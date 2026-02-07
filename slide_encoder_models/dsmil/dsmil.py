import torch
import torch.nn as nn
import torch.nn.functional as F


class FCLayer(nn.Module):
    def __init__(self, dim_in, num_classes):
        super().__init__()
        self.dim_in = dim_in
        self.num_classes = num_classes

        if self.num_classes == 0:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(self.dim_in, self.num_classes)

    def forward(self, feats):
        x = self.fc(feats)
        return feats, x


class BClassifier(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_classes, nonlinear=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.num_classes = num_classes
        self.nonlinear = nonlinear

        if self.nonlinear:
            self.q = nn.Sequential(
                nn.Linear(self.dim_in, self.dim_hidden),
                nn.ReLU(),
                nn.Linear(self.dim_hidden, self.dim_hidden),
                nn.Tanh()
            )
        else:
            self.q = nn.Linear(self.dim_in, self.dim_hidden)
        
        self.fcc = nn.Conv1d(num_classes, num_classes, kernel_size=dim_in)

    def forward(self, feats, classes):
        V = feats
        # Q: query vectors for all instances
        num_instances = feats.shape[0]
        Q = self.q(feats).view(num_instances, -1)

        # Find top instances based on class scores
        _, m_indices = torch.sort(classes, dim=0, descending=True)
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :])
        q_max = self.q(m_feats)

        # Compute attention scores: Q @ q_max^T
        A = torch.mm(Q, q_max.t())
        # Apply scaled softmax
        hidden_dim = Q.shape[1]
        scale_factor = hidden_dim ** 0.5
        A = F.softmax(A / scale_factor, dim=0)
        # B: aggregated bag representation: A^T @ V
        B = torch.mm(A.t(), V)
        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V

        C = self.fcc(B) # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B 




class DSMIL(nn.Module):
    def __init__(self, dim_in, dim_hidden, dropout, num_classes):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dropout = dropout
        self.num_classes = num_classes
        if self.num_classes == 0:
            raise ValueError("DSMIL must have at least one class, please set num_classes > 0")

        self.i_classifier = FCLayer(self.dim_in, self.num_classes)
        self.b_classifier = BClassifier(self.dim_in, self.dim_hidden, self.num_classes, nonlinear=True)

    def forward(self, input_dict):
        x = input_dict["feats"].squeeze(0) # input_dict["feats"]: [1, N, C]
        feats, classes = self.i_classifier(x) 
        output, A, B = self.b_classifier(feats, classes)
        return output


if __name__ == "__main__":
    model = DSMIL(dim_in=1024, dim_hidden=512, dropout=0.5, num_classes=10)
    input_dict = {"feats": torch.randn(1, 500, 1024)}
    output = model(input_dict)
    print(output.shape)