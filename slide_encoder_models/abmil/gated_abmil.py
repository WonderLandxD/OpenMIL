import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedABMIL(nn.Module):
    def __init__(self, dim_in, dim_hidden, dropout, num_classes):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dropout = dropout
        self.num_classes = num_classes

        self.attn_1 = nn.Sequential(
            nn.Linear(self.dim_in, self.dim_hidden),
            nn.Tanh(),
            nn.Dropout(self.dropout)
        )

        self.attn_2 = nn.Sequential(
            nn.Linear(self.dim_in, self.dim_hidden),
            nn.Sigmoid(),
            nn.Dropout(self.dropout)
        )

        self.attn_3 = nn.Linear(self.dim_hidden, 1)

        if self.num_classes == 0:
            self.classifier = nn.Identity()
        else:
            self.classifier = nn.Linear(self.dim_in, self.num_classes)

    def forward(self, input_dict):
        x = input_dict["feats"] # input_dict["feats"]: [1, N, C]
        attn_1 = self.attn_1(x)
        attn_2 = self.attn_2(x)
        attn = attn_1.mul(attn_2)
        attn = self.attn_3(attn)
        A = torch.transpose(attn, -1, -2)
        A = torch.softmax(A, dim=-1)
        slide_feats = torch.matmul(A, x).squeeze(1)
        output = self.classifier(slide_feats)
        
        return output


if __name__ == "__main__":
    model = GatedABMIL(dim_in=1024, dim_hidden=512, dropout=0.5, num_classes=10)
    input_dict = {"feats": torch.randn(1, 500, 1024)}
    output = model(input_dict)
    print(output.shape)