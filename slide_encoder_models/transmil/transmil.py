import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention

class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, dim_in, dim_hidden=None, num_classes=2, num_heads=8, dropout=0.25):
        super().__init__()
        

        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.dropout = dropout

        self.pos_layer = PPEG(dim=self.dim_hidden)
        self._fc1 = nn.Sequential(nn.Linear(self.dim_in, self.dim_hidden), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim_hidden))

        self.layer1 = TransLayer(dim=self.dim_hidden)
        self.layer2 = TransLayer(dim=self.dim_hidden)
        self.norm = nn.LayerNorm(self.dim_hidden)
        # Handle the case when num_classes=0 (no classification head)
        if self.num_classes == 0:
            self._fc2 = nn.Identity()
        else:
            self._fc2 = nn.Linear(self.dim_hidden, self.num_classes)

    def forward(self, input_dict):

        h = input_dict['feats'] #[B, n, C]
        
        h = self._fc1(h) #[B, n, self.dim_hidden]
        
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, self.dim_hidden]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h) #[B, N, self.dim_hidden]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, self.dim_hidden]
        
        #---->Translayer x2
        h = self.layer2(h) #[B, N, self.dim_hidden]

        #---->cls_token
        h = self.norm(h)[:,0]

        #---->predict
        logits = self._fc2(h) #[B, self.num_classes]

        return logits


if __name__ == "__main__":
    model = TransMIL(dim_in=1024, dim_hidden=512, num_classes=2, num_heads=8, dropout=0.25)
    input_dict = {"feats": torch.randn(1, 500, 1024)}
    output = model(input_dict)
    print(output.shape)