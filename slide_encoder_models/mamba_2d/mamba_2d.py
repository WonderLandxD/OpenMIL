import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .mamba_simple import MambaConfig as SimpleMambaConfig
from .mamba_simple import Mamba as SimpleMamba


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class MambaMIL_2D(nn.Module):
    def __init__(self, dim_in=1024, dim_hidden=128, dropout=0.25, num_classes=2, pos_emb_type=None):   
        super(MambaMIL_2D, self).__init__()
        
        # self.args = args
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dropout = dropout

        self._fc1 = [nn.Linear(self.dim_in, self.dim_hidden)]
        self._fc1 += [nn.GELU()]
        if self.dropout > 0:
            self._fc1 += [nn.Dropout(self.dropout)]

        self._fc1 = nn.Sequential(*self._fc1)
        
        self.norm = nn.LayerNorm(self.dim_hidden)
        
        self.layers = nn.ModuleList()
        config = SimpleMambaConfig(
            d_model = self.dim_hidden,
            n_layers = 2,
            d_state = 16,
            inner_layernorms = False,
            pscan = True,
            use_cuda = False,
            mamba_2d = True,
            mamba_2d_max_w = 100000,
            mamba_2d_max_h = 100000,
            mamba_2d_pad_token = 'trainable',
            mamba_2d_patch_size = 512
        )
        self.layers = SimpleMamba(config)
        self.config = config

        self.n_classes = num_classes

        self.attention = nn.Sequential(
                nn.Linear(self.dim_hidden, self.dim_hidden),
                nn.Tanh(),
                nn.Linear(self.dim_hidden, 1)
            )
        # Handle the case when num_classes=0 (no classification head)
        if num_classes == 0:
            self.classifier = nn.Identity()
        else:
            self.classifier = nn.Linear(self.dim_hidden, self.n_classes)

        self.pos_emb_type = pos_emb_type
        if pos_emb_type == 'linear':
            self.pos_embs = nn.Linear(2, self.dim_hidden)
            self.norm_pe = nn.LayerNorm(self.dim_hidden)
            self.pos_emb_dropout = nn.Dropout(0.25)
        else:
            self.pos_embs = None

        self.apply(initialize_weights)

    def forward(self, input_dict):
        x = input_dict['feats']   # [1, N, C]
        coords = input_dict['coords'].squeeze(0).to(x.dtype)  # [1, N, 2]
   
        h = x  # [1, num_patch, feature_dim]

        h = self._fc1(h)  # [1, num_patch, mamba_dim];   project from feature_dim -> mamba_dim

        # Add Pos_emb
        if self.pos_emb_type == 'linear':
            pos_embs = self.pos_embs(coords)
            h = h + pos_embs.unsqueeze(0)
            h = self.pos_emb_dropout(h)

        coords = coords.to(x.dtype)
        h = self.layers(h, coords, self.pos_embs)

        h = self.norm(h)   # LayerNorm
        A = self.attention(h) # [1, W, H, 1]

        if len(A.shape) == 3:
            A = torch.transpose(A, 1, 2)
        else:  
            A = A.permute(0,3,1,2)
            A = A.view(1,1,-1)
            h = h.view(1,-1,self.config.d_model)

        A = F.softmax(A, dim=-1)  # [1, 1, num_patch]  # A: attention weights of patches
        h = torch.bmm(A, h) # [1, 1, 512] , weighted combination to obtain slide feature
        h = h.squeeze(0)  # [1, 512], 512 is the slide dim

        logits = self.classifier(h)  # [1, n_classes]

        return logits
    
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fc1 = self._fc1.to(device)
        self.layers  = self.layers.to(device)
        
        self.attention = self.attention.to(device)
        self.norm = self.norm.to(device)
        self.classifier = self.classifier.to(device)



    
if __name__ == '__main__':
    model = MambaMIL_2D(dim_in=1024, dim_hidden=128, drop_out=0.25, num_classes=2, pos_emb_type=None)
    input_dict = {"feats": torch.randn(1, 500, 1024), "coords": torch.randn(1, 500, 2)}
    output = model(input_dict)
    print(output.shape)