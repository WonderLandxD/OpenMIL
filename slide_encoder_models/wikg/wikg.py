import torch
import torch.nn as nn
import torch.nn.functional as F


# Replace torch_geometric pooling functions with pure PyTorch implementations
def global_mean_pool(x, batch=None):
    """
    Global mean pooling: average over the first dimension (nodes)
    Args:
        x: [N, dim] tensor
        batch: ignored (for compatibility)
    Returns:
        [1, dim] tensor
    """
    return x.mean(dim=0, keepdim=True)


def global_max_pool(x, batch=None):
    """
    Global max pooling: max over the first dimension (nodes)
    Args:
        x: [N, dim] tensor
        batch: ignored (for compatibility)
    Returns:
        [1, dim] tensor
    """
    return x.max(dim=0, keepdim=True)[0]


class GlobalAttention(nn.Module):
    """
    Global attention pooling: weighted sum using attention mechanism
    Replaces torch_geometric.nn.GlobalAttention
    """
    def __init__(self, gate_nn):
        super(GlobalAttention, self).__init__()
        self.gate_nn = gate_nn
    
    def forward(self, x, batch=None):
        """
        Args:
            x: [N, dim] tensor
            batch: ignored (for compatibility)
        Returns:
            [1, dim] tensor
        """
        # Compute attention weights: [N, 1]
        gate = self.gate_nn(x)
        
        # Apply softmax to get attention probabilities
        gate = F.softmax(gate, dim=0)
        
        # Weighted sum: [N, dim] * [N, 1] -> [1, dim]
        out = (x * gate).sum(dim=0, keepdim=True)
        
        return out

class WiKG(nn.Module):
    def __init__(self, dim_in=384, dim_hidden=None, num_classes=2, topk=6, agg_type='bi-interaction', dropout=0.3, pool='mean'):
        super().__init__()
        
        # Network configuration parameters
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.num_classes = num_classes
        self.topk = topk
        self.agg_type = agg_type
        self.pool = pool
        
        # Validate aggregation type
        valid_agg_types = ['gcn', 'sage', 'bi-interaction']
        if agg_type not in valid_agg_types:
            raise ValueError(f"Invalid agg_type: '{agg_type}'. "
                           f"Only supports the following options: {valid_agg_types}")
        
        # Validate pooling type
        valid_pool_types = ['mean', 'max', 'attn']
        if pool not in valid_pool_types:
            raise ValueError(f"Invalid pool type: '{pool}'. "
                           f"Only supports the following options: {valid_pool_types}")

        # Feature transformation
        self._fc1 = nn.Sequential(nn.Linear(self.dim_in, self.dim_hidden), nn.LeakyReLU())
        
        # Head and tail transformation for knowledge graph
        self.W_head = nn.Linear(self.dim_hidden, self.dim_hidden)
        self.W_tail = nn.Linear(self.dim_hidden, self.dim_hidden)

        # Attention scaling
        self.scale = self.dim_hidden ** -0.5

        # Aggregation layers based on type
        if self.agg_type == 'gcn':
            self.linear = nn.Linear(self.dim_hidden, self.dim_hidden)
        elif self.agg_type == 'sage':
            self.linear = nn.Linear(self.dim_hidden * 2, self.dim_hidden)
        elif self.agg_type == 'bi-interaction':
            self.linear1 = nn.Linear(self.dim_hidden, self.dim_hidden)
            self.linear2 = nn.Linear(self.dim_hidden, self.dim_hidden)
        
        # Activation and dropout
        self.activation = nn.LeakyReLU()
        self.message_dropout = nn.Dropout(dropout)

        # Normalization and classification
        self.norm = nn.LayerNorm(self.dim_hidden)
        # Handle the case when num_classes=0 (no classification head)
        if self.num_classes == 0:
            self.classifier = nn.Identity()
        else:
            self.classifier = nn.Linear(self.dim_hidden, self.num_classes)

        # Readout/pooling layer
        if pool == "mean":
            self.readout = global_mean_pool 
        elif pool == "max":
            self.readout = global_max_pool 
        elif pool == "attn":
            att_net = nn.Sequential(
                nn.Linear(self.dim_hidden, self.dim_hidden // 2), 
                nn.LeakyReLU(), 
                nn.Linear(self.dim_hidden // 2, 1)
            )     
            self.readout = GlobalAttention(att_net)
        
    def forward(self, input_dict):
        x = input_dict['feats']
        x = self._fc1(x)    # [B, N, self.dim_hidden]

        # Global context integration
        x = (x + x.mean(dim=1, keepdim=True)) * 0.5  

        # Head and tail embeddings for knowledge graph construction
        e_h = self.W_head(x)
        e_t = self.W_tail(x)

        # Construct neighbor relationships using attention
        attn_logit = (e_h * self.scale) @ e_t.transpose(-2, -1)
        topk_weight, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)

        # Prepare indices for advanced indexing
        topk_index = topk_index.to(torch.long)
        topk_index_expanded = topk_index.expand(e_t.size(0), -1, -1)
        batch_indices = torch.arange(e_t.size(0)).view(-1, 1, 1).to(topk_index.device)

        # Get neighbor embeddings
        Nb_h = e_t[batch_indices, topk_index_expanded, :]  # [B, N, topk, dim_hidden]

        # Apply softmax to get neighbor probabilities
        topk_prob = F.softmax(topk_weight, dim=2)
        eh_r = torch.mul(topk_prob.unsqueeze(-1), Nb_h) + torch.matmul((1 - topk_prob).unsqueeze(-1), e_h.unsqueeze(2))

        # Gated knowledge attention
        e_h_expand = e_h.unsqueeze(2).expand(-1, -1, self.topk, -1)
        gate = torch.tanh(e_h_expand + eh_r)
        ka_weight = torch.einsum('ijkl,ijkm->ijk', Nb_h, gate)

        ka_prob = F.softmax(ka_weight, dim=2).unsqueeze(dim=2)
        e_Nh = torch.matmul(ka_prob, Nb_h).squeeze(dim=2)

        # Aggregation based on type
        if self.agg_type == 'gcn':
            embedding = e_h + e_Nh
            embedding = self.activation(self.linear(embedding))
        elif self.agg_type == 'sage':
            embedding = torch.cat([e_h, e_Nh], dim=2)
            embedding = self.activation(self.linear(embedding))
        elif self.agg_type == 'bi-interaction':
            sum_embedding = self.activation(self.linear1(e_h + e_Nh))
            bi_embedding = self.activation(self.linear2(e_h * e_Nh))
            embedding = sum_embedding + bi_embedding

        # Apply dropout
        h = self.message_dropout(embedding)
        # Global readout/pooling
        h = self.readout(h.squeeze(0), batch=None)
        h = h.squeeze(1)

        h = self.norm(h)
        
        # Classification
        logits = self.classifier(h)
        
        return logits


if __name__ == "__main__":
    model = WiKG(dim_in=1024, dim_hidden=512, num_classes=2, topk=6, agg_type='bi-interaction', dropout=0.3, pool='mean')
    input_dict = {"feats": torch.randn(1, 500, 1024)}
    output = model(input_dict)
    print(output.shape)