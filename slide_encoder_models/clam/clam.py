import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""
class CLAM_SB(nn.Module):
    def __init__(self, dim_in, dim_hidden, dropout, num_classes, k_sample):
        super().__init__()

        self.k_sample = k_sample
        self.num_classes = num_classes
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        
        self.attention_net = Attn_Net_Gated(
            L=self.dim_in,
            D=self.dim_hidden,
            dropout=dropout,
            n_classes=1
        )
        
        # Handle the case when num_classes=0 (no classification head)
        if num_classes == 0:
            raise ValueError("CLAM must have at least one class, please set num_classes > 0")
        else:
            self.classifier = nn.Linear(self.dim_in, num_classes)
        
        # Handle the case when num_classes=0 (no classification head) for instance classifiers
        instance_classifiers = [nn.Linear(self.dim_in, 2) for _ in range(num_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        
        self.instance_loss_fn = nn.CrossEntropyLoss()
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()
    
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()
    
    def inst_eval(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, input_dict, label=None):
        h = input_dict["feats"].squeeze(0) # input_dict["feats"]: [1, N, C]
        label = label

        A, h = self.attention_net(h)
        A = torch.transpose(A, 1, 0)
        A_raw = A
        A = F.softmax(A, dim=1)

        if label is not None:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.num_classes).squeeze()
            
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                total_inst_loss += instance_loss

            instance_loss = total_inst_loss
            inst_labels = np.array(all_targets)
            inst_preds = np.array(all_preds)

        M = torch.mm(A, h)
        output = self.classifier(M)
        
        if label is not None:
            return output, instance_loss, inst_labels, inst_preds
        else:
            return output
    

class CLAM_MB(CLAM_SB):
    def __init__(self, dim_in, dim_hidden, dropout, num_classes, k_sample):
        super().__init__(dim_in, dim_hidden, dropout, num_classes, k_sample)
        
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.num_classes = num_classes
        self.k_sample = k_sample

        self.attention_net = Attn_Net_Gated(
            L=self.dim_in,
            D=self.dim_hidden,
            dropout=dropout,
            n_classes=num_classes
        )
        
        # Handle the case when num_classes=0 (no classification head)
        if num_classes == 0:
            raise ValueError("CLAM must have at least one class, please set num_classes > 0")
        else:
            bag_classifiers = [nn.Linear(self.dim_in, 1) for _ in range(num_classes)]
            instance_classifiers = [nn.Linear(self.dim_in, 2) for _ in range(num_classes)]
        self.classifiers = nn.ModuleList(bag_classifiers)
        self.instance_classifiers = nn.ModuleList(instance_classifiers)

        self.instance_loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_dict, label=None):
        h = input_dict["feats"].squeeze(0) # input_dict["feats"]: [1, N, C]
        label = label

        A, h = self.attention_net(h)
        A = torch.transpose(A, 1, 0)
        A_raw = A
        A = F.softmax(A, dim=1)

        if label is not None:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.num_classes).squeeze()
            
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                total_inst_loss += instance_loss

            instance_loss = total_inst_loss
            inst_labels = np.array(all_targets)
            inst_preds = np.array(all_preds)

        M = torch.mm(A, h)
        
        output = torch.empty(1, self.num_classes, dtype=h.dtype, device=h.device)
        for c in range(self.num_classes):
            output[0, c] = self.classifiers[c](M[c])
        
        if label is not None:
            return output, instance_loss, inst_labels, inst_preds
        else:
            return output


if __name__ == "__main__":
    model = CLAM_MB(dim_in=1024, dim_hidden=512, dropout=0.25, num_classes=2, k_sample=6)
    input_dict = {"feats": torch.randn(1, 500, 1024)}
    output, instance_loss, inst_labels, inst_preds = model(input_dict, label=torch.tensor([0]))
    # output = model(input_dict)
    print(output.shape)
    print(instance_loss)
    print(inst_labels)
    print(inst_preds)