import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

try:
    import safetensors.torch as safetensors_torch
    _HAS_SAFETENSORS = True
except Exception:
    _HAS_SAFETENSORS = False

def create_mlp(
    in_dim=768,
    hid_dims=None,
    out_dim=512,
    act=nn.ReLU(),
    dropout=0.0,
    end_with_fc=True,
    end_with_dropout=False,
    bias=True,
):
    if hid_dims is None:
        hid_dims = [512, 512]

    layers = []
    if len(hid_dims) < 0:
        mlp = nn.Identity()
    else:
        if len(hid_dims) > 0:
            for hid_dim in hid_dims:
                layers.append(nn.Linear(in_dim, hid_dim, bias=bias))
                layers.append(act)
                layers.append(nn.Dropout(dropout))
                in_dim = hid_dim
        layers.append(nn.Linear(in_dim, out_dim))
        if not end_with_fc:
            layers.append(act)
        if end_with_dropout:
            layers.append(nn.Dropout(dropout))
        mlp = nn.Sequential(*layers)
    return mlp


class GlobalAttention(nn.Module):
    def __init__(self, L=1024, D=256, dropout=0.0, num_classes=1):
        super().__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(D, num_classes),
        ]
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x)


class GlobalGatedAttention(nn.Module):
    def __init__(self, L=1024, D=256, dropout=0.0, num_classes=1):
        super().__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(dropout),
        ]
        self.attention_b = [
            nn.Linear(L, D),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        ]
        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, num_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        return A


class ABMIL(nn.Module):
    def __init__(
        self,
        in_dim=1024,
        embed_dim=512,
        num_fc_layers=1,
        dropout=0.25,
        attn_dim=384,
        gate=True,
        num_classes=2,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] * (num_fc_layers - 1),
            dropout=dropout,
            out_dim=embed_dim,
            end_with_fc=False,
        )

        attn_func = GlobalGatedAttention if gate else GlobalAttention
        self.global_attn = attn_func(
            L=embed_dim,
            D=attn_dim,
            dropout=dropout,
            num_classes=1,
        )

        if num_classes > 0:
            self.classifier = nn.Linear(embed_dim, num_classes)
        else:
            self.classifier = nn.Identity()
        self.initialize_weights()

    @staticmethod
    def compute_loss(loss_fn, logits, label):
        if loss_fn is None or logits is None:
            return None
        return loss_fn(logits, label)

    def initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.LayerNorm):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward_attention(self, h, attn_mask=None, attn_only=True):
        h = self.patch_embed(h)
        A = self.global_attn(h)
        A = torch.transpose(A, -2, -1)
        if attn_mask is not None:
            A = A + (1 - attn_mask).unsqueeze(dim=1) * torch.finfo(A.dtype).min
        if attn_only:
            return A
        return h, A

    def forward_features(self, h, attn_mask=None, return_attention=True):
        h, A_base = self.forward_attention(h, attn_mask=attn_mask, attn_only=False)
        A = F.softmax(A_base, dim=-1)
        h = torch.bmm(A, h).squeeze(dim=1)
        log_dict = {"attention": A_base if return_attention else None}
        return h, log_dict

    def forward_head(self, h):
        logits = self.classifier(h)
        return logits

    def forward(
        self,
        h,
        loss_fn=None,
        label=None,
        attn_mask=None,
        return_attention=False,
        return_slide_feats=False,
    ):
        wsi_feats, log_dict = self.forward_features(
            h, attn_mask=attn_mask, return_attention=return_attention
        )
        logits = self.forward_head(wsi_feats)
        cls_loss = self.compute_loss(loss_fn, logits, label)
        results_dict = {"logits": logits, "loss": cls_loss}
        log_dict["loss"] = cls_loss.item() if cls_loss is not None else -1
        if return_slide_feats:
            log_dict["slide_feats"] = wsi_feats
        return results_dict, log_dict


def _strip_classifier_weights(state_dict):
    return {k: v for k, v in state_dict.items() if not k.startswith("classifier.")}


def _unwrap_state_dict(state_dict):
    if isinstance(state_dict, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in state_dict and isinstance(state_dict[key], dict):
                return state_dict[key]
    return state_dict


def _remove_prefix(state_dict, prefix):
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            cleaned[k[len(prefix):]] = v
        else:
            cleaned[k] = v
    return cleaned


def _normalize_state_dict_keys(state_dict):
    state_dict = _unwrap_state_dict(state_dict)
    state_dict = _remove_prefix(state_dict, "module.")
    state_dict = _remove_prefix(state_dict, "model.")
    return state_dict


def _load_state_dict_file(path):
    if path.endswith(".safetensors"):
        if not _HAS_SAFETENSORS:
            raise ImportError("safetensors is required to load .safetensors weights.")
        return safetensors_torch.load_file(path)
    return torch.load(path, map_location="cpu")


def _download_state_dict_from_hf(model_id, filename=None, cache_dir=None):
    if filename:
        local_path = hf_hub_download(repo_id=model_id, filename=filename, cache_dir=cache_dir)
        return _load_state_dict_file(local_path)

    if _HAS_SAFETENSORS:
        try:
            local_path = hf_hub_download(
                repo_id=model_id, filename="model.safetensors", cache_dir=cache_dir
            )
            return _load_state_dict_file(local_path)
        except Exception:
            pass

    local_path = hf_hub_download(
        repo_id=model_id, filename="pytorch_model.bin", cache_dir=cache_dir
    )
    return _load_state_dict_file(local_path)


class FEATHER_UNIV2_Model(nn.Module):
    def __init__(
        self,
        model_id="MahmoodLab/abmil.base.uni_v2.pc108-24k",
        num_classes=0,
        checkpoint_path=None,
        weights_filename=None,
        cache_dir=None,
    ):
        super().__init__()
        self.model_id = model_id
        print(
            "WARNING: FEATHER_UNIV2 requires patch features extracted by UNI-v2 (1536-dim). "
            "Using other encoders is unreasonable."
        )
        
        self.slide_model = ABMIL(
            in_dim=1536,
            embed_dim=512,
            num_fc_layers=1,
            dropout=0.25,
            attn_dim=384,
            gate=True,
            num_classes=0,
        )

        if checkpoint_path:
            state_dict = _load_state_dict_file(checkpoint_path)
        else:
            state_dict = _download_state_dict_from_hf(
                model_id, filename=weights_filename, cache_dir=cache_dir
            )
        state_dict = _normalize_state_dict_keys(state_dict)
        state_dict = _strip_classifier_weights(state_dict)
        self.slide_model.load_state_dict(state_dict, strict=True)

        if num_classes > 0:
            self.classifier = nn.Linear(512, num_classes)
        else:
            self.classifier = nn.Identity()

    def forward(self, input_dict):
        patch_features = input_dict["feats"]
        assert len(patch_features.shape) == 3
        _, log_dict = self.slide_model(
            patch_features,
            loss_fn=None,
            label=None,
            return_attention=False,
            return_slide_feats=True,
        )
        slide_feats = log_dict["slide_feats"]
        output = self.classifier(slide_feats)
        return output


if __name__ == "__main__":
    model = FEATHER_UNIV2_Model(num_classes=0)
    features = torch.randn(1, 100, 1536)
    output = model({"feats": features})
    print(output.shape)
