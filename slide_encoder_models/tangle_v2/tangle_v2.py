import json
import os
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _read_json(path: str | os.PathLike) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    has_module = any(k.startswith("module.") for k in state_dict.keys())
    if not has_module:
        return state_dict
    return {k[len("module.") :]: v for k, v in state_dict.items()}


class BatchedABMIL(nn.Module):
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        dropout: bool = False,
        n_classes: int = 1,
        activation: str = "softmax",
    ):
        super().__init__()
        self.activation = activation

        attn_a = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        attn_b = [nn.Linear(input_dim, hidden_dim), nn.Sigmoid()]
        if dropout:
            attn_a.append(nn.Dropout(0.25))
            attn_b.append(nn.Dropout(0.25))
        self.attention_a = nn.Sequential(*attn_a)
        self.attention_b = nn.Sequential(*attn_b)
        self.attention_c = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor, return_raw_attention: bool = False):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = self.attention_c(a.mul(b))

        if self.activation == "softmax":
            activated_A = F.softmax(A, dim=1)
        elif self.activation == "leaky_relu":
            activated_A = F.leaky_relu(A)
        elif self.activation == "relu":
            activated_A = F.relu(A)
        elif self.activation == "sigmoid":
            activated_A = torch.sigmoid(A)
        else:
            raise NotImplementedError("Activation not implemented.")

        if return_raw_attention:
            return activated_A, A
        return activated_A


class ABMILEmbedder_MH(nn.Module):
    def __init__(self, pre_attention_params: Dict[str, Any], attention_params: Dict[str, Any], aggregation: str = "regular"):
        super().__init__()
        self.pre_attention_params = pre_attention_params
        self.attention_params = attention_params
        self.n_heads = int(attention_params["params"]["n_heads"])
        self.agg_type = aggregation

        in_dim = int(pre_attention_params["input_dim"])
        hid = int(pre_attention_params["hidden_dim"])

        self.pre_attn = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.LayerNorm(hid),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hid, hid),
            nn.LayerNorm(hid),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hid, hid * self.n_heads),
            nn.LayerNorm(hid * self.n_heads),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        params = dict(attention_params["params"])
        params.pop("n_heads", None)
        self.attn = nn.ModuleList([BatchedABMIL(**params) for _ in range(self.n_heads)])

        self.proj_multihead = nn.Linear(in_features=hid * self.n_heads, out_features=hid)

    def forward(self, bags: torch.Tensor, return_attention: bool = False, n_views: int = 1) -> torch.Tensor:
        # bags: (B, N, D)
        embeddings = self.pre_attn(bags)  # (B, N, H*nh)
        b, n, dh = embeddings.shape
        d = dh // self.n_heads
        embeddings = embeddings.view(b, n, d, self.n_heads)

        attention = []
        raw_attention = []
        for i, attn_net in enumerate(self.attn):
            c_attention, c_raw_attention = attn_net(embeddings[:, :, :, i], return_raw_attention=True)
            attention.append(c_attention)
            raw_attention.append(c_raw_attention)
        attention = torch.stack(attention, dim=-1)
        raw_attention = torch.stack(raw_attention, dim=-1)

        if self.agg_type != "regular" or n_views != 1:
            raise NotImplementedError("Only regular aggregation with n_views=1 is supported.")

        embeddings = embeddings * attention
        slide_embeddings = torch.sum(embeddings, dim=1)  # (B, D, nh)

        slide_embeddings = slide_embeddings.reshape(b, -1)  # (B, D*nh)
        slide_embeddings = self.proj_multihead(slide_embeddings)  # (B, D)

        if return_attention:
            return slide_embeddings, raw_attention
        return slide_embeddings


class TANGLE_V2_Model(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        pretrained_dir: str | os.PathLike = None,
        config_path: str | os.PathLike = None,
    ):
        super().__init__()

        print("WARNING: TANGLEv2 requires patch features extracted by UNI (1024-dim). Using other encoders is unreasonable.")

        here = Path(__file__).resolve()
        default_cfg = here.parent / "tangle_v2_config" / "config.json"
        config_path = Path(config_path) if config_path else default_cfg
        cfg = _read_json(config_path)
        cfg = dict(cfg)
        cfg.setdefault("n_tokens", int(cfg.get("token_size", 4096)))

        # Minimal MMSSL(wsi_encoder=abmil_mh) compatible module structure.
        pre_params = {"input_dim": int(cfg["embedding_dim"]), "hidden_dim": int(cfg["hidden_dim"])}
        attn_params = {
            "model": "ABMIL",
            "params": {
                "input_dim": int(cfg["hidden_dim"]),
                "hidden_dim": int(cfg["hidden_dim"]),
                "dropout": True,
                "activation": str(cfg.get("activation", "softmax")),
                "n_heads": int(cfg.get("n_heads", 4)),
                "n_classes": 1,
            },
        }
        self.wsi_embedder = ABMILEmbedder_MH(pre_attention_params=pre_params, attention_params=attn_params)

        feat_dim = int(cfg["hidden_dim"])
        self.classifier = nn.Identity() if num_classes == 0 else nn.Linear(feat_dim, num_classes)

        # Load pretrained weights (expects an MMSSL-like state dict with 'wsi_embedder.*').
        pretrained_dir = Path(pretrained_dir) if pretrained_dir else config_path.parent
        ckpt_path = pretrained_dir / "model.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}. Please place TANGLEv2 'model.pt' into {pretrained_dir}.")
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        sd = _strip_module_prefix(sd)
        # Accept either full MMSSL dict (wsi_embedder.*) or bare wsi_embedder dict.
        if any(k.startswith("wsi_embedder.") for k in sd.keys()):
            self.load_state_dict(sd, strict=False)
        else:
            self.wsi_embedder.load_state_dict(sd, strict=False)

    def forward(self, input_dict: Dict[str, torch.Tensor]):
        feats = input_dict["feats"]
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)
        slide_emb = self.wsi_embedder(feats)
        out = self.classifier(slide_emb)
        return out



if __name__ == "__main__":
    model = TANGLE_V2_Model(num_classes=0)
    input_dict = {
        "feats": torch.randn(1, 1024, 1024)
    }
    output = model(input_dict)
    print(output.shape)

