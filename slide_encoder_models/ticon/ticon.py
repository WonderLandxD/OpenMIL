from __future__ import annotations

import math
from typing import Any, Dict, Mapping

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download


TICON_MODEL_ID = "varunb/TICON"
TICON_WEIGHTS = "backbone/checkpoint.pth"
TICON_TILE_DIMS = {
    "conchv15": 768,
    "hoptimus1": 1536,
    "uni2h": 1536,
    "gigapath": 1536,
    "virchow2": 1280,
}
TICON_ALIASES = {
    "conch_v1_5": "conchv15",
    "conch-v1-5": "conchv15",
    "conchv15": "conchv15",
    "hoptimus1": "hoptimus1",
    "h_optimus_1": "hoptimus1",
    "h-optimus-1": "hoptimus1",
    "uni2h": "uni2h",
    "uni_v2": "uni2h",
    "uni2_h": "uni2h",
    "gigapath": "gigapath",
    "prov_gigapath": "gigapath",
    "virchow2": "virchow2",
    "virchow_2": "virchow2",
}


def _normalize_tile_encoder_key(tile_encoder_key: str) -> str:
    key = str(tile_encoder_key).lower().replace(".", "_")
    key = TICON_ALIASES.get(key, key)
    if key not in TICON_TILE_DIMS:
        available = ", ".join(sorted(TICON_TILE_DIMS))
        raise ValueError(f"Unknown TICON tile encoder key '{tile_encoder_key}'. Available: {available}")
    return key


def _get_slopes(n: int):
    def get_slopes_power_of_2(k: int):
        start = 2 ** (-(2 ** -(math.log2(k) - 3)))
        return [start * start**i for i in range(k)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    closest_power_of_2 = 2 ** math.floor(math.log2(n))
    return get_slopes_power_of_2(closest_power_of_2) + _get_slopes(2 * closest_power_of_2)[0::2][
        : n - closest_power_of_2
    ]


def _strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    return {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in state_dict.items()}


def _unwrap_state_dict(state_dict):
    if isinstance(state_dict, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if isinstance(state_dict.get(key), dict):
                return state_dict[key]
    return state_dict


def _load_state_dict(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_ticon_weights(checkpoint_path: str | None, cache_dir: str | None):
    if checkpoint_path is None:
        checkpoint_path = hf_hub_download(
            repo_id=TICON_MODEL_ID,
            filename=TICON_WEIGHTS,
            cache_dir=cache_dir,
        )
    state_dict = _unwrap_state_dict(_load_state_dict(checkpoint_path))
    state_dict = _strip_prefix(state_dict, "module.")
    if any(k.startswith("backbone.") for k in state_dict.keys()):
        state_dict = {k[len("backbone.") :]: v for k, v in state_dict.items() if k.startswith("backbone.")}
    return state_dict


class Mlp(nn.Module):
    def __init__(self, in_features: int, mlp_ratio: float = 16 / 3):
        super().__init__()
        hidden_features = int(in_features * mlp_ratio)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_features // 2, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(self.act(x1) * x2)


class ProjectionMlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.fc2(self.act(self.fc1(x))))


class Attention(nn.Module):
    def __init__(self, dim: int = 1536, num_heads: int = 24, context_dim: int | None = None):
        super().__init__()
        self.num_heads = num_heads
        context_dim = context_dim or dim
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(context_dim, dim)
        self.v_proj = nn.Linear(context_dim, dim)
        self.proj = nn.Linear(dim, dim)
        slopes = torch.tensor(_get_slopes(num_heads), dtype=torch.float32)
        self.register_buffer("slopes", slopes[None, :, None, None], persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        context: torch.Tensor | None = None,
        context_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context is None or context_coords is None:
            context = x
            context_coords = coords

        b, n_q, d = x.shape
        n_k = context.shape[1]
        h = self.num_heads
        q = self.q_proj(x).reshape(b, n_q, h, d // h).transpose(1, 2)
        k = self.k_proj(context).reshape(b, n_k, h, d // h).transpose(1, 2)
        v = self.v_proj(context).reshape(b, n_k, h, d // h).transpose(1, 2)

        dist = torch.cdist(coords.float(), context_coords.float(), p=2)
        attn_bias = -self.slopes.to(dist.device) * dist[:, None, :, :]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        attn = attn + attn_bias.to(dtype=attn.dtype)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b, n_q, d)
        return self.proj(x)


class Residual(nn.Module):
    def __init__(self, norm: nn.Module, fn: nn.Module, dim: int):
        super().__init__()
        self.norm = norm
        self.fn = fn
        self.gamma = nn.Parameter(torch.ones(dim), requires_grad=True)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return x + self.gamma * self.fn(self.norm(x), **kwargs)


class Block(nn.Module):
    def __init__(self, dim: int = 1536, num_heads: int = 24, context_dim: int | None = None):
        super().__init__()
        self.residual1 = Residual(
            nn.LayerNorm(dim, eps=1e-5),
            Attention(dim=dim, num_heads=num_heads, context_dim=context_dim),
            dim,
        )
        self.residual2 = Residual(nn.LayerNorm(dim, eps=1e-5), Mlp(in_features=dim), dim)

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        context: torch.Tensor | None = None,
        context_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.residual1(x, context=context, coords=coords, context_coords=context_coords)
        return self.residual2(x)


class Transformer(nn.Module):
    def __init__(self, embed_dim: int = 1536, depth: int = 6, num_heads: int = 24, context_dim: int | None = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_blocks = depth
        self.blocks = nn.ModuleList(
            [Block(dim=embed_dim, num_heads=num_heads, context_dim=context_dim) for _ in range(depth)]
        )

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        return_layers: set[int],
        contexts: list[torch.Tensor] | None = None,
        context_coords: torch.Tensor | None = None,
    ) -> Dict[int, torch.Tensor]:
        outputs = {}
        if 0 in return_layers:
            outputs[0] = x
        for block_idx, block in enumerate(self.blocks):
            context = contexts[block_idx] if contexts is not None else None
            x = block(x, coords=coords, context=context, context_coords=context_coords)
            if block_idx + 1 in return_layers:
                outputs[block_idx + 1] = x
        return outputs


class TICONBackbone(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1536,
        depth: int = 6,
        num_heads: int = 24,
        in_dims: Mapping[str, int] = TICON_TILE_DIMS,
        out_layer: int = -1,
    ):
        super().__init__()
        self.encoder = Transformer(embed_dim=embed_dim, depth=depth, num_heads=num_heads)
        self.tile_encoder_keys = list(in_dims.keys())
        self.embed_dim = embed_dim
        self.n_blocks = depth
        self.out_layer = out_layer % (depth + 1)
        self.enc_norm = nn.LayerNorm(embed_dim, eps=1e-5, elementwise_affine=True)
        self.input_proj_dict = nn.ModuleDict(
            {
                f"input_proj_{key}": ProjectionMlp(
                    in_features=in_dim,
                    hidden_features=embed_dim,
                    out_features=embed_dim,
                )
                for key, in_dim in in_dims.items()
            }
        )

    def forward(self, x: torch.Tensor, relative_coords: torch.Tensor, tile_encoder_key: str) -> torch.Tensor:
        x = self.input_proj_dict[f"input_proj_{tile_encoder_key}"](x)
        outputs = self.encoder(x, coords=relative_coords, return_layers={self.out_layer})
        return self.enc_norm(outputs[self.out_layer])


class TICONModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        tile_encoder_key: str = "conchv15",
        checkpoint_path: str | None = None,
        cache_dir: str | None = None,
        pretrained: bool = True,
    ):
        super().__init__()
        self.tile_encoder_key = _normalize_tile_encoder_key(tile_encoder_key)
        print(
            "WARNING: TICON contextualizes tile features from supported encoders "
            "(conchv15, hoptimus1, uni2h, gigapath, virchow2). "
            f"Current tile_encoder_key={self.tile_encoder_key}."
        )

        self.backbone = TICONBackbone()
        if pretrained:
            state_dict = _load_ticon_weights(checkpoint_path=checkpoint_path, cache_dir=cache_dir)
            self.backbone.load_state_dict(state_dict, strict=True)

        self.classifier = nn.Identity() if num_classes == 0 else nn.Linear(self.backbone.embed_dim, num_classes)

    @staticmethod
    def _relative_coords(input_dict: Dict[str, Any], feats: torch.Tensor) -> torch.Tensor:
        coords = input_dict["coords"]
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
        coords = coords.to(device=feats.device, dtype=torch.float32)
        patch_size_lv0 = input_dict.get("patch_size_lv0")
        if patch_size_lv0 is not None:
            if not torch.is_tensor(patch_size_lv0):
                patch_size_lv0 = torch.tensor(patch_size_lv0, device=feats.device)
            patch_size_lv0 = patch_size_lv0.to(device=feats.device, dtype=torch.float32).reshape(-1, 1, 1)
            coords = coords / patch_size_lv0.clamp_min(1.0)
        return coords

    def forward(self, input_dict: Dict[str, Any]) -> torch.Tensor:
        feats = input_dict["feats"]
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)

        tile_encoder_key = _normalize_tile_encoder_key(input_dict.get("tile_encoder_key", self.tile_encoder_key))
        expected_dim = TICON_TILE_DIMS[tile_encoder_key]
        if feats.shape[-1] != expected_dim:
            raise ValueError(
                f"TICON tile_encoder_key={tile_encoder_key} expects {expected_dim}-dim features, "
                f"got {feats.shape[-1]}."
            )

        relative_coords = self._relative_coords(input_dict, feats)
        tile_embeddings = self.backbone(feats, relative_coords=relative_coords, tile_encoder_key=tile_encoder_key)
        slide_embedding = tile_embeddings.mean(dim=1)
        return self.classifier(slide_embedding)


if __name__ == "__main__":
    model = TICONModel(num_classes=0, pretrained=True, tile_encoder_key="conchv15")
    patch_size_lv0 = 1024
    coords = torch.tensor([[0, 0], [1024, 0], [0, 1024], [1024, 1024]], dtype=torch.long).unsqueeze(0)
    print(coords.shape)
    print(patch_size_lv0)
    input_dict = {
        "feats": torch.randn(1, 4, 768),
        "coords": coords,
        "patch_size_lv0": patch_size_lv0,
    }
    print(input_dict["feats"].shape)
    print(model(input_dict).shape)
