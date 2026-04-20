import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from transformers import AutoModel


class CAREModel(nn.Module):
    def __init__(
        self,
        num_classes=2,
        model_name="Zipper-1/CARE",
        local_files_only=True,
    ):
        super().__init__()

        print(
            "WARNING: CARE requires patch features extracted by CONCH-1.5 (768-dim). "
            "Using other encoders is unreasonable. "
            "\033[33mAlso, CARE requires patch coordinates and patch size level 0. "
            "To see how to get them, please use h5 file with "
            "`file['coords'].attrs['patch_size_level0']`\033[0m"
        )

        os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton")
        Path(os.environ["TRITON_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
        # Zipper CARE ships care_core.py / configuration_care.py next to modeling_care.py; they are not pip packages.
        # Transformers' import check runs before those names resolve unless the snapshot root is on sys.path.
        repo_dir = snapshot_download(
            repo_id=model_name,
            local_files_only=local_files_only,
        )
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)

        self.backbone = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )

        if num_classes == 0:
            self.classifier = nn.Identity()
        else:
            self.classifier = nn.Linear(512, num_classes)

    def forward(self, input):
        patch_features = input["feats"]
        patch_coords = input["coords"]
        patch_size_lv0 = input["patch_size_lv0"]

        if patch_features.dim() == 2:
            patch_features = patch_features.unsqueeze(0)
        if patch_coords.dim() == 2:
            patch_coords = patch_coords.unsqueeze(0)

        patch_coords = patch_coords.long()
        if isinstance(patch_size_lv0, torch.Tensor):
            ps = int(patch_size_lv0.reshape(-1)[0].item())
        else:
            ps = int(patch_size_lv0)
        coords = patch_coords // ps

        N_values = torch.tensor([coords.shape[1]], dtype=torch.long, device=coords.device)
        out = self.backbone(patch_features, N_values, coords)
        slide_embedding = out.wsi_embedding
        output = self.classifier(slide_embedding)
        return output


if __name__ == "__main__":
    model = CAREModel(num_classes=2)
    patch_size_lv0 = 1024
    n_patches = 500
    grid_w = int(n_patches**0.5)
    grid_h = (n_patches + grid_w - 1) // grid_w
    coords_list = []
    for i in range(n_patches):
        row = i // grid_w
        col = i % grid_w
        x = col * patch_size_lv0
        y = row * patch_size_lv0
        coords_list.append([x, y])
    coords = torch.tensor(coords_list, dtype=torch.int64).unsqueeze(0)
    print(f"Coords shape: {coords.shape}, min: {coords.min()}, max: {coords.max()}")
    input_dict = {
        "feats": torch.randn(1, 500, 768),
        "coords": coords,
        "patch_size_lv0": patch_size_lv0,
    }
    output = model(input_dict)
    print(f"Output shape: {output.shape}")

