import torch
import torch.nn as nn
from transformers import AutoModel


class TITANModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        print("WARNING: TITAN requires patch features extracted by CONCH-1.5 (768-dim). "
              "Using other encoders is unreasonable. "
              "\033[33mAlso, TITAN requires patch coordinates and patch size level 0. To see how to get them, please use h5 file with `file['coords'].attrs['patch_size_level0']`\033[0m")

        self.backbone = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)

        if num_classes == 0:
            self.classifier = nn.Identity()
        else:
            self.classifier = nn.Linear(768, num_classes)
    
    def forward(self, input):
        patch_features = input['feats']
        coordinates = input['coords']
        patch_size_lv0 = input['patch_size_lv0']
        output = self.backbone.encode_slide_from_patch_features(patch_features, coordinates, patch_size_lv0)
        output = self.classifier(output)
        return output 


if __name__ == "__main__":
    model = TITANModel(num_classes=2)
    # TITAN expects coordinates to be actual patch pixel coordinates
    # Create a grid-like layout: 20x25 patches with patch_size_lv0 spacing
    patch_size_lv0 = 1024
    n_patches = 500
    # Create coordinates in a grid pattern
    grid_w = int(n_patches ** 0.5)  # ~22
    grid_h = (n_patches + grid_w - 1) // grid_w  # ~23
    coords_list = []
    for i in range(n_patches):
        row = i // grid_w
        col = i % grid_w
        # Coordinates should be pixel positions, not grid indices
        x = col * patch_size_lv0
        y = row * patch_size_lv0
        coords_list.append([x, y])
    coords = torch.tensor(coords_list, dtype=torch.int64).unsqueeze(0)  # (1, 500, 2)
    print(f"Coords shape: {coords.shape}, min: {coords.min()}, max: {coords.max()}")
    input_dict = {"feats": torch.randn(1, 500, 768), "coords": coords, "patch_size_lv0": patch_size_lv0}
    output = model(input_dict)
    print(f"Output shape: {output.shape}")