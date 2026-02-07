import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
import os

from .GigaPath.slide_encoder import create_model


class GigaPath_Model(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        print("WARNING: GigaPath requires patch features extracted by Prov-GigaPath (1536-dim). "
              "Using other encoders is unreasonable.")
        print("WARNING: GigaPath requires Flash Attention and CUDA with bfloat16 support.")

        slide_path = hf_hub_download(repo_id="prov-gigapath/prov-gigapath", filename="slide_encoder.pth")
        self.slide_model = create_model(slide_path, "gigapath_slide_enc12l768d", 1536)

        if num_classes == 0:
            self.classifier = nn.Identity()
        else:
            self.classifier = nn.Linear(768, num_classes)
    
    def forward(self, input_dict):
        patch_features = input_dict['feats']
        coordinates = input_dict['coords']
        output = self.slide_model(patch_features, coordinates)[0]
        output = self.classifier(output)
        return output 


if __name__ == "__main__":
    model = GigaPath_Model(num_classes=2).cuda().to(torch.bfloat16)
    input_dict = {"feats": torch.randn(1, 500, 1536).cuda().to(torch.bfloat16), "coords": torch.randn(1, 500, 2).cuda().to(torch.bfloat16)}
    output = model(input_dict)
    print(output.shape)