import torch
import torch.nn as nn
from transformers import AutoModel


class PRISMModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        print("WARNING: PRISM requires patch features extracted by Virchow-1 (1280-dim CLS token + 1280-dim patch tokens with mean pooling). "
              "Using other encoders is unreasonable.")
        
        self.backbone = AutoModel.from_pretrained('paige-ai/Prism', trust_remote_code=True)

        if num_classes == 0:
            self.classifier = nn.Identity()
        else:
            self.classifier = nn.Linear(1280, num_classes)
    
    def forward(self, input_dict):
        patch_features = input_dict['feats']
        result = self.backbone.slide_representations(patch_features)
        output = result['image_embedding']
        output = self.classifier(output)
        return output 


if __name__ == "__main__":
    model = PRISMModel(num_classes=2)
    input_dict = {"feats": torch.randn(1, 500, 2560)}
    output = model(input_dict)
    print(output.shape)