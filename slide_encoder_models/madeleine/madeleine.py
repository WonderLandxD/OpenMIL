import torch
import torch.nn as nn
from .backbone.factory import create_model_from_pretrained


class MADELEINEModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        print("WARNING: MADELEINE requires patch features extracted by CONCH-1 (512-dim). "
              "Using other encoders is unreasonable.")
        
        # Create the MADELEINE model using the factory function
        self.backbone = create_model_from_pretrained()

        if num_classes == 0:
            self.classifier = nn.Identity()
        else:
            self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, input_dict):
        # Get the features from input
        feats = input_dict['feats'] 
        
        # Use the encode_he method as specified by the user
        output = self.backbone.encode_he(feats)
        output = self.classifier(output)
        return output 


if __name__ == "__main__":
    model = MADELEINEModel(num_classes=0)
    input_dict = {"feats": torch.randn(1, 500, 512)}
    output = model(input_dict)
    print(output.shape)