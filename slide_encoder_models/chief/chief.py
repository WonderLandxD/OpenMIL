import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from .backbone import CHIEF


class CHIEF_Model(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        print("WARNING: CHIEF requires patch features extracted by CTransPath (768-dim). "
              "Using other encoders is unreasonable.")

        word_embedding_path = hf_hub_download(repo_id="JWonderLand/CHIEF_unofficial", filename="Text_emdding.pth")
        td = torch.load(hf_hub_download(repo_id="JWonderLand/CHIEF_unofficial", filename="CHIEF_pretraining.pth"), weights_only=False)

        self.backbone = CHIEF(size_arg='small', word_embedding_path=word_embedding_path, dropout=True, n_classes=2)
        self.backbone.load_state_dict(td, strict=True)

        if num_classes == 0:
            self.classifier = nn.Identity()
        else:
            self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_dict):
        feats = input_dict['feats']
        results = self.backbone(feats, x_anatomic=13)
        output = results['WSI_feature']
        return output


if __name__ == "__main__":
    model = CHIEF_Model(num_classes=2)
    input_dict = {"feats": torch.randn(1, 500, 768)}
    output = model(input_dict)
    print(output.shape)

        