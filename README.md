# MIL Bench

This repo provides multiple slide-level models for whole-slide image (WSI) analysis, including classic MIL architectures and pre-trained slide encoders that operate on patch features.

## Models

**MIL models**
- abmil
- gated_abmil
- transmil
- amdmil
- clam_sb
- clam_mb
- dsmil
- 2dmamba
- retmil
- wikg
- ilramil

**Pre-trained slide encoders**
- titan
- prism
- gigapath
- madeleine
- chief
- feather_uni_v1
- feather_uni_v2
- feather_conch_v1_5

## Usage

Run the demo to verify all models:
```bash
python demo_test_model.py
```

Create a model in Python:
```python
from slide_encoder_models.model_registry import create_slide_encoder, list_models

print(list_models())
model = create_slide_encoder("abmil", dim_in=1024, dim_hidden=512, num_classes=2)
```
