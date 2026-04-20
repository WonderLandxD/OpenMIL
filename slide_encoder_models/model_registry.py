from __future__ import annotations

from typing import Dict, List, Type, Callable, Any
import torch.nn as nn

# Import all model classes
from .abmil.abmil import ABMIL
from .abmil.gated_abmil import GatedABMIL
from .amdmil.amdmil import AMD_MIL
from .chief.chief import CHIEF_Model
from .clam.clam import CLAM_SB, CLAM_MB
from .dsmil.dsmil import DSMIL
from .transmil.transmil import TransMIL
from .mamba_2d.mamba_2d import MambaMIL_2D
from .feather.feather_uni_v1_model import FEATHER_UNIV1_Model
from .feather.feather_uni_v2_model import FEATHER_UNIV2_Model
from .feather.feather_conch_v1_5_model import FEATHER_CONCH_V1_5_Model
from .gigapath.gigapath_model import GigaPath_Model
from .madeleine.madeleine import MADELEINEModel
from .prism.prism import PRISMModel
from .titan.titan import TITANModel
from .care.care import CAREModel
from .tangle_v2.tangle_v2 import TANGLE_V2_Model

# Import other models if they exist
try:
    from .retmil.retmil import RetMIL
except ImportError:
    RetMIL = None

try:
    from .wikg.wikg import WiKG
except ImportError:
    WiKG = None

try:
    from .ilramil.ilramil import ILRAMIL
except ImportError:
    ILRAMIL = None


# Centralized default hyperparameters for each model
MODEL_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "abmil": {
        "dim_in": 1024,
        "dim_hidden": 512,
        "dropout": 0.5,
        "num_classes": 2,
    },
    "gated_abmil": {
        "dim_in": 1024,
        "dim_hidden": 512,
        "dropout": 0.5,
        "num_classes": 2,
    },
    "transmil": {
        "dim_in": 1024,
        "dim_hidden": 512,
        "num_heads": 8,
        "dropout": 0.25,
        "num_classes": 2,
    },
    "amdmil": {
        "dim_in": 1024,
        "dim_hidden": 512,
        "dropout": 0.25,
        "agent_num": 256,
        "num_classes": 2,
    },
    "clam_sb": {
        "dim_in": 1024,
        "dim_hidden": 512,
        "dropout": 0.25,
        "k_sample": 8,
        "num_classes": 2,
    },
    "clam_mb": {
        "dim_in": 1024,
        "dim_hidden": 512,
        "dropout": 0.25,
        "k_sample": 8,
        "num_classes": 2,
    },
    "dsmil": {
        "dim_in": 1024,
        "dim_hidden": 512,
        "dropout": 0.5,
        "num_classes": 2,
    },
    "2dmamba": {
        "dim_in": 1024,
        "dim_hidden": 128,
        "dropout": 0.25,
        "num_classes": 2,
        "pos_emb_type": None,
    },
    "titan": {
        "num_classes": 2,
    },
    "prism": {
        "num_classes": 2,
    },
    "gigapath": {
        "num_classes": 2,
    },
    "madeleine": {
        "num_classes": 2,
    },
    "chief": {
        "num_classes": 2,
    },
    "care": {
        "num_classes": 2,
        "model_name": "Zipper-1/CARE",
        "local_files_only": True,
    },
    "tangle_v2": {
        "num_classes": 2,
        "pretrained_dir": None,
        "config_path": None,
    },
    "feather_uni_v1": {
        "model_id": "MahmoodLab/abmil.base.uni.pc108-24k",
        "num_classes": 0,
        "checkpoint_path": None,
        "weights_filename": None,
        "cache_dir": None,
    },
    "feather_uni_v2": {
        "model_id": "MahmoodLab/abmil.base.uni_v2.pc108-24k",
        "num_classes": 0,
        "checkpoint_path": None,
        "weights_filename": None,
        "cache_dir": None,
    },
    "feather_conch_v1_5": {
        "model_id": "MahmoodLab/abmil.base.conch_v15.pc108-24k",
        "num_classes": 0,
        "checkpoint_path": None,
        "weights_filename": None,
        "cache_dir": None,
    },
    "retmil": {
        "dim_in": 1024,
        "dim_hidden": 512,
        "num_heads": 8,
        "chunk_size": 512,
        "num_classes": 2,
    },
    "wikg": {
        "dim_in": 1024,
        "dim_hidden": 512,
        "num_classes": 2,
        "topk": 6,
        "agg_type": "bi-interaction",
        "dropout": 0.3,
        "pool": "mean",
    },
    "ilramil": {
        "dim_in": 1024,
        "dim_hidden": 512,
        "num_classes": 2,
        "num_layers": 2,
        "num_heads": 8,
        "topk": 2,
        "ln": False,
        "dropout": 0.25,
    },
}


def _merge_defaults(model_key: str, **kwargs: Any) -> Dict[str, Any]:
    defaults = MODEL_DEFAULTS.get(model_key, {})
    return {**defaults, **kwargs}


# Model registry with factory functions
MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    # Basic MIL models
    "abmil": lambda **kwargs: ABMIL(**_merge_defaults("abmil", **kwargs)),
    "gated_abmil": lambda **kwargs: GatedABMIL(**_merge_defaults("gated_abmil", **kwargs)),
    "transmil": lambda **kwargs: TransMIL(**_merge_defaults("transmil", **kwargs)),
    "amdmil": lambda **kwargs: AMD_MIL(**_merge_defaults("amdmil", **kwargs)),
    "clam_sb": lambda **kwargs: CLAM_SB(**_merge_defaults("clam_sb", **kwargs)),
    "clam_mb": lambda **kwargs: CLAM_MB(**_merge_defaults("clam_mb", **kwargs)),
    "dsmil": lambda **kwargs: DSMIL(**_merge_defaults("dsmil", **kwargs)),
    "2dmamba": lambda **kwargs: MambaMIL_2D(**_merge_defaults("2dmamba", **kwargs)),
    "retmil": lambda **kwargs: RetMIL(**_merge_defaults("retmil", **kwargs)),
    "wikg": lambda **kwargs: WiKG(**_merge_defaults("wikg", **kwargs)),
    "ilramil": lambda **kwargs: ILRAMIL(**_merge_defaults("ilramil", **kwargs)),
    
    # Pre-trained slide encoder models
    "titan": lambda **kwargs: TITANModel(**_merge_defaults("titan", **kwargs)),
    "prism": lambda **kwargs: PRISMModel(**_merge_defaults("prism", **kwargs)),
    "gigapath": lambda **kwargs: GigaPath_Model(**_merge_defaults("gigapath", **kwargs)),
    "madeleine": lambda **kwargs: MADELEINEModel(**_merge_defaults("madeleine", **kwargs)),
    "chief": lambda **kwargs: CHIEF_Model(**_merge_defaults("chief", **kwargs)),
    "feather_uni_v1": lambda **kwargs: FEATHER_UNIV1_Model(**_merge_defaults("feather_uni_v1", **kwargs)),
    "feather_uni_v2": lambda **kwargs: FEATHER_UNIV2_Model(**_merge_defaults("feather_uni_v2", **kwargs)),
    "feather_conch_v1_5": lambda **kwargs: FEATHER_CONCH_V1_5_Model(**_merge_defaults("feather_conch_v1_5", **kwargs)),
    "care": lambda **kwargs: CAREModel(**_merge_defaults("care", **kwargs)),
    "tangle_v2": lambda **kwargs: TANGLE_V2_Model(**_merge_defaults("tangle_v2", **kwargs)),
}


def list_models() -> List[str]:
    """List all available slide encoder models."""
    return sorted(MODEL_REGISTRY.keys())


def create_slide_encoder(name: str, **kwargs) -> nn.Module:
    """
    Create a slide encoder model by name.
    
    Args:
        name: Name of the model (case-insensitive). Available models:
            - Basic MIL models: abmil, gated_abmil, transmil, amdmil, clam_sb, clam_mb, dsmil, 2dmamba
            - Pre-trained models: titan, prism, gigapath, madeleine, chief, feather_uni_v1, feather_uni_v2, feather_conch_v1_5
        **kwargs: Additional arguments passed to the model constructor.
            Common arguments:
            - num_classes: Number of output classes (default: 2, set to 0 for feature extraction only)
            - dim_in: Input feature dimension (for basic MIL models)
            - dim_hidden: Hidden dimension (for basic MIL models)
            - dropout: Dropout rate (for basic MIL models)
            - Other model-specific arguments
    
    Returns:
        A PyTorch model instance.
    
    Examples:
        >>> # Create a basic ABMIL model
        >>> model = create_slide_encoder("abmil", dim_in=1024, dim_hidden=512, num_classes=2)
        
        >>> # Create a pre-trained TITAN model for feature extraction
        >>> model = create_slide_encoder("titan", num_classes=0)
        
        >>> # Create a TransMIL model with custom parameters
        >>> model = create_slide_encoder("transmil", dim_in=768, dim_hidden=256, num_classes=10)
    """
    key = name.lower().replace("-", "_")
    if key not in MODEL_REGISTRY:
        available = ", ".join(list_models())
        raise ValueError(
            f"Unknown slide encoder: '{name}'. Available models: {available}"
        )
    
    factory = MODEL_REGISTRY[key]
    return factory(**kwargs)
