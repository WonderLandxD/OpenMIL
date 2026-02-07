"""
Slide Encoder Models - Unified interface for creating slide encoder models.

This package provides a unified interface to create various slide encoder models
for whole slide image analysis in multiple instance learning (MIL) settings.
"""

from .model_registry import (
    create_slide_encoder,
    list_models,
    MODEL_REGISTRY,
)

__all__ = [
    "create_slide_encoder",
    "list_models",
    "MODEL_REGISTRY",
]
