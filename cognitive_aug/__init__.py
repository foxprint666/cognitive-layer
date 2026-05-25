"""
Cognitive Augmentation Layers (cognitive_aug)

Modular, brain-inspired cognitive augmentation layers for PyTorch neural networks.
"""

from .engine import CognitiveAugEngine
from .adapters import ModuleAdapter
from .gwt import GlobalWorkspace, AttentionSelector, BroadcastEngine
from .selectors import (
    BaseSelector,
    CosineSimilaritySelector,
    VectorizedCrossAttentionSelector,
    EfficientGumbelSoftmaxSelector,
)
from .salience import (
    global_pool_latent,
    BaseSalience,
    MagnitudeSalience,
    EntropySalience,
    TemporalSurpriseSalience,
)
from .memory import DecayWorkingMemory
from .routing import CognitiveOutputRouter

__all__ = [
    "CognitiveAugEngine",
    "ModuleAdapter",
    "GlobalWorkspace",
    "AttentionSelector",
    "BroadcastEngine",
    "BaseSelector",
    "CosineSimilaritySelector",
    "VectorizedCrossAttentionSelector",
    "EfficientGumbelSoftmaxSelector",
    "global_pool_latent",
    "BaseSalience",
    "MagnitudeSalience",
    "EntropySalience",
    "TemporalSurpriseSalience",
    "DecayWorkingMemory",
    "CognitiveOutputRouter",
]
