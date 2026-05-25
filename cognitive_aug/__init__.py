"""
Cognitive Augmentation Layers (cognitive_aug)

Modular, brain-inspired cognitive augmentation layers for PyTorch neural networks.
"""

from .engine import CognitiveAugEngine
from .adapters import ModuleAdapter
from .gwt import GlobalWorkspace, AttentionSelector, BroadcastEngine

__all__ = [
    "CognitiveAugEngine",
    "ModuleAdapter",
    "GlobalWorkspace",
    "AttentionSelector",
    "BroadcastEngine",
]
