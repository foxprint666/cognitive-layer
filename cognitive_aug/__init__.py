"""
Cognitive Augmentation Layers (cognitive_aug)

Modular, brain-inspired cognitive augmentation layers for PyTorch neural networks.
Shim package wrapping the optimized 'gwt' package for seamless backward compatibility.
"""

from gwt import (
    CognitiveAugEngine,
    ModuleAdapter,
    GlobalWorkspace,
    AttentionSelector,
    BroadcastEngine,
    BaseSelector,
    CosineSimilaritySelector,
    VectorizedCrossAttentionSelector,
    EfficientGumbelSoftmaxSelector,
    global_pool_latent,
    BaseSalience,
    MagnitudeSalience,
    EntropySalience,
    TemporalSurpriseSalience,
    DecayWorkingMemory,
    CognitiveOutputRouter,
    ActiveDendriteGate,
    DendriticModuleAdapter,
    get_dendritic_status,
    CognitiveReplayBuffer,
    ConsolidationEngine,
    prune_dendrites,
    MetacognitiveMonitor,
    DynamicThresholdAdapter,
)

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
    "ActiveDendriteGate",
    "DendriticModuleAdapter",
    "get_dendritic_status",
    "CognitiveReplayBuffer",
    "ConsolidationEngine",
    "prune_dendrites",
    "MetacognitiveMonitor",
    "DynamicThresholdAdapter",
]


