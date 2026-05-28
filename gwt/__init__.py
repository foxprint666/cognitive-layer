"""
gwt/__init__.py
===============
Public API for the ``gwt`` (Global Workspace Theory) package.

Exposes all primary user-facing classes so they can be imported directly
from ``gwt`` without navigating sub-module filenames::

    from gwt import CognitiveAugEngine, GlobalWorkspace
    from gwt import VectorizedCrossAttentionSelector, TemporalSurpriseSalience
    from gwt import DecayWorkingMemory, CognitiveOutputRouter

Internal relative import map
-----------------------------
.engine    -> CognitiveAugEngine, ModuleAdapter, GlobalWorkspace,
              AttentionSelector, BroadcastEngine
.selectors -> BaseSelector, CosineSimilaritySelector,
              VectorizedCrossAttentionSelector, EfficientGumbelSoftmaxSelector
.salience  -> global_pool_latent, BaseSalience, MagnitudeSalience,
              EntropySalience, TemporalSurpriseSalience
.memory    -> DecayWorkingMemory
.routing   -> CognitiveOutputRouter
"""

# ── Core engine & workspace infrastructure ────────────────────────────────────
from .engine import (
    CognitiveAugEngine,
    ModuleAdapter,
    GlobalWorkspace,
    AttentionSelector,
    BroadcastEngine,
)

# ── Plug-and-play attention selectors ─────────────────────────────────────────
from .selectors import (
    BaseSelector,
    CosineSimilaritySelector,
    VectorizedCrossAttentionSelector,
    EfficientGumbelSoftmaxSelector,
)

# ── Low-overhead salience metrics ─────────────────────────────────────────────
from .salience import (
    global_pool_latent,
    BaseSalience,
    MagnitudeSalience,
    EntropySalience,
    TemporalSurpriseSalience,
)

# ── Stateful working memory ───────────────────────────────────────────────────
from .memory import DecayWorkingMemory

# ── Parallel downstream output routing ───────────────────────────────────────
from .routing import CognitiveOutputRouter

# ── Active Dendritic Gating ───────────────────────────────────────────────────
from .dendrite import (
    ActiveDendriteGate,
    DendriticModuleAdapter,
    get_dendritic_status,
)

# ── Sleep & Memory Consolidation ──────────────────────────────────────────────
from .sleep import (
    CognitiveReplayBuffer,
    ConsolidationEngine,
    prune_dendrites,
)

# ── Metacognitive Neuromodulation ─────────────────────────────────────────────
from .neuromod import (
    MetacognitiveMonitor,
    DynamicThresholdAdapter,
)

__all__ = [
    # Engine & workspace
    "CognitiveAugEngine",
    "ModuleAdapter",
    "GlobalWorkspace",
    "AttentionSelector",
    "BroadcastEngine",
    # Selectors
    "BaseSelector",
    "CosineSimilaritySelector",
    "VectorizedCrossAttentionSelector",
    "EfficientGumbelSoftmaxSelector",
    # Salience
    "global_pool_latent",
    "BaseSalience",
    "MagnitudeSalience",
    "EntropySalience",
    "TemporalSurpriseSalience",
    # Memory & routing
    "DecayWorkingMemory",
    "CognitiveOutputRouter",
    # Dendritic computation
    "ActiveDendriteGate",
    "DendriticModuleAdapter",
    "get_dendritic_status",
    # Sleep & Memory Consolidation
    "CognitiveReplayBuffer",
    "ConsolidationEngine",
    "prune_dendrites",
    # Metacognitive Neuromodulation
    "MetacognitiveMonitor",
    "DynamicThresholdAdapter",
]


