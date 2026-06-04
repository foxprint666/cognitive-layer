"""
cognitive_aug/__init__.py
===============
Public API for the ``gwt`` (Global Workspace Theory) package.

Exposes all primary user-facing classes so they can be imported directly
from ``gwt`` without navigating sub-module filenames::

    from cognitive_aug import CognitiveAugEngine, GlobalWorkspace
    from cognitive_aug import VectorizedCrossAttentionSelector, TemporalSurpriseSalience
    from cognitive_aug import DecayWorkingMemory, CognitiveOutputRouter

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

# ── Glial-Inspired Learning Regulation ────────────────────────────────────────
from .glia import (
    AstrocyteManager,
    GradientSanitizerHook,
)

# ── Concept-Level Representation and Abstraction Layer ────────────────────────
from .concept import (
    ConceptLayer,
    ConceptInterventionEngine,
)

# ── Cross-Modal Cognitive Crossbar ────────────────────────────────────────────
from .crossbar import (
    CognitiveCrossbar,
    CrossbarModuleAdapter,
)

# ── Distributed State Stores (Phase v0.8 Enterprise) ──────────────────────────
from .state import (
    BaseStateStore,
    BaseReplayBufferStore,
    InMemoryStateStore,
    InMemoryReplayBufferStore,
    RedisStateStore,
    RedisReplayBufferStore,
)

# ── Asynchronous Execution Task Workers ───────────────────────────────────────
from .async_tasks import (
    start_async_worker,
    stop_async_worker,
    offloaded_enter_sleep_phase,
)

# ── Selective Hooking & Profiling ─────────────────────────────────────────────
from .profiler import (
    profile_submodules,
    register_selective_hooks,
    discover_transformer_layers,
)

# ── OpenTelemetry Observability Telemetry ──────────────────────────────────────
from .telemetry import (
    GWTTelemetryLogger,
    get_telemetry_logger,
)

# ── Autonomous Computational Neurogenesis ──────────────────────────────────────
from .neurogenesis import (
    ExtendedDendriticModuleAdapter,
    NeurogenesisManager,
    ConsolidationEngine as NeurogenesisConsolidationEngine,
    NeurogenesisReplayBuffer,
    OpenWeightsAdapterHook,
    dynamic_register_parameters,
    dynamic_deregister_parameters,
    NeurogenesisGradientSanitizerHook,
    NeurogenesisAstrocyteManager,
    TransferSalienceCalculator,
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
    # Glial-Inspired Learning Regulation
    "AstrocyteManager",
    "GradientSanitizerHook",
    # Concept-Level Representation and Abstraction Layer
    "ConceptLayer",
    "ConceptInterventionEngine",
    # Cross-Modal Cognitive Crossbar
    "CognitiveCrossbar",
    "CrossbarModuleAdapter",
    # Distributed State Stores (Phase v0.8 Enterprise)
    "BaseStateStore",
    "BaseReplayBufferStore",
    "InMemoryStateStore",
    "InMemoryReplayBufferStore",
    "RedisStateStore",
    "RedisReplayBufferStore",
    # Asynchronous Workers
    "start_async_worker",
    "stop_async_worker",
    "offloaded_enter_sleep_phase",
    # Selective Hooking & Profiling
    "profile_submodules",
    "register_selective_hooks",
    "discover_transformer_layers",
    # Observability Telemetry
    "GWTTelemetryLogger",
    "get_telemetry_logger",
    # Neurogenesis
    "ExtendedDendriticModuleAdapter",
    "NeurogenesisManager",
    "NeurogenesisConsolidationEngine",
    "NeurogenesisReplayBuffer",
    "OpenWeightsAdapterHook",
    "dynamic_register_parameters",
    "dynamic_deregister_parameters",
    "NeurogenesisGradientSanitizerHook",
    "NeurogenesisAstrocyteManager",
    "TransferSalienceCalculator",
]
