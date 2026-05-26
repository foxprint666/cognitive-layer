import pytest
import torch
import torch.nn as nn
from gwt import (
    CognitiveAugEngine,
    ModuleAdapter,
    GlobalWorkspace,
    AttentionSelector,
    BroadcastEngine,
    CosineSimilaritySelector,
    VectorizedCrossAttentionSelector,
    EfficientGumbelSoftmaxSelector,
    global_pool_latent,
    MagnitudeSalience,
    EntropySalience,
    TemporalSurpriseSalience,
    DecayWorkingMemory,
    CognitiveOutputRouter,
)


class SimpleModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 8)

    def forward(self, x):
        return self.fc(x)


def test_gwt_direct_engine_and_workspace():
    engine = CognitiveAugEngine()
    module = SimpleModule()
    
    # Register with defaults
    adapter = engine.register_module("simple", module, latent_dim=8)
    assert adapter.name == "simple"
    assert adapter.latent_dim == 8
    
    # Create workspace with matching latent space but smaller key_dim to test key auto-alignment
    workspace = GlobalWorkspace(
        latent_dim=8,
        key_dim=16,
        attention_type="key-query"
    )
    
    engine.attach_workspace(workspace)
    
    # Verify that the adapter's key projection was dynamically aligned from 64 to 16
    assert adapter.key_dim == 16
    assert adapter.key_proj.out_features == 16
    
    # Run a forward pass
    x = torch.randn(2, 4)
    _ = module(x)
    
    # Step the engine
    broadcast = engine.step()
    assert broadcast.shape == (2, 8)
    assert torch.allclose(adapter.get_last_broadcast(), broadcast)


def test_gwt_direct_components():
    # Test selectors
    keys = torch.randn(2, 3, 16)
    query = torch.randn(2, 16)
    
    selector = VectorizedCrossAttentionSelector(key_dim=16, num_heads=2)
    weights = selector(keys, query)
    assert weights.shape == (2, 3)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2))
    
    # Test salience
    salience = MagnitudeSalience(ignition_threshold=1.0)
    states = {"simple": torch.randn(2, 8)}
    scores = salience(states)
    assert scores.shape == (2, 1)
    
    # Test memory
    memory = DecayWorkingMemory(latent_dim=8, decay_rate=0.5, blend_weight=0.8)
    workspace_in = torch.randn(2, 8)
    ignited = torch.ones(2, 1)
    out = memory(workspace_in, ignited)
    assert out.shape == (2, 8)
    
    # Test router
    router = CognitiveOutputRouter(latent_dim=8, output_specs={"head_a": 5})
    routed = router(workspace_in)
    assert routed["head_a"].shape == (2, 5)
