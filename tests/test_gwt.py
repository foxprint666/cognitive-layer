import pytest
import torch
import torch.nn as nn
from cognitive_aug import (
    CognitiveAugEngine,
    GlobalWorkspace,
    AttentionSelector,
    BroadcastEngine,
)


class DummyFeatureExtractor(nn.Module):
    def __init__(self, size=10):
        super().__init__()
        self.fc = nn.Linear(5, size)

    def forward(self, x):
        return self.fc(x)


def test_attention_selector_salience():
    # Test bottom-up salience attention
    selector = AttentionSelector(key_dim=8, attention_type="salience")
    
    # 2 batches, 3 modules, 8 key_dim
    keys = torch.randn(2, 3, 8)
    weights = selector(keys)
    
    assert weights.shape == (2, 3)
    # Check that weights sum to 1 across modules
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2))


def test_attention_selector_ignition():
    # Test ignition threshold masking
    # Set threshold to 0.4. Since we have 3 modules, at least one is likely to be below 0.4
    selector = AttentionSelector(key_dim=8, attention_type="salience", ignition_threshold=0.4)
    
    # Create extreme keys so one module gets very high score and others get very low
    keys = torch.tensor([
        [[10.0, 10.0], [0.1, 0.1], [0.1, 0.1]],
        [[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]]
    ], dtype=torch.float32) # Shape: [2, 3, 2]
    
    # Re-init selector with key_dim=2
    selector = AttentionSelector(key_dim=2, attention_type="salience", ignition_threshold=0.4)
    nn.init.constant_(selector.salience_proj.weight, 1.0)
    weights = selector(keys)
    
    # Batch 1 module 1 should ignite and dominate
    assert weights[0, 0] > 0.9
    assert weights[0, 1] == 0.0
    assert weights[0, 2] == 0.0
    
    # Batch 2 all modules are identical, so they all get 1/3 ~ 0.33 which falls below threshold 0.4.
    # The selector should fallback and keep them active (prevent complete silence/nans)
    assert torch.allclose(weights[1], torch.tensor([1/3, 1/3, 1/3]))


def test_hard_selection_straight_through():
    # Verify that selection_mode='hard' produces hard selection but propagates gradients
    workspace = GlobalWorkspace(
        latent_dim=10, key_dim=4, selection_mode="hard", attention_type="salience"
    )
    
    # Setup inputs with gradients
    latent_a = torch.randn(1, 10, requires_grad=True)
    latent_b = torch.randn(1, 10, requires_grad=True)
    
    key_a = torch.randn(1, 4)
    key_b = torch.randn(1, 4)
    
    latents = {"mod_a": latent_a, "mod_b": latent_b}
    keys = {"mod_a": key_a, "mod_b": key_b}
    
    out = workspace(latents, keys)
    
    # Output should exactly match one of the inputs (hard selection)
    assert torch.allclose(out, latent_a) or torch.allclose(out, latent_b)
    
    # Let's perform a backward pass
    loss = out.sum()
    loss.backward()
    
    # Gradients must have successfully flowed back to the latent inputs
    assert latent_a.grad is not None
    assert latent_b.grad is not None
    assert latent_a.grad.sum() != 0 or latent_b.grad.sum() != 0


def test_full_engine_gwt_cycle():
    # Setup full engine integration test
    engine = CognitiveAugEngine()
    
    mod_a = DummyFeatureExtractor(size=12)
    mod_b = DummyFeatureExtractor(size=12)
    
    adapter_a = engine.register_module(name="mod_a", module=mod_a, latent_dim=12)
    adapter_b = engine.register_module(name="mod_b", module=mod_b, latent_dim=12)
    
    workspace = GlobalWorkspace(
        latent_dim=12,
        key_dim=64, # Default adapter key_dim is 64
        attention_type="key-query",
        selection_mode="soft",
        ignition_threshold=0.2
    )
    
    engine.attach_workspace(workspace)
    
    # Run forwards on both modules
    x_a = torch.randn(2, 5)
    x_b = torch.randn(2, 5)
    
    out_a = mod_a(x_a)
    out_b = mod_b(x_b)
    
    # Perform cognitive step
    broadcast_state = engine.step()
    
    assert broadcast_state.shape == (2, 12)
    
    # Verify that adapters successfully received the workspace broadcast
    assert torch.equal(adapter_a.get_last_broadcast(), broadcast_state)
    assert torch.equal(adapter_b.get_last_broadcast(), broadcast_state)
    
    # Verify gradient flow all the way to mod_a and mod_b parameters
    loss = broadcast_state.sum()
    loss.backward()
    
    # Ensure parameter weights have computed gradients
    for name, param in mod_a.named_parameters():
        assert param.grad is not None
        assert param.grad.sum() != 0
        
    for name, param in mod_b.named_parameters():
        assert param.grad is not None
        assert param.grad.sum() != 0
