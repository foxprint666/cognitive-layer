import torch
import torch.nn as nn
import pytest

from cognitive_aug import (
    CognitiveAugEngine,
    GlobalWorkspace,
)
from cognitive_aug.crossbar import CognitiveCrossbar, CrossbarModuleAdapter


def test_crossbar_vectorized_parallel_routing():
    """Verify that CognitiveCrossbar routes stacked cross-modal inputs correctly and records routing weights."""
    slot_dim = 8
    num_slots = 3
    batch_size = 2

    crossbar = CognitiveCrossbar(slot_dim=slot_dim, num_slots=num_slots)
    
    # Input shape: [B, num_slots, slot_dim]
    x = torch.randn(batch_size, num_slots, slot_dim)
    out = crossbar(x)

    # Output shape should match inputs
    assert out.shape == (batch_size, num_slots, slot_dim)
    
    # Verify weights are captured
    assert crossbar.last_weights is not None
    assert crossbar.last_weights.shape == (batch_size, num_slots, num_slots)


def test_crossbar_dynamic_binding_and_gradient_flow():
    """
    Verify that backpropagating a loss on a single slot propagates gradients cleanly
    across all input slots without leaks or explosions.
    """
    slot_dim = 6
    num_slots = 3
    batch_size = 2

    crossbar = CognitiveCrossbar(slot_dim=slot_dim, num_slots=num_slots)
    x = torch.randn(batch_size, num_slots, slot_dim, requires_grad=True)

    out = crossbar(x)

    # Compute loss on target slot 1 only
    loss = out[:, 1].sum()
    loss.backward()

    # Verify that gradients are propagated across all input slot lines
    # because slot 1 integrates contextual broadcasts from slot 0, 1, and 2!
    assert x.grad is not None
    assert torch.any(x.grad[:, 0] != 0.0)  # Gradient flowed to Slot 0
    assert torch.any(x.grad[:, 1] != 0.0)  # Gradient flowed to Slot 1
    assert torch.any(x.grad[:, 2] != 0.0)  # Gradient flowed to Slot 2
    
    # Verify no NaN or infinite exploding gradients
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()


def test_crossbar_adapter_write_and_distribute():
    """Verify that adapters write slot data correctly and engine step distributes routed context."""
    engine = CognitiveAugEngine()
    latent_dim = 8
    workspace = GlobalWorkspace(latent_dim=latent_dim, key_dim=4)
    engine.attach_workspace(workspace)

    # Setup Crossbar with named slots
    slot_names = ["Vision", "Text"]
    crossbar = CognitiveCrossbar(slot_dim=latent_dim, num_slots=2, slot_names=slot_names)
    engine.attach_crossbar(crossbar)

    # Create and register standard layers wrapped with CrossbarModuleAdapter
    vis_layer = nn.Linear(5, latent_dim)
    txt_layer = nn.Linear(5, latent_dim)

    vis_adapter = CrossbarModuleAdapter("vision", vis_layer, latent_dim, engine.data_flow, slot_idx=0)
    txt_adapter = CrossbarModuleAdapter("text", txt_layer, latent_dim, engine.data_flow, slot_idx=1)

    engine.registry.register("vision", vis_adapter)
    engine.registry.register("text", txt_adapter)

    # Attach references
    vis_adapter.engine = engine
    txt_adapter.engine = engine

    # Run forward passes
    inputs_vis = torch.randn(2, 5)
    inputs_txt = torch.randn(2, 5)

    _ = vis_layer(inputs_vis)
    _ = txt_layer(inputs_txt)

    # Step the GWT engine
    _ = engine.step()

    # 1. Verify that adapters correctly wrote their latents to the crossbar stacked buffer
    assert crossbar._stacked_latents is not None
    assert crossbar._stacked_latents.shape == (2, 2, latent_dim)

    # 2. Verify that dynamic crossbar routing distributed context back to last_broadcast
    vis_broadcast = vis_adapter.get_last_broadcast()
    txt_broadcast = txt_adapter.get_last_broadcast()

    assert vis_broadcast is not None
    assert txt_broadcast is not None
    assert vis_broadcast.shape == (2, latent_dim)
    assert txt_broadcast.shape == (2, latent_dim)

    # 3. Verify inspect connectivity map contains the text-based connecting lines
    report = engine.inspect()
    assert "Crossbar Connectivity Map:" in report
    assert "Vision" in report
    assert "Text" in report
    assert "──>" in report
