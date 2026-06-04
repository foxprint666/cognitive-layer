import pytest
import torch
import torch.nn as nn
from cognitive_aug import CognitiveAugEngine, GlobalWorkspace
from cognitive_aug.engine import DataFlowManager
from cognitive_aug.dendrite import (
    ActiveDendriteGate,
    DendriticModuleAdapter,
    get_dendritic_status,
)


class ToyNet(nn.Module):
    def __init__(self, in_dim=10, out_dim=16):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


class NestedToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.sub = ToyNet(in_dim=5, out_dim=12)

    def forward(self, x):
        return self.sub(x)


def test_active_dendrite_gate_modulatory_gain():
    """Verify that modulatory-gain sigmoidal gating scales features and records activations."""
    feedforward_dim = 8
    context_dim = 4
    num_branches = 3

    gate = ActiveDendriteGate(
        feedforward_dim=feedforward_dim,
        context_dim=context_dim,
        num_branches=num_branches,
        spike_type="modulatory-gain",
    )

    x = torch.ones(2, feedforward_dim)
    context = torch.ones(2, context_dim)

    out = gate(x, context)
    assert out.shape == (2, feedforward_dim)

    # Gating should be in [0, 1] since it uses sigmoid, scaling features element-wise
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)

    # Telemetry should be saved
    assert gate.latest_branch_activations is not None
    assert gate.latest_branch_activations.shape == (2, num_branches, feedforward_dim)


def test_active_dendrite_gate_nmda_threshold():
    """Verify nmda-threshold spikes, mutes inactive branches, and supports STE gradient flow."""
    feedforward_dim = 8
    context_dim = 4
    num_branches = 2
    threshold = 0.6

    gate = ActiveDendriteGate(
        feedforward_dim=feedforward_dim,
        context_dim=context_dim,
        num_branches=num_branches,
        spike_type="nmda-threshold",
        threshold=threshold,
    )

    x = torch.ones(1, feedforward_dim)
    # Set weights to extreme values to force one branch to exceed threshold and another to be below
    with torch.no_grad():
        # First branch weight high: high depolarization
        gate.context_proj.weight[:feedforward_dim].fill_(5.0)
        gate.context_proj.bias[:feedforward_dim].fill_(5.0)
        # Second branch weight very negative: low depolarization
        gate.context_proj.weight[feedforward_dim:].fill_(-5.0)
        gate.context_proj.bias[feedforward_dim:].fill_(-5.0)

    context = torch.ones(1, context_dim)
    gate(x, context)

    # Verify that the second branch is completely zeroed out (muted)
    # Since only branch 1 spiked, branch 2 gates should be 0.0
    # Our recorded branch activations should show branch 1 highly active, branch 2 completely zeroed
    activations = gate.latest_branch_activations
    assert activations is not None

    # Branch 1 (index 0) spiked, so it is > 0
    assert torch.all(activations[0, 0] > 0.0)
    # Branch 2 (index 1) did not spike, so it is exactly 0
    assert torch.all(activations[0, 1] == 0.0)

    # Test STE gradient flow
    x_grad = torch.ones(1, feedforward_dim, requires_grad=True)
    context_grad = torch.ones(1, context_dim, requires_grad=True)

    out_grad = gate(x_grad, context_grad)
    loss = out_grad.sum()
    loss.backward()

    # Gradients should flow cleanly back to the context projection layer and input
    assert context_grad.grad is not None
    assert x_grad.grad is not None
    assert gate.context_proj.weight.grad is not None


def test_shape_agnostic_vectorization():
    """Verify that element-wise gating is vectorized and shape-agnostic (supports batched + sequence data)."""
    feedforward_dim = 16
    context_dim = 8

    gate = ActiveDendriteGate(
        feedforward_dim=feedforward_dim,
        context_dim=context_dim,
        num_branches=4,
        spike_type="modulatory-gain",
    )

    # Batch representation: [B, D]
    x_batched = torch.randn(4, feedforward_dim)
    context = torch.randn(4, context_dim)
    out_batched = gate(x_batched, context)
    assert out_batched.shape == (4, feedforward_dim)

    # Sequence representation: [B, S, D]
    x_seq = torch.randn(4, 5, feedforward_dim)
    out_seq = gate(x_seq, context)
    assert out_seq.shape == (4, 5, feedforward_dim)


def test_memory_safety_detached_tracking():
    """Ensure that historical tracking tensors use .detach() and do not hold backward graphs."""
    feedforward_dim = 6
    context_dim = 4

    gate = ActiveDendriteGate(
        feedforward_dim=feedforward_dim,
        context_dim=context_dim,
        num_branches=2,
        spike_type="modulatory-gain",
    )

    x = torch.randn(2, feedforward_dim, requires_grad=True)
    context = torch.randn(2, context_dim, requires_grad=True)

    out = gate(x, context)
    loss = out.sum()
    loss.backward()

    # The tracking buffer must not have gradient history or require grads
    assert gate.latest_branch_activations is not None
    assert not gate.latest_branch_activations.requires_grad
    assert gate.latest_branch_activations.grad_fn is None


def test_dendritic_adapter_auto_dimension_detection():
    """Verify that DendriticModuleAdapter statically and dynamically detects model dimensions."""
    data_flow = DataFlowManager()

    # 1. Static inspection of nn.Linear
    linear_toy = ToyNet(in_dim=5, out_dim=12)
    adapter1 = DendriticModuleAdapter(
        name="linear_toy",
        module=linear_toy,
        latent_dim=12,
        data_flow=data_flow,
    )
    assert adapter1.dendrite_gate is not None
    assert adapter1.dendrite_gate.feedforward_dim == 12

    # 2. Static inspection of a nested model (recursively finding last layer)
    nested_toy = NestedToyNet()
    adapter2 = DendriticModuleAdapter(
        name="nested_toy",
        module=nested_toy,
        latent_dim=12,
        data_flow=data_flow,
    )
    assert adapter2.dendrite_gate is not None
    assert adapter2.dendrite_gate.feedforward_dim == 12

    # 3. Dynamic lazy-initialization fallback for uninspectable models
    class UninspectableNet(nn.Module):
        def forward(self, x):
            return x * 2.0  # Output dimension cannot be statically inspected

        # Prevent children or standard attributes from being found
        def __getattr__(self, name):
            if name in [
                "out_features",
                "out_channels",
                "hidden_size",
                "output_dim",
                "latent_dim",
                "d_model",
            ]:
                raise AttributeError
            return super().__getattr__(name)

    uninspectable = UninspectableNet()
    adapter3 = DendriticModuleAdapter(
        name="uninspectable",
        module=uninspectable,
        latent_dim=10,
        data_flow=data_flow,
    )
    # Statically should be None
    assert adapter3.dendrite_gate is None

    # Run forward pass; dynamic initialization should trigger
    dummy_input = torch.randn(3, 10)
    uninspectable(dummy_input)  # runs via standard PyTorch forward
    out_gated = adapter3.module(dummy_input)  # runs hook, builds gate dynamically

    assert adapter3.dendrite_gate is not None
    assert adapter3.dendrite_gate.feedforward_dim == 10
    assert out_gated.shape == (3, 10)


def test_dendritic_adapter_hook_pipeline():
    """Verify that running a model hooks up active dendrites and automatically routes GWT context."""
    engine = CognitiveAugEngine()
    model = ToyNet(in_dim=5, out_dim=8)

    # Standard registration with minimal signature
    adapter = DendriticModuleAdapter(
        name="toy",
        module=model,
        latent_dim=8,
        data_flow=engine.data_flow,
        num_branches=4,
        spike_type="nmda-threshold",
    )
    engine.registry.register("toy", adapter)

    # Attach a workspace to test global broadcast distribution
    workspace = GlobalWorkspace(latent_dim=8)
    engine.attach_workspace(workspace)

    # 1. First forward pass (generates features and routes to engine data flow)
    x = torch.randn(2, 5)
    _ = model(x)

    # Buffer should be updated
    assert engine.data_flow.get_buffer("toy").shape == (2, 8)

    # 2. Step the engine to compute Global Workspace Broadcast
    broadcast_state = engine.step()
    assert broadcast_state.shape == (2, 8)

    # Verify the broadcast was delivered back to the adapter buffer
    assert torch.equal(adapter.get_last_broadcast(), broadcast_state)

    # 3. Second forward pass: ActiveDendriteGate should now apply the active context gating
    gated_out = model(x)
    assert gated_out.shape == (2, 8)

    # Verify telemetry shows activation has occurred
    status = get_dendritic_status(adapter)
    assert "active_pct" in status
    assert "muted_pct" in status
    assert status["active_pct"] + status["muted_pct"] == pytest.approx(100.0)


def test_telemetry_integration():
    """Verify get_dendritic_status outputs accurate active/muted statistics."""
    # Create simple gate
    gate = ActiveDendriteGate(
        feedforward_dim=4,
        context_dim=2,
        num_branches=2,
        spike_type="nmda-threshold",
        threshold=0.5,
    )

    # Status before forward pass should be default muted
    status = get_dendritic_status(gate)
    assert status["active_pct"] == 0.0
    assert status["muted_pct"] == 100.0

    # Populating latest branch activations manually to test telemetry calculation
    # Branch 1 (index 0) fully active (1.0 >= 0.5) -> Active
    # Branch 2 (index 1) fully muted (0.0 < 0.5) -> Muted
    # Mean active percentage should be exactly 50.0%
    gate.latest_branch_activations = torch.tensor(
        [[[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]]
    )

    status = get_dendritic_status(gate)
    assert status["active_pct"] == 50.0
    assert status["muted_pct"] == 50.0
