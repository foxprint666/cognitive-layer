import pytest
import torch
import torch.nn as nn
from cognitive_aug import CognitiveAugEngine, GlobalWorkspace
from cognitive_aug.dendrite import (
    ActiveDendriteGate,
    DendriticModuleAdapter,
    get_dendritic_status,
)
from cognitive_aug.sleep import (
    CognitiveReplayBuffer,
    ConsolidationEngine,
    prune_dendrites,
)


class ToyNet(nn.Module):
    def __init__(self, in_dim=5, out_dim=8):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


def test_replay_buffer_bounded_and_prioritized():
    """Verify that CognitiveReplayBuffer enforces capacity limits, stores detached elements, and samples based on salience."""
    buffer = CognitiveReplayBuffer(max_size=3)
    
    latent_states = {"mod": torch.randn(2, 4, requires_grad=True)}
    context = torch.randn(2, 4, requires_grad=True)
    
    # 1. Verify detached storing to prevent memory leaks
    buffer.add_trace(latent_states, context, salience=1.5)
    trace = buffer.buffer[0]
    
    assert not trace["latent_states"]["mod"].requires_grad
    assert not trace["context"].requires_grad
    assert trace["latent_states"]["mod"].grad_fn is None
    assert trace["context"].grad_fn is None

    # 2. Verify bounding size limits by adding more items
    buffer.add_trace(latent_states, context, salience=0.5)
    buffer.add_trace(latent_states, context, salience=2.5)
    
    # Current size is 3 (max capacity)
    assert len(buffer) == 3
    
    # Add a 4th trace with a high salience. It should evict the lowest salience trace (0.5)
    buffer.add_trace(latent_states, context, salience=3.0)
    assert len(buffer) == 3
    
    # Verify that the trace with salience=0.5 was evicted
    saliences = [t["salience"] for t in buffer.buffer]
    assert 0.5 not in saliences
    assert 1.5 in saliences
    assert 2.5 in saliences
    assert 3.0 in saliences

    # 3. Verify prioritized sampling (high-salience trace should be sampled more often)
    # Clear and populate with very distinct saliences
    buffer.clear()
    buffer.add_trace(latent_states, context, salience=0.0)      # very low probability
    buffer.add_trace(latent_states, context, salience=100.0)    # very high probability
    
    samples = buffer.sample_batch(100)
    salience_samples = [s["salience"] for s in samples]
    
    # The high salience item should dominate the samples
    assert salience_samples.count(100.0) > 95


def test_sleep_consolidation_reduces_loss():
    """Verify that ConsolidationEngine updates workspace weights and reduces GWT reconstruction/stability loss."""
    engine = CognitiveAugEngine()
    workspace = GlobalWorkspace(latent_dim=8)
    engine.attach_workspace(workspace)
    
    model = ToyNet(in_dim=5, out_dim=8)
    adapter = DendriticModuleAdapter(
        name="toy",
        module=model,
        latent_dim=8,
        data_flow=engine.data_flow,
        num_branches=2,
        spike_type="modulatory-gain"
    )
    engine.registry.register("toy", adapter)

    # Attach replay buffer and populate with dummy waking traces
    buffer = CognitiveReplayBuffer(max_size=10)
    
    # Setup some waking states
    x = torch.randn(2, 5)
    _ = model(x)  # populates adapter buffer and triggers gate initialization
    
    # Prepare dummy waking traces
    latent_states = {"toy": torch.randn(2, 8, requires_grad=True)}
    context = torch.randn(2, 8, requires_grad=True)
    for _ in range(5):
        buffer.add_trace(latent_states, context, salience=1.0)

    consolidator = ConsolidationEngine(engine, buffer)
    
    # Run offline sleep consolidation steps
    telemetry = consolidator.sleep_cycle(steps=20, learning_rate=0.01, batch_size=4)
    
    # Confirm that memory traces were successfully replayed
    assert telemetry["memory_traces_replayed"] > 0
    # Confirm that reconstruction/stability loss decreased
    assert telemetry["loss_delta"] > 0.0
    assert telemetry["final_loss"] < telemetry["initial_loss"]


def test_dendritic_pruning_homeostasis():
    """Verify that prune_dendrites zero-out weak branches, severs backprop pathways, and locks out gradient recalculation."""
    feedforward_dim = 6
    context_dim = 4
    num_branches = 3
    pruning_threshold = 0.4
    
    gate = ActiveDendriteGate(
        feedforward_dim=feedforward_dim,
        context_dim=context_dim,
        num_branches=num_branches,
        spike_type="modulatory-gain"
    )
    
    # Manually configure latest_branch_activations to simulate waking history:
    # - Branch 0 (index 0) average activation = 0.1 (< 0.4) -> Should be pruned!
    # - Branch 1 (index 1) average activation = 0.8 (>= 0.4) -> Should remain active!
    # - Branch 2 (index 2) average activation = 0.9 (>= 0.4) -> Should remain active!
    # Shape: [B=1, num_branches=3, feedforward_dim=6]
    gate.latest_branch_activations = torch.tensor([[[0.1]*6, [0.8]*6, [0.9]*6]])

    # Verify initially all branches have active parameter slices
    assert not torch.all(gate.context_proj.weight[:feedforward_dim] == 0.0)
    
    # Trigger structural pruning
    pruned_count = prune_dendrites(gate, pruning_threshold=pruning_threshold)
    assert pruned_count == 1  # Exactly 1 branch (index 0) should be pruned

    # Verify branch 0 weight and bias are zeroed out
    assert torch.all(gate.context_proj.weight[:feedforward_dim] == 0.0)
    assert torch.all(gate.pruning_mask[:feedforward_dim] == 0.0)
    if gate.context_proj.bias is not None:
        assert torch.all(gate.context_proj.bias[:feedforward_dim] == 0.0)
        assert torch.all(gate.bias_pruning_mask[:feedforward_dim] == 0.0)

    # Verify branch 1 & 2 weight slices are untouched/active
    assert not torch.all(gate.context_proj.weight[feedforward_dim:] == 0.0)
    assert torch.all(gate.pruning_mask[feedforward_dim:] == 1.0)

    # Verify permanently severed backpropagation gradient lockout:
    # 1. Run a backward pass with active gradients
    x = torch.ones(1, feedforward_dim)
    context = torch.ones(1, context_dim)
    
    out = gate(x, context)
    loss = out.sum()
    loss.backward()
    
    # 2. Check weight gradients
    grad_weight = gate.context_proj.weight.grad
    grad_bias = gate.context_proj.bias.grad
    
    assert grad_weight is not None
    # The gradient for the pruned branch (index 0) MUST be exactly zeroed out by the hook!
    assert torch.all(grad_weight[:feedforward_dim] == 0.0)
    
    # Gradients for active branches (indexes 1 and 2) should flow normally
    assert not torch.all(grad_weight[feedforward_dim:] == 0.0)
    
    if grad_bias is not None:
        assert torch.all(grad_bias[:feedforward_dim] == 0.0)
        assert not torch.all(grad_bias[feedforward_dim:] == 0.0)


def test_engine_integration_sleep_flow():
    """Verify that main CognitiveAugEngine enter_sleep_phase workflow triggers recording, offline consolidation, pruning, and buffer flush."""
    engine = CognitiveAugEngine()
    workspace = GlobalWorkspace(latent_dim=6)
    engine.attach_workspace(workspace)
    
    model = ToyNet(in_dim=3, out_dim=6)
    adapter = DendriticModuleAdapter(
        name="toy",
        module=model,
        latent_dim=6,
        data_flow=engine.data_flow,
        num_branches=2,
        spike_type="modulatory-gain"
    )
    engine.registry.register("toy", adapter)

    # Attach episodic buffer to engine
    buffer = CognitiveReplayBuffer(max_size=10)
    engine.attach_replay_buffer(buffer)

    # 1. Waking mode step: running model and stepping engine should automatically record traces
    x = torch.randn(2, 3)
    _ = model(x)
    _ = engine.step()
    
    # Trace should be captured automatically in buffer
    assert len(buffer) == 1
    
    # Run a few more steps to populate memory
    for _ in range(4):
        _ = model(x)
        _ = engine.step()
        
    assert len(buffer) == 5

    # Configure dendritic gate activations to trigger pruning during sleep
    adapter.dendrite_gate.latest_branch_activations = torch.tensor([[[0.02]*6, [0.85]*6]])

    # 2. Enter Sleep Phase: pause online and consolidate!
    telemetry = engine.enter_sleep_phase(steps=10, learning_rate=0.01, pruning_threshold=0.05)
    
    assert telemetry["memory_traces_replayed"] > 0
    assert telemetry["dendritic_branches_pruned"] == 1
    assert "loss_delta" in telemetry

    # 3. Verify buffer and engine cache flush
    assert len(buffer) == 0
    assert len(engine.data_flow.list_buffers()) == 0
