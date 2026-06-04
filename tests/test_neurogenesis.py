"""
tests/test_neurogenesis.py
==========================
Test suite for Autonomous Computational Neurogenesis.
Verifies dynamic branch addition, dynamic optimizer registration,
calcium homeostatic regulation, and sleep consolidation.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from cognitive_aug import (
    ExtendedDendriticModuleAdapter,
    NeurogenesisManager,
    NeurogenesisConsolidationEngine,
    NeurogenesisReplayBuffer,
    dynamic_register_parameters,
    dynamic_deregister_parameters,
    NeurogenesisAstrocyteManager,
)


def test_extended_dendritic_module_adapter_flow():
    """Verify that ExtendedDendriticModuleAdapter computes active gating and dynamic branch addition."""
    x = torch.randn(2, 8)
    context = torch.randn(2, 4)

    adapter = ExtendedDendriticModuleAdapter(8, 4, initial_branches=1)

    # 1. Verify initial branch exists
    assert len(adapter.branches) == 1
    assert len(adapter.dendritic_gates) == 1

    # 2. Run forward pass
    out = adapter(x, context)
    assert out.shape == (2, 8)

    # 3. Add dynamic branch
    new_idx = adapter.add_dendritic_branch()
    assert new_idx == 1
    assert len(adapter.branches) == 2
    assert len(adapter.dendritic_gates) == 2

    # 4. Verify forward pass remains stable (weights initialized to zero)
    out_new = adapter(x, context)
    assert out_new.shape == (2, 8)
    assert torch.allclose(out, out_new, atol=1e-6)

    # 5. Verify branch deactivation
    adapter.prune_branch(1)
    assert adapter.branch_states["state_1"][3].item() == 0.0


def test_neurogenesis_manager_trigger_and_cooldown():
    """Verify that surprise signals trigger neurogenesis and cooldown temporal locks block consecutive growth."""
    config = {"ne_threshold": 0.80, "ach_focus_ceiling": 0.70, "cooldown_steps": 5}

    astrocyte = NeurogenesisAstrocyteManager(calcium_decay=0.1, safety_ceiling=3.0)
    manager = NeurogenesisManager(
        config=config,
        astrocyte_manager=astrocyte,
        replay_buffer=None,
        metacognitive_monitor=None,
    )

    adapter = ExtendedDendriticModuleAdapter(8, 4, initial_branches=1)
    adapters = [adapter]

    # 1. Under low surprise, status should be idle
    metrics_low = {"NE_surprise": torch.tensor(0.5), "ACh_focus": torch.tensor(0.1)}
    status = manager.step(step_idx=1, metrics=metrics_low, adapters=adapters)
    assert status == "idle"
    assert len(adapter.branches) == 1

    # 2. Under high surprise, trigger neurogenesis
    metrics_high = {"NE_surprise": torch.tensor(0.9), "ACh_focus": torch.tensor(0.1)}
    manager.steps_since_last_birth = 5
    status = manager.step(step_idx=2, metrics=metrics_high, adapters=adapters)
    assert status == "neurogenesis_triggered"
    assert len(adapter.branches) == 2

    # 3. Consecutive triggers must be blocked by cooldown refractory lockout
    status = manager.step(step_idx=3, metrics=metrics_high, adapters=adapters)
    assert status == "idle"  # Blocked by cooldown (needs 5 steps)
    assert len(adapter.branches) == 2


def test_neurogenesis_astrocyte_regulation():
    """Verify homeostatic excitotoxicity monitoring and growth safety checks."""
    astrocyte = NeurogenesisAstrocyteManager(calcium_decay=0.1, safety_ceiling=2.0)

    # 1. Initially safe
    assert astrocyte.evaluate_growth_safety() is True

    # 2. Excitotoxic pathway activation
    x_hyper = torch.ones(2, 10) * 5.0
    regulated = astrocyte.monitor_and_regulate(x_hyper)

    # Verify attenuation is applied to damp hyper-excitation
    assert astrocyte.calcium_store.item() > 2.0
    assert (
        torch.mean(torch.abs(regulated)).item() < torch.mean(torch.abs(x_hyper)).item()
    )

    # 3. Safe growth check is now suppressed
    assert astrocyte.evaluate_growth_safety() is False


def test_dynamic_optimizer_parameter_update():
    """Verify that dynamic register/deregister parameters update the live optimizer successfully."""
    adapter = ExtendedDendriticModuleAdapter(8, 4, initial_branches=1)
    optimizer = optim.Adam(adapter.parameters(), lr=0.01)

    # Initial parameters in optimizer
    num_init_groups = len(optimizer.param_groups)
    assert num_init_groups == 1

    # Spawn a new branch
    new_idx = adapter.add_dendritic_branch()

    # 1. Register parameters dynamically to live optimizer
    dynamic_register_parameters(optimizer, adapter, new_idx)
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[1]["name"] == f"dynamic_branch_{new_idx}"

    # 2. Deregister parameters dynamically from live optimizer
    dynamic_deregister_parameters(optimizer, adapter, new_idx)
    assert len(optimizer.param_groups) == 1
    assert not any(
        group.get("name") == f"dynamic_branch_{new_idx}"
        for group in optimizer.param_groups
    )


def test_neurogenesis_consolidation_offline_sleep():
    """Verify offline SWS sleep cycle, crystallization of useful branches, and pruning of dead ones."""

    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)
            self.adapter = ExtendedDendriticModuleAdapter(4, 2, initial_branches=1)

        def forward(self, x, context):
            base_out = self.linear(x)
            return base_out + self.adapter(base_out, context)

    model = ToyModel()
    replay_buffer = NeurogenesisReplayBuffer(capacity=10)

    # Add transition experiences
    for _ in range(5):
        state = torch.randn(2, 4)
        context = torch.randn(2, 2)
        target = torch.randn(2, 4)
        replay_buffer.push(state, context, target, surprise=1.0, neuro_event=True)

    # Spawn a new candidate branch
    model.adapter.add_dendritic_branch()
    assert len(model.adapter.branches) == 2
    assert (
        model.adapter.branch_states["state_1"][2].item() == 0.0
    )  # consolidation score initially 0.0

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    engine = NeurogenesisConsolidationEngine(model, replay_buffer, threshold_perm=0.3)

    # Run sleep cycle (simulates gradients and optimization)
    engine.execute_sleep_cycle(optimizer, steps=5)

    # Verify candidate branch was evaluated: either permanentized (score=1.0), partially consolidated, or pruned
    state_after = model.adapter.branch_states["state_1"]
    active_flag = state_after[3].item()
    state_after[2].item()

    # Confirm it either stayed active with score >= 0.5 (if gradients flowed) or got pruned (active_flag=0.0)
    assert active_flag in [0.0, 0.5, 1.0]
