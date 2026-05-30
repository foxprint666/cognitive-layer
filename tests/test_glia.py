import torch
import torch.nn as nn
import pytest

from gwt import (
    CognitiveAugEngine,
    GlobalWorkspace,
    ModuleAdapter,
    DendriticModuleAdapter,
    MetacognitiveMonitor,
    AstrocyteManager,
    GradientSanitizerHook,
)


def test_glial_manager_attach_registers_hooks():
    """Verify that attaching AstrocyteManager successfully registers hooks on all adapter parameters."""
    engine = CognitiveAugEngine()
    latent_dim = 8
    workspace = GlobalWorkspace(latent_dim=latent_dim, key_dim=4)
    engine.attach_workspace(workspace)

    toy_model = nn.Linear(5, latent_dim)
    adapter = DendriticModuleAdapter(
        name="toy",
        module=toy_model,
        latent_dim=latent_dim,
        data_flow=engine.data_flow,
    )
    engine.registry.register("toy", adapter)

    manager = AstrocyteManager()
    engine.attach_glial_manager(manager)

    # Verify that the manager has registered a hook for the "toy" adapter
    assert "toy" in manager._sanitizer_hooks
    hook = manager._sanitizer_hooks["toy"]
    assert len(hook.hook_handles) > 0
    assert hook.grad_status == "Stable"

    # Verify that inspect telemetry contains Glia details
    report = engine.inspect()
    assert "toy" in report
    assert "Local Plasticity: 1.0x" in report
    assert "Grad Status: Stable" in report

    # Verify cleanup on destruction
    manager.remove_all_hooks()
    assert len(hook.hook_handles) == 0


def test_lr_scales_down_for_stable_module():
    """Verify that in high ACh (focus) + low NE (surprise) state, learning rates scale down automatically."""
    engine = CognitiveAugEngine()
    latent_dim = 8
    workspace = GlobalWorkspace(latent_dim=latent_dim, key_dim=4)
    engine.attach_workspace(workspace)

    toy_model = nn.Linear(5, latent_dim)
    adapter = ModuleAdapter(
        name="toy",
        module=toy_model,
        latent_dim=latent_dim,
        data_flow=engine.data_flow,
    )
    engine.registry.register("toy", adapter)

    monitor = MetacognitiveMonitor()
    engine.attach_neuromodulator(monitor)

    manager = AstrocyteManager(lr_lock_scale=0.5)
    engine.attach_glial_manager(manager)

    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)

    # Step once to call engine updates
    _ = toy_model(torch.randn(2, 5))
    _ = engine.step()

    # Force a high-ACh, low-NE stable focus state AFTER step
    monitor.ach = 1.0
    monitor.ne = 0.0

    # Apply glial learning rate modifications
    manager.adjust_learning_rates(optimizer)

    # Learning rate should be locked down by 0.5x
    assert pytest.approx(optimizer.param_groups[0]["lr"], abs=1e-5) == 0.05


def test_lr_scales_up_for_surprise_module():
    """Verify that in high NE (surprise) + low ACh + high salience, learning rates scale up automatically."""
    engine = CognitiveAugEngine()
    latent_dim = 8
    workspace = GlobalWorkspace(latent_dim=latent_dim, key_dim=4)
    engine.attach_workspace(workspace)

    toy_model = nn.Linear(5, latent_dim)
    adapter = ModuleAdapter(
        name="toy",
        module=toy_model,
        latent_dim=latent_dim,
        data_flow=engine.data_flow,
    )
    engine.registry.register("toy", adapter)

    monitor = MetacognitiveMonitor()
    engine.attach_neuromodulator(monitor)

    manager = AstrocyteManager(lr_unlock_scale=1.5)
    engine.attach_glial_manager(manager)

    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)

    # Step once to compute updates
    _ = toy_model(torch.randn(2, 5))
    _ = engine.step()

    # Force a high-NE surprise, low-ACh state AFTER step
    monitor.ach = 0.0
    monitor.ne = 1.0

    # Ensure module salience and its EMA are high
    engine.data_flow.update_salience("toy", 1.0)
    manager._ema_saliences["toy"] = 1.0

    # Apply glial learning rate modifications
    manager.adjust_learning_rates(optimizer)

    # Learning rate should be unlocked by 1.5x
    assert pytest.approx(optimizer.param_groups[0]["lr"], abs=1e-5) == 0.15


def test_grad_sanitizer_damps_spikes():
    """Verify that the backward gradient sanitizer hook successfully isolates and dampens gradient spikes."""
    engine = CognitiveAugEngine()
    latent_dim = 8
    workspace = GlobalWorkspace(latent_dim=latent_dim, key_dim=4)
    engine.attach_workspace(workspace)

    toy_model = nn.Linear(2, latent_dim)
    adapter = ModuleAdapter(
        name="toy",
        module=toy_model,
        latent_dim=latent_dim,
        data_flow=engine.data_flow,
    )
    engine.registry.register("toy", adapter)

    # Attach manager with max variance threshold of 2.0 stds and damping of 0.1
    manager = AstrocyteManager(max_variance_threshold=2.0, damping_factor=0.1)
    engine.attach_glial_manager(manager)

    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)

    # 1. Establish a stable running mean and variance baseline for 6 steps
    for _ in range(6):
        optimizer.zero_grad()
        inputs = torch.randn(1, 2)
        # Small stable forward outputs
        outputs = adapter.module(inputs)
        loss = outputs.sum()
        loss.backward()
        
        # Calling adjust_learning_rates resets the hook status to Stable
        manager.adjust_learning_rates(optimizer)
        optimizer.step()

    # Ensure status is Stable
    assert manager._sanitizer_hooks["toy"].grad_status == "Stable"

    # 2. Inject a massive artificial gradient spike by scaling the loss
    optimizer.zero_grad()
    inputs = torch.randn(1, 2)
    outputs = adapter.module(inputs)
    # A massive scale factor triggers the standard deviation threshold
    loss = outputs.sum() * 50000.0
    loss.backward()

    # The backward hook should have detected the spike, scaled it down, and updated status to Damped
    assert manager._sanitizer_hooks["toy"].grad_status == "Damped"
