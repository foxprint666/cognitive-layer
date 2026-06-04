import torch
import torch.nn as nn
import pytest

from cognitive_aug import (
    CognitiveAugEngine,
    GlobalWorkspace,
    ModuleAdapter,
    DendriticModuleAdapter,
)
from cognitive_aug.neuromod import MetacognitiveMonitor, make_ascii_bar


def test_ascii_bar_formatter():
    """Verify that make_ascii_bar formats numbers correctly according to progress lengths."""
    assert make_ascii_bar(0.0) == "░░░░░░░░"
    assert make_ascii_bar(1.0) == "▇▇▇▇▇▇▇▇"
    assert make_ascii_bar(0.5) == "▇▇▇▇░░░░"
    assert make_ascii_bar(0.25) == "▇▇░░░░░░"
    assert make_ascii_bar(0.75) == "▇▇▇▇▇▇░░"


def test_chemical_monitor_curves():
    """Asserts that NE and ACh follow mathematical smoothed differential curves accurately under surprise/entropy changes."""
    # 1. Setup minimal engine
    engine = CognitiveAugEngine()
    latent_dim = 8
    workspace = GlobalWorkspace(latent_dim=latent_dim, key_dim=4, selection_mode="soft")
    engine.attach_workspace(workspace)

    toy_model = nn.Linear(5, latent_dim)
    adapter = ModuleAdapter(
        name="toy",
        module=toy_model,
        latent_dim=latent_dim,
        data_flow=engine.data_flow,
    )
    engine.registry.register("toy", adapter)

    # 2. Attach neuromodulator
    monitor = MetacognitiveMonitor(alpha_ne=0.5, alpha_ach=0.5)
    engine.attach_neuromodulator(monitor)

    # Initially chemicals should be at 0.0 or baseline
    assert monitor.ne == 0.0
    assert monitor.ach == 0.0

    # 3. Simulate first forward step to initialize cache
    noisy_input1 = torch.randn(2, 5)
    _ = toy_model(noisy_input1)
    _ = engine.step()

    # First step: no history, surprise = 0, so NE should be virtually 0.0
    assert monitor.ne < 1e-6

    # 4. Step 2: Feed a completely different input to trigger a surprise spike
    noisy_input2 = torch.randn(2, 5) * 10.0 + 5.0
    _ = toy_model(noisy_input2)
    _ = engine.step()

    # NE and ACh should both rise now
    assert monitor.ne > 0.0
    assert monitor.ach > 0.0

    prev_ne = monitor.ne

    # 5. Step 3: Feed the same input again -> surprise drops to 0.0, NE should decay
    _ = toy_model(noisy_input2)
    _ = engine.step()

    # NE should decay exactly (since surprise is exactly 0)
    # NE_3 = 0.5 * NE_2 + 0.5 * 0.0 = 0.5 * NE_2
    assert pytest.approx(monitor.ne, rel=1e-3) == 0.5 * prev_ne


def test_dynamic_threshold_modulation():
    """Feeds a sudden surprise spike into the engine and asserts that ignition_threshold drops automatically on the next step."""
    engine = CognitiveAugEngine()
    latent_dim = 8
    # Start with a high ignition threshold baseline
    workspace = GlobalWorkspace(
        latent_dim=latent_dim, key_dim=4, ignition_threshold=0.8
    )
    engine.attach_workspace(workspace)

    toy_model = nn.Linear(5, latent_dim)
    adapter = DendriticModuleAdapter(
        name="toy",
        module=toy_model,
        latent_dim=latent_dim,
        data_flow=engine.data_flow,
    )
    engine.registry.register("toy", adapter)

    monitor = MetacognitiveMonitor(alpha_ne=0.1, alpha_ach=0.9, beta=0.5, gamma=0.0)
    engine.attach_neuromodulator(monitor)

    # First step: initialize the state
    _ = toy_model(torch.ones(1, 5))
    _ = engine.step()

    initial_thresh = workspace.selector.ignition_threshold
    assert initial_thresh <= 0.8

    # Second step: feed a very different state to trigger a massive surprise spike
    _ = toy_model(torch.ones(1, 5) * -50.0)
    _ = engine.step()

    # The surprise spike should cause NE to surge, which drops ignition_threshold
    final_thresh = workspace.selector.ignition_threshold
    assert final_thresh < initial_thresh


def test_dendritic_gain_and_sparsity_modulation():
    """Asserts that high focus (ACh rise) increases dendritic spike thresholds and sharpens sigmoid gain temperature."""
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
        spike_type="nmda-threshold",
        threshold=0.4,
    )
    engine.registry.register("toy", adapter)

    # Set up a monitor with high gamma and threshold coefficients to see significant shifts
    monitor = MetacognitiveMonitor(alpha_ach=0.0, threshold_coef=0.3, temp_coef=0.5)
    engine.attach_neuromodulator(monitor)

    # Step once to initialize everything in-place
    _ = toy_model(torch.ones(1, 5))
    _ = engine.step()

    # Manually set ACh to 1.0 to simulate maximum focus and call modulation
    monitor.ach = 1.0
    monitor.adapter.apply(engine, monitor.ne, monitor.ach)

    gate = adapter.dendrite_gate
    # Check that gate threshold increased from 0.4 by 0.3 (0.4 + 0.3 * 1.0 = 0.7)
    assert pytest.approx(gate.threshold, abs=1e-5) == 0.7
    # Check that gain temperature decreased to 0.5 (1.0 - 0.5 * 1.0 = 0.5)
    assert pytest.approx(gate.gain_temperature, abs=1e-5) == 0.5


def test_inspect_dashboard_telemetry():
    """Verifies that .inspect() returns a gorgeous string containing the formatted chemical bars."""
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

    monitor = MetacognitiveMonitor()
    engine.attach_neuromodulator(monitor)

    # Set artificial chemical levels to verify ASCII telemetry rendering
    monitor.ach = 0.52
    monitor.ne = 0.21

    report = engine.inspect()

    # Verify chemical bar and diagnostic structures
    assert "COGNITIVE ENGINE DIAGNOSTIC PANEL" in report
    assert "ACh:" in report
    assert "NE:" in report
    assert "▇▇▇▇░░░░" in report  # ACh 0.52 bar
    assert "▇▇░░░░░░" in report  # NE 0.21 bar
    assert "toy" in report
