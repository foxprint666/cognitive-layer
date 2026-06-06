import torch
from cognitive_aug.iit import IITIntegrationMonitor
from cognitive_aug.engine import CognitiveAugEngine, GlobalWorkspace
from cognitive_aug.crossbar import CognitiveCrossbar


def test_iit_reducible_matrix():
    monitor = IITIntegrationMonitor()

    # Simulate a highly sparse/reducible matrix where modules don't interact
    # except with themselves (identity-like logits)
    # Shape: [B=1, N=4, N=4]
    logits = torch.eye(4).unsqueeze(0) * 10.0  # Sharp diagonal logits

    phi = monitor.calculate_phi(logits)
    # Since the full distribution and the partitioned distribution (which keeps diagonals)
    # are almost identical for an identity matrix, the Cosine distance should be near 0.
    assert phi < 1e-3, f"Expected near 0 Phi for reducible matrix, got {phi}"


def test_iit_dense_matrix():
    monitor = IITIntegrationMonitor()

    # Simulate a dense cross-modal matrix where all modules interact
    logits = torch.ones(1, 4, 4) * 5.0

    phi = monitor.calculate_phi(logits)
    # The partitioned matrix will zero out off-diagonals, creating a vastly different
    # probability distribution than the uniform dense matrix. Phi should be high.
    assert phi > 0.1, f"Expected high Phi for dense matrix, got {phi}"


def test_iit_1d_star_topology():
    monitor = IITIntegrationMonitor()

    # Simulate a 1D star topology output [B=1, N=6]
    logits = torch.randn(1, 6)

    phi = monitor.calculate_phi(logits)
    # Phi should be calculable and >= 0
    assert phi >= 0.0


def test_engine_integration():
    engine = CognitiveAugEngine()
    monitor = IITIntegrationMonitor()
    engine.attach_iit_monitor(monitor)

    assert engine.iit_monitor is not None

    # Mock workspace with last_logits
    ws = GlobalWorkspace(latent_dim=8)
    # We must run a fake step or inject last_logits directly
    ws.last_logits = torch.randn(2, 4)
    engine.workspace = ws

    # Mock crossbar with last_logits
    cb = CognitiveCrossbar(slot_dim=8, num_slots=4)
    cb.last_logits = torch.randn(2, 4, 4)
    engine.crossbar = cb

    # Verify inspect doesn't crash and includes Phi
    engine.latest_phi = 0.42
    dashboard = engine.inspect()
    assert "System Integration (Φ):" in dashboard
    assert "0.42" in dashboard
