# tests/test_distributed.py
import pytest
import torch
import torch.nn as nn
from cognitive_aug.engine import CognitiveAugEngine, GlobalWorkspace


def test_device_alignment_on_mismatched_inputs():
    """Verify that inputs on arbitrary CUDA/CPU devices align with the internal GWT components."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable. Skipping multi-device simulation.")

    device_a = torch.device("cuda:0")
    device_cpu = torch.device("cpu")

    # Initialize GWT engine on CPU
    engine = CognitiveAugEngine()
    workspace = GlobalWorkspace(latent_dim=4, key_dim=4).to(device_cpu)
    engine.attach_workspace(workspace)

    # Initialize model on GPU
    model = nn.Linear(4, 4).to(device_a)

    # Register model adapter
    engine.register_module(
        name="gpu_layer", module=model, latent_dim=4, device=device_a
    )

    # Execute step with inputs generated on GPU
    inputs = torch.randn(2, 4, device=device_a)
    _ = model(inputs)

    # This should evaluate, auto-cast data to CPU/GPU, and complete without device-mismatch errors
    try:
        broadcast = engine.step()
        assert broadcast.device == device_cpu or broadcast.device == device_a
    except Exception as ex:
        pytest.fail(f"Dynamic GWT device-assignment failed: {ex}")
