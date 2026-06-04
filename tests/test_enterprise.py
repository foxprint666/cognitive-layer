"""
tests/test_enterprise.py
========================
Enterprise-Grade Scalability, Asynchrony, and Resilience Refactoring Test Suite.

Verifies:
1. Hook try-catch fault tolerance and GWT step fail-safe graceful bypass.
2. PyTorch tensor binary serialization on simulated RedisStateStore.
3. Micro-profiling and TTFT selective hook registration.
4. Local background thread-safe asynchronous consolidation loops.
5. OpenTelemetry-compatible structured JSON telemetry logging.
"""

import io
import json
import time
from typing import Any, Dict, List, Optional

import pytest
import torch
import torch.nn as nn

from cognitive_aug import (
    CognitiveAugEngine,
    GlobalWorkspace,
    ModuleAdapter,
    DendriticModuleAdapter,
    CognitiveReplayBuffer,
    InMemoryStateStore,
    InMemoryReplayBufferStore,
    start_async_worker,
    stop_async_worker,
    offloaded_enter_sleep_phase,
    profile_submodules,
    register_selective_hooks,
    GWTTelemetryLogger,
    get_telemetry_logger,
)
from cognitive_aug.state import _serialize_tensor, _deserialize_tensor


# ── Mock Infrastructure ───────────────────────────────────────────────────────

class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_a = nn.Linear(8, 4)
        self.layer_b = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Intentionally double branch
        out_a = self.layer_a(x)
        out_b = self.layer_b(x)
        return out_a + out_b


# ── 1. Fault Tolerance & Graceful Fallbacks ───────────────────────────────────

def test_fault_tolerant_bypass() -> None:
    """Verify that forward hooks and step loops intercept exceptions, log them, and bypass GWT gracefully."""
    engine = CognitiveAugEngine()
    workspace = GlobalWorkspace(latent_dim=4, key_dim=8)
    engine.attach_workspace(workspace)

    model = ToyModel()
    
    # Register standard module adapter
    adapter = ModuleAdapter(
        name="toy",
        module=model.layer_a,
        latent_dim=4,
        data_flow=engine.data_flow,
    )
    engine.registry.register("toy", adapter)

    # 1. Test Forward Hook Resilience:
    # We forcefully trigger a forward pass with a bad input shape that causes nn.Linear to crash.
    # PyTorch should raise a RuntimeError inside the forward hook, but the adapter hook must catch it,
    # log it to GWT telemetry, and return standard outputs without throwing a system crash!
    try:
        # Send bad inputs (size 3 matches 8 in out-features but inputs are 3 -> shape error)
        bad_input = torch.randn(2, 3)
        # Even though the forward hook executes, the try-except wrapper catches the crash!
        # Note: the linear layer itself might throw an error because of matrix multiplication,
        # so let's mock a crash inside the hook's serialization or GWT buffer to test GWT isolation specifically.
        # We replace update_buffer with a method that raises an exception!
        def crash_update(name: str, tensor: torch.Tensor):
            raise RuntimeError("Database connection timed out or GPU OOM.")
        
        engine.data_flow._state_store.update_buffer = crash_update
        
        # This forward pass triggers the hook, which triggers the crashed update_buffer.
        # It must complete successfully without throwing the RuntimeError!
        x_good = torch.randn(2, 8)
        outputs = model.layer_a(x_good)
        assert outputs is not None
        assert outputs.shape == (2, 4)
        # Restore original update_buffer method before leaving try block
        engine.data_flow._state_store.update_buffer = engine.data_flow._state_store.__class__.update_buffer.__get__(
            engine.data_flow._state_store, engine.data_flow._state_store.__class__
        )
    except Exception as e:
        pytest.fail(f"ModuleAdapter forward hook did not isolate the runtime crash: {e}")

    # 2. Test Step Loop Resilience:
    # If GWT engine.step() crashes (e.g. workspace selection mode error), it must capture it,
    # log a structured telemetry warning, and return a clean fallback zero-filled broadcast.
    def crash_workspace(*args, **kwargs):
        raise ValueError("CUDA out of memory exception.")
        
    engine.workspace = crash_workspace

    # Ensure data flow has at least one buffer to check batch size
    engine.data_flow.update_buffer("toy", torch.randn(3, 4))
    
    fallback_broadcast = engine.step()
    assert fallback_broadcast is not None
    # Must gracefully return a zero vector matching the active batch size (3) and GWT latent dim (4)
    assert fallback_broadcast.shape == (3, 4)
    assert torch.all(fallback_broadcast == 0.0)


# ── 2. Distributed State Serialization ───────────────────────────────────────

def test_redis_mock_state_store() -> None:
    """Verify PyTorch tensor serialization and deserialization used in our RedisStateStore."""
    pytest.importorskip("safetensors")
    tensor = torch.randn(4, 8)
    
    # 1. Test binary serialization / deserialization loop
    serialized = _serialize_tensor(tensor)
    assert isinstance(serialized, bytes)
    
    deserialized = _deserialize_tensor(serialized)
    assert torch.allclose(tensor, deserialized)
    
    # 2. Verify graph isolation (deserialized tensor must have zero autograd histories)
    x = torch.randn(2, 2, requires_grad=True)
    y = x * 2.0
    
    serialized_y = _serialize_tensor(y)
    deserialized_y = _deserialize_tensor(serialized_y)
    
    assert not deserialized_y.requires_grad
    assert deserialized_y.grad_fn is None


# ── 3. Micro-Benchmarking & Selective Hooking ─────────────────────────────────

def test_selective_hook_profiler() -> None:
    """Verify that our micro-profiler ranks submodules by variance and hooks only target subnets."""
    engine = CognitiveAugEngine()
    model = ToyModel()
    
    dummy_input = torch.randn(4, 8)
    
    # 1. Profile submodules
    stats = profile_submodules(model, dummy_input)
    assert len(stats) == 2  # layer_a and layer_b
    assert stats[0][0] in ["layer_a", "layer_b"]
    
    # 2. Selective hook top 50% of model layers (1 out of 2)
    adapters = register_selective_hooks(
        engine=engine,
        model=model,
        latent_dim=4,
        dummy_input=dummy_input,
        selective_ratio=0.5,
        use_dendritic=False
    )
    
    assert len(adapters) == 1
    assert len(engine.registry.list_adapters()) == 1


# ── 4. Asynchronous Execution Strategy ────────────────────────────────────────

def test_async_task_consolidation() -> None:
    """Verify that Slow-Wave Sleep consolidation offloads asynchronously to the worker queue."""
    engine = CognitiveAugEngine()
    workspace = GlobalWorkspace(latent_dim=4, key_dim=8)
    engine.attach_workspace(workspace)

    replay_buffer = CognitiveReplayBuffer(max_size=10)
    engine.attach_replay_buffer(replay_buffer)

    # Register toy module adapter with the engine registry
    toy_model = ToyModel()
    adapter = ModuleAdapter(
        name="toy",
        module=toy_model.layer_a,
        latent_dim=4,
        data_flow=engine.data_flow,
        key_dim=8,
    )
    engine.registry.register("toy", adapter)

    # Populate traces
    for _ in range(5):
        latent_states = {"toy": torch.randn(2, 4)}
        context = torch.randn(2, 4)
        engine.replay_buffer.add_trace(latent_states, context, salience=1.0)

    assert len(engine.replay_buffer) == 5

    # Trigger asynchronous SWS consolidation
    task_info = offloaded_enter_sleep_phase(
        engine=engine,
        steps=2,
        learning_rate=0.01,
        batch_size=2,
        use_celery=False
    )

    assert task_info["status"] in ["Queued", "Offloaded"]
    assert task_info["backend"] == "Local Thread"

    # Wait for local worker thread to process background queue
    for _ in range(30):
        if len(engine.replay_buffer) == 0:
            break
        time.sleep(0.1)
    
    # Replay buffer should be flushed after sleep consolidation completes!
    assert len(engine.replay_buffer) == 0
    stop_async_worker()


# ── 5. Production Telemetry & Observability ───────────────────────────────────

def test_json_observability_telemetry() -> None:
    """Verify GWTTelemetryLogger formats and streams metrics as standard JSON logs."""
    log_stream = io.StringIO()
    telemetry = GWTTelemetryLogger(stream=log_stream)

    modules_telemetry = {
        "vision_subsystem": {
            "dendritic_active_pct": 75.0,
            "plasticity_scale": 0.5,
            "grad_status": "Stable"
        }
    }

    # Record waking step GWT broadcast
    telemetry.record_step(
        step_idx=42,
        ne=0.25,
        ach=0.82,
        ignition_threshold=0.45,
        modules_telemetry=modules_telemetry
    )

    log_stream.seek(0)
    log_line = log_stream.readline().strip()
    assert log_line != ""

    # Parse and assert structured JSON keys
    parsed = json.loads(log_line)
    assert parsed["record_type"] == "WakingStep"
    assert parsed["step"] == 42
    assert parsed["neuromodulators"]["norepinephrine_surprise"] == 0.25
    assert parsed["neuromodulators"]["acetylcholine_focus"] == 0.82
    assert parsed["modules"]["vision_subsystem"]["dendritic_active_pct"] == 75.0


def test_precision_discovery_telemetry_scaling() -> None:
    """Verify precision-awareness, backbone discovery engine, and scientific notation scaling."""
    # 1. Test Backbone Discovery
    class DummyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Sequential(nn.Linear(8, 8))])
    
    class MockLLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = DummyTransformer()

    model = MockLLM()
    from cognitive_aug.profiler import discover_transformer_layers
    layers = discover_transformer_layers(model)
    assert isinstance(layers, nn.ModuleList)
    assert len(layers) == 1

    # 2. Test Precision and Device Constructors
    from cognitive_aug import DendriticModuleAdapter, CognitiveAugEngine
    engine = CognitiveAugEngine()
    
    # We create a half-precision (float16) target layer to mimic host weights
    target_layer = nn.Linear(8, 8).to(dtype=torch.float16)
    
    adapter = DendriticModuleAdapter(
        name="dendrite_half",
        module=target_layer,
        latent_dim=8,
        data_flow=engine.data_flow,
        device=torch.device("cpu"),
        dtype=torch.float16,
    )
    
    # Check that keys and projections inside adapters are cast to float16
    assert adapter.key_proj.weight.dtype == torch.float16
    assert adapter.dendrite_gate.context_proj.weight.dtype == torch.float16

    # 3. Test Telemetry Scaling below 0.001
    from cognitive_aug import MetacognitiveMonitor
    monitor = MetacognitiveMonitor()
    monitor.ach = 1.95e-4
    monitor.ne = 0.0005
    
    chems = monitor.get_chemical_levels()
    # Check that they format to scientific format (contain "e-")
    assert "1.95e-04" in chems["dashboard"]
    assert "5.00e-04" in chems["dashboard"]
