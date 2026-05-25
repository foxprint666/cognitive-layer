import pytest
import torch
import torch.nn as nn
from cognitive_aug import CognitiveAugEngine, ModuleAdapter
from cognitive_aug.engine import ModuleRegistry, DataFlowManager


class ToyModule(nn.Module):
    def __init__(self, in_dim=10, out_dim=10):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


def test_registry_basic_operations():
    registry = ModuleRegistry()
    assert len(registry.list_names()) == 0

    # Create dummy adapter
    dummy_module = ToyModule()
    data_flow = DataFlowManager()
    adapter = ModuleAdapter(
        name="test_mod", module=dummy_module, latent_dim=10, data_flow=data_flow
    )

    registry.register("test_mod", adapter)
    assert "test_mod" in registry.list_names()
    assert registry.get("test_mod") == adapter

    with pytest.raises(KeyError):
        registry.get("invalid_name")

    registry.clear()
    assert len(registry.list_names()) == 0


def test_data_flow_manager():
    dfm = DataFlowManager()
    t = torch.randn(2, 5)

    dfm.update_buffer("mod_a", t)
    assert torch.equal(dfm.get_buffer("mod_a"), t)

    with pytest.raises(TypeError):
        dfm.update_buffer("mod_b", [1, 2, 3])  # type: ignore

    with pytest.raises(KeyError):
        dfm.get_buffer("non_existent")

    dfm.clear_buffers()
    with pytest.raises(KeyError):
        dfm.get_buffer("mod_a")


def test_engine_module_registration():
    engine = CognitiveAugEngine()
    toy = ToyModule(in_dim=5, out_dim=5)

    adapter = engine.register_module(name="toy", module=toy, latent_dim=10, projection_in_dim=5)
    assert "toy" in engine.registry.list_names()
    assert adapter.latent_dim == 10
    assert adapter.projection is not None

    # Verify forward pass hook updates buffers automatically
    dummy_input = torch.randn(4, 5)
    toy_out = toy(dummy_input)

    # Output should have been intercepted and projected to 10
    buffer_state = engine.data_flow.get_buffer("toy")
    assert buffer_state.shape == (4, 10)
