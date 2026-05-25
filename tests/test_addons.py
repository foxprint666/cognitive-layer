import pytest
import torch
import torch.nn as nn
from cognitive_aug import (
    global_pool_latent,
    CosineSimilaritySelector,
    VectorizedCrossAttentionSelector,
    EfficientGumbelSoftmaxSelector,
    MagnitudeSalience,
    EntropySalience,
    TemporalSurpriseSalience,
    DecayWorkingMemory,
    CognitiveOutputRouter,
)


def test_global_pool_latent():
    # 2D case [B, D]
    x_2d = torch.randn(4, 8)
    assert torch.equal(global_pool_latent(x_2d), x_2d)

    # 3D case [B, T, D]
    x_3d = torch.randn(4, 5, 8)
    out_3d = global_pool_latent(x_3d)
    assert out_3d.shape == (4, 8)
    assert torch.allclose(out_3d, x_3d.mean(dim=1))

    # 4D case [B, C, H, W]
    x_4d = torch.randn(4, 8, 3, 3)
    out_4d = global_pool_latent(x_4d)
    assert out_4d.shape == (4, 8)
    assert torch.allclose(out_4d, x_4d.mean(dim=[2, 3]))


def test_cosine_similarity_selector():
    selector = CosineSimilaritySelector(key_dim=6)
    
    # [B=2, num_modules=3, key_dim=6]
    keys = torch.randn(2, 3, 6)
    query = torch.randn(2, 6)
    
    weights = selector(keys, query)
    assert weights.shape == (2, 3)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2))


def test_vectorized_cross_attention_selector():
    selector = VectorizedCrossAttentionSelector(key_dim=8, num_heads=2)
    
    # Enable gradient tracking on query and keys
    keys = torch.randn(2, 3, 8, requires_grad=True)
    query = torch.randn(2, 8, requires_grad=True)
    
    weights = selector(keys, query)
    assert weights.shape == (2, 3)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2))
    
    # Backprop test
    loss = weights.sum()
    loss.backward()
    assert keys.grad is not None
    assert query.grad is not None


def test_efficient_gumbel_softmax_selector():
    # Test hard Gumbel-Softmax selection
    selector = EfficientGumbelSoftmaxSelector(key_dim=6, tau=1.0, hard=True)
    
    keys = torch.randn(2, 3, 6, requires_grad=True)
    query = torch.randn(2, 6, requires_grad=True)
    
    weights = selector(keys, query)
    assert weights.shape == (2, 3)
    
    # Check that it's a one-hot vector (hard selection)
    assert torch.all((weights == 0.0) | (weights == 1.0))
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2))
    
    # Backprop test (straight-through estimator allows grads to flow back)
    loss = weights.sum()
    loss.backward()
    assert keys.grad is not None
    assert query.grad is not None


def test_magnitude_salience():
    salience = MagnitudeSalience(ignition_threshold=2.0)
    
    # Latents with shapes [B=2, D=4]
    latents = {
        "mod_a": torch.ones(2, 4),  # norm = sqrt(4) = 2.0
        "mod_b": torch.zeros(2, 4), # norm = 0.0
    }
    
    scores = salience(latents)
    assert scores.shape == (2, 2)
    assert torch.allclose(scores[:, 0], torch.tensor([2.0, 2.0]))
    assert torch.allclose(scores[:, 1], torch.tensor([0.0, 0.0]))
    
    # Ignition gating test
    mask = salience.gate(scores)
    assert mask.shape == (2, 2)
    assert torch.all(mask[:, 0] == 1.0) # mod_a >= 2.0 -> ignited
    assert torch.all(mask[:, 1] == 0.0) # mod_b < 2.0 -> suppressed


def test_entropy_salience():
    # Low threshold
    salience = EntropySalience(ignition_threshold=0.5)
    
    # mod_a is high confidence (one active dimension), mod_b is flat uniform noise
    mod_a_state = torch.zeros(2, 8)
    mod_a_state[:, 0] = 100.0  # soft-max will be highly spiked
    
    mod_b_state = torch.zeros(2, 8)  # soft-max will be flat uniform
    
    latents = {
        "mod_a": mod_a_state,
        "mod_b": mod_b_state,
    }
    
    scores = salience(latents)
    assert scores.shape == (2, 2)
    
    # mod_a confidence should be close to 1.0
    assert scores[0, 0] > 0.9
    # mod_b confidence should be close to 0.0
    assert scores[0, 1] < 0.1
    
    mask = salience.gate(scores)
    assert mask.shape == (2, 2)
    assert torch.all(mask[:, 0] == 1.0) # mod_a should ignite
    assert torch.all(mask[:, 1] == 0.0) # mod_b should be masked out


def test_temporal_surprise_salience():
    salience = TemporalSurpriseSalience(ignition_threshold=0.3)
    
    # Latents on step 1
    state_a1 = torch.ones(2, 4)
    state_b1 = torch.ones(2, 4)
    
    latents1 = {
        "mod_a": state_a1,
        "mod_b": state_b1,
    }
    
    # First step: surprise should be 0 since cache initializes with current state
    scores1 = salience(latents1)
    assert torch.allclose(scores1, torch.zeros(2, 2))
    
    # Latents on step 2: mod_a remains identical, mod_b changes significantly
    state_a2 = torch.ones(2, 4)
    state_b2 = -torch.ones(2, 4)  # Opposite direction -> cosine distance = 2.0
    
    latents2 = {
        "mod_a": state_a2,
        "mod_b": state_b2,
    }
    
    scores2 = salience(latents2)
    # mod_a surprise close to 0.0
    assert torch.allclose(scores2[:, 0], torch.zeros(2), atol=1e-5)
    # mod_b surprise close to 2.0
    assert torch.allclose(scores2[:, 1], torch.tensor([2.0, 2.0]))
    
    mask = salience.gate(scores2)
    assert torch.all(mask[:, 0] == 0.0) # mod_a (no surprise) -> suppressed
    assert torch.all(mask[:, 1] == 1.0) # mod_b (surprise) -> ignited


def test_decay_working_memory():
    # latent_dim = 4, decay_rate = 0.5, blend_weight = 0.8
    memory = DecayWorkingMemory(latent_dim=4, decay_rate=0.5, blend_weight=0.8)
    
    # Step 1: Winner proposed, ignited is True
    winner1 = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    ignited1 = torch.tensor([1.0])
    
    out1 = memory(winner1, ignited1)
    # workspace state should be winner1. Blended is 0.8 * winner1 + 0.2 * winner1 = winner1
    assert torch.allclose(out1, winner1)
    assert torch.allclose(memory.workspace_state, winner1)
    
    # Step 2: Winner proposed, but ignited is False -> decays workspace state in-place
    winner2 = torch.tensor([[5.0, 6.0, 7.0, 8.0]])
    ignited2 = torch.tensor([0.0])
    
    out2 = memory(winner2, ignited2)
    # Decayed workspace state should be winner1 * 0.5 = [[0.5, 1.0, 1.5, 2.0]]
    expected_state = winner1 * 0.5
    assert torch.allclose(memory.workspace_state, expected_state)
    
    # Blended = 0.8 * winner2 + 0.2 * expected_state
    expected_blend = 0.8 * winner2 + 0.2 * expected_state
    assert torch.allclose(out2, expected_blend)


def test_cognitive_output_router():
    output_specs = {
        "classification": 3,
        "actions": 5,
        "salience_feedback": 2
    }
    
    router = CognitiveOutputRouter(latent_dim=10, output_specs=output_specs)
    
    workspace_tensor = torch.randn(4, 10)
    outputs = router(workspace_tensor)
    
    assert list(outputs.keys()) == ["classification", "actions", "salience_feedback"]
    
    # Check shapes
    assert outputs["classification"].shape == (4, 3)
    assert outputs["actions"].shape == (4, 5)
    assert outputs["salience_feedback"].shape == (4, 2)
    
    # Verify that they share the same underlying storage (zero-copy slicing)
    try:
        assert outputs["classification"].untyped_storage().data_ptr() == outputs["actions"].untyped_storage().data_ptr()
    except AttributeError:
        assert outputs["classification"].storage().data_ptr() == outputs["actions"].storage().data_ptr()
