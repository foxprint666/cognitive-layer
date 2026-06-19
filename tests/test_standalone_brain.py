import torch
from cognitive_aug.models.autonomic_brain import StandaloneBrainModel

def test_standalone_brain_model():
    model = StandaloneBrainModel(latent_dim=16, key_dim=4)
    visual_input = torch.randn(2, 3, 32, 32)

    # Test forward pass
    output = model(visual_input)
    assert output.shape == (2, 10)

    # Test reset_context
    model.reset_context()
    assert torch.all(model.internal_context == 0)

    # Test training mode and graph accumulation
    model.train()
    output = model(visual_input)
    loss = output.sum()
    loss.backward()

    # Check if gradients are populated
    for name, param in model.named_parameters():
        if param.requires_grad and ("visual_lobe" in name or "language_lobe" in name):
            assert param.grad is not None

    print("StandaloneBrainModel test passed!")

if __name__ == "__main__":
    test_standalone_brain_model()
