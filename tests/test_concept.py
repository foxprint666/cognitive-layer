import torch
import torch.nn as nn
import pytest

from cognitive_aug import (
    CognitiveAugEngine,
    GlobalWorkspace,
)
from cognitive_aug.concept import ConceptLayer, ConceptInterventionEngine


def test_concept_projection_boundaries():
    """Verify that different abstraction types project inputs into valid [0, 1] concept boundaries."""
    input_dim = 16
    num_concepts = 4
    batch_size = 3

    # Test all abstraction types
    for abstraction in ["projection", "linear", "softmax", "threshold"]:
        layer = ConceptLayer(
            input_dim=input_dim,
            num_concepts=num_concepts,
            abstraction_type=abstraction,
        )

        inputs = torch.randn(batch_size, input_dim)
        outputs = layer(inputs)

        # Check shape
        assert outputs.shape == (batch_size, num_concepts)

        # Check value bounds [0, 1]
        assert torch.all(outputs >= 0.0)
        assert torch.all(outputs <= 1.0)

        # For threshold, values must be exactly 0.0 or 1.0 (within precision)
        if abstraction == "threshold":
            assert torch.all((outputs == 0.0) | (outputs == 1.0))


def test_concept_intervention_causal_override_and_gradient_flow():
    """
    Verify that concept overrides forcefully override activations, detaching gradients
    on overridden indices while preserving full gradient flow on uninvolved indices.
    """
    input_dim = 8
    num_concepts = 4
    batch_size = 2

    layer = ConceptLayer(
        input_dim=input_dim,
        num_concepts=num_concepts,
        abstraction_type="projection",
    )

    # 1. Register causal interventions
    # Index 1 forced to 0.85
    layer.intervention_engine.set_intervention(1, 0.85)
    # Index 3 forced to 0.15
    layer.intervention_engine.set_intervention(3, 0.15)

    inputs = torch.randn(batch_size, input_dim, requires_grad=True)
    outputs = layer(inputs)

    # Verify overrides are exact
    assert torch.allclose(outputs[:, 1], torch.tensor(0.85))
    assert torch.allclose(outputs[:, 3], torch.tensor(0.15))

    # Verify normal activations remain variable
    assert not torch.allclose(outputs[:, 0], torch.tensor(0.85))
    assert not torch.allclose(outputs[:, 2], torch.tensor(0.15))

    # 2. Verify gradient flow on UNINVOLVED elements
    loss_uninvolved = outputs[:, 0].sum() + outputs[:, 2].sum()
    loss_uninvolved.backward(retain_graph=True)

    # Gradients should propagate through to inputs perfectly
    assert inputs.grad is not None
    assert torch.any(inputs.grad != 0.0)

    # Reset gradient
    inputs.grad.zero_()

    # 3. Verify gradient flow is DETACHED for OVERRIDDEN elements
    loss_overridden = outputs[:, 1].sum() + outputs[:, 3].sum()
    loss_overridden.backward()

    # Since they are clamped to constants, gradient should be exactly 0.0
    assert torch.all(inputs.grad == 0.0)


def test_inspect_conceptual_dashboard():
    """Verify that engine.inspect() outputs custom conceptual maps with overridden markings."""
    engine = CognitiveAugEngine()
    latent_dim = 8
    workspace = GlobalWorkspace(latent_dim=latent_dim, key_dim=4)
    engine.attach_workspace(workspace)

    # Attach a concept layer with custom concept names
    concept_names = ["Saliency Core", "Noise Subtraction"]
    layer = ConceptLayer(
        input_dim=5,
        num_concepts=2,
        abstraction_type="projection",
        concept_names=concept_names,
    )
    engine.attach_concept_layer("my_concept_layer", layer)

    # Verify attachment populated engine attributes
    assert layer.name == "my_concept_layer"
    assert layer.data_flow == engine.data_flow

    # 1. Run forward pass
    inputs = torch.randn(2, 5)
    _ = layer(inputs)

    # Check that inspect renders the maps correctly
    report = engine.inspect()
    assert "Conceptual Maps:" in report
    assert "Saliency Core" in report
    assert "Noise Subtraction" in report

    # 2. Apply causal intervention
    layer.intervention_engine.set_intervention(1, 1.0)
    
    # Run forward pass again
    _ = layer(inputs)

    # Inspect again to verify OVERRIDDEN output tag
    report_overridden = engine.inspect()
    assert "(OVERRIDDEN -> 1.0)" in report_overridden


def test_concept_intervention_string_name_override():
    """Verify that causal interventions can be registered and resolved by string concept names."""
    input_dim = 6
    num_concepts = 3
    batch_size = 2
    concept_names = ["Anomalous", "Exploratory", "Focused"]
    
    layer = ConceptLayer(
        input_dim=input_dim,
        num_concepts=num_concepts,
        abstraction_type="projection",
        concept_names=concept_names,
    )
    
    # Register overrides: one by string name, one by integer index
    layer.intervention_engine.set_intervention("Anomalous", 1.0)
    layer.intervention_engine.set_intervention(2, 0.0)
    
    inputs = torch.randn(batch_size, input_dim)
    outputs = layer(inputs)
    
    # Assert column 0 ("Anomalous") is exactly clamped to 1.0
    assert torch.allclose(outputs[:, 0], torch.tensor(1.0))
    # Assert column 2 ("Focused" / index 2) is exactly clamped to 0.0
    assert torch.allclose(outputs[:, 2], torch.tensor(0.0))
    # Assert column 1 ("Exploratory") remains free and variable
    assert not torch.allclose(outputs[:, 1], torch.tensor(1.0))
    assert not torch.allclose(outputs[:, 1], torch.tensor(0.0))
