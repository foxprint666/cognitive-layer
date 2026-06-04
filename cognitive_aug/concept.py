"""
cognitive_aug/concept.py
==============
Concept-Level Representation and Abstraction Layer for GWT (Phase v0.6).

Projects high-dimensional hidden states into low-dimensional, structured
concept vectors to support causal interventions, explicit tracking, and
information bottleneck analysis.
"""

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ConceptInterventionEngine:
    """
    Concept Intervention Engine (Causal Override).

    Manages explicit causal overrides of concept activations. Allows programmatic or
    human-in-the-loop intervention to forcefully clamp concept values.
    """

    def __init__(self) -> None:
        # Map of concept index -> override value
        self.interventions: Dict[int, float] = {}

    def set_intervention(self, concept_idx: int, forced_value: float) -> None:
        """Registers a causal override for a specific concept index."""
        self.interventions[concept_idx] = float(forced_value)
        logger.debug(
            f"Registered concept {concept_idx} intervention -> {forced_value:.4f}"
        )

    def clear_interventions(self) -> None:
        """Clears all active overrides."""
        self.interventions.clear()
        logger.debug("Cleared all active concept interventions.")

    def get_interventions(self) -> Dict[int, float]:
        """Retrieves active concept overrides."""
        return self.interventions


class ConceptLayer(nn.Module):
    """
    ConceptLayer Bottleneck Projector.

    Inherits from torch.nn.Module. Projects high-dimensional hidden tensors
    into explicit low-dimensional concept activation scores between 0.0 and 1.0.
    Integrates causal interventions and updates GWT DataFlowManager telemetry.
    """

    def __init__(
        self,
        input_dim: int,
        num_concepts: int,
        abstraction_type: str = "projection",
        concept_names: Optional[List[str]] = None,
        threshold: float = 0.5,
    ) -> None:
        """
        Args:
            input_dim        : Dimensionality of high-dimensional input hidden states.
            num_concepts     : Dimensionality of low-dimensional concept score space.
            abstraction_type : Bottleneck type:
                               - 'projection': Sigmoid-bounded learnable projection (default).
                               - 'linear': Clamped learnable linear output [0, 1].
                               - 'softmax': Softmax-normalized probability score output.
                               - 'threshold': Binary thresholded activation output using STE.
            concept_names    : Optional custom names for concepts.
            threshold        : Activation threshold for 'threshold' GWT bottleneck.
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_concepts = num_concepts
        self.abstraction_type = abstraction_type.lower()
        self.threshold = threshold

        if self.abstraction_type not in [
            "projection",
            "linear",
            "softmax",
            "threshold",
        ]:
            raise ValueError(
                f"Unsupported abstraction_type '{self.abstraction_type}'. "
                "Must be one of ['projection', 'linear', 'softmax', 'threshold']."
            )

        # Learnable projection parameter matrix mapping input_dim -> num_concepts
        self.projection = nn.Linear(input_dim, num_concepts)

        # Causal intervention override manager
        self.intervention_engine = ConceptInterventionEngine()

        # Human-readable concept labels for telemetry dashboards
        if concept_names is not None:
            if len(concept_names) != num_concepts:
                raise ValueError(
                    f"concept_names length ({len(concept_names)}) must match num_concepts ({num_concepts})."
                )
            self.concept_names = list(concept_names)
        else:
            self.concept_names = [f"Concept {i}" for i in range(num_concepts)]

        # System integration variables
        self.name: Optional[str] = None
        self.data_flow: Optional[Any] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects hidden states, applies active overrides, and registers activations to GWT.

        Args:
            x : [B, ..., input_dim] hidden activation tensors.

        Returns:
            [B, ..., num_concepts] low-dimensional concept scores.
        """
        proj = self.projection(x)

        # 1. Apply chosen abstraction activation function
        if self.abstraction_type == "projection":
            activations = torch.sigmoid(proj)
        elif self.abstraction_type == "linear":
            activations = torch.clamp(proj, 0.0, 1.0)
        elif self.abstraction_type == "softmax":
            activations = F.softmax(proj, dim=-1)
        elif self.abstraction_type == "threshold":
            sig = torch.sigmoid(proj)
            spiked = (sig >= self.threshold).to(sig.dtype)
            # Straight-Through Estimator (STE) to preserve backpropagation gradient paths
            activations = spiked + (sig - sig.detach())
        else:
            activations = torch.sigmoid(proj)

        # 2. Causal intervention clamping under a torch.no_grad() block
        interventions = self.intervention_engine.get_interventions()
        if interventions:
            # Clamped target tensor matching activations shape
            clamped_activations = activations.detach().clone()

            with torch.no_grad():
                # Build index mask and target values
                override_mask = torch.zeros(
                    self.num_concepts, device=activations.device, dtype=torch.bool
                )
                override_vals = torch.zeros(
                    self.num_concepts,
                    device=activations.device,
                    dtype=activations.dtype,
                )
                for idx, val in interventions.items():
                    actual_idx = idx
                    if isinstance(idx, str):
                        if idx in self.concept_names:
                            actual_idx = self.concept_names.index(idx)
                        else:
                            logger.warning(
                                f"Concept override key '{idx}' not found in concept names."
                            )
                            continue

                    if isinstance(actual_idx, (int, torch.Tensor)):
                        # Convert to standard Python int if it is a single-element tensor
                        if isinstance(actual_idx, torch.Tensor):
                            actual_idx = int(actual_idx.item())

                        if 0 <= actual_idx < self.num_concepts:
                            override_mask[actual_idx] = True
                            override_vals[actual_idx] = val

                clamped_activations[..., override_mask] = override_vals[override_mask]

            # Differentiable selection: zeroes out gradients for overridden concepts,
            # while fully preserving gradients on all other active nodes
            activations = torch.where(override_mask, clamped_activations, activations)

        # 3. Dynamic registration cache to GWT DataFlowManager
        if self.data_flow is not None and self.name is not None:
            # We store a detached copy to eliminate stale graph memory footprint
            self.data_flow.update_buffer(self.name, activations.detach())

        return activations
