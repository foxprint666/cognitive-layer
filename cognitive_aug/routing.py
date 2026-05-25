import logging
from typing import Dict
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CognitiveOutputRouter(nn.Module):
    """
    A highly performant linear projection mapping layer that safely broadcasts the final
    unified workspace tensor back into dedicated output heads (e.g., classification, actions)
    using a single parallel projection pass to achieve zero-copy/loop-free forward routing.
    """

    def __init__(self, latent_dim: int, output_specs: Dict[str, int]) -> None:
        """
        Args:
            latent_dim: Dimensionality of the global workspace representation.
            output_specs: Dictionary mapping output head name (str) to its dimension (int).
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.output_specs = output_specs

        # 1. Compute total output dimension across all heads
        self.total_out_dim = sum(output_specs.values())
        
        # 2. Define a single parallel projection layer
        self.parallel_proj = nn.Linear(latent_dim, self.total_out_dim)

        # 3. Pre-calculate split sections to slice the unified projection without copying tensors
        self.slices = {}
        current_idx = 0
        for name, dim in output_specs.items():
            self.slices[name] = (current_idx, current_idx + dim)
            current_idx += dim

        logger.info(
            f"Created CognitiveOutputRouter: parallel projection {latent_dim} -> {self.total_out_dim} "
            f"for {len(output_specs)} output heads."
        )

    def forward(self, workspace_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Projects the workspace tensor into multiple output heads in a single vectorized pass,
        returning views of the projection to avoid copy operations.

        Args:
            workspace_tensor: Unified workspace state tensor of shape [B, latent_dim].

        Returns:
            Dict mapping head name (str) to projected output tensor of shape [B, head_dim].
        """
        # A single parallel projection to compute all outputs simultaneously
        projected = self.parallel_proj(workspace_tensor)  # [B, total_out_dim]

        # Slice the outputs into dedicated heads. Slicing returns a view,
        # which satisfies the zero-copy requirement.
        outputs = {}
        for name, (start, end) in self.slices.items():
            outputs[name] = projected[..., start:end]

        return outputs
