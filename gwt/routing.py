"""
gwt/routing.py
==============
Parallel output routing from unified workspace state to downstream heads.

CognitiveOutputRouter maps a single workspace tensor to multiple dedicated
output heads (e.g. classification, actions, feedback) in one vectorized
matrix projection — zero loops, zero tensor copies.
"""
import logging
from typing import Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CognitiveOutputRouter(nn.Module):
    """
    Single-pass parallel projection layer that broadcasts the final unified
    workspace tensor into multiple dedicated output heads simultaneously.

    Achieves zero-copy output slicing by pre-computing index ranges and
    returning tensor *views* (not copies) of the joint projection::

        router = CognitiveOutputRouter(
            latent_dim=256,
            output_specs={"class_label": 10, "action_vector": 4, "attention_feedback": 3},
        )
        outputs = router(workspace_state)
        # outputs["class_label"]        -> [B, 10]
        # outputs["action_vector"]      -> [B, 4]
        # outputs["attention_feedback"] -> [B, 3]
    """

    def __init__(
        self,
        latent_dim: int,
        output_specs: Dict[str, int],
    ) -> None:
        """
        Args:
            latent_dim   : Dimensionality of the incoming workspace state.
            output_specs : ``{head_name: output_dim}`` mapping.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.output_specs = output_specs

        # Single parallel projection: latent_dim -> sum(all output dims)
        self.total_out_dim = sum(output_specs.values())
        self.parallel_proj = nn.Linear(latent_dim, self.total_out_dim)

        # Pre-compute zero-copy slice ranges for each head
        self.slices: Dict[str, tuple] = {}
        current_idx = 0
        for name, dim in output_specs.items():
            self.slices[name] = (current_idx, current_idx + dim)
            current_idx += dim

        logger.info(
            f"Created CognitiveOutputRouter: {latent_dim} -> {self.total_out_dim} "
            f"across {len(output_specs)} output heads."
        )

    def forward(self, workspace_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Projects the workspace tensor into all output heads in one pass.

        Args:
            workspace_tensor : Unified workspace state [B, latent_dim].

        Returns:
            ``{head_name: tensor [B, head_dim]}`` — views, not copies.
        """
        projected = self.parallel_proj(workspace_tensor)    # [B, total_out_dim]

        # Slice into head views — no data copied, memory shared with `projected`
        return {
            name: projected[..., start:end]
            for name, (start, end) in self.slices.items()
        }
