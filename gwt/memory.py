"""
gwt/memory.py
=============
Short-term working memory with exponential decay for workspace slots.

DecayWorkingMemory maintains a stateful buffer between GWT steps.
Non-ignited slots decay in-place; ignited slots are overwritten by the
new winner. Returns a blended context combining history and new winner.
"""
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DecayWorkingMemory(nn.Module):
    """
    Short-term working memory layer with exponential decay for workspace slots.

    On each forward call:
    - Ignited slots  : decayed by factor 1.0 (no decay), then overwritten by new winner.
    - Non-ignited slots: decayed in-place via ``workspace_state.mul_(decay_rate)``.

    Returns a blended context vector::

        output = blend_weight * new_winner + (1 - blend_weight) * decayed_trace
    """

    def __init__(
        self,
        latent_dim: int,
        decay_rate: float = 0.9,
        blend_weight: float = 0.5,
    ) -> None:
        """
        Args:
            latent_dim   : Dimensionality of the global workspace representation.
            decay_rate   : Exponential decay factor for non-ignited slots (0 < rate <= 1).
            blend_weight : Weight given to new winner in the blended output (0–1).
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.decay_rate = decay_rate
        self.blend_weight = blend_weight

        # Stateful memory buffer — lazily resized on first forward call
        self.register_buffer("workspace_state", torch.zeros(1, latent_dim))
        self.has_initialized = False

    def forward(
        self,
        new_winner: torch.Tensor,
        ignited: torch.Tensor,
    ) -> torch.Tensor:
        """
        Processes one workspace step, updating and decaying working memory slots.

        Args:
            new_winner : [B, latent_dim] — newly selected workspace state.
            ignited    : [B] or [B, 1]  — float/bool ignition mask per batch element.

        Returns:
            Blended context tensor [B, latent_dim].
        """
        batch_size = new_winner.shape[0]

        # Lazy init / resize when batch size or device changes
        if (
            not self.has_initialized
            or self.workspace_state.shape[0] != batch_size
        ):
            self.workspace_state = torch.zeros(
                batch_size,
                self.latent_dim,
                device=new_winner.device,
                dtype=new_winner.dtype,
            )
            self.has_initialized = True

        # Ensure ignited is [B, 1] for broadcasting across latent dim
        if ignited.ndim == 1:
            ignited = ignited.unsqueeze(-1)
        ignited_mask = ignited.to(dtype=new_winner.dtype)

        # Vectorized decay: ignited slots keep full value, others decay
        decay_factor = torch.where(ignited_mask > 0.0, 1.0, self.decay_rate)
        self.workspace_state.mul_(decay_factor)

        # Overwrite ignited slots with new winner; leave others at decayed trace
        self.workspace_state = torch.where(
            ignited_mask > 0.0, new_winner, self.workspace_state
        )

        # Blended output: blend_weight * new_winner + (1 - blend_weight) * decayed_trace
        return (self.blend_weight * new_winner) + (
            (1.0 - self.blend_weight) * self.workspace_state
        )

    def reset_memory(self) -> None:
        """
        Resets the working memory state flag.
        The buffer will be zeroed out on the next forward call.
        """
        self.has_initialized = False
