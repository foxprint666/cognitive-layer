import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DecayWorkingMemory(nn.Module):
    """
    A short-term working memory layer with exponential decay for workspace slots.
    If a slot fails to ignite, its old state decays exponentially using an in-place mutation:
    `workspace_state.mul_(decay_rate)`.
    Returns a blended context vector of the new winner and the decaying trace of the past.
    """

    def __init__(
        self, latent_dim: int, decay_rate: float = 0.9, blend_weight: float = 0.5
    ) -> None:
        """
        Args:
            latent_dim: Dimensionality of the global workspace representation.
            decay_rate: Exponential decay rate (0.0 < decay_rate <= 1.0).
            blend_weight: Weight given to the new winner in the blended output (0.0 to 1.0).
                          blend_weight * new_winner + (1.0 - blend_weight) * decayed_trace.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.decay_rate = decay_rate
        self.blend_weight = blend_weight

        # Stateful buffer to store the current working memory trace
        self.register_buffer("workspace_state", torch.zeros(1, latent_dim))
        self.has_initialized = False

    def forward(self, new_winner: torch.Tensor, ignited: torch.Tensor) -> torch.Tensor:
        """
        Processes a workspace step, updating and decaying working memory slots.

        Args:
            new_winner: Tensor of shape [B, latent_dim] representing the newly selected workspace state.
            ignited: Boolean/float tensor of shape [B] or [B, 1] indicating whether each batch element
                     successfully ignited.

        Returns:
            Blended context tensor of shape [B, latent_dim].
        """
        batch_size = new_winner.shape[0]

        # 1. Initialize or resize the workspace state buffer if batch size or device changed
        if not self.has_initialized or self.workspace_state.shape[0] != batch_size:
            self.workspace_state = torch.zeros(
                batch_size, self.latent_dim, device=new_winner.device, dtype=new_winner.dtype
            )
            self.has_initialized = True

        # 2. Reshape ignited to [B, 1] for broadcasting across latent dimension
        if ignited.ndim == 1:
            ignited = ignited.unsqueeze(-1)

        ignited_mask = ignited.to(dtype=new_winner.dtype)

        # 3. Decay the non-ignited slots in-place
        # For ignited slots: decay factor is 1.0 (no decay, will be overwritten by new winner)
        # For non-ignited slots: decay factor is decay_rate
        decay_factor = torch.where(ignited_mask > 0.0, 1.0, self.decay_rate)
        self.workspace_state.mul_(decay_factor)

        # 4. Update the state with the new winner where ignition succeeded
        # If ignited, overwrite with new winner. If not ignited, retain the decayed old trace.
        # This keeps the update fully vectorized and fast.
        self.workspace_state = torch.where(ignited_mask > 0.0, new_winner, self.workspace_state)

        # 5. Return blended context vector: blend_weight * new_winner + (1.0 - blend_weight) * decayed_trace
        blended_state = (self.blend_weight * new_winner) + (
            (1.0 - self.blend_weight) * self.workspace_state
        )

        return blended_state

    def reset_memory(self) -> None:
        """
        Resets the working memory stateful flag.
        The buffer will be zeroed out on the next forward pass.
        """
        self.has_initialized = False
