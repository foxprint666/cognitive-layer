import logging
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def global_pool_latent(latent: torch.Tensor) -> torch.Tensor:
    """
    Apply fast spatial/temporal global averaging (torch.mean) to convert
    various model tensor structures into a clean standard [B, D] state.
    
    Handles:
    - 2D: [B, D] -> Remains [B, D]
    - 3D (Transformers / Sequences): [B, T, D] -> Mean over T (dim 1) -> [B, D]
    - 4D (CNNs / Images): [B, C, H, W] -> Mean over H, W (dims [2, 3]) -> [B, C]
    - Higher dimensions: Mean over intermediate dimensions between batch and feature.
    """
    if latent.ndim <= 2:
        return latent
    elif latent.ndim == 3:
        # [B, T, D] -> [B, D]
        return latent.mean(dim=1)
    elif latent.ndim == 4:
        # [B, C, H, W] -> [B, C]
        return latent.mean(dim=[2, 3])
    else:
        # Fallback for arbitrary higher dimensions
        dims = list(range(1, latent.ndim - 1))
        return latent.mean(dim=dims)


class BaseSalience(nn.Module):
    """
    Abstract base class for high-speed salience evaluation metrics.
    All salience metrics must inherit from this module.
    """

    def __init__(self, ignition_threshold: float = 0.0) -> None:
        """
        Args:
            ignition_threshold: Hard threshold below which states fail to ignite.
        """
        super().__init__()
        self.ignition_threshold = ignition_threshold

    def forward(self, latent_states: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Computes salience scores for all module proposals.

        Args:
            latent_states: Dict of module_name -> tensor [B, latent_dim] (or higher dim raw states).

        Returns:
            Salience scores of shape [B, num_modules].
        """
        raise NotImplementedError("Subclasses must implement forward")

    def gate(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Applies GWT ignition threshold gating to salience scores.
        If all modules fall below the threshold, a fallback is triggered
        to keep all modules active, preventing complete silence or division by zero.

        Args:
            scores: Tensor of shape [B, num_modules].

        Returns:
            Gating float mask of shape [B, num_modules] where 1.0 indicates ignited states,
            and 0.0 indicates suppressed states.
        """
        if self.ignition_threshold <= 0.0:
            return torch.ones_like(scores)

        # Mask out elements below threshold
        mask = (scores >= self.ignition_threshold).to(dtype=scores.dtype)
        
        # Fallback: if all modules fall below threshold, activate them all to maintain flow
        fallback = (mask.sum(dim=-1, keepdim=True) == 0).to(dtype=scores.dtype)
        mask = mask + fallback
        
        return torch.clamp(mask, 0.0, 1.0)


class MagnitudeSalience(BaseSalience):
    """
    Computes salience as the L2 norm of the latent states using torch.linalg.vector_norm(dim=-1).
    """

    def __init__(self, ignition_threshold: float = 0.0) -> None:
        super().__init__(ignition_threshold)

    def forward(self, latent_states: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Align, pool and stack latents
        names = list(latent_states.keys())
        
        # Apply global pooling first to be dimension-agnostic, then stack
        pooled = [global_pool_latent(latent_states[name]) for name in names]
        stacked = torch.stack(pooled, dim=1) # [B, num_modules, latent_dim]
        
        # Compute L2 norm in a single vectorized pass: [B, num_modules]
        return torch.linalg.vector_norm(stacked, dim=-1)


class EntropySalience(BaseSalience):
    """
    Computes Shannon entropy to penalize noisy, unconfident representations.
    Normalizes entropy between 0.0 and 1.0 confidence.
    """

    def __init__(self, ignition_threshold: float = 0.0) -> None:
        super().__init__(ignition_threshold)

    def forward(self, latent_states: Dict[str, torch.Tensor]) -> torch.Tensor:
        names = list(latent_states.keys())
        
        # Apply global pooling first, then stack
        pooled = [global_pool_latent(latent_states[name]) for name in names]
        stacked = torch.stack(pooled, dim=1) # [B, num_modules, latent_dim]
        
        # Convert states to a probability distribution over the feature dimension
        p = torch.softmax(stacked, dim=-1)
        
        # Compute Shannon entropy
        try:
            entropy = torch.special.entr(p).sum(dim=-1)
        except AttributeError:
            log_p = torch.log_softmax(stacked, dim=-1)
            entropy = -(p * log_p).sum(dim=-1)

        # Normalize confidence to [0, 1] range: Confidence = 1.0 - (Entropy / ln(D))
        latent_dim = stacked.shape[-1]
        max_entropy = torch.log(torch.tensor(latent_dim, dtype=stacked.dtype, device=stacked.device))
        
        confidence = 1.0 - (entropy / max_entropy)
        
        return confidence


class TemporalSurpriseSalience(BaseSalience):
    """
    Maintains an in-memory, stateful cache of previous steps' latent vectors
    per module. Uses rapid cosine distance math to calculate temporal surprise
    (shifts) across steps. Keeping it lightweight and detached.
    """

    def __init__(self, ignition_threshold: float = 0.0) -> None:
        super().__init__(ignition_threshold)
        self.cache: Dict[str, torch.Tensor] = {}

    def forward(self, latent_states: Dict[str, torch.Tensor]) -> torch.Tensor:
        names = list(latent_states.keys())
        
        # 1. Pool current states: [B, num_modules, latent_dim]
        current_pooled = [global_pool_latent(latent_states[name]) for name in names]
        current_stacked = torch.stack(current_pooled, dim=1)

        # 2. Retrieve or initialize cached states
        cached_list = []
        for name in names:
            prev = self.cache.get(name)
            if prev is None:
                # Initialize cache with current state on first forward (surprise = 0)
                # Ensure it's detached to prevent backward graph retention across iterations
                prev = latent_states[name].detach()
            cached_list.append(global_pool_latent(prev))
            
        cached_stacked = torch.stack(cached_list, dim=1)

        # 3. Update the stateful cache with current states (detached)
        for name in names:
            self.cache[name] = latent_states[name].detach()

        # 4. Compute cosine similarity in a single vectorized pass: [B, num_modules]
        cos_sim = F.cosine_similarity(current_stacked, cached_stacked, dim=-1)
        
        # Surprise is the cosine distance (1.0 - similarity)
        return 1.0 - cos_sim

    def clear_cache(self) -> None:
        """
        Clears the in-memory stateful cache.
        """
        self.cache.clear()
