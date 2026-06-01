"""
cognitive_aug/salience.py
===============
Low-overhead, vectorized salience evaluation metrics for GWT ignition gating.

All salience classes inherit from BaseSalience and expose a ``gate()`` method
for threshold-based ignition masking.

Available metrics
-----------------
global_pool_latent      : Dimension-agnostic helper — maps any tensor to [B, D].
BaseSalience            : Abstract base class.
MagnitudeSalience       : L2-norm of latent vectors (single vectorized pass).
EntropySalience         : Normalized Shannon entropy confidence [0, 1].
TemporalSurpriseSalience: Stateful cosine-distance surprise across time steps.
"""
import logging
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def global_pool_latent(latent: torch.Tensor) -> torch.Tensor:
    """
    Dimension-agnostic global averaging — converts any model tensor into [B, D].

    Handles:
    - 2D ``[B, D]``        -> returned unchanged.
    - 3D ``[B, T, D]``     -> mean over T  (sequences / transformers).
    - 4D ``[B, C, H, W]``  -> mean over H, W (CNN spatial maps).
    - Higher dims           -> mean over all intermediate dims.
    """
    if latent.ndim <= 2:
        return latent
    elif latent.ndim == 3:
        return latent.mean(dim=1)           # [B, T, D] -> [B, D]
    elif latent.ndim == 4:
        return latent.mean(dim=[2, 3])      # [B, C, H, W] -> [B, C]
    else:
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
            latent_states: ``{module_name: tensor}`` — raw module outputs of any shape.

        Returns:
            Salience scores [B, num_modules].
        """
        raise NotImplementedError("Subclasses must implement forward")

    def gate(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Applies GWT ignition threshold gating to salience scores.

        If all modules fall below the threshold, a fallback activates all modules
        to prevent complete silence or division-by-zero.

        Args:
            scores : Tensor [B, num_modules].

        Returns:
            Float gating mask [B, num_modules] — 1.0 = ignited, 0.0 = suppressed.
        """
        if self.ignition_threshold <= 0.0:
            return torch.ones_like(scores)

        mask = (scores >= self.ignition_threshold).to(dtype=scores.dtype)

        # Fallback: if every module falls below threshold, activate all
        fallback = (mask.sum(dim=-1, keepdim=True) == 0).to(dtype=scores.dtype)
        mask = mask + fallback

        return torch.clamp(mask, 0.0, 1.0)


class MagnitudeSalience(BaseSalience):
    """
    Computes salience as the L2 norm of latent states in a single vectorized pass
    using ``torch.linalg.vector_norm``.
    """

    def __init__(self, ignition_threshold: float = 0.0) -> None:
        super().__init__(ignition_threshold)

    def forward(self, latent_states: Dict[str, torch.Tensor]) -> torch.Tensor:
        names = list(latent_states.keys())
        pooled = [global_pool_latent(latent_states[n]) for n in names]
        stacked = torch.stack(pooled, dim=1)            # [B, num_modules, D]
        return torch.linalg.vector_norm(stacked, dim=-1)  # [B, num_modules]


class EntropySalience(BaseSalience):
    """
    Computes Shannon entropy confidence to penalize noisy, unconfident
    representations. Normalizes to [0.0, 1.0] where 1.0 = maximally confident.
    """

    def __init__(self, ignition_threshold: float = 0.0) -> None:
        super().__init__(ignition_threshold)

    def forward(self, latent_states: Dict[str, torch.Tensor]) -> torch.Tensor:
        names = list(latent_states.keys())
        pooled = [global_pool_latent(latent_states[n]) for n in names]
        stacked = torch.stack(pooled, dim=1)    # [B, num_modules, D]

        p = torch.softmax(stacked, dim=-1)

        try:
            entropy = torch.special.entr(p).sum(dim=-1)
        except AttributeError:
            log_p = torch.log_softmax(stacked, dim=-1)
            entropy = -(p * log_p).sum(dim=-1)

        latent_dim = stacked.shape[-1]
        max_entropy = torch.log(
            torch.tensor(latent_dim, dtype=stacked.dtype, device=stacked.device)
        )

        # Confidence = 1 - normalized_entropy  (higher = more confident)
        return 1.0 - (entropy / max_entropy)    # [B, num_modules]


class TemporalSurpriseSalience(BaseSalience):
    """
    Stateful salience metric that caches previous step latent vectors per module
    and measures temporal surprise as cosine distance across consecutive steps.

    The cache is kept detached to prevent backward-graph retention across iterations.
    """

    def __init__(self, ignition_threshold: float = 0.0) -> None:
        super().__init__(ignition_threshold)
        self.cache: Dict[str, torch.Tensor] = {}

    def forward(self, latent_states: Dict[str, torch.Tensor]) -> torch.Tensor:
        names = list(latent_states.keys())

        # Pool current states -> [B, num_modules, D]
        current_pooled = [global_pool_latent(latent_states[n]) for n in names]
        current_stacked = torch.stack(current_pooled, dim=1)

        # Retrieve (or initialize) cached previous states
        cached_list = []
        for name in names:
            prev = self.cache.get(name)
            if prev is None:
                # First call: no surprise yet — initialize to current state
                prev = latent_states[name].detach()
            cached_list.append(global_pool_latent(prev))

        cached_stacked = torch.stack(cached_list, dim=1)

        # Update cache (detached to prevent stale graph retention)
        for name in names:
            self.cache[name] = latent_states[name].detach()

        # Surprise = cosine distance = 1 - cosine_similarity  [B, num_modules]
        cos_sim = F.cosine_similarity(current_stacked, cached_stacked, dim=-1)
        return 1.0 - cos_sim

    def clear_cache(self) -> None:
        """Clears the in-memory stateful cache between episodes."""
        self.cache.clear()
