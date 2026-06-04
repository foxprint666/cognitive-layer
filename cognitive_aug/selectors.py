"""
cognitive_aug/selectors.py
================
Optimized, plug-and-play attention selector modules for the Global Workspace.

All selectors inherit from BaseSelector and are hot-swappable via::

    workspace.selector = VectorizedCrossAttentionSelector(key_dim=64, num_heads=4)

Available selectors
-------------------
BaseSelector                   : Abstract base class — subclass to create custom selectors.
CosineSimilaritySelector       : Vectorized cosine similarity against a top-down query.
VectorizedCrossAttentionSelector: Native scaled_dot_product_attention (FlashAttention-speed).
EfficientGumbelSoftmaxSelector : Hard winner-take-all with full differentiability.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BaseSelector(nn.Module):
    """
    Abstract base selector class for GWT attention selection mechanisms.
    All selectors must inherit from this module.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        keys: torch.Tensor,
        query: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes attentional selection weights over multiple module proposals.

        Args:
            keys  : Tensor [B, num_modules, key_dim] — proposal keys.
            query : Optional tensor [B, key_dim]     — top-down bias query.

        Returns:
            Attention weights tensor [B, num_modules].
        """
        raise NotImplementedError("Subclasses must implement forward")


class CosineSimilaritySelector(BaseSelector):
    """
    Highly efficient selector using vectorized cosine similarity to score
    incoming states against a top-down query vector.
    """

    def __init__(self, key_dim: int, temperature: float = 1.0) -> None:
        """
        Args:
            key_dim     : Dimensionality of keys used for attention.
            temperature : Softmax temperature scaling parameter.
        """
        super().__init__()
        self.key_dim = key_dim
        self.temperature = temperature

    def forward(
        self,
        keys: torch.Tensor,
        query: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # keys  : [B, num_modules, key_dim]
        # query : [B, key_dim]
        batch_size = keys.shape[0]

        if query is None:
            query = torch.zeros(
                batch_size, self.key_dim, device=keys.device, dtype=keys.dtype
            )

        # q_proj: [B, 1, key_dim]  —  broadcast cosine similarity over all modules
        q_proj = query.unsqueeze(1)
        scores = F.cosine_similarity(keys, q_proj, dim=-1)  # [B, num_modules]

        if self.temperature != 1.0:
            scores = scores / self.temperature

        return F.softmax(scores, dim=-1)


class VectorizedCrossAttentionSelector(BaseSelector):
    """
    Performance-tuned multi-head cross-attention using PyTorch's native
    ``scaled_dot_product_attention`` (unlocks FlashAttention on supported hardware).

    By using an identity matrix as the Value tensor, it computes pure attention
    weights in a single vectorized pass — fully differentiable, zero extra copies.
    """

    def __init__(self, key_dim: int, num_heads: int = 4) -> None:
        """
        Args:
            key_dim   : Dimensionality of keys (must be divisible by num_heads).
            num_heads : Number of attention heads.
        """
        super().__init__()
        self.key_dim = key_dim
        self.num_heads = num_heads

        if key_dim % num_heads != 0:
            raise ValueError(
                f"key_dim ({key_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.head_dim = key_dim // num_heads
        self.q_proj = nn.Linear(key_dim, key_dim, bias=False)
        self.k_proj = nn.Linear(key_dim, key_dim, bias=False)

    def forward(
        self,
        keys: torch.Tensor,
        query: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # keys  : [B, num_modules, key_dim]
        # query : [B, key_dim]
        batch_size, num_modules, _ = keys.shape

        if query is None:
            query = torch.zeros(
                batch_size, self.key_dim, device=keys.device, dtype=keys.dtype
            )

        # Project & reshape Query: [B, 1, key_dim] -> [B, H, 1, head_dim]
        q = self.q_proj(query).unsqueeze(1)
        q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # Project & reshape Keys: [B, num_modules, key_dim] -> [B, H, num_modules, head_dim]
        k = self.k_proj(keys)
        k = k.view(batch_size, num_modules, self.num_heads, self.head_dim).transpose(
            1, 2
        )

        # Identity matrix as Value — avoids value mixing, extracts pure weights
        # [B, H, num_modules, num_modules]  (broadcast, no copy)
        v = torch.eye(num_modules, device=keys.device, dtype=keys.dtype)
        v = v.view(1, 1, num_modules, num_modules).expand(
            batch_size, self.num_heads, -1, -1
        )

        # Native scaled dot-product attention: output [B, H, 1, num_modules]
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        # Average over heads, squeeze sequence dim -> [B, num_modules]
        return attn_out.squeeze(2).mean(dim=1)


class EfficientGumbelSoftmaxSelector(BaseSelector):
    """
    Implements a differentiable hard winner-take-all routing mechanism using
    Gumbel-Softmax, selecting a single winning module per step while preserving
    full gradient flow.
    """

    def __init__(self, key_dim: int, tau: float = 1.0, hard: bool = True) -> None:
        """
        Args:
            key_dim : Dimensionality of keys.
            tau     : Gumbel-Softmax temperature parameter.
            hard    : If True, performs hard argmax in forward while keeping gradients.
        """
        super().__init__()
        self.key_dim = key_dim
        self.tau = tau
        self.hard = hard
        self.query_proj = nn.Linear(key_dim, key_dim, bias=False)

    def forward(
        self,
        keys: torch.Tensor,
        query: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # keys  : [B, num_modules, key_dim]
        # query : [B, key_dim]
        batch_size, num_modules, _ = keys.shape

        if query is None:
            query = torch.zeros(
                batch_size, self.key_dim, device=keys.device, dtype=keys.dtype
            )

        # Project query and compute dot-product logits: [B, num_modules]
        q_proj = self.query_proj(query).unsqueeze(-1)  # [B, key_dim, 1]
        logits = torch.matmul(keys, q_proj).squeeze(-1)  # [B, num_modules]
        logits = logits / (self.key_dim**0.5)

        return F.gumbel_softmax(logits, tau=self.tau, hard=self.hard, dim=-1)
