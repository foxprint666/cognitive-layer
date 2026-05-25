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

    def forward(self, keys: torch.Tensor, query: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes attentional selection weights over multiple module proposals.

        Args:
            keys: Tensor of shape [B, num_modules, key_dim] representing proposal keys.
            query: Optional query tensor of shape [B, key_dim] (top-down bias).

        Returns:
            Attention weights tensor of shape [B, num_modules].
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
            key_dim: Dimensionality of keys used for attention computation.
            temperature: Softmax temperature scaling parameter.
        """
        super().__init__()
        self.key_dim = key_dim
        self.temperature = temperature

    def forward(self, keys: torch.Tensor, query: Optional[torch.Tensor] = None) -> torch.Tensor:
        # keys shape: [B, num_modules, key_dim]
        # query shape: [B, key_dim]
        batch_size = keys.shape[0]

        if query is None:
            # Default to a zero tensor if no query is provided
            query = torch.zeros(batch_size, self.key_dim, device=keys.device, dtype=keys.dtype)

        # Vectorized cosine similarity with broadcasting
        # q_proj: shape [B, 1, key_dim]
        q_proj = query.unsqueeze(1)
        
        # Calculate cosine similarity along key_dim: [B, num_modules]
        scores = F.cosine_similarity(keys, q_proj, dim=-1)

        # Scale by temperature and apply softmax
        if self.temperature != 1.0:
            scores = scores / self.temperature

        return F.softmax(scores, dim=-1)


class VectorizedCrossAttentionSelector(BaseSelector):
    """
    A performance-tuned multi-head cross-attention layer using PyTorch's native
    optimized scaled_dot_product_attention. By using an identity matrix as the Value tensor,
    it computes FlashAttention-compatible attention weights in a single vectorized pass.
    """

    def __init__(self, key_dim: int, num_heads: int = 4) -> None:
        """
        Args:
            key_dim: Dimensionality of keys (must be divisible by num_heads).
            num_heads: Number of attention heads.
        """
        super().__init__()
        self.key_dim = key_dim
        self.num_heads = num_heads
        
        if key_dim % num_heads != 0:
            raise ValueError(f"key_dim ({key_dim}) must be divisible by num_heads ({num_heads}).")
            
        self.head_dim = key_dim // num_heads

        # Key and Query linear projections
        self.q_proj = nn.Linear(key_dim, key_dim, bias=False)
        self.k_proj = nn.Linear(key_dim, key_dim, bias=False)

    def forward(self, keys: torch.Tensor, query: Optional[torch.Tensor] = None) -> torch.Tensor:
        # keys shape: [B, num_modules, key_dim]
        # query shape: [B, key_dim]
        batch_size, num_modules, _ = keys.shape

        if query is None:
            query = torch.zeros(batch_size, self.key_dim, device=keys.device, dtype=keys.dtype)

        # 1. Project and reshape Query: [B, 1, key_dim] -> [B, H, 1, head_dim]
        q = self.q_proj(query).unsqueeze(1)
        q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Project and reshape Keys: [B, num_modules, key_dim] -> [B, H, num_modules, head_dim]
        k = self.k_proj(keys)
        k = k.view(batch_size, num_modules, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Create broadcastable Identity matrix for Value: [B, H, num_modules, num_modules]
        # Using eye and view to prevent redundant allocations, broadcasting to batch and heads
        v = torch.eye(num_modules, device=keys.device, dtype=keys.dtype)
        v = v.view(1, 1, num_modules, num_modules).expand(batch_size, self.num_heads, -1, -1)

        # 4. Compute native scaled dot product attention.
        # Output shape: [B, H, 1, num_modules]
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        # 5. Average weights across heads and squeeze sequence dimension: [B, num_modules]
        return attn_out.squeeze(2).mean(dim=1)


class EfficientGumbelSoftmaxSelector(BaseSelector):
    """
    Implements a differentiable, hard winner-take-all routing mechanism using
    Gumbel-Softmax selection to pick a single winning module.
    """

    def __init__(self, key_dim: int, tau: float = 1.0, hard: bool = True) -> None:
        """
        Args:
            key_dim: Dimensionality of keys.
            tau: Gumbel-Softmax temperature parameter.
            hard: If True, performs hard argmax in forward pass while keeping gradients.
        """
        super().__init__()
        self.key_dim = key_dim
        self.tau = tau
        self.hard = hard
        self.query_proj = nn.Linear(key_dim, key_dim, bias=False)

    def forward(self, keys: torch.Tensor, query: Optional[torch.Tensor] = None) -> torch.Tensor:
        # keys shape: [B, num_modules, key_dim]
        # query shape: [B, key_dim]
        batch_size, num_modules, _ = keys.shape

        if query is None:
            query = torch.zeros(batch_size, self.key_dim, device=keys.device, dtype=keys.dtype)

        # Project query and perform dot-product scores (logits)
        # q_proj: shape [B, key_dim, 1]
        q_proj = self.query_proj(query).unsqueeze(-1)
        
        # Calculate dot product logits: [B, num_modules]
        logits = torch.matmul(keys, q_proj).squeeze(-1)
        
        # Scale scores to avoid exploding/vanishing gradients
        logits = logits / (self.key_dim**0.5)

        # Single-pass Gumbel-Softmax selection
        return F.gumbel_softmax(logits, tau=self.tau, hard=self.hard, dim=-1)
