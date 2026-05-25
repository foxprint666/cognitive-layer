import logging
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class AttentionSelector(nn.Module):
    """
    Computes attentional selection weights over multiple module proposals.
    Supports Top-Down Key-Query matching and Bottom-Up salience scoring.
    Implements a non-linear "ignition" threshold to simulate all-or-none conscious access.
    """

    def __init__(
        self,
        key_dim: int,
        attention_type: str = "key-query",
        ignition_threshold: float = 0.0,
    ) -> None:
        """
        Args:
            key_dim: Dimensionality of keys used for attention computation.
            attention_type: Type of attention to compute ('key-query' or 'salience').
            ignition_threshold: Attentional weight threshold. Modules below this
                                threshold are suppressed (masked out).
        """
        super().__init__()
        self.key_dim = key_dim
        self.attention_type = attention_type
        self.ignition_threshold = ignition_threshold

        # For bottom-up salience: learns to evaluate a module's key directly to compute attention scores
        if attention_type == "salience":
            self.salience_proj = nn.Linear(key_dim, 1, bias=False)
        else:
            # For key-query: learns to project a query vector matching keys
            self.query_proj = nn.Linear(key_dim, key_dim, bias=False)

    def forward(
        self, keys: torch.Tensor, query: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            keys: Tensor of shape [B, num_modules, key_dim] representing proposals' keys.
            query: Optional query tensor of shape [B, key_dim] (required for 'key-query' mode).
            
        Returns:
            Attention weights tensor of shape [B, num_modules].
        """
        batch_size, num_modules, key_dim = keys.shape

        if self.attention_type == "salience":
            # Bottom-up salience scores: project keys to scalars
            # Shape: [B, num_modules, 1] -> squeeze to [B, num_modules]
            scores = self.salience_proj(keys).squeeze(-1)
        else:
            # Top-down key-query matching
            if query is None:
                # Default to matching against a zero query if none provided
                query = torch.zeros(batch_size, key_dim, device=keys.device)

            # Project query and compute dot product similarity
            # query_proj(query): shape [B, key_dim] -> unsqueeze to [B, key_dim, 1]
            q_proj = self.query_proj(query).unsqueeze(-1)
            # keys: shape [B, num_modules, key_dim]
            # scores = matmul(keys, q_proj): shape [B, num_modules, 1] -> squeeze to [B, num_modules]
            scores = torch.matmul(keys, q_proj).squeeze(-1)

        # Scale scores to prevent vanishing/exploding gradients
        scores = scores / (key_dim**0.5)

        # Compute initial softmax weights
        weights = F.softmax(scores, dim=-1)

        # Apply GWT Ignition threshold gating (non-linear thresholding)
        if self.ignition_threshold > 0.0:
            # Mask out weights below threshold
            mask = (weights >= self.ignition_threshold).float()
            # If all modules fall below threshold, keep them as is (prevent divide-by-zero or complete silence)
            fallback = (mask.sum(dim=-1, keepdim=True) == 0).float()
            mask = mask + fallback
            mask = torch.clamp(mask, 0.0, 1.0)

            # Re-normalize masked weights
            weights = (weights * mask) / (weights * mask).sum(dim=-1, keepdim=True).clamp(min=1e-9)

        return weights


class BroadcastEngine(nn.Module):
    """
    Translates selected module latents into the unified global workspace state
    and orchestrates distribution back to module adapters.
    Supports multi-slot working memories.
    """

    def __init__(
        self,
        latent_dim: int,
        workspace_slots: int = 1,
    ) -> None:
        """
        Args:
            latent_dim: Dimensionality of the global workspace representation.
            workspace_slots: Number of working memory slots in the workspace.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.workspace_slots = workspace_slots

        # Slot-based projections to compile modular information
        if workspace_slots > 1:
            self.slot_projs = nn.ModuleList(
                [nn.Linear(latent_dim, latent_dim) for _ in range(workspace_slots)]
            )
            # Aggregation linear layer to map slot states back to single broadcast vector
            self.aggregation = nn.Linear(latent_dim * workspace_slots, latent_dim)

    def forward(
        self, latents: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            latents: Tensor of shape [B, num_modules, latent_dim] containing all module latent vectors.
            weights: Tensor of shape [B, num_modules] containing attentional selection weights.
            
        Returns:
            Broadcasted global workspace state tensor of shape [B, latent_dim].
        """
        # latents: [B, num_modules, latent_dim]
        # weights: [B, num_modules] -> unsqueeze to [B, num_modules, 1]
        w = weights.unsqueeze(-1)

        if self.workspace_slots == 1:
            # Single slot GWT: compute weighted sum of modules
            # Shape: [B, latent_dim]
            broadcast_state = (latents * w).sum(dim=1)
            return broadcast_state
        else:
            # Multi-slot GWT: distribute attention weights to populate parallel slots
            # For simplicity, each slot uses parallel projections of the weighted combination
            # mimicking memory buffers holding distinct components
            slot_outputs = []
            base_mix = (latents * w).sum(dim=1)  # [B, latent_dim]
            for proj in self.slot_projs:
                slot_outputs.append(proj(base_mix))

            # Concatenate parallel slots and project back to latent space
            # Shape: [B, latent_dim * slots]
            concat_slots = torch.cat(slot_outputs, dim=-1)
            # Shape: [B, latent_dim]
            return self.aggregation(concat_slots)


class GlobalWorkspace(nn.Module):
    """
    Main Global Workspace module coordinating selection (AttentionSelector)
    and distribution (BroadcastEngine).
    """

    def __init__(
        self,
        latent_dim: int,
        key_dim: int = 64,
        attention_type: str = "key-query",
        selection_mode: str = "soft",
        ignition_threshold: float = 0.0,
        workspace_slots: int = 1,
    ) -> None:
        """
        Args:
            latent_dim: Dimensionality of the global workspace representation.
            key_dim: Dimensionality of module keys for attention matching.
            attention_type: Type of attention ('key-query' or 'salience').
            selection_mode: Selection mode ('soft' for weighted average, 'hard' for argmax straight-through).
            ignition_threshold: Attentional weight threshold for non-linear ignition.
            workspace_slots: Number of parallel working memory slots.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.key_dim = key_dim
        self.selection_mode = selection_mode

        self.selector = AttentionSelector(
            key_dim=key_dim,
            attention_type=attention_type,
            ignition_threshold=ignition_threshold,
        )

        self.broadcaster = BroadcastEngine(
            latent_dim=latent_dim,
            workspace_slots=workspace_slots,
        )

        # Learnable global query representation for top-down key-query matching
        self.global_query = nn.Parameter(torch.randn(1, key_dim))

    def forward(
        self,
        latent_states: Dict[str, torch.Tensor],
        keys: Dict[str, torch.Tensor],
        custom_query: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            latent_states: Dict of name -> tensor [B, latent_dim] representing modules' states.
            keys: Dict of name -> tensor [B, key_dim] representing modules' key signals.
            custom_query: Optional external query tensor of shape [B, key_dim] for dynamic top-down selection.
            
        Returns:
            Broadcasted global workspace tensor of shape [B, latent_dim].
        """
        names = list(latent_states.keys())
        device = next(self.parameters()).device

        from .salience import global_pool_latent

        # Stack latent states and keys after applying global pooling to ensure shape compatibility
        # latents: [B, num_modules, latent_dim]
        # keys: [B, num_modules, key_dim]
        latents_list = [global_pool_latent(latent_states[name]).to(device) for name in names]
        keys_list = [global_pool_latent(keys[name]).to(device) for name in names]

        latents_stacked = torch.stack(latents_list, dim=1)
        keys_stacked = torch.stack(keys_list, dim=1)

        batch_size = latents_stacked.shape[0]

        # Determine attention query (custom query or global learned query parameter)
        if custom_query is not None:
            query = custom_query.to(device)
        else:
            # Expand global query to match batch size
            query = self.global_query.expand(batch_size, -1)

        # Compute selection weights: [B, num_modules]
        weights = self.selector(keys_stacked, query)

        # Apply selection mode
        if self.selection_mode == "hard":
            # Hard GWT: select only the single winning module proposal
            # Incorporates a straight-through estimator to preserve continuous backpropagation gradients
            winner_idx = torch.argmax(weights, dim=-1)
            one_hot = F.one_hot(winner_idx, num_classes=len(names)).to(weights.dtype)
            
            # Straight-through gradient trick
            # In forward pass, it acts as one_hot. In backward pass, it propagates gradients through weights.
            selection_weights = one_hot + weights - weights.detach()
        else:
            # Soft GWT: continuous weighted average mixture
            selection_weights = weights

        # Broadcast the selected representation
        broadcast_state = self.broadcaster(latents_stacked, selection_weights)

        return broadcast_state
