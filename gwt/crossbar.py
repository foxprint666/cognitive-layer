"""
gwt/crossbar.py
===============
The Cross-Modal Cognitive Crossbar for GWT (Phase v0.7).

Implements a parallel distributed routing bus that allows concurrent cross-modal
message passing and dynamic binding between separate registered modules using
an optimized multi-slot attention routing crossbar.
"""

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .engine import ModuleAdapter

logger = logging.getLogger(__name__)


class CognitiveCrossbar(nn.Module):
    """
    CognitiveCrossbar Layer.
    
    A parallel distributed routing bus that implements an optimized multi-slot
    attention routing crossbar to enable all-to-all cross-modal dynamic binding.
    """

    def __init__(
        self,
        slot_dim: int,
        num_slots: int,
        slot_names: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            slot_dim   : Dimension of the parallel routing slots.
            num_slots  : Number of parallel concurrent routing slot lines.
            slot_names : Optional human-readable names for the slot indices.
        """
        super().__init__()
        self.slot_dim = slot_dim
        self.num_slots = num_slots

        # Query, Key, and Value projection layers for concurrent cross-attention
        self.q_proj = nn.Linear(slot_dim, slot_dim, bias=False)
        self.k_proj = nn.Linear(slot_dim, slot_dim, bias=False)
        self.v_proj = nn.Linear(slot_dim, slot_dim, bias=False)

        # Parameterized binding affinity matrix: [num_slots, num_slots]
        self.binding_weight = nn.Parameter(torch.ones(num_slots, num_slots))

        if slot_names is not None:
            if len(slot_names) != num_slots:
                raise ValueError(
                    f"slot_names length ({len(slot_names)}) must match num_slots ({num_slots})."
                )
            self.slot_names = list(slot_names)
        else:
            self.slot_names = [f"Slot {i}" for i in range(num_slots)]

        # Dynamic state buffers
        self._stacked_latents: Optional[torch.Tensor] = None
        self.last_weights: Optional[torch.Tensor] = None

        # Engine reference
        self.engine: Optional[Any] = None

    def write_slot(self, slot_idx: int, latent: torch.Tensor) -> None:
        """Writes a latent vector directly into a designated slot index."""
        batch_size = latent.shape[0]
        device = latent.device
        dtype = latent.dtype

        # Lazy initialize or resize the internal stacked latent tensor
        if (
            self._stacked_latents is None
            or self._stacked_latents.shape[0] != batch_size
            or self._stacked_latents.device != device
            or self._stacked_latents.dtype != dtype
        ):
            self._stacked_latents = torch.zeros(
                batch_size, self.num_slots, self.slot_dim, device=device, dtype=dtype
            )

        self._stacked_latents[:, slot_idx] = latent

    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Vectorized Parallel Routing.
        
        Args:
            x : Optional [B, num_slots, slot_dim] stacked source latents.
                If None, uses the internally stacked latents written by adapters.
                
        Returns:
            [B, num_slots, slot_dim] routed crossbar representations.
        """
        if x is None:
            if self._stacked_latents is None:
                raise ValueError("No latents written to CognitiveCrossbar. Run forward pass first.")
            x = self._stacked_latents

        batch_size, num_slots, slot_dim = x.shape

        # 1. Parallel projections: [B, num_slots, slot_dim] -> [B, num_slots, slot_dim]
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 2. All-to-all cross-attention scoring
        # scores shape: [B, num_slots, num_slots]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (slot_dim ** 0.5)
        weights = F.softmax(scores, dim=-1)

        # 3. Dynamic scaling of binding weights driven by global neuromodulators (if attached)
        ach = 1.0
        if self.engine is not None and self.engine.neuromodulator is not None:
            ach = getattr(self.engine.neuromodulator, "ach", 1.0)

        # Multiply attention weights by the parameterized binding affinity matrix
        # binding_weight is [num_slots, num_slots], expanded to [1, num_slots, num_slots]
        # and dynamically scaled by ACh to increase/decrease crossbar throughput
        scaled_binding = self.binding_weight.unsqueeze(0) * ach
        routed_weights = weights * scaled_binding

        # Save detached weights for connectivity diagnostics
        self.last_weights = routed_weights.detach()

        # 4. Integrate contextual broadcast across modalities
        # output shape: [B, num_slots, slot_dim]
        routed_latents = torch.matmul(routed_weights, V)

        return routed_latents


class CrossbarModuleAdapter(ModuleAdapter):
    """
    CrossbarModuleAdapter.
    
    Extends ModuleAdapter to support writing to and reading from a dedicated
    slot line on the CognitiveCrossbar.
    """

    def __init__(
        self,
        name: str,
        module: nn.Module,
        latent_dim: int,
        data_flow: Any,
        slot_idx: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            name       : Unique name of the module adapter.
            module     : The target PyTorch module to wrap.
            latent_dim : Representation dimension.
            data_flow  : Active DataFlowManager instance.
            slot_idx   : Dedicated crossbar slot index.
            device     : The PyTorch device context.
            dtype      : The PyTorch dtype context.
        """
        super().__init__(
            name=name,
            module=module,
            latent_dim=latent_dim,
            data_flow=data_flow,
            device=device,
            dtype=dtype,
            **kwargs,
        )
        self.slot_idx = slot_idx
        self.engine: Optional[Any] = None

        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

    def _register_forward_hook(self) -> None:
        """Registers a hook that intercepts latent representations and writes to the crossbar."""

        def hook_fn(module: nn.Module, inputs: Any, outputs: Any) -> Any:
            latent = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

            if not isinstance(latent, torch.Tensor):
                raise TypeError(
                    f"Expected tensor output from module '{self.name}', got {type(outputs)}"
                )

            # Dimension Agnosticism: Pool varying ranks (3D sequence, 4D vision) to flat 2D
            from .salience import global_pool_latent
            latent = global_pool_latent(latent)

            # Optional projection layer
            if self.projection is not None:
                latent = self.projection(latent)

            # Write raw latent directly into designated crossbar slot
            if self.engine is not None and self.engine.crossbar is not None:
                self.engine.crossbar.write_slot(self.slot_idx, latent)

            self.data_flow.update_buffer(self.name, latent)
            return outputs

        self._hook_handle = self.module.register_forward_hook(hook_fn)
        logger.debug(f"Crossbar forward hook successfully registered on module '{self.name}'.")
