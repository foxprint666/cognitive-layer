"""
gwt/engine.py
=============
Core Global Workspace Theory (GWT) infrastructure — self-contained module.

Merges what was previously split across three files (engine.py, adapters.py,
gwt.py) into one coherent unit, eliminating all circular imports.

Contains
--------
ModuleRegistry      : Tracks registered cognitive module adapters.
DataFlowManager     : High-performance latent-state routing buffers.
ModuleAdapter       : Non-intrusive PyTorch forward-hook wrapper.
AttentionSelector   : Key-query / salience attention with ignition gating.
BroadcastEngine     : Weighted-sum broadcaster (single or multi-slot workspace).
GlobalWorkspace     : Top-level GWT module coordinating selection + broadcast.
CognitiveAugEngine  : Main orchestrator tying all components together.
"""
import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Registry & Data Flow
# ─────────────────────────────────────────────────────────────────────────────

class ModuleRegistry:
    """Registry for managing brain-inspired cognitive modules and their adapters."""

    def __init__(self) -> None:
        self._adapters: Dict[str, Any] = {}

    def register(self, name: str, adapter: Any) -> None:
        """Register a ModuleAdapter with a unique name."""
        if name in self._adapters:
            logger.warning(f"Overwriting already registered module adapter: {name}")
        self._adapters[name] = adapter
        logger.debug(f"Successfully registered module: {name}")

    def get(self, name: str) -> Any:
        """Retrieve a registered ModuleAdapter by name."""
        if name not in self._adapters:
            raise KeyError(f"Module '{name}' is not registered.")
        return self._adapters[name]

    def list_names(self) -> List[str]:
        """List names of all registered modules."""
        return list(self._adapters.keys())

    def list_adapters(self) -> List[Any]:
        """List all registered ModuleAdapter instances."""
        return list(self._adapters.values())

    def clear(self) -> None:
        """Clear the registry."""
        self._adapters.clear()


class DataFlowManager:
    """
    Manages communication buffers, dynamic latent spaces, and shapes of all
    registered modules. Ensures high-performance tensor transfers and routing.
    """

    def __init__(self) -> None:
        self._buffers: Dict[str, torch.Tensor] = {}

    def update_buffer(self, name: str, tensor: torch.Tensor) -> None:
        """Update the latent space buffer for a specific module."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Buffer content must be a PyTorch Tensor.")
        self._buffers[name] = tensor

    def get_buffer(self, name: str) -> torch.Tensor:
        """Retrieve the latent space buffer for a specific module."""
        if name not in self._buffers:
            raise KeyError(
                f"No latent buffer found for module '{name}'. "
                "Ensure the module has run a forward pass."
            )
        return self._buffers[name]

    def list_buffers(self) -> Dict[str, torch.Tensor]:
        """Get all current module latent buffers."""
        return self._buffers

    def clear_buffers(self) -> None:
        """Clear all cached latent state buffers."""
        self._buffers.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Module Adapter
# ─────────────────────────────────────────────────────────────────────────────

class ModuleAdapter(nn.Module):
    """
    Adapter wrapper that hooks into any PyTorch nn.Module.
    Automatically intercepts latent representations via forward hooks and routes
    them through the DataFlowManager — without altering the original module.
    """

    def __init__(
        self,
        name: str,
        module: nn.Module,
        latent_dim: int,
        data_flow: DataFlowManager,
        key_dim: int = 64,
        projection_in_dim: Optional[int] = None,
    ) -> None:
        """
        Args:
            name             : Unique name of the module.
            module           : The PyTorch nn.Module to wrap.
            latent_dim       : The workspace latent representation dimension.
            data_flow        : The active DataFlowManager instance.
            key_dim          : Dimensionality of module keys for attention matching.
            projection_in_dim: If the module's raw output dimension does not match
                               latent_dim, specify it here to auto-build a learnable
                               projection layer.
        """
        super().__init__()
        self.name = name
        self.module = module
        self.latent_dim = latent_dim
        self.data_flow = data_flow
        self.key_dim = key_dim

        # Optional learnable projection: raw_output_dim -> latent_dim
        self.projection: Optional[nn.Linear] = None
        if projection_in_dim is not None and projection_in_dim != latent_dim:
            self.projection = nn.Linear(projection_in_dim, latent_dim)
            logger.info(
                f"Created auto-projection layer for module '{name}': "
                f"{projection_in_dim} -> {latent_dim}"
            )

        # Key projection: maps latent state -> attention key space
        self.key_proj = nn.Linear(latent_dim, key_dim)

        # Buffer storing the latest broadcasted workspace state for this module
        self.register_buffer("last_broadcast", torch.zeros(1, latent_dim))

        self._hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._register_forward_hook()

    def _register_forward_hook(self) -> None:
        """Register a PyTorch forward hook to intercept latent outputs."""

        def hook_fn(module: nn.Module, inputs: Any, outputs: Any) -> Any:
            # Unwrap tuple/list outputs (e.g. LSTM hidden states)
            if isinstance(outputs, (tuple, list)):
                latent = outputs[0]
            else:
                latent = outputs

            if not isinstance(latent, torch.Tensor):
                raise TypeError(
                    f"Expected tensor output from module '{self.name}', "
                    f"got {type(outputs)}"
                )

            if self.projection is not None:
                latent = self.projection(latent)

            self.data_flow.update_buffer(self.name, latent)
            return outputs  # always return original outputs unmodified

        self._hook_handle = self.module.register_forward_hook(hook_fn)
        logger.debug(f"Forward hook successfully registered on module '{self.name}'.")

    def get_key(self, latent: torch.Tensor) -> torch.Tensor:
        """Generate a key vector representing the current module state."""
        return self.key_proj(latent)

    def receive_broadcast(self, broadcast_state: torch.Tensor) -> None:
        """Called by CognitiveAugEngine to deliver the latest workspace broadcast."""
        self.last_broadcast = broadcast_state.clone()
        logger.debug(f"Module '{self.name}' received workspace broadcast.")

    def get_last_broadcast(self) -> torch.Tensor:
        """Retrieve the latest broadcasted state from the workspace."""
        return self.last_broadcast

    def remove_hooks(self) -> None:
        """Clean up registered forward hooks."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
            logger.debug(f"Removed forward hook from module '{self.name}'.")

    def __del__(self) -> None:
        self.remove_hooks()


# ─────────────────────────────────────────────────────────────────────────────
# Attention Selector
# ─────────────────────────────────────────────────────────────────────────────

class AttentionSelector(nn.Module):
    """
    Computes attentional selection weights over multiple module proposals.

    Supports:
      - ``'key-query'``  : Top-down key-query dot-product matching.
      - ``'salience'``   : Bottom-up learned salience scoring.

    Implements a non-linear ignition threshold to simulate all-or-none
    conscious access (GWT ignition).
    """

    def __init__(
        self,
        key_dim: int,
        attention_type: str = "key-query",
        ignition_threshold: float = 0.0,
    ) -> None:
        """
        Args:
            key_dim            : Dimensionality of keys used for attention.
            attention_type     : ``'key-query'`` or ``'salience'``.
            ignition_threshold : Weights below this value are suppressed.
        """
        super().__init__()
        self.key_dim = key_dim
        self.attention_type = attention_type
        self.ignition_threshold = ignition_threshold

        if attention_type == "salience":
            self.salience_proj = nn.Linear(key_dim, 1, bias=False)
        else:
            self.query_proj = nn.Linear(key_dim, key_dim, bias=False)

    def forward(
        self,
        keys: torch.Tensor,
        query: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            keys  : [B, num_modules, key_dim] — proposal keys.
            query : [B, key_dim]              — optional top-down query.

        Returns:
            Attention weights [B, num_modules].
        """
        batch_size, num_modules, key_dim = keys.shape

        if self.attention_type == "salience":
            scores = self.salience_proj(keys).squeeze(-1)
        else:
            if query is None:
                query = torch.zeros(batch_size, key_dim, device=keys.device)
            q_proj = self.query_proj(query).unsqueeze(-1)          # [B, key_dim, 1]
            scores = torch.matmul(keys, q_proj).squeeze(-1)        # [B, num_modules]

        scores = scores / (key_dim ** 0.5)
        weights = F.softmax(scores, dim=-1)

        if self.ignition_threshold > 0.0:
            mask = (weights >= self.ignition_threshold).float()
            fallback = (mask.sum(dim=-1, keepdim=True) == 0).float()
            mask = torch.clamp(mask + fallback, 0.0, 1.0)
            weights = (weights * mask) / (
                (weights * mask).sum(dim=-1, keepdim=True).clamp(min=1e-9)
            )

        return weights


# ─────────────────────────────────────────────────────────────────────────────
# Broadcast Engine
# ─────────────────────────────────────────────────────────────────────────────

class BroadcastEngine(nn.Module):
    """
    Translates selected module latents into the unified global workspace state.
    Supports single-slot (biologically faithful) and multi-slot (engineering) modes.
    """

    def __init__(self, latent_dim: int, workspace_slots: int = 1) -> None:
        """
        Args:
            latent_dim      : Dimensionality of the global workspace.
            workspace_slots : Number of parallel working memory slots.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.workspace_slots = workspace_slots

        if workspace_slots > 1:
            self.slot_projs = nn.ModuleList(
                [nn.Linear(latent_dim, latent_dim) for _ in range(workspace_slots)]
            )
            self.aggregation = nn.Linear(latent_dim * workspace_slots, latent_dim)

    def forward(self, latents: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latents : [B, num_modules, latent_dim]
            weights : [B, num_modules]

        Returns:
            Broadcasted workspace state [B, latent_dim].
        """
        w = weights.unsqueeze(-1)   # [B, num_modules, 1]

        if self.workspace_slots == 1:
            return (latents * w).sum(dim=1)

        base_mix = (latents * w).sum(dim=1)     # [B, latent_dim]
        slot_outputs = [proj(base_mix) for proj in self.slot_projs]
        return self.aggregation(torch.cat(slot_outputs, dim=-1))


# ─────────────────────────────────────────────────────────────────────────────
# Global Workspace
# ─────────────────────────────────────────────────────────────────────────────

class GlobalWorkspace(nn.Module):
    """
    Main Global Workspace module coordinating attention selection (AttentionSelector)
    and global broadcast (BroadcastEngine).

    The selector can be hot-swapped at runtime since it is a standard nn.Module::

        workspace.selector = VectorizedCrossAttentionSelector(key_dim=64, num_heads=4)
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
        # Learnable global query for top-down key-query matching
        self.global_query = nn.Parameter(torch.randn(1, key_dim))

    def forward(
        self,
        latent_states: Dict[str, torch.Tensor],
        keys: Dict[str, torch.Tensor],
        custom_query: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            latent_states : name -> [B, latent_dim] module state tensors.
            keys          : name -> [B, key_dim]    module key tensors.
            custom_query  : Optional [B, key_dim] external top-down query.

        Returns:
            Broadcasted global workspace tensor [B, latent_dim].
        """
        # Lazy relative import — keeps salience.py decoupled from this file
        from .salience import global_pool_latent

        names = list(latent_states.keys())
        device = next(self.parameters()).device

        latents_stacked = torch.stack(
            [global_pool_latent(latent_states[n]).to(device) for n in names], dim=1
        )   # [B, num_modules, latent_dim]

        keys_stacked = torch.stack(
            [global_pool_latent(keys[n]).to(device) for n in names], dim=1
        )   # [B, num_modules, key_dim]

        batch_size = latents_stacked.shape[0]
        query = (
            custom_query.to(device)
            if custom_query is not None
            else self.global_query.expand(batch_size, -1)
        )

        weights = self.selector(keys_stacked, query)    # [B, num_modules]

        if self.selection_mode == "hard":
            # Straight-through estimator: hard selection in forward, soft gradients
            winner_idx = torch.argmax(weights, dim=-1)
            one_hot = F.one_hot(winner_idx, num_classes=len(names)).to(weights.dtype)
            selection_weights = one_hot + weights - weights.detach()
        else:
            selection_weights = weights

        return self.broadcaster(latents_stacked, selection_weights)


# ─────────────────────────────────────────────────────────────────────────────
# Cognitive Aug Engine  (top-level orchestrator)
# ─────────────────────────────────────────────────────────────────────────────

class CognitiveAugEngine:
    """
    The main orchestrator engine of the cognitive augmentation library.

    Manages module lifecycles, registries, data flows, and the GWT communication
    cycle (selection -> broadcast -> distribute).
    """

    def __init__(self) -> None:
        self.registry: ModuleRegistry = ModuleRegistry()
        self.data_flow: DataFlowManager = DataFlowManager()
        self.workspace: Optional[nn.Module] = None

    def register_module(
        self,
        name: str,
        module: nn.Module,
        latent_dim: int,
        **kwargs: Any,
    ) -> ModuleAdapter:
        """
        Wrap an existing PyTorch nn.Module with a ModuleAdapter and register it.

        Args:
            name      : Unique identifier for the module.
            module    : The PyTorch module to wrap.
            latent_dim: Dimension of the latent space for this module.

        Returns:
            The created ModuleAdapter instance.
        """
        adapter = ModuleAdapter(
            name=name,
            module=module,
            latent_dim=latent_dim,
            data_flow=self.data_flow,
            **kwargs,
        )
        self.registry.register(name, adapter)
        return adapter

    def attach_workspace(self, workspace: nn.Module) -> None:
        """Attach a GlobalWorkspace to orchestrate attention and broadcasting."""
        self.workspace = workspace
        logger.info("Successfully attached workspace to CognitiveAugEngine.")
        
        # Dynamically align registered adapters' key_proj if their key_dim doesn't match the workspace's key_dim
        if hasattr(workspace, "key_dim"):
            target_key_dim = workspace.key_dim
            for adapter in self.registry.list_adapters():
                if hasattr(adapter, "key_dim") and adapter.key_dim != target_key_dim:
                    logger.info(
                        f"Re-projecting key space for module '{adapter.name}' from {adapter.key_dim} to {target_key_dim} "
                        "to match the attached workspace."
                    )
                    adapter.key_dim = target_key_dim
                    # Recreate key_proj linear layer with new key_dim
                    device = next(adapter.parameters()).device if list(adapter.parameters()) else torch.device("cpu")
                    adapter.key_proj = nn.Linear(adapter.latent_dim, target_key_dim).to(device)

    def step(self) -> torch.Tensor:
        """
        Perform one full GWT cycle:
        1. Collect latent states from every registered module's buffer.
        2. Feed states + keys to the GlobalWorkspace (selection & broadcast).
        3. Distribute the broadcast state back to all ModuleAdapters.

        Returns:
            Broadcasted workspace state tensor [B, latent_dim].
        """
        if self.workspace is None:
            raise ValueError(
                "No workspace attached. Call `attach_workspace` before stepping."
            )

        adapters = self.registry.list_adapters()
        if not adapters:
            raise ValueError("No modules registered with the engine.")

        latent_states: Dict[str, torch.Tensor] = {}
        keys: Dict[str, torch.Tensor] = {}

        for adapter in adapters:
            try:
                latent = self.data_flow.get_buffer(adapter.name)
                latent_states[adapter.name] = latent
                keys[adapter.name] = adapter.get_key(latent)
            except KeyError:
                logger.warning(
                    f"Module '{adapter.name}' has not run a forward pass this step. "
                    "Falling back to zero latent vector."
                )
                batch_size = next(
                    (b.shape[0] for b in self.data_flow.list_buffers().values()), 1
                )
                device = (
                    next(adapter.module.parameters()).device
                    if list(adapter.module.parameters())
                    else torch.device("cpu")
                )
                latent = torch.zeros(batch_size, adapter.latent_dim, device=device)
                latent_states[adapter.name] = latent
                keys[adapter.name] = adapter.get_key(latent)

        broadcast_state = self.workspace(latent_states, keys)

        for adapter in adapters:
            adapter.receive_broadcast(broadcast_state)

        return broadcast_state
