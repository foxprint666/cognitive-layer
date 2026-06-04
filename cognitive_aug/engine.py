"""
cognitive_aug/engine.py
=============
Core Global Workspace Theory (GWT) infrastructure — self-contained module.

Merges what was previously split across three files (engine.py, adapters.py,
cognitive_aug.py) into one coherent unit, eliminating all circular imports.

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

from .exceptions import RegistryError, DeviceMismatchError

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
            raise RegistryError(f"Module '{name}' is not registered.")
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
    
    Delegates dynamically to a GWT StateStore (local or distributed Redis) to
    support horizontally scaled application pods safely.
    """

    def __init__(self, state_store: Optional[Any] = None) -> None:
        from .state import InMemoryStateStore
        self._state_store = state_store if state_store is not None else InMemoryStateStore()

    def update_buffer(self, name: str, tensor: torch.Tensor) -> None:
        """Update the latent space buffer for a specific module."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Buffer content must be a PyTorch Tensor.")
        self._state_store.update_buffer(name, tensor)

    def get_buffer(self, name: str) -> torch.Tensor:
        """Retrieve the latent space buffer for a specific module."""
        return self._state_store.get_buffer(name)

    def list_buffers(self) -> Dict[str, torch.Tensor]:
        """Get all current module latent buffers."""
        return self._state_store.list_buffers()

    def clear_buffers(self) -> None:
        """Clear all cached latent state buffers and saliences."""
        self._state_store.clear()

    def update_salience(self, name: str, score: float) -> None:
        """Update the salience score for a specific module."""
        self._state_store.update_salience(name, score)

    def get_salience(self, name: str) -> float:
        """Retrieve the salience score for a specific module."""
        return self._state_store.get_salience(name)

    def list_saliences(self) -> Dict[str, float]:
        """Get all current module saliences."""
        return self._state_store.list_saliences()

    def clear_saliences(self) -> None:
        """Clear all cached salience scores."""
        self._state_store.clear()


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
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
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
            device           : The PyTorch device to initialize on.
            dtype            : The PyTorch dtype to initialize with.
        """
        super().__init__()
        self.name = name
        self.module = module
        self.latent_dim = latent_dim
        self.data_flow = data_flow
        self.key_dim = key_dim

        factory_kwargs = {'device': device, 'dtype': dtype}

        # Optional learnable projection: raw_output_dim -> latent_dim
        self.projection: Optional[nn.Linear] = None
        if projection_in_dim is not None and projection_in_dim != latent_dim:
            self.projection = nn.Linear(projection_in_dim, latent_dim, **factory_kwargs)
            logger.info(
                f"Created auto-projection layer for module '{name}': "
                f"{projection_in_dim} -> {latent_dim}"
            )

        # Key projection: maps latent state -> attention key space
        self.key_proj = nn.Linear(latent_dim, key_dim, **factory_kwargs)

        # Buffer storing the latest broadcasted workspace state for this module
        self.register_buffer("last_broadcast", torch.zeros(1, latent_dim, **factory_kwargs))

        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

        self._hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._register_forward_hook()

    def _register_forward_hook(self) -> None:
        """Register a PyTorch forward hook to intercept latent outputs with strict fault tolerance."""

        def hook_fn(module: nn.Module, inputs: Any, outputs: Any) -> Any:
            try:
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

                # Dimension Agnosticism: Pool varying ranks (3D sequence, 4D vision) to flat 2D
                from .salience import global_pool_latent
                latent = global_pool_latent(latent)

                if self.projection is not None:
                    module_output_dtype = latent.dtype
                    try:
                        latent = latent.to(dtype=self.projection.weight.dtype, device=self.projection.weight.device)
                        latent = self.projection(latent)
                        latent = latent.to(dtype=module_output_dtype)
                    except Exception as e:
                        raise DeviceMismatchError(f"Device/Dtype mismatch during latent projection: {e}")

                self.data_flow.update_buffer(self.name, latent)
            except Exception as e:
                # Graceful fallback: log the error and bypass GWT, returning base outputs unmodified
                from .telemetry import get_telemetry_logger
                get_telemetry_logger().record_error(
                    error_msg=str(e),
                    phase="ModuleAdapter Forward Hook Interception",
                    details=f"Gracefully bypassing GWT for module '{self.name}'."
                )
            return outputs  # always return original outputs unmodified

        self._hook_handle = self.module.register_forward_hook(hook_fn)
        logger.debug(f"Forward hook successfully registered on module '{self.name}'.")

    def get_key(self, latent: torch.Tensor) -> torch.Tensor:
        """Generate a key vector representing the current module state."""
        return self.key_proj(latent)

    def receive_broadcast(self, broadcast_state: torch.Tensor) -> None:
        """Called by CognitiveAugEngine to deliver the latest workspace broadcast."""
        self.last_broadcast = broadcast_state.detach().clone()
        logger.debug(f"Module '{self.name}' received workspace broadcast.")

    def get_last_broadcast(self) -> torch.Tensor:
        """Retrieve the latest broadcasted state from the workspace."""
        return self.last_broadcast

    def remove_hooks(self) -> None:
        """Clean up registered forward hooks."""
        if getattr(self, "_hook_handle", None) is not None:
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

        try:
            latents_stacked = torch.stack(
                [global_pool_latent(latent_states[n]).to(device) for n in names], dim=1
            )   # [B, num_modules, latent_dim]

            keys_stacked = torch.stack(
                [global_pool_latent(keys[n]).to(device) for n in names], dim=1
            )   # [B, num_modules, key_dim]
        except Exception as e:
            raise DeviceMismatchError(f"Failed to stack latents/keys in workspace: {e}")

        batch_size = latents_stacked.shape[0]
        query = (
            custom_query.to(device)
            if custom_query is not None
            else self.global_query.expand(batch_size, -1)
        )

        weights = self.selector(keys_stacked, query)    # [B, num_modules]

        # Cache detached parameters for metacognitive neuromodulation telemetry
        self.last_weights = weights.detach()
        self.last_query = query.detach()
        self.last_keys = keys_stacked.detach()

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
        self.replay_buffer: Optional[Any] = None
        self.neuromodulator: Optional[Any] = None
        self.glial_manager: Optional[Any] = None
        self.concept_layers: Dict[str, Any] = {}
        self.crossbar: Optional[Any] = None
        self._step_counter: int = 0

    def attach_neuromodulator(self, monitor: Any) -> None:
        """Attach a MetacognitiveMonitor to dynamically tune GWT thresholds."""
        self.neuromodulator = monitor
        logger.info("Successfully attached MetacognitiveMonitor to CognitiveAugEngine.")

    def attach_glial_manager(self, manager: Any) -> None:
        """Attach an AstrocyteManager to dynamically track saliences and stabilize gradients."""
        self.glial_manager = manager
        manager.attach(self)
        logger.info("Successfully attached AstrocyteManager to CognitiveAugEngine.")

    def attach_concept_layer(self, name: str, layer: Any) -> None:
        """Attach a ConceptLayer to the engine."""
        self.concept_layers[name] = layer
        layer.name = name
        layer.data_flow = self.data_flow
        logger.info(f"Successfully attached ConceptLayer '{name}' to CognitiveAugEngine.")

    def attach_crossbar(self, crossbar: Any) -> None:
        """Attach a CognitiveCrossbar to dynamically route cross-modal states."""
        self.crossbar = crossbar
        crossbar.engine = self
        for adapter in self.registry.list_adapters():
            if hasattr(adapter, "slot_idx"):
                adapter.engine = self
        logger.info("Successfully attached CognitiveCrossbar to CognitiveAugEngine.")

    def inspect(self) -> str:
        """
        Builds a gorgeous diagnostic panel showing module states, GWT slot buffers,
        dendritic telemetry, and chemical bars.
        """
        # Lazy import inside boundary to prevent circular dependencies
        from .neuromod import make_ascii_bar

        lines = []
        lines.append("=" * 60)
        lines.append("         COGNITIVE ENGINE DIAGNOSTIC PANEL")
        lines.append("=" * 60)

        # 1. Chemical Telemetry
        if self.neuromodulator is not None:
            chems = self.neuromodulator.get_chemical_levels()
            lines.append("Neuromodulator: Active")
            lines.append(f"  {chems['dashboard']}")
        else:
            lines.append("Neuromodulator: Inactive")
        lines.append("-" * 60)

        # 2. Registered Modules
        adapters = self.registry.list_adapters()
        lines.append(f"Registered Modules ({len(adapters)}):")
        for adapter in adapters:
            # Retrieve glia stabilization telemetry if attached
            glia_info = ""
            if self.glial_manager is not None:
                scale = self.glial_manager._plasticity_scales.get(adapter.name, 1.0)
                hook = self.glial_manager._sanitizer_hooks.get(adapter.name)
                grad_status = hook.grad_status if hook is not None else "Stable"
                lock_char = "🔒" if scale < 0.9 else ("🔓" if scale > 1.1 else "")
                lock_str = f" {lock_char}" if lock_char else ""
                glia_info = f" | Local Plasticity: {scale:.1f}x{lock_str} | Grad Status: {grad_status}"

            # Check if dendritic gate is present
            gate = getattr(adapter, "dendrite_gate", None)
            if gate is not None:
                status = gate.get_status()
                active_pct = status.get("active_pct", 0.0)
                pruned_branches = 0
                if hasattr(gate, "pruning_mask"):
                    pruned_branches = int((gate.pruning_mask == 0).sum().item())
                lines.append(
                    f"  - {adapter.name:<15} [Dendritic Active: {active_pct:5.1f}%{glia_info} | "
                    f"Pruned weights: {pruned_branches}]"
                )
            else:
                lines.append(f"  - {adapter.name:<15} [Standard Layer{glia_info}]")
        lines.append("-" * 60)

        # 3. Workspace state
        if self.workspace is not None:
            lines.append("Workspace: Attached")
            if hasattr(self.workspace, "broadcaster"):
                slots = getattr(self.workspace.broadcaster, "workspace_slots", 1)
                lines.append(f"  - Slots: {slots}")
            if hasattr(self.workspace, "selector"):
                sel_type = getattr(self.workspace.selector, "attention_type", "unknown")
                thresh = getattr(self.workspace.selector, "ignition_threshold", 0.0)
                lines.append(f"  - Attention: {sel_type} (threshold={thresh:.4f})")
        else:
            lines.append("Workspace: Unattached")
        lines.append("-" * 60)

        # 4. Conceptual Maps
        if self.concept_layers:
            lines.append("Conceptual Maps:")
            for layer_name, layer in self.concept_layers.items():
                try:
                    activations = self.data_flow.get_buffer(layer_name)
                    mean_acts = activations.view(-1, layer.num_concepts).mean(dim=0)
                except KeyError:
                    mean_acts = torch.zeros(layer.num_concepts)

                interventions = layer.intervention_engine.get_interventions()
                for i in range(layer.num_concepts):
                    val = float(mean_acts[i].item())
                    bar = make_ascii_bar(val)
                    name_str = layer.concept_names[i]
                    if i in interventions:
                        forced = interventions[i]
                        lines.append(f"  - [{name_str}: {bar} {val:.2f} (OVERRIDDEN -> {forced:.1f})]")
                    else:
                        lines.append(f"  - [{name_str}: {bar} {val:.2f}]")
            lines.append("-" * 60)

        # 5. Crossbar Connectivity Map
        if self.crossbar is not None and self.crossbar.last_weights is not None:
            lines.append("Crossbar Connectivity Map:")
            weights = self.crossbar.last_weights
            if weights.dim() == 3:
                weights = weights.mean(dim=0)
            
            for i in range(self.crossbar.num_slots):
                for j in range(self.crossbar.num_slots):
                    if i != j:
                        w = float(weights[i, j].item())
                        if w > 0.01:
                            bar = make_ascii_bar(w)
                            src = self.crossbar.slot_names[i]
                            tgt = self.crossbar.slot_names[j]
                            lines.append(f"  [{src:<8} ── {bar} {w:.2f} ──> {tgt}]")
            lines.append("-" * 60)

        lines.append("=" * 60)

        return "\n".join(lines)

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
        4. Log structured OpenTelemetry-compatible JSON metrics.

        Returns:
            Broadcasted workspace state tensor [B, latent_dim].
        """
        try:
            if self.workspace is None:
                raise ValueError(
                    "No workspace attached. Call `attach_workspace` before stepping."
                )

            adapters = self.registry.list_adapters()
            if not adapters:
                raise ValueError("No modules registered with the engine.")

            # Self-healing alignment: Ensure all adapter key dimensions match workspace key_dim
            if hasattr(self.workspace, "key_dim"):
                target_key_dim = self.workspace.key_dim
                for adapter in adapters:
                    if hasattr(adapter, "key_dim") and adapter.key_dim != target_key_dim:
                        logger.info(
                            f"Dynamic alignment: Re-projecting key space for module '{adapter.name}' "
                            f"from {adapter.key_dim} to {target_key_dim} to match the workspace."
                        )
                        adapter.key_dim = target_key_dim
                        device = next(adapter.parameters()).device if list(adapter.parameters()) else torch.device("cpu")
                        dtype = next(adapter.parameters()).dtype if list(adapter.parameters()) else torch.float32
                        adapter.key_proj = nn.Linear(adapter.latent_dim, target_key_dim).to(device=device, dtype=dtype)

            latent_states: Dict[str, torch.Tensor] = {}
            keys: Dict[str, torch.Tensor] = {}

            for adapter in adapters:
                try:
                    latent = self.data_flow.get_buffer(adapter.name)
                    latent_states[adapter.name] = latent
                    keys[adapter.name] = adapter.get_key(latent)
                except RegistryError:
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

            if self.neuromodulator is not None:
                self.neuromodulator.modulate(self)

            if self.glial_manager is not None:
                self.glial_manager.update(self)

            # Perform Crossbar message passing if attached
            if self.crossbar is not None:
                for adapter in adapters:
                    if hasattr(adapter, "slot_idx"):
                        adapter.engine = self
                # Vectorized parallel routing
                routed_latents = self.crossbar() # [B, num_slots, slot_dim]
                # Distribute routed slot context back to their adapters as waking GWT context
                for adapter in adapters:
                    if hasattr(adapter, "slot_idx"):
                        adapter.receive_broadcast(routed_latents[:, adapter.slot_idx])

            broadcast_state = self.workspace(latent_states, keys)

            for adapter in adapters:
                adapter.receive_broadcast(broadcast_state)

            # Record waking transition to episodic cache if buffer is attached (O(1) overhead)
            if self.replay_buffer is not None:
                self.record_transition()

            # ── Enterprise Telemetry Logging ──
            try:
                from .telemetry import get_telemetry_logger
                modules_telemetry = {}
                for adapter in adapters:
                    gate_pct = 0.0
                    if hasattr(adapter, "dendrite_gate") and adapter.dendrite_gate is not None:
                        gate_pct = adapter.dendrite_gate.get_status().get("active_pct", 0.0)
                    
                    scale = 1.0
                    grad_status = "Stable"
                    if self.glial_manager is not None:
                        scale = self.glial_manager._plasticity_scales.get(adapter.name, 1.0)
                        hook = self.glial_manager._sanitizer_hooks.get(adapter.name)
                        if hook is not None:
                            grad_status = hook.grad_status

                    modules_telemetry[adapter.name] = {
                        "dendritic_active_pct": gate_pct,
                        "plasticity_scale": scale,
                        "grad_status": grad_status
                    }

                crossbar_data = None
                if self.crossbar is not None and self.crossbar.last_weights is not None:
                    crossbar_data = {"mean_routing_weight": float(self.crossbar.last_weights.mean().item())}

                ne = getattr(self.neuromodulator, "ne", 0.0) if self.neuromodulator is not None else 0.0
                ach = getattr(self.neuromodulator, "ach", 0.0) if self.neuromodulator is not None else 0.0
                thresh = getattr(self.workspace.selector, "ignition_threshold", 0.0) if hasattr(self.workspace, "selector") else 0.0

                get_telemetry_logger().record_step(
                    step_idx=self._step_counter,
                    ne=ne,
                    ach=ach,
                    ignition_threshold=thresh,
                    modules_telemetry=modules_telemetry,
                    crossbar_weights=crossbar_data
                )
                self._step_counter += 1
            except Exception as tel_ex:
                logger.debug(f"Telemetry logging failed: {tel_ex}")

            return broadcast_state

        except Exception as e:
            # ── GWT Step Fail-Safe Fallback Bypass ──
            from .telemetry import get_telemetry_logger
            get_telemetry_logger().record_error(
                error_msg=str(e),
                phase="GWT step execution",
                details="Graceful fail-safe fallback bypass triggered."
            )
            
            # Construct a safe zero-filled broadcast fallback tensor
            batch_size = 1
            try:
                for buf in self.data_flow.list_buffers().values():
                    batch_size = buf.shape[0]
                    break
            except Exception:
                pass
            
            latent_dim = getattr(self.workspace, "latent_dim", 4) if self.workspace is not None else 4
            device = torch.device("cpu")
            if self.workspace is not None and hasattr(self.workspace, "parameters"):
                try:
                    params = list(self.workspace.parameters())
                    if params:
                        device = params[0].device
                except Exception:
                    pass
            fallback_broadcast = torch.zeros(batch_size, latent_dim, device=device)
            
            # Deliver broadcast back to adapters to guarantee forward pathway continuity
            try:
                for adapter in self.registry.list_adapters():
                    adapter.receive_broadcast(fallback_broadcast)
            except Exception:
                pass
                
            return fallback_broadcast

    def attach_replay_buffer(self, replay_buffer: Any) -> None:
        """Attach an episodic replay buffer to record waking transitions."""
        self.replay_buffer = replay_buffer
        logger.info("Successfully attached CognitiveReplayBuffer to CognitiveAugEngine.")

    def record_transition(self) -> None:
        """
        Record the current step's latent states, GWT broadcast state, and salience
        into the attached episodic replay buffer in O(1) time.
        """
        if self.replay_buffer is None:
            return

        adapters = self.registry.list_adapters()
        if not adapters:
            return

        latent_states = {}
        for adapter in adapters:
            try:
                # Retrieve detached waking latent outputs from buffer
                latent_states[adapter.name] = self.data_flow.get_buffer(adapter.name).detach()
            except KeyError:
                continue

        if not latent_states:
            return

        # Fetch latest global broadcast context (detached)
        broadcast_state = adapters[0].get_last_broadcast().detach()

        # Compute salience rank from DataFlowManager (defaulting to average L2 norm fallback)
        saliences = self.data_flow.list_saliences()
        if saliences:
            total_salience = sum(saliences.values())
        else:
            # Fallback salience metric: mean L2 magnitude of latent states
            total_salience = 0.0
            for latent in latent_states.values():
                total_salience += torch.linalg.vector_norm(latent.mean(dim=0)).item()

        # Write to episodic buffer with detached elements (O(1) complexity)
        self.replay_buffer.add_trace(latent_states, broadcast_state, total_salience)

    def enter_sleep_phase(
        self,
        steps: int = 100,
        learning_rate: float = 0.001,
        pruning_threshold: float = 0.05,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """
        Pauses online processing and executes an offline sleep and memory consolidation cycle.
        
        Replays high-salience experiences to stabilize GWT attention routing weights,
        applies structural pruning to under-utilized dendritic branches (permanently
        severing parameters and backprop gradients), and flushes the replay buffer.
        
        Args:
            steps             : Number of slow-wave sleep steps.
            learning_rate     : Learning rate for consolidation.
            pruning_threshold : Average activation threshold below which dendritic branches are pruned.
            batch_size        : Replay mini-batch size.
            
        Returns:
            Telemetry dictionary summarizing replayed items, pruned branches, and loss delta.
        """
        if self.replay_buffer is None:
            logger.warning("No replay buffer is attached. Sleep cycle aborted.")
            return {
                "memory_traces_replayed": 0,
                "dendritic_branches_pruned": 0,
                "loss_delta": 0.0,
            }

        # Lazy imports inside sleep phase boundary to guarantee zero circular import risks
        from .sleep import ConsolidationEngine, prune_dendrites

        # Cache the waking latest_branch_activations for each gate before they are overwritten by sleep cycle replays
        waking_activations = {}
        for adapter in self.registry.list_adapters():
            if hasattr(adapter, "dendrite_gate") and adapter.dendrite_gate is not None:
                if adapter.dendrite_gate.latest_branch_activations is not None:
                    waking_activations[adapter.name] = adapter.dendrite_gate.latest_branch_activations.clone()

        # 1. Initialize offline ConsolidationEngine and execute SWS cycles
        consolidator = ConsolidationEngine(self, self.replay_buffer)
        telemetry = consolidator.sleep_cycle(
            steps=steps,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )

        # 2. Execute synaptic homeostasis: structural pruning of weak dendritic branches (using cached waking activations)
        pruned_count = prune_dendrites(
            self,
            pruning_threshold=pruning_threshold,
            waking_activations=waking_activations,
        )
        telemetry["dendritic_branches_pruned"] = pruned_count

        # 3. Flush memory buffers to prepare for new learning cycles in waking state
        self.replay_buffer.clear()
        self.data_flow.clear_buffers()

        return telemetry
