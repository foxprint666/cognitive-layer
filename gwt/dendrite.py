"""
gwt/dendrite.py
===============
Biologically-inspired Active Dendritic Gating for Phase v0.2.

Active dendrites act as non-linear pre-processors that use global context
(e.g., GWT broadcast state) to dynamically gate or amplify local feedforward
pathways. This prevents catastrophic forgetting and enhances parameter efficiency.
"""

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .engine import ModuleAdapter, DataFlowManager

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core Component: ActiveDendriteGate
# ─────────────────────────────────────────────────────────────────────────────

class ActiveDendriteGate(nn.Module):
    """
    Highly optimized, fully vectorized Active Dendritic Gating layer.
    
    In biological systems, dendritic branches act as non-linear pre-processors that
    use global context to dynamically gate or amplify local feedforward pathways.
    This module implements modulatory gain scaling and thresholded NMDA spiking.
    """

    def __init__(
        self,
        feedforward_dim: int,
        context_dim: int,
        num_branches: int = 4,
        spike_type: str = "modulatory-gain",
        threshold: float = 0.5,
    ) -> None:
        """
        Args:
            feedforward_dim : Dimension of the target module's latent representation.
            context_dim     : Dimension of the incoming GWT broadcast tensor.
            num_branches    : Number of sub-integration dendritic zones.
            spike_type      : Gating type: 'modulatory-gain' or 'nmda-threshold'.
            threshold       : Spike threshold for 'nmda-threshold' gating.
        """
        super().__init__()
        self.feedforward_dim = feedforward_dim
        self.context_dim = context_dim
        self.num_branches = num_branches
        self.spike_type = spike_type
        self.threshold = threshold
        self.gain_temperature = 1.0

        if spike_type not in ["modulatory-gain", "nmda-threshold"]:
            raise ValueError(
                f"Unsupported spike_type '{spike_type}'. "
                "Must be one of ['modulatory-gain', 'nmda-threshold']."
            )

        # Vectorized stacked linear projection mapping context -> branches * features
        self.context_proj = nn.Linear(context_dim, num_branches * feedforward_dim)
        
        # Structural pruning masks to permanently sever pruned connections
        self.register_buffer("pruning_mask", torch.ones_like(self.context_proj.weight.data))
        self.context_proj.weight.register_hook(lambda grad: grad * self.pruning_mask)
        
        if self.context_proj.bias is not None:
            self.register_buffer("bias_pruning_mask", torch.ones_like(self.context_proj.bias.data))
            self.context_proj.bias.register_hook(lambda grad: grad * self.bias_pruning_mask)

        # Detached branch activation scores for telemetry / diagnostic dashboards
        # Shape: [B, num_branches, feedforward_dim]
        self.latest_branch_activations: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x       : [B, ..., feedforward_dim] — incoming feedforward features.
            context : [B, context_dim]          — incoming GWT broadcast tensor.

        Returns:
            Gating features of same shape as x.
        """
        batch_size = x.shape[0]

        # Stacked vectorized projection
        # Input: [B, context_dim] -> Output: [B, num_branches * feedforward_dim]
        proj = self.context_proj(context)

        # Reshape to separate dendritic branches: [B, num_branches, feedforward_dim]
        branch_raw = proj.view(batch_size, self.num_branches, self.feedforward_dim)
        
        # Sigmoid gain temperature scaling (ACh focus modulation)
        gain_temp = getattr(self, "gain_temperature", 1.0)
        if isinstance(gain_temp, torch.Tensor):
            gain_temp = gain_temp.to(device=branch_raw.device, dtype=branch_raw.dtype)
        branch_score = torch.sigmoid(branch_raw / max(gain_temp, 1e-5))

        if self.spike_type == "modulatory-gain":
            branch_gates = branch_score
        elif self.spike_type == "nmda-threshold":
            # NMDA Spiking Activation:
            # 1. Compute branch depolarization level (average activation across feature channels)
            # Shape: [B, num_branches, 1]
            branch_depolarization = branch_score.mean(dim=-1, keepdim=True)

            # 2. Binary heaviside spike mask
            spiked_mask = (branch_depolarization >= self.threshold).to(branch_score.dtype)

            # 3. Straight-Through Estimator (STE) for differentiable gradient flow
            ste_gate = spiked_mask + (branch_depolarization - branch_depolarization.detach())

            # 4. Gate active branches; mute inactive branches to zero
            branch_gates = ste_gate * branch_score
        else:
            raise ValueError(f"Unknown spike type: {self.spike_type}")

        # Vectorized average across all dendritic branches -> [B, feedforward_dim]
        gating_factors = branch_gates.mean(dim=1)

        # Save detached latest branch activations for diagnostics without holding graph memory
        with torch.no_grad():
            self.latest_branch_activations = branch_gates.detach().clone()

        # Handle broadcasting for multi-dimensional inputs (e.g. sequence data [B, S, D])
        # If input has extra dimensions in between, unsqueeze gating_factors to match
        if x.dim() > gating_factors.dim():
            unsqueezed_dims = x.dim() - gating_factors.dim()
            for _ in range(unsqueezed_dims):
                gating_factors = gating_factors.unsqueeze(1)

        # Element-wise gate application
        return x * gating_factors

    def get_status(self) -> Dict[str, float]:
        """
        Retrieve percentage of active vs. muted pathways.
        """
        if self.latest_branch_activations is None:
            return {"active_pct": 0.0, "muted_pct": 100.0}

        # Determine threshold for active pathways
        thresh = self.threshold if self.spike_type == "nmda-threshold" else 0.5

        # Pathway-level statistics (averaged over batch, branch, and feature dimensions)
        active_mask = self.latest_branch_activations >= thresh
        active_pct = active_mask.float().mean().item() * 100.0
        muted_pct = 100.0 - active_pct

        status = {
            "active_pct": active_pct,
            "muted_pct": muted_pct,
        }

        # For NMDA, add branch-level spike statistics
        if self.spike_type == "nmda-threshold":
            # A branch spiked if its average feature gating is non-zero
            branch_spikes = (self.latest_branch_activations.mean(dim=-1) > 0.0).float()
            branch_active_pct = branch_spikes.mean().item() * 100.0
            status["branch_active_pct"] = branch_active_pct
            status["branch_muted_pct"] = 100.0 - branch_active_pct

        return status


# ─────────────────────────────────────────────────────────────────────────────
# Ease-of-Use Component: DendriticModuleAdapter
# ─────────────────────────────────────────────────────────────────────────────

class DendriticModuleAdapter(ModuleAdapter):
    """
    Subclass of ModuleAdapter that automatically appends ActiveDendriteGate
    to the wrapped module's execution hook pipeline.
    """

    def __init__(
        self,
        name_or_module: Any = None,
        module: Optional[nn.Module] = None,
        latent_dim: Optional[int] = None,
        data_flow: Optional[DataFlowManager] = None,
        num_branches: int = 4,
        spike_type: str = "modulatory-gain",
        key_dim: int = 64,
        projection_in_dim: Optional[int] = None,
        threshold: float = 0.5,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Supports dual and flexible signatures for maximum developer convenience:
        1. Standard: DendriticModuleAdapter(name, module, latent_dim, data_flow, ...)
        2. Minimal:  DendriticModuleAdapter(module, spike_type="nmda-threshold")
        3. Keywords: DendriticModuleAdapter(name="toy", module=model, latent_dim=8, data_flow=engine.data_flow)
        """
        # Resolve name and module flexibly
        if isinstance(name_or_module, str):
            resolved_name = name_or_module
            resolved_module = module
            if resolved_module is None:
                raise ValueError("module must be provided when the first argument is a name.")
        elif isinstance(name_or_module, nn.Module):
            resolved_module = name_or_module
            resolved_name = name if name is not None else f"dendritic_{resolved_module.__class__.__name__.lower()}"
        elif name_or_module is None:
            resolved_module = module
            resolved_name = name
            if resolved_module is None:
                raise ValueError("A PyTorch Module must be provided as a positional or keyword argument.")
        else:
            raise ValueError(f"Invalid first argument type: {type(name_or_module)}")

        if resolved_name is None:
            resolved_name = f"dendritic_{resolved_module.__class__.__name__.lower()}"

        # Statically inspect output size if possible
        detected_dim = self._detect_output_dim(resolved_module)

        if latent_dim is None:
            latent_dim = detected_dim if detected_dim is not None else 64

        if data_flow is None:
            data_flow = DataFlowManager()

        # Initialize parent ModuleAdapter
        super().__init__(
            name=resolved_name,
            module=resolved_module,
            latent_dim=latent_dim,
            data_flow=data_flow,
            key_dim=key_dim,
            projection_in_dim=projection_in_dim,
            **kwargs,
        )

        self.num_branches = num_branches
        self.spike_type = spike_type
        self.threshold = threshold

        # ActiveDendriteGate instance (lazy initialized if static inspection fails)
        self.dendrite_gate: Optional[ActiveDendriteGate] = None

        if detected_dim is not None:
            self._init_dendrite_gate(detected_dim)
        else:
            logger.warning(
                f"Could not statically inspect output dimension for module '{resolved_name}'. "
                "ActiveDendriteGate will be lazy-initialized during the first forward pass. "
                "WARNING: If you are training this module, ensure you create the optimizer AFTER "
                "the first forward pass, or explicitly provide 'feedforward_dim' during initialization."
            )

    def _detect_output_dim(self, module: nn.Module) -> Optional[int]:
        """Tries to statically inspect the output dimension of the module layers."""
        # 1. Direct standard layers check
        if isinstance(module, nn.Linear):
            return module.out_features
        for conv_cls in [nn.Conv1d, nn.Conv2d, nn.Conv3d]:
            if isinstance(module, conv_cls):
                return module.out_channels
        for rnn_cls in [nn.LSTM, nn.GRU, nn.RNN]:
            if isinstance(module, rnn_cls):
                return module.hidden_size

        # 2. Attribute-level check
        for attr in ["out_features", "out_channels", "hidden_size", "output_dim", "latent_dim", "d_model"]:
            if hasattr(module, attr):
                val = getattr(module, attr)
                if isinstance(val, int):
                    return val

        # 3. Recursive inspection of children (last child)
        for child in reversed(list(module.children())):
            dim = self._detect_output_dim(child)
            if dim is not None:
                return dim

        return None

    def _init_dendrite_gate(self, feedforward_dim: int) -> None:
        """Helper to construct and register the active dendrite gate."""
        self.dendrite_gate = ActiveDendriteGate(
            feedforward_dim=feedforward_dim,
            context_dim=self.latent_dim,  # Context is the GWT broadcast vector
            num_branches=self.num_branches,
            spike_type=self.spike_type,
            threshold=self.threshold,
        )
        
        # Register as a submodule so parameters appear in adapter.parameters()
        self.add_module("dendrite_gate", self.dendrite_gate)

        # Cast to correct device & dtype to match core module
        device = next(self.module.parameters()).device if list(self.module.parameters()) else torch.device("cpu")
        dtype = next(self.module.parameters()).dtype if list(self.module.parameters()) else torch.float32
        self.dendrite_gate.to(device=device, dtype=dtype)
        logger.debug(f"ActiveDendriteGate successfully initialized for module '{self.name}'.")

    def _register_forward_hook(self) -> None:
        """Overrides parent forward hook to intercept and apply active dendritic gating."""

        def hook_fn(module: nn.Module, inputs: Any, outputs: Any) -> Any:
            # 1. Extract the primary latent tensor (handling tuples or lists)
            is_tuple = isinstance(outputs, tuple)
            is_list = isinstance(outputs, list)
            if is_tuple or is_list:
                latent = outputs[0]
            else:
                latent = outputs

            if not isinstance(latent, torch.Tensor):
                raise TypeError(
                    f"Expected tensor output from module '{self.name}', "
                    f"got {type(outputs)}"
                )

            # 2. Lazy-initialization fallback if output dim could not be statically inspected
            if self.dendrite_gate is None:
                feedforward_dim = latent.shape[-1]
                logger.info(f"Dynamically initializing ActiveDendriteGate for '{self.name}' with feedforward_dim={feedforward_dim}")
                self._init_dendrite_gate(feedforward_dim)

            # 3. Retrieve global GWT context broadcast vector
            context = self.get_last_broadcast()

            # 4. Handle mismatch in batch dimension (e.g., initial pass before first step)
            batch_size = latent.shape[0]
            if context.shape[0] == 1 and batch_size > 1:
                context = context.expand(batch_size, -1)
            elif context.shape[0] != batch_size:
                context = torch.zeros(batch_size, self.latent_dim, device=latent.device, dtype=latent.dtype)

            # 5. Apply dendritic gating element-wise onto features
            gated_latent = self.dendrite_gate(latent, context)

            # Dimension Agnosticism: Pool varying ranks (3D sequence, 4D vision) to flat 2D
            from .salience import global_pool_latent
            gwt_latent = global_pool_latent(gated_latent)

            # 6. Apply optional learnable projection to workspace dimension
            if self.projection is not None:
                gwt_latent = self.projection(gwt_latent)

            # Store the GWT representation inside the DataFlowManager
            self.data_flow.update_buffer(self.name, gwt_latent)

            # 7. Return the gated features preserving the original output wrapper structure
            if is_tuple:
                return (gated_latent,) + outputs[1:]
            elif is_list:
                return [gated_latent] + list(outputs[1:])
            else:
                return gated_latent

        self._hook_handle = self.module.register_forward_hook(hook_fn)
        logger.debug(f"Dendritic forward hook successfully registered on module '{self.name}'.")


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry Integration
# ─────────────────────────────────────────────────────────────────────────────

def get_dendritic_status(model_or_adapter: nn.Module) -> Dict[str, float]:
    """
    Helper function to aggregate the percentage of active vs. muted dendritic
    pathways across all ActiveDendriteGate layers in the model.
    """
    gates = []
    if isinstance(model_or_adapter, ActiveDendriteGate):
        gates.append(model_or_adapter)
    else:
        for m in model_or_adapter.modules():
            if isinstance(m, ActiveDendriteGate):
                gates.append(m)

    if not gates:
        return {"active_pct": 0.0, "muted_pct": 100.0}

    total_active = 0.0
    total_muted = 0.0
    for g in gates:
        status = g.get_status()
        total_active += status["active_pct"]
        total_muted += status["muted_pct"]

    num_gates = len(gates)
    return {
        "active_pct": total_active / num_gates,
        "muted_pct": total_muted / num_gates,
    }
