"""
cognitive_aug/neurogenesis.py
===================
Autonomous Computational Neurogenesis in PyTorch Cognitive Augmentation Layers.

Enables dynamic structural growth (adding dendritic branches) and selective pruning
under Locus Coeruleus (NE) unexpected uncertainty, regulated by glial protection
(calcium regulation) and metacognitive acetylcholine focus suppression.
"""

import logging
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .engine import ModuleAdapter

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 0. Transfer Salience Calculator
# ─────────────────────────────────────────────────────────────────────────────


class TransferSalienceCalculator:
    """
    Before growing a new branch, measures how much
    existing pathways overlap with the new domain.

    High overlap = fast pathway bridging
    Low overlap = slow growth from scratch
    """

    def calculate_transfer_potential(
        self, existing_adapters: List[nn.Module], new_domain_latent: torch.Tensor
    ) -> tuple[float, Optional[nn.Module], Optional[int]]:
        if not existing_adapters or new_domain_latent is None:
            return 0.0, None, None

        best_score = -2.0
        best_adapter = None
        best_branch_idx = None

        # Pool/average the latent state across batch and sequence dims to a single vector
        latent_vec = new_domain_latent.mean(dim=0)
        if latent_vec.dim() > 1:
            latent_vec = latent_vec.mean(dim=0)

        for adapter in existing_adapters:
            target = adapter
            if hasattr(adapter, "adapter"):
                target = getattr(adapter, "adapter")

            if hasattr(target, "branches"):
                for idx in range(len(target.branches)):
                    state = target.branch_states[f"state_{idx}"]
                    if state[3].item() < 0.5:
                        continue  # Skip inactive/pruned branches

                    rep = getattr(target, f"consolidated_rep_{idx}", None)
                    if rep is not None:
                        # Ensure device compatibility
                        rep_aligned = rep.to(
                            device=latent_vec.device, dtype=latent_vec.dtype
                        )

                        # Check for zero norm to prevent NaNs
                        if rep_aligned.norm() > 0 and latent_vec.norm() > 0:
                            similarity = F.cosine_similarity(
                                rep_aligned.unsqueeze(0), latent_vec.unsqueeze(0)
                            ).item()
                            if similarity > best_score:
                                best_score = similarity
                                best_adapter = target
                                best_branch_idx = idx

        if best_adapter is None:
            return 0.0, None, None

        return best_score, best_adapter, best_branch_idx


# ─────────────────────────────────────────────────────────────────────────────
# 1. Extended Dendritic Module Adapter & Active Gating
# ─────────────────────────────────────────────────────────────────────────────


class ExtendedDendriticModuleAdapter(ModuleAdapter):
    """
    Extends the static DendriticModuleAdapter to support runtime branch allocation,
    active NMDA-threshold gating, and branch pruning.
    Can be instantiated as a standard nn.Module or a GWT ModuleAdapter hook.
    """

    def __init__(
        self,
        name_or_feedforward_dim: Any,
        module_or_context_dim: Any = None,
        latent_dim: Optional[int] = None,
        data_flow: Optional[Any] = None,
        key_dim: int = 64,
        projection_in_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        initial_branches: int = 1,
        **kwargs: Any,
    ):
        if isinstance(name_or_feedforward_dim, str):
            # GWT Hook mode
            name = name_or_feedforward_dim
            module = module_or_context_dim
            assert isinstance(module, nn.Module), (
                "module must be a nn.Module in GWT Hook mode"
            )

            super().__init__(
                name=name,
                module=module,
                latent_dim=latent_dim,
                data_flow=data_flow,
                key_dim=key_dim,
                projection_in_dim=projection_in_dim,
                device=device,
                dtype=dtype,
                **kwargs,
            )
            self.is_gwt_hook = True
            self.in_dim = self._detect_output_dim(module) or latent_dim
            self.ctx_dim = latent_dim
        else:
            # Standalone module mode (e.g. OpenWeightsAdapterHook)
            nn.Module.__init__(self)
            self.is_gwt_hook = False
            self.in_dim = name_or_feedforward_dim
            self.ctx_dim = module_or_context_dim
            self.latent_dim = self.ctx_dim
            self.name = f"extended_dendrite_{id(self)}"

        self.theta_nmda = nn.Parameter(torch.tensor(0.5, device=device, dtype=dtype))

        # Parallel computational paths
        self.branches = nn.ModuleList(
            [
                nn.Linear(
                    self.in_dim, self.in_dim, bias=False, device=device, dtype=dtype
                )
                for _ in range(initial_branches)
            ]
        )
        self.dendritic_gates = nn.ModuleList(
            [
                nn.Linear(
                    self.ctx_dim, self.in_dim, bias=True, device=device, dtype=dtype
                )
                for _ in range(initial_branches)
            ]
        )

        # State tracking via PyTorch ParameterDict and Buffer representations
        self.branch_states = nn.ParameterDict()
        for idx in range(initial_branches):
            self._init_branch_state(idx, permanent=True)

    def _detect_output_dim(self, module: nn.Module) -> Optional[int]:
        """Tries to statically inspect the output dimension of the module layers."""
        if isinstance(module, nn.Linear):
            return module.out_features
        for conv_cls in [nn.Conv1d, nn.Conv2d, nn.Conv3d]:
            if isinstance(module, conv_cls):
                return module.out_channels
        for rnn_cls in [nn.LSTM, nn.GRU, nn.RNN]:
            if isinstance(module, rnn_cls):
                return module.hidden_size

        for attr in [
            "out_features",
            "out_channels",
            "hidden_size",
            "output_dim",
            "latent_dim",
            "d_model",
        ]:
            if hasattr(module, attr):
                val = getattr(module, attr)
                if isinstance(val, int):
                    return val

        for child in reversed(list(module.children())):
            dim = self._detect_output_dim(child)
            if dim is not None:
                return dim
        return None

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

            # 2. Retrieve global GWT context broadcast vector
            context = self.get_last_broadcast()

            # 3. Handle mismatch in batch dimension
            batch_size = latent.shape[0]
            if context.shape[0] == 1 and batch_size > 1:
                context = context.expand(batch_size, -1)
            elif context.shape[0] != batch_size:
                context = torch.zeros(
                    batch_size,
                    self.latent_dim,
                    device=latent.device,
                    dtype=latent.dtype,
                )

            context = context.to(device=latent.device, dtype=latent.dtype)

            # 4. Apply dendritic gating/branches
            gated_latent = self.forward_dendrite(latent, context)

            # Dimension Agnosticism: Pool varying ranks (3D sequence, 4D vision) to flat 2D
            from .salience import global_pool_latent

            gwt_latent = global_pool_latent(gated_latent)

            # 5. Apply optional learnable projection to workspace dimension
            if self.projection is not None:
                gwt_latent = self.projection(gwt_latent)

            # Store the GWT representation inside the DataFlowManager
            self.data_flow.update_buffer(self.name, gwt_latent)

            # 6. Return the gated features preserving the original output wrapper structure
            if is_tuple:
                return (gated_latent,) + outputs[1:]
            elif is_list:
                return [gated_latent] + list(outputs[1:])
            else:
                return gated_latent

        self._hook_handle = self.module.register_forward_hook(hook_fn)
        logger.debug(
            f"Extended Dendritic forward hook successfully registered on module '{self.name}'."
        )

    def _init_branch_state(self, idx: int, permanent: bool) -> None:
        # State tensor format: [age, utilization, consolidation_score, active_flag]
        device = self.theta_nmda.device
        dtype = self.theta_nmda.dtype
        state = nn.Parameter(
            torch.tensor(
                [0.0, 0.0, 1.0 if permanent else 0.0, 1.0], device=device, dtype=dtype
            ),
            requires_grad=False,
        )
        self.branch_states[f"state_{idx}"] = state
        self.register_buffer(
            f"consolidated_rep_{idx}",
            torch.zeros(self.ctx_dim, device=device, dtype=dtype),
        )

    def add_dendritic_branch(self) -> int:
        """
        Dynamically adds a new branch. The weights are zero-initialized
        to ensure the forward pass remains stable at the moment of integration.
        """
        new_idx = len(self.branches)
        device = self.theta_nmda.device
        dtype = self.theta_nmda.dtype

        new_proj = nn.Linear(
            self.in_dim, self.in_dim, bias=False, device=device, dtype=dtype
        )
        nn.init.zeros_(new_proj.weight)

        new_gate = nn.Linear(
            self.ctx_dim, self.in_dim, bias=True, device=device, dtype=dtype
        )
        nn.init.zeros_(new_gate.weight)
        nn.init.zeros_(new_gate.bias)

        self.branches.append(new_proj)
        self.dendritic_gates.append(new_gate)
        self._init_branch_state(new_idx, permanent=False)
        return new_idx

    def prune_branch(self, idx: int) -> None:
        """Zeroes out parameters and deactivates the target branch."""
        state = self.branch_states[f"state_{idx}"]
        state.data[3] = 0.0  # Deactivate active_flag
        self.branches[idx].weight.data.zero_()
        if hasattr(self.branches[idx], "bias") and self.branches[idx].bias is not None:
            self.branches[idx].bias.data.zero_()

    def forward_dendrite(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        for idx, (proj, gate) in enumerate(zip(self.branches, self.dendritic_gates)):
            state = self.branch_states[f"state_{idx}"]
            active_flag = state[3].item()

            if active_flag > 0.5:
                # Active NMDA Gating calculation
                dendritic_activation = gate(context)

                # Align dendritic_activation's dimensions with x for sequence models
                if x.dim() > dendritic_activation.dim():
                    unsqueezed_dims = x.dim() - dendritic_activation.dim()
                    for _ in range(unsqueezed_dims):
                        dendritic_activation = dendritic_activation.unsqueeze(1)

                # Sigmoidal NMDA threshold activation function
                nmda_gate = torch.sigmoid(dendritic_activation - self.theta_nmda)
                branch_output = proj(x) * nmda_gate
                out = out + branch_output

                # Update state statistics during training passes
                if self.training:
                    state.data[0] += 1.0  # Age
                    state.data[1] += torch.mean(nmda_gate).detach()  # Utilization

                    # Masked branch representation update
                    mean_gate_per_batch = nmda_gate.detach().mean(dim=-1, keepdim=True)
                    flat_gate = mean_gate_per_batch.view(-1, 1)
                    flat_context = context.detach().view(-1, self.ctx_dim)

                    total_weight = flat_gate.sum()
                    if total_weight > 1e-6:
                        weighted_context = (flat_context * flat_gate).sum(
                            dim=0
                        ) / total_weight
                        rep_buffer = getattr(self, f"consolidated_rep_{idx}")
                        rep_buffer.data = (
                            0.95 * rep_buffer.data + 0.05 * weighted_context
                        )
        return x + out  # Residual connection

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.forward_dendrite(x, context)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Centralized Neurogenesis Manager
# ─────────────────────────────────────────────────────────────────────────────


class NeurogenesisManager:
    """
    Centralized controller for structural neurogenesis. Integrates with the GWT
    module registry, monitors surprise signals, and manages the lifecycle of dynamic branches.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        astrocyte_manager: nn.Module,
        replay_buffer: Any,
        metacognitive_monitor: Any,
    ) -> None:
        self.config = config
        self.astrocyte_manager = astrocyte_manager
        self.replay_buffer = replay_buffer
        self.monitor = metacognitive_monitor

        # Core Neurogenesis Hyperparameters
        self.ne_threshold = config.get("ne_threshold", 0.75)
        self.max_growth_rate = config.get("max_growth_rate", 0.05)
        self.cooldown_steps = config.get("cooldown_steps", 500)

        # State Tracking
        self.steps_since_last_birth = 0
        self.active_neuro_events = []

        # NEW: Transfer Salience Calculator & Suppression States
        self.transfer_calculator = TransferSalienceCalculator()
        self.suppressed_adapters = {}  # target_id -> remaining_steps
        self.suppressed_adapters_refs = {}  # target_id -> (adapter_ref, branch_idx)

    def step(
        self, step_idx: int, metrics: Dict[str, Any], adapters: List[nn.Module]
    ) -> str:
        """
        Monitors incoming metrics, evaluates the birth criteria, and coordinates
        new dendritic path generation across registered adapters.
        """
        self.steps_since_last_birth += 1

        # Decrement interference suppression cooldowns statefully
        for target_id, remaining in list(self.suppressed_adapters.items()):
            if remaining <= 1:
                # Restore active state
                adapter_ref, branch_idx = self.suppressed_adapters_refs[target_id]
                state = adapter_ref.branch_states[f"state_{branch_idx}"]
                state.data[3] = 1.0  # Reactivate active_flag
                del self.suppressed_adapters[target_id]
                del self.suppressed_adapters_refs[target_id]
                print(
                    f"    [Restoration] Reactivated suppressed branch {branch_idx} on module {adapter_ref.name}"
                )
            else:
                self.suppressed_adapters[target_id] = remaining - 1

        ne_surprise = metrics.get("NE_surprise")
        ach_focus = metrics.get("ACh_focus", torch.tensor(0.0))
        current_latent = metrics.get("current_latent")

        if ne_surprise is None:
            return "idle"

        # Evaluate suppression and threshold criteria
        if (
            ne_surprise.item() > self.ne_threshold
            and self.steps_since_last_birth >= self.cooldown_steps
        ):
            if ach_focus.item() > self.config.get("ach_focus_ceiling", 0.8):
                return "suppressed_by_focus"

            # Perform safety checks via the AstrocyteManager
            if not self.astrocyte_manager.evaluate_growth_safety():
                return "suppressed_by_excitotoxicity_protection"

            # NEW: Calculate transfer potential using latent overlap
            transfer_potential, best_adapter, best_branch_idx = 0.0, None, None
            if current_latent is not None:
                transfer_potential, best_adapter, best_branch_idx = (
                    self.transfer_calculator.calculate_transfer_potential(
                        adapters, current_latent
                    )
                )

            print(
                f"    [Transfer Salience] Evaluated transfer potential: {transfer_potential:.4f}"
            )

            # Route based on computed transfer potential
            if transfer_potential > 0.7 and best_adapter is not None:
                # INSTANT ADAPTER: High overlap (>0.7) - bridge existing pathways
                self._bridge_to_existing_pathway(
                    adapters, best_adapter, best_branch_idx, transfer_potential
                )
                self.steps_since_last_birth = 0
                return "neurogenesis_triggered"

            elif transfer_potential > 0.3 and best_adapter is not None:
                # AVERAGE LEARNER: Medium overlap (0.3-0.7) - grow with transfer initialization
                self._grow_with_transfer_init(
                    adapters, best_adapter, best_branch_idx, transfer_potential
                )
                self.steps_since_last_birth = 0
                return "neurogenesis_triggered"

            else:
                # Check for Negative Transfer interference zone (0.0 < transfer_potential < 0.15)
                # Note: standard neurogenesis runs, but we suppress the closest pathway to prevent interference
                if 0.0 < transfer_potential < 0.15 and best_adapter is not None:
                    self._apply_interference_suppression(
                        best_adapter, best_branch_idx, duration_steps=20
                    )

                # Standard zero-init neurogenesis
                self._standard_neurogenesis(adapters, ne_surprise.item())
                self.steps_since_last_birth = 0
                return "neurogenesis_triggered"

        return "idle"

    def _bridge_to_existing_pathway(
        self,
        adapters: List[nn.Module],
        closest_target: nn.Module,
        closest_idx: int,
        transfer_potential: float,
    ) -> None:
        for adapter in adapters:
            target = adapter
            if hasattr(adapter, "adapter"):
                target = getattr(adapter, "adapter")

            if hasattr(target, "add_dendritic_branch"):
                new_idx = target.add_dendritic_branch()

                # Copy closest pathway weights directly
                target.branches[new_idx].weight.data.copy_(
                    closest_target.branches[closest_idx].weight.data
                )

                # Also copy NMDA gating weights for the Instant Adapter
                target.dendritic_gates[new_idx].weight.data.copy_(
                    closest_target.dendritic_gates[closest_idx].weight.data
                )
                target.dendritic_gates[new_idx].bias.data.copy_(
                    closest_target.dendritic_gates[closest_idx].bias.data
                )

                print(
                    f"    [Instant Adapter] Bridged {target.name} branch {new_idx} to closest pathway {closest_target.name} branch {closest_idx}"
                )

                # Set consolidation score to 1.0 (permanent immediately)
                target.branch_states[f"state_{new_idx}"].data[2] = 1.0

    def _grow_with_transfer_init(
        self,
        adapters: List[nn.Module],
        closest_target: nn.Module,
        closest_idx: int,
        transfer_potential: float,
    ) -> None:
        for adapter in adapters:
            target = adapter
            if hasattr(adapter, "adapter"):
                target = getattr(adapter, "adapter")

            if hasattr(target, "add_dendritic_branch"):
                new_idx = target.add_dendritic_branch()

                # Initialize new branch weights scaled by similarity
                target.branches[new_idx].weight.data.copy_(
                    closest_target.branches[closest_idx].weight.data
                    * transfer_potential
                )
                print(
                    f"    [Average Learner] Init {target.name} branch {new_idx} from {closest_target.name} branch {closest_idx} scaled by {transfer_potential:.2f}"
                )

                # Initialize consolidation score to 0.5 (semi-consolidated)
                target.branch_states[f"state_{new_idx}"].data[2] = 0.5

    def _standard_neurogenesis(
        self, adapters: List[nn.Module], surprise: float
    ) -> None:
        for adapter in adapters:
            target = adapter
            if hasattr(adapter, "adapter"):
                target = getattr(adapter, "adapter")

            if hasattr(target, "add_dendritic_branch"):
                new_idx = target.add_dendritic_branch()
                # Zero initialized standard branch (starts with consolidation_score = 0.0)
                target.branch_states[f"state_{new_idx}"].data[2] = 0.0
                print(
                    f"    [Slow Adapter] Spawned standard zero-init branch {new_idx} on {target.name}"
                )

    def _apply_interference_suppression(
        self, closest_target: nn.Module, branch_idx: int, duration_steps: int = 20
    ) -> None:
        state = closest_target.branch_states[f"state_{branch_idx}"]
        state.data[3] = 0.0  # Temporarily suppress active_flag to 0.0

        target_id = f"{id(closest_target)}_branch_{branch_idx}"
        self.suppressed_adapters[target_id] = duration_steps
        self.suppressed_adapters_refs[target_id] = (closest_target, branch_idx)
        print(
            f"    [Negative Transfer] Suppressing similar pathway '{closest_target.name}' "
            f"branch {branch_idx} for {duration_steps} steps to prevent interference."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Consolidation & Pruning Engine
# ─────────────────────────────────────────────────────────────────────────────


class ConsolidationEngine:
    """
    Manages offline sleep cycles. Replays memories of surprise events,
    evaluates branch performance, and executes pruning or permanentization.
    """

    def __init__(
        self, model: nn.Module, replay_buffer: Any, threshold_perm: float = 0.4
    ) -> None:
        self.model = model
        self.replay_buffer = replay_buffer
        self.threshold_perm = threshold_perm

    def execute_sleep_cycle(
        self, optimizer: torch.optim.Optimizer, steps: int = 100
    ) -> None:
        self.model.train()
        for _ in range(steps):
            # Sample surprise traces from the replay buffer
            batch = self.replay_buffer.sample_neurogenesis_traces(batch_size=16)
            if batch is None:
                continue

            x, context, target = batch["state"], batch["context"], batch["target"]
            optimizer.zero_grad()
            out = self.model(x, context)
            loss = torch.nn.functional.mse_loss(out, target)
            loss.backward()

            # Accumulate performance metrics for un-consolidated branches
            self._update_consolidation_scores()
            optimizer.step()

        self._crystallize_or_prune()

    def _update_consolidation_scores(self) -> None:
        for name, adapter in self.model.named_modules():
            target = adapter
            if hasattr(adapter, "adapter"):
                target = getattr(adapter, "adapter")

            if isinstance(target, ExtendedDendriticModuleAdapter):
                for key, state in target.branch_states.items():
                    if state[2] < 1.0:  # Branch is not yet permanent (index 2)
                        idx = int(key.split("_")[-1])
                        weight_grad = target.branches[idx].weight.grad
                        if weight_grad is not None:
                            grad_norm = torch.norm(weight_grad).item()
                            state.data[2] += 0.05 * grad_norm

    def _crystallize_or_prune(self) -> None:
        for name, adapter in self.model.named_modules():
            target = adapter
            if hasattr(adapter, "adapter"):
                target = getattr(adapter, "adapter")

            if isinstance(target, ExtendedDendriticModuleAdapter):
                for key, state in list(target.branch_states.items()):
                    if state[2] < 1.0:  # Evaluate candidate branches
                        idx = int(key.split("_")[-1])
                        if state[2] >= self.threshold_perm:
                            state.data[2] = 1.0  # Set status to permanent
                            state.data[3] = 1.0  # Keep active_flag to 1.0
                        elif state[2] > 0.15:
                            self._apply_partial_consolidation(state, target, idx)
                        else:
                            target.prune_branch(idx)

    def _apply_partial_consolidation(
        self, state: nn.Parameter, adapter: nn.Module, idx: int
    ) -> None:
        # Partially consolidated: Retained but heavily regularized
        state.data[2] = 0.5  # Lock to partial consolidation state
        # Apply L1 weight decay to the target branch and gates to enforce representation sparsity
        adapter.branches[idx].weight.data.mul_(0.90)
        adapter.dendritic_gates[idx].weight.data.mul_(0.90)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Cognitive Replay Buffer with Dynamic Tagging
# ─────────────────────────────────────────────────────────────────────────────


class NeurogenesisReplayBuffer:
    """
    Extends experience replay to prioritize consolidation of memories
    associated with structural modifications.
    """

    def __init__(self, capacity: int = 5000) -> None:
        self.capacity = capacity
        self.buffer = []

    def push(
        self,
        state: torch.Tensor,
        context: torch.Tensor,
        target: torch.Tensor,
        surprise: float,
        neuro_event: bool,
    ) -> None:
        experience = {
            "state": state.detach().clone().cpu(),
            "context": context.detach().clone().cpu(),
            "target": target.detach().clone().cpu(),
            "surprise": surprise,
            "neuro_event": float(neuro_event),
        }
        if len(self.buffer) >= self.capacity:
            # Drop the lowest-surprise experience to preserve critical traces
            self.buffer.sort(key=lambda x: x["surprise"] + x["neuro_event"] * 10.0)
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample_neurogenesis_traces(
        self, batch_size: int
    ) -> Optional[Dict[str, torch.Tensor]]:
        if len(self.buffer) < batch_size:
            return None

        # Draw samples biased toward surprise events
        weights = [x["surprise"] + x["neuro_event"] * 5.0 for x in self.buffer]
        probs = torch.softmax(torch.tensor(weights, dtype=torch.float32), dim=0)
        indices = torch.multinomial(
            probs, num_samples=batch_size, replacement=True
        ).tolist()

        samples = [self.buffer[i] for i in indices]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        return {
            "state": torch.stack([s["state"] for s in samples]).to(device),
            "context": torch.stack([s["context"] for s in samples]).to(device),
            "target": torch.stack([s["target"] for s in samples]).to(device),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Open-Weights Substrate Integration & Hooks
# ─────────────────────────────────────────────────────────────────────────────


class OpenWeightsAdapterHook(nn.Module):
    """
    Wraps standard projection layers in Llama or Mistral to run parallel,
    dynamically growing dendritic branches.
    """

    def __init__(self, target_linear: nn.Linear, context_dim: int) -> None:
        super().__init__()
        self.base_layer = target_linear
        # Keep original backbone weights frozen to preserve base capabilities
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        # Parallel adaptive pathway
        self.adapter = ExtendedDendriticModuleAdapter(
            feedforward_dim=target_linear.out_features,
            context_dim=context_dim,
            initial_branches=1,
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        # Route baseline outputs through the adapter to process contextual adjustments
        adapter_out = self.adapter(base_out, context)
        return base_out + adapter_out


def dynamic_register_parameters(
    optimizer: torch.optim.Optimizer,
    adapter: ExtendedDendriticModuleAdapter,
    branch_idx: int,
) -> None:
    """
    Safely registers the parameters of a newly spawned branch to the active
    optimizer during runtime, preserving existing momentum buffers.
    """
    new_params = list(adapter.branches[branch_idx].parameters()) + list(
        adapter.dendritic_gates[branch_idx].parameters()
    )

    # Isolate parameters in a new group with custom learning rates
    new_group = {
        "params": new_params,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "name": f"dynamic_branch_{branch_idx}",
    }
    optimizer.add_param_group(new_group)


def dynamic_deregister_parameters(
    optimizer: torch.optim.Optimizer,
    adapter: ExtendedDendriticModuleAdapter,
    branch_idx: int,
) -> None:
    """
    Safely removes parameters from a live optimizer during pruning,
    re-indexing parameter groups without breaking active autograd traces.
    """
    params_to_remove = set(adapter.branches[branch_idx].parameters()) | set(
        adapter.dendritic_gates[branch_idx].parameters()
    )

    # Rebuild parameter groups while filtering out pruned parameters
    new_groups = []
    for group in optimizer.param_groups:
        filtered_params = [p for p in group["params"] if p not in params_to_remove]
        if filtered_params:
            group["params"] = filtered_params
            new_groups.append(group)

    optimizer.param_groups = new_groups

    # Clear historical optimizer states corresponding to the pruned parameters
    for param in params_to_remove:
        if param in optimizer.state:
            del optimizer.state[param]


# ─────────────────────────────────────────────────────────────────────────────
# 6. Metaplastic Safety & Excitotoxicity Regulators
# ─────────────────────────────────────────────────────────────────────────────


class NeurogenesisGradientSanitizerHook:
    """
    Extends metaplasticity controls. New branches have their gradients scaled
    based on their maturity level to protect existing representations.
    """

    def __init__(self, maturation_rate: float = 0.01) -> None:
        self.maturation_rate = maturation_rate

    def register_sanitizer(
        self, module: ExtendedDendriticModuleAdapter, idx: int
    ) -> Any:
        # Register a backward hook to scale gradients dynamically during training
        def backward_hook(grad: torch.Tensor) -> torch.Tensor:
            state = module.branch_states[f"state_{idx}"]
            age = state[0].item()
            # Gradual transition scale based on branch age
            scale_factor = 1.0 - torch.exp(
                torch.tensor(
                    -self.maturation_rate * age, device=grad.device, dtype=grad.dtype
                )
            )
            # Clip gradients to prevent numeric instability
            sanitized_grad = torch.clamp(grad, -1.0, 1.0)
            return sanitized_grad * scale_factor

        target_param = module.branches[idx].weight
        return target_param.register_hook(backward_hook)


class NeurogenesisAstrocyteManager(nn.Module):
    """
    Implements glia-inspired homeostatic regulation to protect newly spawned
    pathways from activation-driven excitotoxicity.
    """

    def __init__(
        self, calcium_decay: float = 0.12, safety_ceiling: float = 4.0
    ) -> None:
        super().__init__()
        self.calcium_decay = calcium_decay
        self.safety_ceiling = safety_ceiling
        self.calcium_store = nn.Parameter(torch.zeros(1), requires_grad=False)

    def monitor_and_regulate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Monitors activations, updates internal calcium metrics, and dampens
        excessive output to maintain stable operating ranges.
        """
        mean_activation = torch.mean(torch.abs(x))
        # Update calcium metrics using leaky integration
        self.calcium_store.data = (
            1.0 - self.calcium_decay
        ) * self.calcium_store.data + mean_activation
        if self.calcium_store.item() > self.safety_ceiling:
            # Multiplicative downscaling to damp hyper-excitation
            attenuation = self.safety_ceiling / (self.calcium_store.item() + 1e-6)
            return x * attenuation
        return x

    def evaluate_growth_safety(self) -> bool:
        """Prevents neurogenesis events if the network is currently hyper-excited."""
        return self.calcium_store.item() < self.safety_ceiling * 0.8
