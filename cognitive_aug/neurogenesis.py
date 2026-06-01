"""
cognitive_aug/neurogenesis.py
===================
Autonomous Computational Neurogenesis in PyTorch Cognitive Augmentation Layers.

Enables dynamic structural growth (adding dendritic branches) and selective pruning
under Locus Coeruleus (NE) unexpected uncertainty, regulated by glial protection
(calcium regulation) and metacognitive acetylcholine focus suppression.
"""

import math
import logging
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Extended Dendritic Module Adapter & Active Gating
# ─────────────────────────────────────────────────────────────────────────────

class ExtendedDendriticModuleAdapter(nn.Module):
    """
    Extends the static DendriticModuleAdapter to support runtime branch allocation,
    active NMDA-threshold gating, and branch pruning.
    """
    def __init__(self, feedforward_dim: int, context_dim: int, initial_branches: int = 1):
        super().__init__()
        self.in_dim = feedforward_dim
        self.ctx_dim = context_dim
        self.theta_nmda = nn.Parameter(torch.tensor(0.5))  # NMDA threshold

        # Parallel computational paths
        self.branches = nn.ModuleList([
            nn.Linear(self.in_dim, self.in_dim, bias=False) for _ in range(initial_branches)
        ])
        self.dendritic_gates = nn.ModuleList([
            nn.Linear(self.ctx_dim, self.in_dim, bias=True) for _ in range(initial_branches)
        ])

        # State tracking via PyTorch ParameterDict
        self.branch_states = nn.ParameterDict()
        for idx in range(initial_branches):
            self._init_branch_state(idx, permanent=True)

    def _init_branch_state(self, idx: int, permanent: bool):
        # State tensor format: [age, utilization, consolidation_score, active_flag]
        state = nn.Parameter(torch.tensor([
            0.0, 0.0, 1.0 if permanent else 0.0, 1.0
        ]), requires_grad=False)
        self.branch_states[f"state_{idx}"] = state

    def add_dendritic_branch(self) -> int:
        """
        Dynamically adds a new branch. The weights are zero-initialized
        to ensure the forward pass remains stable at the moment of integration.
        """
        new_idx = len(self.branches)
        new_proj = nn.Linear(self.in_dim, self.in_dim, bias=False)
        nn.init.zeros_(new_proj.weight)

        new_gate = nn.Linear(self.ctx_dim, self.in_dim, bias=True)
        nn.init.zeros_(new_gate.weight)
        nn.init.zeros_(new_gate.bias)

        self.branches.append(new_proj)
        self.dendritic_gates.append(new_gate)
        self._init_branch_state(new_idx, permanent=False)
        return new_idx

    def prune_branch(self, idx: int):
        """Zeroes out parameters and deactivates the target branch."""
        state = self.branch_states[f"state_{idx}"]
        state.data[3] = 0.0  # Deactivate active_flag
        self.branches[idx].weight.data.zero_()
        if hasattr(self.branches[idx], "bias") and self.branches[idx].bias is not None:
            self.branches[idx].bias.data.zero_()

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        for idx, (proj, gate) in enumerate(zip(self.branches, self.dendritic_gates)):
            state = self.branch_states[f"state_{idx}"]
            active_flag = state[3].item()

            if active_flag > 0.5:
                # Active NMDA Gating calculation
                dendritic_activation = gate(context)
                # Sigmoidal NMDA threshold activation function
                nmda_gate = torch.sigmoid(dendritic_activation - self.theta_nmda)
                branch_output = proj(x) * nmda_gate
                out = out + branch_output

                # Update state statistics during training passes
                if self.training:
                    state.data[0] += 1.0  # Age
                    state.data[1] += torch.mean(nmda_gate).detach()  # Utilization
        return out


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
        metacognitive_monitor: Any
    ):
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
        self.neuron_registry = {}

    def step(self, step_idx: int, metrics: Dict[str, Any], adapters: List[nn.Module]) -> str:
        """
        Monitors incoming metrics, evaluates the birth criteria, and coordinates
        new dendritic path generation across registered adapters.
        """
        self.steps_since_last_birth += 1
        ne_surprise = metrics.get("NE_surprise")
        ach_focus = metrics.get("ACh_focus", torch.tensor(0.0))

        if ne_surprise is None:
            return "idle"

        # Evaluate suppression and threshold criteria
        if ne_surprise.item() > self.ne_threshold and self.steps_since_last_birth >= self.cooldown_steps:
            if ach_focus.item() > self.config.get("ach_focus_ceiling", 0.8):
                return "suppressed_by_focus"

            # Perform safety checks via the AstrocyteManager
            if not self.astrocyte_manager.evaluate_growth_safety():
                return "suppressed_by_excitotoxicity_protection"

            # Trigger neurogenesis across target adapters
            for adapter in adapters:
                # Target adapter itself or its internal adapter
                target = adapter
                if hasattr(adapter, "adapter"):
                    target = getattr(adapter, "adapter")

                if hasattr(target, "add_dendritic_branch"):
                    new_idx = target.add_dendritic_branch()
                    self._register_neuron_state(target, new_idx, ne_surprise.item())

            self.steps_since_last_birth = 0
            return "neurogenesis_triggered"

        return "idle"

    def _register_neuron_state(self, adapter: nn.Module, idx: int, surprise: float):
        neuron_id = f"{id(adapter)}_branch_{idx}"
        self.neuron_registry[neuron_id] = {
            "age": 0,
            "utilization": 0.0,
            "consolidation_score": 0.0,
            "active_flag": 1.0,
            "origin_surprise": surprise
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Consolidation & Pruning Engine
# ─────────────────────────────────────────────────────────────────────────────

class ConsolidationEngine:
    """
    Manages offline sleep cycles. Replays memories of surprise events,
    evaluates branch performance, and executes pruning or permanentization.
    """
    def __init__(self, model: nn.Module, replay_buffer: Any, threshold_perm: float = 0.4):
        self.model = model
        self.replay_buffer = replay_buffer
        self.threshold_perm = threshold_perm

    def execute_sleep_cycle(self, optimizer: torch.optim.Optimizer, steps: int = 100):
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

    def _update_consolidation_scores(self):
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

    def _crystallize_or_prune(self):
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

    def _apply_partial_consolidation(self, state: nn.Parameter, adapter: nn.Module, idx: int):
        # Partially consolidated: Retained but heavily regularized
        state.data[2] = 0.5  # Lock to partial consolidation state
        # Apply L1 weight decay to the target branch to enforce representation sparsity
        adapter.branches[idx].weight.data.mul_(0.90)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Cognitive Replay Buffer with Dynamic Tagging
# ─────────────────────────────────────────────────────────────────────────────

class NeurogenesisReplayBuffer:
    """
    Extends experience replay to prioritize consolidation of memories
    associated with structural modifications.
    """
    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self.buffer = []

    def push(self, state: torch.Tensor, context: torch.Tensor, target: torch.Tensor, surprise: float, neuro_event: bool):
        experience = {
            "state": state.detach().clone().cpu(),
            "context": context.detach().clone().cpu(),
            "target": target.detach().clone().cpu(),
            "surprise": surprise,
            "neuro_event": float(neuro_event)
        }
        if len(self.buffer) >= self.capacity:
            # Drop the lowest-surprise experience to preserve critical traces
            self.buffer.sort(key=lambda x: x["surprise"] + x["neuro_event"] * 10.0)
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample_neurogenesis_traces(self, batch_size: int) -> Optional[Dict[str, torch.Tensor]]:
        if len(self.buffer) < batch_size:
            return None

        # Draw samples biased toward surprise events
        weights = [x["surprise"] + x["neuro_event"] * 5.0 for x in self.buffer]
        probs = torch.softmax(torch.tensor(weights, dtype=torch.float32), dim=0)
        indices = torch.multinomial(probs, num_samples=batch_size, replacement=True).tolist()

        samples = [self.buffer[i] for i in indices]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        return {
            "state": torch.stack([s["state"] for s in samples]).to(device),
            "context": torch.stack([s["context"] for s in samples]).to(device),
            "target": torch.stack([s["target"] for s in samples]).to(device)
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Open-Weights Substrate Integration & Hooks
# ─────────────────────────────────────────────────────────────────────────────

class OpenWeightsAdapterHook(nn.Module):
    """
    Wraps standard projection layers in Llama or Mistral to run parallel,
    dynamically growing dendritic branches.
    """
    def __init__(self, target_linear: nn.Linear, context_dim: int):
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
            initial_branches=1
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        # Route baseline outputs through the adapter to process contextual adjustments
        adapter_out = self.adapter(base_out, context)
        return base_out + adapter_out


def dynamic_register_parameters(optimizer: torch.optim.Optimizer, adapter: ExtendedDendriticModuleAdapter, branch_idx: int):
    """
    Safely registers the parameters of a newly spawned branch to the active
    optimizer during runtime, preserving existing momentum buffers.
    """
    new_params = list(adapter.branches[branch_idx].parameters()) + \
                 list(adapter.dendritic_gates[branch_idx].parameters())

    # Isolate parameters in a new group with custom learning rates
    new_group = {
        "params": new_params,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "name": f"dynamic_branch_{branch_idx}"
    }
    optimizer.add_param_group(new_group)


def dynamic_deregister_parameters(optimizer: torch.optim.Optimizer, adapter: ExtendedDendriticModuleAdapter, branch_idx: int):
    """
    Safely removes parameters from a live optimizer during pruning,
    re-indexing parameter groups without breaking active autograd traces.
    """
    params_to_remove = set(adapter.branches[branch_idx].parameters()) | \
                       set(adapter.dendritic_gates[branch_idx].parameters())

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
    def __init__(self, maturation_rate: float = 0.01):
        self.maturation_rate = maturation_rate

    def register_sanitizer(self, module: ExtendedDendriticModuleAdapter, idx: int):
        # Register a backward hook to scale gradients dynamically during training
        def backward_hook(grad: torch.Tensor) -> torch.Tensor:
            state = module.branch_states[f"state_{idx}"]
            age = state[0].item()
            # Gradual transition scale based on branch age
            scale_factor = 1.0 - torch.exp(torch.tensor(-self.maturation_rate * age, device=grad.device, dtype=grad.dtype))
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
    def __init__(self, calcium_decay: float = 0.12, safety_ceiling: float = 4.0):
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
        self.calcium_store.data = (1.0 - self.calcium_decay) * self.calcium_store.data + mean_activation
        if self.calcium_store.item() > self.safety_ceiling:
            # Multiplicative downscaling to damp hyper-excitation
            attenuation = self.safety_ceiling / (self.calcium_store.item() + 1e-6)
            return x * attenuation
        return x

    def evaluate_growth_safety(self) -> bool:
        """Prevents neurogenesis events if the network is currently hyper-excited."""
        return self.calcium_store.item() < self.safety_ceiling * 0.8
