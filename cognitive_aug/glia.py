"""
cognitive_aug/glia.py
===========
Glial-Inspired Learning Regulation for GWT (Phase v0.5).

Astrocytes form tripartite synapses that actively monitor synaptic traffic to
implement metaplasticity (modulating local learning rates), clear excess
excitotoxicity (gradient stabilization), and optimize the localized training process.
"""

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class GradientSanitizerHook:
    """
    Gradient Sanitizer Hook (Excitotoxicity Defense).

    Automatically registers backward hooks onto parameter tensors of wrapped module
    subsystems. Instead of a static global clip, it tracks running gradient norm
    statistics locally under `with torch.no_grad():`. If sudden gradient spikes occur,
    it isolatedly scales down the gradients of ONLY that specific subsystem.
    """

    def __init__(
        self,
        adapter: nn.Module,
        max_variance_threshold: float = 3.0,
        damping_factor: float = 0.2,
        alpha: float = 0.95,
    ) -> None:
        """
        Args:
            adapter                : The module adapter subsystem to protect.
            max_variance_threshold : Number of standard deviations above the running mean
                                     at which a gradient is flagged as an excitotoxic spike.
            damping_factor         : Scale factor to multiply the spiking gradient by (e.g. 0.2).
            alpha                  : Smoothing factor for the exponential moving average (EMA)
                                     of gradient norm mean and variance.
        """
        self.adapter = adapter
        self.max_variance_threshold = max_variance_threshold
        self.damping_factor = damping_factor
        self.alpha = alpha

        # State tracking: "Stable" or "Damped"
        self.grad_status = "Stable"

        # Running statistics for gradient norm per parameter
        self.running_mean: Dict[int, float] = {}
        self.running_var: Dict[int, float] = {}
        self.step_count: Dict[int, int] = {}

        self.hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Finds all parameters in the adapter subsystem and registers backward hooks."""
        for param in self.adapter.parameters():
            if param.requires_grad:
                param_id = id(param)
                self.running_mean[param_id] = 0.0
                # Initialize running variance to a small positive epsilon
                self.running_var[param_id] = 1e-4
                self.step_count[param_id] = 0

                # Register the PyTorch backward hook
                handle = param.register_hook(self._make_hook(param_id))
                self.hook_handles.append(handle)

        logger.debug(
            f"Registered {len(self.hook_handles)} gradient sanitizer hooks on module '{self.adapter.name}'."
        )

    def _make_hook(self, param_id: int):
        """Creates the closure hook function for a specific parameter."""

        def hook_fn(grad: torch.Tensor) -> Optional[torch.Tensor]:
            # Always run monitoring and calculations under no_grad to prevent graph overhead
            with torch.no_grad():
                if grad is None:
                    return None

                # 1. Compute local gradient norm (L2 norm)
                g = float(torch.linalg.vector_norm(grad).item())

                mean = self.running_mean[param_id]
                var = self.running_var[param_id]
                std = var**0.5
                steps = self.step_count[param_id]

                # 2. Excitotoxicity spike check:
                # If we have collected a baseline (e.g. > 5 steps) and the norm spikes
                # beyond max_variance_threshold standard deviations, damp it.
                if steps > 5 and g > mean + self.max_variance_threshold * std:
                    self.grad_status = "Damped"
                    logger.warning(
                        f"Excitotoxicity detected in subsystem '{self.adapter.name}'! "
                        f"Gradient norm spiked to {g:.4f} (mean={mean:.4f}, std={std:.4f}). "
                        f"Damping by factor {self.damping_factor}."
                    )
                    return grad * self.damping_factor

                # 3. Update running statistics with EMA
                self.step_count[param_id] += 1
                self.running_mean[param_id] = self.alpha * mean + (1.0 - self.alpha) * g
                # Variance: EMA of squared deviations
                self.running_var[param_id] = self.alpha * var + (1.0 - self.alpha) * (
                    (g - mean) ** 2
                )

                return None

        return hook_fn

    def remove_hooks(self) -> None:
        """Safely removes all registered backward hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        logger.debug(
            f"Removed all gradient sanitizer hooks from '{self.adapter.name}'."
        )

    def __del__(self) -> None:
        self.remove_hooks()


class AstrocyteManager(nn.Module):
    """
    AstrocyteManager Layer.

    A tripartite synapse model that actively monitors synaptic traffic to implement
    metaplasticity (modulating local learning rates based on ACh focus vs surprise NE)
    and handles dynamic gradient stabilization.
    """

    def __init__(
        self,
        ema_alpha: float = 0.9,
        lr_lock_scale: float = 0.5,
        lr_unlock_scale: float = 1.5,
        max_variance_threshold: float = 3.0,
        damping_factor: float = 0.2,
    ) -> None:
        """
        Args:
            ema_alpha              : Smoothing factor for module saliences EMA tracking.
            lr_lock_scale          : Scale factor to damp learning rate in highly stable states (ACh high, NE low).
            lr_unlock_scale        : Scale factor to boost learning rate in surprising/salient states (NE high).
            max_variance_threshold : STD multiplier for gradient spike filtering inside Sanitizer Hooks.
            damping_factor         : Scale factor for gradient spike damping inside Sanitizer Hooks.
        """
        super().__init__()
        self.ema_alpha = ema_alpha
        self.lr_lock_scale = lr_lock_scale
        self.lr_unlock_scale = lr_unlock_scale
        self.max_variance_threshold = max_variance_threshold
        self.damping_factor = damping_factor

        self.engine: Optional[Any] = None

        # State tracking: module name -> EMA tracking score
        self._ema_saliences: Dict[str, float] = {}
        # Subsystem protection: module name -> GradientSanitizerHook
        self._sanitizer_hooks: Dict[str, GradientSanitizerHook] = {}
        # Dynamic plasticity scales for diagnostic printout: module name -> scale float
        self._plasticity_scales: Dict[str, float] = {}

        # Cache baseline and last applied learning rates to support idempotent scaling and schedulers
        self._baseline_lrs: Dict[int, float] = {}
        self._last_applied_lrs: Dict[int, float] = {}

    def attach(self, engine: Any) -> None:
        """
        Attaches the AstrocyteManager to a CognitiveAugEngine and registers sanitizers.
        """
        self.engine = engine
        self._ema_saliences.clear()
        self._baseline_lrs.clear()
        self._last_applied_lrs.clear()

        # Clean up any existing hook handles
        for hook in self._sanitizer_hooks.values():
            hook.remove_hooks()
        self._sanitizer_hooks.clear()

        # Register sanitizer hooks for all registered modules
        for adapter in engine.registry.list_adapters():
            hook = GradientSanitizerHook(
                adapter=adapter,
                max_variance_threshold=self.max_variance_threshold,
                damping_factor=self.damping_factor,
            )
            self._sanitizer_hooks[adapter.name] = hook
            self._plasticity_scales[adapter.name] = 1.0

        logger.info("Successfully attached AstrocyteManager to CognitiveAugEngine.")

    def update(self, engine: Any) -> None:
        """
        Updates running tracking states (EMA of module-level saliences) from DataFlowManager.
        Runs entirely under torch.no_grad(). Called automatically during engine.step().
        """
        with torch.no_grad():
            # Dynamically register hooks for newly registered module adapters
            for adapter in engine.registry.list_adapters():
                if adapter.name not in self._sanitizer_hooks:
                    hook = GradientSanitizerHook(
                        adapter=adapter,
                        max_variance_threshold=self.max_variance_threshold,
                        damping_factor=self.damping_factor,
                    )
                    self._sanitizer_hooks[adapter.name] = hook
                    self._plasticity_scales[adapter.name] = 1.0

            # Compute and track EMA of saliences
            current_saliences = engine.data_flow.list_saliences()
            for name in engine.registry.list_names():
                score = current_saliences.get(name, 0.5)
                if name not in self._ema_saliences:
                    self._ema_saliences[name] = score
                else:
                    self._ema_saliences[name] = (
                        self.ema_alpha * self._ema_saliences[name]
                        + (1.0 - self.ema_alpha) * score
                    )

    def adjust_learning_rates(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Metaplasticity Optimizer Modifier.

        Computes localized learning rate scaling factors in a highly optimized,
        vectorized PyTorch manner, and modulates the learning rates inside the
        optimizer's corresponding param_groups in-place.

        Schedules:
            - High focus (ACh) + Low surprise (NE) = lock down stable memories (lr * 0.5)
            - High NE + High salience = unlock weights to accelerate learning (lr * 1.5)
        """
        if self.engine is None:
            raise ValueError(
                "AstrocyteManager is not attached to any CognitiveAugEngine."
            )

        # Reset all gradient sanitizer statuses to Stable at the start of each optimization step
        for hook in self._sanitizer_hooks.values():
            hook.grad_status = "Stable"

        # Fetch global metacognitive neuromodulator levels (if active)
        ach = 0.0
        ne = 0.0
        if self.engine.neuromodulator is not None:
            ach = getattr(self.engine.neuromodulator, "ach", 0.0)
            ne = getattr(self.engine.neuromodulator, "ne", 0.0)

        names = self.engine.registry.list_names()
        if not names:
            return

        # Vectorized calculation under no_grad
        with torch.no_grad():
            ach_t = torch.tensor(ach, dtype=torch.float32, device="cpu")
            ne_t = torch.tensor(ne, dtype=torch.float32, device="cpu")
            saliences_t = torch.tensor(
                [self._ema_saliences.get(name, 0.5) for name in names],
                dtype=torch.float32,
                device="cpu",
            )

            # Continuous vectorized metaplasticity formula:
            # eta = 1.0 + (unlock_scale - 1) * NE * salience - (1 - lock_scale) * ACh * (1 - NE)
            # High ACh + Low NE -> 1.0 - 0.5 * 1.0 * 1.0 = 0.5
            # High NE + High salience -> 1.0 + 0.5 * 1.0 * 1.0 = 1.5
            etas = (
                1.0
                + (self.lr_unlock_scale - 1.0) * ne_t * saliences_t
                - (1.0 - self.lr_lock_scale) * ach_t * (1.0 - ne_t)
            )
            # Bound learning rate scaling factor to avoid zero/negative learning rates
            etas = torch.clamp(etas, min=1e-3)

            # Store the computed scaling factors
            scales = {name: float(etas[i].item()) for i, name in enumerate(names)}
            self._plasticity_scales = scales

            # Modulate optimizer param_groups in-place
            for group in optimizer.param_groups:
                group_id = id(group)

                module_name = self._get_group_module_name(group, self.engine)
                if module_name is None:
                    continue

                scale = scales.get(module_name, 1.0)
                current_lr = group["lr"]

                # Handle dynamic scheduler or external changes self-healingly
                if group_id in self._last_applied_lrs:
                    # If current_lr does not match what we set last time, someone else
                    # (like a scheduler) changed it. Treat that as the new baseline!
                    if abs(current_lr - self._last_applied_lrs[group_id]) > 1e-12:
                        self._baseline_lrs[group_id] = current_lr
                else:
                    self._baseline_lrs[group_id] = current_lr

                # Scale the baseline learning rate in-place
                new_lr = self._baseline_lrs[group_id] * scale
                group["lr"] = new_lr
                self._last_applied_lrs[group_id] = new_lr

    def _get_group_module_name(
        self, group: Dict[str, Any], engine: Any
    ) -> Optional[str]:
        """Maps an optimizer param_group to a registered module adapter name."""
        # 1. Custom explicit metadata tag
        if "module_name" in group:
            return group["module_name"]

        # 2. Identity intersection of parameters
        group_params = set(group["params"])
        for adapter in engine.registry.list_adapters():
            adapter_params = set(adapter.parameters())
            if group_params.intersection(adapter_params):
                return adapter.name

        return None

    def remove_all_hooks(self) -> None:
        """Safely removes all registered backward sanitizer hooks."""
        for hook in self._sanitizer_hooks.values():
            hook.remove_hooks()
        self._sanitizer_hooks.clear()

    def __del__(self) -> None:
        self.remove_all_hooks()
