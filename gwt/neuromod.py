"""
gwt/neuromod.py
===============
Metacognitive Neuromodulation for GWT (Phase v0.4).

Mimics chemical neuromodulation (Acetylcholine and Norepinephrine) to adjust
attention selection thresholds and active dendritic gating parameters dynamically
in-place based on metacognitive surprise, representation entropy, and goal alignment.
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .salience import EntropySalience, TemporalSurpriseSalience

logger = logging.getLogger(__name__)


def make_ascii_bar(value: float, length: int = 8) -> str:
    """
    Generates a high-contrast horizontal progress bar of a given length.
    
    Args:
        value  : Value between 0.0 and 1.0.
        length : Total block characters in the bar.
        
    Returns:
        ASCII progress bar string (e.g. '▇▇▇▇░░░░').
    """
    val = max(0.0, min(1.0, float(value)))
    num_blocks = int(round(val * length))
    return "▇" * num_blocks + "░" * (length - num_blocks)


class DynamicThresholdAdapter:
    """
    Applies in-place modifications to CognitiveAugEngine parameters based on chemical levels.
    
    Tunes GWT workspace selector ignition thresholds and active dendritic gate
    depolarization spike thresholds and gain temperatures.
    """

    def __init__(
        self,
        beta: float = 0.4,
        gamma: float = 0.3,
        threshold_coef: float = 0.3,
        temp_coef: float = 0.5,
    ) -> None:
        """
        Args:
            beta           : Coefficient for NE drop on GWT ignition threshold.
            gamma          : Coefficient for ACh rise on GWT ignition threshold.
            threshold_coef : Coefficient for ACh focus on dendritic spike threshold.
            temp_coef      : Coefficient for ACh focus on dendritic gain temperature.
        """
        self.beta = beta
        self.gamma = gamma
        self.threshold_coef = threshold_coef
        self.temp_coef = temp_coef

        # Cache baseline parameters to apply modulations relative to initial states
        self.baseline_ignition_threshold: Optional[float] = None
        self.baseline_dendrite_thresholds: Dict[nn.Module, float] = {}

    def apply(self, engine: Any, ne: float, ach: float) -> None:
        """
        Applies in-place modulations to engine parameters under torch.no_grad().
        
        Args:
            engine : CognitiveAugEngine instance.
            ne     : Current Norepinephrine level [0, 1].
            ach    : Current Acetylcholine level [0, 1].
        """
        # 1. Modulate GlobalWorkspace ignition threshold
        if engine.workspace is not None:
            selector = getattr(engine.workspace, "selector", None)
            if selector is not None and hasattr(selector, "ignition_threshold"):
                if self.baseline_ignition_threshold is None:
                    self.baseline_ignition_threshold = float(selector.ignition_threshold)
                
                # Formula: ignition_threshold = baseline - (beta * NE) + (gamma * ACh)
                new_thresh = self.baseline_ignition_threshold - (self.beta * ne) + (self.gamma * ach)
                selector.ignition_threshold = max(0.0, min(1.0, float(new_thresh)))
                logger.debug(f"Modulated GWT ignition threshold to {selector.ignition_threshold:.4f}")

        # 2. Modulate ActiveDendriteGate layers in registered adapters
        for adapter in engine.registry.list_adapters():
            gate = getattr(adapter, "dendrite_gate", None)
            if gate is not None:
                if gate not in self.baseline_dendrite_thresholds:
                    self.baseline_dendrite_thresholds[gate] = float(gate.threshold)
                
                # Formula 1: NMDA threshold spiking increases with ACh focus
                # threshold_t = baseline_threshold + 0.3 * ACh_t
                base_thresh = self.baseline_dendrite_thresholds[gate]
                gate.threshold = max(0.0, min(1.0, float(base_thresh + self.threshold_coef * ach)))
                
                # Formula 2: Gain Sigmoid Temperature drops with ACh focus
                # temp_t = 1.0 - 0.5 * ACh_t
                gate.gain_temperature = max(1e-5, min(1.0, float(1.0 - self.temp_coef * ach)))
                logger.debug(
                    f"Modulated dendrite gate '{adapter.name}': "
                    f"threshold={gate.threshold:.4f}, temperature={gate.gain_temperature:.4f}"
                )


class MetacognitiveMonitor:
    """
    Metacognitive chemical controller managing Acetylcholine (ACh) and Norepinephrine (NE).
    
    Monitors engine buffers and workspace queries, computes neurotransmitter curves,
    and invokes parameter adapters to regulate attention gating in-place.
    """

    def __init__(
        self,
        alpha_ne: float = 0.8,
        alpha_ach: float = 0.8,
        beta: float = 0.4,
        gamma: float = 0.3,
        threshold_coef: float = 0.3,
        temp_coef: float = 0.5,
    ) -> None:
        """
        Args:
            alpha_ne       : Smoothing factor for Norepinephrine (decay rate).
            alpha_ach      : Smoothing factor for Acetylcholine (decay rate).
            beta           : Dynamic ignition threshold drop coefficient for NE.
            gamma          : Dynamic ignition threshold rise coefficient for ACh.
            threshold_coef : Dendritic spike threshold increase coefficient for ACh.
            temp_coef      : Dendritic temperature sharpening coefficient for ACh.
        """
        self.alpha_ne = alpha_ne
        self.alpha_ach = alpha_ach

        # Virtual chemical neurotransmitter levels [0.0, 1.0]
        self.ne = 0.0
        self.ach = 0.0

        # Stateful salience metric modules
        self.surprise_metric = TemporalSurpriseSalience()
        self.entropy_metric = EntropySalience()

        # Knob adapter for in-place modulation
        self.adapter = DynamicThresholdAdapter(
            beta=beta,
            gamma=gamma,
            threshold_coef=threshold_coef,
            temp_coef=temp_coef,
        )

    def modulate(self, engine: Any) -> None:
        """
        Evaluates current step telemetry, updates chemical levels, and applies
        in-place parameter modulations to the engine. Runs entirely under torch.no_grad().
        
        Args:
            engine : CognitiveAugEngine instance.
        """
        with torch.no_grad():
            # 1. Collect currently active module latent states
            latent_states: Dict[str, torch.Tensor] = {}
            for name in engine.registry.list_names():
                try:
                    latent_states[name] = engine.data_flow.get_buffer(name)
                except KeyError:
                    continue

            # Default values if no active modules are running yet
            surprise_val = 0.0
            confidence_val = 1.0

            if latent_states:
                # Surprise: measured via temporal surprise salience over time steps
                surprise_tensor = self.surprise_metric(latent_states)
                surprise_val = float(surprise_tensor.mean().item())

                # Entropy focus: 1.0 - normalized entropy (higher = more confident)
                confidence_tensor = self.entropy_metric(latent_states)
                confidence_val = float(confidence_tensor.mean().item())

            # 2. Alignment: cosine similarity of GWT broadcast to workspace's target query
            alignment = 1.0
            workspace = engine.workspace
            if workspace is not None:
                last_weights = getattr(workspace, "last_weights", None)
                last_query = getattr(workspace, "last_query", None)
                last_keys = getattr(workspace, "last_keys", None)

                if last_weights is not None and last_query is not None and last_keys is not None:
                    # Reconstruct GWT key from stack using attention weights
                    # last_keys shape: [B, num_modules, key_dim]
                    # last_weights shape: [B, num_modules]
                    workspace_key = (last_keys * last_weights.unsqueeze(-1)).sum(dim=1)  # [B, key_dim]
                    cos_sim = F.cosine_similarity(workspace_key, last_query, dim=-1)     # [B]
                    alignment = max(0.0, min(1.0, float(cos_sim.mean().item())))

            # 3. Focus = Alignment * (1.0 - Entropy) = Alignment * Confidence
            focus = alignment * confidence_val

            # 4. Exponential smoothing chemical curves
            # NE_t = alpha_ne * NE_{t-1} + (1 - alpha_ne) * Surprise_t
            self.ne = self.alpha_ne * self.ne + (1.0 - self.alpha_ne) * surprise_val
            # ACh_t = alpha_ach * ACh_{t-1} + (1 - alpha_ach) * Focus_t
            self.ach = self.alpha_ach * self.ach + (1.0 - self.alpha_ach) * focus

            # Clamp chemicals to keep values in strict [0, 1] range
            self.ne = max(0.0, min(1.0, self.ne))
            self.ach = max(0.0, min(1.0, self.ach))

            # 5. Apply dynamic modulations in-place
            self.adapter.apply(engine, self.ne, self.ach)

    def get_chemical_levels(self) -> Dict[str, Any]:
        """
        Retrieves current raw levels and ASCII progress bar representations.
        
        Returns:
            Dictionary containing chemical statuses.
        """
        ach_bar = make_ascii_bar(self.ach)
        ne_bar = make_ascii_bar(self.ne)
        dashboard = f"[ ACh: {ach_bar} {self.ach:.2f} | NE: {ne_bar} {self.ne:.2f} ]"

        return {
            "ne": self.ne,
            "ach": self.ach,
            "ach_bar": ach_bar,
            "ne_bar": ne_bar,
            "dashboard": dashboard,
        }
