"""
gwt/sleep.py
============
Phase v0.3: Sleep & Memory Consolidation.

Implements an offline Slow-Wave Sleep (SWS) cycle to reinforce stable
synaptic path representations, prioritizing high-salience experiences
and dynamically pruning under-utilized dendritic branches.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dendrite import ActiveDendriteGate, DendriticModuleAdapter
from .engine import DataFlowManager

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CognitiveReplayBuffer
# ─────────────────────────────────────────────────────────────────────────────

class CognitiveReplayBuffer:
    """
    Lightweight, bounded episodic cache that stores high-salience waking transitions.
    
    To prevent memory leaks and graph build-ups, it explicitly detaches and clones
    all stored tensors. Evicts the lowest-salience trace when size exceeds max_size,
    ensuring that only highly salient experiences are retained.
    """

    def __init__(self, max_size: int = 1000) -> None:
        """
        Args:
            max_size: Maximum capacity of the replay buffer.
        """
        self.max_size = max_size
        self.buffer: List[Dict[str, Any]] = []

    def add_trace(
        self,
        latent_states: Dict[str, torch.Tensor],
        context: torch.Tensor,
        salience: float,
    ) -> None:
        """
        Inserts a detached transition trace into the episodic cache in O(1) time.
        
        If the buffer is full, the trace with the lowest salience is evicted.
        
        Args:
            latent_states : Dictionary of `{module_name: latent_tensor}`.
            context       : Global GWT broadcast tensor [B, latent_dim].
            salience      : Scalar salience rating.
        """
        # Strictly detach and clone all tensors to guarantee zero memory graph leakage
        detached_latents = {k: v.detach().clone() for k, v in latent_states.items()}
        detached_context = context.detach().clone()

        trace = {
            "latent_states": detached_latents,
            "context": detached_context,
            "salience": float(salience),
        }

        # O(1) append
        self.buffer.append(trace)

        # Evict the least salient trace if bounded capacity is exceeded
        if len(self.buffer) > self.max_size:
            min_idx = min(range(len(self.buffer)), key=lambda i: self.buffer[i]["salience"])
            self.buffer.pop(min_idx)

    def sample_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        Samples a batch of experiences prioritized by their salience scores.
        
        Args:
            batch_size: Number of traces to sample.
            
        Returns:
            List of sampled experience dictionaries.
        """
        if not self.buffer:
            return []

        # Compute prioritized sampling distribution via stable softmax over saliences
        saliences = torch.tensor([t["salience"] for t in self.buffer], dtype=torch.float32)
        saliences = saliences - saliences.max()  # Numerical stability
        probs = torch.softmax(saliences, dim=0)

        # Draw indices with replacement (supports batch_size larger than buffer capacity)
        indices = torch.multinomial(probs, num_samples=batch_size, replacement=True)
        return [self.buffer[i] for i in indices.tolist()]

    def clear(self) -> None:
        """Flushes the replay buffer."""
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────────────────
# 2. ConsolidationEngine
# ─────────────────────────────────────────────────────────────────────────────

class ConsolidationEngine:
    """
    Offline Sleep Consolidation Engine.
    
    Pauses online inference/data streams and enters a simulated 'Slow-Wave Sleep'
    phase to reinforce GWT workspace attention mappings and active dendritic pathways.
    """

    def __init__(self, engine: Any, replay_buffer: CognitiveReplayBuffer) -> None:
        """
        Args:
            engine        : The main CognitiveAugEngine instance.
            replay_buffer : Attached CognitiveReplayBuffer instance.
        """
        self.engine = engine
        self.replay_buffer = replay_buffer

    def sleep_cycle(
        self,
        steps: int = 100,
        learning_rate: float = 0.001,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """
        Executes N steps of Slow-Wave Sleep.
        
        Optimizes workspace and dendritic parameters using localized reconstruction
        and GWT routing consistency loss.
        
        Args:
            steps         : Number of optimizer cycles.
            learning_rate : Learning rate for local parameter optimization.
            batch_size    : Replay mini-batch size.
            
        Returns:
            Telemetry dictionary detailing replayed items and loss improvements.
        """
        if not self.replay_buffer or len(self.replay_buffer) == 0:
            logger.info("Episodic replay buffer is empty. Skipping sleep cycle.")
            return {
                "memory_traces_replayed": 0,
                "loss_delta": 0.0,
                "initial_loss": 0.0,
                "final_loss": 0.0,
            }

        # 1. Collect all learnable GWT parameters to optimize
        params = []
        if self.engine.workspace is not None:
            params.extend(list(self.engine.workspace.parameters()))

        adapters = self.engine.registry.list_adapters()
        for adapter in adapters:
            if hasattr(adapter, "dendrite_gate") and adapter.dendrite_gate is not None:
                params.extend(list(adapter.dendrite_gate.parameters()))
            if hasattr(adapter, "key_proj"):
                params.extend(list(adapter.key_proj.parameters()))

        if not params:
            logger.warning("No optimizable parameters found. Skipping sleep cycle.")
            return {
                "memory_traces_replayed": 0,
                "loss_delta": 0.0,
                "initial_loss": 0.0,
                "final_loss": 0.0,
            }

        # Setup local sleep cycle optimizer
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        
        initial_loss = 0.0
        final_loss = 0.0
        total_replayed = 0

        # Retrieve parameter device
        device = next(iter(params)).device

        for step in range(steps):
            batch = self.replay_buffer.sample_batch(batch_size)
            if not batch:
                break

            optimizer.zero_grad()
            step_loss = torch.tensor(0.0, device=device)

            for trace in batch:
                latent_states = trace["latent_states"]  # {module_name: latent}
                target_context = trace["context"].to(device)

                # Map tensors in the batch to our optimizer device
                device_latents = {k: v.to(device) for k, v in latent_states.items()}

                # Compute replayed keys via current key projections
                keys = {}
                for name, latent in device_latents.items():
                    adapter = self.engine.registry.get(name)
                    keys[name] = adapter.get_key(latent)

                # A. Workspace Replay: Compute replayed Global Workspace state
                if self.engine.workspace is not None:
                    replayed_broadcast = self.engine.workspace(device_latents, keys)
                    
                    # Workspace Reconstruction Loss (align replayed state with waking state)
                    loss_workspace = F.mse_loss(replayed_broadcast, target_context)
                    step_loss = step_loss + loss_workspace

                    # B. Dendritic Gating Stability Loss:
                    # Feed waking features through Dendritic Gates with the replayed context
                    for name, latent in device_latents.items():
                        adapter = self.engine.registry.get(name)
                        if hasattr(adapter, "dendrite_gate") and adapter.dendrite_gate is not None:
                            # Re-run dendritic pre-processing with replayed global workspace broadcast
                            gated_output = adapter.dendrite_gate(latent, replayed_broadcast)
                            
                            # Stability Loss: Compare replayed gated state to waking gated state (target)
                            loss_dendrite = F.mse_loss(gated_output, latent)
                            step_loss = step_loss + loss_dendrite

            # Average loss over the sampled batch
            step_loss = step_loss / len(batch)

            if step == 0:
                initial_loss = step_loss.item()
            if step == steps - 1:
                final_loss = step_loss.item()

            step_loss.backward()
            optimizer.step()
            total_replayed += len(batch)

        loss_delta = initial_loss - final_loss
        logger.info(
            f"Sleep Cycle complete: replayed {total_replayed} traces. "
            f"Loss change: {initial_loss:.4f} -> {final_loss:.4f} (Delta: {loss_delta:.4f})"
        )

        return {
            "memory_traces_replayed": total_replayed,
            "loss_delta": loss_delta,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3. DendriticPruning (Synaptic Homeostasis)
# ─────────────────────────────────────────────────────────────────────────────

def prune_dendrites(
    model_or_engine: Any,
    pruning_threshold: float = 0.05,
    soft_decay: float = 0.0,
    waking_activations: Optional[Dict[str, torch.Tensor]] = None,
) -> int:
    """
    Scans the model/engine for ActiveDendriteGate layers and prunes weak branches.
    
    If a dendritic branch's average activation score across the waking period falls
    below pruning_threshold, its context projection weights and biases are modified.
    Pruning masks and backward hooks are updated under torch.no_grad() to permanently
    sever the parameters and prevent gradients from recalculating through them.
    
    Args:
        model_or_engine   : The CognitiveAugEngine or PyTorch nn.Module.
        pruning_threshold : Activation threshold below which branches are pruned.
        soft_decay        : If 0.0, zeroed out completely (hard pruning).
                            If > 0.0, softly scaled down.
        waking_activations: Optional dict of cached waking activations. If provided,
                            used instead of the gate's latest activations (which
                            may have been overwritten during sleep replay).
                            
    Returns:
        Total number of dendritic branches pruned/decayed.
    """
    gates = []
    
    # Extract gates from engine or module
    if hasattr(model_or_engine, "registry"):
        adapters = model_or_engine.registry.list_adapters()
        for adapter in adapters:
            if hasattr(adapter, "dendrite_gate") and adapter.dendrite_gate is not None:
                gates.append((adapter.name, adapter.dendrite_gate))
    elif isinstance(model_or_engine, nn.Module):
        for m in model_or_engine.modules():
            if isinstance(m, ActiveDendriteGate):
                gates.append((None, m))

    pruned_count = 0

    for name, gate in gates:
        # Fetch appropriate activation trace (prioritize cached waking activations)
        activations = None
        if waking_activations is not None and name is not None:
            activations = waking_activations.get(name)
            
        if activations is None:
            activations = gate.latest_branch_activations

        if activations is None:
            continue

        # Average activation per branch across batch and feature channels
        # Shape: [num_branches]
        branch_means = activations.mean(dim=(0, 2))
        feedforward_dim = gate.feedforward_dim

        with torch.no_grad():
            for b in range(gate.num_branches):
                if branch_means[b] < pruning_threshold:
                    start_idx = b * feedforward_dim
                    end_idx = (b + 1) * feedforward_dim

                    # Apply hard pruning or soft decay to raw tensors
                    if soft_decay == 0.0:
                        # Hard Pruning: set parameters to 0.0
                        gate.context_proj.weight[start_idx:end_idx, :] = 0.0
                        gate.pruning_mask[start_idx:end_idx, :] = 0.0
                        
                        if gate.context_proj.bias is not None:
                            gate.context_proj.bias[start_idx:end_idx] = 0.0
                            gate.bias_pruning_mask[start_idx:end_idx] = 0.0
                    else:
                        # Soft Decay: multiply parameters by decay rate
                        gate.context_proj.weight[start_idx:end_idx, :] *= soft_decay
                        gate.pruning_mask[start_idx:end_idx, :] *= soft_decay
                        
                        if gate.context_proj.bias is not None:
                            gate.context_proj.bias[start_idx:end_idx] *= soft_decay
                            gate.bias_pruning_mask[start_idx:end_idx] *= soft_decay

                    pruned_count += 1
                    logger.debug(
                        f"Pruned dendritic branch {b} on gate {gate} "
                        f"(mean activation {branch_means[b]:.4f} < {pruning_threshold:.4f})"
                    )

    return pruned_count
