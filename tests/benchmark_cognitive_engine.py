"""
tests/benchmark_cognitive_engine.py
===================================
Concise, production-ready benchmarking script to verify the end-to-end
execution of the 7-phase brain-inspired GWT cognitive architecture.

Verifies:
1. Active Dendritic Gating (Phase v0.2)
2. Sleep & Memory Consolidation (Phase v0.3)
3. Metacognitive Neuromodulation (Phase v0.4)
4. Glial-Inspired Learning Regulation (Phase v0.5)
5. Concept-Level Representation & Intervention (Phase v0.6)
6. Cross-Modal Cognitive Crossbar Routing (Phase v0.7)

This script is entirely self-contained, device-agnostic, and prints a beautiful
ASCII terminal summary containing performance wall time, pre vs. post shift losses,
and the full output of engine.inspect().
"""

import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from gwt import (
    CognitiveAugEngine,
    GlobalWorkspace,
    MetacognitiveMonitor,
    AstrocyteManager,
    ConceptLayer,
    CognitiveCrossbar,
    CognitiveReplayBuffer,
    DendriticModuleAdapter,
    CrossbarModuleAdapter,
    global_pool_latent,
)


class BenchmarkHybridAdapter(DendriticModuleAdapter):
    """
    Custom hybrid adapter combining ActiveDendriteGate (Phase v0.2) and
    CognitiveCrossbar writing (Phase v0.7) for efficient multi-modal routing.
    """

    def __init__(
        self,
        name: str,
        module: nn.Module,
        latent_dim: int,
        data_flow: Any,
        slot_idx: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            module=module,
            latent_dim=latent_dim,
            data_flow=data_flow,
            **kwargs,
        )
        self.slot_idx = slot_idx
        self.engine: Optional[Any] = None

    def _register_forward_hook(self) -> None:
        """Hooks up both active dendritic gating and crossbar slot writing."""

        def hook_fn(module: nn.Module, inputs: Any, outputs: Any) -> Any:
            latent = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

            if not isinstance(latent, torch.Tensor):
                raise TypeError(f"Expected tensor output from module '{self.name}', got {type(outputs)}")

            # Initialize dendritic gate lazily if needed
            if self.dendrite_gate is None:
                self._init_dendrite_gate(latent.shape[-1])

            # Retrieve dynamic context from GWT
            context = self.get_last_broadcast()
            batch_size = latent.shape[0]
            if context.shape[0] == 1 and batch_size > 1:
                context = context.expand(batch_size, -1)
            elif context.shape[0] != batch_size:
                context = torch.zeros(batch_size, self.latent_dim, device=latent.device, dtype=latent.dtype)

            # 1. Apply Active Dendritic Gating (Phase v0.2)
            gated_latent = self.dendrite_gate(latent, context)

            # 2. Pool to flat 2D workspace format (Dimension Agnosticism)
            gwt_latent = global_pool_latent(gated_latent)

            # Apply projection layer if dimension mismatch exists
            if self.projection is not None:
                gwt_latent = self.projection(gwt_latent)

            # 3. Write to Cognitive Crossbar (Phase v0.7)
            if self.engine is not None and self.engine.crossbar is not None:
                self.engine.crossbar.write_slot(self.slot_idx, gwt_latent)

            # 4. Route to standard GWT DataFlow buffer
            self.data_flow.update_buffer(self.name, gwt_latent)

            return gated_latent

        self._hook_handle = self.module.register_forward_hook(hook_fn)


def main() -> None:
    # Set console encoding to UTF-8 to handle GWT dashboard emojis and symbols on Windows
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # ── Device Setup ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running benchmark on device: {device}")

    # ── 1. Lightweight Multi-Task Setup ──────────────────────────────────────
    torch.manual_seed(42)

    # Task A: Spatial/Vision input representation
    task_a_input = torch.randn(32, 8, device=device)
    task_a_target = torch.randn(32, 4, device=device)

    # Task B: Semantic/Text input representation
    task_b_input = torch.randn(32, 8, device=device)
    task_b_target = torch.randn(32, 4, device=device)

    # Shared model tower Linear layer mapping [B, 8] -> [B, 4]
    tower = nn.Linear(8, 4).to(device)

    # ── 2. Initialize GWT Cognitive Architecture ──────────────────────────────
    latent_dim = 4

    engine = CognitiveAugEngine()

    # Phase v0.1: Global Workspace attention selection & broadcast
    workspace = GlobalWorkspace(
        latent_dim=latent_dim,
        key_dim=8,
        attention_type="key-query",
        selection_mode="soft",
        ignition_threshold=0.5,
    ).to(device)
    engine.attach_workspace(workspace)

    # Phase v0.3: Memory Replay Buffer
    replay_buffer = CognitiveReplayBuffer(max_size=100)
    engine.attach_replay_buffer(replay_buffer)

    # Phase v0.4: Metacognitive Monitor
    monitor = MetacognitiveMonitor(alpha_ne=0.3, alpha_ach=0.3)
    engine.attach_neuromodulator(monitor)

    # Phase v0.5: Glial-Inspired learning rate local modifier
    glia_manager = AstrocyteManager(lr_lock_scale=0.5, lr_unlock_scale=1.5)
    engine.attach_glial_manager(glia_manager)

    # Phase v0.6: Concept-Level Representation Layer
    concept_names = ["VisionFocused", "TextFocused", "Exploratory", "Consolidated"]
    concept_layer = ConceptLayer(
        input_dim=latent_dim,
        num_concepts=4,
        abstraction_type="projection",
        concept_names=concept_names,
    ).to(device)
    engine.attach_concept_layer("concept_layer", concept_layer)

    # Phase v0.7: Cross-Modal Cognitive Crossbar
    slot_names = ["TowerSlot", "ConceptSlot"]
    crossbar = CognitiveCrossbar(slot_dim=latent_dim, num_slots=2, slot_names=slot_names).to(device)
    engine.attach_crossbar(crossbar)

    # Wrap the tower using the custom benchmark hybrid adapter (slot 0)
    tower_adapter = BenchmarkHybridAdapter(
        name="tower_module",
        module=tower,
        latent_dim=latent_dim,
        data_flow=engine.data_flow,
        slot_idx=0,
        num_branches=4,
        spike_type="nmda-threshold",
        threshold=0.4,
    )
    engine.registry.register("tower_module", tower_adapter)
    tower_adapter.engine = engine

    # Wrap the concept layer with standard CrossbarModuleAdapter (slot 1)
    concept_adapter = CrossbarModuleAdapter(
        name="concept_module",
        module=concept_layer,
        latent_dim=latent_dim,
        data_flow=engine.data_flow,
        slot_idx=1,
    )
    engine.registry.register("concept_module", concept_adapter)
    concept_adapter.engine = engine

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # Standard PyTorch SGD Optimizer covering the model tower parameters
    optimizer = torch.optim.SGD(tower.parameters(), lr=0.1)

    # ── 3. Rapid Verification Execution Loop ──────────────────────────────────
    start_time = time.perf_counter()

    pre_shift_loss = 0.0
    post_shift_loss = 0.0
    initial_ne = monitor.ne
    initial_ach = monitor.ach

    print("[*] Starting waking training loop on Task A (Steps 1-5)...")
    for step in range(1, 6):
        optimizer.zero_grad()

        # 1. Forward pass on Task A
        tower_out = tower(task_a_input)  # Triggers tower_adapter hook: writes to Crossbar slot 0
        concept_out = concept_layer(tower_out)  # Triggers concept_adapter hook: writes to Crossbar slot 1

        # 2. Step GWT engine to perform cross-attention routing & metacognitive chemistry
        broadcast = engine.step()

        # 3. Compute loss
        loss = F.mse_loss(tower_out, task_a_target)
        loss.backward()

        # 4. Astrocyte localized learning rate adaptation (Phase v0.5)
        engine.glial_manager.adjust_learning_rates(optimizer)

        optimizer.step()

        if step == 5:
            pre_shift_loss = loss.item()
            print(f"    -> Step 5 Loss (Task A): {pre_shift_loss:.6f}")

    print("\n[*] CUTOVER: Instantly switching to Task B (Step 6) to trigger NE surprise...")
    optimizer.zero_grad()

    # Cutover forward pass on Task B
    tower_out = tower(task_b_input)
    concept_out = concept_layer(tower_out)

    # Step GWT engine
    broadcast = engine.step()

    # Compute Task B loss
    loss = F.mse_loss(tower_out, task_b_target)
    loss.backward()

    # Astrocyte update
    engine.glial_manager.adjust_learning_rates(optimizer)
    optimizer.step()

    post_shift_loss = loss.item()
    surprise_ne = monitor.ne
    surprise_ach = monitor.ach
    print(f"    -> Step 6 Loss (Task B): {post_shift_loss:.6f}")
    print(f"    -> Neuromodulator State: ACh = {surprise_ach:.4f} | NE (Surprise Spike!) = {surprise_ne:.4f}")

    # Apply a manual causal intervention on concept 2 ("Exploratory") to demonstrate Phase v0.6 clamping
    print("\n[*] Applying Causal Intervention: Forcing Concept 2 ('Exploratory') to 1.0...")
    concept_layer.intervention_engine.set_intervention(2, 1.0)

    # Re-run a forward pass to apply the causal override
    tower_out = tower(task_b_input)
    concept_out = concept_layer(tower_out)
    engine.step()

    # ── 4. Offline Sleep & Memory Consolidation ────────────────────────────────
    print("\n[*] Entering Offline Sleep Phase (SWS replay & structural dendritic pruning)...")
    sleep_telemetry = engine.enter_sleep_phase(steps=5, learning_rate=0.01, pruning_threshold=0.2)

    wall_time = time.perf_counter() - start_time

    # ── 5. Renders ASCII terminal status summary ──────────────────────────────
    print("\n" + "=" * 60)
    print("               GWT BENCHMARK PERFORMANCE REPORT")
    print("=" * 60)
    print(f"  Total Benchmarking Wall Time : {wall_time:.4f} seconds")
    print(f"  Pre-Shift Loss (Task A Step 5) : {pre_shift_loss:.6f}")
    print(f"  Post-Shift Loss (Task B Step 6): {post_shift_loss:.6f}")
    print(f"  Initial Chemical Levels      : ACh = {initial_ach:.4f} | NE = {initial_ne:.4f}")
    print(f"  Surprise Chemical Levels     : ACh = {surprise_ach:.4f} | NE = {surprise_ne:.4f}")
    print(f"  Memory Traces Replayed       : {sleep_telemetry.get('memory_traces_replayed', 0)}")
    print(f"  Dendritic Branches Pruned    : {sleep_telemetry.get('dendritic_branches_pruned', 0)}")
    print("=" * 60)
    print("\n" + engine.inspect())


if __name__ == "__main__":
    main()
