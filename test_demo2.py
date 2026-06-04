import torch
import torch.optim as optim
from cognitive_aug import (
    ExtendedDendriticModuleAdapter,
    NeurogenesisManager,
    NeurogenesisConsolidationEngine,
    NeurogenesisReplayBuffer,
    dynamic_register_parameters,
    NeurogenesisAstrocyteManager,
)

# 1. Initialize dynamic module adapter and homeostatic astrocyte regulator
in_dim, context_dim = 8, 4
adapter = ExtendedDendriticModuleAdapter(in_dim, context_dim, initial_branches=1)
astrocyte = NeurogenesisAstrocyteManager(calcium_decay=0.1, safety_ceiling=3.0)
replay_buffer = NeurogenesisReplayBuffer(capacity=10)

# 2. Setup dynamic live optimizer
optimizer = optim.Adam(adapter.parameters(), lr=0.01)

# 3. Setup Neurogenesis lifecycle manager
config = {"ne_threshold": 0.80, "ach_focus_ceiling": 0.70, "cooldown_steps": 1}
manager = NeurogenesisManager(
    config=config,
    astrocyte_manager=astrocyte,
    replay_buffer=replay_buffer,
    metacognitive_monitor=None,
)

# 4. Simulate a training pass under unexpected high uncertainty (NE surprise spike)
x = torch.randn(2, in_dim)
context = torch.randn(2, context_dim)
metrics = {
    "NE_surprise": torch.tensor(0.95),
    "ACh_focus": torch.tensor(0.1),
    "current_latent": context,
}

print(f"Pre-neurogenesis branches count: {len(adapter.branches)}")

# 5. Evaluate and trigger neurogenesis
status = manager.step(step_idx=1, metrics=metrics, adapters=[adapter])
print(f"Neurogenesis status: {status}")

if status == "neurogenesis_triggered":
    new_idx = len(adapter.branches) - 1
    print(f"Successfully spawned new branch at index: {new_idx}")

    # Register the newly spawned branch parameters in the active optimizer
    dynamic_register_parameters(optimizer, adapter, new_idx)
    print("Registered new parameters inside the live running Optimizer.")

# 6. Execute forward pass with active calcium monitoring
raw_outputs = adapter(x, context)
regulated_outputs = astrocyte.monitor_and_regulate(raw_outputs)
print(f"Calcium Level: {astrocyte.calcium_store.item():.4f}")

# 7. Simulate offline Slow-Wave Sleep (SWS) crystallization consolidation
target = torch.randn(2, in_dim)
replay_buffer.push(x, context, target, surprise=1.0, neuro_event=True)

sleep_engine = NeurogenesisConsolidationEngine(
    adapter, replay_buffer, threshold_perm=0.3
)
sleep_engine.execute_sleep_cycle(optimizer, steps=2)
print("Completed offline sleep-cycle consolidation evaluation.")
