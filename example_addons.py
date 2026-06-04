"""
Example script demonstrating how to cleanly attach and use the new optimized
cognitive layer components (selectors, salience, memory decay, and routers)
as plug-and-play add-ons in a PyTorch execution loop.
"""

import torch
import torch.nn as nn
from cognitive_aug import (
    CognitiveAugEngine,
    GlobalWorkspace,
    VectorizedCrossAttentionSelector,
    TemporalSurpriseSalience,
    DecayWorkingMemory,
    CognitiveOutputRouter,
)


# 1. Define standard neural network modules (subsystems)
class VisionSubsystem(nn.Module):
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        # Simulated CNN outputting [B, C, H, W]
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.fc = nn.Linear(16 * 8 * 8, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [B, 3, 8, 8]
        feat = torch.relu(self.conv(x))
        feat = feat.view(feat.size(0), -1)
        # Returns [B, feature_dim]
        return self.fc(feat)


class AuditorySubsystem(nn.Module):
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        # Simulated Transformer outputting sequence representations [B, T, D]
        self.emb = nn.Linear(40, feature_dim)
        self.lstm = nn.LSTM(feature_dim, feature_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [B, 5, 40]
        emb = self.emb(x)
        out, _ = self.lstm(emb)
        # Returns a sequence of shape [B, T=5, D=128]
        return out


def main():
    print("=== Initializing Cognitive Augmentation Engine and Modules ===")

    # 2. Setup the Cognitive Engine
    engine = CognitiveAugEngine()

    # Initialize our subsystems
    vision_sub = VisionSubsystem(feature_dim=128)
    auditory_sub = AuditorySubsystem(feature_dim=128)

    # Register subsystems with the engine
    # ModuleAdapter automatically captures the outputs via forward hooks
    engine.register_module(name="vision_mod", module=vision_sub, latent_dim=128)

    engine.register_module(name="audio_mod", module=auditory_sub, latent_dim=128)

    # 3. Create our high-performance GWT workspace
    # We configure it with our new high-speed VectorizedCrossAttentionSelector
    workspace = GlobalWorkspace(latent_dim=128, key_dim=64, attention_type="key-query")

    # Attach our high-performance selector to the workspace
    # Since selectors are standard PyTorch Modules, we can hot-swap them!
    workspace.selector = VectorizedCrossAttentionSelector(key_dim=64, num_heads=4)
    engine.attach_workspace(workspace)

    # 4. Instantiate our stateful Low-Overhead Salience and working memory decay layers
    # We will use TemporalSurpriseSalience to measure step-to-step changes
    salience_metric = TemporalSurpriseSalience(ignition_threshold=0.25)

    # Memory Decay maintains slot states statefully
    working_memory = DecayWorkingMemory(
        latent_dim=128, decay_rate=0.7, blend_weight=0.6
    )

    # 5. Define dedicated output heads using our single-pass parallel CognitiveOutputRouter
    output_specs = {
        "class_label": 10,  # Classification output (10 classes)
        "action_vector": 4,  # Downstream actions (4 actions)
        "attention_feedback": 3,  # Attention scores feedback
    }
    output_router = CognitiveOutputRouter(latent_dim=128, output_specs=output_specs)

    print("\n=== Executing Multiple Steps in the Cognitive Loop ===")

    # Simulate dynamic inputs across 3 steps
    batch_size = 4

    for step in range(1, 4):
        print(f"\n--- Step {step} ---")

        # Simulated sensory inputs
        img_input = torch.randn(batch_size, 3, 8, 8)
        # At step 2, let's keep audio inputs steady, but at step 3, introduce a major shift (high surprise)
        if step == 3:
            audio_input = torch.randn(batch_size, 5, 40) * 10.0  # Surprise!
        else:
            audio_input = torch.ones(batch_size, 5, 40)

        # Standard forward pass on modules
        # Forward hooks automatically intercept outputs and buffer them in engine.data_flow
        _ = vision_sub(img_input)
        _ = auditory_sub(audio_input)

        # 6. Step 1 of Cognitive cycle: Run attention selection and GWT broadcast
        raw_broadcast = engine.step()
        print(f"  [GWT Broadcast] Captured raw workspace state: {raw_broadcast.shape}")

        # 7. Step 2: Compute low-overhead salience (surprise) over current subsystem states
        # Data flow buffers contain raw outputs (e.g. vision: [B, 128], audio: [B, T=5, D=128])
        # Salience handles different dimensions via global pooling internally
        latent_buffers = engine.data_flow.list_buffers()
        surprise_scores = salience_metric(latent_buffers)
        print(f"  [Salience] Cosine surprise scores per module:\n    {surprise_scores}")

        # 8. Step 3: Determine which modules ignited and update the decay working memory
        # We can sum/max surprise scores to determine overall workspace ignition status
        # If any module surprise score exceeds the ignition threshold, the workspace ignites!
        ignition_mask = salience_metric.gate(surprise_scores)
        overall_ignition = (ignition_mask.sum(dim=-1, keepdim=True) > 0.0).float()
        print(
            f"  [Ignition Gating] Gated workspace ignition status:\n    {overall_ignition.squeeze(-1)}"
        )

        # Apply exponential working memory decay and blend with winner representation
        blended_workspace = working_memory(raw_broadcast, overall_ignition)
        print(
            f"  [Working Memory] Blended workspace memory trace shape: {blended_workspace.shape}"
        )

        # 9. Step 4: Route final unified workspace state into dedicated downstream heads in parallel
        downstream_outputs = output_router(blended_workspace)
        print("  [Downstream Routing] Outputs projected in a single vectorized pass:")
        for head, output_tensor in downstream_outputs.items():
            print(f"    - Head '{head}': shape {list(output_tensor.shape)}")

        # Clean up data flow buffers for next step
        engine.data_flow.clear_buffers()

    print("\n=== Cognitive Execution Completed Successfully! ===")


if __name__ == "__main__":
    main()
