"""
cognitive_aug/models/autonomic_brain.py
===============================================================================
A Standalone, Recurrent Cognitive Brain Lobe Architecture implementing 
Phase v0.1 - Phase v1.1 framework properties natively without hooks.
===============================================================================
"""

import math
from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

# Native package alignments
from cognitive_aug.engine import GlobalWorkspace
from cognitive_aug.crossbar import CognitiveCrossbar
from cognitive_aug.neuromod import MetacognitiveMonitor
from cognitive_aug.glia import GradientSanitizerHook
# Assuming global_pool_latent isn't strictly used in the code block but was imported
# from cognitive_aug.salience import global_pool_latent

class StandaloneBrainModel(nn.Module):
    def __init__(
        self, 
        latent_dim: int = 512, 
        key_dim: int = 64, 
        num_concepts: int = 5,
        max_variance_threshold: float = 3.0,
        damping_factor: float = 0.2
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.key_dim = key_dim
        
        # 1. Structural Brain Lobes (Stripped Foundation Networks)
        self.visual_lobe = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, latent_dim)
        )
        self.language_lobe = nn.Linear(latent_dim, latent_dim)
        self.motor_lobe = nn.Linear(latent_dim, 10) # 10 Action/Policy Decoders
        
        # 2. Central Routing Architecture & Cognitive Crossbar Bus (Phase v0.7)
        # Note: Added a 4th slot to dynamically store localized recurrent memory context
        self.crossbar = CognitiveCrossbar(
            slot_dim=latent_dim, 
            num_slots=4, 
            slot_names=["vision", "language", "motor_prep", "recurrent_context"]
        )
        
        # 3. Global Workspace Bottleneck Engine (Phase v0.1)
        self.workspace = GlobalWorkspace(
            latent_dim=latent_dim, 
            key_dim=key_dim, 
            attention_type="key-query", 
            ignition_threshold=0.2
        )
        
        # 4. Neuromodulation & Glial Regulation Safeties (Phase v0.4 & v0.5)
        self.monitor = MetacognitiveMonitor(alpha_ne=0.8, alpha_ach=0.8)
        self.sanitizer = GradientSanitizerHook(
            max_variance_threshold=max_variance_threshold, 
            damping_factor=damping_factor
        )
        
        # Structural Key Projections mapping routed states to key space
        self.key_projs = nn.ModuleDict({
            "vision": nn.Linear(latent_dim, key_dim),
            "language": nn.Linear(latent_dim, key_dim),
            "recurrent": nn.Linear(latent_dim, key_dim)
        })
        
        # Phase v0.8: Autograd Graph Isolated Recurrent Working Memory Anchor
        self.register_buffer("internal_context", torch.zeros(1, latent_dim))
        
        # Initialize and register Glial Backward Sanitizer Hooks to avoid Excitotoxicity
        self._register_protective_glial_hooks()

    def _register_protective_glial_hooks(self) -> None:
        """Phase v0.5: Attaches virtual astrocyte protectors to core weights."""
        for name, param in self.named_parameters():
            if param.requires_grad and ("lobe" in name or "crossbar" in name):
                param.register_hook(self.sanitizer)

    def forward(self, visual_input: torch.Tensor, recurrent_steps: int = 3) -> torch.Tensor:
        batch_size = visual_input.shape[0]
        device = visual_input.device
        
        # Sync working memory allocation sizes to match the runtime batch dimension
        if self.internal_context.shape[0] != batch_size:
            self.internal_context = torch.zeros(batch_size, self.latent_dim, device=device)
            
        # Step A: Extract raw feedforward features from current sensory arrays
        vis_features = self.visual_lobe(visual_input)
        
        # Step B: Begin the Conscious Deliberation Loop (Recurrent Processing Cycle)
        for t in range(recurrent_steps):
            # Run language/internal thought updates fueled by historical context
            lang_features = self.language_lobe(self.internal_context)
            
            # Use package-compliant crossbar writing mechanics
            # Map raw vectors into dedicated slots concurrently
            self.crossbar.write_slot(0, vis_features)
            self.crossbar.write_slot(1, lang_features)
            self.crossbar.write_slot(2, torch.zeros_like(vis_features)) # Motor target preparation lane
            self.crossbar.write_slot(3, self.internal_context) # Loop back historic memory vector
            
            # Execute all-to-all crossbar message passing
            routed_slots = self.crossbar() # Layout outcome matrix: [B, 4, latent_dim]
            
            # Isolate slot groupings for Workspace Competition
            latent_states = {
                "vision": routed_slots[:, 0],
                "language": routed_slots[:, 1],
                "recurrent": routed_slots[:, 3]
            }
            
            # Transform states to alignment key frames via mapped projections
            keys = {name: self.key_projs[name](state) for name, state in latent_states.items()}
            
            # Central GWT Gated Access Selection & Competitive Broadcast
            # The workspace evaluates proposals against the dynamic goals of the network
            next_broadcast = self.workspace(latent_states, keys)
            
            # Phase v0.8: Graph Isolation Rule Enforcement
            # We preserve gradient updates inside the step, but clone detached features 
            # to insulate state cross-over boundaries from accumulating massive graph histories.
            if self.training:
                self.internal_context = next_broadcast
            else:
                self.internal_context = next_broadcast.detach().clone()
                
            # Step C: Update Metacognitive Neuromodulation (Phase v0.4 Chemical Metrics)
            if hasattr(self.workspace, "last_weights"):
                # Track attention entropy to calculate dynamic concentration levels
                self.monitor.modulate_from_weights(self.workspace.last_weights)

        # Step D: Emit motor instructions from finalized consolidated working state
        action_logits = self.motor_lobe(self.internal_context)
        return action_logits
