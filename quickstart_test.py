import torch
import torch.nn as nn
from cognitive_aug import (
    CognitiveAugEngine,
    GlobalWorkspace,
    DendriticModuleAdapter,
    MetacognitiveMonitor,
)

# 1. Define a standard PyTorch module
class VisionEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.fc = nn.Linear(8 * 4 * 4, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.conv(x))
        out = out.view(out.size(0), -1)
        return self.fc(out)

# 2. Instantiate modules and engine
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionEncoder().to(device)

engine = CognitiveAugEngine()

# 3. Register standard module with a Dendritic Adapter
adapter = DendriticModuleAdapter(
    name="vision_encoder",
    module=model,
    latent_dim=16,
    data_flow=engine.data_flow,
    num_branches=4,
    spike_type="nmda-threshold",
    threshold=0.4
)
engine.registry.register("vision_encoder", adapter)

# 4. Attach Workspace and Neuromodulator
workspace = GlobalWorkspace(latent_dim=16, key_dim=32, ignition_threshold=0.3).to(device)
engine.attach_workspace(workspace)

monitor = MetacognitiveMonitor(alpha_ne=0.2, alpha_ach=0.2)
engine.attach_neuromodulator(monitor)

# 5. Run standard forward pass & step GWT cycle
inputs = torch.randn(4, 3, 4, 4, device=device)
outputs = model(inputs)  # Adapter automatically intercepts latents via hooks

broadcast_state = engine.step()  # Computes GWT routing and ACh/NE curves

print(f"[*] Broadcast Context Shape: {broadcast_state.shape}")
print(engine.inspect())  # Displays gorgeous live terminal telemetry dashboard!
