# Cognitive Augmentation Layers (`cognitive_aug`)

Modular, brain-inspired cognitive augmentation layers for neural networks, built atop PyTorch.

This library provides plug-and-play components to endow existing deep learning architectures with biologically inspired capabilities, starting with **Global Workspace Theory (GWT)** mechanisms for attention, integration, and broad dynamic routing.

---

## Features (Phase v0.1 MVP)

* **Module Registry**: Track and coordinate active neural networks acting as specialized cognitive modules.
* **Data Flow Manager**: High-performance, framework-integrated routing of latent representations across dynamic computation cycles.
* **Global Workspace Theory (GWT) Layer**:
  - Configurable `single-slot` (high biological fidelity) and `multi-slot` (engineering focus) workspace bottleneck.
  - Pluggable `AttentionSelector` supporting Top-Down Key-Query matching and Bottom-Up salience selection.
  - **Dynamic Ignition**: Non-linear, threshold-gated attentional selection where only highly active features are broadcast.
  - `BroadcastEngine` that distributes unified cognitive states back to all modules as context.
* **Non-Intrusive Wrappers**: `ModuleAdapter` wraps standard PyTorch `nn.Module`s using forward/backward hooks without polluting or altering original model classes.

---

## Installation

To install in editable mode with development dependencies:

```bash
pip install -e ".[dev]"
```

---

## Quickstart Example

Here is a simple example showing how to register an image classification module and a text processing module, enabling them to communicate via a shared Global Workspace:

```python
import torch
import torch.nn as nn
from cognitive_aug import CognitiveAugEngine, GlobalWorkspace, ModuleAdapter

# 1. Define your standard PyTorch modules
class VisualModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.fc = nn.Linear(16 * 8 * 8, 256)
    def forward(self, x):
        features = torch.relu(self.conv(x))
        features = features.view(features.size(0), -1)
        return self.fc(features)

class TextModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(1000, 64)
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.fc = nn.Linear(128, 256)
    def forward(self, x):
        emb = self.emb(x)
        out, _ = self.lstm(emb)
        return self.fc(out[:, -1, :])

# 2. Instantiate modules
vis_mod = VisualModule()
txt_mod = TextModule()

# 3. Create the Cognitive Engine and Register Modules
engine = CognitiveAugEngine()

# Register modules with their latent dimensions (must match workspace target or be mapped)
vis_adapter = engine.register_module(
    name="vision",
    module=vis_mod,
    latent_dim=256
)

txt_adapter = engine.register_module(
    name="text",
    module=txt_mod,
    latent_dim=256
)

# 4. Instantiate Global Workspace
workspace = GlobalWorkspace(
    latent_dim=256,
    attention_type="key-query",
    ignition_threshold=0.5,
    workspace_slots=1  # Biological single-slot GWT bottleneck
)

# Attach workspace to the engine
engine.attach_workspace(workspace)

# 5. Run standard forward propagation
# Adapters will automatically capture latent representations via PyTorch forward hooks!
dummy_img = torch.randn(4, 3, 8, 8)
dummy_txt = torch.randint(0, 1000, (4, 10))

# Forward pass as usual
vis_out = vis_mod(dummy_img)
txt_out = txt_mod(dummy_txt)

# Perform GWT cycle: Selection & Global Broadcast!
workspace_state = engine.step()

print("Global Workspace broadcast vector shape:", workspace_state.shape)
```

---

## License

This project is licensed under the MIT License - see the LICENSE details.
