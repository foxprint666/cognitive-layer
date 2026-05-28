# Cognitive Augmentation Layers (`cognitive_aug`)

Modular, brain-inspired cognitive augmentation layers for neural networks, built atop PyTorch.

This library provides plug-and-play components to endow existing deep learning architectures with biologically inspired capabilities, starting with **Global Workspace Theory (GWT)** mechanisms for attention, integration, and broad dynamic routing.

---

## Features (Phase v0.4 - Metacognitive Neuromodulation)

* **Module Registry**: Track and coordinate active neural networks acting as specialized cognitive modules.
* **Data Flow Manager**: High-performance, framework-integrated routing of latent representations across dynamic computation cycles and salience caching.
* **Global Workspace Theory (GWT) Layer**:
  - Configurable `single-slot` (high biological fidelity) and `multi-slot` (engineering focus) workspace bottleneck.
  - Pluggable `AttentionSelector` supporting Top-Down Key-Query matching and Bottom-Up salience selection.
  - **Dynamic Ignition**: Non-linear, threshold-gated attentional selection where only highly active features are broadcast.
  - `BroadcastEngine` that distributes unified cognitive states back to all modules as context.
* **Active Dendritic Gating (Phase v0.2)**:
  - **`ActiveDendriteGate`**: Vectorized dendritic pre-processing that uses GWT context to modulate feedforward features.
  - **Modulatory Gain**: Smooth sigmoidal scaling mapping context directly to features.
  - **NMDA Threshold Spiking**: Sharp thresholding mimicking biological NMDA spikes (zeroing out inactive branches) with **Straight-Through Estimators (STE)** to ensure clean backpropagation.
  - **Telemetry Dashboard**: Dynamic pathway inspection and active/muted statistics.
* **Sleep & Memory Consolidation (Phase v0.3)**:
  - **`CognitiveReplayBuffer`**: Bounded episodic memory buffer storing high-salience waking transitions via O(1) detached clones to guarantee zero graph leaks.
  - **`ConsolidationEngine`**: Offline Slow-Wave Sleep training cycle replaying high-salience experiences to reinforce GWT parameter routing under a joint reconstruction/stability loss.
  - **Synaptic Homeostasis**: Structural pruning of weak dendritic branches with **backward gradient lockout hooks** that permanently sever parameter backprop paths.
* **Metacognitive Neuromodulation (Phase v0.4)**:
  - **`MetacognitiveMonitor`**: Fully gradient-free chemical controller (all logic runs under `torch.no_grad()`) that tracks surprise and entropy to drive two virtual neurotransmitter curves — **Norepinephrine (NE)** driven by temporal cosine-distance surprise, and **Acetylcholine (ACh)** driven by workspace entropy focus and top-down goal alignment.
  - **`DynamicThresholdAdapter`**: Applies in-place modulations to the GWT ignition threshold and active dendritic gate parameters every step: `ignition_threshold = baseline − 0.4·NE + 0.3·ACh`, `nmda_threshold = baseline + 0.3·ACh`, sigmoid temperature `= 1.0 − 0.5·ACh`.
  - **`engine.attach_neuromodulator(monitor)`**: One-line hook — thresholds self-tune every GWT cycle automatically.
  - **`engine.inspect()`**: Prints a high-contrast ASCII diagnostic panel with live chemical progress bars: `[ ACh: ▇▇▇▇░░░░ 0.52 | NE: ▇▇░░░░░░ 0.21 ]`, module states, workspace slot info, and dendritic telemetry.
* **Non-Intrusive Wrappers**: `ModuleAdapter` wraps standard PyTorch `nn.Module`s using forward/backward hooks without polluting or altering original model classes.
* **Optimized Cognitive Add-ons**:
  - **Differentiable Selection**: `CosineSimilaritySelector`, `VectorizedCrossAttentionSelector` (using native FlashAttention speeds), and `EfficientGumbelSoftmaxSelector` (hard winner-take-all routing).
  - **Low-Overhead Salience**: `MagnitudeSalience` (L2 norm), `EntropySalience` (Shannon entropy confidence), and stateful `TemporalSurpriseSalience` (temporal cosine distance tracking).
  - **Decay Working Memory**: `DecayWorkingMemory` stateful wrapper with in-place exponential decay mutations and blended historical traces.
  - **Parallel Downstream Routing**: `CognitiveOutputRouter` broadcasting workspace representations back into multiple output heads in a single vectorized matrix projection pass.


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

## High-Performance Modular Add-ons

For advanced cognitive systems or performance-critical setups, the package includes highly optimized, zero-copy, fully vectorized components.

### 1. Advanced Attention Selectors (`selectors.py`)
Replace the default selector with one of three high-speed subclasses inheriting from `BaseSelector`:
- **`CosineSimilaritySelector`**: Uses vectorized matrix-based `cosine_similarity` to rapidly match incoming states to top-down goals.
- **`VectorizedCrossAttentionSelector`**: Uses PyTorch's native `scaled_dot_product_attention` with the Value tensor designed as an identity matrix, unlocking native FlashAttention speeds while remaining fully differentiable.
- **`EfficientGumbelSoftmaxSelector`**: A hard, winner-take-all routing mechanism using single-pass `gumbel_softmax` that retains full differentiability.

### 2. Low-Overhead Salience Metrics (`salience.py`)
Evaluate modular activation to compute confidence and decide workspace ignition:
- **`MagnitudeSalience`**: Computes L2 norms of states in a single vectorized pass using `torch.linalg.vector_norm`.
- **`EntropySalience`**: Computes Shannon entropy normalized to $[0, 1]$ confidence, penalizing noisy, unconfident representations.
- **`TemporalSurpriseSalience`**: Stateful cache that detaches gradients across iterations and uses cosine distance to measure step-to-step temporal shifts.
- **`global_pool_latent`**: A dimension-agnostic helper that automatically maps spatial/temporal latent tensor dimensions (e.g. `[B, T, D]` or `[B, C, H, W]`) into `[B, D]` vectors, avoiding sequential loops.

### 3. Short-Term Decay Working Memory (`memory.py`)
- **`DecayWorkingMemory`**: Exponentially decays inactive workspace slots in-place (`workspace_state.mul_(decay_rate)`) and returns a blended context vector of the active new winner and the decaying trace of the past.

### 4. Parallel Output Router (`routing.py`)
- **`CognitiveOutputRouter`**: Maps unified workspace representations back into dedicated output heads simultaneously using a **single parallel linear projection layer** and returns views using zero-copy slicing.

### 5. Active Dendritic Gating (`gwt/dendrite.py`) [Phase v0.2]
Biologically-inspired active dendritic pre-processors that dynamically modulate local feedforward pathways using GWT context:
- **`ActiveDendriteGate`**: Fully vectorized module that projects context `[B, context_dim]` onto dendritic branches.
  - `"modulatory-gain"`: Smooth sigmoidal contextual scaling.
  - `"nmda-threshold"`: Sharp thresholded biological spiking. Spikes if local depolarization $\ge \text{threshold}$, otherwise zeroed out. Uses a **Straight-Through Estimator (STE)** for clean training gradient flow.
- **`DendriticModuleAdapter`**: Special subclass of `ModuleAdapter` that automatically detects model output dimensions statically (or dynamically on the first forward pass) and seamlessly appends dendritic context-gating onto the module's execution hook pipeline.
- **`get_dendritic_status`**: Telemetry helper scanning modules recursively to report active vs. muted pathway percentages across all dendritic gates.

### 6. Sleep & Memory Consolidation (`gwt/sleep.py`) [Phase v0.3]
Offline Slow-Wave Sleep (SWS) and memory consolidation mechanisms to reinforce stable representations and enforce synaptic homeostasis:
- **`CognitiveReplayBuffer`**: Bounded episodic memory buffer that detaches and clones waking states in $O(1)$ time, evicting the lowest-salience experiences when capacity is reached and sampling batch items prioritized by salience.
- **`ConsolidationEngine`**: Offline slow-wave sleep training loop replaying prioritized experiences to GWT modules and minimizing a combined reconstruction and stability loss.
- **`prune_dendrites`**: Synaptic pruning that zero-prunes or softly decays underperforming branches under `torch.no_grad()`, registering **backward hooks on parameter gradients** to permanently sever backpropagation through pruned channels.
- **`engine.enter_sleep_phase()`**: Single command that caches waking activations, runs consolidation cycles, structurally prunes dead dendritic links, and flushes replay memory buffers.

### 7. Metacognitive Neuromodulation (`gwt/neuromod.py`) [Phase v0.4]
Chemical neuromodulation controller that removes hardcoded thresholds and dynamically self-tunes GWT ignition and dendritic gating parameters:
- **`MetacognitiveMonitor`**: Tracks real-time telemetry from `DataFlowManager` and computes exponential smoothing chemical curves under `torch.no_grad()`:
  - **Norepinephrine (NE):** `NE_t = α_NE · NE_{t-1} + (1-α_NE) · Surprise_t` — surges on surprise spikes, decays back to baseline when inputs are predictable.
  - **Acetylcholine (ACh):** `ACh_t = α_ACh · ACh_{t-1} + (1-α_ACh) · Focus_t` — rises on high target clarity and low workspace entropy, sharpening selective attention.
- **`DynamicThresholdAdapter`**: Applies in-place modulations every step:
  - GWT ignition threshold: `baseline − 0.4·NE + 0.3·ACh`
  - NMDA spike threshold: `baseline_threshold + 0.3·ACh`
  - Dendritic sigmoid temperature: `1.0 − 0.5·ACh`
- **`engine.attach_neuromodulator(monitor)`**: One-line API to enable dynamic self-tuning.
- **`engine.inspect()`**: High-contrast ASCII diagnostic panel with live chemical progress bars.

```python
from cognitive_aug.neuromod import MetacognitiveMonitor

monitor = MetacognitiveMonitor(alpha_ne=0.8, alpha_ach=0.8)
engine.attach_neuromodulator(monitor)

# After running steps:
print(engine.inspect())
# ============================================================
#          COGNITIVE ENGINE DIAGNOSTIC PANEL
# ============================================================
# Neuromodulator: Active
#   [ ACh: ▇▇▇▇░░░░ 0.52 | NE: ▇▇░░░░░░ 0.21 ]
# ------------------------------------------------------------
# Registered Modules (1):
#   - my_layer       [Dendritic Active:  72.4% | Pruned weights: 0]
# ------------------------------------------------------------
# Workspace: Attached
#   - Slots: 1
#   - Attention: key-query (threshold=0.2340)
# ============================================================
```

For a full demo showing how to attach and run these components in an active training loop, see [example_addons.py](file:///c:/Users/ASHLEY%20ALLEN/OneDrive/pypack/example_addons.py).

---

## License

This project is licensed under the MIT License - see the LICENSE details.

