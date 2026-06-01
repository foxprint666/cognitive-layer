# Cognitive Augmentation Layer (GWT)
### Google Colab Proof-of-Concept: Microsoft Phi-2

This guide provides a complete, copy-pasteable Google Colab walkthrough to demonstrate the *before-and-after* benefits of wrapping an open-weights LLM (`microsoft/phi-2`) with the Cognitive Augmentation Layer (`cognitive-aug`). 

By applying active dendritic gating and the Global Workspace Theory (GWT) architecture, we can protect the model from catastrophic forgetting during sequential task learning and actively intervene in its "thoughts" dynamically.

## 1. Environment Setup

Run this in your first Colab cell to install the required packages:

```python
!pip install -q torch transformers accelerate
!pip install -q cognitive-aug>=0.2.9
```

## 2. Model Initialization (Base Phi-2)

Initialize the base Phi-2 model. To fit comfortably in Colab's free T4 GPU tier, we load it in `float16`.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_id = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Load base model in half-precision
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    trust_remote_code=True
).to(device)

# Ensure the model is in eval mode for inference testing
base_model.eval()
print("Base model loaded successfully.")
```

## 3. Cognitive Engine Wrapping (The "After")

Now, we wrap the base model with the `CognitiveAugEngine`. This attaches active dendritic gates to the Transformer blocks and hooks them into a centralized Global Workspace.

```python
from cognitive_aug import (
    CognitiveAugEngine,
    GlobalWorkspace,
    VectorizedCrossAttentionSelector,
    TemporalSurpriseSalience,
    DecayWorkingMemory
)
import cognitive_aug.profiler as profiler

print("Initializing Cognitive Augmentation Engine...")

# Initialize the cognitive engine components
engine = CognitiveAugEngine()
engine.workspace = GlobalWorkspace(latent_dim=128, key_dim=64)
engine.workspace.selector = VectorizedCrossAttentionSelector(workspace_dim=64, key_dim=64)
engine.salience_metric = TemporalSurpriseSalience(decay_rate=0.9)
engine.memory = DecayWorkingMemory(max_capacity=50, decay_rate=0.1)
engine.model = base_model

# Dummy input to trace the computational graph
dummy_input = {
    "input_ids": torch.randint(0, 1000, (1, 8)).to(device),
    "attention_mask": torch.ones(1, 8).to(device)
}

# Automatically discover and hook into Phi-2's transformer blocks
profiler.register_selective_hooks(
    engine=engine,
    model=base_model,
    latent_dim=128,
    dummy_input=dummy_input,
    selective_ratio=0.1,
    use_dendritic=True,
    num_branches=4
)

print("Cognitive Engine successfully attached to Phi-2!")
```

## 4. Before-and-After: Generation & Inference

Let's test generation. The cognitive engine is completely transparent to the Hugging Face `generate` API, meaning you can use it exactly as you would use the base model.

```python
prompt = "The key to building artificial general intelligence is"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

print("--- Generating with Base Phi-2 (No Context) ---")
# If you run this on a fresh base model, it will output standard text
with torch.no_grad():
    outputs = base_model.generate(**inputs, max_length=50)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

print("\n--- Generating with Augmented Phi-2 (Cognitive Loop) ---")
# The engine intercepts the forward passes, computes global salience, 
# and selectively gates dendritic branches across all transformer layers.
with torch.no_grad():
    # Pass the inputs through the engine's cognitive step
    # Note: For text generation, the underlying model.generate still works!
    augmented_outputs = engine.model.generate(**inputs, max_length=50)
    print(tokenizer.decode(augmented_outputs[0], skip_special_tokens=True))
```

## 5. Structural Pruning (Simulated Sleep Phase)

One of the primary benefits of `cognitive-aug` is the ability to prune under-utilized neural pathways based on real activation data (salience) collected during the "waking" generation phase.

```python
from cognitive_aug import prune_dendrites

print("Entering Slow-Wave Sleep (SWS) Phase...")

# Prune dendritic branches that had an average activation below 0.5
# This natively utilizes torch.nn.utils.prune to permanently sever the weights
pruned_branches = prune_dendrites(engine, pruning_threshold=0.5)

print(f"Sleep Phase Complete. Successfully pruned {pruned_branches} inactive dendritic branches.")
print("The model is now leaner and highly specialized for the contexts it just experienced.")
```

## 6. How to Collaborate on the Repository
If you'd like to contribute to the `cognitive-aug` project or adapt it for your own architectures (e.g., Llama 3, Mixtral MoE):
1. **Clone the Repository**: `git clone https://github.com/foxprint666/cognitive-layer.git`
2. **Install locally for development**: `pip install -e .`
3. **Submit a Pull Request**: Create a branch with your structural hooks, MoE mitigations, or custom `SalienceMetrics` and submit a PR!
