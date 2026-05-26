"""
Smoke test: verifies cognitive-aug 0.2.2 installs and imports correctly from PyPI.
Run this inside a fresh venv AFTER: pip install cognitive-aug==0.2.2
"""
import sys

print(f"Python: {sys.version}")

# ── 1. Import check ───────────────────────────────────────────────────────────
print("\n[1/4] Importing all public symbols from cognitive_aug ...")
from cognitive_aug import (
    CognitiveAugEngine,
    ModuleAdapter,
    GlobalWorkspace,
    AttentionSelector,
    BroadcastEngine,
    BaseSelector,
    CosineSimilaritySelector,
    VectorizedCrossAttentionSelector,
    EfficientGumbelSoftmaxSelector,
    global_pool_latent,
    BaseSalience,
    MagnitudeSalience,
    EntropySalience,
    TemporalSurpriseSalience,
    DecayWorkingMemory,
    CognitiveOutputRouter,
)
print("  ✓ All symbols imported successfully")

# ── 2. Version check ──────────────────────────────────────────────────────────
print("\n[2/4] Checking installed version ...")
import importlib.metadata
version = importlib.metadata.version("cognitive-aug")
assert version == "0.2.2", f"Expected 0.2.2, got {version}"
print(f"  ✓ Version: {version}")

# ── 3. Addon module forward pass ──────────────────────────────────────────────
print("\n[3/4] Running mini forward pass through addon components ...")
import torch
import torch.nn as nn

B, T, D, KEY = 2, 5, 64, 32

# Selectors
keys   = torch.randn(B, 3, KEY)
query  = torch.randn(B, KEY)

cos_sel  = CosineSimilaritySelector(key_dim=KEY)
vec_sel  = VectorizedCrossAttentionSelector(key_dim=KEY, num_heads=4)
gum_sel  = EfficientGumbelSoftmaxSelector(key_dim=KEY)

w1 = cos_sel(keys, query);  assert w1.shape == (B, 3), f"CosineSel shape: {w1.shape}"
w2 = vec_sel(keys, query);  assert w2.shape == (B, 3), f"VecCrossAttn shape: {w2.shape}"
w3 = gum_sel(keys, query);  assert w3.shape == (B, 3), f"GumbelSel shape: {w3.shape}"
print("  ✓ Selectors OK")

# Salience (mixed dims: [B,D] and [B,T,D])
states = {
    "vision": torch.randn(B, D),       # 2-D
    "audio":  torch.randn(B, T, D),    # 3-D sequence
}
mag_sal  = MagnitudeSalience(ignition_threshold=0.0)
ent_sal  = EntropySalience(ignition_threshold=0.0)
temp_sal = TemporalSurpriseSalience(ignition_threshold=0.25)

s1 = mag_sal(states);   assert s1.shape == (B, 2)
s2 = ent_sal(states);   assert s2.shape == (B, 2)
s3 = temp_sal(states);  assert s3.shape == (B, 2)
gate = temp_sal.gate(s3); assert gate.shape == (B, 2)
print("  ✓ Salience metrics OK")

# Working memory
workspace_in = torch.randn(B, D)
ignited      = torch.ones(B, 1)
mem = DecayWorkingMemory(latent_dim=D, decay_rate=0.8, blend_weight=0.6)
out = mem(workspace_in, ignited)
assert out.shape == (B, D)
print("  ✓ DecayWorkingMemory OK")

# Output router
router = CognitiveOutputRouter(latent_dim=D, output_specs={"cls": 10, "act": 4})
routed = router(workspace_in)
assert routed["cls"].shape == (B, 10)
assert routed["act"].shape == (B, 4)
print("  ✓ CognitiveOutputRouter OK")

# ── 4. Engine + GWT integration ───────────────────────────────────────────────
print("\n[4/4] Engine + GlobalWorkspace integration test ...")

class TinyMod(nn.Module):
    def __init__(self): super().__init__(); self.fc = nn.Linear(8, D)
    def forward(self, x): return self.fc(x)

engine = CognitiveAugEngine()
mod = TinyMod()
engine.register_module("tiny", mod, latent_dim=D)

ws = GlobalWorkspace(latent_dim=D, key_dim=KEY, attention_type="key-query")
ws.selector = VectorizedCrossAttentionSelector(key_dim=KEY, num_heads=4)
engine.attach_workspace(ws)

x = torch.randn(B, 8)
_ = mod(x)
broadcast = engine.step()
assert broadcast.shape[0] == B
print(f"  ✓ Engine broadcast shape: {broadcast.shape}")

print("\n========================================")
print("  ALL TESTS PASSED — cognitive-aug 0.2.2")
print("========================================")
