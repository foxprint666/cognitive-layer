"""
gwt/profiler.py
===============
Micro-Benchmarking & Selective Hook Registration Layer (Phase v0.8 Enterprise).

Profiles model submodules to selectively register GWT hook adapters on the most
active, high-salience, or high-variance layers. Bypasses stable subnets to
minimize Time-To-First-Token (TTFT) latency overhead in production systems.
"""

import logging
import time
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Layer Profiling Infrastructure
# ─────────────────────────────────────────────────────────────────────────────

def profile_submodules(
    model: nn.Module,
    dummy_input: Any,
) -> List[Tuple[str, nn.Module, Dict[str, float]]]:
    """
    Runs a lightweight dry-run forward pass, measuring latency, activation magnitude,
    and variance across all submodules.
    
    Args:
        model       : The target PyTorch model to profile.
        dummy_input : Representative batch inputs for the forward pass.
        
    Returns:
        Sorted list of tuples: `(submodule_name, submodule_ref, statistics_dict)`.
    """
    logger.info("GWT Micro-Profiler: Initiating layer profiling dry-run...")
    stats: List[Dict[str, Any]] = []
    handles = []

    # Temporary forward hook to profile activations
    def make_profile_hook(name: str, module: nn.Module):
        def hook_fn(mod: nn.Module, inputs: Any, outputs: Any) -> Any:
            t_start = time.perf_counter()
            latent = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            duration = time.perf_counter() - t_start
            
            if isinstance(latent, torch.Tensor):
                flat = latent.detach().clone().float()
                variance = float(flat.var().item()) if flat.numel() > 1 else 0.0
                mean_mag = float(flat.abs().mean().item())
            else:
                variance = 0.0
                mean_mag = 0.0

            stats.append({
                "name": name,
                "module": module,
                "duration": duration,
                "variance": variance,
                "magnitude": mean_mag,
                "salience_score": variance * 0.7 + mean_mag * 0.3
            })
            return outputs
        return hook_fn

    # Register profiling hooks recursively on all leaf nodes
    for name, module in model.named_modules():
        # Hook nn.Linear, nn.Conv2d, and custom Attention layers only
        if len(list(module.children())) == 0 and isinstance(module, (nn.Linear, nn.Conv2d)):
            handle = module.register_forward_hook(make_profile_hook(name, module))
            handles.append(handle)

    # Execute dummy forward pass (under no_grad to keep it lightweight)
    with torch.no_grad():
        try:
            if isinstance(dummy_input, tuple):
                model(*dummy_input)
            elif isinstance(dummy_input, dict):
                model(**dummy_input)
            else:
                model(dummy_input)
        except Exception as e:
            logger.error(f"GWT Micro-Profiler: Error during dummy profiling forward pass: {e}")
        finally:
            # Remove all profiling hooks immediately
            for h in handles:
                h.remove()

    # Sort layers by computed salience_score (highest variance/magnitude first)
    sorted_stats = sorted(stats, key=lambda s: s["salience_score"], reverse=True)
    
    result = []
    for item in sorted_stats:
        metrics = {
            "duration": item["duration"],
            "variance": item["variance"],
            "magnitude": item["magnitude"],
            "salience_score": item["salience_score"]
        }
        result.append((item["name"], item["module"], metrics))
        logger.info(
            f"  -> Module: {item['name']:<30} | Variance: {item['variance']:.6f} | "
            f"Mag: {item['magnitude']:.6f} | Salience: {item['salience_score']:.6f}"
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. Selective Hook Registration
# ─────────────────────────────────────────────────────────────────────────────

def register_selective_hooks(
    engine: Any,
    model: nn.Module,
    latent_dim: int,
    dummy_input: Any,
    selective_ratio: float = 0.3,
    use_dendritic: bool = True,
    **kwargs: Any,
) -> List[Any]:
    """
    Profiles the model and selectively registers GWT adapters only on the top
    highly active, high-salience attention or feedforward subnets.
    
    Bypasses the remaining stable, low-impact layers to minimize TTFT.
    
    Args:
        engine          : Active CognitiveAugEngine instance.
        model           : Target model tower to profile and hook.
        latent_dim      : The GWT workspace latent representation dimension.
        dummy_input     : A representative batch of inputs.
        selective_ratio : Fraction of profiled layers to hook (e.g. 0.3 = top 30% of layers).
        use_dendritic   : Whether to use DendriticModuleAdapter (Phase v0.2) or standard ModuleAdapter.
        
    Returns:
        List of registered adapter instances.
    """
    # 1. Profile submodules
    profiled_layers = profile_submodules(model, dummy_input)
    if not profiled_layers:
        logger.warning("GWT Micro-Profiler: No hookable leaf modules identified. Hooking cancelled.")
        return []

    # 2. Determine target count based on ratio
    num_to_hook = max(1, int(len(profiled_layers) * selective_ratio))
    target_layers = profiled_layers[:num_to_hook]
    
    logger.info(
        f"GWT Micro-Profiler: Hooking top {num_to_hook}/{len(profiled_layers)} "
        f"layers ({selective_ratio*100:.1f}% selective ratio) to optimize TTFT."
    )

    registered_adapters = []
    
    # 3. Register GWT adapters on selected top-salience submodules
    for name, module, metrics in target_layers:
        clean_name = name.replace(".", "_")
        
        # Dynamically select adapter type
        if use_dendritic:
            from gwt import DendriticModuleAdapter
            adapter = DendriticModuleAdapter(
                name=clean_name,
                module=module,
                latent_dim=latent_dim,
                data_flow=engine.data_flow,
                **kwargs
            )
        else:
            from gwt import ModuleAdapter
            adapter = ModuleAdapter(
                name=clean_name,
                module=module,
                latent_dim=latent_dim,
                data_flow=engine.data_flow,
                **kwargs
            )
            
        engine.registry.register(clean_name, adapter)
        registered_adapters.append(adapter)
        logger.info(f"  [+] GWT hook registered selectively on high-salience layer: '{name}'")

    return registered_adapters
