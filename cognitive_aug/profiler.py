"""
cognitive_aug/profiler.py
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
# 1. Automated Backbone Discovery Engine & Layer Scanner
# ─────────────────────────────────────────────────────────────────────────────


def discover_transformer_layers(model: nn.Module) -> Any:
    """
    Dynamically discover structural sequence paths within open-weights transformer blocks.
    """
    common_paths = [
        "model.layers",
        "transformer.h",
        "transformer.layers",
        "model.decoder.layers",
        "transformer.encoder.layers",  # ChatGLM
        "blocks",  # General fallback
    ]
    for path in common_paths:
        try:
            parts = path.split(".")
            target = model
            for part in parts:
                target = getattr(target, part)
            if isinstance(target, (torch.nn.ModuleList, torch.nn.Sequential, list)):
                return target
        except AttributeError:
            continue

    # Deep iterative discovery fallback if paths are custom-mapped
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > 0:
            # Ensure we don't accidentally capture high-level outer wrappers
            if any(isinstance(child, torch.nn.Linear) for child in module[0].modules()):
                return module

    raise AttributeError(
        "cognitive-aug could not automatically resolve the model structure block sequence layers."
    )


def cast_input_to_device_and_dtype(
    x: Any, device: torch.device, dtype: torch.dtype
) -> Any:
    """
    Recursively casts input tensors to the target device and formats.
    """
    if isinstance(x, torch.Tensor):
        if torch.is_floating_point(x):
            return x.to(device=device, dtype=dtype)
        else:
            return x.to(device=device)
    elif isinstance(x, dict):
        return {
            k: cast_input_to_device_and_dtype(v, device, dtype) for k, v in x.items()
        }
    elif isinstance(x, list):
        return [cast_input_to_device_and_dtype(v, device, dtype) for v in x]
    elif isinstance(x, tuple):
        return tuple(cast_input_to_device_and_dtype(v, device, dtype) for v in x)
    return x


# ─────────────────────────────────────────────────────────────────────────────
# 2. Layer Profiling Infrastructure
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

    # Infer host model precision and device
    try:
        model_dtype = next(model.parameters()).dtype
        model_device = next(model.parameters()).device
    except StopIteration:
        model_dtype = torch.float32
        model_device = torch.device("cpu")

    # Ensure dummy_input matches device and format
    dummy_input = cast_input_to_device_and_dtype(dummy_input, model_device, model_dtype)

    # Attempt architecture-agnostic layer discovery to narrow search space
    try:
        target_blocks = discover_transformer_layers(model)
        block_submodules = set(target_blocks.modules())
        logger.info(
            "GWT Micro-Profiler: Dynamic layer scanner discovered sequence backbone structure inside model."
        )
    except AttributeError:
        block_submodules = None
        logger.info(
            "GWT Micro-Profiler: Dynamic layer scanner fell back to standard named modules profiling."
        )

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

            stats.append(
                {
                    "name": name,
                    "module": module,
                    "duration": duration,
                    "variance": variance,
                    "magnitude": mean_mag,
                    "salience_score": variance * 0.7 + mean_mag * 0.3,
                }
            )
            return outputs

        return hook_fn

    for name, module in model.named_modules():
        # MoE Exclusion: Do not hook internal expert networks to prevent registry explosion and OOM
        if any(
            keyword in name.lower()
            for keyword in ["expert", "mlp.gate", "mlp_gate", "mlp"]
        ):
            continue

        if block_submodules is not None and module not in block_submodules:
            continue
        # Hook nn.Linear, nn.Conv2d, and custom Attention layers only
        if len(list(module.children())) == 0 and isinstance(
            module, (nn.Linear, nn.Conv2d)
        ):
            handle = module.register_forward_hook(make_profile_hook(name, module))
            handles.append(handle)

    # Execute dummy forward pass (under no_grad and autocast context to prevent OOM/dtype crashes)
    try:
        with torch.no_grad():
            is_half = model_dtype in (torch.float16, torch.bfloat16)
            is_cuda = model_device.type == "cuda"
            if is_half or is_cuda:
                with torch.autocast(device_type=model_device.type, dtype=model_dtype):
                    if isinstance(dummy_input, tuple):
                        model(*dummy_input)
                    elif isinstance(dummy_input, dict):
                        model(**dummy_input)
                    else:
                        model(dummy_input)
            else:
                if isinstance(dummy_input, tuple):
                    model(*dummy_input)
                elif isinstance(dummy_input, dict):
                    model(**dummy_input)
                else:
                    model(dummy_input)
    except Exception as e:
        logger.error(
            f"GWT Micro-Profiler: Error during dummy profiling forward pass: {e}"
        )
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
            "salience_score": item["salience_score"],
        }
        result.append((item["name"], item["module"], metrics))
        logger.info(
            f"  -> Module: {item['name']:<30} | Variance: {item['variance']:.6f} | "
            f"Mag: {item['magnitude']:.6f} | Salience: {item['salience_score']:.6f}"
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. Selective Hook Registration
# ─────────────────────────────────────────────────────────────────────────────


def register_selective_hooks(
    engine: Any,
    model: nn.Module,
    latent_dim: int,
    dummy_input: Any,
    selective_ratio: float = 0.3,
    use_dendritic: bool = True,
    use_extended_dendritic: bool = False,
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
        use_extended_dendritic: Whether to use ExtendedDendriticModuleAdapter (Phase v0.8) for neurogenesis.

    Returns:
        List of registered adapter instances.
    """
    # 1. Profile submodules
    profiled_layers = profile_submodules(model, dummy_input)
    if not profiled_layers:
        logger.warning(
            "GWT Micro-Profiler: No hookable leaf modules identified. Hooking cancelled."
        )
        return []

    # 2. Determine target count based on ratio
    num_to_hook = max(1, int(len(profiled_layers) * selective_ratio))
    target_layers = profiled_layers[:num_to_hook]

    logger.info(
        f"GWT Micro-Profiler: Hooking top {num_to_hook}/{len(profiled_layers)} "
        f"layers ({selective_ratio * 100:.1f}% selective ratio) to optimize TTFT."
    )

    try:
        model_dtype = next(model.parameters()).dtype
        model_device = next(model.parameters()).device
    except StopIteration:
        model_dtype = torch.float32
        model_device = torch.device("cpu")

    registered_adapters = []

    # 3. Register GWT adapters on selected top-salience submodules
    for name, module, metrics in target_layers:
        clean_name = name.replace(".", "_")

        # Dynamically select adapter type and construct with matching device/dtype
        if use_extended_dendritic:
            from cognitive_aug.neurogenesis import ExtendedDendriticModuleAdapter

            adapter = ExtendedDendriticModuleAdapter(
                name_or_feedforward_dim=clean_name,
                module_or_context_dim=module,
                latent_dim=latent_dim,
                data_flow=engine.data_flow,
                device=model_device,
                dtype=model_dtype,
                **kwargs,
            )
        elif use_dendritic:
            from cognitive_aug import DendriticModuleAdapter

            adapter = DendriticModuleAdapter(
                name=clean_name,
                module=module,
                latent_dim=latent_dim,
                data_flow=engine.data_flow,
                device=model_device,
                dtype=model_dtype,
                **kwargs,
            )
        else:
            from cognitive_aug import ModuleAdapter

            adapter = ModuleAdapter(
                name=clean_name,
                module=module,
                latent_dim=latent_dim,
                data_flow=engine.data_flow,
                device=model_device,
                dtype=model_dtype,
                **kwargs,
            )

        adapter.to(device=model_device, dtype=model_dtype)
        engine.registry.register(clean_name, adapter)
        registered_adapters.append(adapter)
        logger.info(
            f"  [+] GWT hook registered selectively on high-salience layer: '{name}'"
        )

    return registered_adapters
