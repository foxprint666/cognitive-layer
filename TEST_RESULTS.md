# Verification Test Results

This document contains the execution log for the verification suite of the Phase v0.1 GWT MVP of `cognitive_aug`.

## Test Environment
* **Platform:** Windows (`win32`)
* **Python version:** 3.14.3
* **Pytest version:** 9.0.3
* **PyTorch version:** 2.11.0
* **NumPy version:** 2.4.3

---

## Pytest Execution Log

Executed inside the workspace using `py -m pytest tests/`:

```text
============================= test session starts =============================
platform win32 -- Python 3.14.3, pytest-9.0.3, pluggy-1.6.0
rootdir: C:\Users\ASHLEY ALLEN\OneDrive\pypack
configfile: pyproject.toml
collected 7 items

tests\test_engine.py ...                                                 [ 42%]
tests\test_gwt.py ....                                                   [100%]

============================== 7 passed in 1.32s ==============================
```

## Detailed Passing Test Cases

1. **`test_registry_basic_operations`** (`tests/test_engine.py`):
   - Verifies dynamic registration and lifecycle management of cognitive modules inside `ModuleRegistry`.
2. **`test_data_flow_manager`** (`tests/test_engine.py`):
   - Confirms correctness of in-memory latent buffer transfers and robust type-checking within `DataFlowManager`.
3. **`test_engine_module_registration`** (`tests/test_engine.py`):
   - Verifies the auto-projection configuration for modules with mismatched latent spaces and validates forward-pass captures via hooks.
4. **`test_attention_selector_salience`** (`tests/test_gwt.py`):
   - Validates bottom-up salience attention calculation and row-wise softmax normalizations.
5. **`test_attention_selector_ignition`** (`tests/test_gwt.py`):
   - Validates the dynamic attentional ignition threshold gating and fallback routing when all module weights are suppressed.
6. **`test_hard_selection_straight_through`** (`tests/test_gwt.py`):
   - Confirms that hard `selection_mode` selects exact candidate tensors while successfully propagating continuous backward gradients using straight-through estimation.
7. **`test_full_engine_gwt_cycle`** (`tests/test_gwt.py`):
   - Verifies the full forward-backward cycle across the entire integrated GWT system with active learning updates.
