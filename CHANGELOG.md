# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-06-04

### Added
- **Global Workspace Theory (GWT)** core infrastructure in `engine.py`.
- **Neurogenesis Manager** in `neurogenesis.py` to enable autonomous computational neurogenesis, transferring salience and dendritic gating.
- **Sleep & Memory Consolidation** offline Slow-Wave Sleep (SWS) cycles with dynamic dendritic branch pruning.
- **Glial Protection Mechanisms** via `NeurogenesisAstrocyteManager` to regulate excitotoxicity via calcium decay.
- **Custom Exceptions** (`RegistryError`, `DeviceMismatchError`, `DependencyMissingError`) for secure error handling without silent failures.
- **Type Annotations & Stubs** (`py.typed` marker) for static analysis.
- **CI / CD Pipeline** via GitHub Actions to validate standard code formatting and tests.

### Changed
- Refactored `state.py` to use `safetensors` over `pickle` to mitigate arbitrary code execution vulnerabilities when interacting with Redis. Optional group `[redis]` introduced.
- Optimized latent buffers and routing in `engine.py` for distributed capabilities.
- Dynamically register/deregister model parameters into live optimizers avoiding momentum corruption during structural growth.
