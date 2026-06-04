"""
cognitive_aug/exceptions.py
===========================
Custom exception hierarchy for the cognitive_aug package.
"""


class CognitiveAugException(Exception):
    """Base exception for all errors within cognitive_aug."""

    pass


class RegistryError(CognitiveAugException):
    """Raised when a module is not found in the DataFlowManager registry or buffers."""

    pass


class DeviceMismatchError(CognitiveAugException):
    """Raised when a tensor operation fails due to device or dtype mismatch, preventing silent GPU failures."""

    pass


class DependencyMissingError(CognitiveAugException):
    """Raised when an optional dependency (like redis or safetensors) is not installed."""

    pass
