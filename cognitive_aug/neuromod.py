"""
cognitive_aug/neuromod.py
=========================
Compatibility shim for neuromodulation direct imports.
"""

from gwt.neuromod import (
    MetacognitiveMonitor,
    DynamicThresholdAdapter,
    make_ascii_bar,
)

__all__ = [
    "MetacognitiveMonitor",
    "DynamicThresholdAdapter",
    "make_ascii_bar",
]
