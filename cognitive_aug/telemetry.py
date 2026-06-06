"""
cognitive_aug/telemetry.py
================
Structured JSON Observability & Telemetry (Phase v0.8 Enterprise Expansion).

Replaces terminal-only ASCII dashboards with OpenTelemetry-compatible JSON logging.
Streams internal chemical monitor levels, dendritic gate spikes, astrocyte scales,
and memory consolidation performance metrics to enterprise observability stacks.
"""

import json
import logging
import sys
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("cognitive_aug.telemetry")

# Global singleton logger
_telemetry_logger: Optional["GWTTelemetryLogger"] = None


class GWTTelemetryLogger:
    """Enterprise-grade GWT metrics collector and structured JSON logger."""

    def __init__(self, stream: Any = sys.stdout) -> None:
        self.stream = stream
        # Internal configuration to toggle live logging printouts
        self.enabled = True

    def _emit(self, record_type: str, data: Dict[str, Any]) -> None:
        """Serializes and streams the log record to stdout/telemetry streams."""
        if not self.enabled:
            return

        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "epoch_ms": int(time.time() * 1000),
            "record_type": record_type,
            **data,
        }

        try:
            serialized = json.dumps(record)
            # Route to both standard logging (at INFO level) and the raw stream
            logger.info(serialized)
            self.stream.write(serialized + "\n")
            self.stream.flush()
        except Exception as e:
            # Fallback to direct sys.stderr logging on serialization failures
            sys.stderr.write(f"GWT Telemetry Failed to Emit: {e}\n")

    def record_step(
        self,
        step_idx: int,
        ne: float,
        ach: float,
        ignition_threshold: float,
        modules_telemetry: Dict[str, Dict[str, Any]],
        crossbar_weights: Optional[Dict[str, Any]] = None,
        iit_phi: Optional[float] = None,
    ) -> None:
        """Logs live details of a waking GWT competitive broadcast step."""
        data = {
            "step": step_idx,
            "neuromodulators": {
                "norepinephrine_surprise": round(ne, 6),
                "acetylcholine_focus": round(ach, 6),
                "ignition_threshold": round(ignition_threshold, 6),
            },
            "modules": modules_telemetry,
        }
        if crossbar_weights:
            data["crossbar_bindings"] = crossbar_weights
        if iit_phi is not None:
            data["system_integration_phi"] = round(iit_phi, 6)

        self._emit("WakingStep", data)

    def record_consolidation(self, sleep_telemetry: Dict[str, Any]) -> None:
        """Logs offline memory consolidation slow-wave sleep performance metrics."""
        self._emit("SleepConsolidation", sleep_telemetry)

    def record_gradient_sanitizer(
        self,
        module_name: str,
        grad_status: str,
        variance: float,
        damped: bool,
    ) -> None:
        """Logs details of local tripartite synapse gradient sanitization events."""
        self._emit(
            "GradientSanitizer",
            {
                "module_name": module_name,
                "grad_status": grad_status,
                "variance": round(variance, 6),
                "damped": damped,
            },
        )

    def record_error(
        self, error_msg: str, phase: str, details: Optional[str] = None
    ) -> None:
        """Logs critical errors encountered in the GWT layer that triggered fail-safes."""
        self._emit(
            "BypassFailure",
            {
                "error": error_msg,
                "phase": phase,
                "details": details or "",
                "action": "Graceful Fallback Bypassed GWT",
            },
        )


def get_telemetry_logger() -> GWTTelemetryLogger:
    """Retrieves or initializes the GWT telemetry logging singleton."""
    global _telemetry_logger
    if _telemetry_logger is None:
        _telemetry_logger = GWTTelemetryLogger()
    return _telemetry_logger
