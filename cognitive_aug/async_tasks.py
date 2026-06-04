"""
cognitive_aug/async_tasks.py
==================
Asynchronous Execution Task Queues & Workers (Phase v0.8 Enterprise Expansion).

Moves high-latency memory consolidation (sleep cycles) and Astrocyte tracking
completely off the live execution thread into background worker queues.
Compatible with Celery/Redis Enterprise for horizontally scaled pods, and provides
a self-contained, thread-safe asynchronous worker fallback.
"""

import logging
import queue
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Global thread-safe queue for local asynchronous processing
_local_task_queue: queue.Queue = queue.Queue()
_local_worker_thread: Optional[threading.Thread] = None
_local_worker_running: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# 1. Self-Contained Multi-Threaded Asynchronous Worker (Process/Thread Fallback)
# ─────────────────────────────────────────────────────────────────────────────


def _async_worker_loop() -> None:
    """Consumes and processes high-latency cognitive tasks in the background."""
    global _local_worker_running
    logger.info("GWT local background cognitive worker thread started.")

    while _local_worker_running:
        try:
            # Block for a short duration to support hot shutdown
            task_fn, args, kwargs = _local_task_queue.get(timeout=1.0)

            try:
                logger.debug(
                    f"GWT Async Worker: Executing background task: {task_fn.__name__}"
                )
                task_fn(*args, **kwargs)
                logger.debug(
                    f"GWT Async Worker: Successfully finished: {task_fn.__name__}"
                )
            except Exception as ex:
                logger.error(
                    f"GWT Async Worker: Error executing task '{task_fn.__name__}': {ex}",
                    exc_info=True,
                )
            finally:
                _local_task_queue.task_done()

        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"GWT Async Worker: Exception in thread loop: {e}")
            time.sleep(1.0)

    logger.info("GWT local background cognitive worker thread stopped.")


def start_async_worker() -> None:
    """Starts the background cognitive worker thread if not already running."""
    global _local_worker_thread, _local_worker_running
    if not _local_worker_running:
        _local_worker_running = True
        _local_worker_thread = threading.Thread(target=_async_worker_loop, daemon=True)
        _local_worker_thread.start()


def stop_async_worker() -> None:
    """Gracefully shuts down the background cognitive worker thread."""
    global _local_worker_running, _local_worker_thread
    if _local_worker_running:
        _local_worker_running = False
        if _local_worker_thread is not None:
            _local_worker_thread.join(timeout=3.0)
            _local_worker_thread = None


def enqueue_background_task(task_fn: Any, *args: Any, **kwargs: Any) -> None:
    """Enqueues a high-latency function to be run asynchronously in the background."""
    start_async_worker()
    _local_task_queue.put((task_fn, args, kwargs))
    logger.debug(
        f"Successfully enqueued task '{task_fn.__name__}' to local GWT async queue."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Celery & Redis Enterprise Task Wrapper Signatures
# ─────────────────────────────────────────────────────────────────────────────

# Celery application placeholder — resolves dynamically if Celery is installed
celery_app: Optional[Any] = None
try:
    from celery import shared_task

    @shared_task(name="cognitive_aug.tasks.async_sleep_consolidation")
    def celery_sleep_consolidation(
        engine_state_prefix: str,
        steps: int = 100,
        learning_rate: float = 0.001,
        pruning_threshold: float = 0.05,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """
        Celery task representing GWT memory consolidation SWS cycles.
        Pulls traces from Redis Enterprise, rebuilds the state, runs replay optimization,
        and returns details to training pods.
        """
        logger.info(f"Celery Sleep task triggered for prefix: {engine_state_prefix}")
        # Dynamic import to avoid circular dependency
        from cognitive_aug.state import RedisReplayBufferStore, RedisStateStore
        from cognitive_aug.engine import CognitiveAugEngine
        from cognitive_aug.sleep import CognitiveReplayBuffer

        # 1. Reconstruct mock engine context in the worker thread
        engine = CognitiveAugEngine()

        # Override state store to pull from shared Redis Enterprise key
        state_store = RedisStateStore(key_prefix=engine_state_prefix)
        engine.data_flow._state_store = state_store

        # Hook up shared distributed replay buffer
        replay_store = RedisReplayBufferStore(
            key=f"{engine_state_prefix}:replay_buffer"
        )
        replay_buffer = CognitiveReplayBuffer()
        replay_buffer.store = replay_store
        engine.attach_replay_buffer(replay_buffer)

        # 2. Execute consolidation
        telemetry = engine.enter_sleep_phase(
            steps=steps,
            learning_rate=learning_rate,
            pruning_threshold=pruning_threshold,
            batch_size=batch_size,
        )

        logger.info(f"Celery Sleep task successfully completed. Telemetry: {telemetry}")
        return telemetry

except ImportError:
    # Celery is not installed or configured on the host machine
    pass


# ─────────────────────────────────────────────────────────────────────────────
# 3. Waking Execution Loop Offloading Helpers
# ─────────────────────────────────────────────────────────────────────────────


def offloaded_enter_sleep_phase(
    engine: Any,
    steps: int = 5,
    learning_rate: float = 0.001,
    pruning_threshold: float = 0.05,
    batch_size: int = 32,
    use_celery: bool = False,
) -> Any:
    """
    Asynchronously triggers GWT sleep consolidation off the main inference thread.

    If use_celery is enabled and Celery is configured, offloads as a distributed task.
    Otherwise, handles it via local multi-threaded queue.
    """
    if use_celery and "celery_sleep_consolidation" in globals():
        # Offload via Celery task queue
        prefix = getattr(
            engine.data_flow._state_store, "key_prefix", "cognitive_aug:state"
        )
        task = globals()["celery_sleep_consolidation"].delay(
            engine_state_prefix=prefix,
            steps=steps,
            learning_rate=learning_rate,
            pruning_threshold=pruning_threshold,
            batch_size=batch_size,
        )
        logger.info(
            f"GWT: Offloaded sleep phase consolidation to Celery (Task ID: {task.id})"
        )
        return {
            "status": "Offloaded",
            "task_id": task.id,
            "backend": "Celery",
        }
    else:
        # Define local target executing consolidation off the GPU thread
        def background_consolidation() -> None:
            # Enforce dynamic parameters
            telemetry = engine.enter_sleep_phase(
                steps=steps,
                learning_rate=learning_rate,
                pruning_threshold=pruning_threshold,
                batch_size=batch_size,
            )
            # Log structured metrics in worker thread
            from cognitive_aug.telemetry import get_telemetry_logger

            json_logger = get_telemetry_logger()
            json_logger.record_consolidation(telemetry)

        # Offload to the background queue
        enqueue_background_task(background_consolidation)
        return {
            "status": "Queued",
            "backend": "Local Thread",
        }
