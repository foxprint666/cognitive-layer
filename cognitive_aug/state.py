"""
cognitive_aug/state.py
============
Distributed State Management Layer (Phase v0.8 Enterprise Expansion).

Abstracts data flow buffers and replay caching away from local Python processes,
allowing horizontal scaling across multiple pods using Redis/distributed caches.
Serializes/deserializes PyTorch tensors as binary blobs with graph isolation.
"""

import io
import json
import logging
import pickle
from typing import Any, Dict, List, Optional, Protocol, Union

import torch

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. State Store Protocols (Abstract Interfaces)
# ─────────────────────────────────────────────────────────────────────────────

class BaseStateStore:
    """Abstract interface for managing GWT transient latent buffers and saliences."""

    def update_buffer(self, name: str, tensor: torch.Tensor) -> None:
        """Update a registered module's latent buffer state."""
        ...

    def get_buffer(self, name: str) -> torch.Tensor:
        """Retrieve a registered module's latent buffer state."""
        ...

    def list_buffers(self) -> Dict[str, torch.Tensor]:
        """List all current module latent buffers."""
        ...

    def update_salience(self, name: str, score: float) -> None:
        """Update the salience score for a specific module."""
        ...

    def get_salience(self, name: str) -> float:
        """Retrieve the salience score for a specific module."""
        ...

    def list_saliences(self) -> Dict[str, float]:
        """List all current module saliences."""
        ...

    def clear(self) -> None:
        """Clear all buffered states and saliences."""
        ...


class BaseReplayBufferStore:
    """Abstract interface for managing episodic memory buffers across training nodes."""

    def add_trace(self, trace: Dict[str, Any]) -> None:
        """Add a transition trace to the buffer."""
        ...

    def sample_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample a batch of traces prioritized by salience."""
        ...

    def clear(self) -> None:
        """Clear all stored traces."""
        ...

    def __len__(self) -> int:
        ...


# ─────────────────────────────────────────────────────────────────────────────
# 2. Local In-Memory Implementations (Default / Backward Compatibility)
# ─────────────────────────────────────────────────────────────────────────────

class InMemoryStateStore(BaseStateStore):
    """Local, in-memory dictionary-based GWT state store."""

    def __init__(self) -> None:
        self._buffers: Dict[str, torch.Tensor] = {}
        self._saliences: Dict[str, float] = {}

    def update_buffer(self, name: str, tensor: torch.Tensor) -> None:
        self._buffers[name] = tensor

    def get_buffer(self, name: str) -> torch.Tensor:
        if name not in self._buffers:
            raise KeyError(f"No latent buffer found for '{name}' in InMemoryStateStore.")
        return self._buffers[name]

    def list_buffers(self) -> Dict[str, torch.Tensor]:
        return self._buffers

    def update_salience(self, name: str, score: float) -> None:
        self._saliences[name] = float(score)

    def get_salience(self, name: str) -> float:
        return self._saliences.get(name, 0.0)

    def list_saliences(self) -> Dict[str, float]:
        return self._saliences

    def clear(self) -> None:
        self._buffers.clear()
        self._saliences.clear()


class InMemoryReplayBufferStore(BaseReplayBufferStore):
    """Local, in-memory list-based episodic GWT replay cache."""

    def __init__(self, max_size: int = 1000) -> None:
        self.max_size = max_size
        self.buffer: List[Dict[str, Any]] = []

    def add_trace(self, trace: Dict[str, Any]) -> None:
        # Enforce copy properties locally
        copied_trace = {
            "latent_states": {k: v.detach().clone() for k, v in trace["latent_states"].items()},
            "context": trace["context"].detach().clone(),
            "salience": float(trace["salience"]),
        }
        self.buffer.append(copied_trace)
        if len(self.buffer) > self.max_size:
            min_idx = min(range(len(self.buffer)), key=lambda i: self.buffer[i]["salience"])
            self.buffer.pop(min_idx)

    def sample_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        if not self.buffer:
            return []
        saliences = torch.tensor([t["salience"] for t in self.buffer], dtype=torch.float32)
        saliences = saliences - saliences.max()
        probs = torch.softmax(saliences, dim=0)
        indices = torch.multinomial(probs, num_samples=batch_size, replacement=True)
        return [self.buffer[i] for i in indices.tolist()]

    def clear(self) -> None:
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Redis Distributed Implementations (Horizontally Scalable)
# ─────────────────────────────────────────────────────────────────────────────

def _serialize_tensor(tensor: torch.Tensor) -> bytes:
    """Serializes a PyTorch tensor into raw binary bytes."""
    buf = io.BytesIO()
    # Detach and clone to be autograd-safe
    torch.save(tensor.detach().clone(), buf)
    return buf.getvalue()


def _deserialize_tensor(data: bytes) -> torch.Tensor:
    """Deserializes binary bytes back into a PyTorch tensor."""
    buf = io.BytesIO(data)
    return torch.load(buf, map_location=torch.device("cpu"))


class RedisStateStore(BaseStateStore):
    """
    Horizontally scalable, production-grade state store utilizing Redis Enterprise.
    Supports cluster deployment and allows GWT states to persist across scaled pods.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "cognitive_aug:state",
        **kwargs: Any,
    ) -> None:
        self.key_prefix = key_prefix
        self.client: Any = None
        self._fallback: Optional[InMemoryStateStore] = None

        try:
            import redis
            self.client = redis.from_url(redis_url, **kwargs)
            # Test connection
            self.client.ping()
            logger.info(f"Successfully connected to Redis state cache: {redis_url}")
        except Exception as e:
            logger.warning(
                f"Redis client connection failed: {e}. "
                "GWT StateStore gracefully falling back to local InMemoryStateStore."
            )
            self._fallback = InMemoryStateStore()

    def _get_key(self, section: str, name: str) -> str:
        return f"{self.key_prefix}:{section}:{name}"

    def update_buffer(self, name: str, tensor: torch.Tensor) -> None:
        if self._fallback is not None:
            return self._fallback.update_buffer(name, tensor)
        
        try:
            key = self._get_key("buffer", name)
            serialized = _serialize_tensor(tensor)
            self.client.set(key, serialized)
        except Exception as e:
            logger.error(f"GWT Redis Error in update_buffer: {e}. Bypassing.")

    def get_buffer(self, name: str) -> torch.Tensor:
        if self._fallback is not None:
            return self._fallback.get_buffer(name)

        try:
            key = self._get_key("buffer", name)
            data = self.client.get(key)
            if data is None:
                raise KeyError(f"No GWT latent buffer found in Redis for '{name}'.")
            return _deserialize_tensor(data)
        except Exception as e:
            if isinstance(e, KeyError):
                raise e
            logger.error(f"GWT Redis Error in get_buffer: {e}. Falling back to zero-filled mock.")
            # Graceful fallback: return a zero tensor of size [1, 4]
            return torch.zeros(1, 4)

    def list_buffers(self) -> Dict[str, torch.Tensor]:
        if self._fallback is not None:
            return self._fallback.list_buffers()

        try:
            pattern = self._get_key("buffer", "*")
            keys = self.client.keys(pattern)
            buffers = {}
            for key in keys:
                # Extract module name from key
                name = key.decode("utf-8").split(":")[-1]
                data = self.client.get(key)
                if data is not None:
                    buffers[name] = _deserialize_tensor(data)
            return buffers
        except Exception as e:
            logger.error(f"GWT Redis Error in list_buffers: {e}.")
            return {}

    def update_salience(self, name: str, score: float) -> None:
        if self._fallback is not None:
            return self._fallback.update_salience(name, score)

        try:
            key = self._get_key("salience", name)
            self.client.set(key, float(score))
        except Exception as e:
            logger.error(f"GWT Redis Error in update_salience: {e}.")

    def get_salience(self, name: str) -> float:
        if self._fallback is not None:
            return self._fallback.get_salience(name)

        try:
            key = self._get_key("salience", name)
            val = self.client.get(key)
            return float(val) if val is not None else 0.0
        except Exception as e:
            logger.error(f"GWT Redis Error in get_salience: {e}.")
            return 0.0

    def list_saliences(self) -> Dict[str, float]:
        if self._fallback is not None:
            return self._fallback.list_saliences()

        try:
            pattern = self._get_key("salience", "*")
            keys = self.client.keys(pattern)
            saliences = {}
            for key in keys:
                name = key.decode("utf-8").split(":")[-1]
                val = self.client.get(key)
                if val is not None:
                    saliences[name] = float(val)
            return saliences
        except Exception as e:
            logger.error(f"GWT Redis Error in list_saliences: {e}.")
            return {}

    def clear(self) -> None:
        if self._fallback is not None:
            return self._fallback.clear()

        try:
            pattern = f"{self.key_prefix}:*"
            keys = self.client.keys(pattern)
            if keys:
                self.client.delete(*keys)
        except Exception as e:
            logger.error(f"GWT Redis Error in clear state: {e}.")


class RedisReplayBufferStore(BaseReplayBufferStore):
    """
    Episodic memory store using Redis.
    Allows horizontally scaled pods to share experiences, preventing local memory bloating.
    """

    def __init__(
        self,
        max_size: int = 1000,
        redis_url: str = "redis://localhost:6379/0",
        key: str = "cognitive_aug:replay_buffer",
        **kwargs: Any,
    ) -> None:
        self.max_size = max_size
        self.key = key
        self.client: Any = None
        self._fallback: Optional[InMemoryReplayBufferStore] = None

        try:
            import redis
            self.client = redis.from_url(redis_url, **kwargs)
            self.client.ping()
        except Exception as e:
            logger.warning(
                f"Redis client connection failed for ReplayBuffer: {e}. "
                "GWT ReplayBuffer falling back to local InMemoryReplayBufferStore."
            )
            self._fallback = InMemoryReplayBufferStore(max_size)

    def add_trace(self, trace: Dict[str, Any]) -> None:
        if self._fallback is not None:
            return self._fallback.add_trace(trace)

        try:
            # Serialize trace tensors to byte dict
            ser_latents = {k: _serialize_tensor(v) for k, v in trace["latent_states"].items()}
            ser_context = _serialize_tensor(trace["context"])
            
            payload = {
                "latent_states": ser_latents,
                "context": ser_context,
                "salience": float(trace["salience"]),
            }
            # Pickle payload to keep bytes intact
            pickled = pickle.dumps(payload)
            
            # Push onto list
            self.client.lpush(self.key, pickled)
            
            # Bound the size (evict oldest from the list right side)
            if self.client.llen(self.key) > self.max_size:
                self.client.rpop(self.key)
        except Exception as e:
            logger.error(f"GWT Redis Error in add_trace: {e}.")

    def sample_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        if self._fallback is not None:
            return self._fallback.sample_batch(batch_size)

        try:
            # Fetch all items to do prioritised sampling
            raw_items = self.client.lrange(self.key, 0, -1)
            if not raw_items:
                return []
            
            traces = []
            saliences = []
            for item in raw_items:
                payload = pickle.loads(item)
                # Parse back to PyTorch tensors
                latents = {k: _deserialize_tensor(v) for k, v in payload["latent_states"].items()}
                context = _deserialize_tensor(payload["context"])
                
                trace = {
                    "latent_states": latents,
                    "context": context,
                    "salience": payload["salience"],
                }
                traces.append(trace)
                saliences.append(payload["salience"])

            salience_tensor = torch.tensor(saliences, dtype=torch.float32)
            salience_tensor = salience_tensor - salience_tensor.max()
            probs = torch.softmax(salience_tensor, dim=0)
            indices = torch.multinomial(probs, num_samples=batch_size, replacement=True)
            
            return [traces[i] for i in indices.tolist()]
        except Exception as e:
            logger.error(f"GWT Redis Error in sample_batch: {e}.")
            return []

    def clear(self) -> None:
        if self._fallback is not None:
            return self._fallback.clear()

        try:
            self.client.delete(self.key)
        except Exception as e:
            logger.error(f"GWT Redis Error in clear replay buffer: {e}.")

    def __len__(self) -> int:
        if self._fallback is not None:
            return len(self._fallback)

        try:
            return self.client.llen(self.key)
        except Exception as e:
            logger.error(f"GWT Redis Error in len: {e}.")
            return 0
