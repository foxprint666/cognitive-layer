import logging
from typing import Optional, Any
import torch
import torch.nn as nn
from .engine import DataFlowManager

logger = logging.getLogger(__name__)


class ModuleAdapter(nn.Module):
    """
    Adapter wrapper that hooks into any PyTorch nn.Module.
    Automatically intercepts latent representations via forward hooks and routes
    them through the DataFlowManager.
    """

    def __init__(
        self,
        name: str,
        module: nn.Module,
        latent_dim: int,
        data_flow: DataFlowManager,
        key_dim: int = 64,
        projection_in_dim: Optional[int] = None,
    ) -> None:
        """
        Args:
            name: Unique name of the module.
            module: The PyTorch nn.Module to wrap.
            latent_dim: The workspace latent representation dimension.
            data_flow: The active DataFlowManager instance.
            key_dim: Dimensionality of module keys for attention matching.
            projection_in_dim: If the module's raw output dimension does not match latent_dim,
                              specify it here to automatically construct a learnable projection.
        """
        super().__init__()
        self.name = name
        self.module = module
        self.latent_dim = latent_dim
        self.data_flow = data_flow
        self.key_dim = key_dim

        # Projection layer: maps module's raw output to GWT workspace latent_dim if needed
        self.projection: Optional[nn.Linear] = None
        if projection_in_dim is not None and projection_in_dim != latent_dim:
            self.projection = nn.Linear(projection_in_dim, latent_dim)
            logger.info(
                f"Created auto-projection layer for module '{name}': {projection_in_dim} -> {latent_dim}"
            )

        # Key projection: represents the module state in attention key space
        self.key_proj = nn.Linear(latent_dim, key_dim)

        # Buffer to store the latest broadcasted state from the workspace
        self.register_buffer("last_broadcast", torch.zeros(1, latent_dim))

        self._hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._register_forward_hook()

    def _register_forward_hook(self) -> None:
        """
        Register a PyTorch forward hook on the wrapped module to intercept latent outputs.
        """

        def hook_fn(
            module: nn.Module, inputs: Any, outputs: Any
        ) -> Any:
            # Unwrap outputs if they are a tuple or list (e.g. LSTM outputs or auxiliary returns)
            if isinstance(outputs, tuple):
                latent = outputs[0]
            elif isinstance(outputs, list):
                latent = outputs[0]
            else:
                latent = outputs

            # Ensure latent is a float tensor
            if not isinstance(latent, torch.Tensor):
                raise TypeError(
                    f"Expected tensor output from module '{self.name}', got {type(outputs)}"
                )

            # Apply projection if defined
            if self.projection is not None:
                # If output is 3D (e.g., sequence [B, T, D]), project the sequence or the final element
                # For Phase v0.1 MVP, we assume standard batches [B, D] or sequence outputs.
                # Project along the last dimension.
                latent = self.projection(latent)

            # Save the captured state to the central data flow manager
            self.data_flow.update_buffer(self.name, latent)

            return outputs

        # Register hook on the wrapped module
        self._hook_handle = self.module.register_forward_hook(hook_fn)
        logger.debug(f"Forward hook successfully registered on module '{self.name}'.")

    def get_key(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Generate a query/key vector representing the current state of the module.
        Used by GWT AttentionSelector.
        """
        # If latent is empty or has mismatched batch size, handle it gracefully
        # Normally latent shape is [B, latent_dim], key is [B, key_dim]
        # We apply key_proj along the last dimension
        return self.key_proj(latent)

    def receive_broadcast(self, broadcast_state: torch.Tensor) -> None:
        """
        Callback called by the CognitiveAugEngine to update this module with
        the latest global workspace broadcast.
        """
        # Ensure correct shape and copy to the buffer
        self.last_broadcast = broadcast_state.clone()
        logger.debug(f"Module '{self.name}' received workspace broadcast.")

    def get_last_broadcast(self) -> torch.Tensor:
        """
        Helper to retrieve the latest broadcasted state from the workspace.
        Can be queried by user code or injected in custom forwards.
        """
        return self.last_broadcast

    def remove_hooks(self) -> None:
        """
        Clean up and remove registered hooks.
        """
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
            logger.debug(f"Removed forward hook from module '{self.name}'.")

    def __del__(self) -> None:
        self.remove_hooks()
