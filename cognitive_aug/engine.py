import logging
from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModuleRegistry:
    """
    Registry for managing brain-inspired cognitive modules and their adapters.
    """

    def __init__(self) -> None:
        self._adapters: Dict[str, Any] = {}

    def register(self, name: str, adapter: Any) -> None:
        """
        Register a ModuleAdapter with a unique name.
        """
        if name in self._adapters:
            logger.warning(f"Overwriting already registered module adapter: {name}")
        self._adapters[name] = adapter
        logger.debug(f"Successfully registered module: {name}")

    def get(self, name: str) -> Any:
        """
        Retrieve a registered ModuleAdapter by name.
        """
        if name not in self._adapters:
            raise KeyError(f"Module '{name}' is not registered.")
        return self._adapters[name]

    def list_names(self) -> List[str]:
        """
        List names of all registered modules.
        """
        return list(self._adapters.keys())

    def list_adapters(self) -> List[Any]:
        """
        List all registered ModuleAdapter instances.
        """
        return list(self._adapters.values())

    def clear(self) -> None:
        """
        Clear the registry.
        """
        self._adapters.clear()


class DataFlowManager:
    """
    Manages communication buffers, dynamic latent spaces, and shapes of all registered modules.
    Ensures framework-integrated, high-performance tensor transfers and routing.
    """

    def __init__(self) -> None:
        self._buffers: Dict[str, torch.Tensor] = {}

    def update_buffer(self, name: str, tensor: torch.Tensor) -> None:
        """
        Update the latent space buffer for a specific module.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Buffer content must be a PyTorch Tensor.")
        self._buffers[name] = tensor

    def get_buffer(self, name: str) -> torch.Tensor:
        """
        Retrieve the latent space buffer for a specific module.
        """
        if name not in self._buffers:
            raise KeyError(f"No latent buffer found for module '{name}'. Ensure the module has run a forward pass.")
        return self._buffers[name]

    def list_buffers(self) -> Dict[str, torch.Tensor]:
        """
        Get all current module latent buffers.
        """
        return self._buffers

    def clear_buffers(self) -> None:
        """
        Clear all cached latent state buffers.
        """
        self._buffers.clear()


class CognitiveAugEngine:
    """
    The main orchestrator engine of the cognitive augmentation library.
    Manages module lifecycles, registries, data flows, and workspace communication steps.
    """

    def __init__(self) -> None:
        self.registry = ModuleRegistry()
        self.data_flow = DataFlowManager()
        self.workspace: Optional[nn.Module] = None

    def register_module(
        self, name: str, module: nn.Module, latent_dim: int, **kwargs: Any
    ) -> Any:
        """
        Wrap an existing PyTorch nn.Module with a ModuleAdapter and register it with the engine.
        
        Args:
            name: Unique identifier for the module.
            module: The PyTorch module to wrap.
            latent_dim: Dimension of the latent space captured/used by this module.
        """
        from .adapters import ModuleAdapter

        adapter = ModuleAdapter(
            name=name,
            module=module,
            latent_dim=latent_dim,
            data_flow=self.data_flow,
            **kwargs,
        )
        self.registry.register(name, adapter)
        return adapter

    def attach_workspace(self, workspace: nn.Module) -> None:
        """
        Attach a global workspace to orchestrate attentional selection and broadcasting.
        """
        self.workspace = workspace
        logger.info("Successfully attached workspace to CognitiveAugEngine.")

    def step(self) -> torch.Tensor:
        """
        Perform one full GWT cycle:
        1. Retrieve captured latent states from each registered module's buffer.
        2. Feed these states to the AttentionSelector inside the Workspace.
        3. Broadcast the winner/selected workspace state back to all registered ModuleAdapters.
        
        Returns:
            The broadcasted workspace state tensor.
        """
        if self.workspace is None:
            raise ValueError("No workspace attached. Call `attach_workspace` before stepping the engine.")

        # Gather all module latent states and keys
        adapters = self.registry.list_adapters()
        if not adapters:
            raise ValueError("No modules registered with the engine.")

        latent_states = {}
        keys = {}

        for adapter in adapters:
            try:
                # Capture the latest forward pass output cached in the data flow manager
                latent = self.data_flow.get_buffer(adapter.name)
                latent_states[adapter.name] = latent
                
                # Fetch module key (or generate a default key projection)
                keys[adapter.name] = adapter.get_key(latent)
            except KeyError:
                # If a module has not run, we skip or use a zero tensor.
                # Let's log a warning and use a default zero tensor based on the registered latent_dim.
                logger.warning(
                    f"Module '{adapter.name}' has not run a forward pass in the current step. "
                    "Using default zero latent vector."
                )
                # Infer batch size from other buffers if possible, default to 1
                batch_size = 1
                for buff in self.data_flow.list_buffers().values():
                    batch_size = buff.shape[0]
                    break
                
                # Retrieve the device of the module
                device = next(adapter.module.parameters()).device if list(adapter.module.parameters()) else torch.device("cpu")
                
                latent = torch.zeros(batch_size, adapter.latent_dim, device=device)
                latent_states[adapter.name] = latent
                keys[adapter.name] = adapter.get_key(latent)

        # Execute workspace selection & broadcast
        # The workspace expects a dictionary of latent representations and module keys
        broadcast_state = self.workspace(latent_states, keys)

        # Distribute workspace contents back to all registered ModuleAdapters
        for adapter in adapters:
            adapter.receive_broadcast(broadcast_state)

        return broadcast_state
