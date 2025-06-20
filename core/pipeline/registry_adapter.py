"""
Pipeline Registry Adapter Module.

This module provides an adapter class that adapts the PipelineRegistry
to implement the BaseService interface.
"""
import logging
from typing import Dict, Any, Optional

from core.service_interface import BaseService
from core.pipeline.registry import PipelineRegistry

logger = logging.getLogger(__name__)

class PipelineRegistryAdapter(BaseService):
    """
    Adapter class that wraps PipelineRegistry to implement BaseService interface.
    
    This allows the PipelineRegistry to be used with the service registry
    without modifying its original implementation.
    """
    
    def __init__(self):
        """Initialize the adapter with a PipelineRegistry instance."""
        self._registry = PipelineRegistry()
        self._initialized = False
    
    @property
    def registry(self) -> PipelineRegistry:
        """Get the underlying PipelineRegistry instance."""
        return self._registry
    
    @property
    def name(self) -> str:
        """Return the name of the service."""
        return "pipeline_registry"
    
    @property
    def is_initialized(self) -> bool:
        """Check if the service has been initialized."""
        return self._initialized
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the registry service."""
        if not self._initialized:
            logger.info("Initializing pipeline registry service")
            # Any specific initialization for the registry could be done here
            self._initialized = True
    
    def shutdown(self) -> None:
        """Shutdown the registry service."""
        logger.info("Shutting down pipeline registry service")
        # Perform any necessary cleanup
        self._initialized = False
    
    # Delegate all other methods to the underlying registry
    def __getattr__(self, name):
        return getattr(self._registry, name)
