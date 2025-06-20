from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseService(ABC):
    """Base interface for all SmartScrape services."""
    
    @abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the service with the given configuration."""
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Perform cleanup and shutdown operations."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the service."""
        pass
    
    @property
    def is_initialized(self) -> bool:
        """Check if the service has been initialized."""
        return getattr(self, '_initialized', False)
    
    def get_service_health(self) -> Dict[str, Any]:
        """
        Get the health status of this service.
        
        Returns:
            Dict with health information including:
            - status: 'healthy', 'degraded', or 'unhealthy'
            - details: Additional information about the service health
            - metrics: Optional service-specific metrics
        """
        # Default implementation returns basic health check
        return {
            'status': 'healthy' if self.is_initialized else 'unhealthy',
            'details': 'Service is initialized' if self.is_initialized else 'Service is not initialized',
            'metrics': {}
        }
    
    def __enter__(self):
        """Context manager entry."""
        if not self.is_initialized:
            self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()