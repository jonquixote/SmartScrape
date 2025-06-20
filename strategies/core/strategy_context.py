"""
Strategy Context module for providing shared execution context to strategies.

This module contains the StrategyContext class which provides shared services,
configuration, and utilities to strategies during execution.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Union, TypeVar, Type, Callable
import copy

from core.service_registry import ServiceRegistry
from core.configuration import get_resource_config

T = TypeVar('T')

class StrategyContext:
    """
    Context object providing access to shared resources and services for strategies.
    
    StrategyContext serves as a central point for strategies to access:
    1. Configuration parameters
    2. Service registry and services
    3. Shared state and storage
    4. Utility methods for common tasks
    
    This creates a controlled dependency injection mechanism where strategies
    receive what they need without being tightly coupled to implementation details.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize a new strategy context.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.service_registry = ServiceRegistry()
        self._logger = logging.getLogger(self.__class__.__name__)
        
        # Shared state between strategies (mutable)
        self._state: Dict[str, Any] = {}
        
        # Resources managed by this context
        self._resources: List[Any] = []
        
        # Ensure resource services config is included
        if 'resource_services' not in self.config:
            resource_config = get_resource_config()
            self.config['resource_services'] = resource_config
        
    @property
    def logger(self) -> logging.Logger:
        """Get the context logger."""
        return self._logger
    
    def get_service(self, service_name: str) -> Any:
        """
        Get a service from the service registry.
        
        Args:
            service_name: Name of the service to retrieve
            
        Returns:
            The requested service instance
            
        Raises:
            KeyError: If the service is not found
        """
        return self.service_registry.get_service(service_name)
    
    def register_service(self, name: str, service: Any) -> None:
        """
        Register a service directly with the service registry.
        
        Args:
            name: Service name
            service: Service instance
        """
        self.service_registry.register_service(name, service)
        self.logger.debug(f"Registered service: {name}")
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value with an optional default.
        
        Args:
            key: Configuration key (supports dot notation for nested access)
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        if '.' not in key:
            return self.config.get(key, default)
        
        # Handle nested configuration with dot notation
        parts = key.split('.')
        current = self.config
        
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
            
        return current
    
    def create_sub_context(self, config: Optional[Dict[str, Any]] = None) -> 'StrategyContext':
        """
        Create a derived context that inherits from this context.
        
        Args:
            config: Additional configuration to overlay on the current config
            
        Returns:
            A new StrategyContext instance with merged configuration
        """
        # Create a deep copy of the current config
        merged_config = copy.deepcopy(self.config)
        
        # Merge in the new config if provided
        if config:
            self._deep_merge(merged_config, config)
            
        # Create a new context with the merged config
        sub_context = StrategyContext(merged_config)
        
        # Share the same service registry
        sub_context.service_registry = self.service_registry
        
        return sub_context
    
    @classmethod
    def with_services(cls, *service_names: str, config: Optional[Dict[str, Any]] = None) -> 'StrategyContext':
        """
        Create a new context with specific services initialized.
        
        Args:
            *service_names: Names of services to initialize
            config: Optional configuration for the context
            
        Returns:
            A new StrategyContext with the specified services initialized
        """
        context = cls(config)
        
        # Initialize the specified services
        for service_name in service_names:
            try:
                # This will initialize the service if not already initialized
                context.get_service(service_name)
                context.logger.debug(f"Initialized service: {service_name}")
            except KeyError:
                context.logger.warning(f"Service not found: {service_name}")
                
        return context
    
    @classmethod
    def with_resource_services(cls, config: Optional[Dict[str, Any]] = None) -> 'StrategyContext':
        """
        Create a new context with all resource services initialized.
        
        Args:
            config: Optional configuration for the context
            
        Returns:
            A new StrategyContext with all resource services initialized
        """
        context = cls(config)
        
        # Register and initialize all resource services
        context.service_registry.register_resource_services(context.config.get('resource_services'))
        
        # Initialize common resource services
        try:
            context.get_session_manager()
            context.get_error_classifier()
            context.get_rate_limiter()
            context.logger.debug("Resource services initialized")
        except Exception as e:
            context.logger.error(f"Failed to initialize resource services: {e}")
        
        return context
    
    def set_state(self, key: str, value: Any) -> None:
        """
        Set a value in the shared state.
        
        Args:
            key: State key
            value: State value
        """
        self._state[key] = value
        
    def get_state(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the shared state.
        
        Args:
            key: State key
            default: Default value if key is not found
            
        Returns:
            State value or default
        """
        return self._state.get(key, default)
    
    def register_resource(self, resource: Any) -> None:
        """
        Register a resource to be managed by this context.
        
        Resources will be properly closed/released when cleanup() is called.
        
        Args:
            resource: Resource to register (should have a close/shutdown method)
        """
        self._resources.append(resource)
        
    def cleanup(self) -> None:
        """Clean up all resources managed by this context."""
        for resource in self._resources:
            try:
                # Try different methods for cleanup
                if hasattr(resource, 'close'):
                    resource.close()
                elif hasattr(resource, 'shutdown'):
                    resource.shutdown()
                elif hasattr(resource, 'cleanup'):
                    resource.cleanup()
            except Exception as e:
                self.logger.warning(f"Error cleaning up resource: {e}")
                
        # Clear the resources list
        self._resources.clear()
        
        # Clear the state
        self._state.clear()
    
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Deep merge dictionaries, recursively merging nested dictionaries.
        
        Args:
            target: Target dictionary to merge into
            source: Source dictionary to merge from
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value
                
    # Resource service access methods
    
    def get_session_manager(self):
        """Get the session manager service."""
        return self.get_service("session_manager")
    
    def get_rate_limiter(self):
        """Get the rate limiter service."""
        return self.get_service("rate_limiter")
    
    def get_proxy_manager(self):
        """Get the proxy manager service."""
        return self.get_service("proxy_manager")
    
    def get_error_classifier(self):
        """Get the error classifier service."""
        return self.get_service("error_classifier")
    
    def get_retry_manager(self):
        """Get the retry manager service."""
        return self.get_service("retry_manager")
    
    def get_circuit_breaker_manager(self):
        """Get the circuit breaker manager service."""
        return self.get_service("circuit_breaker_manager")
    
    # Resource helper methods
    
    def get_session(self, domain: str, session_type: str = 'http'):
        """
        Get a session for the specified domain.
        
        Args:
            domain: The domain to get a session for
            session_type: The type of session ('http' or 'browser')
            
        Returns:
            A session object appropriate for the domain and type
        """
        session_manager = self.get_session_manager()
        
        if session_type == 'browser':
            return session_manager.get_browser_session(domain)
        else:
            return session_manager.get_session(domain)
    
    def get_proxy(self, domain: str = None, tags: List[str] = None):
        """
        Get a proxy for the specified domain.
        
        Args:
            domain: Optional domain to get a proxy for
            tags: Optional list of tags to filter proxies by
            
        Returns:
            A proxy configuration dict or None if proxies are disabled
        """
        try:
            proxy_manager = self.get_proxy_manager()
            if proxy_manager and proxy_manager.is_initialized:
                if domain:
                    return proxy_manager.get_proxy_for_domain(domain, tags)
                else:
                    return proxy_manager.get_proxy(tags)
        except (KeyError, Exception) as e:
            self.logger.debug(f"Failed to get proxy: {e}")
        return None
    
    def should_wait(self, domain: str) -> bool:
        """
        Check if the strategy should wait before making a request to comply with rate limits.
        
        Args:
            domain: The domain to check rate limits for
            
        Returns:
            True if should wait (will wait automatically), False otherwise
        """
        try:
            rate_limiter = self.get_rate_limiter()
            if rate_limiter and rate_limiter.is_initialized:
                return rate_limiter.wait_if_needed(domain)
        except (KeyError, Exception) as e:
            self.logger.debug(f"Failed to check rate limits: {e}")
        return False
    
    def classify_error(self, error: Exception, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Classify an error using the error classifier service.
        
        Args:
            error: The exception to classify
            metadata: Optional metadata about the error context
            
        Returns:
            A dictionary with error classification details
        """
        try:
            error_classifier = self.get_error_classifier()
            if error_classifier and error_classifier.is_initialized:
                return error_classifier.classify_exception(error, metadata)
        except (KeyError, Exception) as e:
            self.logger.debug(f"Failed to classify error: {e}")
        
        # Default classification if service not available
        return {
            'original_exception': error,
            'error_message': str(error),
            'error_type': type(error).__name__,
            'category': 'unknown',
            'severity': 'persistent',
            'is_retryable': False,
            'metadata': metadata or {},
            'suggested_actions': []
        }
    
    def should_retry(self, operation: Callable, max_attempts: int = None, retry_for: List[str] = None) -> Callable:
        """
        Decorator to retry an operation according to retry policies.
        
        Args:
            operation: The function to retry
            max_attempts: Override default max retry attempts
            retry_for: Categories of errors to retry for
            
        Returns:
            A wrapped function that includes retry logic
        """
        try:
            retry_manager = self.get_retry_manager()
            if retry_manager and retry_manager.is_initialized:
                return retry_manager.retry(operation, max_attempts, retry_for)
        except (KeyError, Exception) as e:
            self.logger.debug(f"Failed to apply retry policy: {e}")
        
        # Return the original function if retry manager not available
        return operation
    
    def is_circuit_open(self, circuit_name: str) -> bool:
        """
        Check if a circuit breaker is open for a specific service/domain.
        
        Args:
            circuit_name: The name of the circuit to check
            
        Returns:
            True if the circuit is open (requests not allowed), False otherwise
        """
        try:
            circuit_manager = self.get_circuit_breaker_manager()
            if circuit_manager and circuit_manager.is_initialized:
                circuit = circuit_manager.get_circuit_breaker(circuit_name)
                return not circuit.allow_request()
        except (KeyError, Exception) as e:
            self.logger.debug(f"Failed to check circuit state: {e}")
        
        # Default to closed (allow requests) if service not available
        return False
    
    def with_circuit_breaker(self, circuit_name: str, operation: Callable) -> Any:
        """
        Execute an operation with circuit breaker protection.
        
        Args:
            circuit_name: The name of the circuit to use
            operation: The function to execute with protection
            
        Returns:
            A wrapped function that includes circuit breaker logic
        """
        try:
            circuit_manager = self.get_circuit_breaker_manager()
            if circuit_manager and circuit_manager.is_initialized:
                return circuit_manager.circuit_breaker(circuit_name)(operation)
        except (KeyError, Exception) as e:
            self.logger.debug(f"Failed to apply circuit breaker: {e}")
        
        # Return the original function if circuit breaker not available
        return operation
    
    def extract_domain(self, url: str) -> str:
        """
        Extract domain from URL for service operations.
        
        Args:
            url: The URL to extract domain from
            
        Returns:
            Domain string (e.g., 'example.com')
        """
        pattern = r'(?:https?:\/\/)?(?:www\.)?([^\/\?]+)'
        match = re.search(pattern, url)
        if match:
            domain = match.group(1)
            # Strip port number if present
            domain = domain.split(':')[0]
            return domain
        return url  # Fallback to original URL if extraction fails