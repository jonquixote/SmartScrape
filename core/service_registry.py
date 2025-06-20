import logging
from typing import Dict, Type, Any, Optional, Set, List
import threading
import time
from collections import defaultdict

from core.service_interface import BaseService

logger = logging.getLogger(__name__)

class ServiceRegistry:
    """Registry for all SmartScrape services with dependency management."""
    
    _instance = None
    _lock = threading.RLock()
    
    # Service group definitions for initialization
    SERVICE_GROUP_CORE = "core"
    SERVICE_GROUP_NETWORK = "network"
    SERVICE_GROUP_ERROR_HANDLING = "error_handling"
    SERVICE_GROUP_EXTRACTION = "extraction"
    SERVICE_GROUP_STRATEGY = "strategy"
    
    # Default initialization order for service groups
    DEFAULT_GROUP_ORDER = [
        SERVICE_GROUP_CORE,
        SERVICE_GROUP_ERROR_HANDLING,
        SERVICE_GROUP_NETWORK,
        SERVICE_GROUP_EXTRACTION,
        SERVICE_GROUP_STRATEGY
    ]
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ServiceRegistry, cls).__new__(cls)
                cls._instance._services = {}
                cls._instance._service_classes = {}
                cls._instance._dependencies = {}
                cls._instance._service_groups = defaultdict(set)
                cls._instance._initialized = False
            return cls._instance
    
    def register_service_class(self, service_class: Type[BaseService], 
                               dependencies: Optional[Set[str]] = None,
                               group: str = None) -> None:
        """
        Register a service class in the registry.
        
        Args:
            service_class: The service class to register
            dependencies: Optional set of service names this service depends on
            group: Optional service group for initialization ordering
        """
        if not issubclass(service_class, BaseService):
            raise TypeError(f"{service_class.__name__} is not a subclass of BaseService")
        
        # Create an instance to get the name
        temp_instance = service_class()
        service_name = temp_instance.name
        
        with self._lock:
            if service_name in self._service_classes:
                logger.warning(f"Service class {service_name} already registered. Overwriting.")
            
            self._service_classes[service_name] = service_class
            if dependencies:
                self._dependencies[service_name] = dependencies
            else:
                self._dependencies[service_name] = set()
            
            # Add to service group if specified
            if group:
                self._service_groups[group].add(service_name)
            
            logger.info(f"Registered service class: {service_name}")
    
    def register_service(self, service_name: str, service: BaseService) -> None:
        """
        Register an existing service instance directly.
        
        Args:
            service_name: The name to register the service under
            service: The service instance to register
        """
        # Enhanced BaseService validation to handle complex import scenarios
        is_base_service = isinstance(service, BaseService)
        
        # Fallback validation: check if BaseService is in the MRO
        if not is_base_service:
            from core.service_interface import BaseService as BSCheck
            is_base_service = isinstance(service, BSCheck) or BSCheck in type(service).__mro__
        
        # Additional fallback: check class names in MRO for robustness
        if not is_base_service:
            mro_class_names = [cls.__name__ for cls in type(service).__mro__]
            is_base_service = 'BaseService' in mro_class_names
        
        if not is_base_service:
            logger.error(f"Service validation failed for {service_name}:")
            logger.error(f"  Service type: {type(service)}")
            logger.error(f"  Service MRO: {type(service).__mro__}")
            logger.error(f"  isinstance(service, BaseService): {isinstance(service, BaseService)}")
            raise TypeError(f"Service {service_name} is not an instance of BaseService")
        
        with self._lock:
            if service_name in self._services:
                logger.warning(f"Service {service_name} already registered. Overwriting.")
            
            self._services[service_name] = service
            logger.info(f"Registered service instance: {service_name}")
    
    def get_service(self, service_name: str, 
                   config: Optional[Dict[str, Any]] = None) -> BaseService:
        """Get or create a service instance by name."""
        with self._lock:
            # Check if service is already instantiated
            if service_name in self._services:
                return self._services[service_name]
            
            # Check if service class is registered
            if service_name not in self._service_classes:
                raise KeyError(f"Service {service_name} not registered")
            
            # Check for circular dependencies
            deps_chain = self._resolve_dependencies(service_name, set())
            
            # Initialize all dependencies first
            for dep in deps_chain:
                if dep not in self._services and dep in self._service_classes:
                    self._services[dep] = self._service_classes[dep]()
                    self._services[dep].initialize(config)
            
            # Create and initialize the requested service
            self._services[service_name] = self._service_classes[service_name]()
            self._services[service_name].initialize(config)
            
            return self._services[service_name]
    
    def _resolve_dependencies(self, service_name: str, visited: Set[str]) -> list:
        """Resolve dependencies for a service and check for circular dependencies."""
        if service_name in visited:
            visited_str = " -> ".join(visited) + " -> " + service_name
            raise ValueError(f"Circular dependency detected: {visited_str}")
        
        new_visited = visited.copy()
        new_visited.add(service_name)
        
        # Get direct dependencies
        deps = self._dependencies.get(service_name, set())
        
        # Build dependency chain (dependencies first, then the service)
        deps_chain = []
        for dep in deps:
            if dep not in self._service_classes:
                raise KeyError(f"Dependency {dep} not registered for service {service_name}")
                
            if dep not in deps_chain:
                deps_chain.extend(self._resolve_dependencies(dep, new_visited))
        
        # Add this service to the chain if not already included
        if service_name not in deps_chain:
            deps_chain.append(service_name)
        
        return deps_chain
    
    def initialize_all(self, config: Optional[Dict[str, Any]] = None,
                      init_order: Optional[List[str]] = None) -> None:
        """
        Initialize all registered services.
        
        Args:
            config: Optional configuration dictionary for all services
            init_order: Optional custom initialization order (list of service names)
                        If not provided, services will be initialized by groups
                        or dependency resolution as fallback
        """
        with self._lock:
            if self._initialized:
                return
            
            start_time = time.time()
            logger.info("Starting service initialization")
            
            # Option 1: Custom initialization order
            if init_order:
                for service_name in init_order:
                    if service_name in self._service_classes:
                        try:
                            self.get_service(service_name, config)
                            logger.debug(f"Initialized service: {service_name}")
                        except Exception as e:
                            logger.error(f"Failed to initialize {service_name}: {str(e)}")
            
            # Option 2: Group-based initialization
            elif self._service_groups:
                # Initialize services by group
                for group in self.DEFAULT_GROUP_ORDER:
                    if group in self._service_groups:
                        logger.debug(f"Initializing service group: {group}")
                        for service_name in self._service_groups[group]:
                            if service_name in self._service_classes and service_name not in self._services:
                                try:
                                    self.get_service(service_name, config)
                                    logger.debug(f"Initialized service: {service_name}")
                                except Exception as e:
                                    logger.error(f"Failed to initialize {service_name}: {str(e)}")
                
                # Initialize any services not in a group
                ungrouped = set(self._service_classes.keys()) - {s for g in self._service_groups.values() for s in g}
                for service_name in ungrouped:
                    if service_name not in self._services:
                        try:
                            self.get_service(service_name, config)
                            logger.debug(f"Initialized ungrouped service: {service_name}")
                        except Exception as e:
                            logger.error(f"Failed to initialize {service_name}: {str(e)}")
            
            # Option 3: Default dependency-based initialization
            else:
                for service_name in self._service_classes:
                    try:
                        self.get_service(service_name, config)
                        logger.debug(f"Initialized service: {service_name}")
                    except Exception as e:
                        logger.error(f"Failed to initialize {service_name}: {str(e)}")
            
            self._initialized = True
            elapsed = time.time() - start_time
            logger.info(f"Service initialization completed in {elapsed:.2f}s")
    
    def shutdown_all(self) -> None:
        """Shutdown all initialized services."""
        with self._lock:
            if not self._services:
                return
                
            start_time = time.time()
            logger.info("Starting service shutdown")
            
            # Shutdown in reverse dependency order
            for service_name in reversed(list(self._services.keys())):
                try:
                    self._services[service_name].shutdown()
                    logger.debug(f"Shut down service: {service_name}")
                except Exception as e:
                    logger.error(f"Error shutting down {service_name}: {str(e)}")
            
            self._services.clear()
            self._initialized = False
            
            elapsed = time.time() - start_time
            logger.info(f"Service shutdown completed in {elapsed:.2f}s")
    
    def get_all_service_names(self) -> Set[str]:
        """Get the names of all registered services."""
        return set(self._service_classes.keys())
    
    def is_service_initialized(self, service_name: str) -> bool:
        """Check if a specific service is initialized."""
        return service_name in self._services and self._services[service_name].is_initialized
    
    def check_service_dependencies(self, service_name: str) -> Dict[str, bool]:
        """
        Check if all dependencies for a service are available and initialized.
        
        Args:
            service_name: The name of the service to check dependencies for
            
        Returns:
            Dict mapping dependency names to their initialization status
        """
        if service_name not in self._dependencies:
            raise KeyError(f"Service {service_name} not registered")
            
        result = {}
        for dep in self._dependencies[service_name]:
            result[dep] = self.is_service_initialized(dep)
            
        return result
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get the health status of all initialized services.
        
        Returns:
            Dict containing overall system health and individual service health
        """
        with self._lock:
            service_health = {}
            healthy_count = 0
            degraded_count = 0
            unhealthy_count = 0
            
            # Get health for each initialized service
            for service_name, service in self._services.items():
                try:
                    health = service.get_service_health()
                    service_health[service_name] = health
                    
                    if health['status'] == 'healthy':
                        healthy_count += 1
                    elif health['status'] == 'degraded':
                        degraded_count += 1
                    else:
                        unhealthy_count += 1
                except Exception as e:
                    logger.error(f"Error getting health for {service_name}: {str(e)}")
                    service_health[service_name] = {
                        'status': 'unhealthy',
                        'details': f"Exception during health check: {str(e)}",
                        'metrics': {}
                    }
                    unhealthy_count += 1
            
            # Determine overall system health
            total_services = len(self._services)
            status = 'healthy'
            
            if unhealthy_count > 0:
                status = 'unhealthy'
            elif degraded_count > 0:
                status = 'degraded'
            
            # Build system health report
            return {
                'status': status,
                'details': f"{healthy_count}/{total_services} services healthy, {degraded_count} degraded, {unhealthy_count} unhealthy",
                'initialized_services': len(self._services),
                'registered_services': len(self._service_classes),
                'services': service_health
            }
    
    def register_resource_services(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Register the standard resource management and error handling services.
        This is a convenience method to make integration easier.
        
        Args:
            config: Optional configuration dictionary for the services
        """
        try:
            from core.error_classifier import ErrorClassifier
            from core.circuit_breaker import CircuitBreakerManager
            from core.retry_manager import RetryManager
            from core.session_manager import SessionManager
            from core.rate_limiter import RateLimiter
            from core.proxy_manager import ProxyManager
            
            # Register core services with proper dependencies and groups
            # Error handling services
            self.register_service_class(
                ErrorClassifier, 
                dependencies=None,
                group=self.SERVICE_GROUP_ERROR_HANDLING
            )
            
            self.register_service_class(
                CircuitBreakerManager, 
                dependencies=None,
                group=self.SERVICE_GROUP_ERROR_HANDLING
            )
            
            self.register_service_class(
                RetryManager, 
                dependencies={"error_classifier"},
                group=self.SERVICE_GROUP_ERROR_HANDLING
            )
            
            # Network related services
            self.register_service_class(
                SessionManager, 
                dependencies=None,
                group=self.SERVICE_GROUP_NETWORK
            )
            
            self.register_service_class(
                RateLimiter, 
                dependencies={"error_classifier"},
                group=self.SERVICE_GROUP_NETWORK
            )
            
            self.register_service_class(
                ProxyManager, 
                dependencies={"session_manager"},
                group=self.SERVICE_GROUP_NETWORK
            )
            
            logger.info("Registered resource management and error handling services")
            
        except ImportError as e:
            logger.error(f"Failed to register resource services: {str(e)}")
            logger.error("Make sure all required resource management services are implemented")