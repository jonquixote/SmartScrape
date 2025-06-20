import pytest
from core.service_interface import BaseService
from core.service_registry import ServiceRegistry

class MockService(BaseService):
    def __init__(self):
        self._initialized = False
        self._config = None
    
    def initialize(self, config=None):
        self._initialized = True
        self._config = config
    
    def shutdown(self):
        self._initialized = False
    
    @property
    def name(self):
        return "mock_service"

class DependentService(BaseService):
    def __init__(self):
        self._initialized = False
        self._config = None
    
    def initialize(self, config=None):
        self._initialized = True
        self._config = config
    
    def shutdown(self):
        self._initialized = False
    
    @property
    def name(self):
        return "dependent_service"

class CircularService1(BaseService):
    def initialize(self, config=None): pass
    def shutdown(self): pass
    @property
    def name(self): return "circular1"

class CircularService2(BaseService):
    def initialize(self, config=None): pass
    def shutdown(self): pass
    @property
    def name(self): return "circular2"

def test_service_registry_singleton():
    """Test that ServiceRegistry is a singleton."""
    registry1 = ServiceRegistry()
    registry2 = ServiceRegistry()
    assert registry1 is registry2

def test_register_service():
    """Test registering a service class."""
    registry = ServiceRegistry()
    registry.register_service_class(MockService)
    assert "mock_service" in registry.get_all_service_names()

def test_get_service():
    """Test getting a service instance."""
    registry = ServiceRegistry()
    registry.register_service_class(MockService)
    service = registry.get_service("mock_service")
    assert service.is_initialized
    assert isinstance(service, MockService)

def test_service_dependencies():
    """Test service dependency resolution."""
    registry = ServiceRegistry()
    registry.register_service_class(MockService)
    registry.register_service_class(DependentService, dependencies={"mock_service"})
    
    # Getting dependent service should initialize mock_service first
    service = registry.get_service("dependent_service")
    assert registry.is_service_initialized("mock_service")
    assert registry.is_service_initialized("dependent_service")

def test_circular_dependency_detection():
    """Test detection of circular dependencies."""
    registry = ServiceRegistry()
    registry.register_service_class(CircularService1, dependencies={"circular2"})
    registry.register_service_class(CircularService2, dependencies={"circular1"})
    
    with pytest.raises(ValueError, match="Circular dependency detected"):
        registry.get_service("circular1")

def test_shutdown_all():
    """Test shutting down all services."""
    registry = ServiceRegistry()
    registry.register_service_class(MockService)
    registry.register_service_class(DependentService, dependencies={"mock_service"})
    
    # Initialize both services
    registry.get_service("dependent_service")
    
    # Shutdown all
    registry.shutdown_all()
    
    # Services should no longer be initialized
    assert not registry.is_service_initialized("mock_service")
    assert not registry.is_service_initialized("dependent_service")

def test_initialize_with_config():
    """Test initializing a service with configuration."""
    registry = ServiceRegistry()
    registry.register_service_class(MockService)
    
    config = {"test": "value"}
    service = registry.get_service("mock_service", config)
    
    assert service._config == config

def test_type_checking():
    """Test that only BaseService subclasses can be registered."""
    class NotAService:
        pass
    
    registry = ServiceRegistry()
    with pytest.raises(TypeError):
        registry.register_service_class(NotAService)

def test_nonexistent_service():
    """Test error handling when requesting a non-existent service."""
    registry = ServiceRegistry()
    with pytest.raises(KeyError):
        registry.get_service("nonexistent_service")

def test_complex_dependency_chain():
    """Test a more complex dependency chain."""
    class Service1(BaseService):
        def initialize(self, config=None): self._initialized = True
        def shutdown(self): self._initialized = False
        @property
        def name(self): return "service1"
    
    class Service2(BaseService):
        def initialize(self, config=None): self._initialized = True
        def shutdown(self): self._initialized = False
        @property
        def name(self): return "service2"
    
    class Service3(BaseService):
        def initialize(self, config=None): self._initialized = True
        def shutdown(self): self._initialized = False
        @property
        def name(self): return "service3"
    
    registry = ServiceRegistry()
    registry.register_service_class(Service1)
    registry.register_service_class(Service2, dependencies={"service1"})
    registry.register_service_class(Service3, dependencies={"service2"})
    
    # Getting service3 should initialize all dependencies in order
    service3 = registry.get_service("service3")
    assert registry.is_service_initialized("service1")
    assert registry.is_service_initialized("service2")
    assert registry.is_service_initialized("service3")