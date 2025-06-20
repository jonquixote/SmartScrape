"""
Tests for the StrategyContext class.

These tests verify the functionality of the StrategyContext class, which
provides a shared execution context for strategies.
"""

import pytest
from unittest.mock import MagicMock, patch
import logging
from typing import Dict, Any, Optional

from strategies.core.strategy_context import StrategyContext
from core.service_interface import BaseService


# Mock services for testing
class MockService(BaseService):
    """Mock service for testing."""
    
    def __init__(self):
        self._initialized = False
        self._config = None
        
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._initialized = True
        self._config = config
        
    def shutdown(self) -> None:
        self._initialized = False
        
    @property
    def name(self) -> str:
        return "mock_service"


class MockResource:
    """Mock resource with close method."""
    
    def __init__(self):
        self.closed = False
        
    def close(self):
        self.closed = True


class MockResourceWithShutdown:
    """Mock resource with shutdown method."""
    
    def __init__(self):
        self.shutdown_called = False
        
    def shutdown(self):
        self.shutdown_called = True


class MockResourceWithCleanup:
    """Mock resource with cleanup method."""
    
    def __init__(self):
        self.cleanup_called = False
        
    def cleanup(self):
        self.cleanup_called = True


# Basic initialization tests
def test_strategy_context_init():
    """Test basic initialization of StrategyContext."""
    # Test with no config
    context = StrategyContext()
    assert context.config == {}
    assert hasattr(context, 'service_registry')
    assert hasattr(context, 'logger')
    
    # Test with config
    test_config = {"key": "value"}
    context = StrategyContext(test_config)
    assert context.config == test_config


# Service registry tests
def test_get_service():
    """Test getting a service from the registry."""
    with patch('strategies.core.strategy_context.ServiceRegistry') as mock_registry_class:
        mock_registry = MagicMock()
        mock_registry_class.return_value = mock_registry
        mock_service = MagicMock()
        mock_registry.get_service.return_value = mock_service
        
        context = StrategyContext()
        service = context.get_service("test_service")
        
        mock_registry.get_service.assert_called_once_with("test_service")
        assert service == mock_service


def test_register_service():
    """Test registering a service directly."""
    with patch('strategies.core.strategy_context.ServiceRegistry') as mock_registry_class:
        mock_registry = MagicMock()
        mock_registry_class.return_value = mock_registry
        mock_service = MagicMock()
        
        context = StrategyContext()
        context.register_service("test_service", mock_service)
        
        mock_registry.register_service.assert_called_once_with("test_service", mock_service)


# Configuration tests
def test_get_config():
    """Test getting configuration values with defaults."""
    config = {
        "simple_key": "simple_value",
        "nested": {
            "inner_key": "inner_value",
            "deeper": {
                "deepest_key": "deepest_value"
            }
        }
    }
    
    context = StrategyContext(config)
    
    # Test simple key
    assert context.get_config("simple_key") == "simple_value"
    
    # Test nested key with dot notation
    assert context.get_config("nested.inner_key") == "inner_value"
    assert context.get_config("nested.deeper.deepest_key") == "deepest_value"
    
    # Test default value for non-existent key
    assert context.get_config("non_existent", "default") == "default"
    assert context.get_config("nested.non_existent", "default") == "default"
    
    # Test accessing a non-dict with dot notation
    assert context.get_config("simple_key.non_existent", "default") == "default"


# Sub-context tests
def test_create_sub_context():
    """Test creating a sub-context with merged configuration."""
    parent_config = {
        "key1": "parent_value",
        "shared": {
            "sub_key1": "parent_sub_value",
            "sub_key2": "parent_sub_value2"
        }
    }
    
    child_config = {
        "key2": "child_value",
        "shared": {
            "sub_key1": "child_override",
            "sub_key3": "child_sub_value"
        }
    }
    
    parent_context = StrategyContext(parent_config)
    child_context = parent_context.create_sub_context(child_config)
    
    # Verify the child has its own config with merged values
    assert child_context.get_config("key1") == "parent_value"  # Inherited from parent
    assert child_context.get_config("key2") == "child_value"   # Added in child
    
    # Check nested merging
    assert child_context.get_config("shared.sub_key1") == "child_override"  # Overridden in child
    assert child_context.get_config("shared.sub_key2") == "parent_sub_value2"  # Inherited from parent
    assert child_context.get_config("shared.sub_key3") == "child_sub_value"  # Added in child
    
    # Verify service registry is shared
    assert parent_context.service_registry is child_context.service_registry


def test_with_services():
    """Test creating a context with specific services initialized."""
    # Create a context with a mock service registry
    with patch('strategies.core.strategy_context.ServiceRegistry') as mock_registry_class:
        mock_registry = MagicMock()
        mock_registry_class.return_value = mock_registry
        
        # Configure the mock to return different services
        service1 = MagicMock()
        service2 = MagicMock()
        
        def get_service_side_effect(name):
            if name == "service1":
                return service1
            elif name == "service2":
                return service2
            else:
                raise KeyError(f"Service {name} not found")
                
        mock_registry.get_service.side_effect = get_service_side_effect
        
        # Call the with_services class method
        config = {"key": "value"}
        context = StrategyContext.with_services("service1", "service2", "non_existent", config=config)
        
        # Verify services were requested
        mock_registry.get_service.assert_any_call("service1")
        mock_registry.get_service.assert_any_call("service2")
        mock_registry.get_service.assert_any_call("non_existent")
        
        # Verify config was set
        assert context.config == config


# State management tests
def test_state_management():
    """Test setting and getting state values."""
    context = StrategyContext()
    
    # Test setting and getting simple values
    context.set_state("key1", "value1")
    assert context.get_state("key1") == "value1"
    
    # Test default for non-existent key
    assert context.get_state("non_existent") is None
    assert context.get_state("non_existent", "default") == "default"
    
    # Test complex objects
    complex_obj = {"a": [1, 2, 3], "b": {"nested": True}}
    context.set_state("complex", complex_obj)
    assert context.get_state("complex") == complex_obj
    
    # Test that states don't leak between contexts
    context2 = StrategyContext()
    assert context2.get_state("key1") is None


# Resource management tests
def test_resource_management():
    """Test registering and cleaning up resources."""
    context = StrategyContext()
    
    # Create and register different types of resources
    resource1 = MockResource()
    resource2 = MockResourceWithShutdown()
    resource3 = MockResourceWithCleanup()
    
    context.register_resource(resource1)
    context.register_resource(resource2)
    context.register_resource(resource3)
    
    # Clean up all resources
    context.cleanup()
    
    # Verify each resource was properly cleaned up
    assert resource1.closed
    assert resource2.shutdown_called
    assert resource3.cleanup_called
    
    # Verify the resources list is cleared
    assert len(context._resources) == 0
    
    # Verify the state is cleared
    context.set_state("key", "value")
    context.cleanup()
    assert context.get_state("key") is None


def test_cleanup_error_handling():
    """Test error handling during resource cleanup."""
    context = StrategyContext()
    
    # Create a resource that raises an exception during cleanup
    failing_resource = MagicMock()
    failing_resource.close.side_effect = Exception("Cleanup failed")
    
    context.register_resource(failing_resource)
    
    # Clean up should not raise exceptions
    context.cleanup()
    
    # Verify the close method was called
    failing_resource.close.assert_called_once()
    
    # Verify resources list is still cleared
    assert len(context._resources) == 0


# Integration test with real service registry
def test_integration_with_service_registry():
    """Integration test with actual ServiceRegistry."""
    from core.service_registry import ServiceRegistry
    
    # Patch the ServiceRegistry's get_service to avoid calling actual services
    original_get_service = ServiceRegistry.get_service
    
    def mock_get_service(self, service_name, config=None):
        # For testing, just return a mock service
        mock_service = MagicMock()
        mock_service.name = service_name
        return mock_service
    
    # Apply the patch temporarily
    ServiceRegistry.get_service = mock_get_service
    
    try:
        # Create a context and try to get a service
        context = StrategyContext()
        service = context.get_service("test_service")
        
        # Verify the service was returned
        assert service.name == "test_service"
        
    finally:
        # Restore the original method
        ServiceRegistry.get_service = original_get_service


# Logger tests
def test_logger_property():
    """Test the logger property returns a logger."""
    context = StrategyContext()
    logger = context.logger
    
    assert isinstance(logger, logging.Logger)
    assert logger.name == "StrategyContext"


# Deep merge tests
def test_deep_merge():
    """Test the deep merge functionality."""
    context = StrategyContext()
    
    target = {
        "key1": "value1",
        "nested": {
            "a": 1,
            "b": 2
        },
        "list": [1, 2, 3]
    }
    
    source = {
        "key2": "value2",
        "nested": {
            "b": 3,
            "c": 4
        },
        "list": [4, 5, 6]  # Lists should be replaced, not merged
    }
    
    # Perform the deep merge
    context._deep_merge(target, source)
    
    # Verify the result
    assert target["key1"] == "value1"  # Unchanged
    assert target["key2"] == "value2"  # Added from source
    assert target["nested"]["a"] == 1  # Unchanged nested
    assert target["nested"]["b"] == 3  # Updated nested
    assert target["nested"]["c"] == 4  # Added nested
    assert target["list"] == [4, 5, 6]  # Replaced, not merged