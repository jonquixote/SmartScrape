"""Tests for the core monitoring system."""

import pytest
import time
import threading
from unittest.mock import MagicMock, patch

from core.monitoring import Monitoring, HealthCheck, HealthStatus
from core.service_interface import BaseService
from core.service_registry import ServiceRegistry

# Mock service for testing
class MockService(BaseService):
    def __init__(self, name, health_status="healthy"):
        self._initialized = True
        self._name = name
        self._health_status = health_status
    
    def initialize(self, config=None):
        self._initialized = True
    
    def shutdown(self):
        self._initialized = False
    
    @property
    def name(self):
        return self._name
    
    def get_service_health(self):
        return {
            "status": self._health_status,
            "details": f"Mock service {self._name} is {self._health_status}",
            "metrics": {"test_metric": 42}
        }

# Mock health check for testing
class MockHealthCheck(HealthCheck):
    def __init__(self, name, status=HealthStatus.HEALTHY, metrics=None):
        super().__init__(name)
        self.status = status
        self.metrics = metrics or {"test_metric": 100}
    
    def check(self):
        return {
            "status": self.status.value,
            "details": f"Mock health check returned {self.status.value}",
            "metrics": self.metrics
        }

# Fixture for a monitoring service with mocks
@pytest.fixture
def monitoring_service():
    # Create a monitoring service
    monitoring = Monitoring()
    
    # Mock the service registry
    mock_registry = MagicMock(spec=ServiceRegistry)
    mock_registry.get_system_health.return_value = {
        "status": "healthy",
        "details": "All services healthy",
        "initialized_services": 5,
        "registered_services": 5,
        "services": {
            "session_manager": {"status": "healthy", "details": "Session manager is healthy", "metrics": {}},
            "proxy_manager": {"status": "healthy", "details": "Proxy manager is healthy", "metrics": {}}
        }
    }
    
    # Set the mocked registry
    monitoring._registry = mock_registry
    
    # Register some mock health checks
    monitoring._health_checks = {
        "test_healthy": MockHealthCheck("test_healthy", HealthStatus.HEALTHY),
        "test_degraded": MockHealthCheck("test_degraded", HealthStatus.DEGRADED),
        "test_unhealthy": MockHealthCheck("test_unhealthy", HealthStatus.UNHEALTHY)
    }
    
    # Initialize with test config
    monitoring.initialize({
        "monitoring_interval": 0.1,  # Fast interval for testing
        "auto_start": False  # Don't auto-start monitoring thread
    })
    
    yield monitoring
    
    # Clean up
    monitoring.shutdown()

def test_monitoring_initialization():
    """Test that the monitoring service initializes correctly."""
    monitoring = Monitoring()
    
    # Should not be initialized initially
    assert not monitoring.is_initialized
    
    # Initialize and check
    monitoring.initialize()
    assert monitoring.is_initialized
    assert monitoring.name == "monitoring"
    
    # Should have default configuration
    assert monitoring._monitoring_interval == 60
    assert len(monitoring._exporters) > 0
    
    # Cleanup
    monitoring.shutdown()
    assert not monitoring.is_initialized

def test_run_health_check(monitoring_service):
    """Test running health checks."""
    # Run health checks
    health_data = monitoring_service.run_health_check()
    
    # Verify structure
    assert "timestamp" in health_data
    assert "status" in health_data
    assert "components" in health_data
    assert "metrics" in health_data
    
    # Verify components
    assert "test_healthy" in health_data["components"]
    assert "test_degraded" in health_data["components"]
    assert "test_unhealthy" in health_data["components"]
    
    # Verify overall status is unhealthy (because one check is unhealthy)
    assert health_data["status"] == "unhealthy"
    
    # Verify metrics collection
    assert "test_healthy" in health_data["metrics"]
    assert health_data["metrics"]["test_healthy"]["test_metric"] == 100

def test_collect_metrics(monitoring_service):
    """Test collecting metrics from services."""
    # Mock the _get_services method to return some services
    mock_services = {
        "service1": MockService("service1"),
        "service2": MockService("service2")
    }
    monitoring_service._get_services = lambda: mock_services
    
    # Collect metrics
    metrics = monitoring_service.collect_metrics()
    
    # Verify structure
    assert "timestamp" in metrics
    assert "service1" in metrics
    assert "service2" in metrics
    
    # Verify metrics content
    assert metrics["service1"]["test_metric"] == 42
    assert metrics["service2"]["test_metric"] == 42

def test_monitoring_thread(monitoring_service):
    """Test the monitoring thread behavior."""
    # Mock the monitoring functions to track calls
    monitoring_service.run_health_check = MagicMock(return_value={"status": "healthy"})
    monitoring_service.export_metrics = MagicMock()
    
    # Start monitoring thread
    monitoring_service.start_monitoring()
    
    # Wait a bit to allow thread to run
    time.sleep(0.3)
    
    # Stop monitoring thread
    monitoring_service.stop_monitoring()
    
    # Verify that functions were called
    assert monitoring_service.run_health_check.call_count >= 1
    assert monitoring_service.export_metrics.call_count >= 1

def test_health_status_calculation(monitoring_service):
    """Test calculation of overall health status."""
    # Test with all healthy
    components = {
        "comp1": {"status": "healthy"},
        "comp2": {"status": "healthy"}
    }
    assert monitoring_service._calculate_overall_status(components) == "healthy"
    
    # Test with one degraded
    components["comp3"] = {"status": "degraded"}
    assert monitoring_service._calculate_overall_status(components) == "degraded"
    
    # Test with one unhealthy
    components["comp4"] = {"status": "unhealthy"}
    assert monitoring_service._calculate_overall_status(components) == "unhealthy"
    
    # Test with unknown only
    components = {
        "comp1": {"status": "unknown"},
        "comp2": {"status": "unknown"}
    }
    assert monitoring_service._calculate_overall_status(components) == "unknown"

def test_register_health_check(monitoring_service):
    """Test registering a custom health check."""
    # Create a new health check
    custom_check = MockHealthCheck("custom_check")
    
    # Register it
    monitoring_service.register_health_check("custom_check", custom_check)
    
    # Verify it was registered
    assert "custom_check" in monitoring_service._health_checks
    
    # Run health checks to make sure it's included
    health_data = monitoring_service.run_health_check()
    assert "custom_check" in health_data["components"]

def test_get_health_history(monitoring_service):
    """Test retrieving health history."""
    # Store some mock health data
    monitoring_service._metrics_history = {
        time.time() - 200: {"timestamp": "2025-05-06T10:00:00", "status": "healthy", "components": {"comp1": {"status": "healthy"}}},
        time.time() - 100: {"timestamp": "2025-05-06T10:01:00", "status": "degraded", "components": {"comp1": {"status": "degraded"}}},
        time.time(): {"timestamp": "2025-05-06T10:02:00", "status": "unhealthy", "components": {"comp1": {"status": "unhealthy"}}}
    }
    
    # Get all history
    history = monitoring_service.get_health_history()
    assert len(history) == 3
    
    # Get recent history
    recent = monitoring_service.get_health_history(period=150)
    assert len(recent) == 2
    
    # Get history for specific component
    comp_history = monitoring_service.get_health_history(component="comp1", period=300)
    assert len(comp_history) == 3
    assert "component" in comp_history[0]

def test_reset_service(monitoring_service):
    """Test resetting a service."""
    # Create a mock service registry and service to reset
    registry = MagicMock(spec=ServiceRegistry)
    mock_service = MagicMock(spec=BaseService)
    registry.get_service.return_value = mock_service
    
    # Mock the ServiceRegistry instantiation
    with patch('core.monitoring.ServiceRegistry', return_value=registry):
        # Test resetting a regular service
        result = monitoring_service.reset_service("test_service")
        assert result["service"] == "test_service"
        assert mock_service.shutdown.call_count == 1
        assert mock_service.initialize.call_count == 1
        
        # Test resetting circuit breaker manager
        circuit_breaker_service = MagicMock(spec=BaseService)
        circuit_breaker_service._circuit_breakers = {
            "circuit1": MagicMock(reset=MagicMock()),
            "circuit2": MagicMock(reset=MagicMock())
        }
        registry.get_service.return_value = circuit_breaker_service
        
        result = monitoring_service.reset_service("circuit_breaker_manager")
        assert result["service"] == "circuit_breaker_manager"
        assert result["success"] is True
        for circuit in circuit_breaker_service._circuit_breakers.values():
            assert circuit.reset.call_count == 1
            
        # Test resetting proxy manager
        proxy_manager = MagicMock(spec=BaseService)
        proxy_manager._check_all_proxies = MagicMock()
        registry.get_service.return_value = proxy_manager
        
        result = monitoring_service.reset_service("proxy_manager")
        assert result["service"] == "proxy_manager"
        assert result["success"] is True
        assert proxy_manager._check_all_proxies.call_count == 1