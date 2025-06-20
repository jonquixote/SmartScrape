"""
Test for circuit breaker functionality.

This test verifies that the circuit breaker properly detects failures
and prevents cascading failures by blocking execution when failure thresholds are met.
"""

import asyncio
import unittest
import time
from unittest.mock import MagicMock, patch

from core.pipeline.circuit_breaker import (
    CircuitBreaker, 
    CircuitState, 
    CircuitBreakerRegistry,
    CircuitBreakerStage
)
from core.pipeline.stage import PipelineStage
from core.pipeline.context import PipelineContext


class SimpleStage(PipelineStage):
    """Simple stage for testing."""
    
    def __init__(self, name, should_succeed=True):
        """Initialize stage."""
        super().__init__()
        self.name = name
        self.should_succeed = should_succeed
        self.process_called = False
        
    async def process(self, context):
        """Process this stage."""
        self.process_called = True
        context.set(f"{self.name}_called", True)
        return self.should_succeed
        
    async def validate_input(self, context):
        """Validate input."""
        return True
        
    async def validate_output(self, context):
        """Validate output."""
        return True
        
    async def handle_error(self, context, error):
        """Handle error."""
        context.add_error(self.name, str(error))
        return False


class CircuitBreakerTest(unittest.TestCase):
    """Test suite for circuit breaker functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.circuit_breaker = CircuitBreaker(
            name="test_circuit",
            failure_threshold=3,
            reset_timeout=1,
            half_open_max_calls=2
        )
        self.context = PipelineContext()
        
    def test_initial_state(self):
        """Test initial state of circuit breaker."""
        # Circuit should start in CLOSED state
        self.assertEqual(self.circuit_breaker.state, CircuitState.CLOSED)
        self.assertTrue(self.circuit_breaker.allow_request())
        
        # Check metrics
        metrics = self.circuit_breaker.get_metrics()
        self.assertEqual(metrics["state"], "closed")
        self.assertEqual(metrics["failure_count"], 0)
        self.assertEqual(metrics["success_count"], 0)
    
    def test_circuit_opens_after_failures(self):
        """Test that circuit opens after reaching failure threshold."""
        # Register failures
        for i in range(3):
            self.circuit_breaker.on_failure()
            
        # Circuit should now be OPEN
        self.assertEqual(self.circuit_breaker.state, CircuitState.OPEN)
        self.assertFalse(self.circuit_breaker.allow_request())
        
        # Check metrics
        metrics = self.circuit_breaker.get_metrics()
        self.assertEqual(metrics["state"], "open")
        self.assertEqual(metrics["failure_count"], 3)
    
    def test_circuit_transitions_to_half_open(self):
        """Test transition to HALF_OPEN state after timeout."""
        # Open the circuit
        for i in range(3):
            self.circuit_breaker.on_failure()
            
        self.assertEqual(self.circuit_breaker.state, CircuitState.OPEN)
        
        # Fast-forward time by mocking the timeout check
        with patch.object(self.circuit_breaker, '_last_failure_time', time.time() - 2):
            # Should transition to HALF_OPEN when checked
            self.assertEqual(self.circuit_breaker.state, CircuitState.HALF_OPEN)
            self.assertTrue(self.circuit_breaker.allow_request())
    
    def test_circuit_closes_after_success_in_half_open(self):
        """Test circuit closes after successful calls in HALF_OPEN state."""
        # Open the circuit
        for i in range(3):
            self.circuit_breaker.on_failure()
            
        # Transition to HALF_OPEN
        with patch.object(self.circuit_breaker, '_last_failure_time', time.time() - 2):
            self.assertEqual(self.circuit_breaker.state, CircuitState.HALF_OPEN)
            
            # Register successful calls
            self.circuit_breaker.on_success()
            self.circuit_breaker.on_success()
            
            # Circuit should close after enough successes
            self.assertEqual(self.circuit_breaker.state, CircuitState.CLOSED)
    
    def test_circuit_reopens_on_failure_in_half_open(self):
        """Test circuit reopens on failure in HALF_OPEN state."""
        # Open the circuit
        for i in range(3):
            self.circuit_breaker.on_failure()
            
        # Transition to HALF_OPEN
        with patch.object(self.circuit_breaker, '_last_failure_time', time.time() - 2):
            self.assertEqual(self.circuit_breaker.state, CircuitState.HALF_OPEN)
            
            # Register a failure
            self.circuit_breaker.on_failure()
            
            # Circuit should reopen
            self.assertEqual(self.circuit_breaker.state, CircuitState.OPEN)
    
    def test_circuit_breaker_stage(self):
        """Test CircuitBreakerStage wrapper."""
        # Create wrapped stage and circuit breaker stage
        wrapped_stage = SimpleStage("test_stage", should_succeed=True)
        cb_stage = CircuitBreakerStage(
            wrapped_stage=wrapped_stage,
            failure_threshold=2,
            reset_timeout=1
        )
        
        # Process should succeed and be passed through
        async def run_stage():
            return await cb_stage.process(self.context)
            
        success = asyncio.run(run_stage())
        self.assertTrue(success)
        self.assertTrue(wrapped_stage.process_called)
        self.assertTrue(self.context.get("test_stage_called"))
    
    def test_circuit_breaker_stage_prevents_execution(self):
        """Test CircuitBreakerStage prevents execution when circuit is open."""
        # Create wrapped stage and circuit breaker stage
        wrapped_stage = SimpleStage("test_stage", should_succeed=False)
        cb_stage = CircuitBreakerStage(
            wrapped_stage=wrapped_stage,
            failure_threshold=2,
            reset_timeout=1
        )
        
        # Fail twice to open circuit
        async def run_stage():
            return await cb_stage.process(self.context)
            
        asyncio.run(run_stage())  # First failure
        wrapped_stage.process_called = False  # Reset for next call
        self.context.set("test_stage_called", False)
        
        asyncio.run(run_stage())  # Second failure
        wrapped_stage.process_called = False  # Reset for next call
        self.context.set("test_stage_called", False)
        
        # Third attempt should be blocked
        success = asyncio.run(run_stage())
        self.assertFalse(success)  # Should fail because circuit is open
        self.assertFalse(wrapped_stage.process_called)  # Stage shouldn't be called
        self.assertFalse(self.context.get("test_stage_called", False))
        
        # Should have an error about circuit being open
        errors = self.context.metadata["errors"]
        self.assertIn("circuit_breaker_test_stage", errors)
    
    def test_circuit_breaker_registry(self):
        """Test the circuit breaker registry."""
        registry = CircuitBreakerRegistry()
        
        # Get a new circuit breaker
        cb1 = registry.get_or_create("service1", failure_threshold=5)
        self.assertEqual(cb1.name, "service1")
        self.assertEqual(cb1.failure_threshold, 5)
        
        # Should get same instance when requesting again
        cb2 = registry.get_or_create("service1")
        self.assertIs(cb1, cb2)
        
        # Get with different name should create new instance
        cb3 = registry.get_or_create("service2")
        self.assertIsNot(cb1, cb3)
        
        # Test reset all
        cb1.on_failure()
        cb3.on_failure()
        self.assertEqual(cb1.get_metrics()["failure_count"], 1)
        self.assertEqual(cb3.get_metrics()["failure_count"], 1)
        
        registry.reset_all()
        
        self.assertEqual(cb1.get_metrics()["failure_count"], 0)
        self.assertEqual(cb3.get_metrics()["failure_count"], 0)


if __name__ == "__main__":
    unittest.main()