"""
Tests for the CircuitBreaker implementation.

This module tests the core functionality of the circuit breaker pattern,
including state transitions, success/failure counting, half-open testing,
timeout handling, and decorator behavior.
"""

import unittest
import time
import asyncio
from unittest.mock import patch, MagicMock

from core.circuit_breaker import (
    CircuitState,
    CircuitBreaker,
    CircuitBreakerManager,
    CircuitOpenError,
    TimeoutError,
    circuit_protected,
    circuit_protected_async,
    fallback_protected,
    fallback_protected_async,
    timeout_protected
)


class TestCircuitBreaker(unittest.TestCase):
    """Test case for the CircuitBreaker class."""
    
    def setUp(self):
        """Set up a fresh circuit breaker for each test."""
        self.circuit_breaker = CircuitBreaker(
            name="test_circuit",
            failure_threshold=3,
            reset_timeout=1,
            half_open_max_calls=2
        )
    
    def test_initial_state(self):
        """Test that the circuit breaker starts in the CLOSED state."""
        self.assertEqual(self.circuit_breaker.state, CircuitState.CLOSED)
        self.assertTrue(self.circuit_breaker.allow_request())
        
        # Check metrics
        metrics = self.circuit_breaker.get_metrics()
        self.assertEqual(metrics["state"], "closed")
        self.assertEqual(metrics["failure_count"], 0)
        self.assertEqual(metrics["success_count"], 0)
    
    def test_success_counting(self):
        """Test that success operations are counted correctly."""
        for i in range(5):
            self.circuit_breaker.success()
        
        metrics = self.circuit_breaker.get_metrics()
        self.assertEqual(metrics["success_count"], 5)
        self.assertEqual(self.circuit_breaker.state, CircuitState.CLOSED)
    
    def test_failure_counting(self):
        """Test that failures are counted correctly."""
        for i in range(2):  # Not enough to trip circuit
            self.circuit_breaker.failure()
        
        self.assertEqual(self.circuit_breaker.state, CircuitState.CLOSED)
        metrics = self.circuit_breaker.get_metrics()
        self.assertEqual(metrics["failure_count"], 2)
    
    def test_circuit_opens_after_threshold(self):
        """Test that the circuit opens after reaching the failure threshold."""
        # Fail until threshold is reached
        for i in range(3):
            self.circuit_breaker.failure()
        
        # Circuit should now be OPEN
        self.assertEqual(self.circuit_breaker.state, CircuitState.OPEN)
        self.assertFalse(self.circuit_breaker.allow_request())
        
        # Check metrics
        metrics = self.circuit_breaker.get_metrics()
        self.assertEqual(metrics["state"], "open")
        self.assertEqual(metrics["failure_count"], 3)
    
    def test_circuit_resets_after_timeout(self):
        """Test that the circuit transitions to HALF_OPEN after the reset timeout."""
        # Open the circuit
        for i in range(3):
            self.circuit_breaker.failure()
        
        self.assertEqual(self.circuit_breaker.state, CircuitState.OPEN)
        
        # Mock the time to simulate timeout
        with patch.object(time, 'time', return_value=time.time() + 2):
            # Should transition to HALF_OPEN when state is checked
            self.assertEqual(self.circuit_breaker.state, CircuitState.HALF_OPEN)
            self.assertTrue(self.circuit_breaker.allow_request())
    
    def test_circuit_allows_limited_calls_in_half_open(self):
        """Test that the circuit allows a limited number of calls in HALF_OPEN state."""
        # Open the circuit
        for i in range(3):
            self.circuit_breaker.failure()
        
        # Transition to HALF_OPEN
        with patch.object(time, 'time', return_value=time.time() + 2):
            self.assertEqual(self.circuit_breaker.state, CircuitState.HALF_OPEN)
            
            # Should allow half_open_max_calls (2) requests
            self.assertTrue(self.circuit_breaker.allow_request())
            self.assertTrue(self.circuit_breaker.allow_request())
            
            # Third request should be blocked
            self.assertFalse(self.circuit_breaker.allow_request())
    
    def test_circuit_closes_after_success_in_half_open(self):
        """Test that the circuit closes after successful calls in HALF_OPEN state."""
        # Open the circuit
        for i in range(3):
            self.circuit_breaker.failure()
        
        # Transition to HALF_OPEN
        with patch.object(time, 'time', return_value=time.time() + 2):
            self.assertEqual(self.circuit_breaker.state, CircuitState.HALF_OPEN)
            
            # Register successful calls to meet half_open_max_calls
            self.circuit_breaker.success()
            self.circuit_breaker.success()
            
            # Circuit should now be CLOSED
            self.assertEqual(self.circuit_breaker.state, CircuitState.CLOSED)
    
    def test_circuit_reopens_on_failure_in_half_open(self):
        """Test that the circuit reopens immediately on failure in HALF_OPEN state."""
        # Open the circuit
        for i in range(3):
            self.circuit_breaker.failure()
        
        # Transition to HALF_OPEN
        with patch.object(time, 'time', return_value=time.time() + 2):
            self.assertEqual(self.circuit_breaker.state, CircuitState.HALF_OPEN)
            
            # Trigger failure in HALF_OPEN
            self.circuit_breaker.failure()
            
            # Circuit should be OPEN again
            self.assertEqual(self.circuit_breaker.state, CircuitState.OPEN)
    
    def test_reset_clears_failure_count(self):
        """Test that reset() clears the failure count and closes the circuit."""
        # Open the circuit
        for i in range(3):
            self.circuit_breaker.failure()
        
        self.assertEqual(self.circuit_breaker.state, CircuitState.OPEN)
        
        # Reset the circuit
        self.circuit_breaker.reset()
        
        # Circuit should be closed and counters reset
        self.assertEqual(self.circuit_breaker.state, CircuitState.CLOSED)
        self.assertTrue(self.circuit_breaker.allow_request())
        
        metrics = self.circuit_breaker.get_metrics()
        self.assertEqual(metrics["failure_count"], 0)
        self.assertEqual(metrics["success_count"], 0)
    
    def test_manual_state_changes(self):
        """Test manual open() and close() operations."""
        # Initially closed
        self.assertEqual(self.circuit_breaker.state, CircuitState.CLOSED)
        
        # Manually open
        self.circuit_breaker.open()
        self.assertEqual(self.circuit_breaker.state, CircuitState.OPEN)
        self.assertFalse(self.circuit_breaker.allow_request())
        
        # Manually close
        self.circuit_breaker.close()
        self.assertEqual(self.circuit_breaker.state, CircuitState.CLOSED)
        self.assertTrue(self.circuit_breaker.allow_request())
    
    def test_reset_failure_count_timeout(self):
        """Test that failure count resets after reset_failure_count_timeout."""
        # Set some failures, but not enough to open
        self.circuit_breaker.failure()
        self.circuit_breaker.failure()
        
        metrics = self.circuit_breaker.get_metrics()
        self.assertEqual(metrics["failure_count"], 2)
        
        # Mock time to be after reset_failure_count_timeout (default 600s)
        with patch.object(time, 'time', return_value=time.time() + 700):
            # Getting state should trigger failure count reset
            self.assertEqual(self.circuit_breaker.state, CircuitState.CLOSED)
            
            metrics = self.circuit_breaker.get_metrics()
            self.assertEqual(metrics["failure_count"], 0)
    
    def test_excluded_exceptions(self):
        """Test that excluded exceptions don't count as failures."""
        # Create a circuit breaker with excluded exceptions
        cb = CircuitBreaker(
            name="test_exclude",
            failure_threshold=2,
            excluded_exceptions={ValueError}
        )
        
        # This should not count as a failure
        try:
            cb.execute(lambda: (_ for _ in ()).throw(ValueError("Expected")))
        except ValueError:
            pass
            
        # This should count as a failure
        try:
            cb.execute(lambda: (_ for _ in ()).throw(TypeError("Expected")))
        except TypeError:
            pass
            
        # Still closed because ValueError didn't count
        self.assertEqual(cb.state, CircuitState.CLOSED)
        
        # Trigger another failure to open the circuit
        try:
            cb.execute(lambda: (_ for _ in ()).throw(TypeError("Expected")))
        except TypeError:
            pass
            
        # Now it should be open
        self.assertEqual(cb.state, CircuitState.OPEN)
    
    def test_event_listeners(self):
        """Test event listeners for state changes, success, and failure."""
        # Mock listeners
        state_listener = MagicMock()
        success_listener = MagicMock()
        failure_listener = MagicMock()
        execution_listener = MagicMock()
        
        # Add listeners
        self.circuit_breaker.add_state_change_listener(state_listener)
        self.circuit_breaker.add_success_listener(success_listener)
        self.circuit_breaker.add_failure_listener(failure_listener)
        self.circuit_breaker.add_execution_listener(execution_listener)
        
        # Test success notification
        self.circuit_breaker.success()
        success_listener.assert_called_once_with("test_circuit")
        execution_listener.assert_not_called()  # Not called directly by success()
        
        # Test failure notification
        error = Exception("Test failure")
        self.circuit_breaker.failure(error)
        failure_listener.assert_called_once_with("test_circuit", error)
        
        # Fail until threshold to trigger state change
        self.circuit_breaker.failure()
        self.circuit_breaker.failure()  # This should open the circuit
        
        # Verify state change notification (CLOSED -> OPEN)
        state_listener.assert_called_once_with(
            "test_circuit", CircuitState.CLOSED, CircuitState.OPEN
        )
        
        # Remove listeners
        self.circuit_breaker.remove_state_change_listener(state_listener)
        self.circuit_breaker.remove_success_listener(success_listener)
        self.circuit_breaker.remove_failure_listener(failure_listener)
        self.circuit_breaker.remove_execution_listener(execution_listener)
        
        # Reset and verify no more calls to removed listeners
        state_listener.reset_mock()
        self.circuit_breaker.reset()
        state_listener.assert_not_called()


class TestCircuitBreakerExecution(unittest.TestCase):
    """Test case for execute methods."""
    
    def setUp(self):
        """Set up a fresh circuit breaker for each test."""
        self.circuit_breaker = CircuitBreaker(
            name="test_execute",
            failure_threshold=2
        )
        
        # Functions for testing
        self.success_func = lambda: "success"
        self.failure_func = lambda: (_ for _ in ()).throw(Exception("Expected"))
        
        # Mock for execution listener
        self.execution_listener = MagicMock()
        self.circuit_breaker.add_execution_listener(self.execution_listener)
    
    def test_execute_success(self):
        """Test successful execution."""
        result = self.circuit_breaker.execute(self.success_func)
        self.assertEqual(result, "success")
        
        # Verify metrics
        metrics = self.circuit_breaker.get_metrics()
        self.assertEqual(metrics["success_count"], 1)
        self.assertEqual(metrics["failure_count"], 0)
        
        # Verify execution listener was called with success
        self.execution_listener.assert_called_once_with("test_execute", True)
    
    def test_execute_failure(self):
        """Test execution that fails."""
        with self.assertRaises(Exception):
            self.circuit_breaker.execute(self.failure_func)
        
        # Verify metrics
        metrics = self.circuit_breaker.get_metrics()
        self.assertEqual(metrics["success_count"], 0)
        self.assertEqual(metrics["failure_count"], 1)
        
        # Verify execution listener was called with failure
        self.execution_listener.assert_called_once_with("test_execute", False)
    
    def test_execute_when_open(self):
        """Test execution is blocked when circuit is open."""
        # Open the circuit
        self.circuit_breaker.failure()
        self.circuit_breaker.failure()
        
        # Execution should be blocked
        with self.assertRaises(CircuitOpenError):
            self.circuit_breaker.execute(self.success_func)
        
        # Verify execution listener was called with failure
        self.execution_listener.assert_called_with("test_execute", False)
    
    def test_execute_with_fallback(self):
        """Test execution with fallback."""
        # Define fallback function
        fallback_func = lambda: "fallback"
        
        # Test successful execution (no fallback used)
        result = self.circuit_breaker.execute_with_fallback(
            self.success_func, fallback_func
        )
        self.assertEqual(result, "success")
        
        # Test fallback used on failure
        result = self.circuit_breaker.execute_with_fallback(
            self.failure_func, fallback_func
        )
        self.assertEqual(result, "fallback")
        
        # Open the circuit
        self.circuit_breaker.failure()  # Second failure opens the circuit
        
        # Test fallback used when circuit open
        result = self.circuit_breaker.execute_with_fallback(
            self.success_func, fallback_func
        )
        self.assertEqual(result, "fallback")


class TestAsyncCircuitBreaker(unittest.IsolatedAsyncioTestCase):
    """Test case for async circuit breaker functions."""
    
    async def asyncSetUp(self):
        """Set up a fresh circuit breaker for each test."""
        self.circuit_breaker = CircuitBreaker(
            name="test_async",
            failure_threshold=2
        )
        
        # Async functions for testing
        self.success_func = self._success_func
        self.failure_func = self._failure_func
        self.slow_func = self._slow_func
    
    async def _success_func(self):
        """Test async function that succeeds."""
        await asyncio.sleep(0.01)
        return "async success"
    
    async def _failure_func(self):
        """Test async function that fails."""
        await asyncio.sleep(0.01)
        raise Exception("Expected async failure")
    
    async def _slow_func(self):
        """Test async function that takes a long time."""
        await asyncio.sleep(0.5)
        return "slow result"
    
    async def test_execute_async_success(self):
        """Test successful async execution."""
        result = await self.circuit_breaker.execute_async(self.success_func)
        self.assertEqual(result, "async success")
        
        metrics = self.circuit_breaker.get_metrics()
        self.assertEqual(metrics["success_count"], 1)
    
    async def test_execute_async_failure(self):
        """Test async execution that fails."""
        with self.assertRaises(Exception):
            await self.circuit_breaker.execute_async(self.failure_func)
        
        metrics = self.circuit_breaker.get_metrics()
        self.assertEqual(metrics["failure_count"], 1)
    
    async def test_execute_with_fallback_async(self):
        """Test async execution with fallback."""
        # Define fallback function
        async def fallback_func():
            return "async fallback"
        
        # Test successful execution (no fallback used)
        result = await self.circuit_breaker.execute_with_fallback_async(
            self.success_func, fallback_func
        )
        self.assertEqual(result, "async success")
        
        # Test fallback used on failure
        result = await self.circuit_breaker.execute_with_fallback_async(
            self.failure_func, fallback_func
        )
        self.assertEqual(result, "async fallback")
    
    async def test_execute_with_timeout(self):
        """Test async execution with timeout."""
        # Should succeed with sufficient timeout
        result = await self.circuit_breaker.execute_with_timeout(
            self.success_func, timeout=1.0
        )
        self.assertEqual(result, "async success")
        
        # Should timeout with short timeout
        with self.assertRaises(TimeoutError):
            await self.circuit_breaker.execute_with_timeout(
                self.slow_func, timeout=0.1
            )
        
        # Circuit should register the timeout as a failure
        metrics = self.circuit_breaker.get_metrics()
        self.assertEqual(metrics["failure_count"], 1)


class TestCircuitBreakerManager(unittest.TestCase):
    """Test case for the CircuitBreakerManager class."""
    
    def setUp(self):
        """Set up a fresh manager for each test."""
        self.manager = CircuitBreakerManager()
    
    def test_get_circuit_breaker(self):
        """Test getting or creating circuit breakers."""
        # Get a new circuit breaker
        cb1 = self.manager.get_circuit_breaker("service1", {"failure_threshold": 10})
        self.assertEqual(cb1.name, "service1")
        self.assertEqual(cb1.failure_threshold, 10)
        
        # Getting again should return the same instance
        cb2 = self.manager.get_circuit_breaker("service1")
        self.assertIs(cb1, cb2)
        
        # Different name should create a new instance
        cb3 = self.manager.get_circuit_breaker("service2")
        self.assertIsNot(cb1, cb3)
    
    def test_register_circuit_breaker(self):
        """Test registering an existing circuit breaker."""
        # Create a circuit breaker directly
        cb = CircuitBreaker(name="external")
        
        # Register it with the manager
        self.manager.register_circuit_breaker("custom", cb)
        
        # Verify it can be retrieved
        retrieved = self.manager.get_circuit_breaker("custom")
        self.assertIs(retrieved, cb)
    
    def test_remove_circuit_breaker(self):
        """Test removing a circuit breaker."""
        # Create a circuit breaker
        self.manager.get_circuit_breaker("temporary")
        
        # Remove it
        self.manager.remove_circuit_breaker("temporary")
        
        # Getting it again should create a new instance
        cb1 = self.manager.get_circuit_breaker("temporary")
        cb2 = self.manager.get_circuit_breaker("temporary")
        self.assertIs(cb1, cb2)  # Same instance, but different from the removed one
    
    def test_reset_all(self):
        """Test resetting all circuit breakers."""
        # Create some circuit breakers
        cb1 = self.manager.get_circuit_breaker("reset1")
        cb2 = self.manager.get_circuit_breaker("reset2")
        
        # Trigger some failures
        cb1.failure()
        cb2.failure()
        cb2.failure()
        
        # Verify failure counts
        self.assertEqual(cb1.get_metrics()["failure_count"], 1)
        self.assertEqual(cb2.get_metrics()["failure_count"], 2)
        
        # Reset all
        self.manager.reset_all()
        
        # Verify counts are reset
        self.assertEqual(cb1.get_metrics()["failure_count"], 0)
        self.assertEqual(cb2.get_metrics()["failure_count"], 0)
    
    def test_get_all_metrics(self):
        """Test getting metrics for all circuit breakers."""
        # Create some circuit breakers
        self.manager.get_circuit_breaker("metrics1")
        self.manager.get_circuit_breaker("metrics2")
        
        # Get all metrics
        metrics = self.manager.get_all_metrics()
        
        # Verify metrics structure
        self.assertIn("metrics1", metrics)
        self.assertIn("metrics2", metrics)
        self.assertEqual(metrics["metrics1"]["state"], "closed")
        self.assertEqual(metrics["metrics2"]["state"], "closed")
    
    def test_get_unhealthy_circuits(self):
        """Test getting only unhealthy circuit breakers."""
        # Create some circuit breakers
        cb1 = self.manager.get_circuit_breaker("healthy", {"failure_threshold": 3})
        cb2 = self.manager.get_circuit_breaker("unhealthy", {"failure_threshold": 1})
        
        # Make one unhealthy
        cb2.failure()  # This should open the circuit
        
        # Get unhealthy circuits
        unhealthy = self.manager.get_unhealthy_circuits()
        
        # Verify only the unhealthy one is returned
        self.assertNotIn("healthy", unhealthy)
        self.assertIn("unhealthy", unhealthy)
    
    def test_initialize_with_config(self):
        """Test initialization with configuration."""
        # Create a manager with configuration
        config = {
            "default_settings": {
                "failure_threshold": 7,
                "reset_timeout": 30
            },
            "circuit_breakers": {
                "preconfigured": {
                    "failure_threshold": 5
                }
            }
        }
        
        manager = CircuitBreakerManager()
        manager.initialize(config)
        
        # Check default settings are applied
        cb1 = manager.get_circuit_breaker("default_test")
        self.assertEqual(cb1.failure_threshold, 7)
        self.assertEqual(cb1.reset_timeout, 30)
        
        # Check preconfigured circuit was created with its settings
        cb2 = manager.get_circuit_breaker("preconfigured")
        self.assertEqual(cb2.failure_threshold, 5)  # From specific config
        self.assertEqual(cb2.reset_timeout, 30)     # From default settings


class TestCircuitBreakerDecorators(unittest.TestCase):
    """Test case for circuit breaker decorators."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a manager for testing
        self.manager = CircuitBreakerManager()
    
    def test_circuit_protected_decorator(self):
        """Test the basic circuit_protected decorator."""
        # Define a test function with the decorator
        @circuit_protected("decorator_test", manager=self.manager)
        def test_func():
            return "protected result"
        
        # Call the function
        result = test_func()
        self.assertEqual(result, "protected result")
        
        # Verify the circuit was created
        cb = self.manager.get_circuit_breaker("decorator_test")
        self.assertEqual(cb.get_metrics()["success_count"], 1)
        
        # Test with a failing function
        @circuit_protected("fail_test", manager=self.manager)
        def fail_func():
            raise ValueError("Expected failure")
        
        # Call it twice to record failures
        for _ in range(2):
            with self.assertRaises(ValueError):
                fail_func()
        
        # Circuit should be open now
        cb = self.manager.get_circuit_breaker("fail_test")
        self.assertEqual(cb.state, CircuitState.OPEN)
        
        # Third call should raise CircuitOpenError
        with self.assertRaises(CircuitOpenError):
            fail_func()
    
    def test_fallback_protected_decorator(self):
        """Test the fallback_protected decorator."""
        # Define fallback function
        def fallback_func():
            return "fallback result"
        
        # Define a test function with the decorator
        @fallback_protected("fallback_test", fallback_func, manager=self.manager)
        def test_func():
            raise ValueError("Expected failure")
        
        # First call should use fallback
        result = test_func()
        self.assertEqual(result, "fallback result")
        
        # Second call should still work via fallback
        result = test_func()
        self.assertEqual(result, "fallback result")


class TestAsyncCircuitBreakerDecorators(unittest.IsolatedAsyncioTestCase):
    """Test case for async circuit breaker decorators."""
    
    async def asyncSetUp(self):
        """Set up test environment."""
        # Create a manager for testing
        self.manager = CircuitBreakerManager()
    
    async def test_circuit_protected_async_decorator(self):
        """Test the circuit_protected_async decorator."""
        # Define a test function with the decorator
        @circuit_protected_async("async_test", manager=self.manager)
        async def test_func():
            await asyncio.sleep(0.01)
            return "async result"
        
        # Call the function
        result = await test_func()
        self.assertEqual(result, "async result")
        
        # Verify the circuit was created
        cb = self.manager.get_circuit_breaker("async_test")
        self.assertEqual(cb.get_metrics()["success_count"], 1)
    
    async def test_fallback_protected_async_decorator(self):
        """Test the fallback_protected_async decorator."""
        # Define fallback function
        async def fallback_func():
            return "async fallback"
        
        # Define a test function with the decorator
        @fallback_protected_async("async_fallback", fallback_func, manager=self.manager)
        async def test_func():
            await asyncio.sleep(0.01)
            raise ValueError("Expected async failure")
        
        # Call should use fallback
        result = await test_func()
        self.assertEqual(result, "async fallback")
    
    async def test_timeout_protected_decorator(self):
        """Test the timeout_protected decorator."""
        # Define a slow function
        @timeout_protected(0.1, "timeout_test", manager=self.manager)
        async def slow_func():
            await asyncio.sleep(0.5)  # Too slow
            return "never reached"
        
        # Call should time out
        with self.assertRaises(TimeoutError):
            await slow_func()
        
        # Define a fast function
        @timeout_protected(0.5, "timeout_fast", manager=self.manager)
        async def fast_func():
            await asyncio.sleep(0.01)  # Fast enough
            return "fast result"
        
        # Call should succeed
        result = await fast_func()
        self.assertEqual(result, "fast result")


if __name__ == "__main__":
    unittest.main()