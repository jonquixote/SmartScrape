"""
Test Protection Stages.

This module tests the circuit breaker and bulkhead protection stages.
"""

import asyncio
import unittest
import time
from unittest.mock import Mock, AsyncMock, patch

from core.pipeline.context import PipelineContext
from core.pipeline.stage import PipelineStage
from core.pipeline.stages.protection import (
    CircuitBreakerStage,
    BulkheadStage,
    with_circuit_breaker,
    with_bulkhead,
    with_combined_protection
)


class MockStage(PipelineStage):
    """Mock stage for testing protection patterns."""
    
    def __init__(self, name="mock_stage", fail=False, delay=0, exception=False):
        super().__init__(name)
        self.fail = fail
        self.delay = delay
        self.exception = exception
        self.process_called = 0
        self.initialized = False
        self.cleaned_up = False
        self.validate_input_called = 0
        self.validate_output_called = 0
        self.handle_error_called = 0
    
    async def process(self, context):
        self.process_called += 1
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        if self.exception:
            raise ValueError("Test exception")
        return not self.fail
    
    async def initialize(self):
        self.initialized = True
    
    async def cleanup(self):
        self.cleaned_up = True
    
    async def validate_input(self, context):
        self.validate_input_called += 1
        return True
    
    async def validate_output(self, context):
        self.validate_output_called += 1
        return True
    
    async def handle_error(self, context, error):
        self.handle_error_called += 1
        return False


class TestCircuitBreakerStage(unittest.IsolatedAsyncioTestCase):
    """Test case for the CircuitBreakerStage."""
    
    async def test_process_success(self):
        """Test that successful processing passes through."""
        mock_stage = MockStage()
        circuit_breaker = CircuitBreakerStage(
            wrapped_stage=mock_stage,
            config={"failure_threshold": 3, "use_registry": False}
        )
        
        context = PipelineContext()
        result = await circuit_breaker.process(context)
        
        self.assertTrue(result)
        self.assertEqual(mock_stage.process_called, 1)
        self.assertEqual(circuit_breaker.circuit_breaker.failure_count, 0)
    
    async def test_process_failure(self):
        """Test that unsuccessful processing is recorded."""
        mock_stage = MockStage(fail=True)
        circuit_breaker = CircuitBreakerStage(
            wrapped_stage=mock_stage,
            config={"failure_threshold": 3, "use_registry": False}
        )
        
        context = PipelineContext()
        result = await circuit_breaker.process(context)
        
        self.assertFalse(result)
        self.assertEqual(mock_stage.process_called, 1)
        self.assertEqual(circuit_breaker.circuit_breaker.failure_count, 1)
    
    async def test_circuit_opens_after_failures(self):
        """Test that circuit opens after threshold failures."""
        mock_stage = MockStage(fail=True)
        circuit_breaker = CircuitBreakerStage(
            wrapped_stage=mock_stage,
            config={"failure_threshold": 3, "use_registry": False}
        )
        
        context = PipelineContext()
        
        # Fail 3 times to reach threshold
        for _ in range(3):
            await circuit_breaker.process(context)
        
        # Circuit should be open now
        self.assertFalse(circuit_breaker.circuit_breaker.allow_request())
        self.assertEqual(mock_stage.process_called, 3)
        
        # Process should not be called when circuit is open
        result = await circuit_breaker.process(context)
        self.assertFalse(result)
        self.assertEqual(mock_stage.process_called, 3)  # Still 3, not increased
    
    async def test_circuit_half_open_after_timeout(self):
        """Test that circuit transitions to half-open after timeout."""
        mock_stage = MockStage(fail=True)
        circuit_breaker = CircuitBreakerStage(
            wrapped_stage=mock_stage,
            config={"failure_threshold": 2, "reset_timeout": 0.1, "use_registry": False}
        )
        
        context = PipelineContext()
        
        # Fail 2 times to open circuit
        for _ in range(2):
            await circuit_breaker.process(context)
        
        # Circuit should be open
        self.assertFalse(circuit_breaker.circuit_breaker.allow_request())
        
        # Wait for reset timeout
        await asyncio.sleep(0.2)
        
        # Circuit should be half-open now and allow one request
        self.assertTrue(circuit_breaker.circuit_breaker.allow_request())
    
    async def test_exception_in_wrapped_stage(self):
        """Test handling of exceptions in the wrapped stage."""
        mock_stage = MockStage(exception=True)
        circuit_breaker = CircuitBreakerStage(
            wrapped_stage=mock_stage,
            config={"failure_threshold": 3, "use_registry": False}
        )
        
        context = PipelineContext()
        result = await circuit_breaker.process(context)
        
        self.assertFalse(result)
        self.assertEqual(mock_stage.process_called, 1)
        self.assertEqual(mock_stage.handle_error_called, 1)
        self.assertEqual(circuit_breaker.circuit_breaker.failure_count, 1)
    
    async def test_fallback_when_circuit_open(self):
        """Test fallback behavior when circuit is open."""
        mock_stage = MockStage(fail=True)
        circuit_breaker = CircuitBreakerStage(
            wrapped_stage=mock_stage,
            config={
                "failure_threshold": 2,
                "use_registry": False,
                "fallback_enabled": True,
                "fallback_data": {"fallback_applied": True, "test_value": 42}
            }
        )
        
        context = PipelineContext()
        
        # Fail twice to open circuit
        for _ in range(2):
            await circuit_breaker.process(context)
        
        # Now process again with open circuit
        result = await circuit_breaker.process(context)
        
        # Should succeed with fallback
        self.assertTrue(result)
        self.assertEqual(mock_stage.process_called, 2)  # Not increased
        self.assertTrue(context.get("fallback_applied"))
        self.assertEqual(context.get("test_value"), 42)
    
    async def test_with_circuit_breaker_factory(self):
        """Test the with_circuit_breaker factory function."""
        mock_stage = MockStage()
        circuit_breaker = with_circuit_breaker(
            stage=mock_stage,
            failure_threshold=3,
            use_registry=False,
            name="test_circuit"
        )
        
        self.assertIsInstance(circuit_breaker, CircuitBreakerStage)
        self.assertEqual(circuit_breaker.name, "test_circuit")
        self.assertEqual(circuit_breaker.failure_threshold, 3)
        self.assertFalse(circuit_breaker.use_registry)


class TestBulkheadStage(unittest.IsolatedAsyncioTestCase):
    """Test case for the BulkheadStage."""
    
    async def test_process_success(self):
        """Test successful processing through bulkhead."""
        mock_stage = MockStage()
        bulkhead = BulkheadStage(
            wrapped_stage=mock_stage,
            config={"max_concurrent_executions": 2}
        )
        
        context = PipelineContext()
        result = await bulkhead.process(context)
        
        self.assertTrue(result)
        self.assertEqual(mock_stage.process_called, 1)
        self.assertEqual(bulkhead._total_executions, 1)
        self.assertEqual(bulkhead._current_executions, 0)  # Reset after completion
    
    async def test_concurrent_executions(self):
        """Test handling of concurrent executions."""
        mock_stage = MockStage(delay=0.1)
        bulkhead = BulkheadStage(
            wrapped_stage=mock_stage,
            config={"max_concurrent_executions": 2}
        )
        
        async def run_task():
            context = PipelineContext()
            return await bulkhead.process(context)
        
        # Start 3 concurrent tasks
        tasks = [run_task() for _ in range(3)]
        results = await asyncio.gather(*tasks)
        
        # All should succeed because max_concurrent_executions is 2 and queue_size is default 20
        self.assertEqual(results, [True, True, True])
        self.assertEqual(mock_stage.process_called, 3)
        self.assertEqual(bulkhead._total_executions, 3)
        self.assertEqual(bulkhead._current_executions, 0)  # Reset after completion
    
    async def test_execution_timeout(self):
        """Test timeout for long-running executions."""
        mock_stage = MockStage(delay=0.5)
        bulkhead = BulkheadStage(
            wrapped_stage=mock_stage,
            config={"max_concurrent_executions": 1, "execution_timeout_seconds": 0.1}
        )
        
        context = PipelineContext()
        result = await bulkhead.process(context)
        
        self.assertFalse(result)
        self.assertEqual(mock_stage.process_called, 1)
        self.assertEqual(bulkhead._timed_out_executions, 1)
        self.assertTrue(any("Execution timed out" in err for err in context.errors.values()))
    
    async def test_queue_timeout(self):
        """Test timeout for operations waiting in queue."""
        mock_stage = MockStage(delay=0.5)
        bulkhead = BulkheadStage(
            wrapped_stage=mock_stage,
            config={
                "max_concurrent_executions": 1,
                "queue_timeout_seconds": 0.1
            }
        )
        
        # Create a task that will hold the semaphore
        first_context = PipelineContext()
        first_task = asyncio.create_task(bulkhead.process(first_context))
        
        # Give time for the first task to acquire the semaphore
        await asyncio.sleep(0.05)
        
        # This should time out waiting for the semaphore
        second_context = PipelineContext()
        result = await bulkhead.process(second_context)
        
        self.assertFalse(result)
        self.assertEqual(bulkhead._queue_timeouts, 1)
        self.assertTrue(any("Bulkhead queue timeout" in err for err in second_context.errors.values()))
        
        # Clean up first task
        await first_task
    
    async def test_with_bulkhead_factory(self):
        """Test the with_bulkhead factory function."""
        mock_stage = MockStage()
        bulkhead = with_bulkhead(
            stage=mock_stage,
            max_concurrent_executions=5,
            max_queue_size=10,
            name="test_bulkhead"
        )
        
        self.assertIsInstance(bulkhead, BulkheadStage)
        self.assertEqual(bulkhead.name, "test_bulkhead")
        self.assertEqual(bulkhead.max_concurrent_executions, 5)
        self.assertEqual(bulkhead.max_queue_size, 10)


class TestCombinedProtectionPatterns(unittest.IsolatedAsyncioTestCase):
    """Test case for combined protection patterns."""
    
    async def test_with_combined_protection_factory(self):
        """Test the with_combined_protection factory function."""
        mock_stage = MockStage()
        protected_stage = with_combined_protection(
            stage=mock_stage,
            circuit_breaker_config={"failure_threshold": 3, "use_registry": False},
            bulkhead_config={"max_concurrent_executions": 2},
            name_prefix="test_"
        )
        
        # Verify the structure: CircuitBreaker -> Bulkhead -> MockStage
        self.assertIsInstance(protected_stage, CircuitBreakerStage)
        self.assertTrue(protected_stage.name.startswith("test_circuit_breaker_"))
        
        bulkhead_stage = protected_stage.wrapped_stage
        self.assertIsInstance(bulkhead_stage, BulkheadStage)
        self.assertTrue(bulkhead_stage.name.startswith("test_bulkhead_"))
        
        self.assertIs(bulkhead_stage.wrapped_stage, mock_stage)
        
        # Test that it works
        context = PipelineContext()
        result = await protected_stage.process(context)
        
        self.assertTrue(result)
        self.assertEqual(mock_stage.process_called, 1)
    
    async def test_cascading_failures_handled(self):
        """Test that cascading failures are properly handled with combined protection."""
        # Create a mock stage that fails
        mock_stage = MockStage(fail=True)
        
        # Create a protected stage with tight limits
        protected_stage = with_combined_protection(
            stage=mock_stage,
            circuit_breaker_config={"failure_threshold": 2, "use_registry": False},
            bulkhead_config={"max_concurrent_executions": 1, "queue_timeout_seconds": 0.1}
        )
        
        # Fail twice to open the circuit
        context1 = PipelineContext()
        context2 = PipelineContext()
        await protected_stage.process(context1)
        await protected_stage.process(context2)
        
        # Circuit should be open now, preventing further calls
        context3 = PipelineContext()
        result = await protected_stage.process(context3)
        
        self.assertFalse(result)
        self.assertEqual(mock_stage.process_called, 2)  # Not increased
        self.assertTrue(context3.contains("circuit_breaker_open"))
        
        # Create more concurrent requests than bulkhead allows
        # (This test is more for illustration, as the circuit breaker
        # will prevent calls from reaching the bulkhead)
        tasks = []
        for _ in range(3):
            context = PipelineContext()
            tasks.append(protected_stage.process(context))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should fail because the circuit is open
        for result in results:
            self.assertFalse(result)
        
        # No additional calls to the underlying stage
        self.assertEqual(mock_stage.process_called, 2)


if __name__ == "__main__":
    unittest.main()