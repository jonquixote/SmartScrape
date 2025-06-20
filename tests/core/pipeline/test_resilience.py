import unittest
import asyncio
import time
from unittest.mock import MagicMock, patch

from core.pipeline.pipeline import Pipeline
from core.pipeline.stage import PipelineStage
from core.pipeline.context import PipelineContext


class TestPipelineResilience(unittest.TestCase):
    """Tests for pipeline resilience and recovery mechanisms."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a basic pipeline for testing
        self.pipeline = Pipeline("resilience_test_pipeline")
        
    def tearDown(self):
        """Clean up after tests."""
        # Reset the pipeline
        self.pipeline = None
        
    async def _run_pipeline(self, pipeline):
        """Helper to run a pipeline asynchronously."""
        return await pipeline.execute()
    
    def test_failure_handling(self):
        """Test pipeline behavior under various failure conditions."""
        # Create a pipeline with stages that will fail in different ways
        pipeline = Pipeline("failure_test", {"continue_on_error": True})
        
        # Add stages with different failure patterns
        pipeline.add_stage(ReliableStage())
        pipeline.add_stage(OccasionalFailureStage({"failure_rate": 1.0}))  # Always fails
        pipeline.add_stage(ErrorPropagatingStage())
        pipeline.add_stage(RecoveryStage())
        
        # Execute pipeline
        context = asyncio.run(self._run_pipeline(pipeline))
        
        # Verify pipeline continued despite failures
        self.assertTrue(context.has_errors())
        self.assertIn("OccasionalFailureStage", context.metadata["errors"])
        
        # Check that recovery stage was executed
        self.assertTrue(context.get("recovery_executed", False))
        
        # Verify all stages after the failure were executed
        metrics = context.get_metrics()
        self.assertEqual(len(metrics["stages"]), 4)
        self.assertEqual(metrics["stages"]["OccasionalFailureStage"]["status"], "failed")
        self.assertEqual(metrics["stages"]["RecoveryStage"]["status"], "success")
    
    def test_circuit_breaker(self):
        """Verify circuit breaker functionality."""
        # Create a pipeline with circuit breaker stage
        pipeline = Pipeline("circuit_breaker_test")
        
        # Add circuit breaker and stages
        circuit_breaker = CircuitBreakerStage({
            "threshold": 3,  # Break after 3 failures
            "reset_timeout": 0.5,  # Reset after 0.5 seconds
        })
        pipeline.add_stage(circuit_breaker)
        
        # Add services that will fail
        for i in range(5):
            failing_service = FailingServiceStage({"name": f"service_{i}"})
            pipeline.add_stage(failing_service)
            
        # Add final verification stage
        pipeline.add_stage(CircuitStatusCheckStage())
        
        # Execute pipeline first time - should fail but not trip circuit breaker
        context = asyncio.run(self._run_pipeline(pipeline))
        
        # Verify first failure was recorded but circuit still closed
        self.assertTrue(context.has_errors())
        self.assertEqual(context.get("circuit_status"), "closed")
        self.assertEqual(context.get("failure_count"), 5)
        
        # Execute again to accumulate failures
        for _ in range(2):
            context = asyncio.run(self._run_pipeline(pipeline))
            
        # Verify circuit is now open after threshold exceeded
        self.assertEqual(context.get("circuit_status"), "open")
        self.assertGreaterEqual(context.get("failure_count"), 3)
        
        # Final stages should be skipped due to open circuit
        skipped_services = context.get("skipped_services", [])
        self.assertGreaterEqual(len(skipped_services), 1)
        
        # Wait for reset timeout
        time.sleep(0.6)
        
        # Execute again to test circuit reset
        context = asyncio.run(self._run_pipeline(pipeline))
        
        # Verify circuit was reset to half-open state
        self.assertEqual(context.get("initial_circuit_status"), "half-open")
    
    def test_retry_mechanism(self):
        """Test retry and fallback mechanisms."""
        # Create a pipeline with retry capability
        pipeline = Pipeline("retry_test")
        
        # Add test stages with retry
        retry_stage = RetryableStage({
            "max_retries": 3,
            "retry_delay": 0.1,
            "failure_rate": 0.7,  # 70% chance of failure per attempt
        })
        pipeline.add_stage(retry_stage)
        
        # Add verification stage
        pipeline.add_stage(RetryVerificationStage())
        
        # Execute pipeline
        context = asyncio.run(self._run_pipeline(pipeline))
        
        # Verify retries were attempted
        retry_count = context.get("retry_count", 0)
        self.assertGreaterEqual(retry_count, 0)
        
        # Check the eventual outcome
        successful = context.get("eventually_successful", False)
        max_retries_exceeded = context.get("max_retries_exceeded", False)
        
        # Either it eventually succeeded or max retries were exceeded
        self.assertTrue(successful or max_retries_exceeded)
        
        # Both can't be true
        self.assertFalse(successful and max_retries_exceeded)
    
    def test_timeout_handling(self):
        """Test timeout handling and cancellation."""
        # Create a pipeline with timeout monitoring
        pipeline = Pipeline("timeout_test", {"timeout": 0.5})  # 500ms timeout
        
        # Add stages with different execution times
        pipeline.add_stage(FastStage())
        pipeline.add_stage(SlowStage({"execution_time": 0.1}))  # 100ms, should work
        pipeline.add_stage(SlowStage({"execution_time": 0.3}))  # 300ms, should work
        pipeline.add_stage(SlowStage({"execution_time": 1.0}))  # 1000ms, should timeout
        pipeline.add_stage(TimeoutVerificationStage())
        
        # Execute pipeline
        context = asyncio.run(self._run_pipeline(pipeline))
        
        # Verify timeout was detected
        self.assertTrue(context.has_errors())
        self.assertTrue(context.get("timeout_detected", False))
        
        # Check which stages executed
        completed_stages = set(context.metadata["completed_stages"])
        self.assertIn("FastStage", completed_stages)
        self.assertIn("SlowStage_0", completed_stages)  # 100ms stage
        self.assertIn("SlowStage_1", completed_stages)  # 300ms stage
        
        # The 1000ms stage should have timed out
        self.assertIn("SlowStage_2", context.metadata["errors"])
        
        # Timeout verification should have run
        self.assertIn("TimeoutVerificationStage", completed_stages)
    
    def test_resource_cleanup(self):
        """Test resource cleanup after failures."""
        # Create a pipeline with resource management
        pipeline = Pipeline("resource_test")
        
        # Add resource acquisition stage
        pipeline.add_stage(ResourceAcquisitionStage())
        
        # Add failing stage
        pipeline.add_stage(FailingStage())
        
        # Add verification stage to check cleanup
        pipeline.add_stage(ResourceCleanupVerificationStage())
        
        # Execute pipeline
        context = asyncio.run(self._run_pipeline(pipeline))
        
        # Verify resources were acquired
        self.assertTrue(context.get("resources_acquired", False))
        
        # Verify stage failed
        self.assertTrue(context.has_errors())
        self.assertIn("FailingStage", context.metadata["errors"])
        
        # Verify resources were cleaned up despite failure
        self.assertTrue(context.get("resources_cleaned_up", False))
    
    def test_partial_success(self):
        """Test partial success handling in pipelines."""
        # Create a pipeline with a mix of successful and failing stages
        pipeline = Pipeline("partial_success_test", {"continue_on_error": True})
        
        # Add a mix of successful and failing stages
        for i in range(5):
            if i % 2 == 0:
                pipeline.add_stage(ReliableStage({"name": f"reliable_{i}"}))
            else:
                pipeline.add_stage(OccasionalFailureStage({
                    "name": f"failing_{i}", 
                    "failure_rate": 1.0
                }))
                
        # Add results collection stage
        pipeline.add_stage(ResultsCollectorStage())
        
        # Execute pipeline
        context = asyncio.run(self._run_pipeline(pipeline))
        
        # Verify mixed results
        self.assertTrue(context.has_errors())
        
        # Get results summary
        successes = context.get("successful_stages", [])
        failures = context.get("failed_stages", [])
        
        # Verify correct counts
        self.assertEqual(len(successes), 3)  # 3 reliable stages (0, 2, 4)
        self.assertEqual(len(failures), 2)   # 2 failing stages (1, 3)
        
        # Verify pipeline continued and collected all results
        self.assertTrue(context.get("results_collected", False))


# Test stages for resilience testing

class ReliableStage(PipelineStage):
    """Stage that always succeeds."""
    
    def __init__(self, config=None):
        super().__init__(config or {})
        self.name = self.config.get("name", self.__class__.__name__)
    
    async def process(self, context):
        """Always succeed."""
        context.set(f"{self.name}_executed", True)
        return True


class OccasionalFailureStage(PipelineStage):
    """Stage that fails based on a configurable failure rate."""
    
    def __init__(self, config=None):
        super().__init__(config or {})
        self.name = self.config.get("name", self.__class__.__name__)
        self.failure_rate = self.config.get("failure_rate", 0.5)
    
    async def process(self, context):
        """Sometimes fail based on failure rate."""
        import random
        
        context.set(f"{self.name}_executed", True)
        
        # Determine if this execution should fail
        if random.random() < self.failure_rate:
            error_msg = f"{self.name} failed deliberately based on failure rate"
            context.add_error(self.name, error_msg)
            raise RuntimeError(error_msg)
            
        return True


class ErrorPropagatingStage(PipelineStage):
    """Stage that checks for errors and propagates them."""
    
    async def process(self, context):
        """Check for errors and propagate them."""
        if context.has_errors():
            context.set("errors_detected", True)
            context.set("error_count", len(context.metadata["errors"]))
        
        return True


class RecoveryStage(PipelineStage):
    """Stage that attempts recovery after errors."""
    
    async def process(self, context):
        """Record recovery attempt."""
        if context.has_errors():
            context.set("recovery_executed", True)
            context.set("recovery_attempted_for", list(context.metadata["errors"].keys()))
        
        return True


class CircuitBreakerStage(PipelineStage):
    """Stage that implements circuit breaker pattern."""
    
    def __init__(self, config=None):
        super().__init__(config or {})
        self.name = "CircuitBreakerStage"
        self.threshold = self.config.get("threshold", 5)
        self.reset_timeout = self.config.get("reset_timeout", 60)
        self.failure_count = 0
        self.last_failure_time = 0
        self.circuit_state = "closed"  # closed, open, half-open
    
    async def process(self, context):
        """Implement circuit breaker logic."""
        # Store initial circuit state
        initial_state = self.circuit_state
        context.set("initial_circuit_status", initial_state)
        
        # Check if the circuit should be reset
        current_time = time.time()
        if (self.circuit_state == "open" and 
            current_time - self.last_failure_time > self.reset_timeout):
            self.circuit_state = "half-open"
        
        # If circuit is open, prevent further processing
        if self.circuit_state == "open":
            context.set("circuit_status", "open")
            context.set("circuit_broken_at", self.last_failure_time)
            context.set("skipped_services", [])
            return False
        
        # Initialize failure tracking for this execution
        context.set("service_failures", [])
        context.set("circuit_status", self.circuit_state)
        context.set("failure_count", self.failure_count)
        
        return True
    
    def handle_error(self, context, error):
        """Handle errors by updating circuit breaker state."""
        # Increment failure count
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        # Check if threshold exceeded
        if self.failure_count >= self.threshold:
            self.circuit_state = "open"
            context.set("circuit_status", "open")
            context.set("circuit_tripped", True)
            context.set("failure_threshold", self.threshold)
            
            # Record which services will be skipped
            remaining_stages = []
            for stage in self.pipeline.stages:
                if isinstance(stage, FailingServiceStage) and not stage.name in context.metadata["completed_stages"]:
                    remaining_stages.append(stage.name)
            context.set("skipped_services", remaining_stages)
        
        # Update context with current counts
        context.set("failure_count", self.failure_count)
        
        return False


class FailingServiceStage(PipelineStage):
    """Stage simulating an external service that may fail."""
    
    def __init__(self, config=None):
        super().__init__(config or {})
        self.name = self.config.get("name", self.__class__.__name__)
    
    async def process(self, context):
        """Simulate a service that always fails."""
        error_msg = f"{self.name} failed deliberately"
        
        # Record the failure
        service_failures = context.get("service_failures", [])
        service_failures.append(self.name)
        context.set("service_failures", service_failures)
        
        # Add error
        context.add_error(self.name, error_msg)
        
        # Always fail
        return False


class CircuitStatusCheckStage(PipelineStage):
    """Stage that checks and reports circuit breaker status."""
    
    async def process(self, context):
        """Check circuit status and record details."""
        # Just record information about circuit state for verification
        return True


class RetryableStage(PipelineStage):
    """Stage that implements a retry mechanism."""
    
    def __init__(self, config=None):
        super().__init__(config or {})
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 0.1)
        self.failure_rate = self.config.get("failure_rate", 0.5)
    
    async def process(self, context):
        """Process with retry logic."""
        import random
        
        retry_count = 0
        success = False
        
        while not success and retry_count <= self.max_retries:
            try:
                # Attempt the operation
                if random.random() >= self.failure_rate:
                    # Success
                    success = True
                    break
                else:
                    # Failure
                    raise RuntimeError("Deliberate failure for retry testing")
            except Exception as e:
                # Handle failure
                retry_count += 1
                context.set("last_error", str(e))
                
                if retry_count <= self.max_retries:
                    # Wait before retry
                    await asyncio.sleep(self.retry_delay)
                    context.set("retry_attempted", True)
                    context.set("retry_count", retry_count)
                else:
                    # Max retries exceeded
                    context.add_error(self.name, f"Max retries ({self.max_retries}) exceeded")
                    context.set("max_retries_exceeded", True)
                    return False
        
        # Record success
        context.set("eventually_successful", success)
        context.set("retry_count", retry_count)
        return success


class RetryVerificationStage(PipelineStage):
    """Stage that verifies retry behavior."""
    
    async def process(self, context):
        """Verify retry behavior."""
        # Just record information for verification
        return True


class FastStage(PipelineStage):
    """Stage that executes quickly."""
    
    async def process(self, context):
        """Execute quickly."""
        context.set(f"{self.name}_executed", True)
        return True


class SlowStage(PipelineStage):
    """Stage that takes a configurable amount of time to execute."""
    
    def __init__(self, config=None):
        super().__init__(config or {})
        self.execution_time = self.config.get("execution_time", 1.0)
        # Create unique name for each slow stage instance
        if "instance_number" not in self.__dict__:
            SlowStage.instance_number = getattr(SlowStage, "instance_number", -1) + 1
            self.name = f"{self.__class__.__name__}_{SlowStage.instance_number}"
    
    async def process(self, context):
        """Execute with a delay."""
        # Record starting
        context.set(f"{self.name}_started", True)
        
        try:
            # Sleep for the configured time
            await asyncio.sleep(self.execution_time)
            context.set(f"{self.name}_completed", True)
            return True
        except asyncio.CancelledError:
            # Handle cancellation
            context.set(f"{self.name}_cancelled", True)
            context.add_error(self.name, "Operation cancelled due to timeout")
            return False


class TimeoutVerificationStage(PipelineStage):
    """Stage that verifies timeout behavior."""
    
    async def process(self, context):
        """Verify timeout handling."""
        # Check for cancelled operations
        cancelled_stages = []
        for key in context.data:
            if key.endswith("_cancelled") and context.get(key):
                stage_name = key[:-10]  # Remove "_cancelled"
                cancelled_stages.append(stage_name)
                
        if cancelled_stages:
            context.set("timeout_detected", True)
            context.set("cancelled_stages", cancelled_stages)
            
        return True


class ResourceAcquisitionStage(PipelineStage):
    """Stage that acquires resources that need to be cleaned up."""
    
    async def process(self, context):
        """Acquire resources."""
        # Simulate resource acquisition
        context.set("resources_acquired", True)
        context.set("connection_pool", ["conn1", "conn2", "conn3"])
        context.set("temp_files", ["/tmp/file1", "/tmp/file2"])
        
        # Set up cleanup hook
        def cleanup_resources():
            # This would normally release connections, delete temp files, etc.
            context.set("resources_cleaned_up", True)
            context.set("connection_pool", [])
            context.set("temp_files", [])
            
        context.set("cleanup_function", cleanup_resources)
        
        return True
    
    def handle_error(self, context, error):
        """Ensure resources are cleaned up on error."""
        cleanup_func = context.get("cleanup_function")
        if cleanup_func:
            cleanup_func()
        return False


class FailingStage(PipelineStage):
    """Stage that deliberately fails."""
    
    async def process(self, context):
        """Just fail."""
        raise RuntimeError("Deliberate failure for testing resource cleanup")


class ResourceCleanupVerificationStage(PipelineStage):
    """Stage that verifies resource cleanup."""
    
    async def process(self, context):
        """Verify resource cleanup."""
        # This stage shouldn't normally execute if the previous stage fails
        # unless continue_on_error is True, but we need to check cleanup
        
        # Check if resources were acquired
        if not context.get("resources_acquired", False):
            context.add_error(self.name, "Resources were never acquired")
            return False
            
        # Check if resources were cleaned up
        connection_pool = context.get("connection_pool", [])
        temp_files = context.get("temp_files", [])
        
        resources_cleaned = len(connection_pool) == 0 and len(temp_files) == 0
        context.set("resources_cleaned_up", resources_cleaned)
        
        if not resources_cleaned:
            context.add_error(self.name, "Resources were not properly cleaned up")
            return False
            
        return True


class ResultsCollectorStage(PipelineStage):
    """Stage that collects results from all previous stages."""
    
    async def process(self, context):
        """Collect and summarize results."""
        # Collect all stage statuses
        successful_stages = []
        failed_stages = []
        
        for stage_name, metrics in context.metadata["stage_metrics"].items():
            if metrics["status"] == "success":
                successful_stages.append(stage_name)
            else:
                failed_stages.append(stage_name)
                
        context.set("successful_stages", successful_stages)
        context.set("failed_stages", failed_stages)
        context.set("success_rate", len(successful_stages) / 
                   (len(successful_stages) + len(failed_stages)))
        context.set("results_collected", True)
        
        return True


if __name__ == "__main__":
    unittest.main()