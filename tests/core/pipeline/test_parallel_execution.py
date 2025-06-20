"""
Test for parallel execution functionality in the Pipeline class.

This test verifies that the parallel execution capabilities of the Pipeline class
work correctly, including dependency management, timeout handling, and error handling.
"""

import asyncio
import unittest
import time
from unittest.mock import MagicMock, patch

from core.pipeline.pipeline import Pipeline, PipelineError
from core.pipeline.stage import PipelineStage
from core.pipeline.context import PipelineContext


class TestStage(PipelineStage):
    """Test stage for pipeline testing."""
    
    def __init__(self, name, execution_time=0.1, success=True, dependencies=None):
        """Initialize test stage."""
        super().__init__()
        self.name = name
        self.execution_time = execution_time
        self.should_succeed = success
        self.dependencies = dependencies or []
        self.initialize_mock = MagicMock()
        self.cleanup_mock = MagicMock()
        self.pre_process_mock = MagicMock()
        self.post_process_mock = MagicMock()
        self.validate_input_mock = MagicMock(return_value=True)
        self.validate_output_mock = MagicMock(return_value=True)
        
    async def initialize(self):
        """Initialize the stage."""
        self.initialize_mock()
        
    async def cleanup(self):
        """Clean up the stage."""
        self.cleanup_mock()
        
    async def pre_process(self, context):
        """Pre-process hook."""
        self.pre_process_mock(context)
        
    async def post_process(self, context, success):
        """Post-process hook."""
        self.post_process_mock(context, success)
        
    async def validate_input(self, context):
        """Validate input."""
        return self.validate_input_mock(context)
        
    async def validate_output(self, context):
        """Validate output."""
        return self.validate_output_mock(context)
    
    async def process(self, context):
        """Process this stage."""
        # Set a marker in the context to track execution
        executed_stages = context.get("executed_stages", [])
        executed_stages.append(self.name)
        context.set("executed_stages", executed_stages)
        
        # Simulate work
        await asyncio.sleep(self.execution_time)
        
        # Track timing
        context.set(f"timing_{self.name}", time.time())
        
        return self.should_succeed
        
    def get_dependencies(self):
        """Get stage dependencies."""
        return self.dependencies


class TimeoutStage(TestStage):
    """A stage that times out."""
    
    async def process(self, context):
        """Process that takes longer than the timeout."""
        await asyncio.sleep(10)  # Long delay to trigger timeout
        return True


class ErrorStage(TestStage):
    """A stage that raises an exception."""
    
    async def process(self, context):
        """Process that raises an exception."""
        raise ValueError("Test error")


class ParallelExecutionTest(unittest.TestCase):
    """Test suite for parallel execution in Pipeline."""
    
    def setUp(self):
        """Set up the test environment."""
        self.context = PipelineContext()
        
    def tearDown(self):
        """Clean up after each test."""
        pass
    
    async def _run_test_pipeline(self, pipeline):
        """Run a test pipeline."""
        return await pipeline.execute()
    
    def test_parallel_execution_independent_stages(self):
        """Test execution of independent stages in parallel."""
        # Create pipeline with parallel execution
        pipeline = Pipeline("test_pipeline", {
            "parallel_execution": True,
            "max_workers": 3
        })
        
        # Add three independent stages
        pipeline.add_stage(TestStage("stage1", execution_time=0.2))
        pipeline.add_stage(TestStage("stage2", execution_time=0.2))
        pipeline.add_stage(TestStage("stage3", execution_time=0.2))
        
        # Run pipeline
        context = asyncio.run(self._run_test_pipeline(pipeline))
        
        # Verify all stages executed
        executed_stages = context.get("executed_stages", [])
        self.assertEqual(len(executed_stages), 3)
        self.assertIn("stage1", executed_stages)
        self.assertIn("stage2", executed_stages)
        self.assertIn("stage3", executed_stages)
        
        # Verify metrics
        metrics = context.get_metrics()
        self.assertEqual(metrics["successful_stages"], 3)
        
        # In parallel, total time should be close to the max stage time, not the sum
        self.assertLess(metrics["total_time"], 0.6)  # Less than sum of all stage times
    
    def test_parallel_execution_with_dependencies(self):
        """Test execution of stages with dependencies in parallel."""
        # Create pipeline with parallel execution
        pipeline = Pipeline("test_pipeline", {
            "parallel_execution": True,
            "max_workers": 3
        })
        
        # Create stages with dependencies
        stage1 = TestStage("stage1", execution_time=0.1)
        stage2 = TestStage("stage2", execution_time=0.1, dependencies=["stage1"])
        stage3 = TestStage("stage3", execution_time=0.1, dependencies=["stage2"])
        stage4 = TestStage("stage4", execution_time=0.1, dependencies=["stage1"])
        
        # Add stages (order shouldn't matter due to dependencies)
        pipeline.add_stages([stage3, stage1, stage4, stage2])
        
        # Run pipeline
        context = asyncio.run(self._run_test_pipeline(pipeline))
        
        # Verify all stages executed
        executed_stages = context.get("executed_stages", [])
        self.assertEqual(len(executed_stages), 4)
        
        # Verify order respects dependencies
        stage1_index = executed_stages.index("stage1")
        stage2_index = executed_stages.index("stage2")
        stage3_index = executed_stages.index("stage3")
        
        self.assertLess(stage1_index, stage2_index)
        self.assertLess(stage2_index, stage3_index)
        
        # Check timing to verify parallel execution of stage4 with stage2/stage3
        timing_stage2 = context.get("timing_stage2")
        timing_stage4 = context.get("timing_stage4")
        
        # stage4 should start after stage1 but can run in parallel with stage2/stage3
        self.assertIsNotNone(timing_stage2)
        self.assertIsNotNone(timing_stage4)
    
    def test_parallel_execution_with_stage_failure(self):
        """Test handling of stage failures in parallel execution."""
        # Create pipeline with parallel execution
        pipeline = Pipeline("test_pipeline", {
            "parallel_execution": True,
            "max_workers": 3,
            "continue_on_error": True  # Continue despite failures
        })
        
        # Add three stages, one will fail
        pipeline.add_stage(TestStage("stage1", execution_time=0.1))
        pipeline.add_stage(TestStage("stage2", execution_time=0.1, success=False))
        pipeline.add_stage(TestStage("stage3", execution_time=0.1))
        
        # Run pipeline
        context = asyncio.run(self._run_test_pipeline(pipeline))
        
        # Verify all stages executed
        executed_stages = context.get("executed_stages", [])
        self.assertEqual(len(executed_stages), 3)
        
        # Verify metrics
        metrics = context.get_metrics()
        self.assertEqual(metrics["successful_stages"], 2)
        self.assertTrue(metrics["has_errors"])
        
        # Check that stage2 has an error
        errors = context.metadata["errors"]
        self.assertIn("stage2", errors)
    
    def test_parallel_execution_with_timeout(self):
        """Test handling of stage timeout in parallel execution."""
        # Create pipeline with parallel execution and short timeout
        pipeline = Pipeline("test_pipeline", {
            "parallel_execution": True,
            "max_workers": 3,
            "stage_timeout": 0.5,  # Short timeout
            "continue_on_error": True
        })
        
        # Add normal stage and one that will timeout
        pipeline.add_stage(TestStage("normal_stage", execution_time=0.1))
        pipeline.add_stage(TimeoutStage("timeout_stage", execution_time=10.0))
        
        # Run pipeline
        context = asyncio.run(self._run_test_pipeline(pipeline))
        
        # Verify normal stage executed
        executed_stages = context.get("executed_stages", [])
        self.assertIn("normal_stage", executed_stages)
        
        # Verify timeout stage has an error
        errors = context.metadata["errors"]
        self.assertIn("timeout_stage", errors)
        
        # Verify metrics
        metrics = context.get_metrics()
        self.assertEqual(metrics["successful_stages"], 1)
        self.assertTrue(metrics["has_errors"])
    
    def test_parallel_execution_with_exception(self):
        """Test handling of exceptions in parallel execution."""
        # Create pipeline with parallel execution
        pipeline = Pipeline("test_pipeline", {
            "parallel_execution": True,
            "max_workers": 3,
            "continue_on_error": True
        })
        
        # Add normal stage and one that raises an exception
        pipeline.add_stage(TestStage("normal_stage", execution_time=0.1))
        pipeline.add_stage(ErrorStage("error_stage"))
        
        # Run pipeline
        context = asyncio.run(self._run_test_pipeline(pipeline))
        
        # Verify normal stage executed
        executed_stages = context.get("executed_stages", [])
        self.assertIn("normal_stage", executed_stages)
        
        # Verify error stage has an error
        errors = context.metadata["errors"]
        self.assertIn("error_stage", errors)
        
        # Verify metrics
        metrics = context.get_metrics()
        self.assertEqual(metrics["successful_stages"], 1)
        self.assertTrue(metrics["has_errors"])
    
    def test_parallel_execution_stops_on_failure(self):
        """Test that parallel execution stops on failure when continue_on_error is False."""
        # Create pipeline with parallel execution that stops on error
        pipeline = Pipeline("test_pipeline", {
            "parallel_execution": True,
            "max_workers": 3,
            "continue_on_error": False
        })
        
        # Add a quick failing stage and a slow stage that should be canceled
        failing_stage = TestStage("failing_stage", execution_time=0.1, success=False)
        slow_stage = TestStage("slow_stage", execution_time=10.0)  # Should be canceled
        
        pipeline.add_stages([failing_stage, slow_stage])
        
        # Run pipeline and expect PipelineError
        with self.assertRaises(PipelineError):
            asyncio.run(self._run_test_pipeline(pipeline))


if __name__ == "__main__":
    unittest.main()