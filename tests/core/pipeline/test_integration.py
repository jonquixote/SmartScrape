import unittest
import asyncio
import json
from unittest.mock import MagicMock, patch

from core.pipeline.pipeline import Pipeline
from core.pipeline.stage import PipelineStage
from core.pipeline.context import PipelineContext
from core.pipeline.registry import PipelineRegistry

class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline architecture."""
    
    def setUp(self):
        """Set up test environment."""
        self.registry = PipelineRegistry()
        # Register test stages in the registry
        self.registry.load_default_stages()
    
    def tearDown(self):
        """Clean up after tests."""
        # Reset the registry
        self.registry.clear()
    
    async def _run_pipeline(self, pipeline):
        """Helper to run a pipeline asynchronously."""
        return await pipeline.execute()
    
    def test_complete_pipeline(self):
        """Test a complete pipeline with multiple stages."""
        # Create a pipeline with multiple stages
        pipeline = Pipeline("test_complete_pipeline")
        
        # Add input, processing, and output stages
        pipeline.add_stage(TestInputStage({"test_data": "input_value"}))
        pipeline.add_stage(TestProcessingStage())
        pipeline.add_stage(TestOutputStage())
        
        # Execute pipeline
        context = asyncio.run(self._run_pipeline(pipeline))
        
        # Verify the pipeline execution completed successfully
        self.assertFalse(context.has_errors())
        self.assertEqual(context.get("processed_data"), "PROCESSED: input_value")
        self.assertTrue(context.get("output_saved", False))
        
        # Check execution flow through metrics
        metrics = context.get_metrics()
        self.assertEqual(len(metrics["stages"]), 3)
        for stage_name, stage_metrics in metrics["stages"].items():
            self.assertEqual(stage_metrics["status"], "success")
    
    def test_data_flow_between_stages(self):
        """Test correct data flow between pipeline stages."""
        # Create pipeline with stages that pass data through
        pipeline = Pipeline("test_data_flow")
        
        # Add stages with data transformations
        pipeline.add_stage(DataGeneratorStage({"initial_value": 5}))
        pipeline.add_stage(DataMultiplierStage({"multiplier": 2}))
        pipeline.add_stage(DataVerifierStage({"expected_value": 10}))
        
        # Execute pipeline
        context = asyncio.run(self._run_pipeline(pipeline))
        
        # Verify data flow and transformations
        self.assertFalse(context.has_errors())
        self.assertEqual(context.get("value"), 10)
        self.assertTrue(context.get("verification_passed", False))
    
    def test_conditional_execution(self):
        """Test conditional execution paths in a pipeline."""
        # Create a pipeline with conditional stages
        pipeline = Pipeline("test_conditional")
        
        # Add stages with conditions
        pipeline.add_stage(TestInputStage({"test_data": "conditional_test"}))
        pipeline.add_stage(ConditionalStage({
            "condition_key": "test_data", 
            "condition_value": "conditional_test",
            "next_stage": "path_a"
        }))
        
        # Add path A stages
        path_a_stage = TestProcessingStage({"path": "A"})
        path_a_stage.name = "path_a"
        pipeline.add_stage(path_a_stage)
        
        # Add path B stages
        path_b_stage = TestProcessingStage({"path": "B"})
        path_b_stage.name = "path_b"
        pipeline.add_stage(path_b_stage)
        
        # Execute pipeline
        context = asyncio.run(self._run_pipeline(pipeline))
        
        # Verify conditional path was taken
        self.assertFalse(context.has_errors())
        self.assertEqual(context.get("path_taken"), "A")
        self.assertFalse("path_b_executed" in context.data)
    
    def test_context_data_management(self):
        """Test context data management during pipeline execution."""
        # Create a pipeline that manipulates context data
        pipeline = Pipeline("test_context_management")
        
        # Add stages that manipulate context
        pipeline.add_stage(ContextManipulationStage({
            "operations": [
                {"op": "set", "key": "test_key", "value": "test_value"},
                {"op": "set", "key": "counter", "value": 0}
            ]
        }))
        
        pipeline.add_stage(ContextManipulationStage({
            "operations": [
                {"op": "increment", "key": "counter", "value": 1},
                {"op": "append", "key": "list_data", "value": "item1"}
            ]
        }))
        
        pipeline.add_stage(ContextManipulationStage({
            "operations": [
                {"op": "increment", "key": "counter", "value": 2},
                {"op": "append", "key": "list_data", "value": "item2"}
            ]
        }))
        
        # Execute pipeline
        context = asyncio.run(self._run_pipeline(pipeline))
        
        # Verify context manipulation
        self.assertFalse(context.has_errors())
        self.assertEqual(context.get("test_key"), "test_value")
        self.assertEqual(context.get("counter"), 3)
        self.assertEqual(context.get("list_data"), ["item1", "item2"])
    
    def test_error_propagation(self):
        """Test error propagation and handling in pipelines."""
        # Create a pipeline with a failing stage
        pipeline = Pipeline("test_error_handling", {"continue_on_error": True})
        
        # Add stages with the second one failing
        pipeline.add_stage(TestInputStage({"test_data": "will_fail"}))
        pipeline.add_stage(FailingStage({"error_message": "Deliberate test failure"}))
        pipeline.add_stage(ErrorCheckingStage())
        
        # Execute pipeline
        context = asyncio.run(self._run_pipeline(pipeline))
        
        # Verify error handling
        self.assertTrue(context.has_errors())
        self.assertIn("FailingStage", context.metadata["errors"])
        self.assertTrue(context.get("error_detected", False))
        
        # Verify pipeline continued despite error
        metrics = context.get_metrics()
        self.assertEqual(len(metrics["stages"]), 3)
        self.assertEqual(metrics["stages"]["FailingStage"]["status"], "failed")
        self.assertEqual(metrics["stages"]["ErrorCheckingStage"]["status"], "success")
    
    def test_pipeline_composition(self):
        """Test pipeline composition and nesting."""
        # Create a sub-pipeline
        sub_pipeline = Pipeline("sub_pipeline")
        sub_pipeline.add_stage(TestInputStage({"sub_data": "sub_value"}))
        sub_pipeline.add_stage(TestProcessingStage())
        
        # Create a main pipeline that includes the sub-pipeline
        main_pipeline = Pipeline("main_pipeline")
        main_pipeline.add_stage(TestInputStage({"main_data": "main_value"}))
        main_pipeline.add_stage(SubPipelineStage({"pipeline": sub_pipeline}))
        main_pipeline.add_stage(CompositionVerifierStage())
        
        # Execute main pipeline
        context = asyncio.run(self._run_pipeline(main_pipeline))
        
        # Verify both pipelines executed and data was preserved
        self.assertFalse(context.has_errors())
        self.assertEqual(context.get("main_data"), "main_value")
        self.assertEqual(context.get("sub_data"), "sub_value")
        self.assertEqual(context.get("processed_data"), "PROCESSED: sub_value")
        self.assertTrue(context.get("composition_verified", False))


# Test stage implementations for integration testing

class TestInputStage(PipelineStage):
    """Test input stage that provides test data."""
    
    async def process(self, context):
        """Process method that adds test data to context."""
        # Add config data to the context
        for key, value in self.config.items():
            if key != "name":  # Skip the name config
                context.set(key, value)
        return True


class TestProcessingStage(PipelineStage):
    """Test processing stage."""
    
    async def process(self, context):
        """Process method that transforms data."""
        path = self.config.get("path")
        if path:
            context.set("path_taken", path)
            context.set(f"path_{path.lower()}_executed", True)
            return True
            
        # Get data from context and process it
        test_data = context.get("test_data")
        if test_data:
            processed = f"PROCESSED: {test_data}"
            context.set("processed_data", processed)
        
        # Get sub_data if it exists
        sub_data = context.get("sub_data")
        if sub_data:
            processed = f"PROCESSED: {sub_data}"
            context.set("processed_data", processed)
            
        return True


class TestOutputStage(PipelineStage):
    """Test output stage."""
    
    async def process(self, context):
        """Process method that simulates saving output."""
        # Simulate saving processed data
        if context.get("processed_data"):
            context.set("output_saved", True)
        return True


class DataGeneratorStage(PipelineStage):
    """Stage that generates initial data."""
    
    async def process(self, context):
        """Set initial value in the context."""
        initial_value = self.config.get("initial_value", 0)
        context.set("value", initial_value)
        return True


class DataMultiplierStage(PipelineStage):
    """Stage that multiplies data by a factor."""
    
    async def process(self, context):
        """Multiply the value in the context."""
        value = context.get("value", 0)
        multiplier = self.config.get("multiplier", 1)
        context.set("value", value * multiplier)
        return True


class DataVerifierStage(PipelineStage):
    """Stage that verifies data matches expected value."""
    
    async def process(self, context):
        """Verify the value in context matches expected."""
        value = context.get("value", 0)
        expected = self.config.get("expected_value")
        verification = value == expected
        context.set("verification_passed", verification)
        return verification


class ConditionalStage(PipelineStage):
    """Stage that conditionally executes different paths."""
    
    async def process(self, context):
        """Determine which path to take based on condition."""
        key = self.config.get("condition_key")
        expected_value = self.config.get("condition_value")
        next_stage = self.config.get("next_stage")
        
        if context.get(key) == expected_value:
            context.set("selected_path", next_stage)
            return True
        
        # Condition not met, skip the next stage
        context.set("selected_path", None)
        return True


class ContextManipulationStage(PipelineStage):
    """Stage that performs various context manipulation operations."""
    
    async def process(self, context):
        """Apply operations to the context."""
        operations = self.config.get("operations", [])
        
        for op in operations:
            op_type = op.get("op")
            key = op.get("key")
            value = op.get("value")
            
            if op_type == "set":
                context.set(key, value)
            elif op_type == "increment":
                current = context.get(key, 0)
                context.set(key, current + value)
            elif op_type == "append":
                current = context.get(key, [])
                if not isinstance(current, list):
                    current = [current]
                current.append(value)
                context.set(key, current)
                
        return True


class FailingStage(PipelineStage):
    """Stage that deliberately fails."""
    
    async def process(self, context):
        """Raise an exception to simulate failure."""
        error_message = self.config.get("error_message", "Test failure")
        raise RuntimeError(error_message)


class ErrorCheckingStage(PipelineStage):
    """Stage that checks for errors in the context."""
    
    async def process(self, context):
        """Check if errors exist in the context."""
        has_errors = context.has_errors()
        context.set("error_detected", has_errors)
        return True


class SubPipelineStage(PipelineStage):
    """Stage that executes a sub-pipeline."""
    
    async def process(self, context):
        """Execute a sub-pipeline with the current context."""
        sub_pipeline = self.config.get("pipeline")
        if not sub_pipeline:
            context.add_error(self.name, "No sub-pipeline configured")
            return False
            
        # Execute the sub-pipeline with the same context
        await sub_pipeline.execute(context.data)
        return True


class CompositionVerifierStage(PipelineStage):
    """Stage that verifies pipeline composition was successful."""
    
    async def process(self, context):
        """Verify data from both main and sub pipelines exists."""
        main_data = context.get("main_data")
        sub_data = context.get("sub_data")
        processed_data = context.get("processed_data")
        
        if main_data and sub_data and processed_data:
            context.set("composition_verified", True)
            return True
            
        return False


if __name__ == "__main__":
    unittest.main()