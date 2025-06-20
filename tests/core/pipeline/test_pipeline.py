"""
Test module for the pipeline functionality.

This module contains tests for the Pipeline, PipelineStage, and PipelineContext classes.
"""

import unittest
import asyncio
from typing import Dict, Any, Optional

from core.pipeline.pipeline import Pipeline, PipelineError
from core.pipeline.stage import PipelineStage
from core.pipeline.context import PipelineContext


class TestStage(PipelineStage):
    """A simple test stage that performs actions based on configuration."""
    
    async def process(self, context: PipelineContext) -> bool:
        """
        Process the test stage.
        
        This implementation can be configured to:
        - Add data to the context
        - Fail based on a condition
        - Raise an exception
        - Simulate processing time
        
        Args:
            context: The pipeline context
            
        Returns:
            bool: Success or failure
            
        Raises:
            Exception: If configured to raise an exception
        """
        # Simulate processing time if configured
        if 'delay' in self.config:
            await asyncio.sleep(self.config['delay'])
            
        # Add data to the context if configured
        if 'add_data' in self.config:
            for key, value in self.config['add_data'].items():
                context.set(key, value)
                
        # Raise an exception if configured
        if self.config.get('raise_exception'):
            raise Exception(self.config.get('exception_message', 'Test exception'))
            
        # Return success or failure based on configuration
        return not self.config.get('fail', False)


class FailingInputValidationStage(PipelineStage):
    """A test stage that fails input validation."""
    
    async def validate_input(self, context: PipelineContext) -> bool:
        return False
        
    async def process(self, context: PipelineContext) -> bool:
        # This should never be called if validation fails
        context.set('validation_failed', True)
        return True


class PipelineTestCase(unittest.TestCase):
    """Test cases for the Pipeline, PipelineStage, and PipelineContext classes."""
    
    def setUp(self):
        """Set up the test environment."""
        # Configure a simple test pipeline
        self.pipeline = Pipeline(
            name="test_pipeline",
            config={"continue_on_error": False}
        )
        
    async def run_pipeline(self, stages, initial_data=None):
        """Helper to run a pipeline with the given stages and initial data."""
        pipeline = Pipeline(name="test_pipeline")
        pipeline.add_stages(stages)
        return await pipeline.execute(initial_data)

    def test_pipeline_context(self):
        """Test basic functionality of the PipelineContext class."""
        context = PipelineContext({"key1": "value1"})
        
        # Test get and set
        self.assertEqual(context.get("key1"), "value1")
        context.set("key2", "value2")
        self.assertEqual(context.get("key2"), "value2")
        
        # Test update
        context.update({"key3": "value3", "key4": "value4"})
        self.assertEqual(context.get("key3"), "value3")
        self.assertEqual(context.get("key4"), "value4")
        
        # Test has_key and remove
        self.assertTrue(context.has_key("key1"))
        context.remove("key1")
        self.assertFalse(context.has_key("key1"))
        
        # Test error handling
        self.assertFalse(context.has_errors())
        context.add_error("test_source", "test_error")
        self.assertTrue(context.has_errors())
        errors = context.get_errors()
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors["test_source"][0], "test_error")

    def test_pipeline_execution_sync(self):
        """Test synchronous execution of a pipeline with event loop."""
        async def _test():
            # Create test stages
            stage1 = TestStage(name="stage1", config={"add_data": {"result1": "value1"}})
            stage2 = TestStage(name="stage2", config={"add_data": {"result2": "value2"}})
            
            # Add stages to pipeline
            self.pipeline.add_stage(stage1)
            self.pipeline.add_stage(stage2)
            
            # Execute pipeline
            context = await self.pipeline.execute({"initial": "data"})
            
            # Verify results
            self.assertEqual(context.get("initial"), "data")
            self.assertEqual(context.get("result1"), "value1")
            self.assertEqual(context.get("result2"), "value2")
            
            # Verify metrics
            metrics = context.get_metrics()
            self.assertEqual(metrics["successful_stages"], 2)
            self.assertEqual(metrics["total_stages"], 2)
            self.assertFalse(metrics["has_errors"])
            
        # Run the async test
        asyncio.run(_test())
        
    def test_pipeline_error_handling(self):
        """Test pipeline error handling."""
        async def _test():
            # Create test stages
            stage1 = TestStage(name="stage1", config={"add_data": {"result1": "value1"}})
            stage2 = TestStage(name="stage2", config={"fail": True})
            stage3 = TestStage(name="stage3", config={"add_data": {"result3": "value3"}})
            
            # Test without continue_on_error
            pipeline = Pipeline(name="error_test", config={"continue_on_error": False})
            pipeline.add_stages([stage1, stage2, stage3])
            
            try:
                await pipeline.execute()
                self.fail("Pipeline should have raised an exception")
            except PipelineError:
                pass  # Expected behavior
            
            # Test with continue_on_error
            pipeline = Pipeline(name="error_test", config={"continue_on_error": True})
            pipeline.add_stages([stage1, stage2, stage3])
            
            context = await pipeline.execute()
            
            # Verify results - stage3 should have executed despite stage2 failure
            self.assertEqual(context.get("result1"), "value1")
            self.assertEqual(context.get("result3"), "value3")
            
            # Verify metrics
            metrics = context.get_metrics()
            self.assertEqual(metrics["successful_stages"], 2)
            self.assertEqual(metrics["total_stages"], 3)
            self.assertTrue(metrics["has_errors"])
            
        # Run the async test
        asyncio.run(_test())
        
    def test_pipeline_exception_handling(self):
        """Test pipeline exception handling."""
        async def _test():
            # Create test stages
            stage1 = TestStage(name="stage1", config={"add_data": {"result1": "value1"}})
            stage2 = TestStage(name="stage2", config={"raise_exception": True, "exception_message": "Test exception"})
            stage3 = TestStage(name="stage3", config={"add_data": {"result3": "value3"}})
            
            # Test without continue_on_error
            pipeline = Pipeline(name="exception_test", config={"continue_on_error": False})
            pipeline.add_stages([stage1, stage2, stage3])
            
            try:
                await pipeline.execute()
                self.fail("Pipeline should have raised an exception")
            except PipelineError:
                pass  # Expected behavior
            
            # Test with continue_on_error
            pipeline = Pipeline(name="exception_test", config={"continue_on_error": True})
            pipeline.add_stages([stage1, stage2, stage3])
            
            context = await pipeline.execute()
            
            # Verify results - stage3 should have executed despite stage2 exception
            self.assertEqual(context.get("result1"), "value1")
            self.assertEqual(context.get("result3"), "value3")
            
            # Verify metrics
            metrics = context.get_metrics()
            self.assertEqual(metrics["successful_stages"], 2)
            self.assertEqual(metrics["total_stages"], 3)
            self.assertTrue(metrics["has_errors"])
            
        # Run the async test
        asyncio.run(_test())
        
    def test_input_validation(self):
        """Test input validation in pipeline stages."""
        async def _test():
            # Create test stages
            stage1 = TestStage(name="stage1", config={"add_data": {"result1": "value1"}})
            stage2 = FailingInputValidationStage(name="stage2")
            stage3 = TestStage(name="stage3", config={"add_data": {"result3": "value3"}})
            
            # Test without continue_on_error
            pipeline = Pipeline(name="validation_test", config={"continue_on_error": False})
            pipeline.add_stages([stage1, stage2, stage3])
            
            try:
                await pipeline.execute()
                self.fail("Pipeline should have raised an exception")
            except PipelineError:
                pass  # Expected behavior
            
            # Test with continue_on_error
            pipeline = Pipeline(name="validation_test", config={"continue_on_error": True})
            pipeline.add_stages([stage1, stage2, stage3])
            
            context = await pipeline.execute()
            
            # Verify results - stage2 should not have executed its process method
            self.assertEqual(context.get("result1"), "value1")
            self.assertIsNone(context.get("validation_failed"))
            self.assertEqual(context.get("result3"), "value3")
            
            # Verify metrics
            metrics = context.get_metrics()
            self.assertEqual(metrics["successful_stages"], 2)
            self.assertEqual(metrics["total_stages"], 3)
            self.assertTrue(metrics["has_errors"])
            
        # Run the async test
        asyncio.run(_test())


if __name__ == "__main__":
    unittest.main()