import unittest
import asyncio
from unittest.mock import MagicMock, patch

from core.pipeline.templates.extraction_pipeline import ExtractionPipeline
from core.pipeline.context import PipelineContext
from core.pipeline.stage import PipelineStage


class MockInputStage(PipelineStage):
    """Mock input stage for testing."""
    
    async def process(self, context: PipelineContext) -> bool:
        context.set("source_content", "Test content")
        context.set("source_url", self.config.get("url", "http://example.com"))
        return True


class MockProcessingStage(PipelineStage):
    """Mock processing stage for testing."""
    
    async def process(self, context: PipelineContext) -> bool:
        # Simulate processing the content
        content = context.get("source_content")
        processed = f"Processed: {content}"
        context.set("processed_content", processed)
        return True


class MockOutputStage(PipelineStage):
    """Mock output stage for testing."""
    
    async def process(self, context: PipelineContext) -> bool:
        # Simulate formatting the output
        processed = context.get("processed_content")
        result = {"result": processed}
        context.set("extraction_result", result)
        return True


class TestExtractionPipeline(unittest.TestCase):
    """Test cases for the ExtractionPipeline class and its factory methods."""
    
    def setUp(self):
        """Set up test environment."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Clean up after tests."""
        self.loop.close()
    
    def test_pipeline_initialization(self):
        """Test basic pipeline initialization."""
        pipeline = ExtractionPipeline("test_pipeline")
        
        self.assertEqual(pipeline.name, "test_pipeline")
        self.assertTrue(isinstance(pipeline.extraction_config, dict))
        self.assertTrue(pipeline.extraction_config["clean_html"])
        self.assertTrue(pipeline.extraction_config["extract_metadata"])
        
        # Test with custom config
        custom_config = {
            "extraction_config": {
                "clean_html": False,
                "extract_metadata": False
            }
        }
        pipeline = ExtractionPipeline("custom_pipeline", custom_config)
        self.assertFalse(pipeline.extraction_config["clean_html"])
        self.assertFalse(pipeline.extraction_config["extract_metadata"])
    
    def test_basic_pipeline_execution(self):
        """Test basic pipeline execution flow."""
        pipeline = ExtractionPipeline("test_execution")
        
        # Add mock stages
        pipeline.add_stages([
            MockInputStage(),
            MockProcessingStage(),
            MockOutputStage()
        ])
        
        # Execute the pipeline
        context = self.loop.run_until_complete(pipeline.execute())
        
        # Check results
        self.assertFalse(context.has_errors())
        self.assertEqual(context.get("source_content"), "Test content")
        self.assertEqual(context.get("processed_content"), "Processed: Test content")
        self.assertEqual(context.get("extraction_result"), {"result": "Processed: Test content"})
    
    def test_error_handling(self):
        """Test pipeline error handling."""
        pipeline = ExtractionPipeline("test_errors")
        
        # Create a stage that raises an exception
        error_stage = MagicMock(spec=PipelineStage)
        error_stage.name = "error_stage"
        error_stage.validate_input.return_value = True
        error_stage.process.side_effect = ValueError("Test error")
        error_stage.handle_error.return_value = True
        
        pipeline.add_stages([
            MockInputStage(),
            error_stage
        ])
        
        # Configure pipeline to continue after errors
        pipeline.config["continue_on_error"] = True
        
        # Execute the pipeline
        context = self.loop.run_until_complete(pipeline.execute())
        
        # Check that the error was recorded
        self.assertTrue(context.has_errors())
        self.assertIn("error_stage", context.metadata["errors"])
    
    @patch('core.pipeline.stages.input.http_input.HTTPInputStage')
    @patch('core.pipeline.stages.processing.html_processing.HTMLCleaningStage')
    @patch('core.pipeline.stages.processing.html_processing.ContentExtractionStage')
    def test_create_html_extraction_pipeline(self, mock_content_extraction, mock_html_cleaning, mock_http_input):
        """Test factory method for HTML extraction pipeline."""
        # Configure mocks
        mock_http_input.return_value = MockInputStage({"url": "http://test.com"})
        mock_html_cleaning.return_value = MockProcessingStage()
        mock_content_extraction.return_value = MockOutputStage()
        
        # Create HTML extraction pipeline
        pipeline = ExtractionPipeline.create_html_extraction_pipeline("http://test.com")
        
        # Verify pipeline configuration
        self.assertIn("html_extraction_test.com", pipeline.name)
        self.assertEqual(pipeline.extraction_config["source_url"], "http://test.com")
        self.assertEqual(pipeline.extraction_config["source_type"], "html")
        
        # Execute the pipeline
        context = self.loop.run_until_complete(pipeline.execute())
        
        # Verify pipeline execution
        self.assertFalse(context.has_errors())
    
    @patch('core.pipeline.stages.input.http_input.HTTPInputStage')
    @patch('core.pipeline.stages.processing.content_extraction.StructuredDataExtractionStage')
    def test_create_api_extraction_pipeline(self, mock_structured_extraction, mock_http_input):
        """Test factory method for API extraction pipeline."""
        # Configure mocks
        mock_http_input.return_value = MockInputStage({"url": "http://api.test.com/v1"})
        mock_structured_extraction.return_value = MockProcessingStage()
        
        # Create API extraction pipeline
        pipeline = ExtractionPipeline.create_api_extraction_pipeline(
            "http://api.test.com/v1",
            {"method": "POST", "headers": {"Content-Type": "application/json"}}
        )
        
        # Verify pipeline configuration
        self.assertIn("api_extraction_api.test.com", pipeline.name)
        self.assertEqual(pipeline.extraction_config["source_url"], "http://api.test.com/v1")
        self.assertEqual(pipeline.extraction_config["source_type"], "api")
        
        # Execute the pipeline
        context = self.loop.run_until_complete(pipeline.execute())
        
        # Verify pipeline execution
        self.assertFalse(context.has_errors())
    
    @patch('core.pipeline.stages.input.http_input.HTTPInputStage')
    @patch('core.pipeline.stages.processing.content_extraction.StructuredDataExtractionStage')
    def test_create_feed_extraction_pipeline(self, mock_structured_extraction, mock_http_input):
        """Test factory method for feed extraction pipeline."""
        # Configure mocks
        mock_http_input.return_value = MockInputStage({"url": "http://blog.test.com/rss"})
        mock_structured_extraction.return_value = MockProcessingStage()
        
        # Create feed extraction pipeline
        pipeline = ExtractionPipeline.create_feed_extraction_pipeline("http://blog.test.com/rss")
        
        # Verify pipeline configuration
        self.assertIn("feed_extraction_blog.test.com", pipeline.name)
        self.assertEqual(pipeline.extraction_config["source_url"], "http://blog.test.com/rss")
        self.assertEqual(pipeline.extraction_config["source_type"], "feed")
        
        # Execute the pipeline
        context = self.loop.run_until_complete(pipeline.execute())
        
        # Verify pipeline execution
        self.assertFalse(context.has_errors())
    
    @patch('core.pipeline.stages.input.http_input.HTTPInputStage')
    @patch('core.pipeline.stages.processing.content_extraction.StructuredDataExtractionStage')
    def test_create_sitemap_extraction_pipeline(self, mock_structured_extraction, mock_http_input):
        """Test factory method for sitemap extraction pipeline."""
        # Configure mocks
        mock_http_input.return_value = MockInputStage({"url": "http://site.test.com/sitemap.xml"})
        mock_structured_extraction.return_value = MockProcessingStage()
        
        # Create sitemap extraction pipeline
        pipeline = ExtractionPipeline.create_sitemap_extraction_pipeline(
            "http://site.test.com/sitemap.xml",
            {"recursive": True, "url_limit": 50}
        )
        
        # Verify pipeline configuration
        self.assertIn("sitemap_extraction_site.test.com", pipeline.name)
        self.assertEqual(pipeline.extraction_config["source_url"], "http://site.test.com/sitemap.xml")
        self.assertEqual(pipeline.extraction_config["source_type"], "sitemap")
        self.assertTrue(pipeline.extraction_config["recursive"])
        self.assertEqual(pipeline.extraction_config["url_limit"], 50)
        
        # Execute the pipeline
        context = self.loop.run_until_complete(pipeline.execute())
        
        # Verify pipeline execution
        self.assertFalse(context.has_errors())
    
    def test_handle_extraction_error(self):
        """Test the specialized extraction error handling."""
        pipeline = ExtractionPipeline("test_error_handling")
        context = PipelineContext({"retry_count": 0})
        
        # Test with retry enabled
        pipeline.extraction_config["retry_failed_stages"] = True
        pipeline.extraction_config["max_retries"] = 3
        
        result = pipeline.handle_extraction_error(context, ValueError("Test error"))
        
        # Should handle the error and allow retry
        self.assertTrue(result)
        self.assertEqual(context.get("retry_count"), 1)
        
        # Test with fallback
        context = PipelineContext({})
        pipeline.extraction_config["retry_failed_stages"] = False
        pipeline.extraction_config["use_fallback"] = True
        
        result = pipeline.handle_extraction_error(context, ValueError("Test error"))
        
        # Should use fallback
        self.assertTrue(result)
        
        # Test with no retry or fallback
        context = PipelineContext({})
        pipeline.extraction_config["retry_failed_stages"] = False
        pipeline.extraction_config["use_fallback"] = False
        
        result = pipeline.handle_extraction_error(context, ValueError("Test error"))
        
        # Should not handle the error
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()