"""
Unit Tests for Semantic Extraction.

This test suite validates that the semantic extractor correctly extracts
structured data from content using AI semantic understanding.
"""

import unittest
import pytest
import asyncio
import json
from unittest.mock import MagicMock, patch, AsyncMock
from bs4 import BeautifulSoup

from extraction.core.extraction_interface import SemanticExtractor
from strategies.core.strategy_context import StrategyContext
from extraction.semantic_extractor import AISemanticExtractor


class TestSemanticExtraction(unittest.TestCase):
    """Test suite for SemanticExtractor."""

    def setUp(self):
        """Set up test environment before each test case."""
        # Create strategy context with mock services
        self.strategy_context = StrategyContext()
        
        # Mock AI service
        self.ai_service = AsyncMock()
        self.ai_service.generate_response = AsyncMock(return_value={
            "content": {
                "title": "Premium Test Product",
                "price": {"amount": 99.99, "currency": "USD"},
                "description": "This is a high-quality test product with amazing features.",
                "specifications": {
                    "weight": "1.5 kg",
                    "dimensions": "10 x 20 x 30 cm",
                    "color": "Silver"
                }
            },
            "_metadata": {
                "model": "test-model",
                "total_tokens": 150
            }
        })
        
        # Mock HTML service
        self.html_service = MagicMock()
        self.html_service.clean_html.return_value = "<html><body>Cleaned HTML</body></html>"
        self.html_service.extract_main_content.return_value = "<body>Main content</body>"
        
        # Mock model selector
        self.model_selector = MagicMock()
        self.model_selector.select_model.return_value = "test-model"
        
        # Mock content processor
        self.content_processor = MagicMock()
        self.content_processor.preprocess_html.return_value = "Preprocessed content"
        
        # Mock error classifier
        self.error_classifier = MagicMock()
        self.error_classifier.classify_exception.return_value = {
            "category": "test_error",
            "retryable": False
        }
        
        # Mock schema validator
        self.schema_validator = MagicMock()
        self.schema_validator.validate.side_effect = lambda data, schema: {
            **data,
            "_metadata": {
                "valid": True,
                "schema": "test_schema"
            }
        }

        # Register mock services
        self.strategy_context.register_service("ai_service", self.ai_service)
        self.strategy_context.register_service("html_service", self.html_service)
        self.strategy_context.register_service("model_selector", self.model_selector)
        self.strategy_context.register_service("content_processor", self.content_processor)
        self.strategy_context.register_service("error_classifier", self.error_classifier)
        self.strategy_context.register_service("schema_validator", self.schema_validator)
        
        # Sample HTML content for testing
        self.sample_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Product Page</title>
            <meta name="description" content="This is a test product page">
        </head>
        <body>
            <h1 class="product-title">Premium Test Product</h1>
            <div class="price">$99.99</div>
            <div class="description">
                <p>This is a high-quality test product with amazing features.</p>
            </div>
        </body>
        </html>
        """
        
        # Create extractor
        self.extractor = AISemanticExtractor(context=self.strategy_context)
        self.extractor.initialize()
        
        # Sample schema
        self.sample_schema = {
            "title": {"type": "string", "description": "Product title", "required": True},
            "price": {"type": "object", "description": "Product price"},
            "description": {"type": "string", "description": "Product description"}
        }

    @pytest.mark.asyncio
    async def test_extract_semantic_content(self):
        """Test extracting semantic content from HTML."""
        # Call extract method
        result = await self.extractor.extract(self.sample_html, self.sample_schema)
        
        # Verify results
        self.assertIsNotNone(result)
        self.assertIn("title", result)
        self.assertEqual("Premium Test Product", result["title"])
        self.assertIn("price", result)
        self.assertIsInstance(result["price"], dict)
        self.assertEqual(99.99, result["price"]["amount"])
        
        # Verify services were used correctly
        self.html_service.clean_html.assert_called_once()
        self.html_service.extract_main_content.assert_called_once()
        self.content_processor.preprocess_html.assert_called_once()
        self.model_selector.select_model.assert_called_once()
        self.ai_service.generate_response.assert_called_once()
        
        # Verify metadata was captured
        self.assertIn("_metadata", result)
        self.assertIn("extractor", result["_metadata"])
        self.assertEqual("AISemanticExtractor", result["_metadata"]["extractor"])

    @pytest.mark.asyncio
    async def test_can_handle_different_content_types(self):
        """Test the can_handle method with different content types."""
        # Test HTML content
        self.assertTrue(self.extractor.can_handle("<html></html>", "html"))
        
        # Test text content
        self.assertTrue(self.extractor.can_handle("Plain text content", "text"))
        
        # Test JSON content
        self.assertTrue(self.extractor.can_handle("{\"key\": \"value\"}", "json"))
        
        # Test unsupported content
        self.assertFalse(self.extractor.can_handle(b"binary data", "binary"))

    @pytest.mark.asyncio
    async def test_extract_with_error_handling(self):
        """Test extraction with error handling."""
        # Make AI service raise an exception
        self.ai_service.generate_response.side_effect = ValueError("Test error")
        
        # Call extract method
        result = await self.extractor.extract(self.sample_html, self.sample_schema)
        
        # Verify error was handled
        self.assertIn("_metadata", result)
        self.assertIn("error", result["_metadata"])
        self.assertIn("success", result["_metadata"])
        self.assertFalse(result["_metadata"]["success"])
        
        # Verify error classifier was called
        self.error_classifier.classify_exception.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_with_different_content_formats(self):
        """Test extraction with different input formats."""
        # Test with BeautifulSoup object
        soup = BeautifulSoup(self.sample_html, "html.parser")
        result_soup = await self.extractor.extract(soup, self.sample_schema)
        self.assertIn("title", result_soup)
        
        # Test with JSON object
        json_data = {"product": {"name": "Test Product", "price": 99.99}}
        result_json = await self.extractor.extract(json_data, self.sample_schema)
        self.assertIn("title", result_json)

    @patch('extraction.semantic_extractor.retry')
    @pytest.mark.asyncio
    async def test_retry_mechanism(self, mock_retry):
        """Test that retry mechanism is used for extraction."""
        # Setup mock retry decorator
        mock_retry_decorator = MagicMock()
        mock_retry_decorator.return_value = lambda func: func
        mock_retry.return_value = mock_retry_decorator
        
        # Create a new extractor instance to use the mocked retry
        extractor = AISemanticExtractor(context=self.strategy_context)
        extractor.initialize()
        
        # Call extract
        await extractor.extract(self.sample_html, self.sample_schema)
        
        # Verify retry was configured
        mock_retry.assert_called_once()
        args, kwargs = mock_retry.call_args
        self.assertIn('retry_on', kwargs)
        self.assertIn('max_attempts', kwargs)

    @pytest.mark.asyncio
    async def test_generate_extraction_prompt(self):
        """Test prompt generation for extraction."""
        # Access the private method for testing (not ideal but necessary)
        prompt = self.extractor._generate_extraction_prompt("Test content", self.sample_schema)
        
        # Verify prompt contains expected components
        self.assertIn("Extract structured data", prompt)
        self.assertIn("title", prompt)
        self.assertIn("price", prompt)
        self.assertIn("description", prompt)
        self.assertIn("Test content", prompt)
        self.assertIn("required", prompt)

    @pytest.mark.asyncio
    async def test_extract_without_schema(self):
        """Test extraction without a schema."""
        # Call extract without schema
        result = await self.extractor.extract(self.sample_html)
        
        # Verify extraction works without schema
        self.assertIsNotNone(result)
        self.assertIn("title", result)
        
        # Verify prompt generation was called without schema guidance
        prompt_args = self.ai_service.generate_response.call_args[1]['prompt']
        self.assertIn("Identify and extract key information", prompt_args)

    @patch('extraction.semantic_extractor.json')
    @pytest.mark.asyncio
    async def test_json_parsing_error_handling(self, mock_json):
        """Test handling of JSON parsing errors."""
        # Setup mock to simulate JSON parsing error
        mock_json.loads.side_effect = json.JSONDecodeError("Test JSON error", "", 0)
        
        # Configure AI service to return text response
        self.ai_service.generate_response.return_value = {
            "content": "This is not valid JSON"
        }
        
        # Call extract method
        result = await self.extractor.extract(self.sample_html, self.sample_schema)
        
        # Verify error handling
        self.assertIn("_error", result)
        self.assertIn("Failed to parse", result["_error"])


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])