"""
Tests for AISemanticExtractor

This module tests the AI-based semantic extraction capabilities
of the SmartScrape system.
"""

import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import json
import asyncio

from extraction.semantic_extractor import AISemanticExtractor
from core.ai_service import AIService
from core.model_selector import ModelSelector


class TestAISemanticExtractor(unittest.TestCase):
    """Test cases for AISemanticExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock AI service and model selector
        self.mock_ai_service = MagicMock(spec=AIService)
        self.mock_model_selector = MagicMock(spec=ModelSelector)
        
        # Configure model selector mock
        self.mock_model_selector.select_model.return_value = "gpt-4"
        
        # Configure AI service mock for synchronous responses
        self.mock_ai_service.generate_completion.return_value = json.dumps({
            "title": "Test Product",
            "price": 99.99,
            "description": "A test product description"
        })
        
        # Configure AI service mock for async responses
        async_response = AsyncMock()
        async_response.return_value = json.dumps({
            "title": "Test Product",
            "price": 99.99,
            "description": "A test product description"
        })
        self.mock_ai_service.generate_completion_async = async_response
        
        # Create extractor instance with mocks
        self.extractor = AISemanticExtractor(
            ai_service=self.mock_ai_service,
            model_selector=self.mock_model_selector
        )
        
        # Test data
        self.test_html = "<html><body><h1>Test Product</h1><p>$99.99</p></body></html>"
        self.test_schema = {
            "title": "string",
            "price": "number",
            "features": "array"
        }
    
    def test_initialization(self):
        """Test proper initialization of AISemanticExtractor."""
        self.assertEqual(self.extractor.ai_service, self.mock_ai_service)
        self.assertEqual(self.extractor.model_selector, self.mock_model_selector)
        self.assertIsNotNone(self.extractor.logger)
    
    def test_extract_content(self):
        """Test basic content extraction."""
        result = self.extractor.extract_content(self.test_html)
        
        # Verify that AI service was called
        self.mock_ai_service.generate_completion.assert_called_once()
        
        # Check that we got parsed JSON results
        self.assertIsInstance(result, dict)
        self.assertEqual(result["title"], "Test Product")
        self.assertEqual(result["price"], 99.99)
    
    def test_extract_content_with_schema(self):
        """Test content extraction with a provided schema."""
        result = self.extractor.extract_content(
            self.test_html,
            extraction_schema=self.test_schema
        )
        
        # Verify schema was used in prompt generation
        prompt_arg = self.mock_ai_service.generate_completion.call_args[0][0]
        self.assertIn(json.dumps(self.test_schema), prompt_arg)
        
        # Check results
        self.assertIsInstance(result, dict)
        self.assertEqual(result["title"], "Test Product")
    
    def test_extract_with_content_type(self):
        """Test extraction with different content types."""
        # Test with product type
        self.extractor.extract_content(
            self.test_html,
            content_type="product"
        )
        
        # Verify product-specific prompt was used
        prompt_arg = self.mock_ai_service.generate_completion.call_args[0][0]
        self.assertIn("product", prompt_arg.lower())
        
        # Test with article type
        self.mock_ai_service.generate_completion.reset_mock()
        self.extractor.extract_content(
            self.test_html,
            content_type="article"
        )
        
        # Verify article-specific prompt was used
        prompt_arg = self.mock_ai_service.generate_completion.call_args[0][0]
        self.assertIn("article", prompt_arg.lower())
    
    def test_extract_with_custom_model(self):
        """Test extraction with custom model selection."""
        self.extractor.extract_content(
            self.test_html,
            model="gpt-3.5-turbo"
        )
        
        # Verify custom model was used
        model_arg = self.mock_ai_service.generate_completion.call_args[1]["model"]
        self.assertEqual(model_arg, "gpt-3.5-turbo")
    
    def test_handle_extraction_error(self):
        """Test error handling during extraction."""
        # Set up AI service to raise an exception
        self.mock_ai_service.generate_completion.side_effect = Exception("AI service error")
        
        # Call extract_content and verify error handling
        result = self.extractor.extract_content(self.test_html)
        
        # Should return empty dict on error
        self.assertEqual(result, {})
    
    def test_handle_invalid_json_response(self):
        """Test handling of invalid JSON responses."""
        # Set up AI service to return invalid JSON
        self.mock_ai_service.generate_completion.return_value = "Not valid JSON!"
        
        # Call extract_content and verify error handling
        result = self.extractor.extract_content(self.test_html)
        
        # Should return empty dict on error
        self.assertEqual(result, {})
    
    def test_preprocessing(self):
        """Test content preprocessing."""
        # Test with HTML that needs preprocessing
        complex_html = """
        <html>
            <head><title>Test</title></head>
            <body>
                <script>console.log('test');</script>
                <div><h1>Product Title</h1></div>
            </body>
        </html>
        """
        
        self.extractor.extract_content(complex_html)
        
        # Verify preprocessing occurred (scripts removed)
        prompt_arg = self.mock_ai_service.generate_completion.call_args[0][0]
        self.assertNotIn("<script>", prompt_arg)
        self.assertIn("<h1>Product Title</h1>", prompt_arg)
    
    @patch('extraction.semantic_extractor.time.sleep', return_value=None)
    def test_extraction_with_retry(self, mock_sleep):
        """Test extraction with retry mechanism."""
        # First call fails, second succeeds
        self.mock_ai_service.generate_completion.side_effect = [
            Exception("Temporary error"),
            json.dumps({"title": "Retry Success"})
        ]
        
        result = self.extractor.extract_content(
            self.test_html,
            max_retries=2,
            retry_delay=1
        )
        
        # Verify retry occurred
        self.assertEqual(self.mock_ai_service.generate_completion.call_count, 2)
        mock_sleep.assert_called_once_with(1)
        
        # Check final result
        self.assertEqual(result["title"], "Retry Success")
    
    def test_extraction_with_fallback_model(self):
        """Test extraction with model fallback."""
        # Configure model selector for fallback scenario
        self.mock_model_selector.select_fallback_model.return_value = "gpt-3.5-turbo"
        
        # First call with primary model fails
        self.mock_ai_service.generate_completion.side_effect = [
            Exception("Model capacity error"),
            json.dumps({"title": "Fallback Success"})
        ]
        
        result = self.extractor.extract_content(
            self.test_html,
            use_fallback=True
        )
        
        # Verify fallback occurred
        self.mock_model_selector.select_fallback_model.assert_called_once()
        
        # Check that second call used fallback model
        fallback_call = self.mock_ai_service.generate_completion.call_args_list[1]
        self.assertEqual(fallback_call[1]["model"], "gpt-3.5-turbo")
        
        # Check final result
        self.assertEqual(result["title"], "Fallback Success")
    
    def test_async_extraction(self):
        """Test asynchronous extraction."""
        # Run async method through event loop
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            self.extractor.extract_content_async(self.test_html)
        )
        
        # Verify async method was called
        self.mock_ai_service.generate_completion_async.assert_called_once()
        
        # Check results
        self.assertIsInstance(result, dict)
        self.assertEqual(result["title"], "Test Product")
    
    def test_batch_extraction(self):
        """Test batch extraction capability."""
        # Test data
        html_batch = [
            "<div>Product 1</div>",
            "<div>Product 2</div>",
            "<div>Product 3</div>"
        ]
        
        # Configure async mock for batch processing
        async def async_side_effect(*args, **kwargs):
            # Return different responses based on input content
            content = args[0]
            if "Product 1" in content:
                return json.dumps({"title": "Product 1"})
            elif "Product 2" in content:
                return json.dumps({"title": "Product 2"})
            else:
                return json.dumps({"title": "Product 3"})
        
        self.mock_ai_service.generate_completion_async.side_effect = async_side_effect
        
        # Run batch extraction
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(
            self.extractor.extract_batch(html_batch)
        )
        
        # Verify we got 3 results
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["title"], "Product 1")
        self.assertEqual(results[1]["title"], "Product 2")
        self.assertEqual(results[2]["title"], "Product 3")
    
    def test_extraction_with_confidence_scores(self):
        """Test extraction with confidence scoring."""
        # Configure AI service to return response with confidence scores
        self.mock_ai_service.generate_completion.return_value = json.dumps({
            "title": {"value": "Test Product", "confidence": 0.95},
            "price": {"value": 99.99, "confidence": 0.80},
            "description": {"value": "A test product description", "confidence": 0.70}
        })
        
        result = self.extractor.extract_content(
            self.test_html,
            include_confidence=True
        )
        
        # Verify confidence scores are included
        self.assertEqual(result["title"]["value"], "Test Product")
        self.assertEqual(result["title"]["confidence"], 0.95)
        self.assertEqual(result["price"]["value"], 99.99)
        self.assertEqual(result["price"]["confidence"], 0.80)
        
        # Verify confidence threshold works
        filtered_result = self.extractor.extract_content(
            self.test_html,
            include_confidence=True,
            confidence_threshold=0.90
        )
        
        # Only title should be included as it exceeds the threshold
        self.assertIn("title", filtered_result)
        self.assertNotIn("description", filtered_result)
    

if __name__ == "__main__":
    unittest.main()