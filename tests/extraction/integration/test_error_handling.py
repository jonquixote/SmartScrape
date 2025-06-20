"""
Integration tests for SmartScrape extraction error handling.

These tests focus on error scenarios and recovery mechanisms in the extraction framework,
including service failures, timeouts, and invalid inputs.
"""

import os
import sys
import json
import time
import pytest
import unittest
from unittest.mock import Mock, MagicMock, patch
from bs4 import BeautifulSoup
import asyncio
from typing import Dict, Any, List, Optional
import logging

# Add parent directory to path to allow importing from SmartScrape modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Import SmartScrape components
from extraction.core.extraction_interface import BaseExtractor
from extraction.pattern_extractor import DOMPatternExtractor
from extraction.semantic_extractor import AISemanticExtractor
from extraction.schema_manager import SchemaManager

from core.pipeline.pipeline_interface import Pipeline, PipelineStage, PipelineContext
from extraction.stages.semantic_extraction_stage import SemanticExtractionStage
from extraction.stages.pattern_extraction_stage import PatternExtractionStage
from extraction.stages.schema_validation_stage import SchemaValidationStage

from core.circuit_breaker import CircuitBreakerManager, OpenCircuitError
from core.retry_manager import RetryManager, RetryableError
from core.error_classifier import ErrorClassifier
from strategies.core.strategy_context import StrategyContext

# Use the common mocks and utilities
from test_extraction_pipelines import (
    MockAIService, 
    MockHTMLService,
    create_mock_strategy_context,
    MockPipeline,
    load_fixture
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestExtractionErrorHandling(unittest.TestCase):
    """Test suite for extraction error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.strategy_context, self.services = create_mock_strategy_context()
        self.pipeline_context = PipelineContext(strategy_context=self.strategy_context)
        
        # Create services with proper error handling capabilities
        self.services["circuit_breaker_manager"] = self._create_real_circuit_breaker()
        self.services["retry_manager"] = self._create_real_retry_manager()
        
        # Setup semantic extraction stage for testing
        self.semantic_stage = SemanticExtractionStage()
        self.semantic_stage.initialize({})
        
        # Sample HTML content for testing
        self.test_html = """
        <html>
        <head><title>Error Test Page</title></head>
        <body>
            <h1>Test Product</h1>
            <div class="price">$99.99</div>
            <div class="description">This is a test product description.</div>
        </body>
        </html>
        """
    
    def tearDown(self):
        """Clean up after tests."""
        self.strategy_context = None
        self.services = None
        self.pipeline_context = None
        self.semantic_stage = None
    
    def _create_real_circuit_breaker(self):
        """Create a real circuit breaker manager for testing."""
        circuit_breaker = Mock(spec=CircuitBreakerManager)
        
        # Track circuit state
        circuit_states = {"ai_extraction": {"open": False, "failures": 0, "threshold": 3}}
        
        # Implement circuit breaker decorator with actual logic
        def circuit_breaker_decorator(name):
            def decorator(func):
                async def wrapper(*args, **kwargs):
                    if circuit_states[name]["open"]:
                        raise OpenCircuitError(f"Circuit {name} is open")
                    
                    try:
                        result = await func(*args, **kwargs)
                        # Reset failure count on success
                        circuit_states[name]["failures"] = 0
                        return result
                    except Exception as e:
                        # Increment failure count
                        circuit_states[name]["failures"] += 1
                        if circuit_states[name]["failures"] >= circuit_states[name]["threshold"]:
                            circuit_states[name]["open"] = True
                            logger.warning(f"Circuit {name} is now open after {circuit_states[name]['failures']} failures")
                        raise e
                return wrapper
            return decorator
        
        circuit_breaker.circuit_breaker = circuit_breaker_decorator
        circuit_breaker.get_circuit_state = lambda name: circuit_states.get(name, {"open": False, "failures": 0})
        circuit_breaker.reset_circuit = lambda name: circuit_states.update({name: {"open": False, "failures": 0, "threshold": 3}})
        
        return circuit_breaker
    
    def _create_real_retry_manager(self):
        """Create a real retry manager for testing."""
        retry_manager = Mock(spec=RetryManager)
        
        # Implement retry decorator with actual logic
        def retry_decorator(max_attempts=3, retry_delay=1, exceptions=None):
            exceptions = exceptions or (Exception,)
            
            def decorator(func):
                async def wrapper(*args, **kwargs):
                    attempts = 0
                    last_error = None
                    
                    while attempts < max_attempts:
                        try:
                            return await func(*args, **kwargs)
                        except exceptions as e:
                            attempts += 1
                            last_error = e
                            if attempts < max_attempts:
                                await asyncio.sleep(retry_delay)
                    
                    # If we get here, all retries failed
                    if last_error:
                        raise last_error
                return wrapper
            return decorator
        
        retry_manager.retry = retry_decorator
        
        return retry_manager
    
    @pytest.mark.asyncio
    async def test_ai_service_failure_recovery(self):
        """Test recovery from AI service failures."""
        # Mock AI service that fails twice and then succeeds
        failure_count = [0]
        
        def failing_ai_service_response(prompt, options):
            failure_count[0] += 1
            if failure_count[0] <= 2:
                raise RetryableError("AI service temporarily unavailable")
            return {"content": {"title": "Test Product", "price": 99.99}}
        
        self.services["ai_service"] = MockAIService(response=failing_ai_service_response)
        
        # Execute the semantic extraction stage
        result = await self.semantic_stage.process({
            "content": self.test_html,
            "content_type": "product",
            "url": "https://example.com/product/123"
        }, self.pipeline_context)
        
        # Verify the result - should eventually succeed after retries
        self.assertIn("extracted_data", result)
        self.assertEqual(result["extracted_data"].get("title"), "Test Product")
        self.assertEqual(result["extracted_data"].get("price"), 99.99)
        self.assertEqual(failure_count[0], 3)  # First two fail, third succeeds
    
    @pytest.mark.asyncio
    async def test_pattern_extraction_failure_recovery(self):
        """Test recovery from pattern extraction failures."""
        # Mock a pattern extraction stage that fails on first attempt
        pattern_stage = PatternExtractionStage()
        pattern_stage.initialize({})
        
        # Patch the extract method to fail on first call
        original_process = pattern_stage.process
        failure_count = [0]
        
        async def failing_process(data, context):
            failure_count[0] += 1
            if failure_count[0] == 1:
                raise Exception("Pattern extraction failed")
            return await original_process(data, context)
        
        pattern_stage.process = failing_process
        
        # Create a pipeline that retries on stage failure
        resilient_pipeline = MockPipeline(
            name="resilient_pipeline",
            stages=[pattern_stage]
        )
        
        # Custom error handler that retries once
        def handle_stage_error(stage, data, error):
            if failure_count[0] <= 1:  # Only retry once
                return True  # Continue pipeline
            return False
        
        resilient_pipeline.handle_stage_error = handle_stage_error
        resilient_pipeline.initialize(self.pipeline_context)
        
        # Execute pipeline
        result = await resilient_pipeline.execute({
            "content": self.test_html,
            "content_type": "product",
            "url": "https://example.com/product/123"
        }, self.pipeline_context)
        
        # Verify that the pipeline continued after error
        self.assertEqual(failure_count[0], 2)  # Should have retried once
        self.assertEqual(len(resilient_pipeline.results), 1)
        self.assertTrue(resilient_pipeline.results[0]["success"])
    
    @pytest.mark.asyncio
    async def test_invalid_schema_handling(self):
        """Test handling of invalid schema definitions."""
        # Mock schema manager to return an invalid schema
        self.services["schema_manager"].get_schema.return_value = {
            "type": "invalid_type",  # Invalid schema type
            "properties": {
                "title": {"type": "string"},
                "price": {"type": "number"}
            }
        }
        
        # Create a pipeline with schema validation
        validation_pipeline = MockPipeline(
            name="validation_pipeline",
            stages=[
                SemanticExtractionStage(),
                SchemaValidationStage()  # Should handle invalid schema
            ]
        )
        
        # Custom error handler that logs errors but continues
        def handle_stage_error(stage, data, error):
            logger.error(f"Stage {stage.name} failed: {str(error)}")
            data["_errors"] = data.get("_errors", []) + [{
                "stage": stage.name,
                "error": str(error)
            }]
            return True  # Continue pipeline
        
        validation_pipeline.handle_stage_error = handle_stage_error
        validation_pipeline.initialize(self.pipeline_context)
        
        # Mock AI service to return valid data
        self.services["ai_service"] = MockAIService(
            response={"content": {"title": "Test Product", "price": 99.99}}
        )
        
        # Execute pipeline with invalid schema
        result = await validation_pipeline.execute({
            "content": self.test_html,
            "content_type": "product",
            "url": "https://example.com/product/123",
            "schema": "invalid_schema"
        }, self.pipeline_context)
        
        # Verify error was captured but pipeline continued
        self.assertIn("_errors", result)
        self.assertEqual(len(result["_errors"]), 1)
        self.assertEqual(result["_errors"][0]["stage"], "schema_validation")
        
        # Extracted data should still be present
        self.assertIn("extracted_data", result)
        self.assertEqual(result["extracted_data"].get("title"), "Test Product")
    
    @pytest.mark.asyncio
    async def test_partial_content_extraction(self):
        """Test extraction from partial or incomplete content."""
        # Test with truncated HTML content
        truncated_html = """
        <html>
        <head><title>Truncated Test</title></head>
        <body>
            <h1>Truncated Product</h1>
            <div class="price">$49.99</div>
            <!-- Content is truncated here -->
        """
        
        # Mock AI service to extract from truncated content
        self.services["ai_service"] = MockAIService(
            response={"content": {
                "title": "Truncated Product",
                "price": 49.99,
                "_partial": True
            }}
        )
        
        # Execute extraction stage with truncated content
        result = await self.semantic_stage.process({
            "content": truncated_html,
            "content_type": "product",
            "url": "https://example.com/truncated"
        }, self.pipeline_context)
        
        # Verify partial extraction succeeded despite truncated content
        self.assertIn("extracted_data", result)
        self.assertEqual(result["extracted_data"].get("title"), "Truncated Product")
        self.assertEqual(result["extracted_data"].get("price"), 49.99)
        self.assertTrue(result["extracted_data"].get("_partial", False))
    
    @pytest.mark.asyncio
    async def test_timeout_recovery(self):
        """Test recovery from timeout errors."""
        # Mock AI service that times out and then succeeds
        timeout_count = [0]
        
        def timeout_then_succeed(prompt, options):
            timeout_count[0] += 1
            if timeout_count[0] == 1:
                raise asyncio.TimeoutError("AI service request timed out")
            return {"content": {"title": "Timeout Test", "price": 59.99}}
        
        self.services["ai_service"] = MockAIService(response=timeout_then_succeed)
        
        # Mock retry manager to recognize timeout errors
        self.services["retry_manager"].retry = lambda max_attempts=3, retry_delay=1, exceptions=None: \
            lambda func: self._create_timeout_retry_wrapper(func, max_attempts, retry_delay)
        
        # Execute extraction with timeout recovery
        result = await self.semantic_stage.process({
            "content": self.test_html,
            "content_type": "product",
            "url": "https://example.com/timeout-test"
        }, self.pipeline_context)
        
        # Verify recovery from timeout
        self.assertEqual(timeout_count[0], 2)  # Should have retried once
        self.assertIn("extracted_data", result)
        self.assertEqual(result["extracted_data"].get("title"), "Timeout Test")
    
    def _create_timeout_retry_wrapper(self, func, max_attempts, retry_delay):
        """Create a wrapper that retries on timeout."""
        async def wrapper(*args, **kwargs):
            attempts = 0
            last_error = None
            
            while attempts < max_attempts:
                try:
                    return await func(*args, **kwargs)
                except (asyncio.TimeoutError, RetryableError) as e:
                    attempts += 1
                    last_error = e
                    if attempts < max_attempts:
                        await asyncio.sleep(retry_delay)
            
            # If we get here, all retries failed
            if last_error:
                raise last_error
        return wrapper
    
    @pytest.mark.asyncio
    async def test_malformed_html_recovery(self):
        """Test recovery from malformed HTML parsing errors."""
        # Malformed HTML content
        malformed_html = """
        <html>
        <head><title>Malformed Page</
        <body>
            <h1>Malformed Test</h1
            <div class="price">$39.99</div>
            <span class="missing-quote attribute="value">Error</span>
        </body>
        </html>
        """
        
        # Create a pipeline with error-tolerant parsing
        resilient_pipeline = MockPipeline(
            name="resilient_pipeline",
            stages=[
                PatternExtractionStage(),  # HTML parsing should be resilient
                SemanticExtractionStage()  # AI extraction as fallback
            ]
        )
        
        # Mock services
        self.services["ai_service"] = MockAIService(
            response={"content": {"title": "Malformed Test", "price": 39.99}}
        )
        
        # HTML service should handle malformed HTML gracefully
        original_clean_html = self.services["html_service"].clean_html
        
        def robust_clean_html(html):
            try:
                # Try to fix common issues
                fixed_html = html.replace("<title>Malformed Page</", "<title>Malformed Page</title>")
                fixed_html = fixed_html.replace("<h1>Malformed Test</h1", "<h1>Malformed Test</h1>")
                fixed_html = fixed_html.replace('span class="missing-quote attribute="value"', 'span class="error" attribute="value"')
                return fixed_html
            except Exception as e:
                logger.warning(f"Error cleaning HTML: {str(e)}")
                return html  # Return original as fallback
        
        self.services["html_service"].clean_html = robust_clean_html
        
        # Execute pipeline with malformed HTML
        result = await resilient_pipeline.execute({
            "content": malformed_html,
            "content_type": "product",
            "url": "https://example.com/malformed"
        }, self.pipeline_context)
        
        # Verify successful extraction despite malformed HTML
        self.assertIn("extracted_data", result)
        self.assertEqual(result["extracted_data"].get("title"), "Malformed Test")
        self.assertEqual(result["extracted_data"].get("price"), 39.99)
        
        # Restore original method
        self.services["html_service"].clean_html = original_clean_html

if __name__ == '__main__':
    pytest.main(['-xvs', __file__])