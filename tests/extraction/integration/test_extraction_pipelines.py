"""
Integration tests for SmartScrape extraction pipelines.

These tests validate the complete extraction pipeline flow, including
content preprocessing, extraction, normalization, validation, and error handling.
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

# Add parent directory to path to allow importing from SmartScrape modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Import SmartScrape components
from extraction.core.extraction_interface import BaseExtractor
from extraction.pattern_extractor import DOMPatternExtractor
from extraction.semantic_extractor import AISemanticExtractor
from extraction.schema_manager import SchemaManager

from core.pipeline.pipeline_interface import Pipeline, PipelineStage, PipelineContext
from core.pipeline.registry import PipelineRegistry
from core.pipeline.factory import PipelineFactory

from extraction.stages.structural_analysis_stage import StructuralAnalysisStage
from extraction.stages.metadata_extraction_stage import MetadataExtractionStage
from extraction.stages.pattern_extraction_stage import PatternExtractionStage
from extraction.stages.semantic_extraction_stage import SemanticExtractionStage
from extraction.stages.content_normalization_stage import ContentNormalizationStage
from extraction.stages.quality_evaluation_stage import QualityEvaluationStage
from extraction.stages.schema_validation_stage import SchemaValidationStage

from strategies.core.strategy_context import StrategyContext

from core.circuit_breaker import CircuitBreakerManager, OpenCircuitError
from core.retry_manager import RetryManager, RetryableError
from core.error_classifier import ErrorClassifier

# Define path to test fixtures
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), '../fixtures')

def load_fixture(filename: str) -> str:
    """Load a test fixture from the fixtures directory."""
    fixture_path = os.path.join(FIXTURES_DIR, filename)
    try:
        with open(fixture_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        # If fixture doesn't exist, create a minimal example for testing
        if filename.endswith('.html'):
            return f"""
            <html>
            <head><title>Test {filename}</title></head>
            <body>
                <h1>Test Content for {filename}</h1>
                <p>This is a placeholder for {filename}</p>
            </body>
            </html>
            """
        elif filename.endswith('.json'):
            return '{}'
        else:
            return f"Test content for {filename}"

class MockAIService:
    """Mock for AI Service to use in tests."""
    
    def __init__(self, response=None, error=None):
        self.response = response or {"content": {"title": "Mocked Title", "price": 99.99}}
        self.error = error
        self.calls = []
    
    async def generate_response(self, prompt, options=None):
        """Generate a mock response or raise an error."""
        self.calls.append({"prompt": prompt, "options": options})
        
        if self.error:
            raise self.error
        
        return self.response

class MockHTMLService:
    """Mock for HTML Service to use in tests."""
    
    def __init__(self, error=None):
        self.error = error
        self.calls = []
    
    def clean_html(self, html):
        """Clean HTML content."""
        self.calls.append({"method": "clean_html", "html": html[:100]})
        
        if self.error:
            raise self.error
        
        return html
    
    def extract_main_content(self, html):
        """Extract main content from HTML."""
        self.calls.append({"method": "extract_main_content", "html": html[:100]})
        
        if self.error:
            raise self.error
        
        return html
    
    def generate_selector(self, element, optimized=False):
        """Generate a CSS selector for an element."""
        self.calls.append({"method": "generate_selector", "element": str(element)[:100]})
        
        if self.error:
            raise self.error
        
        if isinstance(element, str):
            return f"div.{element}"
        
        # If it's a BS4 element, use its name
        tag_name = getattr(element, 'name', 'div')
        return f"{tag_name}.test-selector"

def create_mock_strategy_context():
    """Create a mock StrategyContext with required services."""
    # Create mock services
    services = {
        "ai_service": MockAIService(),
        "html_service": MockHTMLService(),
        "circuit_breaker_manager": MagicMock(spec=CircuitBreakerManager),
        "retry_manager": MagicMock(spec=RetryManager),
        "error_classifier": MagicMock(spec=ErrorClassifier),
        "schema_manager": MagicMock(spec=SchemaManager),
        "content_processor": MagicMock(),
        "model_selector": MagicMock(),
        "fallback_framework": MagicMock()
    }
    
    # Mock the circuit breaker decorator
    def mock_circuit_breaker(circuit_name):
        def decorator(func):
            async def wrapper(*args, **kwargs):
                return await func(*args, **kwargs)
            return wrapper
        return decorator
    
    services["circuit_breaker_manager"].circuit_breaker = mock_circuit_breaker
    
    # Create the mock context
    context = MagicMock(spec=StrategyContext)
    
    # Set up get_service to return appropriate mock
    context.get_service = lambda service_name: services.get(service_name)
    
    return context, services

class MockPipelineStage(PipelineStage):
    """Mock pipeline stage for testing."""
    
    def __init__(self, name="mock_stage", succeeds=True, result=None, error=None):
        super().__init__()
        self._name = name
        self.succeeds = succeeds
        self.result = result or {}
        self.error = error
        self.process_called = 0
    
    @property
    def name(self):
        return self._name
    
    async def process(self, data, context):
        """Process data in this stage."""
        self.process_called += 1
        
        if not self.succeeds:
            if self.error:
                raise self.error
            raise Exception(f"Mock stage {self.name} failed")
        
        # Merge our result into data
        data.update(self.result)
        return data

class MockPipeline(Pipeline):
    """Mock pipeline for testing."""
    
    def __init__(self, name="mock_pipeline", stages=None):
        super().__init__(name=name)
        self.stages = stages or []
        self.execute_called = 0
    
    def initialize(self, context):
        """Initialize the pipeline."""
        self._initialized = True
        for stage in self.stages:
            stage.initialize({})
    
    async def execute(self, data, context):
        """Execute the pipeline on the given data."""
        self.execute_called += 1
        
        result = dict(data)
        for stage in self.stages:
            try:
                result = await stage.process(result, context)
            except Exception as e:
                # Handle the error (could delegate to specific error handler)
                result["_error"] = str(e)
                result["_failed_stage"] = stage.name
                break
        
        return result

class TestExtractionPipelines(unittest.TestCase):
    """Test suite for Extraction Pipelines."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.strategy_context, self.services = create_mock_strategy_context()
        self.pipeline_context = PipelineContext(strategy_context=self.strategy_context)
        
        # Load test HTML fixtures
        self.product_html = load_fixture("product_page.html")
        self.article_html = load_fixture("article_page.html")
        self.listing_html = load_fixture("listing_page.html")
        self.mixed_html = load_fixture("mixed_content_page.html")
        self.minimal_html = load_fixture("minimal_page.html")
        self.malformed_html = load_fixture("malformed_page.html")
        
        # Load schema fixtures
        self.product_schema = {
            "type": "object",
            "required": ["title", "price"],
            "properties": {
                "title": {"type": "string"},
                "price": {"type": "number"},
                "description": {"type": "string"},
                "features": {"type": "array", "items": {"type": "string"}}
            }
        }
        
        self.article_schema = {
            "type": "object",
            "required": ["title", "content"],
            "properties": {
                "title": {"type": "string"},
                "content": {"type": "string"},
                "author": {"type": "string"},
                "date": {"type": "string", "format": "date"}
            }
        }
        
        self.listing_schema = {
            "type": "object",
            "required": ["items"],
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "price": {"type": "number"},
                            "url": {"type": "string"}
                        }
                    }
                },
                "pagination": {"type": "object"}
            }
        }
        
        # Mock schema manager to return schemas
        self.services["schema_manager"].get_schema.side_effect = lambda name: {
            "product": self.product_schema,
            "article": self.article_schema,
            "listing": self.listing_schema
        }.get(name, {})
    
    def tearDown(self):
        """Clean up after tests."""
        self.strategy_context = None
        self.services = None
        self.pipeline_context = None
    
    @pytest.mark.asyncio
    async def test_pipeline_registry_initialization(self):
        """Test that the pipeline registry correctly initializes and registers components."""
        # Create a registry
        registry = PipelineRegistry()
        
        # Register extraction stages
        registry.register_stage("structural_analysis", StructuralAnalysisStage)
        registry.register_stage("metadata_extraction", MetadataExtractionStage)
        registry.register_stage("pattern_extraction", PatternExtractionStage)
        registry.register_stage("semantic_extraction", SemanticExtractionStage)
        registry.register_stage("content_normalization", ContentNormalizationStage)
        registry.register_stage("quality_evaluation", QualityEvaluationStage)
        registry.register_stage("schema_validation", SchemaValidationStage)
        
        # Register extraction pipelines
        registry.register_pipeline_config(
            "product_extraction",
            [
                "structural_analysis",
                "metadata_extraction",
                "pattern_extraction",
                "semantic_extraction",
                "content_normalization",
                "quality_evaluation",
                "schema_validation"
            ],
            {
                "schema": "product"
            }
        )
        
        # Create pipeline factory with the registry
        factory = PipelineFactory(registry)
        
        # Create a pipeline
        pipeline = factory.create_pipeline("product_extraction")
        
        # Verify pipeline was created with the correct stages
        self.assertIsNotNone(pipeline)
        self.assertEqual(len(pipeline.stages), 7)
        self.assertEqual(pipeline.name, "product_extraction")
        
        # Verify stages are in the correct order
        stage_names = [stage.name for stage in pipeline.stages]
        self.assertEqual(stage_names, [
            "structural_analysis",
            "metadata_extraction",
            "pattern_extraction",
            "semantic_extraction",
            "content_normalization",
            "quality_evaluation",
            "schema_validation"
        ])
    
    @pytest.mark.asyncio
    async def test_product_page_extraction(self):
        """Test extraction from a product page."""
        # Create mock stages for product extraction
        structural_stage = MockPipelineStage(
            name="structural_analysis",
            result={"content_type": "product", "structure": {"main_content": "#main"}}
        )
        
        pattern_stage = MockPipelineStage(
            name="pattern_extraction",
            result={"extracted_data": {"title": "Test Product", "price": 99.99}}
        )
        
        semantic_stage = MockPipelineStage(
            name="semantic_extraction",
            result={"extracted_data": {
                "title": "Test Product Enhanced",
                "price": 99.99,
                "description": "This is a test product description with AI enhancement.",
                "features": ["Feature 1", "Feature 2"]
            }}
        )
        
        validation_stage = MockPipelineStage(
            name="schema_validation",
            result={"validation_result": {"valid": True}}
        )
        
        # Create product extraction pipeline
        pipeline = MockPipeline(
            name="product_extraction",
            stages=[structural_stage, pattern_stage, semantic_stage, validation_stage]
        )
        pipeline.initialize(self.pipeline_context)
        
        # Execute the pipeline with product HTML
        result = await pipeline.execute({
            "content": self.product_html,
            "url": "https://example.com/product/123"
        }, self.pipeline_context)
        
        # Verify all stages were called
        self.assertEqual(structural_stage.process_called, 1)
        self.assertEqual(pattern_stage.process_called, 1)
        self.assertEqual(semantic_stage.process_called, 1)
        self.assertEqual(validation_stage.process_called, 1)
        
        # Verify extraction result
        self.assertIn("extracted_data", result)
        self.assertEqual(result["extracted_data"]["title"], "Test Product Enhanced")
        self.assertEqual(result["extracted_data"]["price"], 99.99)
        self.assertIn("description", result["extracted_data"])
        self.assertIn("features", result["extracted_data"])
        self.assertEqual(len(result["extracted_data"]["features"]), 2)
        
        # Verify content type identification
        self.assertEqual(result["content_type"], "product")
        
        # Verify validation result
        self.assertIn("validation_result", result)
        self.assertTrue(result["validation_result"]["valid"])
    
    @pytest.mark.asyncio
    async def test_article_extraction(self):
        """Test extraction from an article page."""
        # Create mock stages for article extraction
        structural_stage = MockPipelineStage(
            name="structural_analysis",
            result={"content_type": "article", "structure": {"main_content": ".article"}}
        )
        
        metadata_stage = MockPipelineStage(
            name="metadata_extraction",
            result={"metadata": {
                "title": "Test Article",
                "author": "Test Author",
                "published_date": "2023-01-01"
            }}
        )
        
        semantic_stage = MockPipelineStage(
            name="semantic_extraction",
            result={"extracted_data": {
                "title": "Test Article",
                "content": "This is a test article content.",
                "author": "Test Author",
                "date": "2023-01-01"
            }}
        )
        
        normalization_stage = MockPipelineStage(
            name="content_normalization",
            result={"normalized_data": {
                "title": "Test Article",
                "content": "This is a test article content.",
                "author": "Test Author",
                "date": "2023-01-01"  # Normalized date format
            }}
        )
        
        # Create article extraction pipeline
        pipeline = MockPipeline(
            name="article_extraction",
            stages=[structural_stage, metadata_stage, semantic_stage, normalization_stage]
        )
        pipeline.initialize(self.pipeline_context)
        
        # Execute the pipeline with article HTML
        result = await pipeline.execute({
            "content": self.article_html,
            "url": "https://example.com/article/123"
        }, self.pipeline_context)
        
        # Verify all stages were called
        self.assertEqual(structural_stage.process_called, 1)
        self.assertEqual(metadata_stage.process_called, 1)
        self.assertEqual(semantic_stage.process_called, 1)
        self.assertEqual(normalization_stage.process_called, 1)
        
        # Verify extraction result contains article data
        self.assertIn("extracted_data", result)
        self.assertEqual(result["extracted_data"]["title"], "Test Article")
        self.assertIn("content", result["extracted_data"])
        self.assertEqual(result["extracted_data"]["author"], "Test Author")
        self.assertEqual(result["extracted_data"]["date"], "2023-01-01")
        
        # Verify content type identification
        self.assertEqual(result["content_type"], "article")
        
        # Verify normalized data
        self.assertIn("normalized_data", result)
        self.assertEqual(result["normalized_data"]["date"], "2023-01-01")
    
    @pytest.mark.asyncio
    async def test_listing_page_extraction(self):
        """Test extraction from a listing page."""
        # Create mock stages for listing extraction
        structural_stage = MockPipelineStage(
            name="structural_analysis",
            result={
                "content_type": "listing", 
                "structure": {
                    "listing_container": ".products",
                    "item_selector": ".product-item",
                    "pagination": ".pagination"
                }
            }
        )
        
        pattern_stage = MockPipelineStage(
            name="pattern_extraction",
            result={"extracted_data": {
                "items": [
                    {"title": "Product 1", "price": 19.99, "url": "/product/1"},
                    {"title": "Product 2", "price": 29.99, "url": "/product/2"},
                    {"title": "Product 3", "price": 39.99, "url": "/product/3"}
                ],
                "pagination": {
                    "current_page": 1,
                    "total_pages": 5,
                    "next_page_url": "/products?page=2"
                }
            }}
        )
        
        validation_stage = MockPipelineStage(
            name="schema_validation",
            result={"validation_result": {"valid": True}}
        )
        
        # Create listing extraction pipeline
        pipeline = MockPipeline(
            name="listing_extraction",
            stages=[structural_stage, pattern_stage, validation_stage]
        )
        pipeline.initialize(self.pipeline_context)
        
        # Execute the pipeline with listing HTML
        result = await pipeline.execute({
            "content": self.listing_html,
            "url": "https://example.com/products"
        }, self.pipeline_context)
        
        # Verify all stages were called
        self.assertEqual(structural_stage.process_called, 1)
        self.assertEqual(pattern_stage.process_called, 1)
        self.assertEqual(validation_stage.process_called, 1)
        
        # Verify extraction result contains listing data
        self.assertIn("extracted_data", result)
        self.assertIn("items", result["extracted_data"])
        self.assertEqual(len(result["extracted_data"]["items"]), 3)
        self.assertEqual(result["extracted_data"]["items"][0]["title"], "Product 1")
        self.assertEqual(result["extracted_data"]["items"][1]["price"], 29.99)
        
        # Verify pagination data
        self.assertIn("pagination", result["extracted_data"])
        self.assertEqual(result["extracted_data"]["pagination"]["total_pages"], 5)
        
        # Verify content type identification
        self.assertEqual(result["content_type"], "listing")
        
        # Verify validation result
        self.assertIn("validation_result", result)
        self.assertTrue(result["validation_result"]["valid"])
    
    @pytest.mark.asyncio
    async def test_mixed_content_extraction(self):
        """Test extraction from a page with mixed content types."""
        # Create mock stages for mixed content extraction
        structural_stage = MockPipelineStage(
            name="structural_analysis",
            result={
                "content_type": "mixed", 
                "structure": {
                    "product_section": ".featured-product",
                    "article_section": ".blog-preview",
                    "listing_section": ".related-products"
                }
            }
        )
        
        # Mixed content extractor that can handle multiple content types
        mixed_extractor_stage = MockPipelineStage(
            name="mixed_content_extraction",
            result={"extracted_data": {
                "product": {
                    "title": "Featured Product",
                    "price": 99.99,
                    "description": "Featured product description."
                },
                "article": {
                    "title": "Related Article",
                    "excerpt": "Article preview text...",
                    "url": "/blog/article"
                },
                "related_products": [
                    {"title": "Related Product 1", "price": 19.99},
                    {"title": "Related Product 2", "price": 29.99}
                ]
            }}
        )
        
        # Create mixed content extraction pipeline
        pipeline = MockPipeline(
            name="mixed_content_extraction",
            stages=[structural_stage, mixed_extractor_stage]
        )
        pipeline.initialize(self.pipeline_context)
        
        # Execute the pipeline with mixed content HTML
        result = await pipeline.execute({
            "content": self.mixed_html,
            "url": "https://example.com/featured"
        }, self.pipeline_context)
        
        # Verify all stages were called
        self.assertEqual(structural_stage.process_called, 1)
        self.assertEqual(mixed_extractor_stage.process_called, 1)
        
        # Verify extraction result contains mixed content data
        self.assertIn("extracted_data", result)
        self.assertIn("product", result["extracted_data"])
        self.assertIn("article", result["extracted_data"])
        self.assertIn("related_products", result["extracted_data"])
        
        # Verify product data
        self.assertEqual(result["extracted_data"]["product"]["title"], "Featured Product")
        self.assertEqual(result["extracted_data"]["product"]["price"], 99.99)
        
        # Verify article data
        self.assertEqual(result["extracted_data"]["article"]["title"], "Related Article")
        
        # Verify related products data
        self.assertEqual(len(result["extracted_data"]["related_products"]), 2)
        
        # Verify content type identification
        self.assertEqual(result["content_type"], "mixed")
    
    @pytest.mark.asyncio
    async def test_minimal_content_extraction(self):
        """Test extraction from a page with minimal content."""
        # Create mock stages for minimal content extraction
        structural_stage = MockPipelineStage(
            name="structural_analysis",
            result={"content_type": "unknown", "structure": {}}
        )
        
        # Generic extractor for minimal content
        generic_extractor_stage = MockPipelineStage(
            name="generic_extraction",
            result={"extracted_data": {
                "title": "Minimal Page",
                "text_content": "This is a minimal page with little content."
            }}
        )
        
        # Create minimal content extraction pipeline
        pipeline = MockPipeline(
            name="minimal_content_extraction",
            stages=[structural_stage, generic_extractor_stage]
        )
        pipeline.initialize(self.pipeline_context)
        
        # Execute the pipeline with minimal HTML
        result = await pipeline.execute({
            "content": self.minimal_html,
            "url": "https://example.com/minimal"
        }, self.pipeline_context)
        
        # Verify all stages were called
        self.assertEqual(structural_stage.process_called, 1)
        self.assertEqual(generic_extractor_stage.process_called, 1)
        
        # Verify extraction result contains some basic data despite minimal content
        self.assertIn("extracted_data", result)
        self.assertIn("title", result["extracted_data"])
        self.assertIn("text_content", result["extracted_data"])
        
        # Verify content type is unknown for minimal content
        self.assertEqual(result["content_type"], "unknown")
    
    @pytest.mark.asyncio
    async def test_malformed_html_extraction(self):
        """Test extraction from a page with malformed HTML."""
        # Create mock stages for malformed HTML extraction
        error_handler_stage = MockPipelineStage(
            name="error_handler",
            result={"error_handled": True, "error_type": "malformed_html"}
        )
        
        # Extractor that can handle malformed HTML
        robust_extractor_stage = MockPipelineStage(
            name="robust_extraction",
            result={"extracted_data": {
                "title": "Partial Title",
                "text_fragments": ["Extracted text fragment 1", "Extracted text fragment 2"]
            }}
        )
        
        # Create robust extraction pipeline for malformed HTML
        pipeline = MockPipeline(
            name="malformed_html_extraction",
            stages=[error_handler_stage, robust_extractor_stage]
        )
        pipeline.initialize(self.pipeline_context)
        
        # Execute the pipeline with malformed HTML
        result = await pipeline.execute({
            "content": self.malformed_html,
            "url": "https://example.com/malformed"
        }, self.pipeline_context)
        
        # Verify all stages were called
        self.assertEqual(error_handler_stage.process_called, 1)
        self.assertEqual(robust_extractor_stage.process_called, 1)
        
        # Verify error was handled
        self.assertTrue(result["error_handled"])
        self.assertEqual(result["error_type"], "malformed_html")
        
        # Verify some data was still extracted despite malformed HTML
        self.assertIn("extracted_data", result)
        self.assertIn("title", result["extracted_data"])
        self.assertIn("text_fragments", result["extracted_data"])
        self.assertEqual(len(result["extracted_data"]["text_fragments"]), 2)
    
    @pytest.mark.asyncio
    async def test_integration_with_core_components(self):
        """Test integration between extraction components and core services."""
        # Create stages that use core services
        circuit_breaker_stage = MockPipelineStage(
            name="circuit_breaking_extraction",
            result={"circuit_breaker_used": True}
        )
        
        retry_stage = MockPipelineStage(
            name="retrying_extraction",
            result={"retry_used": True}
        )
        
        error_classification_stage = MockPipelineStage(
            name="error_classifying_extraction",
            result={"error_classified": True}
        )
        
        # Create pipeline that uses core components
        pipeline = MockPipeline(
            name="core_integration_pipeline",
            stages=[circuit_breaker_stage, retry_stage, error_classification_stage]
        )
        pipeline.initialize(self.pipeline_context)
        
        # Execute the pipeline
        result = await pipeline.execute({
            "content": self.product_html,
            "url": "https://example.com/product/123"
        }, self.pipeline_context)
        
        # Verify all stages were called
        self.assertEqual(circuit_breaker_stage.process_called, 1)
        self.assertEqual(retry_stage.process_called, 1)
        self.assertEqual(error_classification_stage.process_called, 1)
        
        # Verify core components were used
        self.assertTrue(result["circuit_breaker_used"])
        self.assertTrue(result["retry_used"])
        self.assertTrue(result["error_classified"])
    
    @pytest.mark.asyncio
    async def test_schema_validation(self):
        """Test schema validation of extraction results."""
        # Create a stage that produces data with schema validation
        extraction_stage = MockPipelineStage(
            name="extraction",
            result={"extracted_data": {
                "title": "Test Product",
                "price": 99.99,
                "description": "Test description"
            }}
        )
        
        # Create a validation stage that applies schema
        validation_stage = MockPipelineStage(
            name="schema_validation",
            result={"validation_result": {
                "valid": True,
                "schema": "product",
                "fields_validated": ["title", "price", "description"]
            }}
        )
        
        # Create pipeline with validation
        pipeline = MockPipeline(
            name="validation_pipeline",
            stages=[extraction_stage, validation_stage]
        )
        pipeline.initialize(self.pipeline_context)
        
        # Set up context with schema
        self.pipeline_context.set("schema", "product")
        
        # Execute the pipeline
        result = await pipeline.execute({
            "content": self.product_html,
            "url": "https://example.com/product/123"
        }, self.pipeline_context)
        
        # Verify stages were called
        self.assertEqual(extraction_stage.process_called, 1)
        self.assertEqual(validation_stage.process_called, 1)
        
        # Verify validation results
        self.assertIn("validation_result", result)
        self.assertTrue(result["validation_result"]["valid"])
        self.assertEqual(result["validation_result"]["schema"], "product")
        self.assertEqual(len(result["validation_result"]["fields_validated"]), 3)
    
    @pytest.mark.asyncio
    async def test_quality_metrics_evaluation(self):
        """Test evaluation of extraction quality metrics."""
        # Create stages for extraction and quality evaluation
        extraction_stage = MockPipelineStage(
            name="extraction",
            result={"extracted_data": {
                "title": "Test Product",
                "price": 99.99,
                # Missing description and features
            }}
        )
        
        quality_stage = MockPipelineStage(
            name="quality_evaluation",
            result={"quality_metrics": {
                "completeness": 0.6,  # 2 out of 4 fields = 50%
                "confidence": 0.85,
                "fields_quality": {
                    "title": 0.9,
                    "price": 0.95
                },
                "overall_quality": 0.75
            }}
        )
        
        # Create pipeline with quality evaluation
        pipeline = MockPipeline(
            name="quality_pipeline",
            stages=[extraction_stage, quality_stage]
        )
        pipeline.initialize(self.pipeline_context)
        
        # Execute the pipeline
        result = await pipeline.execute({
            "content": self.product_html,
            "url": "https://example.com/product/123"
        }, self.pipeline_context)
        
        # Verify stages were called
        self.assertEqual(extraction_stage.process_called, 1)
        self.assertEqual(quality_stage.process_called, 1)
        
        # Verify quality metrics
        self.assertIn("quality_metrics", result)
        self.assertEqual(result["quality_metrics"]["completeness"], 0.6)
        self.assertEqual(result["quality_metrics"]["confidence"], 0.85)
        self.assertEqual(len(result["quality_metrics"]["fields_quality"]), 2)
        self.assertEqual(result["quality_metrics"]["overall_quality"], 0.75)
    
    @pytest.mark.asyncio
    async def test_batch_extraction(self):
        """Test batch extraction functionality."""
        # Create mock batch extraction stage
        batch_stage = MockPipelineStage(
            name="batch_extraction",
            result={"batch_processed": True, "results_count": 3}
        )
        
        # Create pipeline with batch processing
        pipeline = MockPipeline(
            name="batch_pipeline",
            stages=[batch_stage]
        )
        pipeline.initialize(self.pipeline_context)
        
        # Prepare batch of URLs
        batch_data = {
            "urls": [
                "https://example.com/product/1",
                "https://example.com/product/2",
                "https://example.com/product/3"
            ],
            "content_type": "product",
            "batch_id": "test-batch-001"
        }
        
        # Execute the pipeline
        result = await pipeline.execute(batch_data, self.pipeline_context)
        
        # Verify batch stage was called
        self.assertEqual(batch_stage.process_called, 1)
        
        # Verify batch processing results
        self.assertTrue(result["batch_processed"])
        self.assertEqual(result["results_count"], 3)

if __name__ == '__main__':
    pytest.main(['-xvs', __file__])