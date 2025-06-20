"""
Integration tests for SmartScrape extraction fallback mechanisms.

These tests focus on fallback behaviors in the extraction framework, including
graceful degradation, schema relaxation, and alternative extraction paths.
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
from extraction.fallback_framework import (
    ExtractionFallbackRegistry, 
    ExtractionFallbackChain,
    QualityBasedCondition,
    SchemaComplianceCondition,
    FieldSubsetExtractor,
    SchemaRelaxationTransformer
)

from core.pipeline.pipeline_interface import Pipeline, PipelineStage, PipelineContext
from extraction.stages.semantic_extraction_stage import SemanticExtractionStage
from extraction.stages.pattern_extraction_stage import PatternExtractionStage
from extraction.stages.schema_validation_stage import SchemaValidationStage

from core.circuit_breaker import CircuitBreakerManager, OpenCircuitError
from core.retry_manager import RetryManager, RetryableError
from strategies.core.strategy_context import StrategyContext

# Use the common mocks and utilities
from test_extraction_pipelines import (
    MockAIService, 
    MockHTMLService,
    create_mock_strategy_context,
    MockPipeline,
    load_fixture
)

class MockExtractor(BaseExtractor):
    """Mock extractor for testing fallback scenarios."""
    
    def __init__(self, name="MockExtractor", success=True, data=None, quality=0.8, exception=None):
        super().__init__()
        self.name = name
        self.success = success
        self.data = data or {}
        self.quality = quality
        self.exception = exception
        self.extract_called = 0
        self.can_handle_called = 0
    
    def can_handle(self, content, content_type=None):
        """Check if this extractor can handle the given content type."""
        self.can_handle_called += 1
        return True
    
    async def extract(self, content, schema=None, options=None):
        """Extract data from content."""
        self.extract_called += 1
        
        if self.exception:
            raise self.exception
        
        result = {
            "_metadata": {
                "extractor": self.name,
                "quality": self.quality,
                "success": self.success
            }
        }
        
        if self.success:
            result.update(self.data)
        
        return result
    
    def initialize(self):
        """Initialize the extractor."""
        self._initialized = True
    
    def shutdown(self):
        """Clean up resources."""
        self._initialized = False

class TestExtractionFallbackMechanisms(unittest.TestCase):
    """Test suite for extraction fallback mechanisms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.strategy_context, self.services = create_mock_strategy_context()
        self.pipeline_context = PipelineContext(strategy_context=self.strategy_context)
        
        # Sample HTML content for testing
        self.test_html = """
        <html>
        <head><title>Fallback Test Page</title></head>
        <body>
            <h1>Test Product</h1>
            <div class="price">$99.99</div>
            <div class="description">This is a test product description.</div>
            <div class="features">
                <ul>
                    <li>Feature 1</li>
                    <li>Feature 2</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        # Product schema for testing
        self.product_schema = {
            "type": "object",
            "required": ["title", "price"],
            "properties": {
                "title": {"type": "string"},
                "price": {"type": "number"},
                "description": {"type": "string"},
                "features": {"type": "array", "items": {"type": "string"}},
                "brand": {"type": "string"},
                "sku": {"type": "string"}
            }
        }
        
        # Setup the fallback registry
        self.fallback_registry = ExtractionFallbackRegistry()
    
    def tearDown(self):
        """Clean up after tests."""
        self.strategy_context = None
        self.services = None
        self.pipeline_context = None
        self.fallback_registry = None
    
    @pytest.mark.asyncio
    async def test_semantic_to_pattern_fallback(self):
        """Test fallback from semantic extraction to pattern extraction."""
        # Create extractors for the fallback chain
        semantic_extractor = MockExtractor(
            name="SemanticExtractor",
            success=False,  # Will fail
            exception=RetryableError("AI service unavailable")
        )
        
        pattern_extractor = MockExtractor(
            name="PatternExtractor",
            success=True,  # Will succeed
            data={
                "title": "Test Product",
                "price": 99.99,
                "description": "This is a test product description."
            },
            quality=0.7
        )
        
        # Create and execute fallback chain
        fallback_chain = ExtractionFallbackChain(
            extractors=[semantic_extractor, pattern_extractor],
            fallback_condition=QualityBasedCondition(0.5),
            aggregate_results=False  # Use first successful extractor
        )
        
        # Execute the chain
        result = await fallback_chain.extract(self.test_html, self.product_schema)
        
        # Verify semantic extractor was called but failed
        self.assertEqual(semantic_extractor.extract_called, 1)
        
        # Verify pattern extractor was used as fallback
        self.assertEqual(pattern_extractor.extract_called, 1)
        
        # Verify result is from pattern extractor
        self.assertEqual(result["_metadata"]["extractor"], "PatternExtractor")
        self.assertEqual(result["title"], "Test Product")
        self.assertEqual(result["price"], 99.99)
    
    @pytest.mark.asyncio
    async def test_strict_to_relaxed_schema_fallback(self):
        """Test fallback from strict schema to relaxed schema."""
        # Create a schema relaxation transformer
        relaxation_transformer = SchemaRelaxationTransformer(name="SchemaRelaxer")
        
        # Create a strict extractor that requires all schema fields
        strict_extractor = MockExtractor(
            name="StrictExtractor",
            success=False,  # Will fail due to missing fields
            data={
                "title": "Test Product",
                "price": 99.99,
                # Missing required fields from schema
            },
            quality=0.5
        )
        
        # Create a relaxed extractor that can work with partial data
        relaxed_extractor = MockExtractor(
            name="RelaxedExtractor",
            success=True,  # Will succeed with partial data
            data={
                "title": "Test Product",
                "price": 99.99,
                # Still missing some fields but these are required
            },
            quality=0.6
        )
        
        # Create fallback chain with schema compliance condition
        fallback_chain = ExtractionFallbackChain(
            extractors=[strict_extractor, relaxation_transformer, relaxed_extractor],
            fallback_condition=SchemaComplianceCondition(["title", "price"]),  # Only require these fields
            aggregate_results=False
        )
        
        # Execute the chain
        result = await fallback_chain.extract(self.test_html, self.product_schema)
        
        # Verify strict extractor was tried but failed
        self.assertEqual(strict_extractor.extract_called, 1)
        
        # Verify relaxed extractor was used as fallback
        self.assertEqual(relaxed_extractor.extract_called, 1)
        
        # Verify the schema relaxation occurred
        self.assertEqual(result["_metadata"]["extractor"], "RelaxedExtractor")
        self.assertTrue("_schema_relaxed" in result["_metadata"] or 
                       "relaxed" in result["_metadata"] or
                       "fallback" in result["_metadata"])
    
    @pytest.mark.asyncio
    async def test_field_subset_extraction(self):
        """Test extraction of a subset of fields when full extraction fails."""
        # Create field subset extractor
        complete_extractor = MockExtractor(
            name="CompleteExtractor",
            success=False,  # Will fail to extract all fields
            exception=ValueError("Cannot extract all fields")
        )
        
        # Critical fields to extract
        critical_fields = ["title", "price"]
        
        # Create a subset extractor focusing on critical fields
        subset_extractor = FieldSubsetExtractor(
            core_extractor=MockExtractor(
                name="SubsetExtractor",
                success=True,
                data={
                    "title": "Test Product",
                    "price": 99.99
                },
                quality=0.9
            ),
            critical_fields=critical_fields
        )
        
        # Create fallback chain
        fallback_chain = ExtractionFallbackChain(
            extractors=[complete_extractor, subset_extractor],
            fallback_condition=QualityBasedCondition(0.5),
            aggregate_results=False
        )
        
        # Execute the chain
        result = await fallback_chain.extract(self.test_html, self.product_schema)
        
        # Verify complete extractor was tried but failed
        self.assertEqual(complete_extractor.extract_called, 1)
        
        # Verify result contains only critical fields
        self.assertIn("title", result)
        self.assertIn("price", result)
        self.assertNotIn("description", result)  # Not in critical fields
        
        # Verify metadata indicates subset extraction
        self.assertIn("_metadata", result)
        self.assertTrue(any(key in result["_metadata"] for key in 
                          ["subset", "critical_fields", "partial"]))
    
    @pytest.mark.asyncio
    async def test_format_simplification_fallback(self):
        """Test fallback to simpler data format when complex format fails."""
        # Create extractors for different format complexities
        complex_extractor = MockExtractor(
            name="ComplexExtractor",
            success=False,  # Will fail to extract complex format
            exception=ValueError("Cannot parse complex structure")
        )
        
        # Simple flat structure extractor
        simple_extractor = MockExtractor(
            name="SimpleExtractor",
            success=True,  # Will succeed with simpler format
            data={
                "title": "Test Product",
                "price": 99.99,
                "description": "This is a test product description.",
                "features_text": "Feature 1, Feature 2"  # Simplified format
            },
            quality=0.7
        )
        
        # Create fallback chain
        fallback_chain = ExtractionFallbackChain(
            extractors=[complex_extractor, simple_extractor],
            fallback_condition=QualityBasedCondition(0.5),
            aggregate_results=False
        )
        
        # Execute the chain
        result = await fallback_chain.extract(self.test_html, self.product_schema)
        
        # Verify complex extractor was tried but failed
        self.assertEqual(complex_extractor.extract_called, 1)
        
        # Verify simple extractor was used as fallback
        self.assertEqual(simple_extractor.extract_called, 1)
        
        # Verify result has simplified format
        self.assertEqual(result["title"], "Test Product")
        self.assertEqual(result["price"], 99.99)
        self.assertIn("features_text", result)  # Simplified format
        self.assertNotIn("features", result)  # Complex format not present
    
    @pytest.mark.asyncio
    async def test_extraction_quality_thresholds(self):
        """Test fallback based on extraction quality thresholds."""
        # Create extractors with different quality levels
        low_quality_extractor = MockExtractor(
            name="LowQualityExtractor",
            success=True,
            data={
                "title": "Test Product",
                "price": 99.99,
                # Missing many fields
            },
            quality=0.4  # Below threshold
        )
        
        medium_quality_extractor = MockExtractor(
            name="MediumQualityExtractor",
            success=True,
            data={
                "title": "Test Product",
                "price": 99.99,
                "description": "This is a test product description."
                # More fields but still not complete
            },
            quality=0.7  # Above threshold
        )
        
        high_quality_extractor = MockExtractor(
            name="HighQualityExtractor",
            success=True,
            data={
                "title": "Test Product",
                "price": 99.99,
                "description": "This is a test product description.",
                "features": ["Feature 1", "Feature 2"],
                "brand": "Test Brand"
                # Most complete
            },
            quality=0.9  # High quality
        )
        
        # Create fallback chain with quality threshold
        fallback_chain = ExtractionFallbackChain(
            extractors=[low_quality_extractor, medium_quality_extractor, high_quality_extractor],
            fallback_condition=QualityBasedCondition(0.6),  # Require at least medium quality
            aggregate_results=True  # Combine results
        )
        
        # Execute the chain
        result = await fallback_chain.extract(self.test_html, self.product_schema)
        
        # Verify all extractors were called
        self.assertEqual(low_quality_extractor.extract_called, 1)
        self.assertEqual(medium_quality_extractor.extract_called, 1)
        self.assertEqual(high_quality_extractor.extract_called, 1)
        
        # Verify low quality result was rejected (not meeting threshold)
        self.assertNotEqual(result["_metadata"]["extractor"], "LowQualityExtractor")
        
        # Check if we have fields from both medium and high quality extractors
        # (when aggregate_results=True)
        self.assertIn("title", result)
        self.assertIn("price", result)
        self.assertIn("description", result)
        self.assertIn("features", result)
        self.assertIn("brand", result)
    
    @pytest.mark.asyncio
    async def test_progressive_extraction_degradation(self):
        """Test graceful degradation through a sequence of fallbacks."""
        # Create a chain of extractors with progressive degradation
        extractors = [
            # Primary extractor - fails completely
            MockExtractor(
                name="PrimaryExtractor",
                success=False,
                exception=Exception("Primary extraction failed")
            ),
            
            # Secondary extractor - gets some fields but not all required
            MockExtractor(
                name="SecondaryExtractor",
                success=True,
                data={
                    "title": "Test Product",
                    # Missing price which is required
                },
                quality=0.5
            ),
            
            # Tertiary extractor - gets required fields but low confidence
            MockExtractor(
                name="TertiaryExtractor",
                success=True,
                data={
                    "title": "Test Product",
                    "price": 99.99,
                    # Only required fields
                },
                quality=0.6
            ),
            
            # Quaternary extractor - gets minimal data with high confidence
            MockExtractor(
                name="QuaternaryExtractor",
                success=True,
                data={
                    "title": "Test Product",
                    "price": 99.99,
                    "description": "This is a test product description."
                },
                quality=0.8
            ),
        ]
        
        # Create fallback chain with progressive conditions
        fallback_chain = ExtractionFallbackChain(
            extractors=extractors,
            # Will try all extractors and use best result
            fallback_condition=SchemaComplianceCondition(["title", "price"]),
            aggregate_results=True
        )
        
        # Register with fallback registry to test registry integration
        for i, extractor in enumerate(extractors[:-1]):
            self.fallback_registry.register_fallback(
                extractor.name, 
                extractors[i+1].name
            )
        
        # Execute the chain
        result = await fallback_chain.extract(self.test_html, self.product_schema)
        
        # Verify all extractors were tried
        for extractor in extractors:
            self.assertEqual(extractor.extract_called, 1)
        
        # Verify we got the required fields
        self.assertIn("title", result)
        self.assertIn("price", result)
        
        # Verify metadata shows the progressive fallback
        self.assertIn("_metadata", result)
        self.assertTrue(any(term in str(result["_metadata"]) for term in 
                          ["fallback", "progressive", "degradation", "aggregated"]))
        
        # Verify we have the description from the quaternary extractor
        self.assertIn("description", result)
    
    @pytest.mark.asyncio
    async def test_extraction_chain_with_pipeline_integration(self):
        """Test extraction fallback chain integration with pipeline stages."""
        # Setup a semantic extraction stage that will fail
        semantic_stage = SemanticExtractionStage()
        semantic_stage.initialize({})
        
        # Mock the AI service to fail
        self.services["ai_service"] = MockAIService(
            error=Exception("AI service unavailable")
        )
        
        # Mock the fallback framework to provide a pattern extractor
        pattern_extractor = DOMPatternExtractor()
        self.services["fallback_framework"].get_fallback.return_value = lambda: pattern_extractor
        
        # Create a pipeline with semantic and fallback
        pipeline = MockPipeline(
            name="fallback_pipeline",
            stages=[semantic_stage]
        )
        pipeline.initialize(self.pipeline_context)
        
        # Execute pipeline
        result = await pipeline.execute({
            "content": self.test_html,
            "content_type": "product",
            "url": "https://example.com/product/123"
        }, self.pipeline_context)
        
        # Verify fallback was triggered
        self.assertIn("_metadata", result)
        self.assertEqual(result["_metadata"].get("extraction_method"), "fallback")
        
        # Verify the fallback framework was called
        self.services["fallback_framework"].get_fallback.assert_called_once()
        
        # Verify extraction occurred despite AI service failure
        self.assertIn("extracted_data", result)
    
    @pytest.mark.asyncio
    async def test_fallback_registry_api(self):
        """Test extraction fallback registry API and functionality."""
        # Register some extractors
        extractor_a = MockExtractor(name="ExtractorA", success=False)
        extractor_b = MockExtractor(name="ExtractorB", success=True, data={"test": "value"})
        
        # Register with registry
        self.fallback_registry.register_extractor("ExtractorA", lambda: extractor_a)
        self.fallback_registry.register_extractor("ExtractorB", lambda: extractor_b)
        self.fallback_registry.register_fallback("ExtractorA", "ExtractorB")
        
        # Get fallbacks for ExtractorA
        fallbacks = self.fallback_registry.get_fallbacks_for_extractor("ExtractorA")
        self.assertEqual(fallbacks, ["ExtractorB"])
        
        # Create fallback chain from registry
        chain = self.fallback_registry.create_fallback_chain("ExtractorA")
        
        # Execute the chain
        result = await chain.extract(self.test_html)
        
        # Verify both extractors were used and fallback worked
        self.assertEqual(extractor_a.extract_called, 1)
        self.assertEqual(extractor_b.extract_called, 1)
        self.assertEqual(result["test"], "value")
        
        # Test fallback suggestion
        fallback = self.fallback_registry.suggest_fallback(
            extractor_a, 
            Exception("Test error"), 
            self.test_html
        )
        
        # Verify correct fallback was suggested
        self.assertEqual(fallback.name, "ExtractorB")

if __name__ == '__main__':
    pytest.main(['-xvs', __file__])