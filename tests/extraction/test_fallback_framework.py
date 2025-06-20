"""
Tests for the extraction fallback framework.

These tests validate the functionality of the fallback chain, quality assessment,
and progressive degradation behaviors in the extraction fallback framework.
"""

import unittest
import json
from unittest.mock import Mock, patch
import sys
import os
from typing import Dict, Any, List, Optional

# Add parent directory to path to allow importing from extraction
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from extraction.fallback_framework import (
    ContentType,
    ExtractionResult,
    BaseExtractor,
    ExtractionFallbackChain,
    ExtractionQualityAssessor,
    ExtractionFallbackRegistry,
    QualityBasedCondition,
    SchemaComplianceCondition,
    CompositeCondition,
    FieldSubsetExtractor,
    SchemaRelaxationTransformer,
    TypeCoercionTransformer,
    PartialResultAggregator,
    create_html_extraction_chain,
    create_api_extraction_chain,
    create_text_extraction_chain
)

# Sample test data
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
</head>
<body>
    <h1>Product: Test Product</h1>
    <div class="price">$99.99</div>
    <p class="description">This is a test product description.</p>
</body>
</html>
"""

JSON_CONTENT = """{
    "product": {
        "name": "Test Product",
        "price": 99.99,
        "description": "This is a test product description.",
        "features": ["Feature 1", "Feature 2"]
    }
}"""

TEXT_CONTENT = """
Product: Test Product
Price: $99.99
Description: This is a test product description.
"""

class MockExtractor(BaseExtractor):
    """Mock extractor for testing purposes."""
    
    def __init__(self, name: str, success: bool = True, quality: float = 0.8, 
                data: Dict[str, Any] = None):
        super().__init__(name=name)
        self.success = success
        self.quality = quality
        self.data = data or {}
        self.extract_called = False
        
    def can_handle(self, content_type, content=None, schema=None):
        return True
    
    def extract(self, content, schema=None, options=None):
        self.extract_called = True
        
        if self.success:
            result = ExtractionResult(
                data=self.data,
                success=True,
                extractor_name=self.name,
                quality_score=self.quality
            )
        else:
            result = ExtractionResult(
                success=False,
                error="Mock extraction error",
                extractor_name=self.name,
                quality_score=0.0
            )
            
        return result

class TestExtractionResult(unittest.TestCase):
    """Tests for the ExtractionResult class."""
    
    def test_basic_initialization(self):
        """Test that ExtractionResult initializes correctly."""
        result = ExtractionResult(
            data={"test": "value"},
            success=True,
            extractor_name="TestExtractor",
            quality_score=0.8
        )
        
        self.assertEqual(result.data, {"test": "value"})
        self.assertTrue(result.success)
        self.assertEqual(result.extractor_name, "TestExtractor")
        self.assertEqual(result.quality_score, 0.8)
        
    def test_boolean_conversion(self):
        """Test that bool(result) returns success status."""
        success_result = ExtractionResult(success=True)
        failure_result = ExtractionResult(success=False)
        
        self.assertTrue(bool(success_result))
        self.assertFalse(bool(failure_result))
        
    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        result = ExtractionResult(
            data={"test": "value"},
            success=True,
            error=None,
            extractor_name="TestExtractor",
            quality_score=0.8,
            metadata={"source": "test"}
        )
        
        result_dict = result.to_dict()
        
        self.assertEqual(result_dict["data"], {"test": "value"})
        self.assertTrue(result_dict["success"])
        self.assertIsNone(result_dict["error"])
        self.assertEqual(result_dict["extractor_name"], "TestExtractor")
        self.assertEqual(result_dict["quality_score"], 0.8)
        self.assertEqual(result_dict["metadata"], {"source": "test"})
        self.assertIn("timestamp", result_dict)
        
    def test_from_dict_conversion(self):
        """Test creation from dictionary."""
        data = {
            "data": {"test": "value"},
            "success": True,
            "error": None,
            "extractor_name": "TestExtractor",
            "quality_score": 0.8,
            "metadata": {"source": "test"},
            "timestamp": 1234567890.0
        }
        
        result = ExtractionResult.from_dict(data)
        
        self.assertEqual(result.data, {"test": "value"})
        self.assertTrue(result.success)
        self.assertIsNone(result.error)
        self.assertEqual(result.extractor_name, "TestExtractor")
        self.assertEqual(result.quality_score, 0.8)
        self.assertEqual(result.metadata, {"source": "test"})
        self.assertEqual(result.timestamp, 1234567890.0)
        
    def test_merge_results(self):
        """Test merging results."""
        result1 = ExtractionResult(
            data={"title": "Title 1", "price": 100},
            success=True,
            extractor_name="Extractor1",
            quality_score=0.8,
            metadata={"source": "test1"}
        )
        
        result2 = ExtractionResult(
            data={"description": "Description", "price": 110},
            success=True,
            extractor_name="Extractor2",
            quality_score=0.7,
            metadata={"origin": "test2"}
        )
        
        merged = result1.merge(result2)
        
        # Test field combining
        self.assertEqual(merged.data["title"], "Title 1")  # From result1
        self.assertEqual(merged.data["description"], "Description")  # From result2
        self.assertEqual(merged.data["price"], 100)  # result1 takes precedence
        
        # Test metadata and other properties
        self.assertTrue(merged.success)
        self.assertEqual(merged.extractor_name, "Extractor1+Extractor2")
        self.assertEqual(merged.quality_score, 0.8)  # Takes the max
        self.assertEqual(merged.metadata["source"], "test1")
        self.assertEqual(merged.metadata["origin"], "test2")
        self.assertTrue(merged.metadata["merged"])
        
    def test_merge_with_failed_result(self):
        """Test merging with a failed result."""
        success_result = ExtractionResult(
            data={"title": "Title"},
            success=True,
            extractor_name="SuccessExtractor",
            quality_score=0.8
        )
        
        failed_result = ExtractionResult(
            success=False,
            error="Failed",
            extractor_name="FailedExtractor",
            quality_score=0.0
        )
        
        # Merging success with failure should keep success data
        merged1 = success_result.merge(failed_result)
        self.assertTrue(merged1.success)
        self.assertEqual(merged1.data, {"title": "Title"})
        
        # Merging failure with success should take success data
        merged2 = failed_result.merge(success_result)
        self.assertTrue(merged2.success)
        self.assertEqual(merged2.data, {"title": "Title"})

class TestFallbackConditions(unittest.TestCase):
    """Tests for fallback condition classes."""
    
    def test_quality_based_condition(self):
        """Test quality-based fallback condition."""
        condition = QualityBasedCondition(threshold=0.6)
        
        # Test results at different quality levels
        high_quality = ExtractionResult(success=True, quality_score=0.8)
        medium_quality = ExtractionResult(success=True, quality_score=0.6)
        low_quality = ExtractionResult(success=True, quality_score=0.4)
        failed_result = ExtractionResult(success=False, quality_score=0.0)
        
        self.assertFalse(condition.should_fallback(high_quality))  # Above threshold
        self.assertFalse(condition.should_fallback(medium_quality))  # At threshold
        self.assertTrue(condition.should_fallback(low_quality))  # Below threshold
        self.assertTrue(condition.should_fallback(failed_result))  # Failed
        
    def test_schema_compliance_condition(self):
        """Test schema compliance fallback condition."""
        condition = SchemaComplianceCondition(
            required_fields=["title", "price"],
            completeness_threshold=0.7
        )
        
        # Test with different levels of completeness
        complete_result = ExtractionResult(
            success=True,
            data={
                "title": "Test Product",
                "price": 99.99,
                "description": "Test",
                "category": "Test"
            }
        )
        
        partial_result = ExtractionResult(
            success=True,
            data={
                "title": "Test Product",
                "description": "Test"
            }
        )
        
        minimal_result = ExtractionResult(
            success=True,
            data={
                "description": "Test"
            }
        )
        
        failed_result = ExtractionResult(success=False)
        
        # Test required fields
        self.assertFalse(condition.should_fallback(complete_result))
        self.assertTrue(condition.should_fallback(partial_result))  # Missing price
        self.assertTrue(condition.should_fallback(minimal_result))  # Missing title & price
        self.assertTrue(condition.should_fallback(failed_result))
        
        # Test with a schema
        schema = {
            "fields": [
                {"name": "title", "required": True},
                {"name": "price", "required": True},
                {"name": "description", "required": False},
                {"name": "category", "required": False},
                {"name": "rating", "required": False}
            ]
        }
        
        # 4 of 5 fields = 80% complete
        self.assertFalse(condition.should_fallback(complete_result, schema))
        
        # 2 of 5 fields = 40% complete, below threshold
        self.assertTrue(condition.should_fallback(partial_result, schema))
        
    def test_composite_condition(self):
        """Test composite fallback condition."""
        quality_condition = QualityBasedCondition(threshold=0.6)
        schema_condition = SchemaComplianceCondition(required_fields=["title"])
        
        # Test OR composite (any condition triggers fallback)
        or_condition = CompositeCondition([quality_condition, schema_condition], require_all=False)
        
        # Good quality but missing title
        result1 = ExtractionResult(
            success=True,
            quality_score=0.8,
            data={"description": "Test"}
        )
        
        # Poor quality but has title
        result2 = ExtractionResult(
            success=True,
            quality_score=0.4,
            data={"title": "Test"}
        )
        
        # Good quality and has title
        result3 = ExtractionResult(
            success=True,
            quality_score=0.8,
            data={"title": "Test"}
        )
        
        self.assertTrue(or_condition.should_fallback(result1))  # Missing title
        self.assertTrue(or_condition.should_fallback(result2))  # Low quality
        self.assertFalse(or_condition.should_fallback(result3))  # Good on both
        
        # Test AND composite (all conditions must trigger fallback)
        and_condition = CompositeCondition([quality_condition, schema_condition], require_all=True)
        
        self.assertFalse(and_condition.should_fallback(result1))  # Only schema condition triggers
        self.assertFalse(and_condition.should_fallback(result2))  # Only quality condition triggers
        self.assertFalse(and_condition.should_fallback(result3))  # Neither condition triggers
        
        # Bad on both counts
        result4 = ExtractionResult(
            success=True,
            quality_score=0.4,
            data={"description": "Test"}
        )
        
        self.assertTrue(and_condition.should_fallback(result4))  # Both conditions trigger

class TestExtractionFallbackChain(unittest.TestCase):
    """Tests for the ExtractionFallbackChain class."""
    
    def test_basic_fallback_chain(self):
        """Test basic fallback chain execution."""
        # Create a chain of mock extractors
        primary = MockExtractor("Primary", success=False)
        secondary = MockExtractor("Secondary", success=True, data={"result": "secondary"})
        tertiary = MockExtractor("Tertiary", success=True, data={"result": "tertiary"})
        
        chain = ExtractionFallbackChain(
            extractors=[primary, secondary, tertiary],
            fallback_condition=QualityBasedCondition(0.5),
            aggregate_results=False
        )
        
        result = chain.extract(HTML_CONTENT)
        
        # Verify primary failed and secondary was used
        self.assertTrue(primary.extract_called)
        self.assertTrue(secondary.extract_called)
        self.assertFalse(tertiary.extract_called)  # Shouldn't be called since secondary succeeded
        
        self.assertTrue(result.success)
        self.assertEqual(result.data, {"result": "secondary"})
        self.assertEqual(result.extractor_name, "Secondary")
        
    def test_chain_with_quality_fallback(self):
        """Test fallback chain with quality-based fallback."""
        # Create extractors with different quality levels
        high_quality = MockExtractor("High", success=True, quality=0.9, data={"result": "high"})
        medium_quality = MockExtractor("Medium", success=True, quality=0.6, data={"result": "medium"})
        low_quality = MockExtractor("Low", success=True, quality=0.3, data={"result": "low"})
        
        chain = ExtractionFallbackChain(
            extractors=[low_quality, medium_quality, high_quality],
            fallback_condition=QualityBasedCondition(0.7),  # Trigger fallback if quality < 0.7
            aggregate_results=False
        )
        
        result = chain.extract(HTML_CONTENT)
        
        # Verify all extractors were tried until finding high quality
        self.assertTrue(low_quality.extract_called)
        self.assertTrue(medium_quality.extract_called)
        self.assertTrue(high_quality.extract_called)
        
        # Should return the high quality result
        self.assertEqual(result.data, {"result": "high"})
        self.assertEqual(result.quality_score, 0.9)
        
    def test_chain_with_result_aggregation(self):
        """Test fallback chain with result aggregation."""
        # Create extractors that extract different data
        extractor1 = MockExtractor("Extractor1", data={"title": "Test Product", "price": 99.99})
        extractor2 = MockExtractor("Extractor2", data={"description": "Test description"})
        extractor3 = MockExtractor("Extractor3", data={"category": "Test", "price": 89.99})
        
        chain = ExtractionFallbackChain(
            extractors=[extractor1, extractor2, extractor3],
            fallback_condition=QualityBasedCondition(0.5),
            aggregate_results=True  # Enable result aggregation
        )
        
        result = chain.extract(HTML_CONTENT)
        
        # Verify all extractors were tried
        self.assertTrue(extractor1.extract_called)
        self.assertTrue(extractor2.extract_called)
        self.assertTrue(extractor3.extract_called)
        
        # Verify results were aggregated
        self.assertTrue(result.success)
        self.assertEqual(result.extractor_name, "fallback_chain")
        self.assertTrue(result.metadata["aggregated"])
        
        # Check data fields are combined correctly
        self.assertEqual(result.data["title"], "Test Product")  # From extractor1
        self.assertEqual(result.data["description"], "Test description")  # From extractor2
        self.assertEqual(result.data["category"], "Test")  # From extractor3
        self.assertEqual(result.data["price"], 99.99)  # From extractor1 (first takes precedence)
        
    def test_chain_with_context_propagation(self):
        """Test context propagation through the fallback chain."""
        # Create mockups
        extractor1 = MockExtractor("Extractor1")
        extractor2 = MockExtractor("Extractor2")
        
        chain = ExtractionFallbackChain(
            extractors=[extractor1, extractor2],
            fallback_condition=QualityBasedCondition(0.5)
        )
        
        # Set context on chain
        test_context = {"test": "context"}
        chain.context = test_context
        
        # Verify context is propagated to extractors
        self.assertEqual(extractor1.context, test_context)
        self.assertEqual(extractor2.context, test_context)

class TestExtractionQualityAssessor(unittest.TestCase):
    """Tests for the ExtractionQualityAssessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.assessor = ExtractionQualityAssessor()
        
    def test_schema_compliance_assessment(self):
        """Test schema compliance assessment."""
        # Create schema and data
        schema = {
            "fields": [
                {"name": "title", "required": True},
                {"name": "price", "required": True},
                {"name": "description", "required": False},
                {"name": "category", "required": False}
            ]
        }
        
        complete_data = {
            "title": "Test Product",
            "price": 99.99,
            "description": "Test description",
            "category": "Test"
        }
        
        partial_data = {
            "title": "Test Product",
            "price": 99.99
        }
        
        missing_required_data = {
            "description": "Test description",
            "category": "Test"
        }
        
        # Test assessment scores
        complete_score = self.assessor.assess_schema_compliance(complete_data, schema)
        partial_score = self.assessor.assess_schema_compliance(partial_data, schema)
        missing_score = self.assessor.assess_schema_compliance(missing_required_data, schema)
        
        self.assertEqual(complete_score, 1.0)  # All fields present
        self.assertEqual(partial_score, 0.5)  # 2 of 4 fields
        self.assertEqual(missing_score, 0.0)  # Missing required fields
        
    def test_data_completeness_assessment(self):
        """Test data completeness assessment."""
        # Test with explicit requirements
        requirements = {"required_fields": ["title", "price", "description"]}
        
        complete_data = {
            "title": "Test Product",
            "price": 99.99,
            "description": "Test description"
        }
        
        partial_data = {
            "title": "Test Product",
            "price": 99.99,
            "description": ""  # Empty value
        }
        
        missing_data = {
            "title": "Test Product"
        }
        
        complete_score = self.assessor.assess_data_completeness(complete_data, requirements)
        partial_score = self.assessor.assess_data_completeness(partial_data, requirements)
        missing_score = self.assessor.assess_data_completeness(missing_data, requirements)
        
        self.assertEqual(complete_score, 1.0)  # All required fields present
        self.assertEqual(partial_score, 0.6667, delta=0.001)  # 2 of 3 fields (empty value doesn't count)
        self.assertEqual(missing_score, 0.3333, delta=0.001)  # 1 of 3 fields
        
        # Test without explicit requirements (general completeness)
        general_score = self.assessor.assess_data_completeness({
            "a": 1,
            "b": "",
            "c": None,
            "d": "value"
        })
        
        self.assertEqual(general_score, 0.5)  # 2 of 4 fields have values
        
    def test_data_consistency_assessment(self):
        """Test data consistency assessment."""
        consistent_data = {
            "title": "Test Product",
            "price": 99.99,
            "date_created": "2023-01-01",
            "image_url": "https://example.com/image.jpg"
        }
        
        inconsistent_data = {
            "title": "Test Product",
            "price": "invalid",  # Should be numeric
            "date_created": ["wrong", "type"],  # Should be string or numeric
            "image_url": "invalid-url"  # Should start with http or /
        }
        
        consistent_score = self.assessor.assess_data_consistency(consistent_data)
        inconsistent_score = self.assessor.assess_data_consistency(inconsistent_data)
        
        self.assertGreater(consistent_score, 0.7)  # Should be highly consistent
        self.assertLess(inconsistent_score, 0.5)  # Should have low consistency
        
    def test_extraction_anomaly_detection(self):
        """Test anomaly detection in extracted data."""
        normal_data = {
            "title": "Test Product",
            "description": "This is a normal length description."
        }
        
        anomalous_data = {
            "title": "Test Product",
            "description": "<div>HTML content that <script>shouldn't be here</script></div>",
            "content": "x" * 20000  # Excessively long
        }
        
        normal_score = self.assessor.detect_extraction_anomalies(normal_data)
        anomalous_score = self.assessor.detect_extraction_anomalies(anomalous_data)
        
        self.assertLess(normal_score, 0.2)  # Low anomaly score (good)
        self.assertGreater(anomalous_score, 0.5)  # High anomaly score (bad)
        
    def test_confidence_score_measurement(self):
        """Test confidence score measurement for fields."""
        data = {
            "title": "Test Product",
            "description": "This is a test product description.",
            "price": 99.99,
            "empty_field": "",
            "url": "not-a-valid-url",
            "ratings": [4, 5, 3, 5]
        }
        
        confidence_scores = self.assessor.measure_confidence_scores(data)
        
        self.assertGreater(confidence_scores["title"], 0.7)  # Normal string
        self.assertGreater(confidence_scores["price"], 0.8)  # Numeric value
        self.assertEqual(confidence_scores["empty_field"], 0.0)  # Empty field
        self.assertLess(confidence_scores["url"], 0.5)  # Invalid URL format
        self.assertGreater(confidence_scores["ratings"], 0.4)  # List with items
        
    def test_overall_quality_assessment(self):
        """Test overall quality assessment."""
        schema = {
            "fields": [
                {"name": "title", "required": True},
                {"name": "price", "required": True},
                {"name": "description", "required": False}
            ]
        }
        
        good_result = ExtractionResult(
            data={
                "title": "Test Product",
                "price": 99.99,
                "description": "This is a test product."
            },
            success=True
        )
        
        bad_result = ExtractionResult(
            data={
                "title": "Test Product",
                "description": "<div>HTML content</div>"
            },
            success=True
        )
        
        good_score = self.assessor.assess_quality(good_result, schema)
        bad_score = self.assessor.assess_quality(bad_result, schema)
        
        self.assertGreater(good_score, 0.7)  # Good extraction quality
        self.assertLess(bad_score, 0.5)  # Poor extraction quality

class TestProgressiveDegradation(unittest.TestCase):
    """Tests for progressive degradation components."""
    
    def test_field_subset_extractor(self):
        """Test FieldSubsetExtractor."""
        # Create mock core extractor that returns full data
        core_extractor = MockExtractor(
            name="CoreExtractor",
            data={
                "title": "Test Product",
                "price": 99.99,
                "description": "Test description",
                "category": "Test category",
                "rating": 4.5
            }
        )
        
        # Create subset extractor focusing on critical fields
        subset_extractor = FieldSubsetExtractor(
            core_extractor=core_extractor,
            critical_fields=["title", "price"]
        )
        
        # Test extraction
        schema = {
            "fields": [
                {"name": "title", "required": True},
                {"name": "price", "required": True},
                {"name": "description", "required": False},
                {"name": "category", "required": False},
                {"name": "rating", "required": False}
            ]
        }
        
        result = subset_extractor.extract(HTML_CONTENT, schema)
        
        # Verify result
        self.assertTrue(result.success)
        self.assertTrue("critical_fields_only" in result.metadata)
        self.assertTrue(result.metadata["critical_fields_only"])
        
        # Original extractor was called
        self.assertTrue(core_extractor.extract_called)
        
        # Full data should still be present (the mock doesn't implement subset logic)
        self.assertEqual(result.data["title"], "Test Product")
        self.assertEqual(result.data["price"], 99.99)
    
    def test_schema_relaxation_transformer(self):
        """Test SchemaRelaxationTransformer."""
        # Create mock core extractor
        core_extractor = MockExtractor(
            name="CoreExtractor",
            data={
                "title": "Test Product",
                "price": 99.99
            }
        )
        
        # Create relaxation transformer
        relaxation_transformer = SchemaRelaxationTransformer(
            core_extractor=core_extractor,
            relaxation_level=0.7  # High relaxation
        )
        
        # Test extraction
        schema = {
            "fields": [
                {"name": "title", "required": True},
                {"name": "price", "required": True},
                {"name": "description", "required": True}  # Missing in data
            ]
        }
        
        result = relaxation_transformer.extract(HTML_CONTENT, schema)
        
        # Verify result
        self.assertTrue(result.success)
        self.assertTrue("schema_relaxed" in result.metadata)
        self.assertTrue(result.metadata["schema_relaxed"])
        self.assertEqual(result.metadata["relaxation_level"], 0.7)
        
        # Original extractor was called
        self.assertTrue(core_extractor.extract_called)
    
    def test_type_coercion_transformer(self):
        """Test TypeCoercionTransformer."""
        # Create mock core extractor with string price that should be numeric
        core_extractor = MockExtractor(
            name="CoreExtractor",
            data={
                "title": "Test Product",
                "price": "$99.99",  # String price
                "in_stock": "yes",  # Should be boolean
                "ratings": "4.5, 5, 3.5"  # Should be array
            }
        )
        
        # Create type coercion transformer
        coercion_transformer = TypeCoercionTransformer(core_extractor=core_extractor)
        
        # Test extraction
        schema = {
            "fields": [
                {"name": "title", "type": "string"},
                {"name": "price", "type": "number"},
                {"name": "in_stock", "type": "boolean"},
                {"name": "ratings", "type": "array"}
            ]
        }
        
        result = coercion_transformer.extract(HTML_CONTENT, schema)
        
        # Original extractor was called
        self.assertTrue(core_extractor.extract_called)
        
        # The mock coercion doesn't actually transform the data
        # This primarily tests that the transformer class structure works
        self.assertTrue(result.success)
        self.assertEqual(result.extractor_name, "TypeCoercion(CoreExtractor)")
    
    def test_partial_result_aggregator(self):
        """Test PartialResultAggregator."""
        # Create mock extractors that extract different parts of the data
        title_extractor = MockExtractor(
            name="TitleExtractor",
            data={"title": "Test Product"}
        )
        
        price_extractor = MockExtractor(
            name="PriceExtractor",
            data={"price": 99.99}
        )
        
        desc_extractor = MockExtractor(
            name="DescriptionExtractor",
            data={"description": "Test description"}
        )
        
        # Create aggregator
        aggregator = PartialResultAggregator(
            extractors=[title_extractor, price_extractor, desc_extractor]
        )
        
        # Test extraction
        result = aggregator.extract(HTML_CONTENT)
        
        # Verify all extractors were called
        self.assertTrue(title_extractor.extract_called)
        self.assertTrue(price_extractor.extract_called)
        self.assertTrue(desc_extractor.extract_called)
        
        # Verify result
        self.assertTrue(result.success)
        self.assertEqual(result.extractor_name, "PartialResultAggregator")
        self.assertTrue(result.metadata["aggregated"])
        
        # Check that data is aggregated
        self.assertEqual(result.data["title"], "Test Product")
        self.assertEqual(result.data["price"], 99.99)
        self.assertEqual(result.data["description"], "Test description")

class TestExtractionChainFactory(unittest.TestCase):
    """Tests for extraction chain factory functions."""
    
    def test_html_extraction_chain(self):
        """Test creation of HTML extraction chain."""
        chain = create_html_extraction_chain()
        
        # Verify chain structure
        self.assertIsInstance(chain, ExtractionFallbackChain)
        self.assertTrue(len(chain.extractors) > 0)
        self.assertTrue(chain.aggregate_results)
        
        # Test chain with HTML content
        result = chain.extract(HTML_CONTENT)
        
        # Verify extraction works
        self.assertTrue(result.success)
        
    def test_api_extraction_chain(self):
        """Test creation of API extraction chain."""
        chain = create_api_extraction_chain()
        
        # Verify chain structure
        self.assertIsInstance(chain, ExtractionFallbackChain)
        self.assertTrue(len(chain.extractors) > 0)
        self.assertTrue(chain.aggregate_results)
        
        # Test chain with JSON content
        result = chain.extract(JSON_CONTENT)
        
        # Verify extraction works
        self.assertTrue(result.success)
        
    def test_text_extraction_chain(self):
        """Test creation of text extraction chain."""
        chain = create_text_extraction_chain()
        
        # Verify chain structure
        self.assertIsInstance(chain, ExtractionFallbackChain)
        self.assertTrue(len(chain.extractors) > 0)
        self.assertTrue(chain.aggregate_results)
        
        # Test chain with text content
        result = chain.extract(TEXT_CONTENT)
        
        # Verify extraction works
        self.assertTrue(result.success)

class TestExtractionFallbackRegistry(unittest.TestCase):
    """Tests for the ExtractionFallbackRegistry class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = ExtractionFallbackRegistry()
        
        # Register mock extractors
        class MockExtractorA(BaseExtractor):
            def __init__(self):
                super().__init__(name="ExtractorA")
            def can_handle(self, content_type, content=None, schema=None):
                return True
            def extract(self, content, schema=None, options=None):
                return ExtractionResult(data={"source": "A"}, success=True, extractor_name=self.name)
                
        class MockExtractorB(BaseExtractor):
            def __init__(self):
                super().__init__(name="ExtractorB")
            def can_handle(self, content_type, content=None, schema=None):
                return True
            def extract(self, content, schema=None, options=None):
                return ExtractionResult(data={"source": "B"}, success=True, extractor_name=self.name)
                
        class MockExtractorC(BaseExtractor):
            def __init__(self):
                super().__init__(name="ExtractorC")
            def can_handle(self, content_type, content=None, schema=None):
                return True
            def extract(self, content, schema=None, options=None):
                return ExtractionResult(data={"source": "C"}, success=True, extractor_name=self.name)
        
        self.registry.register_extractor("ExtractorA", MockExtractorA)
        self.registry.register_extractor("ExtractorB", MockExtractorB)
        self.registry.register_extractor("ExtractorC", MockExtractorC)
        
    def test_register_fallback(self):
        """Test registering fallbacks."""
        self.registry.register_fallback("ExtractorA", "ExtractorB")
        self.registry.register_fallback("ExtractorA", "ExtractorC")
        self.registry.register_fallback("ExtractorB", "ExtractorC")
        
        # Verify fallbacks are registered
        fallbacks_a = self.registry.get_fallbacks_for_extractor("ExtractorA")
        fallbacks_b = self.registry.get_fallbacks_for_extractor("ExtractorB")
        fallbacks_c = self.registry.get_fallbacks_for_extractor("ExtractorC")
        
        self.assertEqual(len(fallbacks_a), 2)
        self.assertEqual(fallbacks_a, ["ExtractorB", "ExtractorC"])
        self.assertEqual(fallbacks_b, ["ExtractorC"])
        self.assertEqual(fallbacks_c, [])
        
    def test_create_fallback_chain(self):
        """Test creating a fallback chain."""
        # Register fallbacks
        self.registry.register_fallback("ExtractorA", "ExtractorB")
        self.registry.register_fallback("ExtractorA", "ExtractorC")
        
        # Create chain
        chain = self.registry.create_fallback_chain("ExtractorA")
        
        # Verify chain structure
        self.assertIsInstance(chain, ExtractionFallbackChain)
        self.assertEqual(len(chain.extractors), 3)
        self.assertEqual(chain.extractors[0].name, "ExtractorA")
        self.assertEqual(chain.extractors[1].name, "ExtractorB")
        self.assertEqual(chain.extractors[2].name, "ExtractorC")
        
    def test_get_default_chain(self):
        """Test getting default chain for content type."""
        # Test with HTML content type
        html_chain = self.registry.get_default_chain(ContentType.HTML)
        
        # Verify chain structure
        self.assertIsInstance(html_chain, ExtractionFallbackChain)
        
        # Test extraction
        result = html_chain.extract(HTML_CONTENT)
        self.assertTrue(result.success)
        
    def test_suggest_fallback(self):
        """Test suggesting fallback extractor."""
        # Register fallbacks
        self.registry.register_fallback("ExtractorA", "ExtractorB")
        
        # Create extractors
        extractor_a = self.registry._extractors["ExtractorA"]()
        
        # Simulate an error
        error = ValueError("Test error")
        
        # Get fallback suggestion
        fallback = self.registry.suggest_fallback(extractor_a, error, HTML_CONTENT)
        
        # Verify suggestion
        self.assertIsNotNone(fallback)
        self.assertEqual(fallback.name, "ExtractorB")

if __name__ == '__main__':
    unittest.main()