"""
Test module for the quality evaluation components.

This module tests the quality evaluation system for assessing extracted data
quality and reliability, including completeness, confidence scoring, anomaly
detection, type validation, and schema compliance.
"""

import unittest
import json
import sys
import os
from unittest.mock import MagicMock, patch
from datetime import datetime

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from extraction.quality_evaluator import ExtractedDataQualityEvaluator
from extraction.helpers.quality_metrics import (
    calculate_text_quality, calculate_field_confidence,
    measure_numerical_plausibility, check_date_validity, check_url_validity,
    measure_enum_validity, calculate_overall_quality_score, detect_outliers
)
from extraction.core.extraction_result import ExtractionResult


class TestQualityMetrics(unittest.TestCase):
    """Test individual quality metric functions."""

    def test_calculate_text_quality(self):
        """Test text quality calculation."""
        # High quality text
        high_quality = "This is a well-formatted text with good content and proper capitalization."
        self.assertGreaterEqual(calculate_text_quality(high_quality), 0.8)

        # Low quality text (all caps)
        low_quality = "THIS IS ALL CAPS TEXT WHICH IS CONSIDERED LOWER QUALITY!!!"
        self.assertLessEqual(calculate_text_quality(low_quality), 0.7)

        # Very short text
        short_text = "Hi"
        self.assertLessEqual(calculate_text_quality(short_text), 0.6)

        # HTML remnants
        html_text = "Some <div>text with</div> <span>HTML tags</span>"
        self.assertLessEqual(calculate_text_quality(html_text), 0.7)

        # Common error values
        error_text = "undefined"
        self.assertLessEqual(calculate_text_quality(error_text), 0.5)

        # Empty text
        self.assertEqual(calculate_text_quality(""), 0.0)
        self.assertEqual(calculate_text_quality(None), 0.0)

    def test_calculate_field_confidence(self):
        """Test field confidence calculation."""
        # Test with various field values
        self.assertGreater(calculate_field_confidence("Valid value"), 0.6)
        self.assertEqual(calculate_field_confidence(""), 0.0)
        self.assertEqual(calculate_field_confidence(None), 0.0)

        # Test with pattern matching
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        self.assertGreater(calculate_field_confidence("user@example.com", email_pattern), 0.9)
        self.assertLess(calculate_field_confidence("not-an-email", email_pattern), 0.5)

        # Test with non-string values
        self.assertGreater(calculate_field_confidence(123), 0.7)
        self.assertGreater(calculate_field_confidence(True), 0.7)
        self.assertGreater(calculate_field_confidence(["item1", "item2"]), 0.7)

    def test_measure_numerical_plausibility(self):
        """Test numerical plausibility measurement."""
        # Test within expected range
        self.assertEqual(measure_numerical_plausibility(50, (0, 100)), 1.0)

        # Test outside expected range
        self.assertLess(measure_numerical_plausibility(150, (0, 100)), 1.0)
        self.assertLess(measure_numerical_plausibility(-10, (0, 100)), 1.0)

        # Test with non-numeric value
        self.assertEqual(measure_numerical_plausibility("not a number"), 0.0)

        # Test with no range
        self.assertGreater(measure_numerical_plausibility(100), 0.0)

    def test_check_date_validity(self):
        """Test date validation."""
        # Valid dates in different formats
        self.assertGreater(check_date_validity("2023-01-15"), 0.9)
        self.assertGreater(check_date_validity("01/15/2023"), 0.7)
        self.assertGreater(check_date_validity("January 15, 2023"), 0.7)

        # Invalid dates
        self.assertLess(check_date_validity("not a date"), 0.3)
        self.assertLess(check_date_validity("2023-13-45"), 0.5)  # Invalid month and day

        # Empty or non-string
        self.assertEqual(check_date_validity(""), 0.0)
        self.assertEqual(check_date_validity(None), 0.0)
        self.assertEqual(check_date_validity(12345), 0.0)

    def test_check_url_validity(self):
        """Test URL validation."""
        # Valid URLs
        self.assertGreater(check_url_validity("https://example.com"), 0.9)
        self.assertGreater(check_url_validity("http://example.com"), 0.8)
        self.assertGreater(check_url_validity("www.example.com"), 0.6)

        # Invalid URLs
        self.assertLess(check_url_validity("not a url"), 0.3)
        self.assertLess(check_url_validity("example"), 0.3)

        # Empty or non-string
        self.assertEqual(check_url_validity(""), 0.0)
        self.assertEqual(check_url_validity(None), 0.0)

    def test_measure_enum_validity(self):
        """Test enum validation."""
        allowed_values = ["apple", "banana", "cherry"]

        # Direct match
        self.assertEqual(measure_enum_validity("apple", allowed_values), 1.0)

        # Case-insensitive match
        self.assertGreater(measure_enum_validity("APPLE", allowed_values), 0.8)

        # Partial match
        self.assertGreater(measure_enum_validity("app", allowed_values), 0.0)
        self.assertLessEqual(measure_enum_validity("app", allowed_values), 0.7)

        # No match
        self.assertEqual(measure_enum_validity("orange", allowed_values), 0.0)

        # Empty or None
        self.assertEqual(measure_enum_validity("", allowed_values), 0.0)
        self.assertEqual(measure_enum_validity(None, allowed_values), 0.0)

    def test_calculate_overall_quality_score(self):
        """Test overall quality score calculation."""
        # High quality metrics
        high_metrics = {
            "completeness": 0.95,
            "confidence": 0.9,
            "consistency": 0.85,
            "relevance": 0.8
        }
        high_score = calculate_overall_quality_score(high_metrics)
        self.assertGreater(high_score, 0.8)

        # Low quality metrics
        low_metrics = {
            "completeness": 0.3,
            "confidence": 0.4,
            "consistency": 0.5,
            "relevance": 0.2
        }
        low_score = calculate_overall_quality_score(low_metrics)
        self.assertLess(low_score, 0.5)

        # Empty metrics
        self.assertEqual(calculate_overall_quality_score({}), 0.0)

    def test_detect_outliers(self):
        """Test outlier detection."""
        # Data with outliers
        data_with_outliers = {
            "normal1": 10,
            "normal2": 12,
            "normal3": 9,
            "normal4": 11,
            "outlier": 100  # Clear outlier
        }
        outliers = detect_outliers(data_with_outliers)
        self.assertIn("outlier", outliers)

        # Data without outliers
        data_without_outliers = {
            "value1": 10,
            "value2": 12,
            "value3": 9,
            "value4": 11
        }
        outliers = detect_outliers(data_without_outliers)
        self.assertEqual(len(outliers), 0)

        # Data with string length outliers
        data_with_text_outliers = {
            "text1": "Short text",
            "text2": "Another short text",
            "text3": "A" * 1000  # Very long text
        }
        outliers = detect_outliers(data_with_text_outliers)
        self.assertIn("text3", outliers)


class TestQualityEvaluator(unittest.TestCase):
    """Test the quality evaluator class."""

    def setUp(self):
        """Set up test environment before each test."""
        self.evaluator = ExtractedDataQualityEvaluator()
        self.evaluator.initialize()

        # Sample data for testing
        self.complete_data = {
            "title": "Sample Product",
            "price": 29.99,
            "description": "This is a sample product description.",
            "rating": 4.5,
            "in_stock": True,
            "categories": ["Electronics", "Gadgets"],
            "url": "https://example.com/product",
            "created_at": "2023-01-15",
            "image": "https://example.com/image.jpg"
        }

        self.partial_data = {
            "title": "Sample Product",
            "price": 29.99,
            "description": "",  # Empty field
            "rating": None,  # Missing field
            "categories": ["Electronics"]
        }

        self.low_quality_data = {
            "title": "SAMPLE PRODUCT!!!",
            "price": -29.99,  # Negative price (anomalous)
            "description": "<div>Product description with HTML</div>",
            "rating": 999,  # Implausible rating
            "created_at": "not a date"  # Invalid date
        }

        # Sample schema for testing
        self.schema = {
            "type": "object",
            "required": ["title", "price", "description"],
            "properties": {
                "title": {"type": "string", "minLength": 3},
                "price": {"type": "number", "minimum": 0},
                "description": {"type": "string"},
                "rating": {"type": "number", "minimum": 0, "maximum": 5},
                "in_stock": {"type": "boolean"},
                "categories": {"type": "array", "items": {"type": "string"}},
                "url": {"type": "string", "format": "uri"},
                "created_at": {"type": "string", "format": "date"},
                "image": {"type": "string"}
            }
        }

    def test_initialization(self):
        """Test evaluator initialization."""
        # Test with default configuration
        evaluator = ExtractedDataQualityEvaluator()
        evaluator.initialize()
        self.assertTrue(evaluator.is_initialized)

        # Test with custom configuration
        custom_config = {
            "confidence_thresholds": {
                "high": 0.9,
                "medium": 0.7,
                "low": 0.5
            },
            "criteria_weights": {
                "completeness": 0.4,
                "confidence": 0.3,
                "relevance": 0.1,
                "consistency": 0.1,
                "schema_compliance": 0.1
            }
        }
        evaluator = ExtractedDataQualityEvaluator()
        evaluator.initialize(custom_config)
        self.assertTrue(evaluator.is_initialized)

    def test_can_handle(self):
        """Test the can_handle method."""
        # Should handle dictionaries
        self.assertTrue(self.evaluator.can_handle({}))
        self.assertTrue(self.evaluator.can_handle(self.complete_data))

        # Should handle ExtractionResult objects
        extraction_result = ExtractionResult(data=self.complete_data)
        self.assertTrue(self.evaluator.can_handle(extraction_result))

        # Should not handle other types
        self.assertFalse(self.evaluator.can_handle("string"))
        self.assertFalse(self.evaluator.can_handle(123))
        self.assertFalse(self.evaluator.can_handle(None))

    def test_extract_method(self):
        """Test the extract method (proxy to evaluate)."""
        result = self.evaluator.extract(self.complete_data, {"schema": self.schema})
        self.assertIsInstance(result, dict)
        self.assertIn("metrics", result)
        self.assertIn("quality_score", result)

    def test_completeness_evaluation(self):
        """Test completeness evaluation."""
        # Complete data should have high completeness
        completeness = self.evaluator.calculate_completeness_score(self.complete_data, self.schema)
        self.assertGreater(completeness, 0.9)

        # Partial data should have lower completeness
        completeness = self.evaluator.calculate_completeness_score(self.partial_data, self.schema)
        self.assertLess(completeness, 0.8)

        # Missing required fields should significantly lower completeness
        incomplete_data = {"title": "Just a title"}  # Missing required price and description
        completeness = self.evaluator.calculate_completeness_score(incomplete_data, self.schema)
        self.assertLess(completeness, 0.5)

    def test_confidence_scoring(self):
        """Test confidence scoring."""
        # Good data should have high confidence
        confidence = self.evaluator.calculate_confidence_score(self.complete_data)
        self.assertGreater(confidence, 0.7)

        # Data with anomalies should have lower confidence
        confidence = self.evaluator.calculate_confidence_score(self.low_quality_data)
        self.assertLess(confidence, 0.7)

        # Test with existing confidence metadata
        data_with_confidence = {
            "title": "Sample",
            "_metadata": {"confidence_score": 0.95}
        }
        confidence = self.evaluator.calculate_confidence_score(data_with_confidence)
        self.assertEqual(confidence, 0.95)

    def test_anomaly_detection(self):
        """Test anomaly detection."""
        # Good data should have few or no anomalies
        anomalies = self.evaluator.detect_anomalies(self.complete_data, self.schema)
        self.assertEqual(len(anomalies), 0)

        # Low quality data should have multiple anomalies
        anomalies = self.evaluator.detect_anomalies(self.low_quality_data, self.schema)
        self.assertGreater(len(anomalies), 0)
        # Should detect negative price
        self.assertIn("price", anomalies)
        # Should detect implausible rating
        self.assertIn("rating", anomalies)

        # Test with empty fields
        data_with_empty = {"title": "", "price": 0}
        anomalies = self.evaluator.detect_anomalies(data_with_empty, self.schema)
        self.assertIn("title", anomalies)

    def test_type_validation(self):
        """Test type validation against schema."""
        # Correct types should validate successfully
        result = self.evaluator.validate_field_types(self.complete_data, self.schema)
        self.assertGreater(result["valid_rate"], 0.9)
        self.assertEqual(len(result["invalid_fields"]), 0)

        # Incorrect types should fail validation
        invalid_data = {
            "title": 123,  # Should be string
            "price": "29.99",  # Should be number
            "description": ["Not a string"],  # Should be string
            "rating": "4.5"  # Should be number
        }
        result = self.evaluator.validate_field_types(invalid_data, self.schema)
        self.assertLess(result["valid_rate"], 0.5)
        self.assertGreater(len(result["invalid_fields"]), 0)
        self.assertIn("title", result["invalid_fields"])
        self.assertIn("price", result["invalid_fields"])

    def test_consistency_evaluation(self):
        """Test consistency evaluation."""
        # Consistent data should have high consistency score
        consistency = self.evaluator.evaluate_consistency(self.complete_data)
        self.assertGreater(consistency, 0.7)

        # Inconsistent date formats
        inconsistent_data = {
            "created_at": "2023-01-15",  # ISO format
            "updated_at": "01/20/2023",  # US format
            "published_at": "20-02-2023"  # Another format
        }
        consistency = self.evaluator.evaluate_consistency(inconsistent_data)
        self.assertLess(consistency, 0.7)

    def test_quality_report_generation(self):
        """Test quality report generation."""
        report = self.evaluator.get_quality_report(self.complete_data, self.schema)
        self.assertIsInstance(report, dict)
        self.assertIn("summary", report)
        self.assertIn("field_analysis", report)
        self.assertIn("schema_compliance", report)
        self.assertIn("recommendations", report)

        # Low quality data should generate recommendations
        report = self.evaluator.get_quality_report(self.low_quality_data, self.schema)
        self.assertGreater(len(report["recommendations"]), 0)
        self.assertGreater(len(report["anomalies"]), 0)

    def test_missing_required_fields(self):
        """Test detection of missing required fields."""
        # Complete data should have no missing fields
        missing = self.evaluator.detect_missing_required_fields(self.complete_data, self.schema)
        self.assertEqual(len(missing), 0)

        # Data missing required fields should detect them
        incomplete_data = {"title": "Just a title"}  # Missing required price and description
        missing = self.evaluator.detect_missing_required_fields(incomplete_data, self.schema)
        self.assertEqual(len(missing), 2)
        self.assertIn("price", missing)
        self.assertIn("description", missing)

    def test_text_quality_evaluation(self):
        """Test text quality evaluation."""
        # Good text
        good_text = "This is a well-written description of the product."
        quality = self.evaluator.evaluate_text_quality(good_text)
        self.assertGreater(quality, 0.8)

        # Poor text (ALL CAPS)
        poor_text = "THIS IS A POORLY WRITTEN DESCRIPTION!!!"
        quality = self.evaluator.evaluate_text_quality(poor_text)
        self.assertLess(quality, 0.8)

        # Text with HTML
        html_text = "<p>Text with HTML tags</p>"
        quality = self.evaluator.evaluate_text_quality(html_text)
        self.assertLess(quality, 0.8)

    def test_structural_consistency(self):
        """Test structural consistency evaluation."""
        # Consistent data
        consistent_data = {
            "url1": "https://example.com/page1",
            "url2": "https://example.com/page2",
            "url3": "https://example.com/page3"
        }
        consistency = self.evaluator.evaluate_structural_consistency(consistent_data)
        self.assertEqual(consistency, 1.0)  # All URLs have the same format

        # Inconsistent data
        inconsistent_data = {
            "url1": "https://example.com/page1",
            "url2": "http://different.com/page2",  # Different protocol and domain
            "url3": "www.example.com/page3"  # Missing protocol
        }
        consistency = self.evaluator.evaluate_structural_consistency(inconsistent_data)
        self.assertLess(consistency, 1.0)

    def test_evaluate_with_extraction_result(self):
        """Test evaluate method with ExtractionResult objects."""
        # Create an ExtractionResult
        extraction_result = ExtractionResult(
            data=self.complete_data,
            metadata={"extractor": "TestExtractor", "confidence": 0.9}
        )

        # Evaluate
        result = self.evaluator.evaluate(extraction_result, self.schema)
        self.assertIsInstance(result, dict)
        self.assertIn("metrics", result)
        self.assertIn("quality_score", result)
        self.assertGreater(result["quality_score"], 0.7)  # Should have good quality

    def test_partial_data_handling(self):
        """Test handling of partial data."""
        # Partial data should be evaluated without errors
        result = self.evaluator.evaluate(self.partial_data, self.schema)
        self.assertIsInstance(result, dict)
        self.assertIn("metrics", result)
        
        # Missing fields should be reflected in completeness
        self.assertLess(result["metrics"]["completeness"], 0.8)
        
        # Missing required fields should be detected
        self.assertIn("missing_required_fields", result["metrics"])
        self.assertIn("description", result["metrics"]["missing_required_fields"])

    def test_scoring_algorithm_accuracy(self):
        """Test accuracy of scoring algorithms by comparing samples."""
        # Create samples with known quality issues
        samples = [
            # Complete, valid data (highest quality)
            self.complete_data,
            
            # Missing optional fields (high quality)
            {key: value for key, value in self.complete_data.items() 
             if key in ["title", "price", "description", "in_stock"]},
            
            # Missing a required field (medium quality)
            {key: value for key, value in self.complete_data.items() if key != "description"},
            
            # Type mismatch (low quality)
            {**self.complete_data, "price": "not a number", "rating": "not a number"},
            
            # Multiple issues (lowest quality)
            {"title": "X", "price": -1, "rating": 999, "created_at": "invalid"}
        ]
        
        # Evaluate all samples
        results = [self.evaluator.evaluate(sample, self.schema) for sample in samples]
        scores = [result["quality_score"] for result in results]
        
        # Check that scores decrease as quality decreases
        for i in range(1, len(scores)):
            self.assertLessEqual(scores[i], scores[i-1])
        
        # Particularly check the best and worst samples
        self.assertGreater(scores[0], 0.8)  # Best sample should have high score
        self.assertLess(scores[-1], 0.5)    # Worst sample should have low score

    def test_evaluate_data_relationships(self):
        """Test evaluation of data relationships."""
        # Data with consistent relationships
        related_data = {
            "start_date": "2023-01-01",
            "end_date": "2023-01-15",
            "discount_price": 24.99,
            "regular_price": 29.99
        }
        
        result = self.evaluator.evaluate_data_relationships(related_data)
        self.assertGreater(result["relationship_score"], 0.8)
        
        # Data with inconsistent relationships
        inconsistent_data = {
            "start_date": "2023-01-15",  # Later than end_date
            "end_date": "2023-01-01",
            "discount_price": 34.99,     # Higher than regular_price
            "regular_price": 29.99
        }
        
        result = self.evaluator.evaluate_data_relationships(inconsistent_data)
        self.assertLess(result["relationship_score"], 0.5)

    def test_field_distribution_analysis(self):
        """Test analysis of field value distributions."""
        data = {
            "price1": 10.99,
            "price2": 11.99,
            "price3": 9.99,
            "price4": 12.99,
            "price5": 100.99  # Outlier
        }
        
        analysis = self.evaluator.analyze_field_distribution(data)
        self.assertIn("type_distribution", analysis)
        self.assertIn("numeric_values", analysis)
        
        # Should detect the numeric distribution
        self.assertEqual(analysis["type_distribution"]["float"], 5)
        
        # Should provide meaningful statistics
        self.assertLess(analysis["numeric_values"]["min"], analysis["numeric_values"]["max"])
        
        # Test with string fields
        string_data = {
            "text1": "Short",
            "text2": "Medium length text",
            "text3": "Very long text that has significantly more characters than the others"
        }
        
        analysis = self.evaluator.analyze_field_distribution(string_data)
        self.assertIn("string_length", analysis)
        self.assertLess(analysis["string_length"]["min"], analysis["string_length"]["max"])

    @patch('extraction.quality_evaluator.detect_outliers')
    def test_configurable_evaluation_criteria(self, mock_detect_outliers):
        """Test that evaluation criteria are configurable."""
        # Configure the evaluator with custom thresholds
        custom_config = {
            "confidence_thresholds": {
                "high": 0.9,  # Stricter than default
                "medium": 0.7,
                "low": 0.5
            },
            "criteria_weights": {
                "completeness": 0.5,  # Higher weight on completeness
                "confidence": 0.2,
                "relevance": 0.1,
                "consistency": 0.1,
                "schema_compliance": 0.1
            }
        }
        
        # Create evaluator with custom config
        custom_evaluator = ExtractedDataQualityEvaluator()
        custom_evaluator.initialize(custom_config)
        
        # Default evaluator for comparison
        default_evaluator = ExtractedDataQualityEvaluator()
        default_evaluator.initialize()
        
        # Evaluate same data with both evaluators
        custom_result = custom_evaluator.evaluate(self.partial_data, self.schema)
        default_result = default_evaluator.evaluate(self.partial_data, self.schema)
        
        # Due to different weights, scores should be different
        self.assertNotEqual(custom_result["quality_score"], default_result["quality_score"])
        
        # With higher weight on completeness and lower completeness score for partial data,
        # custom evaluator should give a lower overall score
        if custom_result["metrics"]["completeness"] < 0.8:
            self.assertLess(custom_result["quality_score"], default_result["quality_score"])


if __name__ == '__main__':
    unittest.main()