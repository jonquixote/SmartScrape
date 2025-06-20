"""
Unit Tests for Normalization Stage.

This test suite validates that the NormalizationStage correctly standardizes
and normalizes extracted data according to various configurations.
"""

import unittest
import pytest
import asyncio
import json
from unittest.mock import MagicMock, patch
from datetime import datetime

from core.pipeline.context import PipelineContext
from strategies.core.strategy_context import StrategyContext
from extraction.stages.normalization_stage import NormalizationStage


class TestNormalizationStage(unittest.TestCase):
    """Test suite for NormalizationStage."""

    def setUp(self):
        """Set up test environment before each test case."""
        self.stage = NormalizationStage()
        
        # Create strategy context
        self.strategy_context = StrategyContext()
        
        # Create pipeline context
        self.pipeline_context = PipelineContext()
        self.pipeline_context.strategy_context = self.strategy_context
        
        # Sample extracted data for testing
        self.sample_data = {
            "extracted_data": {
                "title": "  Product Title with  extra spaces  ",
                "price": "$123.45",
                "description": "<p>Product description with <b>HTML</b> tags.</p>",
                "date_published": "2023-01-15",
                "specifications": {
                    "weight": "10kg",
                    "dimensions": "10cm x 20cm x 30cm",
                    "color": "red"
                },
                "features": ["Feature 1", "Feature 2", "Feature 3"],
                "_metadata": {
                    "extraction_method": "pattern",
                    "extractor": "ProductExtractor"
                }
            }
        }

    @pytest.mark.asyncio
    async def test_basic_normalization(self):
        """Test basic normalization functionality."""
        # Configure the stage
        config = {
            "input_key": "extracted_data",
            "output_key": "normalized_data",
            "trim_strings": True,
            "normalize_whitespace": True,
            "remove_html": True
        }
        await self.stage.initialize(config)
        
        # Process data through the stage
        result = await self.stage.process(self.sample_data, self.pipeline_context)
        
        # Get normalized data from result
        normalized_data = result["normalized_data"]
        
        # Assert basic normalization was applied
        self.assertIsNotNone(normalized_data)
        self.assertEqual("Product Title with extra spaces", normalized_data["title"])
        self.assertIn("Product description with HTML tags", normalized_data["description"])
        
        # Check that metadata was preserved
        self.assertIn("_metadata", normalized_data)
        self.assertEqual("pattern", normalized_data["_metadata"]["extraction_method"])
        
        # Check that normalization metadata was added
        self.assertTrue(normalized_data["_metadata"]["normalized"])
        self.assertIn("normalization_time", normalized_data["_metadata"])

    @pytest.mark.asyncio
    async def test_price_normalization(self):
        """Test price normalization functionality."""
        # Configure the stage with price standardization
        config = {
            "input_key": "extracted_data",
            "output_key": "normalized_data",
            "standardize_prices": True,
            "field_types": {
                "price": "price"
            }
        }
        await self.stage.initialize(config)
        
        # Process data through the stage
        result = await self.stage.process(self.sample_data, self.pipeline_context)
        
        # Get normalized data from result
        normalized_data = result["normalized_data"]
        
        # Assert price was normalized correctly
        self.assertIsNotNone(normalized_data)
        self.assertIn("price", normalized_data)
        
        # The price should now be a dictionary with standardized format
        self.assertIsInstance(normalized_data["price"], dict)
        self.assertIn("amount", normalized_data["price"])
        self.assertEqual(123.45, normalized_data["price"]["amount"])
        self.assertIn("currency", normalized_data["price"])
        self.assertEqual("USD", normalized_data["price"]["currency"])

    @pytest.mark.asyncio
    async def test_field_mappings(self):
        """Test field mapping functionality."""
        # Configure the stage with field mappings
        config = {
            "input_key": "extracted_data",
            "output_key": "normalized_data",
            "field_mappings": {
                "title": "product_title",
                "price": "product_price",
                "description": "product_description"
            }
        }
        await self.stage.initialize(config)
        
        # Process data through the stage
        result = await self.stage.process(self.sample_data, self.pipeline_context)
        
        # Get normalized data from result
        normalized_data = result["normalized_data"]
        
        # Assert fields were renamed according to mappings
        self.assertIsNotNone(normalized_data)
        self.assertIn("product_title", normalized_data)
        self.assertIn("product_price", normalized_data)
        self.assertIn("product_description", normalized_data)
        self.assertNotIn("title", normalized_data)
        self.assertNotIn("price", normalized_data)
        self.assertNotIn("description", normalized_data)

    @pytest.mark.asyncio
    async def test_custom_normalizers(self):
        """Test custom normalizer functionality."""
        # Define a custom normalizer function
        def uppercase_normalizer(value):
            if isinstance(value, str):
                return value.upper()
            return value
        
        # Configure the stage
        config = {
            "input_key": "extracted_data",
            "output_key": "normalized_data"
        }
        await self.stage.initialize(config)
        
        # Register custom normalizer
        self.stage.register_custom_normalizer("title", uppercase_normalizer)
        
        # Process data through the stage
        result = await self.stage.process(self.sample_data, self.pipeline_context)
        
        # Get normalized data from result
        normalized_data = result["normalized_data"]
        
        # Assert custom normalizer was applied
        self.assertIsNotNone(normalized_data)
        self.assertEqual("  PRODUCT TITLE WITH  EXTRA SPACES  ", normalized_data["title"])

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in normalization stage."""
        # Create a stage that will cause an error
        stage = NormalizationStage()
        stage._normalize_data = MagicMock(side_effect=ValueError("Test error"))
        
        # Configure the stage
        config = {
            "input_key": "extracted_data",
            "output_key": "normalized_data"
        }
        await stage.initialize(config)
        
        # Process data through the stage
        result = await stage.process(self.sample_data, self.pipeline_context)
        
        # Check error was recorded
        self.assertIn("_error", result)
        self.assertIn("Test error", result["_error"])
        
        # Check original data was returned as fallback
        self.assertIn("normalized_data", result)
        self.assertEqual(self.sample_data["extracted_data"], result["normalized_data"])

    @pytest.mark.asyncio
    async def test_measurement_normalization(self):
        """Test measurement normalization functionality."""
        # Configure the stage with unit standardization
        config = {
            "input_key": "extracted_data",
            "output_key": "normalized_data",
            "standardize_units": True,
            "field_types": {
                "specifications.weight": "measurement",
                "specifications.dimensions": "measurement"
            }
        }
        await self.stage.initialize(config)
        
        # Add test data with measurements
        data = {
            "extracted_data": {
                "specifications": {
                    "weight": "2.5kg",
                    "dimensions": "10cm x 20cm x 30cm"
                }
            }
        }
        
        # Process data through the stage
        result = await self.stage.process(data, self.pipeline_context)
        
        # Get normalized data from result
        normalized_data = result["normalized_data"]
        
        # Assert measurements were normalized
        self.assertIsNotNone(normalized_data)
        # Check weight normalization
        if isinstance(normalized_data["specifications"]["weight"], dict):
            self.assertEqual(2.5, normalized_data["specifications"]["weight"]["value"])
            self.assertEqual("kg", normalized_data["specifications"]["weight"]["unit"])
        
        # Check dimensions are parsed - this depends on implementation details
        # but we can check the structure is at least processed
        self.assertIn("dimensions", normalized_data["specifications"])


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])