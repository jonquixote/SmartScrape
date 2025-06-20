"""
Test module for content normalization pipeline stages.

This module contains tests for the DataNormalizationStage, DataValidationStage,
and DataTransformationStage classes.
"""

import pytest
import json
import datetime
import re
from typing import Dict, Any, List

from core.pipeline.context import PipelineContext
from core.pipeline.stages.processing.content_normalization import (
    DataNormalizationStage,
    DataValidationStage,
    DataTransformationStage
)

# Sample data for testing
SAMPLE_DATA = {
    "title": "  Test Product  ",
    "price": "$199.99",
    "date_published": "2023-05-15",
    "in_stock": "Yes",
    "description": "<p>This is a <b>test</b> description.</p>",
    "tags": "electronics,gadgets,new",
    "rating": "4.5 out of 5",
    "location": "New York, NY 10001, USA",
    "metadata": {
        "source_url": "https://example.com/product/123",
        "extractor": "test"
    }
}

SAMPLE_INVALID_DATA = {
    "title": "",  # Empty title
    "price": "unknown",  # Invalid price
    "date_published": "not a date",  # Invalid date
    "in_stock": "maybe",  # Invalid boolean
    "description": "<p>Too short</p>",  # Too short
    "rating": "invalid"  # Invalid rating
}

# Create test fixtures
@pytest.fixture
def pipeline_context():
    """Create a pipeline context with sample data."""
    context = PipelineContext()
    context.set("input_data", SAMPLE_DATA)
    return context

@pytest.fixture
def pipeline_context_with_normalized_data():
    """Create a pipeline context with already normalized data."""
    context = PipelineContext()
    
    # Normalized data
    normalized_data = {
        "title": "Test Product",
        "price": {
            "amount": 199.99,
            "currency": "USD",
            "formatted": "$199.99"
        },
        "date": {
            "iso": "2023-05-15T00:00:00",
            "timestamp": 1684108800,
            "formatted": "May 15, 2023"
        },
        "in_stock": True,
        "description": "This is a test description.",
        "tags": ["electronics", "gadgets", "new"],
        "rating": {
            "value": 4.5,
            "scale": 5.0,
            "count": None
        },
        "location": {
            "full_address": "New York, NY 10001, USA",
            "city": "New York",
            "state": "NY",
            "postal_code": "10001",
            "country": "USA"
        }
    }
    
    context.set("normalized_data", normalized_data)
    return context

@pytest.fixture
def pipeline_context_with_invalid_data():
    """Create a pipeline context with invalid data for validation testing."""
    context = PipelineContext()
    context.set("normalized_data", SAMPLE_INVALID_DATA)
    return context


class TestDataNormalizationStage:
    """Tests for the DataNormalizationStage class."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test initialization with different configurations."""
        # Default initialization
        stage = DataNormalizationStage()
        assert stage.input_key == "input_data"
        assert stage.output_key == "normalized_data"
        
        # Custom initialization
        config = {
            "input_key": "custom_input",
            "output_key": "custom_output",
            "normalization_rules": {
                "price": {"type": "price"},
                "date_published": {"type": "date"}
            },
            "schema_mappings": {
                "date_published": "published_date"
            },
            "preserve_original": True
        }
        stage = DataNormalizationStage(name="custom_normalizer", config=config)
        assert stage.name == "custom_normalizer"
        assert stage.input_key == "custom_input"
        assert stage.output_key == "custom_output"
        assert "price" in stage.normalization_rules
        assert "date_published" in stage.schema_mappings
        assert stage.preserve_original == True
    
    @pytest.mark.asyncio
    async def test_validate_input(self, pipeline_context):
        """Test input validation."""
        stage = DataNormalizationStage()
        # Valid input
        result = await stage.validate_input(pipeline_context)
        assert result == True
        
        # Invalid input
        empty_context = PipelineContext()
        result = await stage.validate_input(empty_context)
        assert result == False
    
    @pytest.mark.asyncio
    async def test_process(self, pipeline_context):
        """Test the main processing method."""
        # Basic test with default config
        stage = DataNormalizationStage()
        result = await stage.process(pipeline_context)
        
        # Check success
        assert result == True
        
        # Check normalized data
        normalized_data = pipeline_context.get("normalized_data")
        assert normalized_data is not None
        assert isinstance(normalized_data, dict)
        
        # Check specific normalizations
        assert normalized_data["title"] == "Test Product"  # Whitespace trimmed
        assert isinstance(normalized_data["price"], dict)
        assert normalized_data["price"]["amount"] == 199.99
        assert isinstance(normalized_data["date_published"], dict)
        assert normalized_data["in_stock"] is True
        assert isinstance(normalized_data["tags"], list)
        assert len(normalized_data["tags"]) == 3
        
        # Check metadata
        metadata = pipeline_context.get("normalization_metadata")
        assert metadata is not None
        assert "fields_normalized" in metadata
        assert metadata["fields_normalized"] > 0
    
    @pytest.mark.asyncio
    async def test_normalization_with_rules(self, pipeline_context):
        """Test normalization with specific rules."""
        config = {
            "normalization_rules": {
                "price": {
                    "type": "price",
                    "currency": "USD"
                },
                "date_published": {
                    "type": "date",
                    "output_format": "%Y-%m-%d"
                },
                "description": {
                    "type": "text",
                    "strip_html": True
                }
            }
        }
        stage = DataNormalizationStage(config=config)
        result = await stage.process(pipeline_context)
        
        # Check normalized data with rules applied
        normalized_data = pipeline_context.get("normalized_data")
        
        # Price should be normalized to a dict with USD currency
        assert normalized_data["price"]["currency"] == "USD"
        
        # Date should be in the specified format
        assert "date_published" in normalized_data
        assert normalized_data["date_published"]["iso"].startswith("2023-05-15")
        
        # Description should have HTML removed
        assert "<p>" not in normalized_data["description"]
        assert "<b>" not in normalized_data["description"]
    
    @pytest.mark.asyncio
    async def test_schema_mapping(self, pipeline_context):
        """Test schema mapping functionality."""
        config = {
            "schema_mappings": {
                "date_published": "published_date",
                "in_stock": "availability",
                "tags": "categories"
            }
        }
        stage = DataNormalizationStage(config=config)
        result = await stage.process(pipeline_context)
        
        # Check that schema mapping was applied
        normalized_data = pipeline_context.get("normalized_data")
        assert "published_date" in normalized_data
        assert "availability" in normalized_data
        assert "categories" in normalized_data
        
        # Original fields should be removed
        assert "date_published" not in normalized_data
        assert "in_stock" not in normalized_data
        assert "tags" not in normalized_data
    
    @pytest.mark.asyncio
    async def test_preserve_original(self, pipeline_context):
        """Test preserving original values."""
        config = {
            "preserve_original": True
        }
        stage = DataNormalizationStage(config=config)
        result = await stage.process(pipeline_context)
        
        # Check that original values were preserved
        normalized_data = pipeline_context.get("normalized_data")
        
        # Original field values should be under _original key
        assert "_original" in normalized_data
        assert "price" in normalized_data["_original"]
        assert normalized_data["_original"]["price"] == "$199.99"


class TestDataValidationStage:
    """Tests for the DataValidationStage class."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test initialization with different configurations."""
        # Default initialization
        stage = DataValidationStage()
        assert stage.input_key == "normalized_data"
        assert stage.output_key == "validation_results"
        
        # Custom initialization
        config = {
            "input_key": "custom_input",
            "output_key": "custom_output",
            "validation_schema": {
                "title": {"required": True, "min_length": 5},
                "price": {"required": True, "type": "price"},
                "date": {"required": True, "type": "date"}
            },
            "additional_validations": [
                {"field": "rating", "validator": "range", "min": 0, "max": 5}
            ],
            "strict_mode": True
        }
        stage = DataValidationStage(name="custom_validator", config=config)
        assert stage.name == "custom_validator"
        assert stage.input_key == "custom_input"
        assert stage.output_key == "custom_output"
        assert "title" in stage.validation_schema
        assert len(stage.additional_validations) == 1
        assert stage.strict_mode == True
    
    @pytest.mark.asyncio
    async def test_validate_input(self, pipeline_context_with_normalized_data):
        """Test input validation."""
        stage = DataValidationStage()
        # Valid input
        result = await stage.validate_input(pipeline_context_with_normalized_data)
        assert result == True
        
        # Invalid input
        empty_context = PipelineContext()
        result = await stage.validate_input(empty_context)
        assert result == False
    
    @pytest.mark.asyncio
    async def test_process(self, pipeline_context_with_normalized_data):
        """Test the main processing method with valid data."""
        # Configuration with validation rules
        config = {
            "validation_schema": {
                "title": {"required": True, "min_length": 3},
                "price": {"required": True, "type": "price"},
                "date": {"required": True, "type": "date"},
                "description": {"required": True, "min_length": 10},
                "tags": {"required": True, "min_items": 1}
            },
            "strict_mode": False
        }
        stage = DataValidationStage(config=config)
        result = await stage.process(pipeline_context_with_normalized_data)
        
        # Check success
        assert result == True
        
        # Check validation results
        validation_results = pipeline_context_with_normalized_data.get("validation_results")
        assert validation_results is not None
        assert validation_results["is_valid"] == True
        assert validation_results["pass_count"] > 0
        assert validation_results["fail_count"] == 0
        assert len(validation_results["errors"]) == 0
    
    @pytest.mark.asyncio
    async def test_process_with_invalid_data(self, pipeline_context_with_invalid_data):
        """Test the main processing method with invalid data."""
        # Configuration with validation rules
        config = {
            "validation_schema": {
                "title": {"required": True, "min_length": 3},
                "price": {"required": True, "type": "price"},
                "date_published": {"required": True, "type": "date"},
                "description": {"required": True, "min_length": 10},
            },
            "strict_mode": False
        }
        stage = DataValidationStage(config=config)
        result = await stage.process(pipeline_context_with_invalid_data)
        
        # In non-strict mode, process should still return True
        assert result == True
        
        # Check validation results
        validation_results = pipeline_context_with_invalid_data.get("validation_results")
        assert validation_results is not None
        assert validation_results["is_valid"] == False
        assert validation_results["fail_count"] > 0
        assert len(validation_results["errors"]) > 0
        
        # Check specific errors
        errors = validation_results["errors"]
        assert any(e["field"] == "title" for e in errors)
        assert any(e["field"] == "price" for e in errors)
        assert any(e["field"] == "description" for e in errors)
    
    @pytest.mark.asyncio
    async def test_strict_mode(self, pipeline_context_with_invalid_data):
        """Test validation in strict mode."""
        config = {
            "validation_schema": {
                "title": {"required": True, "min_length": 3},
            },
            "strict_mode": True
        }
        stage = DataValidationStage(config=config)
        result = await stage.process(pipeline_context_with_invalid_data)
        
        # In strict mode, process should return False for invalid data
        assert result == False
    
    @pytest.mark.asyncio
    async def test_custom_validators(self, pipeline_context_with_normalized_data):
        """Test custom validation functions."""
        # Add a custom validator function
        def validate_tags_contain_electronics(value, context):
            if isinstance(value, list) and "electronics" in value:
                return True, None
            return False, "Tags must contain 'electronics'"
        
        config = {
            "custom_validators": {
                "tags_electronics": validate_tags_contain_electronics
            },
            "additional_validations": [
                {"field": "tags", "validator": "tags_electronics"}
            ]
        }
        
        # We need to get the class to register our custom validator
        stage = DataValidationStage(config=config)
        stage.register_custom_validator("tags_electronics", validate_tags_contain_electronics)
        
        result = await stage.process(pipeline_context_with_normalized_data)
        
        # Should pass since our sample data has "electronics" in tags
        validation_results = pipeline_context_with_normalized_data.get("validation_results")
        assert validation_results["is_valid"] == True


class TestDataTransformationStage:
    """Tests for the DataTransformationStage class."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test initialization with different configurations."""
        # Default initialization
        stage = DataTransformationStage()
        assert stage.input_key == "normalized_data"
        assert stage.output_key == "transformed_data"
        
        # Custom initialization
        config = {
            "input_key": "custom_input",
            "output_key": "custom_output",
            "transformations": [
                {"type": "rename", "from": "old_name", "to": "new_name"},
                {"type": "filter", "include": ["field1", "field2"]},
                {"type": "modify", "field": "price", "transformer": "multiply", "value": 0.9}
            ],
            "field_mapping": {
                "title": "product_name",
                "description": "product_description"
            }
        }
        stage = DataTransformationStage(name="custom_transformer", config=config)
        assert stage.name == "custom_transformer"
        assert stage.input_key == "custom_input"
        assert stage.output_key == "custom_output"
        assert len(stage.transformations) == 3
        assert "title" in stage.field_mapping
    
    @pytest.mark.asyncio
    async def test_validate_input(self, pipeline_context_with_normalized_data):
        """Test input validation."""
        stage = DataTransformationStage()
        # Valid input
        result = await stage.validate_input(pipeline_context_with_normalized_data)
        assert result == True
        
        # Invalid input
        empty_context = PipelineContext()
        result = await stage.validate_input(empty_context)
        assert result == False
    
    @pytest.mark.asyncio
    async def test_process(self, pipeline_context_with_normalized_data):
        """Test the main processing method."""
        # Basic test with field mapping
        config = {
            "field_mapping": {
                "title": "product_name",
                "price": "product_price",
                "description": "product_description"
            }
        }
        stage = DataTransformationStage(config=config)
        result = await stage.process(pipeline_context_with_normalized_data)
        
        # Check success
        assert result == True
        
        # Check transformed data
        transformed_data = pipeline_context_with_normalized_data.get("transformed_data")
        assert transformed_data is not None
        
        # Check field mapping
        assert "product_name" in transformed_data
        assert "product_price" in transformed_data
        assert "product_description" in transformed_data
        
        # Original fields should be removed
        assert "title" not in transformed_data
        assert "price" not in transformed_data
        assert "description" not in transformed_data
    
    @pytest.mark.asyncio
    async def test_transformation_types(self, pipeline_context_with_normalized_data):
        """Test different transformation types."""
        config = {
            "transformations": [
                {"type": "rename", "from": "title", "to": "product_name"},
                {"type": "filter", "include": ["product_name", "price", "date", "description"]},
                {"type": "modify", "field": "price", "transformer": "extract_value", "path": "amount"},
                {"type": "add", "field": "discount_price", "value": 180.0},
                {"type": "combine", "fields": ["title", "description"], "destination": "full_text", "separator": " - "}
            ]
        }
        stage = DataTransformationStage(config=config)
        result = await stage.process(pipeline_context_with_normalized_data)
        
        # Check transformed data
        transformed_data = pipeline_context_with_normalized_data.get("transformed_data")
        
        # Check rename
        assert "product_name" in transformed_data
        
        # Check filter - only included fields should be present
        assert len(transformed_data) <= 5  # +1 for the new combined field
        
        # Check modify - price should now be just the amount value
        assert isinstance(transformed_data["price"], (int, float))
        
        # Check add - new field should be present
        assert "discount_price" in transformed_data
        assert transformed_data["discount_price"] == 180.0
        
        # Check combine - should have combined fields
        assert "full_text" in transformed_data
    
    @pytest.mark.asyncio
    async def test_nested_transformations(self, pipeline_context_with_normalized_data):
        """Test transformations on nested fields."""
        config = {
            "transformations": [
                {"type": "extract", "field": "price.amount", "destination": "price_value"},
                {"type": "extract", "field": "location.city", "destination": "city"}
            ]
        }
        stage = DataTransformationStage(config=config)
        result = await stage.process(pipeline_context_with_normalized_data)
        
        # Check transformed data
        transformed_data = pipeline_context_with_normalized_data.get("transformed_data")
        
        # Check extracted fields
        assert "price_value" in transformed_data
        assert transformed_data["price_value"] == 199.99
        
        assert "city" in transformed_data
        assert transformed_data["city"] == "New York"
    
    @pytest.mark.asyncio
    async def test_custom_transformer(self, pipeline_context_with_normalized_data):
        """Test custom transformer function."""
        # Define a custom transformer
        def price_with_tax(value, context=None, tax_rate=0.1):
            if isinstance(value, dict) and "amount" in value:
                amount = value["amount"]
                return amount * (1 + tax_rate)
            return value
        
        # Configure transformation with custom transformer
        config = {
            "transformations": [
                {"type": "custom", "field": "price", "destination": "price_with_tax", "function": "add_tax", "tax_rate": 0.1}
            ]
        }
        
        # Create stage and register custom transformer
        stage = DataTransformationStage(config=config)
        stage.register_transformer("add_tax", price_with_tax)
        
        result = await stage.process(pipeline_context_with_normalized_data)
        
        # Check transformed data
        transformed_data = pipeline_context_with_normalized_data.get("transformed_data")
        assert "price_with_tax" in transformed_data
        assert abs(transformed_data["price_with_tax"] - 219.989) < 0.01  # Allow for floating point imprecision


if __name__ == "__main__":
    pytest.main()