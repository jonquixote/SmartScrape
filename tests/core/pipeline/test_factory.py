"""
Tests for the PipelineFactory class.

This module contains tests for creating pipelines with different configuration sources,
validation, versioning, and other factory features.
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from core.pipeline.factory import (
    PipelineFactory, 
    ConfigurationValidationError,
    TemplateNotFoundError,
    VersionCompatibilityError
)
from core.pipeline.pipeline import Pipeline
from core.pipeline.stage import PipelineStage
from core.pipeline.registry import PipelineRegistry
from core.pipeline.context import PipelineContext


# Helper classes for testing
class TestStage(PipelineStage):
    """A test pipeline stage for testing."""
    
    async def process(self, context: PipelineContext) -> bool:
        """Process the stage."""
        context.set('test_stage_output', 'processed')
        return True


# Fixtures
@pytest.fixture
def factory():
    """Create a PipelineFactory instance with a mock registry."""
    registry = PipelineRegistry()
    return PipelineFactory(registry)


@pytest.fixture
def valid_config():
    """Return a valid pipeline configuration."""
    return {
        "name": "test_pipeline",
        "description": "Pipeline for testing",
        "version": "1.0.0",
        "pipeline_config": {
            "parallel_execution": False,
            "continue_on_error": True
        },
        "stages": [
            {
                "stage": "TestStage",
                "config": {
                    "name": "test_stage_1"
                }
            },
            {
                "stage": "TestStage",
                "config": {
                    "name": "test_stage_2"
                }
            }
        ]
    }


@pytest.fixture
def mock_registry():
    """Create a mock registry for testing."""
    registry = MagicMock(spec=PipelineRegistry)
    
    # Set up the create_pipeline method as an AsyncMock
    registry.create_pipeline = AsyncMock()
    registry.create_pipeline.return_value = Pipeline("test_pipeline")
    
    # Set up the register_pipeline method
    registry.register_pipeline = MagicMock()
    
    # Set up get_registered_pipelines method
    registry.get_registered_pipelines = MagicMock(return_value=["existing_pipeline"])
    
    # Set up load_pipeline_config_from_file method
    registry.load_pipeline_config_from_file = MagicMock()
    
    return registry


# Tests for create_pipeline_from_config
@pytest.mark.asyncio
async def test_create_pipeline_from_config(mock_registry, valid_config):
    """Test creating a pipeline from a configuration dictionary."""
    factory = PipelineFactory(mock_registry)
    
    # Mock the validation method to avoid validation errors
    factory.validate_pipeline_config = MagicMock()
    
    pipeline = await factory.create_pipeline_from_config(valid_config)
    
    # Verify the pipeline was created
    assert pipeline is not None
    
    # Verify the registry methods were called correctly
    mock_registry.register_pipeline.assert_called_once_with("test_pipeline", valid_config)
    mock_registry.create_pipeline.assert_called_once_with("test_pipeline")


@pytest.mark.asyncio
async def test_create_pipeline_from_config_validation_error(mock_registry, valid_config):
    """Test that validation errors are raised when creating a pipeline."""
    factory = PipelineFactory(mock_registry)
    
    # Mock the validation method to raise an error
    error_message = "Validation failed"
    factory.validate_pipeline_config = MagicMock(side_effect=ConfigurationValidationError(error_message))
    
    # Verify the error is raised
    with pytest.raises(ConfigurationValidationError, match=error_message):
        await factory.create_pipeline_from_config(valid_config)


# Tests for create_pipeline_from_file
@pytest.mark.asyncio
async def test_create_pipeline_from_file(mock_registry, valid_config, tmp_path):
    """Test creating a pipeline from a configuration file."""
    factory = PipelineFactory(mock_registry)
    
    # Create a temporary config file
    file_path = tmp_path / "test_config.json"
    with open(file_path, 'w') as f:
        json.dump(valid_config, f)
    
    # Mock the validation method to avoid validation errors
    factory.validate_pipeline_config = MagicMock()
    
    # Mock the registry to return the config
    mock_registry.load_pipeline_config_from_file.return_value = valid_config
    
    # Create the pipeline from the file
    pipeline = await factory.create_pipeline_from_file(str(file_path))
    
    # Verify the pipeline was created
    assert pipeline is not None
    
    # Verify the registry methods were called correctly
    mock_registry.load_pipeline_config_from_file.assert_called_once_with(str(file_path))
    mock_registry.create_pipeline.assert_called_once()


@pytest.mark.asyncio
async def test_create_pipeline_from_file_not_found(mock_registry):
    """Test that FileNotFoundError is raised when file doesn't exist."""
    factory = PipelineFactory(mock_registry)
    
    # Mock the registry to raise FileNotFoundError
    mock_registry.load_pipeline_config_from_file.side_effect = FileNotFoundError("File not found")
    
    # Verify the error is raised
    with pytest.raises(FileNotFoundError):
        await factory.create_pipeline_from_file("non_existent_file.json")


# Tests for create_pipeline_from_template
@pytest.mark.asyncio
async def test_create_pipeline_from_template(mock_registry):
    """Test creating a pipeline from a template."""
    factory = PipelineFactory(mock_registry)
    
    # Mock the _load_template_module method
    template_module = MagicMock()
    create_pipeline_func = AsyncMock(return_value=Pipeline("template_pipeline"))
    template_module.create_pipeline = create_pipeline_func
    factory._load_template_module = MagicMock(return_value=template_module)
    
    # Create the pipeline from the template
    params = {"param1": "value1", "param2": "value2"}
    pipeline = await factory.create_pipeline_from_template("test_template", params)
    
    # Verify the pipeline was created
    assert pipeline is not None
    assert pipeline.name == "template_pipeline"
    
    # Verify the template function was called with the parameters
    create_pipeline_func.assert_called_once_with(**params)


@pytest.mark.asyncio
async def test_create_pipeline_from_template_not_found(mock_registry):
    """Test that TemplateNotFoundError is raised when template doesn't exist."""
    factory = PipelineFactory(mock_registry)
    
    # Mock the _load_template_module method to return None
    factory._load_template_module = MagicMock(return_value=None)
    
    # Verify the error is raised
    with pytest.raises(TemplateNotFoundError):
        await factory.create_pipeline_from_template("non_existent_template")


# Tests for create_custom_pipeline
@pytest.mark.asyncio
async def test_create_custom_pipeline_with_pipeline_return(mock_registry):
    """Test creating a pipeline with a builder function that returns a Pipeline."""
    factory = PipelineFactory(mock_registry)
    
    # Create a builder function that returns a Pipeline
    async def builder_func(param1=None, param2=None):
        pipeline = Pipeline("custom_pipeline")
        pipeline.add_stage(TestStage({"name": "test_stage"}))
        return pipeline
    
    # Create the pipeline with the builder function
    params = {"param1": "value1", "param2": "value2"}
    pipeline = await factory.create_custom_pipeline(builder_func, params)
    
    # Verify the pipeline was created
    assert pipeline is not None
    assert pipeline.name == "custom_pipeline"
    assert len(pipeline.stages) == 1


@pytest.mark.asyncio
async def test_create_custom_pipeline_with_config_return(mock_registry, valid_config):
    """Test creating a pipeline with a builder function that returns a config dict."""
    factory = PipelineFactory(mock_registry)
    
    # Create a builder function that returns a config dict
    async def builder_func(param1=None, param2=None):
        return valid_config
    
    # Mock the create_pipeline_from_config method
    factory.create_pipeline_from_config = AsyncMock(return_value=Pipeline("custom_pipeline"))
    
    # Create the pipeline with the builder function
    params = {"param1": "value1", "param2": "value2"}
    pipeline = await factory.create_custom_pipeline(builder_func, params)
    
    # Verify the pipeline was created
    assert pipeline is not None
    
    # Verify create_pipeline_from_config was called with the config
    factory.create_pipeline_from_config.assert_called_once_with(valid_config, None)


# Tests for pipeline builder creation
def test_get_pipeline_builder(factory):
    """Test getting a pipeline builder instance."""
    builder = factory.get_pipeline_builder()
    
    # Verify the builder was created and has a reference to the factory
    assert builder is not None
    assert builder.factory == factory


# Tests for configuration validation
def test_validate_pipeline_config_basic(factory, valid_config):
    """Test basic pipeline configuration validation."""
    # Mock the registry's validation method
    factory.registry._validate_pipeline_config = MagicMock()
    
    # Call the validate method
    factory.validate_pipeline_config(valid_config)
    
    # Verify the registry's validation method was called
    factory.registry._validate_pipeline_config.assert_called_once_with(valid_config)


def test_validate_pipeline_config_version_error(factory, valid_config):
    """Test pipeline configuration validation with version incompatibility."""
    # Set an incompatible version
    valid_config["version"] = "99.0.0"
    
    # Verify the error is raised
    with pytest.raises(ConfigurationValidationError, match="Pipeline version 99.0.0 is not supported"):
        factory.validate_pipeline_config(valid_config)


def test_validate_pipeline_config_flow_error(factory, valid_config):
    """Test pipeline configuration validation with flow errors."""
    # Add invalid connections
    valid_config["connections"] = [
        {"from": "existing_stage", "to": "non_existent_stage"}
    ]
    
    # Verify the error is raised
    with pytest.raises(ConfigurationValidationError, match="Connection references non-existent 'to' stage"):
        factory.validate_pipeline_config(valid_config)


def test_validate_placeholders(factory):
    """Test placeholder validation in configuration."""
    # Create a config with placeholders
    config = {
        "name": "test",
        "variables": {
            "var1": "value1"
        },
        "stage_config": {
            "param1": "${var1}",           # Valid placeholder
            "param2": "${non_existent}",   # Invalid placeholder
        }
    }
    
    # Verify the error is raised
    with pytest.raises(ConfigurationValidationError, match="Undefined placeholder"):
        factory._validate_placeholders(config)


# Tests for configuration utilities
def test_merge_configurations(factory):
    """Test merging configurations."""
    base = {
        "name": "base",
        "config": {
            "param1": "value1",
            "param2": "value2"
        }
    }
    
    override = {
        "config": {
            "param2": "new_value",
            "param3": "value3"
        }
    }
    
    # Mock the registry's merge method
    factory.registry._merge_configurations = MagicMock(return_value={
        "name": "base",
        "config": {
            "param1": "value1",
            "param2": "new_value",
            "param3": "value3"
        }
    })
    
    # Call the merge method
    result = factory.merge_configurations(base, override)
    
    # Verify the registry's merge method was called
    factory.registry._merge_configurations.assert_called_once_with(base, override)
    
    # Verify the result
    assert result["config"]["param2"] == "new_value"
    assert result["config"]["param3"] == "value3"


def test_resolve_placeholder(factory):
    """Test resolving placeholders in configuration."""
    # Create a context for substitution
    context = {
        "var1": "value1",
        "var2": "value2"
    }
    
    # Test simple string placeholder
    value = "This is ${var1} and ${var2}"
    result = factory.resolve_placeholder(value, context)
    assert result == "This is value1 and value2"
    
    # Test nested dictionary placeholder
    value = {
        "param1": "${var1}",
        "nested": {
            "param2": "${var2}"
        }
    }
    result = factory.resolve_placeholder(value, context)
    assert result["param1"] == "value1"
    assert result["nested"]["param2"] == "value2"
    
    # Test list placeholder
    value = ["${var1}", "${var2}"]
    result = factory.resolve_placeholder(value, context)
    assert result[0] == "value1"
    assert result[1] == "value2"


def test_analyze_pipeline(factory):
    """Test pipeline analysis."""
    # Create a pipeline with some stages
    pipeline = Pipeline("test_pipeline")
    pipeline.add_stage(TestStage({"name": "stage1"}))
    pipeline.add_stage(TestStage({"name": "stage2"}))
    
    # Analyze the pipeline
    analysis = factory.analyze_pipeline(pipeline)
    
    # Verify the analysis results
    assert analysis["name"] == "test_pipeline"
    assert analysis["stage_count"] == 2
    assert len(analysis["stages"]) == 2
    assert analysis["stages"][0]["name"] == "stage1"
    assert analysis["stages"][1]["name"] == "stage2"