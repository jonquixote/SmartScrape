"""
Pipeline Registry Test Module.

This module contains tests for the PipelineRegistry class that manages
pipeline configurations and stage registrations.
"""

import asyncio
import json
import os
import tempfile
import threading
import pytest
from typing import Any, Dict, List, Optional, Type

from core.pipeline.registry import (
    PipelineRegistry, PipelineRegistryError, PipelineNotFoundError,
    StageNotFoundError, ConfigurationError
)
from core.pipeline.stage import PipelineStage
from core.pipeline.context import PipelineContext
from core.pipeline.pipeline import Pipeline


# Mock stage classes for testing
class MockStage(PipelineStage):
    """A simple mock stage for testing the registry."""

    async def process(self, context: PipelineContext) -> bool:
        """Process the mock stage."""
        context.set("mock_processed", True)
        return True


class MockStageWithSchema(PipelineStage):
    """A mock stage with a configuration schema."""

    async def process(self, context: PipelineContext) -> bool:
        """Process the mock stage with schema."""
        # Use configuration values if present
        if "message" in self.config:
            context.set("message", self.config["message"])
        if "count" in self.config:
            context.set("count", self.config["count"])
        return True

    def get_config_schema(self) -> Dict[str, Any]:
        """Get JSON schema for this stage's configuration."""
        return {
            "type": "object",
            "required": ["message"],
            "properties": {
                "message": {"type": "string"},
                "count": {"type": "integer"}
            }
        }


class FailingStage(PipelineStage):
    """A mock stage that always fails."""

    async def process(self, context: PipelineContext) -> bool:
        """Process the failing stage."""
        context.set("failing_processed", True)
        return False


class ConditionalStage(PipelineStage):
    """A mock stage that can be configured to succeed or fail."""

    async def process(self, context: PipelineContext) -> bool:
        """Process the conditional stage."""
        context.set("conditional_processed", True)
        # Succeed or fail based on configuration
        return self.config.get("succeed", True)


# Test fixtures
@pytest.fixture
def registry():
    """Create a fresh registry for each test."""
    registry = PipelineRegistry()
    registry.clear()  # Ensure it's empty
    return registry


@pytest.fixture
def sample_config():
    """Create a sample pipeline configuration."""
    return {
        "description": "A test pipeline",
        "version": "1.0.0",
        "tags": ["test"],
        "pipeline_config": {
            "continue_on_error": True
        },
        "stages": [
            {
                "stage": "MockStage",
                "config": {}
            }
        ]
    }


@pytest.fixture
def complex_config():
    """Create a more complex pipeline configuration with multiple stages."""
    return {
        "description": "A complex test pipeline",
        "version": "2.0.0",
        "tags": ["test", "complex"],
        "pipeline_config": {
            "continue_on_error": False,
            "parallel_execution": True,
            "max_workers": 3
        },
        "stages": [
            {
                "stage": "MockStage",
                "config": {}
            },
            {
                "stage": "MockStageWithSchema",
                "config": {
                    "message": "Hello from config",
                    "count": 42
                }
            },
            {
                "stage": "ConditionalStage",
                "config": {
                    "succeed": True
                }
            }
        ]
    }


# Basic registration tests
def test_registry_singleton():
    """Test that PipelineRegistry is a singleton."""
    registry1 = PipelineRegistry()
    registry2 = PipelineRegistry()
    assert registry1 is registry2


def test_register_pipeline(registry, sample_config):
    """Test registering a pipeline configuration."""
    registry.register_pipeline("test_pipeline", sample_config)
    assert "test_pipeline" in registry.get_registered_pipelines()


def test_register_stage(registry):
    """Test registering a stage class."""
    registry.register_stage(MockStage)
    assert "MockStage" in registry.get_registered_stages()


def test_register_multiple_stages(registry):
    """Test registering multiple stage classes."""
    registry.register_stage(MockStage)
    registry.register_stage(MockStageWithSchema)
    registry.register_stage(FailingStage)
    assert set(registry.get_registered_stages()) == {"MockStage", "MockStageWithSchema", "FailingStage"}


def test_get_pipeline_info(registry, sample_config):
    """Test getting pipeline information."""
    registry.register_pipeline("test_pipeline", sample_config)
    info = registry.get_pipeline_info("test_pipeline")
    assert info["name"] == "test_pipeline"
    assert info["description"] == "A test pipeline"
    assert info["version"] == "1.0.0"
    assert "test" in info["tags"]


def test_get_stage_info(registry):
    """Test getting stage information."""
    registry.register_stage(MockStageWithSchema)
    info = registry.get_stage_info("MockStageWithSchema")
    assert info["name"] == "MockStageWithSchema"
    assert "configuration schema" in info["description"].lower() or "config schema" in info["description"].lower()
    assert "config_schema" in info


# Pipeline creation tests
@pytest.mark.asyncio
async def test_create_pipeline(registry, sample_config):
    """Test creating a pipeline from a registered configuration."""
    registry.register_stage(MockStage)
    registry.register_pipeline("test_pipeline", sample_config)
    
    pipeline = await registry.create_pipeline("test_pipeline")
    assert pipeline.name == "test_pipeline"
    assert len(pipeline.stages) == 1
    assert isinstance(pipeline.stages[0], MockStage)


@pytest.mark.asyncio
async def test_create_pipeline_with_override(registry, sample_config):
    """Test creating a pipeline with configuration overrides."""
    registry.register_stage(MockStage)
    registry.register_pipeline("test_pipeline", sample_config)
    
    override = {
        "pipeline_config": {
            "continue_on_error": False,
            "enable_monitoring": True
        }
    }
    
    pipeline = await registry.create_pipeline("test_pipeline", override)
    assert not pipeline.config.get("continue_on_error")
    assert pipeline.config.get("enable_monitoring")


@pytest.mark.asyncio
async def test_create_complex_pipeline(registry, complex_config):
    """Test creating a pipeline with multiple stages."""
    registry.register_stage(MockStage)
    registry.register_stage(MockStageWithSchema)
    registry.register_stage(ConditionalStage)
    registry.register_pipeline("complex_pipeline", complex_config)
    
    pipeline = await registry.create_pipeline("complex_pipeline")
    assert pipeline.name == "complex_pipeline"
    assert len(pipeline.stages) == 3
    assert isinstance(pipeline.stages[0], MockStage)
    assert isinstance(pipeline.stages[1], MockStageWithSchema)
    assert isinstance(pipeline.stages[2], ConditionalStage)
    
    # Test that configuration was properly passed to stages
    stage = pipeline.stages[1]
    assert stage.config["message"] == "Hello from config"
    assert stage.config["count"] == 42


@pytest.mark.asyncio
async def test_pipeline_execution(registry, complex_config):
    """Test that a created pipeline can be executed correctly."""
    registry.register_stage(MockStage)
    registry.register_stage(MockStageWithSchema)
    registry.register_stage(ConditionalStage)
    registry.register_pipeline("complex_pipeline", complex_config)
    
    pipeline = await registry.create_pipeline("complex_pipeline")
    context = await pipeline.execute()
    
    # Check that all stages were executed
    assert context.get("mock_processed") == True
    assert context.get("message") == "Hello from config"
    assert context.get("count") == 42
    assert context.get("conditional_processed") == True


# Error handling tests
def test_register_invalid_stage_class(registry):
    """Test registering an invalid stage class."""
    class NotAStage:
        pass
    
    with pytest.raises(TypeError):
        registry.register_stage(NotAStage)


def test_register_pipeline_with_invalid_config(registry):
    """Test registering a pipeline with an invalid configuration."""
    # Missing stages section
    invalid_config = {
        "description": "Invalid pipeline"
    }
    
    with pytest.raises(ConfigurationError):
        registry.register_pipeline("invalid_pipeline", invalid_config)


def test_register_pipeline_with_invalid_stage_config(registry):
    """Test registering a pipeline with invalid stage configuration."""
    # Stage config is not a dictionary
    invalid_config = {
        "stages": [
            {
                "stage": "MockStage", 
                "config": "not a dict"
            }
        ]
    }
    
    with pytest.raises(ConfigurationError):
        registry.register_pipeline("invalid_pipeline", invalid_config)


def test_create_nonexistent_pipeline(registry):
    """Test creating a pipeline that doesn't exist."""
    with pytest.raises(PipelineNotFoundError):
        asyncio.run(registry.create_pipeline("nonexistent_pipeline"))


@pytest.mark.asyncio
async def test_create_pipeline_with_missing_stage(registry, sample_config):
    """Test creating a pipeline with a stage that's not registered."""
    # Don't register the stage
    registry.register_pipeline("test_pipeline", sample_config)
    
    with pytest.raises(StageNotFoundError):
        await registry.create_pipeline("test_pipeline")


def test_validate_schema_with_missing_required(registry):
    """Test schema validation with missing required fields."""
    registry.register_stage(MockStageWithSchema)
    
    # Missing required 'message' field
    invalid_config = {
        "stages": [
            {
                "stage": "MockStageWithSchema",
                "config": {
                    "count": 42  # missing 'message'
                }
            }
        ]
    }
    
    with pytest.raises(ConfigurationError) as excinfo:
        registry.register_pipeline("invalid_pipeline", invalid_config)
    
    # Check that the error message mentions the missing field
    assert "Missing required field 'message'" in str(excinfo.value)


def test_validate_schema_with_wrong_type(registry):
    """Test schema validation with wrong field types."""
    registry.register_stage(MockStageWithSchema)
    
    # Wrong type for 'count' field (should be integer)
    invalid_config = {
        "stages": [
            {
                "stage": "MockStageWithSchema",
                "config": {
                    "message": "Hello",
                    "count": "not an integer"
                }
            }
        ]
    }
    
    with pytest.raises(ConfigurationError) as excinfo:
        registry.register_pipeline("invalid_pipeline", invalid_config)
    
    # Check that the error message mentions the type issue
    assert "Field 'count' must be an integer" in str(excinfo.value)


# File handling tests
def test_load_config_from_json_file(registry):
    """Test loading a pipeline configuration from a JSON file."""
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w+", delete=False) as temp:
        config = {
            "stages": [
                {
                    "stage": "MockStage",
                    "config": {}
                }
            ]
        }
        json.dump(config, temp)
        temp_path = temp.name
    
    try:
        loaded_config = registry.load_pipeline_config_from_file(temp_path)
        assert loaded_config["stages"][0]["stage"] == "MockStage"
    finally:
        os.unlink(temp_path)


def test_load_config_from_nonexistent_file(registry):
    """Test loading from a file that doesn't exist."""
    with pytest.raises(FileNotFoundError):
        registry.load_pipeline_config_from_file("/path/to/nonexistent/file.json")


def test_load_config_from_invalid_json(registry):
    """Test loading a configuration from an invalid JSON file."""
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w+", delete=False) as temp:
        temp.write("This is not valid JSON")
        temp_path = temp.name
    
    try:
        with pytest.raises(ConfigurationError) as excinfo:
            registry.load_pipeline_config_from_file(temp_path)
        assert "Invalid JSON" in str(excinfo.value)
    finally:
        os.unlink(temp_path)


def test_register_pipeline_from_file(registry):
    """Test registering a pipeline from a file in one step."""
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w+", delete=False) as temp:
        config = {
            "stages": [
                {
                    "stage": "MockStage",
                    "config": {}
                }
            ]
        }
        json.dump(config, temp)
        temp_path = temp.name
    
    try:
        registry.register_pipeline_from_file("file_pipeline", temp_path)
        assert "file_pipeline" in registry.get_registered_pipelines()
    finally:
        os.unlink(temp_path)


# Thread safety tests
def test_concurrent_registration(registry):
    """Test concurrent registration of pipelines and stages."""
    def register_pipelines():
        """Register multiple pipelines in a thread."""
        for i in range(10):
            config = {
                "stages": [
                    {
                        "stage": "MockStage",
                        "config": {}
                    }
                ]
            }
            registry.register_pipeline(f"thread_pipeline_{i}", config)
    
    def register_stages():
        """Register multiple stage classes in a thread."""
        # Create and register some dynamic stage classes
        for i in range(10):
            # Create a dynamic stage class
            stage_name = f"DynamicStage_{i}"
            
            # Use type() to create a new class dynamically
            stage_class = type(
                stage_name,
                (PipelineStage,),
                {
                    "process": lambda self, context: asyncio.create_task(asyncio.sleep(0)),
                    "name": stage_name
                }
            )
            
            registry.register_stage(stage_class)
    
    # Register MockStage first (needed for pipeline registration)
    registry.register_stage(MockStage)
    
    # Create and start threads
    threads = []
    for _ in range(3):
        t1 = threading.Thread(target=register_pipelines)
        t2 = threading.Thread(target=register_stages)
        threads.extend([t1, t2])
        t1.start()
        t2.start()
    
    # Wait for all threads to complete
    for t in threads:
        t.join()
    
    # Verify that all registrations succeeded
    registered_pipelines = registry.get_registered_pipelines()
    registered_stages = registry.get_registered_stages()
    
    for i in range(10):
        assert f"thread_pipeline_{i}" in registered_pipelines
        assert f"DynamicStage_{i}" in registered_stages


# Configuration merging tests
def test_merge_configurations(registry):
    """Test merging of pipeline configurations."""
    base_config = {
        "description": "Base pipeline",
        "pipeline_config": {
            "continue_on_error": True,
            "base_setting": "value"
        },
        "stages": [
            {"stage": "StageA", "config": {}},
            {"stage": "StageB", "config": {}}
        ]
    }
    
    override_config = {
        "description": "Override pipeline",
        "pipeline_config": {
            "continue_on_error": False,
            "new_setting": "new_value"
        },
        "stages": [
            {"stage": "StageC", "config": {}},
        ]
    }
    
    # Merge the configurations using the registry's private method
    # We're testing an implementation detail here, which is not ideal
    # but necessary to fully test the merging logic
    merged = registry._merge_configurations(base_config, override_config)
    
    # Check that overrides took effect
    assert merged["description"] == "Override pipeline"
    assert merged["pipeline_config"]["continue_on_error"] == False
    
    # Check that new values were added
    assert merged["pipeline_config"]["new_setting"] == "new_value"
    
    # Check that base values not in override were preserved
    assert merged["pipeline_config"]["base_setting"] == "value"
    
    # Check that stages were completely replaced
    assert len(merged["stages"]) == 1
    assert merged["stages"][0]["stage"] == "StageC"