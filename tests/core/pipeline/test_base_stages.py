"""
Tests for the base pipeline stages.

This module contains tests for the InputStage, ProcessingStage, OutputStage,
and ConditionalStage base classes.
"""

import asyncio
import json
import os
import pytest
import tempfile
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from core.pipeline.context import PipelineContext
from core.pipeline.stage import PipelineStage
from core.pipeline.stages.base_stages import (
    InputStage, ProcessingStage, OutputStage, ConditionalStage,
    ProcessingMode, OutputMode, ConditionalOperator
)
from core.pipeline.dto import (
    PipelineRequest, PipelineResponse, PipelineMetrics,
    ResponseStatus, RequestMethod, StageMetrics
)


# ====== Test Input Stage ======

class TestHttpInputStage(InputStage):
    """A test HTTP input stage implementation."""
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, config)
        self.mock_responses = config.get("mock_responses", {})
        self.fail_on = config.get("fail_on", set())
        
    async def acquire_data(self, request: PipelineRequest, context: PipelineContext) -> Optional[PipelineResponse]:
        """Simulate HTTP request and return mock response."""
        url = request.source
        
        # Simulate network delay
        await asyncio.sleep(0.1)
        
        # Check if we should fail on this URL
        if url in self.fail_on:
            raise Exception(f"Simulated failure for URL: {url}")
            
        # Return mock response if we have one
        if url in self.mock_responses:
            return self.mock_responses[url]
            
        # Default response
        return PipelineResponse(
            status=ResponseStatus.SUCCESS,
            data={"url": url, "content": f"Content from {url}"},
            source=url,
            headers={"Content-Type": "text/html"},
            status_code=200
        )


class TestFileInputStage(InputStage):
    """A test file input stage implementation."""
    
    async def acquire_data(self, request: PipelineRequest, context: PipelineContext) -> Optional[PipelineResponse]:
        """Simulate file read and return content."""
        file_path = request.source
        
        # Simulate file read delay
        await asyncio.sleep(0.05)
        
        # Check if the file path should fail
        if file_path.endswith("error.txt"):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Return mock file content
        return PipelineResponse(
            status=ResponseStatus.SUCCESS,
            data={"path": file_path, "content": f"Content from {file_path}"},
            source=file_path
        )


# ====== Test Processing Stage ======

class TestHtmlProcessingStage(ProcessingStage):
    """A test HTML processing stage implementation."""
    
    async def transform_data(self, data: Any, context: PipelineContext) -> Optional[Any]:
        """Transform HTML content by extracting a title."""
        # Check if we have HTML content
        if not data or "content" not in data:
            return None
            
        content = data["content"]
        
        # Simulate processing delay
        await asyncio.sleep(0.05)
        
        # Simulate title extraction with a simple pattern
        title = "Extracted Title"
        if "title:" in content.lower():
            title = content.split("title:")[1].strip().split("\n")[0]
            
        # Return processed data
        return {
            "original_content": content,
            "title": title,
            "processing_mode": self.processing_mode.name
        }
        
    async def validate_content(self, context: PipelineContext) -> bool:
        """Validate that we have content to process."""
        # Check for required data in context
        if self.config.get("input_key"):
            return context.has_key(self.config.get("input_key"))
        
        # If we're looking at the whole context, check for content
        return "content" in context.data or any("content" in v for v in context.data.values() if isinstance(v, dict))


class TestDataTransformStage(ProcessingStage):
    """A test data transformation stage implementation."""
    
    async def transform_data(self, data: Any, context: PipelineContext) -> Optional[Any]:
        """Apply data transformations based on configuration."""
        if not data:
            return None
            
        # Get transformation type from config
        transform_type = self.config.get("transform_type", "uppercase")
        
        # Apply different transformations based on type
        result = {}
        
        if transform_type == "uppercase" and isinstance(data, dict):
            # Convert string values to uppercase
            for key, value in data.items():
                if isinstance(value, str):
                    result[key] = value.upper()
                else:
                    result[key] = value
                    
        elif transform_type == "extract_fields" and isinstance(data, dict):
            # Extract only specified fields
            fields = self.config.get("fields", [])
            for field in fields:
                if field in data:
                    result[field] = data[field]
                    
        elif transform_type == "count_items" and isinstance(data, dict):
            # Count items in collections
            for key, value in data.items():
                if isinstance(value, (list, dict, str)):
                    result[f"{key}_count"] = len(value)
                else:
                    result[key] = value
        
        # Simulate processing delay
        await asyncio.sleep(0.05)
        
        return result or data  # Return result or original data if no transformation


# ====== Test Output Stage ======

class TestFileOutputStage(OutputStage):
    """A test file output stage implementation."""
    
    async def deliver_output(self, data: Any, context: PipelineContext) -> Optional[Any]:
        """Simulate writing data to a file."""
        # Get output file path from config
        file_path = self.config.get("file_path", "output.json")
        
        # For testing, we won't actually write to a file, but simulate it
        await asyncio.sleep(0.05)  # Simulate file write delay
        
        # Check if we should simulate failure
        if "fail" in file_path:
            raise PermissionError(f"Permission denied: {file_path}")
            
        # Store the "written" data in context for test verification
        output_format = self.config.get("format", "json")
        if output_format == "json":
            result = {"file_path": file_path, "format": "json", "data": data}
        else:
            result = {"file_path": file_path, "format": output_format, "data": str(data)}
            
        return result
        
    async def validate_output_format(self, context: PipelineContext) -> bool:
        """Validate that output format is supported."""
        output_format = self.config.get("format", "json")
        return output_format in {"json", "text", "csv"}
        
    async def _create_backup(self, context: PipelineContext) -> bool:
        """Simulate creating a backup file."""
        if not self.backup_enabled:
            return True
            
        file_path = self.config.get("file_path", "output.json")
        backup_path = self.backup_path_template.format(path=file_path)
        
        # Simulate backup operation
        await asyncio.sleep(0.02)
        
        # Add backup info to context for verification
        context.set("backup_created", {
            "original_path": file_path,
            "backup_path": backup_path,
            "timestamp": time.time()
        })
        
        # Simulate failure if requested
        return not self.config.get("backup_fails", False)


class TestApiOutputStage(OutputStage):
    """A test API output stage implementation."""
    
    async def deliver_output(self, data: Any, context: PipelineContext) -> Optional[Any]:
        """Simulate sending data to an API."""
        # Get API endpoint from config
        endpoint = self.config.get("endpoint", "https://example.com/api")
        
        # Simulate network delay
        await asyncio.sleep(0.1)
        
        # Check if we should simulate failure
        if "error" in endpoint:
            raise Exception(f"API error: {endpoint}")
            
        # Return simulated API response
        return {
            "endpoint": endpoint,
            "status_code": 200,
            "response": {"success": True, "message": "Data received"}
        }


# ====== Test Conditional Stage ======

@pytest.fixture
def sample_context():
    """Create a sample pipeline context for testing."""
    context = PipelineContext({
        "title": "Sample Title",
        "count": 42,
        "tags": ["test", "sample", "pipeline"],
        "metadata": {
            "author": "Test User",
            "version": "1.0"
        },
        "flag": True,
        "status": "active"
    })
    return context


# ====== Input Stage Tests ======

@pytest.mark.asyncio
async def test_input_stage_success():
    """Test successful data acquisition in input stage."""
    # Setup
    config = {
        "source": "https://example.com",
        "retry_count": 2,
        "throttle_rate": 5.0
    }
    stage = TestHttpInputStage(name="http_input", config=config)
    context = PipelineContext()
    
    # Execute
    result = await stage.process(context)
    
    # Verify
    assert result is True
    assert context.has_key("response")
    assert context.has_key("url")
    assert context.has_key("content")
    assert context.get("url") == "https://example.com"
    
    # Check metrics
    metrics = context.get("input_metrics")
    assert "http_input" in metrics
    assert metrics["http_input"]["count"] == 1
    assert metrics["http_input"]["success"] == 1


@pytest.mark.asyncio
async def test_input_stage_retry():
    """Test retry behavior for transient failures."""
    # Setup
    mock_response = PipelineResponse(
        status=ResponseStatus.SUCCESS,
        data={"url": "https://example.com/retry", "content": "Retry content"},
        source="https://example.com/retry"
    )
    
    config = {
        "retry_count": 2,
        "retry_delay": 0.1,
        "fail_on": {"https://example.com/retry"},
        "mock_responses": {"https://example.com/retry": mock_response}
    }
    stage = TestHttpInputStage(name="retry_test", config=config)
    context = PipelineContext()
    
    # Create a version of acquire_data that fails once then succeeds
    original_acquire = stage.acquire_data
    failure_count = [0]
    
    async def mock_acquire(request, ctx):
        if request.source == "https://example.com/retry" and failure_count[0] < 1:
            failure_count[0] += 1
            raise Exception("Simulated transient error")
        return await original_acquire(request, ctx)
    
    # Replace the method temporarily
    stage.acquire_data = mock_acquire
    
    # Execute
    result = await stage.process(context)
    
    # Restore the original method
    stage.acquire_data = original_acquire
    
    # Verify
    assert result is True
    assert failure_count[0] == 1
    assert context.has_key("content")
    assert "Retry content" in context.get("content")


@pytest.mark.asyncio
async def test_input_stage_throttling():
    """Test request throttling in input stage."""
    # Setup
    config = {
        "throttle_rate": 10.0  # 10 requests per second = 0.1s between requests
    }
    stage = TestHttpInputStage(name="throttle_test", config=config)
    context = PipelineContext()
    
    # Execute multiple requests and measure time
    start_time = time.time()
    await stage.process(context)
    await stage.process(context)
    elapsed = time.time() - start_time
    
    # Verify there was throttling delay
    # With 10 req/sec, two sequential requests should take at least 0.1s
    # Plus the simulated network delay of 0.1s per request
    assert elapsed >= 0.2, f"Expected at least 0.2s delay, got {elapsed}s"


@pytest.mark.asyncio
async def test_input_stage_failure():
    """Test handling of failures in input stage."""
    # Setup
    config = {
        "fail_on": {"https://example.com/error"},
        "retry_count": 1
    }
    stage = TestHttpInputStage(name="error_test", config=config)
    context = PipelineContext()
    
    # Set source to one that will fail
    request = PipelineRequest(source="https://example.com/error")
    context.set("request", request)
    
    # Execute
    result = await stage.process(context)
    
    # Verify
    assert result is False
    assert "input_metrics" in context.data
    assert context.get("input_metrics")["error_test"]["success"] == 0


# ====== Processing Stage Tests ======

@pytest.mark.asyncio
async def test_processing_stage_transform():
    """Test data transformation in processing stage."""
    # Setup
    config = {
        "processing_mode": "COPY"
    }
    stage = TestHtmlProcessingStage(name="html_processor", config=config)
    context = PipelineContext({
        "content": "This is a test page with title: Test Page Title\nAnd some content."
    })
    
    # Execute
    result = await stage.process(context)
    
    # Verify
    assert result is True
    assert context.has_key("title")
    assert context.get("title") == "Test Page Title"
    assert context.get("processing_mode") == "COPY"
    assert "processing_metrics" in context.data
    
    # Original content should still be present (copy mode)
    assert context.get("original_content") == "This is a test page with title: Test Page Title\nAnd some content."


@pytest.mark.asyncio
async def test_processing_stage_validation():
    """Test content validation in processing stage."""
    # Setup
    stage = TestHtmlProcessingStage(name="validation_test")
    
    # Empty context should fail validation
    empty_context = PipelineContext({})
    
    # Context with content should pass validation
    valid_context = PipelineContext({"content": "Test content"})
    
    # Execute and verify
    assert await stage.validate_content(empty_context) is False
    assert await stage.validate_content(valid_context) is True
    
    # Test validation in process method
    result = await stage.process(empty_context)
    assert result is False
    assert empty_context.has_key("processing_metrics")
    
    # There should be an error in the context
    assert any("validation_test" in source for source in empty_context.metadata["errors"])


@pytest.mark.asyncio
async def test_processing_stage_caching():
    """Test result caching in processing stage."""
    # Setup
    config = {
        "enable_caching": True,
        "processing_mode": "IN_PLACE"
    }
    stage = TestHtmlProcessingStage(name="cache_test", config=config)
    context = PipelineContext({
        "content": "Page with title: Cached Title\nContent here."
    })
    
    # Execute twice with the same input
    start_time = time.time()
    await stage.process(context)
    first_duration = time.time() - start_time
    
    # Reset the context but keep the same content to test caching
    context = PipelineContext({
        "content": "Page with title: Cached Title\nContent here."
    })
    
    start_time = time.time()
    await stage.process(context)
    second_duration = time.time() - start_time
    
    # Verify
    # The second execution should be much faster due to caching
    assert second_duration < first_duration
    assert context.get("title") == "Cached Title"
    
    # Check metrics
    metrics = context.get("processing_metrics")["cache_test"]
    assert metrics["cached_hits"] == 1
    

@pytest.mark.asyncio
async def test_processing_stage_modes():
    """Test different processing modes."""
    # Test IN_PLACE mode
    in_place_config = {"processing_mode": "IN_PLACE"}
    in_place_stage = TestDataTransformStage(name="in_place_test", config=in_place_config)
    in_place_context = PipelineContext({"field": "value"})
    
    await in_place_stage.process(in_place_context)
    assert in_place_context.get("field") == "VALUE"  # Transformed in place
    
    # Test COPY mode
    copy_config = {"processing_mode": "COPY", "transform_type": "extract_fields", "fields": ["field1"]}
    copy_stage = TestDataTransformStage(name="copy_test", config=copy_config)
    copy_context = PipelineContext({"field1": "value1", "field2": "value2"})
    
    await copy_stage.process(copy_context)
    assert copy_context.has_key("field1")  # From the copied and transformed result
    assert copy_context.has_key("field2")  # Original field still present
    
    # Test NEW mode with output_key
    new_config = {
        "processing_mode": "NEW", 
        "transform_type": "count_items",
        "output_key": "counts"
    }
    new_stage = TestDataTransformStage(name="new_test", config=new_config)
    new_context = PipelineContext({
        "list": [1, 2, 3],
        "dict": {"a": 1, "b": 2},
        "string": "test"
    })
    
    await new_stage.process(new_context)
    assert new_context.has_key("counts")
    assert new_context.get("counts").get("list_count") == 3
    assert new_context.get("counts").get("dict_count") == 2
    assert new_context.get("counts").get("string_count") == 4


# ====== Output Stage Tests ======

@pytest.mark.asyncio
async def test_output_stage_success():
    """Test successful output delivery."""
    # Setup
    config = {
        "file_path": "test_output.json",
        "format": "json"
    }
    stage = TestFileOutputStage(name="file_output", config=config)
    context = PipelineContext({
        "result": {"key": "value", "count": 42}
    })
    
    # Execute
    result = await stage.process(context)
    
    # Verify
    assert result is True
    assert context.has_key("file_output_result")
    assert context.get("file_output_result")["file_path"] == "test_output.json"
    assert context.get("file_output_result")["format"] == "json"
    assert context.get("file_output_result")["data"]["result"]["key"] == "value"
    
    # Check metrics
    metrics = context.get("output_metrics")
    assert metrics["file_output"]["count"] == 1
    assert metrics["file_output"]["success"] == 1
    assert metrics["file_output"]["operations"]["success"] == 1


@pytest.mark.asyncio
async def test_output_stage_backup():
    """Test backup creation and rollback."""
    # Setup
    config = {
        "file_path": "backup_test.json",
        "backup_enabled": True
    }
    stage = TestFileOutputStage(name="backup_test", config=config)
    context = PipelineContext({"data": "backup test"})
    
    # Execute
    result = await stage.process(context)
    
    # Verify backup was created
    assert result is True
    assert context.has_key("backup_created")
    assert context.get("backup_created")["original_path"] == "backup_test.json"
    assert context.get("backup_created")["backup_path"] == "backup_test.json.bak"
    
    # Test backup failure
    fail_config = {
        "file_path": "backup_fail.json",
        "backup_enabled": True,
        "backup_fails": True,
        "require_backup": True  # This should cause the stage to fail
    }
    fail_stage = TestFileOutputStage(name="fail_backup", config=fail_config)
    fail_context = PipelineContext({"data": "backup failure test"})
    
    fail_result = await fail_stage.process(fail_context)
    assert fail_result is False
    assert any("fail_backup" in source for source in fail_context.metadata["errors"])


@pytest.mark.asyncio
async def test_output_stage_failure():
    """Test handling of output delivery failures."""
    # Setup - configuration that will cause delivery to fail
    config = {
        "file_path": "fail.json",
        "format": "json"
    }
    stage = TestFileOutputStage(name="failure_test", config=config)
    context = PipelineContext({"data": "test data"})
    
    # Execute
    result = await stage.process(context)
    
    # Verify
    assert result is False
    assert "output_metrics" in context.data
    assert context.get("output_metrics")["failure_test"]["success"] == 0
    assert context.get("output_metrics")["failure_test"]["operations"]["failure"] == 1


@pytest.mark.asyncio
async def test_output_mode():
    """Test different output modes."""
    # We can't actually test file operations fully in a unit test,
    # but we can verify that the mode is correctly passed to the deliver_output method
    
    # Setup
    overwrite_config = {"output_mode": "OVERWRITE", "file_path": "overwrite.json"}
    append_config = {"output_mode": "APPEND", "file_path": "append.json"}
    update_config = {"output_mode": "UPDATE", "file_path": "update.json"}
    
    overwrite_stage = TestFileOutputStage(name="overwrite_test", config=overwrite_config)
    append_stage = TestFileOutputStage(name="append_test", config=append_config)
    update_stage = TestFileOutputStage(name="update_test", config=update_config)
    
    context = PipelineContext({"data": "mode test"})
    
    # Execute
    await overwrite_stage.process(context)
    await append_stage.process(context)
    await update_stage.process(context)
    
    # Verify
    assert overwrite_stage.output_mode == OutputMode.OVERWRITE
    assert append_stage.output_mode == OutputMode.APPEND
    assert update_stage.output_mode == OutputMode.UPDATE


# ====== Conditional Stage Tests ======

@pytest.mark.asyncio
async def test_conditional_equals(sample_context):
    """Test EQUALS operator in conditional stage."""
    config = {
        "conditions": [
            {"key": "title", "operator": "EQUALS", "value": "Sample Title"}
        ]
    }
    stage = ConditionalStage(name="equals_test", config=config)
    
    # Execute
    result = await stage.process(sample_context)
    
    # Verify
    assert result is True
    assert sample_context.get("equals_test_result") is True
    
    # Test with non-matching value
    config["conditions"][0]["value"] = "Different Title"
    stage = ConditionalStage(name="equals_false", config=config)
    result = await stage.process(sample_context)
    assert result is False


@pytest.mark.asyncio
async def test_conditional_complex(sample_context):
    """Test complex conditions with logical operators."""
    # Test AND operator (default)
    and_config = {
        "conditions": [
            {"key": "count", "operator": "GREATER_THAN", "value": 40},
            {"key": "status", "operator": "EQUALS", "value": "active"}
        ],
        "logical_operator": "AND"
    }
    and_stage = ConditionalStage(name="and_test", config=and_config)
    and_result = await and_stage.process(sample_context)
    assert and_result is True
    
    # Test OR operator
    or_config = {
        "conditions": [
            {"key": "count", "operator": "LESS_THAN", "value": 40},  # False
            {"key": "status", "operator": "EQUALS", "value": "active"}  # True
        ],
        "logical_operator": "OR"
    }
    or_stage = ConditionalStage(name="or_test", config=or_config)
    or_result = await or_stage.process(sample_context)
    assert or_result is True
    
    # Test negation
    not_config = {
        "conditions": [
            {"key": "flag", "operator": "EQUALS", "value": True, "negate": True}
        ]
    }
    not_stage = ConditionalStage(name="not_test", config=not_config)
    not_result = await not_stage.process(sample_context)
    assert not_result is False


@pytest.mark.asyncio
async def test_conditional_exists_and_contains(sample_context):
    """Test EXISTS and CONTAINS operators."""
    # Test EXISTS
    exists_config = {
        "conditions": [
            {"key": "nonexistent", "operator": "EXISTS"}
        ]
    }
    exists_stage = ConditionalStage(name="exists_test", config=exists_config)
    exists_result = await exists_stage.process(sample_context)
    assert exists_result is False
    
    # Test NOT_EXISTS
    not_exists_config = {
        "conditions": [
            {"key": "nonexistent", "operator": "NOT_EXISTS"}
        ]
    }
    not_exists_stage = ConditionalStage(name="not_exists_test", config=not_exists_config)
    not_exists_result = await not_exists_stage.process(sample_context)
    assert not_exists_result is True
    
    # Test CONTAINS with list
    contains_config = {
        "conditions": [
            {"key": "tags", "operator": "CONTAINS", "value": "sample"}
        ]
    }
    contains_stage = ConditionalStage(name="contains_test", config=contains_config)
    contains_result = await contains_stage.process(sample_context)
    assert contains_result is True
    
    # Test NOT_CONTAINS
    not_contains_config = {
        "conditions": [
            {"key": "tags", "operator": "NOT_CONTAINS", "value": "nonexistent"}
        ]
    }
    not_contains_stage = ConditionalStage(name="not_contains_test", config=not_contains_config)
    not_contains_result = await not_contains_stage.process(sample_context)
    assert not_contains_result is True


@pytest.mark.asyncio
async def test_conditional_advanced_operators(sample_context):
    """Test advanced operators like REGEX_MATCH and IS_TYPE."""
    # Test REGEX_MATCH
    regex_config = {
        "conditions": [
            {"key": "title", "operator": "REGEX_MATCH", "value": "Sample.*"}
        ]
    }
    regex_stage = ConditionalStage(name="regex_test", config=regex_config)
    regex_result = await regex_stage.process(sample_context)
    assert regex_result is True
    
    # Test IS_TYPE
    type_config = {
        "conditions": [
            {"key": "count", "operator": "IS_TYPE", "value": "int"}
        ]
    }
    type_stage = ConditionalStage(name="type_test", config=type_config)
    type_result = await type_stage.process(sample_context)
    assert type_result is True
    
    # Test with wrong type
    wrong_type_config = {
        "conditions": [
            {"key": "count", "operator": "IS_TYPE", "value": "str"}
        ]
    }
    wrong_type_stage = ConditionalStage(name="wrong_type_test", config=wrong_type_config)
    wrong_type_result = await wrong_type_stage.process(sample_context)
    assert wrong_type_result is False


@pytest.mark.asyncio
async def test_conditional_metrics(sample_context):
    """Test that conditional stages record metrics properly."""
    config = {
        "conditions": [
            {"key": "flag", "operator": "EQUALS", "value": True}
        ]
    }
    stage = ConditionalStage(name="metrics_test", config=config)
    
    # Execute multiple times to generate metrics
    for _ in range(3):
        await stage.process(sample_context)
    
    # Change the flag and process again
    sample_context.set("flag", False)
    await stage.process(sample_context)
    
    # Verify metrics
    metrics = sample_context.get("conditional_metrics")["metrics_test"]
    assert metrics["count"] == 4
    assert metrics["success"] == 4
    assert metrics["true_count"] == 3
    assert metrics["false_count"] == 1
    assert metrics["true_percentage"] == 75.0