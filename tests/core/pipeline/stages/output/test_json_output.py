"""
Tests for the JSON output stage.

This module contains tests for the JSONOutputStage that handles formatting
and writing pipeline data as JSON.
"""

import asyncio
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from core.pipeline.context import PipelineContext
from core.pipeline.stages.output.json_output import JSONOutputStage, JSONFormattingMode


class TestJSONOutputStage(unittest.TestCase):
    """Test cases for the JSONOutputStage."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_data = {
            "string": "test",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "array": [1, 2, 3],
            "object": {"a": 1, "b": 2},
            "null": None
        }
        self.context = PipelineContext()
        self.context.set("data", self.test_data)
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        stage = JSONOutputStage()
        self.assertEqual(stage.formatting_mode, JSONFormattingMode.PRETTY)
        self.assertEqual(stage.indent, 2)
        self.assertFalse(stage.ensure_ascii)
        self.assertFalse(stage.sort_keys)
        self.assertIsNone(stage.schema)
        self.assertFalse(stage.streaming)
        self.assertIsNone(stage.file_path)
        self.assertEqual(stage.encoding, "utf-8")
    
    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        config = {
            "formatting_mode": "COMPACT",
            "indent": 4,
            "ensure_ascii": True,
            "sort_keys": True,
            "streaming": True,
            "streaming_batch_size": 500,
            "file_path": "/tmp/test.json",
            "encoding": "latin-1"
        }
        stage = JSONOutputStage(config=config)
        self.assertEqual(stage.formatting_mode, JSONFormattingMode.COMPACT)
        self.assertEqual(stage.indent, 4)
        self.assertTrue(stage.ensure_ascii)
        self.assertTrue(stage.sort_keys)
        self.assertTrue(stage.streaming)
        self.assertEqual(stage.streaming_batch_size, 500)
        self.assertEqual(stage.file_path, "/tmp/test.json")
        self.assertEqual(stage.encoding, "latin-1")
    
    def test_init_with_invalid_mode(self):
        """Test initialization with an invalid formatting mode."""
        config = {"formatting_mode": "INVALID"}
        stage = JSONOutputStage(config=config)
        self.assertEqual(stage.formatting_mode, JSONFormattingMode.PRETTY)
    
    def test_validate_output_format_without_schema(self):
        """Test output validation without a schema."""
        stage = JSONOutputStage()
        asyncio.run(stage.validate_output_format(self.context))
        self.assertFalse(self.context.has_errors())
    
    @patch('jsonschema.Draft7Validator')
    def test_validate_output_format_with_valid_schema(self, mock_validator):
        """Test output validation with a valid schema."""
        # Set up mock validator that passes validation
        mock_validator_instance = MagicMock()
        mock_validator_instance.iter_errors.return_value = []
        mock_validator.return_value = mock_validator_instance
        
        # Create stage with schema
        schema = {"type": "object", "properties": {"string": {"type": "string"}}}
        stage = JSONOutputStage(config={"schema": schema})
        
        # Run validation
        result = asyncio.run(stage.validate_output_format(self.context))
        
        # Check results
        self.assertTrue(result)
        self.assertFalse(self.context.has_errors())
        mock_validator_instance.iter_errors.assert_called_once()
    
    @patch('jsonschema.Draft7Validator')
    def test_validate_output_format_with_invalid_data(self, mock_validator):
        """Test output validation with invalid data against schema."""
        # Set up mock validator that fails validation
        error = MagicMock()
        error.path = ["string"]
        error.message = "validation error"
        
        mock_validator_instance = MagicMock()
        mock_validator_instance.iter_errors.return_value = [error]
        mock_validator.return_value = mock_validator_instance
        
        # Create stage with schema and allow_validation_errors=True
        schema = {"type": "object", "properties": {"string": {"type": "number"}}}
        stage = JSONOutputStage(config={
            "schema": schema,
            "allow_validation_errors": True
        })
        
        # Run validation
        result = asyncio.run(stage.validate_output_format(self.context))
        
        # Check results - should pass despite errors because allow_validation_errors=True
        self.assertTrue(result)
        self.assertTrue(self.context.has_errors())
        
        # Test with allow_validation_errors=False
        stage = JSONOutputStage(config={
            "schema": schema,
            "allow_validation_errors": False
        })
        
        # Run validation
        result = asyncio.run(stage.validate_output_format(self.context))
        
        # Check results - should fail
        self.assertFalse(result)
    
    def test_format_json_pretty(self):
        """Test formatting JSON in pretty mode."""
        stage = JSONOutputStage(config={"formatting_mode": "PRETTY", "indent": 2})
        result = asyncio.run(stage._format_json(self.test_data, self.context))
        
        # Verify the result
        self.assertEqual(result["format"], "json")
        self.assertEqual(result["mode"], "PRETTY")
        
        # Check if the JSON was stored in context
        formatted_json = self.context.get("JSONOutputStage_json")
        self.assertIsNotNone(formatted_json)
        
        # Parse and validate
        parsed = json.loads(formatted_json)
        self.assertEqual(parsed, self.test_data)
    
    def test_format_json_compact(self):
        """Test formatting JSON in compact mode."""
        stage = JSONOutputStage(config={"formatting_mode": "COMPACT"})
        result = asyncio.run(stage._format_json(self.test_data, self.context))
        
        # Verify the result
        self.assertEqual(result["format"], "json")
        self.assertEqual(result["mode"], "COMPACT")
        
        # Check if the JSON was stored in context
        formatted_json = self.context.get("JSONOutputStage_json")
        self.assertIsNotNone(formatted_json)
        
        # Verify no extra whitespace
        self.assertNotIn("  ", formatted_json)
        
        # Parse and validate
        parsed = json.loads(formatted_json)
        self.assertEqual(parsed, self.test_data)
    
    def test_format_json_lines(self):
        """Test formatting JSON in lines mode."""
        # Test with list data
        list_data = [{"id": 1}, {"id": 2}, {"id": 3}]
        self.context.set("data", list_data)
        
        stage = JSONOutputStage(config={"formatting_mode": "LINES"})
        result = asyncio.run(stage._format_json(list_data, self.context))
        
        # Verify the result
        self.assertEqual(result["format"], "json")
        self.assertEqual(result["mode"], "LINES")
        
        # Check if the JSON was stored in context
        formatted_json = self.context.get("JSONOutputStage_json")
        self.assertIsNotNone(formatted_json)
        
        # Should have newlines between items
        self.assertEqual(formatted_json.count("\n"), len(list_data) - 1)
        
        # Parse and validate each line
        lines = formatted_json.split("\n")
        for i, line in enumerate(lines):
            parsed = json.loads(line)
            self.assertEqual(parsed, list_data[i])
    
    def test_write_to_file(self):
        """Test writing JSON to a file."""
        # Create a file path in the temporary directory
        file_path = os.path.join(self.temp_dir.name, "test_output.json")
        
        # Create stage with file output
        stage = JSONOutputStage(config={
            "formatting_mode": "PRETTY",
            "file_path": file_path
        })
        
        # Write to file
        result = asyncio.run(stage._write_to_file(self.test_data, self.context))
        
        # Verify the result
        self.assertEqual(result["format"], "json")
        self.assertEqual(result["mode"], "PRETTY")
        self.assertEqual(result["file_path"], file_path)
        self.assertTrue(os.path.exists(file_path))
        
        # Read back and verify
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            parsed = json.loads(content)
            self.assertEqual(parsed, self.test_data)
    
    def test_stream_to_file(self):
        """Test streaming large JSON data to a file."""
        # Create a file path in the temporary directory
        file_path = os.path.join(self.temp_dir.name, "test_stream.json")
        
        # Create large test data
        large_data = [{"id": i, "value": f"test{i}"} for i in range(1000)]
        
        # Create stage with streaming configuration
        stage = JSONOutputStage(config={
            "formatting_mode": "PRETTY",
            "file_path": file_path,
            "streaming": True,
            "streaming_batch_size": 100
        })
        
        # Stream to file
        result = asyncio.run(stage._stream_to_file(large_data, self.context))
        
        # Verify the result
        self.assertEqual(result["format"], "json")
        self.assertEqual(result["mode"], "PRETTY")
        self.assertEqual(result["file_path"], file_path)
        self.assertTrue(result["streaming"])
        self.assertEqual(result["items"], 1000)
        self.assertTrue(os.path.exists(file_path))
        
        # Read back and verify
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            parsed = json.loads(content)
            self.assertEqual(parsed, large_data)
    
    def test_json_streaming_lines_mode(self):
        """Test streaming JSON in lines mode."""
        # Create a file path in the temporary directory
        file_path = os.path.join(self.temp_dir.name, "test_lines.jsonl")
        
        # Create test data
        line_data = [{"id": i, "value": f"test{i}"} for i in range(100)]
        
        # Create stage with lines mode
        stage = JSONOutputStage(config={
            "formatting_mode": "LINES",
            "file_path": file_path,
            "streaming": True,
            "streaming_batch_size": 10
        })
        
        # Stream to file
        result = asyncio.run(stage._stream_to_file(line_data, self.context))
        
        # Verify the result
        self.assertEqual(result["format"], "json")
        self.assertEqual(result["mode"], "LINES")
        self.assertTrue(os.path.exists(file_path))
        
        # Read back and verify - each line should be a valid JSON object
        lines = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                lines.append(json.loads(line.strip()))
        
        self.assertEqual(len(lines), 100)
        self.assertEqual(lines, line_data)
    
    def test_deliver_output_direct_response(self):
        """Test deliver_output with direct response (no file)."""
        stage = JSONOutputStage(config={"formatting_mode": "PRETTY"})
        result = asyncio.run(stage.deliver_output(self.test_data, self.context))
        
        # Verify the result
        self.assertEqual(result["format"], "json")
        self.assertEqual(result["mode"], "PRETTY")
        
        # Check JSON in context
        formatted_json = self.context.get("JSONOutputStage_json")
        self.assertIsNotNone(formatted_json)
    
    def test_deliver_output_to_file(self):
        """Test deliver_output with file output."""
        file_path = os.path.join(self.temp_dir.name, "output.json")
        stage = JSONOutputStage(config={
            "formatting_mode": "PRETTY",
            "file_path": file_path
        })
        
        result = asyncio.run(stage.deliver_output(self.test_data, self.context))
        
        # Verify the result
        self.assertEqual(result["format"], "json")
        self.assertEqual(result["file_path"], file_path)
        self.assertTrue(os.path.exists(file_path))
    
    def test_deliver_output_with_error(self):
        """Test deliver_output with an error during processing."""
        # Create a stage with an invalid file path to cause an error
        stage = JSONOutputStage(config={
            "file_path": "/nonexistent/directory/file.json",
            "create_directories": False  # Prevent directory creation
        })
        
        # Should handle the error gracefully
        result = asyncio.run(stage.deliver_output(self.test_data, self.context))
        
        # Result should be None due to error
        self.assertIsNone(result)
        
        # Context should have an error
        self.assertTrue(self.context.has_errors())
    
    def test_get_config_schema(self):
        """Test getting the configuration schema."""
        stage = JSONOutputStage()
        schema = stage.get_config_schema()
        
        # Verify schema structure
        self.assertIsInstance(schema, dict)
        self.assertIn("properties", schema)
        
        # Check for JSON-specific properties
        props = schema["properties"]
        self.assertIn("formatting_mode", props)
        self.assertIn("indent", props)
        self.assertIn("ensure_ascii", props)
        self.assertIn("sort_keys", props)
        self.assertIn("schema", props)
        self.assertIn("streaming", props)
        self.assertIn("file_path", props)
        self.assertIn("encoding", props)


if __name__ == "__main__":
    unittest.main()