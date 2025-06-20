"""
Tests for File Input Stage Module.

This module contains tests for the FileInputStage class, testing its ability to acquire
data from local files with various formats, encodings, and error scenarios.
"""

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from unittest import mock

import chardet
import pytest
import yaml
import xml.etree.ElementTree as ET
from filelock import FileLock

from core.pipeline.stages.input.file_input import FileInputStage, FileFormat
from core.pipeline.context import PipelineContext
from core.pipeline.dto import PipelineRequest, PipelineResponse, ResponseStatus


class TestFileInputStage:
    """Test cases for FileInputStage."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Clean up after tests
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def text_file(self, temp_dir):
        """Create a temporary text file."""
        file_path = os.path.join(temp_dir, "sample.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("This is a sample text file.\nIt has multiple lines.\nLine three.")
        return file_path
    
    @pytest.fixture
    def json_file(self, temp_dir):
        """Create a temporary JSON file."""
        file_path = os.path.join(temp_dir, "sample.json")
        data = {
            "name": "Test Data",
            "values": [1, 2, 3, 4, 5],
            "nested": {
                "key": "value",
                "array": ["a", "b", "c"]
            },
            "boolean": True
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return file_path
    
    @pytest.fixture
    def csv_file(self, temp_dir):
        """Create a temporary CSV file."""
        file_path = os.path.join(temp_dir, "sample.csv")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("id,name,age,city\n")
            f.write("1,John Doe,30,New York\n")
            f.write("2,Jane Smith,25,Los Angeles\n")
            f.write("3,Bob Johnson,40,Chicago\n")
        return file_path
    
    @pytest.fixture
    def xml_file(self, temp_dir):
        """Create a temporary XML file."""
        file_path = os.path.join(temp_dir, "sample.xml")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write("<root>\n")
            f.write("  <person id=\"1\">\n")
            f.write("    <name>John Doe</name>\n")
            f.write("    <age>30</age>\n")
            f.write("    <city>New York</city>\n")
            f.write("  </person>\n")
            f.write("  <person id=\"2\">\n")
            f.write("    <name>Jane Smith</name>\n")
            f.write("    <age>25</age>\n")
            f.write("    <city>Los Angeles</city>\n")
            f.write("  </person>\n")
            f.write("</root>\n")
        return file_path
    
    @pytest.fixture
    def yaml_file(self, temp_dir):
        """Create a temporary YAML file."""
        file_path = os.path.join(temp_dir, "sample.yaml")
        data = {
            "config": {
                "server": "localhost",
                "port": 8080,
                "debug": True
            },
            "users": [
                {"name": "admin", "role": "administrator"},
                {"name": "guest", "role": "viewer"}
            ]
        }
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f)
        return file_path
    
    @pytest.fixture
    def binary_file(self, temp_dir):
        """Create a temporary binary file."""
        file_path = os.path.join(temp_dir, "sample.bin")
        with open(file_path, "wb") as f:
            f.write(b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09")
            f.write(b"\xFF\xFE\xFD\xFC\xFB\xFA\xF9\xF8\xF7\xF6")
        return file_path
    
    @pytest.fixture
    def latin1_file(self, temp_dir):
        """Create a temporary file with Latin-1 encoding."""
        file_path = os.path.join(temp_dir, "latin1.txt")
        with open(file_path, "w", encoding="latin-1") as f:
            f.write("This is Latin-1 encoded text with special characters: é è ê ë à â")
        return file_path
    
    @pytest.fixture
    def multiple_files(self, temp_dir):
        """Create multiple files for batch processing tests."""
        files = []
        for i in range(5):
            file_path = os.path.join(temp_dir, f"batch_file_{i}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"This is batch file {i}\n")
                f.write(f"It contains test data for batch processing.")
            files.append(file_path)
        return files
    
    @pytest.fixture
    def directory_with_files(self, temp_dir):
        """Create a directory with nested files and subdirectories."""
        parent_dir = os.path.join(temp_dir, "test_dir")
        os.makedirs(parent_dir)
        
        # Create files in main directory
        with open(os.path.join(parent_dir, "file1.txt"), "w") as f:
            f.write("File 1 content")
        
        with open(os.path.join(parent_dir, "file2.json"), "w") as f:
            f.write('{"key": "value"}')
        
        # Create subdirectory
        subdir = os.path.join(parent_dir, "subdir")
        os.makedirs(subdir)
        
        # Create files in subdirectory
        with open(os.path.join(subdir, "subfile1.txt"), "w") as f:
            f.write("Subfile 1 content")
        
        with open(os.path.join(subdir, "subfile2.csv"), "w") as f:
            f.write("id,name\n1,John")
        
        return parent_dir
    
    @pytest.fixture
    def symlink_file(self, temp_dir, text_file):
        """Create a symlink to a file."""
        link_path = os.path.join(temp_dir, "symlink.txt")
        os.symlink(text_file, link_path)
        return link_path
    
    @pytest.fixture
    def basic_file_stage(self):
        """Create a basic file input stage."""
        config = {
            "file_path": "",  # Will be set per test
            "encoding": None,  # Auto-detect
            "max_size": 1024 * 1024,  # 1MB
            "follow_symlinks": False
        }
        return FileInputStage(name="test_file_stage", config=config)
    
    @pytest.fixture
    def pattern_file_stage(self):
        """Create a file input stage with pattern matching."""
        config = {
            "pattern": "",  # Will be set per test
            "encoding": None,  # Auto-detect
            "max_size": 1024 * 1024,  # 1MB
            "batch_enabled": False
        }
        return FileInputStage(name="pattern_file_stage", config=config)
    
    @pytest.fixture
    def batch_file_stage(self):
        """Create a file input stage with batch processing."""
        config = {
            "pattern": "",  # Will be set per test
            "encoding": None,  # Auto-detect
            "max_size": 1024 * 1024,  # 1MB
            "batch_enabled": True,
            "max_workers": 2
        }
        return FileInputStage(name="batch_file_stage", config=config)
    
    @pytest.fixture
    def directory_file_stage(self):
        """Create a file input stage for directory processing."""
        config = {
            "file_path": "",  # Will be set per test
            "traversal_options": {
                "process_directories": True,
                "max_depth": 2,
                "include_hidden": False
            }
        }
        return FileInputStage(name="directory_file_stage", config=config)
    
    @pytest.mark.asyncio
    async def test_text_file_reading(self, basic_file_stage, text_file):
        """Test reading a text file."""
        # Update file path in config
        basic_file_stage.file_path = text_file
        
        # Create request and context
        request = PipelineRequest()
        context = PipelineContext({})
        
        # Execute the stage
        response = await basic_file_stage.acquire_data(request, context)
        
        # Verify the response
        assert response is not None
        assert response.status == ResponseStatus.SUCCESS
        assert response.data is not None
        assert "text" in response.data
        assert "line_count" in response.data
        assert response.data["line_count"] == 3
        assert "This is a sample text file." in response.data["text"]
        assert response.source == text_file
        assert response.metadata["format"] == "text"
    
    @pytest.mark.asyncio
    async def test_json_file_reading(self, basic_file_stage, json_file):
        """Test reading a JSON file."""
        # Update file path in config
        basic_file_stage.file_path = json_file
        
        # Create request and context
        request = PipelineRequest()
        context = PipelineContext({})
        
        # Execute the stage
        response = await basic_file_stage.acquire_data(request, context)
        
        # Verify the response
        assert response is not None
        assert response.status == ResponseStatus.SUCCESS
        assert response.data is not None
        assert "json" in response.data
        assert response.data["json"]["name"] == "Test Data"
        assert response.data["json"]["values"] == [1, 2, 3, 4, 5]
        assert response.data["json"]["nested"]["key"] == "value"
        assert response.source == json_file
        assert response.metadata["format"] == "json"
    
    @pytest.mark.asyncio
    async def test_csv_file_reading(self, basic_file_stage, csv_file):
        """Test reading a CSV file."""
        # Update file path in config
        basic_file_stage.file_path = csv_file
        
        # Create request and context
        request = PipelineRequest()
        context = PipelineContext({})
        
        # Execute the stage
        response = await basic_file_stage.acquire_data(request, context)
        
        # Verify the response
        assert response is not None
        assert response.status == ResponseStatus.SUCCESS
        assert response.data is not None
        assert "csv" in response.data
        assert "fieldnames" in response.data
        assert "row_count" in response.data
        assert response.data["row_count"] == 3  # 3 data rows (excluding header)
        assert response.data["fieldnames"] == ["id", "name", "age", "city"]
        assert response.data["csv"][0]["name"] == "John Doe"
        assert response.source == csv_file
        assert response.metadata["format"] == "csv"
    
    @pytest.mark.asyncio
    async def test_xml_file_reading(self, basic_file_stage, xml_file):
        """Test reading an XML file."""
        # Update file path in config
        basic_file_stage.file_path = xml_file
        
        # Create request and context
        request = PipelineRequest()
        context = PipelineContext({})
        
        # Execute the stage
        response = await basic_file_stage.acquire_data(request, context)
        
        # Verify the response
        assert response is not None
        assert response.status == ResponseStatus.SUCCESS
        assert response.data is not None
        assert "xml" in response.data
        assert "xml_string" in response.data
        assert "root_tag" in response.data
        assert response.data["root_tag"] == "root"
        assert response.source == xml_file
        assert response.metadata["format"] == "xml"
    
    @pytest.mark.asyncio
    async def test_yaml_file_reading(self, basic_file_stage, yaml_file):
        """Test reading a YAML file."""
        # Update file path in config
        basic_file_stage.file_path = yaml_file
        
        # Create request and context
        request = PipelineRequest()
        context = PipelineContext({})
        
        # Execute the stage
        response = await basic_file_stage.acquire_data(request, context)
        
        # Verify the response
        assert response is not None
        assert response.status == ResponseStatus.SUCCESS
        assert response.data is not None
        assert "yaml" in response.data
        assert response.data["yaml"]["config"]["server"] == "localhost"
        assert response.data["yaml"]["config"]["port"] == 8080
        assert response.data["yaml"]["users"][0]["name"] == "admin"
        assert response.source == yaml_file
        assert response.metadata["format"] == "yaml"
    
    @pytest.mark.asyncio
    async def test_binary_file_reading(self, basic_file_stage, binary_file):
        """Test reading a binary file."""
        # Update file path in config
        basic_file_stage.file_path = binary_file
        
        # Create request and context
        request = PipelineRequest()
        context = PipelineContext({})
        
        # Execute the stage
        response = await basic_file_stage.acquire_data(request, context)
        
        # Verify the response
        assert response is not None
        assert response.status == ResponseStatus.SUCCESS
        assert response.data is not None
        assert "binary" in response.data
        assert "content_type" in response.data
        assert len(response.data["binary"]) == 20  # 20 bytes written to file
        assert response.source == binary_file
        assert response.metadata["format"] == "binary"
    
    @pytest.mark.asyncio
    async def test_encoding_detection(self, basic_file_stage, latin1_file, monkeypatch):
        """Test automatic encoding detection."""
        # Update file path in config
        basic_file_stage.file_path = latin1_file
        
        # Mock chardet.detect to return latin-1 encoding
        def mock_detect(data):
            return {"encoding": "iso-8859-1", "confidence": 0.99}
        
        # Apply the monkeypatch for chardet.detect
        monkeypatch.setattr(chardet, "detect", mock_detect)
        
        # Create request and context
        request = PipelineRequest()
        context = PipelineContext({})
        
        # Execute the stage
        response = await basic_file_stage.acquire_data(request, context)
        
        # Verify the response
        assert response is not None
        assert response.status == ResponseStatus.SUCCESS
        assert response.data is not None
        assert "text" in response.data
        assert "é è ê ë à â" in response.data["text"], "Special characters not properly decoded"
        assert response.metadata["encoding"] == "iso-8859-1"
    
    @pytest.mark.asyncio
    async def test_explicit_encoding(self, basic_file_stage, latin1_file):
        """Test using explicitly specified encoding."""
        # Update file path in config with explicit encoding
        basic_file_stage.file_path = latin1_file
        basic_file_stage.encoding = "latin-1"
        
        # Create request and context
        request = PipelineRequest()
        context = PipelineContext({})
        
        # Execute the stage
        response = await basic_file_stage.acquire_data(request, context)
        
        # Verify the response
        assert response is not None
        assert response.status == ResponseStatus.SUCCESS
        assert response.data is not None
        assert "text" in response.data
        assert "é è ê ë à â" in response.data["text"], "Special characters not properly decoded"
        assert response.metadata["encoding"] == "latin-1"
    
    @pytest.mark.asyncio
    async def test_pattern_matching(self, pattern_file_stage, temp_dir, multiple_files):
        """Test file pattern matching."""
        # Update pattern in config
        pattern = os.path.join(temp_dir, "batch_file_*.txt")
        pattern_file_stage.pattern = pattern
        
        # Create request and context
        request = PipelineRequest()
        context = PipelineContext({})
        
        # Execute the stage
        response = await pattern_file_stage.acquire_data(request, context)
        
        # Verify the response
        assert response is not None
        assert response.status == ResponseStatus.SUCCESS
        assert response.data is not None
        assert "files" in response.data
        assert "file_count" in response.data
        assert response.data["file_count"] == 5
        assert len(response.data["files"]) == 5
        
        # Check if metadata contains file info
        assert "file_metadata" in response.metadata
        assert len(response.metadata["file_metadata"]) == 5
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, batch_file_stage, temp_dir, multiple_files):
        """Test batch processing of multiple files."""
        # Update pattern in config
        pattern = os.path.join(temp_dir, "batch_file_*.txt")
        batch_file_stage.pattern = pattern
        
        # Create request and context
        request = PipelineRequest()
        context = PipelineContext({})
        
        # Execute the stage
        response = await batch_file_stage.acquire_data(request, context)
        
        # Verify the response
        assert response is not None
        assert response.status == ResponseStatus.SUCCESS
        assert response.data is not None
        assert "batch_results" in response.data
        assert "batch_size" in response.data
        assert response.data["batch_size"] == 5
        assert len(response.data["batch_results"]) == 5
        
        # Check metadata counts
        assert response.metadata["total_files"] == 5
        assert response.metadata["successful_files"] == 5
    
    @pytest.mark.asyncio
    async def test_directory_processing(self, directory_file_stage, directory_with_files):
        """Test directory traversal and processing."""
        # Update file path in config
        directory_file_stage.file_path = directory_with_files
        
        # Create request and context
        request = PipelineRequest()
        context = PipelineContext({})
        
        # Execute the stage
        response = await directory_file_stage.acquire_data(request, context)
        
        # Verify the response
        assert response is not None
        assert response.status == ResponseStatus.SUCCESS
        assert response.data is not None
        assert "directory" in response.data
        
        # Check directory contents
        assert response.data["directory"]["name"] == os.path.basename(directory_with_files)
        assert len(response.data["directory"]["files"]) == 2  # Two files in main directory
        assert len(response.data["directory"]["subdirectories"]) == 1  # One subdirectory
        
        # Check metadata
        assert response.metadata["file_count"] == 2
        assert response.metadata["subdir_count"] == 1
    
    @pytest.mark.asyncio
    async def test_symlink_handling(self, basic_file_stage, symlink_file):
        """Test symlink handling when symlinks are disabled."""
        # Update file path in config
        basic_file_stage.file_path = symlink_file
        basic_file_stage.follow_symlinks = False
        
        # Create request and context
        request = PipelineRequest()
        context = PipelineContext({})
        
        # Execute the stage
        response = await basic_file_stage.acquire_data(request, context)
        
        # Verify the response indicates symlink error
        assert response is not None
        assert response.status == ResponseStatus.ERROR
        assert "symlink" in response.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_symlink_following(self, basic_file_stage, symlink_file):
        """Test symlink handling when symlinks are enabled."""
        # Update file path in config
        basic_file_stage.file_path = symlink_file
        basic_file_stage.follow_symlinks = True
        
        # Create request and context
        request = PipelineRequest()
        context = PipelineContext({})
        
        # Execute the stage
        response = await basic_file_stage.acquire_data(request, context)
        
        # Verify the response shows successful file reading through symlink
        assert response is not None
        assert response.status == ResponseStatus.SUCCESS
        assert response.data is not None
        assert "text" in response.data
        assert "This is a sample text file." in response.data["text"]
    
    @pytest.mark.asyncio
    async def test_file_not_found(self, basic_file_stage, temp_dir):
        """Test handling of non-existent file."""
        # Set non-existent file path
        basic_file_stage.file_path = os.path.join(temp_dir, "nonexistent.txt")
        
        # Create request and context
        request = PipelineRequest()
        context = PipelineContext({})
        
        # Execute the stage
        response = await basic_file_stage.acquire_data(request, context)
        
        # Verify the response indicates file not found
        assert response is not None
        assert response.status == ResponseStatus.NOT_FOUND
        assert "not found" in response.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_file_too_large(self, basic_file_stage, text_file):
        """Test handling of file exceeding size limit."""
        # Update file path in config with small size limit
        basic_file_stage.file_path = text_file
        basic_file_stage.max_size = 10  # 10 bytes, file is larger
        
        # Create request and context
        request = PipelineRequest()
        context = PipelineContext({})
        
        # Execute the stage
        response = await basic_file_stage.acquire_data(request, context)
        
        # Verify the response indicates file too large
        assert response is not None
        assert response.status == ResponseStatus.ERROR
        assert "too large" in response.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_invalid_json(self, basic_file_stage, temp_dir):
        """Test handling of invalid JSON file."""
        # Create invalid JSON file
        file_path = os.path.join(temp_dir, "invalid.json")
        with open(file_path, "w") as f:
            f.write('{"key": "value", "broken": }')  # Invalid JSON
        
        # Update file path in config
        basic_file_stage.file_path = file_path
        
        # Create request and context
        request = PipelineRequest()
        context = PipelineContext({})
        
        # Execute the stage
        response = await basic_file_stage.acquire_data(request, context)
        
        # Verify the response indicates JSON parsing error
        assert response is not None
        assert response.status == ResponseStatus.ERROR
        assert "json" in response.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_format_detection(self, basic_file_stage, temp_dir):
        """Test file format detection."""
        # Create test files with different extensions
        files = {}
        
        for ext, content in [
            (".txt", "Plain text content"),
            (".json", '{"key": "value"}'),
            (".csv", "id,name\n1,Test"),
            (".xml", "<root><item>value</item></root>"),
            (".yaml", "key: value\nlist:\n  - item1\n  - item2"),
            (".jpg", b"\xFF\xD8\xFF\xE0" + b"\x00" * 16)  # Fake JPEG header
        ]:
            file_path = os.path.join(temp_dir, f"test{ext}")
            mode = "w" if isinstance(content, str) else "wb"
            with open(file_path, mode) as f:
                f.write(content)
            files[ext] = file_path
        
        # Expected format mapping
        format_mapping = {
            ".txt": FileFormat.TEXT,
            ".json": FileFormat.JSON,
            ".csv": FileFormat.CSV,
            ".xml": FileFormat.XML,
            ".yaml": FileFormat.YAML,
            ".jpg": FileFormat.BINARY
        }
        
        # Test format detection for each file
        for ext, file_path in files.items():
            expected_format = format_mapping[ext]
            detected_format = basic_file_stage._detect_format(file_path)
            assert detected_format == expected_format, f"Format detection failed for {ext}: got {detected_format}, expected {expected_format}"
    
    @pytest.mark.asyncio
    async def test_file_locking(self, basic_file_stage, text_file, monkeypatch):
        """Test file locking mechanism."""
        # Create a mock for FileLock
        lock_acquired = False
        
        class MockFileLock:
            def __init__(self, path, timeout):
                self.path = path
                self.timeout = timeout
                self.is_locked = False
            
            def acquire(self):
                nonlocal lock_acquired
                self.is_locked = True
                lock_acquired = True
            
            def release(self):
                self.is_locked = False
        
        # Apply monkeypatch
        monkeypatch.setattr(FileLock, "__init__", MockFileLock.__init__)
        monkeypatch.setattr(FileLock, "acquire", MockFileLock.acquire)
        monkeypatch.setattr(FileLock, "release", MockFileLock.release)
        monkeypatch.setattr(FileLock, "is_locked", property(lambda self: self.is_locked))
        
        # Update file path in config
        basic_file_stage.file_path = text_file
        
        # Create request and context
        request = PipelineRequest()
        context = PipelineContext({})
        
        # Execute the stage
        response = await basic_file_stage.acquire_data(request, context)
        
        # Verify lock was acquired
        assert lock_acquired is True
        assert response.status == ResponseStatus.SUCCESS
    
    @pytest.mark.asyncio
    async def test_file_request_source(self, basic_file_stage, text_file):
        """Test using file path from request source."""
        # Set an invalid file path in config
        basic_file_stage.file_path = "invalid_path.txt"
        
        # Create request with valid file path in source
        request = PipelineRequest(source=text_file)
        context = PipelineContext({})
        
        # Execute the stage
        response = await basic_file_stage.acquire_data(request, context)
        
        # Verify the stage used the file path from request source
        assert response is not None
        assert response.status == ResponseStatus.SUCCESS
        assert response.data is not None
        assert "text" in response.data
        assert "This is a sample text file." in response.data["text"]
        assert response.source == text_file
    
    @pytest.mark.asyncio
    async def test_template_pattern_resolution(self, pattern_file_stage, temp_dir, multiple_files):
        """Test pattern template resolution with context variables."""
        # Create a template pattern
        pattern_file_stage.pattern = os.path.join(temp_dir, "batch_file_{file_index}.txt")
        
        # Create request and context with template variable
        request = PipelineRequest(params={"file_index": "2"})
        context = PipelineContext({})
        
        # Execute the stage
        response = await pattern_file_stage.acquire_data(request, context)
        
        # Verify the response shows only one file was matched
        assert response is not None
        assert response.status == ResponseStatus.SUCCESS
        assert response.data is not None
        assert "files" in response.data
        assert response.data["file_count"] == 1
        
        # Verify the correct file was matched
        file_key = os.path.basename(multiple_files[2])
        assert file_key in response.data["files"]
    
    @pytest.mark.asyncio
    async def test_validate_source_config(self, basic_file_stage, temp_dir):
        """Test configuration validation."""
        # Test with no file path or pattern
        assert await basic_file_stage.validate_source_config() is False
        
        # Test with valid file path
        file_path = os.path.join(temp_dir, "test.txt")
        with open(file_path, "w") as f:
            f.write("Test content")
        
        basic_file_stage.file_path = file_path
        assert await basic_file_stage.validate_source_config() is True
        
        # Test with invalid format
        basic_file_stage.format = "INVALID_FORMAT"
        assert await basic_file_stage.validate_source_config() is False
        
        # Test with valid format
        basic_file_stage.format = "TEXT"
        assert await basic_file_stage.validate_source_config() is True
    
    @pytest.mark.asyncio
    async def test_shutdown(self, basic_file_stage):
        """Test resource cleanup on shutdown."""
        # Create mock locks
        basic_file_stage._locks = {
            "file1.lock": mock.MagicMock(is_locked=True),
            "file2.lock": mock.MagicMock(is_locked=False)
        }
        
        # Call shutdown
        await basic_file_stage.shutdown()
        
        # Verify locks were released
        basic_file_stage._locks["file1.lock"].release.assert_called_once()
        basic_file_stage._locks["file2.lock"].release.assert_not_called()
        
        # Verify locks dictionary is cleared
        assert len(basic_file_stage._locks) == 0