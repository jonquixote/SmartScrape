"""
Tests for the file output stages.

This module contains tests for the FileOutputStage base class and its
specialized subclasses for different file formats.
"""

import asyncio
import csv
import os
import stat
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.pipeline.context import PipelineContext
from core.pipeline.stages.output.file_output import (
    FileOutputStage, FileNamingStrategy,
    CSVOutputStage, XMLOutputStage, TextOutputStage
)

# Check if PyYAML is available for YAMLOutputStage tests
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class TestFileOutputStage(unittest.TestCase):
    """Test cases for the FileOutputStage base class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_data = "This is test content"
        self.context = PipelineContext()
        self.context.set("data", self.test_data)
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        file_path = os.path.join(self.temp_dir.name, "test.txt")
        stage = FileOutputStage(config={"file_path": file_path})
        
        self.assertEqual(stage.file_path, file_path)
        self.assertEqual(stage.naming_strategy, FileNamingStrategy.FIXED)
        self.assertIsNone(stage.file_permissions)
        self.assertTrue(stage.overwrite_existing)
        self.assertTrue(stage.create_directories)
        self.assertTrue(stage.atomic_write)
        self.assertEqual(stage.encoding, "utf-8")
    
    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        file_path = os.path.join(self.temp_dir.name, "test.txt")
        config = {
            "file_path": file_path,
            "naming_strategy": "TIMESTAMP",
            "file_permissions": 0o644,
            "overwrite_existing": False,
            "create_directories": False,
            "atomic_write": False,
            "encoding": "latin-1"
        }
        
        stage = FileOutputStage(config=config)
        
        self.assertEqual(stage.file_path, file_path)
        self.assertEqual(stage.naming_strategy, FileNamingStrategy.TIMESTAMP)
        self.assertEqual(stage.file_permissions, 0o644)
        self.assertFalse(stage.overwrite_existing)
        self.assertFalse(stage.create_directories)
        self.assertFalse(stage.atomic_write)
        self.assertEqual(stage.encoding, "latin-1")
    
    def test_init_with_invalid_naming_strategy(self):
        """Test initialization with an invalid naming strategy."""
        file_path = os.path.join(self.temp_dir.name, "test.txt")
        config = {
            "file_path": file_path,
            "naming_strategy": "INVALID"
        }
        
        stage = FileOutputStage(config=config)
        self.assertEqual(stage.naming_strategy, FileNamingStrategy.FIXED)
    
    def test_init_without_file_path(self):
        """Test initialization without a file path."""
        with self.assertRaises(ValueError):
            FileOutputStage(config={})
    
    def test_validate_output_format_success(self):
        """Test successful output validation."""
        file_path = os.path.join(self.temp_dir.name, "test.txt")
        stage = FileOutputStage(config={"file_path": file_path})
        
        result = asyncio.run(stage.validate_output_format(self.context))
        self.assertTrue(result)
    
    def test_validate_output_format_nonexistent_directory(self):
        """Test validation with nonexistent directory."""
        file_path = os.path.join(self.temp_dir.name, "nonexistent", "test.txt")
        
        # Test with create_directories=True
        stage = FileOutputStage(config={
            "file_path": file_path,
            "create_directories": True
        })
        
        result = asyncio.run(stage.validate_output_format(self.context))
        self.assertTrue(result)
        
        # Test with create_directories=False
        stage = FileOutputStage(config={
            "file_path": file_path,
            "create_directories": False
        })
        
        result = asyncio.run(stage.validate_output_format(self.context))
        self.assertFalse(result)
        self.assertTrue(self.context.has_errors())
    
    def test_get_output_path_fixed(self):
        """Test getting output path with FIXED naming strategy."""
        file_path = os.path.join(self.temp_dir.name, "test.txt")
        stage = FileOutputStage(config={
            "file_path": file_path,
            "naming_strategy": "FIXED"
        })
        
        output_path = asyncio.run(stage._get_output_path())
        self.assertEqual(output_path, file_path)
    
    def test_get_output_path_timestamp(self):
        """Test getting output path with TIMESTAMP naming strategy."""
        file_path = os.path.join(self.temp_dir.name, "test.txt")
        stage = FileOutputStage(config={
            "file_path": file_path,
            "naming_strategy": "TIMESTAMP"
        })
        
        output_path = asyncio.run(stage._get_output_path())
        self.assertNotEqual(output_path, file_path)
        self.assertTrue(os.path.dirname(output_path), os.path.dirname(file_path))
        self.assertTrue(Path(output_path).name.startswith("test_"))
        self.assertTrue(Path(output_path).name.endswith(".txt"))
    
    def test_get_output_path_sequential(self):
        """Test getting output path with SEQUENTIAL naming strategy."""
        file_path = os.path.join(self.temp_dir.name, "test.txt")
        stage = FileOutputStage(config={
            "file_path": file_path,
            "naming_strategy": "SEQUENTIAL"
        })
        
        # First call should return test_0001.txt
        output_path1 = asyncio.run(stage._get_output_path())
        self.assertTrue(Path(output_path1).name.endswith("_0001.txt"))
        
        # Second call should return test_0002.txt
        output_path2 = asyncio.run(stage._get_output_path())
        self.assertTrue(Path(output_path2).name.endswith("_0002.txt"))
    
    def test_get_output_path_content_hash(self):
        """Test getting output path with CONTENT_HASH naming strategy."""
        file_path = os.path.join(self.temp_dir.name, "test.txt")
        stage = FileOutputStage(config={
            "file_path": file_path,
            "naming_strategy": "CONTENT_HASH"
        })
        
        data = "test content"
        output_path = asyncio.run(stage._get_output_path(data))
        self.assertNotEqual(output_path, file_path)
        
        # Same content should produce same hash
        output_path2 = asyncio.run(stage._get_output_path(data))
        self.assertEqual(output_path, output_path2)
        
        # Different content should produce different hash
        output_path3 = asyncio.run(stage._get_output_path("different content"))
        self.assertNotEqual(output_path, output_path3)
    
    def test_deliver_output_basic(self):
        """Test basic deliver_output functionality."""
        file_path = os.path.join(self.temp_dir.name, "test.txt")
        stage = FileOutputStage(config={"file_path": file_path})
        
        result = asyncio.run(stage.deliver_output(self.test_data, self.context))
        
        # Verify the result
        self.assertEqual(result["file_path"], file_path)
        self.assertEqual(result["format"], "text")
        self.assertTrue(os.path.exists(file_path))
        
        # Read back and verify
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertEqual(content, self.test_data)
    
    def test_deliver_output_with_permissions(self):
        """Test deliver_output with file permissions."""
        file_path = os.path.join(self.temp_dir.name, "test.txt")
        permissions = 0o600  # User read/write only
        
        stage = FileOutputStage(config={
            "file_path": file_path,
            "file_permissions": permissions
        })
        
        result = asyncio.run(stage.deliver_output(self.test_data, self.context))
        
        # Verify the file was created with correct permissions
        self.assertTrue(os.path.exists(file_path))
        file_stat = os.stat(file_path)
        self.assertEqual(file_stat.st_mode & 0o777, permissions)
    
    def test_deliver_output_with_directory_creation(self):
        """Test deliver_output with directory creation."""
        file_path = os.path.join(self.temp_dir.name, "subdir", "test.txt")
        
        stage = FileOutputStage(config={
            "file_path": file_path,
            "create_directories": True
        })
        
        result = asyncio.run(stage.deliver_output(self.test_data, self.context))
        
        # Verify the directories were created
        self.assertTrue(os.path.exists(os.path.dirname(file_path)))
        self.assertTrue(os.path.exists(file_path))
    
    def test_deliver_output_without_atomic_write(self):
        """Test deliver_output without atomic write."""
        file_path = os.path.join(self.temp_dir.name, "test.txt")
        
        stage = FileOutputStage(config={
            "file_path": file_path,
            "atomic_write": False
        })
        
        result = asyncio.run(stage.deliver_output(self.test_data, self.context))
        
        # Verify the file was created
        self.assertTrue(os.path.exists(file_path))
        
        # Read back and verify
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertEqual(content, self.test_data)
    
    def test_deliver_output_with_error(self):
        """Test deliver_output with an error during processing."""
        # Use a directory as file path to cause an error
        file_path = self.temp_dir.name  # This is a directory, not a file
        
        stage = FileOutputStage(config={"file_path": file_path})
        
        result = asyncio.run(stage.deliver_output(self.test_data, self.context))
        
        # Result should be None due to error
        self.assertIsNone(result)
        
        # Context should have an error
        self.assertTrue(self.context.has_errors())


class TestCSVOutputStage(unittest.TestCase):
    """Test cases for the CSVOutputStage."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Test data as list of dictionaries
        self.dict_data = [
            {"id": 1, "name": "Alice", "age": 30},
            {"id": 2, "name": "Bob", "age": 25},
            {"id": 3, "name": "Charlie", "age": 35}
        ]
        
        # Test data as list of lists
        self.list_data = [
            ["id", "name", "age"],
            [1, "Alice", 30],
            [2, "Bob", 25],
            [3, "Charlie", 35]
        ]
        
        # Dictionary of lists
        self.columns_data = {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [30, 25, 35]
        }
        
        self.context = PipelineContext()
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        file_path = os.path.join(self.temp_dir.name, "test")
        stage = CSVOutputStage(config={"file_path": file_path})
        
        self.assertEqual(stage.file_path, file_path + ".csv")
        self.assertEqual(stage.delimiter, ",")
        self.assertTrue(stage.has_header)
        self.assertIsNone(stage.columns)
    
    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        file_path = os.path.join(self.temp_dir.name, "test.csv")
        config = {
            "file_path": file_path,
            "delimiter": ";",
            "quoting": "QUOTE_ALL",
            "has_header": False,
            "columns": ["id", "name"]
        }
        
        stage = CSVOutputStage(config=config)
        
        self.assertEqual(stage.file_path, file_path)
        self.assertEqual(stage.delimiter, ";")
        self.assertEqual(stage.quoting, csv.QUOTE_ALL)
        self.assertFalse(stage.has_header)
        self.assertEqual(stage.columns, ["id", "name"])
    
    def test_validate_output_format_list_of_dicts(self):
        """Test validation with list of dictionaries."""
        file_path = os.path.join(self.temp_dir.name, "test.csv")
        stage = CSVOutputStage(config={"file_path": file_path})
        
        self.context.set("data", self.dict_data)
        result = asyncio.run(stage.validate_output_format(self.context))
        self.assertTrue(result)
    
    def test_validate_output_format_list_of_lists(self):
        """Test validation with list of lists."""
        file_path = os.path.join(self.temp_dir.name, "test.csv")
        stage = CSVOutputStage(config={"file_path": file_path})
        
        self.context.set("data", self.list_data)
        result = asyncio.run(stage.validate_output_format(self.context))
        self.assertTrue(result)
    
    def test_validate_output_format_dict_of_lists(self):
        """Test validation with dictionary of lists."""
        file_path = os.path.join(self.temp_dir.name, "test.csv")
        stage = CSVOutputStage(config={"file_path": file_path})
        
        self.context.set("data", self.columns_data)
        result = asyncio.run(stage.validate_output_format(self.context))
        self.assertTrue(result)
    
    def test_validate_output_format_invalid_data(self):
        """Test validation with invalid data."""
        file_path = os.path.join(self.temp_dir.name, "test.csv")
        stage = CSVOutputStage(config={"file_path": file_path})
        
        # Mix of dictionaries and lists
        invalid_data = [{"id": 1}, [2, 3], {"name": "test"}]
        self.context.set("data", invalid_data)
        
        result = asyncio.run(stage.validate_output_format(self.context))
        self.assertFalse(result)
        self.assertTrue(self.context.has_errors())
    
    def test_prepare_csv_data_from_dict_list(self):
        """Test preparing CSV data from list of dictionaries."""
        file_path = os.path.join(self.temp_dir.name, "test.csv")
        stage = CSVOutputStage(config={"file_path": file_path})
        
        rows, headers = stage._prepare_csv_data(self.dict_data)
        
        # Headers should match dictionary keys
        self.assertIn("id", headers)
        self.assertIn("name", headers)
        self.assertIn("age", headers)
        
        # Rows should be transformed from dictionaries
        self.assertEqual(len(rows), len(self.dict_data))
        for i, row in enumerate(rows):
            self.assertEqual(len(row), len(headers))
            record = self.dict_data[i]
            for j, header in enumerate(headers):
                self.assertEqual(row[j], record[header])
    
    def test_prepare_csv_data_from_list_list(self):
        """Test preparing CSV data from list of lists."""
        file_path = os.path.join(self.temp_dir.name, "test.csv")
        stage = CSVOutputStage(config={
            "file_path": file_path,
            "has_header": True
        })
        
        rows, headers = stage._prepare_csv_data(self.list_data)
        
        # First row should be treated as headers
        self.assertEqual(headers, self.list_data[0])
        
        # Remaining rows should be data
        self.assertEqual(len(rows), len(self.list_data) - 1)
        for i, row in enumerate(rows):
            self.assertEqual(row, self.list_data[i + 1])
    
    def test_prepare_csv_data_with_columns(self):
        """Test preparing CSV data with specified columns."""
        file_path = os.path.join(self.temp_dir.name, "test.csv")
        columns = ["id", "name"]  # Exclude age
        
        stage = CSVOutputStage(config={
            "file_path": file_path,
            "columns": columns
        })
        
        rows, headers = stage._prepare_csv_data(self.dict_data)
        
        # Headers should match specified columns
        self.assertEqual(headers, columns)
        
        # Rows should only include specified columns
        self.assertEqual(len(rows), len(self.dict_data))
        for i, row in enumerate(rows):
            self.assertEqual(len(row), len(columns))
            self.assertEqual(row[0], self.dict_data[i]["id"])
            self.assertEqual(row[1], self.dict_data[i]["name"])
    
    def test_deliver_output_dict_list(self):
        """Test delivering output with list of dictionaries."""
        file_path = os.path.join(self.temp_dir.name, "test.csv")
        stage = CSVOutputStage(config={"file_path": file_path})
        
        result = asyncio.run(stage.deliver_output(self.dict_data, self.context))
        
        # Verify the result
        self.assertEqual(result["file_path"], file_path)
        self.assertEqual(result["format"], "csv")
        self.assertTrue(os.path.exists(file_path))
        
        # Read back and verify
        rows = []
        with open(file_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        
        self.assertEqual(len(rows), len(self.dict_data))
        for i, row in enumerate(rows):
            self.assertEqual(int(row["id"]), self.dict_data[i]["id"])
            self.assertEqual(row["name"], self.dict_data[i]["name"])
            self.assertEqual(int(row["age"]), self.dict_data[i]["age"])
    
    def test_deliver_output_list_list(self):
        """Test delivering output with list of lists."""
        file_path = os.path.join(self.temp_dir.name, "test.csv")
        stage = CSVOutputStage(config={"file_path": file_path})
        
        result = asyncio.run(stage.deliver_output(self.list_data, self.context))
        
        # Verify the result
        self.assertEqual(result["file_path"], file_path)
        self.assertEqual(result["format"], "csv")
        self.assertTrue(os.path.exists(file_path))
        
        # Read back and verify
        rows = []
        with open(file_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append(row)
        
        self.assertEqual(len(rows), len(self.list_data))
        for i, row in enumerate(rows):
            expected_row = [str(item) for item in self.list_data[i]]
            self.assertEqual(row, expected_row)
    
    def test_deliver_output_dict_of_lists(self):
        """Test delivering output with dictionary of lists."""
        file_path = os.path.join(self.temp_dir.name, "test.csv")
        stage = CSVOutputStage(config={"file_path": file_path})
        
        result = asyncio.run(stage.deliver_output(self.columns_data, self.context))
        
        # Verify the result
        self.assertEqual(result["file_path"], file_path)
        self.assertEqual(result["format"], "csv")
        self.assertTrue(os.path.exists(file_path))
        
        # Read back and verify
        rows = []
        with open(file_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        
        self.assertEqual(len(rows), 3)  # 3 rows of data
        for i, row in enumerate(rows):
            self.assertEqual(int(row["id"]), self.columns_data["id"][i])
            self.assertEqual(row["name"], self.columns_data["name"][i])
            self.assertEqual(int(row["age"]), self.columns_data["age"][i])


class TestXMLOutputStage(unittest.TestCase):
    """Test cases for the XMLOutputStage."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Test data as dictionary
        self.dict_data = {
            "root": {
                "person": [
                    {"id": 1, "name": "Alice", "age": 30},
                    {"id": 2, "name": "Bob", "age": 25},
                    {"id": 3, "name": "Charlie", "age": 35}
                ]
            }
        }
        
        # Test data as list
        self.list_data = [
            {"id": 1, "name": "Alice", "age": 30},
            {"id": 2, "name": "Bob", "age": 25},
            {"id": 3, "name": "Charlie", "age": 35}
        ]
        
        self.context = PipelineContext()
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        file_path = os.path.join(self.temp_dir.name, "test")
        stage = XMLOutputStage(config={"file_path": file_path})
        
        self.assertEqual(stage.file_path, file_path + ".xml")
        self.assertEqual(stage.root_element, "root")
        self.assertEqual(stage.item_element, "item")
        self.assertTrue(stage.pretty_print)
        self.assertTrue(stage.xml_declaration)
        self.assertEqual(stage.namespaces, {})
        self.assertEqual(stage.indent, 2)
    
    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        file_path = os.path.join(self.temp_dir.name, "test.xml")
        config = {
            "file_path": file_path,
            "root_element": "data",
            "item_element": "record",
            "pretty_print": False,
            "xml_declaration": False,
            "namespaces": {"xs": "http://www.w3.org/2001/XMLSchema"},
            "indent": 4
        }
        
        stage = XMLOutputStage(config=config)
        
        self.assertEqual(stage.file_path, file_path)
        self.assertEqual(stage.root_element, "data")
        self.assertEqual(stage.item_element, "record")
        self.assertFalse(stage.pretty_print)
        self.assertFalse(stage.xml_declaration)
        self.assertEqual(stage.namespaces, {"xs": "http://www.w3.org/2001/XMLSchema"})
        self.assertEqual(stage.indent, 4)
    
    def test_validate_output_format_success(self):
        """Test successful output validation."""
        file_path = os.path.join(self.temp_dir.name, "test.xml")
        stage = XMLOutputStage(config={"file_path": file_path})
        
        self.context.set("data", self.dict_data)
        result = asyncio.run(stage.validate_output_format(self.context))
        self.assertTrue(result)
    
    def test_validate_output_format_invalid_element_names(self):
        """Test validation with invalid element names."""
        file_path = os.path.join(self.temp_dir.name, "test.xml")
        
        # Invalid root element name
        stage = XMLOutputStage(config={
            "file_path": file_path,
            "root_element": "123invalid"  # Starts with number
        })
        
        self.context.set("data", self.dict_data)
        result = asyncio.run(stage.validate_output_format(self.context))
        self.assertFalse(result)
        self.assertTrue(self.context.has_errors())
        
        # Reset context
        self.context = PipelineContext()
        self.context.set("data", self.dict_data)
        
        # Invalid item element name
        stage = XMLOutputStage(config={
            "file_path": file_path,
            "item_element": "@invalid"  # Contains invalid character
        })
        
        result = asyncio.run(stage.validate_output_format(self.context))
        self.assertFalse(result)
        self.assertTrue(self.context.has_errors())
    
    def test_add_elements_dict(self):
        """Test adding elements from a dictionary."""
        file_path = os.path.join(self.temp_dir.name, "test.xml")
        stage = XMLOutputStage(config={"file_path": file_path})
        
        # Create a parent element
        parent = ET.Element("parent")
        
        # Add dictionary elements
        data = {"name": "Alice", "age": 30, "address": {"city": "New York", "zip": "10001"}}
        count = stage._add_elements(parent, data, "person")
        
        # Verify elements were added
        self.assertEqual(count, 1)  # One top-level element
        person = parent.find("person")
        self.assertIsNotNone(person)
        
        # Check attributes
        self.assertEqual(person.attrib["name"], "Alice")
        self.assertEqual(person.attrib["age"], "30")
        
        # Check nested elements
        address = person.find("address")
        self.assertIsNotNone(address)
        self.assertEqual(address.attrib["city"], "New York")
        self.assertEqual(address.attrib["zip"], "10001")
    
    def test_add_elements_list(self):
        """Test adding elements from a list."""
        file_path = os.path.join(self.temp_dir.name, "test.xml")
        stage = XMLOutputStage(config={"file_path": file_path})
        
        # Create a parent element
        parent = ET.Element("parent")
        
        # Add list elements
        data = ["Alice", "Bob", "Charlie"]
        count = stage._add_elements(parent, data, "names")
        
        # Verify elements were added
        self.assertEqual(count, 4)  # One wrapper + three items
        names = parent.find("names")
        self.assertIsNotNone(names)
        
        # Check items
        items = names.findall(stage.item_element)
        self.assertEqual(len(items), 3)
        self.assertEqual(items[0].text, "Alice")
        self.assertEqual(items[1].text, "Bob")
        self.assertEqual(items[2].text, "Charlie")
    
    def test_deliver_output_dict(self):
        """Test delivering output with dictionary data."""
        file_path = os.path.join(self.temp_dir.name, "test.xml")
        stage = XMLOutputStage(config={"file_path": file_path})
        
        result = asyncio.run(stage.deliver_output(self.dict_data, self.context))
        
        # Verify the result
        self.assertEqual(result["file_path"], file_path)
        self.assertEqual(result["format"], "xml")
        self.assertTrue(os.path.exists(file_path))
        
        # Parse and verify
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        self.assertEqual(root.tag, "root")
        persons = root.findall(".//person")
        self.assertEqual(len(persons), 3)
    
    def test_deliver_output_list(self):
        """Test delivering output with list data."""
        file_path = os.path.join(self.temp_dir.name, "test.xml")
        stage = XMLOutputStage(config={"file_path": file_path})
        
        result = asyncio.run(stage.deliver_output(self.list_data, self.context))
        
        # Verify the result
        self.assertEqual(result["file_path"], file_path)
        self.assertEqual(result["format"], "xml")
        self.assertTrue(os.path.exists(file_path))
        
        # Parse and verify
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        self.assertEqual(root.tag, "root")
        items = root.findall(f".//{stage.item_element}")
        self.assertEqual(len(items), 3)
        
        # Verify attributes
        for i, item in enumerate(items):
            self.assertEqual(item.attrib["id"], str(self.list_data[i]["id"]))
            self.assertEqual(item.attrib["name"], self.list_data[i]["name"])
            self.assertEqual(item.attrib["age"], str(self.list_data[i]["age"]))
    
    def test_deliver_output_with_namespaces(self):
        """Test delivering output with XML namespaces."""
        file_path = os.path.join(self.temp_dir.name, "test.xml")
        stage = XMLOutputStage(config={
            "file_path": file_path,
            "namespaces": {
                "": "http://example.org/default",
                "xs": "http://www.w3.org/2001/XMLSchema"
            }
        })
        
        result = asyncio.run(stage.deliver_output(self.list_data, self.context))
        
        # Verify the file exists
        self.assertTrue(os.path.exists(file_path))
        
        # Check content for namespace declarations
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn('xmlns="http://example.org/default"', content)
            self.assertIn('xmlns:xs="http://www.w3.org/2001/XMLSchema"', content)


class TestTextOutputStage(unittest.TestCase):
    """Test cases for the TextOutputStage."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Simple string data
        self.string_data = "This is a test string"
        
        # List data
        self.list_data = ["Line 1", "Line 2", "Line 3"]
        
        # Dictionary data
        self.dict_data = {
            "name": "Alice",
            "age": 30,
            "city": "New York"
        }
        
        self.context = PipelineContext()
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        file_path = os.path.join(self.temp_dir.name, "test")
        stage = TextOutputStage(config={"file_path": file_path})
        
        self.assertEqual(stage.file_path, file_path + ".txt")
        self.assertIsNone(stage.template)
        self.assertEqual(stage.line_ending, "LF")
        self.assertEqual(stage.line_ending_char, "\n")
        self.assertEqual(stage.join_delimiter, "\n")
    
    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        file_path = os.path.join(self.temp_dir.name, "test.txt")
        config = {
            "file_path": file_path,
            "template": "Name: {name}, Age: {age}",
            "line_ending": "CRLF",
            "join_delimiter": ", "
        }
        
        stage = TextOutputStage(config=config)
        
        self.assertEqual(stage.file_path, file_path)
        self.assertEqual(stage.template, "Name: {name}, Age: {age}")
        self.assertEqual(stage.line_ending, "CRLF")
        self.assertEqual(stage.line_ending_char, "\r\n")
        self.assertEqual(stage.join_delimiter, ", ")
    
    def test_validate_output_format_success(self):
        """Test successful output validation."""
        file_path = os.path.join(self.temp_dir.name, "test.txt")
        stage = TextOutputStage(config={"file_path": file_path})
        
        result = asyncio.run(stage.validate_output_format(self.context))
        self.assertTrue(result)
    
    def test_validate_output_format_invalid_template(self):
        """Test validation with invalid template."""
        file_path = os.path.join(self.temp_dir.name, "test.txt")
        stage = TextOutputStage(config={
            "file_path": file_path,
            "template": "Name: {name}, ID: {id"  # Missing closing brace
        })
        
        result = asyncio.run(stage.validate_output_format(self.context))
        self.assertFalse(result)
        self.assertTrue(self.context.has_errors())
    
    def test_format_text_content_string(self):
        """Test formatting string content."""
        file_path = os.path.join(self.temp_dir.name, "test.txt")
        stage = TextOutputStage(config={"file_path": file_path})
        
        result = stage._format_text_content(self.string_data, self.context)
        self.assertEqual(result, self.string_data)
    
    def test_format_text_content_list(self):
        """Test formatting list content."""
        file_path = os.path.join(self.temp_dir.name, "test.txt")
        stage = TextOutputStage(config={"file_path": file_path})
        
        # Default delimiter (newline)
        result = stage._format_text_content(self.list_data, self.context)
        self.assertEqual(result, "Line 1\nLine 2\nLine 3")
        
        # Custom delimiter
        stage = TextOutputStage(config={
            "file_path": file_path,
            "join_delimiter": " | "
        })
        
        result = stage._format_text_content(self.list_data, self.context)
        self.assertEqual(result, "Line 1 | Line 2 | Line 3")
    
    def test_format_text_content_dict(self):
        """Test formatting dictionary content."""
        file_path = os.path.join(self.temp_dir.name, "test.txt")
        
        # Without template
        stage = TextOutputStage(config={"file_path": file_path})
        
        result = stage._format_text_content(self.dict_data, self.context)
        self.assertIn("name: Alice", result)
        self.assertIn("age: 30", result)
        self.assertIn("city: New York", result)
        
        # With template
        stage = TextOutputStage(config={
            "file_path": file_path,
            "template": "Name: {name}, Age: {age}, City: {city}"
        })
        
        result = stage._format_text_content(self.dict_data, self.context)
        self.assertEqual(result, "Name: Alice, Age: 30, City: New York")
        
        # With incomplete template (missing key)
        stage = TextOutputStage(config={
            "file_path": file_path,
            "template": "Country: {country}"  # 'country' key doesn't exist
        })
        
        # Should fall back to string representation
        result = stage._format_text_content(self.dict_data, self.context)
        self.assertIn("name", result)
        self.assertIn("age", result)
        self.assertIn("city", result)
    
    def test_deliver_output_string(self):
        """Test delivering string output."""
        file_path = os.path.join(self.temp_dir.name, "test.txt")
        stage = TextOutputStage(config={"file_path": file_path})
        
        result = asyncio.run(stage.deliver_output(self.string_data, self.context))
        
        # Verify the result
        self.assertEqual(result["file_path"], file_path)
        self.assertEqual(result["format"], "text")
        self.assertTrue(os.path.exists(file_path))
        
        # Read back and verify
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertEqual(content, self.string_data)
    
    def test_deliver_output_list(self):
        """Test delivering list output."""
        file_path = os.path.join(self.temp_dir.name, "test.txt")
        stage = TextOutputStage(config={"file_path": file_path})
        
        result = asyncio.run(stage.deliver_output(self.list_data, self.context))
        
        # Verify the file exists
        self.assertTrue(os.path.exists(file_path))
        
        # Read back and verify
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            expected = "\n".join(self.list_data)
            self.assertEqual(content, expected)
    
    def test_deliver_output_with_crlf(self):
        """Test delivering output with CRLF line endings."""
        file_path = os.path.join(self.temp_dir.name, "test.txt")
        stage = TextOutputStage(config={
            "file_path": file_path,
            "line_ending": "CRLF"
        })
        
        result = asyncio.run(stage.deliver_output(self.list_data, self.context))
        
        # Verify the file exists
        self.assertTrue(os.path.exists(file_path))
        
        # Read back in binary mode to check for CRLF
        with open(file_path, "rb") as f:
            content = f.read()
            self.assertIn(b"\r\n", content)
            self.assertEqual(content.count(b"\r\n"), 2)  # 2 line breaks for 3 lines


if YAML_AVAILABLE:
    class TestYAMLOutputStage(unittest.TestCase):
        """Test cases for the YAMLOutputStage."""
        
        def setUp(self):
            """Set up test fixtures."""
            self.temp_dir = tempfile.TemporaryDirectory()
            
            # Test data
            self.test_data = {
                "config": {
                    "server": {
                        "host": "localhost",
                        "port": 8080
                    },
                    "database": {
                        "host": "db.example.com",
                        "port": 5432,
                        "credentials": {
                            "username": "user",
                            "password": "pass"
                        }
                    },
                    "logging": {
                        "level": "INFO",
                        "file": "/var/log/app.log"
                    }
                },
                "users": [
                    {"name": "Alice", "role": "admin"},
                    {"name": "Bob", "role": "user"}
                ]
            }
            
            self.context = PipelineContext()
            
        def tearDown(self):
            """Clean up after tests."""
            self.temp_dir.cleanup()
            
        def test_init_with_default_config(self):
            """Test initialization with default configuration."""
            from core.pipeline.stages.output.file_output import YAMLOutputStage
            
            file_path = os.path.join(self.temp_dir.name, "test")
            stage = YAMLOutputStage(config={"file_path": file_path})
            
            self.assertEqual(stage.file_path, file_path + ".yaml")
            self.assertFalse(stage.sort_keys)
            self.assertFalse(stage.flow_style)
            self.assertIsNone(stage.default_style)
            self.assertEqual(stage.indent, 2)
            
        def test_init_with_custom_config(self):
            """Test initialization with custom configuration."""
            from core.pipeline.stages.output.file_output import YAMLOutputStage
            
            file_path = os.path.join(self.temp_dir.name, "test.yml")
            config = {
                "file_path": file_path,
                "sort_keys": True,
                "flow_style": True,
                "default_style": '"',
                "indent": 4
            }
            
            stage = YAMLOutputStage(config=config)
            
            self.assertEqual(stage.file_path, file_path)
            self.assertTrue(stage.sort_keys)
            self.assertTrue(stage.flow_style)
            self.assertEqual(stage.default_style, '"')
            self.assertEqual(stage.indent, 4)
            
        def test_validate_output_format(self):
            """Test output validation."""
            from core.pipeline.stages.output.file_output import YAMLOutputStage
            
            file_path = os.path.join(self.temp_dir.name, "test.yaml")
            stage = YAMLOutputStage(config={"file_path": file_path})
            
            self.context.set("data", self.test_data)
            result = asyncio.run(stage.validate_output_format(self.context))
            self.assertTrue(result)
            
        def test_deliver_output(self):
            """Test delivering YAML output."""
            from core.pipeline.stages.output.file_output import YAMLOutputStage
            
            file_path = os.path.join(self.temp_dir.name, "test.yaml")
            stage = YAMLOutputStage(config={"file_path": file_path})
            
            result = asyncio.run(stage.deliver_output(self.test_data, self.context))
            
            # Verify the result
            self.assertEqual(result["file_path"], file_path)
            self.assertEqual(result["format"], "yaml")
            self.assertTrue(os.path.exists(file_path))
            
            # Read back and verify
            with open(file_path, "r", encoding="utf-8") as f:
                loaded_data = yaml.safe_load(f)
                self.assertEqual(loaded_data, self.test_data)
                
        def test_deliver_output_with_options(self):
            """Test delivering YAML output with custom options."""
            from core.pipeline.stages.output.file_output import YAMLOutputStage
            
            file_path = os.path.join(self.temp_dir.name, "test.yaml")
            stage = YAMLOutputStage(config={
                "file_path": file_path,
                "sort_keys": True,
                "flow_style": True
            })
            
            result = asyncio.run(stage.deliver_output(self.test_data, self.context))
            
            # Verify the file exists
            self.assertTrue(os.path.exists(file_path))
            
            # Read back and verify
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                # Flow style will have more curly braces and fewer newlines
                self.assertIn("{", content)
                # Sort keys will maintain a consistent order
                loaded_data = yaml.safe_load(f)
                self.assertEqual(loaded_data, self.test_data)


if __name__ == "__main__":
    unittest.main()