"""
File Output Stage Module.

This module provides specialized output stages for saving data to various file formats.
"""

import asyncio
import csv
import json
import logging
import os
import stat
import tempfile
import xml.dom.minidom
import xml.etree.ElementTree as ET
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, TextIO, BinaryIO

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from core.pipeline.stages.base_stages import OutputStage
from core.pipeline.context import PipelineContext


class FileNamingStrategy(Enum):
    """File naming strategies."""
    FIXED = auto()  # Use fixed filename from configuration
    TIMESTAMP = auto()  # Add timestamp to filename
    SEQUENTIAL = auto()  # Add sequential number to filename
    CONTENT_HASH = auto()  # Add content hash to filename


class FileOutputStage(OutputStage):
    """
    Base output stage for writing data to the file system.
    
    Features:
    - File naming strategies (fixed, timestamped, sequential)
    - Directory creation if needed
    - File permission management
    - Atomic write operations
    - Backup of existing files
    - Comprehensive error handling
    
    Configuration:
    - file_path: Path to output file or directory
    - naming_strategy: Strategy for file naming (FIXED, TIMESTAMP, SEQUENTIAL, CONTENT_HASH)
    - file_permissions: Unix-style file permissions (e.g., 0o644)
    - overwrite_existing: Whether to overwrite existing files
    - create_directories: Whether to create parent directories
    - atomic_write: Whether to use atomic write operations
    - encoding: Character encoding for text files
    """
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize a new file output stage.
        
        Args:
            name: Name of the stage, defaults to class name if not provided
            config: Configuration parameters for this stage
        """
        super().__init__(name, config)
        
        # File path configuration
        self.file_path = self.config.get("file_path")
        if not self.file_path:
            raise ValueError("file_path must be specified in configuration")
            
        # File naming strategy
        naming_strategy = self.config.get("naming_strategy", "FIXED")
        try:
            self.naming_strategy = FileNamingStrategy[naming_strategy] if isinstance(naming_strategy, str) else naming_strategy
        except (KeyError, TypeError):
            self.naming_strategy = FileNamingStrategy.FIXED
            self.logger.warning(f"Invalid file naming strategy: {naming_strategy}, using FIXED")
        
        # File operation configuration
        self.file_permissions = self.config.get("file_permissions")
        self.overwrite_existing = self.config.get("overwrite_existing", True)
        self.create_directories = self.config.get("create_directories", True)
        self.atomic_write = self.config.get("atomic_write", True)
        
        # Text file configuration
        self.encoding = self.config.get("encoding", "utf-8")
        
        # Set up logger
        self.logger = logging.getLogger(f"pipeline.stages.output.file.{self.name}")
        
        # Statistics for monitoring
        self._stats = {
            "bytes_written": 0,
            "files_created": 0,
            "directories_created": 0
        }
        
        # Keep track of sequential numbering if needed
        self._sequence_number = 0
    
    async def validate_output_format(self, context: PipelineContext) -> bool:
        """
        Validate output configuration before writing.
        
        Args:
            context: The shared pipeline context
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        # Check if file path is specified
        if not self.file_path:
            self.logger.error("No file path specified")
            context.add_error(self.name, "No file path specified")
            return False
            
        # Check if parent directory exists or can be created
        parent_dir = os.path.dirname(os.path.abspath(self.file_path))
        if not os.path.exists(parent_dir):
            if not self.create_directories:
                self.logger.error(f"Parent directory does not exist: {parent_dir}")
                context.add_error(self.name, f"Parent directory does not exist: {parent_dir}")
                return False
        
        # Check if file exists and can be overwritten if needed
        if os.path.exists(self.file_path) and not self.overwrite_existing:
            self.logger.error(f"File already exists and overwrite_existing is False: {self.file_path}")
            context.add_error(self.name, f"File already exists: {self.file_path}")
            return False
            
        return True
    
    async def deliver_output(self, data: Any, context: PipelineContext) -> Optional[Dict[str, Any]]:
        """
        Write the data to a file.
        
        Subclasses should override this method to implement format-specific writing.
        
        Args:
            data: The data to write
            context: The shared pipeline context
            
        Returns:
            Dict containing output file metadata or None if output failed
        """
        # Default implementation uses text format
        try:
            # Get actual file path based on naming strategy
            output_path = await self._get_output_path(data)
            
            # Ensure parent directory exists
            if self.create_directories:
                directory = os.path.dirname(os.path.abspath(output_path))
                if not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                    self._stats["directories_created"] += 1
            
            # Write the file
            temp_path = None
            if self.atomic_write:
                # Use atomic write with temporary file
                directory = os.path.dirname(os.path.abspath(output_path))
                fd, temp_path = tempfile.mkstemp(dir=directory, prefix='.tmp')
                try:
                    with os.fdopen(fd, 'w', encoding=self.encoding) as f:
                        if isinstance(data, str):
                            f.write(data)
                        else:
                            f.write(str(data))
                    
                    # Atomically rename the temporary file to the target file
                    os.replace(temp_path, output_path)
                    temp_path = None  # Prevent deletion in finally block
                finally:
                    # Clean up temporary file if still exists
                    if temp_path and os.path.exists(temp_path):
                        try:
                            os.unlink(temp_path)
                        except:
                            pass
            else:
                # Direct write
                with open(output_path, 'w', encoding=self.encoding) as f:
                    if isinstance(data, str):
                        f.write(data)
                    else:
                        f.write(str(data))
            
            # Set file permissions if specified
            if self.file_permissions is not None:
                os.chmod(output_path, self.file_permissions)
            
            # Update statistics
            file_size = os.path.getsize(output_path)
            self._stats["bytes_written"] += file_size
            self._stats["files_created"] += 1
            
            return {
                "file_path": output_path,
                "size": file_size,
                "format": "text"
            }
            
        except Exception as e:
            self.logger.error(f"Error writing to file: {str(e)}")
            context.add_error(self.name, f"File output error: {str(e)}")
            return None
    
    async def _get_output_path(self, data: Any = None) -> str:
        """
        Get the actual output file path based on the naming strategy.
        
        Args:
            data: The data to be written (used for content-hash strategy)
            
        Returns:
            str: The resolved file path
        """
        if self.naming_strategy == FileNamingStrategy.FIXED:
            return self.file_path
            
        # Parse the base path
        base_path = Path(self.file_path)
        stem = base_path.stem
        suffix = base_path.suffix
        parent = base_path.parent
        
        if self.naming_strategy == FileNamingStrategy.TIMESTAMP:
            # Add timestamp to filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{stem}_{timestamp}{suffix}"
            return str(parent / filename)
            
        elif self.naming_strategy == FileNamingStrategy.SEQUENTIAL:
            # Add sequential number to filename
            self._sequence_number += 1
            filename = f"{stem}_{self._sequence_number:04d}{suffix}"
            return str(parent / filename)
            
        elif self.naming_strategy == FileNamingStrategy.CONTENT_HASH:
            # Add content hash to filename
            import hashlib
            
            # Generate a hash based on the data
            if isinstance(data, str):
                content_hash = hashlib.md5(data.encode('utf-8')).hexdigest()[:8]
            elif isinstance(data, bytes):
                content_hash = hashlib.md5(data).hexdigest()[:8]
            else:
                # For other types, convert to JSON first
                try:
                    content_hash = hashlib.md5(json.dumps(data).encode('utf-8')).hexdigest()[:8]
                except:
                    # Fallback to timestamp if we can't hash the content
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    content_hash = f"nohash_{timestamp}"
            
            filename = f"{stem}_{content_hash}{suffix}"
            return str(parent / filename)
            
        # Default to fixed path if strategy is not recognized
        return self.file_path
    
    async def _create_backup(self, context: PipelineContext) -> bool:
        """
        Create a backup of the existing file before writing.
        
        Args:
            context: The shared pipeline context
            
        Returns:
            bool: True if backup was created or not needed, False on error
        """
        if not self.backup_enabled or not os.path.exists(self.file_path):
            return True
            
        try:
            backup_path = self.backup_path_template.format(path=self.file_path)
            
            # Create the backup file
            with open(self.file_path, 'rb') as src:
                with open(backup_path, 'wb') as dst:
                    dst.write(src.read())
            
            # Add backup info to context
            context.set("backup_created", {
                "original_path": self.file_path,
                "backup_path": backup_path,
                "timestamp": datetime.now().isoformat()
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating backup: {str(e)}")
            context.add_error(self.name, f"Backup error: {str(e)}")
            return False
    
    async def _rollback_from_backup(self, context: PipelineContext) -> bool:
        """
        Restore from backup after a failed write operation.
        
        Args:
            context: The shared pipeline context
            
        Returns:
            bool: True if rollback succeeded, False otherwise
        """
        if not self.backup_enabled:
            return False
            
        backup_info = context.get("backup_created")
        if not backup_info:
            return False
            
        backup_path = backup_info["backup_path"]
        if not os.path.exists(backup_path):
            self.logger.error(f"Backup file not found: {backup_path}")
            return False
            
        try:
            # Restore from backup
            with open(backup_path, 'rb') as src:
                with open(self.file_path, 'wb') as dst:
                    dst.write(src.read())
                    
            self.logger.info(f"Successfully rolled back to backup: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during rollback: {str(e)}")
            context.add_error(self.name, f"Rollback error: {str(e)}")
            return False
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for configuration.
        
        Returns:
            Dict containing JSON schema for this stage's configuration
        """
        schema = super().get_config_schema()
        
        # Add file-specific properties
        file_properties = {
            "file_path": {"type": "string"},
            "naming_strategy": {
                "type": "string",
                "enum": ["FIXED", "TIMESTAMP", "SEQUENTIAL", "CONTENT_HASH"]
            },
            "file_permissions": {"type": "integer"},
            "overwrite_existing": {"type": "boolean"},
            "create_directories": {"type": "boolean"},
            "atomic_write": {"type": "boolean"},
            "encoding": {"type": "string"}
        }
        
        # Update the properties in the schema
        schema["properties"].update(file_properties)
        
        return schema


class CSVOutputStage(FileOutputStage):
    """
    Output stage for writing data to CSV files.
    
    Features:
    - Configurable CSV formatting (delimiters, quoting)
    - Support for headers
    - Row filtering
    - Column selection
    
    Configuration:
    - delimiter: Field delimiter (default: ,)
    - quoting: Quoting style (default: QUOTE_MINIMAL)
    - has_header: Whether to include a header row
    - columns: List of columns to include
    - dialect: CSV dialect to use
    """
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize a new CSV output stage.
        
        Args:
            name: Name of the stage, defaults to class name if not provided
            config: Configuration parameters for this stage
        """
        super().__init__(name, config)
        
        # CSV-specific configuration
        self.delimiter = self.config.get("delimiter", ",")
        
        quoting_str = self.config.get("quoting", "QUOTE_MINIMAL")
        self.quoting = {
            "QUOTE_MINIMAL": csv.QUOTE_MINIMAL,
            "QUOTE_ALL": csv.QUOTE_ALL,
            "QUOTE_NONNUMERIC": csv.QUOTE_NONNUMERIC,
            "QUOTE_NONE": csv.QUOTE_NONE
        }.get(quoting_str, csv.QUOTE_MINIMAL)
        
        self.has_header = self.config.get("has_header", True)
        self.columns = self.config.get("columns", None)
        self.dialect = self.config.get("dialect", "excel")
        
        # Ensure file has .csv extension
        if not self.file_path.lower().endswith('.csv'):
            self.file_path += '.csv'
    
    async def validate_output_format(self, context: PipelineContext) -> bool:
        """
        Validate that the data can be written as CSV.
        
        Args:
            context: The shared pipeline context
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        if not await super().validate_output_format(context):
            return False
            
        # Get the data to validate
        data = self._prepare_output_data(context)
        if data is None:
            self.logger.error("No data to write")
            context.add_error(self.name, "No data to write")
            return False
            
        # Check that data is a list or can be converted to one
        if not isinstance(data, (list, tuple)) and not isinstance(data, dict):
            self.logger.error("Data must be a list, tuple, or dict for CSV output")
            context.add_error(self.name, "Invalid data type for CSV output")
            return False
            
        # If it's a dict and not a list of dicts, convert it
        if isinstance(data, dict):
            if not all(isinstance(v, (list, tuple)) for v in data.values()):
                self.logger.error("Dict values must be lists or tuples for CSV output")
                context.add_error(self.name, "Invalid dict values for CSV output")
                return False
                
            # Check that all lists have the same length
            lengths = [len(v) for v in data.values()]
            if len(set(lengths)) > 1:
                self.logger.error("All dict values must have the same length for CSV output")
                context.add_error(self.name, "Inconsistent dict value lengths for CSV output")
                return False
        
        # If it's a list, check that entries are dicts or lists
        elif isinstance(data, (list, tuple)):
            if len(data) > 0:
                first_item = data[0]
                if isinstance(first_item, dict):
                    # List of dicts - check that all items are dicts
                    if not all(isinstance(item, dict) for item in data):
                        self.logger.error("All items must be dicts for CSV output")
                        context.add_error(self.name, "Mixed item types in list for CSV output")
                        return False
                elif isinstance(first_item, (list, tuple)):
                    # List of lists - check that all items are lists of the same length
                    if not all(isinstance(item, (list, tuple)) for item in data):
                        self.logger.error("All items must be lists or tuples for CSV output")
                        context.add_error(self.name, "Mixed item types in list for CSV output")
                        return False
                        
                    # Check that all lists have the same length
                    lengths = [len(item) for item in data]
                    if len(set(lengths)) > 1:
                        self.logger.error("All lists must have the same length for CSV output")
                        context.add_error(self.name, "Inconsistent list lengths for CSV output")
                        return False
                else:
                    # If items are scalars, they will be treated as a single column
                    pass
        
        return True
    
    async def deliver_output(self, data: Any, context: PipelineContext) -> Optional[Dict[str, Any]]:
        """
        Write the data to a CSV file.
        
        Args:
            data: The data to write
            context: The shared pipeline context
            
        Returns:
            Dict containing output file metadata or None if output failed
        """
        try:
            # Get actual file path based on naming strategy
            output_path = await self._get_output_path(data)
            
            # Ensure parent directory exists
            if self.create_directories:
                directory = os.path.dirname(os.path.abspath(output_path))
                if not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                    self._stats["directories_created"] += 1
            
            # Prepare rows and headers
            rows, headers = self._prepare_csv_data(data)
            row_count = len(rows)
            
            # Write the file
            temp_path = None
            if self.atomic_write:
                # Use atomic write with temporary file
                directory = os.path.dirname(os.path.abspath(output_path))
                fd, temp_path = tempfile.mkstemp(dir=directory, prefix='.tmp', suffix='.csv')
                try:
                    with os.fdopen(fd, 'w', encoding=self.encoding, newline='') as f:
                        writer = csv.writer(
                            f, 
                            delimiter=self.delimiter,
                            quoting=self.quoting,
                            dialect=self.dialect
                        )
                        
                        # Write header if enabled
                        if self.has_header and headers:
                            writer.writerow(headers)
                            
                        # Write data rows
                        writer.writerows(rows)
                    
                    # Atomically rename the temporary file to the target file
                    os.replace(temp_path, output_path)
                    temp_path = None  # Prevent deletion in finally block
                finally:
                    # Clean up temporary file if still exists
                    if temp_path and os.path.exists(temp_path):
                        try:
                            os.unlink(temp_path)
                        except:
                            pass
            else:
                # Direct write
                with open(output_path, 'w', encoding=self.encoding, newline='') as f:
                    writer = csv.writer(
                        f, 
                        delimiter=self.delimiter,
                        quoting=self.quoting,
                        dialect=self.dialect
                    )
                    
                    # Write header if enabled
                    if self.has_header and headers:
                        writer.writerow(headers)
                        
                    # Write data rows
                    writer.writerows(rows)
            
            # Set file permissions if specified
            if self.file_permissions is not None:
                os.chmod(output_path, self.file_permissions)
            
            # Update statistics
            file_size = os.path.getsize(output_path)
            self._stats["bytes_written"] += file_size
            self._stats["files_created"] += 1
            
            return {
                "file_path": output_path,
                "size": file_size,
                "format": "csv",
                "rows": row_count,
                "header": self.has_header
            }
            
        except Exception as e:
            self.logger.error(f"Error writing CSV file: {str(e)}")
            context.add_error(self.name, f"CSV output error: {str(e)}")
            return None
    
    def _prepare_csv_data(self, data: Any) -> Tuple[List[List[Any]], List[str]]:
        """
        Transform input data into CSV rows and headers.
        
        Args:
            data: The input data (list of dicts, list of lists, or dict of lists)
            
        Returns:
            Tuple of (rows, headers) where rows is a list of lists and headers is a list of strings
        """
        rows = []
        headers = []
        
        if isinstance(data, dict):
            # Dict of lists/tuples
            headers = list(data.keys())
            if self.columns:
                # Filter to include only specified columns
                headers = [h for h in headers if h in self.columns]
                
            # Transpose the data to create rows
            values = [data[h] for h in headers]
            row_count = len(values[0]) if values else 0
            for i in range(row_count):
                rows.append([values[j][i] for j in range(len(headers))])
                
        elif isinstance(data, (list, tuple)):
            if not data:
                return [], []
                
            first_item = data[0]
            
            if isinstance(first_item, dict):
                # List of dicts
                if self.columns:
                    headers = self.columns
                else:
                    # Get all unique keys from all dicts
                    headers = list(set().union(*(d.keys() for d in data)))
                    
                # Create a row for each dict
                for item in data:
                    rows.append([item.get(h, "") for h in headers])
                    
            elif isinstance(first_item, (list, tuple)):
                # List of lists/tuples
                if self.columns and len(self.columns) == len(first_item):
                    headers = self.columns
                elif self.has_header:
                    # First row is treated as headers
                    headers = list(first_item)
                    data = data[1:]
                    
                # Each list is already a row
                rows = list(data)
                
            else:
                # List of scalar values - single column
                if self.columns and len(self.columns) == 1:
                    headers = self.columns
                else:
                    headers = ["value"]
                    
                # Each value becomes a single-item row
                rows = [[item] for item in data]
                
        return rows, headers
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for configuration.
        
        Returns:
            Dict containing JSON schema for this stage's configuration
        """
        schema = super().get_config_schema()
        
        # Add CSV-specific properties
        csv_properties = {
            "delimiter": {"type": "string"},
            "quoting": {
                "type": "string",
                "enum": ["QUOTE_MINIMAL", "QUOTE_ALL", "QUOTE_NONNUMERIC", "QUOTE_NONE"]
            },
            "has_header": {"type": "boolean"},
            "columns": {
                "type": "array",
                "items": {"type": "string"}
            },
            "dialect": {"type": "string"}
        }
        
        # Update the properties in the schema
        schema["properties"].update(csv_properties)
        
        return schema


class XMLOutputStage(FileOutputStage):
    """
    Output stage for writing data to XML files.
    
    Features:
    - Configurable XML formatting (indentation, encoding)
    - Support for namespaces
    - XML schema validation
    - Pretty printing option
    
    Configuration:
    - root_element: Name of the root XML element
    - item_element: Name of individual item elements
    - pretty_print: Whether to format XML with indentation
    - xml_declaration: Whether to include XML declaration
    - namespaces: Dict of namespace prefixes and URIs
    - indent: Indentation for pretty printing
    """
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize a new XML output stage.
        
        Args:
            name: Name of the stage, defaults to class name if not provided
            config: Configuration parameters for this stage
        """
        super().__init__(name, config)
        
        # XML-specific configuration
        self.root_element = self.config.get("root_element", "root")
        self.item_element = self.config.get("item_element", "item")
        self.pretty_print = self.config.get("pretty_print", True)
        self.xml_declaration = self.config.get("xml_declaration", True)
        self.namespaces = self.config.get("namespaces", {})
        self.indent = self.config.get("indent", 2)
        
        # Ensure file has .xml extension
        if not self.file_path.lower().endswith('.xml'):
            self.file_path += '.xml'
    
    async def validate_output_format(self, context: PipelineContext) -> bool:
        """
        Validate that the data can be written as XML.
        
        Args:
            context: The shared pipeline context
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        if not await super().validate_output_format(context):
            return False
            
        # Check that the data can be converted to XML
        # This is minimal validation since almost any data can be represented as XML
        data = self._prepare_output_data(context)
        if data is None:
            self.logger.error("No data to write")
            context.add_error(self.name, "No data to write")
            return False
            
        # Verify that root and item element names are valid XML names
        import re
        name_pattern = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_.-]*$')
        
        if not name_pattern.match(self.root_element):
            self.logger.error(f"Invalid root element name: {self.root_element}")
            context.add_error(self.name, f"Invalid root element name: {self.root_element}")
            return False
            
        if not name_pattern.match(self.item_element):
            self.logger.error(f"Invalid item element name: {self.item_element}")
            context.add_error(self.name, f"Invalid item element name: {self.item_element}")
            return False
            
        return True
    
    async def deliver_output(self, data: Any, context: PipelineContext) -> Optional[Dict[str, Any]]:
        """
        Write the data to an XML file.
        
        Args:
            data: The data to write
            context: The shared pipeline context
            
        Returns:
            Dict containing output file metadata or None if output failed
        """
        try:
            # Get actual file path based on naming strategy
            output_path = await self._get_output_path(data)
            
            # Ensure parent directory exists
            if self.create_directories:
                directory = os.path.dirname(os.path.abspath(output_path))
                if not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                    self._stats["directories_created"] += 1
            
            # Create XML document
            root = ET.Element(self.root_element)
            
            # Add namespaces
            for prefix, uri in self.namespaces.items():
                if prefix:
                    ET.register_namespace(prefix, uri)
                    root.attrib[f"xmlns:{prefix}"] = uri
                else:
                    root.attrib["xmlns"] = uri
            
            # Convert data to XML elements
            element_count = self._add_elements(root, data)
            
            # Create XML tree
            tree = ET.ElementTree(root)
            
            # Use atomic write with temporary file
            temp_path = None
            if self.atomic_write:
                directory = os.path.dirname(os.path.abspath(output_path))
                fd, temp_path = tempfile.mkstemp(dir=directory, prefix='.tmp', suffix='.xml')
                try:
                    os.close(fd)  # Close the file descriptor since we'll use ElementTree to write
                    
                    if self.pretty_print:
                        # Pretty printing with xml.dom.minidom
                        xml_str = ET.tostring(root, encoding=self.encoding)
                        dom = xml.dom.minidom.parseString(xml_str)
                        pretty_xml = dom.toprettyxml(indent=' ' * self.indent, encoding=self.encoding)
                        
                        with open(temp_path, 'wb') as f:
                            f.write(pretty_xml)
                    else:
                        # Direct write with ElementTree
                        tree.write(temp_path, encoding=self.encoding, xml_declaration=self.xml_declaration)
                    
                    # Atomically rename the temporary file to the target file
                    os.replace(temp_path, output_path)
                    temp_path = None  # Prevent deletion in finally block
                finally:
                    # Clean up temporary file if still exists
                    if temp_path and os.path.exists(temp_path):
                        try:
                            os.unlink(temp_path)
                        except:
                            pass
            else:
                # Direct write
                if self.pretty_print:
                    # Pretty printing with xml.dom.minidom
                    xml_str = ET.tostring(root, encoding=self.encoding)
                    dom = xml.dom.minidom.parseString(xml_str)
                    pretty_xml = dom.toprettyxml(indent=' ' * self.indent, encoding=self.encoding)
                    
                    with open(output_path, 'wb') as f:
                        f.write(pretty_xml)
                else:
                    # Direct write with ElementTree
                    tree.write(output_path, encoding=self.encoding, xml_declaration=self.xml_declaration)
            
            # Set file permissions if specified
            if self.file_permissions is not None:
                os.chmod(output_path, self.file_permissions)
            
            # Update statistics
            file_size = os.path.getsize(output_path)
            self._stats["bytes_written"] += file_size
            self._stats["files_created"] += 1
            
            return {
                "file_path": output_path,
                "size": file_size,
                "format": "xml",
                "elements": element_count
            }
            
        except Exception as e:
            self.logger.error(f"Error writing XML file: {str(e)}")
            context.add_error(self.name, f"XML output error: {str(e)}")
            return None
    
    def _add_elements(self, parent: ET.Element, data: Any, name: str = None) -> int:
        """
        Recursively add XML elements based on the data structure.
        
        Args:
            parent: Parent XML element
            data: Data to convert to XML
            name: Optional element name override
            
        Returns:
            int: Number of elements added
        """
        element_count = 0
        
        if isinstance(data, dict):
            if name:
                # Create a named element for this dict
                element = ET.SubElement(parent, name)
                # Add dict items as child elements or attributes
                for key, value in data.items():
                    if isinstance(value, (dict, list, tuple)):
                        # Complex values are child elements
                        element_count += self._add_elements(element, value, key)
                    else:
                        # Simple values are attributes
                        element.attrib[key] = str(value) if value is not None else ""
                element_count += 1
            else:
                # Add dict items directly to parent
                for key, value in data.items():
                    element_count += self._add_elements(parent, value, key)
        
        elif isinstance(data, (list, tuple)):
            if name:
                # Use the provided name as a wrapper element
                wrapper = ET.SubElement(parent, name)
                element_count += 1
                
                # Add each item with the item_element name
                for item in data:
                    element_count += self._add_elements(wrapper, item, self.item_element)
            else:
                # Add items directly to parent with the item_element name
                for item in data:
                    element_count += self._add_elements(parent, item, self.item_element)
        
        else:
            # For scalar values, create an element with text content
            element = ET.SubElement(parent, name or self.item_element)
            element.text = str(data) if data is not None else ""
            element_count += 1
            
        return element_count
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for configuration.
        
        Returns:
            Dict containing JSON schema for this stage's configuration
        """
        schema = super().get_config_schema()
        
        # Add XML-specific properties
        xml_properties = {
            "root_element": {"type": "string"},
            "item_element": {"type": "string"},
            "pretty_print": {"type": "boolean"},
            "xml_declaration": {"type": "boolean"},
            "namespaces": {"type": "object"},
            "indent": {"type": "integer", "minimum": 0, "maximum": 8}
        }
        
        # Update the properties in the schema
        schema["properties"].update(xml_properties)
        
        return schema


class YAMLOutputStage(FileOutputStage):
    """
    Output stage for writing data to YAML files.
    
    Features:
    - Configurable YAML formatting
    - Support for document metadata
    - Comments in output
    
    Configuration:
    - sort_keys: Whether to sort dictionary keys
    - flow_style: Style for collections (block or flow)
    - default_style: Default scalar style
    - indent: Indentation level
    """
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize a new YAML output stage.
        
        Args:
            name: Name of the stage, defaults to class name if not provided
            config: Configuration parameters for this stage
        """
        super().__init__(name, config)
        
        # Check if PyYAML is available
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required for YAMLOutputStage")
        
        # YAML-specific configuration
        self.sort_keys = self.config.get("sort_keys", False)
        self.flow_style = self.config.get("flow_style", False)
        self.default_style = self.config.get("default_style", None)
        self.indent = self.config.get("indent", 2)
        
        # Ensure file has .yaml or .yml extension
        if not self.file_path.lower().endswith(('.yaml', '.yml')):
            self.file_path += '.yaml'
    
    async def validate_output_format(self, context: PipelineContext) -> bool:
        """
        Validate that the data can be written as YAML.
        
        Args:
            context: The shared pipeline context
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        if not await super().validate_output_format(context):
            return False
            
        # Check that PyYAML is available
        if not YAML_AVAILABLE:
            self.logger.error("PyYAML is required for YAML output")
            context.add_error(self.name, "PyYAML is required for YAML output")
            return False
            
        # Check that the data can be serialized to YAML
        data = self._prepare_output_data(context)
        if data is None:
            self.logger.error("No data to write")
            context.add_error(self.name, "No data to write")
            return False
            
        # Try to serialize the data to YAML to check for any issues
        try:
            yaml.dump(data, default_flow_style=self.flow_style)
            return True
        except Exception as e:
            self.logger.error(f"Data cannot be serialized to YAML: {str(e)}")
            context.add_error(self.name, f"YAML serialization error: {str(e)}")
            return False
    
    async def deliver_output(self, data: Any, context: PipelineContext) -> Optional[Dict[str, Any]]:
        """
        Write the data to a YAML file.
        
        Args:
            data: The data to write
            context: The shared pipeline context
            
        Returns:
            Dict containing output file metadata or None if output failed
        """
        try:
            # Get actual file path based on naming strategy
            output_path = await self._get_output_path(data)
            
            # Ensure parent directory exists
            if self.create_directories:
                directory = os.path.dirname(os.path.abspath(output_path))
                if not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                    self._stats["directories_created"] += 1
            
            # Use atomic write with temporary file
            temp_path = None
            if self.atomic_write:
                directory = os.path.dirname(os.path.abspath(output_path))
                fd, temp_path = tempfile.mkstemp(dir=directory, prefix='.tmp', suffix='.yaml')
                try:
                    with os.fdopen(fd, 'w', encoding=self.encoding) as f:
                        yaml.dump(
                            data,
                            f,
                            default_flow_style=self.flow_style,
                            default_style=self.default_style,
                            indent=self.indent,
                            sort_keys=self.sort_keys
                        )
                    
                    # Atomically rename the temporary file to the target file
                    os.replace(temp_path, output_path)
                    temp_path = None  # Prevent deletion in finally block
                finally:
                    # Clean up temporary file if still exists
                    if temp_path and os.path.exists(temp_path):
                        try:
                            os.unlink(temp_path)
                        except:
                            pass
            else:
                # Direct write
                with open(output_path, 'w', encoding=self.encoding) as f:
                    yaml.dump(
                        data,
                        f,
                        default_flow_style=self.flow_style,
                        default_style=self.default_style,
                        indent=self.indent,
                        sort_keys=self.sort_keys
                    )
            
            # Set file permissions if specified
            if self.file_permissions is not None:
                os.chmod(output_path, self.file_permissions)
            
            # Update statistics
            file_size = os.path.getsize(output_path)
            self._stats["bytes_written"] += file_size
            self._stats["files_created"] += 1
            
            return {
                "file_path": output_path,
                "size": file_size,
                "format": "yaml"
            }
            
        except Exception as e:
            self.logger.error(f"Error writing YAML file: {str(e)}")
            context.add_error(self.name, f"YAML output error: {str(e)}")
            return None
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for configuration.
        
        Returns:
            Dict containing JSON schema for this stage's configuration
        """
        schema = super().get_config_schema()
        
        # Add YAML-specific properties
        yaml_properties = {
            "sort_keys": {"type": "boolean"},
            "flow_style": {"type": "boolean"},
            "default_style": {"type": "string"},
            "indent": {"type": "integer", "minimum": 0, "maximum": 8}
        }
        
        # Update the properties in the schema
        schema["properties"].update(yaml_properties)
        
        return schema


class TextOutputStage(FileOutputStage):
    """
    Output stage for writing data to plain text files.
    
    Features:
    - Configurable text formatting
    - Line ending control
    - Template substitution
    
    Configuration:
    - template: Optional text template with {placeholders}
    - line_ending: Line ending style (LF, CRLF)
    - join_delimiter: Delimiter for joining elements when data is a list
    """
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize a new text output stage.
        
        Args:
            name: Name of the stage, defaults to class name if not provided
            config: Configuration parameters for this stage
        """
        super().__init__(name, config)
        
        # Text-specific configuration
        self.template = self.config.get("template")
        self.line_ending = self.config.get("line_ending", "LF")
        self.join_delimiter = self.config.get("join_delimiter", "\n")
        
        # Convert line endings to actual characters
        self.line_ending_char = "\n" if self.line_ending == "LF" else "\r\n"
        
        # Ensure file has a reasonable extension if none is provided
        if not os.path.splitext(self.file_path)[1]:
            self.file_path += '.txt'
    
    async def validate_output_format(self, context: PipelineContext) -> bool:
        """
        Validate text output configuration.
        
        Args:
            context: The shared pipeline context
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        if not await super().validate_output_format(context):
            return False
            
        # Check template format if provided
        if self.template:
            try:
                # Ensure template can be formatted
                sample_data = {"test": "value"}
                self.template.format(**sample_data)
            except Exception as e:
                self.logger.error(f"Invalid template format: {str(e)}")
                context.add_error(self.name, f"Template format error: {str(e)}")
                return False
                
        return True
    
    async def deliver_output(self, data: Any, context: PipelineContext) -> Optional[Dict[str, Any]]:
        """
        Write the data to a text file.
        
        Args:
            data: The data to write
            context: The shared pipeline context
            
        Returns:
            Dict containing output file metadata or None if output failed
        """
        try:
            # Get actual file path based on naming strategy
            output_path = await self._get_output_path(data)
            
            # Ensure parent directory exists
            if self.create_directories:
                directory = os.path.dirname(os.path.abspath(output_path))
                if not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                    self._stats["directories_created"] += 1
            
            # Format the content as text
            content = self._format_text_content(data, context)
            
            # Use atomic write with temporary file
            temp_path = None
            if self.atomic_write:
                directory = os.path.dirname(os.path.abspath(output_path))
                fd, temp_path = tempfile.mkstemp(dir=directory, prefix='.tmp', suffix='.txt')
                try:
                    with os.fdopen(fd, 'w', encoding=self.encoding, newline='') as f:
                        f.write(content)
                    
                    # Atomically rename the temporary file to the target file
                    os.replace(temp_path, output_path)
                    temp_path = None  # Prevent deletion in finally block
                finally:
                    # Clean up temporary file if still exists
                    if temp_path and os.path.exists(temp_path):
                        try:
                            os.unlink(temp_path)
                        except:
                            pass
            else:
                # Direct write
                with open(output_path, 'w', encoding=self.encoding, newline='') as f:
                    f.write(content)
            
            # Set file permissions if specified
            if self.file_permissions is not None:
                os.chmod(output_path, self.file_permissions)
            
            # Update statistics
            file_size = os.path.getsize(output_path)
            self._stats["bytes_written"] += file_size
            self._stats["files_created"] += 1
            
            # Count lines
            line_count = content.count(self.line_ending_char) + (1 if content else 0)
            
            return {
                "file_path": output_path,
                "size": file_size,
                "format": "text",
                "lines": line_count
            }
            
        except Exception as e:
            self.logger.error(f"Error writing text file: {str(e)}")
            context.add_error(self.name, f"Text output error: {str(e)}")
            return None
    
    def _format_text_content(self, data: Any, context: PipelineContext) -> str:
        """
        Format data as text according to configuration.
        
        Args:
            data: The data to format
            context: The shared pipeline context
            
        Returns:
            str: Formatted text content
        """
        # Convert data to string representation
        if isinstance(data, str):
            content = data
        elif isinstance(data, (list, tuple)):
            # Join list items with the configured delimiter
            items = [str(item) for item in data]
            content = self.join_delimiter.join(items)
        elif isinstance(data, dict):
            if self.template:
                # Use template string with dictionary
                try:
                    content = self.template.format(**data)
                except KeyError:
                    # Fall back to string representation
                    content = str(data)
            else:
                # Format as key-value pairs
                lines = [f"{key}: {value}" for key, value in data.items()]
                content = self.join_delimiter.join(lines)
        else:
            # Default string representation
            content = str(data)
        
        # Normalize line endings
        if self.line_ending == "CRLF":
            content = content.replace("\r\n", "\n").replace("\n", "\r\n")
        elif self.line_ending == "LF":
            content = content.replace("\r\n", "\n")
            
        return content
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for configuration.
        
        Returns:
            Dict containing JSON schema for this stage's configuration
        """
        schema = super().get_config_schema()
        
        # Add text-specific properties
        text_properties = {
            "template": {"type": "string"},
            "line_ending": {
                "type": "string",
                "enum": ["LF", "CRLF"]
            },
            "join_delimiter": {"type": "string"}
        }
        
        # Update the properties in the schema
        schema["properties"].update(text_properties)
        
        return schema