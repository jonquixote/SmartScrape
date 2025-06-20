"""
File Input Stage Module.

This module provides a FileInputStage that specializes in acquiring data from local files
with features like format detection, encoding handling, and efficient processing of large files.
"""

import asyncio
import csv
import glob
import io
import json
import logging
import os
import re
import shutil
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Set, Tuple, Union

import chardet
import fcntl
import yaml
from filelock import FileLock

from core.pipeline.stages.base_stages import InputStage
from core.pipeline.context import PipelineContext
from core.pipeline.dto import PipelineRequest, PipelineResponse, ResponseStatus


class FileFormat(Enum):
    """Enumeration of supported file formats."""
    TEXT = auto()
    JSON = auto()
    CSV = auto()
    XML = auto()
    YAML = auto()
    BINARY = auto()
    UNKNOWN = auto()


class FileInputStage(InputStage):
    """
    File Input Stage for file-based data acquisition.
    
    Features:
    - Support for various file formats (text, JSON, CSV, XML, YAML)
    - Automatic file encoding detection and handling
    - Line-by-line reading for large files
    - Glob pattern support for batch processing
    - File modification time checking
    - File locking for concurrent access
    - Directory traversal capabilities
    - Symlink handling
    
    Configuration:
    - file_path: Path to the file or directory
    - pattern: Glob pattern for selecting multiple files
    - format: Explicit file format (auto-detected if not specified)
    - encoding: Character encoding (auto-detected if not specified)
    - max_size: Maximum file size to process (in bytes)
    - csv_options: Options for CSV parsing
    - traversal_options: Options for directory traversal
    - follow_symlinks: Whether to follow symbolic links
    - lock_timeout: Maximum time to wait for a file lock (in seconds)
    """
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize a new file input stage.
        
        Args:
            name (Optional[str]): Name of the stage, defaults to class name
            config (Optional[Dict[str, Any]]): Configuration parameters
        """
        super().__init__(name, config)
        
        # Store file-specific configuration
        self.file_path = self.config.get("file_path", "")
        self.pattern = self.config.get("pattern", None)
        self.format = self.config.get("format", None)
        self.encoding = self.config.get("encoding", None)
        self.max_size = self.config.get("max_size", 1024 * 1024 * 100)  # 100MB default
        self.csv_options = self.config.get("csv_options", {})
        self.traversal_options = self.config.get("traversal_options", {})
        self.follow_symlinks = self.config.get("follow_symlinks", False)
        self.lock_timeout = self.config.get("lock_timeout", 30)
        
        # Set up max line size for line-by-line reading
        self.max_line_size = self.config.get("max_line_size", 1024 * 1024)  # 1MB per line default
        
        # Set up batch processing options
        self.batch_enabled = self.config.get("batch_enabled", False)
        self.batch_size = self.config.get("batch_size", 1000)
        self.max_workers = self.config.get("max_workers", 4)
        
        # Initialize logger
        self.logger = logging.getLogger(f"pipeline.stages.input.file.{self.name}")
        
        # File locks registry
        self._locks = {}
    
    async def acquire_data(self, request: PipelineRequest, context: PipelineContext) -> Optional[PipelineResponse]:
        """
        Acquire data from a file source.
        
        Args:
            request (PipelineRequest): The input request data
            context (PipelineContext): The shared pipeline context
            
        Returns:
            Optional[PipelineResponse]: The acquired data or None if acquisition failed
        """
        # Get the file path(s) from request or config
        file_paths = self._get_file_paths(request, context)
        if not file_paths:
            self.logger.error("No file paths specified")
            return None
        
        # If we have multiple files, process them based on configuration
        if len(file_paths) > 1:
            return await self._process_multiple_files(file_paths, context)
        
        # Process a single file
        file_path = file_paths[0]
        return await self._process_single_file(file_path, context)
    
    async def _process_single_file(self, file_path: str, context: PipelineContext) -> Optional[PipelineResponse]:
        """
        Process a single file.
        
        Args:
            file_path (str): The file path
            context (PipelineContext): The shared pipeline context
            
        Returns:
            Optional[PipelineResponse]: The acquired data or None if acquisition failed
        """
        # Check if file exists
        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            return self._create_error_response(
                file_path, 
                "File not found", 
                ResponseStatus.NOT_FOUND
            )
        
        # Check if it's a directory
        if os.path.isdir(file_path):
            if self.traversal_options.get("process_directories", False):
                return await self._process_directory(file_path, context)
            else:
                self.logger.error(f"Path is a directory: {file_path}")
                return self._create_error_response(
                    file_path, 
                    "Path is a directory", 
                    ResponseStatus.ERROR
                )
        
        # Check if it's a symlink
        if os.path.islink(file_path) and not self.follow_symlinks:
            self.logger.warning(f"Skipping symlink: {file_path}")
            return self._create_error_response(
                file_path, 
                "Symlink following is disabled", 
                ResponseStatus.ERROR
            )
        
        # Check file size
        try:
            file_size = os.path.getsize(file_path)
            if file_size > self.max_size:
                self.logger.error(f"File too large: {file_path} ({file_size} bytes)")
                return self._create_error_response(
                    file_path, 
                    f"File too large ({file_size} bytes)", 
                    ResponseStatus.ERROR
                )
        except OSError as e:
            self.logger.error(f"Error checking file size: {str(e)}")
            return self._create_error_response(
                file_path, 
                f"Error accessing file: {str(e)}", 
                ResponseStatus.ERROR
            )
        
        # Determine file format
        format_type = self._detect_format(file_path)
        
        # Acquire file lock if needed
        lock_path = f"{file_path}.lock"
        try:
            async with self._get_file_lock(lock_path):
                # Read and process the file based on its format
                if format_type == FileFormat.JSON:
                    return await self._read_json_file(file_path)
                elif format_type == FileFormat.CSV:
                    return await self._read_csv_file(file_path)
                elif format_type == FileFormat.XML:
                    return await self._read_xml_file(file_path)
                elif format_type == FileFormat.YAML:
                    return await self._read_yaml_file(file_path)
                elif format_type == FileFormat.TEXT:
                    return await self._read_text_file(file_path)
                elif format_type == FileFormat.BINARY:
                    return await self._read_binary_file(file_path)
                else:
                    self.logger.error(f"Unsupported file format: {file_path}")
                    return self._create_error_response(
                        file_path, 
                        "Unsupported file format", 
                        ResponseStatus.ERROR
                    )
        except TimeoutError:
            self.logger.error(f"Timeout waiting for file lock: {file_path}")
            return self._create_error_response(
                file_path, 
                "Timeout waiting for file lock", 
                ResponseStatus.TIMEOUT
            )
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")
            return self._create_error_response(
                file_path, 
                f"Error processing file: {str(e)}", 
                ResponseStatus.ERROR
            )
    
    async def _process_multiple_files(self, file_paths: List[str], context: PipelineContext) -> Optional[PipelineResponse]:
        """
        Process multiple files, either combining results or processing in batch.
        
        Args:
            file_paths (List[str]): List of file paths
            context (PipelineContext): The shared pipeline context
            
        Returns:
            Optional[PipelineResponse]: The acquired data or None if acquisition failed
        """
        self.logger.info(f"Processing {len(file_paths)} files")
        
        # Check whether to process in batches or combine results
        if self.batch_enabled:
            return await self._process_batch(file_paths, context)
        else:
            # Process all files and combine results
            all_data = {}
            file_metadata = []
            
            for file_path in file_paths:
                response = await self._process_single_file(file_path, context)
                
                if response and response.is_success and response.data:
                    # Get filename for use as a key
                    filename = os.path.basename(file_path)
                    
                    # Store the data under the filename key
                    all_data[filename] = response.data
                    
                    # Keep metadata for each file
                    file_metadata.append({
                        "path": file_path,
                        "size": os.path.getsize(file_path),
                        "modified": os.path.getmtime(file_path),
                        "format": str(self._detect_format(file_path))
                    })
                elif not response or not response.is_success:
                    self.logger.warning(f"Failed to process file: {file_path}")
            
            if not all_data:
                self.logger.error("No data acquired from any file")
                return self._create_error_response(
                    ",".join(file_paths), 
                    "No data acquired from any file", 
                    ResponseStatus.ERROR
                )
            
            # Return combined results
            return PipelineResponse(
                status=ResponseStatus.SUCCESS,
                data={
                    "files": all_data,
                    "file_count": len(all_data)
                },
                source=",".join(file_paths),
                metadata={
                    "file_metadata": file_metadata,
                    "total_files": len(file_paths),
                    "successful_files": len(all_data)
                }
            )
    
    async def _process_batch(self, file_paths: List[str], context: PipelineContext) -> Optional[PipelineResponse]:
        """
        Process files in batches, executing in parallel using a thread pool.
        
        Args:
            file_paths (List[str]): List of file paths
            context (PipelineContext): The shared pipeline context
            
        Returns:
            Optional[PipelineResponse]: The acquired data with batch results
        """
        self.logger.info(f"Processing {len(file_paths)} files in batches")
        
        # Create result containers
        all_results = []
        errors = []
        
        # Process files in batches using a thread pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all files for processing
            future_to_file = {
                executor.submit(self._process_file_sync, file_path): file_path
                for file_path in file_paths
            }
            
            # Collect results as they complete
            for future in future_to_file:
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        all_results.append(result)
                    else:
                        errors.append({"file": file_path, "error": "Processing failed"})
                except Exception as e:
                    errors.append({"file": file_path, "error": str(e)})
        
        # Check if we have any successful results
        if not all_results:
            self.logger.error("No data acquired from any file in batch")
            return self._create_error_response(
                "batch_processing", 
                "No data acquired from any file in batch", 
                ResponseStatus.ERROR
            )
        
        # Return batch results
        return PipelineResponse(
            status=ResponseStatus.SUCCESS if not errors else ResponseStatus.PARTIAL,
            data={
                "batch_results": all_results,
                "batch_size": len(all_results)
            },
            source="batch_processing",
            metadata={
                "total_files": len(file_paths),
                "successful_files": len(all_results),
                "failed_files": len(errors),
                "errors": errors if errors else None
            }
        )
    
    def _process_file_sync(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Process a file synchronously for batch processing.
        
        Args:
            file_path (str): The file path
            
        Returns:
            Optional[Dict[str, Any]]: The processed file data or None
        """
        # Run the async method in a new event loop
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                response = loop.run_until_complete(
                    self._process_single_file(file_path, PipelineContext({}))
                )
                
                if response and response.is_success and response.data:
                    return {
                        "file": file_path,
                        "data": response.data,
                        "metadata": {
                            "size": os.path.getsize(file_path),
                            "modified": os.path.getmtime(file_path),
                            "format": str(self._detect_format(file_path))
                        }
                    }
                return None
            finally:
                loop.close()
        except Exception as e:
            self.logger.error(f"Error in sync processing of {file_path}: {str(e)}")
            return None
    
    async def _process_directory(self, dir_path: str, context: PipelineContext) -> Optional[PipelineResponse]:
        """
        Process a directory by traversing its contents.
        
        Args:
            dir_path (str): The directory path
            context (PipelineContext): The shared pipeline context
            
        Returns:
            Optional[PipelineResponse]: The acquired data or None if acquisition failed
        """
        # Check traversal depth
        max_depth = self.traversal_options.get("max_depth", 1)
        current_depth = context.get("traversal_depth", 0)
        
        if current_depth >= max_depth:
            self.logger.info(f"Reached maximum traversal depth at {dir_path}")
            return self._create_error_response(
                dir_path, 
                "Maximum traversal depth reached", 
                ResponseStatus.ERROR
            )
        
        # List directory contents
        try:
            contents = os.listdir(dir_path)
        except OSError as e:
            self.logger.error(f"Error listing directory {dir_path}: {str(e)}")
            return self._create_error_response(
                dir_path, 
                f"Error listing directory: {str(e)}", 
                ResponseStatus.ERROR
            )
        
        # Process directory contents
        files_info = []
        subdirs_info = []
        
        for item in contents:
            item_path = os.path.join(dir_path, item)
            
            # Skip hidden files if configured
            if item.startswith('.') and not self.traversal_options.get("include_hidden", False):
                continue
            
            # Skip symlinks if configured
            if os.path.islink(item_path) and not self.follow_symlinks:
                continue
            
            # Collect information based on item type
            if os.path.isfile(item_path):
                files_info.append({
                    "name": item,
                    "path": item_path,
                    "size": os.path.getsize(item_path),
                    "modified": os.path.getmtime(item_path),
                    "format": str(self._detect_format(item_path))
                })
            elif os.path.isdir(item_path):
                subdirs_info.append({
                    "name": item,
                    "path": item_path,
                    "item_count": len(os.listdir(item_path))
                })
        
        # Return directory information
        return PipelineResponse(
            status=ResponseStatus.SUCCESS,
            data={
                "directory": {
                    "path": dir_path,
                    "name": os.path.basename(dir_path),
                    "files": files_info,
                    "subdirectories": subdirs_info
                }
            },
            source=dir_path,
            metadata={
                "file_count": len(files_info),
                "subdir_count": len(subdirs_info),
                "traversal_depth": current_depth
            }
        )
    
    async def _read_text_file(self, file_path: str) -> PipelineResponse:
        """
        Read a text file, handling encoding and large files.
        
        Args:
            file_path (str): The file path
            
        Returns:
            PipelineResponse: The file content
        """
        # Determine encoding if not specified
        encoding = self.encoding or await self._detect_encoding(file_path)
        
        # Check if we should read line by line based on file size
        file_size = os.path.getsize(file_path)
        
        if file_size > self.max_line_size:
            # Read large file line by line
            lines = []
            line_count = 0
            
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    for line in f:
                        lines.append(line.rstrip('\n'))
                        line_count += 1
            except UnicodeDecodeError as e:
                # Try again with binary mode and then decode
                try:
                    with open(file_path, 'rb') as f:
                        content = f.read()
                        text = content.decode(encoding, errors='replace')
                        lines = text.splitlines()
                        line_count = len(lines)
                except Exception as inner_e:
                    self.logger.error(f"Error reading text file {file_path}: {str(inner_e)}")
                    return self._create_error_response(
                        file_path, 
                        f"Error reading text file: {str(inner_e)}", 
                        ResponseStatus.ERROR
                    )
            except Exception as e:
                self.logger.error(f"Error reading text file {file_path}: {str(e)}")
                return self._create_error_response(
                    file_path, 
                    f"Error reading text file: {str(e)}", 
                    ResponseStatus.ERROR
                )
            
            return PipelineResponse(
                status=ResponseStatus.SUCCESS,
                data={
                    "text": lines,
                    "line_count": line_count
                },
                source=file_path,
                metadata={
                    "encoding": encoding,
                    "file_size": file_size,
                    "line_count": line_count,
                    "format": "text"
                }
            )
        else:
            # Read smaller file all at once
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
            except UnicodeDecodeError as e:
                # Try again with binary mode and then decode
                try:
                    with open(file_path, 'rb') as f:
                        content_bytes = f.read()
                        content = content_bytes.decode(encoding, errors='replace')
                except Exception as inner_e:
                    self.logger.error(f"Error reading text file {file_path}: {str(inner_e)}")
                    return self._create_error_response(
                        file_path, 
                        f"Error reading text file: {str(inner_e)}", 
                        ResponseStatus.ERROR
                    )
            except Exception as e:
                self.logger.error(f"Error reading text file {file_path}: {str(e)}")
                return self._create_error_response(
                    file_path, 
                    f"Error reading text file: {str(e)}", 
                    ResponseStatus.ERROR
                )
            
            # Calculate line count
            line_count = content.count('\n') + (0 if content.endswith('\n') else 1)
            
            return PipelineResponse(
                status=ResponseStatus.SUCCESS,
                data={
                    "text": content,
                    "line_count": line_count
                },
                source=file_path,
                metadata={
                    "encoding": encoding,
                    "file_size": file_size,
                    "line_count": line_count,
                    "format": "text"
                }
            )
    
    async def _read_json_file(self, file_path: str) -> PipelineResponse:
        """
        Read and parse a JSON file.
        
        Args:
            file_path (str): The file path
            
        Returns:
            PipelineResponse: The parsed JSON data
        """
        # Determine encoding if not specified
        encoding = self.encoding or await self._detect_encoding(file_path)
        
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                data = json.load(f)
            
            return PipelineResponse(
                status=ResponseStatus.SUCCESS,
                data={
                    "json": data
                },
                source=file_path,
                metadata={
                    "encoding": encoding,
                    "file_size": os.path.getsize(file_path),
                    "format": "json"
                }
            )
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error in {file_path}: {str(e)}")
            return self._create_error_response(
                file_path, 
                f"JSON decode error: {str(e)}", 
                ResponseStatus.ERROR
            )
        except Exception as e:
            self.logger.error(f"Error reading JSON file {file_path}: {str(e)}")
            return self._create_error_response(
                file_path, 
                f"Error reading JSON file: {str(e)}", 
                ResponseStatus.ERROR
            )
    
    async def _read_csv_file(self, file_path: str) -> PipelineResponse:
        """
        Read and parse a CSV file.
        
        Args:
            file_path (str): The file path
            
        Returns:
            PipelineResponse: The parsed CSV data
        """
        # Determine encoding if not specified
        encoding = self.encoding or await self._detect_encoding(file_path)
        
        # Set up CSV options
        csv_options = {
            "delimiter": self.csv_options.get("delimiter", ","),
            "quotechar": self.csv_options.get("quotechar", '"'),
            "quoting": getattr(csv, self.csv_options.get("quoting", "QUOTE_MINIMAL"))
        }
        
        has_header = self.csv_options.get("has_header", True)
        
        try:
            rows = []
            fieldnames = None
            
            with open(file_path, 'r', newline='', encoding=encoding) as f:
                # Determine if we should use the first row as header
                if has_header:
                    reader = csv.DictReader(f, **csv_options)
                    fieldnames = reader.fieldnames
                    rows = list(reader)
                else:
                    reader = csv.reader(f, **csv_options)
                    rows = list(reader)
            
            return PipelineResponse(
                status=ResponseStatus.SUCCESS,
                data={
                    "csv": rows,
                    "fieldnames": fieldnames,
                    "row_count": len(rows)
                },
                source=file_path,
                metadata={
                    "encoding": encoding,
                    "file_size": os.path.getsize(file_path),
                    "format": "csv",
                    "has_header": has_header,
                    "csv_options": csv_options
                }
            )
        except Exception as e:
            self.logger.error(f"Error reading CSV file {file_path}: {str(e)}")
            return self._create_error_response(
                file_path, 
                f"Error reading CSV file: {str(e)}", 
                ResponseStatus.ERROR
            )
    
    async def _read_xml_file(self, file_path: str) -> PipelineResponse:
        """
        Read and parse an XML file.
        
        Args:
            file_path (str): The file path
            
        Returns:
            PipelineResponse: The parsed XML data
        """
        try:
            # Parse XML
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Convert XML to dict (simplified approach)
            def xml_to_dict(element):
                result = {element.tag: {} if element.attrib else None}
                children = list(element)
                
                if children:
                    child_dict = {}
                    for child in children:
                        child_data = xml_to_dict(child)
                        if child.tag in child_dict:
                            # If this tag already exists, convert to list or append
                            if not isinstance(child_dict[child.tag], list):
                                child_dict[child.tag] = [child_dict[child.tag]]
                            child_dict[child.tag].append(child_data[child.tag])
                        else:
                            child_dict.update(child_data)
                    
                    if element.attrib:
                        result[element.tag].update(element.attrib)
                        result[element.tag].update({"__content": child_dict})
                    else:
                        result[element.tag] = child_dict
                else:
                    # Element has no children, just text
                    if element.attrib:
                        result[element.tag].update(element.attrib)
                        if element.text and element.text.strip():
                            result[element.tag]["__text"] = element.text
                    else:
                        result[element.tag] = element.text if element.text and element.text.strip() else {}
                
                return result
            
            # Convert XML to dictionary
            xml_dict = xml_to_dict(root)
            
            # Also include the raw XML string
            with open(file_path, 'r', encoding=self.encoding or 'utf-8') as f:
                xml_string = f.read()
            
            return PipelineResponse(
                status=ResponseStatus.SUCCESS,
                data={
                    "xml": xml_dict,
                    "xml_string": xml_string,
                    "root_tag": root.tag
                },
                source=file_path,
                metadata={
                    "file_size": os.path.getsize(file_path),
                    "format": "xml",
                    "element_count": len(list(root.iter()))
                }
            )
        except ET.ParseError as e:
            self.logger.error(f"XML parse error in {file_path}: {str(e)}")
            return self._create_error_response(
                file_path, 
                f"XML parse error: {str(e)}", 
                ResponseStatus.ERROR
            )
        except Exception as e:
            self.logger.error(f"Error reading XML file {file_path}: {str(e)}")
            return self._create_error_response(
                file_path, 
                f"Error reading XML file: {str(e)}", 
                ResponseStatus.ERROR
            )
    
    async def _read_yaml_file(self, file_path: str) -> PipelineResponse:
        """
        Read and parse a YAML file.
        
        Args:
            file_path (str): The file path
            
        Returns:
            PipelineResponse: The parsed YAML data
        """
        # Determine encoding if not specified
        encoding = self.encoding or await self._detect_encoding(file_path)
        
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                data = yaml.safe_load(f)
            
            return PipelineResponse(
                status=ResponseStatus.SUCCESS,
                data={
                    "yaml": data
                },
                source=file_path,
                metadata={
                    "encoding": encoding,
                    "file_size": os.path.getsize(file_path),
                    "format": "yaml"
                }
            )
        except yaml.YAMLError as e:
            self.logger.error(f"YAML parse error in {file_path}: {str(e)}")
            return self._create_error_response(
                file_path, 
                f"YAML parse error: {str(e)}", 
                ResponseStatus.ERROR
            )
        except Exception as e:
            self.logger.error(f"Error reading YAML file {file_path}: {str(e)}")
            return self._create_error_response(
                file_path, 
                f"Error reading YAML file: {str(e)}", 
                ResponseStatus.ERROR
            )
    
    async def _read_binary_file(self, file_path: str) -> PipelineResponse:
        """
        Read a binary file.
        
        Args:
            file_path (str): The file path
            
        Returns:
            PipelineResponse: The binary file content
        """
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # Determine MIME type based on file extension
            mime_type = self._get_mime_type(file_path)
            
            return PipelineResponse(
                status=ResponseStatus.SUCCESS,
                data={
                    "binary": content,
                    "content_type": mime_type
                },
                source=file_path,
                metadata={
                    "file_size": len(content),
                    "format": "binary",
                    "mime_type": mime_type
                }
            )
        except Exception as e:
            self.logger.error(f"Error reading binary file {file_path}: {str(e)}")
            return self._create_error_response(
                file_path, 
                f"Error reading binary file: {str(e)}", 
                ResponseStatus.ERROR
            )
    
    def _get_file_paths(self, request: PipelineRequest, context: PipelineContext) -> List[str]:
        """
        Get file paths from request, config, or pattern.
        
        Args:
            request (PipelineRequest): The input request
            context (PipelineContext): The shared pipeline context
            
        Returns:
            List[str]: List of file paths
        """
        file_paths = []
        
        # Try to get file path from request source
        if request.source:
            if os.path.exists(request.source):
                file_paths.append(request.source)
            else:
                self.logger.warning(f"File not found: {request.source}")
        
        # Try using glob pattern from config
        if not file_paths and self.pattern:
            pattern = self.pattern
            
            # Replace template variables in pattern
            if '{' in pattern and '}' in pattern:
                try:
                    template_vars = {**context.data, **request.params}
                    pattern = pattern.format(**template_vars)
                except KeyError as e:
                    self.logger.warning(f"Missing template variable for pattern: {str(e)}")
                except Exception as e:
                    self.logger.warning(f"Error applying pattern template: {str(e)}")
            
            # Apply glob pattern
            try:
                glob_paths = glob.glob(pattern, recursive=True)
                if glob_paths:
                    # Filter out directories if not processing them
                    if not self.traversal_options.get("process_directories", False):
                        glob_paths = [p for p in glob_paths if not os.path.isdir(p)]
                    
                    # Filter out symlinks if not following them
                    if not self.follow_symlinks:
                        glob_paths = [p for p in glob_paths if not os.path.islink(p)]
                    
                    file_paths.extend(glob_paths)
                else:
                    self.logger.warning(f"No files found for pattern: {pattern}")
            except Exception as e:
                self.logger.warning(f"Error applying glob pattern: {str(e)}")
        
        # Try file path from config
        if not file_paths and self.file_path:
            if os.path.exists(self.file_path):
                file_paths.append(self.file_path)
            else:
                self.logger.warning(f"File not found: {self.file_path}")
        
        return file_paths
    
    def _detect_format(self, file_path: str) -> FileFormat:
        """
        Detect the format of a file.
        
        Args:
            file_path (str): The file path
            
        Returns:
            FileFormat: The detected file format
        """
        # Use explicit format if specified
        if self.format:
            try:
                return FileFormat[self.format.upper()]
            except (KeyError, AttributeError):
                self.logger.warning(f"Invalid format specified: {self.format}")
        
        # Detect based on file extension
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in {'.json'}:
            return FileFormat.JSON
        elif ext in {'.csv', '.tsv'}:
            return FileFormat.CSV
        elif ext in {'.xml', '.html', '.htm', '.xhtml'}:
            return FileFormat.XML
        elif ext in {'.yaml', '.yml'}:
            return FileFormat.YAML
        elif ext in {'.txt', '.md', '.log', '.py', '.js', '.java', '.c', '.cpp', '.h', '.cs', '.php', '.rb'}:
            return FileFormat.TEXT
        elif ext in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.pdf', '.zip', '.exe', '.bin'}:
            return FileFormat.BINARY
        
        # If extension doesn't give a clear answer, try to peek at content
        try:
            with open(file_path, 'rb') as f:
                header = f.read(4096)  # Read first 4KB
                
                # Check for common file signatures
                if header.startswith(b'{') or header.startswith(b'['):
                    # Looks like JSON
                    try:
                        json.loads(header)
                        return FileFormat.JSON
                    except json.JSONDecodeError:
                        pass
                
                if header.startswith(b'<'):
                    # Might be XML
                    try:
                        ET.fromstring(header)
                        return FileFormat.XML
                    except ET.ParseError:
                        pass
                
                # Check if it's CSV by counting commas in the first few lines
                comma_line_ratio = 0
                newline_count = header.count(b'\n')
                if newline_count > 0:
                    comma_count = header.count(b',')
                    comma_line_ratio = comma_count / newline_count
                    
                    if comma_line_ratio >= 3:  # Arbitrary threshold
                        return FileFormat.CSV
                
                # Check if it might be YAML (harder to detect)
                if b':' in header and b'\n' in header and not b'{' in header[:20]:
                    return FileFormat.YAML
                
                # Check if this looks like text
                is_text = True
                for byte in header:
                    # Check for non-printable, non-whitespace characters
                    if byte < 9 or (byte > 13 and byte < 32) or byte > 126:
                        is_text = False
                        break
                
                if is_text:
                    return FileFormat.TEXT
                
                # Default to binary
                return FileFormat.BINARY
                
        except Exception as e:
            self.logger.warning(f"Error detecting file format: {str(e)}")
            return FileFormat.UNKNOWN
    
    async def _detect_encoding(self, file_path: str) -> str:
        """
        Detect the encoding of a text file.
        
        Args:
            file_path (str): The file path
            
        Returns:
            str: The detected encoding
        """
        # Read a sample of the file to detect encoding
        try:
            with open(file_path, 'rb') as f:
                # Read a sample (up to 100KB)
                sample = f.read(1024 * 100)
            
            # Use chardet to detect encoding
            result = chardet.detect(sample)
            encoding = result['encoding'] or 'utf-8'
            
            # Default to utf-8 for high confidence, otherwise use detected
            if result['confidence'] < 0.7:
                self.logger.warning(
                    f"Low confidence encoding detection ({result['confidence']:.2f}) for {file_path}. "
                    f"Detected {encoding}, using utf-8 as fallback."
                )
                return 'utf-8'
            
            return encoding
        except Exception as e:
            self.logger.warning(f"Error detecting encoding: {str(e)}. Defaulting to utf-8.")
            return 'utf-8'
    
    def _get_mime_type(self, file_path: str) -> str:
        """
        Get MIME type based on file extension.
        
        Args:
            file_path (str): The file path
            
        Returns:
            str: The MIME type
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        # Common MIME types mapping
        mime_types = {
            '.txt': 'text/plain',
            '.html': 'text/html',
            '.htm': 'text/html',
            '.css': 'text/css',
            '.csv': 'text/csv',
            '.json': 'application/json',
            '.xml': 'application/xml',
            '.yaml': 'application/x-yaml',
            '.yml': 'application/x-yaml',
            '.pdf': 'application/pdf',
            '.doc': 'application/msword',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.xls': 'application/vnd.ms-excel',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.svg': 'image/svg+xml',
            '.zip': 'application/zip',
            '.mp3': 'audio/mpeg',
            '.mp4': 'video/mp4',
            '.avi': 'video/x-msvideo',
            '.py': 'text/x-python',
            '.js': 'text/javascript'
        }
        
        return mime_types.get(ext, 'application/octet-stream')
    
    @contextmanager
    def _get_file_lock_sync(self, lock_path: str):
        """
        Acquire a file lock synchronously using a context manager.
        
        Args:
            lock_path (str): Path to the lock file
            
        Yields:
            None
            
        Raises:
            TimeoutError: If lock acquisition times out
        """
        # Use filelock library for more robust file locking
        lock = FileLock(lock_path, timeout=self.lock_timeout)
        
        try:
            lock.acquire()
            yield
        finally:
            if lock.is_locked:
                lock.release()
    
    async def _get_file_lock(self, lock_path: str):
        """
        Async wrapper around the file lock context manager.
        
        Args:
            lock_path (str): Path to the lock file
            
        Returns:
            Context manager for the file lock
            
        Raises:
            TimeoutError: If lock acquisition times out
        """
        return self._get_file_lock_sync(lock_path)
    
    def _create_error_response(self, source: str, message: str, 
                             status: ResponseStatus) -> PipelineResponse:
        """
        Create an error response.
        
        Args:
            source (str): The source of the error
            message (str): The error message
            status (ResponseStatus): The error status
            
        Returns:
            PipelineResponse: The error response
        """
        return PipelineResponse(
            status=status,
            data=None,
            source=source,
            error_message=message,
            metadata={"error_type": status.name}
        )
    
    async def validate_source_config(self) -> bool:
        """
        Validate the file source configuration.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        # Check if we have a file path, pattern, or expecting it from request
        if not self.file_path and not self.pattern and not self.config.get("expect_from_request", False):
            self.logger.error("No file path or pattern specified")
            return False
        
        # If file_path is specified, check that it exists
        if self.file_path and not os.path.exists(self.file_path):
            self.logger.warning(f"File path does not exist: {self.file_path}")
            if not self.config.get("allow_nonexistent", False):
                return False
        
        # Check format if explicitly specified
        if self.format:
            try:
                FileFormat[self.format.upper()]
            except KeyError:
                self.logger.error(f"Invalid format specified: {self.format}")
                return False
        
        return True
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for file input stage configuration.
        
        Returns:
            Dict[str, Any]: Dictionary containing JSON schema
        """
        schema = super().get_config_schema()
        
        # Add file-specific properties
        file_properties = {
            "file_path": {"type": "string"},
            "pattern": {"type": "string"},
            "format": {"type": "string", "enum": [f.name for f in FileFormat]},
            "encoding": {"type": "string"},
            "max_size": {"type": "integer", "minimum": 1},
            "max_line_size": {"type": "integer", "minimum": 1},
            "csv_options": {
                "type": "object",
                "properties": {
                    "delimiter": {"type": "string"},
                    "quotechar": {"type": "string"},
                    "quoting": {"type": "string", "enum": ["QUOTE_MINIMAL", "QUOTE_ALL", "QUOTE_NONNUMERIC", "QUOTE_NONE"]},
                    "has_header": {"type": "boolean"}
                }
            },
            "traversal_options": {
                "type": "object",
                "properties": {
                    "max_depth": {"type": "integer", "minimum": 0},
                    "process_directories": {"type": "boolean"},
                    "include_hidden": {"type": "boolean"}
                }
            },
            "follow_symlinks": {"type": "boolean"},
            "lock_timeout": {"type": "number", "minimum": 0},
            "batch_enabled": {"type": "boolean"},
            "batch_size": {"type": "integer", "minimum": 1},
            "max_workers": {"type": "integer", "minimum": 1},
            "expect_from_request": {"type": "boolean"},
            "allow_nonexistent": {"type": "boolean"}
        }
        
        # Update the properties in the schema
        schema["properties"].update(file_properties)
        
        return schema
    
    async def shutdown(self) -> None:
        """Shutdown the file input stage and clean up resources."""
        # Clean up any locks that might still be held
        for lock_path in self._locks:
            try:
                if self._locks[lock_path].is_locked:
                    self._locks[lock_path].release()
            except Exception as e:
                self.logger.warning(f"Error releasing lock {lock_path}: {str(e)}")
        
        self._locks.clear()