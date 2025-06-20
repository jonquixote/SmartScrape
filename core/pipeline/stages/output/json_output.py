"""
JSON Output Stage Module.

This module provides specialized output stages for JSON formatting and delivery.
"""

import asyncio
import json
import logging
import os
import tempfile
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union, BinaryIO, TextIO
import ijson  # For incremental JSON parsing

from core.pipeline.stages.base_stages import OutputStage
from core.pipeline.context import PipelineContext


class JSONFormattingMode(Enum):
    """JSON formatting modes."""
    COMPACT = auto()  # Minimal whitespace
    PRETTY = auto()   # Indented for readability
    LINES = auto()    # Newline-delimited JSON


class JSONOutputStage(OutputStage):
    """
    Output stage for formatting and writing JSON data.
    
    Features:
    - Configurable formatting (compact, pretty, newline-delimited)
    - Schema validation of output data
    - Direct response or file output
    - JSON streaming for large datasets
    - Support for incremental updates
    
    Configuration:
    - formatting_mode: How to format the JSON output (COMPACT, PRETTY, LINES)
    - indent: Indentation level for PRETTY mode
    - schema: Optional JSON schema for validation
    - streaming: Whether to use streaming for large datasets
    - file_path: Path to output file (if not using direct response)
    - encoding: Character encoding for output
    - ensure_ascii: Whether to escape non-ASCII characters
    """
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize a new JSON output stage.
        
        Args:
            name: Name of the stage, defaults to class name if not provided
            config: Configuration parameters for this stage
        """
        super().__init__(name, config)
        
        # Configure formatting options
        formatting_mode = self.config.get("formatting_mode", "PRETTY")
        try:
            self.formatting_mode = JSONFormattingMode[formatting_mode] if isinstance(formatting_mode, str) else formatting_mode
        except (KeyError, TypeError):
            self.formatting_mode = JSONFormattingMode.PRETTY
            self.logger.warning(f"Invalid JSON formatting mode: {formatting_mode}, using PRETTY")
        
        # JSON-specific configuration
        self.indent = self.config.get("indent", 2)
        self.ensure_ascii = self.config.get("ensure_ascii", False)
        self.sort_keys = self.config.get("sort_keys", False)
        self.separators = (',', ':') if self.formatting_mode == JSONFormattingMode.COMPACT else None
        
        # Schema validation
        self.schema = self.config.get("schema")
        self.schema_validator = None
        
        # Streaming configuration
        self.streaming = self.config.get("streaming", False)
        self.streaming_batch_size = self.config.get("streaming_batch_size", 1000)
        
        # File output options
        self.file_path = self.config.get("file_path")
        self.encoding = self.config.get("encoding", "utf-8")
        
        # Set up logger
        self.logger = logging.getLogger(f"pipeline.stages.output.json.{self.name}")
        
        # Statistics for monitoring
        self._stats = {
            "items_processed": 0,
            "bytes_written": 0,
            "validation_errors": 0
        }
    
    async def validate_output_format(self, context: PipelineContext) -> bool:
        """
        Validate that the output data conforms to a JSON schema if one is provided.
        
        Args:
            context: The shared pipeline context
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        if not self.schema:
            return True
            
        # Load schema validator lazily on first use
        if self.schema_validator is None:
            try:
                import jsonschema
                self.schema_validator = jsonschema.Draft7Validator(self.schema)
            except ImportError:
                self.logger.warning("jsonschema package not available, schema validation disabled")
                return True
            except Exception as e:
                self.logger.error(f"Error initializing JSON schema validator: {str(e)}")
                context.add_error(self.name, f"Schema validation error: {str(e)}")
                return False
        
        # Get the data to validate
        data = self._prepare_output_data(context)
        if data is None:
            return False
            
        # Validate against schema
        try:
            errors = list(self.schema_validator.iter_errors(data))
            if errors:
                for error in errors:
                    error_path = '.'.join([str(p) for p in error.path]) if error.path else 'root'
                    error_msg = f"Validation error at {error_path}: {error.message}"
                    self.logger.warning(error_msg)
                    context.add_error(self.name, error_msg)
                    self._stats["validation_errors"] += 1
                
                # Fail if validation errors are not allowed
                if not self.config.get("allow_validation_errors", False):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error during JSON schema validation: {str(e)}")
            context.add_error(self.name, f"Schema validation error: {str(e)}")
            return False
    
    async def deliver_output(self, data: Any, context: PipelineContext) -> Optional[Dict[str, Any]]:
        """
        Format and deliver the JSON output.
        
        Args:
            data: The data to format as JSON
            context: The shared pipeline context
            
        Returns:
            Dict containing delivery results or None if delivery failed
        """
        try:
            # Check for output destination
            if self.file_path:
                return await self._write_to_file(data, context)
            else:
                # Direct response - format and return JSON data
                return await self._format_json(data, context)
                
        except Exception as e:
            self.logger.error(f"Error delivering JSON output: {str(e)}")
            context.add_error(self.name, f"JSON output error: {str(e)}")
            return None
    
    async def _format_json(self, data: Any, context: PipelineContext) -> Dict[str, Any]:
        """
        Format data as JSON according to configuration.
        
        Args:
            data: The data to format
            context: The shared pipeline context
            
        Returns:
            Dict containing formatted JSON data and metadata
        """
        if self.formatting_mode == JSONFormattingMode.PRETTY:
            formatted_json = json.dumps(
                data, 
                indent=self.indent,
                ensure_ascii=self.ensure_ascii,
                sort_keys=self.sort_keys
            )
        elif self.formatting_mode == JSONFormattingMode.COMPACT:
            formatted_json = json.dumps(
                data,
                separators=self.separators,
                ensure_ascii=self.ensure_ascii,
                sort_keys=self.sort_keys
            )
        elif self.formatting_mode == JSONFormattingMode.LINES:
            if isinstance(data, list):
                lines = [json.dumps(item, ensure_ascii=self.ensure_ascii) for item in data]
                formatted_json = '\n'.join(lines)
            else:
                formatted_json = json.dumps(
                    data,
                    ensure_ascii=self.ensure_ascii,
                    sort_keys=self.sort_keys
                )
        
        self._stats["items_processed"] += 1 if not isinstance(data, list) else len(data)
        self._stats["bytes_written"] += len(formatted_json)
        
        # Store formatted JSON in context for direct use
        context.set(f"{self.name}_json", formatted_json)
        
        return {
            "format": "json",
            "mode": self.formatting_mode.name,
            "size": len(formatted_json),
            "items": self._stats["items_processed"]
        }
    
    async def _write_to_file(self, data: Any, context: PipelineContext) -> Dict[str, Any]:
        """
        Write JSON data to a file.
        
        Args:
            data: The data to write
            context: The shared pipeline context
            
        Returns:
            Dict containing file output metadata
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(self.file_path)), exist_ok=True)
        
        # Determine how to write based on mode and streaming options
        if self.streaming and isinstance(data, list) and len(data) > self.streaming_batch_size:
            return await self._stream_to_file(data, context)
        
        # Use atomic write with temporary file
        temp_path = None
        try:
            # Create a temporary file in the same directory for atomic rename
            directory = os.path.dirname(os.path.abspath(self.file_path))
            fd, temp_path = tempfile.mkstemp(dir=directory, prefix='.tmp', suffix='.json')
            
            # Write data with appropriate formatting
            with os.fdopen(fd, 'w', encoding=self.encoding) as f:
                # Log Point 3.1: Log just before json.dump
                if hasattr(context, 'get') and context.get('url'):
                    url = context.get('url')
                    self.logger.info(f"JSON_SAVE_LOG_3.1 [{url}] About to write JSON to file: {self.file_path}")
                    self.logger.info(f"JSON_SAVE_LOG_3.1 [{url}] Data type: {type(data)}")
                    if isinstance(data, dict):
                        data_keys = list(data.keys())
                        self.logger.info(f"JSON_SAVE_LOG_3.1 [{url}] Data dict keys: {data_keys}")
                        if 'data' in data:
                            data_content = data.get('data', [])
                            self.logger.info(f"JSON_SAVE_LOG_3.1 [{url}] Data['data'] type: {type(data_content)}, length: {len(data_content) if hasattr(data_content, '__len__') else 'N/A'}")
                            if isinstance(data_content, list) and len(data_content) > 0:
                                self.logger.info(f"JSON_SAVE_LOG_3.1 [{url}] Sample data['data'][0]: {str(data_content[0])[:200]}")
                        # Log all top-level keys and their data types/lengths
                        for key, value in data.items():
                            if isinstance(value, list):
                                self.logger.info(f"JSON_SAVE_LOG_3.1 [{url}] Data['{key}'] list length: {len(value)}")
                            elif isinstance(value, dict):
                                self.logger.info(f"JSON_SAVE_LOG_3.1 [{url}] Data['{key}'] dict keys: {list(value.keys())}")
                            else:
                                self.logger.info(f"JSON_SAVE_LOG_3.1 [{url}] Data['{key}'] type: {type(value)}, value: {str(value)[:100]}")
                    elif isinstance(data, list):
                        self.logger.info(f"JSON_SAVE_LOG_3.1 [{url}] Data list length: {len(data)}")
                        if len(data) > 0:
                            self.logger.info(f"JSON_SAVE_LOG_3.1 [{url}] Sample data[0]: {str(data[0])[:200]}")
                    else:
                        self.logger.info(f"JSON_SAVE_LOG_3.1 [{url}] Data value: {str(data)[:200]}")
                
                if self.formatting_mode == JSONFormattingMode.PRETTY:
                    json.dump(
                        data, f,
                        indent=self.indent,
                        ensure_ascii=self.ensure_ascii,
                        sort_keys=self.sort_keys
                    )
                elif self.formatting_mode == JSONFormattingMode.COMPACT:
                    json.dump(
                        data, f,
                        separators=self.separators,
                        ensure_ascii=self.ensure_ascii,
                        sort_keys=self.sort_keys
                    )
                elif self.formatting_mode == JSONFormattingMode.LINES:
                    if isinstance(data, list):
                        for item in data:
                            f.write(json.dumps(item, ensure_ascii=self.ensure_ascii))
                            f.write('\n')
                    else:
                        json.dump(
                            data, f,
                            ensure_ascii=self.ensure_ascii,
                            sort_keys=self.sort_keys
                        )
            
            # Atomically rename the temporary file to the target file
            os.replace(temp_path, self.file_path)
            temp_path = None  # Prevent deletion in finally block
            
            # Update statistics
            self._stats["items_processed"] += 1 if not isinstance(data, list) else len(data)
            file_size = os.path.getsize(self.file_path)
            self._stats["bytes_written"] += file_size
            
            return {
                "file_path": self.file_path,
                "format": "json",
                "mode": self.formatting_mode.name,
                "size": file_size,
                "items": self._stats["items_processed"]
            }
            
        except Exception as e:
            raise Exception(f"Error writing JSON to file: {str(e)}")
            
        finally:
            # Clean up temporary file if still exists
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
    
    async def _stream_to_file(self, data: List[Any], context: PipelineContext) -> Dict[str, Any]:
        """
        Stream large datasets to a JSON file in chunks.
        
        Args:
            data: The list of data items to stream
            context: The shared pipeline context
            
        Returns:
            Dict containing file output metadata
        """
        # Determine if we're writing a JSON array or line-delimited JSON
        is_array_output = self.formatting_mode != JSONFormattingMode.LINES
        
        # Use atomic write with temporary file
        temp_path = None
        try:
            # Create a temporary file in the same directory for atomic rename
            directory = os.path.dirname(os.path.abspath(self.file_path))
            fd, temp_path = tempfile.mkstemp(dir=directory, prefix='.tmp', suffix='.json')
            
            with os.fdopen(fd, 'w', encoding=self.encoding) as f:
                # Log Point 3.1: Log for streaming method before json.dump
                if hasattr(context, 'get') and context.get('url'):
                    url = context.get('url')
                    self.logger.info(f"JSON_SAVE_LOG_3.1_STREAM [{url}] About to stream write JSON to file: {self.file_path}")
                    self.logger.info(f"JSON_SAVE_LOG_3.1_STREAM [{url}] Data type: {type(data)}, length: {len(data) if hasattr(data, '__len__') else 'N/A'}")
                    if isinstance(data, list) and len(data) > 0:
                        self.logger.info(f"JSON_SAVE_LOG_3.1_STREAM [{url}] Sample data[0]: {str(data[0])[:200]}")
                        # Check if list items have data field
                        if isinstance(data[0], dict) and 'data' in data[0]:
                            self.logger.info(f"JSON_SAVE_LOG_3.1_STREAM [{url}] First item has 'data' key with value: {str(data[0]['data'])[:200]}")
                
                # Start array if needed
                if is_array_output:
                    f.write('[')
                
                # Process items in batches
                total_items = len(data)
                for i, item in enumerate(data):
                    if is_array_output and i > 0:
                        f.write(',')
                        
                        # Add newline and indentation for PRETTY mode
                        if self.formatting_mode == JSONFormattingMode.PRETTY:
                            f.write('\n' + ' ' * self.indent)
                    
                    # Write the item
                    if is_array_output:
                        json.dump(
                            item, f,
                            indent=self.indent if self.formatting_mode == JSONFormattingMode.PRETTY else None,
                            ensure_ascii=self.ensure_ascii,
                            sort_keys=self.sort_keys
                        )
                    else:
                        # Line-delimited JSON
                        f.write(json.dumps(
                            item,
                            ensure_ascii=self.ensure_ascii,
                            sort_keys=self.sort_keys
                        ))
                        f.write('\n')
                    
                    # Yield to other tasks occasionally
                    if i % self.streaming_batch_size == 0:
                        await asyncio.sleep(0)
                
                # End array if needed
                if is_array_output:
                    f.write(']')
            
            # Atomically rename the temporary file to the target file
            os.replace(temp_path, self.file_path)
            temp_path = None  # Prevent deletion in finally block
            
            # Update statistics
            self._stats["items_processed"] += total_items
            file_size = os.path.getsize(self.file_path)
            self._stats["bytes_written"] += file_size
            
            return {
                "file_path": self.file_path,
                "format": "json",
                "mode": self.formatting_mode.name,
                "size": file_size,
                "items": total_items,
                "streaming": True
            }
            
        except Exception as e:
            raise Exception(f"Error streaming JSON to file: {str(e)}")
            
        finally:
            # Clean up temporary file if still exists
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for configuration.
        
        Returns:
            Dict containing JSON schema for this stage's configuration
        """
        schema = super().get_config_schema()
        
        # Add JSON-specific properties
        json_properties = {
            "formatting_mode": {
                "type": "string",
                "enum": ["COMPACT", "PRETTY", "LINES"]
            },
            "indent": {
                "type": "integer",
                "minimum": 0,
                "maximum": 8
            },
            "ensure_ascii": {"type": "boolean"},
            "sort_keys": {"type": "boolean"},
            "schema": {"type": "object"},
            "streaming": {"type": "boolean"},
            "streaming_batch_size": {"type": "integer", "minimum": 10},
            "file_path": {"type": "string"},
            "encoding": {"type": "string"},
            "allow_validation_errors": {"type": "boolean"}
        }
        
        # Update the properties in the schema
        schema["properties"].update(json_properties)
        
        return schema