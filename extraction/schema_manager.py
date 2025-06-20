"""
Schema Management Module

This module provides comprehensive schema management capabilities including:
- Schema registration and retrieval
- Schema generation from sample data
- Schema merging and extending
- Schema format conversion
- Schema utilities for working with fields and relationships
"""

import os
import re
import json
import copy
import logging
import datetime
from typing import Dict, Any, List, Optional, Union, Set, Tuple, Callable

from core.service_interface import BaseService

logger = logging.getLogger(__name__)

class SchemaManager(BaseService):
    """
    Manages schemas for data extraction and validation.
    
    This class provides methods to register, retrieve, generate, 
    merge, and extend schemas, as well as utilities for working
    with schema fields and converting between schema formats.
    """
    
    def __init__(self):
        """Initialize the schema manager."""
        super().__init__()
        self._schemas = {}
        self._schema_paths = []
        
    @property
    def name(self) -> str:
        """Get the service name."""
        return "schema_manager"
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the schema manager with configuration.
        
        Args:
            config: Configuration dictionary with schema directories and settings
        """
        if self._initialized:
            return
            
        super().initialize(config)
        
        # Extract configuration
        self._config = config or {}
        
        # Load schema directories from config
        schema_dirs = self._config.get("schema_directories", [])
        if isinstance(schema_dirs, str):
            schema_dirs = [schema_dirs]
        
        # Add default schema directory if it exists
        default_schema_dir = os.path.join(os.path.dirname(__file__), "../schemas")
        if os.path.exists(default_schema_dir) and default_schema_dir not in schema_dirs:
            schema_dirs.append(default_schema_dir)
        
        # Load schemas from all directories
        for schema_dir in schema_dirs:
            if os.path.exists(schema_dir):
                self._schema_paths.append(schema_dir)
                self._load_schemas_from_directory(schema_dir)
        
        # Load inline schemas from config
        inline_schemas = self._config.get("schemas", {})
        for schema_name, schema in inline_schemas.items():
            self.register_schema(schema_name, schema)
            
        logger.info(f"Schema manager initialized with {len(self._schemas)} schemas")
    
    def shutdown(self) -> None:
        """Clean up resources."""
        self._schemas = {}
        self._schema_paths = []
        super().shutdown()
        logger.debug("Schema manager shut down")
    
    def register_schema(self, name: str, schema: Dict[str, Any]) -> None:
        """
        Add a schema to the registry.
        
        Args:
            name: Name to register the schema under
            schema: Schema definition
        """
        # Add metadata if not present
        if "_metadata" not in schema:
            schema["_metadata"] = {}
        
        # Update metadata
        schema["_metadata"].update({
            "name": name,
            "registered_at": datetime.datetime.now().isoformat()
        })
        
        # Store schema
        self._schemas[name] = schema
        logger.debug(f"Registered schema: {name}")
    
    def get_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a schema by name.
        
        Args:
            name: Name of the schema to retrieve
            
        Returns:
            Schema definition or None if not found
        """
        # Check if schema is in memory
        if name in self._schemas:
            return copy.deepcopy(self._schemas[name])
        
        # Try to load from disk
        for path in self._schema_paths:
            schema_path = os.path.join(path, f"{name}.json")
            if os.path.exists(schema_path):
                try:
                    with open(schema_path, 'r') as f:
                        schema = json.load(f)
                    self.register_schema(name, schema)
                    return copy.deepcopy(schema)
                except Exception as e:
                    logger.error(f"Error loading schema {name}: {str(e)}")
        
        logger.warning(f"Schema not found: {name}")
        return None
    
    def has_schema(self, name: str) -> bool:
        """
        Check if a schema exists in the registry.
        
        Args:
            name: Name of the schema to check
            
        Returns:
            True if the schema exists, False otherwise
        """
        # Check in-memory schemas
        if name in self._schemas:
            return True
            
        # Check disk schemas
        for path in self._schema_paths:
            schema_path = os.path.join(path, f"{name}.json")
            if os.path.exists(schema_path):
                return True
                
        return False
    
    def remove_schema(self, name: str) -> bool:
        """
        Remove a schema from the registry.
        
        Args:
            name: Name of the schema to remove
            
        Returns:
            True if schema was removed, False if it didn't exist
        """
        if name in self._schemas:
            del self._schemas[name]
            logger.debug(f"Removed schema: {name}")
            return True
        return False
    
    def generate_schema_from_sample(self, data: Dict[str, Any], name: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a schema from a data sample.
        
        Args:
            data: Sample data to generate schema from
            name: Optional name to register the schema under
            
        Returns:
            Generated schema
        """
        schema = self._infer_schema(data)
        
        # Add metadata
        if "_metadata" not in schema:
            schema["_metadata"] = {}
            
        schema["_metadata"].update({
            "generated": True,
            "generated_at": datetime.datetime.now().isoformat(),
            "source": "sample_data"
        })
        
        # Register if name provided
        if name:
            self.register_schema(name, schema)
            
        return schema
    
    def merge_schemas(self, schema1: Dict[str, Any], schema2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine two schemas into one.
        
        Args:
            schema1: First schema to merge
            schema2: Second schema to merge
            
        Returns:
            Merged schema
        """
        merged = copy.deepcopy(schema1)
        
        # Merge metadata
        if "_metadata" not in merged:
            merged["_metadata"] = {}
            
        if "_metadata" in schema2:
            merged["_metadata"]["merged_from"] = [
                merged["_metadata"].get("name", "unnamed_schema1"),
                schema2["_metadata"].get("name", "unnamed_schema2")
            ]
        
        # Merge fields
        for field_name, field_schema in schema2.items():
            if field_name.startswith("_"):
                # Skip metadata fields
                continue
                
            if field_name not in merged:
                # Field only in schema2, copy it
                merged[field_name] = copy.deepcopy(field_schema)
            else:
                # Field in both schemas, merge properties
                if isinstance(merged[field_name], dict) and isinstance(field_schema, dict):
                    # Merge field properties, schema1 takes precedence for conflicts
                    for prop, value in field_schema.items():
                        if prop not in merged[field_name]:
                            merged[field_name][prop] = copy.deepcopy(value)
                        elif prop == "type" and merged[field_name]["type"] != value:
                            # Type conflict, use union type if possible
                            if isinstance(merged[field_name]["type"], list):
                                if value not in merged[field_name]["type"]:
                                    merged[field_name]["type"].append(value)
                            else:
                                merged[field_name]["type"] = [merged[field_name]["type"], value]
                        elif prop == "enum" and "enum" in merged[field_name]:
                            # Merge enum values
                            merged[field_name]["enum"] = list(set(merged[field_name]["enum"] + value))
        
        # Update metadata
        merged["_metadata"]["merged_at"] = datetime.datetime.now().isoformat()
        
        return merged
    
    def extend_schema(self, base_schema: Dict[str, Any], extension: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extend a base schema with additional fields and properties.
        
        Args:
            base_schema: Base schema to extend
            extension: Extension schema with additional fields/properties
            
        Returns:
            Extended schema
        """
        extended = copy.deepcopy(base_schema)
        
        # Update metadata
        if "_metadata" not in extended:
            extended["_metadata"] = {}
            
        if "_metadata" in extension:
            extended["_metadata"]["extended_from"] = base_schema["_metadata"].get("name", "unnamed_base")
            extended["_metadata"]["extended_with"] = extension["_metadata"].get("name", "unnamed_extension")
        
        # Add/override fields from extension
        for field_name, field_schema in extension.items():
            if field_name.startswith("_"):
                # Skip metadata fields
                continue
                
            # Override or add field
            extended[field_name] = copy.deepcopy(field_schema)
        
        # Update metadata
        extended["_metadata"]["extended_at"] = datetime.datetime.now().isoformat()
        
        return extended
    
    def convert_schema_format(self, schema: Dict[str, Any], target_format: str) -> Dict[str, Any]:
        """
        Convert schema to a different format.
        
        Args:
            schema: Schema to convert
            target_format: Target format (json_schema, avro, xml_schema)
            
        Returns:
            Converted schema
        """
        if target_format.lower() == "json_schema":
            return self.convert_to_json_schema(schema)
        elif target_format.lower() == "avro":
            return self.convert_to_avro_schema(schema)
        elif target_format.lower() == "xml_schema":
            return self.convert_to_xml_schema(schema)
        else:
            logger.warning(f"Unsupported schema format: {target_format}")
            return schema
    
    def get_field_definition(self, schema: Dict[str, Any], field_path: str) -> Optional[Dict[str, Any]]:
        """
        Get field details from a schema by path.
        
        Args:
            schema: Schema to search
            field_path: Path to the field (e.g., "user.address.city")
            
        Returns:
            Field definition or None if not found
        """
        if not field_path:
            return None
            
        # Split path into components
        parts = field_path.split(".")
        current_schema = schema
        
        for i, part in enumerate(parts):
            # Handle array indices in path (e.g., items[0].name)
            array_match = re.match(r"(.+)\[(\d+)\]", part)
            if array_match:
                array_name = array_match.group(1)
                if array_name not in current_schema:
                    return None
                    
                array_schema = current_schema[array_name]
                if not isinstance(array_schema, dict) or "type" not in array_schema or array_schema["type"] != "array":
                    return None
                    
                # Get item schema
                if "items" not in array_schema:
                    return None
                    
                current_schema = array_schema["items"]
            else:
                # Regular field
                if part not in current_schema:
                    return None
                    
                if i == len(parts) - 1:
                    # Last part, return field definition
                    return current_schema[part]
                    
                # Navigate to nested schema
                if isinstance(current_schema[part], dict) and "type" in current_schema[part]:
                    if current_schema[part]["type"] == "object" and "properties" in current_schema[part]:
                        current_schema = current_schema[part]["properties"]
                    else:
                        # Can't navigate further
                        return None
                else:
                    # Not a navigable field
                    return None
        
        return None
    
    def get_required_fields(self, schema: Dict[str, Any]) -> List[str]:
        """
        List required fields from a schema.
        
        Args:
            schema: Schema to analyze
            
        Returns:
            List of required field paths
        """
        required_fields = []
        
        def collect_required_fields(schema_part, prefix=""):
            for field_name, field_schema in schema_part.items():
                if field_name.startswith("_"):
                    continue  # Skip metadata fields
                    
                field_path = f"{prefix}.{field_name}" if prefix else field_name
                
                if isinstance(field_schema, dict) and field_schema.get("required", False):
                    required_fields.append(field_path)
                
                # Check nested objects
                if isinstance(field_schema, dict) and field_schema.get("type") == "object" and "properties" in field_schema:
                    collect_required_fields(field_schema["properties"], field_path)
                
                # Check array of objects
                if isinstance(field_schema, dict) and field_schema.get("type") == "array" and "items" in field_schema:
                    items_schema = field_schema["items"]
                    if isinstance(items_schema, dict) and items_schema.get("type") == "object" and "properties" in items_schema:
                        collect_required_fields(items_schema["properties"], f"{field_path}[]")
        
        collect_required_fields(schema)
        return required_fields
    
    def map_fields(self, source_schema: Dict[str, Any], target_schema: Dict[str, Any]) -> Dict[str, str]:
        """
        Map fields between source and target schemas.
        
        Args:
            source_schema: Source schema
            target_schema: Target schema
            
        Returns:
            Mapping from source fields to target fields
        """
        mapping = {}
        
        # Get all fields from both schemas
        source_fields = set(self._get_all_fields(source_schema))
        target_fields = set(self._get_all_fields(target_schema))
        
        # Direct name matches
        for source_field in source_fields:
            field_name = source_field.split(".")[-1]
            
            # Exact matches
            if source_field in target_fields:
                mapping[source_field] = source_field
                continue
                
            # Check for case-insensitive matches
            source_lower = source_field.lower()
            for target_field in target_fields:
                if target_field.lower() == source_lower:
                    mapping[source_field] = target_field
                    break
            
            # Check for partial field name matches
            if source_field not in mapping:
                for target_field in target_fields:
                    target_name = target_field.split(".")[-1]
                    if field_name.lower() == target_name.lower():
                        mapping[source_field] = target_field
                        break
        
        # Check for semantic matches using field descriptions
        for source_field in source_fields:
            if source_field in mapping:
                continue
                
            source_def = self.get_field_definition(source_schema, source_field)
            if not source_def or not isinstance(source_def, dict) or "description" not in source_def:
                continue
                
            source_desc = source_def["description"].lower()
            
            for target_field in target_fields:
                target_def = self.get_field_definition(target_schema, target_field)
                if not target_def or not isinstance(target_def, dict) or "description" not in target_def:
                    continue
                    
                target_desc = target_def["description"].lower()
                
                # Check for significant description overlap
                if (source_desc in target_desc or target_desc in source_desc or
                    self._description_similarity(source_desc, target_desc) > 0.7):
                    mapping[source_field] = target_field
                    break
        
        return mapping
    
    def flatten_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert nested schema to flat schema with dot notation.
        
        Args:
            schema: Nested schema to flatten
            
        Returns:
            Flattened schema
        """
        flat_schema = {}
        
        def flatten_field(field_schema, prefix=""):
            for field_name, field_def in field_schema.items():
                if field_name.startswith("_"):
                    continue  # Skip metadata fields
                    
                field_path = f"{prefix}.{field_name}" if prefix else field_name
                
                if isinstance(field_def, dict) and field_def.get("type") == "object" and "properties" in field_def:
                    # Flatten nested object
                    flatten_field(field_def["properties"], field_path)
                elif isinstance(field_def, dict) and field_def.get("type") == "array" and "items" in field_def:
                    items_schema = field_def["items"]
                    if isinstance(items_schema, dict) and items_schema.get("type") == "object" and "properties" in items_schema:
                        # Flatten array of objects
                        flatten_field(items_schema["properties"], f"{field_path}[]")
                    else:
                        # Array of primitives
                        flat_schema[field_path] = {
                            "type": "array",
                            "items": items_schema.copy()
                        }
                        if "description" in field_def:
                            flat_schema[field_path]["description"] = field_def["description"]
                        if "required" in field_def:
                            flat_schema[field_path]["required"] = field_def["required"]
                else:
                    # Regular field
                    flat_schema[field_path] = field_def.copy()
        
        flatten_field(schema)
        
        # Add metadata
        if "_metadata" in schema:
            flat_schema["_metadata"] = schema["_metadata"].copy()
            flat_schema["_metadata"]["flattened"] = True
            
        return flat_schema
    
    def get_field_types(self, schema: Dict[str, Any]) -> Dict[str, str]:
        """
        Get field types from a schema.
        
        Args:
            schema: Schema to analyze
            
        Returns:
            Dictionary mapping field paths to types
        """
        field_types = {}
        
        def collect_types(schema_part, prefix=""):
            for field_name, field_schema in schema_part.items():
                if field_name.startswith("_"):
                    continue  # Skip metadata fields
                    
                field_path = f"{prefix}.{field_name}" if prefix else field_name
                
                if isinstance(field_schema, dict) and "type" in field_schema:
                    field_types[field_path] = field_schema["type"]
                    
                    # Handle format and nested types
                    if field_schema["type"] == "string" and "format" in field_schema:
                        field_types[field_path] = field_schema["format"]
                    elif field_schema["type"] == "object" and "properties" in field_schema:
                        collect_types(field_schema["properties"], field_path)
                    elif field_schema["type"] == "array" and "items" in field_schema:
                        items_schema = field_schema["items"]
                        if isinstance(items_schema, dict) and "type" in items_schema:
                            field_types[field_path] = f"array<{items_schema['type']}>"
                            
                            if items_schema["type"] == "object" and "properties" in items_schema:
                                collect_types(items_schema["properties"], f"{field_path}[]")
        
        collect_types(schema)
        return field_types
    
    def add_field_validator(self, schema: Dict[str, Any], field: str, validator: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add custom validation rule to a field.
        
        Args:
            schema: Schema to modify
            field: Field path to add validator to
            validator: Validator configuration
            
        Returns:
            Updated schema
        """
        updated_schema = copy.deepcopy(schema)
        
        # Parse field path
        parts = field.split(".")
        current = updated_schema
        
        # Navigate to the field
        for i, part in enumerate(parts):
            # Handle array notation
            array_match = re.match(r"(.+)\[(\d*)\]", part)
            if array_match:
                array_name = array_match.group(1)
                
                if array_name not in current:
                    logger.warning(f"Field {array_name} not found in schema")
                    return schema
                    
                if i == len(parts) - 1:
                    # Last part, add validator to array field
                    if "validators" not in current[array_name]:
                        current[array_name]["validators"] = []
                    current[array_name]["validators"].append(validator)
                else:
                    # Navigate to array items
                    if "items" not in current[array_name]:
                        logger.warning(f"Array items not defined for {array_name}")
                        return schema
                        
                    current = current[array_name]["items"]
                    
                    if "type" in current and current["type"] == "object" and "properties" in current:
                        current = current["properties"]
                    else:
                        logger.warning(f"Cannot navigate to {field}")
                        return schema
            else:
                # Regular field navigation
                if part not in current:
                    logger.warning(f"Field {part} not found in schema")
                    return schema
                    
                if i == len(parts) - 1:
                    # Last part, add validator to field
                    if "validators" not in current[part]:
                        current[part]["validators"] = []
                    current[part]["validators"].append(validator)
                else:
                    # Navigate to nested object
                    if "type" in current[part] and current[part]["type"] == "object" and "properties" in current[part]:
                        current = current[part]["properties"]
                    else:
                        logger.warning(f"Cannot navigate to {field}")
                        return schema
        
        return updated_schema
    
    def get_schema_differences(self, schema1: Dict[str, Any], schema2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare two schemas and identify differences.
        
        Args:
            schema1: First schema to compare
            schema2: Second schema to compare
            
        Returns:
            Dictionary with differences
        """
        differences = {
            "fields_only_in_schema1": [],
            "fields_only_in_schema2": [],
            "type_differences": [],
            "constraint_differences": []
        }
        
        # Get all fields from both schemas
        fields1 = set(self._get_all_fields(schema1))
        fields2 = set(self._get_all_fields(schema2))
        
        # Fields only in one schema
        differences["fields_only_in_schema1"] = list(fields1 - fields2)
        differences["fields_only_in_schema2"] = list(fields2 - fields1)
        
        # Compare common fields
        common_fields = fields1.intersection(fields2)
        for field in common_fields:
            field_def1 = self.get_field_definition(schema1, field)
            field_def2 = self.get_field_definition(schema2, field)
            
            if not field_def1 or not field_def2:
                continue
                
            # Compare types
            type1 = field_def1.get("type")
            type2 = field_def2.get("type")
            
            if type1 != type2:
                differences["type_differences"].append({
                    "field": field,
                    "schema1_type": type1,
                    "schema2_type": type2
                })
            
            # Compare constraints
            constraints = ["minimum", "maximum", "minLength", "maxLength", "pattern", "format", "enum", "required"]
            for constraint in constraints:
                if constraint in field_def1 or constraint in field_def2:
                    value1 = field_def1.get(constraint)
                    value2 = field_def2.get(constraint)
                    
                    if value1 != value2:
                        differences["constraint_differences"].append({
                            "field": field,
                            "constraint": constraint,
                            "schema1_value": value1,
                            "schema2_value": value2
                        })
        
        return differences
    
    def convert_to_json_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert internal schema to JSON Schema format.
        
        Args:
            schema: Internal schema to convert
            
        Returns:
            JSON Schema representation
        """
        json_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {},
            "required": []
        }
        
        # Add title and description if available
        if "_metadata" in schema and "name" in schema["_metadata"]:
            json_schema["title"] = schema["_metadata"]["name"]
            
        if "_metadata" in schema and "description" in schema["_metadata"]:
            json_schema["description"] = schema["_metadata"]["description"]
        
        # Convert fields
        for field_name, field_schema in schema.items():
            if field_name.startswith("_"):
                continue  # Skip metadata fields
                
            if isinstance(field_schema, dict):
                # Convert field to JSON Schema format
                json_schema["properties"][field_name] = self._convert_field_to_json_schema(field_schema)
                
                # Add to required list if needed
                if field_schema.get("required", False):
                    json_schema["required"].append(field_name)
        
        # Remove empty required array
        if not json_schema["required"]:
            del json_schema["required"]
            
        return json_schema
    
    def convert_from_json_schema(self, json_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert JSON Schema to internal format.
        
        Args:
            json_schema: JSON Schema to convert
            
        Returns:
            Internal schema representation
        """
        internal_schema = {}
        
        # Add metadata
        internal_schema["_metadata"] = {
            "source": "json_schema",
            "converted_at": datetime.datetime.now().isoformat()
        }
        
        # Add title and description to metadata
        if "title" in json_schema:
            internal_schema["_metadata"]["name"] = json_schema["title"]
            
        if "description" in json_schema:
            internal_schema["_metadata"]["description"] = json_schema["description"]
        
        # Get required fields
        required_fields = json_schema.get("required", [])
        
        # Convert properties
        if "properties" in json_schema:
            for field_name, field_schema in json_schema["properties"].items():
                internal_schema[field_name] = self._convert_json_schema_field(field_schema)
                
                # Set required flag
                if field_name in required_fields:
                    internal_schema[field_name]["required"] = True
        
        return internal_schema
    
    def convert_to_avro_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert internal schema to Avro schema format.
        
        Args:
            schema: Internal schema to convert
            
        Returns:
            Avro schema representation
        """
        # Start with basic Avro record
        avro_schema = {
            "type": "record",
            "name": schema.get("_metadata", {}).get("name", "AnonymousRecord"),
            "fields": []
        }
        
        # Add description if available
        if "_metadata" in schema and "description" in schema["_metadata"]:
            avro_schema["doc"] = schema["_metadata"]["description"]
        
        # Convert fields
        for field_name, field_schema in schema.items():
            if field_name.startswith("_"):
                continue  # Skip metadata fields
                
            if isinstance(field_schema, dict):
                avro_field = {
                    "name": field_name,
                    "type": self._convert_field_to_avro_type(field_schema)
                }
                
                # Add description if available
                if "description" in field_schema:
                    avro_field["doc"] = field_schema["description"]
                    
                # Add default if available
                if "default" in field_schema:
                    avro_field["default"] = field_schema["default"]
                elif not field_schema.get("required", False):
                    # Make nullable if not required
                    avro_field["type"] = ["null", avro_field["type"]]
                    avro_field["default"] = None
                
                avro_schema["fields"].append(avro_field)
        
        return avro_schema
    
    def convert_to_xml_schema(self, schema: Dict[str, Any]) -> str:
        """
        Convert internal schema to XML Schema (XSD) format.
        
        Args:
            schema: Internal schema to convert
            
        Returns:
            XML Schema representation as string
        """
        # Create XML Schema header
        xsd = '<?xml version="1.0" encoding="UTF-8"?>\n'
        xsd += '<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">\n'
        
        # Add root element
        schema_name = schema.get("_metadata", {}).get("name", "AnonymousType")
        xsd += f'  <xs:element name="{schema_name}" type="{schema_name}Type"/>\n\n'
        
        # Define the complex type
        xsd += f'  <xs:complexType name="{schema_name}Type">\n'
        xsd += '    <xs:sequence>\n'
        
        # Add fields
        for field_name, field_schema in schema.items():
            if field_name.startswith("_"):
                continue  # Skip metadata fields
                
            if isinstance(field_schema, dict):
                field_type = field_schema.get("type", "string")
                xsd_type = self._get_xsd_type(field_type)
                
                # Add field element
                min_occurs = "0" if not field_schema.get("required", False) else "1"
                max_occurs = "unbounded" if field_type == "array" else "1"
                
                xsd += f'      <xs:element name="{field_name}" type="{xsd_type}" minOccurs="{min_occurs}" maxOccurs="{max_occurs}"'
                
                # Add documentation if available
                if "description" in field_schema:
                    xsd += '>\n'
                    xsd += f'        <xs:annotation>\n'
                    xsd += f'          <xs:documentation>{field_schema["description"]}</xs:documentation>\n'
                    xsd += f'        </xs:annotation>\n'
                    xsd += f'      </xs:element>\n'
                else:
                    xsd += '/>\n'
        
        # Close the schema
        xsd += '    </xs:sequence>\n'
        xsd += '  </xs:complexType>\n'
        xsd += '</xs:schema>'
        
        return xsd
    
    def _load_schemas_from_directory(self, directory: str) -> None:
        """
        Load all JSON schemas from a directory.
        
        Args:
            directory: Directory path to load schemas from
        """
        try:
            if not os.path.exists(directory):
                logger.warning(f"Schema directory does not exist: {directory}")
                return
                
            for filename in os.listdir(directory):
                if filename.endswith(".json"):
                    schema_name = os.path.splitext(filename)[0]
                    schema_path = os.path.join(directory, filename)
                    
                    try:
                        with open(schema_path, 'r') as f:
                            schema = json.load(f)
                        
                        self.register_schema(schema_name, schema)
                    except Exception as e:
                        logger.error(f"Error loading schema {schema_name}: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading schemas from directory {directory}: {str(e)}")
    
    def _infer_schema(self, data: Any, path: str = "") -> Dict[str, Any]:
        """
        Recursively infer schema from data.
        
        Args:
            data: Data to infer schema from
            path: Current path in data structure
            
        Returns:
            Inferred schema
        """
        if data is None:
            return {"type": "null"}
        
        if isinstance(data, bool):
            return {"type": "boolean"}
        
        if isinstance(data, int):
            return {"type": "integer"}
        
        if isinstance(data, float):
            return {"type": "number"}
        
        if isinstance(data, str):
            # Try to detect format
            schema = {"type": "string"}
            
            # Check for common formats
            if self._is_date(data):
                schema["format"] = "date"
            elif self._is_datetime(data):
                schema["format"] = "date-time"
            elif self._is_email(data):
                schema["format"] = "email"
            elif self._is_url(data):
                schema["format"] = "uri"
            
            return schema
        
        if isinstance(data, list):
            if not data:
                return {"type": "array", "items": {}}
            
            # Infer schema from first item
            item_schema = self._infer_schema(data[0], f"{path}[0]")
            
            # Check if all items have the same schema
            uniform = True
            for i, item in enumerate(data[1:], 1):
                if i < 10:  # Only check first few items for performance
                    item_type = self._infer_schema(item, f"{path}[{i}]")
                    if item_type != item_schema:
                        uniform = False
                        break
            
            if uniform:
                return {"type": "array", "items": item_schema}
            else:
                return {"type": "array"}
        
        if isinstance(data, dict):
            properties = {}
            required = []
            
            for key, value in data.items():
                if key.startswith("_"):
                    continue  # Skip metadata fields
                
                properties[key] = self._infer_schema(value, f"{path}.{key}")
                
                # Assume non-null fields are required
                if value is not None:
                    required.append(key)
            
            schema = {
                "type": "object",
                "properties": properties
            }
            
            if required:
                schema["required"] = required
            
            return schema
        
        # Fallback
        return {"type": "string"}
    
    def _get_all_fields(self, schema: Dict[str, Any]) -> List[str]:
        """
        Get all field paths from a schema.
        
        Args:
            schema: Schema to analyze
            
        Returns:
            List of field paths
        """
        fields = []
        
        def collect_fields(schema_part, prefix=""):
            for field_name, field_schema in schema_part.items():
                if field_name.startswith("_"):
                    continue  # Skip metadata fields
                    
                field_path = f"{prefix}.{field_name}" if prefix else field_name
                fields.append(field_path)
                
                # Check nested objects
                if isinstance(field_schema, dict) and field_schema.get("type") == "object" and "properties" in field_schema:
                    collect_fields(field_schema["properties"], field_path)
                
                # Check array of objects
                if isinstance(field_schema, dict) and field_schema.get("type") == "array" and "items" in field_schema:
                    items_schema = field_schema["items"]
                    if isinstance(items_schema, dict) and items_schema.get("type") == "object" and "properties" in items_schema:
                        collect_fields(items_schema["properties"], f"{field_path}[]")
        
        collect_fields(schema)
        return fields
    
    def _description_similarity(self, desc1: str, desc2: str) -> float:
        """
        Calculate similarity between two field descriptions.
        
        Args:
            desc1: First description
            desc2: Second description
            
        Returns:
            Similarity score (0-1)
        """
        # Simple word overlap for now
        words1 = set(re.findall(r"\w+", desc1.lower()))
        words2 = set(re.findall(r"\w+", desc2.lower()))
        
        if not words1 or not words2:
            return 0
            
        # Calculate Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _convert_field_to_json_schema(self, field: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert internal field schema to JSON Schema format.
        
        Args:
            field: Internal field schema
            
        Returns:
            JSON Schema field definition
        """
        json_field = {}
        
        # Copy basic properties
        if "type" in field:
            json_field["type"] = field["type"]
            
        if "description" in field:
            json_field["description"] = field["description"]
            
        if "default" in field:
            json_field["default"] = field["default"]
            
        # Copy constraints
        constraints = ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", 
                      "minLength", "maxLength", "pattern", "format", "enum", "multipleOf"]
                      
        for constraint in constraints:
            if constraint in field:
                json_field[constraint] = field[constraint]
        
        # Handle nested types
        if field.get("type") == "object" and "properties" in field:
            json_field["type"] = "object"
            json_field["properties"] = {}
            json_field["required"] = []
            
            for prop_name, prop_schema in field["properties"].items():
                json_field["properties"][prop_name] = self._convert_field_to_json_schema(prop_schema)
                
                if prop_schema.get("required", False):
                    json_field["required"].append(prop_name)
            
            # Remove empty required array
            if not json_field["required"]:
                del json_field["required"]
                
        elif field.get("type") == "array" and "items" in field:
            json_field["type"] = "array"
            json_field["items"] = self._convert_field_to_json_schema(field["items"])
            
            # Add array constraints
            if "minItems" in field:
                json_field["minItems"] = field["minItems"]
                
            if "maxItems" in field:
                json_field["maxItems"] = field["maxItems"]
                
            if "uniqueItems" in field:
                json_field["uniqueItems"] = field["uniqueItems"]
        
        return json_field
    
    def _convert_json_schema_field(self, field: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert JSON Schema field to internal format.
        
        Args:
            field: JSON Schema field
            
        Returns:
            Internal field schema
        """
        internal_field = {}
        
        # Copy basic properties
        if "type" in field:
            internal_field["type"] = field["type"]
            
        if "description" in field:
            internal_field["description"] = field["description"]
            
        if "default" in field:
            internal_field["default"] = field["default"]
            
        # Copy constraints
        constraints = ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", 
                      "minLength", "maxLength", "pattern", "format", "enum", "multipleOf"]
                      
        for constraint in constraints:
            if constraint in field:
                internal_field[constraint] = field[constraint]
        
        # Handle nested types
        if field.get("type") == "object" and "properties" in field:
            internal_field["type"] = "object"
            internal_field["properties"] = {}
            
            required_props = field.get("required", [])
            
            for prop_name, prop_schema in field["properties"].items():
                internal_field["properties"][prop_name] = self._convert_json_schema_field(prop_schema)
                
                if prop_name in required_props:
                    internal_field["properties"][prop_name]["required"] = True
                
        elif field.get("type") == "array" and "items" in field:
            internal_field["type"] = "array"
            internal_field["items"] = self._convert_json_schema_field(field["items"])
            
            # Add array constraints
            if "minItems" in field:
                internal_field["minItems"] = field["minItems"]
                
            if "maxItems" in field:
                internal_field["maxItems"] = field["maxItems"]
                
            if "uniqueItems" in field:
                internal_field["uniqueItems"] = field["uniqueItems"]
        
        return internal_field
    
    def _convert_field_to_avro_type(self, field: Dict[str, Any]) -> Union[str, Dict[str, Any]]:
        """
        Convert internal field type to Avro type.
        
        Args:
            field: Internal field schema
            
        Returns:
            Avro type definition
        """
        field_type = field.get("type", "string")
        
        # Handle primitive types
        primitive_map = {
            "string": "string",
            "integer": "int",
            "number": "double",
            "boolean": "boolean",
            "null": "null"
        }
        
        if field_type in primitive_map:
            return primitive_map[field_type]
            
        # Handle complex types
        if field_type == "object" and "properties" in field:
            avro_record = {
                "type": "record",
                "name": field.get("name", "AnonymousRecord"),
                "fields": []
            }
            
            for prop_name, prop_schema in field["properties"].items():
                avro_field = {
                    "name": prop_name,
                    "type": self._convert_field_to_avro_type(prop_schema)
                }
                
                # Add description if available
                if "description" in prop_schema:
                    avro_field["doc"] = prop_schema["description"]
                    
                # Add default if available
                if "default" in prop_schema:
                    avro_field["default"] = prop_schema["default"]
                elif not prop_schema.get("required", False):
                    # Make nullable if not required
                    avro_field["type"] = ["null", avro_field["type"]]
                    avro_field["default"] = None
                
                avro_record["fields"].append(avro_field)
                
            return avro_record
            
        elif field_type == "array" and "items" in field:
            return {
                "type": "array",
                "items": self._convert_field_to_avro_type(field["items"])
            }
            
        # Default to string for unknown types
        return "string"
    
    def _get_xsd_type(self, field_type: str) -> str:
        """
        Convert internal type to XML Schema (XSD) type.
        
        Args:
            field_type: Internal field type
            
        Returns:
            XSD type name
        """
        xsd_map = {
            "string": "xs:string",
            "integer": "xs:integer",
            "number": "xs:decimal",
            "boolean": "xs:boolean",
            "object": "xs:complexType",
            "array": "xs:sequence",
            "null": "xs:anyType"
        }
        
        return xsd_map.get(field_type, "xs:string")
    
    def _is_date(self, value: str) -> bool:
        """
        Check if a string is likely a date.
        
        Args:
            value: String to check
            
        Returns:
            True if string is a date, False otherwise
        """
        date_patterns = [
            r"\d{4}-\d{2}-\d{2}",  # ISO format
            r"\d{2}/\d{2}/\d{4}",  # MM/DD/YYYY
            r"\d{2}-\d{2}-\d{4}"   # MM-DD-YYYY
        ]
        
        return any(re.match(pattern, value) for pattern in date_patterns)
    
    def _is_datetime(self, value: str) -> bool:
        """
        Check if a string is likely a datetime.
        
        Args:
            value: String to check
            
        Returns:
            True if string is a datetime, False otherwise
        """
        return "T" in value and self._is_date(value.split("T")[0])
    
    def _is_email(self, value: str) -> bool:
        """
        Check if a string is likely an email.
        
        Args:
            value: String to check
            
        Returns:
            True if string is an email, False otherwise
        """
        return "@" in value and "." in value.split("@")[1]
    
    def _is_url(self, value: str) -> bool:
        """
        Check if a string is likely a URL.
        
        Args:
            value: String to check
            
        Returns:
            True if string is a URL, False otherwise
        """
        return value.startswith(("http://", "https://", "www."))