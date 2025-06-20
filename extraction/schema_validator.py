"""
Schema Validation Module

This module provides comprehensive data validation against schemas including:
- Data validation against schema definitions
- Type coercion for different field types
- Required field validation
- Field relationship validation
- Custom validation rules
"""

import re
import json
import logging
import datetime
from typing import Dict, Any, List, Optional, Union
from urllib.parse import urlparse

from extraction.core.extraction_interface import SchemaValidator as BaseSchemaValidator

logger = logging.getLogger(__name__)

class SchemaValidator(BaseSchemaValidator):
    """
    Validates data against schemas with extensive validation capabilities.
    
    This class provides methods to validate extracted data against schemas,
    coerce types, validate required fields, check constraints, and apply
    custom validation rules.
    """
    
    def __init__(self, context=None):
        """Initialize the schema validator."""
        super().__init__(context)
        self._initialized = False
        self._config = {}
        self._validators = {}
        
    def initialize(self) -> None:
        """Initialize the schema validator."""
        if self._initialized:
            return
            
        # Register specialized validators
        self._validators = {
            "string": StringValidator(),
            "number": NumberValidator(),
            "integer": NumberValidator(),
            "boolean": BooleanValidator(),
            "object": ObjectValidator(),
            "array": ArrayValidator(),
            "date": DateValidator(),
            "datetime": DateValidator(),
            "email": StringValidator(),
            "url": URLValidator(),
            "enum": EnumValidator(),
            "pattern": PatternValidator()
        }
        
        self._initialized = True
        logger.debug("Schema validator initialized")
    
    def shutdown(self) -> None:
        """Clean up resources."""
        self._initialized = False
        logger.debug("Schema validator shut down")
        
    def can_handle(self, content: Any, content_type: Optional[str] = None) -> bool:
        """
        Check if this validator can handle the given content and content type.
        
        Args:
            content: Content to check compatibility with
            content_type: Optional hint about the content type
            
        Returns:
            True if the validator can handle this content, False otherwise
        """
        # Can validate any content against a schema
        return True
    
    def extract(self, content: Any, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract method is not applicable for validator.
        
        SchemaValidator doesn't extract data but validates it.
        Use validate() instead.
        
        Args:
            content: Not used
            options: Not used
            
        Returns:
            Error message indicating this method shouldn't be used
        """
        return {
            "_error": "SchemaValidator doesn't extract data. Use validate() instead."
        }
    
    def validate(self, data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate data against a schema.
        
        Args:
            data: Data to validate
            schema: Schema to validate against
            
        Returns:
            Validation results with errors and warnings
        """
        if not self._initialized:
            self.initialize()
            
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "value": data,
            "metadata": {
                "validated_at": datetime.datetime.now().isoformat(),
                "schema_name": schema.get("_metadata", {}).get("name", "unnamed_schema")
            }
        }
        
        # Skip validation for null data if not required
        if data is None:
            schema_is_required = False
            
            # Check if the schema has a required property
            if isinstance(schema, dict):
                schema_is_required = schema.get("required", False)
            
            if schema_is_required:
                validation_result["valid"] = False
                validation_result["errors"].append({
                    "path": "",
                    "message": "Data is required and cannot be null"
                })
            return validation_result
        
        # Validate required fields
        missing_fields = self.validate_required_fields(data, schema)
        if missing_fields:
            validation_result["valid"] = False
            for field in missing_fields:
                validation_result["errors"].append({
                    "path": field,
                    "message": f"Required field '{field}' is missing"
                })
        
        # Validate data types and constraints
        field_errors = []
        for field_name, field_value in data.items():
            if field_name.startswith("_"):
                continue  # Skip metadata fields
                
            # Get field schema
            field_schema = schema.get(field_name, {})
            if not field_schema:
                # Field not in schema
                validation_result["warnings"].append({
                    "path": field_name,
                    "message": f"Field '{field_name}' not defined in schema"
                })
                continue
            
            # Validate field
            field_result = self.validate_field(field_value, field_schema, field_name)
            if not field_result["valid"]:
                field_errors.extend(field_result["errors"])
        
        if field_errors:
            validation_result["valid"] = False
            validation_result["errors"].extend(field_errors)
        
        # Validate relationships between fields if any are defined
        relationship_errors = self.validate_relationships(data, schema)
        if relationship_errors:
            validation_result["valid"] = False
            validation_result["errors"].extend(relationship_errors)
        
        return validation_result
    
    def validate_field(self, field_value: Any, field_schema: Dict[str, Any], path: str = "") -> Dict[str, Any]:
        """
        Validate a single field against its schema.
        
        Args:
            field_value: Value to validate
            field_schema: Schema for the field
            path: JSON path to the field (for error reporting)
            
        Returns:
            Validation results for the field
        """
        if not self._initialized:
            self.initialize()
            
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "value": field_value
        }
        
        # Skip validation for null value if not required
        if field_value is None:
            field_is_required = field_schema.get("required", False)
            if field_is_required:
                result["valid"] = False
                result["errors"].append({
                    "path": path,
                    "message": f"Field is required and cannot be null"
                })
            return result
        
        # Get field type
        field_type = field_schema.get("type", "string")
        
        # Check for enum first as it trumps other type checks
        if "enum" in field_schema:
            enum_validator = self._validators["enum"]
            enum_result = enum_validator.validate(field_value, field_schema, path)
            if not enum_result["valid"]:
                result["valid"] = False
                result["errors"].extend(enum_result["errors"])
                return result
        
        # Validate type-specific constraints
        if field_type in self._validators:
            # Use the appropriate validator
            validator = self._validators[field_type]
            type_result = validator.validate(field_value, field_schema, path)
            
            if not type_result["valid"]:
                result["valid"] = False
                result["errors"].extend(type_result["errors"])
            
            # Try to coerce value if it's the wrong type
            if not type_result["valid"] and "type" in type_result["errors"][0]["message"]:
                try:
                    # Attempt to coerce
                    coerced_value = validator.coerce(field_value, field_schema)
                    
                    # Validate again with coerced value
                    coerced_result = validator.validate(coerced_value, field_schema, path)
                    
                    if coerced_result["valid"]:
                        # Coercion successful
                        result["valid"] = True
                        result["errors"] = []
                        result["warnings"].append({
                            "path": path,
                            "message": f"Value coerced from {type(field_value).__name__} to {field_type}"
                        })
                        result["value"] = coerced_value
                    else:
                        # Coercion didn't solve the problem
                        result["warnings"].append({
                            "path": path,
                            "message": f"Attempted coercion failed"
                        })
                except Exception as e:
                    # Coercion failed
                    result["warnings"].append({
                        "path": path,
                        "message": f"Coercion attempt failed: {str(e)}"
                    })
        else:
            # Unknown type
            result["warnings"].append({
                "path": path,
                "message": f"Unknown field type: {field_type}"
            })
        
        # Validate pattern if specified (regardless of type)
        if "pattern" in field_schema:
            pattern_validator = self._validators["pattern"]
            pattern_result = pattern_validator.validate(field_value, field_schema, path)
            if not pattern_result["valid"]:
                result["valid"] = False
                result["errors"].extend(pattern_result["errors"])
        
        # For objects, validate nested fields
        if field_type == "object" and isinstance(field_value, dict) and "properties" in field_schema:
            for prop_name, prop_value in field_value.items():
                if prop_name.startswith("_"):
                    continue  # Skip metadata fields
                    
                prop_schema = field_schema["properties"].get(prop_name, {})
                if not prop_schema:
                    # Property not in schema
                    result["warnings"].append({
                        "path": f"{path}.{prop_name}",
                        "message": f"Property '{prop_name}' not defined in schema"
                    })
                    continue
                
                # Validate nested property
                prop_result = self.validate_field(prop_value, prop_schema, f"{path}.{prop_name}")
                if not prop_result["valid"]:
                    result["valid"] = False
                    result["errors"].extend(prop_result["errors"])
        
        # For arrays of objects, validate each item
        if field_type == "array" and isinstance(field_value, list) and "items" in field_schema:
            item_schema = field_schema["items"]
            
            for i, item in enumerate(field_value):
                # Validate array item
                item_result = self.validate_field(item, item_schema, f"{path}[{i}]")
                if not item_result["valid"]:
                    result["valid"] = False
                    result["errors"].extend(item_result["errors"])
        
        return result
    
    def get_validation_errors(self, data: Dict[str, Any], schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get detailed validation errors for data against a schema.
        
        Args:
            data: Data to validate
            schema: Schema to validate against
            
        Returns:
            List of validation errors with path and message
        """
        validation_result = self.validate(data, schema)
        return validation_result["errors"]
    
    def coerce_types(self, data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coerce data values to the types specified in the schema.
        
        Args:
            data: Data to coerce
            schema: Schema with type information
            
        Returns:
            Data with coerced types
        """
        if not self._initialized:
            self.initialize()
            
        if not isinstance(data, dict):
            # Can't coerce non-dict data at the top level
            return data
            
        coerced_data = {}
        
        for field_name, field_value in data.items():
            if field_name.startswith("_"):
                # Preserve metadata fields
                coerced_data[field_name] = field_value
                continue
                
            # Get field schema
            field_schema = schema.get(field_name, {})
            if not field_schema:
                # Field not in schema, copy as is
                coerced_data[field_name] = field_value
                continue
            
            # Handle null values
            if field_value is None:
                coerced_data[field_name] = None
                continue
            
            # Get field type
            field_type = field_schema.get("type", "string")
            
            # Coerce based on type
            if field_type in self._validators:
                try:
                    validator = self._validators[field_type]
                    coerced_value = validator.coerce(field_value, field_schema)
                    coerced_data[field_name] = coerced_value
                except Exception as e:
                    # Coercion failed, use original value
                    logger.warning(f"Coercion failed for field '{field_name}': {str(e)}")
                    coerced_data[field_name] = field_value
            else:
                # Unknown type, copy as is
                coerced_data[field_name] = field_value
            
            # Handle nested objects
            if field_type == "object" and isinstance(coerced_data[field_name], dict) and "properties" in field_schema:
                coerced_data[field_name] = self.coerce_types(coerced_data[field_name], field_schema["properties"])
            
            # Handle arrays
            if field_type == "array" and isinstance(coerced_data[field_name], list) and "items" in field_schema:
                item_schema = field_schema["items"]
                
                # For arrays of objects, coerce each item
                if item_schema.get("type") == "object" and "properties" in item_schema:
                    coerced_items = []
                    for item in coerced_data[field_name]:
                        if isinstance(item, dict):
                            coerced_items.append(self.coerce_types(item, item_schema["properties"]))
                        else:
                            coerced_items.append(item)
                    coerced_data[field_name] = coerced_items
                else:
                    # For arrays of primitives, coerce each item
                    coerced_items = []
                    for item in coerced_data[field_name]:
                        try:
                            if item_schema.get("type") in self._validators:
                                validator = self._validators[item_schema["type"]]
                                coerced_items.append(validator.coerce(item, item_schema))
                            else:
                                coerced_items.append(item)
                        except Exception:
                            coerced_items.append(item)
                    coerced_data[field_name] = coerced_items
        
        return coerced_data
    
    def get_missing_fields(self, data: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
        """
        Identify missing fields that are required by the schema.
        
        Args:
            data: Data to check
            schema: Schema defining required fields
            
        Returns:
            List of missing required field paths
        """
        return self.validate_required_fields(data, schema)
    
    def validate_required_fields(self, data: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
        """
        Validate that all required fields are present.
        
        Args:
            data: Data to validate
            schema: Schema with required field information
            
        Returns:
            List of missing required fields
        """
        if not self._initialized:
            self.initialize()
            
        missing_fields = []
        
        # Check each field in schema
        for field_name, field_schema in schema.items():
            if field_name.startswith("_"):
                continue  # Skip metadata fields
                
            if not isinstance(field_schema, dict):
                continue  # Skip non-dict field schemas
                
            is_required = field_schema.get("required", False)
            
            if is_required and (field_name not in data or data[field_name] is None):
                missing_fields.append(field_name)
            
            # Check nested objects for required fields
            if field_name in data and data[field_name] is not None:
                field_value = data[field_name]
                field_type = field_schema.get("type", "string")
                
                # For nested objects
                if field_type == "object" and isinstance(field_value, dict) and "properties" in field_schema:
                    nested_missing = self.validate_required_fields(field_value, field_schema["properties"])
                    missing_fields.extend([f"{field_name}.{f}" for f in nested_missing])
                
                # For arrays of objects
                if field_type == "array" and isinstance(field_value, list) and "items" in field_schema:
                    item_schema = field_schema["items"]
                    
                    if item_schema.get("type") == "object" and "properties" in item_schema:
                        for i, item in enumerate(field_value):
                            if isinstance(item, dict):
                                nested_missing = self.validate_required_fields(item, item_schema["properties"])
                                missing_fields.extend([f"{field_name}[{i}].{f}" for f in nested_missing])
        
        return missing_fields
    
    def validate_relationships(self, data: Dict[str, Any], schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Validate relationships between data fields.
        
        Args:
            data: Data to validate
            schema: Schema with relationship information
            
        Returns:
            List of relationship validation issues
        """
        if not self._initialized:
            self.initialize()
            
        errors = []
        
        # Check for schema-level relationships
        if "_relationships" in schema:
            relationships = schema["_relationships"]
            
            for relationship in relationships:
                rel_type = relationship.get("type")
                fields = relationship.get("fields", [])
                
                if not rel_type or not fields:
                    continue
                
                # Validate based on relationship type
                if rel_type == "mutually_exclusive":
                    # Only one of the fields should be present
                    present_fields = [f for f in fields if f in data and data[f] is not None]
                    if len(present_fields) > 1:
                        errors.append({
                            "path": ",".join(present_fields),
                            "message": f"Fields are mutually exclusive: {', '.join(present_fields)}"
                        })
                
                elif rel_type == "mutually_required":
                    # Either all fields should be present or none
                    present_fields = [f for f in fields if f in data and data[f] is not None]
                    if 0 < len(present_fields) < len(fields):
                        missing_fields = [f for f in fields if f not in present_fields]
                        errors.append({
                            "path": ",".join(fields),
                            "message": f"Fields are mutually required: {', '.join(missing_fields)} missing"
                        })
                
                elif rel_type == "greater_than":
                    # Field1 should be greater than field2
                    if len(fields) == 2 and fields[0] in data and fields[1] in data:
                        if data[fields[0]] is not None and data[fields[1]] is not None:
                            try:
                                if not data[fields[0]] > data[fields[1]]:
                                    errors.append({
                                        "path": ",".join(fields),
                                        "message": f"'{fields[0]}' must be greater than '{fields[1]}'"
                                    })
                            except TypeError:
                                errors.append({
                                    "path": ",".join(fields),
                                    "message": f"Cannot compare '{fields[0]}' and '{fields[1]}': incompatible types"
                                })
                
                elif rel_type == "less_than":
                    # Field1 should be less than field2
                    if len(fields) == 2 and fields[0] in data and fields[1] in data:
                        if data[fields[0]] is not None and data[fields[1]] is not None:
                            try:
                                if not data[fields[0]] < data[fields[1]]:
                                    errors.append({
                                        "path": ",".join(fields),
                                        "message": f"'{fields[0]}' must be less than '{fields[1]}'"
                                    })
                            except TypeError:
                                errors.append({
                                    "path": ",".join(fields),
                                    "message": f"Cannot compare '{fields[0]}' and '{fields[1]}': incompatible types"
                                })
                
                elif rel_type == "conditional_requirement":
                    # If condition_field has value, then required_field is required
                    if len(fields) == 2:
                        condition_field = fields[0]
                        required_field = fields[1]
                        
                        condition_value = relationship.get("condition_value")
                        condition_operator = relationship.get("condition_operator", "eq")
                        
                        if condition_field in data:
                            condition_met = False
                            
                            # Check condition
                            if condition_operator == "eq" and data[condition_field] == condition_value:
                                condition_met = True
                            elif condition_operator == "ne" and data[condition_field] != condition_value:
                                condition_met = True
                            elif condition_operator == "in" and data[condition_field] in condition_value:
                                condition_met = True
                            elif condition_operator == "gt" and data[condition_field] > condition_value:
                                condition_met = True
                            elif condition_operator == "lt" and data[condition_field] < condition_value:
                                condition_met = True
                            elif condition_operator == "exists" and data[condition_field] is not None:
                                condition_met = True
                            
                            # If condition is met, required_field must be present
                            if condition_met and (required_field not in data or data[required_field] is None):
                                errors.append({
                                    "path": required_field,
                                    "message": f"'{required_field}' is required when '{condition_field}' {condition_operator} {condition_value}"
                                })
        
        # Check for field-level relationships
        for field_name, field_schema in schema.items():
            if field_name.startswith("_") or not isinstance(field_schema, dict):
                continue
                
            if "_relationships" in field_schema:
                field_rels = field_schema["_relationships"]
                
                for rel in field_rels:
                    if rel.get("type") == "dependent_field":
                        dependent_field = rel.get("field")
                        
                        if field_name in data and data[field_name] is not None:
                            if dependent_field not in data or data[dependent_field] is None:
                                errors.append({
                                    "path": f"{field_name},{dependent_field}",
                                    "message": f"Field '{dependent_field}' is required when '{field_name}' is present"
                                })
        
        return errors
    
    def apply_default_values(self, data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fill defaults for missing fields.
        
        Args:
            data: Data to fill defaults in
            schema: Schema with default value information
            
        Returns:
            Data with defaults applied
        """
        if not self._initialized:
            self.initialize()
            
        if not isinstance(data, dict):
            return data
            
        filled_data = dict(data)
        
        # Apply defaults for each field in schema
        for field_name, field_schema in schema.items():
            if field_name.startswith("_") or not isinstance(field_schema, dict):
                continue
                
            # Apply default if field is missing and has default
            if field_name not in filled_data and "default" in field_schema:
                filled_data[field_name] = field_schema["default"]
            
            # Handle nested objects
            if field_name in filled_data and filled_data[field_name] is not None:
                field_value = filled_data[field_name]
                field_type = field_schema.get("type", "string")
                
                # For nested objects
                if field_type == "object" and isinstance(field_value, dict) and "properties" in field_schema:
                    filled_data[field_name] = self.apply_default_values(field_value, field_schema["properties"])
                
                # For arrays of objects
                if field_type == "array" and isinstance(field_value, list) and "items" in field_schema:
                    item_schema = field_schema["items"]
                    
                    if item_schema.get("type") == "object" and "properties" in item_schema:
                        filled_items = []
                        for item in field_value:
                            if isinstance(item, dict):
                                filled_items.append(self.apply_default_values(item, item_schema["properties"]))
                            else:
                                filled_items.append(item)
                        filled_data[field_name] = filled_items
        
        return filled_data
    
    def sanitize_data(self, data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean data according to schema.
        
        Args:
            data: Data to sanitize
            schema: Schema defining sanitization rules
            
        Returns:
            Sanitized data
        """
        if not self._initialized:
            self.initialize()
            
        if not isinstance(data, dict):
            return data
            
        sanitized_data = {}
        
        # Copy only fields defined in schema
        for field_name, field_schema in schema.items():
            if field_name.startswith("_"):
                continue  # Skip metadata fields
                
            if field_name in data:
                field_value = data[field_name]
                
                # Skip null values unless they have a default
                if field_value is None:
                    if "default" in field_schema:
                        sanitized_data[field_name] = field_schema["default"]
                    else:
                        sanitized_data[field_name] = None
                    continue
                
                field_type = field_schema.get("type", "string")
                
                # Sanitize based on type
                if field_type == "string" and isinstance(field_value, str):
                    # Apply string sanitization
                    sanitized_value = field_value.strip()
                    
                    # Apply max length if specified
                    if "maxLength" in field_schema and len(sanitized_value) > field_schema["maxLength"]:
                        sanitized_value = sanitized_value[:field_schema["maxLength"]]
                    
                    sanitized_data[field_name] = sanitized_value
                
                elif field_type == "object" and isinstance(field_value, dict) and "properties" in field_schema:
                    # Sanitize nested object
                    sanitized_data[field_name] = self.sanitize_data(field_value, field_schema["properties"])
                
                elif field_type == "array" and isinstance(field_value, list) and "items" in field_schema:
                    # Sanitize array items
                    item_schema = field_schema["items"]
                    
                    if item_schema.get("type") == "object" and "properties" in item_schema:
                        sanitized_items = []
                        for item in field_value:
                            if isinstance(item, dict):
                                sanitized_items.append(self.sanitize_data(item, item_schema["properties"]))
                            else:
                                sanitized_items.append(item)
                        sanitized_data[field_name] = sanitized_items
                    else:
                        # For arrays of primitives
                        sanitized_data[field_name] = field_value
                else:
                    # Copy other types as is
                    sanitized_data[field_name] = field_value
        
        return sanitized_data


class FieldValidator:
    """Base class for field validators."""
    
    def validate(self, value: Any, schema: Dict[str, Any], path: str = "") -> Dict[str, Any]:
        """
        Validate a value against a schema.
        
        Args:
            value: Value to validate
            schema: Schema to validate against
            path: JSON path for error reporting
            
        Returns:
            Validation result
        """
        raise NotImplementedError("Subclasses must implement validate()")
    
    def coerce(self, value: Any, schema: Dict[str, Any]) -> Any:
        """
        Coerce a value to the correct type.
        
        Args:
            value: Value to coerce
            schema: Schema with type information
            
        Returns:
            Coerced value
        """
        raise NotImplementedError("Subclasses must implement coerce()")


class StringValidator(FieldValidator):
    """Validator for string fields."""
    
    def validate(self, value: Any, schema: Dict[str, Any], path: str = "") -> Dict[str, Any]:
        """Validate a string value."""
        result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Type check
        if not isinstance(value, str):
            result["valid"] = False
            result["errors"].append({
                "path": path,
                "message": f"Expected string but got {type(value).__name__}"
            })
            return result
        
        # Check constraints
        if "minLength" in schema and len(value) < schema["minLength"]:
            result["valid"] = False
            result["errors"].append({
                "path": path,
                "message": f"String is too short (min: {schema['minLength']})"
            })
        
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            result["valid"] = False
            result["errors"].append({
                "path": path,
                "message": f"String is too long (max: {schema['maxLength']})"
            })
        
        # Format validation
        if "format" in schema:
            format_type = schema["format"]
            
            if format_type == "email" and not self._is_valid_email(value):
                result["valid"] = False
                result["errors"].append({
                    "path": path,
                    "message": "Invalid email format"
                })
            elif format_type == "uri" and not self._is_valid_url(value):
                result["valid"] = False
                result["errors"].append({
                    "path": path,
                    "message": "Invalid URL format"
                })
            elif format_type == "date" and not self._is_valid_date(value):
                result["valid"] = False
                result["errors"].append({
                    "path": path,
                    "message": "Invalid date format (expected YYYY-MM-DD)"
                })
            elif format_type == "date-time" and not self._is_valid_datetime(value):
                result["valid"] = False
                result["errors"].append({
                    "path": path,
                    "message": "Invalid datetime format (expected ISO 8601)"
                })
        
        return result
    
    def coerce(self, value: Any, schema: Dict[str, Any]) -> str:
        """Coerce a value to string."""
        if isinstance(value, str):
            return value
            
        return str(value)
    
    def _is_valid_email(self, value: str) -> bool:
        """Check if a string is a valid email."""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(email_pattern, value))
    
    def _is_valid_url(self, value: str) -> bool:
        """Check if a string is a valid URL."""
        try:
            result = urlparse(value)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _is_valid_date(self, value: str) -> bool:
        """Check if a string is a valid date."""
        try:
            datetime.datetime.strptime(value, "%Y-%m-%d")
            return True
        except ValueError:
            return False
    
    def _is_valid_datetime(self, value: str) -> bool:
        """Check if a string is a valid ISO 8601 datetime."""
        try:
            # Try parsing with various ISO formats
            for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"]:
                try:
                    datetime.datetime.strptime(value, fmt)
                    return True
                except:
                    continue
            return False
        except:
            return False


class NumberValidator(FieldValidator):
    """Validator for number fields."""
    
    def validate(self, value: Any, schema: Dict[str, Any], path: str = "") -> Dict[str, Any]:
        """Validate a numeric value."""
        result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Type check
        field_type = schema.get("type", "number")
        expected_type = int if field_type == "integer" else (int, float)
        
        if not isinstance(value, expected_type):
            if field_type == "integer" and isinstance(value, float) and value.is_integer():
                # Allow integer-valued floats for integer fields
                pass
            else:
                result["valid"] = False
                result["errors"].append({
                    "path": path,
                    "message": f"Expected {field_type} but got {type(value).__name__}"
                })
                return result
        
        # Check constraints
        if "minimum" in schema and value < schema["minimum"]:
            result["valid"] = False
            result["errors"].append({
                "path": path,
                "message": f"Value is too small (min: {schema['minimum']})"
            })
        
        if "maximum" in schema and value > schema["maximum"]:
            result["valid"] = False
            result["errors"].append({
                "path": path,
                "message": f"Value is too large (max: {schema['maximum']})"
            })
        
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            result["valid"] = False
            result["errors"].append({
                "path": path,
                "message": f"Value must be greater than {schema['exclusiveMinimum']}"
            })
        
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            result["valid"] = False
            result["errors"].append({
                "path": path,
                "message": f"Value must be less than {schema['exclusiveMaximum']}"
            })
        
        if "multipleOf" in schema and value % schema["multipleOf"] != 0:
            result["valid"] = False
            result["errors"].append({
                "path": path,
                "message": f"Value must be a multiple of {schema['multipleOf']}"
            })
        
        return result
    
    def coerce(self, value: Any, schema: Dict[str, Any]) -> Union[int, float]:
        """Coerce a value to number."""
        field_type = schema.get("type", "number")
        
        if field_type == "integer":
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)
            if isinstance(value, str):
                # Try to parse as integer
                return int(float(value))  # Handle "123.0" strings
            return int(value)
        else:
            # number type
            if isinstance(value, (int, float)):
                return value
            if isinstance(value, str):
                return float(value)
            return float(value)


class BooleanValidator(FieldValidator):
    """Validator for boolean fields."""
    
    def validate(self, value: Any, schema: Dict[str, Any], path: str = "") -> Dict[str, Any]:
        """Validate a boolean value."""
        result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Type check
        if not isinstance(value, bool):
            result["valid"] = False
            result["errors"].append({
                "path": path,
                "message": f"Expected boolean but got {type(value).__name__}"
            })
        
        return result
    
    def coerce(self, value: Any, schema: Dict[str, Any]) -> bool:
        """Coerce a value to boolean."""
        if isinstance(value, bool):
            return value
        
        if isinstance(value, str):
            # Handle common string representations
            if value.lower() in ["true", "yes", "1", "on", "y"]:
                return True
            if value.lower() in ["false", "no", "0", "off", "n"]:
                return False
            # Default to false for other strings
            return bool(value)
        
        # For numbers, 0 is False, everything else is True
        if isinstance(value, (int, float)):
            return bool(value)
        
        # Default to False for null/None
        if value is None:
            return False
        
        # For other types, use Python's truthiness rules
        return bool(value)


class ObjectValidator(FieldValidator):
    """Validator for object fields."""
    
    def validate(self, value: Any, schema: Dict[str, Any], path: str = "") -> Dict[str, Any]:
        """Validate an object value."""
        result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Type check
        if not isinstance(value, dict):
            result["valid"] = False
            result["errors"].append({
                "path": path,
                "message": f"Expected object but got {type(value).__name__}"
            })
            return result
        
        # Check constraints - property count
        if "minProperties" in schema and len(value) < schema["minProperties"]:
            result["valid"] = False
            result["errors"].append({
                "path": path,
                "message": f"Object has too few properties (min: {schema['minProperties']})"
            })
        
        if "maxProperties" in schema and len(value) > schema["maxProperties"]:
            result["valid"] = False
            result["errors"].append({
                "path": path,
                "message": f"Object has too many properties (max: {schema['maxProperties']})"
            })
        
        # Check for required properties - handled by parent validator
        # Check property values - handled by parent validator
        
        return result
    
    def coerce(self, value: Any, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Coerce a value to object."""
        if isinstance(value, dict):
            return value
        
        if isinstance(value, str):
            # Try to parse as JSON
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                raise ValueError(f"Cannot coerce string to object: not valid JSON")
        
        raise ValueError(f"Cannot coerce {type(value).__name__} to object")


class ArrayValidator(FieldValidator):
    """Validator for array fields."""
    
    def validate(self, value: Any, schema: Dict[str, Any], path: str = "") -> Dict[str, Any]:
        """Validate an array value."""
        result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Type check
        if not isinstance(value, list):
            result["valid"] = False
            result["errors"].append({
                "path": path,
                "message": f"Expected array but got {type(value).__name__}"
            })
            return result
        
        # Check constraints
        if "minItems" in schema and len(value) < schema["minItems"]:
            result["valid"] = False
            result["errors"].append({
                "path": path,
                "message": f"Array has too few items (min: {schema['minItems']})"
            })
        
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            result["valid"] = False
            result["errors"].append({
                "path": path,
                "message": f"Array has too many items (max: {schema['maxItems']})"
            })
        
        # Check uniqueness constraint
        if schema.get("uniqueItems", False):
            # Create a list of hashable representations of items
            try:
                hashable_items = [json.dumps(item, sort_keys=True) if isinstance(item, (dict, list)) else item for item in value]
                if len(hashable_items) != len(set(hashable_items)):
                    result["valid"] = False
                    result["errors"].append({
                        "path": path,
                        "message": "Array items must be unique"
                    })
            except TypeError:
                result["warnings"].append({
                    "path": path,
                    "message": "Could not check uniqueness of array items (unhashable types)"
                })
        
        # Item validation is handled by parent validator
        
        return result
    
    def coerce(self, value: Any, schema: Dict[str, Any]) -> List[Any]:
        """Coerce a value to array."""
        if isinstance(value, list):
            return value
        
        if isinstance(value, str):
            # Try to parse as JSON
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
                raise ValueError(f"JSON value is not an array")
            except json.JSONDecodeError:
                # Try comma-separated values
                return [item.strip() for item in value.split(",") if item.strip()]
        
        # Single value becomes a single-item array
        return [value]


class DateValidator(FieldValidator):
    """Validator for date/datetime fields."""
    
    def validate(self, value: Any, schema: Dict[str, Any], path: str = "") -> Dict[str, Any]:
        """Validate a date/datetime value."""
        result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check if it's a string
        if not isinstance(value, str):
            if isinstance(value, (datetime.date, datetime.datetime)):
                # Native date/datetime objects are valid
                return result
                
            result["valid"] = False
            result["errors"].append({
                "path": path,
                "message": f"Expected date string but got {type(value).__name__}"
            })
            return result
        
        # Validate format
        format_type = schema.get("format", "date")
        
        if format_type == "date":
            if not self._is_valid_date(value):
                result["valid"] = False
                result["errors"].append({
                    "path": path,
                    "message": "Invalid date format (expected YYYY-MM-DD)"
                })
        elif format_type == "date-time":
            if not self._is_valid_datetime(value):
                result["valid"] = False
                result["errors"].append({
                    "path": path,
                    "message": "Invalid datetime format (expected ISO 8601)"
                })
        
        return result
    
    def coerce(self, value: Any, schema: Dict[str, Any]) -> str:
        """Coerce a value to date string."""
        if isinstance(value, str):
            # Validate and normalize date string
            format_type = schema.get("format", "date")
            
            if format_type == "date":
                # Try to parse and reformat
                try:
                    date_obj = self._parse_date(value)
                    return date_obj.strftime("%Y-%m-%d")
                except ValueError:
                    raise ValueError(f"Cannot coerce '{value}' to date format")
            elif format_type == "date-time":
                # Try to parse and reformat
                try:
                    dt_obj = self._parse_datetime(value)
                    return dt_obj.isoformat()
                except ValueError:
                    raise ValueError(f"Cannot coerce '{value}' to datetime format")
            
            return value
        
        if isinstance(value, datetime.date) and not isinstance(value, datetime.datetime):
            return value.strftime("%Y-%m-%d")
        
        if isinstance(value, datetime.datetime):
            return value.isoformat()
        
        raise ValueError(f"Cannot coerce {type(value).__name__} to date string")
    
    def _is_valid_date(self, value: str) -> bool:
        """Check if a string is a valid date."""
        try:
            self._parse_date(value)
            return True
        except ValueError:
            return False
    
    def _is_valid_datetime(self, value: str) -> bool:
        """Check if a string is a valid datetime."""
        try:
            self._parse_datetime(value)
            return True
        except ValueError:
            return False
    
    def _parse_date(self, value: str) -> datetime.date:
        """Parse a date string in various formats."""
        # Try ISO format first (YYYY-MM-DD)
        try:
            return datetime.datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError:
            pass
        
        # Try other common formats
        formats = [
            "%d/%m/%Y",  # DD/MM/YYYY
            "%m/%d/%Y",  # MM/DD/YYYY
            "%d-%m-%Y",  # DD-MM-YYYY
            "%m-%d-%Y",  # MM-DD-YYYY
            "%Y/%m/%d",  # YYYY/MM/DD
            "%d %b %Y",  # DD MMM YYYY
            "%d %B %Y"   # DD Month YYYY
        ]
        
        for fmt in formats:
            try:
                return datetime.datetime.strptime(value, fmt).date()
            except ValueError:
                continue
        
        raise ValueError(f"Date string '{value}' does not match any known format")
    
    def _parse_datetime(self, value: str) -> datetime.datetime:
        """Parse a datetime string in various formats."""
        # Try ISO formats
        iso_formats = [
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f"
        ]
        
        for fmt in iso_formats:
            try:
                return datetime.datetime.strptime(value, fmt)
            except ValueError:
                continue
        
        # Try other common formats
        formats = [
            "%d/%m/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M:%S",
            "%d-%m-%Y %H:%M:%S",
            "%m-%d-%Y %H:%M:%S"
        ]
        
        for fmt in formats:
            try:
                return datetime.datetime.strptime(value, fmt)
            except ValueError:
                continue
        
        raise ValueError(f"Datetime string '{value}' does not match any known format")


class URLValidator(FieldValidator):
    """Validator for URL fields."""
    
    def validate(self, value: Any, schema: Dict[str, Any], path: str = "") -> Dict[str, Any]:
        """Validate a URL value."""
        result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Type check
        if not isinstance(value, str):
            result["valid"] = False
            result["errors"].append({
                "path": path,
                "message": f"Expected URL string but got {type(value).__name__}"
            })
            return result
        
        # URL validation
        if not self._is_valid_url(value):
            result["valid"] = False
            result["errors"].append({
                "path": path,
                "message": "Invalid URL format"
            })
        
        return result
    
    def coerce(self, value: Any, schema: Dict[str, Any]) -> str:
        """Coerce a value to URL string."""
        if isinstance(value, str):
            # Check if it's a valid URL
            if self._is_valid_url(value):
                return value
                
            # Try to fix common URL issues
            if not value.startswith(("http://", "https://")) and not value.startswith("//"):
                prefixed = "http://" + value
                if self._is_valid_url(prefixed):
                    return prefixed
            
            raise ValueError(f"Cannot coerce '{value}' to a valid URL")
        
        raise ValueError(f"Cannot coerce {type(value).__name__} to URL string")
    
    def _is_valid_url(self, value: str) -> bool:
        """Check if a string is a valid URL."""
        try:
            result = urlparse(value)
            return all([result.scheme, result.netloc])
        except:
            return False


class EnumValidator(FieldValidator):
    """Validator for enum fields."""
    
    def validate(self, value: Any, schema: Dict[str, Any], path: str = "") -> Dict[str, Any]:
        """Validate a value against enum options."""
        result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        if "enum" not in schema:
            # Not an enum field
            return result
        
        enum_values = schema["enum"]
        
        if value not in enum_values:
            result["valid"] = False
            result["errors"].append({
                "path": path,
                "message": f"Value must be one of: {', '.join(str(v) for v in enum_values)}"
            })
        
        return result
    
    def coerce(self, value: Any, schema: Dict[str, Any]) -> Any:
        """Coerce a value to one of the enum options if possible."""
        if "enum" not in schema:
            return value
            
        enum_values = schema["enum"]
        
        if value in enum_values:
            return value
            
        # Try case-insensitive matching for strings
        if isinstance(value, str):
            for enum_val in enum_values:
                if isinstance(enum_val, str) and value.lower() == enum_val.lower():
                    return enum_val
        
        # Try numeric coercion if the enum values are numeric
        if all(isinstance(x, (int, float)) for x in enum_values) and not isinstance(value, (bool, complex)):
            try:
                numeric_value = float(value) if "." in str(value) else int(value)
                if numeric_value in enum_values:
                    return numeric_value
            except (ValueError, TypeError):
                pass
        
        raise ValueError(f"Cannot coerce {value} to one of the enum values: {enum_values}")


class PatternValidator(FieldValidator):
    """Validator for pattern-matching fields."""
    
    def validate(self, value: Any, schema: Dict[str, Any], path: str = "") -> Dict[str, Any]:
        """Validate a value against a regex pattern."""
        result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        if "pattern" not in schema:
            # No pattern to validate against
            return result
        
        # Pattern validation requires a string
        if not isinstance(value, str):
            result["valid"] = False
            result["errors"].append({
                "path": path,
                "message": f"Expected string for pattern matching but got {type(value).__name__}"
            })
            return result
        
        # Validate against pattern
        pattern = schema["pattern"]
        try:
            if not re.match(pattern, value):
                result["valid"] = False
                result["errors"].append({
                    "path": path,
                    "message": f"Value does not match pattern: {pattern}"
                })
        except re.error as e:
            result["valid"] = False
            result["errors"].append({
                "path": path,
                "message": f"Invalid pattern in schema: {str(e)}"
            })
        
        return result
    
    def coerce(self, value: Any, schema: Dict[str, Any]) -> str:
        """Coerce not really applicable for pattern validation."""
        # Just convert to string for pattern matching
        return str(value)