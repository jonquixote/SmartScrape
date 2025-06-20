"""
Schema Validation Stage Module

This module provides a pipeline stage for validating extracted data against
a schema, ensuring compliance with data structure requirements.
"""

import logging
import json
from typing import Dict, Any, Optional, Union, List
from jsonschema import validate, ValidationError, Draft7Validator, validators

from core.pipeline.stages.base_stages import ProcessingStage
from core.pipeline.context import PipelineContext
from extraction.schema_validator import SchemaValidator

logger = logging.getLogger(__name__)

class SchemaValidationStage(ProcessingStage):
    """
    Pipeline stage that validates extracted data against a schema.
    
    This stage ensures that the extracted data complies with the specified schema,
    validates data types and required fields, and can either reject non-compliant
    data or attempt to fix issues automatically.
    """
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the schema validation stage with configuration.
        
        Args:
            name: Name of this stage (defaults to class name)
            config: Configuration dictionary
        """
        super().__init__(name, config)
        self.validator = None
        self.input_key = self.config.get("input_key", "normalized_data")
        self.output_key = self.config.get("output_key", "validated_data")
        self.schema_key = self.config.get("schema_key", "extraction_schema")
        self.schema_path = self.config.get("schema_path", None)
        self.content_type_schemas = self.config.get("content_type_schemas", {})
        self.fail_on_validation_error = self.config.get("fail_on_validation_error", False)
        self.auto_fix_issues = self.config.get("auto_fix_issues", True)
        self.remove_additional_properties = self.config.get("remove_additional_properties", False)
        self.add_validation_metadata = self.config.get("add_validation_metadata", True)
        
    async def initialize(self) -> None:
        """Initialize the validator and stage resources."""
        if self._initialized:
            return
            
        # Create the schema validator
        self.validator = SchemaValidator()
        
        # Initialize the validator with configuration
        validator_config = {
            "schema_path": self.schema_path,
            "content_type_schemas": self.content_type_schemas,
            "auto_fix_issues": self.auto_fix_issues,
            "remove_additional_properties": self.remove_additional_properties
        }
        self.validator.initialize(validator_config)
        
        await super().initialize()
        logger.debug(f"{self.name} initialized with schema validator")
        
    async def cleanup(self) -> None:
        """Clean up resources used by this stage."""
        if self.validator:
            self.validator.shutdown()
            
        await super().cleanup()
        logger.debug(f"{self.name} cleaned up")
        
    async def validate_input(self, context: PipelineContext) -> bool:
        """
        Validate that the required inputs are present in the context.
        
        Args:
            context: Pipeline context containing data
            
        Returns:
            True if validation passes, False otherwise
        """
        if not context.has_key(self.input_key):
            logger.warning(f"Missing required input '{self.input_key}' in context")
            context.add_error(self.name, f"Missing required input: {self.input_key}")
            return False
            
        # Check if we have a schema (either in context or pre-configured)
        has_schema = (context.has_key(self.schema_key) or 
                    self.schema_path is not None or 
                    context.has_key("content_type"))
        
        if not has_schema:
            logger.warning("No schema available for validation")
            if self.fail_on_validation_error:
                context.add_error(self.name, "No schema available for validation")
                return False
                
        return True
        
    async def transform_data(self, data: Dict[str, Any], context: PipelineContext) -> Optional[Dict[str, Any]]:
        """
        Validate extracted data against a schema.
        
        Args:
            data: Input data (not used, we get data from context)
            context: Pipeline context containing extracted data
            
        Returns:
            Dictionary containing validated data or None if validation fails
        """
        try:
            if not self.validator:
                self.validator = SchemaValidator()
                validator_config = {
                    "schema_path": self.schema_path,
                    "content_type_schemas": self.content_type_schemas,
                    "auto_fix_issues": self.auto_fix_issues,
                    "remove_additional_properties": self.remove_additional_properties
                }
                self.validator.initialize(validator_config)
            
            # Get extracted data from context
            extracted_data = context.get(self.input_key)
            
            # Get schema from context if available
            schema = None
            if context.has_key(self.schema_key):
                schema = context.get(self.schema_key)
                logger.info("Using schema from context")
            elif self.schema_path:
                # Schema will be loaded from path by validator
                logger.info(f"Using schema from configured path: {self.schema_path}")
            elif context.has_key("content_type"):
                # Try to use schema based on content type
                content_type = context.get("content_type")
                if content_type in self.content_type_schemas:
                    logger.info(f"Using schema for content type: {content_type}")
                else:
                    logger.warning(f"No schema available for content type: {content_type}")
                    if self.fail_on_validation_error:
                        context.add_error(self.name, f"No schema for content type: {content_type}")
                        return None
            
            # Prepare validation options
            options = {
                "content_type": context.get("content_type", "unknown"),
                "schema": schema,
                "auto_fix_issues": self.auto_fix_issues,
                "remove_additional_properties": self.remove_additional_properties,
                "add_validation_metadata": self.add_validation_metadata
            }
            
            # Validate data against schema
            logger.info("Validating extracted data against schema")
            validation_result = self.validator.validate(extracted_data, options)
            
            # Check if validation passed
            if not validation_result.get("valid", False):
                errors = validation_result.get("errors", [])
                logger.warning(f"Schema validation failed with {len(errors)} errors")
                
                # Add validation errors to context
                for error in errors:
                    context.add_error(self.name, f"Validation error: {error.get('message')}")
                
                if self.fail_on_validation_error:
                    logger.error("Failing due to validation errors")
                    return None
                else:
                    logger.warning("Continuing despite validation errors")
            else:
                logger.info("Schema validation passed")
            
            # Get validated data
            validated_data = validation_result.get("data", {})
            
            # Add validation metadata if requested
            if self.add_validation_metadata:
                self._add_validation_metadata(validated_data, validation_result, context)
            
            return validated_data
            
        except Exception as e:
            logger.error(f"Error in schema validation: {str(e)}")
            context.add_error(self.name, f"Validation error: {str(e)}")
            
            if self.fail_on_validation_error:
                return None
                
            # Return the original data as a fallback
            return context.get(self.input_key)
    
    def _add_validation_metadata(self, validated_data: Dict[str, Any], 
                              validation_result: Dict[str, Any],
                              context: PipelineContext) -> None:
        """
        Add validation metadata to the output data.
        
        Args:
            validated_data: The validated data to update
            validation_result: The validation result information
            context: The pipeline context
        """
        # Initialize metadata if needed
        if "_metadata" not in validated_data:
            validated_data["_metadata"] = {}
        
        # Add validation metadata
        validation_metadata = {
            "valid": validation_result.get("valid", False),
            "error_count": len(validation_result.get("errors", [])),
            "fixed_issues": validation_result.get("fixed_issues", []),
            "validator": self.__class__.__name__,
            "validation_time": validation_result.get("validation_time", "")
        }
        
        # Include the schemas used
        if "schema" in validation_result:
            schema_id = None
            schema = validation_result["schema"]
            if isinstance(schema, dict):
                schema_id = schema.get("$id") or schema.get("id")
            validation_metadata["schema_id"] = schema_id
        
        validated_data["_metadata"]["validation"] = validation_metadata
        
        # Set validation status in context
        context.set("validation_passed", validation_result.get("valid", False))
        context.set("validation_errors", validation_result.get("errors", []))
    
    async def get_schema_for_content_type(self, content_type: str) -> Optional[Dict[str, Any]]:
        """
        Get a schema for a specific content type.
        
        Args:
            content_type: The content type to get a schema for
            
        Returns:
            Schema as dictionary or None if not found
        """
        # If validator is already initialized, use its method
        if self.validator:
            return self.validator.get_schema_for_content_type(content_type)
        
        # Otherwise, check the content_type_schemas configuration
        if content_type in self.content_type_schemas:
            schema_path = self.content_type_schemas[content_type]
            
            try:
                with open(schema_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading schema for content type {content_type}: {str(e)}")
                
        return None