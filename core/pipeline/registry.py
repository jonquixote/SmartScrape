"""
Pipeline Registry Module.

This module provides the PipelineRegistry class that manages pipeline configurations
and stage registrations, handles pipeline creation, and validates configurations.
"""

import json
import logging
import os
import threading
import yaml
from typing import Any, Dict, List, Optional, Set, Type, Union, cast

from core.pipeline.pipeline import Pipeline, PipelineError
from core.pipeline.stage import PipelineStage


logger = logging.getLogger(__name__)


class PipelineRegistryError(Exception):
    """Base exception for pipeline registry errors."""
    pass


class PipelineNotFoundError(PipelineRegistryError):
    """Exception raised when a requested pipeline is not found."""
    pass


class StageNotFoundError(PipelineRegistryError):
    """Exception raised when a requested stage is not found."""
    pass


class ConfigurationError(PipelineRegistryError):
    """Exception raised for pipeline configuration errors."""
    pass


class PipelineRegistry:
    """
    Registry for managing pipeline configurations and stage implementations.
    
    This class implements the singleton pattern to provide a centralized registry
    for all pipeline components. It handles registration, validation, and creation
    of pipelines and their stages.
    
    Attributes:
        _pipeline_configs (Dict[str, Dict[str, Any]]): Registered pipeline configurations
        _stage_classes (Dict[str, Type[PipelineStage]]): Registered stage classes
        _pipeline_metadata (Dict[str, Dict[str, Any]]): Additional pipeline metadata
        _logger (logging.Logger): Logger instance
    """
    
    # Singleton instance and lock
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls) -> 'PipelineRegistry':
        """Create or return the singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(PipelineRegistry, cls).__new__(cls)
                cls._instance._pipeline_configs = {}
                cls._instance._stage_classes = {}
                cls._instance._pipeline_metadata = {}
                cls._instance._logger = logging.getLogger("pipeline.registry")
                cls._instance._initialized = False
            return cls._instance
    
    def register_pipeline(self, name: str, config: Dict[str, Any], 
                         metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a pipeline configuration.
        
        Args:
            name: Unique name for the pipeline
            config: Pipeline configuration dictionary
            metadata: Optional metadata about the pipeline
            
        Raises:
            ConfigurationError: If the configuration is invalid
        """
        with self._lock:
            # Validate the configuration first
            try:
                self._validate_pipeline_config(config)
            except ConfigurationError as e:
                self._logger.error(f"Failed to register pipeline '{name}': {str(e)}")
                raise
            
            if name in self._pipeline_configs:
                self._logger.warning(f"Pipeline '{name}' already exists, overwriting")
            
            self._pipeline_configs[name] = config
            
            if metadata:
                self._pipeline_metadata[name] = metadata
            elif name not in self._pipeline_metadata:
                self._pipeline_metadata[name] = {
                    "description": config.get("description", ""),
                    "version": config.get("version", "1.0.0"),
                    "tags": config.get("tags", [])
                }
            
            self._logger.info(f"Registered pipeline '{name}'")
    
    def register_stage(self, stage_class: Type[PipelineStage]) -> None:
        """
        Register a pipeline stage class.
        
        Args:
            stage_class: The stage class to register
            
        Raises:
            TypeError: If stage_class is not a subclass of PipelineStage
        """
        with self._lock:
            if not issubclass(stage_class, PipelineStage):
                raise TypeError(f"{stage_class.__name__} is not a subclass of PipelineStage")
            
            # Create a temporary instance to get metadata
            temp_instance = stage_class()
            stage_name = temp_instance.name
            
            if stage_name in self._stage_classes:
                self._logger.warning(f"Stage '{stage_name}' already registered, overwriting")
            
            self._stage_classes[stage_name] = stage_class
            self._logger.info(f"Registered stage '{stage_name}'")
    
    async def create_pipeline(self, name: str, 
                       override_config: Optional[Dict[str, Any]] = None) -> Pipeline:
        """
        Create a pipeline instance from a registered configuration.
        
        Args:
            name: The name of the registered pipeline
            override_config: Optional configuration overrides
            
        Returns:
            An initialized Pipeline instance
            
        Raises:
            PipelineNotFoundError: If the pipeline name is not registered
            StageNotFoundError: If a required stage is not registered
            ConfigurationError: If the configuration is invalid
        """
        with self._lock:
            if name not in self._pipeline_configs:
                raise PipelineNotFoundError(f"Pipeline '{name}' not found")
            
            # Get the base configuration
            config = self._pipeline_configs[name].copy()
            
            # Apply overrides if provided
            if override_config:
                config = self._merge_configurations(config, override_config)
                try:
                    self._validate_pipeline_config(config)
                except ConfigurationError as e:
                    self._logger.error(f"Invalid configuration for pipeline '{name}': {str(e)}")
                    raise
            
            # Create the pipeline
            pipeline = Pipeline(name, config.get("pipeline_config", {}))
            
            # Add stages to the pipeline
            for stage_config in config.get("stages", []):
                stage_name = stage_config["stage"]
                stage_config_dict = stage_config.get("config", {})
                
                if stage_name not in self._stage_classes:
                    raise StageNotFoundError(f"Stage '{stage_name}' not found for pipeline '{name}'")
                
                stage_class = self._stage_classes[stage_name]
                # Pass the configuration directly to the stage constructor
                stage_instance = stage_class(name=stage_name, config=stage_config_dict)
                pipeline.add_stage(stage_instance)
            
            return pipeline
    
    def get_registered_pipelines(self) -> List[str]:
        """
        Get a list of all registered pipeline names.
        
        Returns:
            List of pipeline names
        """
        with self._lock:
            return list(self._pipeline_configs.keys())
    
    def get_pipeline_info(self, name: str) -> Dict[str, Any]:
        """
        Get detailed information about a registered pipeline.
        
        Args:
            name: The name of the pipeline
            
        Returns:
            Dictionary with pipeline information
            
        Raises:
            PipelineNotFoundError: If the pipeline name is not registered
        """
        with self._lock:
            if name not in self._pipeline_configs:
                raise PipelineNotFoundError(f"Pipeline '{name}' not found")
            
            config = self._pipeline_configs[name]
            metadata = self._pipeline_metadata.get(name, {})
            
            # Get information about the stages used
            stages_info = []
            for stage_config in config.get("stages", []):
                stage_name = stage_config["stage"]
                stage_class = self._stage_classes.get(stage_name)
                
                if stage_class:
                    # Try to get metadata from the stage class
                    temp_instance = stage_class()
                    stage_metadata = temp_instance.get_metadata()
                    stages_info.append({
                        "name": stage_name,
                        "description": stage_metadata.get("description", ""),
                        "config": stage_config.get("config", {})
                    })
                else:
                    # Stage is referenced but not registered
                    stages_info.append({
                        "name": stage_name,
                        "description": "Stage not registered",
                        "config": stage_config.get("config", {})
                    })
            
            return {
                "name": name,
                "description": metadata.get("description", ""),
                "version": metadata.get("version", "1.0.0"),
                "tags": metadata.get("tags", []),
                "stages": stages_info,
                "pipeline_config": config.get("pipeline_config", {})
            }
    
    def get_registered_stages(self) -> List[str]:
        """
        Get a list of all registered stage names.
        
        Returns:
            List of stage names
        """
        with self._lock:
            return list(self._stage_classes.keys())
    
    def get_stage_info(self, name: str) -> Dict[str, Any]:
        """
        Get detailed information about a registered stage.
        
        Args:
            name: The name of the stage
            
        Returns:
            Dictionary with stage information
            
        Raises:
            StageNotFoundError: If the stage name is not registered
        """
        with self._lock:
            if name not in self._stage_classes:
                raise StageNotFoundError(f"Stage '{name}' not found")
            
            stage_class = self._stage_classes[name]
            temp_instance = stage_class()
            return temp_instance.get_metadata()
    
    def load_pipeline_config_from_file(self, path: str) -> Dict[str, Any]:
        """
        Load a pipeline configuration from a file.
        
        Args:
            path: Path to the configuration file (JSON or YAML)
            
        Returns:
            The loaded configuration
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ConfigurationError: If the file format is invalid
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        try:
            with open(path, 'r') as f:
                if path.endswith('.json'):
                    config = json.load(f)
                elif path.endswith(('.yaml', '.yml')):
                    config = yaml.safe_load(f)
                else:
                    raise ConfigurationError(f"Unsupported file format: {path}")
            
            # Validate the loaded configuration
            self._validate_pipeline_config(config)
            return config
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in {path}: {str(e)}")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {path}: {str(e)}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration from {path}: {str(e)}")
    
    def register_pipeline_from_file(self, name: str, path: str) -> None:
        """
        Load and register a pipeline configuration from a file.
        
        Args:
            name: Name for the pipeline
            path: Path to the configuration file
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ConfigurationError: If the configuration is invalid
        """
        config = self.load_pipeline_config_from_file(path)
        self.register_pipeline(name, config)
    
    def clear(self) -> None:
        """Clear all registered pipelines and stages."""
        with self._lock:
            self._pipeline_configs.clear()
            self._stage_classes.clear()
            self._pipeline_metadata.clear()
            self._logger.info("Pipeline registry cleared")
    
    def _validate_pipeline_config(self, config: Dict[str, Any]) -> None:
        """
        Validate a pipeline configuration.
        
        Args:
            config: The pipeline configuration to validate
            
        Raises:
            ConfigurationError: If the configuration is invalid
        """
        # Check for required fields
        if not isinstance(config, dict):
            raise ConfigurationError("Pipeline configuration must be a dictionary")
        
        # Validate pipeline configuration section
        if "pipeline_config" in config and not isinstance(config["pipeline_config"], dict):
            raise ConfigurationError("pipeline_config must be a dictionary")
        
        # Validate stages section
        if "stages" not in config:
            raise ConfigurationError("Pipeline configuration missing 'stages' section")
        
        if not isinstance(config["stages"], list):
            raise ConfigurationError("'stages' must be a list")
        
        # Validate each stage configuration
        for i, stage_config in enumerate(config["stages"]):
            if not isinstance(stage_config, dict):
                raise ConfigurationError(f"Stage {i} configuration must be a dictionary")
            
            if "stage" not in stage_config:
                raise ConfigurationError(f"Stage {i} missing 'stage' name")
            
            # Validate stage configuration if present
            if "config" in stage_config and not isinstance(stage_config["config"], dict):
                raise ConfigurationError(f"Stage {i} 'config' must be a dictionary")
            
            # Check for registered stages only if we have them registered
            # (allows validation without requiring registration)
            stage_name = stage_config["stage"]
            if stage_name in self._stage_classes:
                # Validate against schema if the stage has one
                stage_class = self._stage_classes[stage_name]
                temp_instance = stage_class()
                schema = temp_instance.get_config_schema()
                
                if schema and "config" in stage_config:
                    self._validate_against_schema(stage_config["config"], schema, f"Stage {stage_name}")
        
        # Check for pipeline structure (connections between stages)
        # This would depend on how stages connect to each other, 
        # which could be explicit or implicit based on your design
    
    def _validate_against_schema(self, config: Dict[str, Any], 
                                schema: Dict[str, Any], 
                                context: str) -> None:
        """
        Validate a configuration against a JSON schema.
        
        Args:
            config: The configuration to validate
            schema: The JSON schema to validate against
            context: Context for error messages
            
        Raises:
            ConfigurationError: If validation fails
        """
        # Simple implementation - could use a full JSON schema validator like jsonschema
        # Check required fields
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in config:
                raise ConfigurationError(f"{context}: Missing required field '{field}'")
        
        # Check property types
        properties = schema.get("properties", {})
        for field, field_schema in properties.items():
            if field in config:
                field_type = field_schema.get("type")
                if field_type == "string" and not isinstance(config[field], str):
                    raise ConfigurationError(f"{context}: Field '{field}' must be a string")
                elif field_type == "number" and not isinstance(config[field], (int, float)):
                    raise ConfigurationError(f"{context}: Field '{field}' must be a number")
                elif field_type == "integer" and not isinstance(config[field], int):
                    raise ConfigurationError(f"{context}: Field '{field}' must be an integer")
                elif field_type == "boolean" and not isinstance(config[field], bool):
                    raise ConfigurationError(f"{context}: Field '{field}' must be a boolean")
                elif field_type == "array" and not isinstance(config[field], list):
                    raise ConfigurationError(f"{context}: Field '{field}' must be an array")
                elif field_type == "object" and not isinstance(config[field], dict):
                    raise ConfigurationError(f"{context}: Field '{field}' must be an object")
    
    def _merge_configurations(self, base: Dict[str, Any], 
                             override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configurations, with override taking precedence.
        
        Args:
            base: Base configuration
            override: Configuration to override with
            
        Returns:
            Merged configuration
        """
        result = base.copy()
        
        # Merge pipeline_config
        if "pipeline_config" in override:
            if "pipeline_config" not in result:
                result["pipeline_config"] = {}
            result["pipeline_config"].update(override["pipeline_config"])
        
        # Handle stages specially
        if "stages" in override:
            # Replace stages completely if specified in override
            result["stages"] = override["stages"]
        
        # Merge top-level fields
        for key, value in override.items():
            if key not in ["pipeline_config", "stages"]:
                result[key] = value
        
        return result