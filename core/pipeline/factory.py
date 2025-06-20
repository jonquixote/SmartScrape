"""
Pipeline Factory Module.

This module provides the PipelineFactory class for creating and managing pipeline
instances from various configuration sources, with support for validation,
composition, versioning, and conditional construction.
"""

import json
import logging
import os
import importlib
import inspect
from typing import Any, Callable, Dict, List, Optional, Type, Union, cast, TYPE_CHECKING

from core.pipeline.pipeline import Pipeline, PipelineError
from core.pipeline.stage import PipelineStage
from core.pipeline.registry import PipelineRegistry
from core.pipeline.context import PipelineContext
from core.service_interface import BaseService

if TYPE_CHECKING:
    from core.pipeline.builder import PipelineBuilder


logger = logging.getLogger(__name__)


class PipelineFactoryError(Exception):
    """Base exception for pipeline factory errors."""
    pass


class ConfigurationValidationError(PipelineFactoryError):
    """Exception raised when pipeline configuration validation fails."""
    pass


class TemplateNotFoundError(PipelineFactoryError):
    """Exception raised when a requested template is not found."""
    pass


class VersionCompatibilityError(PipelineFactoryError):
    """Exception raised when version compatibility check fails."""
    pass


class PipelineFactory(BaseService):
    """
    Factory for creating and managing pipeline instances.
    
    This class provides a comprehensive API for pipeline creation from various
    sources, configuration validation, versioning, and conditional construction.
    It integrates with the PipelineRegistry for stage and pipeline management.
    
    Attributes:
        registry (PipelineRegistry): The pipeline registry instance
        template_paths (List[str]): Paths to search for pipeline templates
        logger (logging.Logger): Logger instance
    """
    
    def __init__(self, registry: Optional[PipelineRegistry] = None) -> None:
        """
        Initialize the pipeline factory.
        
        Args:
            registry: Optional pipeline registry instance
        """
        self.registry = registry or PipelineRegistry()
        self.template_paths = [
            "core/pipeline/templates",
            "custom/pipeline/templates"
        ]
        self.logger = logging.getLogger("pipeline.factory")
        
        # Dynamic import cache for template modules
        self._template_modules = {}
        self._is_initialized = False
    
    @property
    def name(self) -> str:
        """Return the name of the service."""
        return "pipeline_factory"
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the pipeline factory service.
        
        Args:
            config: Optional configuration dictionary
        """
        if self._is_initialized:
            return
            
        # Ensure registry is initialized
        if hasattr(self.registry, 'initialize') and not getattr(self.registry, 'is_initialized', False):
            self.registry.initialize(config)
            
        # Add any additional initialization
        if config:
            # Configure template paths if specified
            if "template_paths" in config:
                self.template_paths = config["template_paths"]
        
        self._is_initialized = True
        self.logger.info("Pipeline factory initialized")
    
    def shutdown(self) -> None:
        """Shutdown the pipeline factory service."""
        if not self._is_initialized:
            return
            
        # Clear caches
        self._template_modules.clear()
        
        self._is_initialized = False
        self.logger.info("Pipeline factory shut down")
        
    async def create_pipeline_from_config(self, config: Dict[str, Any], 
                                   name: Optional[str] = None) -> Pipeline:
        """
        Create a pipeline from a configuration dictionary.
        
        Args:
            config: Pipeline configuration
            name: Optional name for the pipeline
            
        Returns:
            Initialized Pipeline instance
            
        Raises:
            ConfigurationValidationError: If the configuration is invalid
        """
        # Validate the configuration
        try:
            self.validate_pipeline_config(config)
        except ConfigurationValidationError as e:
            self.logger.error(f"Invalid pipeline configuration: {str(e)}")
            raise
        
        pipeline_name = name or config.get("name", f"pipeline_{id(config)}")
        
        # Register the configuration temporarily if not already registered
        if pipeline_name not in self.registry.get_registered_pipelines():
            self.registry.register_pipeline(pipeline_name, config)
            
        # Create the pipeline from the registry
        return await self.registry.create_pipeline(pipeline_name)
        
    async def create_pipeline_from_file(self, file_path: str, 
                                 name: Optional[str] = None,
                                 override_config: Optional[Dict[str, Any]] = None) -> Pipeline:
        """
        Create a pipeline from a configuration file.
        
        Args:
            file_path: Path to configuration file (JSON or YAML)
            name: Optional name for the pipeline
            override_config: Optional configuration overrides
            
        Returns:
            Initialized Pipeline instance
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ConfigurationValidationError: If the configuration is invalid
        """
        try:
            # Load the configuration
            config = self.registry.load_pipeline_config_from_file(file_path)
            
            # Apply overrides
            if override_config:
                config = self.merge_configurations(config, override_config)
                
            pipeline_name = name or os.path.basename(file_path).split('.')[0]
            
            # Create the pipeline
            return await self.create_pipeline_from_config(config, pipeline_name)
        except Exception as e:
            if isinstance(e, FileNotFoundError):
                raise
            elif isinstance(e, ConfigurationValidationError):
                raise
            else:
                error_msg = f"Failed to create pipeline from file {file_path}: {str(e)}"
                self.logger.error(error_msg)
                raise PipelineFactoryError(error_msg) from e
                
    async def create_pipeline_from_template(self, template_name: str,
                                    params: Optional[Dict[str, Any]] = None,
                                    name: Optional[str] = None) -> Pipeline:
        """
        Create a pipeline from a template.
        
        Args:
            template_name: Name of the template (module or class)
            params: Parameters to pass to the template
            name: Optional name for the pipeline
            
        Returns:
            Initialized Pipeline instance
            
        Raises:
            TemplateNotFoundError: If template cannot be found
            ConfigurationValidationError: If generated configuration is invalid
        """
        params = params or {}
        
        # Try to import as a module with a create_* function
        template_module = self._load_template_module(template_name)
        if template_module:
            # Look for create_pipeline, get_pipeline_config, or create_*_pipeline functions
            for attr_name in dir(template_module):
                if attr_name.startswith("create_") and attr_name.endswith("_pipeline"):
                    factory_func = getattr(template_module, attr_name)
                    if callable(factory_func):
                        try:
                            # Try to call the factory function
                            pipeline = await self._call_async_or_sync(factory_func, params)
                            
                            # If the function returns a Pipeline instance, use it
                            if isinstance(pipeline, Pipeline):
                                if name:
                                    pipeline.name = name
                                return pipeline
                            # If it returns a configuration dict, create from that
                            elif isinstance(pipeline, dict):
                                return await self.create_pipeline_from_config(pipeline, name)
                        except Exception as e:
                            self.logger.warning(
                                f"Error calling {attr_name} in {template_name}: {str(e)}"
                            )
                                
        raise TemplateNotFoundError(f"Cannot find or use template '{template_name}'")
        
    async def create_custom_pipeline(self, 
                             builder_func: Callable[..., Union[Pipeline, Dict[str, Any]]],
                             params: Optional[Dict[str, Any]] = None,
                             name: Optional[str] = None) -> Pipeline:
        """
        Create a pipeline using a custom builder function.
        
        Args:
            builder_func: Function that builds a pipeline or configuration
            params: Parameters to pass to the builder function
            name: Optional name for the pipeline
            
        Returns:
            Initialized Pipeline instance
            
        Raises:
            PipelineFactoryError: If pipeline creation fails
        """
        params = params or {}
        
        try:
            # Call the builder function
            result = await self._call_async_or_sync(builder_func, params)
            
            # If it returns a Pipeline instance, use it
            if isinstance(result, Pipeline):
                if name:
                    result.name = name
                return result
            # If it returns a configuration dict, create from that
            elif isinstance(result, dict):
                return await self.create_pipeline_from_config(result, name)
            else:
                raise PipelineFactoryError(
                    f"Builder function returned {type(result)}, expected Pipeline or dict"
                )
        except Exception as e:
            error_msg = f"Failed to create custom pipeline: {str(e)}"
            self.logger.error(error_msg)
            raise PipelineFactoryError(error_msg) from e
            
    def get_pipeline_builder(self) -> 'PipelineBuilder':
        """
        Get a fluent builder interface for pipeline construction.
        
        Returns:
            New PipelineBuilder instance
        """
        # Import locally to avoid circular import
        from core.pipeline.builder import PipelineBuilder
        return PipelineBuilder(self)
        
    def validate_pipeline_config(self, config: Dict[str, Any]) -> None:
        """
        Perform comprehensive validation of a pipeline configuration.
        
        Args:
            config: Pipeline configuration to validate
            
        Raises:
            ConfigurationValidationError: If validation fails
        """
        # First, use the registry's validation
        try:
            self.registry._validate_pipeline_config(config)
        except Exception as e:
            raise ConfigurationValidationError(f"Basic validation failed: {str(e)}")
            
        # Additional validations beyond what registry provides
            
        # Check version compatibility if specified
        if "version" in config:
            try:
                self._check_version_compatibility(config["version"])
            except VersionCompatibilityError as e:
                raise ConfigurationValidationError(str(e))
                
        # Validate stage connections and flow if the design supports it
        self._validate_pipeline_flow(config)
        
        # Validate placeholders - ensure all can be resolved
        self._validate_placeholders(config)
        
    def merge_configurations(self, base: Dict[str, Any], 
                           override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configurations, with deep merging of dictionaries.
        
        Args:
            base: Base configuration
            override: Configuration to override with
            
        Returns:
            Merged configuration
        """
        # Start with the registry's implementation
        result = self.registry._merge_configurations(base, override)
        
        # Add deep merging of nested dictionaries if needed
        # This is a placeholder - the registry implementation might be sufficient
        
        return result
        
    def resolve_placeholder(self, value: Any, context: Dict[str, Any]) -> Any:
        """
        Resolve placeholder values in configuration using context.
        
        Supports ${variable} format for placeholder substitution.
        
        Args:
            value: Value which might contain placeholders
            context: Context containing values for substitution
            
        Returns:
            Resolved value
        """
        if isinstance(value, str) and "${" in value and "}" in value:
            # Simple placeholder resolution
            result = value
            start_idx = result.find("${")
            while start_idx >= 0:
                end_idx = result.find("}", start_idx)
                if end_idx > start_idx:
                    placeholder = result[start_idx+2:end_idx]
                    if placeholder in context:
                        replacement = str(context[placeholder])
                        result = result[:start_idx] + replacement + result[end_idx+1:]
                        # Update the start_idx for the next search
                        start_idx = result.find("${")
                    else:
                        # Skip this placeholder if not found
                        start_idx = result.find("${", start_idx + 2)
                else:
                    # No closing brace, exit loop
                    break
            return result
        elif isinstance(value, dict):
            # Recursively process dictionaries
            return {k: self.resolve_placeholder(v, context) for k, v in value.items()}
        elif isinstance(value, list):
            # Recursively process lists
            return [self.resolve_placeholder(item, context) for item in value]
        else:
            # Return other types unchanged
            return value
            
    def analyze_pipeline(self, pipeline: Pipeline) -> Dict[str, Any]:
        """
        Analyze a pipeline and return detailed information.
        
        This can be used for visualization, debugging, and performance analysis.
        
        Args:
            pipeline: Pipeline to analyze
            
        Returns:
            Analysis results including stage dependencies, execution paths, etc.
        """
        # Collect basic information
        pipeline_info = {
            "name": pipeline.name,
            "stage_count": len(pipeline.stages),
            "stages": [],
            "execution_paths": [],
            "potential_bottlenecks": [],
            "parallel_execution": pipeline.parallel_execution,
            "max_workers": pipeline.max_workers
        }
        
        # Analyze each stage
        for i, stage in enumerate(pipeline.stages):
            stage_info = {
                "name": stage.name,
                "position": i,
                "type": stage.__class__.__name__,
                "description": stage.__doc__ or "No description available"
            }
            
            # Add to the stages list
            pipeline_info["stages"].append(stage_info)
            
        # Identify linear execution path
        pipeline_info["execution_paths"].append({
            "name": "main_path",
            "description": "Primary linear execution path",
            "stages": [stage["name"] for stage in pipeline_info["stages"]]
        })
        
        # Note: More advanced analysis would be added here based on the
        # actual design of the pipeline system, such as:
        # - Branching/conditional paths
        # - Stage dependencies
        # - Potential parallelizable groups
        # - Expected execution time based on profiling data
            
        return pipeline_info
        
    def _load_template_module(self, template_name: str) -> Any:
        """
        Load a template module by name.
        
        Args:
            template_name: Name of the template
            
        Returns:
            Loaded module or None if not found
        """
        # Check cache first
        if template_name in self._template_modules:
            return self._template_modules[template_name]
            
        # Try to load from various locations
        for base_path in self.template_paths:
            # Try as direct module
            module_name = f"{base_path.replace('/', '.')}.{template_name}"
            try:
                module = importlib.import_module(module_name)
                self._template_modules[template_name] = module
                return module
            except ImportError:
                pass
                
            # Try with _pipeline suffix
            module_name = f"{base_path.replace('/', '.')}.{template_name}_pipeline"
            try:
                module = importlib.import_module(module_name)
                self._template_modules[template_name] = module
                return module
            except ImportError:
                pass
                
        # Not found
        return None
        
    async def _call_async_or_sync(self, func: Callable, params: Dict[str, Any]) -> Any:
        """
        Call a function that might be async or sync.
        
        Args:
            func: Function to call
            params: Parameters to pass
            
        Returns:
            Result of the function call
        """
        # Check if the function is a coroutine function
        if inspect.iscoroutinefunction(func):
            # Call async function
            return await func(**params)
        else:
            # Call sync function
            return func(**params)
            
    def _check_version_compatibility(self, version: str) -> None:
        """
        Check if a pipeline version is compatible with this factory.
        
        Args:
            version: Version string to check
            
        Raises:
            VersionCompatibilityError: If versions are incompatible
        """
        # Simple version check - could use semantic versioning
        supported_versions = ["1.0.0", "1.1.0", "2.0.0"]
        if version not in supported_versions:
            raise VersionCompatibilityError(
                f"Pipeline version {version} is not supported. "
                f"Supported versions: {', '.join(supported_versions)}"
            )
            
    def _validate_pipeline_flow(self, config: Dict[str, Any]) -> None:
        """
        Validate the flow of a pipeline configuration.
        
        Args:
            config: Pipeline configuration
            
        Raises:
            ConfigurationValidationError: If flow validation fails
        """
        # Validate stage connections if explicitly defined
        if "connections" in config:
            connections = config["connections"]
            stage_names = [stage["stage"] for stage in config.get("stages", [])]
            
            for conn in connections:
                from_stage = conn.get("from")
                to_stage = conn.get("to")
                
                if from_stage not in stage_names:
                    raise ConfigurationValidationError(
                        f"Connection references non-existent 'from' stage: {from_stage}"
                    )
                    
                if to_stage not in stage_names:
                    raise ConfigurationValidationError(
                        f"Connection references non-existent 'to' stage: {to_stage}"
                    )
        
        # Default sequential flow doesn't need validation
        # More complex flow validation would be implemented based on the
        # specific flow control mechanisms of your pipeline architecture
        
    def _validate_placeholders(self, config: Dict[str, Any]) -> None:
        """
        Validate that all placeholders in a configuration have valid references.
        
        Args:
            config: Pipeline configuration
            
        Raises:
            ConfigurationValidationError: If placeholder validation fails
        """
        # Extract all available context keys that could be used in placeholders
        # This is a simplified implementation - would need customization based on
        # the actual placeholder system used
        context_keys = set()
        
        # Add standard context variables
        context_keys.update(["pipeline_name", "timestamp", "environment"])
        
        # Add variables from config
        if "variables" in config:
            context_keys.update(config["variables"].keys())
            
        # Scan configuration for placeholder references
        def check_value(value: Any, path: str) -> None:
            """Recursively check values for placeholders"""
            if isinstance(value, str) and "${" in value and "}" in value:
                start_idx = value.find("${")
                while start_idx >= 0:
                    end_idx = value.find("}", start_idx)
                    if end_idx > start_idx:
                        placeholder = value[start_idx+2:end_idx]
                        if placeholder not in context_keys:
                            raise ConfigurationValidationError(
                                f"Undefined placeholder '${placeholder}' at {path}"
                            )
                        start_idx = value.find("${", end_idx)
                    else:
                        # Malformed placeholder
                        raise ConfigurationValidationError(
                            f"Malformed placeholder in '{value}' at {path}"
                        )
            elif isinstance(value, dict):
                for k, v in value.items():
                    check_value(v, f"{path}.{k}" if path else k)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    check_value(item, f"{path}[{i}]")
                    
        # Start validation from the root
        try:
            check_value(config, "")
        except ConfigurationValidationError as e:
            raise ConfigurationValidationError(
                f"Placeholder validation failed: {str(e)}"
            )