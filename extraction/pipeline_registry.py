"""
Pipeline Registry Module

This module provides a central registry for pipeline stages and complete pipelines.
It allows for dynamic registration, lookup, and instantiation of pipeline components.
"""

import logging
from typing import Dict, Any, Optional, List, Type, Callable, Union, Set
import importlib
import inspect
import asyncio
from concurrent.futures import ThreadPoolExecutor

from core.pipeline.pipeline import Pipeline
from core.pipeline.stage import PipelineStage
from core.pipeline.builder import PipelineBuilder
from core.pipeline.context import PipelineContext
from core.service_interface import BaseService

logger = logging.getLogger(__name__)

class PipelineRegistry(BaseService):
    """
    Central registry for pipeline stages and complete pipelines.
    
    This class provides:
    - Registration of pipeline stages by name
    - Registration of complete pipelines
    - Dynamic loading of stage classes from modules
    - Factory methods for instantiating stages and pipelines
    """
    
    def __init__(self):
        """Initialize the pipeline registry."""
        # Stage registries
        self._stage_classes: Dict[str, Type[PipelineStage]] = {}
        self._stage_factories: Dict[str, Callable] = {}
        
        # Pipeline registries
        self._pipeline_templates: Dict[str, Dict[str, Any]] = {}
        self._pipeline_factories: Dict[str, Callable] = {}
        
        # Module registries for auto-discovery
        self._stage_modules: Set[str] = set()
        self._is_initialized = False
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    @property
    def name(self) -> str:
        """Return the name of the service."""
        return "pipeline_registry"
    
    @property
    def is_initialized(self) -> bool:
        """Check if the service has been initialized."""
        return self._is_initialized
        
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the registry and discover components.
        
        Args:
            config: Optional configuration dict
        """
        if self._is_initialized:
            return
        
        try:
            # Auto-discover stages from registered modules
            # This is a synchronous version for BaseService compatibility
            for module_path in self._stage_modules:
                try:
                    module = importlib.import_module(module_path)
                    
                    # Find all classes in the module that inherit from PipelineStage
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, PipelineStage) and 
                            obj is not PipelineStage and
                            not name.startswith('_')):
                            # Register the stage class using its name
                            self.register_stage_class(name, obj)
                            # Also register with common naming patterns
                            if name.endswith('Stage'):
                                self.register_stage_class(name[:-5].lower(), obj)
                    
                except Exception as e:
                    logger.warning(f"Error loading pipeline stages from module {module_path}: {str(e)}")
            
            logger.info(f"Pipeline registry initialized with {len(self._stage_classes)} stage types")
            self._is_initialized = True
        except Exception as e:
            logger.error(f"Error initializing pipeline registry: {str(e)}")
            raise
    
    def shutdown(self) -> None:
        """Shutdown the registry and release resources."""
        self._executor.shutdown(wait=True)
        self._is_initialized = False
        logger.debug("Pipeline registry shut down")
    
    # Keep the async versions for backward compatibility
    async def async_initialize(self) -> None:
        """Initialize the registry asynchronously."""
        if self._is_initialized:
            return
        
        try:
            # Auto-discover stages from registered modules
            await self._discover_stages()
            logger.info(f"Pipeline registry initialized with {len(self._stage_classes)} stage types")
            self._is_initialized = True
        except Exception as e:
            logger.error(f"Error initializing pipeline registry: {str(e)}")
            raise
    
    async def async_shutdown(self) -> None:
        """Shutdown the registry asynchronously."""
        self._executor.shutdown(wait=True)
        self._is_initialized = False
        logger.debug("Pipeline registry shut down")
    
    def register_stage_class(self, stage_name: str, stage_class: Type[PipelineStage]) -> None:
        """
        Register a pipeline stage class with the registry.
        
        Args:
            stage_name: Name to register the stage under
            stage_class: The stage class to register
        """
        if not issubclass(stage_class, PipelineStage):
            raise ValueError(f"Class {stage_class.__name__} is not a subclass of PipelineStage")
        
        self._stage_classes[stage_name] = stage_class
        logger.debug(f"Registered stage class: {stage_name}")
    
    def register_stage_factory(self, stage_name: str, factory_func: Callable) -> None:
        """
        Register a factory function for creating pipeline stages.
        
        Args:
            stage_name: Name to register the factory under
            factory_func: Function that creates a stage instance
        """
        self._stage_factories[stage_name] = factory_func
        logger.debug(f"Registered stage factory: {stage_name}")
    
    def register_pipeline_template(self, template_name: str, template: Dict[str, Any]) -> None:
        """
        Register a pipeline template configuration.
        
        Args:
            template_name: Name to register the template under
            template: Pipeline configuration template
        """
        self._pipeline_templates[template_name] = template
        logger.debug(f"Registered pipeline template: {template_name}")
    
    def register_pipeline_factory(self, pipeline_name: str, factory_func: Callable) -> None:
        """
        Register a factory function for creating complete pipelines.
        
        Args:
            pipeline_name: Name to register the factory under
            factory_func: Function that creates a pipeline instance
        """
        self._pipeline_factories[pipeline_name] = factory_func
        logger.debug(f"Registered pipeline factory: {pipeline_name}")
    
    def register_module(self, module_path: str) -> None:
        """
        Register a module to scan for pipeline stages.
        
        Args:
            module_path: Importable Python module path
        """
        self._stage_modules.add(module_path)
        logger.debug(f"Registered module for stage discovery: {module_path}")
    
    def get_stage_class(self, stage_name: str) -> Optional[Type[PipelineStage]]:
        """
        Get a registered stage class by name.
        
        Args:
            stage_name: Name of the stage class to retrieve
            
        Returns:
            The stage class or None if not found
        """
        return self._stage_classes.get(stage_name)
    
    def create_stage(self, stage_type: str, name: Optional[str] = None, 
                    config: Optional[Dict[str, Any]] = None) -> Optional[PipelineStage]:
        """
        Create a pipeline stage instance by type.
        
        Args:
            stage_type: Type of stage to create
            name: Optional name for the stage
            config: Optional configuration for the stage
            
        Returns:
            Instance of the requested stage or None if not found
        """
        # Try factory first if available
        if stage_type in self._stage_factories:
            try:
                return self._stage_factories[stage_type](name, config)
            except Exception as e:
                logger.error(f"Error creating stage '{stage_type}' using factory: {str(e)}")
                return None
        
        # Fall back to class instantiation
        stage_class = self.get_stage_class(stage_type)
        if not stage_class:
            logger.error(f"Stage type '{stage_type}' not found in registry")
            return None
        
        try:
            return stage_class(name=name, config=config)
        except Exception as e:
            logger.error(f"Error instantiating stage '{stage_type}': {str(e)}")
            return None
    
    def create_pipeline(self, pipeline_name: str, config: Optional[Dict[str, Any]] = None) -> Optional[Pipeline]:
        """
        Create a pipeline instance by name.
        
        Args:
            pipeline_name: Name of the pipeline to create
            config: Optional configuration overrides
            
        Returns:
            Pipeline instance or None if creation failed
        """
        # Use factory if available
        if pipeline_name in self._pipeline_factories:
            try:
                return self._pipeline_factories[pipeline_name](config)
            except Exception as e:
                logger.error(f"Error creating pipeline '{pipeline_name}' using factory: {str(e)}")
                return None
        
        # Use template if available
        if pipeline_name in self._pipeline_templates:
            try:
                return self._create_pipeline_from_template(pipeline_name, config)
            except Exception as e:
                logger.error(f"Error creating pipeline '{pipeline_name}' from template: {str(e)}")
                return None
        
        logger.error(f"Pipeline '{pipeline_name}' not found in registry")
        return None
    
    def list_stage_types(self) -> List[str]:
        """
        Get a list of all registered stage types.
        
        Returns:
            List of stage type names
        """
        return list(set(list(self._stage_classes.keys()) + list(self._stage_factories.keys())))
    
    def list_pipeline_types(self) -> List[str]:
        """
        Get a list of all registered pipeline types.
        
        Returns:
            List of pipeline type names
        """
        return list(set(list(self._pipeline_templates.keys()) + list(self._pipeline_factories.keys())))
    
    async def _discover_stages(self) -> None:
        """
        Discover and register pipeline stages from registered modules.
        """
        if not self._stage_modules:
            logger.debug("No modules registered for stage discovery")
            return
        
        for module_path in self._stage_modules:
            try:
                loop = asyncio.get_event_loop()
                module = await loop.run_in_executor(
                    self._executor, 
                    lambda: importlib.import_module(module_path)
                )
                
                # Find all classes in the module that inherit from PipelineStage
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, PipelineStage) and 
                        obj is not PipelineStage and
                        not name.startswith('_')):
                        # Register the stage class using its name
                        self.register_stage_class(name, obj)
                        # Also register with common naming patterns
                        if name.endswith('Stage'):
                            self.register_stage_class(name[:-5].lower(), obj)
                
            except Exception as e:
                logger.warning(f"Error loading pipeline stages from module {module_path}: {str(e)}")
    
    def _create_pipeline_from_template(self, template_name: str, 
                                     config_overrides: Optional[Dict[str, Any]] = None) -> Pipeline:
        """
        Create a pipeline from a template configuration.
        
        Args:
            template_name: Name of the template to use
            config_overrides: Optional configuration overrides
            
        Returns:
            Configured pipeline instance
        """
        template = self._pipeline_templates.get(template_name)
        if not template:
            raise ValueError(f"Pipeline template '{template_name}' not found")
        
        # Create a deep copy and apply overrides
        import copy
        pipeline_config = copy.deepcopy(template)
        
        # Apply configuration overrides
        if config_overrides:
            self._deep_update(pipeline_config, config_overrides)
        
        # Create pipeline using the builder
        builder = PipelineBuilder(name=pipeline_config.get('name', template_name))
        
        # Add stages
        stages_config = pipeline_config.get('stages', [])
        for stage_config in stages_config:
            stage_type = stage_config.get('type')
            stage_name = stage_config.get('name')
            stage_config = stage_config.get('config', {})
            
            stage = self.create_stage(stage_type, stage_name, stage_config)
            if stage:
                builder.add_stage(stage)
            else:
                logger.warning(f"Failed to create stage '{stage_name}' of type '{stage_type}'")
        
        # Set pipeline properties
        if 'max_retries' in pipeline_config:
            builder.set_max_retries(pipeline_config['max_retries'])
        
        if 'error_handlers' in pipeline_config:
            for handler_config in pipeline_config['error_handlers']:
                handler_type = handler_config.get('type')
                handler_config = handler_config.get('config', {})
                # Add error handler logic here
        
        return builder.build()
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """
        Recursively update a dictionary with another dictionary.
        
        Args:
            base_dict: Dictionary to update
            update_dict: Dictionary with updates
        """
        for key, value in update_dict.items():
            if (key in base_dict and isinstance(base_dict[key], dict) and
                isinstance(value, dict)):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value


# Singleton instance for global access
_registry = None

def get_registry() -> PipelineRegistry:
    """
    Get the global pipeline registry instance.
    
    Returns:
        Global PipelineRegistry instance
    """
    global _registry
    if (_registry is None) or (not _registry._is_initialized):
        _registry = PipelineRegistry()
    return _registry

async def initialize_registry() -> None:
    """Initialize the global pipeline registry."""
    registry = get_registry()
    await registry.async_initialize()

def register_built_in_stages() -> None:
    """Register built-in pipeline stages with the registry."""
    registry = get_registry()
    
    # Register extraction modules for discovery
    registry.register_module('extraction.stages')
    
    # Register core modules for discovery
    registry.register_module('core.pipeline.stages')
    
    # Register built-in stage classes directly
    from extraction.stages.normalization_stage import NormalizationStage
    from extraction.stages.content_normalization_stage import ContentNormalizationStage
    from extraction.stages.pattern_extraction_stage import PatternExtractionStage
    from extraction.stages.semantic_extraction_stage import SemanticExtractionStage
    from extraction.stages.quality_evaluation_stage import QualityEvaluationStage
    from extraction.stages.schema_validation_stage import SchemaValidationStage
    
    registry.register_stage_class('normalization', NormalizationStage)
    registry.register_stage_class('content_normalization', ContentNormalizationStage)
    registry.register_stage_class('pattern_extraction', PatternExtractionStage)
    registry.register_stage_class('semantic_extraction', SemanticExtractionStage)
    registry.register_stage_class('quality_evaluation', QualityEvaluationStage)
    registry.register_stage_class('schema_validation', SchemaValidationStage)
    
    logger.info("Registered built-in pipeline stages")

def register_built_in_pipelines() -> None:
    """Register built-in pipeline templates with the registry."""
    registry = get_registry()
    
    # Import factory methods
    from extraction.stages.pipeline_config import (
        get_extraction_pipeline, 
        ExtractionPipelineFactory
    )
    
    # Register pipeline factories
    registry.register_pipeline_factory('default_extraction', 
                                      ExtractionPipelineFactory.create_default_pipeline)
    registry.register_pipeline_factory('product_extraction', 
                                      ExtractionPipelineFactory.create_product_pipeline)
    registry.register_pipeline_factory('article_extraction', 
                                      ExtractionPipelineFactory.create_article_pipeline)
    
    logger.info("Registered built-in pipeline templates")