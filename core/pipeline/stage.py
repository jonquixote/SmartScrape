"""
Pipeline Stage Module.

This module defines the abstract PipelineStage class that all pipeline stages must implement.
"""

import abc
import time
import logging
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic

from core.pipeline.context import PipelineContext

T = TypeVar('T')


class PipelineStage(abc.ABC):
    """
    Abstract base class for all pipeline stages.
    
    A pipeline stage represents a discrete processing step within a pipeline.
    Each stage takes a context object, performs some processing, and updates the context.
    
    Attributes:
        name (str): The name of this stage.
        logger (logging.Logger): Logger for this stage.
        config (Dict[str, Any]): Configuration parameters for this stage.
    """
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize a new pipeline stage.
        
        Args:
            name (Optional[str]): Name of the stage, defaults to class name if not provided.
            config (Optional[Dict[str, Any]]): Configuration parameters for this stage.
        """
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f"pipeline.stage.{self.name}")
        self.config = config or {}
    
    @abc.abstractmethod
    async def process(self, context: PipelineContext) -> bool:
        """
        Process the current stage.
        
        This is the main method that stages must implement to perform their processing logic.
        
        Args:
            context (PipelineContext): The shared pipeline context.
            
        Returns:
            bool: True if processing succeeded, False otherwise.
        """
        pass
    
    async def validate_input(self, context: PipelineContext) -> bool:
        """
        Validate that the context contains all required inputs for this stage.
        
        Override this method to implement custom input validation logic.
        
        Args:
            context (PipelineContext): The shared pipeline context.
            
        Returns:
            bool: True if validation passes, False otherwise.
        """
        return True
    
    async def validate_output(self, context: PipelineContext) -> bool:
        """
        Validate that the context contains all expected outputs after processing.
        
        Override this method to implement custom output validation logic.
        
        Args:
            context (PipelineContext): The shared pipeline context.
            
        Returns:
            bool: True if validation passes, False otherwise.
        """
        return True
    
    async def pre_process(self, context: PipelineContext) -> None:
        """
        Pre-processing hook executed before the main process method.
        
        Override this method to implement custom pre-processing logic.
        
        Args:
            context (PipelineContext): The shared pipeline context.
        """
        pass
    
    async def post_process(self, context: PipelineContext, success: bool) -> None:
        """
        Post-processing hook executed after the main process method.
        
        Override this method to implement custom post-processing logic.
        
        Args:
            context (PipelineContext): The shared pipeline context.
            success (bool): Whether the process method succeeded.
        """
        pass
    
    async def initialize(self) -> None:
        """
        Initialize resources needed by this stage.
        
        This method is called when the pipeline starts, before any stages are executed.
        Override this method to implement resource initialization.
        """
        pass
    
    async def cleanup(self) -> None:
        """
        Clean up resources used by this stage.
        
        This method is called when the pipeline ends, after all stages are executed.
        Override this method to implement resource cleanup.
        """
        pass
    
    async def handle_error(self, context: PipelineContext, error: Exception) -> bool:
        """
        Handle an error that occurred during processing.
        
        Override this method to implement custom error handling logic.
        
        Args:
            context (PipelineContext): The shared pipeline context.
            error (Exception): The exception that occurred.
            
        Returns:
            bool: True if the error was handled and processing should continue,
                 False if the pipeline should abort.
        """
        self.logger.error(f"Error in stage {self.name}: {str(error)}", exc_info=True)
        context.add_error(self.name, str(error))
        return False
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about this stage for registration and documentation.
        
        Returns:
            Dict[str, Any]: Dictionary containing stage metadata.
        """
        return {
            "name": self.name,
            "description": self.__doc__,
            "config_schema": self.get_config_schema()
        }
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for this stage's configuration.
        
        Override this method to provide a configuration schema for validation.
        
        Returns:
            Dict[str, Any]: Dictionary containing JSON schema.
        """
        return {}