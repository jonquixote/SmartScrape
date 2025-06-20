"""
Pipeline Service Module

This module provides a service layer for managing extraction pipelines.
It handles pipeline creation, execution, and monitoring.
"""

import logging
import asyncio
import time
from typing import Dict, Any, Optional, List, Set, Tuple, Union

from core.pipeline.pipeline import Pipeline
from core.pipeline.context import PipelineContext
from core.service_interface import BaseService
from extraction.pipeline_registry import get_registry, initialize_registry, register_built_in_stages, register_built_in_pipelines

logger = logging.getLogger(__name__)

class PipelineService(BaseService):
    """
    Service for creating and executing extraction pipelines.
    
    This service provides:
    - Creation of pipelines from templates or custom configurations
    - Execution of pipelines with monitoring and error handling
    - Pipeline lifecycle management (initialization, cleanup)
    - Access to pipeline execution statistics
    """
    
    def __init__(self):
        """Initialize the pipeline service."""
        self._active_pipelines: Dict[str, Pipeline] = {}
        self._pipeline_stats: Dict[str, Dict[str, Any]] = {}
        self._is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize the pipeline service and registry."""
        if self._is_initialized:
            return
        
        try:
            # Initialize the pipeline registry
            register_built_in_stages()
            await initialize_registry()
            register_built_in_pipelines()
            
            self._is_initialized = True
            logger.info("Pipeline service initialized")
        except Exception as e:
            logger.error(f"Error initializing pipeline service: {str(e)}")
            raise
    
    async def shutdown(self) -> None:
        """Shut down all pipelines and clean up resources."""
        cleanup_tasks = []
        
        # Clean up all active pipelines
        for pipeline_id, pipeline in self._active_pipelines.items():
            logger.debug(f"Cleaning up pipeline {pipeline_id}")
            cleanup_tasks.append(pipeline.cleanup())
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        self._active_pipelines.clear()
        self._pipeline_stats.clear()
        self._is_initialized = False
        
        logger.info("Pipeline service shut down")
    
    async def create_pipeline(self, pipeline_type: str, config: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a pipeline of the specified type.
        
        Args:
            pipeline_type: Type of pipeline to create
            config: Optional configuration overrides
            
        Returns:
            ID of the created pipeline
            
        Raises:
            ValueError: If pipeline type is not found or creation fails
        """
        if not self._is_initialized:
            await self.initialize()
        
        # Get the pipeline registry
        registry = get_registry()
        
        # Create the pipeline
        pipeline = registry.create_pipeline(pipeline_type, config)
        if not pipeline:
            raise ValueError(f"Failed to create pipeline of type '{pipeline_type}'")
        
        # Generate a unique ID for this pipeline
        pipeline_id = f"{pipeline_type}_{int(time.time())}_{id(pipeline):x}"
        
        # Initialize the pipeline
        try:
            await pipeline.initialize()
        except Exception as e:
            logger.error(f"Error initializing pipeline {pipeline_id}: {str(e)}")
            await pipeline.cleanup()
            raise ValueError(f"Pipeline initialization failed: {str(e)}")
        
        # Store the pipeline
        self._active_pipelines[pipeline_id] = pipeline
        self._pipeline_stats[pipeline_id] = {
            "created_at": time.time(),
            "pipeline_type": pipeline_type,
            "execution_count": 0,
            "success_count": 0,
            "error_count": 0,
            "last_execution": None,
            "avg_execution_time": 0,
            "total_execution_time": 0
        }
        
        logger.info(f"Created pipeline {pipeline_id} of type '{pipeline_type}'")
        return pipeline_id
    
    async def execute_pipeline(self, pipeline_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a pipeline with the given input data.
        
        Args:
            pipeline_id: ID of the pipeline to execute
            input_data: Input data for the pipeline
            
        Returns:
            Pipeline execution results
            
        Raises:
            ValueError: If pipeline ID is not found
        """
        # Get the pipeline
        pipeline = self._active_pipelines.get(pipeline_id)
        if not pipeline:
            raise ValueError(f"Pipeline '{pipeline_id}' not found")
        
        # Create a context with the input data
        context = PipelineContext()
        for key, value in input_data.items():
            context.set(key, value)
        
        # Execute the pipeline
        stats = self._pipeline_stats[pipeline_id]
        start_time = time.time()
        
        try:
            success = await pipeline.process(context)
            
            # Update pipeline statistics
            execution_time = time.time() - start_time
            stats["execution_count"] += 1
            stats["last_execution"] = time.time()
            stats["total_execution_time"] += execution_time
            stats["avg_execution_time"] = stats["total_execution_time"] / stats["execution_count"]
            
            if success:
                stats["success_count"] += 1
            else:
                stats["error_count"] += 1
            
            # Collect output data
            output_data = self._collect_output_data(context)
            
            # Add execution metadata
            output_data["_execution_metadata"] = {
                "pipeline_id": pipeline_id,
                "execution_time": execution_time,
                "success": success,
                "errors": context.get_errors()
            }
            
            return output_data
            
        except Exception as e:
            logger.error(f"Error executing pipeline {pipeline_id}: {str(e)}")
            stats["error_count"] += 1
            stats["execution_count"] += 1
            stats["last_execution"] = time.time()
            
            raise RuntimeError(f"Pipeline execution failed: {str(e)}")
    
    async def release_pipeline(self, pipeline_id: str) -> bool:
        """
        Release a pipeline and free its resources.
        
        Args:
            pipeline_id: ID of the pipeline to release
            
        Returns:
            True if pipeline was released, False otherwise
        """
        pipeline = self._active_pipelines.get(pipeline_id)
        if not pipeline:
            logger.warning(f"Pipeline '{pipeline_id}' not found for release")
            return False
        
        try:
            # Clean up the pipeline
            await pipeline.cleanup()
            
            # Remove from active pipelines
            del self._active_pipelines[pipeline_id]
            
            logger.info(f"Released pipeline {pipeline_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error releasing pipeline {pipeline_id}: {str(e)}")
            return False
    
    def get_pipeline_stats(self, pipeline_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for pipelines.
        
        Args:
            pipeline_id: Optional ID of a specific pipeline
            
        Returns:
            Dictionary of pipeline statistics
        """
        if pipeline_id:
            return self._pipeline_stats.get(pipeline_id, {})
        
        return {
            "active_pipelines": len(self._active_pipelines),
            "pipelines": self._pipeline_stats
        }
    
    def list_available_pipeline_types(self) -> List[str]:
        """
        Get a list of available pipeline types.
        
        Returns:
            List of pipeline type names
        """
        registry = get_registry()
        return registry.list_pipeline_types()
    
    def list_available_stages(self) -> List[str]:
        """
        Get a list of available pipeline stage types.
        
        Returns:
            List of stage type names
        """
        registry = get_registry()
        return registry.list_stage_types()
    
    def _collect_output_data(self, context: PipelineContext) -> Dict[str, Any]:
        """
        Collect output data from the pipeline context.
        
        Args:
            context: Pipeline context with execution results
            
        Returns:
            Dictionary of output data
        """
        output_data = {}
        
        # Get all keys in the context
        for key in context.get_all_keys():
            # Skip internal keys (those starting with underscore)
            if not key.startswith('_'):
                value = context.get(key)
                output_data[key] = value
        
        # Add errors if any
        errors = context.get_errors()
        if errors:
            output_data["_errors"] = errors
        
        return output_data


# Singleton instance for global access
_service = None

def get_pipeline_service() -> PipelineService:
    """
    Get the global pipeline service instance.
    
    Returns:
        Global PipelineService instance
    """
    global _service
    if _service is None:
        _service = PipelineService()
    return _service

async def initialize_pipeline_service() -> None:
    """Initialize the global pipeline service."""
    service = get_pipeline_service()
    await service.initialize()

async def create_extraction_pipeline(pipeline_type: str = "default_extraction", 
                                   config: Optional[Dict[str, Any]] = None) -> str:
    """
    Create an extraction pipeline of the specified type.
    
    Args:
        pipeline_type: Type of extraction pipeline to create
        config: Optional configuration overrides
        
    Returns:
        ID of the created pipeline
    """
    service = get_pipeline_service()
    
    # Initialize if needed
    if not service._is_initialized:
        await service.initialize()
    
    return await service.create_pipeline(pipeline_type, config)

async def execute_extraction(pipeline_id: str, url: str, html_content: Optional[str] = None,
                          options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Execute an extraction pipeline on the given URL or HTML content.
    
    Args:
        pipeline_id: ID of the pipeline to execute
        url: URL to extract from
        html_content: Optional pre-fetched HTML content
        options: Optional extraction options
        
    Returns:
        Extraction results
    """
    service = get_pipeline_service()
    
    # Create input data
    input_data = {
        "url": url,
        "extraction_options": options or {}
    }
    
    if html_content:
        input_data["html_content"] = html_content
    
    # Execute the pipeline
    return await service.execute_pipeline(pipeline_id, input_data)