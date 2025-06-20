"""
Pipeline Executor Module

This module provides functions for executing extraction pipelines with error handling,
monitoring, and customization options.
"""

import logging
import asyncio
import time
from typing import Dict, Any, Optional, List, Union, Callable, Tuple

from core.pipeline.pipeline import Pipeline
from core.pipeline.context import PipelineContext
from core.monitoring import log_execution_metrics
from extraction.pipeline_service import get_pipeline_service, create_extraction_pipeline

logger = logging.getLogger(__name__)

class PipelineExecutor:
    """
    Executor for running extraction pipelines with advanced options.
    
    This class provides:
    - Simplified interface for pipeline execution
    - Error handling with fallback options
    - Execution monitoring and metrics collection
    - Support for callbacks and event hooks
    """
    
    def __init__(self, pipeline_type: str = "default_extraction", 
                pipeline_config: Optional[Dict[str, Any]] = None):
        """
        Initialize a pipeline executor.
        
        Args:
            pipeline_type: Type of pipeline to execute
            pipeline_config: Optional configuration for the pipeline
        """
        self.pipeline_type = pipeline_type
        self.pipeline_config = pipeline_config or {}
        self.pipeline_id = None
        self.before_hooks: List[Callable] = []
        self.after_hooks: List[Callable] = []
        self.error_handlers: List[Callable] = []
        self.timeout = 60  # Default timeout in seconds
        self.retry_count = 2  # Default number of retries
        self.retry_delay = 1.0  # Default delay between retries in seconds
        self.fallback_pipeline_type = None
    
    async def initialize(self) -> None:
        """Initialize and create the pipeline."""
        if self.pipeline_id:
            return
            
        try:
            self.pipeline_id = await create_extraction_pipeline(
                self.pipeline_type, 
                self.pipeline_config
            )
            logger.debug(f"Initialized pipeline executor with pipeline ID: {self.pipeline_id}")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline executor: {str(e)}")
            raise
    
    async def execute(self, url: str, html_content: Optional[str] = None,
                   execution_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the pipeline on the given URL or HTML content.
        
        Args:
            url: URL to extract from
            html_content: Optional pre-fetched HTML content
            execution_options: Optional execution parameters
            
        Returns:
            Extraction results
            
        Raises:
            RuntimeError: If execution fails after retries
        """
        # Ensure pipeline is initialized
        if not self.pipeline_id:
            await self.initialize()
        
        options = execution_options or {}
        
        # Prepare input data
        input_data = {
            "url": url,
            "extraction_options": options
        }
        
        if html_content:
            input_data["html_content"] = html_content
        
        # Set timeout for this execution
        timeout = options.get("timeout", self.timeout)
        
        # Execute hooks before pipeline
        await self._execute_before_hooks(input_data)
        
        # Start timing
        start_time = time.time()
        
        # Try to execute with retries
        result = None
        last_error = None
        retries = options.get("retry_count", self.retry_count)
        retry_delay = options.get("retry_delay", self.retry_delay)
        
        for attempt in range(retries + 1):
            if attempt > 0:
                logger.info(f"Retrying pipeline execution (attempt {attempt}/{retries})")
                # Wait before retry
                await asyncio.sleep(retry_delay * attempt)
            
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    self._execute_pipeline(input_data),
                    timeout=timeout
                )
                
                # If we reach here, execution succeeded
                break
                
            except asyncio.TimeoutError:
                logger.warning(f"Pipeline execution timed out after {timeout} seconds")
                last_error = RuntimeError(f"Pipeline execution timed out after {timeout} seconds")
                
            except Exception as e:
                logger.error(f"Pipeline execution failed: {str(e)}")
                last_error = e
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # If all attempts failed, try fallback or raise error
        if result is None:
            if self.fallback_pipeline_type:
                logger.info(f"Using fallback pipeline '{self.fallback_pipeline_type}'")
                result = await self._execute_fallback(input_data)
            else:
                # Try error handlers
                handled = await self._execute_error_handlers(input_data, last_error)
                if not handled:
                    if last_error:
                        raise last_error
                    else:
                        raise RuntimeError("Pipeline execution failed with unknown error")
        
        # Process and augment results
        if result:
            # Add execution metadata if not present
            if "_execution_metadata" not in result:
                result["_execution_metadata"] = {}
            
            # Update execution metadata
            result["_execution_metadata"]["executor"] = "PipelineExecutor"
            result["_execution_metadata"]["execution_time"] = execution_time
            result["_execution_metadata"]["retries"] = attempt
            
            # Log metrics
            log_execution_metrics(
                component="pipeline_executor",
                operation="execute",
                execution_time=execution_time,
                success=(last_error is None),
                metadata={
                    "pipeline_type": self.pipeline_type,
                    "url": url,
                    "retries": attempt
                }
            )
        
        # Execute hooks after pipeline
        await self._execute_after_hooks(result)
        
        return result
    
    async def _execute_pipeline(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the pipeline with the given input data.
        
        Args:
            input_data: Input data for the pipeline
            
        Returns:
            Pipeline execution results
        """
        service = get_pipeline_service()
        return await service.execute_pipeline(self.pipeline_id, input_data)
    
    async def _execute_fallback(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a fallback pipeline when the primary one fails.
        
        Args:
            input_data: Input data for the pipeline
            
        Returns:
            Fallback pipeline execution results
        """
        if not self.fallback_pipeline_type:
            return None
        
        try:
            # Create a fallback pipeline
            fallback_id = await create_extraction_pipeline(
                self.fallback_pipeline_type,
                self.pipeline_config
            )
            
            # Execute the fallback pipeline
            service = get_pipeline_service()
            result = await service.execute_pipeline(fallback_id, input_data)
            
            # Release the fallback pipeline
            await service.release_pipeline(fallback_id)
            
            return result
            
        except Exception as e:
            logger.error(f"Fallback pipeline execution failed: {str(e)}")
            return None
    
    async def _execute_before_hooks(self, input_data: Dict[str, Any]) -> None:
        """
        Execute hooks before pipeline execution.
        
        Args:
            input_data: Input data for the pipeline
        """
        for hook in self.before_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(input_data)
                else:
                    hook(input_data)
            except Exception as e:
                logger.warning(f"Error in before-execution hook: {str(e)}")
    
    async def _execute_after_hooks(self, result: Dict[str, Any]) -> None:
        """
        Execute hooks after pipeline execution.
        
        Args:
            result: Pipeline execution results
        """
        for hook in self.after_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(result)
                else:
                    hook(result)
            except Exception as e:
                logger.warning(f"Error in after-execution hook: {str(e)}")
    
    async def _execute_error_handlers(self, input_data: Dict[str, Any], error: Exception) -> bool:
        """
        Execute error handlers when pipeline execution fails.
        
        Args:
            input_data: Input data for the pipeline
            error: The exception that occurred
            
        Returns:
            True if an error handler handled the error, False otherwise
        """
        for handler in self.error_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    handled = await handler(input_data, error)
                else:
                    handled = handler(input_data, error)
                    
                if handled:
                    logger.info("Error handler successfully handled the execution error")
                    return True
            except Exception as e:
                logger.warning(f"Error in error handler: {str(e)}")
        
        return False
    
    def add_before_hook(self, hook: Callable) -> None:
        """
        Add a hook to be executed before pipeline execution.
        
        Args:
            hook: Function that takes input_data as argument
        """
        self.before_hooks.append(hook)
    
    def add_after_hook(self, hook: Callable) -> None:
        """
        Add a hook to be executed after pipeline execution.
        
        Args:
            hook: Function that takes results as argument
        """
        self.after_hooks.append(hook)
    
    def add_error_handler(self, handler: Callable) -> None:
        """
        Add an error handler for pipeline execution failures.
        
        Args:
            handler: Function that takes input_data and error as arguments
                    and returns True if handled, False otherwise
        """
        self.error_handlers.append(handler)
    
    def set_fallback_pipeline(self, pipeline_type: str) -> None:
        """
        Set a fallback pipeline type to use if the primary pipeline fails.
        
        Args:
            pipeline_type: Type of pipeline to use as fallback
        """
        self.fallback_pipeline_type = pipeline_type
    
    async def release(self) -> None:
        """Release the pipeline and resources."""
        if self.pipeline_id:
            service = get_pipeline_service()
            await service.release_pipeline(self.pipeline_id)
            self.pipeline_id = None
            logger.debug("Released pipeline executor resources")


async def execute_extraction_pipeline(url: str, 
                                   pipeline_type: str = "default_extraction",
                                   html_content: Optional[str] = None,
                                   options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Execute an extraction pipeline with simplified interface.
    
    Args:
        url: URL to extract from
        pipeline_type: Type of pipeline to use
        html_content: Optional pre-fetched HTML content
        options: Optional extraction and execution options
        
    Returns:
        Extraction results
    """
    executor = PipelineExecutor(pipeline_type=pipeline_type)
    
    try:
        # Execute the pipeline
        result = await executor.execute(url, html_content, options)
        return result
    finally:
        # Release resources
        await executor.release()

async def batch_execute_extraction(urls: List[str], 
                                pipeline_type: str = "default_extraction",
                                options: Optional[Dict[str, Any]] = None,
                                concurrency: int = 5) -> List[Dict[str, Any]]:
    """
    Execute extraction for multiple URLs in parallel.
    
    Args:
        urls: List of URLs to extract from
        pipeline_type: Type of pipeline to use
        options: Optional extraction and execution options
        concurrency: Maximum number of concurrent extractions
        
    Returns:
        List of extraction results
    """
    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(concurrency)
    
    async def _extract_with_semaphore(url: str) -> Dict[str, Any]:
        async with semaphore:
            try:
                return await execute_extraction_pipeline(url, pipeline_type, None, options)
            except Exception as e:
                logger.error(f"Error extracting from {url}: {str(e)}")
                return {
                    "url": url,
                    "success": False,
                    "error": str(e)
                }
    
    # Execute extractions in parallel
    tasks = [_extract_with_semaphore(url) for url in urls]
    results = await asyncio.gather(*tasks)
    
    return results