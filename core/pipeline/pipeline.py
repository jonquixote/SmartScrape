"""
Pipeline Module.

This module defines the Pipeline class that orchestrates the execution of pipeline stages.
"""

import asyncio
import logging
import time
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Type

from core.pipeline.stage import PipelineStage
from core.pipeline.context import PipelineContext


class PipelineError(Exception):
    """Exception raised for errors in the pipeline execution."""
    pass


class Pipeline:
    """
    A pipeline that executes a series of stages on a shared context.
    
    The Pipeline class manages the execution of a sequence of PipelineStage objects,
    orchestrating the flow of data through each stage and handling errors and metrics.
    
    Attributes:
        name (str): The name of this pipeline.
        logger (logging.Logger): Logger for this pipeline.
        stages (List[PipelineStage]): Ordered list of stages in this pipeline.
        config (Dict[str, Any]): Configuration parameters for this pipeline.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize a new pipeline.
        
        Args:
            name (str): Unique name for this pipeline.
            config (Optional[Dict[str, Any]]): Pipeline configuration parameters.
        """
        self.name = name
        self.logger = logging.getLogger(f"pipeline.{name}")
        self.stages: List[PipelineStage] = []
        self.config = config or {}
        
        # Configuration options with defaults
        self.continue_on_error = self.config.get("continue_on_error", False)
        self.enable_metrics = self.config.get("enable_metrics", True)
        self.parallel_execution = self.config.get("parallel_execution", False)
        self.max_workers = self.config.get("max_workers", 5)
    
    def add_stage(self, stage: PipelineStage) -> "Pipeline":
        """
        Add a stage to the pipeline.
        
        Args:
            stage (PipelineStage): The stage to add.
            
        Returns:
            Pipeline: self for method chaining.
        """
        self.stages.append(stage)
        return self
    
    def add_stages(self, stages: List[PipelineStage]) -> "Pipeline":
        """
        Add multiple stages to the pipeline.
        
        Args:
            stages (List[PipelineStage]): List of stages to add.
            
        Returns:
            Pipeline: self for method chaining.
        """
        self.stages.extend(stages)
        return self
    
    async def execute(self, initial_data: Optional[Dict[str, Any]] = None) -> PipelineContext:
        """
        Execute the pipeline with the provided initial data.
        
        Args:
            initial_data (Optional[Dict[str, Any]]): Initial data to populate the context.
            
        Returns:
            PipelineContext: The final pipeline context.
            
        Raises:
            PipelineError: If pipeline execution fails and continue_on_error is False.
        """
        # Create the pipeline context with initial data
        context = PipelineContext(initial_data or {})
        context.start_pipeline(self.name)
        
        self.logger.info(f"Starting pipeline '{self.name}' with {len(self.stages)} stages")
        start_time = time.time()
        
        try:
            # Initialize all stages
            await self._initialize_stages()
            
            # Execute stages (sequentially or in parallel based on configuration)
            if self.parallel_execution:
                await self._execute_parallel(context)
            else:
                await self._execute_sequential(context)
                
        except Exception as e:
            self.logger.error(f"Pipeline '{self.name}' execution failed: {str(e)}")
            context.add_error("pipeline", f"Pipeline execution failed: {str(e)}")
            if not self.continue_on_error:
                raise PipelineError(f"Pipeline '{self.name}' execution failed: {str(e)}") from e
        finally:
            # Clean up all stages regardless of success/failure
            await self._cleanup_stages()
            
            # Mark the end of pipeline execution
            context.end_pipeline()
            
            # Log execution metrics
            if self.enable_metrics:
                self._log_metrics(context)
        
        end_time = time.time()
        execution_time = end_time - start_time
        self.logger.info(f"Pipeline '{self.name}' completed in {execution_time:.2f}s")
        
        return context
    
    async def _initialize_stages(self) -> None:
        """Initialize all stages in the pipeline."""
        for stage in self.stages:
            try:
                await stage.initialize()
            except Exception as e:
                self.logger.error(f"Error initializing stage '{stage.name}': {str(e)}")
                raise PipelineError(f"Failed to initialize stage '{stage.name}': {str(e)}") from e
    
    async def _cleanup_stages(self) -> None:
        """Clean up all stages in the pipeline."""
        for stage in self.stages:
            try:
                await stage.cleanup()
            except Exception as e:
                self.logger.warning(f"Error cleaning up stage '{stage.name}': {str(e)}")
    
    async def _execute_sequential(self, context: PipelineContext) -> None:
        """
        Execute pipeline stages sequentially.
        
        Args:
            context (PipelineContext): The shared pipeline context.
            
        Raises:
            PipelineError: If a stage fails and continue_on_error is False.
        """
        for stage_index, stage in enumerate(self.stages):
            stage_name = stage.name
            self.logger.debug(f"Executing stage {stage_index+1}/{len(self.stages)}: '{stage_name}'")
            
            # Mark the start of this stage
            context.start_stage(stage_name)
            
            try:
                # Validate inputs
                if not await stage.validate_input(context):
                    error_msg = f"Input validation failed for stage '{stage_name}'"
                    self.logger.warning(error_msg)
                    context.add_error(stage_name, error_msg)
                    context.end_stage(success=False)
                    
                    if not self.continue_on_error:
                        raise PipelineError(error_msg)
                    continue
                
                # Execute pre-processing hook
                await stage.pre_process(context)
                
                # Execute the main stage process
                start_time = time.time()
                success = await stage.process(context)
                execution_time = time.time() - start_time
                
                self.logger.debug(f"Stage '{stage_name}' completed in {execution_time:.2f}s with result: {'success' if success else 'failure'}")
                
                # Validate outputs
                if success and not await stage.validate_output(context):
                    error_msg = f"Output validation failed for stage '{stage_name}'"
                    self.logger.warning(error_msg)
                    context.add_error(stage_name, error_msg)
                    success = False
                
                # If the stage failed, record an error
                if not success:
                    error_msg = f"Stage '{stage_name}' failed"
                    self.logger.error(error_msg)
                    context.add_error(stage_name, error_msg)
                
                # Execute post-processing hook
                await stage.post_process(context, success)
                
                # Mark the end of this stage
                context.end_stage(success=success)
                
                # If the stage failed and we're not continuing on error, stop the pipeline
                if not success and not self.continue_on_error:
                    raise PipelineError(error_msg)
                
            except Exception as e:
                self.logger.error(f"Error in stage '{stage_name}': {str(e)}")
                
                # Let the stage try to handle the error
                handled = await stage.handle_error(context, e)
                
                # Mark the stage as failed
                context.end_stage(success=False)
                
                # If the error wasn't handled and we're not continuing on error, stop the pipeline
                if not handled and not self.continue_on_error:
                    raise PipelineError(f"Unhandled error in stage '{stage_name}': {str(e)}") from e
    
    async def _execute_parallel(self, context: PipelineContext) -> None:
        """Execute independent pipeline stages in parallel.
        
        This method provides:
        - Concurrent execution of independent stages
        - Management of dependencies between stages
        - Concurrency control with maximum workers
        - Timeout handling for long-running stages
        - Status updates during execution
        - Task cancellation and cleanup
        
        Args:
            context: The shared pipeline context
            
        Raises:
            PipelineError: If any stage fails and continue_on_error is False
        """
        if not self.stages:
            return

        self.logger.info(f"Executing pipeline '{self.name}' with {len(self.stages)} stages in parallel mode")
        
        # Dictionary to store tasks by stage name
        tasks = {}
        # Dictionary to track stage dependencies
        dependencies = {}
        # Dictionary to track stage completion status
        completed = {}
        # Dictionary to store results from each stage
        results = {}
        # Set of running tasks
        running = set()
        # Stages waiting to be executed
        pending = list(self.stages)
        # Maximum number of concurrent stages
        max_workers = min(len(self.stages), self.max_workers)
        # Track failures for circuit breaker pattern
        failures = []
        
        # Get timeout configuration
        stage_timeout = self.config.get("stage_timeout", 300)  # Default 5 minutes
        
        # Set up dependency graph based on input/output requirements
        for stage in self.stages:
            stage_name = stage.name
            # Default: no dependencies (can execute immediately)
            dependencies[stage_name] = set()
            completed[stage_name] = False
            
            # Check for explicit dependencies if available
            if hasattr(stage, "get_dependencies"):
                dependencies[stage_name] = set(stage.get_dependencies())
                
        # Helper function to execute a single stage with timeout
        async def execute_stage(stage, stage_name):
            try:
                context.start_stage(stage_name)
                self.logger.debug(f"Starting stage: {stage_name}")
                
                # Validate stage inputs
                if not stage.validate_input(context):
                    self.logger.warning(f"Input validation failed for stage: {stage_name}")
                    context.add_error(stage_name, "Input validation failed")
                    context.end_stage(success=False)
                    return stage_name, False, None
                
                # Execute the stage with timeout
                try:
                    start_time = time.time()
                    success = await asyncio.wait_for(
                        stage.process(context), 
                        timeout=stage_timeout
                    )
                    execution_time = time.time() - start_time
                    
                    self.logger.debug(
                        f"Stage '{stage_name}' completed in {execution_time:.2f}s "
                        f"with result: {'success' if success else 'failure'}"
                    )
                    
                    # Validate outputs
                    if success and not await stage.validate_output(context):
                        self.logger.warning(f"Output validation failed for stage: {stage_name}")
                        context.add_error(stage_name, "Output validation failed")
                        success = False
                    
                except asyncio.TimeoutError:
                    self.logger.error(f"Stage '{stage_name}' timed out after {stage_timeout}s")
                    context.add_error(stage_name, f"Execution timed out after {stage_timeout}s")
                    # Call the stage's handle_error method
                    await stage.handle_error(context, asyncio.TimeoutError(f"Stage timed out after {stage_timeout}s"))
                    return stage_name, False, asyncio.TimeoutError(f"Stage timed out")
                
                # Execute post-processing hook
                await stage.post_process(context, success)
                
                # Mark the end of this stage
                context.end_stage(success=success)
                
                # Record failures
                if not success:
                    error_msg = f"Stage '{stage_name}' failed"
                    self.logger.error(error_msg)
                    context.add_error(stage_name, error_msg)
                    return stage_name, False, None
                
                return stage_name, True, None
                
            except Exception as e:
                self.logger.error(f"Error in stage '{stage_name}': {str(e)}")
                tb = traceback.format_exc()
                self.logger.debug(f"Traceback for stage '{stage_name}':\n{tb}")
                
                # Try to handle the error in the stage
                try:
                    handled = await stage.handle_error(context, e)
                except Exception as handle_err:
                    self.logger.error(f"Error in handle_error for stage '{stage_name}': {str(handle_err)}")
                    handled = False
                
                # Mark the stage as failed
                context.end_stage(success=False)
                
                if not handled:
                    return stage_name, False, e
                else:
                    return stage_name, False, None
        
        # Main execution loop
        while pending or running:
            # Schedule new tasks if we have capacity and available stages
            while pending and len(running) < max_workers:
                # Find stages with no unmet dependencies
                ready_stages = []
                for stage in pending:
                    stage_name = stage.name
                    # Check if all dependencies are satisfied
                    if all(completed.get(dep, False) for dep in dependencies[stage_name]):
                        ready_stages.append(stage)
                        
                if not ready_stages:
                    # No stages ready to execute yet
                    break
                    
                # Select a stage to execute (can be optimized with priority)
                stage = ready_stages[0]
                stage_name = stage.name
                
                # Create and schedule the task
                task = asyncio.create_task(execute_stage(stage, stage_name))
                tasks[stage_name] = task
                running.add(task)
                pending.remove(stage)
                
                self.logger.debug(f"Scheduled stage: {stage_name}")
            
            if not running:
                # If no tasks are running but we still have pending stages,
                # we have a dependency cycle
                if pending:
                    pending_names = [s.name for s in pending]
                    error_msg = f"Dependency cycle detected; could not schedule stages: {pending_names}"
                    self.logger.error(error_msg)
                    context.add_error("pipeline", error_msg)
                    raise PipelineError(error_msg)
                break
            
            # Wait for any task to complete
            done, running = await asyncio.wait(
                running, 
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Process completed tasks
            for task in done:
                try:
                    stage_name, success, exception = await task
                    results[stage_name] = (success, exception)
                    completed[stage_name] = True
                    
                    if not success and exception:
                        failures.append((stage_name, exception))
                        
                    # If a stage failed and we're not continuing on error, stop pipeline
                    if not success and not self.continue_on_error:
                        # Cancel all running tasks
                        for running_task in running:
                            running_task.cancel()
                            
                        # Wait for tasks to cancel
                        if running:
                            await asyncio.wait(running, return_when=asyncio.ALL_COMPLETED)
                            
                        # If there's an exception, raise it
                        if exception:
                            raise PipelineError(f"Stage '{stage_name}' failed: {str(exception)}") from exception
                        else:
                            raise PipelineError(f"Stage '{stage_name}' failed")
                    
                except Exception as e:
                    self.logger.error(f"Error processing task result: {str(e)}")
                    if not self.continue_on_error:
                        # Cancel all running tasks
                        for running_task in running:
                            running_task.cancel()
                            
                        # Wait for tasks to cancel
                        if running:
                            await asyncio.wait(running, return_when=asyncio.ALL_COMPLETED)
                            
                        raise PipelineError(f"Error processing task result: {str(e)}") from e
            
        # Check if any stage failed
        if failures and not self.continue_on_error:
            # Just report the first failure for simplicity
            stage_name, exception = failures[0]
            if exception:
                raise PipelineError(f"Stage '{stage_name}' failed: {str(exception)}") from exception
            else:
                raise PipelineError(f"Stage '{stage_name}' failed")
    
    def _log_metrics(self, context: PipelineContext) -> None:
        """
        Log pipeline execution metrics.
        
        Args:
            context (PipelineContext): The completed pipeline context.
        """
        metrics = context.get_metrics()
        
        self.logger.info(f"Pipeline '{self.name}' metrics:")
        self.logger.info(f"  Total time: {metrics['total_time']:.2f}s")
        self.logger.info(f"  Stages: {metrics['total_stages']}, Successful: {metrics['successful_stages']}")
        
        if metrics['has_errors']:
            self.logger.info("  Errors:")
            for source, errors in context.get_errors().items():
                for error in errors:
                    self.logger.info(f"    {source}: {error}")
        
        self.logger.debug("  Stage metrics:")
        for stage_name, stage_metrics in metrics['stages'].items():
            self.logger.debug(
                f"    {stage_name}: {stage_metrics['status']} "
                f"in {stage_metrics['execution_time']:.2f}s"
            )