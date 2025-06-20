"""
Adapter Classes for Pipeline Migration

This module provides adapter classes to facilitate the migration from legacy code to the
pipeline architecture. These adapters enable gradual migration by allowing old and new
components to work together seamlessly.
"""

import asyncio
import inspect
import logging
from typing import Any, Dict, List, Optional, Set, Type, Union, Callable

from core.pipeline.context import PipelineContext
from core.pipeline.pipeline import Pipeline
from core.pipeline.registry import PipelineRegistry
from core.pipeline.stage import PipelineStage

logger = logging.getLogger(__name__)

class StrategyToPipelineAdapter(PipelineStage):
    """
    Adapts a Strategy to be used as a Pipeline Stage.
    
    This allows existing strategy implementations to be used within
    a pipeline without requiring immediate refactoring.
    
    Example:
        ```python
        # Create a pipeline using an existing strategy
        pipeline = Pipeline("extraction_pipeline")
        
        # Add strategy as a stage (with the adapter)
        bfs_strategy = BFSStrategy(context)
        pipeline.add_stage(StrategyToPipelineAdapter(bfs_strategy))
        
        # Execute the pipeline
        result_context = await pipeline.execute({"url": "https://example.com"})
        ```
    """
    
    def __init__(self, 
                 strategy: Any, 
                 method_name: str = "execute", 
                 output_key: str = "result",
                 config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the adapter with a strategy.
        
        Args:
            strategy: The strategy instance to adapt
            method_name: The strategy method to call (default: "execute")
            output_key: The key to use for storing the result in the context
            config: Optional configuration for the stage
        """
        super().__init__(config or {})
        self.strategy = strategy
        self.method_name = method_name
        self.output_key = output_key
        self.strategy_name = getattr(strategy, "name", strategy.__class__.__name__)
        self.name = f"{self.strategy_name}Adapter"
        
    async def process(self, context: PipelineContext) -> bool:
        """
        Execute the strategy as a pipeline stage.
        
        Args:
            context: The pipeline context
            
        Returns:
            bool: True if execution succeeded, False otherwise
        """
        try:
            # Get the strategy method to call
            method = getattr(self.strategy, self.method_name)
            
            # Determine what parameters the method needs
            params = {}
            sig = inspect.signature(method)
            
            # Map context data to method parameters
            for param_name in sig.parameters:
                if param_name != 'self' and param_name in context.data:
                    params[param_name] = context.get(param_name)
            
            # Handle special case for first positional parameter (often 'url')
            if 'url' in sig.parameters and 'url' in context.data and 'url' not in params:
                params['url'] = context.get('url')
                
            # Execute the strategy method
            if asyncio.iscoroutinefunction(method):
                result = await method(**params)
            else:
                result = method(**params)
                
            # Store the result
            context.set(self.output_key, result)
            
            # Also store any results the strategy might have accumulated
            if hasattr(self.strategy, "get_results") and callable(self.strategy.get_results):
                context.set(f"{self.output_key}_all", self.strategy.get_results())
                
            # Handle error reporting
            if hasattr(self.strategy, "get_errors") and callable(self.strategy.get_errors):
                errors = self.strategy.get_errors()
                if errors:
                    for error in errors:
                        context.add_error(self.name, str(error))
                    
            return True if result else False
            
        except Exception as e:
            return self.handle_error(context, e)
            
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for this adapter's configuration.
        
        Returns:
            Dict containing JSON schema
        """
        return {
            "type": "object",
            "properties": {
                "method_name": {"type": "string"},
                "output_key": {"type": "string"}
            }
        }


class LegacyExtractorAdapter(PipelineStage):
    """
    Adapts a legacy extractor class to be used as a pipeline stage.
    
    This allows existing extraction code to be used within a pipeline
    while gradually migrating to the pipeline architecture.
    
    Example:
        ```python
        # Create a pipeline using a legacy extractor
        pipeline = Pipeline("extraction_pipeline")
        
        # Create the adapter with the legacy extractor class
        from extraction.content_extraction import ContentExtractor
        extractor = ContentExtractor()
        pipeline.add_stage(LegacyExtractorAdapter(
            extractor, 
            method_name="extract_content",
            input_mapping={"html_content": "html", "url": "url"},
            output_key="extraction_result"
        ))
        
        # Execute the pipeline
        result_context = await pipeline.execute({
            "html": "<html>...</html>",
            "url": "https://example.com"
        })
        ```
    """
    
    def __init__(self, 
                 extractor: Any, 
                 method_name: str, 
                 input_mapping: Optional[Dict[str, str]] = None,
                 output_key: str = "result",
                 flatten_result: bool = True,
                 config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the adapter with a legacy extractor.
        
        Args:
            extractor: The extractor instance to adapt
            method_name: The extractor method to call
            input_mapping: Mapping from context keys to parameter names
            output_key: The key to use for storing the result
            flatten_result: Whether to flatten dictionary results into context
            config: Optional configuration for the stage
        """
        super().__init__(config or {})
        self.extractor = extractor
        self.method_name = method_name
        self.input_mapping = input_mapping or {}
        self.output_key = output_key
        self.flatten_result = flatten_result
        self.name = f"{extractor.__class__.__name__}Adapter"
        
    def validate_input(self, context: PipelineContext) -> bool:
        """
        Validate that required inputs are present in the context.
        
        Args:
            context: The pipeline context
            
        Returns:
            bool: True if all required inputs are present
        """
        for ctx_key in self.input_mapping.keys():
            if ctx_key not in context.data:
                context.add_error(self.name, f"Missing required input: {ctx_key}")
                return False
        return True
        
    async def process(self, context: PipelineContext) -> bool:
        """
        Execute the legacy extractor as a pipeline stage.
        
        Args:
            context: The pipeline context
            
        Returns:
            bool: True if execution succeeded, False otherwise
        """
        try:
            # Get the extractor method to call
            method = getattr(self.extractor, self.method_name)
            
            # Map context data to method parameters
            params = {}
            for ctx_key, param_name in self.input_mapping.items():
                params[param_name] = context.get(ctx_key)
            
            # Add any additional parameters from config
            if "method_params" in self.config:
                params.update(self.config["method_params"])
                
            # Execute the extractor method
            if asyncio.iscoroutinefunction(method):
                result = await method(**params)
            else:
                result = method(**params)
                
            # Store the result
            if result is None:
                context.add_error(self.name, "Extractor returned None")
                return False
                
            if isinstance(result, dict):
                if self.flatten_result:
                    # Flatten dictionary result into context
                    context.update(result)
                else:
                    # Store as a single value
                    context.set(self.output_key, result)
            else:
                # Store as a single value
                context.set(self.output_key, result)
                
            # Check for success indicator in result
            if isinstance(result, dict) and "success" in result:
                return result["success"]
                
            return True
            
        except Exception as e:
            return self.handle_error(context, e)


class PipelineToLegacyAdapter:
    """
    Adapts a pipeline to be used like a legacy component.
    
    This allows new pipeline implementations to be used by legacy code
    that expects a different interface.
    
    Example:
        ```python
        # Create an adapter that makes a pipeline look like a legacy extractor
        from core.pipeline.registry import PipelineRegistry
        
        adapter = PipelineToLegacyAdapter(
            pipeline_name="extraction_pipeline",
            input_mapping={"html": "html_content", "url": "url"},
            registry=PipelineRegistry()
        )
        
        # Use it like a legacy extractor
        result = await adapter.extract_content(
            html="<html>...</html>",
            url="https://example.com"
        )
        ```
    """
    
    def __init__(self, 
                 pipeline_name: str,
                 method_name: str = "execute",
                 input_mapping: Optional[Dict[str, str]] = None,
                 result_processor: Optional[Callable[[PipelineContext], Any]] = None,
                 registry: Optional[PipelineRegistry] = None) -> None:
        """
        Initialize the adapter with a pipeline name.
        
        Args:
            pipeline_name: Name of the pipeline to use
            method_name: Method to expose (will be created dynamically)
            input_mapping: Mapping from parameter names to context keys
            result_processor: Optional function to process the pipeline result
            registry: Pipeline registry to use (or will create one)
        """
        self.pipeline_name = pipeline_name
        self.method_name = method_name
        self.input_mapping = input_mapping or {}
        self.result_processor = result_processor
        self.registry = registry or PipelineRegistry()
        
        # Create the adapter method dynamically
        setattr(self, method_name, self._execute_pipeline)
        
    async def _execute_pipeline(self, **kwargs) -> Any:
        """
        Execute the pipeline with the provided parameters.
        
        Args:
            **kwargs: Parameters to pass to the pipeline
            
        Returns:
            The pipeline result processed according to configuration
        """
        try:
            # Create the pipeline
            pipeline = self.registry.create_pipeline(self.pipeline_name)
            if not pipeline:
                raise ValueError(f"No pipeline found with name: {self.pipeline_name}")
                
            # Map input parameters to context keys
            context_data = {}
            for param_name, value in kwargs.items():
                ctx_key = self.input_mapping.get(param_name, param_name)
                context_data[ctx_key] = value
                
            # Execute the pipeline
            context = await pipeline.execute(context_data)
            
            # Process the result
            if self.result_processor:
                return self.result_processor(context)
            
            # Default processing: return all data if no processor specified
            if context.has_errors():
                return {
                    "success": False,
                    "errors": context.metadata["errors"],
                    **context.data
                }
            
            return {
                "success": True,
                **context.data
            }
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }


class CompositeExtractorAdapter:
    """
    Creates a unified interface that can switch between legacy and pipeline implementations.
    
    This adapter allows for gradual migration by providing feature flags to control
    which implementation to use, with options for parallel execution and result comparison.
    
    Example:
        ```python
        # Create a composite adapter
        adapter = CompositeExtractorAdapter(
            legacy_extractor=ContentExtractor(),
            legacy_method="extract_content",
            pipeline_name="extraction_pipeline",
            use_pipeline=True,
            compare_results=True
        )
        
        # Use like a normal extractor
        result = await adapter.extract_content(
            html_content="<html>...</html>",
            url="https://example.com"
        )
        ```
    """
    
    def __init__(self,
                 legacy_extractor: Any = None,
                 legacy_method: str = "extract_content",
                 pipeline_name: str = "extraction_pipeline",
                 use_pipeline: bool = True,
                 compare_results: bool = False,
                 fallback_to_legacy: bool = True,
                 registry: Optional[PipelineRegistry] = None) -> None:
        """
        Initialize the composite adapter.
        
        Args:
            legacy_extractor: The legacy extractor instance
            legacy_method: The legacy extractor method to call
            pipeline_name: Name of the pipeline to use
            use_pipeline: Whether to use the pipeline implementation
            compare_results: Whether to compare results from both implementations
            fallback_to_legacy: Whether to fall back to legacy on pipeline failure
            registry: Pipeline registry to use (or will create one)
        """
        self.legacy_extractor = legacy_extractor
        self.legacy_method = legacy_method
        self.pipeline_name = pipeline_name
        self.use_pipeline = use_pipeline
        self.compare_results = compare_results
        self.fallback_to_legacy = fallback_to_legacy
        self.registry = registry or PipelineRegistry()
        
        # Create the method with the same name as the legacy method
        setattr(self, legacy_method, self._execute)
        
        # Set up result comparison logging
        self.comparison_logger = logging.getLogger("pipeline_comparison")
        
    async def _execute(self, **kwargs) -> Any:
        """
        Execute using the appropriate implementation based on configuration.
        
        Args:
            **kwargs: Parameters to pass to the implementation
            
        Returns:
            The result from the selected implementation
        """
        pipeline_result = None
        legacy_result = None
        
        # Determine execution mode
        if self.compare_results:
            # Run both and compare
            pipeline_result, legacy_result = await self._execute_both(**kwargs)
            return legacy_result if not self.use_pipeline else pipeline_result
        elif self.use_pipeline:
            # Run pipeline with legacy fallback
            try:
                pipeline_result = await self._execute_pipeline(**kwargs)
                return pipeline_result
            except Exception as e:
                logger.error(f"Pipeline execution failed: {str(e)}")
                if self.fallback_to_legacy and self.legacy_extractor:
                    logger.info("Falling back to legacy implementation")
                    return await self._execute_legacy(**kwargs)
                raise
        else:
            # Just run legacy
            return await self._execute_legacy(**kwargs)
    
    async def _execute_pipeline(self, **kwargs) -> Any:
        """Execute using the pipeline implementation."""
        pipeline = self.registry.create_pipeline(self.pipeline_name)
        if not pipeline:
            raise ValueError(f"No pipeline found with name: {self.pipeline_name}")
            
        context = await pipeline.execute(kwargs)
        
        if context.has_errors():
            return {
                "success": False,
                "errors": context.metadata["errors"],
                **context.data
            }
        
        return {
            "success": True,
            **context.data
        }
    
    async def _execute_legacy(self, **kwargs) -> Any:
        """Execute using the legacy implementation."""
        if not self.legacy_extractor:
            raise ValueError("No legacy extractor provided")
            
        method = getattr(self.legacy_extractor, self.legacy_method)
        if asyncio.iscoroutinefunction(method):
            return await method(**kwargs)
        return method(**kwargs)
    
    async def _execute_both(self, **kwargs) -> tuple:
        """Execute both implementations and compare results."""
        tasks = []
        
        # Create tasks for both implementations
        if self.legacy_extractor:
            tasks.append(asyncio.create_task(
                self._execute_legacy(**kwargs)
            ))
        else:
            tasks.append(asyncio.create_task(
                asyncio.sleep(0)  # Dummy task
            ))
            
        tasks.append(asyncio.create_task(
            self._execute_pipeline(**kwargs)
        ))
        
        # Run both implementations
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        legacy_result = results[0] if not isinstance(results[0], Exception) else None
        pipeline_result = results[1] if not isinstance(results[1], Exception) else None
        
        # Compare and log differences
        if legacy_result and pipeline_result:
            self._compare_results(legacy_result, pipeline_result, kwargs)
            
        return pipeline_result, legacy_result
        
    def _compare_results(self, legacy_result: Any, pipeline_result: Any, inputs: Dict[str, Any]) -> None:
        """Compare results from both implementations and log differences."""
        try:
            if isinstance(legacy_result, dict) and isinstance(pipeline_result, dict):
                # Compare dictionaries
                all_keys = set(legacy_result.keys()) | set(pipeline_result.keys())
                differences = []
                
                for key in all_keys:
                    legacy_value = legacy_result.get(key, "<missing>")
                    pipeline_value = pipeline_result.get(key, "<missing>")
                    
                    if legacy_value != pipeline_value:
                        differences.append({
                            "key": key,
                            "legacy": legacy_value,
                            "pipeline": pipeline_value
                        })
                
                if differences:
                    self.comparison_logger.warning(
                        f"Result differences found for input {inputs.get('url', '')}:"
                    )
                    for diff in differences:
                        self.comparison_logger.warning(
                            f"  Key: {diff['key']}\n"
                            f"    Legacy: {diff['legacy']}\n"
                            f"    Pipeline: {diff['pipeline']}"
                        )
            elif legacy_result != pipeline_result:
                # Simple comparison for non-dictionaries
                self.comparison_logger.warning(
                    f"Result differences found for input {inputs.get('url', '')}:\n"
                    f"  Legacy: {legacy_result}\n"
                    f"  Pipeline: {pipeline_result}"
                )
        except Exception as e:
            self.comparison_logger.error(f"Error comparing results: {str(e)}")