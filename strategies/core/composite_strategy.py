"""
Composite Strategy Pattern implementation for SmartScrape.

This module implements various ways to compose multiple strategies:
1. SequentialStrategy - Executes strategies in sequence
2. FallbackStrategy - Tries strategies until one succeeds
3. PipelineStrategy - Passes output from one strategy as input to the next
4. ParallelStrategy - Executes strategies concurrently

These composite strategies implement the Composite pattern, allowing treatment
of groups of strategies the same as individual strategies.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Set, Tuple
from concurrent.futures import ThreadPoolExecutor

from strategies.base_strategy import BaseStrategy
from strategies.core.strategy_context import StrategyContext
from strategies.core.strategy_types import StrategyType, StrategyCapability, strategy_metadata

logger = logging.getLogger(__name__)

class CompositeStrategy(BaseStrategy):
    """
    Base class for composite strategies that combine multiple strategies.
    Implements the Composite pattern to allow treating compositions of strategies
    the same as individual strategies.
    """
    
    def __init__(self, context: Optional['StrategyContext'] = None):
        """
        Initialize the composite strategy.
        
        Args:
            context: The strategy context containing shared services and configuration
        """
        super().__init__(context)
        self._child_strategies: Dict[str, BaseStrategy] = {}
        # Initialize logger for composite strategies
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    def add_strategy(self, strategy: BaseStrategy) -> None:
        """
        Add a child strategy to this composite.
        
        Args:
            strategy: The strategy to add
        
        Raises:
            TypeError: If strategy is not a BaseStrategy instance
        """
        if not isinstance(strategy, BaseStrategy):
            raise TypeError(f"Expected BaseStrategy instance, got {type(strategy).__name__}")
        
        self._child_strategies[strategy.name] = strategy
    
    def add_strategy_by_name(self, strategy_name: str) -> None:
        """
        Add a child strategy by name. This will retrieve the strategy
        from the strategy factory in the context.
        
        Args:
            strategy_name: The name of the strategy to add
            
        Raises:
            ValueError: If strategy cannot be found or context is missing
        """
        if not self.context:
            raise ValueError("Cannot add strategy by name: context is missing")
            
        if not hasattr(self.context, 'strategy_factory'):
            raise ValueError("Cannot add strategy by name: strategy factory not available in context")
            
        # Get the strategy from the factory
        strategy = self.context.strategy_factory.get_strategy(strategy_name)
        
        # Add the strategy
        self._child_strategies[strategy.name] = strategy
        
    def remove_strategy(self, strategy_name: str) -> None:
        """
        Remove a child strategy by name.
        
        Args:
            strategy_name: The name of the strategy to remove
        """
        if strategy_name in self._child_strategies:
            del self._child_strategies[strategy_name]
    
    def get_child_strategy(self, strategy_name: str) -> Optional[BaseStrategy]:
        """
        Get a child strategy by name.
        
        Args:
            strategy_name: The name of the strategy to retrieve
            
        Returns:
            The strategy with the given name, or None if not found
        """
        return self._child_strategies.get(strategy_name)
    
    def get_child_strategies(self) -> List[BaseStrategy]:
        """
        Get all child strategies.
        
        Returns:
            List of all child strategies
        """
        return list(self._child_strategies.values())
    
    def get_results(self) -> List[Dict[str, Any]]:
        """
        Get combined results from all child strategies.
        
        Returns:
            Combined list of results from all child strategies
        """
        all_results = []
        for strategy in self._child_strategies.values():
            all_results.extend(strategy.get_results())
        return all_results
    
    @property
    def name(self) -> str:
        """
        Get the name of the strategy.
        
        Returns:
            String name of the strategy
        """
        return "composite_strategy"
    
    def initialize(self) -> None:
        """Initialize all child strategies."""
        super().initialize()
        for strategy in self._child_strategies.values():
            strategy.initialize()
    
    def shutdown(self) -> None:
        """Shutdown all child strategies."""
        for strategy in self._child_strategies.values():
            strategy.shutdown()
        super().shutdown()
    
    def get_combined_capabilities(self) -> Set[StrategyCapability]:
        """
        Get the combined capabilities of all child strategies.
        
        Returns:
            Set of all capabilities from child strategies
        """
        combined_capabilities = set()
        for strategy in self._child_strategies.values():
            if hasattr(strategy, '_metadata') and hasattr(strategy._metadata, 'capabilities'):
                combined_capabilities.update(strategy._metadata.capabilities)
        return combined_capabilities
    
    def can_handle(self, url: str, **kwargs) -> bool:
        """
        Check if this composite strategy can handle the given URL.
        A composite strategy can handle a URL if at least one of its child strategies can.
        
        Args:
            url: URL to check
            **kwargs: Additional arguments
            
        Returns:
            True if at least one child strategy can handle the URL
        """
        for strategy in self._child_strategies.values():
            try:
                if hasattr(strategy, 'can_handle') and strategy.can_handle(url, **kwargs):
                    return True
            except Exception as e:
                self.logger.warning(f"Error checking if strategy {type(strategy).__name__} can handle URL: {e}")
                continue
        return len(self._child_strategies) > 0  # Default to True if we have strategies
    
    def handle_error(self, error: Exception, context: Optional[Any] = None) -> Optional[Any]:
        """
        Handle errors for composite strategies.
        
        Args:
            error: The error that occurred
            context: Optional context information
            
        Returns:
            Recovery result or None
        """
        self.logger.error(f"Error in composite strategy {self.name}: {error}")
        return None
    
    def __iter__(self):
        """
        Make the composite strategy iterable over its child strategies.
        
        Returns:
            Iterator over child strategies
        """
        return iter(self._child_strategies.values())
    
    def __len__(self):
        """
        Get the number of child strategies.
        
        Returns:
            Number of child strategies
        """
        return len(self._child_strategies)


@strategy_metadata(
    strategy_type=StrategyType.SPECIAL_PURPOSE,
    capabilities={
        StrategyCapability.ERROR_HANDLING,
        StrategyCapability.CONTENT_NORMALIZATION
    },
    description="Sequential composition of multiple strategies executed in order."
)
class SequentialStrategy(CompositeStrategy):
    """
    Executes multiple strategies in sequence.
    Each strategy is executed in the order it was added.
    Results from all strategies are combined.
    """
    
    @property
    def name(self) -> str:
        """Get the name of the strategy."""
        return "sequential_strategy"
    
    def execute(self, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Execute all child strategies in sequence.
        
        Args:
            url: The URL to process
            **kwargs: Additional parameters for the strategies
            
        Returns:
            Combined result dictionary or None if all failed
        """
        combined_results = {}
        
        for i, strategy in enumerate(self._child_strategies.values()):
            self.logger.info(f"Executing strategy [{i+1}/{len(self._child_strategies)}]: {strategy.name}")
            
            try:
                result = strategy.execute(url, **kwargs)
                if result:
                    # Merge results, avoiding overwriting existing keys
                    for key, value in result.items():
                        if key not in combined_results:
                            combined_results[key] = value
                        elif isinstance(combined_results[key], list) and isinstance(value, list):
                            # Combine lists
                            combined_results[key].extend(value)
            except Exception as e:
                self.handle_error(
                    error=e,
                    message=f"Error in sequential strategy with child {strategy.name}",
                    category="strategy_execution",
                    severity="warning",
                    url=url,
                    strategy_name=self.name
                )
        
        return combined_results if combined_results else None
    
    def crawl(self, start_url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Crawl with all child strategies in sequence.
        
        Args:
            start_url: The URL to start crawling from
            **kwargs: Additional parameters for the strategies
            
        Returns:
            Combined result dictionary or None if all failed
        """
        combined_results = {}
        
        for i, strategy in enumerate(self._child_strategies.values()):
            self.logger.info(f"Crawling with strategy [{i+1}/{len(self._child_strategies)}]: {strategy.name}")
            
            try:
                result = strategy.crawl(start_url, **kwargs)
                if result:
                    # Merge results
                    for key, value in result.items():
                        if key not in combined_results:
                            combined_results[key] = value
                        elif isinstance(combined_results[key], list) and isinstance(value, list):
                            combined_results[key].extend(value)
            except Exception as e:
                self.handle_error(
                    error=e,
                    message=f"Error in sequential crawl with child {strategy.name}",
                    category="strategy_crawl",
                    severity="warning",
                    url=start_url,
                    strategy_name=self.name
                )
        
        return combined_results if combined_results else None
    
    def extract(self, html_content: str, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Extract data using all child strategies in sequence.
        
        Args:
            html_content: The HTML content to extract data from
            url: The URL the content was fetched from
            **kwargs: Additional parameters for the strategies
            
        Returns:
            Combined result dictionary or None if all failed
        """
        combined_results = {}
        
        for i, strategy in enumerate(self._child_strategies.values()):
            self.logger.info(f"Extracting with strategy [{i+1}/{len(self._child_strategies)}]: {strategy.name}")
            
            try:
                result = strategy.extract(html_content, url, **kwargs)
                if result:
                    # Merge results
                    for key, value in result.items():
                        if key not in combined_results:
                            combined_results[key] = value
                        elif isinstance(combined_results[key], list) and isinstance(value, list):
                            combined_results[key].extend(value)
            except Exception as e:
                self.handle_error(
                    error=e,
                    message=f"Error in sequential extract with child {strategy.name}",
                    category="strategy_extract",
                    severity="warning",
                    url=url,
                    strategy_name=self.name
                )
        
        return combined_results if combined_results else None


@strategy_metadata(
    strategy_type=StrategyType.SPECIAL_PURPOSE,
    capabilities={
        StrategyCapability.ERROR_HANDLING,
        StrategyCapability.CONTENT_NORMALIZATION
    },
    description="Fallback composition executing strategies until one succeeds."
)
class FallbackStrategy(CompositeStrategy):
    """
    Executes strategies in order until one succeeds.
    Useful for implementing fallback mechanisms.
    
    This strategy is particularly useful when you have multiple approaches
    to extract data and want to try them in a specified order until one works.
    """
    
    @property
    def name(self) -> str:
        """Get the name of the strategy."""
        return "fallback_strategy"
    
    def execute(self, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Execute child strategies until one succeeds.
        
        Args:
            url: The URL to process
            **kwargs: Additional parameters for the strategies
            
        Returns:
            Result from the first successful strategy or None if all failed
        """
        for i, strategy in enumerate(self._child_strategies.values()):
            self.logger.info(f"Trying strategy [{i+1}/{len(self._child_strategies)}]: {strategy.name}")
            
            try:
                result = strategy.execute(url, **kwargs)
                if result:
                    self.logger.info(f"Strategy {strategy.name} succeeded")
                    return result
                else:
                    self.logger.info(f"Strategy {strategy.name} failed, trying next")
            except Exception as e:
                self.handle_error(
                    error=e,
                    message=f"Error in fallback strategy with child {strategy.name}",
                    category="strategy_execution",
                    severity="info",  # Using info since fallback expects failures
                    url=url,
                    strategy_name=self.name
                )
        
        self.logger.warning(f"All strategies failed for URL: {url}")
        return None
    
    def crawl(self, start_url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Crawl with child strategies until one succeeds.
        
        Args:
            start_url: The URL to start crawling from
            **kwargs: Additional parameters for the strategies
            
        Returns:
            Result from the first successful strategy or None if all failed
        """
        for i, strategy in enumerate(self._child_strategies.values()):
            self.logger.info(f"Trying crawl with strategy [{i+1}/{len(self._child_strategies)}]: {strategy.name}")
            
            try:
                result = strategy.crawl(start_url, **kwargs)
                if result:
                    self.logger.info(f"Crawl with {strategy.name} succeeded")
                    return result
                else:
                    self.logger.info(f"Crawl with {strategy.name} failed, trying next")
            except Exception as e:
                self.handle_error(
                    error=e,
                    message=f"Error in fallback crawl with child {strategy.name}",
                    category="strategy_crawl",
                    severity="info",
                    url=start_url,
                    strategy_name=self.name
                )
        
        self.logger.warning(f"All crawl strategies failed for URL: {start_url}")
        return None
    
    def extract(self, html_content: str, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Extract data using child strategies until one succeeds.
        
        Args:
            html_content: The HTML content to extract data from
            url: The URL the content was fetched from
            **kwargs: Additional parameters for the strategies
            
        Returns:
            Result from the first successful strategy or None if all failed
        """
        for i, strategy in enumerate(self._child_strategies.values()):
            self.logger.info(f"Trying extraction with strategy [{i+1}/{len(self._child_strategies)}]: {strategy.name}")
            
            try:
                result = strategy.extract(html_content, url, **kwargs)
                if result:
                    self.logger.info(f"Extraction with {strategy.name} succeeded")
                    return result
                else:
                    self.logger.info(f"Extraction with {strategy.name} failed, trying next")
            except Exception as e:
                self.handle_error(
                    error=e,
                    message=f"Error in fallback extract with child {strategy.name}",
                    category="strategy_extract",
                    severity="info",
                    url=url,
                    strategy_name=self.name
                )
        
        self.logger.warning(f"All extraction strategies failed for URL: {url}")
        return None
    
    def handle_error(self, error: Exception, context: Optional[Any] = None, **kwargs) -> Optional[Any]:
        """
        Handle errors for fallback strategy with additional context.
        
        Args:
            error: The error that occurred
            context: Optional context information
            **kwargs: Additional error context like message, category, severity, url, strategy_name
            
        Returns:
            Recovery result or None
        """
        message = kwargs.get('message', str(error))
        severity = kwargs.get('severity', 'error')
        
        if severity == 'info':
            self.logger.info(f"Fallback strategy expected failure: {message}")
        else:
            self.logger.error(f"Error in fallback strategy: {message}")
        
        return None


@strategy_metadata(
    strategy_type=StrategyType.SPECIAL_PURPOSE,
    capabilities={
        StrategyCapability.CONTENT_NORMALIZATION,
        StrategyCapability.DATA_VALIDATION
    },
    description="Pipeline composition connecting the output of one strategy to the input of the next."
)
class PipelineStrategy(CompositeStrategy):
    """
    Creates a pipeline of strategies where each strategy's output is passed as input to the next.
    Order of execution is determined by the order strategies are added.
    
    This is useful for transformations or progressive refinement of data.
    """
    
    @property
    def name(self) -> str:
        """Get the name of the strategy."""
        return "pipeline_strategy"
    
    def execute(self, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Execute strategies as a pipeline, passing results from one to the next.
        
        Args:
            url: The URL to process
            **kwargs: Additional parameters for the strategies
            
        Returns:
            Final result after passing through the pipeline or None if the pipeline failed
        """
        if not self._child_strategies:
            self.logger.warning("No strategies in pipeline")
            return None
        
        # Start with the input kwargs
        current_kwargs = kwargs.copy()
        current_result = None
        
        # For each strategy in the pipeline
        for i, strategy in enumerate(self._child_strategies.values()):
            self.logger.info(f"Pipeline stage [{i+1}/{len(self._child_strategies)}]: {strategy.name}")
            
            try:
                # Execute with the current kwargs, which include results from previous strategy
                current_result = strategy.execute(url, **current_kwargs)
                
                if current_result:
                    # Add results to kwargs for next strategy
                    current_kwargs.update(current_result)
                else:
                    self.logger.warning(f"Pipeline stage {i+1} ({strategy.name}) returned no results")
                    return None  # Pipeline is broken if any stage returns None
            except Exception as e:
                self.handle_error(
                    error=e,
                    message=f"Error in pipeline stage {i+1} ({strategy.name})",
                    category="strategy_execution",
                    severity="error",
                    url=url,
                    strategy_name=self.name
                )
                return None  # Pipeline is broken if any stage fails
        
        return current_result
    
    def crawl(self, start_url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Apply pipeline pattern to crawling, though this may be less common.
        
        Args:
            start_url: The URL to start crawling from
            **kwargs: Additional parameters for the strategies
            
        Returns:
            Final result after passing through the pipeline or None if the pipeline failed
        """
        # Similar to execute, but using crawl method
        if not self._child_strategies:
            self.logger.warning("No strategies in pipeline")
            return None
        
        current_kwargs = kwargs.copy()
        current_result = None
        
        for i, strategy in enumerate(self._child_strategies.values()):
            self.logger.info(f"Pipeline crawl stage [{i+1}/{len(self._child_strategies)}]: {strategy.name}")
            
            try:
                current_result = strategy.crawl(start_url, **current_kwargs)
                
                if current_result:
                    current_kwargs.update(current_result)
                else:
                    self.logger.warning(f"Pipeline crawl stage {i+1} ({strategy.name}) returned no results")
                    return None
            except Exception as e:
                self.handle_error(
                    error=e,
                    message=f"Error in pipeline crawl stage {i+1} ({strategy.name})",
                    category="strategy_crawl",
                    severity="error",
                    url=start_url,
                    strategy_name=self.name
                )
                return None
        
        return current_result
    
    def extract(self, html_content: str, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Extract data using strategies in a pipeline.
        
        Args:
            html_content: The HTML content to extract data from
            url: The URL the content was fetched from
            **kwargs: Additional parameters for the strategies
            
        Returns:
            Final result after passing through the pipeline or None if the pipeline failed
        """
        # Similar to execute, but using extract method
        if not self._child_strategies:
            self.logger.warning("No strategies in pipeline")
            return None
        
        current_kwargs = kwargs.copy()
        current_result = None
        current_html = html_content
        
        for i, strategy in enumerate(self._child_strategies.values()):
            self.logger.info(f"Pipeline extract stage [{i+1}/{len(self._child_strategies)}]: {strategy.name}")
            
            try:
                # Pass the potentially modified HTML along the pipeline
                current_result = strategy.extract(current_html, url, **current_kwargs)
                
                if current_result:
                    current_kwargs.update(current_result)
                    # If a strategy returns modified HTML, use it for the next stage
                    if 'html_content' in current_result:
                        current_html = current_result['html_content']
                else:
                    self.logger.warning(f"Pipeline extract stage {i+1} ({strategy.name}) returned no results")
                    return None
            except Exception as e:
                self.handle_error(
                    error=e,
                    message=f"Error in pipeline extract stage {i+1} ({strategy.name})",
                    category="strategy_extract",
                    severity="error",
                    url=url,
                    strategy_name=self.name
                )
                return None
        
        return current_result


@strategy_metadata(
    strategy_type=StrategyType.SPECIAL_PURPOSE,
    capabilities={
        StrategyCapability.ERROR_HANDLING,
        StrategyCapability.CONTENT_NORMALIZATION
    },
    description="Parallel composition executing strategies concurrently and combining their results."
)
class ParallelStrategy(CompositeStrategy):
    """
    Executes multiple strategies in parallel and combines their results.
    This can significantly improve performance when strategies are independent.
    """
    
    def __init__(self, context: Optional['StrategyContext'] = None, max_workers: int = 5, combine_mode: str = 'merge'):
        """
        Initialize the parallel strategy.
        
        Args:
            context: The strategy context containing shared services
            max_workers: Maximum number of threads for parallel execution
            combine_mode: How to combine results ('merge', 'append', or 'best')
        """
        super().__init__(context)
        self.max_workers = max_workers
        self.combine_mode = combine_mode  # 'merge', 'append', or 'best'
    
    @property
    def name(self) -> str:
        """Get the name of the strategy."""
        return "parallel_strategy"
    
    def _combine_results(self, results: List[Tuple[str, Optional[Dict[str, Any]]]]) -> Optional[Dict[str, Any]]:
        """
        Combine results from parallel strategy execution based on the combine_mode.
        
        Args:
            results: List of (strategy_name, result) tuples
            
        Returns:
            Combined result dictionary or None if all failed
        """
        # Filter out None results
        valid_results = [(name, result) for name, result in results if result is not None]
        
        if not valid_results:
            return None
        
        if self.combine_mode == 'append':
            # Return a list of all results with their strategy names
            return {
                'results': [{'strategy': name, 'data': result} for name, result in valid_results]
            }
        
        elif self.combine_mode == 'best':
            # Choose the "best" result based on some heuristic
            # Here we're using the result with the most fields as a simple heuristic
            best_result = max(valid_results, key=lambda x: len(x[1]) if x[1] else 0)
            return best_result[1]
        
        else:  # Default: 'merge'
            # Merge all results into a single dictionary
            combined = {}
            for name, result in valid_results:
                for key, value in result.items():
                    # If key exists in combined and both are lists, extend the list
                    if key in combined and isinstance(combined[key], list) and isinstance(value, list):
                        combined[key].extend(value)
                    # If key exists in combined and both are dicts, merge them
                    elif key in combined and isinstance(combined[key], dict) and isinstance(value, dict):
                        combined[key].update(value)
                    # Otherwise, only add if key doesn't exist (first strategy wins conflicts)
                    elif key not in combined:
                        combined[key] = value
            return combined
    
    def execute(self, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Execute strategies in parallel.
        
        Args:
            url: The URL to process
            **kwargs: Additional parameters for the strategies
            
        Returns:
            Combined result dictionary based on combine_mode or None if all failed
        """
        if not self._child_strategies:
            self.logger.warning("No strategies to execute in parallel")
            return None
        
        strategies = list(self._child_strategies.items())
        
        # Define a worker function for the thread pool
        def execute_strategy(strategy_tuple):
            name, strategy = strategy_tuple
            try:
                result = strategy.execute(url, **kwargs)
                return (name, result)
            except Exception as e:
                self.handle_error(
                    error=e,
                    message=f"Error in parallel execution of {name}",
                    category="strategy_execution",
                    severity="warning",
                    url=url,
                    strategy_name=self.name
                )
                return (name, None)
        
        # Execute strategies in parallel
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(strategies))) as executor:
            results = list(executor.map(execute_strategy, strategies))
        
        # Combine and return results
        return self._combine_results(results)
    
    def crawl(self, start_url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Crawl with strategies in parallel.
        
        Args:
            start_url: The URL to start crawling from
            **kwargs: Additional parameters for the strategies
            
        Returns:
            Combined result dictionary based on combine_mode or None if all failed
        """
        if not self._child_strategies:
            self.logger.warning("No strategies to crawl with in parallel")
            return None
        
        strategies = list(self._child_strategies.items())
        
        # Define a worker function for the thread pool
        def crawl_strategy(strategy_tuple):
            name, strategy = strategy_tuple
            try:
                result = strategy.crawl(start_url, **kwargs)
                return (name, result)
            except Exception as e:
                self.handle_error(
                    error=e,
                    message=f"Error in parallel crawl of {name}",
                    category="strategy_crawl",
                    severity="warning",
                    url=start_url,
                    strategy_name=self.name
                )
                return (name, None)
        
        # Execute strategies in parallel
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(strategies))) as executor:
            results = list(executor.map(crawl_strategy, strategies))
        
        # Combine and return results
        return self._combine_results(results)
    
    def extract(self, html_content: str, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Extract data using strategies in parallel.
        
        Args:
            html_content: The HTML content to extract data from
            url: The URL the content was fetched from
            **kwargs: Additional parameters for the strategies
            
        Returns:
            Combined result dictionary based on combine_mode or None if all failed
        """
        if not self._child_strategies:
            self.logger.warning("No strategies to extract with in parallel")
            return None
        
        strategies = list(self._child_strategies.items())
        
        # Define a worker function for the thread pool
        def extract_strategy(strategy_tuple):
            name, strategy = strategy_tuple
            try:
                result = strategy.extract(html_content, url, **kwargs)
                return (name, result)
            except Exception as e:
                self.handle_error(
                    error=e,
                    message=f"Error in parallel extract of {name}",
                    category="strategy_extract",
                    severity="warning",
                    url=url,
                    strategy_name=self.name
                )
                return (name, None)
        
        # Execute strategies in parallel
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(strategies))) as executor:
            results = list(executor.map(extract_strategy, strategies))
        
        # Combine and return results
        return self._combine_results(results)


# Additional composite strategies could be added here, such as:
# - VotingStrategy (combines results based on consensus)
# - AdaptiveStrategy (selects strategies based on URL characteristics)
# - SpecializedStrategy (uses different strategies for different types of data)