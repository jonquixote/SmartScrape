"""
Multi-Strategy Implementation using the Composite Pattern.

This module implements the MultiStrategyV2 class which combines multiple extraction strategies
using the composite pattern to maximize data extraction success and quality.
"""

import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from collections import defaultdict

from strategies.core.composite_strategy import CompositeStrategy
from strategies.core.strategy_interface import BaseStrategy
from strategies.core.strategy_context import StrategyContext
from strategies.core.strategy_types import StrategyType, StrategyCapability, strategy_metadata
from strategies.core.strategy_factory import StrategyFactory

# Configure logging
logger = logging.getLogger(__name__)

@strategy_metadata(
    strategy_type=StrategyType.SPECIAL_PURPOSE,
    capabilities={
        StrategyCapability.ERROR_HANDLING,
        StrategyCapability.CONTENT_NORMALIZATION,
        StrategyCapability.DYNAMIC_CONTENT,
        StrategyCapability.PAGINATION_HANDLING,
        StrategyCapability.DATA_VALIDATION
    },
    description="A composite strategy that combines multiple strategies to improve data extraction."
)
class MultiStrategyV2(CompositeStrategy):
    """
    A strategy that combines multiple extraction strategies to improve overall
    extraction results. It can run multiple strategies in sequence or parallel
    and consolidate their results based on confidence scores.
    
    This implementation uses the composite pattern to organize and execute
    child strategies.
    """
    
    def __init__(self, 
                 context: StrategyContext,
                 fallback_threshold: float = 0.4,
                 confidence_threshold: float = 0.7,
                 use_voting: bool = True,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the multi-strategy.
        
        Args:
            context: The strategy context providing access to services
            fallback_threshold: Threshold below which to try fallback strategies
            confidence_threshold: Threshold for accepting extraction results
            use_voting: Whether to use voting to resolve conflicts
            config: Additional configuration
        """
        super().__init__(context)
        
        # Ensure _child_strategies is properly initialized as a dictionary
        # This is a defensive fix to prevent the "'StrategyContext' object is not iterable" error
        if not hasattr(self, '_child_strategies') or not isinstance(self._child_strategies, dict):
            self._child_strategies = {}
        
        # Configuration
        self.fallback_threshold = fallback_threshold
        self.confidence_threshold = confidence_threshold
        self.use_voting = use_voting
        
        # Default configuration
        self.config = {
            'max_depth': 2,
            'max_pages': 100,
            'include_external': False,
            'parallel_execution': True,
            'deduplicate': True,
            'max_concurrent': 5,
            'timeout': 30
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Track individual strategy results and performance
        self.strategy_performance = {}
        
        # Extraction statistics
        self.extraction_stats = {
            "total_attempts": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "fallbacks_triggered": 0,
            "strategy_usage": defaultdict(int),
            "strategy_success": defaultdict(int),
            "avg_confidence": 0.0,
            "execution_time": 0.0
        }
        
        # Store comprehensive results
        self._consolidated_results: List[Dict[str, Any]] = []
    
    @property
    def name(self) -> str:
        return "multi_strategy_v2"
    
    def initialize_strategy_performance_tracking(self) -> None:
        """Initialize performance tracking for all child strategies."""
        for strategy_name in self._child_strategies:
            if strategy_name not in self.strategy_performance:
                self.strategy_performance[strategy_name] = {
                    "total_executions": 0,
                    "successful_executions": 0,
                    "failed_executions": 0,
                    "success_rate": 0.0,
                    "avg_confidence": 0.0,
                    "avg_execution_time": 0.0,
                    "total_confidence": 0.0,
                    "total_execution_time": 0.0
                }
    
    def update_strategy_performance(self, 
                                   strategy_name: str, 
                                   success: bool, 
                                   confidence: float = 0.0,
                                   execution_time: float = 0.0) -> None:
        """
        Update performance metrics for a strategy.
        
        Args:
            strategy_name: The name of the strategy
            success: Whether the execution was successful
            confidence: The confidence score (0-1)
            execution_time: The execution time in seconds
        """
        if strategy_name not in self.strategy_performance:
            self.initialize_strategy_performance_tracking()
        
        performance = self.strategy_performance[strategy_name]
        performance["total_executions"] += 1
        performance["total_execution_time"] += execution_time
        
        if success:
            performance["successful_executions"] += 1
            performance["total_confidence"] += confidence
        else:
            performance["failed_executions"] += 1
        
        # Update derived metrics
        if performance["total_executions"] > 0:
            performance["success_rate"] = (
                performance["successful_executions"] / performance["total_executions"]
            )
            performance["avg_execution_time"] = (
                performance["total_execution_time"] / performance["total_executions"]
            )
        
        if performance["successful_executions"] > 0:
            performance["avg_confidence"] = (
                performance["total_confidence"] / performance["successful_executions"]
            )
    
    def execute(self, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Execute the multi-strategy for the given URL.
        
        Args:
            url: The URL to process
            **kwargs: Additional arguments to pass to child strategies
        
        Returns:
            A dictionary with the execution results, or None if all strategies failed
        """
        logger.info(f"Executing multi-strategy for URL: {url}")
        start_time = time.time()
        
        # Initialize performance tracking if needed
        self.initialize_strategy_performance_tracking()
        
        # Update extraction stats
        self.extraction_stats["total_attempts"] += 1
        
        # Execute child strategies
        if self.config.get('parallel_execution', True) and len(self._child_strategies) > 1:
            # Execute strategies in parallel
            logger.debug("Using parallel strategy execution")
            strategy_results = self._execute_strategies_parallel(url, **kwargs)
        else:
            # Execute strategies sequentially
            logger.debug("Using sequential strategy execution")
            strategy_results = self._execute_strategies_sequential(url, **kwargs)
        
        # Filter out failed results
        strategy_results = [
            (data, confidence, strategy_name) 
            for data, confidence, strategy_name in strategy_results 
            if data and confidence > 0
        ]
        
        # Return early if no successful executions
        if not strategy_results:
            logger.warning(f"No successful executions for URL: {url}")
            self.extraction_stats["failed_extractions"] += 1
            self.extraction_stats["execution_time"] += (time.time() - start_time)
            return None
        
        # Get the best single strategy result
        best_result = max(strategy_results, key=lambda x: x[1])
        best_data, best_confidence, best_strategy = best_result
        
        # If only one strategy succeeded, use that result
        if len(strategy_results) == 1:
            result = {**best_data, "strategy": best_strategy, "confidence": best_confidence}
            self._update_extraction_stats(result)
            self.extraction_stats["execution_time"] += (time.time() - start_time)
            return result
        
        # If high confidence with best strategy, use that
        if best_confidence >= self.confidence_threshold:
            result = {**best_data, "strategy": best_strategy, "confidence": best_confidence}
            self._update_extraction_stats(result)
            self.extraction_stats["execution_time"] += (time.time() - start_time)
            return result
        
        # Otherwise, combine results with voting if enabled
        if self.use_voting:
            combined_data, combined_confidence = self._combine_results_with_voting(strategy_results)
            result = {**combined_data, "strategy": "combined", "confidence": combined_confidence}
            self._update_extraction_stats(result)
            self.extraction_stats["execution_time"] += (time.time() - start_time)
            return result
        
        # If not using voting, fall back to the best result
        result = {**best_data, "strategy": best_strategy, "confidence": best_confidence}
        self._update_extraction_stats(result)
        self.extraction_stats["execution_time"] += (time.time() - start_time)
        return result
    
    def _update_extraction_stats(self, result: Dict[str, Any]) -> None:
        """
        Update extraction statistics based on a result.
        
        Args:
            result: The execution result
        """
        self.extraction_stats["successful_extractions"] += 1
        
        # Add to consolidated results with metadata
        result_with_metadata = {
            "data": result,
            "source_url": result.get("url", ""),
            "score": result.get("confidence", 0.0),
            "strategy": result.get("strategy", "unknown")
        }
        
        self._consolidated_results.append(result_with_metadata)
        
        # Calculate running average confidence
        total_extractions = self.extraction_stats["successful_extractions"]
        if total_extractions > 0:
            current_avg = self.extraction_stats["avg_confidence"]
            new_confidence = result.get("confidence", 0.0)
            self.extraction_stats["avg_confidence"] = (
                (current_avg * (total_extractions - 1) + new_confidence) 
                / total_extractions
            )
    
    def _execute_strategies_sequential(self, url: str, **kwargs) -> List[Tuple[Dict[str, Any], float, str]]:
        """
        Execute child strategies sequentially.
        
        Args:
            url: The URL to process
            **kwargs: Additional arguments to pass to child strategies
        
        Returns:
            A list of (data, confidence, strategy_name) tuples
        """
        strategy_results = []
        
        for strategy_name, strategy in self._child_strategies.items():
            self.extraction_stats["strategy_usage"][strategy_name] += 1
            
            start_time = time.time()
            try:
                # Execute the strategy
                result = strategy.execute(url, **kwargs)
                execution_time = time.time() - start_time
                
                # Process the result
                if result:
                    # Try to extract confidence from the result
                    confidence = result.get("confidence", 0.5)
                    
                    # Update performance metrics
                    self.update_strategy_performance(
                        strategy_name, True, confidence, execution_time
                    )
                    
                    # Track success
                    self.extraction_stats["strategy_success"][strategy_name] += 1
                    
                    # Add to results
                    strategy_results.append((result, confidence, strategy_name))
                else:
                    # Update performance metrics for failure
                    self.update_strategy_performance(
                        strategy_name, False, 0.0, execution_time
                    )
            
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Handle error
                logger.error(f"Error executing strategy {strategy_name}: {str(e)}")
                self.handle_error(
                    error=e,
                    message=f"Error in strategy {strategy_name}: {str(e)}",
                    category="strategy_execution",
                    severity="warning",
                    url=url,
                    strategy_name=strategy_name
                )
                
                # Update performance metrics for failure
                self.update_strategy_performance(
                    strategy_name, False, 0.0, execution_time
                )
        
        return strategy_results
    
    def _execute_strategies_parallel(self, url: str, **kwargs) -> List[Tuple[Dict[str, Any], float, str]]:
        """
        Execute child strategies in parallel.
        
        Args:
            url: The URL to process
            **kwargs: Additional arguments to pass to child strategies
        
        Returns:
            A list of (data, confidence, strategy_name) tuples
        """
        async def _execute_strategy(strategy_name: str, strategy: BaseStrategy) -> Tuple[Optional[Dict[str, Any]], float, str]:
            """Execute a single strategy asynchronously."""
            self.extraction_stats["strategy_usage"][strategy_name] += 1
            
            start_time = time.time()
            try:
                # Execute the strategy
                result = strategy.execute(url, **kwargs)
                execution_time = time.time() - start_time
                
                # Process the result
                if result:
                    # Try to extract confidence from the result
                    confidence = result.get("confidence", 0.5)
                    
                    # Update performance metrics
                    self.update_strategy_performance(
                        strategy_name, True, confidence, execution_time
                    )
                    
                    # Track success
                    self.extraction_stats["strategy_success"][strategy_name] += 1
                    
                    # Return the result
                    return result, confidence, strategy_name
                else:
                    # Update performance metrics for failure
                    self.update_strategy_performance(
                        strategy_name, False, 0.0, execution_time
                    )
                    return None, 0.0, strategy_name
            
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Handle error
                logger.error(f"Error executing strategy {strategy_name}: {str(e)}")
                self.handle_error(
                    error=e,
                    message=f"Error in strategy {strategy_name}: {str(e)}",
                    category="strategy_execution",
                    severity="warning",
                    url=url,
                    strategy_name=strategy_name
                )
                
                # Update performance metrics for failure
                self.update_strategy_performance(
                    strategy_name, False, 0.0, execution_time
                )
                return None, 0.0, strategy_name
        
        # Create tasks for all strategies
        # Additional defensive check to ensure _child_strategies is a dictionary
        if not isinstance(self._child_strategies, dict):
            logger.error(f"_child_strategies is not a dict: {type(self._child_strategies)}")
            self._child_strategies = {}
            return {"url": url, "data": [], "metadata": {"error": "Invalid child strategies configuration"}}
        
        tasks = [
            _execute_strategy(strategy_name, strategy)
            for strategy_name, strategy in self._child_strategies.items()
        ]
        
        # Run all tasks in parallel
        try:
            # If we have an event loop, use it, otherwise create one
            loop = asyncio.get_event_loop()
            results = loop.run_until_complete(asyncio.gather(*tasks))
        except Exception as e:
            logger.error(f"Error executing strategies in parallel: {str(e)}")
            # Fall back to sequential execution
            return self._execute_strategies_sequential(url, **kwargs)
        
        # Filter out failed results
        strategy_results = [
            (data, confidence, strategy_name) 
            for data, confidence, strategy_name in results 
            if data is not None
        ]
        
        return strategy_results
    
    def _combine_results_with_voting(self, 
                                   strategy_results: List[Tuple[Dict[str, Any], float, str]]) -> Tuple[Dict[str, Any], float]:
        """
        Combine results from multiple strategies using weighted voting.
        
        Args:
            strategy_results: List of (data, confidence, strategy_name) tuples
        
        Returns:
            A tuple of (combined_data, combined_confidence)
        """
        logger.debug(f"Combining results from {len(strategy_results)} strategies using voting")
        
        # Initialize for voting
        field_votes = defaultdict(lambda: defaultdict(float))
        field_confidences = defaultdict(float)
        total_confidence = sum(confidence for _, confidence, _ in strategy_results)
        
        # Collect votes for each field value
        for data, confidence, _ in strategy_results:
            # Skip empty results
            if not data:
                continue
                
            # Calculate weight based on confidence
            weight = confidence / total_confidence if total_confidence > 0 else 0
            
            # Cast votes for each field in this result
            for field, value in data.items():
                # Convert value to string for comparison
                value_str = str(value)
                
                # Add weighted vote for this value
                field_votes[field][value_str] += weight
                
                # Track maximum confidence for each field
                field_confidences[field] = max(field_confidences[field], confidence)
        
        # Determine winning values for each field
        combined_data = {}
        field_scores = {}
        
        for field, votes in field_votes.items():
            # Get the value with the highest vote count
            winning_value_str, vote_score = max(votes.items(), key=lambda x: x[1])
            
            # Convert back to original type if possible
            for data, _, _ in strategy_results:
                if field in data and str(data[field]) == winning_value_str:
                    combined_data[field] = data[field]
                    break
            else:
                # If not found in original data, use the string version
                combined_data[field] = winning_value_str
            
            # Store the vote score (agreement level) for this field
            field_scores[field] = vote_score
        
        # Calculate overall confidence based on agreement and original confidences
        if field_scores:
            # Average of field agreement scores weighted by field confidences
            weighted_sum = sum(score * field_confidences[field] for field, score in field_scores.items())
            total_field_confidence = sum(field_confidences.values())
            
            combined_confidence = weighted_sum / total_field_confidence if total_field_confidence > 0 else 0
        else:
            combined_confidence = 0.0
        
        logger.debug(f"Combined confidence: {combined_confidence:.2f}")
        return combined_data, combined_confidence
    
    def crawl(self, start_url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Crawl starting from the given URL.
        
        Args:
            start_url: The URL to start crawling from
            **kwargs: Additional arguments to pass to child strategies
        
        Returns:
            A dictionary with crawling results, or None if crawling failed
        """
        logger.info(f"Starting multi-strategy crawl from URL: {start_url}")
        start_time = time.time()
        
        # Get configuration for crawling
        max_depth = kwargs.get('max_depth', self.config.get('max_depth', 2))
        max_pages = kwargs.get('max_pages', self.config.get('max_pages', 100))
        include_external = kwargs.get('include_external', self.config.get('include_external', False))
        
        # Initialize tracking
        visited_urls = set()
        queue = [{'url': start_url, 'depth': 0, 'score': 1.0}]
        results = []
        
        # Crawling loop
        while queue and len(visited_urls) < max_pages:
            # Sort queue by score (highest first)
            queue.sort(key=lambda x: x['score'], reverse=True)
            
            # Get next URL to process
            current = queue.pop(0)
            url = current['url']
            depth = current['depth']
            
            # Skip if already visited
            if url in visited_urls:
                continue
            
            # Add to visited
            visited_urls.add(url)
            logger.info(f"Processing URL: {url} (Depth: {depth})")
            
            try:
                # Execute strategies on this URL
                execution_result = self.execute(url, **kwargs)
                
                if execution_result:
                    # Add metadata to the result
                    result_with_metadata = {
                        "data": execution_result,
                        "source_url": url,
                        "depth": depth,
                        "score": execution_result.get("confidence", 0.0),
                        "strategy": execution_result.get("strategy", "unknown")
                    }
                    
                    results.append(result_with_metadata)
                
                # Don't go deeper if max depth reached
                if depth >= max_depth:
                    continue
                
                # Get next URLs from all strategies
                next_urls = []
                for strategy_name, strategy in self._child_strategies.items():
                    try:
                        # Check if the strategy has a get_next_urls method
                        if hasattr(strategy, 'get_next_urls'):
                            strategy_next_urls = strategy.get_next_urls(
                                url=url,
                                depth=depth,
                                visited=visited_urls,
                                extraction_result=execution_result,
                                **kwargs
                            )
                            
                            if strategy_next_urls:
                                next_urls.extend(strategy_next_urls)
                    except Exception as e:
                        logger.error(f"Error getting next URLs from strategy {strategy_name}: {str(e)}")
                
                # Add new URLs to the queue
                for next_url_info in next_urls:
                    next_url = next_url_info['url']
                    
                    # Skip if already visited or in queue
                    if next_url in visited_urls:
                        continue
                    
                    # Check if URL already in queue
                    existing = next((item for item in queue if item['url'] == next_url), None)
                    
                    if existing:
                        # Update score if higher
                        if next_url_info['score'] > existing['score']:
                            existing['score'] = next_url_info['score']
                    else:
                        queue.append(next_url_info)
            
            except Exception as e:
                logger.error(f"Error processing URL {url}: {str(e)}")
                self.handle_error(
                    error=e,
                    message=f"Error crawling URL {url}: {str(e)}",
                    category="crawling",
                    severity="warning",
                    url=url
                )
        
        # Update execution time
        self.extraction_stats["execution_time"] += (time.time() - start_time)
        
        logger.info(f"Multi-strategy crawl completed. Visited {len(visited_urls)} URLs")
        
        return {
            "results": results,
            "metrics": self.get_metrics(),
            "visited_urls": list(visited_urls)
        }
    
    def extract(self, html_content: str, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Extract data from HTML content.
        
        Args:
            html_content: The HTML content to extract from
            url: The URL associated with the content
            **kwargs: Additional arguments to pass to child strategies
        
        Returns:
            A dictionary with extraction results, or None if extraction failed
        """
        logger.info(f"Extracting data from HTML for URL: {url}")
        start_time = time.time()
        
        # Initialize performance tracking if needed
        self.initialize_strategy_performance_tracking()
        
        # Update extraction stats
        self.extraction_stats["total_attempts"] += 1
        
        # Execute child strategies for extraction
        strategy_results = []
        
        for strategy_name, strategy in self._child_strategies.items():
            self.extraction_stats["strategy_usage"][strategy_name] += 1
            
            start_strategy_time = time.time()
            try:
                # Execute the strategy's extract method
                result = strategy.extract(html_content, url, **kwargs)
                execution_time = time.time() - start_strategy_time
                
                # Process the result
                if result:
                    # Try to extract confidence from the result
                    confidence = result.get("confidence", 0.5)
                    
                    # Update performance metrics
                    self.update_strategy_performance(
                        strategy_name, True, confidence, execution_time
                    )
                    
                    # Track success
                    self.extraction_stats["strategy_success"][strategy_name] += 1
                    
                    # Add to results
                    strategy_results.append((result, confidence, strategy_name))
                else:
                    # Update performance metrics for failure
                    self.update_strategy_performance(
                        strategy_name, False, 0.0, execution_time
                    )
            
            except Exception as e:
                execution_time = time.time() - start_strategy_time
                
                # Handle error
                logger.error(f"Error extracting with strategy {strategy_name}: {str(e)}")
                self.handle_error(
                    error=e,
                    message=f"Error in strategy {strategy_name} extraction: {str(e)}",
                    category="extraction",
                    severity="warning",
                    url=url,
                    strategy_name=strategy_name
                )
                
                # Update performance metrics for failure
                self.update_strategy_performance(
                    strategy_name, False, 0.0, execution_time
                )
        
        # Update execution time
        self.extraction_stats["execution_time"] += (time.time() - start_time)
        
        # Filter out failed results
        strategy_results = [
            (data, confidence, strategy_name) 
            for data, confidence, strategy_name in strategy_results 
            if data and confidence > 0
        ]
        
        # Return early if no successful extractions
        if not strategy_results:
            logger.warning(f"No successful extractions for URL: {url}")
            self.extraction_stats["failed_extractions"] += 1
            return None
        
        # Get the best single strategy result
        best_result = max(strategy_results, key=lambda x: x[1])
        best_data, best_confidence, best_strategy = best_result
        
        # If only one strategy succeeded, use that result
        if len(strategy_results) == 1:
            result = {**best_data, "strategy": best_strategy, "confidence": best_confidence}
            self._update_extraction_stats(result)
            return result
        
        # If high confidence with best strategy, use that
        if best_confidence >= self.confidence_threshold:
            result = {**best_data, "strategy": best_strategy, "confidence": best_confidence}
            self._update_extraction_stats(result)
            return result
        
        # Otherwise, combine results with voting if enabled
        if self.use_voting:
            combined_data, combined_confidence = self._combine_results_with_voting(strategy_results)
            result = {**combined_data, "strategy": "combined", "confidence": combined_confidence}
            self._update_extraction_stats(result)
            return result
        
        # If not using voting, fall back to the best result
        result = {**best_data, "strategy": best_strategy, "confidence": best_confidence}
        self._update_extraction_stats(result)
        return result
    
    def get_results(self) -> List[Dict[str, Any]]:
        """
        Get combined results from all child strategies.
        
        Returns:
            A list of result dictionaries
        """
        # Get direct results from this strategy
        results = self._consolidated_results.copy()
        
        # Add results from child strategies
        for strategy in self._child_strategies.values():
            strategy_results = strategy.get_results()
            if strategy_results:
                results.extend(strategy_results)
        
        # Deduplicate results if configured
        if self.config.get('deduplicate', True):
            deduplicated_results = []
            seen_urls = set()
            
            for result in results:
                # Skip non-dictionary results
                if not isinstance(result, dict):
                    continue
                
                # Get the URL for deduplication
                result_url = None
                if "url" in result:
                    result_url = result["url"]
                elif "source_url" in result:
                    result_url = result["source_url"]
                elif "data" in result and isinstance(result["data"], dict):
                    result_url = result["data"].get("url")
                
                # Skip if we've seen this URL before
                if result_url and result_url in seen_urls:
                    continue
                
                # Add to deduplicated results
                if result_url:
                    seen_urls.add(result_url)
                deduplicated_results.append(result)
            
            return deduplicated_results
        
        return results
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics about strategy execution.
        
        Returns:
            A dictionary of metrics
        """
        # Get base metrics
        metrics = {
            "extraction_stats": dict(self.extraction_stats),
            "strategy_performance": self.strategy_performance.copy(),
            "child_strategies": list(self._child_strategies.keys()),
            "total_children": len(self._child_strategies),
            "total_results": len(self._consolidated_results),
            "has_errors": self.has_errors()
        }
        
        # Calculate additional metrics
        if metrics["extraction_stats"]["total_attempts"] > 0:
            metrics["success_rate"] = (
                metrics["extraction_stats"]["successful_extractions"] / 
                metrics["extraction_stats"]["total_attempts"]
            )
        else:
            metrics["success_rate"] = 0.0
        
        return metrics
    
    def add_strategy(self, strategy: BaseStrategy) -> None:
        """
        Add a child strategy and initialize performance tracking.
        
        Args:
            strategy: The strategy to add
        """
        super().add_strategy(strategy)
        
        # Initialize performance tracking for the new strategy
        strategy_name = strategy.name
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "success_rate": 0.0,
                "avg_confidence": 0.0,
                "avg_execution_time": 0.0,
                "total_confidence": 0.0,
                "total_execution_time": 0.0
            }
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate the provided configuration.
        
        Args:
            config: The configuration to validate
        
        Returns:
            True if the configuration is valid, False otherwise
        """
        # Validate numeric values
        numeric_fields = ['max_depth', 'max_pages', 'max_concurrent', 'timeout']
        for field in numeric_fields:
            if field in config and not isinstance(config[field], (int, float)):
                logger.warning(f"{field} must be a number")
                return False
        
        # Validate boolean values
        boolean_fields = ['include_external', 'parallel_execution', 'deduplicate']
        for field in boolean_fields:
            if field in config and not isinstance(config[field], bool):
                logger.warning(f"{field} must be a boolean")
                return False
        
        # Validate thresholds
        threshold_fields = ['fallback_threshold', 'confidence_threshold']
        for field in threshold_fields:
            if field in config:
                if not isinstance(config[field], (int, float)):
                    logger.warning(f"{field} must be a number")
                    return False
                if not 0 <= config[field] <= 1:
                    logger.warning(f"{field} must be between 0 and 1")
                    return False
        
        return True


def create_multi_strategy_v2(
    context: StrategyContext,
    strategy_names: List[str],
    strategy_factory: Optional[StrategyFactory] = None,
    config: Optional[Dict[str, Any]] = None
) -> MultiStrategyV2:
    """
    Factory function to create a MultiStrategyV2 with the specified strategies.
    
    Args:
        context: The strategy context
        strategy_names: List of strategy names to include
        strategy_factory: Factory to create strategies (if None, uses context to create one)
        config: Configuration for the multi-strategy
    
    Returns:
        Configured MultiStrategyV2 instance
    """
    # Use provided factory or create one
    if strategy_factory is None:
        strategy_factory = StrategyFactory(context)
    
    # Create the multi-strategy
    multi_strategy = MultiStrategyV2(context=context, config=config)
    
    # Add each strategy
    for strategy_name in strategy_names:
        try:
            strategy = strategy_factory.get_strategy(strategy_name)
            multi_strategy.add_strategy(strategy)
            logger.info(f"Added strategy {strategy_name} to multi-strategy")
        except ValueError as e:
            logger.warning(f"Could not add strategy {strategy_name}: {str(e)}")
    
    return multi_strategy