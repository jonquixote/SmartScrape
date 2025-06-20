"""
Fallback Framework for SmartScrape Strategies

This module provides a framework for implementing graceful degradation in the strategy pattern.
It allows the system to:
1. Define fallback chains for strategies
2. Determine when to trigger fallbacks based on various conditions
3. Assess quality of strategy results
4. Recover and consolidate partial results from multiple strategies

The fallback framework integrates with the existing strategy pattern and error handling
framework to provide robust error recovery and graceful degradation.
"""

import logging
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Any, Optional, Set, Type, Union, Callable, Tuple
import copy

from strategies.core.strategy_interface import BaseStrategy
from strategies.core.strategy_context import StrategyContext
from strategies.core.strategy_types import StrategyType, StrategyCapability
from strategies.core.composite_strategy import FallbackStrategy

# Configure logging
logger = logging.getLogger(__name__)

class FallbackCondition(ABC):
    """
    Abstract base class for conditions that trigger fallback strategies.
    
    Fallback conditions are used to determine when to activate a fallback strategy.
    Different implementations can check for specific error types, retry attempts,
    timeouts, or quality metrics.
    """
    
    @abstractmethod
    def should_fallback(self, context: Dict[str, Any]) -> bool:
        """
        Check if fallback should be triggered based on the provided context.
        
        Args:
            context: Dictionary with information about the current execution state
            
        Returns:
            True if fallback should be triggered, False otherwise
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of this fallback condition."""
        pass
    
    @property
    def description(self) -> str:
        """Get a description of this fallback condition."""
        return f"Fallback condition: {self.name}"


class ErrorBasedCondition(FallbackCondition):
    """Triggers fallback based on specific error categories or severity."""
    
    def __init__(self, 
                error_categories: Optional[Set[str]] = None,
                error_severities: Optional[Set[str]] = None,
                error_messages: Optional[List[str]] = None):
        """
        Initialize with error criteria.
        
        Args:
            error_categories: Set of error categories that trigger fallback
            error_severities: Set of error severities that trigger fallback
            error_messages: List of error message substrings that trigger fallback
        """
        self.error_categories = error_categories or set()
        self.error_severities = error_severities or set()
        self.error_messages = error_messages or []
    
    def should_fallback(self, context: Dict[str, Any]) -> bool:
        """Check if error matches criteria for fallback."""
        error = context.get('error')
        if not error:
            return False
            
        # Check error category
        if 'category' in error and error['category'] in self.error_categories:
            return True
            
        # Check error severity
        if 'severity' in error and error['severity'] in self.error_severities:
            return True
            
        # Check error message
        if 'message' in error:
            for substring in self.error_messages:
                if substring in error['message']:
                    return True
                    
        return False
    
    @property
    def name(self) -> str:
        """Get the name of this fallback condition."""
        return "error_based"


class AttemptBasedCondition(FallbackCondition):
    """Triggers fallback after a specified number of attempts."""
    
    def __init__(self, max_attempts: int = 3):
        """
        Initialize with maximum attempts.
        
        Args:
            max_attempts: Number of attempts before triggering fallback
        """
        self.max_attempts = max_attempts
    
    def should_fallback(self, context: Dict[str, Any]) -> bool:
        """Check if attempt count exceeds max attempts."""
        attempts = context.get('attempts', 0)
        return attempts >= self.max_attempts
    
    @property
    def name(self) -> str:
        """Get the name of this fallback condition."""
        return "attempt_based"


class TimeoutCondition(FallbackCondition):
    """Triggers fallback if execution time exceeds a threshold."""
    
    def __init__(self, timeout_seconds: float = 30.0):
        """
        Initialize with timeout threshold.
        
        Args:
            timeout_seconds: Execution time threshold in seconds
        """
        self.timeout_seconds = timeout_seconds
    
    def should_fallback(self, context: Dict[str, Any]) -> bool:
        """Check if execution time exceeds timeout."""
        start_time = context.get('start_time')
        if not start_time:
            return False
            
        elapsed_time = time.time() - start_time
        return elapsed_time >= self.timeout_seconds
    
    @property
    def name(self) -> str:
        """Get the name of this fallback condition."""
        return "timeout"


class ResultQualityCondition(FallbackCondition):
    """Triggers fallback if result quality is below threshold."""
    
    def __init__(self, min_quality_score: float = 0.5, assessor: Optional['StrategyResultQualityAssessor'] = None):
        """
        Initialize with quality threshold.
        
        Args:
            min_quality_score: Minimum acceptable quality score (0-1)
            assessor: Quality assessor to use
        """
        self.min_quality_score = min_quality_score
        self.assessor = assessor or StrategyResultQualityAssessor()
    
    def should_fallback(self, context: Dict[str, Any]) -> bool:
        """Check if result quality is below threshold."""
        result = context.get('result')
        if not result:
            return True
            
        quality_score = self.assessor.get_quality_score(result)
        return quality_score < self.min_quality_score
    
    @property
    def name(self) -> str:
        """Get the name of this fallback condition."""
        return "quality_based"


class CompositeCondition(FallbackCondition):
    """Combines multiple fallback conditions with AND or OR logic."""
    
    class Logic(Enum):
        """Logic for combining conditions."""
        AND = "and"
        OR = "or"
    
    def __init__(self, conditions: List[FallbackCondition], logic: Logic = Logic.OR):
        """
        Initialize with conditions and logical operator.
        
        Args:
            conditions: List of fallback conditions to combine
            logic: Logic to use when combining conditions (AND or OR)
        """
        self.conditions = conditions
        self.logic = logic
    
    def should_fallback(self, context: Dict[str, Any]) -> bool:
        """Check all conditions based on the specified logic."""
        if not self.conditions:
            return False
            
        if self.logic == self.Logic.AND:
            return all(condition.should_fallback(context) for condition in self.conditions)
        else:  # OR logic
            return any(condition.should_fallback(context) for condition in self.conditions)
    
    @property
    def name(self) -> str:
        """Get the name of this composite fallback condition."""
        condition_names = [condition.name for condition in self.conditions]
        logic_str = " AND " if self.logic == self.Logic.AND else " OR "
        return f"composite({logic_str.join(condition_names)})"


class StrategyResultQualityAssessor:
    """
    Assesses the quality of strategy results based on multiple metrics.
    
    This class provides methods to evaluate results in terms of:
    - Completeness: Checks if required fields are present
    - Confidence: Estimates the reliability of the extracted data
    - Relevance: Checks if the result is relevant to the original request
    - Overall Quality: Combines the above metrics into a single score
    """
    
    def __init__(self, required_fields: Optional[List[str]] = None, 
                weights: Optional[Dict[str, float]] = None):
        """
        Initialize the assessor with optional configuration.
        
        Args:
            required_fields: List of fields required for a complete result
            weights: Dictionary of weight factors for different quality aspects
        """
        self.required_fields = required_fields or []
        
        # Default weights for quality aspects
        self.weights = {
            'completeness': 0.4,
            'confidence': 0.3,
            'relevance': 0.3
        }
        
        # Override with custom weights if provided
        if weights:
            self.weights.update(weights)
            
        # Normalize weights to sum to 1.0
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            for key in self.weights:
                self.weights[key] /= total_weight
    
    def assess_completeness(self, result: Dict[str, Any]) -> float:
        """
        Check how complete the result is based on required fields.
        
        Args:
            result: The result to assess
            
        Returns:
            Completeness score between 0 and 1
        """
        if not result:
            return 0.0
            
        if not self.required_fields:
            # No required fields specified, check if result is not empty
            return 1.0 if result else 0.0
            
        # Count how many required fields are present
        present_count = sum(1 for field in self.required_fields if field in result)
        return present_count / len(self.required_fields)
    
    def assess_confidence(self, result: Dict[str, Any]) -> float:
        """
        Estimate the reliability of the extracted data.
        
        Args:
            result: The result to assess
            
        Returns:
            Confidence score between 0 and 1
        """
        # Check if result contains confidence indicators
        if not result:
            return 0.0
            
        # If result has explicit confidence score, use it
        if 'confidence' in result:
            return min(1.0, max(0.0, float(result['confidence'])))
            
        # Check for other confidence indicators
        confidence_indicators = [
            'score', 'probability', 'certainty', 'accuracy', 'precision'
        ]
        
        for indicator in confidence_indicators:
            if indicator in result:
                try:
                    return min(1.0, max(0.0, float(result[indicator])))
                except (ValueError, TypeError):
                    pass
        
        # Default confidence based on result structure
        # More keys and nested structures suggest higher confidence
        keys_count = len(result.keys())
        nested_count = sum(1 for v in result.values() if isinstance(v, (dict, list)))
        
        # Heuristic: more structured data often indicates higher confidence
        structure_score = min(1.0, (keys_count + nested_count * 2) / 20)
        
        return structure_score
    
    def assess_relevance(self, result: Dict[str, Any]) -> float:
        """
        Check if the result is relevant to the original request.
        
        Args:
            result: The result to assess
            
        Returns:
            Relevance score between 0 and 1
        """
        # This is a placeholder implementation
        # A real implementation would compare result to the original request
        # For now, we'll assume all non-empty results are relevant
        if not result:
            return 0.0
            
        # If result has explicit relevance score, use it
        if 'relevance' in result:
            return min(1.0, max(0.0, float(result['relevance'])))
            
        # Default relevance based on result size
        return min(1.0, len(str(result)) / 1000)
    
    def get_quality_score(self, result: Dict[str, Any]) -> float:
        """
        Calculate overall quality score based on completeness, confidence, and relevance.
        
        Args:
            result: The result to assess
            
        Returns:
            Quality score between 0 and 1
        """
        if not result:
            return 0.0
            
        # Calculate individual scores
        completeness = self.assess_completeness(result)
        confidence = self.assess_confidence(result)
        relevance = self.assess_relevance(result)
        
        # Calculate weighted average
        quality_score = (
            completeness * self.weights['completeness'] +
            confidence * self.weights['confidence'] +
            relevance * self.weights['relevance']
        )
        
        return quality_score


class FallbackStrategyChain:
    """
    Manages an ordered sequence of strategies for fallback execution.
    
    This class:
    - Maintains a prioritized list of strategies to try when primary strategies fail
    - Executes strategies sequentially until success or exhaustion
    - Collects partial results from all executed strategies
    - Provides result aggregation capabilities
    - Tracks execution metrics
    """
    
    def __init__(self, primary_strategy: BaseStrategy, 
                fallback_strategies: List[BaseStrategy],
                condition: Optional[FallbackCondition] = None):
        """
        Initialize the fallback chain.
        
        Args:
            primary_strategy: The main strategy to try first
            fallback_strategies: Ordered list of fallback strategies to try
            condition: Condition that triggers fallback
        """
        self.primary_strategy = primary_strategy
        self.fallback_strategies = fallback_strategies
        self.condition = condition or CompositeCondition([
            ErrorBasedCondition(error_severities={'critical', 'error'}),
            AttemptBasedCondition(max_attempts=3)
        ])
        
        # Metrics tracking
        self.metrics = {
            'attempts': 0,
            'fallbacks_triggered': 0,
            'successful_fallbacks': 0,
            'total_execution_time': 0.0,
            'strategy_times': {},
            'strategy_results': {}
        }
        
        # Partial results from all strategies
        self.partial_results = []
    
    def execute(self, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Execute the strategy chain with fallbacks.
        
        Args:
            url: The URL to process
            **kwargs: Additional parameters for strategies
            
        Returns:
            Result from successful strategy or None if all failed
        """
        start_time = time.time()
        context = {
            'url': url,
            'start_time': start_time,
            'attempts': 0,
            'kwargs': kwargs,
            'error': None,
            'result': None
        }
        
        # Try the primary strategy first
        try:
            logger.info(f"Executing primary strategy: {self.primary_strategy.name}")
            context['attempts'] += 1
            self.metrics['attempts'] += 1
            
            strategy_start = time.time()
            result = self.primary_strategy.execute(url, **kwargs)
            strategy_time = time.time() - strategy_start
            
            self.metrics['strategy_times'][self.primary_strategy.name] = strategy_time
            
            if result:
                logger.info(f"Primary strategy succeeded: {self.primary_strategy.name}")
                context['result'] = result
                self.metrics['strategy_results'][self.primary_strategy.name] = {
                    'success': True,
                    'time': strategy_time
                }
                
                # Store for partial results collection
                self.partial_results.append({
                    'strategy': self.primary_strategy.name,
                    'result': result,
                    'success': True
                })
                
                self.metrics['total_execution_time'] = time.time() - start_time
                return result
            else:
                logger.warning(f"Primary strategy returned no result: {self.primary_strategy.name}")
                context['error'] = {
                    'message': 'Strategy returned no result',
                    'severity': 'warning',
                    'category': 'execution'
                }
                self.metrics['strategy_results'][self.primary_strategy.name] = {
                    'success': False,
                    'time': strategy_time
                }
                
                # Store for partial results collection
                self.partial_results.append({
                    'strategy': self.primary_strategy.name,
                    'result': None,
                    'success': False
                })
        except Exception as e:
            logger.error(f"Primary strategy failed with error: {str(e)}")
            context['error'] = {
                'message': str(e),
                'severity': 'error',
                'category': 'execution',
                'exception': e
            }
            self.metrics['strategy_results'][self.primary_strategy.name] = {
                'success': False,
                'error': str(e)
            }
            
            # Store for partial results collection
            self.partial_results.append({
                'strategy': self.primary_strategy.name,
                'result': None,
                'success': False,
                'error': str(e)
            })
        
        # Check if we should trigger fallbacks
        if not self.condition.should_fallback(context):
            logger.info("Fallback condition not met, not trying fallback strategies")
            self.metrics['total_execution_time'] = time.time() - start_time
            return None
            
        # Try fallback strategies in order
        self.metrics['fallbacks_triggered'] += 1
        logger.info(f"Trying {len(self.fallback_strategies)} fallback strategies")
        
        for i, strategy in enumerate(self.fallback_strategies):
            try:
                logger.info(f"Executing fallback strategy {i+1}/{len(self.fallback_strategies)}: {strategy.name}")
                context['attempts'] += 1
                self.metrics['attempts'] += 1
                
                strategy_start = time.time()
                result = strategy.execute(url, **kwargs)
                strategy_time = time.time() - strategy_start
                
                self.metrics['strategy_times'][strategy.name] = strategy_time
                
                if result:
                    logger.info(f"Fallback strategy succeeded: {strategy.name}")
                    context['result'] = result
                    self.metrics['strategy_results'][strategy.name] = {
                        'success': True,
                        'time': strategy_time
                    }
                    self.metrics['successful_fallbacks'] += 1
                    
                    # Store for partial results collection
                    self.partial_results.append({
                        'strategy': strategy.name,
                        'result': result,
                        'success': True
                    })
                    
                    self.metrics['total_execution_time'] = time.time() - start_time
                    return result
                else:
                    logger.warning(f"Fallback strategy returned no result: {strategy.name}")
                    self.metrics['strategy_results'][strategy.name] = {
                        'success': False,
                        'time': strategy_time
                    }
                    
                    # Store for partial results collection
                    self.partial_results.append({
                        'strategy': strategy.name,
                        'result': None,
                        'success': False
                    })
            except Exception as e:
                logger.error(f"Fallback strategy failed with error: {str(e)}")
                self.metrics['strategy_results'][strategy.name] = {
                    'success': False,
                    'error': str(e)
                }
                
                # Store for partial results collection
                self.partial_results.append({
                    'strategy': strategy.name,
                    'result': None,
                    'success': False,
                    'error': str(e)
                })
        
        # If we get here, all strategies failed
        logger.warning("All strategies in fallback chain failed")
        
        # Try to recover with partial results
        recovered_result = self.recover_with_partial_results()
        if recovered_result:
            logger.info("Successfully recovered partial results")
            
        self.metrics['total_execution_time'] = time.time() - start_time
        return recovered_result
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics for the fallback chain."""
        return copy.deepcopy(self.metrics)
    
    def recover_with_partial_results(self) -> Optional[Dict[str, Any]]:
        """
        Attempt to recover a usable result from partial results.
        
        This method tries to construct a useful result from partial successful results
        across multiple strategies, even if none were fully successful.
        
        Returns:
            A recovered result or None if recovery is not possible
        """
        # Filter for any partial successes
        partial_successes = [
            item for item in self.partial_results 
            if item.get('result') and isinstance(item.get('result'), dict)
        ]
        
        if not partial_successes:
            logger.info("No partial successes to recover from")
            return None
            
        if len(partial_successes) == 1:
            # Only one partial success, return it directly
            return partial_successes[0]['result']
            
        # Multiple partial successes, attempt to merge them
        try:
            return merge_strategy_results([item['result'] for item in partial_successes])
        except Exception as e:
            logger.error(f"Error merging partial results: {str(e)}")
            
            # Fall back to the "best" partial result
            best_result = None
            best_score = -1
            assessor = StrategyResultQualityAssessor()
            
            for item in partial_successes:
                score = assessor.get_quality_score(item['result'])
                if score > best_score:
                    best_score = score
                    best_result = item['result']
                    
            if best_result:
                logger.info(f"Selected best partial result with score {best_score}")
                
            return best_result


class FallbackRegistry:
    """
    Registry for managing available fallback strategies.
    
    This class:
    - Maintains mappings between strategy types and suitable fallback strategies
    - Provides methods to find and create appropriate fallback chains
    - Suggests fallbacks based on the type of error
    """
    
    def __init__(self):
        """Initialize the fallback registry."""
        # Map strategy types to lists of fallback strategy classes
        self._fallbacks_by_type: Dict[StrategyType, List[Type[BaseStrategy]]] = {
            strategy_type: [] for strategy_type in StrategyType
        }
        
        # Map error categories to suggested fallback strategies
        self._fallbacks_by_error: Dict[str, List[Type[BaseStrategy]]] = {}
        
        # Additional metadata about registered fallbacks
        self._fallback_metadata: Dict[str, Dict[str, Any]] = {}
    
    def register_fallback(self, strategy_type: StrategyType, 
                         fallback_strategy: Type[BaseStrategy],
                         error_categories: Optional[List[str]] = None,
                         priority: int = 100,
                         metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a fallback strategy for a specific strategy type.
        
        Args:
            strategy_type: The strategy type this fallback applies to
            fallback_strategy: The fallback strategy class
            error_categories: Optional list of error categories this fallback handles
            priority: Priority value (lower values = higher priority)
            metadata: Additional metadata about the fallback
        """
        # Register for strategy type
        if strategy_type not in self._fallbacks_by_type:
            self._fallbacks_by_type[strategy_type] = []
            
        # Check if already registered to avoid duplicates
        for existing in self._fallbacks_by_type[strategy_type]:
            if existing == fallback_strategy:
                logger.warning(f"Fallback strategy {fallback_strategy.__name__} already registered for {strategy_type}")
                return
                
        self._fallbacks_by_type[strategy_type].append(fallback_strategy)
        
        # Register for error categories if provided
        if error_categories:
            for category in error_categories:
                if category not in self._fallbacks_by_error:
                    self._fallbacks_by_error[category] = []
                self._fallbacks_by_error[category].append(fallback_strategy)
        
        # Store metadata
        strategy_name = fallback_strategy.__name__
        metadata_dict = {
            'priority': priority,
            'error_categories': error_categories or []
        }
        if metadata:
            metadata_dict.update(metadata)
        self._fallback_metadata[strategy_name] = metadata_dict
        
        logger.info(f"Registered fallback {fallback_strategy.__name__} for strategy type {strategy_type}")
    
    def get_fallbacks_for_type(self, strategy_type: StrategyType) -> List[Type[BaseStrategy]]:
        """
        Get fallback strategies for a specific strategy type.
        
        Args:
            strategy_type: The strategy type to get fallbacks for
            
        Returns:
            List of fallback strategy classes, ordered by priority
        """
        fallbacks = self._fallbacks_by_type.get(strategy_type, [])
        
        # Sort by priority
        return sorted(fallbacks, key=lambda s: self._fallback_metadata.get(s.__name__, {}).get('priority', 100))
    
    def create_fallback_chain(self, strategy: BaseStrategy, 
                             context: StrategyContext,
                             condition: Optional[FallbackCondition] = None) -> FallbackStrategyChain:
        """
        Create a fallback chain for a specific strategy.
        
        Args:
            strategy: The primary strategy
            context: Strategy context for creating fallback strategies
            condition: Optional condition for triggering fallbacks
            
        Returns:
            A configured FallbackStrategyChain instance
        """
        # Determine strategy type from strategy metadata
        strategy_type = None
        if hasattr(strategy, '_metadata') and hasattr(strategy._metadata, 'strategy_type'):
            strategy_type = strategy._metadata.strategy_type
        else:
            # Default to SPECIAL_PURPOSE if type can't be determined
            strategy_type = StrategyType.SPECIAL_PURPOSE
            logger.warning(f"Could not determine strategy type for {strategy.name}, using default")
        
        # Get appropriate fallbacks for this strategy type
        fallback_classes = self.get_fallbacks_for_type(strategy_type)
        
        # Create instances of fallback strategies
        fallback_strategies = []
        for fallback_class in fallback_classes:
            try:
                fallback = fallback_class(context)
                fallback.initialize()
                fallback_strategies.append(fallback)
            except Exception as e:
                logger.error(f"Error creating fallback strategy {fallback_class.__name__}: {str(e)}")
        
        return FallbackStrategyChain(strategy, fallback_strategies, condition)
    
    def suggest_fallbacks(self, strategy: BaseStrategy, 
                         error: Dict[str, Any]) -> List[Type[BaseStrategy]]:
        """
        Suggest fallback strategies based on the strategy and error.
        
        Args:
            strategy: The failed strategy
            error: Error information dictionary
            
        Returns:
            List of suggested fallback strategy classes
        """
        suggested = []
        
        # Check error category-specific fallbacks
        error_category = error.get('category')
        if error_category and error_category in self._fallbacks_by_error:
            suggested.extend(self._fallbacks_by_error[error_category])
        
        # Add strategy type-specific fallbacks
        if hasattr(strategy, '_metadata') and hasattr(strategy._metadata, 'strategy_type'):
            strategy_type = strategy._metadata.strategy_type
            suggested.extend(self.get_fallbacks_for_type(strategy_type))
        
        # Remove duplicates
        unique_suggested = []
        seen = set()
        for fallback in suggested:
            if fallback.__name__ not in seen:
                seen.add(fallback.__name__)
                unique_suggested.append(fallback)
        
        return unique_suggested


# Recovery utility functions

def merge_strategy_results(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge results from multiple strategies into a single consolidated result.
    
    This function attempts to intelligently combine partial results from different strategies
    into a more complete overall result.
    
    Args:
        results_list: List of result dictionaries to merge
        
    Returns:
        A merged result dictionary
        
    Raises:
        ValueError: If results_list is empty
    """
    if not results_list:
        raise ValueError("Cannot merge empty results list")
        
    if len(results_list) == 1:
        return results_list[0]
    
    # Start with an empty merged result
    merged = {}
    
    # Collect all keys from all results
    all_keys = set()
    for result in results_list:
        all_keys.update(result.keys())
    
    # Process each key across all results
    for key in all_keys:
        # Collect all non-None values for this key
        values = [result[key] for result in results_list if key in result and result[key] is not None]
        
        if not values:
            # No values for this key
            continue
            
        if len(values) == 1:
            # Only one value, use it directly
            merged[key] = values[0]
            continue
            
        # Multiple values - handle based on type
        if all(isinstance(v, dict) for v in values):
            # For dictionaries, recursively merge
            merged[key] = merge_strategy_results(values)
        elif all(isinstance(v, list) for v in values):
            # For lists, combine and deduplicate
            combined = []
            for v in values:
                for item in v:
                    if item not in combined:
                        combined.append(item)
            merged[key] = combined
        else:
            # For scalar values, use the most common value
            # or if equal frequency, use the one with highest confidence
            value_counts = {}
            for i, result in enumerate(results_list):
                if key in result and result[key] is not None:
                    value = result[key]
                    str_value = str(value)  # Convert to string for counting
                    if str_value not in value_counts:
                        value_counts[str_value] = {
                            'count': 0,
                            'confidence': result.get('confidence', {}).get(key, 0.5),
                            'value': value
                        }
                    value_counts[str_value]['count'] += 1
                    
            # Find most common value
            most_common = sorted(
                value_counts.values(), 
                key=lambda x: (x['count'], x['confidence']), 
                reverse=True
            )
            if most_common:
                merged[key] = most_common[0]['value']
    
    return merged


def extract_best_components(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract the best components from multiple results to create an optimal combined result.
    
    Unlike merge_strategy_results which tries to combine all data, this function
    selectively picks the best version of each component based on quality assessment.
    
    Args:
        results_list: List of result dictionaries
        
    Returns:
        A composite result with the best components
    """
    if not results_list:
        return {}
        
    if len(results_list) == 1:
        return results_list[0]
    
    # Create assessor for evaluating components
    assessor = StrategyResultQualityAssessor()
    
    # Final composite result
    composite = {}
    
    # Get all unique keys
    all_keys = set()
    for result in results_list:
        all_keys.update(result.keys())
    
    # For each key, select the best version
    for key in all_keys:
        best_value = None
        best_score = -1
        
        for result in results_list:
            if key not in result or result[key] is None:
                continue
                
            # Assess this component's quality
            component_quality = 0.0
            
            # If the component is a dictionary, assess it directly
            if isinstance(result[key], dict):
                component_quality = assessor.get_quality_score(result[key])
            else:
                # For non-dict values, use confidence if available or a default score
                component_quality = result.get('confidence', {}).get(key, 0.5)
                
                # Adjust score based on value properties (length for strings, etc.)
                if isinstance(result[key], str):
                    # Longer, non-empty strings often have more information
                    component_quality *= min(1.0, len(result[key]) / 100) if result[key] else 0
                elif isinstance(result[key], list):
                    # More items often means more complete data
                    component_quality *= min(1.0, len(result[key]) / 10) if result[key] else 0
            
            if component_quality > best_score:
                best_score = component_quality
                best_value = result[key]
        
        if best_value is not None:
            composite[key] = best_value
    
    return composite


def rebuild_from_fragments(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Attempt to rebuild a coherent result from fragments across multiple partial results.
    
    This function is more aggressive than merge_strategy_results when dealing with
    incomplete results, trying to fill gaps and reconstruct a complete result.
    
    Args:
        results_list: List of partial result dictionaries
        
    Returns:
        A reconstructed result dictionary
    """
    if not results_list:
        return {}
        
    if len(results_list) == 1:
        return results_list[0]
    
    # Start with the most complete result as a base
    assessor = StrategyResultQualityAssessor()
    base_result = max(results_list, key=assessor.assess_completeness)
    
    # Clone the base to avoid modifying the original
    reconstructed = copy.deepcopy(base_result)
    
    # Identify missing or low-confidence components in the base
    missing_keys = set()
    low_confidence_keys = set()
    
    # Get all keys from all results
    all_keys = set()
    for result in results_list:
        all_keys.update(result.keys())
    
    # Check for missing or low confidence keys in base result
    for key in all_keys:
        if key not in base_result or base_result[key] is None:
            missing_keys.add(key)
        elif isinstance(base_result.get('confidence', {}), dict) and base_result.get('confidence', {}).get(key, 1.0) < 0.5:
            low_confidence_keys.add(key)
    
    # Sort other results by completeness score (descending)
    other_results = sorted(
        [r for r in results_list if r != base_result],
        key=assessor.assess_completeness,
        reverse=True
    )
    
    # Try to fill in missing pieces from other results
    for key in missing_keys.union(low_confidence_keys):
        for result in other_results:
            if key in result and result[key] is not None:
                # For low confidence keys, only replace if other result has higher confidence
                if key in low_confidence_keys:
                    base_confidence = base_result.get('confidence', {}).get(key, 0.5)
                    other_confidence = result.get('confidence', {}).get(key, 0.0)
                    
                    if other_confidence <= base_confidence:
                        continue
                
                # Fill in the missing or low confidence component
                reconstructed[key] = result[key]
                
                # Update confidence if available
                if 'confidence' in result and isinstance(result['confidence'], dict) and key in result['confidence']:
                    if 'confidence' not in reconstructed:
                        reconstructed['confidence'] = {}
                    reconstructed['confidence'][key] = result['confidence'][key]
                
                break
    
    return reconstructed


def estimate_result_quality(result: Dict[str, Any], required_fields: Optional[List[str]] = None) -> float:
    """
    Estimate the quality of a strategy result.
    
    Args:
        result: The result to evaluate
        required_fields: Optional list of fields required for a complete result
        
    Returns:
        Quality score between 0 and 1
    """
    assessor = StrategyResultQualityAssessor(required_fields=required_fields)
    return assessor.get_quality_score(result)


# Integrate with strategy framework

def integrate_with_strategy_context(context: StrategyContext) -> None:
    """
    Integrate fallback framework with StrategyContext.
    
    This function adds fallback-related methods to the strategy context.
    
    Args:
        context: The StrategyContext instance
    """
    # Create a fallback registry
    fallback_registry = FallbackRegistry()
    
    # Register it as a service
    context.register_service("fallback_registry", fallback_registry)
    
    # Create an assessor and register it
    assessor = StrategyResultQualityAssessor()
    context.register_service("result_quality_assessor", assessor)


# Helper to create a composite fallback condition
def create_fallback_condition(
    error_categories: Optional[Set[str]] = None,
    error_severities: Optional[Set[str]] = None,
    max_attempts: Optional[int] = None,
    timeout_seconds: Optional[float] = None,
    min_quality_score: Optional[float] = None,
    logic: CompositeCondition.Logic = CompositeCondition.Logic.OR
) -> FallbackCondition:
    """
    Create a composite fallback condition from various criteria.
    
    Args:
        error_categories: Set of error categories that trigger fallback
        error_severities: Set of error severities that trigger fallback
        max_attempts: Maximum attempts before triggering fallback
        timeout_seconds: Timeout in seconds before triggering fallback
        min_quality_score: Minimum quality score to avoid fallback
        logic: Logic for combining conditions (AND or OR)
        
    Returns:
        A configured fallback condition
    """
    conditions = []
    
    if error_categories or error_severities:
        conditions.append(ErrorBasedCondition(
            error_categories=error_categories,
            error_severities=error_severities
        ))
    
    if max_attempts is not None:
        conditions.append(AttemptBasedCondition(max_attempts=max_attempts))
    
    if timeout_seconds is not None:
        conditions.append(TimeoutCondition(timeout_seconds=timeout_seconds))
    
    if min_quality_score is not None:
        conditions.append(ResultQualityCondition(min_quality_score=min_quality_score))
    
    if not conditions:
        # Default condition
        return CompositeCondition([
            ErrorBasedCondition(error_severities={'critical', 'error'}),
            AttemptBasedCondition(max_attempts=3)
        ], logic=CompositeCondition.Logic.OR)
    
    if len(conditions) == 1:
        return conditions[0]
    
    return CompositeCondition(conditions, logic=logic)