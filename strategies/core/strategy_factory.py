"""
Strategy Factory module for creating and managing strategy instances.

This module provides a factory for registering, creating, and selecting strategy
instances based on capabilities and requirements.
"""

import logging
from typing import Dict, Type, Any, Optional, List, Set

from strategies.core.strategy_interface import BaseStrategy
from strategies.core.strategy_context import StrategyContext
from strategies.core.strategy_types import StrategyCapability, StrategyType, StrategyMetadata

logger = logging.getLogger(__name__)

class StrategyFactory:
    """Factory for creating and managing strategy instances."""

    def __init__(self, context: StrategyContext):
        """
        Initialize the strategy factory.
        
        Args:
            context: The strategy context to use for creating strategies
        """
        self.context = context
        self._strategy_classes: Dict[str, Type[BaseStrategy]] = {}
        self._strategy_metadata: Dict[str, StrategyMetadata] = {}
        self.logger = logging.getLogger(__name__)

    def register_strategy(self, strategy_class: Type[BaseStrategy]) -> None:
        """
        Register a strategy class and its metadata with the factory.
        
        Args:
            strategy_class: The strategy class to register
        
        Raises:
            ValueError: If the strategy class doesn't have valid metadata
        """
        # Create a temporary instance to get the name
        try:
            temp_instance = strategy_class(self.context)
        except TypeError:
            # Fallback: try without context for backward compatibility
            temp_instance = strategy_class()
        strategy_name = temp_instance.name
        
        # Check for metadata
        metadata = getattr(strategy_class, '_metadata', None)
        if not metadata or not isinstance(metadata, StrategyMetadata):
            raise ValueError(f"Strategy class {strategy_class.__name__} is missing valid StrategyMetadata.")
        
        # Register strategy class and metadata
        self._strategy_classes[strategy_name] = strategy_class
        self._strategy_metadata[strategy_name] = metadata
        
        self.logger.info(f"Registered strategy: {strategy_name}")

    def get_strategy(self, strategy_name: str) -> BaseStrategy:
        """
        Get a strategy instance by name.
        
        Args:
            strategy_name: Name of the strategy to get
        
        Returns:
            Strategy instance
        
        Raises:
            ValueError: If the strategy is not registered
        """
        if strategy_name not in self._strategy_classes:
            available_strategies = list(self._strategy_classes.keys())
            self.logger.error(f"Strategy '{strategy_name}' not registered. Available strategies: {available_strategies}")
            
            # Provide suggestions for common misnamed strategies
            suggestions = []
            if strategy_name == "form_strategy":
                suggestions.extend(["form_search_engine"])
            elif strategy_name == "url_param_strategy":
                suggestions.append("url-param-search-engine")
            elif strategy_name == "ai_guided_strategy":
                suggestions.append("ai_guided")
            
            if suggestions:
                self.logger.info(f"Did you mean one of: {suggestions}?")
            
            raise ValueError(f"Strategy '{strategy_name}' not registered. Available strategies: {available_strategies}")
        
        strategy_class_or_factory = self._strategy_classes[strategy_name]
        
        # Handle both class types and factory functions (lambdas)
        if callable(strategy_class_or_factory):
            if isinstance(strategy_class_or_factory, type):
                # It's a class, instantiate it with context
                return strategy_class_or_factory(self.context)
            else:
                # It's a factory function, call it
                return strategy_class_or_factory(self.context)
        else:
            # It's already an instance
            return strategy_class_or_factory

    def get_strategies_by_capability(self, required_capabilities: Set[StrategyCapability]) -> List[Type[BaseStrategy]]:
        """
        Get strategy classes that have all the required capabilities.
        
        Args:
            required_capabilities: Set of capabilities that strategies must have
        
        Returns:
            List of strategy classes that have all required capabilities
        """
        if not required_capabilities:
            return list(self._strategy_classes.values())
        
        matching_strategies = []
        
        for strategy_name, metadata in self._strategy_metadata.items():
            # Check if strategy has all required capabilities
            if all(metadata.has_capability(cap) for cap in required_capabilities):
                matching_strategies.append(self._strategy_classes[strategy_name])
        
        return matching_strategies

    def get_strategy_metadata(self, strategy_name: str) -> Optional[StrategyMetadata]:
        """
        Get metadata for a registered strategy.
        
        Args:
            strategy_name: Name of the strategy
        
        Returns:
            Strategy metadata or None if not found
        """
        return self._strategy_metadata.get(strategy_name)

    def get_all_instances(self) -> List[BaseStrategy]:
        """
        Get all registered strategy instances.
        
        Returns:
            List of strategy instances
        """
        instances = []
        for strategy_name in self._strategy_classes.keys():
            try:
                # Get an instance of each strategy
                instance = self.get_strategy(strategy_name)
                instances.append(instance)
            except Exception as e:
                self.logger.warning(f"Error creating instance of strategy {strategy_name}: {str(e)}")
        
        return instances

    def register_strategy_instance(self, strategy: BaseStrategy) -> None:
        """
        Register a strategy instance with the factory.
        
        Args:
            strategy: The strategy instance to register
        """
        strategy_name = strategy.name
        
        # Register strategy class and create a class factory function that returns this instance
        self._strategy_classes[strategy_name] = lambda ctx=None: strategy
        
        # Register metadata if available, or create basic metadata
        metadata = getattr(strategy.__class__, '_metadata', None)
        if not metadata or not isinstance(metadata, StrategyMetadata):
            # Create basic metadata for the instance
            from strategies.core.strategy_types import strategy_metadata
            metadata = strategy_metadata(
                strategy_type=StrategyType.COMPOSITE,
                capabilities=set(),
                priority=5
            )
        
        self._strategy_metadata[strategy_name] = metadata
        
        self.logger.info(f"Registered strategy instance: {strategy_name}")

    def get_all_strategy_names(self) -> List[str]:
        """
        Get the names of all registered strategies.
        
        Returns:
            List of strategy names
        """
        return list(self._strategy_classes.keys())

    def create_strategies_from_config(self, config_list: List[Dict[str, Any]]) -> List[BaseStrategy]:
        """
        Create a list of strategies based on a configuration list.
        
        Args:
            config_list: List of dictionaries with strategy configurations
        
        Returns:
            List of strategy instances
        """
        strategies = []
        
        for config in config_list:
            strategy_name = config.get('name')
            if not strategy_name or strategy_name not in self._strategy_classes:
                self.logger.warning(f"Strategy '{strategy_name}' not found, skipping")
                continue
            
            try:
                # Create the strategy
                strategy = self.get_strategy(strategy_name)
                
                # Apply configuration if provided
                strategy_config = config.get('config')
                if strategy_config and not strategy.validate_config(strategy_config):
                    self.logger.warning(f"Invalid configuration for strategy '{strategy_name}', skipping")
                    continue
                
                # Initialize the strategy
                strategy.initialize()
                
                # Add to the list
                strategies.append(strategy)
                
            except Exception as e:
                self.logger.error(f"Error creating strategy '{strategy_name}': {str(e)}")
        
        return strategies

    def create_strategy_by_type(self, strategy_type: StrategyType) -> Optional[BaseStrategy]:
        """
        Create a strategy of the specified type.
        
        Args:
            strategy_type: Type of strategy to create
        
        Returns:
            Strategy instance or None if no strategies of that type are registered
        """
        for strategy_name, metadata in self._strategy_metadata.items():
            if metadata.strategy_type == strategy_type:
                return self.get_strategy(strategy_name)
        
        return None

    def select_best_strategy(self, url: str, capabilities: Optional[Set[StrategyCapability]] = None) -> Optional[BaseStrategy]:
        """
        Select the best strategy for a given URL and capabilities.
        
        Args:
            url: URL to process
            capabilities: Optional set of required capabilities
        
        Returns:
            Best strategy for the URL or None if no suitable strategy is found
        """
        # Get strategies with the required capabilities
        candidate_strategies = self.get_strategies_by_capability(capabilities or set())
        
        # Check which strategies can handle the URL
        suitable_strategies = []
        for strategy_class in candidate_strategies:
            try:
                # Create a temporary instance to check if it can handle the URL
                temp_instance = strategy_class(self.context)
                
                # Handle both sync and async can_handle methods
                import inspect
                if inspect.iscoroutinefunction(temp_instance.can_handle):
                    # It's async, we can't await here in a sync method, so skip this check
                    # and assume it can handle (let the actual execution handle the async check)
                    self.logger.warning(f"Strategy {strategy_class.__name__} has async can_handle method, skipping check")
                    suitable_strategies.append(strategy_class)
                else:
                    # It's sync, check the result
                    can_handle_result = temp_instance.can_handle(url)
                    if can_handle_result:
                        suitable_strategies.append(strategy_class)
            except Exception as e:
                self.logger.warning(f"Error checking if strategy {strategy_class.__name__} can handle URL: {str(e)}")
        
        if not suitable_strategies:
            return None
        
        # Sort by priority - we could use metadata to determine this
        # For now, just return the first suitable strategy
        strategy_class = suitable_strategies[0]
        return strategy_class(self.context)