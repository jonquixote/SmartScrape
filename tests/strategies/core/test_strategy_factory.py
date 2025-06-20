"""
Tests for the strategy factory component of the Strategy Pattern.

This module tests the StrategyFactory class which is responsible for:
1. Registering strategy classes
2. Creating strategy instances
3. Finding strategies by name, type or capability
4. Creating strategies from configuration
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List, Set, Optional, Type

# Import the strategy factory and related components
from strategies.core.strategy_factory import StrategyFactory
from strategies.core.strategy_types import (
    StrategyType,
    StrategyCapability,
    StrategyMetadata,
    strategy_metadata
)
from strategies.core.strategy_context import StrategyContext
from strategies.core.strategy_interface import BaseStrategy


# Define mock strategy classes for testing
class MockStrategyContext:
    """Mock strategy context for testing."""
    
    def __init__(self):
        self.logger = MagicMock()
        self.service_registry = MagicMock()
        self.config = {"test_config": "value"}


# Mock strategy classes with metadata for testing
@strategy_metadata(
    strategy_type=StrategyType.TRAVERSAL,
    capabilities={StrategyCapability.ROBOTS_TXT_ADHERENCE},
    description="Mock BFS Strategy for testing."
)
class MockBFSStrategy(BaseStrategy):
    """Mock BFS Strategy for testing the factory."""
    
    def __init__(self, context=None):
        super().__init__(context)
        self._results = []
    
    def execute(self, url, **kwargs):
        """Execute the strategy."""
        result = {"url": url, "strategy": self.name, "executed": True}
        self._results.append(result)
        return result
    
    def crawl(self, start_url, **kwargs):
        """Crawl from a starting URL."""
        result = {"url": start_url, "strategy": self.name, "crawled": True}
        self._results.append(result)
        return result
    
    def extract(self, html_content, url, **kwargs):
        """Extract data from HTML."""
        result = {"url": url, "strategy": self.name, "extracted": True, "content_length": len(html_content)}
        self._results.append(result)
        return result
    
    def get_results(self):
        """Get collected results."""
        return self._results
    
    @property
    def name(self):
        return "mock_bfs"


@strategy_metadata(
    strategy_type=StrategyType.EXTRACTION,
    capabilities={StrategyCapability.AI_ASSISTED, StrategyCapability.SCHEMA_EXTRACTION},
    description="Mock AI Extraction Strategy for testing."
)
class MockAIExtractionStrategy(BaseStrategy):
    """Mock AI Extraction Strategy for testing the factory."""
    
    def __init__(self, context=None):
        super().__init__(context)
        self._results = []
        self.config = {}
    
    def execute(self, url, **kwargs):
        """Execute the strategy."""
        result = {"url": url, "strategy": self.name, "executed": True}
        self._results.append(result)
        return result
    
    def crawl(self, start_url, **kwargs):
        """Crawl from a starting URL."""
        result = {"url": start_url, "strategy": self.name, "crawled": True}
        self._results.append(result)
        return result
    
    def extract(self, html_content, url, **kwargs):
        """Extract data from HTML using AI."""
        result = {"url": url, "strategy": self.name, "extracted": True, "ai_powered": True}
        self._results.append(result)
        return result
    
    def get_results(self):
        """Get collected results."""
        return self._results
    
    def initialize_with_config(self, config):
        """Initialize the strategy with configuration."""
        self.config = config
        return self
    
    @property
    def name(self):
        return "mock_ai_extraction"


class StrategyWithoutMetadata(BaseStrategy):
    """A strategy class without metadata for testing error handling."""
    
    def __init__(self, context=None):
        super().__init__(context)
    
    def execute(self, url, **kwargs):
        pass
    
    def crawl(self, start_url, **kwargs):
        pass
    
    def extract(self, html_content, url, **kwargs):
        pass
    
    def get_results(self):
        return []
    
    @property
    def name(self):
        return "no_metadata_strategy"


# Fixtures for tests
@pytest.fixture
def mock_context():
    """Create a mock context for testing."""
    return MockStrategyContext()


@pytest.fixture
def strategy_factory(mock_context):
    """Create a strategy factory for testing."""
    factory = StrategyFactory(mock_context)
    return factory


@pytest.fixture
def populated_factory(strategy_factory):
    """Create a strategy factory with pre-registered strategies."""
    factory = strategy_factory
    factory.register_strategy(MockBFSStrategy)
    factory.register_strategy(MockAIExtractionStrategy)
    return factory


# Test cases
class TestStrategyFactoryInitialization:
    """Tests for StrategyFactory initialization."""
    
    def test_initialization(self, mock_context):
        """Test that the factory initializes correctly."""
        factory = StrategyFactory(mock_context)
        
        # Check that the factory has the expected properties
        assert factory.context is mock_context
        assert isinstance(factory._strategy_classes, dict)
        assert isinstance(factory._strategy_metadata, dict)
        assert len(factory._strategy_classes) == 0
        assert len(factory._strategy_metadata) == 0


class TestStrategyRegistration:
    """Tests for strategy registration functionality."""
    
    def test_register_strategy(self, strategy_factory):
        """Test registering a strategy class."""
        factory = strategy_factory
        
        # Register a strategy
        factory.register_strategy(MockBFSStrategy)
        
        # Check that the strategy was registered correctly
        assert "mock_bfs" in factory._strategy_classes
        assert factory._strategy_classes["mock_bfs"] is MockBFSStrategy
        assert "mock_bfs" in factory._strategy_metadata
        assert isinstance(factory._strategy_metadata["mock_bfs"], StrategyMetadata)
        assert factory._strategy_metadata["mock_bfs"].strategy_type == StrategyType.TRAVERSAL
        assert StrategyCapability.ROBOTS_TXT_ADHERENCE in factory._strategy_metadata["mock_bfs"].capabilities
    
    def test_register_multiple_strategies(self, strategy_factory):
        """Test registering multiple strategy classes."""
        factory = strategy_factory
        
        # Register multiple strategies
        factory.register_strategy(MockBFSStrategy)
        factory.register_strategy(MockAIExtractionStrategy)
        
        # Check that both strategies were registered correctly
        assert len(factory._strategy_classes) == 2
        assert len(factory._strategy_metadata) == 2
        assert "mock_bfs" in factory._strategy_classes
        assert "mock_ai_extraction" in factory._strategy_classes
    
    def test_register_strategy_without_metadata(self, strategy_factory):
        """Test that registering a strategy without metadata raises an error."""
        factory = strategy_factory
        
        # Attempt to register a strategy without metadata
        with pytest.raises(ValueError, match="missing valid StrategyMetadata"):
            factory.register_strategy(StrategyWithoutMetadata)
    
    def test_register_same_strategy_twice(self, strategy_factory):
        """Test registering the same strategy twice."""
        factory = strategy_factory
        
        # Register a strategy
        factory.register_strategy(MockBFSStrategy)
        
        # Register it again (should replace the first registration)
        factory.register_strategy(MockBFSStrategy)
        
        # Check that there's still only one registration
        assert len(factory._strategy_classes) == 1
        assert len(factory._strategy_metadata) == 1
    
    def test_register_strategy_with_duplicate_name(self, strategy_factory):
        """Test handling conflicting strategy names."""
        factory = strategy_factory
        
        # Create a strategy class with the same name but different metadata
        @strategy_metadata(
            strategy_type=StrategyType.TRAVERSAL,
            capabilities={StrategyCapability.RATE_LIMITING},
            description="Another mock BFS strategy with the same name."
        )
        class AnotherMockBFSStrategy(BaseStrategy):
            @property
            def name(self):
                return "mock_bfs"  # Same name as MockBFSStrategy
            
            def execute(self, url, **kwargs):
                pass
            
            def crawl(self, start_url, **kwargs):
                pass
            
            def extract(self, html_content, url, **kwargs):
                pass
            
            def get_results(self):
                return []
        
        # Register the first strategy
        factory.register_strategy(MockBFSStrategy)
        
        # Register the second strategy with the same name
        factory.register_strategy(AnotherMockBFSStrategy)
        
        # Check that the second registration replaced the first
        assert len(factory._strategy_classes) == 1
        assert factory._strategy_classes["mock_bfs"] is AnotherMockBFSStrategy
        assert StrategyCapability.RATE_LIMITING in factory._strategy_metadata["mock_bfs"].capabilities
        assert StrategyCapability.ROBOTS_TXT_ADHERENCE not in factory._strategy_metadata["mock_bfs"].capabilities


class TestStrategyRetrieval:
    """Tests for strategy retrieval functionality."""
    
    def test_get_strategy(self, populated_factory):
        """Test getting a strategy instance by name."""
        factory = populated_factory
        
        # Get a strategy instance
        strategy = factory.get_strategy("mock_bfs")
        
        # Check that the strategy is of the expected type
        assert isinstance(strategy, MockBFSStrategy)
        assert strategy.name == "mock_bfs"
        assert strategy.context is factory.context
    
    def test_get_nonexistent_strategy(self, populated_factory):
        """Test getting a strategy that doesn't exist."""
        factory = populated_factory
        
        # Attempt to get a non-existent strategy
        with pytest.raises(ValueError, match="not registered"):
            factory.get_strategy("nonexistent_strategy")
    
    def test_get_strategies_by_capability(self, populated_factory):
        """Test getting strategies by capability."""
        factory = populated_factory
        
        # Get strategies with a specific capability
        strategies = factory.get_strategies_by_capability({StrategyCapability.ROBOTS_TXT_ADHERENCE})
        assert len(strategies) == 1
        assert MockBFSStrategy in strategies
        
        # Get strategies with AI_ASSISTED capability
        strategies = factory.get_strategies_by_capability({StrategyCapability.AI_ASSISTED})
        assert len(strategies) == 1
        assert MockAIExtractionStrategy in strategies
        
        # Get strategies with multiple capabilities (must match all)
        strategies = factory.get_strategies_by_capability({
            StrategyCapability.AI_ASSISTED,
            StrategyCapability.SCHEMA_EXTRACTION
        })
        assert len(strategies) == 1
        assert MockAIExtractionStrategy in strategies
        
        # Get strategies with a capability that no registered strategy has
        strategies = factory.get_strategies_by_capability({StrategyCapability.FORM_INTERACTION})
        assert len(strategies) == 0
    
    def test_get_strategy_metadata(self, populated_factory):
        """Test getting strategy metadata by name."""
        factory = populated_factory
        
        # Get metadata for an existing strategy
        metadata = factory.get_strategy_metadata("mock_bfs")
        assert isinstance(metadata, StrategyMetadata)
        assert metadata.strategy_type == StrategyType.TRAVERSAL
        assert metadata.description == "Mock BFS Strategy for testing."
        
        # Get metadata for a non-existent strategy
        metadata = factory.get_strategy_metadata("nonexistent_strategy")
        assert metadata is None
    
    def test_get_all_strategy_names(self, populated_factory):
        """Test getting all strategy names."""
        factory = populated_factory
        
        # Get all strategy names
        names = factory.get_all_strategy_names()
        assert set(names) == {"mock_bfs", "mock_ai_extraction"}
        
        # Register another strategy and check again
        @strategy_metadata(
            strategy_type=StrategyType.INTERACTION,
            capabilities={StrategyCapability.FORM_INTERACTION},
            description="Mock Form Strategy for testing."
        )
        class MockFormStrategy(BaseStrategy):
            @property
            def name(self):
                return "mock_form"
            
            def execute(self, url, **kwargs):
                pass
            
            def crawl(self, start_url, **kwargs):
                pass
            
            def extract(self, html_content, url, **kwargs):
                pass
            
            def get_results(self):
                return []
        
        factory.register_strategy(MockFormStrategy)
        names = factory.get_all_strategy_names()
        assert set(names) == {"mock_bfs", "mock_ai_extraction", "mock_form"}


class TestStrategyConfiguration:
    """Tests for creating strategies from configuration."""
    
    def test_create_strategies_from_config(self, populated_factory):
        """Test creating strategies from a configuration list."""
        factory = populated_factory
        
        # Define a configuration list
        config_list = [
            {"name": "mock_bfs", "config": {"max_depth": 3, "follow_external": False}},
            {"name": "mock_ai_extraction", "config": {"model": "gpt-4", "extract_schema": True}}
        ]
        
        # Create strategies from the configuration
        strategies = factory.create_strategies_from_config(config_list)
        
        # Check that the strategies were created correctly
        assert len(strategies) == 2
        assert isinstance(strategies[0], MockBFSStrategy)
        assert isinstance(strategies[1], MockAIExtractionStrategy)
        
        # Check that the second strategy was initialized with the config
        # (assuming MockAIExtractionStrategy implements initialize_with_config)
        assert strategies[1].config == {"model": "gpt-4", "extract_schema": True}
    
    def test_create_strategies_from_config_with_missing_strategy(self, populated_factory):
        """Test creating strategies from a configuration with a missing strategy."""
        factory = populated_factory
        
        # Define a configuration list with a non-existent strategy
        config_list = [
            {"name": "mock_bfs", "config": {}},
            {"name": "nonexistent_strategy", "config": {}}
        ]
        
        # Create strategies from the configuration
        strategies = factory.create_strategies_from_config(config_list)
        
        # Check that only the existing strategy was created
        assert len(strategies) == 1
        assert isinstance(strategies[0], MockBFSStrategy)
    
    def test_create_strategies_from_config_with_invalid_config(self, populated_factory):
        """Test creating strategies from invalid configuration."""
        factory = populated_factory
        
        # Define an invalid configuration list (missing "name" key)
        invalid_config_list = [
            {"strategy": "mock_bfs", "config": {}}  # Wrong key name
        ]
        
        # Create strategies from the invalid configuration
        strategies = factory.create_strategies_from_config(invalid_config_list)
        
        # Check that no strategies were created
        assert len(strategies) == 0
    
    def test_create_strategies_from_empty_config(self, populated_factory):
        """Test creating strategies from an empty configuration."""
        factory = populated_factory
        
        # Create strategies from an empty configuration
        strategies = factory.create_strategies_from_config([])
        
        # Check that no strategies were created
        assert len(strategies) == 0


class TestAdvancedFeatures:
    """Tests for advanced StrategyFactory features."""
    
    def test_get_strategies_by_type(self, populated_factory):
        """Test getting strategies by type."""
        factory = populated_factory
        
        # Get strategies of a specific type
        traversal_strategies = factory.get_strategies_by_type(StrategyType.TRAVERSAL)
        assert len(traversal_strategies) == 1
        assert MockBFSStrategy in traversal_strategies
        
        # Get strategies of another type
        extraction_strategies = factory.get_strategies_by_type(StrategyType.EXTRACTION)
        assert len(extraction_strategies) == 1
        assert MockAIExtractionStrategy in extraction_strategies
        
        # Get strategies of a type that no registered strategy has
        interaction_strategies = factory.get_strategies_by_type(StrategyType.INTERACTION)
        assert len(interaction_strategies) == 0
    
    def test_create_strategy_with_config(self, populated_factory):
        """Test creating a single strategy with configuration."""
        factory = populated_factory
        
        # Create a strategy with configuration
        config = {"model": "gpt-4", "extract_schema": True}
        strategy = factory.create_strategy_with_config("mock_ai_extraction", config)
        
        # Check that the strategy was created correctly
        assert isinstance(strategy, MockAIExtractionStrategy)
        assert strategy.config == config
    
    def test_create_strategy_with_config_nonexistent(self, populated_factory):
        """Test creating a non-existent strategy with configuration."""
        factory = populated_factory
        
        # Attempt to create a non-existent strategy
        with pytest.raises(ValueError, match="not registered"):
            factory.create_strategy_with_config("nonexistent_strategy", {})