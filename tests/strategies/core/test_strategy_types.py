"""
Tests for strategy type definitions and metadata system.

This module tests the functionality of:
- StrategyType enum
- StrategyCapability enum
- StrategyMetadata class
- strategy_metadata decorator
"""

import pytest
from typing import Set, List, Dict, Any, Optional

from strategies.core.strategy_types import (
    StrategyType, 
    StrategyCapability, 
    StrategyMetadata,
    strategy_metadata
)

# For testing the decorator, we need a minimal class that can be decorated
class MockBaseStrategy:
    """Mock base class for testing the strategy_metadata decorator."""
    def __init__(self, context=None):
        self.context = context


# Test the StrategyType enum
def test_strategy_type_enum():
    """Test that StrategyType enum contains the expected values."""
    # Verify all expected types exist
    assert StrategyType.TRAVERSAL.value == "traversal"
    assert StrategyType.INTERACTION.value == "interaction"
    assert StrategyType.EXTRACTION.value == "extraction"
    assert StrategyType.SPECIAL_PURPOSE.value == "special_purpose"
    
    # Verify the number of types
    assert len(list(StrategyType)) == 4


# Test the StrategyCapability enum
def test_strategy_capability_enum():
    """Test that StrategyCapability enum contains the expected values."""
    # Verify key capabilities exist
    assert StrategyCapability.JAVASCRIPT_EXECUTION.value == "javascript_execution"
    assert StrategyCapability.FORM_INTERACTION.value == "form_interaction"
    assert StrategyCapability.API_INTERACTION.value == "api_interaction"
    assert StrategyCapability.RATE_LIMITING.value == "rate_limiting"
    assert StrategyCapability.ROBOTS_TXT_ADHERENCE.value == "robots_txt_adherence"
    assert StrategyCapability.AI_ASSISTED.value == "ai_assisted"
    
    # Verify the enum has a significant number of capabilities (flexible test)
    assert len(list(StrategyCapability)) >= 10


# Test the StrategyMetadata class
def test_strategy_metadata_init():
    """Test initialization of StrategyMetadata class."""
    capabilities = {
        StrategyCapability.JAVASCRIPT_EXECUTION,
        StrategyCapability.FORM_INTERACTION
    }
    
    metadata = StrategyMetadata(
        strategy_type=StrategyType.INTERACTION,
        capabilities=capabilities,
        description="Test metadata"
    )
    
    # Verify attributes were set correctly
    assert metadata.strategy_type == StrategyType.INTERACTION
    assert metadata.capabilities == capabilities
    assert metadata.description == "Test metadata"
    assert metadata.config_schema == {}  # Default empty dict
    
    # Test with config_schema
    config_schema = {"key": {"type": "string", "required": True}}
    metadata_with_schema = StrategyMetadata(
        strategy_type=StrategyType.INTERACTION,
        capabilities=capabilities,
        description="Test metadata",
        config_schema=config_schema
    )
    assert metadata_with_schema.config_schema == config_schema


def test_strategy_metadata_has_capability():
    """Test the has_capability method."""
    capabilities = {
        StrategyCapability.JAVASCRIPT_EXECUTION,
        StrategyCapability.FORM_INTERACTION
    }
    
    metadata = StrategyMetadata(
        strategy_type=StrategyType.INTERACTION,
        capabilities=capabilities,
        description="Test metadata"
    )
    
    # Test positive cases
    assert metadata.has_capability(StrategyCapability.JAVASCRIPT_EXECUTION) is True
    assert metadata.has_capability(StrategyCapability.FORM_INTERACTION) is True
    
    # Test negative case
    assert metadata.has_capability(StrategyCapability.API_INTERACTION) is False


def test_strategy_metadata_has_any_capability():
    """Test the has_any_capability method."""
    capabilities = {
        StrategyCapability.JAVASCRIPT_EXECUTION,
        StrategyCapability.FORM_INTERACTION
    }
    
    metadata = StrategyMetadata(
        strategy_type=StrategyType.INTERACTION,
        capabilities=capabilities,
        description="Test metadata"
    )
    
    # Test with one matching capability
    assert metadata.has_any_capability({StrategyCapability.JAVASCRIPT_EXECUTION}) is True
    
    # Test with one matching and one non-matching capability
    assert metadata.has_any_capability({
        StrategyCapability.JAVASCRIPT_EXECUTION,
        StrategyCapability.API_INTERACTION
    }) is True
    
    # Test with all non-matching capabilities
    assert metadata.has_any_capability({
        StrategyCapability.API_INTERACTION,
        StrategyCapability.PAGINATION_HANDLING
    }) is False


def test_strategy_metadata_has_all_capabilities():
    """Test the has_all_capabilities method."""
    capabilities = {
        StrategyCapability.JAVASCRIPT_EXECUTION,
        StrategyCapability.FORM_INTERACTION,
        StrategyCapability.ERROR_HANDLING
    }
    
    metadata = StrategyMetadata(
        strategy_type=StrategyType.INTERACTION,
        capabilities=capabilities,
        description="Test metadata"
    )
    
    # Test with subset of capabilities
    assert metadata.has_all_capabilities({
        StrategyCapability.JAVASCRIPT_EXECUTION,
        StrategyCapability.FORM_INTERACTION
    }) is True
    
    # Test with exact match
    assert metadata.has_all_capabilities(capabilities) is True
    
    # Test with one capability not in the set
    assert metadata.has_all_capabilities({
        StrategyCapability.JAVASCRIPT_EXECUTION,
        StrategyCapability.API_INTERACTION
    }) is False
    
    # Test with empty set (should return True as empty set is subset of any set)
    assert metadata.has_all_capabilities(set()) is True


def test_strategy_metadata_to_dict():
    """Test the to_dict method for serialization."""
    capabilities = {
        StrategyCapability.JAVASCRIPT_EXECUTION,
        StrategyCapability.FORM_INTERACTION
    }
    
    config_schema = {"key": {"type": "string", "required": True}}
    
    metadata = StrategyMetadata(
        strategy_type=StrategyType.INTERACTION,
        capabilities=capabilities,
        description="Test metadata",
        config_schema=config_schema
    )
    
    dict_result = metadata.to_dict()
    
    # Verify dictionary structure
    assert dict_result["strategy_type"] == "interaction"
    assert sorted(dict_result["capabilities"]) == sorted([
        "javascript_execution", 
        "form_interaction"
    ])
    assert dict_result["description"] == "Test metadata"
    assert dict_result["config_schema"] == config_schema


# Test the strategy_metadata decorator
def test_strategy_metadata_decorator():
    """Test the strategy_metadata decorator."""
    
    @strategy_metadata(
        strategy_type=StrategyType.TRAVERSAL,
        capabilities={
            StrategyCapability.ROBOTS_TXT_ADHERENCE,
            StrategyCapability.LINK_EXTRACTION
        },
        description="Test traversal strategy"
    )
    class MockTraversalStrategy(MockBaseStrategy):
        pass
    
    # Verify metadata was attached to the class
    assert hasattr(MockTraversalStrategy, '_metadata')
    assert isinstance(MockTraversalStrategy._metadata, StrategyMetadata)
    assert MockTraversalStrategy._metadata.strategy_type == StrategyType.TRAVERSAL
    assert StrategyCapability.ROBOTS_TXT_ADHERENCE in MockTraversalStrategy._metadata.capabilities
    assert StrategyCapability.LINK_EXTRACTION in MockTraversalStrategy._metadata.capabilities
    assert MockTraversalStrategy._metadata.description == "Test traversal strategy"


def test_strategy_metadata_decorator_with_config_schema():
    """Test the strategy_metadata decorator with config schema."""
    
    config_schema = {
        "max_depth": {"type": "integer", "default": 3},
        "include_external": {"type": "boolean", "default": False}
    }
    
    @strategy_metadata(
        strategy_type=StrategyType.TRAVERSAL,
        capabilities={StrategyCapability.ROBOTS_TXT_ADHERENCE},
        description="Test traversal strategy with config",
        config_schema=config_schema
    )
    class MockConfigurableStrategy(MockBaseStrategy):
        pass
    
    # Verify config schema was included in metadata
    assert MockConfigurableStrategy._metadata.config_schema == config_schema


def test_strategy_metadata_decorator_adds_name_property():
    """Test that the decorator adds a name property if not present."""
    
    @strategy_metadata(
        strategy_type=StrategyType.EXTRACTION,
        capabilities={StrategyCapability.SCHEMA_EXTRACTION},
        description="Test extraction strategy"
    )
    class DataExtractionStrategy(MockBaseStrategy):
        pass
    
    # Instantiate to test the property
    strategy = DataExtractionStrategy()
    
    # Verify name property was added and is correct
    assert hasattr(strategy, 'name')
    assert strategy.name == "data_extraction"  # "DataExtractionStrategy" -> "data_extraction"


def test_strategy_metadata_decorator_respects_existing_name_property():
    """Test that the decorator doesn't override an existing name property."""
    
    @strategy_metadata(
        strategy_type=StrategyType.EXTRACTION,
        capabilities={StrategyCapability.SCHEMA_EXTRACTION},
        description="Test extraction strategy with custom name"
    )
    class CustomNameStrategy(MockBaseStrategy):
        @property
        def name(self):
            return "my_custom_name"
    
    # Instantiate to test the property
    strategy = CustomNameStrategy()
    
    # Verify name property is the custom one
    assert strategy.name == "my_custom_name"


def test_multiple_strategy_metadata_decorators():
    """Test that multiple classes can be decorated without interference."""
    
    @strategy_metadata(
        strategy_type=StrategyType.TRAVERSAL,
        capabilities={StrategyCapability.ROBOTS_TXT_ADHERENCE},
        description="Strategy A"
    )
    class StrategyA(MockBaseStrategy):
        pass
    
    @strategy_metadata(
        strategy_type=StrategyType.EXTRACTION,
        capabilities={StrategyCapability.SCHEMA_EXTRACTION},
        description="Strategy B"
    )
    class StrategyB(MockBaseStrategy):
        pass
    
    # Verify metadata is correct and separate for each class
    assert StrategyA._metadata.description == "Strategy A"
    assert StrategyB._metadata.description == "Strategy B"
    assert StrategyA._metadata.strategy_type == StrategyType.TRAVERSAL
    assert StrategyB._metadata.strategy_type == StrategyType.EXTRACTION