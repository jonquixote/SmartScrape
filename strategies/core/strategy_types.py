"""
Strategy type definitions module for the Strategy Pattern in SmartScrape.

This module defines the classification system for strategies, including:
- Strategy types (TRAVERSAL, INTERACTION, etc.)
- Strategy capabilities (what a strategy can do)
- Strategy metadata (properties describing a strategy)
- Decorator for attaching metadata to strategy classes
"""

from enum import Enum
from typing import Set, Dict, Any, Optional, Type, Callable, TypeVar
import inspect
import re

# Type variable for better type hints with decorator
T = TypeVar('T')

class StrategyType(Enum):
    """Defines the primary type and purpose of a strategy."""
    TRAVERSAL = "traversal"           # Strategies for traversing websites (BFS, DFS)
    INTERACTION = "interaction"       # Strategies that interact with forms, inputs, etc.
    EXTRACTION = "extraction"         # Strategies focused on data extraction
    SPECIAL_PURPOSE = "special_purpose"  # Strategies with specialized or combined functionality

class StrategyCapability(Enum):
    """Defines specific capabilities that a strategy may have."""
    # Browser/JavaScript capabilities
    JAVASCRIPT_EXECUTION = "javascript_execution"  # Can execute JavaScript
    DYNAMIC_CONTENT = "dynamic_content"          # Can handle dynamically loaded content
    
    # Interaction capabilities
    FORM_INTERACTION = "form_interaction"        # Can interact with forms
    API_INTERACTION = "api_interaction"          # Can interact with APIs
    LOGIN_HANDLING = "login_handling"            # Can handle login processes
    CAPTCHA_SOLVING = "captcha_solving"          # Can solve or bypass captchas
    
    # Discovery capabilities
    SITEMAP_DISCOVERY = "sitemap_discovery"      # Can discover and use sitemaps
    LINK_EXTRACTION = "link_extraction"          # Can extract links from content
    
    # Compliance capabilities
    ROBOTS_TXT_ADHERENCE = "robots_txt_adherence"  # Follows robots.txt rules
    RATE_LIMITING = "rate_limiting"               # Implements rate limiting
    
    # Proxy & rotation capabilities
    PROXY_SUPPORT = "proxy_support"               # Can use proxies
    USER_AGENT_ROTATION = "user_agent_rotation"   # Can rotate user agents
    
    # Error handling
    ERROR_HANDLING = "error_handling"             # Has robust error handling
    RETRY_MECHANISM = "retry_mechanism"           # Implements retry mechanisms
    
    # Data capabilities
    SCHEMA_EXTRACTION = "schema_extraction"       # Can extract structured schema data
    CONTENT_NORMALIZATION = "content_normalization"  # Normalizes content
    DATA_VALIDATION = "data_validation"           # Validates extracted data
    
    # Advanced capabilities
    AI_ASSISTED = "ai_assisted"                   # Uses AI for extraction/decisions
    PAGINATION_HANDLING = "pagination_handling"   # Can handle pagination
    INFINITE_SCROLL = "infinite_scroll"           # Can handle infinite scroll
    
    # Enhanced AI capabilities for new strategies
    SEMANTIC_SEARCH = "semantic_search"           # Can perform semantic content analysis
    INTENT_ANALYSIS = "intent_analysis"           # Can analyze and understand user intent
    AI_SCHEMA_GENERATION = "ai_schema_generation" # Can generate data schemas using AI
    PROGRESSIVE_CRAWLING = "progressive_crawling" # Can crawl progressively with scope adjustment
    INTELLIGENT_URL_GENERATION = "intelligent_url_generation" # Can generate intelligent URLs
    AI_PATHFINDING = "ai_pathfinding"            # Can find optimal paths using AI
    EARLY_RELEVANCE_TERMINATION = "early_relevance_termination" # Can terminate early based on relevance
    MEMORY_ADAPTIVE = "memory_adaptive"           # Can adapt based on available memory
    CIRCUIT_BREAKER = "circuit_breaker"          # Implements circuit breaker pattern
    CONSOLIDATED_AI_PROCESSING = "consolidated_ai_processing" # Can process multiple pages with AI at once


class StrategyMetadata:
    """Metadata for a strategy, describing its type, capabilities, and configuration."""
    
    def __init__(self,
                 strategy_type: StrategyType,
                 capabilities: Set[StrategyCapability],
                 description: str,
                 config_schema: Optional[Dict[str, Any]] = None):
        """
        Initialize strategy metadata.
        
        Args:
            strategy_type: The primary type of the strategy
            capabilities: Set of capabilities the strategy provides
            description: A description of what the strategy does
            config_schema: Optional schema describing configuration parameters
        """
        self.strategy_type = strategy_type
        self.capabilities = capabilities
        self.description = description
        self.config_schema = config_schema or {}
    
    def has_capability(self, capability: StrategyCapability) -> bool:
        """
        Check if the strategy has a specific capability.
        
        Args:
            capability: The capability to check for
            
        Returns:
            True if the strategy has the capability, False otherwise
        """
        return capability in self.capabilities
    
    def has_any_capability(self, capabilities: Set[StrategyCapability]) -> bool:
        """
        Check if the strategy has any of the specified capabilities.
        
        Args:
            capabilities: Set of capabilities to check for
            
        Returns:
            True if the strategy has any of the capabilities, False otherwise
        """
        return bool(self.capabilities.intersection(capabilities))
    
    def has_all_capabilities(self, capabilities: Set[StrategyCapability]) -> bool:
        """
        Check if the strategy has all of the specified capabilities.
        
        Args:
            capabilities: Set of capabilities to check for
            
        Returns:
            True if the strategy has all of the capabilities, False otherwise
        """
        return capabilities.issubset(self.capabilities)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the metadata to a dictionary for serialization."""
        return {
            "strategy_type": self.strategy_type.value,
            "capabilities": [cap.value for cap in self.capabilities],
            "description": self.description,
            "config_schema": self.config_schema
        }


def strategy_metadata(
    strategy_type: StrategyType,
    capabilities: Set[StrategyCapability],
    description: str,
    config_schema: Optional[Dict[str, Any]] = None
) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to attach metadata to a strategy class.
    
    Args:
        strategy_type: The primary type of the strategy
        capabilities: Set of capabilities the strategy provides
        description: A description of what the strategy does
        config_schema: Optional schema describing configuration parameters
        
    Returns:
        Decorator function that attaches metadata to the class
    """
    def decorator(cls: Type[T]) -> Type[T]:
        # Create and attach metadata to the class
        cls._metadata = StrategyMetadata(
            strategy_type=strategy_type,
            capabilities=capabilities,
            description=description,
            config_schema=config_schema
        )
        
        # Add a name property based on class name if not explicitly defined
        if not hasattr(cls, 'name') or not isinstance(getattr(cls, 'name'), property):
            # Convert CamelCase to snake_case and remove 'strategy' suffix
            def get_default_name(self):
                class_name = cls.__name__
                # Remove 'Strategy' suffix if present
                if class_name.endswith('Strategy'):
                    class_name = class_name[:-8]
                # Convert CamelCase to snake_case
                name = re.sub(r'(?<!^)(?=[A-Z])', '_', class_name).lower()
                return name
                
            cls.name = property(get_default_name)
        
        return cls
        
    return decorator