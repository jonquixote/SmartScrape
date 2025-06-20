"""
Base Pattern Analyzer Module

This module provides the foundation classes and interfaces for pattern analysis
in the SmartScrape framework. It defines abstract base classes that specialized
analyzers will implement.
"""

import abc
import json
import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from urllib.parse import urlparse
from bs4 import BeautifulSoup

from core.service_interface import BaseService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PatternAnalyzer")

class PatternAnalyzer(BaseService, abc.ABC):
    """
    Abstract base class for pattern analyzers.
    
    Pattern analyzers detect specific patterns in web pages and provide
    information that can be used for intelligent extraction.
    """
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize the pattern analyzer.
        
        Args:
            confidence_threshold: Minimum confidence level to consider a pattern valid
        """
        self._initialized = False
        self.confidence_threshold = confidence_threshold
        self.detected_patterns = {}

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the service with the given configuration."""
        if self._initialized:
            return
            
        # Apply configuration if provided
        if config:
            self.confidence_threshold = config.get('confidence_threshold', self.confidence_threshold)
            
        self._initialized = True
        logger.info(f"{self.name} service initialized")
    
    def shutdown(self) -> None:
        """Perform cleanup and shutdown operations."""
        self._initialized = False
        logger.info(f"{self.name} service shut down")
    
    @property
    def name(self) -> str:
        """Return the name of the service."""
        return "pattern_analyzer"

    @abc.abstractmethod
    async def analyze(self, html: str, url: str) -> Dict[str, Any]:
        """
        Analyze a page to detect patterns.
        
        Args:
            html: HTML content of the page
            url: URL of the page
            
        Returns:
            Dictionary with detected patterns and their properties
        """
        pass

    def get_domain(self, url: str) -> str:
        """
        Extract the domain from a URL.
        
        Args:
            url: URL to extract domain from
            
        Returns:
            Domain name
        """
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        return domain

    def calculate_confidence(self, evidence_points: List[float]) -> float:
        """
        Calculate confidence level based on collected evidence points.
        
        Args:
            evidence_points: List of confidence values for different aspects of a pattern
            
        Returns:
            Overall confidence score (0.0 to 1.0)
        """
        if not evidence_points:
            return 0.0
            
        # Weight more heavily towards lower scores to be conservative
        return sum(evidence_points) / len(evidence_points)
    
    def get_pattern_key(self, pattern_type: str, url: str) -> str:
        """
        Generate a unique key for a pattern.
        
        Args:
            pattern_type: Type of pattern
            url: URL where pattern was detected
            
        Returns:
            Unique pattern key
        """
        domain = self.get_domain(url)
        return f"{domain}:{pattern_type}"
    
    def parse_html(self, html: str) -> BeautifulSoup:
        """
        Parse HTML using BeautifulSoup with appropriate parser.
        
        Args:
            html: HTML content
            
        Returns:
            BeautifulSoup object
        """
        # Use lxml for speed if available, fall back to html.parser
        try:
            return BeautifulSoup(html, 'lxml')
        except:
            return BeautifulSoup(html, 'html.parser')
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert analyzer state to dictionary for serialization.
        
        Returns:
            Dictionary representation of analyzer state
        """
        return {
            "analyzer_type": self.__class__.__name__,
            "confidence_threshold": self.confidence_threshold,
            "detected_patterns": self.detected_patterns
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PatternAnalyzer':
        """
        Create an analyzer from dictionary representation.
        
        Args:
            data: Dictionary representation of analyzer state
            
        Returns:
            Pattern analyzer instance
        """
        analyzer = cls(confidence_threshold=data.get("confidence_threshold", 0.7))
        analyzer.detected_patterns = data.get("detected_patterns", {})
        return analyzer


class PatternRegistry:
    """
    Registry for holding detected patterns across different analyzers.
    This allows for sharing pattern information between components.
    """
    
    def __init__(self):
        """Initialize the pattern registry."""
        self.patterns = {}
        self.domain_patterns = {}
        
    def register_pattern(self, pattern_type: str, url: str, pattern_data: Dict[str, Any], 
                         confidence: float) -> bool:
        """
        Register a detected pattern.
        
        Args:
            pattern_type: Type of pattern (e.g., "search_form", "listing", etc.)
            url: URL where pattern was detected
            pattern_data: Pattern information
            confidence: Confidence level in the pattern (0.0 to 1.0)
            
        Returns:
            Whether the pattern was registered successfully
        """
        domain = urlparse(url).netloc
        pattern_key = f"{domain}:{pattern_type}"
        
        # Only register if confidence is high enough
        if confidence < 0.5:
            return False
            
        # Store the pattern
        self.patterns[pattern_key] = {
            "pattern_type": pattern_type,
            "domain": domain,
            "url": url,
            "pattern_data": pattern_data,
            "confidence": confidence,
            "timestamp": None  # Will be filled in when implemented
        }
        
        # Also store by domain for quicker lookup
        if domain not in self.domain_patterns:
            self.domain_patterns[domain] = {}
        
        self.domain_patterns[domain][pattern_type] = self.patterns[pattern_key]
        
        return True
    
    def get_pattern(self, pattern_type: str, url: str) -> Optional[Dict[str, Any]]:
        """
        Get a registered pattern.
        
        Args:
            pattern_type: Type of pattern to retrieve
            url: URL or domain to get pattern for
            
        Returns:
            Pattern data or None if not found
        """
        domain = urlparse(url).netloc
        pattern_key = f"{domain}:{pattern_type}"
        
        return self.patterns.get(pattern_key)
    
    def get_patterns_for_domain(self, domain: str) -> Dict[str, Dict[str, Any]]:
        """
        Get all patterns registered for a domain.
        
        Args:
            domain: Domain to get patterns for
            
        Returns:
            Dictionary of patterns by type
        """
        return self.domain_patterns.get(domain, {})
    
    def to_json(self) -> str:
        """
        Convert registry to JSON for serialization.
        
        Returns:
            JSON string
        """
        return json.dumps({
            "patterns": self.patterns,
            "domain_patterns": self.domain_patterns
        }, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'PatternRegistry':
        """
        Create registry from JSON representation.
        
        Args:
            json_str: JSON string
            
        Returns:
            Pattern registry instance
        """
        data = json.loads(json_str)
        registry = cls()
        registry.patterns = data.get("patterns", {})
        registry.domain_patterns = data.get("domain_patterns", {})
        return registry


class PatternAnalysisResult:
    """
    Class to hold the results of pattern analysis.
    This provides a structured way to access analysis results.
    """
    
    def __init__(self, 
                 url: str,
                 patterns: Dict[str, Any] = None,
                 selectors: Dict[str, List[str]] = None,
                 metadata: Dict[str, Any] = None):
        """
        Initialize pattern analysis result.
        
        Args:
            url: URL that was analyzed
            patterns: Detected patterns by pattern type
            selectors: Generated selectors for different elements
            metadata: Additional metadata about the analysis
        """
        self.url = url
        self.patterns = patterns or {}
        self.selectors = selectors or {}
        self.metadata = metadata or {}
        
    def has_pattern(self, pattern_type: str) -> bool:
        """
        Check if a specific pattern was detected.
        
        Args:
            pattern_type: Type of pattern to check
            
        Returns:
            Whether the pattern was detected
        """
        return pattern_type in self.patterns
    
    def get_pattern(self, pattern_type: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific pattern.
        
        Args:
            pattern_type: Type of pattern to get
            
        Returns:
            Pattern data or None if not found
        """
        return self.patterns.get(pattern_type)
    
    def get_selectors(self, element_type: str) -> List[str]:
        """
        Get selectors for a specific element type.
        
        Args:
            element_type: Type of element to get selectors for
            
        Returns:
            List of selectors or empty list if none found
        """
        return self.selectors.get(element_type, [])
    
    def get_best_selector(self, element_type: str) -> Optional[str]:
        """
        Get the best selector for a specific element type.
        
        Args:
            element_type: Type of element to get selector for
            
        Returns:
            Best selector or None if no selectors found
        """
        selectors = self.get_selectors(element_type)
        return selectors[0] if selectors else None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary for serialization.
        
        Returns:
            Dictionary representation of result
        """
        return {
            "url": self.url,
            "patterns": self.patterns,
            "selectors": self.selectors,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PatternAnalysisResult':
        """
        Create result from dictionary representation.
        
        Args:
            data: Dictionary representation of result
            
        Returns:
            Pattern analysis result instance
        """
        return cls(
            url=data.get("url", ""),
            patterns=data.get("patterns", {}),
            selectors=data.get("selectors", {}),
            metadata=data.get("metadata", {})
        )


# Global registry instance
pattern_registry = PatternRegistry()

def get_registry() -> PatternRegistry:
    """
    Get the global pattern registry.
    
    Returns:
        Global pattern registry instance
    """
    global pattern_registry
    return pattern_registry