"""
Extraction Interface Module

This module defines the core interfaces and abstract base classes for the 
SmartScrape Universal Extraction Framework. These interfaces standardize how
extraction components interact and provide base functionality for specialized extractors.
"""

from abc import ABC, abstractmethod
import re
import html
from typing import Dict, Any, Optional, List, Union, Set, Tuple
import logging

# Import StrategyContext - assuming it's defined in strategies.core.strategy_context
from strategies.core.strategy_context import StrategyContext

# Configure logging
logger = logging.getLogger(__name__)

class BaseExtractor(ABC):
    """
    Abstract base class for all extraction components.
    
    This class defines the standard interface that all extractors must implement
    and provides common utility methods for content handling and metadata extraction.
    """
    
    def __init__(self, context: Optional[StrategyContext] = None):
        """
        Initialize the extractor with an optional strategy context.
        
        Args:
            context: StrategyContext for accessing shared services and configurations
        """
        self._context = context
        self._initialized = False
    
    @abstractmethod
    def extract(self, content: Any, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract data from the provided content.
        
        Args:
            content: Content to extract data from (typically HTML, JSON, or text)
            options: Optional extraction parameters to customize behavior
            
        Returns:
            Dictionary containing extracted data and metadata
        """
        pass
    
    @abstractmethod
    def can_handle(self, content: Any, content_type: Optional[str] = None) -> bool:
        """
        Check if this extractor can handle the given content and content type.
        
        Args:
            content: Content to check compatibility with
            content_type: Optional hint about the content type (e.g., "html", "json", "text")
            
        Returns:
            True if the extractor can handle this content, False otherwise
        """
        pass
    
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the extractor, setting up any required resources or connections.
        
        This method should be called before the extractor is used for the first time.
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """
        Clean up any resources used by the extractor.
        
        This method should be called when the extractor is no longer needed.
        """
        pass
    
    def get_context(self) -> StrategyContext:
        """
        Get the strategy context associated with this extractor.
        
        Returns:
            The strategy context object
            
        Raises:
            ValueError: If no context has been set
        """
        if self._context is None:
            raise ValueError("No context has been set for this extractor")
        return self._context
    
    def set_context(self, context: StrategyContext) -> None:
        """
        Set the strategy context for this extractor.
        
        Args:
            context: StrategyContext to use with this extractor
        """
        self._context = context
    
    def sanitize_content(self, content: str) -> str:
        """
        Sanitize content to remove potentially problematic elements.
        
        Args:
            content: Raw content string to sanitize
            
        Returns:
            Sanitized content string
        """
        if not content:
            return ""
            
        # Unescape HTML entities
        content = html.unescape(content)
        
        # Remove script and style elements
        content = re.sub(r'<script[^>]*>.*?</script>', ' ', content, flags=re.DOTALL)
        content = re.sub(r'<style[^>]*>.*?</style>', ' ', content, flags=re.DOTALL)
        
        # Remove comments
        content = re.sub(r'<!--.*?-->', ' ', content, flags=re.DOTALL)
        
        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content).strip()
        
        return content
    
    def extract_metadata(self, content: str) -> Dict[str, Any]:
        """
        Extract basic metadata from content.
        
        Args:
            content: Content to extract metadata from
            
        Returns:
            Dictionary of metadata fields
        """
        metadata = {
            "content_length": len(content) if content else 0,
            "extractor_type": self.__class__.__name__,
        }
        
        # Try to extract title if it's HTML
        if content and content.strip().startswith("<"):
            title_match = re.search(r'<title[^>]*>(.*?)</title>', content, re.IGNORECASE | re.DOTALL)
            if title_match:
                metadata["title"] = title_match.group(1).strip()
        
        return metadata
    
    def get_confidence_score(self, extraction_result: Dict[str, Any]) -> float:
        """
        Calculate a confidence score for an extraction result.
        
        Args:
            extraction_result: The extraction result to evaluate
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base implementation uses completeness as a proxy for confidence
        if not extraction_result or not isinstance(extraction_result, dict):
            return 0.0
            
        # Check if result already has a confidence score
        if "_metadata" in extraction_result and "confidence" in extraction_result["_metadata"]:
            return float(extraction_result["_metadata"]["confidence"])
        
        # Default heuristic: more data fields generally means higher confidence
        # This should be overridden by specialized extractors
        data_fields = [k for k in extraction_result.keys() if not k.startswith("_")]
        
        if not data_fields:
            return 0.0
            
        # Simple scoring based on number of non-empty fields
        non_empty_fields = [k for k in data_fields if extraction_result.get(k)]
        return min(1.0, len(non_empty_fields) / max(1, len(data_fields)))


class PatternExtractor(BaseExtractor):
    """
    Abstract base class for pattern-based extraction.
    
    This class provides the foundation for extractors that use patterns 
    (CSS selectors, XPath, regex, etc.) to extract structured data.
    """
    
    @abstractmethod
    def detect_patterns(self, content: Any) -> Dict[str, Any]:
        """
        Detect patterns in the provided content.
        
        Args:
            content: Content to analyze for patterns
            
        Returns:
            Dictionary of detected patterns
        """
        pass
    
    @abstractmethod
    def generate_selectors(self, content: Any, target_elements: List[Any]) -> Dict[str, str]:
        """
        Generate selectors for target elements in the content.
        
        Args:
            content: Content containing the target elements
            target_elements: List of elements to generate selectors for
            
        Returns:
            Dictionary mapping field names to selectors
        """
        pass
    
    @abstractmethod
    def validate_pattern(self, content: Any, pattern: Any) -> bool:
        """
        Validate if a pattern is effective for the given content.
        
        Args:
            content: Content to validate the pattern against
            pattern: Pattern to validate
            
        Returns:
            True if pattern is valid and effective, False otherwise
        """
        pass
    
    @abstractmethod
    def extract_with_pattern(self, content: Any, pattern: Any) -> Dict[str, Any]:
        """
        Extract data using a specific pattern.
        
        Args:
            content: Content to extract data from
            pattern: Pattern to use for extraction
            
        Returns:
            Dictionary of extracted data
        """
        pass


class SemanticExtractor(BaseExtractor):
    """
    Abstract base class for AI/semantic understanding extraction.
    
    This class provides the foundation for extractors that use AI models 
    to understand and extract data based on semantic meaning.
    """
    
    @abstractmethod
    def preprocess_content(self, content: Any) -> Any:
        """
        Preprocess content for AI analysis.
        
        Args:
            content: Raw content to preprocess
            
        Returns:
            Preprocessed content ready for AI analysis
        """
        pass
    
    @abstractmethod
    def generate_prompt(self, content: Any, extraction_goal: str) -> str:
        """
        Generate an AI prompt for extraction.
        
        Args:
            content: Content to extract from
            extraction_goal: Description of what to extract
            
        Returns:
            Formatted prompt string
        """
        pass
    
    @abstractmethod
    def parse_ai_response(self, response: str) -> Dict[str, Any]:
        """
        Parse AI model response into structured data.
        
        Args:
            response: Raw response from AI model
            
        Returns:
            Structured data extracted from response
        """
        pass
    
    @abstractmethod
    def calculate_semantic_confidence(self, result: Dict[str, Any], content: Any) -> float:
        """
        Calculate confidence score for semantic extraction.
        
        Args:
            result: Extraction result to evaluate
            content: Original content for reference
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        pass


class StructuralAnalyzer(BaseExtractor):
    """
    Abstract base class for document structure analysis.
    
    This class provides the foundation for analyzing the structure of documents
    to understand their layout, sections, and content organization.
    """
    
    @abstractmethod
    def analyze_structure(self, content: Any) -> Dict[str, Any]:
        """
        Analyze the structure of the provided content.
        
        Args:
            content: Content to analyze
            
        Returns:
            Structural analysis results
        """
        pass
    
    @abstractmethod
    def identify_content_sections(self, content: Any) -> List[Dict[str, Any]]:
        """
        Identify distinct content sections in the document.
        
        Args:
            content: Content to analyze
            
        Returns:
            List of identified content sections with metadata
        """
        pass
    
    @abstractmethod
    def detect_element_relationships(self, content: Any) -> Dict[str, List[Tuple]]:
        """
        Detect relationships between elements in the content.
        
        Args:
            content: Content to analyze
            
        Returns:
            Dictionary mapping relationship types to element pairs
        """
        pass
    
    @abstractmethod
    def cluster_similar_content(self, elements: List[Any]) -> List[List[Any]]:
        """
        Cluster similar content elements together.
        
        Args:
            elements: List of content elements to cluster
            
        Returns:
            List of element clusters
        """
        pass


class MetadataExtractor(BaseExtractor):
    """
    Abstract base class for metadata extraction.
    
    This class provides the foundation for extractors that focus on 
    extracting metadata from various sources and formats.
    """
    
    @abstractmethod
    def extract_jsonld(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract JSON-LD structured data from content.
        
        Args:
            content: Content to extract from
            
        Returns:
            List of extracted JSON-LD objects
        """
        pass
    
    @abstractmethod
    def extract_opengraph(self, content: str) -> Dict[str, Any]:
        """
        Extract OpenGraph metadata from content.
        
        Args:
            content: Content to extract from
            
        Returns:
            Dictionary of OpenGraph metadata
        """
        pass
    
    @abstractmethod
    def extract_microdata(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract Microdata from content.
        
        Args:
            content: Content to extract from
            
        Returns:
            List of extracted Microdata items
        """
        pass
    
    @abstractmethod
    def standardize_metadata(self, metadata: Dict[str, Any], source: str) -> Dict[str, Any]:
        """
        Standardize metadata from different sources into a common format.
        
        Args:
            metadata: Raw metadata to standardize
            source: Source of the metadata (e.g., "jsonld", "opengraph")
            
        Returns:
            Standardized metadata dictionary
        """
        pass


class ContentNormalizer(BaseExtractor):
    """
    Abstract base class for standardizing extracted data.
    
    This class provides the foundation for normalizers that standardize
    extracted data into consistent formats and units.
    """
    
    @abstractmethod
    def normalize_text(self, text: str) -> str:
        """
        Normalize text content.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        pass
    
    @abstractmethod
    def normalize_datetime(self, datetime_str: str) -> str:
        """
        Normalize date and time strings to a standard format.
        
        Args:
            datetime_str: Date/time string to normalize
            
        Returns:
            Normalized date/time string in ISO format
        """
        pass
    
    @abstractmethod
    def normalize_units(self, value: str, unit_type: str) -> Dict[str, Any]:
        """
        Normalize units to standard units.
        
        Args:
            value: Value with unit to normalize
            unit_type: Type of unit (e.g., "length", "weight", "currency")
            
        Returns:
            Dictionary with normalized value and unit
        """
        pass
    
    @abstractmethod
    def normalize_entity(self, entity: str, entity_type: str) -> Dict[str, Any]:
        """
        Normalize entity references.
        
        Args:
            entity: Entity to normalize
            entity_type: Type of entity (e.g., "person", "organization", "location")
            
        Returns:
            Dictionary with normalized entity data
        """
        pass


class QualityEvaluator(BaseExtractor):
    """
    Abstract base class for assessing extraction quality.
    
    This class provides the foundation for evaluators that assess
    the quality, completeness, and relevance of extracted data.
    """
    
    @abstractmethod
    def evaluate_schema_compliance(self, data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate how well extracted data complies with a schema.
        
        Args:
            data: Extracted data to evaluate
            schema: Schema to evaluate against
            
        Returns:
            Evaluation results with compliance metrics
        """
        pass
    
    @abstractmethod
    def evaluate_completeness(self, data: Dict[str, Any], expected_fields: Set[str]) -> float:
        """
        Evaluate the completeness of extracted data.
        
        Args:
            data: Extracted data to evaluate
            expected_fields: Set of expected field names
            
        Returns:
            Completeness score between 0.0 and 1.0
        """
        pass
    
    @abstractmethod
    def evaluate_relevance(self, data: Dict[str, Any], query: str) -> float:
        """
        Evaluate the relevance of extracted data to a query.
        
        Args:
            data: Extracted data to evaluate
            query: Query or extraction goal to evaluate relevance against
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        pass
    
    @abstractmethod
    def calculate_overall_quality(self, evaluations: Dict[str, Any]) -> float:
        """
        Calculate an overall quality score from multiple evaluations.
        
        Args:
            evaluations: Dictionary of evaluation results
            
        Returns:
            Overall quality score between 0.0 and 1.0
        """
        pass


class SchemaValidator(BaseExtractor):
    """
    Abstract base class for validating against schemas.
    
    This class provides the foundation for validators that ensure
    extracted data conforms to defined schemas.
    """
    
    @abstractmethod
    def validate(self, data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate data against a schema.
        
        Args:
            data: Data to validate
            schema: Schema to validate against
            
        Returns:
            Validation results with errors and warnings
        """
        pass
    
    @abstractmethod
    def coerce_types(self, data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coerce data values to the types specified in the schema.
        
        Args:
            data: Data to coerce
            schema: Schema with type information
            
        Returns:
            Data with coerced types
        """
        pass
    
    @abstractmethod
    def validate_required_fields(self, data: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
        """
        Validate that all required fields are present.
        
        Args:
            data: Data to validate
            schema: Schema with required field information
            
        Returns:
            List of missing required fields
        """
        pass
    
    @abstractmethod
    def validate_relationships(self, data: Dict[str, Any], schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Validate relationships between data fields.
        
        Args:
            data: Data to validate
            schema: Schema with relationship information
            
        Returns:
            List of relationship validation issues
        """
        pass