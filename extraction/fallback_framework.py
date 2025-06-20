"""
Extraction Fallback Framework Module

This module provides a comprehensive framework for implementing fallback chains in the
extraction process. It enables graceful degradation from specific to general extraction
methods, helping ensure data extraction succeeds even when primary methods fail.

Key features:
1. Fallback chain management - sequence extraction methods from most specific to most general
2. Quality-based fallback decisions - determine when to trigger fallbacks based on quality metrics
3. Partial result aggregation - collect and consolidate extraction results from multiple methods
4. Progressive degradation - gracefully degrade extraction requirements to ensure some results
"""

import logging
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Any, Optional, Set, Type, Union, Callable, Tuple, TypeVar, Generic
import copy
import json
from dataclasses import dataclass, field

from bs4 import BeautifulSoup

# Try to import core components, but allow fallbacks if they're not available
try:
    from core.service_interface import BaseService
    from core.service_registry import ServiceRegistry
    from extraction.schema_extraction import ExtractionSchema, ExtractionField
    SERVICE_REGISTRY_AVAILABLE = True
except ImportError:
    SERVICE_REGISTRY_AVAILABLE = False
    
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ExtractionFallback")

# Generic type for extractors
T = TypeVar('T')

# Enum for content types
class ContentType(Enum):
    """Enum representing different types of content for extraction."""
    HTML = "html"
    API = "api"
    TEXT = "text"
    JSON = "json"
    XML = "xml"
    TABLE = "table"
    LISTING = "listing"
    ARTICLE = "article"
    PRODUCT = "product"
    GENERIC = "generic"

class ExtractionResult:
    """Class representing the result of an extraction operation."""
    
    def __init__(self, 
                data: Dict[str, Any] = None, 
                success: bool = True, 
                error: Optional[str] = None,
                extractor_name: str = "unknown",
                quality_score: float = 0.0,
                metadata: Dict[str, Any] = None):
        """
        Initialize an extraction result.
        
        Args:
            data: The extracted data
            success: Whether the extraction was successful
            error: Error message if extraction failed
            extractor_name: Name of the extractor used
            quality_score: Quality score of the extraction (0.0 to 1.0)
            metadata: Additional metadata about the extraction
        """
        self.data = data or {}
        self.success = success
        self.error = error
        self.extractor_name = extractor_name
        self.quality_score = quality_score
        self.metadata = metadata or {}
        self.timestamp = time.time()
    
    def __bool__(self) -> bool:
        """Convert to boolean, representing success or failure."""
        return self.success
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "data": self.data,
            "success": self.success,
            "error": self.error,
            "extractor_name": self.extractor_name,
            "quality_score": self.quality_score,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractionResult':
        """Create from dictionary representation."""
        result = cls()
        result.data = data.get("data", {})
        result.success = data.get("success", True)
        result.error = data.get("error")
        result.extractor_name = data.get("extractor_name", "unknown")
        result.quality_score = data.get("quality_score", 0.0)
        result.metadata = data.get("metadata", {})
        result.timestamp = data.get("timestamp", time.time())
        return result
    
    def merge(self, other: 'ExtractionResult') -> 'ExtractionResult':
        """
        Merge with another extraction result, preferring data from this result where available.
        
        Args:
            other: Another ExtractionResult to merge with
            
        Returns:
            New merged ExtractionResult
        """
        if not other.success:
            return self
            
        merged = ExtractionResult()
        merged.success = self.success or other.success
        merged.error = self.error if self.error else other.error
        merged.extractor_name = f"{self.extractor_name}+{other.extractor_name}"
        merged.quality_score = max(self.quality_score, other.quality_score)
        
        # Merge metadata
        merged.metadata = {**other.metadata, **self.metadata}
        merged.metadata["merged"] = True
        
        # Merge data (self takes precedence)
        merged.data = copy.deepcopy(other.data)
        for key, value in self.data.items():
            if isinstance(value, dict) and key in merged.data and isinstance(merged.data[key], dict):
                # Recursively merge nested dictionaries
                merged.data[key] = {**merged.data[key], **value}
            else:
                # Replace or add value
                merged.data[key] = value
                
        return merged

class BaseExtractor(ABC, Generic[T]):
    """Abstract base class for all extractors in the fallback framework."""
    
    def __init__(self, name: str = None):
        """
        Initialize the extractor.
        
        Args:
            name: Optional name for the extractor
        """
        self.name = name or self.__class__.__name__
        self._context = None
    
    @property
    def context(self):
        """Get the current context."""
        return self._context
    
    @context.setter
    def context(self, value):
        """Set the extraction context."""
        self._context = value
    
    @abstractmethod
    def can_handle(self, content_type: Union[str, ContentType], content: Any = None, 
                 schema: Any = None) -> bool:
        """
        Check if this extractor can handle the given content type and schema.
        
        Args:
            content_type: Type of content to extract from
            content: Optional content to evaluate
            schema: Optional schema to evaluate
            
        Returns:
            True if this extractor can handle the content and schema
        """
        pass
    
    @abstractmethod
    def extract(self, content: Any, schema: Any = None, 
               options: Dict[str, Any] = None) -> ExtractionResult:
        """
        Extract data from content according to schema.
        
        Args:
            content: Content to extract data from
            schema: Schema defining what to extract
            options: Additional extraction options
            
        Returns:
            ExtractionResult containing extracted data
        """
        pass
    
    def get_confidence(self, content: Any, schema: Any = None) -> float:
        """
        Get the confidence level for extracting the given content with this extractor.
        
        Args:
            content: Content to evaluate
            schema: Schema to evaluate
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not self.can_handle(self._detect_content_type(content), content, schema):
            return 0.0
        return 0.8  # Default confidence level, override for more accurate estimates
    
    def _detect_content_type(self, content: Any) -> ContentType:
        """
        Detect the content type of the given content.
        
        Args:
            content: Content to analyze
            
        Returns:
            Detected ContentType
        """
        if isinstance(content, str):
            # Try to parse as JSON
            try:
                json.loads(content)
                return ContentType.JSON
            except json.JSONDecodeError:
                pass
                
            # Check if it's HTML
            if "<html" in content.lower() or "<!doctype html" in content.lower():
                return ContentType.HTML
                
            # Check if it's XML
            if "<?xml" in content or "<" in content and "</" in content:
                return ContentType.XML
                
            # Default to TEXT
            return ContentType.TEXT
            
        elif isinstance(content, dict):
            return ContentType.JSON
            
        elif isinstance(content, (list, tuple)):
            return ContentType.JSON
            
        # Default for unknown types
        return ContentType.GENERIC

class FallbackCondition(ABC):
    """Abstract base class for conditions that trigger fallbacks."""
    
    @abstractmethod
    def should_fallback(self, result: ExtractionResult, schema: Any = None) -> bool:
        """
        Determine if a fallback should be triggered based on the extraction result.
        
        Args:
            result: The extraction result to evaluate
            schema: Optional schema for context
            
        Returns:
            True if fallback should be triggered
        """
        pass

class QualityBasedCondition(FallbackCondition):
    """Trigger fallback based on the quality score of the extraction result."""
    
    def __init__(self, quality_threshold: float = 0.5):
        """
        Initialize with a quality threshold.
        
        Args:
            quality_threshold: Minimum quality score (0.0 to 1.0) required to avoid fallback
        """
        self.quality_threshold = quality_threshold
    
    def should_fallback(self, result: ExtractionResult, schema: Any = None) -> bool:
        """
        Determine if fallback should be triggered based on quality score.
        
        Args:
            result: The extraction result to evaluate
            schema: Optional schema for context
            
        Returns:
            True if quality score is below threshold
        """
        if not result.success:
            return True
        return result.quality_score < self.quality_threshold

class SchemaComplianceCondition(FallbackCondition):
    """Trigger fallback based on schema compliance of the extraction result."""
    
    def __init__(self, required_fields: List[str] = None, completeness_threshold: float = 0.7):
        """
        Initialize with required fields and completeness threshold.
        
        Args:
            required_fields: List of field names that must be present
            completeness_threshold: Minimum fraction of schema fields that must be present
        """
        self.required_fields = required_fields or []
        self.completeness_threshold = completeness_threshold
    
    def should_fallback(self, result: ExtractionResult, schema: Any = None) -> bool:
        """
        Determine if fallback should be triggered based on schema compliance.
        
        Args:
            result: The extraction result to evaluate
            schema: Optional schema for context
            
        Returns:
            True if result is not compliant with schema
        """
        if not result.success or not result.data:
            return True
            
        # Check required fields
        for field in self.required_fields:
            if field not in result.data or not result.data[field]:
                return True
                
        # If we have a schema, check overall completeness
        if schema:
            fields = []
            if hasattr(schema, 'fields'):
                # For ExtractionSchema from schema_extraction module
                fields = [f.name for f in schema.fields]
            elif isinstance(schema, dict) and "fields" in schema:
                # For dict-based schema
                fields = [f.get('name') for f in schema["fields"] if 'name' in f]
            elif isinstance(schema, dict):
                # Assume keys are field names
                fields = list(schema.keys())
                
            if fields:
                present_fields = [f for f in fields if f in result.data and result.data[f]]
                completeness = len(present_fields) / len(fields) if fields else 0
                return completeness < self.completeness_threshold
                
        return False

class CompositeCondition(FallbackCondition):
    """Combine multiple fallback conditions with logical operations."""
    
    def __init__(self, conditions: List[FallbackCondition], require_all: bool = False):
        """
        Initialize with a list of conditions.
        
        Args:
            conditions: List of FallbackCondition objects
            require_all: If True, all conditions must be met to trigger fallback (AND),
                         if False, any condition can trigger fallback (OR)
        """
        self.conditions = conditions
        self.require_all = require_all
    
    def should_fallback(self, result: ExtractionResult, schema: Any = None) -> bool:
        """
        Determine if fallback should be triggered based on composite conditions.
        
        Args:
            result: The extraction result to evaluate
            schema: Optional schema for context
            
        Returns:
            True if fallback should be triggered according to the composite logic
        """
        if not self.conditions:
            return False
            
        if self.require_all:
            # All conditions must say yes (AND)
            return all(condition.should_fallback(result, schema) for condition in self.conditions)
        else:
            # Any condition can say yes (OR)
            return any(condition.should_fallback(result, schema) for condition in self.conditions)

class ExtractionFallbackChain:
    """
    Manages a sequence of extractors to be tried in order until successful extraction.
    
    This class implements the Chain of Responsibility pattern for extractors, allowing
    progressive fallback from specific to general extraction methods.
    """
    
    def __init__(self, 
                extractors: List[BaseExtractor],
                fallback_condition: Optional[FallbackCondition] = None,
                aggregate_results: bool = True):
        """
        Initialize the fallback chain.
        
        Args:
            extractors: List of extractors to try in sequence
            fallback_condition: Condition that triggers fallback to next extractor
            aggregate_results: Whether to aggregate results from all extractors
        """
        self.extractors = extractors
        self.fallback_condition = fallback_condition or QualityBasedCondition(0.5)
        self.aggregate_results = aggregate_results
        self.results = []
        self._context = None
    
    @property
    def context(self):
        """Get the current context."""
        return self._context
    
    @context.setter
    def context(self, value):
        """Set the context for all extractors in the chain."""
        self._context = value
        for extractor in self.extractors:
            extractor.context = value
    
    def extract(self, content: Any, schema: Any = None, 
               options: Dict[str, Any] = None) -> ExtractionResult:
        """
        Extract data using the fallback chain.
        
        Args:
            content: Content to extract data from
            schema: Schema defining what to extract
            options: Additional extraction options
            
        Returns:
            ExtractionResult from the first successful extractor that meets quality criteria,
            or an aggregated result if aggregate_results is True
        """
        self.results = []
        options = options or {}
        
        # Track best result
        best_result = None
        best_score = -1.0
        
        # Try each extractor in sequence
        for extractor in self.extractors:
            # Skip extractors that can't handle this content and schema
            if not extractor.can_handle(extractor._detect_content_type(content), content, schema):
                logger.debug(f"Extractor {extractor.name} cannot handle this content, skipping")
                continue
                
            # Try extraction
            try:
                logger.info(f"Trying extractor: {extractor.name}")
                result = extractor.extract(content, schema, options)
                self.results.append(result)
                
                # Update best result if this is better
                if result.quality_score > best_score:
                    best_result = result
                    best_score = result.quality_score
                
                # If this result is good enough, stop here (unless we're aggregating results)
                if not self.fallback_condition.should_fallback(result, schema) and not self.aggregate_results:
                    logger.info(f"Extractor {extractor.name} produced acceptable results, stopping chain")
                    return result
                    
            except Exception as e:
                logger.error(f"Error in extractor {extractor.name}: {str(e)}")
                continue
        
        # If we're aggregating results, combine them
        if self.aggregate_results and len(self.results) > 0:
            return self._aggregate_results(schema)
            
        # Return best result or empty result if none available
        if best_result:
            return best_result
            
        # No successful extractions
        return ExtractionResult(
            success=False, 
            error="All extraction methods failed",
            extractor_name="fallback_chain"
        )
    
    def _aggregate_results(self, schema: Any = None) -> ExtractionResult:
        """
        Aggregate results from all extractors that succeeded.
        
        Args:
            schema: Optional schema for context
            
        Returns:
            Aggregated ExtractionResult
        """
        if not self.results:
            return ExtractionResult(
                success=False, 
                error="No extraction results to aggregate",
                extractor_name="fallback_chain"
            )
            
        # Start with the best result
        successful_results = [r for r in self.results if r.success]
        if not successful_results:
            return ExtractionResult(
                success=False, 
                error="No successful extraction results to aggregate",
                extractor_name="fallback_chain"
            )
            
        # Sort by quality score (highest first)
        sorted_results = sorted(successful_results, key=lambda r: r.quality_score, reverse=True)
        
        # Start with the best result
        aggregated = copy.deepcopy(sorted_results[0])
        
        # Merge in other results
        for result in sorted_results[1:]:
            aggregated = aggregated.merge(result)
            
        # Update metadata
        aggregated.extractor_name = "fallback_chain"
        aggregated.metadata["aggregated"] = True
        aggregated.metadata["extractor_count"] = len(successful_results)
        aggregated.metadata["extractors_used"] = [r.extractor_name for r in successful_results]
        
        return aggregated

class ExtractionQualityAssessor:
    """Evaluates the quality of extraction results using various metrics."""
    
    def __init__(self):
        """Initialize the quality assessor."""
        pass
    
    def assess_quality(self, result: ExtractionResult, schema: Any = None,
                     requirements: Dict[str, Any] = None) -> float:
        """
        Assess the overall quality of an extraction result.
        
        Args:
            result: The extraction result to evaluate
            schema: Optional schema for context
            requirements: Optional extraction requirements
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        if not result.success or not result.data:
            return 0.0
            
        scores = []
        
        # Schema compliance
        if schema:
            schema_score = self.assess_schema_compliance(result.data, schema)
            scores.append(schema_score * 0.4)  # Weight: 40%
            
        # Data completeness
        completeness_score = self.assess_data_completeness(result.data, requirements)
        scores.append(completeness_score * 0.3)  # Weight: 30%
        
        # Data consistency
        consistency_score = self.assess_data_consistency(result.data)
        scores.append(consistency_score * 0.2)  # Weight: 20%
        
        # Anomaly detection
        anomaly_score = 1.0 - self.detect_extraction_anomalies(result.data)
        scores.append(anomaly_score * 0.1)  # Weight: 10%
        
        # Calculate weighted average
        if scores:
            return sum(scores) / sum(weight for score, weight in zip(scores, [0.4, 0.3, 0.2, 0.1]))
        return 0.0
    
    def assess_schema_compliance(self, data: Dict[str, Any], schema: Any) -> float:
        """
        Assess how well the data complies with the given schema.
        
        Args:
            data: Data to evaluate
            schema: Schema to check against
            
        Returns:
            Compliance score (0.0 to 1.0)
        """
        if not data:
            return 0.0
            
        fields = []
        required_fields = []
        
        # Extract field information from schema
        if hasattr(schema, 'fields'):
            # For ExtractionSchema from schema_extraction module
            fields = [f.name for f in schema.fields]
            required_fields = [f.name for f in schema.fields if f.required]
        elif isinstance(schema, dict) and "fields" in schema:
            # For dict-based schema
            fields = [f.get('name') for f in schema["fields"] if 'name' in f]
            required_fields = [f.get('name') for f in schema["fields"] 
                              if 'name' in f and f.get('required', False)]
        elif isinstance(schema, dict):
            # Assume keys are field names
            fields = list(schema.keys())
            required_fields = [k for k, v in schema.items() 
                              if isinstance(v, dict) and v.get('required', False)]
        
        if not fields:
            return 1.0  # No fields to check
            
        # Check required fields
        if required_fields:
            missing_required = [f for f in required_fields if f not in data or not data[f]]
            if missing_required:
                # Missing required fields is a serious issue
                return 0.0
        
        # Check all fields
        present_fields = [f for f in fields if f in data and data[f] is not None]
        
        # Calculate compliance score
        return len(present_fields) / len(fields) if fields else 1.0
    
    def assess_data_completeness(self, data: Dict[str, Any], 
                                requirements: Dict[str, Any] = None) -> float:
        """
        Assess the completeness of extracted data.
        
        Args:
            data: Data to evaluate
            requirements: Optional extraction requirements
            
        Returns:
            Completeness score (0.0 to 1.0)
        """
        if not data:
            return 0.0
            
        # If we have specific requirements, use those
        if requirements and "required_fields" in requirements:
            required_fields = requirements["required_fields"]
            missing_fields = [f for f in required_fields if f not in data or not data[f]]
            return 1.0 - (len(missing_fields) / len(required_fields)) if required_fields else 1.0
            
        # Otherwise assess general completeness
        # Count fields with non-empty values
        non_empty_fields = sum(1 for v in data.values() if v not in (None, "", [], {}))
        total_fields = len(data)
        
        # Calculate completeness ratio
        if total_fields == 0:
            return 0.0
        return non_empty_fields / total_fields
    
    def assess_data_consistency(self, data: Dict[str, Any]) -> float:
        """
        Assess the internal consistency of the extracted data.
        
        Args:
            data: Data to evaluate
            
        Returns:
            Consistency score (0.0 to 1.0)
        """
        if not data:
            return 0.0
            
        inconsistencies = 0
        
        # Check for common inconsistencies
        
        # 1. Title present but empty content
        if 'title' in data and data['title'] and 'content' in data and not data['content']:
            inconsistencies += 1
            
        # 2. URL-related inconsistencies
        if ('url' in data and not data['url'].startswith('http')) or \
           ('image_url' in data and data['image_url'] and not data['image_url'].startswith(('http', '/'))):
            inconsistencies += 1
            
        # 3. Date-related inconsistencies
        date_fields = [f for f in data.keys() if 'date' in f.lower()]
        for date_field in date_fields:
            if data[date_field] and not isinstance(data[date_field], (str, int, float)):
                inconsistencies += 1
                
        # 4. Check numeric fields
        price_fields = [f for f in data.keys() if 'price' in f.lower() or 'amount' in f.lower()]
        for price_field in price_fields:
            if data[price_field] and not isinstance(data[price_field], (int, float, str)):
                inconsistencies += 1
                
        # Calculate consistency score (max 4 checks)
        max_inconsistencies = 4
        return 1.0 - (inconsistencies / max_inconsistencies if inconsistencies <= max_inconsistencies else 1.0)
    
    def detect_extraction_anomalies(self, data: Dict[str, Any]) -> float:
        """
        Detect anomalies in the extracted data that might indicate extraction issues.
        
        Args:
            data: Data to evaluate
            
        Returns:
            Anomaly score (0.0 to 1.0), where 0.0 means no anomalies
        """
        if not data:
            return 1.0
            
        anomalies = 0
        
        # Check for common anomalies
        
        # 1. Excessively long values might indicate extraction errors
        for key, value in data.items():
            if isinstance(value, str) and len(value) > 10000:
                anomalies += 1
                
        # 2. HTML tags in text fields
        text_fields = ['title', 'description', 'content', 'text']
        for field in text_fields:
            if field in data and isinstance(data[field], str) and ('<' in data[field] and '>' in data[field]):
                anomalies += 1
                
        # 3. JavaScript or CSS in content
        code_indicators = ['function(', 'var ', '.js', '.css', '{', '}', ';']
        for field in text_fields:
            if field in data and isinstance(data[field], str):
                if any(indicator in data[field] for indicator in code_indicators):
                    anomalies += 1
                    break
                    
        # 4. Inconsistent data types
        if 'price' in data and not isinstance(data['price'], (int, float, str)):
            anomalies += 1
            
        # Calculate anomaly score (max 4 checks)
        max_anomalies = 4
        return anomalies / max_anomalies if anomalies <= max_anomalies else 1.0
    
    def measure_confidence_scores(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Measure confidence scores for individual fields in the extracted data.
        
        Args:
            data: Data to evaluate
            
        Returns:
            Dictionary mapping field names to confidence scores (0.0 to 1.0)
        """
        if not data:
            return {}
            
        confidence_scores = {}
        
        for key, value in data.items():
            if value is None or value == "" or value == [] or value == {}:
                confidence_scores[key] = 0.0
                continue
                
            # Calculate field-specific confidence
            if isinstance(value, str):
                # String fields
                if len(value) < 2:
                    confidence_scores[key] = 0.3  # Very short values are suspicious
                elif len(value) > 10000:
                    confidence_scores[key] = 0.4  # Very long values might indicate extraction errors
                else:
                    confidence_scores[key] = 0.8  # Normal looking strings
                    
                # Adjust based on field name
                if 'url' in key.lower() and not value.startswith('http'):
                    confidence_scores[key] *= 0.5  # URLs should start with http
                    
            elif isinstance(value, (int, float)):
                # Numeric fields
                confidence_scores[key] = 0.9  # Numbers are usually reliable
                
            elif isinstance(value, list):
                # List fields
                confidence_scores[key] = min(0.9, len(value) / 10)  # More items, more confidence (up to 0.9)
                
            elif isinstance(value, dict):
                # Dict fields - recursive confidence
                sub_confidences = self.measure_confidence_scores(value)
                confidence_scores[key] = sum(sub_confidences.values()) / len(sub_confidences) if sub_confidences else 0.5
                
            else:
                # Unknown type
                confidence_scores[key] = 0.5
                
        return confidence_scores

class ExtractionFallbackRegistry:
    """Registry for managing fallback extractors and creating fallback chains."""
    
    def __init__(self):
        """Initialize the registry."""
        self._fallbacks = {}  # Maps extractor types to fallback extractors
        self._default_chains = {}  # Maps content types to default fallback chains
        self._extractors = {}  # Maps extractor names to extractor classes
        self._quality_assessor = ExtractionQualityAssessor()
    
    def register_fallback(self, extractor_type: str, fallback_extractor_type: str) -> None:
        """
        Register a fallback extractor for a specific extractor type.
        
        Args:
            extractor_type: Type of extractor that might fail
            fallback_extractor_type: Type of extractor to use as fallback
        """
        if extractor_type not in self._fallbacks:
            self._fallbacks[extractor_type] = []
            
        if fallback_extractor_type not in self._fallbacks[extractor_type]:
            self._fallbacks[extractor_type].append(fallback_extractor_type)
            logger.info(f"Registered {fallback_extractor_type} as fallback for {extractor_type}")
    
    def register_extractor(self, name: str, extractor_class: Type[BaseExtractor]) -> None:
        """
        Register an extractor class with the registry.
        
        Args:
            name: Name of the extractor
            extractor_class: Extractor class to register
        """
        self._extractors[name] = extractor_class
        logger.info(f"Registered extractor: {name}")
    
    def get_fallbacks_for_extractor(self, extractor_type: str) -> List[str]:
        """
        Get the fallback extractor types for a specific extractor type.
        
        Args:
            extractor_type: Type of extractor to get fallbacks for
            
        Returns:
            List of fallback extractor types
        """
        return self._fallbacks.get(extractor_type, [])
    
    def create_fallback_chain(self, extractor_type: str, schema: Any = None) -> ExtractionFallbackChain:
        """
        Create a fallback chain for a specific extractor type.
        
        Args:
            extractor_type: Type of primary extractor
            schema: Optional schema to adapt fallbacks
            
        Returns:
            ExtractionFallbackChain with appropriate fallbacks
        """
        extractors = []
        
        # Add primary extractor
        if extractor_type in self._extractors:
            extractors.append(self._extractors[extractor_type]())
            
        # Add fallbacks
        fallback_types = self.get_fallbacks_for_extractor(extractor_type)
        for fallback_type in fallback_types:
            if fallback_type in self._extractors:
                extractors.append(self._extractors[fallback_type]())
                
        # Create appropriate fallback condition based on schema
        if schema:
            # Extract required fields from schema
            required_fields = []
            if hasattr(schema, 'fields'):
                required_fields = [f.name for f in schema.fields if f.required]
            elif isinstance(schema, dict) and "fields" in schema:
                required_fields = [f.get('name') for f in schema["fields"] 
                                  if 'name' in f and f.get('required', False)]
                
            # Create a composite condition with quality and schema requirements
            quality_condition = QualityBasedCondition(0.6)
            schema_condition = SchemaComplianceCondition(required_fields)
            condition = CompositeCondition([quality_condition, schema_condition], require_all=False)
        else:
            # Default to quality-based fallback
            condition = QualityBasedCondition(0.5)
            
        return ExtractionFallbackChain(extractors, condition)
    
    def get_default_chain(self, content_type: Union[str, ContentType]) -> ExtractionFallbackChain:
        """
        Get a default fallback chain for a specific content type.
        
        Args:
            content_type: Type of content to extract from
            
        Returns:
            ExtractionFallbackChain suitable for the content type
        """
        if isinstance(content_type, str):
            content_type = ContentType(content_type) if content_type in ContentType.__members__ else ContentType.GENERIC
            
        # Return cached chain if available
        if content_type in self._default_chains:
            return self._default_chains[content_type]
            
        # Create an appropriate chain based on content type
        if content_type == ContentType.HTML:
            return self._create_html_extraction_chain()
        elif content_type == ContentType.API:
            return self._create_api_extraction_chain()
        elif content_type == ContentType.TEXT:
            return self._create_text_extraction_chain()
        else:
            # For other content types, create a generic chain
            # Collect all extractors that can handle this content type
            compatible_extractors = []
            for name, extractor_class in self._extractors.items():
                extractor = extractor_class()
                if extractor.can_handle(content_type):
                    compatible_extractors.append(extractor)
                    
            # Sort by specificity (more specific extractors first)
            compatible_extractors.sort(key=lambda e: e.get_confidence(None, None), reverse=True)
            
            # Create chain
            chain = ExtractionFallbackChain(compatible_extractors, QualityBasedCondition(0.5))
            self._default_chains[content_type] = chain
            return chain
    
    def suggest_fallback(self, extractor: BaseExtractor, error: Exception, 
                        content: Any) -> Optional[BaseExtractor]:
        """
        Suggest a fallback extractor based on the error and content.
        
        Args:
            extractor: Extractor that failed
            error: Exception that occurred
            content: Content being extracted
            
        Returns:
            Suggested fallback extractor or None if no suitable fallback found
        """
        extractor_type = extractor.__class__.__name__
        content_type = extractor._detect_content_type(content)
        
        # Get fallbacks for this extractor type
        fallback_types = self.get_fallbacks_for_extractor(extractor_type)
        
        # Find a fallback that can handle this content
        for fallback_type in fallback_types:
            if fallback_type in self._extractors:
                fallback = self._extractors[fallback_type]()
                if fallback.can_handle(content_type, content):
                    return fallback
                    
        # If no specific fallback found, return the default fallback for this content type
        for name, extractor_class in self._extractors.items():
            if name != extractor_type:  # Don't suggest the same extractor
                fallback = extractor_class()
                if fallback.can_handle(content_type, content):
                    return fallback
                    
        return None
    
    def _create_html_extraction_chain(self) -> ExtractionFallbackChain:
        """
        Create a fallback chain specifically for HTML content extraction.
        
        Returns:
            ExtractionFallbackChain for HTML content
        """
        extractors = []
        
        # Add extractors in order of specificity
        for name in ['StructureBasedExtractor', 'CssXPathExtractor', 'RegionBasedExtractor', 'FullPageExtractor']:
            if name in self._extractors:
                extractors.append(self._extractors[name]())
                
        # Create chain
        chain = ExtractionFallbackChain(extractors, QualityBasedCondition(0.5))
        self._default_chains[ContentType.HTML] = chain
        return chain
    
    def _create_api_extraction_chain(self) -> ExtractionFallbackChain:
        """
        Create a fallback chain specifically for API response extraction.
        
        Returns:
            ExtractionFallbackChain for API responses
        """
        extractors = []
        
        # Add extractors in order of specificity
        for name in ['DirectPathExtractor', 'JsonPathExtractor', 'PartialPathExtractor', 'FullResponseExtractor']:
            if name in self._extractors:
                extractors.append(self._extractors[name]())
                
        # Create chain
        chain = ExtractionFallbackChain(extractors, QualityBasedCondition(0.6))
        self._default_chains[ContentType.API] = chain
        return chain
    
    def _create_text_extraction_chain(self) -> ExtractionFallbackChain:
        """
        Create a fallback chain specifically for unstructured text extraction.
        
        Returns:
            ExtractionFallbackChain for text content
        """
        extractors = []
        
        # Add extractors in order of specificity
        for name in ['PatternBasedExtractor', 'RuleBasedExtractor', 'NlpBasedExtractor', 'AiBasedExtractor']:
            if name in self._extractors:
                extractors.append(self._extractors[name]())
                
        # Create chain
        chain = ExtractionFallbackChain(extractors, QualityBasedCondition(0.4))
        self._default_chains[ContentType.TEXT] = chain
        return chain

class FieldSubsetExtractor(BaseExtractor):
    """Extractor that focuses on extracting only critical fields from content."""
    
    def __init__(self, core_extractor: BaseExtractor, critical_fields: List[str] = None):
        """
        Initialize field subset extractor.
        
        Args:
            core_extractor: Base extractor to use
            critical_fields: List of critical field names to focus on
        """
        super().__init__(name=f"FieldSubset({core_extractor.name})")
        self.core_extractor = core_extractor
        self.critical_fields = critical_fields or []
    
    def can_handle(self, content_type: Union[str, ContentType], content: Any = None, 
                 schema: Any = None) -> bool:
        """Check if this extractor can handle the given content and schema."""
        return self.core_extractor.can_handle(content_type, content, schema)
    
    def extract(self, content: Any, schema: Any = None, 
               options: Dict[str, Any] = None) -> ExtractionResult:
        """Extract only critical fields from content."""
        options = options or {}
        
        # Create a subset schema if original schema exists
        subset_schema = None
        if schema:
            if hasattr(schema, 'fields'):
                # For ExtractionSchema from schema_extraction module
                subset_fields = [f for f in schema.fields if f.name in self.critical_fields]
                if subset_fields:
                    subset_schema = copy.deepcopy(schema)
                    subset_schema.fields = subset_fields
            elif isinstance(schema, dict) and "fields" in schema:
                # For dict-based schema
                subset_fields = [f for f in schema["fields"] 
                                if "name" in f and f["name"] in self.critical_fields]
                if subset_fields:
                    subset_schema = copy.deepcopy(schema)
                    subset_schema["fields"] = subset_fields
            elif isinstance(schema, dict):
                # Assume keys are field names
                subset_schema = {k: v for k, v in schema.items() if k in self.critical_fields}
                
        # Extract with core extractor
        result = self.core_extractor.extract(content, subset_schema or schema, options)
        
        # Update metadata
        result.extractor_name = self.name
        result.metadata["critical_fields_only"] = True
        result.metadata["critical_fields"] = self.critical_fields
        
        return result

class SchemaRelaxationTransformer(BaseExtractor):
    """Transformer that relaxes schema requirements to allow for more flexible extraction."""
    
    def __init__(self, core_extractor: BaseExtractor, relaxation_level: float = 0.5):
        """
        Initialize schema relaxation transformer.
        
        Args:
            core_extractor: Base extractor to use
            relaxation_level: Degree of schema relaxation (0.0 to 1.0)
        """
        super().__init__(name=f"SchemaRelaxation({core_extractor.name})")
        self.core_extractor = core_extractor
        self.relaxation_level = min(1.0, max(0.0, relaxation_level))
    
    def can_handle(self, content_type: Union[str, ContentType], content: Any = None, 
                 schema: Any = None) -> bool:
        """Check if this extractor can handle the given content and schema."""
        return self.core_extractor.can_handle(content_type, content, schema)
    
    def extract(self, content: Any, schema: Any = None, 
               options: Dict[str, Any] = None) -> ExtractionResult:
        """Extract data with relaxed schema requirements."""
        options = options or {}
        
        # Create a relaxed schema if original schema exists
        relaxed_schema = None
        if schema:
            if hasattr(schema, 'fields'):
                # For ExtractionSchema from schema_extraction module
                relaxed_schema = copy.deepcopy(schema)
                # Make fewer fields required based on relaxation level
                for field in relaxed_schema.fields:
                    if field.required and self.relaxation_level >= 0.5:
                        field.required = False
                    # Also relax confidence thresholds
                    field.confidence_threshold *= (1.0 - self.relaxation_level)
            elif isinstance(schema, dict) and "fields" in schema:
                # For dict-based schema
                relaxed_schema = copy.deepcopy(schema)
                for field in relaxed_schema["fields"]:
                    if "required" in field and field["required"] and self.relaxation_level >= 0.5:
                        field["required"] = False
                    # Also relax other requirements
                    if "confidence_threshold" in field:
                        field["confidence_threshold"] *= (1.0 - self.relaxation_level)
            elif isinstance(schema, dict):
                # Assume keys are field names
                relaxed_schema = copy.deepcopy(schema)
                for key, value in relaxed_schema.items():
                    if isinstance(value, dict) and "required" in value and value["required"] and self.relaxation_level >= 0.5:
                        value["required"] = False
                        
        # Extract with core extractor
        result = self.core_extractor.extract(content, relaxed_schema or schema, options)
        
        # Update metadata
        result.extractor_name = self.name
        result.metadata["schema_relaxed"] = True
        result.metadata["relaxation_level"] = self.relaxation_level
        
        return result

class TypeCoercionTransformer(BaseExtractor):
    """Transformer that helps with type flexibility during extraction."""
    
    def __init__(self, core_extractor: BaseExtractor):
        """
        Initialize type coercion transformer.
        
        Args:
            core_extractor: Base extractor to use
        """
        super().__init__(name=f"TypeCoercion({core_extractor.name})")
        self.core_extractor = core_extractor
    
    def can_handle(self, content_type: Union[str, ContentType], content: Any = None, 
                 schema: Any = None) -> bool:
        """Check if this extractor can handle the given content and schema."""
        return self.core_extractor.can_handle(content_type, content, schema)
    
    def extract(self, content: Any, schema: Any = None, 
               options: Dict[str, Any] = None) -> ExtractionResult:
        """Extract data with type coercion for better schema compliance."""
        options = options or {}
        
        # Extract with core extractor
        result = self.core_extractor.extract(content, schema, options)
        
        # Only process if extraction was successful
        if result.success and result.data and schema:
            # Apply type coercion to match schema
            coerced_data = self._coerce_types(result.data, schema)
            result.data = coerced_data
            result.extractor_name = self.name
            result.metadata["type_coercion_applied"] = True
            
        return result
    
    def _coerce_types(self, data: Dict[str, Any], schema: Any) -> Dict[str, Any]:
        """
        Apply type coercion based on schema expectations.
        
        Args:
            data: Data to coerce
            schema: Schema defining expected types
            
        Returns:
            Data with coerced types
        """
        if not data or not schema:
            return data
            
        coerced_data = copy.deepcopy(data)
        
        # Extract type information from schema
        field_types = {}
        
        if hasattr(schema, 'fields'):
            # For ExtractionSchema from schema_extraction module
            for field in schema.fields:
                field_types[field.name] = field.type
        elif isinstance(schema, dict) and "fields" in schema:
            # For dict-based schema
            for field in schema["fields"]:
                if "name" in field and "type" in field:
                    field_types[field["name"]] = field["type"]
        elif isinstance(schema, dict):
            # Assume keys are field names, look for type info
            for key, value in schema.items():
                if isinstance(value, dict) and "type" in value:
                    field_types[key] = value["type"]
        
        # Apply coercion
        for field, value in coerced_data.items():
            if field in field_types and value is not None:
                expected_type = field_types[field]
                
                # String to number conversions
                if expected_type in ["number", "float", "integer", "int"] and isinstance(value, str):
                    try:
                        # Try to convert string to number
                        if expected_type in ["integer", "int"]:
                            coerced_data[field] = int(value)
                        else:
                            coerced_data[field] = float(value)
                    except ValueError:
                        # If conversion fails, extract numeric part if possible
                        import re
                        numbers = re.findall(r'[-+]?\d*\.\d+|\d+', value)
                        if numbers:
                            if expected_type in ["integer", "int"]:
                                coerced_data[field] = int(float(numbers[0]))
                            else:
                                coerced_data[field] = float(numbers[0])
                
                # Boolean conversions
                elif expected_type == "boolean" and not isinstance(value, bool):
                    if isinstance(value, str):
                        coerced_data[field] = value.lower() in ['true', 'yes', 'y', '1']
                    elif isinstance(value, (int, float)):
                        coerced_data[field] = bool(value)
                
                # List conversions
                elif expected_type == "array" and isinstance(value, str):
                    if ',' in value:
                        coerced_data[field] = [item.strip() for item in value.split(',')]
                    else:
                        coerced_data[field] = [value]
                
                # String conversions
                elif expected_type == "string" and not isinstance(value, str):
                    coerced_data[field] = str(value)
                    
        return coerced_data

class PartialResultAggregator(BaseExtractor):
    """Aggregator that combines partial extraction results from multiple sources."""
    
    def __init__(self, extractors: List[BaseExtractor]):
        """
        Initialize partial result aggregator.
        
        Args:
            extractors: List of extractors to combine results from
        """
        super().__init__(name="PartialResultAggregator")
        self.extractors = extractors
    
    def can_handle(self, content_type: Union[str, ContentType], content: Any = None, 
                 schema: Any = None) -> bool:
        """Check if this aggregator can handle the given content and schema."""
        # Can handle if at least one extractor can
        return any(extractor.can_handle(content_type, content, schema) for extractor in self.extractors)
    
    def extract(self, content: Any, schema: Any = None, 
               options: Dict[str, Any] = None) -> ExtractionResult:
        """Extract and aggregate partial results from multiple extractors."""
        options = options or {}
        results = []
        
        # Extract with each extractor
        for extractor in self.extractors:
            if extractor.can_handle(extractor._detect_content_type(content), content, schema):
                try:
                    result = extractor.extract(content, schema, options)
                    if result.success:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error in extractor {extractor.name}: {str(e)}")
        
        # Aggregate results
        if not results:
            return ExtractionResult(
                success=False, 
                error="No successful extractions to aggregate",
                extractor_name=self.name
            )
            
        # Start with the best result
        sorted_results = sorted(results, key=lambda r: r.quality_score, reverse=True)
        aggregated = copy.deepcopy(sorted_results[0])
        
        # Merge in other results
        for result in sorted_results[1:]:
            aggregated = aggregated.merge(result)
            
        # Update metadata
        aggregated.extractor_name = self.name
        aggregated.metadata["aggregated"] = True
        aggregated.metadata["extractor_count"] = len(results)
        aggregated.metadata["extractors_used"] = [r.extractor_name for r in results]
        
        return aggregated

def create_html_extraction_chain() -> ExtractionFallbackChain:
    """
    Create a fallback chain for HTML content extraction.
    
    Returns:
        ExtractionFallbackChain configured for HTML extraction
    """
    # This is a placeholder implementation - in a real application, this would
    # instantiate concrete extractors from your codebase
    extractors = []
    
    # For the test version, let's define some dummy extractors
    class DummyStructureExtractor(BaseExtractor):
        def __init__(self):
            super().__init__(name="StructureExtractor")
        def can_handle(self, content_type, content=None, schema=None):
            return isinstance(content_type, ContentType) and content_type == ContentType.HTML or \
                   content_type == "html"
        def extract(self, content, schema=None, options=None):
            # Simulate extraction
            return ExtractionResult(
                data={"title": "Example Title", "content": "Example Content"},
                success=True,
                extractor_name=self.name,
                quality_score=0.8
            )
    
    class DummyCssXPathExtractor(BaseExtractor):
        def __init__(self):
            super().__init__(name="CssXPathExtractor")
        def can_handle(self, content_type, content=None, schema=None):
            return isinstance(content_type, ContentType) and content_type == ContentType.HTML or \
                   content_type == "html"
        def extract(self, content, schema=None, options=None):
            # Simulate extraction
            return ExtractionResult(
                data={"title": "Example Title", "content": "Example Content"},
                success=True,
                extractor_name=self.name,
                quality_score=0.7
            )
    
    # Add dummy extractors to the chain
    extractors.append(DummyStructureExtractor())
    extractors.append(DummyCssXPathExtractor())
    
    # Create and return the chain
    return ExtractionFallbackChain(
        extractors=extractors,
        fallback_condition=QualityBasedCondition(0.6),
        aggregate_results=True
    )

def create_api_extraction_chain() -> ExtractionFallbackChain:
    """
    Create a fallback chain for API response extraction.
    
    Returns:
        ExtractionFallbackChain configured for API extraction
    """
    # Similar placeholder implementation
    extractors = []
    
    # For the test version, let's define some dummy extractors
    class DummyDirectPathExtractor(BaseExtractor):
        def __init__(self):
            super().__init__(name="DirectPathExtractor")
        def can_handle(self, content_type, content=None, schema=None):
            return isinstance(content_type, ContentType) and content_type == ContentType.API or \
                   content_type == "api"
        def extract(self, content, schema=None, options=None):
            # Simulate extraction
            return ExtractionResult(
                data={"id": 123, "name": "Example API Result"},
                success=True,
                extractor_name=self.name,
                quality_score=0.9
            )
    
    class DummyJsonPathExtractor(BaseExtractor):
        def __init__(self):
            super().__init__(name="JsonPathExtractor")
        def can_handle(self, content_type, content=None, schema=None):
            return isinstance(content_type, ContentType) and content_type == ContentType.API or \
                   content_type == "api" or content_type == ContentType.JSON or content_type == "json"
        def extract(self, content, schema=None, options=None):
            # Simulate extraction
            return ExtractionResult(
                data={"id": 123, "name": "Example API Result"},
                success=True,
                extractor_name=self.name,
                quality_score=0.8
            )
    
    # Add dummy extractors to the chain
    extractors.append(DummyDirectPathExtractor())
    extractors.append(DummyJsonPathExtractor())
    
    # Create and return the chain
    return ExtractionFallbackChain(
        extractors=extractors,
        fallback_condition=QualityBasedCondition(0.7),
        aggregate_results=True
    )

def create_text_extraction_chain() -> ExtractionFallbackChain:
    """
    Create a fallback chain for unstructured text extraction.
    
    Returns:
        ExtractionFallbackChain configured for text extraction
    """
    # Similar placeholder implementation
    extractors = []
    
    # For the test version, let's define some dummy extractors
    class DummyPatternExtractor(BaseExtractor):
        def __init__(self):
            super().__init__(name="PatternExtractor")
        def can_handle(self, content_type, content=None, schema=None):
            return isinstance(content_type, ContentType) and content_type == ContentType.TEXT or \
                   content_type == "text"
        def extract(self, content, schema=None, options=None):
            # Simulate extraction
            return ExtractionResult(
                data={"entities": ["Example Entity"], "keywords": ["example", "keyword"]},
                success=True,
                extractor_name=self.name,
                quality_score=0.7
            )
    
    class DummyNlpExtractor(BaseExtractor):
        def __init__(self):
            super().__init__(name="NlpExtractor")
        def can_handle(self, content_type, content=None, schema=None):
            return isinstance(content_type, ContentType) and content_type == ContentType.TEXT or \
                   content_type == "text"
        def extract(self, content, schema=None, options=None):
            # Simulate extraction
            return ExtractionResult(
                data={"entities": ["Example Entity"], "sentiment": "positive"},
                success=True,
                extractor_name=self.name,
                quality_score=0.8
            )
    
    # Add dummy extractors to the chain
    extractors.append(DummyPatternExtractor())
    extractors.append(DummyNlpExtractor())
    
    # Create and return the chain
    return ExtractionFallbackChain(
        extractors=extractors,
        fallback_condition=QualityBasedCondition(0.5),
        aggregate_results=True
    )

# For integration with the service registry pattern
if SERVICE_REGISTRY_AVAILABLE:
    class FallbackFrameworkService(BaseService):
        """Service implementation of the extraction fallback framework."""
        
        def __init__(self):
            self._initialized = False
            self._registry = ExtractionFallbackRegistry()
            self._quality_assessor = ExtractionQualityAssessor()
        
        def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
            """Initialize the service with configuration."""
            if self._initialized:
                return
                
            # Register extractors and fallbacks based on config
            if config and "extractors" in config:
                for extractor_config in config["extractors"]:
                    if "name" in extractor_config and "class" in extractor_config:
                        name = extractor_config["name"]
                        class_path = extractor_config["class"]
                        try:
                            module_path, class_name = class_path.rsplit(".", 1)
                            module = __import__(module_path, fromlist=[class_name])
                            extractor_class = getattr(module, class_name)
                            self._registry.register_extractor(name, extractor_class)
                        except Exception as e:
                            logger.error(f"Error registering extractor {name}: {str(e)}")
            
            if config and "fallbacks" in config:
                for fallback_config in config["fallbacks"]:
                    if "extractor" in fallback_config and "fallback" in fallback_config:
                        self._registry.register_fallback(
                            fallback_config["extractor"],
                            fallback_config["fallback"]
                        )
                        
            self._initialized = True
            logger.info("Fallback framework service initialized")
        
        def shutdown(self) -> None:
            """Clean up resources."""
            self._initialized = False
            logger.info("Fallback framework service shut down")
        
        @property
        def name(self) -> str:
            """Return the name of the service."""
            return "fallback_framework"
        
        def get_registry(self) -> ExtractionFallbackRegistry:
            """Get the fallback registry."""
            return self._registry
        
        def get_quality_assessor(self) -> ExtractionQualityAssessor:
            """Get the quality assessor."""
            return self._quality_assessor
        
        def create_fallback_chain(self, extractor_type: str, schema: Any = None) -> ExtractionFallbackChain:
            """Create a fallback chain for a specific extractor type."""
            return self._registry.create_fallback_chain(extractor_type, schema)
        
        def get_default_chain(self, content_type: Union[str, ContentType]) -> ExtractionFallbackChain:
            """Get the default fallback chain for a content type."""
            return self._registry.get_default_chain(content_type)