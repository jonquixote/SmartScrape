"""
Extraction Result Module

This module defines standardized structures for extraction outputs,
providing consistent access to extracted data, metadata, and quality metrics.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

@dataclass
class ExtractionResult:
    """
    Class for standardizing extraction output.
    
    This class provides a structured container for extraction results,
    including the extracted data, metadata about the extraction process,
    quality metrics, source information, and error details.
    """
    
    # Primary extracted data
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Extraction success flag
    success: bool = False
    
    # Name of the extractor that produced this result
    extractor_name: str = ""
    
    # Extraction process metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Quality metrics
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Source information
    source_info: Dict[str, Any] = field(default_factory=dict)
    
    # Error details if extraction failed
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize default values after initialization."""
        # Set default timestamp if not provided
        if "timestamp" not in self.metadata:
            self.metadata["timestamp"] = datetime.now().isoformat()
        
        # Set default confidence if not provided and extraction succeeded
        if "confidence" not in self.quality_metrics and self.success:
            self.quality_metrics["confidence"] = 1.0
    
    @property
    def is_empty(self) -> bool:
        """
        Check if the extraction result contains no data.
        
        Returns:
            True if the result has no data, False otherwise
        """
        return not bool(self.data)
    
    @property
    def confidence(self) -> float:
        """
        Get the confidence score for this extraction result.
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        return self.quality_metrics.get("confidence", 0.0)
    
    def add_error(self, error_type: str, error_message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Add an error to the extraction result.
        
        Args:
            error_type: Type of error (e.g., "validation", "extraction")
            error_message: Human-readable error message
            details: Optional additional error details
        """
        error = {
            "type": error_type,
            "message": error_message,
            "timestamp": datetime.now().isoformat()
        }
        
        if details:
            error["details"] = details
            
        self.errors.append(error)
        self.success = False
    
    def merge(self, other: 'ExtractionResult') -> 'ExtractionResult':
        """
        Merge this extraction result with another one.
        
        Args:
            other: Another ExtractionResult to merge with
            
        Returns:
            New ExtractionResult with merged data
        """
        # Create a new result with merged data
        merged = ExtractionResult(
            data={**self.data, **other.data},
            success=self.success or other.success,
            extractor_name=f"{self.extractor_name}+{other.extractor_name}",
            metadata={**self.metadata, **other.metadata},
            quality_metrics={**self.quality_metrics, **other.quality_metrics},
            source_info={**self.source_info, **other.source_info},
            errors=[*self.errors, *other.errors]
        )
        
        # Update confidence as weighted average
        if "confidence" in merged.quality_metrics:
            self_conf = self.quality_metrics.get("confidence", 0.0)
            other_conf = other.quality_metrics.get("confidence", 0.0)
            merged.quality_metrics["confidence"] = (self_conf + other_conf) / 2
            
        return merged
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the extraction result to a dictionary.
        
        Returns:
            Dictionary representation of the extraction result
        """
        return {
            "data": self.data,
            "success": self.success,
            "extractor_name": self.extractor_name,
            "metadata": self.metadata,
            "quality_metrics": self.quality_metrics,
            "source_info": self.source_info,
            "errors": self.errors
        }
    
    def to_json(self, indent: int = 2) -> str:
        """
        Convert the extraction result to a JSON string.
        
        Args:
            indent: Number of spaces for indentation
            
        Returns:
            JSON string representation of the extraction result
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractionResult':
        """
        Create an ExtractionResult from a dictionary.
        
        Args:
            data: Dictionary representation of an extraction result
            
        Returns:
            ExtractionResult instance
        """
        return cls(
            data=data.get("data", {}),
            success=data.get("success", False),
            extractor_name=data.get("extractor_name", ""),
            metadata=data.get("metadata", {}),
            quality_metrics=data.get("quality_metrics", {}),
            source_info=data.get("source_info", {}),
            errors=data.get("errors", [])
        )