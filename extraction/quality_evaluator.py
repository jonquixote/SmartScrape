"""
Quality Evaluator Module - Clean Implementation

This module provides classes for evaluating the quality of extracted data,
including completeness, confidence, relevance, and anomaly detection.
"""

import logging
import json
import copy
import math
import re
from typing import Dict, Any, List, Optional, Union, Set, Tuple

from extraction.core.extraction_interface import QualityEvaluator, BaseExtractor
from extraction.core.extraction_result import ExtractionResult
from extraction.helpers.quality_metrics import (
    calculate_text_quality, calculate_field_confidence, measure_numerical_plausibility,
    check_date_validity, check_url_validity, measure_enum_validity, check_field_relationships,
    calculate_overall_quality_score, generate_quality_profile, identify_improvement_opportunities,
    calculate_schema_compliance_rate, detect_outliers, measure_data_coherence
)

logger = logging.getLogger(__name__)

class ExtractedDataQualityEvaluator(QualityEvaluator):
    """
    Quality evaluator for assessing extracted data.
    
    This class provides methods to evaluate various quality aspects of
    extracted data, including completeness, relevance, correctness,
    consistency, and overall quality.
    """
    
    def __init__(self, context=None):
        """
        Initialize the quality evaluator.
        
        Args:
            context: Strategy context for accessing shared services
        """
        super().__init__(context)
        self._config = {}
        self._field_type_patterns = {}
        self._confidence_thresholds = {}
        self._criteria_weights = {}
        
    def initialize(self) -> None:
        """Initialize the evaluator, loading configurations."""
        if self._initialized:
            return
            
        logger.info("Initializing Quality Evaluator")
        
        # Set default confidence thresholds
        self._confidence_thresholds = {
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4
        }
        
        # Set default criteria weights for overall quality score
        self._criteria_weights = {
            "completeness": 0.3,
            "confidence": 0.2,
            "relevance": 0.15,
            "consistency": 0.15,
            "schema_compliance": 0.2
        }
        
        # Common field type patterns for validation
        self._field_type_patterns = {
            "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            "url": r"^(https?://)?([a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}(/.*)?$",
            "price": r"^(\$|€|£|¥)?\s*\d+(\.\d{1,2})?(\s*[a-zA-Z]{3})?$",
            "date": r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}$",
            "phone": r"^\+?(\d[\s-]?){8,14}\d$"
        }
        
        # If a context is available, try to load configuration
        if self._context:
            try:
                config_service = self._context.get_service("config_service")
                quality_config = config_service.get_config("quality_evaluation", {})
                
                # Update with custom configuration
                if "confidence_thresholds" in quality_config:
                    self._confidence_thresholds.update(quality_config["confidence_thresholds"])
                
                if "criteria_weights" in quality_config:
                    self._criteria_weights.update(quality_config["criteria_weights"])
                
                if "field_type_patterns" in quality_config:
                    self._field_type_patterns.update(quality_config["field_type_patterns"])
                
                logger.info("Loaded quality evaluation configuration")
            except Exception as e:
                logger.warning(f"Failed to load quality evaluation configuration: {str(e)}")
        
        self._initialized = True
    
    def shutdown(self) -> None:
        """Clean up resources used by the evaluator."""
        if not self._initialized:
            return
            
        logger.info("Shutting down Quality Evaluator")
        self._initialized = False
    
    def can_handle(self, data: Any, data_type: Optional[str] = None) -> bool:
        """
        Check if this evaluator can handle the given data.
        
        Args:
            data: Data to check (usually a dict or ExtractionResult)
            data_type: Optional hint about the data type
            
        Returns:
            True if the evaluator can handle this data, False otherwise
        """
        if isinstance(data, dict):
            return True
        if isinstance(data, ExtractionResult):
            return True
        return False
    
    def extract(self, content: Any, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Implements the extract method required by BaseExtractor.
        
        For QualityEvaluator, this is a proxy to evaluate().
        
        Args:
            content: Data to evaluate (dict or ExtractionResult)
            options: Optional evaluation options
            
        Returns:
            Evaluation results
        """
        if not self._initialized:
            self.initialize()
            
        schema = None
        if options and "schema" in options:
            schema = options["schema"]
            
        return self.evaluate(content, schema, options)
    
    def evaluate(self, extraction_result: Union[Dict[str, Any], ExtractionResult], 
                schema: Optional[Dict[str, Any]] = None,
                options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main evaluation method for extracted data quality.
        
        Args:
            extraction_result: Data to evaluate
            schema: Optional schema for validation
            options: Optional evaluation options
            
        Returns:
            Dictionary with evaluation results and quality metrics
        """
        if not self._initialized:
            self.initialize()
            
        # Convert ExtractionResult to dict if needed
        if isinstance(extraction_result, ExtractionResult):
            data = extraction_result.data
        else:
            data = extraction_result
            
        if not isinstance(data, dict):
            return {
                "quality_score": 0.0,
                "metrics": {
                    "completeness": 0.0,
                    "confidence": 0.0,
                    "relevance": 0.0,
                    "consistency": 0.0,
                    "schema_compliance": 0.0
                },
                "anomalies": ["Invalid data type"],
                "missing_required_fields": [],
                "errors": ["Data is not a dictionary"]
            }
            
        # Special case for truly empty data
        if not data or all(v is None or str(v).strip() == "" for v in data.values()):
            return {
                "quality_score": 0.0,
                "metrics": {
                    "completeness": 0.0,
                    "confidence": 0.0,
                    "relevance": 0.0,
                    "consistency": 0.0,
                    "schema_compliance": 0.0
                },
                "anomalies": [],
                "missing_required_fields": schema.get("required", []) if schema else [],
                "errors": []
            }
        
        # Calculate individual metrics
        completeness = self.calculate_completeness_score(data, schema)
        confidence = self.calculate_confidence_score(data)
        relevance = self.calculate_relevance_score(data, options)
        consistency = self.evaluate_consistency(data)
        schema_compliance = self.calculate_schema_compliance_rate(data, schema)
        
        # Calculate overall quality score
        overall_score = self.calculate_overall_quality({
            "completeness": completeness,
            "confidence": confidence,
            "relevance": relevance,
            "consistency": consistency,
            "schema_compliance": schema_compliance
        })
        
        # Detect anomalies
        anomalies = self.detect_anomalies(data, schema)
        
        # Check for missing required fields
        missing_fields = self.detect_missing_required_fields(data, schema)
        
        return {
            "quality_score": overall_score,
            "overall_score": overall_score,  # Alias for compatibility
            "metrics": {
                "completeness": completeness,
                "confidence": confidence,
                "relevance": relevance,
                "consistency": consistency,
                "schema_compliance": schema_compliance
            },
            "anomalies": anomalies,
            "missing_required_fields": missing_fields,
            "errors": [],
            "_metadata": {
                "evaluator": "ExtractedDataQualityEvaluator",
                "schema_provided": schema is not None
            }
        }
    
    # Abstract method implementations
    def evaluate_schema_compliance(self, data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate how well extracted data complies with a schema."""
        try:
            compliance_result = calculate_schema_compliance_rate(data, schema)
            if isinstance(compliance_result, dict):
                compliance_rate = compliance_result.get("compliance_rate", 0.0)
            else:
                compliance_rate = float(compliance_result)
            
            violations = self.detect_schema_violations(data, schema)
            
            return {
                "compliance_rate": compliance_rate,
                "violations": violations,
                "score": compliance_rate
            }
        except Exception as e:
            logger.warning(f"Error evaluating schema compliance: {e}")
            return {
                "compliance_rate": 0.0,
                "violations": [],
                "score": 0.0
            }
    
    def evaluate_completeness(self, data: Dict[str, Any], expected_fields: set) -> float:
        """Evaluate the completeness of extracted data."""
        if not expected_fields:
            return 1.0
            
        present_fields = set(k for k, v in data.items() if v is not None and str(v).strip() != "")
        return len(present_fields.intersection(expected_fields)) / len(expected_fields)
    
    def evaluate_relevance(self, data: Dict[str, Any], query: str) -> float:
        """Evaluate the relevance of extracted data to a query."""
        if not query:
            return 1.0
            
        # Simple text-based relevance
        text_content = " ".join(str(v) for v in data.values() if isinstance(v, str))
        query_lower = query.lower()
        content_lower = text_content.lower()
        
        # Check for direct matches
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        
        if not query_words:
            return 1.0
            
        matches = len(query_words.intersection(content_words))
        return matches / len(query_words)

    def calculate_overall_quality(self, metrics: Dict[str, float]) -> float:
        """Calculate overall quality score from individual metrics."""
        if not metrics:
            return 0.0
            
        # Apply weights and calculate weighted average
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric, value in metrics.items():
            weight = self._criteria_weights.get(metric, 0.1)  # Default weight
            if value is not None:
                weighted_sum += value * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    # Implementation methods
    def calculate_completeness_score(self, data: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate completeness score for extracted data.
        
        Args:
            data: Extracted data to evaluate
            schema: Optional schema with required fields
            
        Returns:
            Completeness score between 0.0 and 1.0
        """
        if not data:
            return 0.0
            
        # If schema provided, check against required fields
        if schema and "required" in schema:
            required_fields = set(schema["required"])
            return self.evaluate_completeness(data, required_fields)
        
        # Otherwise, calculate based on non-empty fields
        total_fields = len(data)
        if total_fields == 0:
            return 0.0
            
        non_empty_fields = sum(1 for v in data.values() if v is not None and str(v).strip() != "")
        return non_empty_fields / total_fields
    
    def calculate_confidence_score(self, data: Dict[str, Any]) -> float:
        """
        Calculate confidence score for extracted data.
        
        Args:
            data: Extracted data to evaluate
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not data:
            return 0.0
            
        # Check if confidence is already provided in metadata
        if "_metadata" in data and "confidence" in data["_metadata"]:
            return float(data["_metadata"]["confidence"])
        
        # Calculate confidence based on field quality
        field_confidences = []
        for key, value in data.items():
            if key.startswith("_"):  # Skip metadata fields
                continue
                
            confidence = calculate_field_confidence(value, key)
            field_confidences.append(confidence)
        
        if not field_confidences:
            return 0.0
            
        return sum(field_confidences) / len(field_confidences)
    
    def calculate_relevance_score(self, data: Dict[str, Any], options: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate relevance score for extracted data.
        
        Args:
            data: Extracted data to evaluate
            options: Optional evaluation options containing query
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        # If data is empty, relevance should be 0
        if not data or all(v is None or str(v).strip() == "" for v in data.values()):
            return 0.0
            
        if not options or "query" not in options:
            return 1.0  # No query to evaluate against
            
        query = options["query"]
        return self.evaluate_relevance(data, query)
    
    def evaluate_consistency(self, data: Dict[str, Any]) -> float:
        """
        Evaluate internal consistency of extracted data.
        
        Args:
            data: Extracted data to evaluate
            
        Returns:
            Consistency score between 0.0 and 1.0
        """
        # If data is empty, consistency should be 0
        if not data or all(v is None or str(v).strip() == "" for v in data.values()):
            return 0.0
            
        try:
            return measure_data_coherence(data)
        except Exception as e:
            logger.warning(f"Error evaluating consistency: {e}")
            return 0.5  # Default neutral score
    
    def calculate_schema_compliance_rate(self, data: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate how well data complies with a schema.
        
        Args:
            data: Extracted data to evaluate
            schema: Schema to validate against
            
        Returns:
            Compliance rate between 0.0 and 1.0
        """
        # If data is empty, schema compliance should be 0 (unless no schema required)
        if not data or all(v is None or str(v).strip() == "" for v in data.values()):
            if not schema or not schema.get("required"):
                return 1.0  # No data required, no schema required
            return 0.0  # Data required but empty
            
        if not schema:
            return 1.0  # No schema to validate against
            
        try:
            compliance_result = calculate_schema_compliance_rate(data, schema)
            if isinstance(compliance_result, dict):
                return compliance_result.get("compliance_rate", 0.0)
            return float(compliance_result)
        except Exception as e:
            logger.warning(f"Error calculating schema compliance: {e}")
            return 0.5  # Default neutral score
    
    def detect_anomalies(self, data: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Detect anomalies in extracted data.
        
        Args:
            data: Extracted data to evaluate
            schema: Optional schema for validation
            
        Returns:
            List of field names with detected anomalies
        """
        anomalies = []
        
        try:
            outliers = detect_outliers(data)
            anomalies.extend(outliers)
        except Exception as e:
            logger.warning(f"Error detecting outliers: {e}")
        
        # Check for obvious data quality issues
        for key, value in data.items():
            if key.startswith("_"):  # Skip metadata
                continue
                
            # Check for empty required fields
            if schema and "required" in schema and key in schema["required"]:
                if value is None or str(value).strip() == "":
                    anomalies.append(key)
            
            # Check for implausible values
            if isinstance(value, (int, float)):
                if value < 0 and key in ["price", "rating", "count"]:
                    anomalies.append(key)
        
        return list(set(anomalies))  # Remove duplicates
    
    def detect_missing_required_fields(self, data: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Detect missing required fields.
        
        Args:
            data: Extracted data to evaluate
            schema: Schema with required fields
            
        Returns:
            List of missing required field names
        """
        if not schema or "required" not in schema:
            return []
            
        required_fields = set(schema["required"])
        present_fields = set(k for k, v in data.items() if v is not None and str(v).strip() != "")
        
        return list(required_fields - present_fields)
    
    def detect_schema_violations(self, data: Dict[str, Any], schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect schema violations in data.
        
        Args:
            data: Data to validate
            schema: Schema to validate against
            
        Returns:
            List of violation descriptions
        """
        violations = []
        
        if not schema or "properties" not in schema:
            return violations
        
        properties = schema["properties"]
        
        for field, field_schema in properties.items():
            if field not in data:
                continue
                
            value = data[field]
            expected_type = field_schema.get("type")
            
            # Type checking
            if expected_type and value is not None:
                if expected_type == "string" and not isinstance(value, str):
                    violations.append({
                        "field": field,
                        "type": "type_mismatch",
                        "expected": expected_type,
                        "actual": type(value).__name__
                    })
                elif expected_type == "number" and not isinstance(value, (int, float)):
                    violations.append({
                        "field": field,
                        "type": "type_mismatch",
                        "expected": expected_type,
                        "actual": type(value).__name__
                    })
                elif expected_type == "boolean" and not isinstance(value, bool):
                    violations.append({
                        "field": field,
                        "type": "type_mismatch",
                        "expected": expected_type,
                        "actual": type(value).__name__
                    })
        
        return violations
    
    def get_quality_report(self, data: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive quality report.
        
        Args:
            data: Data to evaluate
            schema: Optional schema for validation
            
        Returns:
            Comprehensive quality report
        """
        evaluation = self.evaluate(data, schema)
        
        report = {
            "summary": {
                "overall_quality": evaluation["quality_score"],
                "passes_threshold": evaluation["quality_score"] >= self._confidence_thresholds["medium"],
                "data_fields": len([k for k in data.keys() if not k.startswith("_")]),
                "anomaly_count": len(evaluation["anomalies"])
            },
            "metrics": evaluation["metrics"],
            "anomalies": evaluation["anomalies"],
            "missing_required_fields": evaluation["missing_required_fields"],
            "field_analysis": {},
            "schema_compliance": {},
            "recommendations": []
        }
        
        # Add field-level analysis
        for key, value in data.items():
            if key.startswith("_"):
                continue
                
            field_quality = calculate_field_confidence(value, key)
            report["field_analysis"][key] = {
                "quality": field_quality,
                "type": type(value).__name__,
                "empty": value is None or value == "",
                "length": len(str(value)) if value is not None else 0
            }
        
        # Add schema compliance if schema provided
        if schema:
            report["schema_compliance"] = self.evaluate_schema_compliance(data, schema)
        
        # Add recommendations
        if evaluation["quality_score"] < self._confidence_thresholds["high"]:
            report["recommendations"].append("Consider improving data quality")
        
        if evaluation["anomalies"]:
            report["recommendations"].append(f"Address {len(evaluation['anomalies'])} detected anomalies")
        
        if evaluation["missing_required_fields"]:
            report["recommendations"].append(f"Fill {len(evaluation['missing_required_fields'])} missing required fields")
        
        return report
    
    def evaluate_text_quality(self, text: str) -> float:
        """
        Evaluate the quality of text content.
        
        Args:
            text: Text to evaluate
            
        Returns:
            Text quality score between 0.0 and 1.0
        """
        return calculate_text_quality(text)
    
    def evaluate_structural_consistency(self, data: Dict[str, Any]) -> float:
        """
        Evaluate structural consistency of data.
        
        Args:
            data: Data to evaluate
            
        Returns:
            Structural consistency score between 0.0 and 1.0
        """
        return self.evaluate_consistency(data)

    @property
    def is_initialized(self) -> bool:
        """Check if the evaluator is initialized."""
        return self._initialized
