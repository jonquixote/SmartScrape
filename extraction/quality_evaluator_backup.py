"""
Quality Evaluator Module

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
        
        # Calculate individual metrics
        completeness = self.calculate_completeness_score(data, schema)
        confidence = self.calculate_confidence_score(data)
        relevance = self.calculate_relevance_score(data, options)
        consistency = self.evaluate_consistency(data)
        schema_compliance = self.calculate_schema_compliance_rate(data, schema)
        
        # Calculate overall quality score
        overall_score = self.calculate_overall_quality_score({
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
        return {
            "compliance_rate": self.calculate_schema_compliance_rate(data, schema),
            "violations": self.detect_schema_violations(data, schema),
            "score": self.calculate_schema_compliance_rate(data, schema)
        }
    
    def evaluate_completeness(self, data: Dict[str, Any], expected_fields: set) -> float:
        """Evaluate the completeness of extracted data."""
        if not expected_fields:
            return 1.0
            
        present_fields = set(k for k, v in data.items() if v is not None and v != "")
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
        return self.calculate_overall_quality_score(metrics)
    
    # Implementation methods
    def calculate_completeness_score(self, data: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> float:
                options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate the quality of an extraction result.
        
        Args:
            extraction_result: The extraction result to evaluate
            schema: Optional schema to validate against
            options: Optional evaluation options
            
        Returns:
            Dictionary with quality metrics
        """
        if not self._initialized:
            self.initialize()
            
        options = options or {}
        
        # Extract the data dictionary from ExtractionResult if needed
        if isinstance(extraction_result, ExtractionResult):
            data = extraction_result.data
            metadata = extraction_result.metadata
        else:
            data = extraction_result
            metadata = data.get("_metadata", {}) if isinstance(data, dict) else {}
        
        # Initialize result dictionary
        result = {
            "quality_score": 0.0,
            "metadata": {
                "evaluator": self.__class__.__name__,
                "evaluation_time": options.get("evaluation_time", None),
                "schema_used": bool(schema)
            },
            "metrics": {}
        }
        
        # Skip evaluation for non-dict data
        if not isinstance(data, dict):
            result["error"] = f"Cannot evaluate data of type {type(data).__name__}"
            return result
        
        # Get field metrics
        field_metrics = self._evaluate_fields(data, schema, options)
        result["metrics"].update(field_metrics)
        
        # Get completeness score
        completeness = self.calculate_completeness_score(data, schema)
        result["metrics"]["completeness"] = completeness
        
        # Get confidence score
        confidence = self.calculate_confidence_score(data)
        result["metrics"]["confidence"] = confidence
        
        # Get relevance score if context provided
        context = options.get("context")
        if context:
            relevance = self.calculate_relevance_score(data, context)
            result["metrics"]["relevance"] = relevance
        
        # Validate field types if schema provided
        if schema:
            type_validation = self.validate_field_types(data, schema)
            result["metrics"]["type_validation"] = type_validation
            
            # Check for missing required fields
            missing_fields = self.detect_missing_required_fields(data, schema)
            if missing_fields:
                result["metrics"]["missing_required_fields"] = missing_fields
        
        # Check for anomalies
        anomalies = self.detect_anomalies(data, schema)
        if anomalies:
            result["metrics"]["anomalies"] = anomalies
        
        # Check consistency
        consistency = self.evaluate_consistency(data)
        result["metrics"]["consistency"] = consistency
        
        # Analyze field distribution
        if options.get("analyze_distribution", False):
            distribution = self.analyze_field_distribution(data)
            result["metrics"]["field_distribution"] = distribution
        
        # Evaluate text quality for string fields
        text_quality = {}
        for field, value in data.items():
            if isinstance(value, str) and not field.startswith("_"):
                text_quality[field] = self.evaluate_text_quality(value)
        
        if text_quality:
            result["metrics"]["text_quality"] = text_quality
        
        # Calculate overall quality score
        scores = {}
        
        # Add weighted scores for each available metric
        if "completeness" in result["metrics"]:
            scores["completeness"] = result["metrics"]["completeness"] * self._criteria_weights["completeness"]
            
        if "confidence" in result["metrics"]:
            scores["confidence"] = result["metrics"]["confidence"] * self._criteria_weights["confidence"]
            
        if "relevance" in result["metrics"]:
            scores["relevance"] = result["metrics"]["relevance"] * self._criteria_weights["relevance"]
            
        if "consistency" in result["metrics"]:
            scores["consistency"] = result["metrics"]["consistency"] * self._criteria_weights["consistency"]
            
        if schema and "type_validation" in result["metrics"]:
            schema_score = result["metrics"]["type_validation"].get("valid_rate", 0.0)
            scores["schema_compliance"] = schema_score * self._criteria_weights["schema_compliance"]
        
        # Calculate overall score
        if scores:
            weights_sum = sum(self._criteria_weights[k] for k in scores.keys())
            if weights_sum > 0:
                result["quality_score"] = sum(scores.values()) / weights_sum
        
        # Generate quality report if requested
        if options.get("generate_report", False):
            report = self.get_quality_report(data, schema)
            result["report"] = report
            
            # Identify improvement opportunities
            opportunities = identify_improvement_opportunities(data, result["metrics"])
            if opportunities:
                result["improvement_opportunities"] = opportunities
        
        return result
    
    def calculate_completeness_score(self, data: Dict[str, Any], 
                                   schema: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate a completeness score for the extracted data.
        
        Args:
            data: Extracted data to evaluate
            schema: Optional schema defining required fields
            
        Returns:
            Completeness score between 0.0 and 1.0
        """
        if not data:
            return 0.0
        
        # Exclude metadata fields
        data_fields = {k: v for k, v in data.items() if not k.startswith("_")}
        
        # Count non-empty fields
        non_empty_fields = sum(1 for v in data_fields.values() if v is not None and v != "")
        total_fields = len(data_fields)
        
        # Base completeness on filled fields
        base_completeness = non_empty_fields / max(1, total_fields)
        
        # If schema is provided, check required fields
        if schema:
            required_fields = self._get_required_fields(schema)
            
            if required_fields:
                # Count present required fields
                present_required = sum(1 for field in required_fields if field in data and data[field] is not None and data[field] != "")
                
                # Calculate required field completeness
                required_completeness = present_required / max(1, len(required_fields))
                
                # Final score is weighted average, with required fields weighted higher
                return (required_completeness * 0.7) + (base_completeness * 0.3)
        
        return base_completeness
    
    def calculate_relevance_score(self, data: Dict[str, Any], context: Any) -> float:
        """
        Calculate how relevant the extracted data is to the given context.
        
        Args:
            data: Extracted data to evaluate
            context: Context to evaluate relevance against (query, intent, etc.)
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        if not data or not context:
            return 0.0
        
        context_str = str(context)
        relevance_signals = 0
        data_fields = {k: v for k, v in data.items() if not k.startswith("_")}
        
        # Check field names for relevance to context
        for field in data_fields:
            # Field name appears in context
            if field.lower() in context_str.lower():
                relevance_signals += 1
        
        # Check field values for relevance to context
        for value in data_fields.values():
            if isinstance(value, str) and context_str.lower() in value.lower():
                relevance_signals += 0.5
            elif isinstance(value, (list, tuple)) and any(
                isinstance(item, str) and context_str.lower() in item.lower() 
                for item in value
            ):
                relevance_signals += 0.5
        
        # Simple relevance calculation
        return min(1.0, relevance_signals / max(3, len(data_fields) / 2))
    
    def calculate_confidence_score(self, data: Dict[str, Any]) -> float:
        """
        Calculate a confidence score for the extracted data.
        
        Args:
            data: Extracted data to evaluate
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not data:
            return 0.0
        
        # Check if confidence score already exists in metadata
        if isinstance(data, dict) and "_metadata" in data and "confidence_score" in data["_metadata"]:
            return float(data["_metadata"]["confidence_score"])
        
        # Calculate field-level confidence
        field_confidence = {}
        total_confidence = 0.0
        field_count = 0
        
        for field, value in data.items():
            if field.startswith("_"):
                continue
                
            field_count += 1
            
            # Apply appropriate confidence calculation based on field and value type
            if isinstance(value, str):
                # Apply specific pattern matching if we have a known field type
                pattern = None
                field_lower = field.lower()
                
                for type_name, type_pattern in self._field_type_patterns.items():
                    if type_name in field_lower or field_lower.endswith(f"_{type_name}"):
                        pattern = type_pattern
                        break
                
                field_confidence[field] = calculate_field_confidence(value, pattern)
            else:
                field_confidence[field] = calculate_field_confidence(value)
                
            total_confidence += field_confidence[field]
        
        # Store field-level confidence for detailed reports
        self._field_confidence = field_confidence
        
        # Calculate overall confidence
        average_confidence = total_confidence / max(1, field_count)
        return average_confidence
    
    def detect_anomalies(self, data: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Detect anomalous or suspicious values in the data.
        
        Args:
            data: Extracted data to analyze
            schema: Optional schema for additional validation context
            
        Returns:
            Dictionary of detected anomalies
        """
        anomalies = {}
        
        # Exclude metadata fields
        data_fields = {k: v for k, v in data.items() if not k.startswith("_")}
        
        # Check for empty strings or null values where not expected
        for field, value in data_fields.items():
            # Skip fields that are allowed to be empty
            if schema and self._is_nullable_field(field, schema):
                continue
                
            if value == "":
                anomalies[field] = {
                    "type": "empty_string",
                    "severity": "medium",
                    "message": "Empty string found"
                }
        
        # Check numerical fields for plausibility
        for field, value in data_fields.items():
            if isinstance(value, (int, float)):
                field_lower = field.lower()
                
                # Check price fields for negative values
                if "price" in field_lower or "cost" in field_lower or "amount" in field_lower:
                    if value < 0:
                        anomalies[field] = {
                            "type": "implausible_value",
                            "severity": "high",
                            "message": f"Negative price/cost value: {value}"
                        }
                    elif value > 1000000:  # Very high price
                        anomalies[field] = {
                            "type": "suspicious_value",
                            "severity": "medium",
                            "message": f"Unusually high price/cost value: {value}"
                        }
                
                # Check other numeric fields for extreme values
                elif "rating" in field_lower:
                    if value < 0 or value > 10:
                        anomalies[field] = {
                            "type": "out_of_range",
                            "severity": "medium",
                            "message": f"Rating outside expected range (0-10): {value}"
                        }
                
                # Check year fields
                elif "year" in field_lower:
                    current_year = 2025  # Use a method to get current year in production
                    if value < 1900 or value > current_year + 5:
                        anomalies[field] = {
                            "type": "out_of_range",
                            "severity": "medium",
                            "message": f"Year outside reasonable range: {value}"
                        }
        
        # Check string fields with expected formats
        for field, value in data_fields.items():
            if isinstance(value, str):
                field_lower = field.lower()
                
                # Check email fields
                if "email" in field_lower:
                    if "@" not in value or "." not in value.split("@")[-1]:
                        anomalies[field] = {
                            "type": "format_error",
                            "severity": "high",
                            "message": f"Invalid email format: {value}"
                        }
                
                # Check URL fields
                elif any(term in field_lower for term in ["url", "link", "website"]):
                    if not value.startswith(("http://", "https://", "www.")):
                        anomalies[field] = {
                            "type": "format_error",
                            "severity": "medium",
                            "message": f"Invalid URL format: {value}"
                        }
                
                # Check date fields
                elif any(term in field_lower for term in ["date", "time", "published", "created"]):
                    if check_date_validity(value) < 0.7:
                        anomalies[field] = {
                            "type": "format_error",
                            "severity": "medium",
                            "message": f"Invalid or ambiguous date format: {value}"
                        }
                
                # Check for HTML remnants in content fields
                elif any(term in field_lower for term in ["text", "content", "description", "summary"]):
                    if re.search(r'</?[a-z]+[^>]*>', value):
                        anomalies[field] = {
                            "type": "html_artifact",
                            "severity": "low",
                            "message": "Contains HTML tags"
                        }
        
        # Use statistical methods to detect outliers
        statistical_outliers = detect_outliers(data_fields)
        
        # Add statistical outliers to anomalies with lower severity
        for field, outliers in statistical_outliers.items():
            if field not in anomalies:
                anomalies[field] = {
                    "type": "statistical_outlier",
                    "severity": "low",
                    "message": f"Statistical outlier detected: {outliers}"
                }
        
        return anomalies
    
    def validate_field_types(self, data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that field values match their expected types.
        
        Args:
            data: Extracted data to validate
            schema: Schema defining expected field types
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "valid_fields": [],
            "invalid_fields": [],
            "valid_rate": 0.0,
            "type_mismatches": []
        }
        
        # Extract schema fields
        fields = {}
        if "properties" in schema:
            # JSON Schema format
            fields = schema["properties"]
        elif "fields" in schema:
            # Custom schema format
            fields = {f["name"]: f for f in schema["fields"] if "name" in f}
        else:
            # Assume schema directly maps field names to their specifications
            fields = schema
        
        # Track validation counts
        valid_count = 0
        total_count = 0
        
        # Check each field with defined type
        for field_name, field_schema in fields.items():
            # Skip metadata fields
            if field_name.startswith("_"):
                continue
                
            # Skip if field is not in data
            if field_name not in data:
                continue
                
            total_count += 1
            value = data[field_name]
            
            # Get expected type
            expected_type = field_schema.get("type")
            if not expected_type:
                # Skip fields without defined type
                continue
            
            # Compare actual type with expected type
            type_match = self._check_type_match(value, expected_type)
            
            if type_match:
                valid_count += 1
                results["valid_fields"].append(field_name)
            else:
                results["invalid_fields"].append(field_name)
                results["type_mismatches"].append({
                    "field": field_name,
                    "expected_type": expected_type,
                    "actual_type": type(value).__name__,
                    "value": str(value)[:100]  # Truncate long values
                })
        
        # Calculate validation rate
        results["valid_rate"] = valid_count / max(1, total_count)
        
        return results
    
    def evaluate_consistency(self, data: Dict[str, Any]) -> float:
        """
        Evaluate internal consistency of the data.
        
        Args:
            data: Extracted data to evaluate
            
        Returns:
            Consistency score between 0.0 and 1.0
        """
        if not data:
            return 0.0
        
        # Check for coherence between fields
        coherence_score = measure_data_coherence(data)
        
        # Check for field content consistency (e.g., dates in same format)
        format_consistency = self.evaluate_structural_consistency(data)
        
        # Combine scores
        return (coherence_score + format_consistency) / 2
    
    def get_quality_report(self, data: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive quality report for the data.
        
        Args:
            data: Extracted data to evaluate
            schema: Optional schema for validation
            
        Returns:
            Dictionary with detailed quality report
        """
        report = {
            "summary": {},
            "field_analysis": {},
            "anomalies": [],
            "schema_compliance": None,
            "recommendations": []
        }
        
        # Generate quality profile
        profile = generate_quality_profile(data)
        report["summary"] = {
            "field_count": profile["field_count"],
            "completeness": profile["completeness"],
            "data_types": profile["field_types"],
            "overall_quality": profile["overall_score"]
        }
        
        # Add schema compliance if schema is provided
        if schema:
            compliance = calculate_schema_compliance_rate(data, schema)
            report["schema_compliance"] = {
                "compliance_rate": compliance["compliance_rate"],
                "missing_required_fields": compliance["missing_required_fields"],
                "type_mismatches": compliance["type_mismatches"],
                "constraint_violations": compliance["constraint_violations"]
            }
        
        # Add field-level analysis
        for field, value in data.items():
            if field.startswith("_"):
                continue
                
            field_analysis = {"type": type(value).__name__}
            
            # Add confidence score if available
            if hasattr(self, "_field_confidence"):
                field_analysis["confidence"] = self._field_confidence.get(field, 0.0)
            
            # Add text quality for string fields
            if isinstance(value, str):
                field_analysis["text_quality"] = self.evaluate_text_quality(value)
                
                # Identify field subtypes
                if check_date_validity(value) > 0.7:
                    field_analysis["subtype"] = "date"
                elif check_url_validity(value) > 0.7:
                    field_analysis["subtype"] = "url"
            
            # Add numeric field analysis
            elif isinstance(value, (int, float)):
                field_analysis["value"] = value
                
                # Check if there's a plausible range
                field_lower = field.lower()
                if "price" in field_lower:
                    field_analysis["plausibility"] = measure_numerical_plausibility(
                        value, expected_range=(0.01, 100000)
                    )
                elif "rating" in field_lower:
                    field_analysis["plausibility"] = measure_numerical_plausibility(
                        value, expected_range=(0, 10)
                    )
                elif "year" in field_lower:
                    field_analysis["plausibility"] = measure_numerical_plausibility(
                        value, expected_range=(1900, 2030)
                    )
                else:
                    field_analysis["plausibility"] = measure_numerical_plausibility(value)
            
            report["field_analysis"][field] = field_analysis
        
        # Add anomalies
        anomalies = self.detect_anomalies(data, schema)
        if anomalies:
            for field, anomaly in anomalies.items():
                report["anomalies"].append({
                    "field": field,
                    "type": anomaly["type"],
                    "severity": anomaly["severity"],
                    "message": anomaly["message"]
                })
        
        # Generate recommendations
        if report["summary"]["completeness"] < 0.7:
            report["recommendations"].append({
                "type": "completeness",
                "message": "Data is incomplete. Consider using alternative extraction methods."
            })
            
        if anomalies:
            report["recommendations"].append({
                "type": "anomalies",
                "message": f"Found {len(anomalies)} anomalous fields. Review and validate these fields."
            })
            
        if schema and report["schema_compliance"]["missing_required_fields"]:
            report["recommendations"].append({
                "type": "required_fields",
                "message": "Missing required fields. Try a more comprehensive extraction approach."
            })
        
        # Add overall assessment
        quality_score = profile["overall_score"]
        if quality_score >= 0.8:
            assessment = "Excellent quality data, suitable for production use."
        elif quality_score >= 0.6:
            assessment = "Good quality data, but some fields may need verification."
        elif quality_score >= 0.4:
            assessment = "Fair quality data, review and enhancement recommended."
        else:
            assessment = "Poor quality data, significant improvements needed."
            
        report["summary"]["assessment"] = assessment
        
        return report
    
    def evaluate_text_quality(self, text: str) -> float:
        """
        Evaluate the quality of text content.
        
        Args:
            text: Text to evaluate
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        return calculate_text_quality(text)
    
    def evaluate_structural_consistency(self, data: Dict[str, Any]) -> float:
        """
        Evaluate structural consistency of the data.
        
        Args:
            data: Data to evaluate
            
        Returns:
            Consistency score between 0.0 and 1.0
        """
        if not data:
            return 0.0
        
        # Group fields by apparent type
        date_fields = []
        url_fields = []
        price_fields = []
        
        for field, value in data.items():
            if field.startswith("_") or not isinstance(value, str):
                continue
                
            # Categorize fields by content
            if check_date_validity(value) > 0.7:
                date_fields.append((field, value))
            elif check_url_validity(value) > 0.7:
                url_fields.append((field, value))
            elif re.search(r'^\s*[\$£€¥]?\s*\d+(\.\d{1,2})?\s*$', value):
                price_fields.append((field, value))
        
        consistency_scores = []
        
        # Check date format consistency
        if len(date_fields) > 1:
            formats = set()
            for _, date_str in date_fields:
                # Simple format detection (very basic)
                if re.match(r'^\d{4}-\d{2}-\d{2}', date_str):
                    formats.add("ISO")
                elif re.match(r'^\d{1,2}[/-]\d{1,2}[/-]\d{4}', date_str):
                    formats.add("DD/MM/YYYY")
                elif re.match(r'^\d{1,2}\s+[A-Za-z]+\s+\d{4}', date_str):
                    formats.add("DD Month YYYY")
                else:
                    formats.add("other")
            
            # Calculate consistency score
            date_consistency = 1.0 if len(formats) == 1 else 1.0 / len(formats)
            consistency_scores.append(date_consistency)
        
        # Check URL format consistency
        if len(url_fields) > 1:
            formats = set()
            for _, url in url_fields:
                if url.startswith("https://"):
                    formats.add("https")
                elif url.startswith("http://"):
                    formats.add("http")
                elif url.startswith("www."):
                    formats.add("www")
                else:
                    formats.add("other")
            
            # Calculate consistency score
            url_consistency = 1.0 if len(formats) == 1 else 1.0 / len(formats)
            consistency_scores.append(url_consistency)
        
        # Check price format consistency
        if len(price_fields) > 1:
            formats = set()
            for _, price in price_fields:
                if price.startswith("$"):
                    formats.add("USD")
                elif price.startswith("£"):
                    formats.add("GBP")
                elif price.startswith("€"):
                    formats.add("EUR")
                elif price.startswith("¥"):
                    formats.add("JPY")
                elif re.search(r'^\d', price):
                    formats.add("no_symbol")
                else:
                    formats.add("other")
            
            # Calculate consistency score
            price_consistency = 1.0 if len(formats) == 1 else 1.0 / len(formats)
            consistency_scores.append(price_consistency)
        
        # Calculate overall structural consistency
        if consistency_scores:
            return sum(consistency_scores) / len(consistency_scores)
        else:
            return 1.0  # No consistency issues found
    
    def detect_extraction_errors(self, data: Dict[str, Any]) -> Dict[str, str]:
        """
        Identify likely extraction errors in the data.
        
        Args:
            data: Data to analyze
            
        Returns:
            Dictionary mapping field names to detected errors
        """
        errors = {}
        
        for field, value in data.items():
            if field.startswith("_"):
                continue
                
            # Check for HTML artifacts
            if isinstance(value, str):
                # Check for HTML tags
                if re.search(r'</?[a-z]+[^>]*>', value):
                    errors[field] = "Contains HTML tags"
                
                # Check for encoded entities
                if re.search(r'&[a-z]+;', value):
                    errors[field] = "Contains HTML entities"
                
                # Check for JSON/code artifacts
                if re.search(r'[{}\[\]":,]', value) and (
                    value.startswith(("{", "[")) or
                    re.search(r'"[a-z_]+"\s*:', value, re.IGNORECASE)
                ):
                    errors[field] = "Contains JSON/code fragments"
                
                # Check for template artifacts
                if re.search(r'{{.*?}}|\${.*?}|%\(.*?\)s', value):
                    errors[field] = "Contains template syntax"
                
                # Check for boilerplate text that might indicate extraction error
                boilerplate_patterns = [
                    r'lorem ipsum',
                    r'click here',
                    r'loading',
                    r'undefined',
                    r'null',
                    r'not found',
                    r'no information',
                    r'coming soon'
                ]
                
                for pattern in boilerplate_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        errors[field] = f"Contains boilerplate text: '{pattern}'"
                        break
        
        return errors
    
    def evaluate_data_relationships(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify relational integrity and logic between fields.
        
        Args:
            data: Data to analyze
            
        Returns:
            Dictionary with relationship validation results
        """
        results = {
            "valid_relationships": [],
            "invalid_relationships": [],
            "relationship_score": 1.0
        }
        
        # Define common field relationships
        relationships = [
            {
                "id": "price_currency",
                "type": "dependency",
                "fields": ["price", "currency"],
                "description": "If price exists, currency should also exist"
            },
            {
                "id": "start_end_dates",
                "type": "numeric_comparison",
                "fields": ["start_date", "end_date"],
                "condition": "<=",
                "description": "Start date should be before or equal to end date"
            },
            {
                "id": "dimensions_consistency",
                "type": "consistent_units",
                "fields": ["width", "height", "depth"],
                "description": "Dimension fields should use consistent units"
            }
        ]
        
        # Customize relationships based on field existence
        field_keys = set(data.keys())
        
        # Name field relationships
        if "first_name" in field_keys and "last_name" in field_keys and "name" in field_keys:
            relationships.append({
                "id": "name_consistency",
                "type": "custom",
                "fields": ["first_name", "last_name", "name"],
                "description": "Full name should include first and last name"
            })
        
        # Price relationships
        price_fields = [f for f in field_keys if "price" in f.lower()]
        if len(price_fields) > 1:
            # Check for discount, regular, sale prices
            if "discount_price" in field_keys and "regular_price" in field_keys:
                relationships.append({
                    "id": "discount_price",
                    "type": "numeric_comparison",
                    "fields": ["discount_price", "regular_price"],
                    "condition": "<",
                    "description": "Discount price should be less than regular price"
                })
        
        # Evaluate defined relationships
        relationship_results = check_field_relationships(data, relationships)
        
        # Process results
        valid_count = 0
        total_count = 0
        
        for rel_id, score in relationship_results.items():
            rel_info = next((r for r in relationships if r.get("id") == rel_id), None)
            if not rel_info:
                continue
                
            total_count += 1
            
            if score >= 0.8:  # Consider highly valid
                valid_count += 1
                results["valid_relationships"].append({
                    "id": rel_id,
                    "description": rel_info.get("description", ""),
                    "score": score
                })
            elif score > 0:  # Partially valid
                valid_count += score  # Add partial validity
                results["invalid_relationships"].append({
                    "id": rel_id,
                    "description": rel_info.get("description", ""),
                    "score": score,
                    "fields": rel_info.get("fields", [])
                })
        
        # Custom relationship checks
        for rel in relationships:
            if rel["id"] == "name_consistency" and rel["id"] not in relationship_results:
                # Check if full name contains first and last name
                if all(f in data for f in ["first_name", "last_name", "name"]):
                    first = data["first_name"]
                    last = data["last_name"]
                    full = data["name"]
                    
                    if isinstance(first, str) and isinstance(last, str) and isinstance(full, str):
                        if first in full and last in full:
                            valid_count += 1
                            results["valid_relationships"].append({
                                "id": "name_consistency",
                                "description": "Full name includes first and last name",
                                "score": 1.0
                            })
                        else:
                            results["invalid_relationships"].append({
                                "id": "name_consistency",
                                "description": "Full name doesn't include first and last name",
                                "score": 0.0,
                                "fields": ["first_name", "last_name", "name"]
                            })
                    
                    total_count += 1
        
        # Calculate overall relationship score
        if total_count > 0:
            results["relationship_score"] = valid_count / total_count
        
        return results
    
    def assess_source_reliability(self, source: Any) -> float:
        """
        Evaluate the reliability of the data source.
        
        Args:
            source: Source information (URL, API, etc.)
            
        Returns:
            Reliability score between 0.0 and 1.0
        """
        # This is a placeholder implementation
        # In a real implementation, this could check against known reliable sources,
        # domain reputation, SSL certification, etc.
        
        if not source:
            return 0.5  # Default moderate reliability
        
        reliability = 0.5  # Start with moderate reliability
        
        if isinstance(source, str):
            # Check if it's a URL
            if source.startswith(("http://", "https://", "www.")):
                # Higher trust for HTTPS
                if source.startswith("https://"):
                    reliability += 0.1
                
                # Higher trust for well-known domains
                trusted_domains = [
                    ".gov", ".edu", ".org", 
                    "wikipedia.org", "amazon.com", "nytimes.com"
                ]
                
                if any(domain in source.lower() for domain in trusted_domains):
                    reliability += 0.2
                
                # Lower trust for certain TLDs
                suspicious_tlds = [".xyz", ".info", ".biz", ".top"]
                if any(source.lower().endswith(tld) for tld in suspicious_tlds):
                    reliability -= 0.1
            
            # API source format often indicates structured data source
            elif "api" in source.lower():
                reliability += 0.1
        
        # Ensure result is in 0.0-1.0 range
        return max(0.0, min(1.0, reliability))
    
    def detect_missing_required_fields(self, data: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
        """
        Check for missing required fields.
        
        Args:
            data: Data to check
            schema: Schema defining required fields
            
        Returns:
            List of missing required field names
        """
        missing_fields = []
        required_fields = self._get_required_fields(schema)
        
        for field in required_fields:
            if field not in data or data[field] is None or data[field] == "":
                missing_fields.append(field)
        
        return missing_fields
    
    def analyze_field_distribution(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform statistical analysis of field values.
        
        Args:
            data: Data to analyze
            
        Returns:
            Dictionary with distribution analysis
        """
        analysis = {}
        
        # Count field types
        type_counts = {}
        for field, value in data.items():
            if field.startswith("_"):
                continue
                
            # Get field type
            field_type = type(value).__name__
            type_counts[field_type] = type_counts.get(field_type, 0) + 1
        
        analysis["type_distribution"] = type_counts
        
        # Analyze string lengths if there are string fields
        string_fields = {k: v for k, v in data.items() 
                       if isinstance(v, str) and not k.startswith("_")}
        
        if string_fields:
            string_lengths = [len(v) for v in string_fields.values()]
            
            if string_lengths:
                analysis["string_length"] = {
                    "min": min(string_lengths),
                    "max": max(string_lengths),
                    "mean": sum(string_lengths) / len(string_lengths),
                    "median": sorted(string_lengths)[len(string_lengths) // 2]
                }
                
                # Add distribution by categories
                length_distribution = {
                    "short (1-10)": 0,
                    "medium (11-100)": 0,
                    "long (101-1000)": 0,
                    "very_long (1000+)": 0
                }
                
                for length in string_lengths:
                    if length <= 10:
                        length_distribution["short (1-10)"] += 1
                    elif length <= 100:
                        length_distribution["medium (11-100)"] += 1
                    elif length <= 1000:
                        length_distribution["long (101-1000)"] += 1
                    else:
                        length_distribution["very_long (1000+)"] += 1
                
                analysis["string_length"]["distribution"] = length_distribution
        
        # Analyze numeric values if there are numeric fields
        numeric_fields = {k: v for k, v in data.items() 
                        if isinstance(v, (int, float)) and not k.startswith("_")}
        
        if numeric_fields:
            numeric_values = list(numeric_fields.values())
            
            if numeric_values:
                analysis["numeric_values"] = {
                    "min": min(numeric_values),
                    "max": max(numeric_values),
                    "mean": sum(numeric_values) / len(numeric_values)
                }
                
                # Add distribution by range
                if len(numeric_values) > 1:
                    min_val = min(numeric_values)
                    max_val = max(numeric_values)
                    range_size = max_val - min_val
                    
                    if range_size > 0:
                        # Create quartile distribution
                        quartiles = {
                            "q1": 0,
                            "q2": 0,
                            "q3": 0,
                            "q4": 0
                        }
                        
                        q1_threshold = min_val + (range_size * 0.25)
                        q2_threshold = min_val + (range_size * 0.5)
                        q3_threshold = min_val + (range_size * 0.75)
                        
                        for value in numeric_values:
                            if value < q1_threshold:
                                quartiles["q1"] += 1
                            elif value < q2_threshold:
                                quartiles["q2"] += 1
                            elif value < q3_threshold:
                                quartiles["q3"] += 1
                            else:
                                quartiles["q4"] += 1
                        
                        analysis["numeric_values"]["quartiles"] = quartiles
        
        # Analyze array lengths if there are array fields
        array_fields = {k: v for k, v in data.items() 
                      if isinstance(v, (list, tuple)) and not k.startswith("_")}
        
        if array_fields:
            array_lengths = [len(v) for v in array_fields.values()]
            
            if array_lengths:
                analysis["array_length"] = {
                    "min": min(array_lengths),
                    "max": max(array_lengths),
                    "mean": sum(array_lengths) / len(array_lengths),
                    "empty_arrays": len([l for l in array_lengths if l == 0])
                }
        
        return analysis
    
    def _evaluate_fields(self, data: Dict[str, Any], schema: Optional[Dict[str, Any]] = None,
                       options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate individual fields in the data.
        
        Args:
            data: Data to evaluate
            schema: Optional schema for validation
            options: Optional evaluation options
            
        Returns:
            Dictionary with field evaluation metrics
        """
        field_metrics = {
            "_field_confidence": {},
            "_field_completeness": {},
            "_field_anomalies": {}
        }
        
        # Evaluate each field
        for field, value in data.items():
            if field.startswith("_"):
                continue
                
            # Calculate field confidence
            if options and options.get("pattern_map", {}).get(field):
                pattern = options["pattern_map"][field]
                field_metrics["_field_confidence"][field] = calculate_field_confidence(value, pattern)
            else:
                field_metrics["_field_confidence"][field] = calculate_field_confidence(value)
            
            # Check field completeness
            field_metrics["_field_completeness"][field] = 1.0 if value is not None and value != "" else 0.0
            
            # Check for anomalies
            if isinstance(value, str):
                # Check for empty strings
                if value == "":
                    field_metrics["_field_anomalies"][field] = "empty_string"
                
                # Check for error indicators
                error_indicators = ["error", "undefined", "null", "none", "n/a", "not found", "not available"]
                if any(value.lower() == indicator for indicator in error_indicators):
                    field_metrics["_field_anomalies"][field] = "error_indicator"
                
                # Check for HTML remnants
                if re.search(r'</?[a-z]+[^>]*>', value, re.IGNORECASE):
                    field_metrics["_field_anomalies"][field] = "html_remnant"
            
            # Check numeric fields
            elif isinstance(value, (int, float)):
                if math.isnan(value) or math.isinf(value):
                    field_metrics["_field_anomalies"][field] = "invalid_number"
        
        # Remove empty dictionaries
        for key in list(field_metrics.keys()):
            if not field_metrics[key]:
                del field_metrics[key]
        
        return field_metrics
    
    def _get_required_fields(self, schema: Dict[str, Any]) -> List[str]:
        """
        Extract list of required field names from schema.
        
        Args:
            schema: Schema to analyze
            
        Returns:
            List of required field names
        """
        required_fields = []
        
        # Handle JSON Schema format
        if "required" in schema:
            required_fields.extend(schema["required"])
        
        # Handle custom schema format with fields array
        if "fields" in schema and isinstance(schema["fields"], list):
            required_fields.extend(
                field["name"] for field in schema["fields"] 
                if "name" in field and field.get("required", False)
            )
        
        # Handle direct field specifications
        if "properties" in schema:
            for field_name, field_schema in schema["properties"].items():
                if field_schema.get("required", False):
                    required_fields.append(field_name)
        
        return required_fields
    
    def _is_nullable_field(self, field: str, schema: Dict[str, Any]) -> bool:
        """
        Check if a field is allowed to be null/empty.
        
        Args:
            field: Field name to check
            schema: Schema to check against
            
        Returns:
            True if the field is nullable, False otherwise
        """
        # First check if it's a required field
        required_fields = self._get_required_fields(schema)
        if field in required_fields:
            return False
        
        # Check field-specific nullable property
        field_schema = None
        
        # Handle JSON Schema format
        if "properties" in schema and field in schema["properties"]:
            field_schema = schema["properties"][field]
        
        # Handle custom schema format with fields array
        elif "fields" in schema and isinstance(schema["fields"], list):
            for schema_field in schema["fields"]:
                if schema_field.get("name") == field:
                    field_schema = schema_field
                    break
        
        # Check nullable property
        if field_schema:
            # Schema may use different terms for nullability
            for prop in ["nullable", "allowNull", "allow_null", "optional"]:
                if prop in field_schema:
                    return bool(field_schema[prop])
        
        # Default: non-required fields are nullable
        return True
    
    def _check_type_match(self, value: Any, expected_type: str) -> bool:
        """
        Check if a value matches an expected type.
        
        Args:
            value: Value to check
            expected_type: Expected type string
            
        Returns:
            True if types match, False otherwise
        """
        if value is None:
            return expected_type in ["null", "any"]
        
        if expected_type == "string":
            return isinstance(value, str)
        
        if expected_type == "number":
            return isinstance(value, (int, float))
        
        if expected_type == "integer":
            return isinstance(value, int)
        
        if expected_type == "boolean":
            return isinstance(value, bool)
        
        if expected_type == "array":
            return isinstance(value, (list, tuple))
        
        if expected_type == "object":
            return isinstance(value, dict)
        
        if expected_type == "date":
            if isinstance(value, str):
                return check_date_validity(value) > 0.7
            return False
        
        if expected_type == "url":
            if isinstance(value, str):
                return check_url_validity(value) > 0.7
            return False
        
        # Default to string representation comparison
        return str(type(value).__name__) == expected_type