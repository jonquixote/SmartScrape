"""
Quality Evaluator Implementation

This module provides a complete implementation of the QualityEvaluator 
abstract class for evaluating extraction quality.
"""

import logging
import json
import copy
import math
import re
from typing import Dict, Any, List, Optional, Union, Set, Tuple

from extraction.core.extraction_interface import QualityEvaluator, BaseExtractor
from extraction.core.extraction_result import ExtractionResult
from core.service_interface import BaseService
from extraction.helpers.quality_metrics import (
    calculate_text_quality, calculate_field_confidence, measure_numerical_plausibility,
    check_date_validity, check_url_validity, measure_enum_validity, check_field_relationships,
    calculate_overall_quality_score, generate_quality_profile, identify_improvement_opportunities,
    calculate_schema_compliance_rate, detect_outliers, measure_data_coherence
)

logger = logging.getLogger(__name__)

class QualityEvaluatorImpl(QualityEvaluator, BaseService):
    """
    Complete implementation of the QualityEvaluator interface.
    
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
        self._initialized = False
        
    @property
    def name(self) -> str:
        """Return the service name."""
        return "quality_evaluator"
        
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
        
        # Calculate completeness
        completeness_result = self.evaluate_completeness(data, schema)
        result["metrics"]["completeness"] = completeness_result
        
        # Calculate relevance
        relevance_result = self.evaluate_relevance(data, options.get("query", ""))
        result["metrics"]["relevance"] = relevance_result
        
        # Check schema compliance if schema is provided
        if schema:
            compliance_result = self.evaluate_schema_compliance(data, schema)
            result["metrics"]["schema_compliance"] = compliance_result
        
        # Calculate overall quality score
        result["quality_score"] = self.calculate_overall_quality(result["metrics"])
        
        # Add improvement recommendations
        result["recommendations"] = self._generate_recommendations(data, result["metrics"])
        
        return result
    
    def evaluate_completeness(self, data: Dict[str, Any], 
                      schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate the completeness of extracted data.
        
        Args:
            data: The extracted data to evaluate
            schema: Optional schema to check required fields
            
        Returns:
            Dictionary with completeness metrics
        """
        result = {
            "score": 0.0,
            "field_count": 0,
            "non_empty_fields": 0,
            "empty_fields": [],
            "missing_fields": []
        }
        
        # Count fields and non-empty fields
        field_count = 0
        non_empty_count = 0
        empty_fields = []
        
        for key, value in data.items():
            # Skip metadata fields
            if key.startswith("_"):
                continue
                
            field_count += 1
            
            # Check if field has a non-empty value
            if value is None or value == "" or value == [] or value == {}:
                empty_fields.append(key)
            else:
                non_empty_count += 1
        
        result["field_count"] = field_count
        result["non_empty_fields"] = non_empty_count
        result["empty_fields"] = empty_fields
        
        # Check required fields if schema is provided
        if schema:
            required_fields = self._get_required_fields(schema)
            missing_fields = [field for field in required_fields if field not in data or data[field] is None or data[field] == ""]
            result["missing_fields"] = missing_fields
            
            # Adjust completeness score based on required fields
            if required_fields:
                present_required = len(required_fields) - len(missing_fields)
                required_score = present_required / len(required_fields)
                
                # Weight required fields more heavily
                result["score"] = (required_score * 0.7) + ((non_empty_count / max(1, field_count)) * 0.3)
            else:
                result["score"] = non_empty_count / max(1, field_count)
        else:
            # Without schema, score is just the proportion of non-empty fields
            result["score"] = non_empty_count / max(1, field_count)
        
        return result
    
    def evaluate_relevance(self, data: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Evaluate the relevance of extracted data to the original query.
        
        Args:
            data: The extracted data to evaluate
            query: The original search query
            
        Returns:
            Dictionary with relevance metrics
        """
        result = {
            "score": 0.0,
            "query_term_matches": 0,
            "query_terms_found": []
        }
        
        # If no query, use default relevance
        if not query:
            result["score"] = 0.7  # Default medium-high relevance
            return result
        
        # Prepare query terms
        query_terms = set(query.lower().split())
        data_str = json.dumps(data).lower()
        
        # Count term matches
        matches = 0
        matched_terms = []
        
        for term in query_terms:
            if len(term) <= 2:  # Skip very short terms
                continue
                
            if term in data_str:
                matches += 1
                matched_terms.append(term)
        
        result["query_term_matches"] = matches
        result["query_terms_found"] = matched_terms
        
        # Calculate relevance score
        relevant_terms = len([term for term in query_terms if len(term) > 2])
        if relevant_terms > 0:
            result["score"] = matches / relevant_terms
        else:
            result["score"] = 0.5  # Default medium relevance
        
        return result
    
    def evaluate_schema_compliance(self, data: Dict[str, Any], 
                           schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate how well the extracted data complies with a schema.
        
        Args:
            data: The extracted data to evaluate
            schema: The schema to validate against
            
        Returns:
            Dictionary with schema compliance metrics
        """
        result = {
            "score": 0.0,
            "valid_fields": 0,
            "invalid_fields": [],
            "missing_required_fields": [],
            "type_mismatches": []
        }
        
        # Get required fields
        required_fields = self._get_required_fields(schema)
        
        # Check for missing required fields
        missing = [field for field in required_fields if field not in data or data[field] is None or data[field] == ""]
        result["missing_required_fields"] = missing
        
        # Validate against schema properties
        properties = self._get_schema_properties(schema)
        
        valid_fields = 0
        invalid_fields = []
        type_mismatches = []
        
        for field, spec in properties.items():
            # Skip if field is not in data
            if field not in data:
                continue
                
            # Get expected type
            expected_type = spec.get("type")
            if not expected_type:
                continue
                
            # Compare actual type with expected type
            value = data[field]
            type_match = self._check_type_match(value, expected_type)
            
            if type_match:
                valid_fields += 1
            else:
                invalid_fields.append(field)
                type_mismatches.append({
                    "field": field,
                    "expected_type": expected_type,
                    "actual_type": type(value).__name__,
                    "value": str(value)[:100]  # Truncate long values
                })
        
        result["valid_fields"] = valid_fields
        result["invalid_fields"] = invalid_fields
        result["type_mismatches"] = type_mismatches
        
        # Calculate compliance score
        total_fields = valid_fields + len(invalid_fields)
        required_score = (len(required_fields) - len(missing)) / max(1, len(required_fields)) if required_fields else 1.0
        
        if total_fields > 0:
            type_score = valid_fields / total_fields
            # Weight required fields and type validation equally
            result["score"] = (required_score * 0.5) + (type_score * 0.5)
        else:
            result["score"] = required_score
        
        return result
    
    def calculate_overall_quality(self, metrics: Dict[str, Any]) -> float:
        """
        Calculate an overall quality score based on all quality metrics.
        
        Args:
            metrics: Dictionary of quality metrics
            
        Returns:
            Overall quality score between 0.0 and 1.0
        """
        # Get scores for each metric
        completeness = metrics.get("completeness", {}).get("score", 0.0)
        relevance = metrics.get("relevance", {}).get("score", 0.0)
        schema_compliance = metrics.get("schema_compliance", {}).get("score", 0.0)
        
        # Apply weights
        weighted_sum = 0.0
        weight_sum = 0.0
        
        if "completeness" in metrics:
            weighted_sum += completeness * self._criteria_weights["completeness"]
            weight_sum += self._criteria_weights["completeness"]
            
        if "relevance" in metrics:
            weighted_sum += relevance * self._criteria_weights["relevance"]
            weight_sum += self._criteria_weights["relevance"]
            
        if "schema_compliance" in metrics:
            weighted_sum += schema_compliance * self._criteria_weights["schema_compliance"]
            weight_sum += self._criteria_weights["schema_compliance"]
        
        # Calculate weighted average
        if weight_sum > 0:
            return weighted_sum / weight_sum
        else:
            return 0.0
    
    def _get_required_fields(self, schema: Dict[str, Any]) -> List[str]:
        """
        Extract required fields from a schema.
        
        Args:
            schema: Schema to extract required fields from
            
        Returns:
            List of required field names
        """
        required = []
        
        # Try standard JSON Schema format
        if "required" in schema and isinstance(schema["required"], list):
            required = schema["required"]
        
        # Try custom schema format
        elif "fields" in schema and isinstance(schema["fields"], list):
            for field in schema["fields"]:
                if isinstance(field, dict) and field.get("required", False):
                    field_name = field.get("name")
                    if field_name:
                        required.append(field_name)
        
        return required
    
    def _get_schema_properties(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract properties from a schema.
        
        Args:
            schema: Schema to extract properties from
            
        Returns:
            Dictionary of property specifications
        """
        properties = {}
        
        # Try standard JSON Schema format
        if "properties" in schema and isinstance(schema["properties"], dict):
            properties = schema["properties"]
        
        # Try custom schema format
        elif "fields" in schema and isinstance(schema["fields"], list):
            for field in schema["fields"]:
                if isinstance(field, dict) and "name" in field:
                    properties[field["name"]] = field
        
        return properties
    
    def _check_type_match(self, value: Any, expected_type: str) -> bool:
        """
        Check if a value matches the expected type.
        
        Args:
            value: Value to check
            expected_type: Expected type as string
            
        Returns:
            True if types match, False otherwise
        """
        if value is None:
            return expected_type == "null"
            
        if expected_type == "string":
            return isinstance(value, str)
            
        if expected_type == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
            
        if expected_type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
            
        if expected_type == "boolean":
            return isinstance(value, bool)
            
        if expected_type == "array":
            return isinstance(value, list)
            
        if expected_type == "object":
            return isinstance(value, dict)
            
        # Handle union types (e.g., "string|number")
        if "|" in expected_type:
            types = expected_type.split("|")
            return any(self._check_type_match(value, t) for t in types)
            
        return False
    
    def _generate_recommendations(self, data: Dict[str, Any], 
                          metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate recommendations for improving extraction quality.
        
        Args:
            data: The extracted data
            metrics: Quality metrics
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # Check completeness
        completeness = metrics.get("completeness", {})
        if completeness.get("score", 1.0) < 0.7:
            # Add recommendations for missing fields
            missing_fields = completeness.get("missing_fields", [])
            if missing_fields:
                recommendations.append({
                    "type": "missing_required_fields",
                    "fields": missing_fields,
                    "message": f"Add extraction for required fields: {', '.join(missing_fields)}"
                })
            
            # Add recommendations for empty fields
            empty_fields = completeness.get("empty_fields", [])
            if empty_fields:
                recommendations.append({
                    "type": "empty_fields",
                    "fields": empty_fields,
                    "message": f"Improve extraction for empty fields: {', '.join(empty_fields[:5])}"
                    + ("..." if len(empty_fields) > 5 else "")
                })
        
        # Check schema compliance
        schema_compliance = metrics.get("schema_compliance", {})
        if schema_compliance.get("score", 1.0) < 0.7:
            # Add recommendations for type mismatches
            type_mismatches = schema_compliance.get("type_mismatches", [])
            if type_mismatches:
                fields = [mismatch["field"] for mismatch in type_mismatches]
                recommendations.append({
                    "type": "type_mismatches",
                    "fields": fields,
                    "message": f"Fix type mismatches in fields: {', '.join(fields[:5])}"
                    + ("..." if len(fields) > 5 else "")
                })
        
        return recommendations
