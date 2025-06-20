"""
Quality Evaluation Stage Module

This module provides a pipeline stage for evaluating the quality and completeness
of extracted data, calculating quality scores, and flagging potential issues.
"""

import logging
from typing import Dict, Any, Optional, Union, List, Tuple
import re

from core.pipeline.stages.base_stages import ProcessingStage
from core.pipeline.context import PipelineContext
from extraction.quality_evaluator_impl import QualityEvaluatorImpl

logger = logging.getLogger(__name__)

class QualityEvaluationStage(ProcessingStage):
    """
    Pipeline stage that evaluates the quality and completeness of extracted data.
    
    This stage calculates quality scores for extracted content, checks for missing
    or low-quality fields, evaluates data completeness, and flags potential issues
    to help improve extraction quality.
    """
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the quality evaluation stage with configuration.
        
        Args:
            name: Name of this stage (defaults to class name)
            config: Configuration dictionary
        """
        super().__init__(name, config)
        self.evaluator = None
        self.input_key = self.config.get("input_key", "normalized_data")
        self.output_key = self.config.get("output_key", "quality_results")
        self.schema_key = self.config.get("schema_key", "extraction_schema")
        self.quality_threshold = self.config.get("quality_threshold", 0.7)
        self.required_fields = self.config.get("required_fields", {})
        self.field_weights = self.config.get("field_weights", {})
        self.check_consistency = self.config.get("check_consistency", True)
        self.check_completeness = self.config.get("check_completeness", True)
        self.check_correctness = self.config.get("check_correctness", True)
        self.flag_issues = self.config.get("flag_issues", True)
        
    async def initialize(self) -> None:
        """Initialize the evaluator and stage resources."""
        if self._initialized:
            return
            
        # Create the quality evaluator
        self.evaluator = QualityEvaluatorImpl()
        
        # Initialize the evaluator with configuration
        evaluator_config = {
            "quality_threshold": self.quality_threshold,
            "required_fields": self.required_fields,
            "field_weights": self.field_weights,
            "check_consistency": self.check_consistency,
            "check_completeness": self.check_completeness,
            "check_correctness": self.check_correctness,
            "flag_issues": self.flag_issues
        }
        self.evaluator.initialize(evaluator_config)
        
        await super().initialize()
        logger.debug(f"{self.name} initialized with quality evaluator")
        
    async def cleanup(self) -> None:
        """Clean up resources used by this stage."""
        if self.evaluator:
            self.evaluator.shutdown()
            
        await super().cleanup()
        logger.debug(f"{self.name} cleaned up")
        
    async def validate_input(self, context: PipelineContext) -> bool:
        """
        Validate that the required inputs are present in the context.
        
        Args:
            context: Pipeline context containing data
            
        Returns:
            True if validation passes, False otherwise
        """
        if not context.has_key(self.input_key):
            logger.warning(f"Missing required input '{self.input_key}' in context")
            context.add_error(self.name, f"Missing required input: {self.input_key}")
            return False
            
        # Check if input is a dictionary
        input_data = context.get(self.input_key)
        if not isinstance(input_data, dict):
            logger.warning(f"Invalid input type for quality evaluation: {type(input_data)}")
            context.add_error(self.name, f"Invalid input type: {type(input_data)}")
            return False
            
        return True
        
    async def transform_data(self, data: Dict[str, Any], context: PipelineContext) -> Optional[Dict[str, Any]]:
        """
        Evaluate the quality of extracted data.
        
        Args:
            data: Input data (not used, we get data from context)
            context: Pipeline context containing extracted data
            
        Returns:
            Dictionary containing quality evaluation results or None if evaluation fails
        """
        try:
            if not self.evaluator:
                self.evaluator = QualityEvaluatorImpl()
                evaluator_config = {
                    "quality_threshold": self.quality_threshold,
                    "required_fields": self.required_fields,
                    "field_weights": self.field_weights,
                    "check_consistency": self.check_consistency,
                    "check_completeness": self.check_completeness,
                    "check_correctness": self.check_correctness,
                    "flag_issues": self.flag_issues
                }
                self.evaluator.initialize(evaluator_config)
            
            # Get extracted data from context
            extracted_data = context.get(self.input_key)
            
            # Get content type and schema if available
            content_type = context.get("content_type", "unknown")
            schema = context.get(self.schema_key, None)
            
            # Update required fields based on content type if not already defined
            if not self.required_fields and content_type != "unknown":
                required_fields = self._get_default_required_fields(content_type)
                self.evaluator.config["required_fields"] = required_fields
            
            # Create evaluation options
            options = {
                "content_type": content_type,
                "schema": schema,
                "extraction_hints": context.get("extraction_hints", {}),
                "extraction_method": extracted_data.get("_metadata", {}).get("extraction_method", "unknown")
            }
            
            # Evaluate data quality
            logger.info(f"Evaluating quality of extracted data for content type: {content_type}")
            evaluation_result = self.evaluator.evaluate(extracted_data, options)
            
            # Store reference to original data
            evaluation_result["data"] = extracted_data
            
            # Add quality threshold for reference
            evaluation_result["_metadata"] = evaluation_result.get("_metadata", {})
            evaluation_result["_metadata"]["quality_threshold"] = self.quality_threshold
            
            # Determine if data passes quality threshold
            overall_score = evaluation_result.get("overall_score", 0)
            passes_threshold = overall_score >= self.quality_threshold
            
            evaluation_result["passes_threshold"] = passes_threshold
            
            # Mark whether we have all required fields
            missing_required = evaluation_result.get("missing_required_fields", [])
            evaluation_result["has_all_required"] = len(missing_required) == 0
            
            # Set quality indicators in context for downstream stages
            context.set("quality_score", overall_score)
            context.set("passes_quality_threshold", passes_threshold)
            context.set("missing_required_fields", missing_required)
            
            # Log quality results
            if passes_threshold:
                logger.info(f"Data passes quality threshold: {overall_score:.2f} >= {self.quality_threshold}")
            else:
                logger.warning(f"Data below quality threshold: {overall_score:.2f} < {self.quality_threshold}")
                if missing_required:
                    logger.warning(f"Missing required fields: {', '.join(missing_required)}")
            
            # Add enhancement recommendations
            if self.flag_issues:
                recommendations = self._generate_recommendations(evaluation_result)
                if recommendations:
                    evaluation_result["recommendations"] = recommendations
                    logger.info(f"Generated {len(recommendations)} improvement recommendations")
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Error in quality evaluation: {str(e)}")
            context.add_error(self.name, f"Evaluation error: {str(e)}")
            
            # Create minimal result to avoid breaking the pipeline
            return {
                "success": False,
                "error": str(e),
                "overall_score": 0.0,
                "passes_threshold": False,
                "has_all_required": False,
                "_metadata": {
                    "evaluator": self.__class__.__name__,
                    "quality_threshold": self.quality_threshold
                }
            }
    
    def _get_default_required_fields(self, content_type: str) -> Dict[str, List[str]]:
        """
        Get default required fields based on content type.
        
        Args:
            content_type: Type of content (product, article, listing, etc.)
            
        Returns:
            Dictionary mapping content types to lists of required fields
        """
        default_required = {
            "product": [
                "title", "price"
            ],
            "article": [
                "title", "content"
            ],
            "listing": [
                "items"
            ],
            "search_results": [
                "items"
            ]
        }
        
        # Add additional recommended fields
        default_recommended = {
            "product": [
                "description", "images", "specifications"
            ],
            "article": [
                "date_published", "author"
            ],
            "listing": [
                "pagination", "total_items"
            ],
            "search_results": [
                "pagination", "total_results"
            ]
        }
        
        return {
            "required": default_required.get(content_type, []),
            "recommended": default_recommended.get(content_type, [])
        }
    
    def _generate_recommendations(self, evaluation_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate actionable recommendations to improve extraction quality.
        
        Args:
            evaluation_result: Quality evaluation results
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Get base data
        missing_required = evaluation_result.get("missing_required_fields", [])
        low_quality_fields = evaluation_result.get("low_quality_fields", {})
        issues = evaluation_result.get("issues", [])
        extracted_data = evaluation_result.get("data", {})
        
        # Recommend adding missing required fields
        for field in missing_required:
            recommendations.append({
                "type": "missing_field",
                "field": field,
                "priority": "high",
                "message": f"Required field '{field}' is missing"
            })
        
        # Recommend improving low quality fields
        for field, details in low_quality_fields.items():
            score = details.get("score", 0)
            issue = details.get("issue", "")
            
            if score < 0.5:
                priority = "high"
            elif score < 0.7:
                priority = "medium"
            else:
                priority = "low"
                
            recommendations.append({
                "type": "quality_issue",
                "field": field,
                "priority": priority,
                "score": score,
                "issue": issue,
                "message": f"Field '{field}' has quality issues: {issue}"
            })
        
        # Add recommendations for specific issues
        for issue in issues:
            issue_type = issue.get("type", "")
            field = issue.get("field", "")
            message = issue.get("message", "")
            
            recommendations.append({
                "type": issue_type,
                "field": field,
                "priority": issue.get("priority", "medium"),
                "message": message
            })
        
        # Add schema-specific recommendations
        if "field_completeness" in evaluation_result:
            completeness = evaluation_result["field_completeness"]
            if completeness < 0.7:
                # Extract schema fields that are missing
                data_keys = set(extracted_data.keys())
                schema = evaluation_result.get("_metadata", {}).get("schema", {})
                schema_fields = set(schema.get("properties", {}).keys()) if schema else set()
                
                missing_schema_fields = schema_fields - data_keys
                if missing_schema_fields:
                    recommendations.append({
                        "type": "schema_completeness",
                        "priority": "medium",
                        "missing_fields": list(missing_schema_fields),
                        "message": f"Data is missing {len(missing_schema_fields)} fields defined in schema"
                    })
        
        # Check for potential data quality issues
        has_issues = self._check_for_common_quality_issues(extracted_data)
        recommendations.extend(has_issues)
        
        return recommendations
    
    def _check_for_common_quality_issues(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check for common data quality issues.
        
        Args:
            data: Extracted data
            
        Returns:
            List of identified issues
        """
        issues = []
        
        # Check for truncated text fields
        for field in ["title", "description", "content"]:
            if field in data and isinstance(data[field], str):
                value = data[field]
                if value.endswith("...") or value.endswith("…"):
                    issues.append({
                        "type": "truncated_text",
                        "field": field,
                        "priority": "medium",
                        "message": f"Field '{field}' appears to be truncated"
                    })
        
        # Check for potential HTML in text fields
        for field in ["title", "description", "content"]:
            if field in data and isinstance(data[field], str):
                value = data[field]
                if re.search(r'<\w+[^>]*>', value) or re.search(r'&[a-z]+;', value):
                    issues.append({
                        "type": "html_content",
                        "field": field,
                        "priority": "medium",
                        "message": f"Field '{field}' may contain HTML that should be cleaned"
                    })
        
        # Check for empty lists
        for field in ["images", "variants", "specifications", "tags", "items"]:
            if field in data and isinstance(data[field], list) and not data[field]:
                issues.append({
                    "type": "empty_list",
                    "field": field,
                    "priority": "low",
                    "message": f"Field '{field}' is an empty list"
                })
        
        # Check for suspiciously short content
        if "content" in data and isinstance(data["content"], str) and len(data["content"]) < 100:
            issues.append({
                "type": "short_content",
                "field": "content",
                "priority": "high",
                "message": "Content field is suspiciously short, may indicate extraction failure"
            })
        
        # Check for price inconsistencies
        if "price" in data and "price_value" in data:
            price_str = str(data["price"])
            price_value = data["price_value"]
            
            # Check if the value seems too low for what would be expected
            if price_value < 0.1 and not re.search(r'free|0[.,]0', price_str.lower()):
                issues.append({
                    "type": "price_inconsistency",
                    "field": "price",
                    "priority": "high",
                    "message": "Price value seems incorrectly parsed or unnaturally low"
                })
        
        # Check for items issues in listings
        if "items" in data and isinstance(data["items"], list):
            items = data["items"]
            
            # Check if all items have the same value for a field (might indicate extraction error)
            if len(items) > 1:
                for field in ["title", "price"]:
                    if all(item.get(field) == items[0].get(field) for item in items if field in item):
                        issues.append({
                            "type": "repeated_values",
                            "field": f"items.{field}",
                            "priority": "high",
                            "message": f"All items have identical '{field}' values, possible extraction error"
                        })
        
        return issues