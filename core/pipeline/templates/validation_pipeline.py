import logging
from typing import Any, Dict, List, Optional, Union

from core.pipeline.pipeline import Pipeline
from core.pipeline.stage import PipelineStage
from core.pipeline.context import PipelineContext
from core.service_registry import ServiceRegistry

# Import stages (assuming they exist based on batch4_pipeline_architecture.md)
from core.pipeline.stages.processing.content_validation import DataValidationStage
from core.pipeline.stages.output.json_output import JSONOutputStage


class ValidationPipeline(Pipeline):
    """Pre-configured pipeline for data validation workflows.
    
    This pipeline template provides specialized configurations for validating data
    against schemas, checking data quality, and ensuring data consistency.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the validation pipeline with a name and configuration.
        
        Args:
            name: Unique name for this pipeline
            config: Pipeline configuration with validation-specific settings
        """
        super().__init__(name, config)
        self.logger = logging.getLogger(f"validation_pipeline.{name}")
        
        # Validation-specific configuration defaults
        self.validation_config = {
            "stop_on_first_error": False,
            "generate_report": True,
            "auto_correct": False,
            "correction_level": "safe",  # safe, moderate, aggressive
            "validate_schema": True,
            "validate_quality": True,
            "validate_consistency": True,
            "validation_threshold": 0.8,  # minimum score to pass validation
            **self.config.get("validation_config", {})
        }
        
        # Initialize services if needed
        self.service_registry = ServiceRegistry()
        
    async def execute(self, initial_data: Optional[Dict[str, Any]] = None) -> PipelineContext:
        """Execute the validation pipeline with specialized error handling.
        
        Args:
            initial_data: Initial data to populate the context
            
        Returns:
            The final pipeline context with validation results
        """
        context = await super().execute(initial_data)
        
        # Generate validation report if configured
        if self.validation_config.get("generate_report", True):
            self._generate_validation_report(context)
            
        return context
    
    def _generate_validation_report(self, context: PipelineContext) -> None:
        """Generate a comprehensive validation report.
        
        Args:
            context: The pipeline context with validation results
        """
        # Collect validation results from all stages
        validation_results = {}
        for stage_name, metrics in context.metadata["stage_metrics"].items():
            if "validation_results" in metrics:
                validation_results[stage_name] = metrics["validation_results"]
        
        # Calculate overall validation score
        total_checks = 0
        passed_checks = 0
        errors = []
        warnings = []
        
        for stage, results in validation_results.items():
            if "checks" in results:
                total_checks += results["total_checks"]
                passed_checks += results["passed_checks"]
            
            if "errors" in results:
                for error in results["errors"]:
                    errors.append({
                        "stage": stage,
                        "message": error["message"],
                        "path": error.get("path", ""),
                        "severity": error.get("severity", "error")
                    })
            
            if "warnings" in results:
                for warning in results["warnings"]:
                    warnings.append({
                        "stage": stage,
                        "message": warning["message"],
                        "path": warning.get("path", ""),
                        "severity": "warning"
                    })
        
        # Calculate overall score
        validation_score = (passed_checks / total_checks) if total_checks > 0 else 0
        
        # Create validation report
        validation_report = {
            "validation_score": validation_score,
            "passed": validation_score >= self.validation_config.get("validation_threshold", 0.8),
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "failed_checks": total_checks - passed_checks,
            "errors": errors,
            "warnings": warnings,
            "corrective_actions": context.get("corrective_actions", [])
        }
        
        context.set("validation_report", validation_report)
    
    def auto_correct_data(self, context: PipelineContext) -> bool:
        """Attempt to automatically correct validation issues.
        
        Args:
            context: The pipeline context with validation issues
            
        Returns:
            True if corrections were applied, False otherwise
        """
        if not self.validation_config.get("auto_correct", False):
            return False
            
        correction_level = self.validation_config.get("correction_level", "safe")
        data = context.get("data")
        report = context.get("validation_report", {})
        
        if not data or not report:
            return False
            
        corrective_actions = []
        
        # Apply corrections based on correction level
        if correction_level == "safe":
            # Only apply safe corrections (whitespace, casing, etc.)
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, str):
                        # Trim whitespace
                        data[key] = value.strip()
                        if value != data[key]:
                            corrective_actions.append({
                                "path": key,
                                "action": "trim_whitespace",
                                "original": value,
                                "corrected": data[key]
                            })
        
        elif correction_level == "moderate":
            # Apply safe corrections + data type conversions, format standardization
            # More complex correction logic would go here
            pass
            
        elif correction_level == "aggressive":
            # Apply all corrections + field inferencing, advanced data transformations
            # Most aggressive correction logic would go here
            pass
            
        # Record corrective actions taken
        if corrective_actions:
            context.set("corrective_actions", corrective_actions)
            context.set("data", data)  # Update the data with corrections
            return True
            
        return False
    
    @classmethod
    def create_schema_validation_pipeline(cls, schema: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> 'ValidationPipeline':
        """Create a pipeline configured for schema validation.
        
        Args:
            schema: The JSON schema to validate against
            config: Additional configuration options
            
        Returns:
            Configured ValidationPipeline instance
        """
        name = f"schema_validation_{schema.get('title', 'custom').lower().replace(' ', '_')}"
        pipeline_config = {
            "validation_config": {
                "validation_type": "schema",
                "schema": schema,
                **config.get("validation_config", {}) if config else {}
            },
            **config if config else {}
        }
        
        pipeline = cls(name, pipeline_config)
        
        # Add schema validation stages
        pipeline.add_stages([
            DataValidationStage({
                "validate_schema": True,
                "schema": schema,
                "fail_on_extra_properties": pipeline_config["validation_config"].get("fail_on_extra_properties", False),
                "fail_on_missing_required": pipeline_config["validation_config"].get("fail_on_missing_required", True)
            }),
            JSONOutputStage({
                "format": "json",
                "pretty_print": True,
                "output_field": "validation_results"
            })
        ])
        
        return pipeline
    
    @classmethod
    def create_quality_validation_pipeline(cls, config: Optional[Dict[str, Any]] = None) -> 'ValidationPipeline':
        """Create a pipeline configured for data quality assessment.
        
        Args:
            config: Configuration options
            
        Returns:
            Configured ValidationPipeline instance
        """
        name = "quality_validation"
        pipeline_config = {
            "validation_config": {
                "validation_type": "quality",
                "completeness_threshold": config.get("completeness_threshold", 0.8) if config else 0.8,
                "consistency_threshold": config.get("consistency_threshold", 0.9) if config else 0.9,
                "accuracy_checks": config.get("accuracy_checks", True) if config else True,
                **config.get("validation_config", {}) if config else {}
            },
            **config if config else {}
        }
        
        pipeline = cls(name, pipeline_config)
        
        # Add quality validation stages
        pipeline.add_stages([
            DataValidationStage({
                "validate_quality": True,
                "check_completeness": True,
                "check_consistency": True,
                "check_accuracy": pipeline_config["validation_config"].get("accuracy_checks", True),
                "completeness_threshold": pipeline_config["validation_config"]["completeness_threshold"],
                "consistency_threshold": pipeline_config["validation_config"]["consistency_threshold"],
                "required_fields": pipeline_config["validation_config"].get("required_fields", [])
            }),
            JSONOutputStage({
                "format": "json",
                "pretty_print": True,
                "output_field": "validation_results"
            })
        ])
        
        return pipeline
    
    @classmethod
    def create_consistency_validation_pipeline(cls, config: Optional[Dict[str, Any]] = None) -> 'ValidationPipeline':
        """Create a pipeline configured for data consistency validation.
        
        Args:
            config: Configuration options
            
        Returns:
            Configured ValidationPipeline instance
        """
        name = "consistency_validation"
        pipeline_config = {
            "validation_config": {
                "validation_type": "consistency",
                "reference_data": config.get("reference_data") if config else None,
                "reference_fields": config.get("reference_fields", []) if config else [],
                "check_relational_integrity": config.get("check_relational_integrity", True) if config else True,
                **config.get("validation_config", {}) if config else {}
            },
            **config if config else {}
        }
        
        pipeline = cls(name, pipeline_config)
        
        # Add consistency validation stages
        pipeline.add_stages([
            DataValidationStage({
                "validate_consistency": True,
                "reference_data": pipeline_config["validation_config"].get("reference_data"),
                "reference_fields": pipeline_config["validation_config"].get("reference_fields", []),
                "check_relational_integrity": pipeline_config["validation_config"].get("check_relational_integrity", True),
                "field_format_rules": pipeline_config["validation_config"].get("field_format_rules", {})
            }),
            JSONOutputStage({
                "format": "json",
                "pretty_print": True,
                "output_field": "validation_results"
            })
        ])
        
        return pipeline