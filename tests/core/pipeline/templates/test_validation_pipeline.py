import unittest
import asyncio
from unittest.mock import MagicMock, patch

from core.pipeline.templates.validation_pipeline import ValidationPipeline
from core.pipeline.context import PipelineContext
from core.pipeline.stage import PipelineStage


class MockValidationStage(PipelineStage):
    """Mock validation stage for testing."""
    
    async def process(self, context: PipelineContext) -> bool:
        data = context.get("data", {})
        validation_results = {
            "total_checks": 5,
            "passed_checks": 4,
            "checks": [
                {"field": "name", "status": "pass", "message": "Field exists"},
                {"field": "age", "status": "pass", "message": "Field exists"},
                {"field": "email", "status": "pass", "message": "Field exists"},
                {"field": "url", "status": "pass", "message": "Field exists"},
                {"field": "phone", "status": "fail", "message": "Field missing"}
            ],
            "errors": [
                {"message": "Required field 'phone' is missing", "path": "phone", "severity": "error"}
            ],
            "warnings": [
                {"message": "Email format is unusual", "path": "email", "severity": "warning"}
            ]
        }
        
        # Add validation results to both context and stage metrics
        context.set("validation_results", validation_results)
        
        # Add to stage metrics (normally handled by the Pipeline class)
        if "stage_metrics" not in context.metadata:
            context.metadata["stage_metrics"] = {}
        
        context.metadata["stage_metrics"][self.name] = {
            "validation_results": validation_results
        }
        
        return self.config.get("should_pass", True)


class TestValidationPipeline(unittest.TestCase):
    """Test cases for the ValidationPipeline class and its factory methods."""
    
    def setUp(self):
        """Set up test environment."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        # Sample test data
        self.test_data = {
            "data": {
                "name": "John Doe",
                "age": 30,
                "email": "john@example",
                "url": "http://example.com"
            }
        }
        
        # Sample JSON schema
        self.test_schema = {
            "title": "Person",
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "email": {"type": "string"},
                "url": {"type": "string"},
                "phone": {"type": "string"}
            },
            "required": ["name", "age", "email"]
        }
    
    def tearDown(self):
        """Clean up after tests."""
        self.loop.close()
    
    def test_pipeline_initialization(self):
        """Test basic pipeline initialization."""
        pipeline = ValidationPipeline("test_pipeline")
        
        self.assertEqual(pipeline.name, "test_pipeline")
        self.assertTrue(isinstance(pipeline.validation_config, dict))
        self.assertFalse(pipeline.validation_config["stop_on_first_error"])
        self.assertTrue(pipeline.validation_config["generate_report"])
        
        # Test with custom config
        custom_config = {
            "validation_config": {
                "stop_on_first_error": True,
                "generate_report": False,
                "auto_correct": True
            }
        }
        pipeline = ValidationPipeline("custom_pipeline", custom_config)
        self.assertTrue(pipeline.validation_config["stop_on_first_error"])
        self.assertFalse(pipeline.validation_config["generate_report"])
        self.assertTrue(pipeline.validation_config["auto_correct"])
    
    def test_generate_validation_report(self):
        """Test the validation report generation."""
        pipeline = ValidationPipeline("test_report")
        
        # Set up context with mock validation results
        context = PipelineContext()
        context.metadata["stage_metrics"] = {
            "validation_stage_1": {
                "validation_results": {
                    "total_checks": 3,
                    "passed_checks": 2,
                    "errors": [{"message": "Error 1", "path": "field1"}],
                    "warnings": []
                }
            },
            "validation_stage_2": {
                "validation_results": {
                    "total_checks": 2,
                    "passed_checks": 2,
                    "errors": [],
                    "warnings": [{"message": "Warning 1", "path": "field2"}]
                }
            }
        }
        
        # Generate report
        pipeline._generate_validation_report(context)
        
        # Verify report contents
        report = context.get("validation_report")
        self.assertIsNotNone(report)
        self.assertEqual(report["total_checks"], 5)
        self.assertEqual(report["passed_checks"], 4)
        self.assertEqual(report["failed_checks"], 1)
        self.assertEqual(len(report["errors"]), 1)
        self.assertEqual(len(report["warnings"]), 1)
        self.assertGreaterEqual(report["validation_score"], 0.8)  # 4/5 = 0.8
        self.assertTrue(report["passed"])  # Default threshold is 0.8
    
    def test_auto_correct_data(self):
        """Test the auto-correction functionality."""
        pipeline = ValidationPipeline("test_correction")
        pipeline.validation_config["auto_correct"] = True
        pipeline.validation_config["correction_level"] = "safe"
        
        # Context with data that needs correction
        context = PipelineContext({
            "data": {
                "name": "  John Doe  ",  # Extra whitespace
                "age": 30,
                "email": "john@example.com"
            },
            "validation_report": {
                "errors": [{"path": "name", "message": "Extra whitespace"}]
            }
        })
        
        # Apply corrections
        result = pipeline.auto_correct_data(context)
        
        # Verify corrections
        self.assertTrue(result)
        self.assertEqual(context.get("data")["name"], "John Doe")  # Whitespace trimmed
        self.assertTrue("corrective_actions" in context.data)
        self.assertEqual(len(context.get("corrective_actions")), 1)
    
    @patch('core.pipeline.stages.processing.content_validation.DataValidationStage')
    def test_create_schema_validation_pipeline(self, mock_validation_stage):
        """Test factory method for schema validation pipeline."""
        # Configure mock
        mock_validation_stage.return_value = MockValidationStage({"should_pass": True})
        
        # Create schema validation pipeline
        pipeline = ValidationPipeline.create_schema_validation_pipeline(self.test_schema)
        
        # Verify pipeline configuration
        self.assertIn("schema_validation_person", pipeline.name.lower())
        self.assertEqual(pipeline.validation_config["validation_type"], "schema")
        self.assertEqual(pipeline.validation_config["schema"], self.test_schema)
        
        # Execute the pipeline
        context = self.loop.run_until_complete(pipeline.execute(self.test_data))
        
        # Verify pipeline execution
        self.assertFalse(context.has_errors())
        self.assertIsNotNone(context.get("validation_report"))
    
    @patch('core.pipeline.stages.processing.content_validation.DataValidationStage')
    def test_create_quality_validation_pipeline(self, mock_validation_stage):
        """Test factory method for quality validation pipeline."""
        # Configure mock
        mock_validation_stage.return_value = MockValidationStage({"should_pass": True})
        
        # Create quality validation pipeline
        pipeline = ValidationPipeline.create_quality_validation_pipeline({
            "completeness_threshold": 0.9,
            "consistency_threshold": 0.8,
            "accuracy_checks": True
        })
        
        # Verify pipeline configuration
        self.assertEqual(pipeline.name, "quality_validation")
        self.assertEqual(pipeline.validation_config["validation_type"], "quality")
        self.assertEqual(pipeline.validation_config["completeness_threshold"], 0.9)
        self.assertEqual(pipeline.validation_config["consistency_threshold"], 0.8)
        self.assertTrue(pipeline.validation_config["accuracy_checks"])
        
        # Execute the pipeline
        context = self.loop.run_until_complete(pipeline.execute(self.test_data))
        
        # Verify pipeline execution
        self.assertFalse(context.has_errors())
        self.assertIsNotNone(context.get("validation_report"))
    
    @patch('core.pipeline.stages.processing.content_validation.DataValidationStage')
    def test_create_consistency_validation_pipeline(self, mock_validation_stage):
        """Test factory method for consistency validation pipeline."""
        # Configure mock
        mock_validation_stage.return_value = MockValidationStage({"should_pass": True})
        
        # Reference data for consistency validation
        reference_data = {
            "john@example.com": {"name": "John Doe", "age": 30}
        }
        
        # Create consistency validation pipeline
        pipeline = ValidationPipeline.create_consistency_validation_pipeline({
            "reference_data": reference_data,
            "reference_fields": ["email"],
            "check_relational_integrity": True
        })
        
        # Verify pipeline configuration
        self.assertEqual(pipeline.name, "consistency_validation")
        self.assertEqual(pipeline.validation_config["validation_type"], "consistency")
        self.assertEqual(pipeline.validation_config["reference_data"], reference_data)
        self.assertEqual(pipeline.validation_config["reference_fields"], ["email"])
        self.assertTrue(pipeline.validation_config["check_relational_integrity"])
        
        # Execute the pipeline
        context = self.loop.run_until_complete(pipeline.execute(self.test_data))
        
        # Verify pipeline execution
        self.assertFalse(context.has_errors())
        self.assertIsNotNone(context.get("validation_report"))
    
    def test_pipeline_execution_with_mock_stages(self):
        """Test validation pipeline execution with mock stages."""
        pipeline = ValidationPipeline("test_validation_execution")
        
        # Add mock validation stage
        pipeline.add_stage(MockValidationStage({
            "validate_schema": True,
            "schema": self.test_schema
        }))
        
        # Execute the pipeline
        context = self.loop.run_until_complete(pipeline.execute(self.test_data))
        
        # Check results
        self.assertFalse(context.has_errors())
        self.assertIsNotNone(context.get("validation_results"))
        self.assertIsNotNone(context.get("validation_report"))
        
        # Check report details
        report = context.get("validation_report")
        self.assertEqual(report["total_checks"], 5)
        self.assertEqual(report["passed_checks"], 4)
        self.assertEqual(len(report["errors"]), 1)


if __name__ == '__main__':
    unittest.main()