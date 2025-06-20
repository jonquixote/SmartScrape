import unittest
import asyncio
from unittest.mock import MagicMock, patch

from core.pipeline.templates.transformation_pipeline import TransformationPipeline
from core.pipeline.context import PipelineContext
from core.pipeline.stage import PipelineStage


class MockTransformationStage(PipelineStage):
    """Mock transformation stage for testing."""
    
    async def process(self, context: PipelineContext) -> bool:
        # Get original data
        data = context.get("data", {})
        
        # Simulate transformation based on stage type
        stage_type = self.config.get("stage_type", "generic")
        
        if stage_type == "normalization":
            # Normalize text by trimming whitespace
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, str):
                        data[key] = value.strip()
            
            # Add normalized data to context
            context.set("data", data)
            context.set("normalized_data", data)
        
        elif stage_type == "enrichment":
            # Add additional data from enrichment source
            enrichment_source = self.config.get("source", {})
            
            # Merge enrichment data with original data
            if isinstance(data, dict) and isinstance(enrichment_source, dict):
                for key, value in enrichment_source.items():
                    if key not in data:
                        data[key] = value
            
            # Add enriched data to context
            context.set("data", data)
            context.set("enriched_data", data)
        
        elif stage_type == "restructuring":
            # Apply mapping rules to restructure data
            mapping_rules = self.config.get("mapping_rules", {})
            restructured = {}
            
            if isinstance(data, dict) and mapping_rules:
                for new_key, original_key in mapping_rules.items():
                    if original_key in data:
                        restructured[new_key] = data[original_key]
            
            # Add restructured data to context
            context.set("data", restructured)
            context.set("restructured_data", restructured)
        
        elif stage_type == "aggregation":
            # Apply aggregation rules
            group_by = self.config.get("group_by", [])
            aggregated = {}
            
            # Very simplified grouping for testing
            if isinstance(data, list) and group_by:
                group_field = group_by[0]
                for item in data:
                    if group_field in item:
                        group_value = item[group_field]
                        if group_value not in aggregated:
                            aggregated[group_value] = []
                        aggregated[group_value].append(item)
            
            # Add aggregated data to context
            context.set("data", aggregated)
            context.set("aggregated_data", aggregated)
        
        # Track transformations if configured
        if self.config.get("track_changes", False):
            self._track_changes(context, data)
        
        return True
    
    def _track_changes(self, context: PipelineContext, transformed_data: dict) -> None:
        """Track changes for testing purposes."""
        original_data = context.get("original_data", {})
        
        if not original_data or not transformed_data:
            return
            
        changes = []
        
        # Track field-level changes for dictionaries
        if isinstance(original_data, dict) and isinstance(transformed_data, dict):
            for key, value in transformed_data.items():
                if key in original_data:
                    if original_data[key] != value:
                        changes.append({
                            "field": key,
                            "operation": "modified",
                            "original": original_data[key],
                            "transformed": value
                        })
                else:
                    changes.append({
                        "field": key,
                        "operation": "added",
                        "transformed": value
                    })
        
        context.set("transformation_changes", changes)


class TestTransformationPipeline(unittest.TestCase):
    """Test cases for the TransformationPipeline class and its factory methods."""
    
    def setUp(self):
        """Set up test environment."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        # Sample test data
        self.test_data = {
            "data": {
                "name": "  John Doe  ",  # Has extra whitespace
                "age": 30,
                "email": "john@example.com"
            }
        }
        
        # Sample mapping rules for restructuring
        self.mapping_rules = {
            "fullName": "name",
            "userAge": "age",
            "contactEmail": "email"
        }
        
        # Sample enrichment sources
        self.enrichment_sources = [
            {
                "source": {
                    "location": "New York",
                    "interests": ["coding", "reading"]
                },
                "type": "profile",
                "fields": ["location", "interests"]
            }
        ]
        
        # Sample aggregation data
        self.aggregation_data = {
            "data": [
                {"category": "electronics", "item": "laptop", "price": 1200},
                {"category": "electronics", "item": "phone", "price": 800},
                {"category": "books", "item": "novel", "price": 15},
                {"category": "books", "item": "textbook", "price": 50}
            ]
        }
    
    def tearDown(self):
        """Clean up after tests."""
        self.loop.close()
    
    def test_pipeline_initialization(self):
        """Test basic pipeline initialization."""
        pipeline = TransformationPipeline("test_pipeline")
        
        self.assertEqual(pipeline.name, "test_pipeline")
        self.assertTrue(isinstance(pipeline.transformation_config, dict))
        self.assertTrue(pipeline.transformation_config["preserve_original"])
        self.assertTrue(pipeline.transformation_config["track_changes"])
        
        # Test with custom config
        custom_config = {
            "transformation_config": {
                "preserve_original": False,
                "track_changes": False,
                "validate_after_transform": False
            }
        }
        pipeline = TransformationPipeline("custom_pipeline", custom_config)
        self.assertFalse(pipeline.transformation_config["preserve_original"])
        self.assertFalse(pipeline.transformation_config["track_changes"])
        self.assertFalse(pipeline.transformation_config["validate_after_transform"])
    
    def test_track_transformations(self):
        """Test the transformation tracking functionality."""
        pipeline = TransformationPipeline("test_tracking")
        
        # Set up context with original and transformed data
        context = PipelineContext({
            "original_data": {
                "name": "John Doe",
                "age": 30
            },
            "data": {
                "name": "John M. Doe",  # Modified
                "age": 30,  # Unchanged
                "location": "New York"  # Added
            }
        })
        
        # Track transformations
        pipeline._track_transformations(context)
        
        # Verify tracked changes
        changes = context.get("transformation_changes")
        self.assertIsNotNone(changes)
        self.assertEqual(len(changes), 2)  # One modified, one added
        
        # Check for modified field
        modified = [c for c in changes if c["operation"] == "modified"]
        self.assertEqual(len(modified), 1)
        self.assertEqual(modified[0]["field"], "name")
        self.assertEqual(modified[0]["original"], "John Doe")
        self.assertEqual(modified[0]["transformed"], "John M. Doe")
        
        # Check for added field
        added = [c for c in changes if c["operation"] == "added"]
        self.assertEqual(len(added), 1)
        self.assertEqual(added[0]["field"], "location")
        self.assertEqual(added[0]["transformed"], "New York")
    
    def test_pipeline_execution_with_mock_stages(self):
        """Test transformation pipeline execution with mock stages."""
        pipeline = TransformationPipeline("test_transformation_execution")
        
        # Add mock transformation stage
        pipeline.add_stage(MockTransformationStage({
            "stage_type": "normalization",
            "track_changes": True
        }))
        
        # Execute the pipeline
        context = self.loop.run_until_complete(pipeline.execute(self.test_data))
        
        # Check results
        self.assertFalse(context.has_errors())
        self.assertEqual(context.get("data")["name"], "John Doe")  # Whitespace removed
        self.assertIsNotNone(context.get("normalized_data"))
    
    @patch('core.pipeline.stages.processing.content_normalization.DataNormalizationStage')
    def test_create_normalization_pipeline(self, mock_normalization_stage):
        """Test factory method for normalization pipeline."""
        # Configure mock
        mock_normalization_stage.return_value = MockTransformationStage({
            "stage_type": "normalization",
            "track_changes": True
        })
        
        # Create normalization pipeline
        pipeline = TransformationPipeline.create_normalization_pipeline({
            "normalize_text": True,
            "normalize_dates": True,
            "normalize_urls": False
        })
        
        # Verify pipeline configuration
        self.assertEqual(pipeline.name, "normalization_pipeline")
        self.assertEqual(pipeline.transformation_config["transformation_type"], "normalization")
        self.assertTrue(pipeline.transformation_config["normalize_text"])
        self.assertTrue(pipeline.transformation_config["normalize_dates"])
        self.assertFalse(pipeline.transformation_config["normalize_urls"])
        
        # Execute the pipeline
        context = self.loop.run_until_complete(pipeline.execute(self.test_data))
        
        # Verify pipeline execution
        self.assertFalse(context.has_errors())
        self.assertIsNotNone(context.get("normalized_data"))
    
    @patch('core.pipeline.stages.processing.content_normalization.DataNormalizationStage')
    @patch('core.pipeline.stages.processing.content_extraction.StructuredDataExtractionStage')
    def test_create_enrichment_pipeline(self, mock_extraction_stage, mock_normalization_stage):
        """Test factory method for enrichment pipeline."""
        # Configure mocks
        mock_normalization_stage.return_value = MockTransformationStage({
            "stage_type": "normalization",
            "track_changes": True
        })
        mock_extraction_stage.return_value = MockTransformationStage({
            "stage_type": "enrichment",
            "source": self.enrichment_sources[0]["source"],
            "track_changes": True
        })
        
        # Create enrichment pipeline
        pipeline = TransformationPipeline.create_enrichment_pipeline(
            self.enrichment_sources,
            {"merge_strategy": "overlay", "conflict_resolution": "prefer_new"}
        )
        
        # Verify pipeline configuration
        self.assertEqual(pipeline.name, "enrichment_pipeline")
        self.assertEqual(pipeline.transformation_config["transformation_type"], "enrichment")
        self.assertEqual(pipeline.transformation_config["enrichment_sources"], self.enrichment_sources)
        self.assertEqual(pipeline.transformation_config["merge_strategy"], "overlay")
        self.assertEqual(pipeline.transformation_config["conflict_resolution"], "prefer_new")
        
        # Execute the pipeline
        context = self.loop.run_until_complete(pipeline.execute(self.test_data))
        
        # Verify pipeline execution
        self.assertFalse(context.has_errors())
        self.assertIsNotNone(context.get("enriched_data"))
    
    @patch('core.pipeline.stages.processing.content_extraction.StructuredDataExtractionStage')
    @patch('core.pipeline.stages.processing.content_normalization.DataNormalizationStage')
    def test_create_restructuring_pipeline(self, mock_normalization_stage, mock_extraction_stage):
        """Test factory method for restructuring pipeline."""
        # Configure mocks
        mock_extraction_stage.return_value = MockTransformationStage({
            "stage_type": "restructuring",
            "mapping_rules": self.mapping_rules,
            "track_changes": True
        })
        mock_normalization_stage.return_value = MockTransformationStage({
            "stage_type": "normalization",
            "track_changes": True
        })
        
        # Create restructuring pipeline
        pipeline = TransformationPipeline.create_restructuring_pipeline(
            self.mapping_rules,
            {"include_unmapped_fields": False, "structure_type": "flat"}
        )
        
        # Verify pipeline configuration
        self.assertEqual(pipeline.name, "restructuring_pipeline")
        self.assertEqual(pipeline.transformation_config["transformation_type"], "restructuring")
        self.assertEqual(pipeline.transformation_config["mapping_rules"], self.mapping_rules)
        self.assertFalse(pipeline.transformation_config["include_unmapped_fields"])
        self.assertEqual(pipeline.transformation_config["structure_type"], "flat")
        
        # Execute the pipeline
        context = self.loop.run_until_complete(pipeline.execute(self.test_data))
        
        # Verify pipeline execution
        self.assertFalse(context.has_errors())
        self.assertIsNotNone(context.get("restructured_data"))
    
    @patch('core.pipeline.stages.processing.content_extraction.StructuredDataExtractionStage')
    def test_create_aggregation_pipeline(self, mock_extraction_stage):
        """Test factory method for aggregation pipeline."""
        # Configure mock
        mock_extraction_stage.return_value = MockTransformationStage({
            "stage_type": "aggregation",
            "group_by": ["category"],
            "track_changes": True
        })
        
        # Create aggregation pipeline
        aggregation_rules = {
            "group_by": "category",
            "metrics": ["count", "sum"]
        }
        
        pipeline = TransformationPipeline.create_aggregation_pipeline(
            aggregation_rules,
            {"group_by": ["category"], "sort_by": ["count"]}
        )
        
        # Verify pipeline configuration
        self.assertEqual(pipeline.name, "aggregation_pipeline")
        self.assertEqual(pipeline.transformation_config["transformation_type"], "aggregation")
        self.assertEqual(pipeline.transformation_config["aggregation_rules"], aggregation_rules)
        self.assertEqual(pipeline.transformation_config["group_by"], ["category"])
        self.assertEqual(pipeline.transformation_config["sort_by"], ["count"])
        
        # Execute the pipeline
        context = self.loop.run_until_complete(pipeline.execute(self.aggregation_data))
        
        # Verify pipeline execution
        self.assertFalse(context.has_errors())
        self.assertIsNotNone(context.get("aggregated_data"))


if __name__ == '__main__':
    unittest.main()