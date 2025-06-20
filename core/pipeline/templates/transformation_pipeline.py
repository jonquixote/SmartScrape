import logging
from typing import Any, Dict, List, Optional, Union

from core.pipeline.pipeline import Pipeline
from core.pipeline.stage import PipelineStage
from core.pipeline.context import PipelineContext
from core.service_registry import ServiceRegistry

# Import stages (assuming they exist based on batch4_pipeline_architecture.md)
from core.pipeline.stages.processing.content_normalization import DataNormalizationStage
from core.pipeline.stages.processing.content_extraction import StructuredDataExtractionStage
from core.pipeline.stages.output.json_output import JSONOutputStage


class TransformationPipeline(Pipeline):
    """Pre-configured pipeline for data transformation workflows.
    
    This pipeline template provides specialized configurations for transforming data
    through normalization, enrichment, restructuring, and other operations.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the transformation pipeline with a name and configuration.
        
        Args:
            name: Unique name for this pipeline
            config: Pipeline configuration with transformation-specific settings
        """
        super().__init__(name, config)
        self.logger = logging.getLogger(f"transformation_pipeline.{name}")
        
        # Transformation-specific configuration defaults
        self.transformation_config = {
            "preserve_original": True,
            "track_changes": True,
            "validate_after_transform": True,
            "enrichment_sources": [],
            "transformation_rules": {},
            "include_metadata": True,
            "record_discarded_data": False,
            **self.config.get("transformation_config", {})
        }
        
        # Initialize services if needed
        self.service_registry = ServiceRegistry()
        
    async def execute(self, initial_data: Optional[Dict[str, Any]] = None) -> PipelineContext:
        """Execute the transformation pipeline with specialized tracking.
        
        Args:
            initial_data: Initial data to populate the context
            
        Returns:
            The final pipeline context with transformed data
        """
        # Store original data if configured
        if initial_data and self.transformation_config.get("preserve_original", True):
            initial_data["original_data"] = initial_data.get("data", {})
            
        context = await super().execute(initial_data)
        
        # Track transformation changes if configured
        if self.transformation_config.get("track_changes", True):
            self._track_transformations(context)
            
        return context
    
    def _track_transformations(self, context: PipelineContext) -> None:
        """Track and record transformations applied to the data.
        
        Args:
            context: The pipeline context with transformation results
        """
        original_data = context.get("original_data", {})
        transformed_data = context.get("data", {})
        
        if not original_data or not transformed_data:
            return
            
        changes = []
        discarded = []
        
        # For simple dictionaries, track field-level changes
        if isinstance(original_data, dict) and isinstance(transformed_data, dict):
            # Track modified or added fields
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
            
            # Track removed fields
            if self.transformation_config.get("record_discarded_data", False):
                for key, value in original_data.items():
                    if key not in transformed_data:
                        discarded.append({
                            "field": key,
                            "operation": "removed",
                            "original": value
                        })
        
        # More complex tracking for lists, nested structures, etc. would go here
        
        # Add transformation metadata to context
        context.set("transformation_changes", changes)
        
        if discarded:
            context.set("discarded_data", discarded)
    
    @classmethod
    def create_normalization_pipeline(cls, config: Optional[Dict[str, Any]] = None) -> 'TransformationPipeline':
        """Create a pipeline configured for data normalization.
        
        Args:
            config: Configuration options
            
        Returns:
            Configured TransformationPipeline instance
        """
        name = "normalization_pipeline"
        pipeline_config = {
            "transformation_config": {
                "transformation_type": "normalization",
                "normalize_text": config.get("normalize_text", True) if config else True,
                "normalize_dates": config.get("normalize_dates", True) if config else True,
                "normalize_numbers": config.get("normalize_numbers", True) if config else True,
                "normalize_urls": config.get("normalize_urls", True) if config else True,
                "date_format": config.get("date_format", "ISO8601") if config else "ISO8601",
                "number_format": config.get("number_format", "standard") if config else "standard",
                **config.get("transformation_config", {}) if config else {}
            },
            **config if config else {}
        }
        
        pipeline = cls(name, pipeline_config)
        
        # Add normalization stages
        pipeline.add_stages([
            DataNormalizationStage({
                "normalize_text": pipeline_config["transformation_config"].get("normalize_text", True),
                "normalize_dates": pipeline_config["transformation_config"].get("normalize_dates", True),
                "normalize_numbers": pipeline_config["transformation_config"].get("normalize_numbers", True),
                "normalize_urls": pipeline_config["transformation_config"].get("normalize_urls", True),
                "date_format": pipeline_config["transformation_config"].get("date_format", "ISO8601"),
                "number_format": pipeline_config["transformation_config"].get("number_format", "standard"),
                "track_changes": pipeline_config["transformation_config"].get("track_changes", True)
            }),
            JSONOutputStage({
                "format": "json",
                "pretty_print": True,
                "output_field": "normalized_data"
            })
        ])
        
        return pipeline
    
    @classmethod
    def create_enrichment_pipeline(cls, enrichment_sources: List[Dict[str, Any]], config: Optional[Dict[str, Any]] = None) -> 'TransformationPipeline':
        """Create a pipeline configured for data enrichment.
        
        Args:
            enrichment_sources: List of enrichment sources to use
            config: Additional configuration options
            
        Returns:
            Configured TransformationPipeline instance
        """
        name = "enrichment_pipeline"
        pipeline_config = {
            "transformation_config": {
                "transformation_type": "enrichment",
                "enrichment_sources": enrichment_sources,
                "merge_strategy": config.get("merge_strategy", "overlay") if config else "overlay",
                "conflict_resolution": config.get("conflict_resolution", "prefer_new") if config else "prefer_new",
                **config.get("transformation_config", {}) if config else {}
            },
            **config if config else {}
        }
        
        pipeline = cls(name, pipeline_config)
        
        # Add enrichment stages
        # First, a normalization stage to prepare the data
        pipeline.add_stage(DataNormalizationStage({
            "normalize_keys": True,
            "normalize_dates": True,
            "track_changes": True
        }))
        
        # Add an enrichment stage for each source
        for source in enrichment_sources:
            # Using StructuredDataExtractionStage as a placeholder for a hypothetical EnrichmentStage
            pipeline.add_stage(StructuredDataExtractionStage({
                "enrichment": True,
                "source": source.get("source"),
                "source_type": source.get("type"),
                "fields_to_enrich": source.get("fields", []),
                "mapping": source.get("mapping", {}),
                "merge_strategy": pipeline_config["transformation_config"].get("merge_strategy", "overlay"),
                "conflict_resolution": pipeline_config["transformation_config"].get("conflict_resolution", "prefer_new")
            }))
        
        # Final output stage
        pipeline.add_stage(JSONOutputStage({
            "format": "json",
            "pretty_print": True,
            "output_field": "enriched_data"
        }))
        
        return pipeline
    
    @classmethod
    def create_restructuring_pipeline(cls, mapping_rules: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> 'TransformationPipeline':
        """Create a pipeline configured for data restructuring.
        
        Args:
            mapping_rules: Rules for restructuring data fields
            config: Additional configuration options
            
        Returns:
            Configured TransformationPipeline instance
        """
        name = "restructuring_pipeline"
        pipeline_config = {
            "transformation_config": {
                "transformation_type": "restructuring",
                "mapping_rules": mapping_rules,
                "include_unmapped_fields": config.get("include_unmapped_fields", False) if config else False,
                "structure_type": config.get("structure_type", "flat") if config else "flat",  # flat, nested, array
                **config.get("transformation_config", {}) if config else {}
            },
            **config if config else {}
        }
        
        pipeline = cls(name, pipeline_config)
        
        # Add restructuring stages
        pipeline.add_stages([
            # Using StructuredDataExtractionStage for restructuring functionality 
            # (In a real implementation, you might have a dedicated RestructuringStage)
            StructuredDataExtractionStage({
                "restructuring": True,
                "mapping_rules": mapping_rules,
                "include_unmapped_fields": pipeline_config["transformation_config"].get("include_unmapped_fields", False),
                "structure_type": pipeline_config["transformation_config"].get("structure_type", "flat"),
                "track_changes": pipeline_config["transformation_config"].get("track_changes", True)
            }),
            # Optional normalization after restructuring
            DataNormalizationStage({
                "normalize_keys": True,
                "normalize_text": True
            }),
            JSONOutputStage({
                "format": "json",
                "pretty_print": True,
                "output_field": "restructured_data"
            })
        ])
        
        return pipeline
    
    @classmethod
    def create_aggregation_pipeline(cls, aggregation_rules: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> 'TransformationPipeline':
        """Create a pipeline configured for data aggregation.
        
        Args:
            aggregation_rules: Rules for aggregating data
            config: Additional configuration options
            
        Returns:
            Configured TransformationPipeline instance
        """
        name = "aggregation_pipeline"
        pipeline_config = {
            "transformation_config": {
                "transformation_type": "aggregation",
                "aggregation_rules": aggregation_rules,
                "group_by": config.get("group_by", []) if config else [],
                "sort_by": config.get("sort_by", []) if config else [],
                "filter_rules": config.get("filter_rules", {}) if config else {},
                **config.get("transformation_config", {}) if config else {}
            },
            **config if config else {}
        }
        
        pipeline = cls(name, pipeline_config)
        
        # Add aggregation stages
        pipeline.add_stages([
            # Using StructuredDataExtractionStage for aggregation functionality
            # (In a real implementation, you might have a dedicated AggregationStage)
            StructuredDataExtractionStage({
                "aggregation": True,
                "aggregation_rules": aggregation_rules,
                "group_by": pipeline_config["transformation_config"].get("group_by", []),
                "sort_by": pipeline_config["transformation_config"].get("sort_by", []),
                "filter_rules": pipeline_config["transformation_config"].get("filter_rules", {})
            }),
            JSONOutputStage({
                "format": "json",
                "pretty_print": True,
                "output_field": "aggregated_data"
            })
        ])
        
        return pipeline