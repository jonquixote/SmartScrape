"""
Extraction Stages Package

This package contains pipeline stages for the SmartScrape extraction framework.
These stages work with the pipeline architecture to extract, normalize, and validate data.
"""

from .structural_analysis_stage import StructuralAnalysisStage
from .metadata_extraction_stage import MetadataExtractionStage
from .pattern_extraction_stage import PatternExtractionStage
from .semantic_extraction_stage import SemanticExtractionStage
from .content_normalization_stage import ContentNormalizationStage
from .quality_evaluation_stage import QualityEvaluationStage
from .schema_validation_stage import SchemaValidationStage
from .pipeline_config import ExtractionPipelineFactory, get_extraction_pipeline

__all__ = [
    'StructuralAnalysisStage', 
    'MetadataExtractionStage',
    'PatternExtractionStage',
    'SemanticExtractionStage',
    'ContentNormalizationStage',
    'QualityEvaluationStage',
    'SchemaValidationStage',
    'ExtractionPipelineFactory',
    'get_extraction_pipeline'
]