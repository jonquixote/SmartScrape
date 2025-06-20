"""
Extraction package for SmartScrape

This package provides structural analysis and data extraction capabilities
for HTML documents and other web content.
"""

from extraction.structural_analyzer import DOMStructuralAnalyzer
from extraction.quality_evaluator_impl import QualityEvaluatorImpl
from extraction.metadata_extractor import MetadataExtractorImpl
from extraction.content_normalizer_impl import ContentNormalizerImpl

from extraction.extraction_helpers import (
    create_dynamic_extraction_strategy,
    process_results,
    generate_json_export,
    generate_csv_export,
    generate_excel_export
)

from extraction.fallback_extraction import perform_extraction_with_fallback
from extraction.content_analysis import analyze_site_structure, generate_content_filter_instructions

__all__ = [
    'DOMStructuralAnalyzer',
    'create_dynamic_extraction_strategy',
    'process_results',
    'generate_json_export',
    'generate_csv_export',
    'generate_excel_export',
    'perform_extraction_with_fallback',
    'analyze_site_structure',
    'generate_content_filter_instructions'
]