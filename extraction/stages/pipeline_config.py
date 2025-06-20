"""
Pipeline Configuration Module

This module provides factory methods and configuration templates for creating extraction pipelines.
It defines standard pipeline configurations for different types of extractions.
"""

import logging
from typing import Dict, Any, Optional, List, Union, Type

from core.pipeline.pipeline import Pipeline
from core.pipeline.builder import PipelineBuilder
from core.pipeline.stages.base_stages import ProcessingStage
from extraction.stages.normalization_stage import NormalizationStage
from extraction.stages.content_normalization_stage import ContentNormalizationStage
from extraction.stages.pattern_extraction_stage import PatternExtractionStage
from extraction.stages.semantic_extraction_stage import SemanticExtractionStage
from extraction.stages.quality_evaluation_stage import QualityEvaluationStage
from extraction.stages.schema_validation_stage import SchemaValidationStage

logger = logging.getLogger(__name__)

class ExtractionPipelineFactory:
    """
    Factory for creating and configuring extraction pipelines.
    
    This class provides methods for creating standard pipeline configurations
    for different types of content extraction.
    """
    
    @classmethod
    def create_default_pipeline(cls, config: Optional[Dict[str, Any]] = None) -> Pipeline:
        """
        Create a default extraction pipeline suitable for general-purpose extraction.
        
        Args:
            config: Optional configuration overrides
            
        Returns:
            Configured pipeline instance
        """
        # Start with a new builder
        builder = PipelineBuilder(name="DefaultExtractionPipeline")
        
        # Merge configuration
        merged_config = cls._merge_config(cls.get_default_config(), config or {})
        
        # Add HTML preprocessing stage if configured
        if merged_config.get("use_html_preprocessing", True):
            from core.pipeline.stages.html.html_preprocessing_stage import HTMLPreprocessingStage
            builder.add_stage(HTMLPreprocessingStage(config=merged_config.get("html_preprocessing", {})))
        
        # Add pattern extraction stage
        pattern_config = merged_config.get("pattern_extraction", {})
        builder.add_stage(PatternExtractionStage(config=pattern_config))
        
        # Add semantic extraction if enabled
        if merged_config.get("use_semantic_extraction", True):
            semantic_config = merged_config.get("semantic_extraction", {})
            builder.add_stage(SemanticExtractionStage(config=semantic_config))
        
        # Add normalization stage
        norm_config = merged_config.get("normalization", {})
        builder.add_stage(NormalizationStage(config=norm_config))
        
        # Add quality evaluation if enabled
        if merged_config.get("use_quality_evaluation", True):
            quality_config = merged_config.get("quality_evaluation", {})
            builder.add_stage(QualityEvaluationStage(config=quality_config))
        
        # Add schema validation if enabled
        if merged_config.get("use_schema_validation", True):
            schema_config = merged_config.get("schema_validation", {})
            builder.add_stage(SchemaValidationStage(config=schema_config))
        
        # Set pipeline-level properties
        if "max_retries" in merged_config:
            builder.set_max_retries(merged_config["max_retries"])
        
        if merged_config.get("parallel_execution", False):
            builder.enable_parallel_execution()
        
        # Build and return the pipeline
        pipeline = builder.build()
        logger.info(f"Created default extraction pipeline with {len(pipeline.stages)} stages")
        return pipeline
    
    @classmethod
    def create_product_pipeline(cls, config: Optional[Dict[str, Any]] = None) -> Pipeline:
        """
        Create an extraction pipeline optimized for product data.
        
        Args:
            config: Optional configuration overrides
            
        Returns:
            Configured pipeline instance
        """
        # Start with default configuration
        base_config = cls.get_default_config()
        
        # Apply product-specific configuration
        product_config = {
            "pattern_extraction": {
                "patterns": ["product_price", "product_title", "product_description", "product_specs"],
                "content_type": "product",
                "extract_structured_data": True,
                "structured_data_format": ["json-ld", "microdata"]
            },
            "semantic_extraction": {
                "content_type": "product",
                "extraction_mode": "targeted",
                "target_fields": ["title", "price", "description", "features", "specifications"]
            },
            "normalization": {
                "standardize_prices": True,
                "standardize_units": True,
                "field_types": {
                    "price": "price",
                    "dimensions": "measurement",
                    "weight": "measurement",
                    "specs": "key_value_list"
                }
            },
            "schema_validation": {
                "schema_name": "product_schema",
                "required_fields": ["title", "price"]
            }
        }
        
        # Merge configurations
        merged_config = cls._merge_config(base_config, product_config)
        
        # Apply user overrides
        if config:
            merged_config = cls._merge_config(merged_config, config)
        
        # Create the pipeline
        return cls.create_default_pipeline(merged_config)
    
    @classmethod
    def create_article_pipeline(cls, config: Optional[Dict[str, Any]] = None) -> Pipeline:
        """
        Create an extraction pipeline optimized for article content.
        
        Args:
            config: Optional configuration overrides
            
        Returns:
            Configured pipeline instance
        """
        # Start with default configuration
        base_config = cls.get_default_config()
        
        # Apply article-specific configuration
        article_config = {
            "pattern_extraction": {
                "patterns": ["article_title", "article_author", "article_date", "article_content"],
                "content_type": "article",
                "extract_metadata": True,
                "extract_structured_data": True,
                "structured_data_format": ["json-ld"]
            },
            "semantic_extraction": {
                "content_type": "article",
                "extraction_mode": "comprehensive",
                "target_fields": ["title", "author", "published_date", "content", "summary"]
            },
            "normalization": {
                "standardize_dates": True,
                "normalize_whitespace": True,
                "field_types": {
                    "published_date": "date",
                    "content": "long_text",
                    "summary": "text"
                }
            },
            "schema_validation": {
                "schema_name": "article_schema",
                "required_fields": ["title", "content"]
            }
        }
        
        # Merge configurations
        merged_config = cls._merge_config(base_config, article_config)
        
        # Apply user overrides
        if config:
            merged_config = cls._merge_config(merged_config, config)
        
        # Create the pipeline
        return cls.create_default_pipeline(merged_config)
    
    @classmethod
    def create_listing_pipeline(cls, config: Optional[Dict[str, Any]] = None) -> Pipeline:
        """
        Create an extraction pipeline optimized for product listings/search results.
        
        Args:
            config: Optional configuration overrides
            
        Returns:
            Configured pipeline instance
        """
        # Start with default configuration
        base_config = cls.get_default_config()
        
        # Apply listing-specific configuration
        listing_config = {
            "pattern_extraction": {
                "patterns": ["listing_items", "pagination", "filters"],
                "content_type": "listing",
                "extract_list_items": True,
                "extract_pagination": True
            },
            "semantic_extraction": {
                "content_type": "listing",
                "extraction_mode": "list",
                "target_fields": ["items", "pagination", "total_results", "filters"]
            },
            "normalization": {
                "normalize_lists": True,
                "field_types": {
                    "items": "list",
                    "pagination": "object",
                    "filters": "key_value_list"
                }
            },
            "schema_validation": {
                "schema_name": "listing_schema",
                "required_fields": ["items"],
                "list_item_schema": "product_summary_schema"
            }
        }
        
        # Merge configurations
        merged_config = cls._merge_config(base_config, listing_config)
        
        # Apply user overrides
        if config:
            merged_config = cls._merge_config(merged_config, config)
        
        # Create the pipeline
        return cls.create_default_pipeline(merged_config)
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """
        Get the default pipeline configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            # Pipeline settings
            "max_retries": 2,
            "parallel_execution": False,
            
            # Feature flags
            "use_html_preprocessing": True,
            "use_semantic_extraction": True,
            "use_quality_evaluation": True,
            "use_schema_validation": True,
            
            # HTML preprocessing configuration
            "html_preprocessing": {
                "remove_scripts": True,
                "remove_styles": True,
                "remove_comments": True,
                "convert_encoding": "utf-8"
            },
            
            # Pattern extraction configuration
            "pattern_extraction": {
                "extract_structured_data": True,
                "extract_metadata": True,
                "use_css_selectors": True,
                "use_xpath": True,
                "output_key": "extracted_data"
            },
            
            # Semantic extraction configuration
            "semantic_extraction": {
                "model": "default",
                "extraction_mode": "auto",
                "confidence_threshold": 0.7,
                "max_tokens": 1000,
                "input_key": "extracted_data",
                "output_key": "enhanced_data"
            },
            
            # Normalization configuration
            "normalization": {
                "trim_strings": True,
                "normalize_whitespace": True,
                "standardize_dates": True,
                "standardize_prices": True,
                "remove_html": True,
                "input_key": "enhanced_data",
                "output_key": "normalized_data"
            },
            
            # Quality evaluation configuration
            "quality_evaluation": {
                "min_quality_score": 0.6,
                "check_completeness": True,
                "check_consistency": True,
                "input_key": "normalized_data",
                "output_key": "quality_checked_data"
            },
            
            # Schema validation configuration
            "schema_validation": {
                "strict_validation": False,
                "add_missing_fields": True,
                "input_key": "quality_checked_data",
                "output_key": "validated_data"
            }
        }
    
    @classmethod
    def _merge_config(cls, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge two configuration dictionaries.
        
        Args:
            base_config: Base configuration dictionary
            override_config: Configuration dictionary with overrides
            
        Returns:
            Merged configuration dictionary
        """
        import copy
        result = copy.deepcopy(base_config)
        
        for key, value in override_config.items():
            # If both are dictionaries, merge recursively
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = cls._merge_config(result[key], value)
            else:
                # Otherwise just override
                result[key] = copy.deepcopy(value)
        
        return result


def get_extraction_pipeline(pipeline_type: str = "default", 
                         config: Optional[Dict[str, Any]] = None) -> Pipeline:
    """
    Get an extraction pipeline of the specified type.
    
    Args:
        pipeline_type: Type of pipeline to create
        config: Optional configuration overrides
        
    Returns:
        Configured pipeline instance
    """
    factory = ExtractionPipelineFactory
    
    if pipeline_type == "product":
        return factory.create_product_pipeline(config)
    elif pipeline_type == "article":
        return factory.create_article_pipeline(config)
    elif pipeline_type == "listing":
        return factory.create_listing_pipeline(config)
    else:
        return factory.create_default_pipeline(config)


def register_pipeline_templates(registry):
    """
    Register pipeline templates with the provided registry.
    
    Args:
        registry: Pipeline registry to register templates with
    """
    # Register standard pipeline configurations as templates
    factory = ExtractionPipelineFactory
    
    # Get base configurations
    default_config = factory.get_default_config()
    
    # Register templates
    registry.register_pipeline_template("default_extraction", {
        "name": "Default Extraction Pipeline",
        "stages": [
            {"type": "html_preprocessing", "config": default_config.get("html_preprocessing", {})},
            {"type": "pattern_extraction", "config": default_config.get("pattern_extraction", {})},
            {"type": "semantic_extraction", "config": default_config.get("semantic_extraction", {})},
            {"type": "normalization", "config": default_config.get("normalization", {})},
            {"type": "quality_evaluation", "config": default_config.get("quality_evaluation", {})},
            {"type": "schema_validation", "config": default_config.get("schema_validation", {})}
        ],
        "max_retries": default_config.get("max_retries", 2),
        "parallel_execution": default_config.get("parallel_execution", False)
    })
    
    # Create product extraction pipeline template
    product_config = factory._merge_config(default_config, {
        "pattern_extraction": {
            "patterns": ["product_price", "product_title", "product_description", "product_specs"],
            "content_type": "product"
        },
        "semantic_extraction": {
            "content_type": "product",
            "target_fields": ["title", "price", "description", "features", "specifications"]
        }
    })
    
    registry.register_pipeline_template("product_extraction", {
        "name": "Product Extraction Pipeline",
        "stages": [
            {"type": "html_preprocessing", "config": product_config.get("html_preprocessing", {})},
            {"type": "pattern_extraction", "config": product_config.get("pattern_extraction", {})},
            {"type": "semantic_extraction", "config": product_config.get("semantic_extraction", {})},
            {"type": "normalization", "config": product_config.get("normalization", {})},
            {"type": "quality_evaluation", "config": product_config.get("quality_evaluation", {})},
            {"type": "schema_validation", "config": product_config.get("schema_validation", {})}
        ],
        "max_retries": product_config.get("max_retries", 2)
    })
    
    # Create article extraction pipeline template
    article_config = factory._merge_config(default_config, {
        "pattern_extraction": {
            "patterns": ["article_title", "article_author", "article_date", "article_content"],
            "content_type": "article"
        },
        "semantic_extraction": {
            "content_type": "article",
            "target_fields": ["title", "author", "published_date", "content", "summary"]
        }
    })
    
    registry.register_pipeline_template("article_extraction", {
        "name": "Article Extraction Pipeline",
        "stages": [
            {"type": "html_preprocessing", "config": article_config.get("html_preprocessing", {})},
            {"type": "pattern_extraction", "config": article_config.get("pattern_extraction", {})},
            {"type": "semantic_extraction", "config": article_config.get("semantic_extraction", {})},
            {"type": "normalization", "config": article_config.get("normalization", {})},
            {"type": "quality_evaluation", "config": article_config.get("quality_evaluation", {})},
            {"type": "schema_validation", "config": article_config.get("schema_validation", {})}
        ],
        "max_retries": article_config.get("max_retries", 2)
    })
    
    logger.info("Registered standard pipeline templates with registry")