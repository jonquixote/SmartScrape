"""
Metadata Extraction Stage Module

This module provides a pipeline stage for extracting metadata from HTML content,
consolidating information from multiple sources like meta tags, structured data,
Open Graph, and microdata.
"""

import logging
from typing import Dict, Any, Optional, Union, List
from bs4 import BeautifulSoup, Tag

from core.pipeline.stages.base_stages import ProcessingStage
from core.pipeline.context import PipelineContext
from extraction.metadata_extractor import MetadataExtractorImpl
from core.retry_manager import RetryManager

logger = logging.getLogger(__name__)

class MetadataExtractionStage(ProcessingStage):
    """
    Pipeline stage that extracts metadata from content using multiple extractor types.
    
    This stage uses the MetadataExtractorImpl to extract metadata from various
    sources including HTML meta tags, Open Graph, JSON-LD structured data, and microdata,
    consolidating it to provide comprehensive page metadata.
    """
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the metadata extraction stage with configuration.
        
        Args:
            name: Name of this stage (defaults to class name)
            config: Configuration dictionary
        """
        super().__init__(name, config)
        self.extractor = None
        self.input_key = self.config.get("input_key", "html_content")
        self.output_key = self.config.get("output_key", "metadata")
        self.url_key = self.config.get("url_key", "url")
        self.extractor_types = self.config.get("extractor_types", [
            "html_meta", "open_graph", "json_ld", "microdata"
        ])
        self.confidence_threshold = self.config.get("confidence_threshold", 0.6)
        self.prioritize_metadata = self.config.get("prioritize_metadata", True)
        self.normalize_metadata = self.config.get("normalize_metadata", True)
        self.preserve_sources = self.config.get("preserve_sources", True)
        self.retry_manager = RetryManager()
        
    async def initialize(self) -> None:
        """Initialize the extractor and stage resources."""
        if self._initialized:
            return
            
        # Create the metadata extractor
        self.extractor = MetadataExtractorImpl()
        
        # Initialize the extractor
        self.extractor.initialize()
        
        await super().initialize()
        logger.debug(f"{self.name} initialized with metadata extractor")
        
    async def cleanup(self) -> None:
        """Clean up resources used by this stage."""
        if self.extractor and self.extractor._initialized:
            self.extractor.shutdown()
            
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
            
        html_content = context.get(self.input_key)
        if not html_content or not isinstance(html_content, (str, BeautifulSoup, Tag)):
            logger.warning(f"Invalid HTML content in '{self.input_key}'")
            context.add_error(self.name, f"Invalid HTML content: {type(html_content)}")
            return False
            
        return True
        
    async def transform_data(self, data: Dict[str, Any], context: PipelineContext) -> Optional[Dict[str, Any]]:
        """
        Extract metadata from HTML content.
        
        Args:
            data: Input data (not used, we get data from context)
            context: Pipeline context containing HTML content
            
        Returns:
            Dictionary containing extracted metadata or None if extraction fails
        """
        try:
            if not self.extractor:
                self.extractor = MetadataExtractorImpl()
                self.extractor.initialize()
            
            # Set the context if available
            if hasattr(context, "strategy_context") and context.strategy_context:
                self.extractor.context = context.strategy_context
            
            # Get HTML content from context
            html_content = context.get(self.input_key)
            
            # Get URL from context if available (helps with relative URL resolution)
            url = context.get(self.url_key, "")
            
            # Prepare extraction options
            options = {
                "url": url,
                "prioritize_metadata": self.prioritize_metadata,
                "normalize": self.normalize_metadata,
                "extraction_time": context.get_metadata().get("stage_start_time", {}).get(self.name)
            }
            
            # Extract metadata
            logger.info(f"Extracting metadata with prioritize_metadata={self.prioritize_metadata}, normalize={self.normalize_metadata}")
            extraction_result = self.extractor.extract(html_content, options)
            
            # Calculate confidence scores for each metadata field
            confidence_scores = self._calculate_confidence_scores(extraction_result)
            
            # Add confidence scores to metadata
            if "_metadata" not in extraction_result:
                extraction_result["_metadata"] = {}
            extraction_result["_metadata"]["confidence_scores"] = confidence_scores
            
            # Flag low-confidence fields
            low_confidence_fields = [
                field for field, score in confidence_scores.items() 
                if score < self.confidence_threshold and field not in ["_metadata"]
            ]
            
            if low_confidence_fields:
                extraction_result["_metadata"]["low_confidence_fields"] = low_confidence_fields
                logger.debug(f"Low confidence metadata fields: {', '.join(low_confidence_fields)}")
            
            # Set content type based on metadata if not already set
            self._set_content_type_from_metadata(extraction_result, context)
            
            # Add canonical URL to context if found
            if "canonical_url" in extraction_result:
                context.set("canonical_url", extraction_result["canonical_url"])
            
            # Add title to context if found
            if "title" in extraction_result:
                context.set("page_title", extraction_result["title"])
            
            # Add structured data to context if available
            if self.preserve_sources:
                if "json_ld" in extraction_result:
                    context.set("structured_data", extraction_result["json_ld"])
                if "open_graph" in extraction_result:
                    context.set("open_graph", extraction_result["open_graph"])
                if "microdata" in extraction_result:
                    context.set("microdata", extraction_result["microdata"])
            
            return extraction_result
            
        except Exception as e:
            logger.error(f"Error in metadata extraction: {str(e)}")
            context.add_error(self.name, f"Extraction error: {str(e)}")
            return None
    
    def _calculate_confidence_scores(self, metadata: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate confidence scores for extracted metadata fields.
        
        Args:
            metadata: Extracted metadata
            
        Returns:
            Dictionary mapping field names to confidence scores
        """
        confidence_scores = {}
        sources = metadata.get("_metadata", {}).get("sources", [])
        
        for field, value in metadata.items():
            if field == "_metadata" or field.startswith("_"):
                continue
                
            # Base confidence on field type and presence in multiple sources
            base_confidence = 0.5
            
            # Higher confidence for fields present in structured data
            if "json_ld" in sources and field in metadata.get("json_ld", {}):
                base_confidence = max(base_confidence, 0.8)
            
            # Medium confidence for fields in Open Graph
            elif "open_graph" in sources and field in metadata.get("open_graph", {}):
                base_confidence = max(base_confidence, 0.7)
            
            # Lower confidence for basic meta tags
            elif "html_meta" in sources and field in metadata.get("html_meta", {}):
                base_confidence = max(base_confidence, 0.6)
                
            # Adjust confidence based on value characteristics
            if field in ["title", "description", "image", "url"]:
                if isinstance(value, str) and len(value) > 5:
                    base_confidence += 0.1
                    
            # Add confidence score
            confidence_scores[field] = min(base_confidence, 1.0)  # Cap at 1.0
        
        return confidence_scores
    
    def _set_content_type_from_metadata(self, metadata: Dict[str, Any], context: PipelineContext) -> None:
        """
        Set content type in context based on metadata if not already present.
        
        Args:
            metadata: Extracted metadata
            context: Pipeline context to update
        """
        # Skip if content type is already set with high confidence
        if (context.has_key("content_type") and 
            context.has_key("content_type_confidence") and 
            context.get("content_type_confidence", 0) > 0.6):
            return
            
        content_type = None
        confidence = 0.0
        
        # Check JSON-LD schema type
        jsonld = metadata.get("json_ld", {})
        if isinstance(jsonld, dict) and "@type" in jsonld:
            schema_type = jsonld["@type"]
            if schema_type == "Product":
                content_type = "product"
                confidence = 0.8
            elif schema_type in ["Article", "BlogPosting", "NewsArticle"]:
                content_type = "article"
                confidence = 0.8
            elif schema_type in ["ItemList", "SearchResultsPage"]:
                content_type = "listing"
                confidence = 0.7
                
        # Check Open Graph type
        if not content_type or confidence < 0.7:
            og_type = metadata.get("open_graph", {}).get("type")
            if og_type == "product":
                content_type = "product"
                confidence = 0.7
            elif og_type == "article":
                content_type = "article"
                confidence = 0.7
                
        # Set content type and confidence if found
        if content_type and confidence > context.get("content_type_confidence", 0):
            context.set("content_type", content_type)
            context.set("content_type_confidence", confidence)
            logger.debug(f"Set content type from metadata: {content_type} (confidence: {confidence})")