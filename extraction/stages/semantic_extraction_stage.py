"""
Semantic Extraction Stage Module

This module provides a pipeline stage for extracting data using semantic understanding
and AI-assisted extraction for complex content structures.
"""

import logging
import json
from typing import Dict, Any, Optional, Union, List
from bs4 import BeautifulSoup, Tag

from core.pipeline.stages.base_stages import ProcessingStage
from core.pipeline.context import PipelineContext
from extraction.semantic_extractor import SemanticExtractor
from core.retry_manager import RetryManager

logger = logging.getLogger(__name__)

class SemanticExtractionStage(ProcessingStage):
    """
    Pipeline stage that extracts data using semantic understanding and AI.
    
    This stage uses the SemanticExtractor to perform context-aware extraction
    of information from HTML content, leveraging AI to understand and extract
    data from complex or unstructured content that pattern-based approaches
    might miss.
    """
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the semantic extraction stage with configuration.
        
        Args:
            name: Name of this stage (defaults to class name)
            config: Configuration dictionary
        """
        super().__init__(name, config)
        self.extractor = None
        self.input_key = self.config.get("input_key", "html_content")
        self.output_key = self.config.get("output_key", "semantic_data")
        self.url_key = self.config.get("url_key", "url")
        self.schema_key = self.config.get("schema_key", "extraction_schema")
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.use_extraction_hints = self.config.get("use_extraction_hints", True)
        self.ai_service_name = self.config.get("ai_service_name", "ai_service")
        self.enable_caching = self.config.get("enable_caching", True)
        self.max_token_limit = self.config.get("max_token_limit", 4000)
        self.fallback_to_pattern = self.config.get("fallback_to_pattern", True)
        self.retry_manager = RetryManager()
        
    async def initialize(self) -> None:
        """Initialize the extractor and stage resources."""
        if self._initialized:
            return
            
        # Create the semantic extractor
        self.extractor = SemanticExtractor()
        
        # Initialize the extractor with configuration
        extractor_config = self.config.get("extractor_config", {})
        self.extractor.initialize(extractor_config)
        
        await super().initialize()
        logger.debug(f"{self.name} initialized with semantic extractor")
        
    async def cleanup(self) -> None:
        """Clean up resources used by this stage."""
        if self.extractor:
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
            
        # Check if AI service is available (only if we're not in test mode)
        if not self.config.get("test_mode", False):
            ai_service = self._get_ai_service(context)
            if not ai_service:
                logger.warning("AI service not available for semantic extraction")
                context.add_error(self.name, "AI service not available")
                return False
            
        return True
        
    async def transform_data(self, data: Dict[str, Any], context: PipelineContext) -> Optional[Dict[str, Any]]:
        """
        Extract data using semantic understanding and AI.
        
        Args:
            data: Input data (not used, we get data from context)
            context: Pipeline context containing HTML content
            
        Returns:
            Dictionary containing extracted data or None if extraction fails
        """
        # Use retry management for the actual execution
        try:
            # Define a function to be retried
            async def _extract_with_retry():
                return await self._extract_data(data, context)
                
            # Execute with retry
            return await self.retry_manager.retry_async(
                max_attempts=2, 
                retry_on=["ConnectionError", "TimeoutError", "ServiceUnavailableError"]
            )(_extract_with_retry)()
        except Exception as e:
            logger.error(f"Error in semantic extraction with retry: {str(e)}")
            context.add_error(self.name, f"Extraction error: {str(e)}")
            
            # Fall back to pattern extraction if available
            if self.fallback_to_pattern and context.has_key("extracted_data"):
                logger.info("Falling back to pattern extraction results")
                pattern_results = context.get("extracted_data")
                if "_metadata" not in pattern_results:
                    pattern_results["_metadata"] = {}
                pattern_results["_metadata"]["fallback"] = True
                return pattern_results
                
            return None
            
    async def _extract_data(self, data: Dict[str, Any], context: PipelineContext) -> Optional[Dict[str, Any]]:
        """
        The actual data extraction implementation, which may be retried.
        
        Args:
            data: Input data
            context: Pipeline context
            
        Returns:
            Extracted data or None
        """
        try:
            if not self.extractor:
                self.extractor = SemanticExtractor()
                extractor_config = self.config.get("extractor_config", {})
                self.extractor.initialize(extractor_config)
            
            # Set the context if available
            if hasattr(context, "strategy_context") and context.strategy_context:
                self.extractor.context = context.strategy_context
            
            # Set the AI service if not already set
            if not self.extractor.ai_service:
                ai_service = self._get_ai_service(context)
                if ai_service:
                    self.extractor.ai_service = ai_service
            
            # Get HTML content from context
            html_content = context.get(self.input_key)
            
            # Get URL from context if available
            url = context.get(self.url_key, "")
            
            # Get schema from context if available
            schema = None
            if context.has_key(self.schema_key):
                schema = context.get(self.schema_key)
            
            # Get extraction hints if available and enabled
            extraction_hints = {}
            if self.use_extraction_hints and context.has_key("extraction_hints"):
                extraction_hints = context.get("extraction_hints")
            
            # If we have pattern extraction results, use them as a baseline
            pattern_results = None
            if self.fallback_to_pattern and context.has_key("extracted_data"):
                pattern_results = context.get("extracted_data")
            
            # Prepare extraction options
            options = {
                "url": url,
                "extraction_hints": extraction_hints,
                "content_type": context.get("content_type", "unknown"),
                "enable_caching": self.enable_caching,
                "max_token_limit": self.max_token_limit,
                "baseline_data": pattern_results
            }
            
            # Extract data
            logger.info(f"Extracting data using semantic understanding with schema: {schema is not None}")
            extraction_result = await self.extractor.extract(html_content, schema, options)
            
            # Add confidence scores
            if "_metadata" not in extraction_result:
                extraction_result["_metadata"] = {}
            
            extraction_result["_metadata"]["extraction_method"] = "semantic"
            extraction_result["_metadata"]["confidence_scores"] = self._calculate_confidence_scores(extraction_result)
            
            # Flag low-confidence fields
            low_confidence_fields = [
                field for field, score in extraction_result["_metadata"]["confidence_scores"].items() 
                if score < self.confidence_threshold and field not in ["_metadata"]
            ]
            
            if low_confidence_fields:
                extraction_result["_metadata"]["low_confidence_fields"] = low_confidence_fields
                logger.debug(f"Low confidence semantic fields: {', '.join(low_confidence_fields)}")
            
            # If we have pattern results, merge them with semantic results for completeness
            if pattern_results and isinstance(pattern_results, dict):
                merged_result = self._merge_extraction_results(pattern_results, extraction_result)
                
                # Keep track of data sources
                if "_metadata" not in merged_result:
                    merged_result["_metadata"] = {}
                merged_result["_metadata"]["merged"] = True
                merged_result["_metadata"]["sources"] = ["pattern", "semantic"]
                
                # Use the merged result
                extraction_result = merged_result
            
            # Store extraction statistics
            self._update_extraction_stats(context, extraction_result)
            
            return extraction_result
            
        except Exception as e:
            logger.error(f"Error in semantic extraction: {str(e)}")
            context.add_error(self.name, f"Extraction error: {str(e)}")
            
            # Fall back to pattern extraction if available
            if self.fallback_to_pattern and context.has_key("extracted_data"):
                logger.info("Falling back to pattern extraction results")
                pattern_results = context.get("extracted_data")
                if "_metadata" not in pattern_results:
                    pattern_results["_metadata"] = {}
                pattern_results["_metadata"]["fallback"] = True
                return pattern_results
                
            return None
    
    def _get_ai_service(self, context: PipelineContext) -> Any:
        """
        Get the AI service from the context.
        
        Args:
            context: Pipeline context
            
        Returns:
            AI service or None if not available
        """
        if hasattr(context, "strategy_context") and context.strategy_context:
            try:
                return context.strategy_context.get_service(self.ai_service_name)
            except Exception as e:
                logger.warning(f"Error getting AI service: {str(e)}")
                
        # For test mode, we can use a mock
        if self.config.get("test_mode", False):
            return self._create_mock_ai_service()
                
        return None
    
    def _create_mock_ai_service(self) -> Any:
        """
        Create a mock AI service for testing.
        
        Returns:
            Mock AI service
        """
        # Simple mock that just returns a fixed response
        class MockAIService:
            async def extract_structured_data(self, content, schema=None, options=None):
                return {"title": "Mock Title", "text": "Mock Content", "_metadata": {"mock": True}}
                
            async def analyze_content(self, content, query=None, options=None):
                return {"analysis": "Mock Analysis", "_metadata": {"mock": True}}
                
        return MockAIService()
    
    def _calculate_confidence_scores(self, extraction_result: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate confidence scores for extracted fields.
        
        Args:
            extraction_result: Extracted data
            
        Returns:
            Dictionary mapping field names to confidence scores
        """
        confidence_scores = {}
        
        # If the extractor provided confidence scores, use those
        if "_metadata" in extraction_result and "field_confidences" in extraction_result["_metadata"]:
            return extraction_result["_metadata"]["field_confidences"]
        
        # Otherwise, calculate based on field characteristics
        for field, value in extraction_result.items():
            if field == "_metadata" or field.startswith("_"):
                continue
                
            # Default confidence
            confidence = 0.7
            
            # Adjust based on value type and characteristics
            if isinstance(value, str):
                # Empty or very short strings have lower confidence
                if not value or len(value) < 3:
                    confidence = 0.4
                # Very long strings are likely to be correct
                elif len(value) > 100:
                    confidence = 0.9
            elif isinstance(value, (list, dict)):
                # Complex structures usually have higher confidence
                confidence = 0.8
                # Empty complex structures have lower confidence
                if not value:
                    confidence = 0.5
            elif value is None:
                # Null values have low confidence
                confidence = 0.3
                
            confidence_scores[field] = confidence
            
        return confidence_scores
    
    def _merge_extraction_results(self, pattern_result: Dict[str, Any], 
                                semantic_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge pattern and semantic extraction results, favoring semantic where available.
        
        Args:
            pattern_result: Results from pattern extraction
            semantic_result: Results from semantic extraction
            
        Returns:
            Merged extraction results
        """
        # Start with pattern results as the base
        merged = dict(pattern_result)
        
        # Get confidence scores for semantic results
        semantic_confidences = {}
        if "_metadata" in semantic_result and "confidence_scores" in semantic_result["_metadata"]:
            semantic_confidences = semantic_result["_metadata"]["confidence_scores"]
            
        # Update with semantic results
        for field, value in semantic_result.items():
            if field == "_metadata" or field.startswith("_"):
                continue
                
            # Check if we should use semantic value
            confidence = semantic_confidences.get(field, 0.7)
            if confidence >= self.confidence_threshold:
                merged[field] = value
                
        # Special handling for lists of items
        if "items" in pattern_result and "items" in semantic_result:
            # If semantic analysis found more items, use those
            if len(semantic_result["items"]) > len(pattern_result["items"]):
                merged["items"] = semantic_result["items"]
            # Otherwise augment existing items with semantic information
            else:
                merged["items"] = self._augment_items(pattern_result["items"], semantic_result["items"])
                
        # Merge metadata
        merged["_metadata"] = merged.get("_metadata", {})
        merged["_metadata"].update(semantic_result.get("_metadata", {}))
        
        return merged
    
    def _augment_items(self, pattern_items: List[Dict[str, Any]], 
                     semantic_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Augment pattern-extracted items with semantic information where available.
        
        Args:
            pattern_items: Items from pattern extraction
            semantic_items: Items from semantic extraction
            
        Returns:
            Augmented items
        """
        # If lists are not the same size, we cannot reliably augment
        if len(pattern_items) != len(semantic_items):
            # Return the list with more items
            return semantic_items if len(semantic_items) > len(pattern_items) else pattern_items
            
        # Augment each item
        augmented_items = []
        for pattern_item, semantic_item in zip(pattern_items, semantic_items):
            # Start with pattern item as base
            augmented = dict(pattern_item)
            
            # Add semantic fields that are missing or empty in pattern item
            for field, value in semantic_item.items():
                if field not in pattern_item or not pattern_item[field]:
                    augmented[field] = value
                    
            augmented_items.append(augmented)
            
        return augmented_items
    
    def _update_extraction_stats(self, context: PipelineContext, result: Dict[str, Any]) -> None:
        """
        Update extraction statistics in context.
        
        Args:
            context: Pipeline context
            result: Extraction result
        """
        # Get existing stats or initialize
        stats = context.get("extraction_stats", {})
        
        # Update stats
        stats["semantic_extraction"] = {
            "fields_extracted": len([k for k in result.keys() if not k.startswith("_")]),
            "has_items": "items" in result,
            "item_count": len(result.get("items", [])) if "items" in result else 0,
            "low_confidence_fields": len(result.get("_metadata", {}).get("low_confidence_fields", [])),
            "merged": result.get("_metadata", {}).get("merged", False),
            "fallback": result.get("_metadata", {}).get("fallback", False)
        }
        
        # Store updated stats
        context.set("extraction_stats", stats)