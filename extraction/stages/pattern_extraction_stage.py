"""
Pattern Extraction Stage Module

This module provides a pipeline stage for extracting structured data using pattern
matching techniques, with support for paginated content aggregation.
"""

import logging
from typing import Dict, Any, Optional, Union, List
import asyncio
from bs4 import BeautifulSoup, Tag

from core.pipeline.stages.base_stages import ProcessingStage
from core.pipeline.context import PipelineContext
from extraction.pattern_extractor import DOMPatternExtractor
from core.retry_manager import RetryManager

logger = logging.getLogger(__name__)

class PatternExtractionStage(ProcessingStage):
    """
    Pipeline stage that extracts structured data using pattern matching techniques.
    
    This stage uses the DOMPatternExtractor to identify patterns in HTML content and
    extract structured data, with support for handling paginated content and aggregating
    data across multiple pages.
    """
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the pattern extraction stage with configuration.
        
        Args:
            name: Name of this stage (defaults to class name)
            config: Configuration dictionary
        """
        super().__init__(name, config)
        self.extractor = None
        self.input_key = self.config.get("input_key", "html_content")
        self.output_key = self.config.get("output_key", "extracted_data")
        self.url_key = self.config.get("url_key", "url")
        self.schema_key = self.config.get("schema_key", "extraction_schema")
        self.enable_pagination = self.config.get("enable_pagination", True)
        self.max_pagination_pages = self.config.get("max_pagination_pages", 3)
        self.pagination_delay = self.config.get("pagination_delay", 1.0)
        self.use_extraction_hints = self.config.get("use_extraction_hints", True)
        self.retry_manager = RetryManager()
        
    async def initialize(self) -> None:
        """Initialize the extractor and stage resources."""
        if self._initialized:
            return
            
        # Create the pattern extractor
        self.extractor = DOMPatternExtractor()
        
        # Initialize the extractor with configuration
        extractor_config = self.config.get("extractor_config", {})
        self.extractor.initialize(extractor_config)
        
        await super().initialize()
        logger.debug(f"{self.name} initialized with pattern extractor")
        
    async def cleanup(self) -> None:
        """Clean up resources used by this stage."""
        if self.extractor and self.extractor.is_initialized:
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
        Extract structured data using pattern matching techniques.
        
        Args:
            data: Input data (not used, we get data from context)
            context: Pipeline context containing HTML content
            
        Returns:
            Dictionary containing extracted data or None if extraction fails
        """
        return await self.retry_manager.retry(self._transform_data, data, context)
    
    async def _transform_data(self, data: Dict[str, Any], context: PipelineContext) -> Optional[Dict[str, Any]]:
        """
        Extract structured data using pattern matching techniques.
        
        Args:
            data: Input data (not used, we get data from context)
            context: Pipeline context containing HTML content
            
        Returns:
            Dictionary containing extracted data or None if extraction fails
        """
        try:
            if not self.extractor:
                self.extractor = DOMPatternExtractor()
                extractor_config = self.config.get("extractor_config", {})
                self.extractor.initialize(extractor_config)
            
            # Set the context if available
            if hasattr(context, "strategy_context") and context.strategy_context:
                self.extractor.context = context.strategy_context
            
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
            
            # Prepare extraction options
            options = {
                "url": url,
                "extraction_hints": extraction_hints
            }
            
            # Get content type if available
            if context.has_key("content_type"):
                options["content_type"] = context.get("content_type")
            
            # Extract data
            logger.info(f"Extracting data using pattern matching with schema: {schema is not None}")
            extraction_result = self.extractor.extract(html_content, schema, options)
            
            # Handle pagination if enabled
            if self.enable_pagination and context.has_key("extraction_hints"):
                hints = context.get("extraction_hints")
                if hints.get("has_pagination", False):
                    await self._handle_pagination(context, extraction_result, url, schema, options)
            
            # Store extraction method in context for tracking
            extraction_result["_metadata"] = extraction_result.get("_metadata", {})
            extraction_result["_metadata"]["extraction_method"] = "pattern"
            
            # Store number of extracted items in context
            if "items" in extraction_result:
                item_count = len(extraction_result["items"])
                extraction_result["_metadata"]["item_count"] = item_count
                logger.info(f"Extracted {item_count} items using pattern extraction")
            
            return extraction_result
            
        except Exception as e:
            logger.error(f"Error in pattern extraction: {str(e)}")
            context.add_error(self.name, f"Extraction error: {str(e)}")
            return None
    
    async def _handle_pagination(self, context: PipelineContext, initial_result: Dict[str, Any], 
                              url: str, schema: Optional[Dict[str, Any]], 
                              options: Dict[str, Any]) -> None:
        """
        Handle pagination by extracting and aggregating data from multiple pages.
        
        Args:
            context: Pipeline context
            initial_result: Initial extraction result
            url: URL of the initial page
            schema: Extraction schema
            options: Extraction options
        """
        try:
            # Get session manager from context for making HTTP requests
            session_manager = None
            if hasattr(context, "strategy_context") and context.strategy_context:
                try:
                    session_manager = context.strategy_context.get_service("session_manager")
                except Exception:
                    logger.warning("Could not get session_manager from context")
            
            if not session_manager:
                logger.warning("Cannot handle pagination without session_manager")
                return
            
            # Get pagination information from extraction hints
            hints = context.get("extraction_hints", {})
            pagination_elements = hints.get("pagination_elements", [])
            
            # Find next page link
            next_page_url = None
            for element in pagination_elements:
                if element.get("rel") == "next" or element.get("is_next", False):
                    next_page_url = element.get("url")
                    break
            
            # If no next page found, try to get from initial result
            if not next_page_url and "next_page" in initial_result:
                next_page_url = initial_result.get("next_page")
            
            if not next_page_url:
                logger.debug("No next page URL found for pagination")
                return
            
            # Initialize pagination tracking
            page_count = 1  # We've already processed the first page
            aggregated_items = initial_result.get("items", [])
            
            # Process pagination pages
            while next_page_url and page_count < self.max_pagination_pages:
                # Introduce delay to avoid rate limiting
                await asyncio.sleep(self.pagination_delay)
                
                try:
                    # Fetch next page
                    logger.info(f"Fetching pagination page {page_count + 1}: {next_page_url}")
                    response = await session_manager.get(next_page_url)
                    
                    if response.status_code != 200:
                        logger.warning(f"Failed to fetch pagination page: {response.status_code}")
                        break
                    
                    # Extract data from next page
                    page_html = response.text
                    page_options = dict(options)
                    page_options["url"] = next_page_url
                    
                    # Extract data from the page
                    page_result = self.extractor.extract(page_html, schema, page_options)
                    
                    # Get items from the page
                    page_items = page_result.get("items", [])
                    if page_items:
                        logger.info(f"Extracted {len(page_items)} items from pagination page {page_count + 1}")
                        aggregated_items.extend(page_items)
                    
                    # Update next page URL
                    next_page_url = None
                    if "next_page" in page_result:
                        next_page_url = page_result.get("next_page")
                    
                    # If no next page in result, try to extract from pagination elements
                    if not next_page_url:
                        # Parse the new page HTML to find pagination elements
                        soup = BeautifulSoup(page_html, "lxml")
                        for selector in hints.get("pagination_selectors", []):
                            next_links = soup.select(selector)
                            for link in next_links:
                                if ("next" in link.get_text().lower() or 
                                    "next" in link.get("class", "")):
                                    next_page_url = link.get("href")
                                    break
                            if next_page_url:
                                break
                    
                    page_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing pagination page: {str(e)}")
                    break
            
            # Update the initial result with aggregated items
            if len(aggregated_items) > len(initial_result.get("items", [])):
                initial_result["items"] = aggregated_items
                initial_result["_metadata"]["pagination_pages"] = page_count
                initial_result["_metadata"]["total_items"] = len(aggregated_items)
                logger.info(f"Aggregated {len(aggregated_items)} items from {page_count} pages")
                
        except Exception as e:
            logger.error(f"Error handling pagination: {str(e)}")
    
    async def validate_output(self, context: PipelineContext) -> bool:
        """
        Validate the extraction output.
        
        Args:
            context: Pipeline context containing extracted data
            
        Returns:
            True if validation passes, False otherwise
        """
        output_key = self.output_key or self.name
        if not context.has_key(output_key):
            logger.warning(f"Missing output '{output_key}' in context")
            return False
            
        extracted_data = context.get(output_key)
        
        # Validate that we have some results
        if not extracted_data:
            logger.warning("Empty extraction results")
            return False
            
        # Check if we have items (for listing pages)
        if "items" in extracted_data:
            items = extracted_data["items"]
            if not items or not isinstance(items, list):
                logger.warning(f"Invalid items in extraction results: {type(items)}")
                return False
            
            # Check that we have at least some items if we expected some
            if context.has_key("extraction_hints"):
                hints = context.get("extraction_hints")
                if hints.get("content_type") in ["listing", "search_results"] and len(items) == 0:
                    logger.warning("No items extracted for listing/search page")
                    return False
        
        # For product pages, check that we have basic product info
        if context.has_key("content_type") and context.get("content_type") == "product":
            if not extracted_data.get("title") and not extracted_data.get("name"):
                logger.warning("No product title/name extracted for product page")
                return False
        
        return True