"""
Structural Analysis Stage Module

This module provides a pipeline stage for structural analysis of HTML content, 
identifying page layouts, content sections, and element relationships.
"""

import logging
from typing import Dict, Any, Optional, Union, List
from bs4 import BeautifulSoup, Tag

from core.pipeline.stages.base_stages import ProcessingStage
from core.pipeline.context import PipelineContext
from extraction.structural_analyzer import DOMStructuralAnalyzer
from core.retry_manager import RetryManager

logger = logging.getLogger(__name__)

class StructuralAnalysisStage(ProcessingStage):
    """
    Pipeline stage that analyzes HTML structure to identify layout, sections,
    and content areas to guide downstream extraction.
    
    This stage uses the DOMStructuralAnalyzer to perform comprehensive analysis
    of HTML documents, identifying content sections, layout patterns, and element
    relationships to guide extraction strategies.
    """
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the structural analysis stage with configuration.
        
        Args:
            name: Name of this stage (defaults to class name)
            config: Configuration dictionary
        """
        super().__init__(name, config)
        self.analyzer = None
        self.input_key = self.config.get("input_key", "html_content")
        self.output_key = self.config.get("output_key", "structure_analysis")
        self.url_key = self.config.get("url_key", "url")
        self.analysis_types = self.config.get("analysis_types", [
            "hierarchy", "sections", "content_type", "navigation", 
            "interactive", "result_groups", "layout"
        ])
        self.enable_error_recovery = self.config.get("enable_error_recovery", True)
        
    async def initialize(self) -> None:
        """Initialize the analyzer and stage resources."""
        if self._initialized:
            return
            
        # Create the structural analyzer
        self.analyzer = DOMStructuralAnalyzer()
        
        # Initialize the analyzer
        analyzer_config = self.config.get("analyzer_config", {})
        self.analyzer.initialize(analyzer_config)
        
        await super().initialize()
        logger.debug(f"{self.name} initialized with analyzer")
        
    async def cleanup(self) -> None:
        """Clean up resources used by this stage."""
        if self.analyzer and self.analyzer.is_initialized:
            self.analyzer.shutdown()
            
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
        
    # Create a retry manager for retrying operations
    retry_manager = RetryManager()
    
    async def transform_data(self, data: Dict[str, Any], context: PipelineContext) -> Optional[Dict[str, Any]]:
        """
        Perform structural analysis on HTML content.
        
        Args:
            data: Input data (not used, we get data from context)
            context: Pipeline context containing HTML content
            
        Returns:
            Dictionary containing analysis results or None if processing fails
        """
        try:
            if not self.analyzer:
                self.analyzer = DOMStructuralAnalyzer()
                analyzer_config = self.config.get("analyzer_config", {})
                self.analyzer.initialize(analyzer_config)
            
            # Set the context if available
            if hasattr(context, "strategy_context") and context.strategy_context:
                self.analyzer.context = context.strategy_context
            
            # Get HTML content from context
            html_content = context.get(self.input_key)
            
            # Get URL from context if available (helps with relative URL resolution)
            url = context.get(self.url_key, "")
            
            # Prepare analysis options
            options = {
                "url": url,
                "analysis_types": self.analysis_types
            }
            
            # Run analysis
            logger.info(f"Analyzing HTML structure with {len(self.analysis_types)} analysis types")
            analysis_result = self.analyzer.analyze(html_content, options)
            
            if not analysis_result.get("success", False):
                error_msg = analysis_result.get("error", "Unknown error during structural analysis")
                logger.error(f"Structural analysis failed: {error_msg}")
                context.add_error(self.name, f"Analysis failed: {error_msg}")
                return None
            
            # Add tags based on content type detection
            content_type = analysis_result.get("content_type", {}).get("primary_type", "unknown")
            if content_type != "unknown":
                logger.info(f"Detected content type: {content_type}")
                context.set("content_type", content_type)
                
                # Add confidence score
                confidence = analysis_result.get("content_type", {}).get("confidence", 0.0)
                context.set("content_type_confidence", confidence)
            
            # Tag content sections for downstream extraction
            await self._tag_content_for_extraction(analysis_result, context)
            
            # Store information about main content area for extraction
            main_content_area = analysis_result.get("content_boundaries", {}).get("boundaries", {}).get("main_content")
            if main_content_area:
                context.set("main_content_selector", main_content_area)
            
            # Set content sections for targeted extraction
            content_sections = analysis_result.get("content_sections", [])
            if content_sections:
                # Sort by importance score
                sorted_sections = sorted(content_sections, 
                                       key=lambda x: x.get("importance_score", 0), 
                                       reverse=True)
                context.set("content_sections", sorted_sections)
                
                # Set primary content section
                if sorted_sections:
                    context.set("primary_content_section", sorted_sections[0])
            
            # Set result groups for list/table detection
            result_groups = analysis_result.get("result_groups", [])
            if result_groups:
                context.set("result_groups", result_groups)
                
                # Set primary result group (if any with decent similarity)
                primary_groups = [g for g in result_groups if g.get("structural_similarity", 0) > 0.7]
                if primary_groups:
                    context.set("primary_result_group", primary_groups[0])
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in structural analysis: {str(e)}")
            context.add_error(self.name, f"Analysis error: {str(e)}")
            
            if self.enable_error_recovery:
                # Attempt partial recovery with basic analysis
                return await self._perform_basic_analysis(html_content, context)
                
            return None
    
    async def _tag_content_for_extraction(self, analysis_result: Dict[str, Any], context: PipelineContext) -> None:
        """
        Tag content areas for downstream extraction based on structural analysis.
        
        Args:
            analysis_result: Results from structural analysis
            context: Pipeline context to update with tags
        """
        # Create extraction hints dictionary if it doesn't exist
        extraction_hints = context.get("extraction_hints", {})
        
        # Add content type hints
        content_type = analysis_result.get("content_type", {}).get("primary_type", "unknown")
        extraction_hints["content_type"] = content_type
        
        # Add pagination hints
        pagination = analysis_result.get("pagination", {})
        if pagination.get("has_pagination", False):
            extraction_hints["has_pagination"] = True
            extraction_hints["pagination_elements"] = pagination.get("pagination_elements", [])
            extraction_hints["current_page"] = pagination.get("current_page")
            extraction_hints["total_pages"] = pagination.get("total_pages")
        
        # Add structure hints based on content type
        if content_type == "product":
            self._add_product_extraction_hints(analysis_result, extraction_hints)
        elif content_type == "article":
            self._add_article_extraction_hints(analysis_result, extraction_hints)
        elif content_type == "listing":
            self._add_listing_extraction_hints(analysis_result, extraction_hints)
        elif content_type == "search_results":
            self._add_search_results_extraction_hints(analysis_result, extraction_hints)
        
        # Add general structure hints
        structure_map = analysis_result.get("structure_map", {})
        if structure_map:
            extraction_hints["structure_map"] = structure_map
        
        # Update context with enhanced extraction hints
        context.set("extraction_hints", extraction_hints)
    
    def _add_product_extraction_hints(self, analysis_result: Dict[str, Any], hints: Dict[str, Any]) -> None:
        """Add product-specific extraction hints."""
        hints["target_fields"] = ["title", "price", "description", "images", "specifications", "variants"]
        
        # Check for structured data that might contain product info
        jsonld_items = analysis_result.get("json_ld", {}).get("items", [])
        for item in jsonld_items:
            if item.get("@type") == "Product":
                hints["has_structured_product_data"] = True
                break
    
    def _add_article_extraction_hints(self, analysis_result: Dict[str, Any], hints: Dict[str, Any]) -> None:
        """Add article-specific extraction hints."""
        hints["target_fields"] = ["title", "author", "date_published", "content", "category", "tags"]
        
        # Check for article metadata
        metadata = analysis_result.get("open_graph", {})
        if metadata.get("type") == "article":
            hints["has_article_metadata"] = True
    
    def _add_listing_extraction_hints(self, analysis_result: Dict[str, Any], hints: Dict[str, Any]) -> None:
        """Add listing-specific extraction hints."""
        hints["target_fields"] = ["items", "next_page", "total_items", "filters"]
        
        # Add result group information
        result_groups = analysis_result.get("result_groups", [])
        if result_groups:
            # Find the most likely listing container
            sorted_groups = sorted(result_groups, 
                                 key=lambda x: (x.get("item_count", 0), x.get("structural_similarity", 0)), 
                                 reverse=True)
            if sorted_groups:
                hints["primary_listing_container"] = sorted_groups[0].get("container_selector")
                hints["listing_item_count"] = sorted_groups[0].get("item_count", 0)
    
    def _add_search_results_extraction_hints(self, analysis_result: Dict[str, Any], hints: Dict[str, Any]) -> None:
        """Add search results-specific extraction hints."""
        hints["target_fields"] = ["items", "next_page", "total_results", "filters", "sorting"]
        
        # Add form information for search refinement
        interactive = analysis_result.get("interactive_elements", {})
        if interactive.get("forms"):
            search_forms = [f for f in interactive.get("forms", []) 
                          if f.get("purpose") == "search"]
            if search_forms:
                hints["has_search_form"] = True
                hints["search_form_selector"] = search_forms[0].get("selector")
    
    async def _perform_basic_analysis(self, html_content: Any, context: PipelineContext) -> Dict[str, Any]:
        """
        Perform basic analysis as a fallback when full analysis fails.
        
        Args:
            html_content: HTML content to analyze
            context: Pipeline context
            
        Returns:
            Basic analysis results
        """
        try:
            # Get HTML service if available in context
            html_service = None
            if hasattr(context, "strategy_context") and context.strategy_context:
                try:
                    html_service = context.strategy_context.get_service("html_service")
                except Exception:
                    pass
            
            # Parse HTML if needed
            if isinstance(html_content, str):
                soup = BeautifulSoup(html_content, "lxml")
            elif isinstance(html_content, (BeautifulSoup, Tag)):
                soup = html_content
            else:
                return {"success": False, "error": "Invalid HTML content type"}
            
            # Basic analysis results
            basic_results = {
                "success": True,
                "page_title": self._extract_title(soup),
                "content_type": {
                    "primary_type": "unknown",
                    "confidence": 0.5
                },
                "_metadata": {
                    "analyzer": "BasicFallbackAnalyzer",
                    "analysis_types": ["basic"]
                }
            }
            
            # Try to determine content type based on basic signals
            content_type, confidence = self._basic_content_type_detection(soup)
            basic_results["content_type"]["primary_type"] = content_type
            basic_results["content_type"]["confidence"] = confidence
            
            # Find main content area
            main_content = soup.find("main") or self._find_main_content(soup)
            if main_content:
                basic_results["content_boundaries"] = {
                    "boundaries": {
                        "main_content": self._get_element_path(main_content, html_service)
                    }
                }
            
            # Set content type in context
            context.set("content_type", content_type)
            context.set("content_type_confidence", confidence)
            
            return basic_results
            
        except Exception as e:
            logger.error(f"Error in basic fallback analysis: {str(e)}")
            return {
                "success": False,
                "error": f"Basic analysis failed: {str(e)}",
                "_metadata": {
                    "analyzer": "BasicFallbackAnalyzer"
                }
            }
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title from HTML."""
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            return title_tag.string.strip()
            
        # Try h1 as fallback
        h1_tag = soup.find("h1")
        if h1_tag:
            return h1_tag.get_text(strip=True)
            
        return ""
    
    def _basic_content_type_detection(self, soup: BeautifulSoup) -> tuple:
        """Perform basic content type detection based on simple signals."""
        # Check for product signals
        product_signals = 0
        if soup.select(".product, #product, [itemtype*=Product], .price, .add-to-cart, .buy-now"):
            product_signals += 1
        if soup.find(string=lambda t: t and "Add to Cart" in t):
            product_signals += 1
        
        # Check for article signals
        article_signals = 0
        if soup.find("article") or soup.select("[itemtype*=Article], .article, .post, .blog-post"):
            article_signals += 1
        if soup.select(".author, .byline, .published-date, [itemprop=datePublished]"):
            article_signals += 1
        
        # Check for listing signals
        listing_signals = 0
        product_lists = soup.select(".products, .product-list, .items, .search-results")
        if product_lists:
            listing_signals += 1
            # Check if list items have similar structure
            if any(len(pl.find_all(class_=True)) > 3 for pl in product_lists):
                listing_signals += 1
        
        # Determine content type based on signals
        if product_signals > max(article_signals, listing_signals):
            return "product", 0.6
        elif article_signals > max(product_signals, listing_signals):
            return "article", 0.6
        elif listing_signals > 0:
            return "listing", 0.6
        
        return "unknown", 0.4
    
    def _find_main_content(self, soup: BeautifulSoup) -> Optional[Tag]:
        """Find the main content area using common patterns."""
        # Try common content containers
        for selector in ["#content", ".content", "#main", ".main", "article", ".post", ".product"]:
            content = soup.select_one(selector)
            if content:
                return content
        
        # Try the largest div with substantial content
        candidates = []
        for div in soup.find_all("div", class_=True):
            text_length = len(div.get_text(strip=True))
            if text_length > 200:  # Arbitrary threshold for "substantial" content
                candidates.append((div, text_length))
        
        if candidates:
            # Sort by text length, descending
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        
        return None
    
    def _get_element_path(self, element: Tag, html_service=None) -> str:
        """
        Get a CSS selector path for an element.
        
        Args:
            element: The HTML element
            html_service: Optional HTML service for better selector generation
            
        Returns:
            CSS selector string
        """
        if html_service:
            try:
                return html_service.generate_selector(element)
            except Exception:
                pass
        
        # Basic selector generation fallback
        if element.get("id"):
            return f"#{element['id']}"
        
        # Class-based selector
        if element.get("class"):
            cls = ".".join(element["class"])
            return f"{element.name}.{cls}"
        
        # Position-based selector (less reliable)
        parents = []
        current = element
        while current and current.name != "html":
            parent = current.parent
            if parent:
                siblings = [s for s in parent.find_all(current.name, recursive=False)]
                if len(siblings) > 1:
                    index = siblings.index(current) + 1
                    parents.append(f"{current.name}:nth-of-type({index})")
                else:
                    parents.append(current.name)
            current = parent
        
        return " > ".join(reversed(parents))