"""
HTML Processing Module.

This module provides specialized processing stages for HTML document handling,
including cleaning, content extraction, and element selection.
"""

import copy
import logging
import re
import time
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Pattern

import lxml.html
from bs4 import BeautifulSoup
from lxml import etree
from lxml.html import HtmlElement, fromstring, tostring
from lxml.html.clean import Cleaner

from core.pipeline.stages.base_stages import ProcessingStage
from core.pipeline.context import PipelineContext
from core.pipeline.dto import PipelineRequest, PipelineResponse, ResponseStatus
from core.service_registry import ServiceRegistry
from core.html_service import HTMLService


class CleaningStrategy(Enum):
    """Enumeration of HTML cleaning strategies."""
    LIGHT = auto()  # Remove minimal elements (scripts, styles)
    MODERATE = auto()  # Remove navigation, ads, and other non-content elements
    AGGRESSIVE = auto()  # Remove everything except main content
    CUSTOM = auto()  # Use custom configuration


class ExtractionAlgorithm(Enum):
    """Enumeration of content extraction algorithms."""
    DENSITY_BASED = auto()  # Extract based on text-to-code ratio
    SEMANTIC_BASED = auto()  # Extract based on semantic markup (article, main)
    POSITIONAL_BASED = auto()  # Extract based on position in document
    CUSTOM_SELECTOR = auto()  # Extract using custom CSS/XPath selector


class SelectorStrategy(Enum):
    """Enumeration of element selection strategies."""
    FIRST_MATCH = auto()  # Return first matching element
    ALL_MATCHES = auto()  # Return all matching elements
    PRIORITY_BASED = auto()  # Try selectors in order of priority
    COMPOSITE = auto()  # Combine results from multiple selectors


class HTMLCleaningStage(ProcessingStage):
    """
    HTML Cleaning Stage for removing unwanted elements and attributes from HTML documents.
    
    This stage performs cleaning operations on HTML content, such as removing scripts,
    styles, comments, and other unwanted elements based on configurable strategies.
    
    Features:
    - Multiple cleaning strategies (light, moderate, aggressive)
    - Custom element and attribute filtering
    - Content sanitization for security
    - Metrics for removed elements
    
    Configuration:
    - strategy: The cleaning strategy to use
    - remove_elements: List of element tags to remove
    - safe_attrs: List of attributes to keep (others removed)
    - allow_tags: List of tags to allow (others removed)
    - remove_unknown_tags: Whether to remove unknown tags
    - kill_tags: List of tags to remove including content
    - embedded_content: Whether to keep embedded content (iframes, etc)
    - sandbox_external: Whether to sandbox external elements
    """
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize HTML cleaning stage.
        
        Args:
            name: Name of the stage, defaults to class name
            config: Configuration parameters
        """
        super().__init__(name, config)
        
        # Load strategy
        strategy_name = self.config.get("strategy", "MODERATE")
        try:
            self.strategy = CleaningStrategy[strategy_name] if isinstance(strategy_name, str) else strategy_name
        except (KeyError, TypeError):
            self.strategy = CleaningStrategy.MODERATE
            self.logger.warning(f"Invalid cleaning strategy: {strategy_name}, using MODERATE")
        
        # Configure cleaning options based on strategy
        self._configure_cleaning_options()
        
        # Initialize HTML service if needed
        self._html_service = None
        
        # Initialize metrics
        self.metrics = {
            "elements_removed": 0,
            "attributes_removed": 0,
            "comments_removed": 0,
            "scripts_removed": 0,
            "styles_removed": 0,
            "processing_time": 0
        }
    
    async def process_request(self, request: PipelineRequest, context: PipelineContext) -> Optional[PipelineResponse]:
        """
        Process the HTML document by cleaning it according to the configured strategy.
        
        Args:
            request: The pipeline request containing HTML content
            context: The shared pipeline context
            
        Returns:
            PipelineResponse with cleaned HTML
        """
        # Ensure we have HTML content to process
        if not request.params or "html" not in request.params:
            self.logger.error("No HTML content found in request")
            return PipelineResponse(
                status=ResponseStatus.ERROR,
                error_message="No HTML content found in request",
                source=request.source
            )
        
        # Get the HTML content
        html_content = request.params["html"]
        
        # Ensure HTML content is a string
        if not isinstance(html_content, str):
            self.logger.error("HTML content must be a string")
            return PipelineResponse(
                status=ResponseStatus.ERROR,
                error_message="HTML content must be a string",
                source=request.source
            )
        
        # Initialize the HTML service if it's not already available
        if self._html_service is None:
            try:
                registry = ServiceRegistry()
                self._html_service = registry.get_service("html_service")
            except Exception as e:
                self.logger.warning(f"Could not get HTML service: {str(e)}. Using internal implementation.")
                self._html_service = None
        
        # Track processing time
        start_time = time.time()
        
        try:
            # Use HTML service if available, otherwise use internal implementation
            if self._html_service:
                cleaned_html, metrics = await self._clean_with_service(html_content)
                self.metrics.update(metrics)
            else:
                cleaned_html, metrics = self._clean_html(html_content)
                self.metrics.update(metrics)
            
            # Add processing time to metrics
            processing_time = time.time() - start_time
            self.metrics["processing_time"] = processing_time
            
            # Create response with cleaned HTML
            return PipelineResponse(
                status=ResponseStatus.SUCCESS,
                data={
                    "html": cleaned_html,
                    "original_size": len(html_content),
                    "cleaned_size": len(cleaned_html)
                },
                source=request.source,
                metadata={
                    "cleaning_metrics": self.metrics,
                    "strategy": self.strategy.name
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error cleaning HTML: {str(e)}")
            return PipelineResponse(
                status=ResponseStatus.ERROR,
                error_message=f"Error cleaning HTML: {str(e)}",
                source=request.source
            )
    
    async def _clean_with_service(self, html_content: str) -> Tuple[str, Dict[str, Any]]:
        """
        Clean HTML using the HTML service.
        
        Args:
            html_content: The HTML content to clean
            
        Returns:
            Tuple of (cleaned HTML, metrics)
        """
        # Convert our cleaning options to HTML service format
        service_options = {
            "strategy": self.strategy.name,
            "remove_elements": self.remove_elements,
            "safe_attrs": self.safe_attrs,
            "allow_tags": self.allow_tags,
            "kill_tags": self.kill_tags,
            "remove_unknown_tags": self.remove_unknown_tags
        }
        
        # Call HTML service to clean the content
        result = await self._html_service.clean_html(html_content, service_options)
        
        return result["cleaned_html"], result["metrics"]
    
    def _clean_html(self, html_content: str) -> Tuple[str, Dict[str, Any]]:
        """
        Clean HTML using internal implementation.
        
        Args:
            html_content: The HTML content to clean
            
        Returns:
            Tuple of (cleaned HTML, metrics)
        """
        metrics = {
            "elements_removed": 0,
            "attributes_removed": 0,
            "comments_removed": 0,
            "scripts_removed": 0,
            "styles_removed": 0
        }
        
        try:
            # Parse the HTML using lxml
            doc = fromstring(html_content)
            
            # Count elements before cleaning
            original_elements = len(doc.xpath("//*"))
            original_scripts = len(doc.xpath("//script"))
            original_styles = len(doc.xpath("//style"))
            original_comments = len(doc.xpath("//comment()"))
            
            # Create and configure the cleaner
            cleaner = Cleaner(
                scripts=self.remove_scripts,
                javascript=self.remove_javascript,
                comments=self.remove_comments,
                style=self.remove_styles,
                links=self.remove_links,
                meta=self.remove_meta,
                page_structure=self.remove_page_structure,
                processing_instructions=True,
                embedded=not self.allow_embedded,
                frames=not self.allow_frames,
                forms=not self.allow_forms,
                annoying_tags=self.remove_annoying_tags,
                remove_tags=self.remove_elements,
                kill_tags=self.kill_tags,
                allow_tags=self.allow_tags if self.allow_tags else None,
                remove_unknown_tags=self.remove_unknown_tags,
                safe_attrs_only=bool(self.safe_attrs),
                safe_attrs=set(self.safe_attrs) if self.safe_attrs else None
            )
            
            # Clean the document
            cleaned_doc = cleaner.clean_html(doc)
            
            # Count elements after cleaning
            remaining_elements = len(cleaned_doc.xpath("//*"))
            remaining_scripts = len(cleaned_doc.xpath("//script"))
            remaining_styles = len(cleaned_doc.xpath("//style"))
            remaining_comments = len(cleaned_doc.xpath("//comment()"))
            
            # Set metrics
            metrics["elements_removed"] = original_elements - remaining_elements
            metrics["scripts_removed"] = original_scripts - remaining_scripts
            metrics["styles_removed"] = original_styles - remaining_styles
            metrics["comments_removed"] = original_comments - remaining_comments
            
            # Convert back to string
            cleaned_html = tostring(cleaned_doc, encoding="unicode", method="html")
            
            # Optionally apply additional cleaning steps
            if self.config.get("extra_whitespace_cleanup", False):
                cleaned_html = re.sub(r'\s+', ' ', cleaned_html)
                cleaned_html = re.sub(r'>\s+<', '><', cleaned_html)
            
            return cleaned_html, metrics
            
        except Exception as e:
            self.logger.error(f"Error during HTML cleaning: {str(e)}")
            # Return original content on error
            return html_content, metrics
    
    def _configure_cleaning_options(self) -> None:
        """
        Configure cleaning options based on the selected strategy.
        """
        # Default settings
        self.remove_scripts = True
        self.remove_javascript = True
        self.remove_comments = True
        self.remove_styles = True
        self.remove_links = False
        self.remove_meta = False
        self.remove_page_structure = False
        self.allow_embedded = False
        self.allow_frames = False
        self.allow_forms = True
        self.remove_annoying_tags = True
        self.remove_elements = []
        self.kill_tags = []
        self.allow_tags = []
        self.safe_attrs = []
        self.remove_unknown_tags = False
        
        # Configure based on strategy
        if self.strategy == CleaningStrategy.LIGHT:
            # Light cleaning: remove minimal elements
            self.remove_links = False
            self.remove_meta = False
            self.allow_embedded = True
            self.allow_frames = True
            self.allow_forms = True
            self.remove_annoying_tags = False
            self.kill_tags = ["script", "style", "noscript"]
            
        elif self.strategy == CleaningStrategy.MODERATE:
            # Moderate cleaning: remove navigation, ads, etc.
            self.remove_links = False
            self.remove_meta = True
            self.allow_embedded = False
            self.allow_frames = False
            self.allow_forms = True
            self.remove_annoying_tags = True
            self.kill_tags = ["script", "style", "noscript", "iframe", "object", "embed"]
            self.remove_elements = ["nav", "header", "footer", "aside", "banner", "ads", "figure"]
            
        elif self.strategy == CleaningStrategy.AGGRESSIVE:
            # Aggressive cleaning: keep only main content
            self.remove_links = True
            self.remove_meta = True
            self.remove_page_structure = True
            self.allow_embedded = False
            self.allow_frames = False
            self.allow_forms = False
            self.remove_annoying_tags = True
            self.kill_tags = ["script", "style", "noscript", "iframe", "object", "embed"]
            self.remove_elements = [
                "nav", "header", "footer", "aside", "banner", "ads", "figure",
                "meta", "link", "style", "video", "audio"
            ]
            self.safe_attrs = [
                "src", "href", "alt", "title", "class", "id", "data-src", "data-alt",
                "width", "height", "colspan", "rowspan"
            ]
            
        elif self.strategy == CleaningStrategy.CUSTOM:
            # Custom cleaning: use configuration
            self.remove_scripts = self.config.get("remove_scripts", True)
            self.remove_javascript = self.config.get("remove_javascript", True)
            self.remove_comments = self.config.get("remove_comments", True)
            self.remove_styles = self.config.get("remove_styles", True)
            self.remove_links = self.config.get("remove_links", False)
            self.remove_meta = self.config.get("remove_meta", False)
            self.remove_page_structure = self.config.get("remove_page_structure", False)
            self.allow_embedded = self.config.get("allow_embedded", False)
            self.allow_frames = self.config.get("allow_frames", False)
            self.allow_forms = self.config.get("allow_forms", True)
            self.remove_annoying_tags = self.config.get("remove_annoying_tags", True)
            self.remove_elements = self.config.get("remove_elements", [])
            self.kill_tags = self.config.get("kill_tags", [])
            self.allow_tags = self.config.get("allow_tags", [])
            self.safe_attrs = self.config.get("safe_attrs", [])
            self.remove_unknown_tags = self.config.get("remove_unknown_tags", False)
        
        # Apply any overrides from configuration
        for key in [
            "remove_scripts", "remove_javascript", "remove_comments", "remove_styles",
            "remove_links", "remove_meta", "remove_page_structure", "allow_embedded",
            "allow_frames", "allow_forms", "remove_annoying_tags", "remove_elements",
            "kill_tags", "allow_tags", "safe_attrs", "remove_unknown_tags"
        ]:
            if key in self.config:
                setattr(self, key, self.config[key])


class ContentExtractionStage(ProcessingStage):
    """
    Content Extraction Stage for extracting main content from HTML documents.
    
    This stage extracts the main content from HTML documents using various algorithms
    such as text density analysis, semantic markup analysis, or custom selectors.
    
    Features:
    - Multiple extraction algorithms
    - Support for custom selectors
    - Quality metrics for extracted content
    - Handling of various content structures
    
    Configuration:
    - algorithm: The extraction algorithm to use
    - selectors: List of CSS or XPath selectors to try
    - min_content_length: Minimum content length to consider valid
    - require_heading: Whether main content should contain a heading
    - quality_metrics: Whether to include content quality metrics
    - fallback_to_body: Whether to fallback to body if no content found
    """
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize content extraction stage.
        
        Args:
            name: Name of the stage, defaults to class name
            config: Configuration parameters
        """
        super().__init__(name, config)
        
        # Load algorithm
        algorithm_name = self.config.get("algorithm", "DENSITY_BASED")
        try:
            self.algorithm = ExtractionAlgorithm[algorithm_name] if isinstance(algorithm_name, str) else algorithm_name
        except (KeyError, TypeError):
            self.algorithm = ExtractionAlgorithm.DENSITY_BASED
            self.logger.warning(f"Invalid extraction algorithm: {algorithm_name}, using DENSITY_BASED")
        
        # Configure extraction options
        self.selectors = self.config.get("selectors", [])
        self.min_content_length = self.config.get("min_content_length", 100)
        self.require_heading = self.config.get("require_heading", False)
        self.quality_metrics = self.config.get("quality_metrics", True)
        self.fallback_to_body = self.config.get("fallback_to_body", True)
        
        # Add semantic selectors for common content containers
        self.semantic_selectors = [
            "article", "main", "[role=main]", "#main", "#content", "#main-content",
            ".main", ".content", ".main-content", ".post-content", ".entry-content"
        ]
        
        if not self.selectors:
            self.selectors = self.semantic_selectors
        
        # Initialize HTML service if needed
        self._html_service = None
        
        # Initialize metrics
        self.metrics = {
            "content_length": 0,
            "extraction_method": "",
            "content_to_code_ratio": 0,
            "has_heading": False,
            "processing_time": 0
        }
    
    async def process_request(self, request: PipelineRequest, context: PipelineContext) -> Optional[PipelineResponse]:
        """
        Process the HTML document by extracting main content.
        
        Args:
            request: The pipeline request containing HTML content
            context: The shared pipeline context
            
        Returns:
            PipelineResponse with extracted content
        """
        # Ensure we have HTML content to process
        if not request.params or "html" not in request.params:
            self.logger.error("No HTML content found in request")
            return PipelineResponse(
                status=ResponseStatus.ERROR,
                error_message="No HTML content found in request",
                source=request.source
            )
        
        # Get the HTML content
        html_content = request.params["html"]
        
        # Ensure HTML content is a string
        if not isinstance(html_content, str):
            self.logger.error("HTML content must be a string")
            return PipelineResponse(
                status=ResponseStatus.ERROR,
                error_message="HTML content must be a string",
                source=request.source
            )
        
        # Initialize the HTML service if it's not already available
        if self._html_service is None:
            try:
                registry = ServiceRegistry()
                self._html_service = registry.get_service("html_service")
            except Exception as e:
                self.logger.warning(f"Could not get HTML service: {str(e)}. Using internal implementation.")
                self._html_service = None
        
        # Track processing time
        start_time = time.time()
        
        try:
            # Use HTML service if available, otherwise use internal implementation
            if self._html_service and hasattr(self._html_service, "extract_main_content"):
                main_content, metrics = await self._extract_with_service(html_content)
                self.metrics.update(metrics)
            else:
                main_content, metrics = self._extract_main_content(html_content)
                self.metrics.update(metrics)
            
            # Add processing time to metrics
            processing_time = time.time() - start_time
            self.metrics["processing_time"] = processing_time
            
            # Create response with extracted content
            return PipelineResponse(
                status=ResponseStatus.SUCCESS,
                data={
                    "html": main_content,
                    "original_size": len(html_content),
                    "content_size": len(main_content),
                    "content_ratio": len(main_content) / len(html_content) if len(html_content) > 0 else 0
                },
                source=request.source,
                metadata={
                    "extraction_metrics": self.metrics,
                    "algorithm": self.algorithm.name
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error extracting main content: {str(e)}")
            return PipelineResponse(
                status=ResponseStatus.ERROR,
                error_message=f"Error extracting main content: {str(e)}",
                source=request.source
            )
    
    async def _extract_with_service(self, html_content: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extract main content using the HTML service.
        
        Args:
            html_content: The HTML content to process
            
        Returns:
            Tuple of (main content, metrics)
        """
        # Convert our extraction options to HTML service format
        service_options = {
            "algorithm": self.algorithm.name,
            "selectors": self.selectors,
            "min_content_length": self.min_content_length,
            "require_heading": self.require_heading,
            "quality_metrics": self.quality_metrics,
            "fallback_to_body": self.fallback_to_body
        }
        
        # Call HTML service to extract the content
        result = await self._html_service.extract_main_content(html_content, service_options)
        
        return result["content"], result["metrics"]
    
    def _extract_main_content(self, html_content: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extract main content using internal implementation.
        
        Args:
            html_content: The HTML content to process
            
        Returns:
            Tuple of (main content, metrics)
        """
        metrics = {
            "content_length": 0,
            "extraction_method": "",
            "content_to_code_ratio": 0,
            "has_heading": False
        }
        
        try:
            # Parse the HTML using lxml
            doc = fromstring(html_content)
            
            # Extract main content based on selected algorithm
            if self.algorithm == ExtractionAlgorithm.CUSTOM_SELECTOR:
                content_element, method = self._extract_with_custom_selectors(doc)
            elif self.algorithm == ExtractionAlgorithm.SEMANTIC_BASED:
                content_element, method = self._extract_with_semantic_analysis(doc)
            elif self.algorithm == ExtractionAlgorithm.POSITIONAL_BASED:
                content_element, method = self._extract_with_positional_analysis(doc)
            else:  # DENSITY_BASED or fallback
                content_element, method = self._extract_with_density_analysis(doc)
            
            # If no content found, try fallback methods
            if content_element is None and self.fallback_to_body:
                body_element = doc.xpath("//body")
                if body_element:
                    content_element = body_element[0]
                    method = "fallback_to_body"
            
            # If still no content, return empty string with metrics
            if content_element is None:
                self.logger.warning("No main content found in document")
                return "", metrics
            
            # Check if content has a heading
            has_heading = bool(content_element.xpath(".//h1|.//h2|.//h3"))
            
            # Convert content element to string
            content_html = tostring(content_element, encoding="unicode", method="html")
            
            # Calculate metrics
            content_text_length = len(" ".join(content_element.xpath(".//text()")))
            content_html_length = len(content_html)
            html_length = len(html_content)
            
            metrics["content_length"] = content_text_length
            metrics["extraction_method"] = method
            metrics["content_to_code_ratio"] = content_html_length / html_length if html_length > 0 else 0
            metrics["has_heading"] = has_heading
            
            # Validate content based on requirements
            if content_text_length < self.min_content_length:
                self.logger.warning(f"Extracted content length ({content_text_length}) is below minimum ({self.min_content_length})")
                if self.fallback_to_body:
                    body_element = doc.xpath("//body")
                    if body_element:
                        content_element = body_element[0]
                        content_html = tostring(content_element, encoding="unicode", method="html")
                        metrics["extraction_method"] = "fallback_to_body"
            
            if self.require_heading and not has_heading:
                self.logger.warning("Extracted content doesn't contain a heading")
            
            return content_html, metrics
            
        except Exception as e:
            self.logger.error(f"Error during main content extraction: {str(e)}")
            # Return original content on error
            return html_content, metrics
    
    def _extract_with_custom_selectors(self, doc: HtmlElement) -> Tuple[Optional[HtmlElement], str]:
        """
        Extract content using custom selectors.
        
        Args:
            doc: The HTML document
            
        Returns:
            Tuple of (content element, method used)
        """
        for selector in self.selectors:
            try:
                # Check if it's an XPath expression
                if selector.startswith("//") or selector.startswith("./"):
                    elements = doc.xpath(selector)
                else:
                    # CSS selector
                    elements = doc.cssselect(selector)
                
                if elements:
                    return elements[0], f"custom_selector:{selector}"
            except Exception as e:
                self.logger.warning(f"Error with selector '{selector}': {str(e)}")
        
        return None, ""
    
    def _extract_with_semantic_analysis(self, doc: HtmlElement) -> Tuple[Optional[HtmlElement], str]:
        """
        Extract content based on semantic markup.
        
        Args:
            doc: The HTML document
            
        Returns:
            Tuple of (content element, method used)
        """
        # Try semantic elements
        for selector in self.semantic_selectors:
            try:
                if selector.startswith("[") or selector.startswith(".") or selector.startswith("#"):
                    # CSS selector
                    elements = doc.cssselect(selector)
                else:
                    # Tag name
                    elements = doc.xpath(f"//{selector}")
                
                if elements:
                    return elements[0], f"semantic:{selector}"
            except Exception:
                pass
        
        return None, ""
    
    def _extract_with_positional_analysis(self, doc: HtmlElement) -> Tuple[Optional[HtmlElement], str]:
        """
        Extract content based on positional analysis.
        
        Args:
            doc: The HTML document
            
        Returns:
            Tuple of (content element, method used)
        """
        # Get all div elements
        divs = doc.xpath("//div")
        
        # No divs found
        if not divs:
            return None, ""
        
        # Find divs in the middle of the document (often contains main content)
        middle_index = len(divs) // 2
        candidates = divs[max(0, middle_index - 3):min(len(divs), middle_index + 3)]
        
        best_div = None
        max_text_length = 0
        
        for div in candidates:
            # Calculate text length
            text_content = " ".join(div.xpath(".//text()"))
            text_length = len(text_content)
            
            if text_length > max_text_length:
                max_text_length = text_length
                best_div = div
        
        if best_div is not None:
            return best_div, "positional_analysis"
        
        return None, ""
    
    def _extract_with_density_analysis(self, doc: HtmlElement) -> Tuple[Optional[HtmlElement], str]:
        """
        Extract content based on text density analysis.
        
        Args:
            doc: The HTML document
            
        Returns:
            Tuple of (content element, method used)
        """
        # Get all potential content containers
        containers = doc.xpath("//div | //section | //article | //main")
        
        best_container = None
        best_score = 0
        
        for container in containers:
            # Skip small containers
            if len(container.xpath(".//*")) < 5:
                continue
            
            # Calculate text density
            text_content = " ".join(container.xpath(".//text()"))
            text_length = len(text_content)
            
            # Calculate link density
            links_text = " ".join(container.xpath(".//a//text()"))
            links_length = len(links_text)
            
            link_density = links_length / text_length if text_length > 0 else 1
            
            # Calculate tag density
            tag_count = len(container.xpath(".//*"))
            tag_density = tag_count / text_length if text_length > 0 else 1
            
            # Calculate content score
            # Higher text length, lower link density, and lower tag density is better
            score = text_length * (1 - link_density) * (1 - min(1, tag_density / 10))
            
            if score > best_score:
                best_score = score
                best_container = container
        
        if best_container is not None:
            return best_container, "density_analysis"
        
        return None, ""


class ElementSelectionStage(ProcessingStage):
    """
    Element Selection Stage for extracting specific elements from HTML documents.
    
    This stage extracts specific elements from HTML documents using CSS or XPath selectors,
    with support for dynamic selector generation and priority-based selection.
    
    Features:
    - Selection using CSS or XPath selectors
    - Dynamic selector generation
    - Priority-based selection strategies
    - Element filtering and validation
    - Selection accuracy metrics
    
    Configuration:
    - selectors: List of CSS or XPath selectors to use
    - strategy: The selection strategy to use
    - attribute_inclusion: Optional list of attributes to include
    - element_filters: Optional filters to apply to matched elements
    - min_elements: Minimum number of elements to consider successful
    - max_elements: Maximum number of elements to return
    - include_parent_context: Whether to include parent context with elements
    - extract_attributes: Whether to extract only attributes (no HTML)
    """
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize element selection stage.
        
        Args:
            name: Name of the stage, defaults to class name
            config: Configuration parameters
        """
        super().__init__(name, config)
        
        # Load strategy
        strategy_name = self.config.get("strategy", "ALL_MATCHES")
        try:
            self.strategy = SelectorStrategy[strategy_name] if isinstance(strategy_name, str) else strategy_name
        except (KeyError, TypeError):
            self.strategy = SelectorStrategy.ALL_MATCHES
            self.logger.warning(f"Invalid selector strategy: {strategy_name}, using ALL_MATCHES")
        
        # Configure selection options
        self.selectors = self.config.get("selectors", [])
        self.attribute_inclusion = self.config.get("attribute_inclusion", [])
        self.element_filters = self.config.get("element_filters", {})
        self.min_elements = self.config.get("min_elements", 1)
        self.max_elements = self.config.get("max_elements", 100)
        self.include_parent_context = self.config.get("include_parent_context", False)
        self.extract_attributes = self.config.get("extract_attributes", False)
        
        # Initialize HTML service if needed
        self._html_service = None
        
        # Initialize metrics
        self.metrics = {
            "elements_found": 0,
            "elements_filtered": 0,
            "selectors_used": [],
            "extraction_success": False,
            "processing_time": 0
        }
    
    async def process_request(self, request: PipelineRequest, context: PipelineContext) -> Optional[PipelineResponse]:
        """
        Process the HTML document by selecting specific elements.
        
        Args:
            request: The pipeline request containing HTML content
            context: The shared pipeline context
            
        Returns:
            PipelineResponse with selected elements
        """
        # Ensure we have HTML content to process
        if not request.data or "html" not in request.data:
            self.logger.error("No HTML content found in request")
            return PipelineResponse(
                status=ResponseStatus.ERROR,
                error_message="No HTML content found in request",
                source=request.source
            )
        
        # Get the HTML content
        html_content = request.data["html"]
        
        # Ensure HTML content is a string
        if not isinstance(html_content, str):
            self.logger.error("HTML content must be a string")
            return PipelineResponse(
                status=ResponseStatus.ERROR,
                error_message="HTML content must be a string",
                source=request.source
            )
        
        # Get selectors from request if provided
        if "selectors" in request.params:
            self.selectors = request.params["selectors"]
        
        # Ensure we have selectors to use
        if not self.selectors:
            self.logger.error("No selectors specified")
            return PipelineResponse(
                status=ResponseStatus.ERROR,
                error_message="No selectors specified",
                source=request.source
            )
        
        # Initialize the HTML service if it's not already available
        if self._html_service is None:
            try:
                registry = ServiceRegistry()
                self._html_service = registry.get_service("html_service")
            except Exception as e:
                self.logger.warning(f"Could not get HTML service: {str(e)}. Using internal implementation.")
                self._html_service = None
        
        # Track processing time
        start_time = time.time()
        
        try:
            # Use HTML service if available, otherwise use internal implementation
            if self._html_service and hasattr(self._html_service, "select_elements"):
                selected_elements, metrics = await self._select_with_service(html_content)
                self.metrics.update(metrics)
            else:
                selected_elements, metrics = self._select_elements(html_content)
                self.metrics.update(metrics)
            
            # Add processing time to metrics
            processing_time = time.time() - start_time
            self.metrics["processing_time"] = processing_time
            
            # Create response with selected elements
            return PipelineResponse(
                status=ResponseStatus.SUCCESS if selected_elements else ResponseStatus.NOT_FOUND,
                data={
                    "elements": selected_elements,
                    "count": len(selected_elements)
                },
                source=request.source,
                metadata={
                    "selection_metrics": self.metrics,
                    "strategy": self.strategy.name
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error selecting elements: {str(e)}")
            return PipelineResponse(
                status=ResponseStatus.ERROR,
                error_message=f"Error selecting elements: {str(e)}",
                source=request.source
            )
    
    async def _select_with_service(self, html_content: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Select elements using the HTML service.
        
        Args:
            html_content: The HTML content to process
            
        Returns:
            Tuple of (selected elements, metrics)
        """
        # Convert our selection options to HTML service format
        service_options = {
            "strategy": self.strategy.name,
            "selectors": self.selectors,
            "attribute_inclusion": self.attribute_inclusion,
            "element_filters": self.element_filters,
            "min_elements": self.min_elements,
            "max_elements": self.max_elements,
            "include_parent_context": self.include_parent_context,
            "extract_attributes": self.extract_attributes
        }
        
        # Call HTML service to select elements
        result = await self._html_service.select_elements(html_content, service_options)
        
        return result["elements"], result["metrics"]
    
    def _select_elements(self, html_content: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Select elements using internal implementation.
        
        Args:
            html_content: The HTML content to process
            
        Returns:
            Tuple of (selected elements, metrics)
        """
        metrics = {
            "elements_found": 0,
            "elements_filtered": 0,
            "selectors_used": [],
            "extraction_success": False
        }
        
        try:
            # Parse the HTML using lxml
            doc = fromstring(html_content)
            
            # Select elements based on strategy
            if self.strategy == SelectorStrategy.FIRST_MATCH:
                elements, used_selectors = self._select_first_match(doc)
            elif self.strategy == SelectorStrategy.PRIORITY_BASED:
                elements, used_selectors = self._select_priority_based(doc)
            elif self.strategy == SelectorStrategy.COMPOSITE:
                elements, used_selectors = self._select_composite(doc)
            else:  # ALL_MATCHES or fallback
                elements, used_selectors = self._select_all_matches(doc)
            
            # Update metrics
            metrics["elements_found"] = len(elements)
            metrics["selectors_used"] = used_selectors
            
            # Apply filters if specified
            if self.element_filters:
                original_count = len(elements)
                elements = self._apply_filters(elements)
                metrics["elements_filtered"] = original_count - len(elements)
            
            # Limit number of elements if needed
            if self.max_elements and len(elements) > self.max_elements:
                elements = elements[:self.max_elements]
            
            # Check if we found the minimum required elements
            extraction_success = len(elements) >= self.min_elements
            metrics["extraction_success"] = extraction_success
            
            # Convert elements to the required format
            result_elements = []
            for element in elements:
                if self.extract_attributes:
                    # Extract only attributes
                    element_dict = {"attributes": {}}
                    for attr, value in element.attrib.items():
                        if not self.attribute_inclusion or attr in self.attribute_inclusion:
                            element_dict["attributes"][attr] = value
                    
                    # Add text content as a special attribute
                    element_dict["text"] = " ".join(element.xpath(".//text()")).strip()
                    element_dict["tag"] = element.tag
                    result_elements.append(element_dict)
                else:
                    # Extract HTML
                    element_html = tostring(element, encoding="unicode", method="html")
                    
                    # Include parent context if requested
                    if self.include_parent_context and element.getparent() is not None:
                        parent_path = element.getroottree().getpath(element.getparent())
                        siblings = element.getparent().getchildren()
                        element_index = siblings.index(element) if element in siblings else -1
                        
                        context = {
                            "parent_path": parent_path,
                            "element_index": element_index,
                            "sibling_count": len(siblings)
                        }
                        
                        result_elements.append({
                            "html": element_html,
                            "tag": element.tag,
                            "context": context
                        })
                    else:
                        result_elements.append({
                            "html": element_html,
                            "tag": element.tag
                        })
            
            return result_elements, metrics
            
        except Exception as e:
            self.logger.error(f"Error during element selection: {str(e)}")
            # Return empty list on error
            return [], metrics
    
    def _select_first_match(self, doc: HtmlElement) -> Tuple[List[HtmlElement], List[str]]:
        """
        Select the first element that matches any selector.
        
        Args:
            doc: The HTML document
            
        Returns:
            Tuple of (selected elements, used selectors)
        """
        for selector in self.selectors:
            try:
                # Check if it's an XPath expression
                if selector.startswith("//") or selector.startswith("./"):
                    elements = doc.xpath(selector)
                else:
                    # CSS selector
                    elements = doc.cssselect(selector)
                
                if elements:
                    return [elements[0]], [selector]
            except Exception as e:
                self.logger.warning(f"Error with selector '{selector}': {str(e)}")
        
        return [], []
    
    def _select_all_matches(self, doc: HtmlElement) -> Tuple[List[HtmlElement], List[str]]:
        """
        Select all elements that match any selector.
        
        Args:
            doc: The HTML document
            
        Returns:
            Tuple of (selected elements, used selectors)
        """
        all_elements = []
        used_selectors = []
        
        for selector in self.selectors:
            try:
                # Check if it's an XPath expression
                if selector.startswith("//") or selector.startswith("./"):
                    elements = doc.xpath(selector)
                else:
                    # CSS selector
                    elements = doc.cssselect(selector)
                
                if elements:
                    all_elements.extend(elements)
                    used_selectors.append(selector)
            except Exception as e:
                self.logger.warning(f"Error with selector '{selector}': {str(e)}")
        
        # Remove duplicates while preserving order
        unique_elements = []
        for element in all_elements:
            if element not in unique_elements:
                unique_elements.append(element)
        
        return unique_elements, used_selectors
    
    def _select_priority_based(self, doc: HtmlElement) -> Tuple[List[HtmlElement], List[str]]:
        """
        Select elements based on selector priority.
        
        Args:
            doc: The HTML document
            
        Returns:
            Tuple of (selected elements, used selectors)
        """
        for selector in self.selectors:
            try:
                # Check if it's an XPath expression
                if selector.startswith("//") or selector.startswith("./"):
                    elements = doc.xpath(selector)
                else:
                    # CSS selector
                    elements = doc.cssselect(selector)
                
                if elements and len(elements) >= self.min_elements:
                    return elements, [selector]
            except Exception as e:
                self.logger.warning(f"Error with selector '{selector}': {str(e)}")
        
        # If we reached here, no selector found enough elements
        # Return results from the selector that found the most elements
        best_elements = []
        best_selector = ""
        
        for selector in self.selectors:
            try:
                # Check if it's an XPath expression
                if selector.startswith("//") or selector.startswith("./"):
                    elements = doc.xpath(selector)
                else:
                    # CSS selector
                    elements = doc.cssselect(selector)
                
                if len(elements) > len(best_elements):
                    best_elements = elements
                    best_selector = selector
            except Exception:
                pass
        
        return best_elements, [best_selector] if best_selector else []
    
    def _select_composite(self, doc: HtmlElement) -> Tuple[List[HtmlElement], List[str]]:
        """
        Combine results from multiple selectors.
        
        Args:
            doc: The HTML document
            
        Returns:
            Tuple of (selected elements, used selectors)
        """
        all_elements = []
        used_selectors = []
        
        for selector in self.selectors:
            try:
                # Check if it's an XPath expression
                if selector.startswith("//") or selector.startswith("./"):
                    elements = doc.xpath(selector)
                else:
                    # CSS selector
                    elements = doc.cssselect(selector)
                
                if elements:
                    all_elements.extend(elements)
                    used_selectors.append(selector)
            except Exception as e:
                self.logger.warning(f"Error with selector '{selector}': {str(e)}")
        
        # Remove duplicates while preserving order
        unique_elements = []
        for element in all_elements:
            if element not in unique_elements:
                unique_elements.append(element)
        
        return unique_elements, used_selectors
    
    def _apply_filters(self, elements: List[HtmlElement]) -> List[HtmlElement]:
        """
        Apply filters to the selected elements.
        
        Args:
            elements: List of selected elements
            
        Returns:
            Filtered list of elements
        """
        filtered_elements = []
        
        for element in elements:
            # Check if element passes all filters
            if self._element_passes_filters(element):
                filtered_elements.append(element)
        
        return filtered_elements
    
    def _element_passes_filters(self, element: HtmlElement) -> bool:
        """
        Check if element passes all specified filters.
        
        Args:
            element: The element to check
            
        Returns:
            True if element passes all filters, False otherwise
        """
        # Text content filter
        if "min_text_length" in self.element_filters:
            text_content = " ".join(element.xpath(".//text()")).strip()
            if len(text_content) < self.element_filters["min_text_length"]:
                return False
        
        # Attribute filter
        if "required_attributes" in self.element_filters:
            for attr in self.element_filters["required_attributes"]:
                if attr not in element.attrib:
                    return False
        
        # Attribute value filter
        if "attribute_values" in self.element_filters:
            for attr, value in self.element_filters["attribute_values"].items():
                if attr not in element.attrib or element.attrib[attr] != value:
                    return False
        
        # Child element filter
        if "required_children" in self.element_filters:
            for child_selector in self.element_filters["required_children"]:
                try:
                    if not element.xpath(f"./{child_selector}"):
                        return False
                except Exception:
                    return False
        
        # Regex pattern filter
        if "text_pattern" in self.element_filters:
            text_content = " ".join(element.xpath(".//text()")).strip()
            pattern = re.compile(self.element_filters["text_pattern"])
            if not pattern.search(text_content):
                return False
        
        # Element passes all filters
        return True