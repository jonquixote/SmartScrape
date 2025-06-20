"""
Structural Analyzer Module

This module provides functionality to analyze HTML document structure,
identify content sections, detect element relationships, and understand
page layout.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from bs4 import BeautifulSoup, Tag

from extraction.core.extraction_interface import StructuralAnalyzer

# Configure logging
logger = logging.getLogger(__name__)

class DOMStructuralAnalyzer(StructuralAnalyzer):
    """
    Analyzes HTML document structure to understand layout, content organization,
    and element relationships.
    
    This class provides comprehensive structural analysis of web pages to support
    intelligent data extraction by understanding the document's organization.
    """
    
    def __init__(self, context=None):
        """Initialize the DOM structural analyzer."""
        super().__init__(context)
        self._semantic_tags = {
            'header', 'footer', 'nav', 'main', 'article', 'section', 'aside',
            'figure', 'figcaption', 'details', 'summary', 'mark', 'time'
        }
        self._content_tags = {
            'article', 'section', 'main', 'div', 'p', 'h1', 'h2', 'h3', 
            'h4', 'h5', 'h6', 'ul', 'ol', 'dl', 'table'
        }
        self._last_analysis = {}
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the analyzer with configuration."""
        if self._initialized:
            return
        self._config = config or {}
        self._initialized = True
        logger.debug(f"{self.__class__.__name__} initialized")
    
    def shutdown(self) -> None:
        """Clean up resources."""
        self._initialized = False
        logger.debug(f"{self.__class__.__name__} shut down")
    
    def can_handle(self, content: Any, content_type: Optional[str] = None) -> bool:
        """
        Check if this analyzer can handle the given content type.
        
        Args:
            content: The content to analyze
            content_type: The type of content (optional)
            
        Returns:
            True if the analyzer can handle this content, False otherwise
        """
        if content_type and content_type.lower() in ["html", "xhtml", "xml"]:
            return True
            
        if isinstance(content, str) and ("<html" in content.lower()[:1000] or 
                                       "<body" in content.lower()[:1000]):
            return True
            
        if isinstance(content, (BeautifulSoup, Tag)):
            return True
            
        return False
    
    def analyze_structure(self, content: Any) -> Dict[str, Any]:
        """
        Analyze the structure of the provided content.
        
        This is the implementation of the abstract method from the parent class.
        It calls analyze() for consistency with other methods in this class.
        
        Args:
            content: Content to analyze
            
        Returns:
            Structural analysis results
        """
        return self.analyze(content)
    
    def extract(self, content: Any, schema: Optional[Dict[str, Any]] = None, 
               options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract structural information from content.
        
        Args:
            content: The content to analyze
            schema: Optional schema to guide extraction
            options: Additional options for extraction
            
        Returns:
            Dictionary with structural analysis results
        """
        # For structural analyzer, extract is just a wrapper around analyze
        return self.analyze(content, options)
    
    def analyze(self, html_content: Any, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive structural analysis of HTML content.
        
        Args:
            html_content: HTML content to analyze
            options: Optional analysis options
            
        Returns:
            Dictionary with analysis results
        """
        options = options or {}
        try:
            # Parse HTML content if needed
            soup = self._parse_html(html_content)
            
            # Store original URL if provided
            url = options.get("url", "")
            
            # Determine which analyses to perform
            analysis_types = options.get("analysis_types", [
                "hierarchy", "sections", "content_type", "navigation", 
                "interactive", "result_groups", "layout"
            ])
            
            # Initialize results
            results = {
                "url": url,
                "success": True,
                "page_title": self._extract_title(soup),
                "_metadata": {
                    "analyzer": self.__class__.__name__,
                    "analysis_types": analysis_types
                }
            }
            
            # Perform requested analyses
            if "hierarchy" in analysis_types:
                results["hierarchy"] = self.analyze_dom_hierarchy(soup)
                
            if "sections" in analysis_types:
                results["content_sections"] = self.identify_content_sections(soup)
                results["content_boundaries"] = self.find_content_boundaries(soup)
                
            if "content_type" in analysis_types:
                results["content_type"] = self.detect_content_type(soup)
                
            if "navigation" in analysis_types:
                results["navigation_elements"] = self.identify_navigation_elements(soup)
                results["pagination"] = self.detect_pagination_elements(soup)
                
            if "interactive" in analysis_types:
                results["interactive_elements"] = self.identify_interactive_elements(soup)
                
            if "result_groups" in analysis_types:
                results["result_groups"] = self.detect_result_groups(soup)
                
            if "layout" in analysis_types:
                results["layout"] = self.analyze_layout(soup)
                
            if "template" in analysis_types:
                results["template_structure"] = self.detect_template_structure(soup)
                
            if "content_density" in analysis_types:
                results["content_density"] = self.analyze_content_density(soup)
                
            # Generate overall structure map
            results["structure_map"] = self.generate_structure_map(soup)
            
            # Store analysis results for future reference
            self._last_analysis = results
            
            return results
            
        except Exception as e:
            # Use error handler if available
            if self._context:
                try:
                    error_classifier = self._get_service("error_classifier")
                    classification = error_classifier.classify_exception(e, {
                        "analyzer": self.__class__.__name__,
                        "operation": "analyze"
                    })
                    logger.error(f"Error in DOM structural analysis: {classification['category']}: {str(e)}")
                except Exception:
                    logger.error(f"Error in DOM structural analysis: {str(e)}", exc_info=True)
            else:
                logger.error(f"Error in DOM structural analysis: {str(e)}", exc_info=True)
                
            return {
                "success": False,
                "error": str(e),
                "_metadata": {
                    "analyzer": self.__class__.__name__
                }
            }
    
    def analyze_dom_hierarchy(self, dom: Union[BeautifulSoup, Tag]) -> Dict[str, Any]:
        """
        Map the parent-child relationships in the DOM.
        
        Args:
            dom: BeautifulSoup object or Tag
            
        Returns:
            Dictionary with DOM hierarchy information
        """
        try:
            # Find the root element (typically <body>)
            root = dom.find('body') if dom.find('body') else dom
            
            # Extract basic hierarchy statistics
            element_count = len(list(root.descendants))
            max_depth = self._calculate_max_depth(root)
            semantic_elements = self._count_semantic_elements(root)
            
            # Build a simplified DOM tree
            tree = self._build_simplified_tree(root, max_depth=3)
            
            # Calculate complexity metrics
            complexity = self._calculate_dom_complexity(root)
            
            return {
                "element_count": element_count,
                "max_depth": max_depth,
                "semantic_structure_quality": semantic_elements / max(1, element_count),
                "complexity": complexity,
                "simplified_tree": tree,
                "dominant_patterns": self._find_dominant_patterns(root)
            }
        except Exception as e:
            logger.error(f"Error analyzing DOM hierarchy: {str(e)}")
            return {"error": str(e)}
    
    def identify_content_sections(self, dom: Union[BeautifulSoup, Tag]) -> List[Dict[str, Any]]:
        """
        Identify distinct content sections in the document.
        
        Args:
            dom: BeautifulSoup object or Tag
            
        Returns:
            List of identified content sections with metadata
        """
        try:
            sections = []
            
            # Check for semantic sectioning
            semantic_sections = self._find_semantic_sections(dom)
            if semantic_sections:
                sections.extend(semantic_sections)
            
            # Use visual sections as fallback
            if not sections or len(sections) <= 1:
                visual_sections = self._detect_visual_sections(dom)
                sections.extend(visual_sections)
            
            # Analyze each section
            for section in sections:
                # Calculate text density
                section["text_density"] = self._calculate_text_density(section["element"])
                
                # Check if it's likely boilerplate
                section["is_boilerplate"] = self._detect_boilerplate(section["element"])
                
                # Find headlines in section
                headlines = self._find_headline_elements(section["element"])
                section["headlines"] = [h.get_text(strip=True) for h in headlines]
                
                # Calculate content importance score
                section["importance_score"] = self._calculate_section_importance(
                    section["element"],
                    section["text_density"],
                    section["is_boilerplate"]
                )
                
                # Remove BS4 element to make JSON serializable
                section["element"] = self._get_element_path(section["element"])
            
            # Sort sections by importance score
            sections.sort(key=lambda x: x["importance_score"], reverse=True)
            
            return sections
        except Exception as e:
            logger.error(f"Error identifying content sections: {str(e)}")
            return [{"error": str(e)}]
    
    def detect_result_groups(self, dom: Union[BeautifulSoup, Tag]) -> List[Dict[str, Any]]:
        """
        Find and group related items within the document.
        
        Args:
            dom: BeautifulSoup object or Tag
            
        Returns:
            List of result groups with metadata
        """
        try:
            result_groups = []
            
            # Look for common result containers
            containers = self._find_result_containers(dom)
            
            for container in containers:
                # Try to identify items within container
                items = self._identify_result_items(container)
                
                if items and len(items) >= 2:
                    # Calculate structural similarity between items
                    similarity = self._calculate_structural_similarity(items)
                    
                    # Only consider as a result group if items are similar
                    if similarity >= 0.6:
                        # Extract common fields from items
                        common_fields = self._extract_common_fields(items)
                        
                        # Get container selector
                        container_selector = self._get_element_path(container)
                        
                        result_groups.append({
                            "container_selector": container_selector,
                            "item_count": len(items),
                            "structural_similarity": similarity,
                            "common_fields": common_fields,
                            "is_list": container.name in ['ul', 'ol'] or self._has_list_structure(container),
                            "is_table": container.name == 'table' or self._has_table_structure(container),
                            "is_grid": self._has_grid_structure(container)
                        })
            
            # Sort by item count and structural similarity
            result_groups.sort(key=lambda x: (x["item_count"], x["structural_similarity"]), reverse=True)
            
            return result_groups
        except Exception as e:
            logger.error(f"Error detecting result groups: {str(e)}")
            return [{"error": str(e)}]
    
    def analyze_layout(self, dom: Union[BeautifulSoup, Tag]) -> Dict[str, Any]:
        """
        Analyze the visual layout and organization of the document.
        
        Args:
            dom: BeautifulSoup object or Tag
            
        Returns:
            Dictionary with layout analysis results
        """
        try:
            # Get HTML service if available
            html_service = self._get_service("html_service") if self._context else None
            
            # Identify layout structure
            has_grid = bool(self._detect_layout_grid(dom))
            has_sidebar = self._detect_sidebar(dom)
            
            # Build layout map
            layout_map = {
                "has_grid_layout": has_grid,
                "has_sidebar": has_sidebar,
                "visual_sections": self._detect_visual_sections(dom, include_elements=False),
                "content_flow": self._map_reading_flow(dom),
                "whitespace_distribution": self._analyze_whitespace_distribution(dom),
                "main_content_area": self._identify_main_content_area(dom)
            }
            
            # Analyze CSS for layout patterns if HTML service is available
            if html_service:
                try:
                    layout_map["css_layout_analysis"] = html_service.analyze_css_layout(dom)
                except Exception as css_err:
                    logger.warning(f"Error analyzing CSS layout: {str(css_err)}")
            
            return layout_map
        except Exception as e:
            logger.error(f"Error analyzing layout: {str(e)}")
            return {"error": str(e)}
    
    def generate_structure_map(self, dom: Union[BeautifulSoup, Tag]) -> Dict[str, Any]:
        """
        Create a structured representation of the document.
        
        Args:
            dom: BeautifulSoup object or Tag
            
        Returns:
            Dictionary with structured representation of the document
        """
        try:
            # Find body or root element
            root = dom.find('body') if dom.find('body') else dom
            
            # Generate simplified structure map
            structure_map = {
                "type": "root",
                "tag": root.name,
                "id": root.get('id', ''),
                "classes": root.get('class', []),
                "child_count": len([c for c in root.children if isinstance(c, Tag)]),
                "depth": 0,
                "children": []
            }
            
            # Add semantic sections to the structure map
            for section in self._find_semantic_sections(root, include_elements=False):
                element = section["element"]
                section_map = {
                    "type": "section",
                    "tag": element.name,
                    "id": element.get('id', ''),
                    "classes": element.get('class', []),
                    "section_type": section["section_type"],
                    "content_type": section.get("content_type", "unknown"),
                    "path": self._get_element_path(element),
                    "child_count": len([c for c in element.children if isinstance(c, Tag)])
                }
                structure_map["children"].append(section_map)
            
            # Add result groups to the structure map
            for group in self.detect_result_groups(root):
                if "error" in group:
                    continue
                    
                group_map = {
                    "type": "result_group",
                    "container_selector": group["container_selector"],
                    "item_count": group["item_count"],
                    "group_type": ("list" if group.get("is_list", False) else 
                                  "table" if group.get("is_table", False) else 
                                  "grid" if group.get("is_grid", False) else "generic")
                }
                structure_map["children"].append(group_map)
            
            return structure_map
        except Exception as e:
            logger.error(f"Error generating structure map: {str(e)}")
            return {"error": str(e)}
    
    def find_content_boundaries(self, dom: Union[BeautifulSoup, Tag]) -> Dict[str, Any]:
        """
        Identify content section boundaries within the document.
        
        Args:
            dom: BeautifulSoup object or Tag
            
        Returns:
            Dictionary mapping section types to their boundaries
        """
        try:
            boundaries = {}
            
            # Find body or root element
            root = dom.find('body') if dom.find('body') else dom
            
            # Identify main content boundaries
            main_content = root.find('main')
            if not main_content:
                # Try to find main content using other methods
                main_content = self._identify_main_content_area(root)
            
            if main_content:
                boundaries["main_content"] = self._get_element_path(main_content)
            
            # Identify header boundaries
            header = root.find('header')
            if header:
                boundaries["header"] = self._get_element_path(header)
            
            # Identify footer boundaries
            footer = root.find('footer')
            if footer:
                boundaries["footer"] = self._get_element_path(footer)
            
            # Identify sidebar boundaries
            sidebar_candidates = root.select('aside, [class*=sidebar], [id*=sidebar]')
            if sidebar_candidates:
                boundaries["sidebar"] = self._get_element_path(sidebar_candidates[0])
            
            # Identify navigation boundaries
            nav = root.find('nav')
            if nav:
                boundaries["navigation"] = self._get_element_path(nav)
            
            return {
                "boundaries": boundaries,
                "content_regions": self._identify_content_regions(root)
            }
        except Exception as e:
            logger.error(f"Error finding content boundaries: {str(e)}")
            return {"error": str(e)}
    
    def detect_content_type(self, dom: Union[BeautifulSoup, Tag]) -> Dict[str, Any]:
        """
        Determine the page type (product, article, listing, etc.).
        
        Args:
            dom: BeautifulSoup object or Tag
            
        Returns:
            Dictionary with content type information
        """
        try:
            # Extract metadata for content type detection
            meta_tags = dom.find_all('meta')
            meta_data = {}
            for tag in meta_tags:
                if tag.get('name'):
                    meta_data[tag['name']] = tag.get('content', '')
                elif tag.get('property'):
                    meta_data[tag['property']] = tag.get('content', '')
            
            # Check for Open Graph type
            og_type = meta_data.get('og:type', '')
            
            # Check for structured data
            structured_data_types = self._extract_structured_data_types(dom)
            
            # Look for content patterns
            has_article_structure = self._has_article_structure(dom)
            has_product_structure = self._has_product_structure(dom)
            has_listing_structure = self._has_listing_structure(dom)
            has_form_dominance = self._has_form_dominance(dom)
            has_search_results = self._has_search_results_structure(dom)
            
            # Determine primary content type
            content_type = "unknown"
            confidence = 0.0
            content_signals = {}
            
            # Check for article
            if og_type == 'article' or 'article' in structured_data_types:
                content_type = "article"
                confidence = 0.9
                content_signals["article"] = 0.9
            elif has_article_structure:
                content_type = "article"
                confidence = 0.7
                content_signals["article"] = 0.7
            
            # Check for product
            if og_type == 'product' or 'product' in structured_data_types:
                content_type = "product"
                confidence = 0.9
                content_signals["product"] = 0.9
            elif has_product_structure:
                content_type = "product"
                confidence = 0.7
                content_signals["product"] = 0.7
            
            # Check for listing
            if has_listing_structure:
                listing_confidence = 0.8
                content_signals["listing"] = listing_confidence
                if listing_confidence > confidence:
                    content_type = "listing"
                    confidence = listing_confidence
            
            # Check for search results
            if has_search_results:
                search_confidence = 0.75
                content_signals["search_results"] = search_confidence
                if search_confidence > confidence:
                    content_type = "search_results"
                    confidence = search_confidence
            
            # Check for form/input page
            if has_form_dominance:
                form_confidence = 0.7
                content_signals["form"] = form_confidence
                if form_confidence > confidence:
                    content_type = "form"
                    confidence = form_confidence
            
            return {
                "primary_type": content_type,
                "confidence": confidence,
                "content_signals": content_signals,
                "structured_data_types": structured_data_types,
                "meta_properties": {k: v for k, v in meta_data.items() if 'type' in k.lower()}
            }
        except Exception as e:
            logger.error(f"Error detecting content type: {str(e)}")
            return {"primary_type": "unknown", "error": str(e)}
    
    def identify_navigation_elements(self, dom: Union[BeautifulSoup, Tag]) -> Dict[str, Any]:
        """
        Find navigation components in the document.
        
        Args:
            dom: BeautifulSoup object or Tag
            
        Returns:
            Dictionary with navigation elements information
        """
        try:
            nav_elements = {
                "primary_nav": [],
                "secondary_nav": [],
                "breadcrumbs": [],
                "menu_elements": []
            }
            
            # Find semantic nav elements
            nav_tags = dom.find_all('nav')
            for i, nav in enumerate(nav_tags):
                nav_info = {
                    "selector": self._get_element_path(nav),
                    "item_count": len(nav.find_all('a')),
                    "is_primary": self._is_primary_navigation(nav),
                    "position": self._get_element_position(nav)
                }
                
                if nav_info["is_primary"]:
                    nav_elements["primary_nav"].append(nav_info)
                else:
                    nav_elements["secondary_nav"].append(nav_info)
            
            # Find breadcrumbs
            breadcrumb_candidates = dom.select('.breadcrumb, .breadcrumbs, [itemtype*="Breadcrumb"], ol[class*="bread"], ul[class*="bread"]')
            for crumb in breadcrumb_candidates:
                nav_elements["breadcrumbs"].append({
                    "selector": self._get_element_path(crumb),
                    "items": [{"text": a.get_text(strip=True), "href": a.get('href', '')} 
                              for a in crumb.find_all('a')]
                })
            
            # Find menu elements (dropdowns, etc.)
            menu_candidates = dom.select('ul[class*="menu"], [class*="dropdown"], [role="menu"]')
            for menu in menu_candidates:
                if menu not in nav_tags:  # Avoid duplicates
                    nav_elements["menu_elements"].append({
                        "selector": self._get_element_path(menu),
                        "item_count": len(menu.find_all('a')),
                        "position": self._get_element_position(menu)
                    })
            
            return nav_elements
        except Exception as e:
            logger.error(f"Error identifying navigation elements: {str(e)}")
            return {"error": str(e)}
    
    def detect_pagination_elements(self, dom: Union[BeautifulSoup, Tag]) -> Dict[str, Any]:
        """
        Find pagination controls in the document.
        
        Args:
            dom: BeautifulSoup object or Tag
            
        Returns:
            Dictionary with pagination information
        """
        try:
            pagination = {
                "has_pagination": False,
                "pagination_elements": [],
                "current_page": None,
                "total_pages": None,
                "page_links": []
            }
            
            # Look for common pagination selectors
            pagination_candidates = dom.select('.pagination, [class*="pager"], [class*="pagin"], nav[aria-label*="pag"]')
            
            if not pagination_candidates:
                # Try other patterns
                pagination_candidates = dom.select('ul > li > a[href*="page="], div > a[href*="page="]')
            
            if pagination_candidates:
                pagination["has_pagination"] = True
                
                for p in pagination_candidates:
                    # Extract pagination info
                    pages = []
                    current = None
                    total = None
                    
                    # Find page links
                    links = p.find_all('a')
                    for link in links:
                        href = link.get('href', '')
                        text = link.get_text(strip=True)
                        active = 'active' in link.get('class', []) or 'current' in link.get('class', [])
                        
                        if text.isdigit():
                            pages.append(int(text))
                            if active:
                                current = int(text)
                    
                    # Extract current page from element with 'active' or 'current' class
                    active_element = p.select_one('.active, .current')
                    if active_element and active_element.get_text(strip=True).isdigit():
                        current = int(active_element.get_text(strip=True))
                    
                    # Try to find total pages
                    if pages:
                        total = max(pages)
                    
                    pagination["pagination_elements"].append({
                        "selector": self._get_element_path(p),
                        "page_count": len(pages),
                        "current_page": current,
                        "total_pages": total
                    })
                    
                    # Keep track of page links
                    for link in links:
                        href = link.get('href', '')
                        text = link.get_text(strip=True)
                        if href and (text.isdigit() or text in ['Next', 'Previous', '›', '‹', '»', '«']):
                            pagination["page_links"].append({
                                "text": text,
                                "href": href,
                                "is_current": 'active' in link.get('class', []) or 'current' in link.get('class', [])
                            })
                
                # Set current and total pages from the first pagination element
                if pagination["pagination_elements"]:
                    pagination["current_page"] = pagination["pagination_elements"][0]["current_page"]
                    pagination["total_pages"] = pagination["pagination_elements"][0]["total_pages"]
            
            return pagination
        except Exception as e:
            logger.error(f"Error detecting pagination elements: {str(e)}")
            return {"has_pagination": False, "error": str(e)}
    
    def identify_interactive_elements(self, dom: Union[BeautifulSoup, Tag]) -> Dict[str, Any]:
        """
        Find forms and interactive elements in the document.
        
        Args:
            dom: BeautifulSoup object or Tag
            
        Returns:
            Dictionary with interactive elements information
        """
        try:
            interactive = {
                "forms": [],
                "buttons": [],
                "inputs": {
                    "text": 0,
                    "checkbox": 0,
                    "radio": 0,
                    "select": 0,
                    "textarea": 0,
                    "other": 0
                },
                "interactive_regions": []
            }
            
            # Find forms
            forms = dom.find_all('form')
            for form in forms:
                form_info = {
                    "selector": self._get_element_path(form),
                    "method": form.get('method', 'get').lower(),
                    "action": form.get('action', ''),
                    "inputs": []
                }
                
                # Analyze form inputs
                for input_tag in form.find_all(['input', 'select', 'textarea']):
                    input_type = input_tag.get('type', 'text').lower() if input_tag.name == 'input' else input_tag.name
                    input_info = {
                        "name": input_tag.get('name', ''),
                        "type": input_type,
                        "id": input_tag.get('id', ''),
                        "placeholder": input_tag.get('placeholder', ''),
                        "required": input_tag.has_attr('required')
                    }
                    form_info["inputs"].append(input_info)
                    
                    # Count input types
                    if input_tag.name == 'input':
                        if input_type in interactive["inputs"]:
                            interactive["inputs"][input_type] += 1
                        else:
                            interactive["inputs"]["other"] += 1
                    else:
                        interactive["inputs"][input_tag.name] += 1
                
                # Determine form purpose
                form_info["purpose"] = self._determine_form_purpose(form)
                
                interactive["forms"].append(form_info)
            
            # Find buttons outside forms
            for button in dom.find_all(['button', 'a[class*="btn"], [class*="button"], input[type="button"], input[type="submit"]']):
                if not button.find_parent('form'):
                    interactive["buttons"].append({
                        "selector": self._get_element_path(button),
                        "text": button.get_text(strip=True),
                        "type": button.name
                    })
            
            # Find interactive regions (tabbed content, accordions, etc.)
            interactive_regions = dom.select('[role="tablist"], [role="tabpanel"], [class*="accordion"], [class*="tabs"], [aria-expanded]')
            for region in interactive_regions:
                interactive["interactive_regions"].append({
                    "selector": self._get_element_path(region),
                    "type": self._determine_interactive_region_type(region),
                    "state": "expanded" if region.get('aria-expanded') == 'true' else "collapsed" if region.get('aria-expanded') == 'false' else "unknown"
                })
            
            return interactive
        except Exception as e:
            logger.error(f"Error identifying interactive elements: {str(e)}")
            return {"error": str(e)}
    
    def analyze_content_density(self, dom: Union[BeautifulSoup, Tag]) -> Dict[str, Any]:
        """
        Identify content-rich areas in the document.
        
        Args:
            dom: BeautifulSoup object or Tag
            
        Returns:
            Dictionary with content density information
        """
        try:
            # Get root element
            root = dom.find('body') if dom.find('body') else dom
            
            # Calculate overall text density
            overall_density = self._calculate_text_density(root)
            
            # Find main content area
            main_content = root.find('main')
            if not main_content:
                main_content = self._identify_main_content_area(root)
            
            main_content_density = self._calculate_text_density(main_content) if main_content else 0
            
            # Analyze content sections
            sections = self._find_semantic_sections(root)
            section_densities = []
            
            for section in sections:
                element = section["element"]
                density = self._calculate_text_density(element)
                text_length = len(element.get_text(strip=True))
                
                section_densities.append({
                    "selector": self._get_element_path(element),
                    "section_type": section["section_type"],
                    "density": density,
                    "text_length": text_length
                })
            
            # Sort sections by density
            section_densities.sort(key=lambda x: x["density"], reverse=True)
            
            # Identify content-rich areas
            content_rich_selectors = [s["selector"] for s in section_densities[:3] if s["density"] > 0.3]
            
            return {
                "overall_density": overall_density,
                "main_content_density": main_content_density,
                "section_densities": section_densities,
                "content_rich_areas": content_rich_selectors,
                "text_length": len(root.get_text(strip=True))
            }
        except Exception as e:
            logger.error(f"Error analyzing content density: {str(e)}")
            return {"error": str(e)}
    
    def detect_template_structure(self, dom: Union[BeautifulSoup, Tag]) -> Dict[str, Any]:
        """
        Identify repeated page elements that might be part of a template.
        
        Args:
            dom: BeautifulSoup object or Tag
            
        Returns:
            Dictionary with template structure information
        """
        try:
            # Get root element
            root = dom.find('body') if dom.find('body') else dom
            
            # Detect common template elements
            template_elements = {
                "header": None,
                "footer": None,
                "navigation": None,
                "sidebar": None,
                "repeated_structures": []
            }
            
            # Check for header
            header = root.find('header')
            if header:
                template_elements["header"] = self._get_element_path(header)
            
            # Check for footer
            footer = root.find('footer')
            if footer:
                template_elements["footer"] = self._get_element_path(footer)
            
            # Check for navigation
            nav = root.find('nav')
            if nav:
                template_elements["navigation"] = self._get_element_path(nav)
            
            # Check for sidebar
            sidebar = root.select_one('aside, [class*="sidebar"], [id*="sidebar"]')
            if sidebar:
                template_elements["sidebar"] = self._get_element_path(sidebar)
            
            # Find repeated structures
            repeated = self._find_repeated_elements(root)
            template_elements["repeated_structures"] = repeated
            
            # Determine template regions
            template_regions = {
                "likely_template": [template_elements[k] for k in ["header", "footer", "navigation", "sidebar"] if template_elements[k]],
                "content_specific": []
            }
            
            # Add main content area to content specific regions
            main_content = root.find('main')
            if main_content:
                template_regions["content_specific"].append(self._get_element_path(main_content))
            
            return {
                "template_elements": template_elements,
                "template_regions": template_regions
            }
        except Exception as e:
            logger.error(f"Error detecting template structure: {str(e)}")
            return {"error": str(e)}
    
    def cluster_similar_content(self, elements: List[Tag]) -> List[List[Tag]]:
        """
        Group similar elements together based on structure and content.
        
        Args:
            elements: List of BeautifulSoup Tag objects
            
        Returns:
            List of lists, where each inner list contains similar elements
        """
        if not elements:
            return []
            
        try:
            # Calculate similarity matrix
            n = len(elements)
            similarity_matrix = [[0 for _ in range(n)] for _ in range(n)]
            
            for i in range(n):
                for j in range(i+1, n):
                    similarity = self._calculate_content_similarity(elements[i], elements[j])
                    similarity_matrix[i][j] = similarity
                    similarity_matrix[j][i] = similarity
            
            # Set diagonal to 1 (element is identical to itself)
            for i in range(n):
                similarity_matrix[i][i] = 1.0
            
            # Simple clustering: group elements with similarity > threshold
            clusters = []
            unassigned = set(range(n))
            
            while unassigned:
                # Pick an unassigned element
                i = next(iter(unassigned))
                unassigned.remove(i)
                
                # Create a new cluster with this element
                cluster = [i]
                
                # Find similar elements
                for j in list(unassigned):
                    if similarity_matrix[i][j] >= 0.7:  # Threshold
                        cluster.append(j)
                        unassigned.remove(j)
                
                # Add the cluster to our list of clusters
                clusters.append([elements[idx] for idx in cluster])
            
            return clusters
        except Exception as e:
            logger.error(f"Error clustering similar content: {str(e)}")
            return [[elements[0]]] if elements else []  # Fallback to one cluster with all elements
    
    def detect_element_relationships(self, content: Any) -> Dict[str, List[Tuple]]:
        """
        Detect relationships between elements in the content.
        
        Args:
            content: HTML content or BeautifulSoup object
            
        Returns:
            Dictionary mapping relationship types to element pairs
        """
        try:
            # Parse HTML if needed
            dom = self._parse_html(content)
            
            # Find relationships
            relationships = {
                "parent_child": [],
                "sibling": [],
                "container_item": [],
                "label_field": [],
                "heading_content": []
            }
            
            # Find parent-child relationships for semantic elements
            semantic_elements = dom.find_all(self._semantic_tags)
            for elem in semantic_elements:
                parent_path = self._get_element_path(elem)
                for child in elem.find_all(self._semantic_tags, recursive=False):
                    child_path = self._get_element_path(child)
                    relationships["parent_child"].append((parent_path, child_path))
            
            # Find container-item relationships
            containers = self._find_result_containers(dom)
            for container in containers:
                container_path = self._get_element_path(container)
                items = self._identify_result_items(container)
                for item in items:
                    item_path = self._get_element_path(item)
                    relationships["container_item"].append((container_path, item_path))
            
            # Find label-field relationships
            labels = dom.find_all('label')
            for label in labels:
                if label.get('for'):
                    field = dom.find(id=label['for'])
                    if field:
                        label_path = self._get_element_path(label)
                        field_path = self._get_element_path(field)
                        relationships["label_field"].append((label_path, field_path))
            
            # Find heading-content relationships
            headings = dom.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            for heading in headings:
                # Look for next sibling or parent's next sibling as content
                content_elem = None
                next_elem = heading.find_next_sibling()
                if next_elem and next_elem.name in self._content_tags:
                    content_elem = next_elem
                
                # If no appropriate next sibling, look for a parent's content section
                if not content_elem and heading.parent:
                    section = heading.parent
                    if section.name in ['section', 'article', 'div']:
                        # Get all elements after the heading
                        after_heading = False
                        for child in section.children:
                            if after_heading and isinstance(child, Tag) and child.name in self._content_tags:
                                content_elem = child
                                break
                            if child is heading:
                                after_heading = True
                
                if content_elem:
                    heading_path = self._get_element_path(heading)
                    content_path = self._get_element_path(content_elem)
                    relationships["heading_content"].append((heading_path, content_path))
            
            return relationships
        except Exception as e:
            logger.error(f"Error detecting element relationships: {str(e)}")
            return {"error": [str(e)]}
    
    # Helper methods
    
    def _parse_html(self, html_content: Any) -> BeautifulSoup:
        """Parse HTML content into a BeautifulSoup object."""
        if isinstance(html_content, (BeautifulSoup, Tag)):
            return html_content
            
        if isinstance(html_content, str):
            return BeautifulSoup(html_content, 'html.parser')
            
        raise ValueError(f"Unsupported content type: {type(html_content)}")
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract the title of the page."""
        if soup.title:
            return soup.title.string.strip() if soup.title.string else ""
        
        h1 = soup.find('h1')
        if h1:
            return h1.get_text(strip=True)
            
        return ""
    
    def _calculate_max_depth(self, element: Tag) -> int:
        """Calculate the maximum depth of nested elements."""
        if not element or not hasattr(element, 'contents'):
            return 0
            
        max_child_depth = 0
        for child in element.children:
            if isinstance(child, Tag):
                child_depth = self._calculate_max_depth(child)
                max_child_depth = max(max_child_depth, child_depth)
                
        return 1 + max_child_depth
    
    def _count_semantic_elements(self, element: Tag) -> int:
        """Count the number of semantic HTML5 elements."""
        return len(element.find_all(self._semantic_tags))
    
    def _build_simplified_tree(self, element: Tag, max_depth: int = 3, current_depth: int = 0) -> Dict[str, Any]:
        """Build a simplified representation of the DOM tree."""
        if current_depth > max_depth or not element or not hasattr(element, 'name'):
            return None
            
        node = {
            "tag": element.name,
            "id": element.get('id', ''),
            "classes": element.get('class', []),
            "children": []
        }
        
        for child in element.children:
            if isinstance(child, Tag):
                child_node = self._build_simplified_tree(child, max_depth, current_depth + 1)
                if child_node:
                    node["children"].append(child_node)
                    
        return node
    
    def _calculate_dom_complexity(self, element: Tag) -> float:
        """Calculate a complexity score for the DOM structure."""
        # Count various elements that contribute to complexity
        element_count = len(list(element.descendants))
        unique_tags = len(set(tag.name for tag in element.find_all()))
        max_depth = self._calculate_max_depth(element)
        class_count = len(set(cls for tag in element.find_all() if tag.get('class') 
                            for cls in tag.get('class', [])))
        id_count = len(set(tag.get('id') for tag in element.find_all() if tag.get('id')))
        
        # Normalize and combine factors
        normalized_elements = min(1.0, element_count / 1000)
        normalized_tags = min(1.0, unique_tags / 50)
        normalized_depth = min(1.0, max_depth / 20)
        normalized_attrs = min(1.0, (class_count + id_count) / 200)
        
        # Weight and combine
        complexity = (
            normalized_elements * 0.4 +
            normalized_tags * 0.2 +
            normalized_depth * 0.3 +
            normalized_attrs * 0.1
        )
        
        return complexity
    
    def _find_dominant_patterns(self, element: Tag) -> List[Dict[str, Any]]:
        """Find dominant structural patterns in the DOM."""
        patterns = []
        
        # Look for common patterns
        tag_counts = {}
        for tag in element.find_all():
            if tag.name not in tag_counts:
                tag_counts[tag.name] = 0
            tag_counts[tag.name] += 1
        
        # Find dominant tags
        dominant_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Find repeating class patterns
        class_patterns = {}
        for tag in element.find_all(True, {'class': True}):
            class_str = ' '.join(sorted(tag.get('class')))
            if class_str not in class_patterns:
                class_patterns[class_str] = 0
            class_patterns[class_str] += 1
        
        # Get dominant class patterns
        dominant_classes = sorted(class_patterns.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Add to patterns
        for tag, count in dominant_tags:
            patterns.append({
                "type": "tag",
                "pattern": tag,
                "count": count
            })
            
        for classes, count in dominant_classes:
            if count >= 3:  # Only include patterns that repeat at least 3 times
                patterns.append({
                    "type": "class",
                    "pattern": classes,
                    "count": count
                })
        
        return patterns
    
    def _get_service(self, service_name: str) -> Any:
        """Helper to get a service from the context."""
        if not self._context:
            raise ValueError(f"Cannot access {service_name} service: context not set")
        return self._context.get_service(service_name)
    
    def _calculate_text_density(self, element: Tag) -> float:
        """Calculate the ratio of text content to HTML code."""
        if not element:
            return 0.0
            
        html_length = len(str(element))
        if html_length == 0:
            return 0.0
            
        text_length = len(element.get_text(strip=True))
        return text_length / html_length
    
    def _detect_boilerplate(self, element: Tag) -> bool:
        """Determine if an element is likely boilerplate content."""
        # Check element name
        if element.name in ["nav", "header", "footer", "aside"]:
            return True
            
        # Check for common boilerplate classes/ids
        boilerplate_indicators = [
            "menu", "navbar", "navigation", "sidebar", "footer", "header", 
            "banner", "copyright", "social", "related", "share", "widget"
        ]
        
        # Check element classes and id
        element_attrs = " ".join([
            " ".join(element.get("class", [])), 
            element.get("id", ""),
            element.get("role", "")
        ]).lower()
        
        for indicator in boilerplate_indicators:
            if indicator in element_attrs:
                return True
                
        # Check links density
        links = element.find_all("a")
        if links:
            link_text_length = sum(len(a.get_text(strip=True)) for a in links)
            total_text_length = len(element.get_text(strip=True))
            if total_text_length > 0 and link_text_length / total_text_length > 0.6:
                return True
                
        return False
    
    def _find_headline_elements(self, element: Tag) -> List[Tag]:
        """Find headline elements within the given element."""
        headlines = []
        
        # Look for heading tags
        for heading in element.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            headlines.append(heading)
            
        # Look for elements with headline-like classes
        for headline_elem in element.find_all(class_=lambda c: c and any(x in " ".join(c) for x in ["title", "headline", "heading"])):
            if headline_elem not in headlines:
                headlines.append(headline_elem)
                
        return headlines
    
    def _calculate_section_importance(self, element: Tag, text_density: float, is_boilerplate: bool) -> float:
        """Calculate the importance score of a content section."""
        if is_boilerplate:
            return 0.1
            
        # Base score on text density
        score = text_density * 5  # Scale up to 0-5 range
        
        # Adjust for section size
        text_length = len(element.get_text(strip=True))
        
        # Small sections get reduced importance
        if text_length < 100:
            score *= 0.5
        # Medium sections get normal importance
        elif text_length < 500:
            score *= 1.0
        # Large sections get increased importance
        else:
            score *= 1.5
            
        # Adjust for semantic elements
        if element.name in ["article", "main"]:
            score *= 2.0
        elif element.name in ["section"]:
            score *= 1.5
            
        # Adjust for headlines
        headlines = self._find_headline_elements(element)
        if headlines:
            score *= 1.0 + min(1.0, len(headlines) * 0.25)
            
        # Adjust for position
        position_score = self._calculate_position_score(element)
        score *= position_score
        
        return min(10.0, score)  # Cap at 10
    
    def _calculate_position_score(self, element: Tag) -> float:
        """Calculate a score based on the element's position in the document."""
        # Elements closer to the top of the main content are generally more important
        # A score of 1.0 is neutral, < 1.0 reduces importance, > 1.0 increases importance
        
        # Get the position index among siblings
        siblings = list(element.parent.children) if element.parent else []
        siblings = [s for s in siblings if isinstance(s, Tag)]
        
        if not siblings:
            return 1.0
            
        try:
            position_index = siblings.index(element)
            normalized_position = position_index / len(siblings)
            
            # If near the top, increase importance
            if normalized_position < 0.2:
                return 1.5
            # If near the bottom, decrease importance
            elif normalized_position > 0.8:
                return 0.7
            # Otherwise, neutral impact
            else:
                return 1.0
        except (ValueError, ZeroDivisionError):
            return 1.0
    
    def _get_element_path(self, element: Tag) -> str:
        """
        Generate a unique CSS selector path for an element.
        
        Args:
            element: BeautifulSoup Tag object
            
        Returns:
            CSS selector string
        """
        if not element:
            return ""
            
        # Try to use HTML service if available
        if self._context:
            try:
                html_service = self._get_service("html_service")
                return html_service.generate_selector(element)
            except Exception as e:
                logger.debug(f"Error using HTML service for selector generation: {str(e)}")
        
        # Fallback implementation
        selector_parts = []
        current = element
        
        while current and current.name:
            # Try ID first
            if current.get('id'):
                selector_parts.insert(0, f"#{current['id']}")
                break
            
            # Try to find position among siblings with same tag
            siblings = [sibling for sibling in current.parent.find_all(current.name, recursive=False)] if current.parent else []
            
            if len(siblings) > 1:
                for i, sibling in enumerate(siblings, 1):
                    if sibling is current:
                        selector_parts.insert(0, f"{current.name}:nth-of-type({i})")
                        break
            else:
                selector_parts.insert(0, current.name)
            
            current = current.parent
        
        return " > ".join(selector_parts) if selector_parts else element.name
    
    def _find_semantic_sections(self, element: Tag, include_elements: bool = True) -> List[Dict[str, Any]]:
        """
        Find semantic sections within the document.
        
        Args:
            element: BeautifulSoup Tag object
            include_elements: Whether to include the BS4 element in the result
            
        Returns:
            List of semantic section dictionaries
        """
        sections = []
        
        # Try to find article elements first
        for article in element.find_all('article'):
            sections.append({
                "section_type": "article",
                "element": article
            })
        
        # Main element is usually the main content
        main = element.find('main')
        if main:
            sections.append({
                "section_type": "main",
                "element": main
            })
        
        # Look for section elements
        for section in element.find_all('section'):
            sections.append({
                "section_type": "section",
                "element": section
            })
        
        # Find aside elements (sidebars)
        for aside in element.find_all('aside'):
            sections.append({
                "section_type": "aside",
                "element": aside
            })
        
        # Look for header
        header = element.find('header')
        if header:
            sections.append({
                "section_type": "header",
                "element": header
            })
        
        # Look for footer
        footer = element.find('footer')
        if footer:
            sections.append({
                "section_type": "footer",
                "element": footer
            })
        
        # If no semantic sections found, look for div with meaningful classes or ids
        if not sections:
            section_indicators = [
                "content", "main", "article", "post", "body", "entry",
                "text", "page", "container"
            ]
            
            for div in element.find_all('div'):
                attrs = " ".join([
                    " ".join(div.get("class", [])),
                    div.get("id", "")
                ]).lower()
                
                if any(indicator in attrs for indicator in section_indicators):
                    # Make sure it has some content
                    if len(div.get_text(strip=True)) > 100:
                        sections.append({
                            "section_type": "implicit",
                            "element": div
                        })
        
        # If not include_elements, remove the BS4 element from the result
        if not include_elements:
            for section in sections:
                section["element_path"] = self._get_element_path(section["element"])
        
        return sections
    
    def _detect_visual_sections(self, element: Tag, include_elements: bool = True) -> List[Dict[str, Any]]:
        """
        Detect visual sections of the document based on styling cues.
        
        Args:
            element: BeautifulSoup Tag object
            include_elements: Whether to include the BS4 element in the result
            
        Returns:
            List of visual section dictionaries
        """
        visual_sections = []
        
        # Look for elements with container-like classes
        container_classes = [
            "container", "wrapper", "section", "block", "panel", "box",
            "card", "module", "row", "grid", "column"
        ]
        
        selector = ", ".join([f"[class*={cls}]" for cls in container_classes])
        containers = element.select(selector)
        
        for container in containers:
            # Only consider substantial containers
            if len(container.get_text(strip=True)) < 50:
                continue
                
            # Ignore if it's a semantic element we already captured
            if container.name in self._semantic_tags:
                continue
            
            # Extract container information
            container_info = {
                "section_type": "visual",
                "element": container,
                "container_type": "unknown"
            }
            
            # Try to determine container type from classes
            classes = " ".join(container.get("class", []) if container.get("class") else []).lower()
            
            if "header" in classes or "banner" in classes:
                container_info["container_type"] = "header"
            elif "footer" in classes:
                container_info["container_type"] = "footer"
            elif "sidebar" in classes or "aside" in classes:
                container_info["container_type"] = "sidebar"
            elif "content" in classes or "main" in classes:
                container_info["container_type"] = "content"
            elif "nav" in classes or "menu" in classes:
                container_info["container_type"] = "navigation"
            else:
                # Try to guess based on content and position
                if self._has_heading_followed_by_paragraphs(container):
                    container_info["container_type"] = "content_section"
                elif self._is_primarily_links(container):
                    container_info["container_type"] = "navigation"
            
            # Add to sections list if we haven't already included a parent
            parent_already_included = False
            for section in visual_sections:
                if container in section["element"].descendants:
                    parent_already_included = True
                    break
            
            if not parent_already_included:
                visual_sections.append(container_info)
        
        # If not include_elements, remove the BS4 element from the result
        if not include_elements:
            for section in visual_sections:
                section["element_path"] = self._get_element_path(section["element"])
                section.pop("element", None)
        
        return visual_sections
    
    def _has_heading_followed_by_paragraphs(self, element: Tag) -> bool:
        """Check if the element has a heading followed by paragraphs."""
        headings = element.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        if not headings:
            return False
            
        # Check if there's at least one paragraph
        return bool(element.find('p'))
    
    def _is_primarily_links(self, element: Tag) -> bool:
        """Check if the element is primarily composed of links."""
        links = element.find_all('a')
        if not links:
            return False
            
        # Calculate text ratio
        all_text = element.get_text(strip=True)
        if not all_text:
            return False
            
        link_text = "".join([link.get_text(strip=True) for link in links])
        link_ratio = len(link_text) / len(all_text)
        
        return link_ratio > 0.6  # If > 60% of text is in links
    
    def _find_result_containers(self, element: Tag) -> List[Tag]:
        """Find containers that likely hold result items."""
        containers = []
        
        # Check for semantic list elements
        for list_elem in element.find_all(['ul', 'ol']):
            # Only include lists with multiple items
            items = list_elem.find_all('li')
            if len(items) >= 2:
                containers.append(list_elem)
        
        # Check for tables
        for table in element.find_all('table'):
            rows = table.find_all('tr')
            if len(rows) >= 2:  # Header row + at least one data row
                containers.append(table)
        
        # Check for grid layouts
        grid_selectors = [
            ".grid", ".row", ".products", ".items", ".results", ".cards",
            "[class*=grid]", "[class*=list]", "[class*=result]", "[class*=product]"
        ]
        
        for selector in grid_selectors:
            for grid in element.select(selector):
                # Only include if it has multiple children
                children = [child for child in grid.children if isinstance(child, Tag)]
                if len(children) >= 2:
                    containers.append(grid)
        
        # Check for result-like classes
        result_selectors = [
            "[class*=results]", "[class*=listing]", "[class*=search-results]",
            "[id*=results]", "[id*=search]", "[class*=products]"
        ]
        
        for selector in result_selectors:
            for result_elem in element.select(selector):
                # Check if not already included
                if result_elem not in containers:
                    containers.append(result_elem)
        
        return containers
    
    def _identify_result_items(self, container: Tag) -> List[Tag]:
        """Identify individual result items within a container."""
        items = []
        
        # If it's a list, items are <li> elements
        if container.name in ['ul', 'ol']:
            items = container.find_all('li')
            
        # If it's a table, items are rows
        elif container.name == 'table':
            items = container.find_all('tr')[1:]  # Skip header row
            
        # Otherwise, look for repeating elements
        else:
            # First try to find elements with result-item classes
            item_selectors = [
                "[class*=item]", "[class*=result]", "[class*=product]",
                "[class*=card]", "[class*=box]", ".cell", ".col"
            ]
            
            for selector in item_selectors:
                candidates = container.select(selector)
                if len(candidates) >= 2:
                    items = candidates
                    break
            
            # If no items found, try direct children with consistent types
            if not items:
                direct_children = [child for child in container.children if isinstance(child, Tag)]
                
                # Only consider if there are at least 2 children
                if len(direct_children) >= 2:
                    # Group by tag name
                    by_tag = {}
                    for child in direct_children:
                        if child.name not in by_tag:
                            by_tag[child.name] = []
                        by_tag[child.name].append(child)
                    
                    # Use the most common tag
                    most_common = max(by_tag.items(), key=lambda x: len(x[1]), default=(None, []))
                    if most_common[0] and len(most_common[1]) >= 2:
                        items = most_common[1]
        
        return items
    
    def _calculate_structural_similarity(self, items: List[Tag]) -> float:
        """Calculate structural similarity between a list of elements."""
        if not items or len(items) < 2:
            return 0.0
            
        similarities = []
        for i in range(len(items) - 1):
            for j in range(i + 1, len(items)):
                similarities.append(self._calculate_content_similarity(items[i], items[j]))
                
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _calculate_content_similarity(self, elem1: Tag, elem2: Tag) -> float:
        """Calculate similarity between two elements based on their structure and content."""
        if not elem1 or not elem2:
            return 0.0
            
        # Compare tag names
        tag_similarity = 1.0 if elem1.name == elem2.name else 0.0
        
        # Compare class attributes
        class1 = set(elem1.get('class', []))
        class2 = set(elem2.get('class', []))
        class_similarity = 0.0
        if class1 and class2:
            common_classes = class1.intersection(class2)
            class_similarity = len(common_classes) / max(len(class1), len(class2))
        
        # Compare child structure
        children1 = [child.name for child in elem1.find_all(recursive=False) if isinstance(child, Tag)]
        children2 = [child.name for child in elem2.find_all(recursive=False) if isinstance(child, Tag)]
        
        structure_similarity = 0.0
        if children1 and children2:
            # Compare length
            length_similarity = min(len(children1), len(children2)) / max(len(children1), len(children2))
            
            # Compare elements
            common_tags = set(children1).intersection(set(children2))
            tag_set_similarity = len(common_tags) / max(len(set(children1)), len(set(children2)))
            
            structure_similarity = (length_similarity + tag_set_similarity) / 2
        
        # Calculate overall similarity
        return (tag_similarity * 0.3) + (class_similarity * 0.3) + (structure_similarity * 0.4)
    
    def _extract_common_fields(self, items: List[Tag]) -> List[str]:
        """Extract common fields found in a list of items."""
        if not items:
            return []
            
        common_fields = []
        
        # Check for common elements in the first item
        first_item = items[0]
        
        # Look for common fields in items
        potential_fields = {
            "title": ["h1", "h2", "h3", "h4", "h5", "h6", ".title", ".name", "[class*=title]", "[class*=name]"],
            "image": ["img", ".image", ".thumbnail", "[class*=image]", "[class*=thumb]"],
            "price": [".price", "[class*=price]", "[itemprop=price]"],
            "description": ["p", ".description", "[class*=desc]", "[itemprop=description]"],
            "link": ["a", ".link", "[class*=link]"],
            "rating": [".rating", ".stars", "[class*=rating]", "[class*=stars]"],
            "button": ["button", ".button", "[class*=button]", "[class*=btn]"]
        }
        
        for field, selectors in potential_fields.items():
            # Check if most items have this field
            count = 0
            for item in items:
                for selector in selectors:
                    if item.select(selector):
                        count += 1
                        break
            
            if count >= len(items) * 0.7:  # 70% of items should have the field
                common_fields.append(field)
        
        return common_fields
    
    def _has_list_structure(self, element: Tag) -> bool:
        """Check if an element has a list-like structure."""
        # Check for direct children with similar structure
        children = [child for child in element.children if isinstance(child, Tag)]
        if len(children) < 2:
            return False
            
        # Check for similar tag names
        tag_names = [child.name for child in children]
        most_common_tag = max(set(tag_names), key=tag_names.count, default=None)
        common_tag_count = tag_names.count(most_common_tag)
        
        return common_tag_count >= len(children) * 0.7
    
    def _has_table_structure(self, element: Tag) -> bool:
        """Check if an element has a table-like structure."""
        # Check for table-related classes
        classes = " ".join(element.get("class", [])).lower()
        if any(term in classes for term in ["table", "grid", "row", "column"]):
            return True
            
        # Check for regular structure with rows and cells
        children = [child for child in element.children if isinstance(child, Tag)]
        if not children:
            return False
            
        # Check if each "row" has a similar number of children
        row_child_counts = []
        for child in children:
            if isinstance(child, Tag):
                row_child_counts.append(len([c for c in child.children if isinstance(c, Tag)]))
                
        if not row_child_counts or max(row_child_counts) < 2:
            return False
            
        # Check if most rows have the same number of cells
        most_common_count = max(set(row_child_counts), key=row_child_counts.count, default=0)
        consistent_count = row_child_counts.count(most_common_count)
        
        return consistent_count >= len(row_child_counts) * 0.7
    
    def _has_grid_structure(self, element: Tag) -> bool:
        """Check if an element has a grid layout structure."""
        # Check for grid-related classes
        classes = " ".join(element.get("class", [])).lower()
        if any(term in classes for term in ["grid", "cards", "tiles"]):
            return True
            
        # Check style attribute for grid or display: flex
        style = element.get("style", "").lower()
        if "display: grid" in style or "display:grid" in style or "display: flex" in style or "display:flex" in style:
            return True
            
        # Check for consistent child elements with inline-block or float styling
        children = [child for child in element.children if isinstance(child, Tag)]
        if len(children) < 4:  # Need a minimum number of items for a grid
            return False
            
        # Check if children have similar dimensions/classes
        child_classes = [" ".join(child.get("class", [])).lower() for child in children]
        if all(child_classes) and len(set(child_classes)) < len(children) * 0.5:
            return True
            
        return False
    
    def _identify_main_content_area(self, element: Tag) -> Optional[Dict[str, Any]]:
        """Identify the main content area of the page."""
        # First try semantic elements
        main = element.find('main')
        if main:
            return {
                "selector": self._get_element_path(main),
                "type": "semantic",
                "tag": "main"
            }
            
        # Next try article
        article = element.find('article')
        if article:
            return {
                "selector": self._get_element_path(article),
                "type": "semantic",
                "tag": "article"
            }
            
        # Try content-related classes/IDs
        content_selectors = [
            "#content", "#main-content", ".content", ".main-content",
            "[role=main]", ".main", "#main"
        ]
        
        for selector in content_selectors:
            content = element.select_one(selector)
            if content:
                return {
                    "selector": self._get_element_path(content),
                    "type": "class-based",
                    "tag": content.name
                }
                
        # Last resort: find div with most content and least boilerplate
        candidates = element.find_all(['div', 'section'])
        if candidates:
            # Score candidates by content value
            scored_candidates = []
            for candidate in candidates:
                # Skip tiny elements
                if len(candidate.get_text(strip=True)) < 100:
                    continue
                    
                # Skip if likely boilerplate
                if self._detect_boilerplate(candidate):
                    continue
                    
                # Score based on text density and content size
                text = candidate.get_text(strip=True)
                density = self._calculate_text_density(candidate)
                headings = len(candidate.find_all(['h1', 'h2', 'h3']))
                paragraphs = len(candidate.find_all('p'))
                
                score = (len(text) * 0.001) + (density * 5) + (headings * 2) + (paragraphs * 0.5)
                
                scored_candidates.append((candidate, score))
                
            # Sort by score and return highest
            if scored_candidates:
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                best_candidate = scored_candidates[0][0]
                return {
                    "selector": self._get_element_path(best_candidate),
                    "type": "heuristic",
                    "tag": best_candidate.name,
                    "score": scored_candidates[0][1]
                }
                
        return None

    def _map_reading_flow(self, element: Tag) -> List[Dict[str, Any]]:
        """Map the logical reading flow through the document."""
        flow = []
        
        # Find all headings and content blocks in document order
        headings = element.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        paragraphs = element.find_all(['p', 'div > p', 'article > p', 'section > p'])
        
        # Combine and sort by position in document
        all_elements = [(h, 'heading', int(h.name[1])) for h in headings]
        all_elements.extend([(p, 'paragraph', 0) for p in paragraphs])
        
        # Sort by position in document
        all_elements.sort(key=lambda x: str(x[0]).index)
        
        # Convert to flow items
        for elem, elem_type, level in all_elements:
            text = elem.get_text(strip=True)
            if not text:
                continue
                
            if elem_type == 'heading':
                flow.append({
                    "type": "heading",
                    "level": level,
                    "text": text[:100] + ('...' if len(text) > 100 else ''),
                    "selector": self._get_element_path(elem)
                })
            else:
                flow.append({
                    "type": "content",
                    "text_length": len(text),
                    "preview": text[:100] + ('...' if len(text) > 100 else ''),
                    "selector": self._get_element_path(elem)
                })
                
        return flow[:20]  # Limit to first 20 elements for brevity
        
    def _analyze_whitespace_distribution(self, element: Tag) -> Dict[str, Any]:
        """Analyze the use of whitespace in the document structure."""
        # This is challenging with just HTML without CSS
        # We can look for hints in spacing elements
        
        # Count common whitespace elements
        hr_count = len(element.find_all('hr'))
        br_count = len(element.find_all('br'))
        
        # Look for margin/padding hints in style attributes
        margin_elements = len(element.find_all(lambda tag: tag.get('style') and 
                                             ('margin' in tag.get('style').lower() or
                                              'padding' in tag.get('style').lower())))
        
        # Check for elements with spacing classes
        spacing_classes = len(element.find_all(class_=lambda c: c and any(
            term in ' '.join(c).lower() for term in 
            ['spacer', 'divider', 'gap', 'space', 'mt-', 'mb-', 'my-', 'pt-', 'pb-', 'py-']
        )))
        
        return {
            "horizontal_dividers": hr_count,
            "line_breaks": br_count,
            "styled_spacing": margin_elements,
            "spacing_classes": spacing_classes,
            "implicit_whitespace": hr_count + br_count + margin_elements + spacing_classes
        }
        
    def _detect_sidebar(self, element: Tag) -> bool:
        """Detect if the page has a sidebar layout."""
        # Check for semantic aside elements
        aside = element.find('aside')
        if aside:
            return True
            
        # Check for sidebar classes
        sidebar_selectors = [
            "[class*=sidebar]", "[id*=sidebar]", ".rail", ".column",
            "[class*=rail]", "[class*=aside]", "[class*=side-"
        ]
        
        for selector in sidebar_selectors:
            if element.select(selector):
                return True
                
        return False
    
    def _extract_structured_data_types(self, dom: BeautifulSoup) -> List[str]:
        """Extract types from structured data (JSON-LD, microdata)."""
        types = []
        
        # Look for JSON-LD
        for script in dom.find_all('script', type='application/ld+json'):
            try:
                import json
                data = json.loads(script.string)
                
                # Handle both single objects and arrays
                if isinstance(data, list):
                    items = data
                else:
                    items = [data]
                    
                for item in items:
                    if isinstance(item, dict):
                        # Extract @type field
                        item_type = item.get('@type')
                        if item_type:
                            if isinstance(item_type, list):
                                types.extend([t.lower() for t in item_type])
                            else:
                                types.append(item_type.lower())
            except Exception as e:
                logger.debug(f"Error parsing JSON-LD: {str(e)}")
        
        # Look for microdata
        for element in dom.find_all(itemtype=True):
            itemtype = element.get('itemtype', '')
            if itemtype:
                # Extract type from URL (e.g., http://schema.org/Product -> product)
                parts = itemtype.strip('/').split('/')
                if parts:
                    type_name = parts[-1].lower()
                    types.append(type_name)
        
        return types
    
    def _has_article_structure(self, dom: BeautifulSoup) -> bool:
        """Check if the document has an article-like structure."""
        # Check for article element
        if dom.find('article'):
            return True
            
        # Check for typical article components
        has_headline = bool(dom.find(['h1', 'h2']))
        has_paragraphs = len(dom.find_all('p')) >= 3
        has_author = bool(dom.select('.author, .byline, [itemprop="author"]'))
        has_date = bool(dom.select('.date, .published, time, [itemprop="datePublished"]'))
        
        # Check content structure (looking for good amount of text)
        text_length = len(dom.get_text(strip=True))
        paragraphs = dom.find_all('p')
        avg_paragraph_length = sum(len(p.get_text(strip=True)) for p in paragraphs) / max(1, len(paragraphs))
        
        # Calculate article likelihood score
        article_score = 0
        if has_headline: article_score += 1
        if has_paragraphs: article_score += 1
        if has_author: article_score += 1
        if has_date: article_score += 1
        if text_length > 1000: article_score += 1
        if avg_paragraph_length > 100: article_score += 1
        
        return article_score >= 3
    
    def _has_product_structure(self, dom: BeautifulSoup) -> bool:
        """Check if the document has a product page structure."""
        # Check for product schema
        product_schema = any('product' in t.lower() for t in self._extract_structured_data_types(dom))
        if product_schema:
            return True
            
        # Check for typical product components
        has_price = bool(dom.select('.price, [itemprop="price"], [class*="price"]'))
        has_product_image = bool(dom.select('.product-image, [itemprop="image"], [class*="product"] img'))
        has_product_title = bool(dom.select('[itemprop="name"], .product-title, .product-name, [class*="product-title"]'))
        has_add_to_cart = bool(dom.select('button[class*="cart"], [class*="add-to-cart"], [id*="add-to-cart"]'))
        has_product_description = bool(dom.select('[itemprop="description"], .product-description, .description'))
        
        # Calculate product likelihood score
        product_score = 0
        if has_price: product_score += 2  # Price is a strong indicator
        if has_product_image: product_score += 1
        if has_product_title: product_score += 1
        if has_add_to_cart: product_score += 2  # Add to cart is a strong indicator
        if has_product_description: product_score += 1
        
        return product_score >= 3
    
    def _has_listing_structure(self, dom: BeautifulSoup) -> bool:
        """Check if the document has a listing page structure."""
        # Look for product listings
        result_containers = self._find_result_containers(dom)
        
        if not result_containers:
            return False
            
        # Check if containers have multiple similar items
        for container in result_containers:
            items = self._identify_result_items(container)
            if len(items) >= 3:  # Need at least 3 items to be a listing
                similarity = self._calculate_structural_similarity(items)
                if similarity > 0.6:  # Items should be similar
                    return True
        
        return False
    
    def _has_form_dominance(self, dom: BeautifulSoup) -> bool:
        """Check if the page is dominated by form elements."""
        forms = dom.find_all('form')
        
        if not forms:
            return False
            
        # Check if forms are a significant part of the page
        all_text_length = len(dom.get_text(strip=True))
        if all_text_length == 0:
            return False
            
        # Calculate form-related elements
        form_elements = []
        for form in forms:
            form_elements.extend(form.find_all(['input', 'select', 'textarea', 'button', 'label']))
            
        # Look for key form pages
        form_keywords = ['login', 'register', 'sign up', 'contact', 'checkout', 'subscribe']
        page_text = dom.get_text(strip=True).lower()
        has_form_keywords = any(keyword in page_text for keyword in form_keywords)
        
        return len(form_elements) >= 5 or (has_form_keywords and len(forms) > 0)
    
    def _has_search_results_structure(self, dom: BeautifulSoup) -> bool:
        """Check if the page has a search results structure."""
        # Look for search results indicators
        search_indicators = [
            "search results", "search result", "we found", "results for", 
            "items found", "products found", "matching your search"
        ]
        
        # Check text for search indicators
        page_text = dom.get_text(strip=True).lower()
        has_search_text = any(indicator in page_text for indicator in search_indicators)
        
        # Look for search form
        has_search_form = bool(dom.select('form[action*="search"], form input[name*="search"], form input[placeholder*="search"]'))
        
        # Look for URL parameters
        has_search_url = False
        if dom.find('link', rel='canonical'):
            canonical = dom.find('link', rel='canonical').get('href', '')
            has_search_url = 'search' in canonical or 'query' in canonical or 'q=' in canonical
        
        # Check for result list structure
        has_result_list = self._has_listing_structure(dom)
        
        # Combine signals
        return (has_search_text and has_result_list) or (has_search_form and has_result_list) or (has_search_url and has_result_list)
    
    def _detect_layout_grid(self, element: Tag) -> List[Dict[str, Any]]:
        """Detect grid-based layouts in the document."""
        grids = []
        
        # Look for CSS grid layouts
        grid_selectors = [
            "[class*=grid]", "[class*=row]", "[class*=column]",
            "[style*='display: grid']", "[style*='display:grid']",
            "[style*='display: flex']", "[style*='display:flex']"
        ]
        
        for selector in grid_selectors:
            for grid in element.select(selector):
                children = [child for child in grid.children if isinstance(child, Tag)]
                if len(children) >= 3:  # Need at least 3 children for a grid
                    grids.append({
                        "selector": self._get_element_path(grid),
                        "type": "css-based",
                        "child_count": len(children)
                    })
        
        # Look for semantic grid-like structures
        if not grids:
            # Check for evenly distributed children
            for container in element.find_all('div'):
                children = [child for child in container.children if isinstance(child, Tag)]
                if len(children) >= 3:
                    # Check if children have similar structures
                    if self._has_similar_children(container):
                        grids.append({
                            "selector": self._get_element_path(container),
                            "type": "structural",
                            "child_count": len(children)
                        })
        
        return grids
    
    def _has_similar_children(self, element: Tag) -> bool:
        """Check if an element has children with similar structure."""
        children = [child for child in element.children if isinstance(child, Tag)]
        if len(children) < 3:
            return False
            
        # Get tag names and classes
        tag_names = [child.name for child in children]
        child_classes = [tuple(sorted(child.get('class', []))) for child in children]
        
        # Check for tag name consistency
        if len(set(tag_names)) == 1:
            return True
            
        # Check for class consistency
        if len(set(child_classes)) == 1 and child_classes[0]:
            return True
            
        # Check for structural similarity
        structural_similarity = self._calculate_structural_similarity(children)
        return structural_similarity > 0.7
    
    def _identify_content_regions(self, element: Tag) -> List[Dict[str, Any]]:
        """Identify distinct content regions in the document."""
        regions = []
        
        # Identify semantic regions first
        if element.find('main'):
            regions.append({
                "type": "main_content",
                "selector": self._get_element_path(element.find('main'))
            })
            
        if element.find('header'):
            regions.append({
                "type": "header",
                "selector": self._get_element_path(element.find('header'))
            })
            
        if element.find('footer'):
            regions.append({
                "type": "footer",
                "selector": self._get_element_path(element.find('footer'))
            })
            
        if element.find('nav'):
            regions.append({
                "type": "navigation",
                "selector": self._get_element_path(element.find('nav'))
            })
            
        if element.find('aside'):
            regions.append({
                "type": "sidebar",
                "selector": self._get_element_path(element.find('aside'))
            })
        
        # Add content sections
        for section in element.find_all('section'):
            regions.append({
                "type": "content_section",
                "selector": self._get_element_path(section),
                "heading": section.find(['h1', 'h2', 'h3']).get_text(strip=True) if section.find(['h1', 'h2', 'h3']) else None
            })
            
        # Identify article elements
        for article in element.find_all('article'):
            regions.append({
                "type": "article",
                "selector": self._get_element_path(article),
                "heading": article.find(['h1', 'h2', 'h3']).get_text(strip=True) if article.find(['h1', 'h2', 'h3']) else None
            })
        
        return regions
    
    def _is_primary_navigation(self, nav: Tag) -> bool:
        """Determine if a navigation element is the primary navigation."""
        # Check for primary nav indicators in classes or IDs
        attrs = " ".join([
            " ".join(nav.get("class", [])), 
            nav.get("id", "")
        ]).lower()
        
        primary_indicators = ["primary", "main", "top", "header", "principal", "global"]
        if any(indicator in attrs for indicator in primary_indicators):
            return True
            
        # Check if it's in the header
        if nav.find_parent('header'):
            return True
            
        # Check if it's the first nav element
        siblings = list(nav.parent.find_all('nav')) if nav.parent else []
        if siblings and siblings[0] == nav:
            return True
            
        # Check if it has more links than other navs
        siblings = [s for s in siblings if s != nav]
        if siblings:
            nav_links = len(nav.find_all('a'))
            sibling_links = [len(s.find_all('a')) for s in siblings]
            if nav_links > max(sibling_links, default=0):
                return True
        
        # Check position
        if self._is_top_section(nav):
            return True
            
        return False
    
    def _is_top_section(self, element: Tag) -> bool:
        """Check if an element is in the top section of the document."""
        if not element.parent or not element.parent.parent:
            return True
            
        # Calculate position in document
        siblings_before = len(list(element.find_all_previous()))
        total_elements = len(list(element.parent.find_all()))
        
        # If it's in the first 20% of the document
        return siblings_before / total_elements < 0.2 if total_elements else True
    
    def _get_element_position(self, element: Tag) -> Dict[str, Any]:
        """Get the position of an element in the document."""
        if not element:
            return {"relative_position": "unknown"}
            
        try:
            siblings_before = len(list(element.find_all_previous()))
            total_elements = len(list(element.parent.find_all())) if element.parent else 0
            
            if total_elements:
                relative_pos = siblings_before / total_elements
                if relative_pos < 0.25:
                    position = "top"
                elif relative_pos < 0.75:
                    position = "middle"
                else:
                    position = "bottom"
            else:
                position = "unknown"
                
            # Check if it's in a specific section
            in_header = bool(element.find_parent('header'))
            in_footer = bool(element.find_parent('footer'))
            in_sidebar = bool(element.find_parent('aside'))
            in_main = bool(element.find_parent('main'))
            
            container = "unknown"
            if in_header:
                container = "header"
            elif in_footer:
                container = "footer"
            elif in_sidebar:
                container = "sidebar"
            elif in_main:
                container = "main"
                
            return {
                "relative_position": position,
                "container": container
            }
        except Exception:
            return {"relative_position": "unknown"}
    
    def _determine_form_purpose(self, form: Tag) -> str:
        """Determine the purpose of a form based on its content."""
        form_text = form.get_text().lower()
        form_attrs = " ".join([form.get('id', ''), form.get('class', ''), form.get('action', '')]).lower()
        
        # Check for common form types
        if any(term in form_text or term in form_attrs for term in ['login', 'sign in']):
            return "login"
        if any(term in form_text or term in form_attrs for term in ['register', 'sign up', 'create account']):
            return "registration"
        if any(term in form_text or term in form_attrs for term in ['search']):
            return "search"
        if any(term in form_text or term in form_attrs for term in ['comment']):
            return "comment"
        if any(term in form_text or term in form_attrs for term in ['contact', 'feedback']):
            return "contact"
        if any(term in form_text or term in form_attrs for term in ['subscribe', 'newsletter']):
            return "subscription"
        if any(term in form_text or term in form_attrs for term in ['checkout', 'payment', 'billing']):
            return "checkout"
        
        # Check for form elements that give clues
        has_password = bool(form.find('input', {'type': 'password'}))
        has_email = bool(form.find('input', {'type': 'email'}))
        has_search = bool(form.find('input', {'type': 'search'}))
        
        if has_password and has_email:
            return "login_or_registration"
        if has_search:
            return "search"
            
        return "unknown"
    
    def _determine_interactive_region_type(self, element: Tag) -> str:
        """Determine the type of an interactive region."""
        element_attrs = " ".join([
            element.name,
            " ".join(element.get("class", [])),
            element.get("id", ""),
            element.get("role", "")
        ]).lower()
        
        if 'tab' in element_attrs:
            return "tabs"
        if 'accordion' in element_attrs:
            return "accordion"
        if 'dropdown' in element_attrs or 'menu' in element_attrs:
            return "dropdown"
        if 'modal' in element_attrs or 'dialog' in element_attrs:
            return "modal"
        if 'carousel' in element_attrs or 'slider' in element_attrs:
            return "carousel"
        
        return "unknown"
    
    def _find_repeated_elements(self, element: Tag) -> List[Dict[str, Any]]:
        """Find repeated structural elements that might be part of a template."""
        repeated = []
        
        # Look for common repeating patterns
        # 1. Social media links/icons
        social_selectors = [
            "[class*=social]", "[id*=social]", "[class*=share]", "[id*=share]",
            "a[href*=facebook]", "a[href*=twitter]", "a[href*=instagram]", 
            "a[href*=linkedin]", "a[href*=youtube]"
        ]
        
        social_links = element.select(", ".join(social_selectors))
        if len(social_links) >= 2:
            repeated.append({
                "type": "social_links",
                "count": len(social_links),
                "example": self._get_element_path(social_links[0])
            })
        
        # 2. Navigation menus
        nav_repeated = {}
        for nav in element.find_all('nav'):
            menu_items = nav.find_all('li')
            if len(menu_items) >= 3:
                nav_repeated[self._get_element_path(nav)] = len(menu_items)
        
        if nav_repeated:
            repeated.append({
                "type": "navigation_menu",
                "instances": nav_repeated
            })
        
        # 3. Footer links
        footer = element.find('footer')
        if footer:
            footer_links = footer.find_all('a')
            if len(footer_links) >= 4:
                repeated.append({
                    "type": "footer_links",
                    "count": len(footer_links),
                    "container": self._get_element_path(footer)
                })
        
        return repeated