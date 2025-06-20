"""
Pattern Extractor Module

This module provides pattern-based content extraction capabilities for identifying
and extracting structured data from HTML content based on repeating patterns.
"""

import re
import logging
import html
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from bs4 import BeautifulSoup, Tag
import json

from extraction.core.extraction_interface import PatternExtractor
from extraction.core.extraction_result import ExtractionResult
from extraction.helpers.selector_generator import SelectorGenerator
from core.html_service import HTMLService
from components.pattern_analyzer.listing_analyzer import ListingAnalyzer
from core.service_interface import BaseService

# Configure logging
logger = logging.getLogger(__name__)

class DOMPatternExtractor(PatternExtractor, BaseService):
    """
    Extractor for identifying and extracting structured data from HTML 
    content based on repeating DOM patterns.
    """
    
    def __init__(self, context=None):
        """
        Initialize the DOM pattern extractor.
        
        Args:
            context: Strategy context for accessing shared services
        """
        super().__init__(context)
        self.html_service = HTMLService()
        self.selector_generator = SelectorGenerator()
        self.listing_analyzer = ListingAnalyzer(confidence_threshold=0.6, min_items=2)
        self._initialized = False

    @property
    def name(self) -> str:
        """Return the name of the service."""
        return "pattern_extractor"
        
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the service with the given configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        if not self._initialized:
            logger.info("Initializing DOMPatternExtractor service")
            
            if config:
                # Apply any configuration settings if provided
                if "confidence_threshold" in config:
                    self.listing_analyzer.confidence_threshold = config["confidence_threshold"]
                if "min_items" in config:
                    self.listing_analyzer.min_items = config["min_items"]
                    
            self._initialized = True
    
    def shutdown(self) -> None:
        """Clean up any resources used by the extractor."""
        if self._initialized:
            logger.info("Shutting down DOMPatternExtractor")
            self._initialized = False
    
    def can_handle(self, content: Any, content_type: Optional[str] = None) -> bool:
        """
        Check if this extractor can handle the given content.
        
        Args:
            content: Content to check compatibility with
            content_type: Optional hint about the content type
            
        Returns:
            True if the extractor can handle this content, False otherwise
        """
        # Check if content is HTML or HTML-like
        if content_type in ['html', 'text/html']:
            return True
            
        # Check if content looks like HTML
        if isinstance(content, str):
            # Check for HTML tags
            return bool(re.search(r'<[a-z]+[^>]*>.*?</[a-z]+>', content, re.DOTALL | re.IGNORECASE))
            
        return False
    
    def extract(self, content: Any, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract data from HTML content using pattern-based extraction.
        
        Args:
            content: HTML content to extract from
            options: Optional extraction parameters
                - target_pattern: CSS selector to use (bypasses pattern detection)
                - max_items: Maximum number of items to extract
                - extract_pagination: Whether to extract pagination information
                - fields: List of fields to extract for each item
                
        Returns:
            Dictionary containing extracted data and metadata
        """
        if not self._initialized:
            self.initialize()
            
        if not content:
            logger.warning("Empty content provided to extractor")
            return self._create_empty_result("No content provided")
            
        # Process options
        options = options or {}
        max_items = options.get('max_items', 100)  # Default limit
        extract_pagination = options.get('extract_pagination', True)
        target_pattern = options.get('target_pattern')
        fields = options.get('fields', [])
        
        try:
            # If a target pattern is provided, use it directly
            if target_pattern:
                return self._extract_with_pattern(content, target_pattern, max_items, fields)
                
            # Otherwise, detect patterns and extract
            containers = self.identify_result_containers(content)
            
            if not containers:
                logger.info("No result containers identified in content")
                return self._create_empty_result("No pattern containers found")
                
            # Use the highest confidence container
            best_container = max(containers, key=lambda x: x.get('confidence', 0))
            container_selector = best_container.get('selector')
            
            if not container_selector:
                logger.warning("No valid selector found for identified container")
                return self._create_empty_result("Failed to generate container selector")
                
            # Extract data with the identified pattern
            result = self._extract_with_pattern(content, container_selector, max_items, fields)
            
            # Add pagination information if requested
            if extract_pagination:
                pagination_info = self.detect_pagination_pattern(content)
                if pagination_info:
                    result["pagination"] = pagination_info
                    
            # Add pattern analysis metadata
            result["pattern_metadata"] = {
                "container_type": best_container.get("element"),
                "container_selector": container_selector,
                "confidence_score": best_container.get("confidence"),
                "content_type": best_container.get("content_type", "unknown"),
                "detected_structure": self.analyze_result_structure(content, container_selector)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting data: {str(e)}", exc_info=True)
            return self._create_empty_result(f"Extraction error: {str(e)}")
    
    def identify_result_containers(self, html_content: str) -> List[Dict[str, Any]]:
        """
        Find potential result containers in the HTML content.
        
        Args:
            html_content: HTML content to analyze
            
        Returns:
            List of potential container elements with metadata
        """
        try:
            result_containers = []
            
            # Use the listing analyzer to identify containers
            containers = self.listing_analyzer.identify_result_containers(html_content)
            
            # Convert the results to our format
            for container in containers:
                result_containers.append({
                    'selector': container.get('selector'),
                    'confidence': container.get('confidence', 0),
                    'element': container.get('element'),
                    'content_type': container.get('content_type', 'unknown'),
                    'child_count': container.get('child_count', 0)
                })
                
            # As a backup, try to find containers through content analysis methods
            if not result_containers:
                soup = BeautifulSoup(html_content, 'lxml')
                
                # Find dense areas with repeating content
                dense_areas = self.analyze_element_density(soup)
                for area in dense_areas:
                    container = {
                        'selector': self.selector_generator.generate_css_selector(area, html_content),
                        'confidence': 0.6,  # Lower confidence for density-based detection
                        'element': area.name,
                        'content_type': 'unknown',
                        'child_count': len([c for c in area.children if isinstance(c, Tag)])
                    }
                    result_containers.append(container)
                    
                # Find areas with repeating patterns
                repeating_patterns = self.find_repeating_patterns(soup)
                for pattern in repeating_patterns:
                    if (pattern.get('container')):
                        container = {
                            'selector': self.selector_generator.generate_css_selector(pattern['container'], html_content),
                            'confidence': pattern.get('similarity', 0.5),
                            'element': pattern['container'].name,
                            'content_type': 'unknown',
                            'child_count': len(pattern.get('items', []))
                        }
                        # Check if we already have this container
                        if not any(c['selector'] == container['selector'] for c in result_containers):
                            result_containers.append(container)
                            
            # Sort by confidence
            result_containers.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            return result_containers
            
        except Exception as e:
            logger.error(f"Error identifying result containers: {str(e)}", exc_info=True)
            return []
    
    def analyze_result_structure(self, html_content: str, container_selector: str) -> Dict[str, Any]:
        """
        Determine the common structure of results.
        
        Args:
            html_content: HTML content containing the results
            container_selector: CSS selector for the container
            
        Returns:
            Dictionary with structure analysis
        """
        try:
            soup = BeautifulSoup(html_content, 'lxml')
            container = soup.select_one(container_selector)
            
            if not container:
                logger.warning(f"Container not found with selector: {container_selector}")
                return {"error": "Container not found"}
                
            # Try to identify the item selector using listing_analyzer
            item_selector_data = self.listing_analyzer.generate_item_selectors(html_content, container_selector)
            
            # If that doesn't work, try our own methods
            if not item_selector_data.get('success') or not item_selector_data.get('item_selectors'):
                items = self._identify_items_in_container(container)
                item_selector = self._generate_item_selector(container, items)
                
                # Generate field selectors
                field_selectors = self.selector_generator.create_field_selectors(container, items[:3] if items else [])
                
                item_selector_data = {
                    "success": bool(items and item_selector),
                    "item_selectors": [item_selector] if item_selector else [],
                    "field_selectors": field_selectors,
                    "item_count": len(items)
                }
                
            # Convert container markup to a more standardized structure
            structure_analysis = self._analyze_container_structure(container, item_selector_data)
            
            return structure_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing result structure: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    def generate_selectors(self, container: Tag, target_elements: List[Tag]) -> Dict[str, str]:
        """
        Create optimal selectors for the container and its items.
        
        Args:
            container: The container element
            target_elements: The elements to generate selectors for
            
        Returns:
            Dictionary mapping field names to selectors
        """
        result = {}
        
        # Generate container selector
        container_selector = self.selector_generator.generate_css_selector(container, str(container))
        if container_selector:
            result["container"] = container_selector
            
        # If no target elements, just return container selector
        if not target_elements:
            return result
            
        # Generate individual item selector
        first_element = target_elements[0]
        item_selector = self.selector_generator.generate_css_selector(first_element, str(container))
        
        if item_selector:
            result["item"] = item_selector
            
        # Generate field selectors
        field_selectors = self.selector_generator.create_field_selectors(container, target_elements[:3])
        
        # Add field selectors to result
        for field, selector_info in field_selectors.items():
            result[field] = selector_info["selector"]
            
        return result
    
    def validate_selector(self, selector: str, html_content: str) -> Dict[str, Any]:
        """
        Test the reliability of a selector.
        
        Args:
            selector: CSS selector to test
            html_content: HTML content to test against
            
        Returns:
            Dictionary with validation results
        """
        try:
            soup = BeautifulSoup(html_content, 'lxml')
            elements = soup.select(selector)
            
            # Basic validity check
            if not elements:
                return {
                    "valid": False,
                    "count": 0,
                    "message": "Selector doesn't match any elements"
                }
                
            # Check consistency of matched elements
            if len(elements) > 1:
                # Check if elements have similar structure
                structural_similarity = self._calculate_structural_similarity(elements)
                
                return {
                    "valid": True,
                    "count": len(elements),
                    "similarity": structural_similarity,
                    "consistent": structural_similarity > 0.7,
                    "message": "Valid selector" if structural_similarity > 0.7 else "Elements have inconsistent structure"
                }
            else:
                return {
                    "valid": True,
                    "count": 1,
                    "message": "Selector matches a single element"
                }
                
        except Exception as e:
            logger.error(f"Error validating selector: {str(e)}", exc_info=True)
            return {
                "valid": False,
                "message": f"Error validating selector: {str(e)}"
            }
    
    def extract_items(self, container: Tag, selector: str) -> List[Dict[str, Any]]:
        """
        Extract structured data from matched elements.
        
        Args:
            container: Container element
            selector: CSS selector for items within the container
            
        Returns:
            List of extracted items
        """
        extracted_items = []
        
        try:
            # Find all matching elements
            items = container.select(selector)
            
            if not items:
                logger.warning(f"No elements matched selector: {selector}")
                return []
                
            # Generate field selectors from sample items
            field_selectors = self.selector_generator.create_field_selectors(container, items[:3])
            
            # Extract data from each item
            for item in items:
                extracted_item = {}
                
                for field_name, selector_info in field_selectors.items():
                    field_selector = selector_info["selector"]
                    attribute = selector_info["attribute"]
                    
                    try:
                        # Find the element for this field
                        field_elements = item.select(field_selector)
                        
                        if field_elements:
                            field_element = field_elements[0]
                            
                            # Extract the appropriate attribute
                            if attribute == "text":
                                value = field_element.get_text(strip=True)
                            elif attribute == "html":
                                value = str(field_element)
                            else:
                                value = field_element.get(attribute, "")
                                
                            extracted_item[field_name] = value
                    except Exception as e:
                        logger.debug(f"Error extracting field {field_name}: {str(e)}")
                        
                # Only add items that have extracted data
                if extracted_item:
                    extracted_items.append(extracted_item)
                    
            return extracted_items
            
        except Exception as e:
            logger.error(f"Error extracting items: {str(e)}", exc_info=True)
            return []
    
    def map_item_fields(self, item_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Map extracted fields to a standardized schema.
        
        Args:
            item_elements: List of extracted item data
            
        Returns:
            List of items with standardized field names
        """
        mapped_items = []
        
        # Define field mapping patterns
        field_mappings = {
            "title": ["title", "name", "heading", "product_name", "product_title"],
            "price": ["price", "cost", "amount", "product_price"],
            "description": ["description", "desc", "summary", "content", "text"],
            "image": ["image", "img", "thumbnail", "photo", "picture"],
            "url": ["url", "link", "href"]
        }
        
        # Reverse mappings for lookup
        reverse_mappings = {}
        for standard_name, variations in field_mappings.items():
            for variation in variations:
                reverse_mappings[variation] = standard_name
                
        # Add case-insensitive mappings
        for field in list(reverse_mappings.keys()):
            reverse_mappings[field.lower()] = reverse_mappings[field]
        
        for item in item_elements:
            mapped_item = {}
            
            # Copy all fields
            for field, value in item.items():
                # Try to map to standard field name
                standard_field = reverse_mappings.get(field.lower(), field)
                mapped_item[standard_field] = value
                
            # Add the mapped item
            mapped_items.append(mapped_item)
            
        return mapped_items
    
    def find_repeating_patterns(self, container: Union[BeautifulSoup, Tag]) -> List[Dict[str, Any]]:
        """
        Identify recurring DOM patterns within a container.
        
        Args:
            container: BeautifulSoup container to analyze
            
        Returns:
            List of identified repeating patterns
        """
        patterns = []
        
        # Handle both BeautifulSoup and Tag objects
        soup = container
        if not isinstance(container, BeautifulSoup):
            # If it's a Tag, we'll treat it as the container
            soup = BeautifulSoup("<html></html>", "lxml")
            soup.html.append(container)
            
        # Common container tags
        container_tags = ['div', 'ul', 'ol', 'table', 'section', 'article']
        
        # Look for containers with repeating children
        for tag in container_tags:
            for element in soup.find_all(tag):
                # Skip small elements
                if len(element.find_all()) < 5:
                    continue
                    
                # Get direct children that are tags
                children = [c for c in element.children if isinstance(c, Tag)]
                
                # Count occurrences of each tag
                tag_counts = {}
                for child in children:
                    tag_counts[child.name] = tag_counts.get(child.name, 0) + 1
                    
                # Look for tags that appear multiple times
                for tag_name, count in tag_counts.items():
                    if count >= 3:  # At least 3 occurrences to be considered a pattern
                        # Get items with the repeating tag
                        items = [c for c in children if c.name == tag_name]
                        
                        # Calculate similarity between items
                        similarity = self._calculate_structural_similarity(items)
                        
                        # If items are similar enough, consider it a pattern
                        if similarity > 0.6:
                            patterns.append({
                                'container': element,
                                'items': items,
                                'tag_name': tag_name,
                                'count': count,
                                'similarity': similarity
                            })
        
        # Sort by similarity and count
        patterns.sort(key=lambda x: (x['similarity'], x['count']), reverse=True)
        
        return patterns
    
    def analyze_element_density(self, container: Union[BeautifulSoup, Tag]) -> List[Tag]:
        """
        Find content-rich sections based on element density.
        
        Args:
            container: BeautifulSoup container to analyze
            
        Returns:
            List of content-rich elements
        """
        dense_areas = []
        
        # Handle both BeautifulSoup and Tag objects
        soup = container
        if not isinstance(container, BeautifulSoup):
            # If it's a Tag, we'll treat it as the container
            soup = BeautifulSoup("<html></html>", "lxml")
            soup.html.append(container)
            
        # Get all elements that might contain substantive content
        potential_containers = soup.find_all(['div', 'section', 'article', 'main', 'ul', 'ol'])
        
        for element in potential_containers:
            # Skip small elements
            if len(element.find_all()) < 5:
                continue
                
            # Calculate metrics
            total_descendants = len(element.find_all())
            text_length = len(element.get_text())
            sibling_count = len([s for s in element.find_all_next(element.name, recursive=False)])
            
            # Calculate density score
            density_score = (total_descendants * 0.5) + (text_length * 0.0005) + (sibling_count * 0.2)
            
            # Check for indicators of content-rich areas
            has_headings = bool(element.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']))
            has_images = bool(element.find('img'))
            has_links = bool(element.find('a'))
            
            # Adjust score based on content indicators
            if has_headings:
                density_score += 5
            if has_images:
                density_score += 3
            if has_links:
                density_score += 2
                
            # Set a minimum threshold
            if density_score > 10:
                dense_areas.append((element, density_score))
                
        # Sort by density score (highest first)
        dense_areas.sort(key=lambda x: x[1], reverse=True)
        
        # Return the elements, without scores
        return [area[0] for area in dense_areas[:5]]  # Return top 5 densest areas
    
    def detect_pagination_pattern(self, html_content: str) -> Dict[str, Any]:
        """
        Find pagination elements and determine the pattern.
        
        Args:
            html_content: HTML content to analyze
            
        Returns:
            Dictionary with pagination information
        """
        try:
            soup = BeautifulSoup(html_content, 'lxml')
            
            # Dictionary to store pagination info
            pagination_info = {
                "has_pagination": False,
                "next_page": None,
                "prev_page": None,
                "current_page": None,
                "total_pages": None,
                "page_numbers": []
            }
            
            # Look for pagination containers
            pagination_containers = []
            
            # Common pagination container patterns
            pagination_selectors = [
                '.pagination', '.pager', '.pages', '.page-numbers', 
                '[class*="pagination"]', '[class*="pager"]',
                'nav[aria-label*="pagination" i]', '[role="navigation"]'
            ]
            
            for selector in pagination_selectors:
                elements = soup.select(selector)
                pagination_containers.extend(elements)
                
            if not pagination_containers:
                # Try to find other pagination indicators if no container found
                pagination_elements = []
                
                # Look for elements that might be page number links
                number_links = soup.find_all('a', string=re.compile(r'^\d+$'))
                if number_links and len(number_links) > 1:
                    pagination_elements.extend(number_links)
                    
                # Look for next/prev navigation
                next_links = soup.find_all('a', string=re.compile(r'next|>>', re.I)) 
                next_links.extend(soup.find_all('a', title=re.compile(r'next|>>', re.I)))
                
                prev_links = soup.find_all('a', string=re.compile(r'prev|<<|previous', re.I))
                prev_links.extend(soup.find_all('a', title=re.compile(r'prev|<<|previous', re.I)))
                
                pagination_elements.extend(next_links)
                pagination_elements.extend(prev_links)
                
                if pagination_elements:
                    # Find a common parent for these elements
                    parent_elements = []
                    for element in pagination_elements:
                        parent = element.parent
                        if parent not in parent_elements:
                            parent_elements.append(parent)
                            
                    if parent_elements:
                        pagination_containers.extend(parent_elements)
            
            # If we found pagination elements, extract information
            if pagination_containers:
                pagination_info["has_pagination"] = True
                
                # Get the most promising container
                container = pagination_containers[0]
                
                # Find next page link
                next_link = container.find('a', string=re.compile(r'next|>>', re.I))
                if not next_link:
                    next_link = container.find('a', title=re.compile(r'next|>>', re.I))
                    
                if next_link and next_link.get('href'):
                    pagination_info["next_page"] = next_link['href']
                    
                # Find prev page link
                prev_link = container.find('a', string=re.compile(r'prev|<<|previous', re.I))
                if not prev_link:
                    prev_link = container.find('a', title=re.compile(r'prev|<<|previous', re.I))
                    
                if prev_link and prev_link.get('href'):
                    pagination_info["prev_page"] = prev_link['href']
                    
                # Find current page indicator
                current_indicators = container.find_all(class_=re.compile(r'current|active|selected', re.I))
                
                if current_indicators:
                    current_text = current_indicators[0].get_text(strip=True)
                    # Try to extract the page number
                    page_match = re.search(r'\d+', current_text)
                    if page_match:
                        pagination_info["current_page"] = int(page_match.group(0))
                        
                # Find page number links
                page_links = []
                for link in container.find_all('a'):
                    text = link.get_text(strip=True)
                    if re.match(r'^\d+$', text):
                        page_links.append({
                            "number": int(text),
                            "url": link.get('href', '')
                        })
                        
                pagination_info["page_numbers"] = sorted(page_links, key=lambda x: x["number"])
                
                # Try to determine total pages
                if page_links:
                    pagination_info["total_pages"] = max(link["number"] for link in page_links)
            
            return pagination_info
            
        except Exception as e:
            logger.error(f"Error detecting pagination: {str(e)}", exc_info=True)
            return {"has_pagination": False, "error": str(e)}
    
    def extract_list_structure(self, container: Tag) -> Dict[str, Any]:
        """
        Analyze list-like content structures.
        
        Args:
            container: Container element to analyze
            
        Returns:
            Dictionary with list structure analysis
        """
        result = {
            "is_list": False,
            "list_type": None,
            "item_count": 0,
            "items": []
        }
        
        try:
            # Check if container is a list element
            if container.name in ['ul', 'ol']:
                result["is_list"] = True
                result["list_type"] = container.name
                
                # Find list items
                list_items = container.find_all('li', recursive=False)
                result["item_count"] = len(list_items)
                
                # Analyze a sample of list items
                for i, item in enumerate(list_items[:5]):  # Limit to 5 items
                    item_analysis = {
                        "index": i,
                        "text_length": len(item.get_text(strip=True)),
                        "has_links": bool(item.find('a')),
                        "has_images": bool(item.find('img')),
                        "child_elements": len([c for c in item.children if isinstance(c, Tag)])
                    }
                    result["items"].append(item_analysis)
                    
            # Also check for div-based lists
            elif container.name == 'div':
                # Look for repeated child elements
                children = [c for c in container.children if isinstance(c, Tag)]
                
                if children:
                    # Count occurrences of each tag
                    tag_counts = {}
                    for child in children:
                        tag_counts[child.name] = tag_counts.get(child.name, 0) + 1
                        
                    # Find most common tag
                    most_common = None
                    max_count = 0
                    for tag, count in tag_counts.items():
                        if count > max_count:
                            most_common = tag
                            max_count = count
                            
                    # If a tag appears multiple times, treat as a list
                    if max_count >= 3:
                        result["is_list"] = True
                        result["list_type"] = f"div-{most_common}"
                        
                        # Get items of the most common type
                        list_items = [c for c in children if c.name == most_common]
                        result["item_count"] = len(list_items)
                        
                        # Analyze items
                        for i, item in enumerate(list_items[:5]):  # Limit to 5 items
                            item_analysis = {
                                "index": i,
                                "text_length": len(item.get_text(strip=True)),
                                "has_links": bool(item.find('a')),
                                "has_images": bool(item.find('img')),
                                "child_elements": len([c for c in item.children if isinstance(c, Tag)])
                            }
                            result["items"].append(item_analysis)
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing list structure: {str(e)}", exc_info=True)
            return {"is_list": False, "error": str(e)}
    
    def analyze_tabular_data(self, container: Tag) -> Dict[str, Any]:
        """
        Handle and analyze table-based content.
        
        Args:
            container: Container element to analyze
            
        Returns:
            Dictionary with table structure analysis
        """
        result = {
            "is_table": False,
            "headers": [],
            "rows": 0,
            "columns": 0,
            "sample_data": []
        }
        
        try:
            # Check if container is a table or contains a table
            tables = [container] if container.name == 'table' else container.find_all('table')
            
            if not tables:
                return result
                
            # Analyze the first table
            table = tables[0]
            result["is_table"] = True
            
            # Find headers
            headers = []
            header_row = table.find('tr')
            
            if header_row:
                header_cells = header_row.find_all(['th', 'td'])
                for cell in header_cells:
                    headers.append(cell.get_text(strip=True))
                    
            result["headers"] = headers
            
            # Count rows and columns
            rows = table.find_all('tr')
            result["rows"] = len(rows)
            
            if rows:
                # Count columns in the first row
                first_row = rows[0]
                columns = first_row.find_all(['th', 'td'])
                result["columns"] = len(columns)
                
            # Extract sample data (up to 5 rows)
            for i, row in enumerate(rows[:5]):
                if i == 0 and header_row == row:
                    # Skip header row in sample data
                    continue
                    
                row_data = []
                cells = row.find_all(['td', 'th'])
                
                for cell in cells:
                    row_data.append(cell.get_text(strip=True))
                    
                result["sample_data"].append(row_data)
                
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing tabular data: {str(e)}", exc_info=True)
            return {"is_table": False, "error": str(e)}
    
    def detect_pattern(self, content: Any, pattern: Any) -> List[Dict[str, Any]]:
        """
        Detect patterns in the provided content.
        
        Args:
            content: Content to analyze for patterns
            pattern: Pattern to detect (typically a selector or regex)
            
        Returns:
            Dictionary of detected patterns
        """
        if isinstance(pattern, str) and self.can_handle(content):
            # Assuming pattern is a CSS selector
            soup = BeautifulSoup(content, 'lxml')
            elements = soup.select(pattern)
            
            results = []
            for element in elements:
                # Extract data from the element
                element_data = {}
                
                # Extract text content
                text = element.get_text(strip=True)
                if text:
                    element_data["text"] = text
                    
                # Extract attributes
                for attr, value in element.attrs.items():
                    element_data[attr] = value
                    
                # For images, extract src and alt
                if element.name == 'img':
                    element_data["src"] = element.get("src", "")
                    element_data["alt"] = element.get("alt", "")
                    
                # For links, extract href and text
                elif element.name == 'a':
                    element_data["href"] = element.get("href", "")
                    element_data["link_text"] = element.get_text(strip=True)
                    
                results.append(element_data)
                
            return results
            
        return []
    
    def validate_pattern(self, content: Any, pattern: Any) -> bool:
        """
        Validate if a pattern is effective for the given content.
        
        Args:
            content: Content to validate the pattern against
            pattern: Pattern to validate
            
        Returns:
            True if pattern is valid and effective, False otherwise
        """
        if isinstance(pattern, str) and self.can_handle(content):
            # Assuming pattern is a CSS selector
            validation = self.validate_selector(pattern, content)
            return validation.get("valid", False)
            
        return False
    
    def extract_with_pattern(self, content: Any, pattern: Any) -> Dict[str, Any]:
        """
        Extract data using a specific pattern.
        
        Args:
            content: Content to extract data from
            pattern: Pattern to use for extraction
            
        Returns:
            Dictionary of extracted data
        """
        if isinstance(pattern, str) and self.can_handle(content):
            # Use existing method, assuming pattern is a CSS selector
            return self._extract_with_pattern(content, pattern)
            
        return {}
    
    def _extract_with_pattern(self, html_content: str, container_selector: str, 
                           max_items: int = 100, fields: List[str] = None) -> Dict[str, Any]:
        """
        Extract data using a specific container selector.
        
        Args:
            html_content: HTML content to extract from
            container_selector: CSS selector for the container
            max_items: Maximum number of items to extract
            fields: List of specific fields to extract
            
        Returns:
            Dictionary with extracted items and metadata
        """
        try:
            soup = BeautifulSoup(html_content, 'lxml')
            container = soup.select_one(container_selector)
            
            if not container:
                logger.warning(f"Container not found with selector: {container_selector}")
                return self._create_empty_result("Container not found")
                
            # Get item selector using the pattern analyzer
            item_data = self.listing_analyzer.generate_item_selectors(html_content, container_selector)
            
            # If no item selector found, try our own methods
            if not item_data.get('success') or not item_data.get('item_selectors'):
                items = self._identify_items_in_container(container)
                if not items:
                    logger.warning("No items found in container")
                    return self._create_empty_result("No items found in container")
                    
                item_selector = self._generate_item_selector(container, items)
                
            else:
                # Use the item selector from the analyzer
                item_selector = item_data['item_selectors'][0] if item_data['item_selectors'] else None
                
            if not item_selector:
                logger.warning("Failed to generate item selector")
                return self._create_empty_result("Failed to generate item selector")
                
            # Extract items using the selector
            extracted_items = self.extract_items(container, item_selector)
            
            # Limit the number of items
            if max_items and len(extracted_items) > max_items:
                extracted_items = extracted_items[:max_items]
                
            # Map to standard fields
            mapped_items = self.map_item_fields(extracted_items)
            
            # Filter fields if requested
            if fields:
                filtered_items = []
                for item in mapped_items:
                    filtered_item = {field: item.get(field, "") for field in fields if field in item}
                    filtered_items.append(filtered_item)
                mapped_items = filtered_items
                
            # Calculate confidence based on number of items and fields
            confidence = min(1.0, len(mapped_items) / 10) if mapped_items else 0.0
                
            result = {
                "items": mapped_items,
                "count": len(mapped_items),
                "container_selector": container_selector,
                "item_selector": item_selector,
                "confidence": confidence
            }
            
            # Add field information
            if mapped_items:
                sample_item = mapped_items[0]
                result["fields"] = list(sample_item.keys())
                
            return result
            
        except Exception as e:
            logger.error(f"Error extracting with pattern: {str(e)}", exc_info=True)
            return self._create_empty_result(f"Extraction error: {str(e)}")
    
    def _identify_items_in_container(self, container: Tag) -> List[Tag]:
        """
        Identify probable list items within a container.
        
        Args:
            container: Container element
            
        Returns:
            List of item elements
        """
        # If it's already a list, use the list items
        if container.name in ['ul', 'ol']:
            items = container.find_all('li', recursive=False)
            if items:
                return items
                
        # Look for repeating structures
        children = [c for c in container.children if isinstance(c, Tag)]
        
        # Count occurrences of each tag
        tag_counts = {}
        for child in children:
            tag_counts[child.name] = tag_counts.get(child.name, 0) + 1
            
        # Find the most common tag with multiple occurrences
        common_tags = [(tag, count) for tag, count in tag_counts.items() if count >= 2]
        
        if common_tags:
            # Sort by frequency
            common_tags.sort(key=lambda x: x[1], reverse=True)
            most_common_tag = common_tags[0][0]
            
            # Get items with the most common tag
            items = [c for c in children if c.name == most_common_tag]
            
            # Verify items have similar structure
            if self._calculate_structural_similarity(items) > 0.5:
                return items
                
        # Look for items with semantic classes
        item_classes = ['item', 'product', 'result', 'card', 'article', 'post']
        for cls in item_classes:
            items = container.find_all(class_=re.compile(rf'{cls}', re.I))
            if len(items) >= 2:
                return items
                
        # Last resort: any direct children that look consistent
        if children and len(children) >= 2:
            # Check for structural similarity
            if self._calculate_structural_similarity(children) > 0.5:
                return children
                
        return []
    
    def _generate_item_selector(self, container: Tag, items: List[Tag]) -> str:
        """
        Generate a CSS selector for list items.
        
        Args:
            container: Container element
            items: List of item elements
            
        Returns:
            CSS selector string
        """
        if not items:
            return ""
            
        # Get a sample item
        sample = items[0]
        
        # Try tag and class selector
        if sample.get('class'):
            selector = f"{sample.name}.{'.'.join(sample['class'])}"
            return selector
            
        # Try direct child selector if items are direct children
        if sample.parent == container:
            # Check if all items have the same tag
            if all(item.name == sample.name for item in items):
                return f"{sample.name}"
                
        # Try any tag in container
        return f"{sample.name}"
    
    def _calculate_structural_similarity(self, elements: List[Tag]) -> float:
        """
        Calculate how structurally similar a set of elements are.
        
        Args:
            elements: List of elements to compare
            
        Returns:
            Similarity score (0-1)
        """
        if not elements or len(elements) < 2:
            return 0.0
            
        # Generate structural signatures
        signatures = []
        for element in elements:
            # Create a simple signature based on tag names and counts
            tags = {}
            for child in element.find_all():
                tags[child.name] = tags.get(child.name, 0) + 1
                
            # Convert to a sorted list of (tag, count) for comparison
            sig = sorted(tags.items())
            signatures.append(sig)
            
        # Compare each signature to the first one and calculate similarity
        similarities = []
        first_sig = signatures[0]
        
        for sig in signatures[1:]:
            # Calculate Jaccard similarity
            first_tags = set(t[0] for t in first_sig)
            other_tags = set(t[0] for t in sig)
            
            if not first_tags and not other_tags:
                similarities.append(1.0)  # Both empty
            elif not first_tags or not other_tags:
                similarities.append(0.0)  # One is empty
            else:
                intersection = first_tags.intersection(other_tags)
                union = first_tags.union(other_tags)
                similarities.append(len(intersection) / len(union))
                
        # Return average similarity
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _analyze_container_structure(self, container: Tag, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the structure of a container and its items.
        
        Args:
            container: Container element
            item_data: Item selector data
            
        Returns:
            Dictionary with structure analysis
        """
        structure = {
            "container_tag": container.name,
            "item_selectors": item_data.get("item_selectors", []),
            "field_selectors": item_data.get("field_selectors", {}),
            "item_count": item_data.get("item_count", 0)
        }
        
        # Check if it's a list
        list_structure = self.extract_list_structure(container)
        if list_structure.get("is_list"):
            structure["is_list"] = True
            structure["list_type"] = list_structure.get("list_type")
            structure["list_items"] = list_structure.get("items", [])
            
        # Check if it's a table
        table_structure = self.analyze_tabular_data(container)
        if table_structure.get("is_table"):
            structure["is_table"] = True
            structure["table_headers"] = table_structure.get("headers", [])
            structure["rows"] = table_structure.get("rows", 0)
            structure["columns"] = table_structure.get("columns", 0)
            structure["sample_data"] = table_structure.get("sample_data", [])
            
        # If it's neither a list nor a table, provide general structure info
        if not structure.get("is_list") and not structure.get("is_table"):
            # Count element types
            tag_counts = {}
            for element in container.find_all():
                tag_counts[element.name] = tag_counts.get(element.name, 0) + 1
                
            structure["element_counts"] = tag_counts
            
            # Get hierarchy depth
            depth = 0
            deepest = container
            while deepest.find():
                depth += 1
                deepest = deepest.find()
                
            structure["hierarchy_depth"] = depth
            
        return structure
    
    def _create_empty_result(self, message: str) -> Dict[str, Any]:
        """
        Create an empty result with an error message.
        
        Args:
            message: Error message
            
        Returns:
            Empty result dictionary
        """
        return {
            "items": [],
            "count": 0,
            "error": message,
            "confidence": 0.0
        }

    def detect_patterns(self, content: Any) -> List[Dict[str, Any]]:
        """
        Detect patterns in the provided content.
        
        Args:
            content: Content to analyze for patterns
            
        Returns:
            List of detected patterns
        """
        if not self.can_handle(content):
            return []
            
        try:
            # Use the existing methods to identify containers and patterns
            containers = self.identify_result_containers(content)
            
            results = []
            for container in containers:
                pattern_info = {
                    "type": "container",
                    "selector": container.get('selector'),
                    "confidence": container.get('confidence', 0),
                    "element": container.get('element'),
                    "content_type": container.get('content_type', 'unknown'),
                    "child_count": container.get('child_count', 0)
                }
                results.append(pattern_info)
                
            # If we found containers, also try to find pagination patterns
            if results:
                pagination_info = self.detect_pagination_pattern(content)
                if pagination_info and pagination_info.get("has_pagination"):
                    pagination_pattern = {
                        "type": "pagination",
                        "has_pagination": True,
                        "next_page": pagination_info.get("next_page"),
                        "prev_page": pagination_info.get("prev_page"),
                        "current_page": pagination_info.get("current_page"),
                        "confidence": 0.8 if pagination_info.get("next_page") else 0.5
                    }
                    results.append(pagination_pattern)
            
            return results
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {str(e)}", exc_info=True)
            return []