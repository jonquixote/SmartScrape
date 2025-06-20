"""
DOM Analyzer Utilities

This module provides utility functions for analyzing HTML DOM structures,
calculating text density, detecting boilerplate elements, and other
DOM-related analytical operations.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from bs4 import BeautifulSoup, Tag, NavigableString

# Configure logging
logger = logging.getLogger(__name__)

def calculate_text_density(element: Tag) -> float:
    """
    Calculate the ratio of text content to HTML code.
    
    Args:
        element: BeautifulSoup Tag to analyze
        
    Returns:
        Float value representing text-to-code ratio (0.0-1.0)
    """
    if not element:
        return 0.0
        
    try:
        # Get HTML length (length of the string representation of the tag)
        html_length = len(str(element))
        if html_length == 0:
            return 0.0
            
        # Get text length (stripped of whitespace)
        text_length = len(element.get_text(strip=True))
        
        # Calculate density ratio
        return text_length / html_length
    except Exception as e:
        logger.warning(f"Error calculating text density: {str(e)}")
        return 0.0

def detect_boilerplate(element: Tag) -> bool:
    """
    Identify if an element is likely to be boilerplate content.
    
    Args:
        element: BeautifulSoup Tag to analyze
        
    Returns:
        True if element is likely boilerplate, False otherwise
    """
    if not element:
        return False
        
    try:
        # Check element name - common boilerplate elements
        if element.name in ["nav", "header", "footer", "aside"]:
            return True
            
        # Check element classes and ID for boilerplate indicators
        boilerplate_indicators = [
            "menu", "navbar", "navigation", "footer", "header", 
            "sidebar", "banner", "copyright", "social", "share",
            "related", "widget", "ad", "advertisement", "promo"
        ]
        
        # Combine classes and ID into a single string for checking
        element_attrs = " ".join([
            " ".join(element.get("class", [])),
            element.get("id", ""),
            element.get("role", "")
        ]).lower()
        
        # Check for boilerplate indicators in attributes
        for indicator in boilerplate_indicators:
            if indicator in element_attrs:
                return True
        
        # Check link density - high link density often indicates navigation/boilerplate
        links = element.find_all("a")
        text = element.get_text(strip=True)
        
        if links and text:
            link_text = "".join([link.get_text(strip=True) for link in links])
            link_ratio = len(link_text) / len(text) if len(text) > 0 else 0
            
            # If more than 60% of text is in links, likely boilerplate
            if link_ratio > 0.6:
                return True
                
        return False
    except Exception as e:
        logger.warning(f"Error detecting boilerplate: {str(e)}")
        return False

def find_headline_elements(dom: Union[BeautifulSoup, Tag]) -> List[Tag]:
    """
    Locate headline and title elements in the DOM.
    
    Args:
        dom: BeautifulSoup or Tag object to analyze
        
    Returns:
        List of tags that represent headlines or titles
    """
    headlines = []
    
    try:
        # Find standard heading tags
        heading_tags = dom.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        headlines.extend(heading_tags)
        
        # Find elements with title-related classes or IDs
        title_elements = dom.find_all(class_=lambda c: c and any(
            title_indicator in " ".join(c).lower() 
            for title_indicator in ["title", "headline", "heading"]
        ))
        
        # Add unique title elements that aren't already in headlines
        for elem in title_elements:
            if elem not in headlines:
                headlines.append(elem)
                
        # Find elements with title-related IDs
        title_by_id = dom.find_all(id=lambda i: i and any(
            title_indicator in i.lower() 
            for title_indicator in ["title", "headline", "heading"]
        ))
        
        # Add unique title elements that aren't already in headlines
        for elem in title_by_id:
            if elem not in headlines:
                headlines.append(elem)
                
        # Find elements with schema.org title markup
        schema_titles = dom.find_all(itemprop="name") + dom.find_all(itemprop="headline")
        for elem in schema_titles:
            if elem not in headlines:
                headlines.append(elem)
                
        return headlines
    except Exception as e:
        logger.warning(f"Error finding headline elements: {str(e)}")
        return []

def analyze_element_attributes(element: Tag) -> Dict[str, Any]:
    """
    Extract semantic attributes and metadata from an element.
    
    Args:
        element: BeautifulSoup Tag to analyze
        
    Returns:
        Dictionary of semantic attributes and metadata
    """
    attributes = {
        "tag_name": element.name,
        "id": element.get("id", ""),
        "classes": element.get("class", []),
        "semantic_role": element.get("role", ""),
        "aria_attributes": {},
        "microdata": {},
        "link_data": {},
        "text_content": element.get_text(strip=True)[:100]  # Truncate long text
    }
    
    try:
        # Extract ARIA attributes
        for attr_name in element.attrs:
            if attr_name.startswith("aria-"):
                attributes["aria_attributes"][attr_name] = element[attr_name]
                
        # Extract microdata attributes
        if element.has_attr("itemprop"):
            attributes["microdata"]["itemprop"] = element["itemprop"]
        if element.has_attr("itemtype"):
            attributes["microdata"]["itemtype"] = element["itemtype"]
        if element.has_attr("itemscope"):
            attributes["microdata"]["itemscope"] = True
            
        # Extract link data if element is a link
        if element.name == "a":
            attributes["link_data"] = {
                "href": element.get("href", ""),
                "target": element.get("target", ""),
                "rel": element.get("rel", []),
                "title": element.get("title", ""),
                "text": element.get_text(strip=True)
            }
            
        # Extract image data if element is an image
        elif element.name == "img":
            attributes["image_data"] = {
                "src": element.get("src", ""),
                "alt": element.get("alt", ""),
                "width": element.get("width", ""),
                "height": element.get("height", ""),
                "loading": element.get("loading", "")
            }
            
        # Extract form field data for input elements
        elif element.name == "input":
            attributes["form_field_data"] = {
                "type": element.get("type", "text"),
                "name": element.get("name", ""),
                "value": element.get("value", ""),
                "placeholder": element.get("placeholder", ""),
                "required": element.has_attr("required")
            }
            
        return attributes
    except Exception as e:
        logger.warning(f"Error analyzing element attributes: {str(e)}")
        return attributes

def detect_list_structures(dom: Union[BeautifulSoup, Tag]) -> List[Dict[str, Any]]:
    """
    Identify list patterns and structures in the DOM.
    
    Args:
        dom: BeautifulSoup or Tag object to analyze
        
    Returns:
        List of dictionaries containing list structure information
    """
    list_structures = []
    
    try:
        # Find semantic list elements
        semantic_lists = dom.find_all(['ul', 'ol', 'dl'])
        for list_elem in semantic_lists:
            items = list_elem.find_all(['li', 'dt', 'dd'])
            if len(items) >= 2:  # Only consider lists with at least 2 items
                list_structures.append({
                    "type": "semantic",
                    "tag": list_elem.name,
                    "selector": generate_selector(list_elem),
                    "item_count": len(items),
                    "item_selector": f"{generate_selector(list_elem)} > {items[0].name}"
                })
                
        # Find non-semantic lists (repeated div/section patterns)
        containers = dom.find_all(['div', 'section', 'article'])
        for container in containers:
            # Skip small containers
            children = [c for c in container.children if isinstance(c, Tag)]
            if len(children) < 3:
                continue
                
            # Check if children have similar structure (indicating a list)
            similarity_score = calculate_child_similarity(children)
            if similarity_score > 0.7:  # If 70% similarity, likely a list
                list_structures.append({
                    "type": "non-semantic",
                    "tag": container.name,
                    "selector": generate_selector(container),
                    "item_count": len(children),
                    "item_selector": f"{generate_selector(container)} > {children[0].name}",
                    "similarity_score": similarity_score
                })
                
        # Detect table structures which often represent lists
        tables = dom.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            if len(rows) > 1:  # Need more than one row
                list_structures.append({
                    "type": "table",
                    "tag": "table",
                    "selector": generate_selector(table),
                    "item_count": len(rows),
                    "item_selector": f"{generate_selector(table)} > tbody > tr"
                })
                
        return list_structures
    except Exception as e:
        logger.warning(f"Error detecting list structures: {str(e)}")
        return []

def calculate_content_similarity(element1: Tag, element2: Tag) -> float:
    """
    Calculate similarity between two elements based on structure and content.
    
    Args:
        element1: First BeautifulSoup Tag to compare
        element2: Second BeautifulSoup Tag to compare
        
    Returns:
        Float value between 0.0 and 1.0 representing similarity
    """
    if not element1 or not element2:
        return 0.0
        
    try:
        similarity_scores = []
        
        # Compare tag names (1.0 if same, 0.0 if different)
        tag_similarity = 1.0 if element1.name == element2.name else 0.0
        similarity_scores.append(tag_similarity)
        
        # Compare classes
        classes1 = set(element1.get('class', []))
        classes2 = set(element2.get('class', []))
        if classes1 and classes2:
            # Calculate Jaccard similarity: intersection / union
            class_similarity = len(classes1.intersection(classes2)) / len(classes1.union(classes2))
        else:
            class_similarity = 1.0 if not classes1 and not classes2 else 0.0
        similarity_scores.append(class_similarity)
        
        # Compare direct children types
        children1 = [child.name for child in element1.children if isinstance(child, Tag)]
        children2 = [child.name for child in element2.children if isinstance(child, Tag)]
        
        if children1 and children2:
            # Calculate child tag frequency similarity
            children1_count = {tag: children1.count(tag) for tag in set(children1)}
            children2_count = {tag: children2.count(tag) for tag in set(children2)}
            
            # Get all unique tags
            all_tags = set(children1_count.keys()).union(set(children2_count.keys()))
            
            # Calculate difference for each tag
            tag_diffs = []
            for tag in all_tags:
                count1 = children1_count.get(tag, 0)
                count2 = children2_count.get(tag, 0)
                max_count = max(count1, count2)
                if max_count > 0:
                    diff = abs(count1 - count2) / max_count
                    tag_diffs.append(1.0 - diff)  # Convert to similarity
            
            # Average the tag differences
            children_similarity = sum(tag_diffs) / len(tag_diffs) if tag_diffs else 0.0
        else:
            children_similarity = 1.0 if not children1 and not children2 else 0.0
        similarity_scores.append(children_similarity)
        
        # Compare text length (as a rough content indicator)
        text1 = element1.get_text(strip=True)
        text2 = element2.get_text(strip=True)
        if text1 and text2:
            len1, len2 = len(text1), len(text2)
            text_length_similarity = min(len1, len2) / max(len1, len2)
        else:
            text_length_similarity = 1.0 if not text1 and not text2 else 0.0
        similarity_scores.append(text_length_similarity)
        
        # Calculate weighted average
        weights = [0.3, 0.2, 0.3, 0.2]  # Weights for different factors
        overall_similarity = sum(score * weight for score, weight in zip(similarity_scores, weights))
        
        return overall_similarity
    except Exception as e:
        logger.warning(f"Error calculating content similarity: {str(e)}")
        return 0.0

def detect_visual_sections(dom: Union[BeautifulSoup, Tag]) -> List[Dict[str, Any]]:
    """
    Use styling cues to identify visual sections in the document.
    
    Args:
        dom: BeautifulSoup or Tag object to analyze
        
    Returns:
        List of dictionaries representing visual sections
    """
    visual_sections = []
    
    try:
        # Look for elements with container-like classes
        container_classes = [
            "container", "section", "wrapper", "box", "panel", "card", 
            "block", "module", "row", "column", "grid", "content"
        ]
        
        # Build a CSS selector for container-like elements
        container_selector = ", ".join([f"[class*={cls}]" for cls in container_classes])
        container_elements = dom.select(container_selector)
        
        # Filter out small containers and analyze each potential visual section
        for container in container_elements:
            # Skip if container has no substantial content
            if len(container.get_text(strip=True)) < 50:
                continue
                
            section_info = {
                "type": "visual_section",
                "tag": container.name,
                "selector": generate_selector(container),
                "classes": container.get("class", []),
                "text_length": len(container.get_text(strip=True)),
                "visual_type": "unknown"
            }
            
            # Try to determine section type from classes
            container_class_str = " ".join(container.get("class", [])).lower()
            
            if any(term in container_class_str for term in ["header", "banner", "top"]):
                section_info["visual_type"] = "header"
            elif any(term in container_class_str for term in ["footer", "bottom"]):
                section_info["visual_type"] = "footer"
            elif any(term in container_class_str for term in ["sidebar", "aside", "rail"]):
                section_info["visual_type"] = "sidebar"
            elif any(term in container_class_str for term in ["main", "content", "article"]):
                section_info["visual_type"] = "main_content"
            elif any(term in container_class_str for term in ["nav", "menu", "navbar"]):
                section_info["visual_type"] = "navigation"
            elif any(term in container_class_str for term in ["hero", "banner", "jumbotron"]):
                section_info["visual_type"] = "hero"
            elif any(term in container_class_str for term in ["feature", "card", "product"]):
                section_info["visual_type"] = "feature"
            
            # Add content structure information
            section_info["has_heading"] = bool(container.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']))
            section_info["link_count"] = len(container.find_all('a'))
            section_info["image_count"] = len(container.find_all('img'))
            section_info["paragraph_count"] = len(container.find_all('p'))
            
            visual_sections.append(section_info)
            
        return visual_sections
    except Exception as e:
        logger.warning(f"Error detecting visual sections: {str(e)}")
        return []

def analyze_whitespace_distribution(dom: Union[BeautifulSoup, Tag]) -> Dict[str, Any]:
    """
    Analyze the distribution of whitespace to identify content boundaries.
    
    Args:
        dom: BeautifulSoup or Tag object to analyze
        
    Returns:
        Dictionary with whitespace analysis results
    """
    try:
        # Count whitespace indicators
        hr_elements = len(dom.find_all('hr'))  # Horizontal rules
        br_elements = len(dom.find_all('br'))  # Line breaks
        
        # Find elements with margin/padding style attributes
        margin_elements = dom.find_all(lambda tag: tag.get('style') and 
                                               ('margin' in tag.get('style').lower() or
                                                'padding' in tag.get('style').lower()))
        
        # Find elements with spacing-related classes
        spacing_classes = dom.find_all(class_=lambda c: c and any(
            term in " ".join(c).lower() for term in 
            ["spacer", "divider", "gap", "space", "mt-", "mb-", "my-", "pt-", "pb-", "py-"]
        ))
        
        # Check for CSS Grid or Flexbox layout indicators
        grid_elements = dom.find_all(lambda tag: tag.get('style') and 
                                            ('display: grid' in tag.get('style').lower() or
                                             'display: flex' in tag.get('style').lower()))
        
        # Count elements with explicit width/height styling
        sized_elements = dom.find_all(lambda tag: tag.get('style') and 
                                             ('width' in tag.get('style').lower() or
                                              'height' in tag.get('style').lower()))
        
        # Return whitespace distribution analysis
        return {
            "horizontal_dividers": hr_elements,
            "line_breaks": br_elements,
            "margin_padding_elements": len(margin_elements),
            "spacing_classes": len(spacing_classes),
            "grid_flex_layouts": len(grid_elements),
            "explicit_sizing": len(sized_elements),
            "total_whitespace_indicators": (hr_elements + br_elements + 
                                           len(margin_elements) + len(spacing_classes)),
            "layout_complexity": "high" if len(grid_elements) > 5 else "medium" if len(grid_elements) > 0 else "low"
        }
    except Exception as e:
        logger.warning(f"Error analyzing whitespace distribution: {str(e)}")
        return {
            "error": str(e),
            "horizontal_dividers": 0,
            "line_breaks": 0,
            "margin_padding_elements": 0,
            "spacing_classes": 0,
            "grid_flex_layouts": 0,
            "explicit_sizing": 0,
            "total_whitespace_indicators": 0,
            "layout_complexity": "unknown"
        }

def detect_layout_grid(dom: Union[BeautifulSoup, Tag]) -> List[Dict[str, Any]]:
    """
    Identify grid-based layouts in the document.
    
    Args:
        dom: BeautifulSoup or Tag object to analyze
        
    Returns:
        List of dictionaries representing grid layouts
    """
    grids = []
    
    try:
        # Find explicit CSS grid containers
        css_grid_selectors = [
            "[style*='display: grid']", 
            "[style*='display:grid']",
            ".grid",
            "[class*='grid-']",
            "[class*='row']"
        ]
        
        grid_selector = ", ".join(css_grid_selectors)
        grid_elements = dom.select(grid_selector)
        
        for grid in grid_elements:
            grid_info = {
                "type": "css_grid",
                "tag": grid.name,
                "selector": generate_selector(grid),
                "classes": grid.get("class", []),
                "child_count": len([c for c in grid.children if isinstance(c, Tag)])
            }
            
            # Extract grid styling if available
            if grid.get('style'):
                style = grid.get('style')
                grid_info["style"] = style
                
                # Try to extract grid properties
                if 'grid-template-columns' in style:
                    grid_info["columns"] = style.split('grid-template-columns:')[1].split(';')[0].strip()
                if 'grid-template-rows' in style:
                    grid_info["rows"] = style.split('grid-template-rows:')[1].split(';')[0].strip()
                    
            grids.append(grid_info)
        
        # Find flex containers
        flex_selectors = [
            "[style*='display: flex']", 
            "[style*='display:flex']",
            ".flex", 
            "[class*='flex-']", 
            ".d-flex"
        ]
        
        flex_selector = ", ".join(flex_selectors)
        flex_elements = dom.select(flex_selector)
        
        for flex in flex_elements:
            flex_info = {
                "type": "flexbox",
                "tag": flex.name,
                "selector": generate_selector(flex),
                "classes": flex.get("class", []),
                "child_count": len([c for c in flex.children if isinstance(c, Tag)])
            }
            
            # Extract flex styling if available
            if flex.get('style'):
                style = flex.get('style')
                flex_info["style"] = style
                
                # Try to extract flex properties
                if 'flex-direction' in style:
                    flex_info["direction"] = style.split('flex-direction:')[1].split(';')[0].strip()
                if 'justify-content' in style:
                    flex_info["justify"] = style.split('justify-content:')[1].split(';')[0].strip()
                    
            grids.append(flex_info)
            
        # Detect implicit grids (elements with similar children in a grid-like layout)
        containers = dom.find_all(['div', 'section', 'ul'])
        for container in containers:
            if container in grid_elements or container in flex_elements:
                continue  # Skip already identified grids
                
            children = [c for c in container.children if isinstance(c, Tag)]
            if len(children) >= 3:  # Need at least 3 children for a grid
                # Check if children have similar structure
                if has_similar_items(children):
                    grids.append({
                        "type": "implicit_grid",
                        "tag": container.name,
                        "selector": generate_selector(container),
                        "classes": container.get("class", []),
                        "child_count": len(children),
                        "similarity": calculate_child_similarity(children)
                    })
        
        return grids
    except Exception as e:
        logger.warning(f"Error detecting layout grids: {str(e)}")
        return []

def map_reading_flow(dom: Union[BeautifulSoup, Tag]) -> List[Dict[str, Any]]:
    """
    Map the logical reading flow through the document.
    
    Args:
        dom: BeautifulSoup or Tag object to analyze
        
    Returns:
        List of dictionaries representing content in reading order
    """
    reading_flow = []
    
    try:
        # Get all content elements in document order
        content_elements = []
        
        # Find headings
        headings = dom.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        for heading in headings:
            heading_level = int(heading.name[1])
            content_elements.append({
                "type": "heading",
                "level": heading_level,
                "element": heading,
                "text": heading.get_text(strip=True),
                "position": get_element_position(heading)
            })
            
        # Find paragraphs
        paragraphs = dom.find_all('p')
        for para in paragraphs:
            # Skip empty paragraphs
            if not para.get_text(strip=True):
                continue
                
            content_elements.append({
                "type": "paragraph",
                "element": para,
                "text": para.get_text(strip=True),
                "position": get_element_position(para),
                "length": len(para.get_text(strip=True))
            })
            
        # Find lists
        lists = dom.find_all(['ul', 'ol'])
        for list_elem in lists:
            content_elements.append({
                "type": "list",
                "element": list_elem,
                "items": len(list_elem.find_all('li')),
                "position": get_element_position(list_elem),
                "text": list_elem.get_text(strip=True)[:100] + '...' if len(list_elem.get_text(strip=True)) > 100 else list_elem.get_text(strip=True)
            })
            
        # Find tables
        tables = dom.find_all('table')
        for table in tables:
            content_elements.append({
                "type": "table",
                "element": table,
                "rows": len(table.find_all('tr')),
                "position": get_element_position(table)
            })
            
        # Find images with captions
        images = dom.find_all('img')
        for img in images:
            # Only include substantial images (skip icons, etc.)
            if img.get('width') and img.get('height'):
                try:
                    width = int(img['width'])
                    height = int(img['height'])
                    if width < 50 or height < 50:
                        continue  # Skip small images
                except (ValueError, TypeError):
                    pass  # Continue if we can't determine size
                    
            # Look for captions (figcaption or nearby text)
            caption = ""
            if img.parent and img.parent.name == 'figure':
                figcaption = img.parent.find('figcaption')
                if figcaption:
                    caption = figcaption.get_text(strip=True)
                    
            content_elements.append({
                "type": "image",
                "element": img,
                "position": get_element_position(img),
                "caption": caption,
                "alt": img.get('alt', '')
            })
            
        # Sort elements by their position in the document
        # We can't directly sort by position in the DOM,
        # so we'll use selector as a proxy for document order
        for i, elem in enumerate(content_elements):
            elem["order"] = i
            # Clean up by removing the BeautifulSoup element
            elem["selector"] = generate_selector(elem["element"])
            del elem["element"]
            reading_flow.append(elem)
            
        return reading_flow
    except Exception as e:
        logger.warning(f"Error mapping reading flow: {str(e)}")
        return []

# Helper functions

def generate_selector(element: Tag) -> str:
    """Generate a CSS selector for an element."""
    try:
        # Try to use HTML service if available
        from strategies.core.strategy_context import StrategyContext
        from core.service_registry import ServiceRegistry
        
        html_service = ServiceRegistry.get_instance().get_service("html_service")
        if html_service:
            return html_service.generate_selector(element)
    except ImportError:
        pass  # Fall back to a basic implementation
        
    # Simple selector generation
    selector_parts = []
    current = element
    
    while current and current.name and hasattr(current, 'parent'):
        # Use ID if available
        if current.get('id'):
            selector_parts.insert(0, f"#{current['id']}")
            break
            
        # Otherwise use tag and position
        siblings = [sibling for sibling in current.parent.find_all(current.name, recursive=False)] if current.parent else []
        
        if len(siblings) > 1:
            index = siblings.index(current) + 1
            selector_parts.insert(0, f"{current.name}:nth-of-type({index})")
        else:
            selector_parts.insert(0, current.name)
            
        current = current.parent
        
    return " > ".join(selector_parts)

def get_element_position(element: Tag) -> Dict[str, Any]:
    """Get the position information for an element in the document."""
    position = {
        "selector": generate_selector(element)
    }
    
    # Count preceding elements as a proxy for vertical position
    position["preceding_elements"] = len(list(element.find_all_previous()))
    
    return position

def calculate_child_similarity(children: List[Tag]) -> float:
    """Calculate the similarity between a group of child elements."""
    if not children or len(children) < 2:
        return 0.0
        
    # If all tags are the same, there's high baseline similarity
    tags = [child.name for child in children]
    if len(set(tags)) == 1:
        tag_similarity = 1.0
    else:
        # Calculate tag name diversity
        most_common_tag = max(set(tags), key=tags.count)
        tag_similarity = tags.count(most_common_tag) / len(tags)
        
    # Calculate class similarity
    class_similarities = []
    for i in range(len(children) - 1):
        for j in range(i + 1, len(children)):
            class_similarities.append(calculate_content_similarity(children[i], children[j]))
            
    avg_class_similarity = sum(class_similarities) / len(class_similarities) if class_similarities else 0.0
    
    # Weight the similarities
    return (tag_similarity * 0.4) + (avg_class_similarity * 0.6)

def has_similar_items(items: List[Tag]) -> bool:
    """Check if a list of items has similar structure."""
    similarity = calculate_child_similarity(items)
    return similarity > 0.7  # 70% similarity threshold