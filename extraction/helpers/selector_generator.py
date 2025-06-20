"""
Selector Generator Module

This module provides utilities for generating optimal CSS and XPath selectors
for DOM elements to ensure reliable extraction across different web pages.
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Set
from bs4 import BeautifulSoup, Tag

class SelectorGenerator:
    """
    Utility class for generating and optimizing CSS and XPath selectors
    for web page elements to ensure reliable extraction.
    """
    
    def __init__(self):
        """Initialize the SelectorGenerator."""
        self.preferred_attributes = [
            'id', 'data-id', 'data-testid', 'data-automation', 'data-qa',
            'data-test', 'name', 'class', 'data-element', 'data-cy'
        ]
        
    def generate_css_selector(self, element: Tag, html: str) -> str:
        """
        Generate an optimal CSS selector for the given element.
        
        Args:
            element: The BeautifulSoup Tag to generate a selector for
            html: The full HTML document (for testing uniqueness)
            
        Returns:
            A CSS selector string
        """
        # Try ID-based selector first (most reliable)
        if element.get('id'):
            selector = f"#{element['id']}"
            if self.test_selector_uniqueness(selector, html):
                return selector
                
        # Try class-based selector with element type
        if element.get('class'):
            classes = ' '.join(element['class'])
            selector = f"{element.name}.{classes.replace(' ', '.')}"
            if self.test_selector_uniqueness(selector, html):
                return selector
                
        # Try data attributes
        for attr in self.preferred_attributes:
            if attr != 'id' and attr != 'class' and element.get(attr):
                selector = f"{element.name}[{attr}='{element[attr]}']"
                if self.test_selector_uniqueness(selector, html):
                    return selector
                    
        # Generate a position-based selector as a fallback
        return self._generate_position_path(element)
        
    def generate_xpath_selector(self, element: Tag) -> str:
        """
        Generate an optimal XPath selector for the given element.
        
        Args:
            element: The BeautifulSoup Tag to generate a selector for
            
        Returns:
            An XPath selector string
        """
        # Try ID-based XPath first
        if element.get('id'):
            return f"//*[@id='{element['id']}']"
            
        # Try attributes-based XPath
        for attr in self.preferred_attributes:
            if attr != 'id' and element.get(attr):
                # For class, need to handle multiple classes
                if attr == 'class':
                    classes = element['class']
                    if classes:
                        return f"//{element.name}[contains(@class, '{classes[0]}')]"
                else:
                    return f"//{element.name}[@{attr}='{element[attr]}']"
                    
        # Generate a position path as fallback
        return self._generate_xpath_position_path(element)
        
    def generate_robust_selector(self, element: Tag, html: str) -> Dict[str, str]:
        """
        Generate multiple selectors with fallbacks for maximum reliability.
        
        Args:
            element: The BeautifulSoup Tag to generate selectors for
            html: The full HTML document (for testing uniqueness)
            
        Returns:
            Dictionary with primary and fallback selectors
        """
        selectors = {
            'css': self.generate_css_selector(element, html),
            'xpath': self.generate_xpath_selector(element),
            'fallbacks': []
        }
        
        # Add attribute-based fallbacks
        for attr in self.preferred_attributes:
            if attr != 'id' and attr != 'class' and element.get(attr):
                selectors['fallbacks'].append({
                    'type': 'css', 
                    'selector': f"{element.name}[{attr}='{element[attr]}']"
                })
                
        # Add structural fallback
        selectors['fallbacks'].append({
            'type': 'xpath',
            'selector': self._generate_xpath_position_path(element)
        })
        
        return selectors
        
    def test_selector_uniqueness(self, selector: str, html: str) -> bool:
        """
        Test if a CSS selector uniquely identifies a single element.
        
        Args:
            selector: CSS selector to test
            html: HTML content to test against
            
        Returns:
            True if selector matches exactly one element, False otherwise
        """
        soup = BeautifulSoup(html, 'html.parser')
        try:
            elements = soup.select(selector)
            return len(elements) == 1
        except:
            return False
            
    def optimize_selector(self, selector: str, html: str) -> str:
        """
        Optimize a selector by making it as short as possible while maintaining uniqueness.
        
        Args:
            selector: CSS selector to optimize
            html: HTML content to test against
            
        Returns:
            Optimized CSS selector
        """
        if not selector or '>' not in selector:
            return selector
            
        # Try to shorten by removing ancestors
        parts = selector.split(' > ')
        
        # Try with just the last part (most specific)
        last_part = parts[-1]
        if self.test_selector_uniqueness(last_part, html):
            return last_part
            
        # Try with progressively more ancestors
        for i in range(len(parts) - 2, -1, -1):
            shorter = ' > '.join(parts[i:])
            if self.test_selector_uniqueness(shorter, html):
                return shorter
                
        return selector
        
    def find_parent_container(self, element: Tag, max_levels: int = 5) -> Tag:
        """
        Find the logical parent container for an element.
        
        Args:
            element: The element to find a container for
            max_levels: Maximum levels to traverse up
            
        Returns:
            Parent container Tag
        """
        current = element
        levels = 0
        
        # Container tags that typically group related content
        container_tags = {'div', 'section', 'article', 'main', 'ul', 'ol', 'table'}
        
        while current.parent and levels < max_levels:
            parent = current.parent
            
            # Check if parent has container-like tag
            if parent.name in container_tags:
                # Check if parent has multiple similar children
                siblings = [s for s in parent.children if isinstance(s, Tag) and s.name == current.name]
                if len(siblings) > 1:
                    return parent
                    
                # Check for container attributes
                if parent.get('class') and any(c in str(parent['class']).lower() for c in 
                                            ['container', 'list', 'results', 'items', 'content', 'grid']):
                    return parent
                    
            current = parent
            levels += 1
            
        return current
        
    def create_field_selectors(self, container: Tag, samples: List[Tag]) -> Dict[str, Dict[str, Any]]:
        """
        Generate field-specific selectors based on sample items.
        
        Args:
            container: The container element
            samples: List of sample item elements
            
        Returns:
            Dictionary mapping field names to selector information
        """
        if not samples or len(samples) < 1:
            return {}
            
        # Analyze the first sample to identify potential fields
        sample = samples[0]
        
        # Common field mapping patterns
        field_mapping = {
            'title': ['h1', 'h2', 'h3', 'h4', '.title', '.name', '[class*="title"]', '[class*="name"]'],
            'price': ['.price', '[class*="price"]', '[itemprop="price"]', 'span[class*="price"]'],
            'description': ['.description', '[class*="description"]', '[class*="desc"]', 'p'],
            'image': ['img', '[class*="image"]', '[class*="img"]', '[class*="photo"]'],
            'link': ['a', 'a[href]'],
            'date': ['[class*="date"]', '[class*="time"]', 'time'],
            'author': ['[class*="author"]', '[class*="by"]', '[rel="author"]'],
            'rating': ['[class*="rating"]', '[class*="stars"]', '[class*="score"]']
        }
        
        field_selectors = {}
        html_str = str(container)
        
        # Generate selectors for each field type
        for field_name, potential_selectors in field_mapping.items():
            for selector in potential_selectors:
                try:
                    elements = sample.select(selector)
                    if elements:
                        # Verify this selector works for all samples
                        valid_for_all = all(len(s.select(selector)) > 0 for s in samples)
                        
                        if valid_for_all:
                            # Create relative selector
                            relative_selector = selector
                            field_selectors[field_name] = {
                                'selector': relative_selector,
                                'type': 'css',
                                'attribute': 'text' if field_name not in ['image', 'link'] else 'src' if field_name == 'image' else 'href'
                            }
                            break
                except:
                    continue
                    
        return field_selectors
        
    def _generate_position_path(self, element: Tag) -> str:
        """
        Generate a CSS selector using the element's position in the document.
        
        Args:
            element: The BeautifulSoup Tag to generate a selector for
            
        Returns:
            A CSS selector string using the element's position
        """
        parts = []
        current = element
        
        # Traverse up to 5 levels or until we hit body
        for _ in range(5):
            if not current or current.name == 'body' or current.name == 'html':
                break
                
            # Get the element's position among siblings of the same type
            siblings = [s for s in current.parent.children if isinstance(s, Tag) and s.name == current.name]
            position = siblings.index(current) + 1
            
            # Add tag with position to selector
            if current.get('class'):
                class_str = '.'.join(current['class'])
                parts.append(f"{current.name}.{class_str}")
            else:
                parts.append(f"{current.name}:nth-of-type({position})")
                
            current = current.parent
            
        # Reverse the parts and join them
        return ' > '.join(reversed(parts))
        
    def _generate_xpath_position_path(self, element: Tag) -> str:
        """
        Generate an XPath selector using the element's position in the document.
        
        Args:
            element: The BeautifulSoup Tag to generate a selector for
            
        Returns:
            An XPath selector string using the element's position
        """
        parts = []
        current = element
        
        # Traverse up to 5 levels or until we hit body
        for _ in range(5):
            if not current or current.name == 'body' or current.name == 'html':
                break
                
            # Get the element's position among siblings of the same type
            siblings = [s for s in current.parent.children if isinstance(s, Tag) and s.name == current.name]
            position = siblings.index(current) + 1
            
            # Add tag with position to selector
            parts.append(f"{current.name}[{position}]")
            current = current.parent
            
        # Reverse the parts and join them
        return '//' + '/'.join(reversed(parts))