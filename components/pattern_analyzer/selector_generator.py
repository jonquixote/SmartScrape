"""
Selector Generator Module

This module provides utilities for generating, testing, and optimizing CSS selectors
for web scraping. It includes mechanisms to validate selector reliability and dynamically
adjust selectors based on extraction success.
"""

import re
import logging
import random
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Callable
from urllib.parse import urlparse
from bs4 import BeautifulSoup, Tag

from components.pattern_analyzer.base_analyzer import get_registry

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SelectorGenerator")


class SelectorGenerator:
    """
    Utility class for generating, testing, and optimizing CSS selectors.
    Helps create reliable selectors that work across different pages.
    """
    
    def __init__(self, stability_threshold: float = 0.7):
        """
        Initialize the selector generator.
        
        Args:
            stability_threshold: Threshold for selector stability (0.0 to 1.0)
        """
        self.stability_threshold = stability_threshold
        
        # Strategies for generating selectors, in order of priority
        self.selector_strategies = [
            self._generate_id_selector,
            self._generate_class_selector,
            self._generate_attribute_selector,
            self._generate_parent_child_selector,
            self._generate_positional_selector
        ]
        
        # Common classes to avoid in selectors (too generic)
        self.common_classes = {
            'container', 'wrapper', 'content', 'main', 'item', 'row', 'col',
            'box', 'card', 'panel', 'section', 'block', 'module', 'widget',
            'inner', 'outer', 'body', 'active', 'hidden', 'visible'
        }
    
    def generate_selector(self, element: Tag, html_context: str = None) -> str:
        """
        Generate the best selector for a given element.
        
        Args:
            element: BeautifulSoup Tag object
            html_context: HTML context for testing (optional)
            
        Returns:
            CSS selector string
        """
        selectors = []
        
        # Try each strategy in order
        for strategy in self.selector_strategies:
            selector = strategy(element)
            if selector:
                selectors.append(selector)
        
        # If we have a HTML context, test selectors and pick the best one
        if html_context:
            for selector in selectors:
                if self.test_selector(selector, element, html_context):
                    return selector
        
        # Otherwise return the first (most specific) selector
        return selectors[0] if selectors else f"{element.name}"
    
    def test_selector(self, selector: str, target_element: Tag, html: str) -> bool:
        """
        Test if a selector uniquely identifies the target element.
        
        Args:
            selector: CSS selector to test
            target_element: Target element that should be matched
            html: HTML content for testing
            
        Returns:
            True if the selector uniquely identifies the element, False otherwise
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            matches = soup.select(selector)
            
            # Successful if we get exactly one match
            if len(matches) == 1:
                # Verify it's the right element by comparing content
                if target_element.get_text().strip() == matches[0].get_text().strip():
                    return True
            
            return False
        except Exception as e:
            logger.error(f"Error testing selector {selector}: {str(e)}")
            return False
    
    def validate_selector_stability(self, selector: str, sample_htmls: List[str]) -> float:
        """
        Validate how stable a selector is across multiple HTML samples.
        
        Args:
            selector: CSS selector to validate
            sample_htmls: List of HTML samples to test against
            
        Returns:
            Stability score (0.0 to 1.0)
        """
        if not sample_htmls:
            return 0.0
            
        match_count = 0
        
        for html in sample_htmls:
            try:
                soup = BeautifulSoup(html, 'html.parser')
                matches = soup.select(selector)
                
                # Count as a match if we get exactly one element
                if len(matches) == 1:
                    match_count += 1
            except Exception as e:
                logger.error(f"Error validating selector {selector}: {str(e)}")
        
        # Calculate stability as percentage of samples where selector worked
        return match_count / len(sample_htmls)
    
    def optimize_selector(self, initial_selector: str, sample_htmls: List[str], 
                        expected_content_pattern: Optional[str] = None) -> str:
        """
        Optimize a selector for better stability across samples.
        
        Args:
            initial_selector: Initial CSS selector
            sample_htmls: List of HTML samples to test against
            expected_content_pattern: Regex pattern that the content should match (optional)
            
        Returns:
            Optimized CSS selector
        """
        if not sample_htmls:
            return initial_selector
            
        # Test the initial selector
        initial_stability = self.validate_selector_stability(initial_selector, sample_htmls)
        
        if initial_stability >= self.stability_threshold:
            return initial_selector
        
        # Try simplified versions of the selector
        simplified_selectors = self._generate_simplified_selectors(initial_selector)
        
        best_selector = initial_selector
        best_stability = initial_stability
        
        for selector in simplified_selectors:
            stability = self.validate_selector_stability(selector, sample_htmls)
            
            # If we have an expected content pattern, validate content too
            if expected_content_pattern and stability > 0:
                content_match_score = self._validate_content_match(
                    selector, sample_htmls, expected_content_pattern
                )
                # Combine stability with content matching
                stability = (stability + content_match_score) / 2
            
            if stability > best_stability:
                best_selector = selector
                best_stability = stability
                
                # If we reach high stability, we're done
                if stability >= self.stability_threshold:
                    break
        
        return best_selector
    
    def adjust_selector_dynamically(self, base_selector: str, html: str, 
                                  validation_func: Callable[[Tag], bool]) -> str:
        """
        Dynamically adjust a selector based on extraction results.
        
        Args:
            base_selector: Base CSS selector to adjust
            html: Current HTML content
            validation_func: Function that takes an element and returns True if it's valid
            
        Returns:
            Adjusted CSS selector
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            matches = soup.select(base_selector)
            
            # If we got no matches or too many, try adjusting the selector
            if not matches or len(matches) > 3:
                # Generate alternative selectors
                alternatives = self._generate_alternative_selectors(base_selector, soup)
                
                for alt_selector in alternatives:
                    alt_matches = soup.select(alt_selector)
                    
                    # Check if we get a reasonable number of matches
                    if alt_matches and len(alt_matches) <= 3:
                        # Validate matches with the validation function
                        valid_matches = [e for e in alt_matches if validation_func(e)]
                        
                        if valid_matches:
                            return alt_selector
            
            # If base selector finds valid elements, keep it
            elif any(validation_func(e) for e in matches):
                return base_selector
                
        except Exception as e:
            logger.error(f"Error adjusting selector {base_selector}: {str(e)}")
        
        # Fall back to the original selector if adjustments failed
        return base_selector
    
    def generate_selectors_from_pattern(self, pattern_type: str, url: str) -> Dict[str, str]:
        """
        Generate selectors from a registered pattern.
        
        Args:
            pattern_type: Type of pattern to use
            url: URL of the page
            
        Returns:
            Dictionary of element type to selector
        """
        # Get the domain from the URL
        domain = urlparse(url).netloc
        
        # Check the registry for patterns of this type for this domain
        registry = get_registry()
        pattern = registry.get_pattern(pattern_type, url)
        
        if not pattern:
            # Try getting domain-level patterns
            domain_patterns = registry.get_patterns_for_domain(domain)
            if pattern_type in domain_patterns:
                pattern = domain_patterns[pattern_type]
        
        if pattern:
            # Extract selectors from the pattern
            if 'selector' in pattern:
                return {pattern_type: pattern['selector']}
            elif 'selectors' in pattern:
                return pattern['selectors']
        
        # Return empty dict if no pattern found
        return {}
    
    def _generate_id_selector(self, element: Tag) -> Optional[str]:
        """
        Generate a selector using the element's ID.
        
        Args:
            element: BeautifulSoup Tag object
            
        Returns:
            CSS selector string or None
        """
        if element.get('id'):
            return f"#{element['id']}"
        return None
    
    def _generate_class_selector(self, element: Tag) -> Optional[str]:
        """
        Generate a selector using the element's classes.
        
        Args:
            element: BeautifulSoup Tag object
            
        Returns:
            CSS selector string or None
        """
        if element.get('class'):
            classes = element['class']
            
            # Filter out common non-specific classes
            specific_classes = [c for c in classes if c.lower() not in self.common_classes]
            
            if specific_classes:
                # Use the most specific (usually longest) class
                sorted_classes = sorted(specific_classes, key=len, reverse=True)
                return f"{element.name}.{sorted_classes[0]}"
            elif classes:
                # If we only have common classes, use the first one
                return f"{element.name}.{classes[0]}"
        
        return None
    
    def _generate_attribute_selector(self, element: Tag) -> Optional[str]:
        """
        Generate a selector using distinctive attributes.
        
        Args:
            element: BeautifulSoup Tag object
            
        Returns:
            CSS selector string or None
        """
        # Priority attributes that often distinguish elements
        priority_attrs = ['role', 'data-id', 'type', 'name', 'itemprop']
        
        for attr in priority_attrs:
            if element.get(attr):
                return f"{element.name}[{attr}='{element[attr]}']"
        
        return None
    
    def _generate_parent_child_selector(self, element: Tag) -> Optional[str]:
        """
        Generate a selector using parent-child relationship.
        
        Args:
            element: BeautifulSoup Tag object
            
        Returns:
            CSS selector string or None
        """
        parent = element.parent
        
        if parent and parent.name != 'body':
            # Try to use parent ID
            if parent.get('id'):
                return f"#{parent['id']} > {element.name}"
            
            # Try to use specific parent class
            if parent.get('class'):
                classes = parent['class']
                specific_classes = [c for c in classes if c.lower() not in self.common_classes]
                
                if specific_classes:
                    return f"{parent.name}.{specific_classes[0]} > {element.name}"
                elif classes:
                    return f"{parent.name}.{classes[0]} > {element.name}"
        
        return None
    
    def _generate_positional_selector(self, element: Tag) -> str:
        """
        Generate a selector using element position.
        
        Args:
            element: BeautifulSoup Tag object
            
        Returns:
            CSS selector string
        """
        siblings = list(element.find_previous_siblings(element.name))
        return f"{element.name}:nth-of-type({len(siblings) + 1})"
    
    def _generate_simplified_selectors(self, selector: str) -> List[str]:
        """
        Generate simplified versions of a selector.
        
        Args:
            selector: Original CSS selector
            
        Returns:
            List of simplified selectors
        """
        simplified = []
        
        # Remove nth-of-type
        nth_pattern = re.compile(r':nth-of-type\(\d+\)')
        nth_removed = nth_pattern.sub('', selector)
        if nth_removed != selector:
            simplified.append(nth_removed)
        
        # Extract just the class or ID part
        id_match = re.search(r'#([\w-]+)', selector)
        if id_match:
            simplified.append(f"#{id_match.group(1)}")
        
        class_match = re.search(r'\.([\w-]+)', selector)
        if class_match:
            class_name = class_match.group(1)
            if class_name.lower() not in self.common_classes:
                simplified.append(f".{class_name}")
        
        # Try attribute only
        attr_match = re.search(r'\[(\w+)=[\'"]([^\'"]+)[\'"]\]', selector)
        if attr_match:
            attr_name, attr_value = attr_match.groups()
            simplified.append(f"[{attr_name}='{attr_value}']")
        
        return simplified
    
    def _generate_alternative_selectors(self, selector: str, soup: BeautifulSoup) -> List[str]:
        """
        Generate alternative selectors when the original fails.
        
        Args:
            selector: Original CSS selector
            soup: BeautifulSoup object of the current page
            
        Returns:
            List of alternative selectors
        """
        alternatives = []
        
        # Try with just the tag name if the selector has more
        tag_match = re.match(r'^(\w+)', selector)
        if tag_match:
            tag_name = tag_match.group(1)
            if tag_name not in ['div', 'span', 'p', 'a']:  # Skip too generic tags
                alternatives.append(tag_name)
        
        # Try with just the class or just the ID
        id_match = re.search(r'#([\w-]+)', selector)
        if id_match:
            alternatives.append(f"#{id_match.group(1)}")
        
        class_match = re.search(r'\.([\w-]+)', selector)
        if class_match:
            class_name = class_match.group(1)
            if class_name.lower() not in self.common_classes:
                alternatives.append(f".{class_name}")
        
        # Try attribute with partial matching
        attr_match = re.search(r'\[(\w+)=[\'"]([^\'"]+)[\'"]\]', selector)
        if attr_match:
            attr_name, attr_value = attr_match.groups()
            alternatives.append(f"[{attr_name}*='{attr_value}']")
        
        # Try parent selector with different child
        parent_child_match = re.match(r'((?:\w+(?:\.\w+|\#\w+)?)) > (\w+)', selector)
        if parent_child_match:
            parent, child = parent_child_match.groups()
            # Try with just parent
            alternatives.append(parent)
            # Try parent with any descendant (not just direct child)
            alternatives.append(f"{parent} {child}")
        
        # If selector is very specific, try a more general version
        if len(selector.split()) >= 3:
            parts = selector.split()
            alternatives.append(' '.join(parts[0:2]))  # First two parts
            alternatives.append(parts[-1])  # Just the last part
        
        return alternatives
    
    def _validate_content_match(self, selector: str, sample_htmls: List[str], 
                              pattern: str) -> float:
        """
        Validate if elements matched by a selector contain expected content.
        
        Args:
            selector: CSS selector to test
            sample_htmls: List of HTML samples to test against
            pattern: Regex pattern that content should match
            
        Returns:
            Content match score (0.0 to 1.0)
        """
        if not sample_htmls:
            return 0.0
            
        try:
            pattern_re = re.compile(pattern, re.I)
        except:
            logger.error(f"Invalid regex pattern: {pattern}")
            return 0.0
        
        match_count = 0
        total_samples = 0
        
        for html in sample_htmls:
            try:
                soup = BeautifulSoup(html, 'html.parser')
                matches = soup.select(selector)
                
                if matches:
                    total_samples += 1
                    
                    # Check if any match contains content matching the pattern
                    if any(pattern_re.search(m.get_text()) for m in matches):
                        match_count += 1
            except Exception as e:
                logger.error(f"Error validating content for selector {selector}: {str(e)}")
        
        # Return ratio of samples where content matched
        return match_count / total_samples if total_samples > 0 else 0.0