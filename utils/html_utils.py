"""
Optimized HTML processing utilities for SmartScrape.

This module provides optimized HTML parsing and processing tools:
- Fast HTML parsing with lxml
- XPath support for efficient element selection
- Performance-optimized BeautifulSoup configurations
- Parallel processing for large HTML documents
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from functools import lru_cache
import re
from urllib.parse import urlparse

# Import lxml and BeautifulSoup with lxml parser
from lxml import etree, html

try:
    # Import cssselect for CSS selector support
    from lxml.cssselect import CSSSelector
    CSSSELECT_AVAILABLE = True
except ImportError:
    CSSSELECT_AVAILABLE = False
    logging.warning("lxml.cssselect not installed, CSS selector functionality will be limited")

from bs4 import BeautifulSoup

# Configure logging
logger = logging.getLogger(__name__)

# Lazy import for domain intelligence to avoid circular dependencies
_domain_intelligence = None

def get_domain_intelligence():
    """
    Lazy initialization of domain intelligence component.
    """
    global _domain_intelligence
    if _domain_intelligence is None:
        try:
            from components.domain_intelligence import DomainIntelligence
            _domain_intelligence = DomainIntelligence()
        except ImportError:
            logger.warning("DomainIntelligence not available, using basic functionality")
            _domain_intelligence = None
    return _domain_intelligence

# Lazy import for site type detection to avoid circular dependencies
_site_type_detector = None

def get_site_type_detector():
    """
    Lazy initialization of site type detector.
    """
    global _site_type_detector
    if _site_type_detector is None:
        try:
            from strategies.ai_guided.site_type import detect_site_type
            _site_type_detector = detect_site_type
        except ImportError:
            logger.warning("site_type.detect_site_type not available, using basic functionality")
            _site_type_detector = None
    return _site_type_detector

def parse_html(html_content: str, parser: str = 'lxml') -> BeautifulSoup:
    """
    Parse HTML content with BeautifulSoup using the lxml parser for better performance.
    
    Args:
        html_content: HTML content to parse
        parser: Parser to use ('lxml' recommended for performance)
        
    Returns:
        BeautifulSoup object
    """
    # Verify lxml is being used - fallback to html.parser if needed
    try:
        return BeautifulSoup(html_content, parser)
    except Exception as e:
        logger.warning(f"Failed to parse with {parser}: {str(e)}. Falling back to html.parser")
        return BeautifulSoup(html_content, 'html.parser')

def parse_html_with_lxml(html_content: str) -> etree.ElementTree:
    """
    Parse HTML using lxml directly for maximum performance.
    
    Args:
        html_content: HTML content to parse
        
    Returns:
        lxml.etree.ElementTree object
    """
    return html.fromstring(html_content)

@lru_cache(maxsize=100)
def compile_xpath(xpath_expression: str) -> etree.XPath:
    """
    Compile an XPath expression for repeated use.
    
    Args:
        xpath_expression: XPath expression to compile
        
    Returns:
        Compiled XPath object
    """
    return etree.XPath(xpath_expression)

def find_by_xpath(document: Union[etree.ElementTree, BeautifulSoup, str], xpath: str) -> List[Any]:
    """
    Find elements using XPath for faster selection.
    
    Args:
        document: HTML document (lxml ElementTree, BeautifulSoup object, or HTML string)
        xpath: XPath expression
        
    Returns:
        List of matching elements
    """
    # Convert to lxml Element if needed
    if isinstance(document, str):
        doc = html.fromstring(document)
    elif isinstance(document, BeautifulSoup):
        # Convert BeautifulSoup to string then parse with lxml
        doc = html.fromstring(str(document))
    else:
        doc = document
    
    # Use compiled XPath for better performance
    compiled_xpath = compile_xpath(xpath)
    return compiled_xpath(doc)

def select_with_css(document: Union[etree.ElementTree, BeautifulSoup, str], css_selector: str, url: Optional[str] = None) -> List[Any]:
    """
    Select elements using CSS selectors with domain intelligence enhancements.
    
    Args:
        document: HTML document (lxml ElementTree, BeautifulSoup object, or HTML string)
        css_selector: CSS selector
        url: Optional URL for domain-specific selector enhancements
        
    Returns:
        List of matching elements
    """
    # Convert to lxml Element if needed
    if isinstance(document, str):
        doc = html.fromstring(document)
    elif isinstance(document, BeautifulSoup):
        # Convert BeautifulSoup to string then parse with lxml
        doc = html.fromstring(str(document))
    else:
        doc = document
    
    # If cssselect is available, use it directly
    if CSSSELECT_AVAILABLE:
        try:
            # Create CSS selector and use it to find elements
            sel = CSSSelector(css_selector)
            return sel(doc)
        except Exception as e:
            logger.warning(f"Error using CSSSelector: {str(e)}. Falling back to XPath.")
            # Fall back to XPath conversion on error
    
    # If we have a URL, try to use domain intelligence for enhanced selectors
    if url and css_selector:
        # Get domain intelligence for enhanced selection (if available)
        domain_intelligence = get_domain_intelligence()
        site_type_detector = get_site_type_detector()
        
        if domain_intelligence and site_type_detector:
            try:
                # Detect site type
                site_type = site_type_detector(url)
                
                # Get specialized extraction config for the site type
                config = domain_intelligence.get_specialized_extraction_config(site_type)
                
                # Check if config has CSS selectors for the type of content we're looking for
                if "css_selectors" in config:
                    # Look for a matching selector category based on our input selector
                    for category, selectors in config["css_selectors"].items():
                        # If our selector looks like it's targeting this category
                        if any(term in css_selector.lower() for term in category.split('_')):
                            # Try each specialized selector for this category
                            for specialized_selector in selectors:
                                try:
                                    # Try with cssselect if available
                                    if CSSSELECT_AVAILABLE:
                                        sel = CSSSelector(specialized_selector)
                                        result = sel(doc)
                                        if result:
                                            logger.info(f"Using specialized selector {specialized_selector} for {category}")
                                            return result
                                except Exception:
                                    continue  # Try next selector
            except Exception as e:
                logger.debug(f"Error using domain intelligence: {str(e)}")
    
    # Fallback method: convert CSS to XPath (basic support for simple selectors)
    try:
        # For very basic CSS selectors, attempt a simple conversion to XPath
        xpath = _css_to_xpath(css_selector)
        return doc.xpath(xpath)
    except Exception as e:
        logger.warning(f"Error converting CSS to XPath: {str(e)}")
        
        # Last resort: use a very basic find by tag
        if re.match(r'^[a-zA-Z0-9]+$', css_selector):  # Simple tag name
            return doc.xpath(f'//{css_selector}')
        
        # Empty result if all methods fail
        return []

def _css_to_xpath(css: str) -> str:
    """
    Very basic CSS to XPath converter for common simple selectors.
    
    Args:
        css: CSS selector
        
    Returns:
        XPath equivalent
    """
    # Handle ID selectors (#id)
    if css.startswith('#'):
        return f"//*[@id='{css[1:]}']"
        
    # Handle class selectors (.class)
    if css.startswith('.'):
        return f"//*[contains(@class, '{css[1:]}')]"
        
    # Handle tag selectors with attributes (tag[attr='value'])
    tag_attr_match = re.match(r'^([a-zA-Z0-9]+)\[([^=]+)=["|\']([^"\']+)["|\']', css)
    if tag_attr_match:
        tag, attr, value = tag_attr_match.groups()
        return f"//{tag}[@{attr}='{value}']"
        
    # Handle tag selectors with class (tag.class)
    tag_class_match = re.match(r'^([a-zA-Z0-9]+)\.([a-zA-Z0-9_-]+)', css)
    if tag_class_match:
        tag, cls = tag_class_match.groups()
        return f"//{tag}[contains(@class, '{cls}')]"
        
    # Handle simple tag selectors
    if re.match(r'^[a-zA-Z0-9]+$', css):
        return f'//{css}'
        
    # Default for more complex selectors that we can't handle
    # Just wrap with // to make a very basic attempt
    return f'//{css}'

def extract_text_fast(element: Union[etree.Element, BeautifulSoup]) -> str:
    """
    Extract text from an element faster than BeautifulSoup's get_text().
    
    Args:
        element: HTML element (lxml Element or BeautifulSoup Tag)
        
    Returns:
        Extracted text
    """
    if isinstance(element, BeautifulSoup) or hasattr(element, 'get_text'):
        # BeautifulSoup element - convert to string and parse with lxml for speed
        html_str = str(element)
        lxml_elem = html.fromstring(html_str)
        return " ".join(lxml_elem.xpath('.//text()'))
    else:
        # lxml Element
        return " ".join(element.xpath('.//text()'))

def extract_links_fast(document: Union[etree.ElementTree, BeautifulSoup, str], base_url: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Extract links from a document using lxml for speed.
    
    Args:
        document: HTML document (lxml ElementTree, BeautifulSoup object, or HTML string)
        base_url: Optional base URL for resolving relative links
        
    Returns:
        List of dictionaries with href, text, and title
    """
    # Convert to lxml Element if needed
    if isinstance(document, str):
        doc = html.fromstring(document)
    elif isinstance(document, BeautifulSoup):
        doc = html.fromstring(str(document))
    else:
        doc = document
    
    links = []
    for link in doc.xpath('//a'):
        href = link.get('href')
        
        # Skip empty links
        if not href:
            continue
            
        # Resolve relative URLs if base_url is provided
        if base_url and href.startswith('/'):
            if base_url.endswith('/'):
                href = base_url + href[1:]
            else:
                href = base_url + href
                
        links.append({
            'href': href,
            'text': " ".join(link.xpath('.//text()')),
            'title': link.get('title', '')
        })
    
    return links

def extract_tables_fast(document: Union[etree.ElementTree, BeautifulSoup, str]) -> List[List[List[str]]]:
    """
    Extract tables from a document using lxml for speed.
    
    Args:
        document: HTML document (lxml ElementTree, BeautifulSoup object, or HTML string)
        
    Returns:
        List of tables, where each table is a list of rows, and each row is a list of cell text
    """
    # Convert to lxml Element if needed
    if isinstance(document, str):
        doc = html.fromstring(document)
    elif isinstance(document, BeautifulSoup):
        doc = html.fromstring(str(document))
    else:
        doc = document
    
    tables = []
    for table in doc.xpath('//table'):
        rows = []
        for tr in table.xpath('.//tr'):
            cells = []
            for td in tr.xpath('.//td | .//th'):
                cells.append(" ".join(td.xpath('.//text()')).strip())
            if cells:
                rows.append(cells)
        if rows:
            tables.append(rows)
    
    return tables

def clean_html_fast(html_content: str) -> str:
    """
    Clean HTML content using lxml for speed.
    
    Args:
        html_content: HTML content to clean
        
    Returns:
        Cleaned HTML content
    """
    # Parse HTML
    doc = html.fromstring(html_content)
    
    # Remove script and style elements
    for element in doc.xpath('//script | //style | //noscript'):
        element.getparent().remove(element)
    
    # Return cleaned HTML
    return html.tostring(doc, pretty_print=True, encoding='unicode')

def extract_meta_tags(document: Union[etree.ElementTree, BeautifulSoup, str]) -> Dict[str, str]:
    """
    Extract meta tags from a document using lxml for speed.
    
    Args:
        document: HTML document (lxml ElementTree, BeautifulSoup object, or HTML string)
        
    Returns:
        Dictionary of meta tag name/property values
    """
    # Convert to lxml Element if needed
    if isinstance(document, str):
        doc = html.fromstring(document)
    elif isinstance(document, BeautifulSoup):
        doc = html.fromstring(str(document))
    else:
        doc = document
    
    meta_tags = {}
    
    # Extract standard meta tags with name attribute
    for meta in doc.xpath('//meta[@name]'):
        name = meta.get('name', '').lower()
        content = meta.get('content', '')
        if name and content:
            meta_tags[name] = content
    
    # Extract Open Graph and other meta tags with property attribute
    for meta in doc.xpath('//meta[@property]'):
        prop = meta.get('property', '').lower()
        content = meta.get('content', '')
        if prop and content:
            meta_tags[prop] = content
    
    return meta_tags

class SimpleHTMLService:
    """
    Simple HTML service that provides essential HTML processing functions.
    
    This service provides common HTML processing methods used by strategies
    and extractors, leveraging the optimized functions in this module.
    """
    
    def clean_html(self, html_content: str) -> str:
        """
        Clean HTML by removing scripts, styles, and unwanted elements.
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            Cleaned HTML content
        """
        return clean_html_fast(html_content)
    
    def extract_main_content(self, html_content: str) -> str:
        """
        Extract the main content from an HTML document.
        
        This method attempts to identify and extract the main content area
        of a webpage, filtering out navigation, headers, footers, etc.
        
        Args:
            html_content: HTML content
            
        Returns:
            HTML string containing just the main content
        """
        # Parse the document
        if isinstance(html_content, str):
            doc = html.fromstring(html_content)
        else:
            doc = html_content
            
        # Content score for each element
        content_scores = {}
        
        # Elements likely to contain main content
        main_content_candidates = doc.xpath('//div | //article | //section | //main')
        
        max_score = 0
        best_candidate = None
        
        # Score each candidate based on content density and positioning
        for element in main_content_candidates:
            # Skip hidden elements
            display = element.get('style', '').lower()
            if 'display: none' in display or 'visibility: hidden' in display:
                continue
                
            # Score based on tag name
            tag_name = element.tag.lower()
            score = {
                'main': 20,
                'article': 15,
                'section': 10,
                'div': 5
            }.get(tag_name, 0)
            
            # Score based on ID and class
            attributes = ' '.join([
                element.get('id', '').lower(),
                element.get('class', '').lower()
            ])
            
            # Positive indicators in ID/class
            for term in ['content', 'main', 'article', 'story', 'post', 'entry', 'text', 'body']:
                if term in attributes:
                    score += 25
                    
            # Negative indicators in ID/class
            for term in ['comment', 'sidebar', 'footer', 'nav', 'menu', 'header', 'banner']:
                if term in attributes:
                    score -= 25
            
            # Score based on text-to-tag ratio
            text_content = ' '.join(element.xpath('.//text()'))
            text_length = len(text_content)
            html_length = len(html.tostring(element, encoding='unicode'))
            
            if html_length > 0:
                ratio = text_length / html_length
                score += ratio * 50
            
            # Consider descendant paragraphs
            p_count = len(element.xpath('.//p'))
            if p_count > 0:
                score += min(p_count * 3, 50)  # Cap at 50 points
                
            content_scores[element] = score
            
            if score > max_score:
                max_score = score
                best_candidate = element
        
        # If we found a good candidate, extract its content
        if best_candidate is not None and max_score > 50:
            return html.tostring(best_candidate, encoding='unicode')
        
        # Fallback to selecting content blocks with substantial text
        content_blocks = []
        for p in doc.xpath('//p'):
            text = ' '.join(p.xpath('.//text()'))
            if len(text.strip()) > 150:  # Paragraph with substantial text
                content_blocks.append(html.tostring(p, encoding='unicode'))
        
        if content_blocks:
            return '\n'.join(content_blocks)
            
        # If all else fails, return the original content
        return html_content