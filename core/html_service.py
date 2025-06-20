import logging
import re
from typing import Dict, List, Optional, Any, Tuple, Set, Union
import hashlib
import threading

from bs4 import BeautifulSoup, Tag, NavigableString, Comment
import cssselect
import lxml.html
from lxml.cssselect import CSSSelector

from core.service_interface import BaseService

logger = logging.getLogger(__name__)

class HTMLService(BaseService):
    """Service for HTML operations including cleaning, selector generation, and content extraction."""
    
    def __init__(self):
        self._initialized = False
        self._config = None
        self._selector_cache = {}
        self._lock = threading.RLock()
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the HTML service."""
        if self._initialized:
            return
        
        self._config = config or {}
        self._initialized = True
        logger.info("HTML service initialized")
    
    def shutdown(self) -> None:
        """Shutdown the HTML service."""
        if not self._initialized:
            return
        
        self._selector_cache.clear()
        self._initialized = False
        logger.info("HTML service shut down")
    
    @property
    def name(self) -> str:
        """Return the name of the service."""
        return "html_service"
    
    def clean_html(self, html: str, remove_js: bool = True, remove_css: bool = True, 
                 remove_comments: bool = True, comprehensive_security: bool = True,
                 preserve_event_handlers: bool = False) -> str:
        """Clean HTML by removing unwanted elements and normalizing structure with comprehensive security."""
        if not html:
            return ""
        
        # Check if the content looks like plain text (no HTML tags)
        # More sophisticated check for actual HTML tags vs angle brackets
        # HTML tags must start with < followed by letter, not have spaces before the tag name
        html_tag_pattern = r'<[a-zA-Z]+[a-zA-Z0-9]*(?:\s+[^>]*)?\s*/?>'
        has_html_tags = re.search(html_tag_pattern, html)
        
        # Special case for XML - if it's XML but not HTML, preserve it
        if html.strip().startswith('<?xml'):
            try:
                soup = BeautifulSoup(html, 'xml')  # Use XML parser
                return str(soup)
            except:
                return html.strip()
        
        if not has_html_tags:
            # For plain text content, just return it as-is
            return html.strip()
        
        # Check if it's very simple content that doesn't need full HTML processing
        simple_tag_pattern = r'^[^<]*<[^>]+>[^<]*$|^[^<]*<[^>]+>[^<]*<\/[^>]+>[^<]*$'
        if len(html) < 200 and not any(tag in html.lower() for tag in ['<html', '<body', '<head', '<script', '<style']):
            # For simple content, use minimal processing
            try:
                soup = BeautifulSoup(html, 'html.parser')  # Use html.parser for simple content
                return str(soup)
            except:
                return html.strip()
        
        # Parse the HTML
        try:
            soup = BeautifulSoup(html, 'lxml')
            
            # Remove script tags
            if remove_js:
                for script in soup.find_all('script'):
                    script.extract()
            
            # Remove style tags
            if remove_css:
                for style in soup.find_all('style'):
                    style.extract()
            
            # Remove stylesheet links if comprehensive security is enabled
            if comprehensive_security and remove_css:
                for link in soup.find_all('link', rel='stylesheet'):
                    link.extract()
                for link in soup.find_all('link', rel=lambda x: x and 'stylesheet' in str(x).lower()):
                    link.extract()
            
            # Remove comments
            if remove_comments:
                for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                    comment.extract()
            
            # Remove hidden elements (always enabled)
            for hidden in soup.find_all(attrs={"hidden": True}):
                hidden.extract()
            
            # Remove elements with CSS hiding BEFORE removing style attributes
            if remove_css:
                self._remove_css_hidden_elements(soup)
            
            # Comprehensive security cleaning
            if comprehensive_security:
                self._remove_security_risks(soup, preserve_event_handlers)
            
            # Normalize whitespace
            for element in soup.find_all(string=True):
                if element.parent.name not in ['script', 'style', 'pre', 'code']:
                    element.replace_with(NavigableString(' '.join(element.strip().split())))
            
            # Clean up style attributes if needed (do this LAST)
            if remove_css:
                for tag in soup.find_all(attrs={"style": True}):
                    del tag['style']
            
            return str(soup)
            
        except Exception as e:
            logger.error(f"Error cleaning HTML: {str(e)}")
            return html
    
    def _remove_security_risks(self, soup: BeautifulSoup, preserve_event_handlers: bool = False) -> None:
        """Remove various security risks from HTML."""
        # Define dangerous event attributes that could contain JavaScript
        dangerous_events = [
            'onclick', 'ondblclick', 'onmousedown', 'onmouseup', 'onmouseover', 
            'onmouseout', 'onmousemove', 'onkeydown', 'onkeyup', 'onkeypress',
            'onfocus', 'onblur', 'onload', 'onunload', 'onsubmit', 'onreset',
            'onchange', 'onselect', 'onerror', 'onabort', 'onresize', 'onscroll',
            'oncontextmenu', 'ondrag', 'ondrop', 'onwheel', 'ontouchstart',
            'ontouchend', 'ontouchmove', 'ontouchcancel', 'onpointerdown',
            'onpointerup', 'onpointermove', 'onpointerover', 'onpointerout',
            'onanimationstart', 'onanimationend', 'onanimationiteration',
            'ontransitionend', 'ontransitionstart', 'oninput', 'oninvalid'
        ]
        
        # Remove event handlers unless preservation is requested
        if not preserve_event_handlers:
            for event_attr in dangerous_events:
                for element in soup.find_all(attrs={event_attr: True}):
                    # Remove only the dangerous attribute, not the entire element
                    del element[event_attr]
        
        # Remove javascript: URLs from href, src, and action attributes
        for attr in ['href', 'src', 'action', 'formaction']:
            for element in soup.find_all(attrs={attr: True}):
                attr_value = element[attr]
                if isinstance(attr_value, str) and attr_value.lower().strip().startswith('javascript:'):
                    element[attr] = '#'  # Replace with safe placeholder
        
        # Remove data: URLs that might contain scripts
        for attr in ['src', 'href']:
            for element in soup.find_all(attrs={attr: True}):
                attr_value = element[attr]
                if isinstance(attr_value, str) and attr_value.lower().strip().startswith('data:'):
                    # Check if data URL contains script content
                    if any(dangerous in attr_value.lower() for dangerous in ['script', 'javascript', 'vbscript']):
                        element[attr] = '#'  # Replace with safe placeholder
        
        # Remove elements that commonly host malicious content
        dangerous_tags = ['object', 'embed', 'applet']
        for tag_name in dangerous_tags:
            for element in soup.find_all(tag_name):
                element.extract()
        
        # Remove meta refresh that could redirect to malicious sites
        for meta in soup.find_all('meta', attrs={'http-equiv': 'refresh'}):
            meta.extract()
        
        # Clean style attributes for CSS-based attacks
        for element in soup.find_all(attrs={'style': True}):
            style_value = element['style']
            if isinstance(style_value, str):
                # Remove expressions and javascript URLs from styles
                dangerous_css_patterns = [
                    'expression(', 'javascript:', 'vbscript:', 'mocha:',
                    'livescript:', 'data:', 'url(javascript:', 'url(data:'
                ]
                
                for pattern in dangerous_css_patterns:
                    if pattern in style_value.lower():
                        # Remove the entire style attribute if it contains dangerous content
                        del element['style']
                        break
        
        # Remove injection patterns from attributes
        self._remove_injection_patterns(soup)
    
    def _remove_css_hidden_elements(self, soup: BeautifulSoup) -> None:
        """Remove elements that are hidden via CSS."""
        # Remove elements with display:none or visibility:hidden in style attribute
        for element in soup.find_all(attrs={"style": lambda value: value and self._is_css_hidden(value)}):
            element.extract()
    
    def _is_css_hidden(self, style_value: str) -> bool:
        """Check if a CSS style value indicates the element is hidden."""
        if not isinstance(style_value, str):
            return False
        
        style_lower = style_value.lower().replace(' ', '').replace('\t', '').replace('\n', '')
        
        # Check for display:none
        if 'display:none' in style_lower or 'display:hidden' in style_lower:
            return True
        
        # Check for visibility:hidden
        if 'visibility:hidden' in style_lower:
            return True
        
        # Check for zero dimensions
        if ('width:0' in style_lower and 'height:0' in style_lower) or \
           ('width:0px' in style_lower and 'height:0px' in style_lower):
            return True
        
        # Check for off-screen positioning
        if 'left:-9999' in style_lower or 'left:-999' in style_lower or \
           'text-indent:-9999' in style_lower or 'text-indent:-999' in style_lower:
            return True
        
        return False
    
    def generate_selector(self, element: Union[Tag, str], html: Optional[str] = None, 
                        method: str = 'css', optimized: bool = True) -> str:
        """Generate a CSS or XPath selector for an element."""
        if isinstance(element, str) and html:
            # If element is a string, parse it with the provided HTML context
            soup = BeautifulSoup(html, 'lxml')
            matches = soup.select(element)
            if not matches:
                return ""
            element = matches[0]
        
        if not isinstance(element, Tag):
            return ""
        
        # Generate a unique key for caching
        element_str = str(element)
        cache_key = f"{hashlib.md5(element_str.encode()).hexdigest()}_{method}_{optimized}"
        
        with self._lock:
            # Check if we have a cached selector
            if cache_key in self._selector_cache:
                return self._selector_cache[cache_key]
        
            if method == 'css':
                selector = self._generate_css_selector(element, optimized)
            elif method == 'xpath':
                selector = self._generate_xpath_selector(element, optimized)
            else:
                raise ValueError(f"Unsupported selector method: {method}")
            
            # Cache the selector
            self._selector_cache[cache_key] = selector
            return selector
    
    def _generate_css_selector(self, element: Tag, optimized: bool = True) -> str:
        """Generate a CSS selector for an element."""
        # Try id first for optimized selectors
        if optimized and element.get('id'):
            return f"#{element['id']}"
        
        # Try a combination of tag and class
        if optimized and element.get('class'):
            classes = '.'.join(element['class'])
            selector = f"{element.name}.{classes}"
            
            # Check if this uniquely identifies the element
            parent = element.parent
            if parent and len(parent.select(selector)) == 1:
                return selector
        
        # Generate a full path selector
        path = []
        current = element
        
        while current and current.name != '[document]':
            # Get the tag
            selector = current.name
            
            # Add id if present
            if current.get('id'):
                selector = f"{selector}#{current['id']}"
                path.insert(0, selector)
                break
            
            # Add classes if present
            if current.get('class'):
                classes = '.'.join(current['class'])
                selector = f"{selector}.{classes}"
            
            # Add position if needed
            if not optimized or not current.get('id') and not current.get('class'):
                siblings = [sibling for sibling in current.parent.children if sibling.name == current.name]
                if len(siblings) > 1:
                    position = siblings.index(current) + 1
                    selector = f"{selector}:nth-of-type({position})"
            
            path.insert(0, selector)
            current = current.parent
        
        return ' > '.join(path)
    
    def _generate_xpath_selector(self, element: Tag, optimized: bool = True) -> str:
        """Generate an XPath selector for an element."""
        # Try id first for optimized selectors
        if optimized and element.get('id'):
            return f"//*[@id='{element['id']}']"
        
        # Generate a full path selector
        path = []
        current = element
        
        while current and current.name != '[document]':
            # Get the tag
            selector = current.name
            
            # Add conditions
            conditions = []
            
            # Add id if present
            if current.get('id'):
                conditions.append(f"@id='{current['id']}'")
            
            # Add classes if present
            if current.get('class'):
                classes = ' '.join(current['class'])
                conditions.append(f"contains(@class, '{classes}')")
            
            # Add position if needed
            if not optimized or not conditions:
                siblings = [sibling for sibling in current.parent.children if sibling.name == current.name]
                if len(siblings) > 1:
                    position = siblings.index(current) + 1
                    conditions.append(f"position()={position}")
            
            # Combine conditions
            if conditions:
                selector = f"{selector}[{' and '.join(conditions)}]"
            
            path.insert(0, selector)
            current = current.parent
        
        return '//' + '/'.join(path)
    
    def compare_elements(self, element1: Tag, element2: Tag) -> float:
        """Compare two elements and return a similarity score (0-1)."""
        # Compare tag names
        if element1.name != element2.name:
            return 0.0
        
        # Compare attributes (excluding content-specific ones)
        attrs1 = {k: v for k, v in element1.attrs.items() if k not in ['id', 'style']}
        attrs2 = {k: v for k, v in element2.attrs.items() if k not in ['id', 'style']}
        
        # Calculate attribute similarity
        attr_similarity = 0.0
        if attrs1 or attrs2:
            common_attrs = set(attrs1.keys()) & set(attrs2.keys())
            total_attrs = set(attrs1.keys()) | set(attrs2.keys())
            
            # Check values of common attributes
            matching_values = sum(1 for attr in common_attrs if attrs1[attr] == attrs2[attr])
            
            if total_attrs:
                attr_similarity = matching_values / len(total_attrs)
        else:
            # If no attributes, they're similar in this respect
            attr_similarity = 1.0
        
        # Compare structure (number of children of each type)
        children1 = [child.name for child in element1.children if child.name]
        children2 = [child.name for child in element2.children if child.name]
        
        structure_similarity = 0.0
        if children1 or children2:
            # Count children by tag type
            count1 = {tag: children1.count(tag) for tag in set(children1)}
            count2 = {tag: children2.count(tag) for tag in set(children2)}
            
            # Find common tags
            common_tags = set(count1.keys()) & set(count2.keys())
            total_tags = set(count1.keys()) | set(count2.keys())
            
            # Calculate similarity based on tag counts
            if total_tags:
                tag_similarity = sum(min(count1.get(tag, 0), count2.get(tag, 0)) for tag in total_tags) / sum(max(count1.get(tag, 0), count2.get(tag, 0)) for tag in total_tags)
                structure_similarity = tag_similarity
            else:
                structure_similarity = 1.0  # Both have no children
        else:
            # If no children, they're similar in this respect
            structure_similarity = 1.0
        
        # Compare text length and content (basic)
        text1 = element1.get_text(strip=True)
        text2 = element2.get_text(strip=True)
        
        text_similarity = 0.0
        if text1 or text2:
            # Compare text lengths
            max_len = max(len(text1), len(text2))
            min_len = min(len(text1), len(text2))
            
            if max_len > 0:
                length_ratio = min_len / max_len
                text_similarity = length_ratio
            else:
                text_similarity = 1.0  # Both empty
        else:
            # If no text, they're similar in this respect
            text_similarity = 1.0
        
        # Weighted combination of similarities
        overall_similarity = 0.4 * attr_similarity + 0.4 * structure_similarity + 0.2 * text_similarity
        return overall_similarity
    
    def extract_main_content(self, html: str) -> str:
        """Extract the main content area from an HTML document."""
        if not html:
            return ""
        
        try:
            # Parse the HTML
            soup = BeautifulSoup(html, 'lxml')
            
            # Remove navigation, header, footer, sidebar, etc.
            for tag in soup.find_all(['nav', 'header', 'footer', 'aside']):
                tag.extract()
            
            for tag in soup.find_all(attrs={"role": ["navigation", "banner", "contentinfo"]}):
                tag.extract()
            
            # Look for common content container ids/classes
            content_selectors = [
                "#content", "#main", "#main-content", ".content", ".main", ".main-content",
                "article", "[role=main]", ".post", ".entry", ".entry-content", ".post-content"
            ]
            
            for selector in content_selectors:
                content = soup.select_one(selector)
                if content:
                    # Check if this contains a significant portion of the content
                    content_text = content.get_text(strip=True)
                    total_text = soup.get_text(strip=True)
                    
                    if len(content_text) > 0.3 * len(total_text):
                        return str(content)
            
            # If no suitable container found, return the body without obvious non-content
            return str(soup.body) if soup.body else str(soup)
            
        except Exception as e:
            logger.error(f"Error extracting main content: {str(e)}")
            return html
    
    def extract_tables(self, html: str) -> List[Dict[str, Any]]:
        """Extract table data from HTML."""
        if not html:
            return []
        
        try:
            soup = BeautifulSoup(html, 'lxml')
            tables = []
            
            for i, table in enumerate(soup.find_all('table')):
                headers = []
                rows = []
                
                # Get headers
                thead = table.find('thead')
                if thead:
                    header_row = thead.find('tr')
                    if header_row:
                        headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
                
                # If no headers in thead, try the first row
                if not headers:
                    first_row = table.find('tr')
                    if first_row:
                        headers = [th.get_text(strip=True) for th in first_row.find_all(['th', 'td'])]
                        
                # Process tbody if it exists, otherwise process all rows
                tbody = table.find('tbody') or table
                
                for tr in tbody.find_all('tr'):
                    # Skip header row if we're processing all rows
                    if not headers or tr != table.find('tr'):
                        row = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                        if row:  # Skip empty rows
                            rows.append(row)
                
                # Create table data structure
                table_data = {
                    'id': i,
                    'caption': table.find('caption').get_text(strip=True) if table.find('caption') else None,
                    'headers': headers,
                    'rows': rows
                }
                
                # If headers exist, create dictionaries for each row
                if headers and all(len(row) == len(headers) for row in rows):
                    table_data['data'] = [dict(zip(headers, row)) for row in rows]
                
                tables.append(table_data)
            
            return tables
            
        except Exception as e:
            logger.error(f"Error extracting tables: {str(e)}")
            return []
    
    def extract_links(self, html: str, base_url: Optional[str] = None) -> List[Dict[str, Any]]:
        """Extract links from HTML with metadata."""
        if not html:
            return []
        
        try:
            from urllib.parse import urljoin, urlparse
            
            soup = BeautifulSoup(html, 'lxml')
            links = []
            
            for a in soup.find_all('a', href=True):
                href = a['href']
                original_href = href
                
                # Handle relative URLs if base_url is provided
                if base_url and not href.startswith(('http://', 'https://', 'mailto:', 'tel:', '//')):
                    # Use urljoin to correctly handle all types of relative URLs
                    href = urljoin(base_url, href)
                
                # Extract link metadata
                link_data = {
                    'url': href,
                    'text': a.get_text(strip=True),
                    'title': a.get('title', ''),
                    'rel': a.get('rel', []),
                }
                
                # Determine if link is internal
                if base_url:
                    # For base_url https://test.com/subdir/
                    # Extract the domain part (https://test.com)
                    base_domain = urlparse(base_url).scheme + "://" + urlparse(base_url).netloc
                    link_data['is_internal'] = href.startswith(base_domain)
                else:
                    # Without base_url, consider only relative links as internal
                    link_data['is_internal'] = not original_href.startswith(('http://', 'https://', 'mailto:', 'tel:', '//'))
                
                links.append(link_data)
            
            return links
            
        except Exception as e:
            logger.error(f"Error extracting links: {str(e)}")
            return []
    
    def _remove_injection_patterns(self, soup: BeautifulSoup) -> None:
        """Remove various injection patterns from HTML attributes and content."""
        # Define dangerous injection patterns - more specific to avoid false positives
        sql_patterns = [
            'DROP TABLE', 'DELETE FROM', 'INSERT INTO', 'UPDATE SET',
            'UNION SELECT', '; DROP', 'OR 1=1', 'OR \'1\'=\'1\'',
            'UNION ALL', ') UNION', ' UNION '
        ]
        
        command_patterns = [
            'rm -rf', 'cat /etc/', '/etc/passwd', '/bin/sh', '/usr/bin/',
            'chmod 777', 'chown root', 'su -', 'sudo su', '/etc/shadow',
            '$(rm', '$(cat', '`rm', '`cat'
        ]
        
        # Only check for template patterns in attributes, not content
        template_patterns = [
            '<%=', '%>', '#{', '}#',
            '__import__(', 'eval(', 'exec(', 'compile('
        ]
        
        xml_patterns = [
            '<!DOCTYPE', '<!ENTITY', 'SYSTEM "file://', '&xxe;'
        ]
        
        # Patterns for attributes only (more restrictive)
        attribute_patterns = sql_patterns + command_patterns + template_patterns + xml_patterns
        
        # Patterns for content (very restrictive to avoid false positives)
        content_patterns = [
            '; DROP TABLE', 'rm -rf /', 'cat /etc/passwd', 
            '<!DOCTYPE.*<!ENTITY', '&xxe;', '<%=.*%>',
            '{{', '}}', '${', '<%=', '%>', '#{', '}#'
        ]
        
        # Check all attributes of all elements
        for element in soup.find_all():
            for attr_name, attr_value in list(element.attrs.items()):
                if isinstance(attr_value, str):
                    # Check for dangerous patterns in attribute values
                    attr_lower = attr_value.lower()
                    for pattern in attribute_patterns:
                        if pattern.lower() in attr_lower:
                            # Remove the dangerous attribute
                            del element[attr_name]
                            break
                elif isinstance(attr_value, list):
                    # Handle list attributes (like class)
                    attr_str = ' '.join(attr_value).lower()
                    for pattern in attribute_patterns:
                        if pattern.lower() in attr_str:
                            del element[attr_name]
                            break
        
        # Check text content for injection patterns (only very specific dangerous patterns)
        for text_node in soup.find_all(string=True):
            if hasattr(text_node, 'string') and text_node.string:
                text_lower = text_node.string.lower()
                for pattern in content_patterns:
                    if pattern.lower() in text_lower:
                        # Replace dangerous text with safe placeholder
                        text_node.replace_with('[CONTENT REMOVED FOR SECURITY]')
                        break