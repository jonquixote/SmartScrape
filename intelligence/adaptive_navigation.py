"""
Adaptive Navigation System

This module provides intelligent navigation capabilities that work with the Universal Hunter
to adaptively navigate websites based on content patterns and user intent.
"""

import logging
import asyncio
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse, parse_qs
import hashlib

from bs4 import BeautifulSoup

logger = logging.getLogger("AdaptiveNavigation")

@dataclass
class NavigationPattern:
    """Represents a navigation pattern discovered on a site"""
    pattern_type: str  # pagination, infinite_scroll, category_nav, search_form
    selector: str  # CSS selector for the pattern
    confidence: float  # Confidence in pattern detection
    metadata: Dict[str, Any]  # Additional pattern-specific data

@dataclass
class NavigationAction:
    """Represents a navigation action to take"""
    action_type: str  # click, scroll, form_submit, url_navigate
    target: str  # CSS selector or URL
    parameters: Dict[str, Any]  # Action parameters
    expected_result: str  # What we expect to happen

class SiteNavigationAnalyzer:
    """Analyzes sites to understand their navigation patterns"""
    
    def __init__(self):
        self.pattern_cache = {}
    
    def analyze_navigation(self, html: str, url: str) -> List[NavigationPattern]:
        """Analyze a page to identify navigation patterns"""
        soup = BeautifulSoup(html, 'html.parser')
        patterns = []
        
        # Detect pagination
        pagination_pattern = self._detect_pagination(soup, url)
        if pagination_pattern:
            patterns.append(pagination_pattern)
        
        # Detect infinite scroll
        infinite_scroll_pattern = self._detect_infinite_scroll(soup)
        if infinite_scroll_pattern:
            patterns.append(infinite_scroll_pattern)
        
        # Detect category navigation
        category_nav_pattern = self._detect_category_navigation(soup)
        if category_nav_pattern:
            patterns.append(category_nav_pattern)
        
        # Detect search forms
        search_form_pattern = self._detect_search_forms(soup)
        if search_form_pattern:
            patterns.append(search_form_pattern)
        
        # Detect load more buttons
        load_more_pattern = self._detect_load_more_buttons(soup)
        if load_more_pattern:
            patterns.append(load_more_pattern)
        
        return patterns
    
    def _detect_pagination(self, soup: BeautifulSoup, url: str) -> Optional[NavigationPattern]:
        """Detect pagination patterns"""
        # Look for common pagination indicators
        pagination_indicators = [
            '.pagination', '.pager', '.page-numbers', '.paginate',
            '[class*="page"]', '[class*="next"]', '[class*="prev"]'
        ]
        
        for indicator in pagination_indicators:
            elements = soup.select(indicator)
            if elements:
                # Look for next page links
                for element in elements:
                    next_links = element.find_all('a', href=True, string=re.compile(r'next|>|»', re.I))
                    if next_links:
                        return NavigationPattern(
                            pattern_type='pagination',
                            selector=indicator,
                            confidence=0.8,
                            metadata={
                                'next_selector': f"{indicator} a[href]",
                                'current_page': self._extract_current_page(element),
                                'total_pages': self._extract_total_pages(element)
                            }
                        )
        
        # Look for numbered pagination
        page_numbers = soup.find_all('a', href=True, string=re.compile(r'^\d+$'))
        if len(page_numbers) >= 3:
            return NavigationPattern(
                pattern_type='pagination',
                selector='a[href]',
                confidence=0.7,
                metadata={
                    'next_selector': 'a[href]',
                    'pattern': 'numbered'
                }
            )
        
        return None
    
    def _detect_infinite_scroll(self, soup: BeautifulSoup) -> Optional[NavigationPattern]:
        """Detect infinite scroll patterns"""
        # Look for infinite scroll indicators
        infinite_scroll_indicators = [
            'data-scroll', 'infinite-scroll', 'lazy-load',
            'data-ajax-url', 'data-next-page'
        ]
        
        for indicator in infinite_scroll_indicators:
            elements = soup.find_all(attrs={indicator: True})
            if elements:
                return NavigationPattern(
                    pattern_type='infinite_scroll',
                    selector=f'[{indicator}]',
                    confidence=0.7,
                    metadata={'trigger_method': 'scroll'}
                )
        
        # Look for JavaScript patterns
        scripts = soup.find_all('script')
        for script in scripts:
            if script.string:
                script_text = script.string.lower()
                if any(pattern in script_text for pattern in ['infinite', 'scroll', 'loadmore']):
                    return NavigationPattern(
                        pattern_type='infinite_scroll',
                        selector='body',
                        confidence=0.5,
                        metadata={'trigger_method': 'scroll', 'js_driven': True}
                    )
        
        return None
    
    def _detect_category_navigation(self, soup: BeautifulSoup) -> Optional[NavigationPattern]:
        """Detect category navigation menus"""
        # Look for navigation menus
        nav_selectors = ['nav', '.nav', '.navigation', '.menu', '.categories']
        
        for selector in nav_selectors:
            nav_elements = soup.select(selector)
            for nav in nav_elements:
                links = nav.find_all('a', href=True)
                if len(links) >= 3:
                    # Check if links look like categories
                    link_texts = [link.get_text().strip() for link in links]
                    if any(text.lower() in ['category', 'section', 'topic', 'tag'] for text in link_texts):
                        return NavigationPattern(
                            pattern_type='category_nav',
                            selector=selector,
                            confidence=0.6,
                            metadata={
                                'categories': link_texts[:10],
                                'link_selector': f"{selector} a[href]"
                            }
                        )
        
        return None
    
    def _detect_search_forms(self, soup: BeautifulSoup) -> Optional[NavigationPattern]:
        """Detect search forms"""
        search_forms = soup.find_all('form')
        
        for form in search_forms:
            # Look for search indicators
            search_inputs = form.find_all('input', attrs={'type': 'search'})
            search_inputs.extend(form.find_all('input', attrs={'name': re.compile(r'search|query|q', re.I)}))
            search_inputs.extend(form.find_all('input', attrs={'placeholder': re.compile(r'search', re.I)}))
            
            if search_inputs:
                return NavigationPattern(
                    pattern_type='search_form',
                    selector='form',
                    confidence=0.8,
                    metadata={
                        'form_action': form.get('action', ''),
                        'form_method': form.get('method', 'get'),
                        'search_input': search_inputs[0].get('name', 'q')
                    }
                )
        
        return None
    
    def _detect_load_more_buttons(self, soup: BeautifulSoup) -> Optional[NavigationPattern]:
        """Detect load more buttons"""
        load_more_patterns = [
            'load more', 'show more', 'view more', 'see more',
            'load additional', 'more results'
        ]
        
        for pattern in load_more_patterns:
            buttons = soup.find_all(['button', 'a'], string=re.compile(pattern, re.I))
            if buttons:
                button = buttons[0]
                return NavigationPattern(
                    pattern_type='load_more',
                    selector=self._generate_selector(button),
                    confidence=0.7,
                    metadata={'button_text': button.get_text().strip()}
                )
        
        return None
    
    def _extract_current_page(self, pagination_element) -> Optional[int]:
        """Extract current page number from pagination element"""
        # Look for current/active page indicators
        current_indicators = pagination_element.find_all(
            attrs={'class': re.compile(r'current|active|selected', re.I)}
        )
        
        for indicator in current_indicators:
            text = indicator.get_text().strip()
            if text.isdigit():
                return int(text)
        
        return None
    
    def _extract_total_pages(self, pagination_element) -> Optional[int]:
        """Extract total pages from pagination element"""
        # Look for page numbers
        page_links = pagination_element.find_all('a', href=True, string=re.compile(r'^\d+$'))
        if page_links:
            page_numbers = [int(link.get_text()) for link in page_links if link.get_text().isdigit()]
            if page_numbers:
                return max(page_numbers)
        
        return None
    
    def _generate_selector(self, element) -> str:
        """Generate a CSS selector for an element"""
        if element.get('id'):
            return f"#{element['id']}"
        elif element.get('class'):
            return f".{element['class'][0]}"
        else:
            return element.name

class AdaptiveNavigator:
    """Handles adaptive navigation based on detected patterns"""
    
    def __init__(self, session=None):
        self.session = session
        self.analyzer = SiteNavigationAnalyzer()
        self.navigation_history = []
    
    async def navigate_intelligently(self, url: str, intent_keywords: List[str], 
                                   max_pages: int = 5) -> List[str]:
        """
        Navigate a site intelligently to find relevant content URLs
        
        Args:
            url: Starting URL
            intent_keywords: Keywords representing user intent
            max_pages: Maximum pages to navigate
            
        Returns:
            List of relevant content URLs found
        """
        relevant_urls = []
        visited_urls = set()
        urls_to_visit = [url]
        
        logger.info(f"🧭 Starting intelligent navigation from {url}")
        
        for page_count in range(max_pages):
            if not urls_to_visit:
                break
            
            current_url = urls_to_visit.pop(0)
            if current_url in visited_urls:
                continue
            
            visited_urls.add(current_url)
            
            try:
                # Get page content
                if not self.session:
                    logger.warning("No session available for navigation")
                    break
                
                async with self.session.get(current_url, timeout=10) as response:
                    if response.status != 200:
                        continue
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                
                # Analyze navigation patterns
                patterns = self.analyzer.analyze_navigation(html, current_url)
                logger.info(f"📊 Found {len(patterns)} navigation patterns on {current_url}")
                
                # Extract content URLs from current page
                content_urls = self._extract_content_urls(soup, current_url, intent_keywords)
                relevant_urls.extend(content_urls)
                
                # Navigate based on patterns
                next_urls = await self._navigate_patterns(
                    patterns, current_url, html, intent_keywords
                )
                
                # Add new URLs to visit (avoid duplicates)
                for next_url in next_urls:
                    if next_url not in visited_urls and next_url not in urls_to_visit:
                        urls_to_visit.append(next_url)
                
                # Limit total URLs to visit
                urls_to_visit = urls_to_visit[:10]
                
            except Exception as e:
                logger.error(f"❌ Navigation error on {current_url}: {e}")
                continue
        
        logger.info(f"🎯 Navigation complete: Found {len(relevant_urls)} relevant URLs")
        return relevant_urls
    
    def _extract_content_urls(self, soup: BeautifulSoup, base_url: str, 
                            intent_keywords: List[str]) -> List[str]:
        """Extract relevant content URLs from current page"""
        content_urls = []
        
        # Look for article/content links
        content_selectors = [
            'article a[href]', '.post a[href]', '.entry a[href]',
            '.item a[href]', '.product a[href]', 'h2 a[href]', 'h3 a[href]'
        ]
        
        for selector in content_selectors:
            links = soup.select(selector)
            for link in links:
                href = link.get('href')
                if not href:
                    continue
                
                full_url = urljoin(base_url, href)
                link_text = link.get_text().strip()
                
                # Check relevance to intent
                if self._is_url_relevant(link_text, full_url, intent_keywords):
                    content_urls.append(full_url)
        
        return content_urls[:20]  # Limit per page
    
    def _is_url_relevant(self, link_text: str, url: str, intent_keywords: List[str]) -> bool:
        """Check if a URL is relevant to the intent keywords"""
        combined_text = (link_text + ' ' + url).lower()
        
        # Count keyword matches
        matches = sum(1 for keyword in intent_keywords 
                     if keyword.lower() in combined_text)
        
        # Require at least one match
        return matches > 0
    
    async def _navigate_patterns(self, patterns: List[NavigationPattern], 
                               current_url: str, html: str, 
                               intent_keywords: List[str]) -> List[str]:
        """Navigate using detected patterns to find more content"""
        next_urls = []
        
        for pattern in patterns:
            if pattern.pattern_type == 'pagination':
                pagination_urls = await self._handle_pagination(
                    pattern, current_url, html
                )
                next_urls.extend(pagination_urls)
            
            elif pattern.pattern_type == 'category_nav':
                category_urls = await self._handle_category_navigation(
                    pattern, current_url, html, intent_keywords
                )
                next_urls.extend(category_urls)
            
            elif pattern.pattern_type == 'search_form':
                search_urls = await self._handle_search_form(
                    pattern, current_url, intent_keywords
                )
                next_urls.extend(search_urls)
        
        return next_urls
    
    async def _handle_pagination(self, pattern: NavigationPattern, 
                               current_url: str, html: str) -> List[str]:
        """Handle pagination navigation"""
        next_urls = []
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find next page links
        next_selector = pattern.metadata.get('next_selector', pattern.selector)
        next_links = soup.select(next_selector)
        
        for link in next_links[:3]:  # Limit to first 3 next pages
            href = link.get('href')
            if href:
                next_url = urljoin(current_url, href)
                # Avoid circular navigation
                if next_url != current_url:
                    next_urls.append(next_url)
        
        return next_urls
    
    async def _handle_category_navigation(self, pattern: NavigationPattern,
                                        current_url: str, html: str,
                                        intent_keywords: List[str]) -> List[str]:
        """Handle category navigation"""
        category_urls = []
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find category links
        link_selector = pattern.metadata.get('link_selector', f"{pattern.selector} a[href]")
        category_links = soup.select(link_selector)
        
        for link in category_links:
            href = link.get('href')
            link_text = link.get_text().strip()
            
            if href and self._is_url_relevant(link_text, href, intent_keywords):
                category_url = urljoin(current_url, href)
                category_urls.append(category_url)
        
        return category_urls[:5]  # Limit category exploration
    
    async def _handle_search_form(self, pattern: NavigationPattern,
                                current_url: str, intent_keywords: List[str]) -> List[str]:
        """Handle search form navigation"""
        search_urls = []
        
        # Construct search URL
        form_action = pattern.metadata.get('form_action', '')
        search_input = pattern.metadata.get('search_input', 'q')
        
        if not form_action:
            form_action = current_url
        
        # Use first intent keyword as search term
        if intent_keywords:
            search_term = intent_keywords[0]
            
            # Build search URL
            if '?' in form_action:
                search_url = f"{form_action}&{search_input}={search_term}"
            else:
                search_url = f"{form_action}?{search_input}={search_term}"
            
            search_urls.append(search_url)
        
        return search_urls

class ContentLinkExtractor:
    """Extracts content links from pages using various strategies"""
    
    def __init__(self):
        self.extraction_patterns = {
            'news': [
                'article a[href]', '.post a[href]', '.entry a[href]',
                '.news-item a[href]', '.story a[href]', 'h2 a[href]'
            ],
            'ecommerce': [
                '.product a[href]', '.item a[href]', '.product-item a[href]',
                '.product-card a[href]', '.listing a[href]'
            ],
            'blog': [
                '.post a[href]', '.entry a[href]', '.blog-post a[href]',
                'article a[href]', 'h2 a[href]', 'h3 a[href]'
            ],
            'general': [
                'article a[href]', '.content a[href]', '.main a[href]',
                'h1 a[href]', 'h2 a[href]', 'h3 a[href]'
            ]
        }
    
    def extract_content_links(self, html: str, base_url: str, 
                            site_type: str = 'general') -> List[Tuple[str, str]]:
        """
        Extract content links from HTML
        
        Args:
            html: HTML content
            base_url: Base URL for resolving relative links
            site_type: Type of site (news, ecommerce, blog, general)
            
        Returns:
            List of (url, title) tuples
        """
        soup = BeautifulSoup(html, 'html.parser')
        links = []
        
        # Get patterns for site type
        patterns = self.extraction_patterns.get(site_type, self.extraction_patterns['general'])
        
        for pattern in patterns:
            elements = soup.select(pattern)
            for element in elements:
                href = element.get('href')
                if href:
                    full_url = urljoin(base_url, href)
                    title = element.get_text().strip() or element.get('title', '')
                    
                    if title and len(title) > 5:  # Filter out empty or very short titles
                        links.append((full_url, title))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_links = []
        for url, title in links:
            if url not in seen:
                seen.add(url)
                unique_links.append((url, title))
        
        return unique_links[:50]  # Limit results

# Enhanced navigation methods for Universal Intelligence
    
    def analyze_link_relationships(self, html: str, url: str) -> Dict[str, List[str]]:
        """
        Analyze link relationships on a page to understand content hierarchy.
        
        Returns:
            Dictionary with relationship types and associated URLs
        """
        soup = BeautifulSoup(html, 'html.parser')
        relationships = {
            'parent_pages': [],
            'child_pages': [],
            'related_content': [],
            'external_links': [],
            'pagination_links': [],
            'category_links': []
        }
        
        base_domain = urlparse(url).netloc
        
        # Analyze all links
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            absolute_url = urljoin(url, href)
            link_domain = urlparse(absolute_url).netloc
            link_text = link.get_text().strip().lower()
            link_classes = ' '.join(link.get('class', []))
            
            # Parent page indicators
            if any(indicator in link_text for indicator in ['back', 'up', 'parent', 'home', 'main']):
                relationships['parent_pages'].append(absolute_url)
            
            # Child/detail page indicators
            elif any(indicator in link_classes for indicator in ['detail', 'read-more', 'view', 'article']):
                relationships['child_pages'].append(absolute_url)
            
            # Category/section links
            elif any(indicator in link_classes for indicator in ['category', 'section', 'tag']):
                relationships['category_links'].append(absolute_url)
            
            # Pagination links
            elif any(indicator in link_text for indicator in ['next', 'previous', 'page']) or \
                 any(indicator in link_classes for indicator in ['page', 'pagination']):
                relationships['pagination_links'].append(absolute_url)
            
            # External links
            elif link_domain != base_domain:
                relationships['external_links'].append(absolute_url)
            
            # Related content (same domain, not navigation)
            elif link_domain == base_domain and \
                 not any(nav in link_classes for nav in ['nav', 'menu', 'footer', 'header']):
                relationships['related_content'].append(absolute_url)
        
        return relationships
    
    def detect_breadcrumbs(self, html: str) -> Optional[List[Dict[str, str]]]:
        """
        Detect and parse breadcrumb navigation.
        
        Returns:
            List of breadcrumb items with text and URL
        """
        soup = BeautifulSoup(html, 'html.parser')
        breadcrumbs = []
        
        # Common breadcrumb selectors
        breadcrumb_selectors = [
            '.breadcrumb', '.breadcrumbs', '.breadcrumb-nav',
            '.crumbs', '[aria-label*="breadcrumb"]', '.path',
            '.navigation-path', '.page-path'
        ]
        
        for selector in breadcrumb_selectors:
            breadcrumb_container = soup.select_one(selector)
            if breadcrumb_container:
                # Extract breadcrumb items
                links = breadcrumb_container.find_all('a', href=True)
                for link in links:
                    breadcrumbs.append({
                        'text': link.get_text().strip(),
                        'url': link.get('href'),
                        'is_current': 'current' in ' '.join(link.get('class', []))
                    })
                
                # Check for current page (non-linked text)
                if breadcrumbs:
                    text_nodes = breadcrumb_container.find_all(text=True)
                    current_text = ''.join(text_nodes).strip()
                    # Look for text after the last separator
                    separators = ['>', '/', '→', '»', '|']
                    for sep in separators:
                        if sep in current_text:
                            parts = current_text.split(sep)
                            if len(parts) > len(breadcrumbs):
                                breadcrumbs.append({
                                    'text': parts[-1].strip(),
                                    'url': None,
                                    'is_current': True
                                })
                            break
                
                if breadcrumbs:
                    return breadcrumbs
        
        return None
    
    def find_content_discovery_links(self, html: str, url: str, content_type: str) -> List[Dict[str, Any]]:
        """
        Find links that are likely to lead to the desired content type.
        
        Args:
            html: Page HTML
            url: Current page URL
            content_type: Type of content to find (news, products, jobs, etc.)
            
        Returns:
            List of promising links with metadata
        """
        soup = BeautifulSoup(html, 'html.parser')
        content_links = []
        
        # Content-type specific patterns
        content_patterns = {
            'news': {
                'keywords': ['news', 'article', 'story', 'report', 'update', 'breaking'],
                'selectors': ['.news-item', '.article-link', '.story-link'],
                'avoid': ['archive', 'old', 'category']
            },
            'products': {
                'keywords': ['product', 'item', 'buy', 'shop', 'price', 'details'],
                'selectors': ['.product-link', '.item-link', '.product-title'],
                'avoid': ['category', 'brand', 'filter']
            },
            'jobs': {
                'keywords': ['job', 'position', 'career', 'opening', 'role', 'apply'],
                'selectors': ['.job-link', '.position-link', '.career-link'],
                'avoid': ['company', 'department', 'location']
            },
            'contact': {
                'keywords': ['contact', 'about', 'location', 'office', 'phone'],
                'selectors': ['.contact-link', '.about-link'],
                'avoid': ['careers', 'news', 'products']
            }
        }
        
        if content_type not in content_patterns:
            return content_links
        
        patterns = content_patterns[content_type]
        
        # Find links using specific selectors
        for selector in patterns['selectors']:
            for link in soup.select(f"{selector} a[href]"):
                href = link.get('href')
                text = link.get_text().strip()
                if href and text:
                    content_links.append({
                        'url': urljoin(url, href),
                        'text': text,
                        'confidence': 0.8,
                        'discovery_method': 'selector',
                        'selector': selector
                    })
        
        # Find links using keyword matching
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            text = link.get_text().strip().lower()
            classes = ' '.join(link.get('class', [])).lower()
            
            if not href or not text:
                continue
            
            # Check for positive keywords
            keyword_score = sum(1 for keyword in patterns['keywords'] if keyword in text or keyword in classes)
            
            # Check for negative keywords
            avoid_score = sum(1 for avoid in patterns['avoid'] if avoid in text or avoid in classes)
            
            # Calculate confidence
            confidence = (keyword_score * 0.3) - (avoid_score * 0.2)
            
            if confidence > 0.2:
                content_links.append({
                    'url': urljoin(url, href),
                    'text': link.get_text().strip(),
                    'confidence': min(confidence, 0.9),
                    'discovery_method': 'keyword',
                    'keywords_found': [kw for kw in patterns['keywords'] if kw in text]
                })
        
        # Sort by confidence and remove duplicates
        seen_urls = set()
        unique_links = []
        for link in sorted(content_links, key=lambda x: x['confidence'], reverse=True):
            if link['url'] not in seen_urls:
                seen_urls.add(link['url'])
                unique_links.append(link)
        
        return unique_links[:10]  # Return top 10 most promising links
    
    def detect_search_interface(self, html: str) -> Optional[Dict[str, Any]]:
        """
        Detect and analyze search interfaces on the page.
        
        Returns:
            Search interface configuration if found
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find search forms
        search_forms = []
        for form in soup.find_all('form'):
            search_inputs = []
            
            # Look for search input fields
            for input_elem in form.find_all('input'):
                input_type = input_elem.get('type', '').lower()
                input_name = input_elem.get('name', '').lower()
                input_placeholder = input_elem.get('placeholder', '').lower()
                
                if (input_type in ['search', 'text'] and 
                    ('search' in input_name or 'q' in input_name or 'query' in input_name or
                     'search' in input_placeholder)):
                    search_inputs.append({
                        'name': input_elem.get('name', 'q'),
                        'type': input_type,
                        'placeholder': input_elem.get('placeholder', ''),
                        'required': input_elem.has_attr('required')
                    })
            
            if search_inputs:
                # Look for filters and additional fields
                filters = []
                for select in form.find_all('select'):
                    select_name = select.get('name', '').lower()
                    if any(filter_term in select_name for filter_term in ['category', 'type', 'sort', 'filter']):
                        options = [opt.get('value') for opt in select.find_all('option') if opt.get('value')]
                        filters.append({
                            'name': select.get('name'),
                            'type': 'select',
                            'options': options
                        })
                
                search_forms.append({
                    'action': form.get('action', ''),
                    'method': form.get('method', 'get').lower(),
                    'search_inputs': search_inputs,
                    'filters': filters,
                    'submit_selector': 'input[type="submit"], button[type="submit"]'
                })
        
        if search_forms:
            # Return the most comprehensive search form
            best_form = max(search_forms, key=lambda x: len(x['search_inputs']) + len(x['filters']))
            return {
                'available': True,
                'forms': search_forms,
                'recommended_form': best_form,
                'can_filter': len(best_form['filters']) > 0
            }
        
        return None
    
    def detect_filter_options(self, html: str) -> List[Dict[str, Any]]:
        """
        Detect available filter options on the page.
        
        Returns:
            List of available filters with their options
        """
        soup = BeautifulSoup(html, 'html.parser')
        filters = []
        
        # Look for filter forms and controls
        filter_selectors = [
            '.filter', '.filters', '.facet', '.facets',
            '.sidebar', '.filter-panel', '.search-filters'
        ]
        
        for selector in filter_selectors:
            filter_container = soup.select_one(selector)
            if filter_container:
                # Find select dropdowns
                for select in filter_container.find_all('select'):
                    options = []
                    for option in select.find_all('option'):
                        if option.get('value') and option.get('value') != '':
                            options.append({
                                'value': option.get('value'),
                                'text': option.get_text().strip()
                            })
                    
                    if options:
                        filters.append({
                            'type': 'select',
                            'name': select.get('name', ''),
                            'label': self._extract_filter_label(select),
                            'options': options,
                            'selector': f"select[name='{select.get('name')}']"
                        })
                
                # Find checkbox groups
                checkbox_groups = {}
                for checkbox in filter_container.find_all('input', type='checkbox'):
                    name = checkbox.get('name', '')
                    if name:
                        if name not in checkbox_groups:
                            checkbox_groups[name] = []
                        checkbox_groups[name].append({
                            'value': checkbox.get('value', ''),
                            'text': self._extract_checkbox_label(checkbox),
                            'checked': checkbox.has_attr('checked')
                        })
                
                for name, options in checkbox_groups.items():
                    filters.append({
                        'type': 'checkbox_group',
                        'name': name,
                        'label': name.replace('_', ' ').title(),
                        'options': options,
                        'selector': f"input[name='{name}']"
                    })
        
        return filters
    
    def _extract_filter_label(self, element) -> str:
        """Extract label for a filter element"""
        # Look for associated label
        element_id = element.get('id')
        if element_id:
            label = element.find_previous('label', {'for': element_id})
            if label:
                return label.get_text().strip()
        
        # Look for preceding text
        prev_text = []
        for prev in element.previous_siblings:
            if hasattr(prev, 'get_text'):
                text = prev.get_text().strip()
                if text:
                    prev_text.append(text)
                    break
            elif isinstance(prev, str):
                text = prev.strip()
                if text:
                    prev_text.append(text)
                    break
        
        return prev_text[0] if prev_text else element.get('name', 'Unknown')
    
    def _extract_checkbox_label(self, checkbox) -> str:
        """Extract label text for a checkbox"""
        # Look for associated label
        checkbox_id = checkbox.get('id')
        if checkbox_id:
            label = checkbox.find_next('label', {'for': checkbox_id})
            if label:
                return label.get_text().strip()
        
        # Look for label wrapping the checkbox
        parent_label = checkbox.find_parent('label')
        if parent_label:
            # Get text excluding the checkbox itself
            text_parts = []
            for content in parent_label.contents:
                if hasattr(content, 'name') and content.name == 'input':
                    continue
                text_parts.append(str(content).strip())
            return ''.join(text_parts).strip()
        
        # Look for following text
        next_sibling = checkbox.next_sibling
        if next_sibling:
            return str(next_sibling).strip()
        
        return checkbox.get('value', 'Unknown')

class DynamicContentHandler:
    """Handles dynamic content loading and interaction"""
    
    def __init__(self, session=None):
        self.session = session
        self.js_patterns = {
            'infinite_scroll': [
                'infinite', 'scroll', 'loadmore', 'load-more',
                'ajax-load', 'lazy-load', 'auto-load'
            ],
            'tab_content': [
                'tab', 'tabs', 'accordion', 'toggle', 'expand'
            ],
            'modal_content': [
                'modal', 'popup', 'overlay', 'lightbox', 'dialog'
            ],
            'dynamic_form': [
                'ajax-form', 'dynamic-form', 'live-search', 'autocomplete'
            ]
        }
    
    def detect_dynamic_content_patterns(self, html: str) -> List[Dict[str, Any]]:
        """
        Detect patterns that indicate dynamic content loading.
        
        Returns:
            List of dynamic content patterns found
        """
        soup = BeautifulSoup(html, 'html.parser')
        patterns = []
        
        # Check for infinite scroll indicators
        infinite_scroll = self._detect_infinite_scroll_advanced(soup)
        if infinite_scroll:
            patterns.append(infinite_scroll)
        
        # Check for tab/accordion content
        tab_content = self._detect_tab_accordion_content(soup)
        if tab_content:
            patterns.extend(tab_content)
        
        # Check for modal/popup triggers
        modal_triggers = self._detect_modal_triggers(soup)
        if modal_triggers:
            patterns.extend(modal_triggers)
        
        # Check for AJAX-loaded content
        ajax_content = self._detect_ajax_content(soup)
        if ajax_content:
            patterns.extend(ajax_content)
        
        return patterns
    
    def _detect_infinite_scroll_advanced(self, soup: BeautifulSoup) -> Optional[Dict[str, Any]]:
        """Advanced infinite scroll detection"""
        indicators = []
        
        # Check for data attributes
        data_attrs = [
            'data-infinite-scroll', 'data-auto-load', 'data-lazy-load',
            'data-next-page', 'data-load-more', 'data-ajax-url'
        ]
        
        for attr in data_attrs:
            elements = soup.find_all(attrs={attr: True})
            if elements:
                indicators.append({
                    'type': 'data_attribute',
                    'attribute': attr,
                    'elements': len(elements),
                    'confidence': 0.9
                })
        
        # Check JavaScript for scroll patterns
        scripts = soup.find_all('script')
        for script in scripts:
            if script.string:
                script_content = script.string.lower()
                scroll_patterns = [
                    'window.onscroll', 'scroll', 'loadmore', 'infinite',
                    'ajax.*scroll', 'lazy.*load'
                ]
                
                for pattern in scroll_patterns:
                    if re.search(pattern, script_content):
                        indicators.append({
                            'type': 'javascript_pattern',
                            'pattern': pattern,
                            'confidence': 0.7
                        })
        
        # Check for CSS classes
        css_classes = [
            'infinite-scroll', 'lazy-load', 'auto-load',
            'scroll-load', 'endless-scroll'
        ]
        
        for css_class in css_classes:
            elements = soup.find_all(class_=re.compile(css_class, re.I))
            if elements:
                indicators.append({
                    'type': 'css_class',
                    'class': css_class,
                    'elements': len(elements),
                    'confidence': 0.8
                })
        
        if indicators:
            return {
                'pattern_type': 'infinite_scroll',
                'indicators': indicators,
                'confidence': max(ind['confidence'] for ind in indicators),
                'trigger_method': 'scroll_to_bottom',
                'interaction_required': True
            }
        
        return None
    
    def _detect_tab_accordion_content(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Detect tab and accordion content that needs to be activated"""
        tab_patterns = []
        
        # Look for tab structures
        tab_containers = soup.find_all(['div', 'ul'], class_=re.compile(r'tab|accordion', re.I))
        
        for container in tab_containers:
            tabs = container.find_all(['a', 'button', 'li'], href=True)
            if not tabs:
                tabs = container.find_all(['a', 'button', 'li'], attrs={'data-toggle': True})
            
            if tabs:
                tab_info = {
                    'pattern_type': 'tab_content',
                    'container_selector': self._generate_selector(container),
                    'tabs': [],
                    'confidence': 0.8,
                    'interaction_required': True
                }
                
                for tab in tabs[:5]:  # Limit to first 5 tabs
                    tab_data = {
                        'selector': self._generate_selector(tab),
                        'text': tab.get_text().strip(),
                        'target': tab.get('href') or tab.get('data-target') or tab.get('data-toggle'),
                        'active': 'active' in tab.get('class', [])
                    }
                    tab_info['tabs'].append(tab_data)
                
                if tab_info['tabs']:
                    tab_patterns.append(tab_info)
        
        return tab_patterns
    
    def _detect_modal_triggers(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Detect elements that trigger modal/popup content"""
        modal_triggers = []
        
        # Look for modal trigger attributes
        modal_attrs = ['data-toggle', 'data-modal', 'data-popup', 'data-lightbox']
        
        for attr in modal_attrs:
            triggers = soup.find_all(attrs={attr: True})
            for trigger in triggers:
                modal_info = {
                    'pattern_type': 'modal_content',
                    'trigger_selector': self._generate_selector(trigger),
                    'trigger_text': trigger.get_text().strip(),
                    'modal_target': trigger.get(attr),
                    'confidence': 0.7,
                    'interaction_required': True
                }
                modal_triggers.append(modal_info)
        
        # Look for popup/modal CSS classes
        popup_classes = ['popup', 'modal', 'lightbox', 'overlay', 'dialog']
        for css_class in popup_classes:
            triggers = soup.find_all(class_=re.compile(css_class, re.I))
            for trigger in triggers:
                if trigger.name in ['a', 'button']:
                    modal_info = {
                        'pattern_type': 'modal_content',
                        'trigger_selector': self._generate_selector(trigger),
                        'trigger_text': trigger.get_text().strip(),
                        'css_class': css_class,
                        'confidence': 0.6,
                        'interaction_required': True
                    }
                    modal_triggers.append(modal_info)
        
        return modal_triggers
    
    def _detect_ajax_content(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Detect AJAX-loaded content areas"""
        ajax_patterns = []
        
        # Look for AJAX indicators
        ajax_attrs = [
            'data-ajax-url', 'data-remote', 'data-src',
            'data-load-url', 'data-content-url'
        ]
        
        for attr in ajax_attrs:
            elements = soup.find_all(attrs={attr: True})
            for element in elements:
                ajax_info = {
                    'pattern_type': 'ajax_content',
                    'container_selector': self._generate_selector(element),
                    'ajax_url': element.get(attr),
                    'confidence': 0.8,
                    'interaction_required': False,
                    'load_trigger': 'page_load'
                }
                ajax_patterns.append(ajax_info)
        
        # Look for load more buttons
        load_more_selectors = [
            'button', 'a', 'div'
        ]
        
        load_more_text_patterns = [
            r'load\s+more', r'show\s+more', r'view\s+more',
            r'see\s+more', r'more\s+results', r'load\s+additional'
        ]
        
        for selector in load_more_selectors:
            for element in soup.find_all(selector):
                element_text = element.get_text().strip().lower()
                for pattern in load_more_text_patterns:
                    if re.search(pattern, element_text):
                        ajax_info = {
                            'pattern_type': 'load_more_button',
                            'trigger_selector': self._generate_selector(element),
                            'trigger_text': element.get_text().strip(),
                            'confidence': 0.7,
                            'interaction_required': True,
                            'load_trigger': 'click'
                        }
                        ajax_patterns.append(ajax_info)
                        break
        
        return ajax_patterns
    
    def _generate_selector(self, element) -> str:
        """Generate a CSS selector for an element"""
        selectors = []
        
        # Add ID if available
        if element.get('id'):
            return f"#{element['id']}"
        
        # Add tag name
        selectors.append(element.name)
        
        # Add classes
        if element.get('class'):
            classes = element['class']
            if isinstance(classes, list):
                classes = ' '.join(classes)
            selectors.append(f".{classes.replace(' ', '.')}")
        
        # Add attribute selectors for unique identification
        for attr in ['data-id', 'data-target', 'href']:
            if element.get(attr):
                selectors.append(f"[{attr}='{element[attr]}']")
                break
        
        return ''.join(selectors)
    
    def create_interaction_plan(self, dynamic_patterns: List[Dict[str, Any]], content_goal: str) -> List[Dict[str, Any]]:
        """
        Create a plan for interacting with dynamic content to achieve content goals.
        
        Args:
            dynamic_patterns: List of detected dynamic content patterns
            content_goal: What type of content we're trying to access
            
        Returns:
            List of interaction steps
        """
        interaction_plan = []
        
        for pattern in dynamic_patterns:
            pattern_type = pattern['pattern_type']
            
            if pattern_type == 'infinite_scroll':
                interaction_plan.append({
                    'action': 'scroll_to_trigger_load',
                    'method': 'scroll_to_bottom',
                    'repeat': True,
                    'max_iterations': 5,
                    'wait_time': 2,
                    'success_indicator': 'new_content_loaded',
                    'priority': 1 if 'more' in content_goal.lower() else 2
                })
            
            elif pattern_type == 'tab_content':
                for tab in pattern['tabs']:
                    if not tab['active']:
                        interaction_plan.append({
                            'action': 'click_tab',
                            'selector': tab['selector'],
                            'tab_text': tab['text'],
                            'wait_time': 1,
                            'success_indicator': 'tab_content_visible',
                            'priority': 2
                        })
            
            elif pattern_type == 'modal_content':
                interaction_plan.append({
                    'action': 'click_modal_trigger',
                    'selector': pattern['trigger_selector'],
                    'trigger_text': pattern['trigger_text'],
                    'wait_time': 1,
                    'success_indicator': 'modal_visible',
                    'priority': 3
                })
            
            elif pattern_type == 'load_more_button':
                interaction_plan.append({
                    'action': 'click_load_more',
                    'selector': pattern['trigger_selector'],
                    'repeat': True,
                    'max_iterations': 3,
                    'wait_time': 2,
                    'success_indicator': 'new_content_loaded',
                    'priority': 1
                })
        
        # Sort by priority (lower number = higher priority)
        interaction_plan.sort(key=lambda x: x['priority'])
        
        return interaction_plan

    def detect_spa_patterns(self, html: str, url: str) -> Dict[str, Any]:
        """
        Detect Single Page Application patterns that require JavaScript rendering.
        
        Args:
            html: Page HTML content
            url: Page URL
            
        Returns:
            Dictionary with SPA detection results
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        spa_indicators = {
            'is_spa': False,
            'confidence': 0.0,
            'indicators': [],
            'framework': 'unknown',
            'rendering_required': False
        }
        
        # Check for SPA frameworks
        frameworks = {
            'react': ['react', 'reactdom', '__REACT_DEVTOOLS__'],
            'angular': ['angular', 'ng-app', 'ng-controller'],
            'vue': ['vue', 'vuejs', '__VUE__'],
            'ember': ['ember', 'emberjs'],
            'svelte': ['svelte'],
            'next.js': ['__NEXT_DATA__', '_next'],
            'nuxt': ['__NUXT__'],
            'gatsby': ['__GATSBY']
        }
        
        scripts = soup.find_all('script')
        for script in scripts:
            script_content = script.string or ''
            script_src = script.get('src', '')
            
            for framework, indicators in frameworks.items():
                for indicator in indicators:
                    if (indicator in script_content.lower() or 
                        indicator in script_src.lower()):
                        spa_indicators['framework'] = framework
                        spa_indicators['indicators'].append({
                            'type': 'framework_detection',
                            'framework': framework,
                            'indicator': indicator,
                            'confidence': 0.9
                        })
        
        # Check for minimal HTML content (common in SPAs)
        body = soup.find('body')
        if body:
            body_text = body.get_text(strip=True)
            if len(body_text) < 200:  # Very little static content
                spa_indicators['indicators'].append({
                    'type': 'minimal_content',
                    'content_length': len(body_text),
                    'confidence': 0.6
                })
        
        # Check for app containers
        app_containers = soup.find_all(['div'], id=re.compile(r'app|root|main', re.I))
        if app_containers:
            for container in app_containers:
                if not container.get_text(strip=True):  # Empty container
                    spa_indicators['indicators'].append({
                        'type': 'empty_app_container',
                        'container_id': container.get('id'),
                        'confidence': 0.8
                    })
        
        # Check for route handling indicators
        if any('history.pushstate' in str(script).lower() or 
               'router' in str(script).lower() for script in scripts):
            spa_indicators['indicators'].append({
                'type': 'client_side_routing',
                'confidence': 0.7
            })
        
        # Calculate overall confidence
        if spa_indicators['indicators']:
            spa_indicators['confidence'] = max(
                indicator['confidence'] for indicator in spa_indicators['indicators']
            )
            spa_indicators['is_spa'] = spa_indicators['confidence'] > 0.6
            spa_indicators['rendering_required'] = spa_indicators['confidence'] > 0.7
        
        return spa_indicators
    
    def detect_captcha_challenges(self, html: str) -> List[Dict[str, Any]]:
        """
        Detect CAPTCHA challenges that need to be handled.
        
        Args:
            html: Page HTML content
            
        Returns:
            List of detected CAPTCHA challenges
        """
        soup = BeautifulSoup(html, 'html.parser')
        captcha_challenges = []
        
        # Common CAPTCHA services
        captcha_services = {
            'recaptcha': {
                'indicators': ['g-recaptcha', 'recaptcha', 'grecaptcha'],
                'selectors': ['.g-recaptcha', '[data-sitekey]', '#recaptcha'],
                'confidence': 0.9
            },
            'hcaptcha': {
                'indicators': ['h-captcha', 'hcaptcha'],
                'selectors': ['.h-captcha', '[data-hcaptcha-sitekey]'],
                'confidence': 0.9
            },
            'cloudflare': {
                'indicators': ['cf-challenge', 'cloudflare'],
                'selectors': ['.cf-challenge-running', '#cf-challenge-stage'],
                'confidence': 0.8
            },
            'custom': {
                'indicators': ['captcha', 'verification', 'prove you are human'],
                'selectors': ['.captcha', '.verification', '.challenge'],
                'confidence': 0.6
            }
        }
        
        for service, config in captcha_services.items():
            found = False
            
            # Check text content
            page_text = soup.get_text().lower()
            for indicator in config['indicators']:
                if indicator in page_text:
                    found = True
                    break
            
            # Check for specific elements
            for selector in config['selectors']:
                if soup.select(selector):
                    found = True
                    break
            
            if found:
                captcha_challenges.append({
                    'type': service,
                    'confidence': config['confidence'],
                    'selectors': config['selectors'],
                    'bypass_required': True,
                    'complexity': 'high' if service in ['recaptcha', 'hcaptcha'] else 'medium'
                })
        
        return captcha_challenges
    
    def analyze_form_interactions(self, html: str) -> List[Dict[str, Any]]:
        """
        Analyze forms that may require interaction for content access.
        
        Args:
            html: Page HTML content
            
        Returns:
            List of form interaction opportunities
        """
        soup = BeautifulSoup(html, 'html.parser')
        form_interactions = []
        
        forms = soup.find_all('form')
        
        for form in forms:
            form_analysis = {
                'form_selector': self._generate_selector(form),
                'action': form.get('action', ''),
                'method': form.get('method', 'get').lower(),
                'inputs': [],
                'form_type': 'unknown',
                'interaction_required': False,
                'confidence': 0.5
            }
            
            # Analyze form inputs
            inputs = form.find_all(['input', 'select', 'textarea'])
            
            for input_elem in inputs:
                input_type = input_elem.get('type', 'text')
                input_name = input_elem.get('name', '')
                input_placeholder = input_elem.get('placeholder', '')
                
                input_data = {
                    'type': input_type,
                    'name': input_name,
                    'placeholder': input_placeholder,
                    'required': input_elem.get('required') is not None,
                    'selector': self._generate_selector(input_elem)
                }
                
                form_analysis['inputs'].append(input_data)
                
                # Determine form type based on inputs
                if input_type == 'search' or 'search' in input_name.lower():
                    form_analysis['form_type'] = 'search'
                    form_analysis['confidence'] = 0.9
                    form_analysis['interaction_required'] = True
                elif input_type == 'email' or 'email' in input_name.lower():
                    form_analysis['form_type'] = 'email_signup'
                    form_analysis['confidence'] = 0.8
                elif input_type == 'password':
                    form_analysis['form_type'] = 'login'
                    form_analysis['confidence'] = 0.9
                elif 'filter' in input_name.lower() or 'sort' in input_name.lower():
                    form_analysis['form_type'] = 'filter'
                    form_analysis['confidence'] = 0.8
                    form_analysis['interaction_required'] = True
            
            # Check submit buttons
            submit_buttons = form.find_all(['button', 'input'], type='submit')
            submit_buttons.extend(form.find_all('button', type=None))
            
            for button in submit_buttons:
                button_text = button.get_text().strip().lower()
                if any(keyword in button_text for keyword in ['search', 'find', 'filter', 'submit']):
                    form_analysis['interaction_required'] = True
                    form_analysis['confidence'] = min(1.0, form_analysis['confidence'] + 0.2)
            
            if form_analysis['inputs']:
                form_interactions.append(form_analysis)
        
        return form_interactions
    
    def create_javascript_rendering_strategy(self, html: str, url: str) -> Dict[str, Any]:
        """
        Create a strategy for handling JavaScript rendering requirements.
        
        Args:
            html: Page HTML content
            url: Page URL
            
        Returns:
            JavaScript rendering strategy
        """
        spa_analysis = self.detect_spa_patterns(html, url)
        dynamic_patterns = self.detect_dynamic_content_patterns(html)
        captcha_challenges = self.detect_captcha_challenges(html)
        
        strategy = {
            'rendering_required': False,
            'rendering_type': 'static',
            'wait_strategy': 'immediate',
            'interaction_steps': [],
            'challenges': [],
            'estimated_load_time': 1
        }
        
        # Determine if rendering is required
        if spa_analysis['rendering_required']:
            strategy['rendering_required'] = True
            strategy['rendering_type'] = 'spa'
            strategy['wait_strategy'] = 'dynamic_content_loaded'
            strategy['estimated_load_time'] = 5
        elif any(pattern['interaction_required'] for pattern in dynamic_patterns):
            strategy['rendering_required'] = True
            strategy['rendering_type'] = 'dynamic_interactions'
            strategy['wait_strategy'] = 'interaction_based'
            strategy['estimated_load_time'] = 3
        
        # Add interaction steps
        if dynamic_patterns:
            interaction_plan = self.create_interaction_plan(dynamic_patterns, 'comprehensive')
            strategy['interaction_steps'] = interaction_plan[:5]  # Limit interactions
        
        # Add challenges
        if captcha_challenges:
            strategy['challenges'].extend(captcha_challenges)
            strategy['estimated_load_time'] += 10  # CAPTCHAs take time
        
        return strategy
