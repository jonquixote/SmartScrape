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
