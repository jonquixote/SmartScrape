"""
Aggressive Multi-Source URL Discovery System

This module implements a comprehensive URL discovery system that uses multiple
search engines and sources to ensure we always get enough high-quality URLs
for content hunting, regardless of rate limits or API issues.

Sources:
1. DuckDuckGo (primary)
2. Bing Search API (if available)
3. Google Custom Search (if available) 
4. SearX instances (public search engines)
5. Direct news site search (for news queries)
6. GitHub/Reddit search (for tech queries)
7. Fallback domain lists
"""

import asyncio
import logging
import time
import random
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urlparse, urljoin, quote
from dataclasses import dataclass
import aiohttp
import requests

@dataclass
class SearchSource:
    """Represents a search source with its configuration"""
    name: str
    url_template: str
    parser_func: str
    max_results: int = 20
    timeout: int = 10
    rate_limit_delay: float = 1.0
    active: bool = True

@dataclass 
class DiscoveredURL:
    """Represents a discovered URL with metadata"""
    url: str
    title: str = ""
    description: str = ""
    source: str = ""
    relevance_score: float = 0.0
    ranking: int = 0

class AggressiveURLDiscovery:
    """
    Aggressive multi-source URL discovery system that ensures we always
    get enough URLs for content hunting.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session = None
        
        # Search sources configuration - prioritizing fastest and most reliable
        self.search_sources = {
            # Primary DuckDuckGo (most reliable)
            'duckduckgo_instant': SearchSource(
                name='DuckDuckGo Instant',
                url_template='https://duckduckgo.com/html/?q={query}&dc=42',
                parser_func='_parse_duckduckgo_html',
                max_results=40,
                timeout=8
            ),
            
            # Alternative DuckDuckGo endpoints  
            'duckduckgo_lite': SearchSource(
                name='DuckDuckGo Lite',
                url_template='https://lite.duckduckgo.com/lite/?q={query}',
                parser_func='_parse_duckduckgo_lite_html',
                max_results=30,
                timeout=6
            ),
            
            # Bing search (no API key needed for basic search)
            'bing_web': SearchSource(
                name='Bing Web Search',
                url_template='https://www.bing.com/search?q={query}&count=50',
                parser_func='_parse_bing_html',
                max_results=50,
                timeout=10
            ),
            
            # Yahoo search
            'yahoo_search': SearchSource(
                name='Yahoo Search',
                url_template='https://search.yahoo.com/search?p={query}&n=30',
                parser_func='_parse_yahoo_html',
                max_results=30,
                timeout=10
            ),
            
            # Yandex search (very reliable)
            'yandex_search': SearchSource(
                name='Yandex Search',
                url_template='https://yandex.com/search/?text={query}&lr=84',
                parser_func='_parse_yandex_html',
                max_results=30,
                timeout=10
            ),
            
            # Startpage (privacy-focused Google proxy)
            'startpage': SearchSource(
                name='Startpage',
                url_template='https://www.startpage.com/sp/search?query={query}&cat=web&pl=opensearch',
                parser_func='_parse_startpage_html',
                max_results=25,
                timeout=12
            ),
            
            # Archive.org for historical content
            'archive_search': SearchSource(
                name='Internet Archive',
                url_template='https://web.archive.org/web/sitemap/https://example.com?q={query}',
                parser_func='_parse_archive_html',
                max_results=15,
                timeout=15
            ),
            
            # News aggregators that don't block easily
            'news_google': SearchSource(
                name='Google News',
                url_template='https://news.google.com/search?q={query}&hl=en-US&gl=US&ceid=US:en',
                parser_func='_parse_google_news_html',
                max_results=20,
                timeout=10
            ),
        }
        
        # Domain-specific fallbacks for different query types
        self.domain_fallbacks = {
            'news': [
                'https://www.reuters.com/search/news?blob={query}',
                'https://www.bbc.com/search?q={query}',
                'https://www.cnn.com/search?q={query}',
                'https://www.npr.org/search?query={query}',
                'https://www.theverge.com/search?q={query}',
                'https://techcrunch.com/search/{query}',
                'https://www.bloomberg.com/search?query={query}',
                'https://www.wsj.com/search?query={query}'
            ],
            'tech': [
                'https://news.ycombinator.com/search?q={query}',
                'https://www.wired.com/search/?q={query}',
                'https://arstechnica.com/search/?query={query}',
                'https://www.techradar.com/search?searchTerm={query}',
                'https://www.zdnet.com/search/?q={query}',
                'https://stackoverflow.com/search?q={query}',
                'https://github.com/search?q={query}'
            ],
            'finance': [
                'https://finance.yahoo.com/search?p={query}',
                'https://www.marketwatch.com/search?q={query}',
                'https://seekingalpha.com/search?q={query}',
                'https://www.fool.com/search/?q={query}',
                'https://www.cnbc.com/search/?query={query}'
            ],
            'science': [
                'https://www.nature.com/search?q={query}',
                'https://www.sciencedaily.com/search/?keyword={query}',
                'https://www.newscientist.com/search/?q={query}',
                'https://phys.org/search/?search={query}'
            ]
        }
        
    async def discover_urls(self, query: str, max_urls: int = 50, 
                          query_type: str = 'general') -> List[DiscoveredURL]:
        """
        Aggressively discover URLs from multiple sources
        
        Args:
            query: Search query
            max_urls: Target number of URLs to discover
            query_type: Type of query (news, tech, finance, etc.)
            
        Returns:
            List of DiscoveredURL objects
        """
        self.logger.info(f"🚀 Starting aggressive URL discovery for: '{query}' (target: {max_urls})")
        
        all_urls = []
        seen_urls = set()
        
        # Create aiohttp session for parallel requests with better headers
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) Gecko/20100101 Firefox/120.0',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]
        
        headers = {
            'User-Agent': random.choice(user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        timeout = aiohttp.ClientTimeout(total=45)  # Increased timeout
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=3)  # Limit connections
        
        async with aiohttp.ClientSession(timeout=timeout, headers=headers, connector=connector) as session:
            self.session = session
            
            # Phase 1: Parallel search across all active sources
            search_tasks = []
            for source_key, source in self.search_sources.items():
                if source.active:
                    task = self._search_source(source, query, session)
                    search_tasks.append(task)
            
            self.logger.info(f"🔍 Launching {len(search_tasks)} parallel searches...")
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Collect results from parallel searches
            for i, result in enumerate(search_results):
                if isinstance(result, Exception):
                    source_name = list(self.search_sources.keys())[i]
                    self.logger.warning(f"Search source {source_name} failed: {result}")
                    continue
                
                if isinstance(result, list):
                    for url_obj in result:
                        if url_obj.url not in seen_urls:
                            seen_urls.add(url_obj.url)
                            all_urls.append(url_obj)
            
            self.logger.info(f"📊 Phase 1 complete: {len(all_urls)} URLs from search engines")
            
            # Phase 2: Domain-specific fallbacks if we need more URLs
            if len(all_urls) < max_urls:
                needed = max_urls - len(all_urls)
                self.logger.info(f"🎯 Need {needed} more URLs, activating domain fallbacks...")
                
                fallback_urls = await self._get_domain_fallback_urls(query, query_type, needed, session)
                for url_obj in fallback_urls:
                    if url_obj.url not in seen_urls:
                        seen_urls.add(url_obj.url)
                        all_urls.append(url_obj)
            
            # Phase 3: Emergency URL generation if still not enough
            if len(all_urls) < max_urls * 0.6:  # If we have less than 60% of target
                needed = max_urls - len(all_urls)
                self.logger.info(f"⚡ Emergency mode: need {needed} more URLs")
                
                emergency_urls = self._generate_emergency_urls(query, query_type, needed)
                for url_obj in emergency_urls:
                    if url_obj.url not in seen_urls:
                        seen_urls.add(url_obj.url)
                        all_urls.append(url_obj)
        
        # Rank and filter URLs
        ranked_urls = self._rank_and_filter_urls(all_urls, query, max_urls)
        
        self.logger.info(f"🏆 URL discovery complete: {len(ranked_urls)} high-quality URLs")
        return ranked_urls
    
    async def _search_source(self, source: SearchSource, query: str, session: aiohttp.ClientSession) -> List[DiscoveredURL]:
        """Search a specific source and return URLs"""
        try:
            # Format the URL
            search_url = source.url_template.format(query=quote(query))
            
            # Make the request
            async with session.get(search_url, timeout=source.timeout) as response:
                if response.status == 200:
                    content = await response.text()
                    
                    # Parse using the appropriate parser
                    parser_method = getattr(self, source.parser_func, None)
                    if parser_method:
                        urls = parser_method(content, source.name)
                        self.logger.info(f"✅ {source.name}: {len(urls)} URLs")
                        return urls[:source.max_results]
                    else:
                        self.logger.warning(f"Parser {source.parser_func} not found")
                        return []
                else:
                    self.logger.warning(f"{source.name} returned status {response.status}")
                    return []
                    
        except Exception as e:
            self.logger.warning(f"Error searching {source.name}: {e}")
            return []
    
    def _parse_duckduckgo_html(self, html: str, source: str) -> List[DiscoveredURL]:
        """Parse DuckDuckGo HTML results"""
        urls = []
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find result links
            for result in soup.find_all('a', class_='result__a'):
                href = result.get('href', '')
                title = result.get_text(strip=True)
                
                if href and self._is_valid_url(href):
                    urls.append(DiscoveredURL(
                        url=href,
                        title=title,
                        source=source
                    ))
                    
        except Exception as e:
            self.logger.warning(f"Error parsing DuckDuckGo HTML: {e}")
            
        return urls
    
    def _parse_duckduckgo_lite_html(self, html: str, source: str) -> List[DiscoveredURL]:
        """Parse DuckDuckGo Lite HTML results"""
        urls = []
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # DuckDuckGo Lite has simpler structure
            for link in soup.find_all('a', href=True):
                href = link.get('href', '')
                title = link.get_text(strip=True)
                
                # Filter for result links (skip navigation)
                if href and href.startswith('http') and 'duckduckgo.com' not in href:
                    if self._is_valid_url(href):
                        urls.append(DiscoveredURL(
                            url=href,
                            title=title,
                            source=source
                        ))
                        
        except Exception as e:
            self.logger.warning(f"Error parsing DuckDuckGo Lite HTML: {e}")
            
        return urls
    
    def _parse_bing_html(self, html: str, source: str) -> List[DiscoveredURL]:
        """Parse Bing search HTML results"""
        urls = []
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # Bing result containers
            for result in soup.find_all(['h2', 'h3']):
                link = result.find('a', href=True)
                if link:
                    href = link.get('href', '')
                    title = link.get_text(strip=True)
                    
                    if href:
                        # Decode Bing redirect URLs
                        actual_url = self._decode_bing_url(href)
                        if actual_url and self._is_valid_url(actual_url):
                            urls.append(DiscoveredURL(
                                url=actual_url,
                                title=title,
                                source=source
                            ))
                        
        except Exception as e:
            self.logger.warning(f"Error parsing Bing HTML: {e}")
            
        return urls
    
    def _decode_bing_url(self, bing_url: str) -> Optional[str]:
        """Decode Bing redirect URLs to get the actual destination URL"""
        try:
            if 'bing.com/ck/a?' in bing_url:
                # Parse the Bing redirect URL
                from urllib.parse import parse_qs, urlparse
                import base64
                
                parsed = urlparse(bing_url)
                params = parse_qs(parsed.query)
                
                # The 'u' parameter contains the base64-encoded URL
                if 'u' in params:
                    encoded_url = params['u'][0]
                    try:
                        # Decode the base64 URL
                        decoded_bytes = base64.b64decode(encoded_url + '==')  # Add padding if needed
                        decoded_url = decoded_bytes.decode('utf-8')
                        
                        # Remove the 'a1' prefix if present
                        if decoded_url.startswith('a1'):
                            decoded_url = decoded_url[2:]
                        
                        # Ensure it's a valid HTTP/HTTPS URL
                        if decoded_url.startswith(('http://', 'https://')):
                            return decoded_url
                    except Exception:
                        pass
            
            # If it's not a Bing redirect or decoding failed, return the original URL if it's valid
            if bing_url.startswith(('http://', 'https://')) and 'bing.com' not in bing_url:
                return bing_url
                
        except Exception:
            pass
        
        return None
    
    def _parse_yahoo_html(self, html: str, source: str) -> List[DiscoveredURL]:
        """Parse Yahoo search HTML results"""
        urls = []
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # Yahoo result links
            for link in soup.find_all('a', href=True):
                href = link.get('href', '')
                title = link.get_text(strip=True)
                
                # Filter Yahoo result URLs
                if href and href.startswith('http') and 'yahoo.com' not in href:
                    if self._is_valid_url(href):
                        urls.append(DiscoveredURL(
                            url=href,
                            title=title,
                            source=source
                        ))
                        
        except Exception as e:
            self.logger.warning(f"Error parsing Yahoo HTML: {e}")
            
        return urls
    
    def _parse_yandex_html(self, html: str, source: str) -> List[DiscoveredURL]:
        """Parse Yandex search HTML results"""
        urls = []
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # Yandex result structure
            for result in soup.find_all('a', href=True):
                href = result.get('href', '')
                title = result.get_text(strip=True)
                
                # Clean Yandex redirect URLs
                if href and href.startswith('http') and 'yandex.' not in href:
                    if self._is_valid_url(href):
                        urls.append(DiscoveredURL(
                            url=href,
                            title=title,
                            source=source
                        ))
                        
        except Exception as e:
            self.logger.warning(f"Error parsing Yandex HTML: {e}")
            
        return urls
    
    def _parse_startpage_html(self, html: str, source: str) -> List[DiscoveredURL]:
        """Parse Startpage HTML results"""
        urls = []
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # Startpage result links
            for link in soup.find_all('a', class_='w-gl__result-title'):
                href = link.get('href', '')
                title = link.get_text(strip=True)
                
                if href and self._is_valid_url(href):
                    urls.append(DiscoveredURL(
                        url=href,
                        title=title,
                        source=source
                    ))
                    
        except Exception as e:
            self.logger.warning(f"Error parsing Startpage HTML: {e}")
            
        return urls
    
    def _parse_archive_html(self, html: str, source: str) -> List[DiscoveredURL]:
        """Parse Internet Archive HTML results"""
        urls = []
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # Archive.org links
            for link in soup.find_all('a', href=True):
                href = link.get('href', '')
                title = link.get_text(strip=True)
                
                if href and href.startswith('http') and 'archive.org' not in href:
                    if self._is_valid_url(href):
                        urls.append(DiscoveredURL(
                            url=href,
                            title=title,
                            source=source
                        ))
                        
        except Exception as e:
            self.logger.warning(f"Error parsing Archive HTML: {e}")
            
        return urls
    
    def _parse_google_news_html(self, html: str, source: str) -> List[DiscoveredURL]:
        """Parse Google News HTML results"""
        urls = []
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # Google News article links
            for link in soup.find_all('a', href=True):
                href = link.get('href', '')
                title = link.get_text(strip=True)
                
                # Extract actual URL from Google News redirect
                if href and ('news.google.com' in href or href.startswith('./')):
                    # Try to extract the actual article URL
                    if 'url=' in href:
                        import urllib.parse
                        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                        actual_url = parsed.get('url', [None])[0]
                        if actual_url and self._is_valid_url(actual_url):
                            urls.append(DiscoveredURL(
                                url=actual_url,
                                title=title,
                                source=source
                            ))
                            
        except Exception as e:
            self.logger.warning(f"Error parsing Google News HTML: {e}")
            
        return urls

    async def _get_domain_fallback_urls(self, query: str, query_type: str, 
                                      needed: int, session: aiohttp.ClientSession) -> List[DiscoveredURL]:
        """Get URLs from domain-specific fallback sources"""
        urls = []
        
        # Get relevant domain templates
        domain_templates = self.domain_fallbacks.get(query_type, self.domain_fallbacks['news'])
        
        # Try each domain template
        for template in domain_templates[:6]:  # Limit to 6 domains to avoid taking too long
            try:
                search_url = template.format(query=quote(query))
                
                async with session.get(search_url, timeout=15) as response:
                    if response.status == 200:
                        html = await response.text()
                        
                        # Extract URLs from the page
                        page_urls = self._extract_urls_from_html(html, search_url, f"fallback_{query_type}")
                        urls.extend(page_urls[:3])  # Take top 3 from each domain
                        
                        if len(urls) >= needed:
                            break
                            
            except Exception as e:
                self.logger.debug(f"Fallback domain failed: {e}")
                continue
        
        return urls
    
    def _extract_urls_from_html(self, html: str, base_url: str, source: str) -> List[DiscoveredURL]:
        """Extract URLs from HTML content"""
        urls = []
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find all links
            for link in soup.find_all('a', href=True):
                href = link.get('href', '')
                title = link.get_text(strip=True)[:100]  # Limit title length
                
                # Convert relative URLs to absolute
                if href.startswith('/'):
                    href = urljoin(base_url, href)
                
                # Filter for valid external URLs
                if (href.startswith('http') and 
                    self._is_valid_url(href) and 
                    not self._is_social_media_url(href)):
                    
                    urls.append(DiscoveredURL(
                        url=href,
                        title=title,
                        source=source
                    ))
                    
        except Exception as e:
            self.logger.debug(f"Error extracting URLs from HTML: {e}")
            
        return urls
    
    def _generate_emergency_urls(self, query: str, query_type: str, needed: int) -> List[DiscoveredURL]:
        """Generate emergency URLs when all other methods fail"""
        urls = []
        
        # Comprehensive emergency URL patterns - DIRECT CONTENT URLS ONLY
        emergency_patterns = [
            # Top news sources - DIRECT CONTENT PAGES
            'https://www.reuters.com/technology/',
            'https://www.reuters.com/business/',
            'https://www.bbc.com/news/technology',
            'https://www.bbc.com/news/business',
            'https://www.cnn.com/business',
            'https://www.npr.org/sections/technology/',
            'https://www.theguardian.com/technology',
            'https://www.nytimes.com/section/technology',
            'https://www.washingtonpost.com/technology/',
            'https://www.wsj.com/news/technology',
            
            # Tech sources - DIRECT CONTENT PAGES
            'https://techcrunch.com/',
            'https://www.theverge.com/',
            'https://www.wired.com/category/business/',
            'https://arstechnica.com/tech-policy/',
            'https://www.engadget.com/',
            'https://venturebeat.com/',
            'https://www.zdnet.com/topic/innovation/',
            'https://www.cnet.com/news/',
            'https://www.pcmag.com/news',
            'https://mashable.com/tech',
            
            # Science sources - DIRECT CONTENT PAGES
            'https://www.nature.com/news',
            'https://www.sciencedaily.com/news/',
            'https://www.newscientist.com/subject/technology/',
            'https://www.scientificamerican.com/technology/',
            'https://phys.org/technology-news/',
            'https://www.space.com/news',
            
            # Business/Finance - DIRECT CONTENT PAGES
            'https://finance.yahoo.com/news/',
            'https://www.marketwatch.com/latest-news',
            'https://seekingalpha.com/news',
            'https://www.fool.com/investing/',
            'https://www.cnbc.com/technology/',
            'https://fortune.com/section/technology/',
            'https://www.businessinsider.com/sai',
            
            # Tech-specific sources - DIRECT CONTENT PAGES
            'https://www.technologyreview.com/',
            'https://spectrum.ieee.org/',
            'https://www.techrepublic.com/',
            'https://www.computerworld.com/',
            'https://www.infoworld.com/',
            'https://www.networkworld.com/',
            'https://www.informationweek.com/',
            
            # Energy/Climate sources - DIRECT CONTENT PAGES
            'https://www.greentechmedia.com/',
            'https://www.renewableenergyworld.com/',
            'https://www.pv-magazine.com/',
            'https://www.windpowerengineering.com/',
            'https://cleantechnica.com/',
            'https://electrek.co/',
            
            # General tech/innovation - DIRECT CONTENT PAGES
            'https://www.fastcompany.com/technology',
            'https://www.technologyreview.com/topic/artificial-intelligence/',
            'https://hai.stanford.edu/news',
            'https://ai.googleblog.com/',
            'https://openai.com/blog/',
            'https://www.deepmind.com/blog',
        ]
        
        # Query-type specific URL selection
        query_specific_patterns = []
        if query_type == 'tech':
            query_specific_patterns = emergency_patterns[10:30]  # Tech and AI sources
        elif query_type == 'science':
            query_specific_patterns = emergency_patterns[20:26]  # Science sources  
        elif query_type == 'finance':
            query_specific_patterns = emergency_patterns[26:33]  # Business sources
        elif query_type == 'news':
            query_specific_patterns = emergency_patterns[0:15]   # News sources
        elif 'energy' in query.lower() or 'renewable' in query.lower():
            query_specific_patterns = emergency_patterns[34:40]  # Energy sources
        else:
            query_specific_patterns = emergency_patterns[0:20]   # Mix of top sources
        
        # Combine specific + general patterns
        all_patterns = query_specific_patterns + [p for p in emergency_patterns if p not in query_specific_patterns]
        
        # Generate URLs up to needed amount
        for i, pattern in enumerate(all_patterns[:needed]):
            try:
                urls.append(DiscoveredURL(
                    url=pattern,  # Use URL directly, no query formatting needed
                    title=f"Emergency content source {i+1}",
                    source="emergency_generator",
                    ranking=i+1000  # Lower priority
                ))
            except Exception as e:
                self.logger.debug(f"Error creating emergency URL: {e}")
                continue
        
        self.logger.info(f"Generated {len(urls)} emergency content URLs")
        return urls
    
    def _rank_and_filter_urls(self, urls: List[DiscoveredURL], query: str, max_urls: int) -> List[DiscoveredURL]:
        """Rank and filter URLs by relevance and quality"""
        
        # Calculate relevance scores
        query_words = set(query.lower().split())
        
        filtered_candidates = []
        
        for url_obj in urls:
            # First apply validity filter
            if not self._is_valid_url(url_obj.url):
                continue
                
            score = 0.3  # Base score
            
            # Title relevance (higher weight)
            title_words = set(url_obj.title.lower().split())
            title_overlap = len(query_words.intersection(title_words))
            if title_overlap > 0:
                score += min(0.4, title_overlap * 0.15)  # Increased weight
            
            # Description relevance
            desc_words = set(url_obj.description.lower().split())
            desc_overlap = len(query_words.intersection(desc_words))
            if desc_overlap > 0:
                score += min(0.2, desc_overlap * 0.1)
            
            # High-quality domain bonus (significant boost)
            if self._is_high_quality_domain(url_obj.url):
                score += 0.3  # Increased bonus
            
            # Source reliability bonus
            if url_obj.source in ['DuckDuckGo', 'SearX Instance 1', 'SearX Instance 2']:
                score += 0.1
            elif url_obj.source in ['Bing Web Search', 'Internet Archive']:
                score += 0.05
            
            # Penalize generic or low-quality patterns
            url_lower = url_obj.url.lower()
            if any(pattern in url_lower for pattern in ['/search', '/browse', '/index', '/list']):
                score -= 0.2
            
            # Boost for specific content indicators
            if any(indicator in url_obj.title.lower() for indicator in ['article', 'news', 'report', 'analysis', 'study']):
                score += 0.15
                
            # Boost for recent content indicators
            if any(year in url_obj.title.lower() for year in ['2024', '2025']):
                score += 0.1
            
            url_obj.relevance_score = min(1.0, max(0.0, score))
            
            # Only include URLs with minimum score
            if url_obj.relevance_score >= 0.4:
                filtered_candidates.append(url_obj)
        
        # Sort by relevance score
        sorted_urls = sorted(filtered_candidates, key=lambda x: x.relevance_score, reverse=True)
        
        # Remove duplicates while preserving order
        seen = set()
        final_urls = []
        for url_obj in sorted_urls:
            if url_obj.url not in seen:
                seen.add(url_obj.url)
                final_urls.append(url_obj)
        
        return final_urls[:max_urls]
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and useful"""
        if not url or not url.startswith('http'):
            return False
        
        url_lower = url.lower()
        
        # Filter out unwanted URLs and patterns
        unwanted_patterns = [
            'javascript:', 'mailto:', 'tel:', '#',
            '.pdf', '.doc', '.ppt', '.zip', '.exe',
            # Social media and video platforms
            'facebook.com', 'twitter.com', 'instagram.com',
            'youtube.com/watch', 'tiktok.com', 'linkedin.com',
            # Generic search and aggregation pages - MORE AGGRESSIVE
            '/search?', '/search/', '/topics/', '/search=', '?q=', '?query=',
            'search?p=', 'search?text=', 'search/?', '/search/', 'search.php',
            'github.com/topics/', 'github.com/search', 'github.com/explore',
            'reddit.com/search', 'reddit.com/r/search', 'reddit.com/r/all',
            'stackoverflow.com/questions', 'stackoverflow.com/search',
            'stackexchange.com', 'superuser.com', 'serverfault.com',
            # Search engines and search result pages
            'duckduckgo.com/?q=', 'bing.com/search', 'google.com/search',
            'yahoo.com/search', 'yandex.com/search', 'startpage.com/search',
            'searx.', '/searx/', 'metager.org', 'mojeek.com',
            # Generic landing and navigation pages
            'docs.python.org', 'developer.mozilla.org', 'w3schools.com',
            '/latest', '/browse', '/directory', '/sitemap', '/index.php',
            '/category/', '/categories/', '/tag/', '/tags/', '/archive/',
            # Ad and tracking URLs
            'doubleclick.net', 'googleadservices.com', 'googlesyndication.com',
            'amazon.com/dp/', 'amazon.com/gp/', 'affiliate', '/ref=',
            # Generic news search pages (not specific articles)
            'news.google.com', 'news.yahoo.com', 'news.bing.com',
            # Wikipedia disambiguation and search pages
            'wikipedia.org/wiki/Category:', 'wikipedia.org/w/index.php',
            # Forum/discussion aggregators
            'quora.com/search', 'medium.com/search', 'medium.com/tag/',
            # Academic search engines (not specific papers)
            'scholar.google.com', 'arxiv.org/search', 'researchgate.net/search',
            'pubmed.ncbi.nlm.nih.gov/?term=', 'jstor.org/action/doBasicSearch',
        ]
        
        # Check for unwanted patterns
        if any(pattern in url_lower for pattern in unwanted_patterns):
            return False
        
        # Additional checks for low-quality URLs
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()
        query_params = parsed.query.lower()
        
        # Filter out root domains and very short paths (unless from quality domains)
        if path in ['', '/', '/home', '/index', '/main', '/default']:
            # Allow root pages only from high-quality domains
            if not self._is_high_quality_domain(url):
                return False
            
        # Filter out URLs that are clearly search results or listings
        if any(keyword in path for keyword in ['/list', '/index', '/browse', '/directory', '/search']):
            return False
            
        # Filter out URLs with search-like query parameters
        if any(param in query_params for param in ['q=', 'query=', 'search=', 'keyword=', 's=']):
            return False
            
        return True
    
    def _is_social_media_url(self, url: str) -> bool:
        """Check if URL is from social media"""
        social_domains = [
            'facebook.com', 'twitter.com', 'instagram.com',
            'linkedin.com', 'tiktok.com', 'snapchat.com'
        ]
        return any(domain in url.lower() for domain in social_domains)
    
    def _is_high_quality_domain(self, url: str) -> bool:
        """Check if URL is from a high-quality domain"""
        try:
            domain = urlparse(url).netloc.lower()
            
            # High-quality news and content domains
            quality_domains = [
                # Major news outlets
                'reuters.com', 'bbc.com', 'cnn.com', 'npr.org',
                'nytimes.com', 'washingtonpost.com', 'guardian.com',
                'wsj.com', 'bloomberg.com', 'forbes.com',
                'economist.com', 'ft.com', 'reuters.co.uk',
                
                # Technology and science
                'techcrunch.com', 'theverge.com', 'wired.com',
                'arstechnica.com', 'engadget.com', 'gizmodo.com',
                'zdnet.com', 'computerworld.com', 'infoworld.com',
                'technologyreview.com', 'spectrum.ieee.org',
                
                # Science and research
                'nature.com', 'sciencedaily.com', 'newscientist.com',
                'sciencemag.org', 'phys.org', 'livescience.com',
                'scientificamerican.com', 'nationalgeographic.com',
                
                # Business and finance
                'harvard.edu', 'mit.edu', 'stanford.edu',
                'businessinsider.com', 'fastcompany.com',
                'mckinsey.com', 'pwc.com', 'deloitte.com',
                
                # Specialized publications
                'venturebeat.com', 'mashable.com', 'readwrite.com',
                'thenextweb.com', 'digitaltrends.com',
                'techradar.com', 'pcmag.com', 'cnet.com'
            ]
            
            return any(quality_domain in domain for quality_domain in quality_domains)
            
        except:
            return False
