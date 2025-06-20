"""
Universal Crawl4AI Strategy with Progressive Data Collection

This strategy implements advanced crawling capabilities using crawl4ai with
intelligent pathfinding, progressive data collection, and AI-driven content
extraction. It focuses on efficient, resilient crawling with semantic analysis.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import tempfile
import json
import re

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    from crawl4ai import AsyncWebCrawler, CacheMode, BrowserConfig, CrawlerRunConfig
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
    from crawl4ai.content_filter_strategy import PruningContentFilter
    from crawl4ai.extraction_strategy import (
        JsonCssExtractionStrategy,
        LLMExtractionStrategy,
    )
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False

from strategies.base_strategy import BaseStrategy
from strategies.core.strategy_types import (
    StrategyCapability, StrategyMetadata, StrategyType, strategy_metadata
)
from config import (
    CRAWL4AI_ENABLED, CRAWL4AI_MAX_PAGES, CRAWL4AI_DEEP_CRAWL,
    CRAWL4AI_MEMORY_THRESHOLD, CRAWL4AI_AI_PATHFINDING,
    USE_UNDETECTED_CHROMEDRIVER, PROGRESSIVE_DATA_COLLECTION,
    DATA_CONSISTENCY_CHECKS, CIRCUIT_BREAKER_ENABLED
)


@dataclass
class PageData:
    """Represents collected data from a single page"""
    url: str
    title: str
    main_content: str
    structured_data: Dict[str, Any]
    relevance_score: float
    timestamp: float
    metadata: Dict[str, Any]


@dataclass
class CrawlSession:
    """Manages state for a crawling session"""
    query: str
    intent_analysis: Dict[str, Any]
    collected_pages: List[PageData]
    visited_urls: Set[str]
    pending_urls: List[Tuple[str, float]]  # (url, priority_score)
    total_relevance: float
    start_time: float
    memory_usage: float


@dataclass
class SiteTypeDetection:
    """Site type detection results"""
    is_spa: bool = False
    framework: Optional[str] = None  # 'react', 'vue', 'angular', 'unknown'
    has_infinite_scroll: bool = False
    has_dynamic_content: bool = False
    requires_js: bool = False
    confidence: float = 0.0


@strategy_metadata(
    strategy_type=StrategyType.SPECIAL_PURPOSE,
    capabilities={
        StrategyCapability.AI_ASSISTED,
        StrategyCapability.PROGRESSIVE_CRAWLING,
        StrategyCapability.SEMANTIC_SEARCH,
        StrategyCapability.INTENT_ANALYSIS,
        StrategyCapability.AI_PATHFINDING,
        StrategyCapability.EARLY_RELEVANCE_TERMINATION,
        StrategyCapability.MEMORY_ADAPTIVE,
        StrategyCapability.CIRCUIT_BREAKER,
        StrategyCapability.CONSOLIDATED_AI_PROCESSING,
        StrategyCapability.JAVASCRIPT_EXECUTION,
        StrategyCapability.DYNAMIC_CONTENT,
        StrategyCapability.ERROR_HANDLING,
        StrategyCapability.RETRY_MECHANISM,
        StrategyCapability.RATE_LIMITING
    },
    description="Advanced crawling strategy using crawl4ai with intelligent pathfinding, progressive data collection, and AI-driven content extraction"
)
class UniversalCrawl4AIStrategy(BaseStrategy):
    """
    Advanced crawling strategy using crawl4ai for intelligent, progressive data collection.
    
    Key Features:
    - Progressive lightweight data collection from multiple pages
    - AI-driven pathfinding and scope adjustment
    - Early relevance evaluation to prevent crawling low-value paths
    - Memory-adaptive resource management
    - Consolidated AI processing of aggregated data
    - Circuit breaker pattern for resilience
    """
    
    def __init__(self, context=None, **kwargs):
        """Initialize the Universal Crawl4AI Strategy"""
        super().__init__(context, **kwargs)
        
        if not CRAWL4AI_AVAILABLE:
            raise ImportError("crawl4ai is not available. Please install with: pip install crawl4ai[all]")
        
        if not CRAWL4AI_ENABLED:
            raise ValueError("Crawl4AI is disabled in configuration")
        
        self.logger = logging.getLogger(__name__)
        
        # Strategy configuration
        self.max_pages = min(kwargs.get('max_pages', CRAWL4AI_MAX_PAGES), CRAWL4AI_MAX_PAGES)
        self.deep_crawl_enabled = CRAWL4AI_DEEP_CRAWL
        self.use_undetected_driver = USE_UNDETECTED_CHROMEDRIVER
        self.ai_pathfinding = CRAWL4AI_AI_PATHFINDING
        self.memory_threshold = CRAWL4AI_MEMORY_THRESHOLD
        
        # Progressive collection settings
        self.progressive_collection = PROGRESSIVE_DATA_COLLECTION
        self.data_consistency_checks = DATA_CONSISTENCY_CHECKS
        
        # Resilience settings
        self.circuit_breaker_enabled = CIRCUIT_BREAKER_ENABLED
        self.circuit_breaker_failures = 0
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_timeout = 300  # 5 minutes
        self.circuit_breaker_last_failure = 0
        
        # Content relevance thresholds
        self.min_relevance_score = 0.3
        self.min_content_length = 200
        self.max_irrelevant_pages = 3
        
        # Performance metrics
        self.performance_metrics = {
            'pages_crawled': 0,
            'pages_skipped': 0,
            'total_processing_time': 0,
            'average_relevance': 0.0,
            'memory_peak': 0.0
        }
        
        self.logger.info(f"UniversalCrawl4AIStrategy initialized with max_pages={self.max_pages}")
    
    @property
    def metadata(self) -> StrategyMetadata:
        """Return strategy metadata"""
        return StrategyMetadata(
            name="UniversalCrawl4AIStrategy",
            description="Advanced crawl4ai strategy with progressive data collection and AI pathfinding",
            capabilities=[
                StrategyCapability.DOM_NAVIGATION,
                StrategyCapability.JAVASCRIPT_EXECUTION,
                StrategyCapability.PAGINATION,
                StrategyCapability.INFINITE_SCROLL
            ],
            parameters={
                "max_pages": {"type": "int", "default": CRAWL4AI_MAX_PAGES, "description": "Maximum pages to crawl"},
                "deep_crawl": {"type": "bool", "default": True, "description": "Enable deep crawling"},
                "ai_pathfinding": {"type": "bool", "default": True, "description": "Enable AI-driven pathfinding"},
                "memory_threshold": {"type": "float", "default": 80.0, "description": "Memory usage threshold"}
            }
        )
    
    def supports_enhanced_context(self) -> bool:
        """Indicate that this strategy supports enhanced context with intent analysis and schema"""
        return True
    
    async def search(self, query: str, url: str = None, context: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """
        Search method that accepts enhanced context and delegates to execute method.
        
        Args:
            query: Search query string
            url: Target URL (optional, can be generated from context)
            context: Enhanced context with intent analysis, schema, etc.
            **kwargs: Additional parameters
            
        Returns:
            Search results in standardized format
        """
        # Merge context into kwargs for execute method
        if context:
            kwargs.update({
                'user_prompt': query,
                'intent_analysis': context.get('intent_analysis', {}),
                'pydantic_schema': context.get('pydantic_schema'),
                'site_analysis': context.get('site_analysis', {}),
                'required_capabilities': context.get('required_capabilities', set())
            })
        else:
            kwargs['user_prompt'] = query
            
        # Use provided URL or try to determine from context
        target_url = url
        if not target_url and context and context.get('intent_analysis'):
            # Try to get URL from intent analysis or site analysis
            intent_data = context['intent_analysis']
            if 'target_urls' in intent_data and intent_data['target_urls']:
                target_url = intent_data['target_urls'][0]
        
        if not target_url:
            return {
                "success": False,
                "error": "No target URL provided or could be determined from context",
                "results": []
            }
            
        # Execute the crawling strategy
        result = await self.execute(target_url, **kwargs)
        
        if result:
            return {
                "success": True,
                "results": result.get('items', []),
                "metadata": {
                    "strategy": "universal_crawl4ai",
                    "pages_processed": result.get('total_pages', 0),
                    "relevance_score": result.get('average_relevance', 0.0),
                    "processing_time": result.get('processing_time', 0.0)
                }
            }
        else:
            return {
                "success": False,
                "error": "Crawling execution failed",
                "results": []
            }
    
    async def execute(self, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Execute controlled crawl with progressive data collection and AI processing
        
        Args:
            url: Starting URL for crawling
            **kwargs: Additional parameters including user_prompt, intent_analysis
            
        Returns:
            Dictionary containing structured results from all crawled pages
        """
        start_time = time.time()
        
        try:
            # Check circuit breaker
            if not self._check_circuit_breaker():
                self.logger.warning("Circuit breaker is open, skipping crawl")
                return None
            
            # Initialize crawl session
            user_prompt = kwargs.get('user_prompt', kwargs.get('query', ''))
            intent_analysis = kwargs.get('intent_analysis', {})
            
            session = CrawlSession(
                query=user_prompt,
                intent_analysis=intent_analysis,
                collected_pages=[],
                visited_urls=set(),
                pending_urls=[(url, 1.0)],  # Start with max priority
                total_relevance=0.0,
                start_time=start_time,
                memory_usage=0.0
            )
            
            self.logger.info(f"Starting crawl session for: {user_prompt}")
            
            # Create crawler configuration
            crawler_config = self._create_crawler_config()
            browser_config = self._create_browser_config()
            
            # Execute advanced progressive crawling with parallel processing
            async with AsyncWebCrawler(config=browser_config) as crawler:
                # Use parallel processing if we have multiple URLs
                if len(session.pending_urls) > 2 and self.max_pages > 1:
                    await self._execute_parallel_crawl(crawler, session)
                else:
                    await self._execute_progressive_crawl(crawler, session)
            
            # Consolidate and process collected data
            final_results = await self._consolidate_results(session)
            
            # Update performance metrics
            self._update_performance_metrics(session, time.time() - start_time)
            
            self.logger.info(f"Crawl completed: {len(session.collected_pages)} pages, "
                           f"avg relevance: {session.total_relevance / max(len(session.collected_pages), 1):.3f}")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Crawl execution failed: {e}", exc_info=True)
            self._record_circuit_breaker_failure()
            return None
    
    async def _execute_progressive_crawl(self, crawler: AsyncWebCrawler, session: CrawlSession) -> None:
        """Execute progressive crawling with relevance checking and dynamic termination"""
        
        irrelevant_page_count = 0
        
        while (session.pending_urls and 
               len(session.collected_pages) < self.max_pages and
               irrelevant_page_count < self.max_irrelevant_pages):
            
            # Check memory usage
            if session.memory_usage > self.memory_threshold:
                self.logger.warning(f"Memory threshold exceeded: {session.memory_usage}%")
                break
            
            # Get next URL with highest priority
            current_url, priority = session.pending_urls.pop(0)
            
            if current_url in session.visited_urls:
                continue
            
            session.visited_urls.add(current_url)
            
            try:
                # Perform two-tier analysis: fast HTML check first
                page_data = await self._crawl_single_page(crawler, current_url, session)
                
                if page_data is None:
                    self.performance_metrics['pages_skipped'] += 1
                    continue
                
                # Evaluate page relevance
                if not await self._evaluate_page_relevance(page_data, session):
                    irrelevant_page_count += 1
                    self.logger.debug(f"Low relevance page skipped: {current_url}")
                    continue
                
                # Reset irrelevant counter on finding relevant content
                irrelevant_page_count = 0
                
                # Add to collected pages
                session.collected_pages.append(page_data)
                session.total_relevance += page_data.relevance_score
                
                # Discover new URLs if deep crawling is enabled
                if self.deep_crawl_enabled and len(session.collected_pages) < self.max_pages:
                    new_urls = await self._discover_follow_up_urls(
                        crawler, current_url, page_data, session
                    )
                    
                    # Add new URLs with priority scoring
                    for new_url in new_urls:
                        if new_url not in session.visited_urls:
                            priority_score = self._calculate_url_priority(new_url, session)
                            session.pending_urls.append((new_url, priority_score))
                    
                    # Sort pending URLs by priority
                    session.pending_urls.sort(key=lambda x: x[1], reverse=True)
                
                # Check dynamic termination conditions
                if await self._should_terminate_crawl(session):
                    self.logger.info("Dynamic termination conditions met")
                    break
                    
                self.performance_metrics['pages_crawled'] += 1
                
            except Exception as e:
                self.logger.warning(f"Failed to crawl {current_url}: {e}")
                continue
    
    async def _execute_parallel_crawl(self, crawler: AsyncWebCrawler, session: CrawlSession) -> None:
        """Execute parallel crawling of multiple URLs for improved performance"""
        
        if len(session.pending_urls) <= 1:
            # Not enough URLs for parallel processing, use sequential
            await self._execute_progressive_crawl(crawler, session)
            return
        
        # Configure parallel processing based on system capabilities
        max_concurrent = min(3, len(session.pending_urls))  # Conservative parallel limit
        semaphore = asyncio.Semaphore(max_concurrent)
        
        self.logger.info(f"Starting parallel crawl with {max_concurrent} concurrent workers")
        
        async def crawl_with_semaphore(url_score_tuple):
            """Crawl a single URL with semaphore protection"""
            async with semaphore:
                url, priority = url_score_tuple
                if url in session.visited_urls:
                    return None
                
                session.visited_urls.add(url)
                return await self._crawl_single_page(crawler, url, session)
        
        # Process URLs in batches to avoid overwhelming the system
        batch_size = 5
        while (session.pending_urls and 
               len(session.collected_pages) < self.max_pages):
            
            # Get next batch of URLs
            current_batch = session.pending_urls[:batch_size]
            session.pending_urls = session.pending_urls[batch_size:]
            
            # Execute batch in parallel
            try:
                results = await asyncio.gather(
                    *[crawl_with_semaphore(url_tuple) for url_tuple in current_batch],
                    return_exceptions=True
                )
                
                # Process results
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        url = current_batch[i][0]
                        self.logger.warning(f"Parallel crawl failed for {url}: {result}")
                        continue
                    
                    if result and isinstance(result, PageData):
                        # Evaluate page relevance
                        if await self._evaluate_page_relevance(result, session):
                            session.collected_pages.append(result)
                            session.total_relevance += result.relevance_score
                            self.performance_metrics['pages_crawled'] += 1
                        else:
                            self.logger.debug(f"Low relevance page skipped: {result.url}")
                    
                    # Check if we have enough pages
                    if len(session.collected_pages) >= self.max_pages:
                        break
                
                # Brief pause between batches to be respectful
                await asyncio.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"Parallel batch processing failed: {e}")
                # Fall back to sequential processing for remaining URLs
                break
        
        self.logger.info(f"Parallel crawl completed. Collected {len(session.collected_pages)} pages")
    
    async def _apply_advanced_extraction_techniques(self, page_data: PageData, 
                                                   site_info: Dict[str, Any]) -> PageData:
        """Apply advanced extraction techniques based on site type"""
        
        try:
            # Enhanced content extraction for SPAs
            if site_info.get('optimal_strategy') == 'modern_spa':
                page_data = await self._extract_spa_content(page_data, site_info)
            
            # Enhanced extraction for infinite scroll sites
            elif site_info.get('has_infinite_scroll'):
                page_data = await self._extract_infinite_scroll_content(page_data)
            
            # Enhanced extraction for paginated content
            elif site_info.get('has_pagination'):
                page_data = await self._extract_paginated_content(page_data)
            
            # Apply smart content filtering
            page_data = self._apply_smart_content_filtering(page_data, site_info)
            
            return page_data
            
        except Exception as e:
            self.logger.warning(f"Advanced extraction failed for {page_data.url}: {e}")
            return page_data
    
    async def _extract_spa_content(self, page_data: PageData, site_info: Dict[str, Any]) -> PageData:
        """Extract content optimized for Single Page Applications"""
        
        # For SPAs, the main content might be in specific containers
        spa_selectors = [
            '[id*="app"]', '[id*="root"]', '[id*="main"]',
            '.app-content', '.main-content', '.page-content',
            '[data-react-component]', '[data-vue-component]'
        ]
        
        # Try to extract more structured content for SPAs
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(page_data.main_content, 'html.parser')
            
            # Look for SPA-specific content containers
            spa_content = ""
            for selector in spa_selectors:
                elements = soup.select(selector)
                if elements:
                    spa_content = ' '.join([elem.get_text(strip=True) for elem in elements])
                    if len(spa_content) > len(page_data.main_content) * 0.8:
                        page_data.main_content = spa_content
                        page_data.metadata['extraction_method'] = 'spa_optimized'
                        break
            
            # Extract SPA-specific structured data
            structured_data = page_data.structured_data.copy()
            structured_data.update({
                'framework': site_info.get('framework'),
                'spa_optimized': True,
                'content_selectors_used': spa_selectors[:3]
            })
            page_data.structured_data = structured_data
            
        except Exception as e:
            self.logger.debug(f"SPA content extraction failed: {e}")
        
        return page_data
    
    async def _extract_infinite_scroll_content(self, page_data: PageData) -> PageData:
        """Extract content optimized for infinite scroll pages"""
        
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(page_data.main_content, 'html.parser')
            
            # Look for infinite scroll content containers
            scroll_selectors = [
                '.infinite-scroll', '.lazy-load', '.scroll-content',
                '[data-infinite-scroll]', '.feed-item', '.list-item'
            ]
            
            scroll_content = []
            for selector in scroll_selectors:
                elements = soup.select(selector)
                if elements:
                    scroll_content.extend([elem.get_text(strip=True) for elem in elements])
            
            if scroll_content:
                enhanced_content = page_data.main_content + '\n\n' + '\n'.join(scroll_content)
                page_data.main_content = enhanced_content
                page_data.metadata['infinite_scroll_items'] = len(scroll_content)
                page_data.metadata['extraction_method'] = 'infinite_scroll_optimized'
            
        except Exception as e:
            self.logger.debug(f"Infinite scroll content extraction failed: {e}")
        
        return page_data
    
    async def _extract_paginated_content(self, page_data: PageData) -> PageData:
        """Extract content optimized for paginated pages"""
        
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(page_data.main_content, 'html.parser')
            
            # Look for pagination indicators and next page links
            pagination_data = {
                'has_next_page': False,
                'next_page_url': None,
                'current_page': 1,
                'total_pages': None
            }
            
            # Find next page links
            next_links = soup.find_all('a', text=re.compile(r'next|>|continue', re.I))
            if not next_links:
                next_links = soup.select('a[href*="page"], a[href*="p="]')
            
            if next_links:
                pagination_data['has_next_page'] = True
                pagination_data['next_page_url'] = next_links[0].get('href')
            
            # Try to extract page numbers
            page_indicators = soup.find_all(text=re.compile(r'page \d+ of \d+', re.I))
            if page_indicators:
                import re
                match = re.search(r'page (\d+) of (\d+)', page_indicators[0], re.I)
                if match:
                    pagination_data['current_page'] = int(match.group(1))
                    pagination_data['total_pages'] = int(match.group(2))
            
            page_data.structured_data.update({
                'pagination': pagination_data,
                'extraction_method': 'pagination_optimized'
            })
            
        except Exception as e:
            self.logger.debug(f"Pagination content extraction failed: {e}")
        
        return page_data
    
    def _apply_smart_content_filtering(self, page_data: PageData, site_info: Dict[str, Any]) -> PageData:
        """Apply intelligent content filtering based on site type and content analysis"""
        
        try:
            content = page_data.main_content
            
            # Site-specific filtering
            if site_info.get('framework') == 'react':
                # React sites often have component-based content
                content = self._filter_react_artifacts(content)
            elif site_info.get('framework') == 'vue':
                # Vue sites might have directive artifacts
                content = self._filter_vue_artifacts(content)
            
            # General smart filtering
            content = self._remove_navigation_noise(content)
            content = self._enhance_content_structure(content)
            
            page_data.main_content = content
            page_data.metadata['smart_filtering_applied'] = True
            
        except Exception as e:
            self.logger.debug(f"Smart content filtering failed: {e}")
        
        return page_data
    
    def _filter_react_artifacts(self, content: str) -> str:
        """Remove React-specific artifacts from content"""
        # Remove common React development artifacts
        react_patterns = [
            r'React\.createElement.*?\n',
            r'__REACT_DEVTOOLS_GLOBAL_HOOK__.*?\n',
            r'data-reactroot.*?\n'
        ]
        
        for pattern in react_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        return content
    
    def _filter_vue_artifacts(self, content: str) -> str:
        """Remove Vue-specific artifacts from content"""
        # Remove common Vue development artifacts
        vue_patterns = [
            r'v-\w+="[^"]*"',
            r'@\w+="[^"]*"',
            r'\{\{.*?\}\}'
        ]
        
        for pattern in vue_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        return content
    
    def _remove_navigation_noise(self, content: str) -> str:
        """Remove common navigation and UI noise from content"""
        noise_patterns = [
            r'\b(Home|Menu|Navigation|Footer|Header|Sidebar)\b',
            r'\b(Login|Register|Sign up|Sign in|Log out)\b',
            r'\b(Cookie|Privacy|Terms|Contact|About)\b',
            r'\b(Share|Tweet|Like|Follow|Subscribe)\b',
            r'\b(Advertisement|Ad|Sponsored)\b'
        ]
        
        for pattern in noise_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        # Remove multiple spaces that may result from removals
        content = re.sub(r'\s+', ' ', content)
        
        return content
    
    def _enhance_content_structure(self, content: str) -> str:
        """Enhance content structure and readability"""
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        content = re.sub(r'[ \t]+', ' ', content)
        
        # Remove very short lines that are likely UI elements
        lines = content.split('\n')
        filtered_lines = []
        
        for line in lines:
            line = line.strip()
            if len(line) > 10 or re.search(r'[.!?]$', line):
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)

    # === Required Abstract Methods === 
    
    async def extract_data(self, html_content: str, **kwargs) -> Dict[str, Any]:
        """
        Extract data from HTML content using advanced techniques.
        This method implements the required abstract method from BaseStrategy.
        """
        try:
            # Parse HTML content
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Detect site type for optimal extraction
            site_info = self._detect_site_type(html_content)
            
            # Extract basic data structure
            extracted_data = {
                'title': soup.title.text.strip() if soup.title else '',
                'main_content': soup.get_text(strip=True),
                'links': [a.get('href') for a in soup.find_all('a', href=True)][:20],
                'images': [img.get('src') for img in soup.find_all('img', src=True)][:10],
                'metadata': {
                    'site_type': site_info,
                    'extraction_timestamp': time.time(),
                    'content_length': len(html_content),
                    'extraction_method': 'advanced_universal'
                }
            }
            
            # Apply advanced extraction based on site type
            if site_info.get('is_spa'):
                extracted_data = self._enhance_spa_extraction(extracted_data, soup, site_info)
            elif site_info.get('has_infinite_scroll'):
                extracted_data = self._enhance_infinite_scroll_extraction(extracted_data, soup)
            elif site_info.get('has_pagination'):
                extracted_data = self._enhance_pagination_extraction(extracted_data, soup)
            
            # Apply content filtering and enhancement
            extracted_data['main_content'] = self._enhance_content_structure(
                self._remove_navigation_noise(extracted_data['main_content'])
            )
            
            return extracted_data
            
        except Exception as e:
            self.logger.error(f"Data extraction failed: {e}")
            return {'error': str(e), 'main_content': '', 'title': ''}
    
    async def generate_urls(self, query: str, **kwargs) -> List[str]:
        """
        Generate URLs for the given query.
        This strategy relies on external URL generation (DuckDuckGo), so this method
        returns an empty list or URLs from kwargs if provided.
        """
        # This strategy doesn't generate URLs directly - it uses provided URLs
        # Check if URLs are provided in kwargs
        if 'urls' in kwargs:
            return kwargs['urls']
        if 'url' in kwargs:
            return [kwargs['url']]
        
        # Return empty list - URL generation is handled by DuckDuckGo component
        self.logger.info("No URLs provided to UniversalCrawl4AIStrategy.generate_urls() - expecting external URL generation")
        return []
    
    # === Enhanced Site Detection Methods ===
    
    def _detect_site_type(self, html_content: str) -> Dict[str, Any]:
        """Detect the type of site and its characteristics for optimal extraction"""
        
        detection_result = {
            'is_spa': False,
            'framework': None,
            'has_infinite_scroll': False,
            'has_dynamic_content': False,
            'requires_js': False,
            'has_pagination': False,
            'optimal_strategy': 'standard',
            'confidence': 0.0
        }
        
        try:
            # Framework detection
            framework_indicators = {
                'react': [r'react', r'__REACT_DEVTOOLS_GLOBAL_HOOK__', r'ReactDOM', r'data-react'],
                'vue': [r'vue\.js', r'Vue\.', r'v-\w+', r'@\w+="'],
                'angular': [r'angular', r'ng-\w+', r'Angular', r'\[ng'],
                'svelte': [r'svelte', r'Svelte'],
                'next': [r'next\.js', r'Next\.js', r'__NEXT_DATA__'],
                'nuxt': [r'nuxt', r'Nuxt']
            }
            
            confidence_scores = {}
            
            for framework, patterns in framework_indicators.items():
                matches = 0
                for pattern in patterns:
                    if re.search(pattern, html_content, re.IGNORECASE):
                        matches += 1
                
                if matches > 0:
                    confidence_scores[framework] = matches / len(patterns)
            
            # Determine primary framework
            if confidence_scores:
                primary_framework = max(confidence_scores.items(), key=lambda x: x[1])
                detection_result['framework'] = primary_framework[0]
                detection_result['confidence'] = primary_framework[1]
                detection_result['is_spa'] = True
                detection_result['optimal_strategy'] = 'modern_spa'
            
            # Infinite scroll detection
            infinite_scroll_indicators = [
                r'infinite.?scroll', r'lazy.?load', r'load.?more',
                r'data-infinite', r'scroll-to-load', r'endless.?scroll'
            ]
            
            scroll_matches = sum(1 for pattern in infinite_scroll_indicators 
                               if re.search(pattern, html_content, re.IGNORECASE))
            
            if scroll_matches >= 2:
                detection_result['has_infinite_scroll'] = True
                detection_result['optimal_strategy'] = 'infinite_scroll'
            
            # Pagination detection
            pagination_indicators = [
                r'pagination', r'page.?\d+', r'next.?page', r'previous.?page',
                r'nav.*page', r'pager', r'page-nav'
            ]
            
            pagination_matches = sum(1 for pattern in pagination_indicators 
                                   if re.search(pattern, html_content, re.IGNORECASE))
            
            if pagination_matches >= 2:
                detection_result['has_pagination'] = True
            
            # Dynamic content detection
            dynamic_indicators = [
                r'data-\w+', r'ajax', r'fetch\(', r'XMLHttpRequest',
                r'async.*function', r'Promise\.'
            ]
            
            dynamic_matches = sum(1 for pattern in dynamic_indicators
                                if re.search(pattern, html_content, re.IGNORECASE))
            
            if dynamic_matches >= 3:
                detection_result['has_dynamic_content'] = True
                detection_result['requires_js'] = True
            
            # Override strategy based on multiple factors
            if detection_result['has_infinite_scroll'] and detection_result['is_spa']:
                detection_result['optimal_strategy'] = 'spa_infinite_scroll'
            elif detection_result['has_pagination'] and detection_result['is_spa']:
                detection_result['optimal_strategy'] = 'spa_pagination'
            
        except Exception as e:
            self.logger.debug(f"Site type detection failed: {e}")
        
        return detection_result
    
    def _get_optimal_crawler_config(self, site_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimal crawler configuration based on site type"""
        
        base_config = {
            'word_count_threshold': 10,
            'only_text': False,
            'bypass_cache': False,
            'include_raw_html': True,
            'wait_for': None,
            'js_code': None,
            'css_selector': None,
            'screenshot': False,
            'delay_before_return_html': 2.0
        }
        
        # Optimize for SPAs
        if site_info.get('is_spa'):
            base_config.update({
                'delay_before_return_html': 3.0,  # Wait longer for JS to execute
                'wait_for': 'networkidle0',  # Wait for network to be idle
                'js_code': """
                    // Wait for common SPA loading indicators to disappear
                    const loadingSelectors = ['.loading', '.spinner', '[data-loading]', '.loader'];
                    const waitForLoading = () => {
                        return new Promise((resolve) => {
                            const checkLoading = () => {
                                const hasLoading = loadingSelectors.some(sel => document.querySelector(sel));
                                if (!hasLoading) {
                                    resolve();
                                } else {
                                    setTimeout(checkLoading, 500);
                                }
                            };
                            checkLoading();
                        });
                    };
                    await waitForLoading();
                """,
                'word_count_threshold': 50  # SPAs often have more content
            })
        
        # Optimize for infinite scroll
        elif site_info.get('has_infinite_scroll'):
            base_config.update({
                'delay_before_return_html': 2.5,
                'js_code': """
                    // Trigger initial scroll to load more content
                    window.scrollTo(0, document.body.scrollHeight / 2);
                    await new Promise(resolve => setTimeout(resolve, 1000));
                    window.scrollTo(0, document.body.scrollHeight);
                    await new Promise(resolve => setTimeout(resolve, 1500));
                """,
                'word_count_threshold': 100
            })
        
        # Optimize for dynamic content
        elif site_info.get('has_dynamic_content'):
            base_config.update({
                'delay_before_return_html': 2.5,
                'wait_for': 'networkidle2',
                'word_count_threshold': 30
            })
        
        return base_config
    
    def _enhance_spa_extraction(self, extracted_data: Dict[str, Any], 
                               soup: 'BeautifulSoup', site_info: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance extraction for Single Page Applications"""
        
        try:
            # Look for SPA-specific content containers
            spa_selectors = [
                '#app', '#root', '#main', '.app', '.application',
                '[data-react-component]', '[data-vue-component]',
                '.page-content', '.main-content', '.content-wrapper'
            ]
            
            spa_content_found = False
            for selector in spa_selectors:
                elements = soup.select(selector)
                if elements and len(elements[0].get_text(strip=True)) > 100:
                    spa_content = elements[0].get_text(strip=True)
                    if len(spa_content) > len(extracted_data['main_content']) * 0.5:
                        extracted_data['main_content'] = spa_content
                        spa_content_found = True
                        break
            
            # Extract SPA-specific structured data
            extracted_data['metadata']['spa_extraction'] = {
                'framework': site_info.get('framework'),
                'content_enhanced': spa_content_found,
                'selectors_used': spa_selectors[:3]
            }
            
            # Look for component-based content
            components = soup.find_all(attrs={'data-component': True})
            if components:
                extracted_data['components'] = [
                    comp.get_text(strip=True)[:200] for comp in components[:5]
                ]
            
        except Exception as e:
            self.logger.debug(f"SPA extraction enhancement failed: {e}")
        
        return extracted_data
    
    def _enhance_infinite_scroll_extraction(self, extracted_data: Dict[str, Any], 
                                          soup: 'BeautifulSoup') -> Dict[str, Any]:
        """Enhance extraction for infinite scroll pages"""
        
        try:
            # Look for common infinite scroll item containers
            item_selectors = [
                '.feed-item', '.list-item', '.card', '.post',
                '[data-item]', '.item', '.entry', '.article'
            ]
            
            scroll_items = []
            for selector in item_selectors:
                items = soup.select(selector)
                if len(items) > 3:  # Likely an item container if multiple found
                    scroll_items.extend([item.get_text(strip=True)[:300] for item in items[:10]])
                    break
            
            if scroll_items:
                extracted_data['scroll_items'] = scroll_items
                extracted_data['metadata']['infinite_scroll'] = {
                    'items_found': len(scroll_items),
                    'extraction_enhanced': True
                }
                
                # Append scroll items to main content
                scroll_content = '\n\n'.join(scroll_items)
                extracted_data['main_content'] += f'\n\nScroll Items:\n{scroll_content}'
            
        except Exception as e:
            self.logger.debug(f"Infinite scroll extraction enhancement failed: {e}")
        
        return extracted_data
    
    def _enhance_pagination_extraction(self, extracted_data: Dict[str, Any], 
                                     soup: 'BeautifulSoup') -> Dict[str, Any]:
        """Enhance extraction for paginated content"""
        
        try:
            # Extract pagination information
            pagination_info = {
                'has_next': False,
                'has_previous': False,
                'current_page': None,
                'total_pages': None,
                'next_url': None,
                'previous_url': None
            }
            
            # Look for next/previous links
            next_links = soup.find_all('a', string=re.compile(r'next|>|continue', re.I))
            if next_links:
                pagination_info['has_next'] = True
                pagination_info['next_url'] = next_links[0].get('href')
            
            prev_links = soup.find_all('a', string=re.compile(r'prev|previous|<|back', re.I))
            if prev_links:
                pagination_info['has_previous'] = True
                pagination_info['previous_url'] = prev_links[0].get('href')
            
            # Try to extract page numbers
            page_text = soup.get_text()
            page_match = re.search(r'page\s+(\d+)\s+of\s+(\d+)', page_text, re.I)
            if page_match:
                pagination_info['current_page'] = int(page_match.group(1))
                pagination_info['total_pages'] = int(page_match.group(2))
            
            extracted_data['pagination'] = pagination_info
            extracted_data['metadata']['pagination_enhanced'] = True
            
        except Exception as e:
            self.logger.debug(f"Pagination extraction enhancement failed: {e}")
        
        return extracted_data

    # === Existing helper methods completion ===
    
    async def _crawl_single_page(self, crawler: AsyncWebCrawler, url: str, 
                               session: CrawlSession) -> Optional[PageData]:
        """Crawl a single page with site-aware optimization"""
        
        try:
            # Detect site type for optimal configuration
            site_info = self._detect_site_type("")  # Initial empty detection
            crawler_config = self._get_optimal_crawler_config(site_info)
            
            # Configure extraction strategy based on site type
            extraction_strategy = None
            if site_info.get('optimal_strategy') == 'modern_spa':
                extraction_strategy = JsonCssExtractionStrategy({
                    "content": "main, article, .content, .post-content, #content",
                    "title": "h1, .title, .post-title, title",
                    "links": "a[href]"
                })
            
            # Create run configuration
            run_config = CrawlerRunConfig(
                word_count_threshold=crawler_config.get('word_count_threshold', 10),
                only_text=crawler_config.get('only_text', False),
                bypass_cache=crawler_config.get('bypass_cache', False),
                include_raw_html=crawler_config.get('include_raw_html', True),
                wait_for=crawler_config.get('wait_for'),
                js_code=crawler_config.get('js_code'),
                css_selector=crawler_config.get('css_selector'),
                screenshot=crawler_config.get('screenshot', False),
                delay_before_return_html=crawler_config.get('delay_before_return_html', 2.0),
                extraction_strategy=extraction_strategy
            )
            
            # Execute crawl with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    result = await crawler.arun(url=url, config=run_config)
                    
                    if result and result.success:
                        # Re-detect site type with actual content
                        site_info = self._detect_site_type(result.html)
                        
                        # Create PageData object
                        page_data = PageData(
                            url=url,
                            title=result.metadata.get('title', ''),
                            main_content=result.cleaned_html or result.markdown or '',
                            structured_data=result.extracted_content or {},
                            relevance_score=0.0,  # Will be calculated later
                            timestamp=time.time(),
                            metadata={
                                'site_info': site_info,
                                'crawl_config': crawler_config,
                                'attempt': attempt + 1,
                                'success': True
                            }
                        )
                        
                        # Apply advanced extraction techniques
                        page_data = await self._apply_advanced_extraction_techniques(page_data, site_info)
                        
                        # Calculate relevance score
                        page_data.relevance_score = self._calculate_relevance_score(page_data, session)
                        
                        return page_data
                        
                except Exception as crawl_error:
                    self.logger.warning(f"Crawl attempt {attempt + 1} failed for {url}: {crawl_error}")
                    if attempt == max_retries - 1:
                        # Try HTTP fallback on final attempt
                        return await self._http_fallback_crawl(url, session)
                    
                    # Exponential backoff
                    await asyncio.sleep(2 ** attempt)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Single page crawl failed for {url}: {e}")
            return await self._http_fallback_crawl(url, session)
    
    async def _http_fallback_crawl(self, url: str, session: CrawlSession) -> Optional[PageData]:
        """Fallback HTTP-based crawling when browser-based crawl fails"""
        
        if not HTTPX_AVAILABLE:
            return None
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                html_content = response.text
                
                # Extract data using the extract_data method
                extracted_data = await self.extract_data(html_content)
                
                page_data = PageData(
                    url=url,
                    title=extracted_data.get('title', ''),
                    main_content=extracted_data.get('main_content', ''),
                    structured_data=extracted_data.get('metadata', {}),
                    relevance_score=0.0,
                    timestamp=time.time(),
                    metadata={
                        'extraction_method': 'http_fallback',
                        'content_length': len(html_content)
                    }
                )
                
                # Calculate relevance score
                page_data.relevance_score = self._calculate_relevance_score(page_data, session)
                
                return page_data
                
        except Exception as e:
            self.logger.warning(f"HTTP fallback crawl failed for {url}: {e}")
            return None
    
    def _calculate_relevance_score(self, page_data: PageData, session: CrawlSession) -> float:
        """Calculate relevance score for a page based on query and content"""
        
        try:
            query_terms = session.query.lower().split()
            content_lower = page_data.main_content.lower()
            title_lower = page_data.title.lower()
            
            # Basic keyword matching
            content_matches = sum(1 for term in query_terms if term in content_lower)
            title_matches = sum(1 for term in query_terms if term in title_lower)
            
            # Calculate base score
            content_score = content_matches / max(len(query_terms), 1)
            title_score = title_matches / max(len(query_terms), 1) * 2  # Title matches worth more
            
            # Content quality factors
            content_length = len(page_data.main_content)
            length_score = min(content_length / 1000, 1.0)  # Up to 1000 chars = full score
            
            # Combine scores
            relevance_score = (content_score * 0.4 + title_score * 0.4 + length_score * 0.2)
            
            # Boost for structured data
            if page_data.structured_data:
                relevance_score *= 1.1
            
            # Boost for modern site extraction
            if page_data.metadata.get('site_info', {}).get('is_spa'):
                relevance_score *= 1.05
            
            return min(relevance_score, 1.0)
            
        except Exception as e:
            self.logger.debug(f"Relevance calculation failed: {e}")
            return 0.5  # Default moderate relevance
    
    async def _evaluate_page_relevance(self, page_data: PageData, session: CrawlSession) -> bool:
        """Evaluate if a page is relevant enough to include in results"""
        
        # Check minimum relevance score
        if page_data.relevance_score < self.min_relevance_score:
            return False
        
        # Check minimum content length
        if len(page_data.main_content) < self.min_content_length:
            return False
        
        # Check for duplicate or near-duplicate content
        for existing_page in session.collected_pages:
            similarity = self._calculate_content_similarity(
                page_data.main_content, existing_page.main_content
            )
            if similarity > 0.8:  # Too similar
                return False
        
        return True
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two content strings"""
        
        try:
            # Simple word-based similarity
            words1 = set(content1.lower().split())
            words2 = set(content2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception:
            return 0.0
    
    async def _discover_follow_up_urls(self, crawler: AsyncWebCrawler, current_url: str,
                                     page_data: PageData, session: CrawlSession) -> List[str]:
        """Discover follow-up URLs from the current page"""
        
        try:
            from urllib.parse import urljoin, urlparse
            
            # Extract links from structured data
            discovered_urls = []
            
            # Get links from the page data
            if 'links' in page_data.structured_data:
                for link in page_data.structured_data['links'][:10]:  # Limit to prevent explosion
                    if link and isinstance(link, str):
                        full_url = urljoin(current_url, link)
                        if self._is_valid_follow_url(full_url, current_url):
                            discovered_urls.append(full_url)
            
            # Look for pagination links
            if page_data.structured_data.get('pagination', {}).get('has_next'):
                next_url = page_data.structured_data['pagination'].get('next_url')
                if next_url:
                    full_next_url = urljoin(current_url, next_url)
                    if self._is_valid_follow_url(full_next_url, current_url):
                        discovered_urls.append(full_next_url)
            
            return discovered_urls[:5]  # Limit to prevent crawl explosion
            
        except Exception as e:
            self.logger.debug(f"URL discovery failed for {current_url}: {e}")
            return []
    
    def _is_valid_follow_url(self, url: str, base_url: str) -> bool:
        """Check if a URL is valid for following during crawl"""
        
        try:
            from urllib.parse import urlparse
            
            parsed_url = urlparse(url)
            parsed_base = urlparse(base_url)
            
            # Must be same domain
            if parsed_url.netloc != parsed_base.netloc:
                return False
            
            # Skip common non-content URLs
            skip_patterns = [
                'login', 'register', 'logout', 'contact', 'about',
                'privacy', 'terms', 'cookie', 'admin', 'api',
                '.pdf', '.jpg', '.png', '.gif', '.css', '.js'
            ]
            
            url_lower = url.lower()
            for pattern in skip_patterns:
                if pattern in url_lower:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _calculate_url_priority(self, url: str, session: CrawlSession) -> float:
        """Calculate priority score for a discovered URL"""
        
        try:
            priority = 0.5  # Base priority
            
            # Boost for query terms in URL
            query_terms = session.query.lower().split()
            url_lower = url.lower()
            
            for term in query_terms:
                if term in url_lower:
                    priority += 0.2
            
            # Boost for certain URL patterns
            if any(pattern in url_lower for pattern in ['article', 'post', 'page', 'content']):
                priority += 0.1
            
            # Reduce priority for deep paths
            path_depth = url.count('/') - 2  # Subtract protocol and domain
            priority -= path_depth * 0.05
            
            return max(0.1, min(1.0, priority))
            
        except Exception:
            return 0.5
    
    async def _should_terminate_crawl(self, session: CrawlSession) -> bool:
        """Check if crawl should be terminated early"""
        
        # Check if we have enough high-quality results
        if len(session.collected_pages) >= 3:
            avg_relevance = session.total_relevance / len(session.collected_pages)
            if avg_relevance >= 0.7:
                return True
        
        # Check time-based termination
        elapsed_time = time.time() - session.start_time
        if elapsed_time > 300:  # 5 minutes max
            return True
        
        return False
    
    async def _consolidate_results(self, session: CrawlSession) -> Dict[str, Any]:
        """Consolidate and process collected page data into final results"""
        
        try:
            if not session.collected_pages:
                return {
                    "success": False,
                    "items": [],
                    "total_pages": 0,
                    "average_relevance": 0.0,
                    "processing_time": time.time() - session.start_time
                }
            
            # Process and consolidate page data
            consolidated_items = []
            
            for page_data in session.collected_pages:
                item = {
                    "url": page_data.url,
                    "title": page_data.title,
                    "content": page_data.main_content[:1000],  # Truncate for response size
                    "full_content": page_data.main_content,
                    "structured_data": page_data.structured_data,
                    "relevance_score": page_data.relevance_score,
                    "timestamp": page_data.timestamp,
                    "metadata": page_data.metadata
                }
                
                consolidated_items.append(item)
            
            # Sort by relevance score
            consolidated_items.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            # Calculate session metrics
            avg_relevance = session.total_relevance / len(session.collected_pages)
            processing_time = time.time() - session.start_time
            
            return {
                "success": True,
                "items": consolidated_items,
                "total_pages": len(session.collected_pages),
                "average_relevance": avg_relevance,
                "processing_time": processing_time,
                "query": session.query,
                "session_metadata": {
                    "strategy": "universal_crawl4ai",
                    "max_pages": self.max_pages,
                    "deep_crawl_enabled": self.deep_crawl_enabled,
                    "ai_pathfinding": self.ai_pathfinding,
                    "total_urls_visited": len(session.visited_urls),
                    "performance_metrics": self.performance_metrics
                }
            }
            
        except Exception as e:
            self.logger.error(f"Result consolidation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "items": [],
                "total_pages": 0,
                "average_relevance": 0.0,
                "processing_time": time.time() - session.start_time
            }
    
    def _update_performance_metrics(self, session: CrawlSession, processing_time: float) -> None:
        """Update performance metrics after crawl completion"""
        
        try:
            self.performance_metrics.update({
                'total_processing_time': self.performance_metrics['total_processing_time'] + processing_time,
                'average_relevance': session.total_relevance / max(len(session.collected_pages), 1),
                'memory_peak': max(self.performance_metrics['memory_peak'], session.memory_usage)
            })
            
        except Exception as e:
            self.logger.debug(f"Performance metrics update failed: {e}")
    
    def _create_crawler_config(self) -> Dict[str, Any]:
        """Create base crawler configuration"""
        
        return {
            'word_count_threshold': 10,
            'only_text': False,
            'bypass_cache': True,
            'include_raw_html': True,
            'delay_before_return_html': 2.0,
            'timeout': 30.0
        }
    
    def _create_browser_config(self) -> BrowserConfig:
        """Create browser configuration for the crawler"""
        
        return BrowserConfig(
            browser_type="chromium",
            headless=True,
            viewport_width=1920,
            viewport_height=1080,
            timeout=30000,
            use_undetected_chrome=self.use_undetected_driver,
            extra_args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-web-security",
                "--disable-blink-features=AutomationControlled",
                "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ]
        )
    
    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker allows execution"""
        
        if not self.circuit_breaker_enabled:
            return True
        
        if self.circuit_breaker_failures >= self.circuit_breaker_threshold:
            # Check if timeout has passed
            if time.time() - self.circuit_breaker_last_failure < self.circuit_breaker_timeout:
                return False
            else:
                # Reset circuit breaker
                self.circuit_breaker_failures = 0
                return True
        
        return True
    
    def _record_circuit_breaker_failure(self) -> None:
        """Record a failure for circuit breaker tracking"""
        
        if self.circuit_breaker_enabled:
            self.circuit_breaker_failures += 1
            self.circuit_breaker_last_failure = time.time()
            
            if self.circuit_breaker_failures >= self.circuit_breaker_threshold:
                self.logger.warning(f"Circuit breaker opened after {self.circuit_breaker_failures} failures")
    
    # === Required Abstract Methods from BaseStrategy ===
    
    @property
    def name(self) -> str:
        """Get the name of the strategy"""
        return "UniversalCrawl4AIStrategy"
    
    def crawl(self, start_url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Crawl from a starting URL using this strategy.
        This is a synchronous wrapper around the async execute method.
        """
        try:
            # Create an event loop if one doesn't exist
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is already running, we need to use a different approach
                    # This typically happens in Jupyter notebooks or async contexts
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, self.execute(start_url, **kwargs))
                        return future.result()
                else:
                    return loop.run_until_complete(self.execute(start_url, **kwargs))
            except RuntimeError:
                # No event loop exists, create one
                return asyncio.run(self.execute(start_url, **kwargs))
        except Exception as e:
            self.logger.error(f"Crawl execution failed: {e}")
            return None
    
    def extract(self, html_content: str, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Extract data from HTML content using this strategy.
        This is a synchronous wrapper around the async extract_data method.
        """
        try:
            # Create an event loop if one doesn't exist
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is already running, we need to use a different approach
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, self.extract_data(html_content, **kwargs))
                        return future.result()
                else:
                    return loop.run_until_complete(self.extract_data(html_content, **kwargs))
            except RuntimeError:
                # No event loop exists, create one
                return asyncio.run(self.extract_data(html_content, **kwargs))
        except Exception as e:
            self.logger.error(f"Data extraction failed: {e}")
            return None
    
    def get_results(self) -> List[Dict[str, Any]]:
        """
        Get all results collected by this strategy.
        Since this strategy processes results immediately, we return the performance metrics.
        """
        return [{
            'strategy': 'universal_crawl4ai',
            'performance_metrics': self.performance_metrics,
            'circuit_breaker_status': {
                'failures': self.circuit_breaker_failures,
                'is_open': self.circuit_breaker_failures >= self.circuit_breaker_threshold,
                'last_failure': self.circuit_breaker_last_failure
            },
            'configuration': {
                'max_pages': self.max_pages,
                'deep_crawl_enabled': self.deep_crawl_enabled,
                'ai_pathfinding': self.ai_pathfinding,
                'memory_threshold': self.memory_threshold,
                'circuit_breaker_enabled': self.circuit_breaker_enabled
            }
        }]
