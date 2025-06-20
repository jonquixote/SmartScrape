"""
Modern Site Extraction Strategy

Specialized strategy for handling modern web applications including:
- React, Vue, Angular, and other SPA frameworks
- Dynamic content loading and lazy loading
- JavaScript-heavy sites with AJAX/fetch requests
- Infinite scroll and dynamic pagination
- Real-time content updates

This strategy extends the universal crawler with enhanced JavaScript execution,
intelligent wait strategies, and framework-specific detection.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from bs4 import BeautifulSoup

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.async_crawler_strategy import AsyncCrawlerStrategy

from .base_strategy import BaseStrategy
from ..extraction.content_analysis import detect_javascript_dependency, identify_dynamic_content_areas
from ..core.content_processor import ContentProcessor
from ..core.error_handler import handle_errors_gracefully


@dataclass
class ModernSiteConfig:
    """Configuration for modern site extraction"""
    # JavaScript execution settings
    javascript_enabled: bool = True
    wait_for_selector: Optional[str] = None
    wait_timeout: int = 15000  # 15 seconds
    page_load_timeout: int = 30000  # 30 seconds
    
    # Framework detection
    detect_frameworks: bool = True
    framework_specific_waits: bool = True
    
    # Dynamic content handling
    lazy_loading_scroll: bool = True
    infinite_scroll_detection: bool = True
    ajax_wait_time: float = 2.0
    
    # Performance settings
    max_retries: int = 3
    headless: bool = True
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


class ModernSiteStrategy(BaseStrategy):
    """
    Advanced extraction strategy for modern web applications and SPAs.
    
    This strategy provides:
    - Framework detection (React, Vue, Angular, etc.)
    - Intelligent wait strategies for dynamic content
    - Lazy loading and infinite scroll handling
    - AJAX/fetch request monitoring
    - Progressive enhancement fallbacks
    """
    
    def __init__(self, config: Optional[ModernSiteConfig] = None):
        super().__init__()
        self.config = config or ModernSiteConfig()
        self.content_processor = ContentProcessor()
        self.logger = logging.getLogger(__name__)
        
        # Framework detection patterns
        self.framework_patterns = {
            'react': [
                'data-reactroot', 'data-react-checksum', '_reactInternalInstance',
                'React.createElement', 'ReactDOM.render', '__REACT_DEVTOOLS_GLOBAL_HOOK__'
            ],
            'vue': [
                'v-for', 'v-if', 'v-bind', 'v-model', 'data-v-', 'Vue.createApp', 
                '__VUE__', 'vue.runtime', '$mount'
            ],
            'angular': [
                'ng-app', 'ng-controller', 'ng-repeat', 'ng-if', 'ng-model',
                'angular.module', 'ng-version', '[ng-'
            ],
            'svelte': [
                'svelte-', '__svelte', 'svelte.js', 'SvelteComponent'
            ],
            'ember': [
                'ember-view', 'ember-application', 'Ember.Application', 'ember.js'
            ]
        }
        
        # Dynamic content indicators
        self.dynamic_indicators = [
            'loading', 'spinner', 'skeleton', 'placeholder', 'lazy-load',
            'data-src', 'data-lazy', 'intersection-observer', 'infinite-scroll'
        ]

    async def extract_data(self, url: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract data from modern web applications with enhanced JavaScript support.
        """
        try:
            self.logger.info(f"Starting modern site extraction for: {url}")
            
            # Initial analysis
            analysis_result = await self._analyze_site_requirements(url)
            extraction_strategy = self._determine_extraction_strategy(analysis_result)
            
            # Execute extraction based on strategy
            if extraction_strategy == 'spa_intensive':
                return await self._extract_spa_content(url, context, analysis_result)
            elif extraction_strategy == 'dynamic_moderate':
                return await self._extract_dynamic_content(url, context, analysis_result)
            else:
                return await self._extract_with_minimal_js(url, context)
                
        except Exception as e:
            self.logger.error(f"Modern site extraction failed for {url}: {str(e)}")
            return {"error": str(e), "extraction_method": "modern_site_failed"}

    async def generate_urls(self, query: str, context: Dict[str, Any]) -> List[str]:
        """
        Generate URLs for modern site extraction (delegates to DuckDuckGo).
        """
        # This strategy focuses on extraction, not URL generation
        # URL generation is handled by DuckDuckGo components
        return []

    async def _analyze_site_requirements(self, url: str) -> Dict[str, Any]:
        """
        Analyze the site to determine its modern web application requirements.
        """
        try:
            # First, do a lightweight fetch to analyze structure
            browser_config = BrowserConfig(
                headless=self.config.headless,
                java_script_enabled=False,  # Initial analysis without JS
                browser_type="chromium"
            )
            
            async with AsyncWebCrawler(config=browser_config) as crawler:
                result = await crawler.arun(
                    url=url,
                    cache_mode=CacheMode.BYPASS,
                    crawler_config=CrawlerRunConfig()
                )
                
                if not result.success:
                    return {"analysis_failed": True, "error": result.error_message}
                
                return await self._analyze_html_structure(result.html, url)
                
        except Exception as e:
            self.logger.error(f"Site analysis failed for {url}: {str(e)}")
            return {"analysis_failed": True, "error": str(e)}

    async def _analyze_html_structure(self, html_content: str, url: str) -> Dict[str, Any]:
        """
        Analyze HTML structure for modern web application characteristics.
        """
        soup = BeautifulSoup(html_content, 'lxml')
        
        analysis = {
            'detected_frameworks': [],
            'js_dependency_level': 0,
            'has_dynamic_content': False,
            'lazy_loading_present': False,
            'infinite_scroll_indicators': False,
            'ajax_patterns': False,
            'spa_characteristics': False,
            'content_sparse': False,
            'wait_selectors': []
        }
        
        # Framework detection
        html_str = str(soup).lower()
        for framework, patterns in self.framework_patterns.items():
            if any(pattern.lower() in html_str for pattern in patterns):
                analysis['detected_frameworks'].append(framework)
        
        # JavaScript dependency analysis
        analysis['js_dependency_level'] = detect_javascript_dependency(soup)
        analysis['has_dynamic_content'] = analysis['js_dependency_level'] >= 2
        
        # Dynamic content area identification
        dynamic_areas = identify_dynamic_content_areas(soup)
        analysis['dynamic_content_areas'] = dynamic_areas
        
        # Check for lazy loading indicators
        lazy_indicators = soup.select('[loading="lazy"], [data-src], [data-lazy]')
        analysis['lazy_loading_present'] = len(lazy_indicators) > 0
        
        # Check for infinite scroll indicators
        infinite_scroll_patterns = ['infinite', 'load-more', 'pagination-dynamic']
        analysis['infinite_scroll_indicators'] = any(
            pattern in html_str for pattern in infinite_scroll_patterns
        )
        
        # Analyze content sparsity
        text_content = soup.get_text(strip=True)
        script_tags = soup.find_all('script')
        analysis['content_sparse'] = len(text_content) < 1000 and len(script_tags) > 5
        
        # SPA characteristics
        analysis['spa_characteristics'] = (
            len(analysis['detected_frameworks']) > 0 or
            analysis['content_sparse'] or
            analysis['js_dependency_level'] >= 3
        )
        
        # Identify potential wait selectors
        analysis['wait_selectors'] = self._identify_wait_selectors(soup)
        
        return analysis

    def _identify_wait_selectors(self, soup: BeautifulSoup) -> List[str]:
        """
        Identify CSS selectors that would indicate content has loaded.
        """
        selectors = []
        
        # Common content containers
        content_containers = soup.select('[id*="content"], [class*="content"], main, article')
        for container in content_containers[:3]:  # Limit to first 3
            if container.get('id'):
                selectors.append(f"#{container['id']}")
            elif container.get('class'):
                selectors.append(f".{container['class'][0]}")
        
        # List containers
        list_containers = soup.select('[class*="list"], [class*="grid"], [class*="items"]')
        for container in list_containers[:2]:
            if container.get('class'):
                selectors.append(f".{container['class'][0]}")
        
        return selectors

    def _determine_extraction_strategy(self, analysis: Dict[str, Any]) -> str:
        """
        Determine the appropriate extraction strategy based on site analysis.
        """
        if analysis.get('analysis_failed'):
            return 'fallback'
        
        if analysis.get('spa_characteristics') or analysis['js_dependency_level'] >= 3:
            return 'spa_intensive'
        elif analysis.get('has_dynamic_content') or analysis['js_dependency_level'] >= 2:
            return 'dynamic_moderate'
        else:
            return 'minimal_js'

    async def _extract_spa_content(self, url: str, context: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract content from Single Page Applications with intensive JavaScript.
        """
        try:
            self.logger.info(f"Extracting SPA content from: {url}")
            
            browser_config = BrowserConfig(
                headless=self.config.headless,
                java_script_enabled=True,
                browser_type="chromium",
                user_agent=self.config.user_agent,
                viewport_width=1920,
                viewport_height=1080
            )
            
            # Configure crawler for SPA handling
            crawler_config = CrawlerRunConfig(
                page_timeout=self.config.page_load_timeout,
                cache_mode=CacheMode.BYPASS
            )
            
            # Add framework-specific JavaScript if detected
            js_code = self._generate_framework_specific_js(analysis)
            if js_code:
                crawler_config.js_code = js_code
            
            # Set wait selector if available
            wait_selectors = analysis.get('wait_selectors', [])
            if wait_selectors:
                crawler_config.wait_for = wait_selectors[0]
                crawler_config.delay_before_return_html = self.config.ajax_wait_time
            
            async with AsyncWebCrawler(config=browser_config) as crawler:
                result = await crawler.arun(url=url, crawler_config=crawler_config)
                
                if not result.success:
                    return {"error": result.error_message, "extraction_method": "spa_failed"}
                
                # Process extracted content
                return await self._process_extracted_content(result, url, "spa_intensive")
                
        except Exception as e:
            self.logger.error(f"SPA extraction failed for {url}: {str(e)}")
            return {"error": str(e), "extraction_method": "spa_failed"}

    async def _extract_dynamic_content(self, url: str, context: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract content from sites with moderate dynamic content.
        """
        try:
            self.logger.info(f"Extracting dynamic content from: {url}")
            
            browser_config = BrowserConfig(
                headless=self.config.headless,
                java_script_enabled=True,
                browser_type="chromium",
                user_agent=self.config.user_agent
            )
            
            # JavaScript for handling lazy loading and dynamic content
            js_code = """
            // Scroll to trigger lazy loading
            window.scrollTo(0, document.body.scrollHeight / 3);
            await new Promise(resolve => setTimeout(resolve, 1000));
            window.scrollTo(0, document.body.scrollHeight / 2);
            await new Promise(resolve => setTimeout(resolve, 1000));
            window.scrollTo(0, document.body.scrollHeight);
            await new Promise(resolve => setTimeout(resolve, 2000));
            
            // Wait for any loading indicators to disappear
            const loadingElements = document.querySelectorAll('[class*="loading"], [class*="spinner"]');
            for (let element of loadingElements) {
                if (element.style.display !== 'none') {
                    await new Promise(resolve => {
                        const observer = new MutationObserver(() => {
                            if (element.style.display === 'none' || !document.contains(element)) {
                                observer.disconnect();
                                resolve();
                            }
                        });
                        observer.observe(element, { attributes: true, childList: true });
                        setTimeout(() => { observer.disconnect(); resolve(); }, 5000);
                    });
                }
            }
            """
            
            crawler_config = CrawlerRunConfig(
                js_code=js_code,
                page_timeout=self.config.page_load_timeout,
                delay_before_return_html=self.config.ajax_wait_time,
                cache_mode=CacheMode.BYPASS
            )
            
            async with AsyncWebCrawler(config=browser_config) as crawler:
                result = await crawler.arun(url=url, crawler_config=crawler_config)
                
                if not result.success:
                    return {"error": result.error_message, "extraction_method": "dynamic_failed"}
                
                return await self._process_extracted_content(result, url, "dynamic_moderate")
                
        except Exception as e:
            self.logger.error(f"Dynamic content extraction failed for {url}: {str(e)}")
            return {"error": str(e), "extraction_method": "dynamic_failed"}

    async def _extract_with_minimal_js(self, url: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract content with minimal JavaScript requirements.
        """
        try:
            self.logger.info(f"Extracting with minimal JS from: {url}")
            
            browser_config = BrowserConfig(
                headless=self.config.headless,
                java_script_enabled=True,
                browser_type="chromium"
            )
            
            crawler_config = CrawlerRunConfig(
                page_timeout=20000,  # Shorter timeout for simpler sites
                delay_before_return_html=1.0,
                cache_mode=CacheMode.BYPASS
            )
            
            async with AsyncWebCrawler(config=browser_config) as crawler:
                result = await crawler.arun(url=url, crawler_config=crawler_config)
                
                if not result.success:
                    return {"error": result.error_message, "extraction_method": "minimal_js_failed"}
                
                return await self._process_extracted_content(result, url, "minimal_js")
                
        except Exception as e:
            self.logger.error(f"Minimal JS extraction failed for {url}: {str(e)}")
            return {"error": str(e), "extraction_method": "minimal_js_failed"}

    def _generate_framework_specific_js(self, analysis: Dict[str, Any]) -> Optional[str]:
        """
        Generate framework-specific JavaScript code for better extraction.
        """
        detected_frameworks = analysis.get('detected_frameworks', [])
        
        js_snippets = []
        
        if 'react' in detected_frameworks:
            js_snippets.append("""
            // Wait for React to finish rendering
            if (window.React || window.__REACT_DEVTOOLS_GLOBAL_HOOK__) {
                await new Promise(resolve => {
                    if (window.requestIdleCallback) {
                        window.requestIdleCallback(resolve, { timeout: 3000 });
                    } else {
                        setTimeout(resolve, 2000);
                    }
                });
            }
            """)
        
        if 'vue' in detected_frameworks:
            js_snippets.append("""
            // Wait for Vue to finish rendering
            if (window.Vue || window.__VUE__) {
                await new Promise(resolve => setTimeout(resolve, 2000));
            }
            """)
        
        if 'angular' in detected_frameworks:
            js_snippets.append("""
            // Wait for Angular to stabilize
            if (window.angular || window.ng) {
                await new Promise(resolve => {
                    const checkStability = () => {
                        if (window.getAllAngularTestabilities) {
                            const testabilities = window.getAllAngularTestabilities();
                            const stable = testabilities.every(t => t.isStable());
                            if (stable) {
                                resolve();
                            } else {
                                setTimeout(checkStability, 500);
                            }
                        } else {
                            setTimeout(resolve, 2000);
                        }
                    };
                    checkStability();
                });
            }
            """)
        
        return '\n'.join(js_snippets) if js_snippets else None

    async def _process_extracted_content(self, result: Any, url: str, method: str) -> Dict[str, Any]:
        """
        Process the extracted content and format the response.
        """
        try:
            # Extract basic content
            content_data = {
                "url": url,
                "extraction_method": method,
                "success": result.success,
                "html_content": result.html,
                "raw_text": result.cleaned_html,
                "links": result.links.get("internal", []) + result.links.get("external", []),
                "images": result.media.get("images", []),
                "title": "",
                "content": "",
                "structured_data": {}
            }
            
            # Process with BeautifulSoup for additional analysis
            if result.html:
                soup = BeautifulSoup(result.html, 'lxml')
                
                # Extract title
                title_tag = soup.find('title')
                if title_tag:
                    content_data["title"] = title_tag.get_text(strip=True)
                
                # Extract main content
                content_data["content"] = await self._extract_main_content(soup)
                
                # Extract structured data
                content_data["structured_data"] = self._extract_structured_data(soup)
                
                # Content quality metrics
                content_data["quality_metrics"] = self._calculate_quality_metrics(soup)
            
            return content_data
            
        except Exception as e:
            self.logger.error(f"Content processing failed: {str(e)}")
            return {
                "url": url,
                "extraction_method": method,
                "success": False,
                "error": str(e)
            }

    async def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """
        Extract the main content from the page using intelligent selectors.
        """
        # Try multiple content extraction strategies
        content_selectors = [
            'main', '[role="main"]', '.main-content', '#main-content',
            'article', '.article', '.content', '#content',
            '.post-content', '.entry-content', '.page-content'
        ]
        
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                return elements[0].get_text(strip=True, separator=' ')
        
        # Fallback to body content
        body = soup.find('body')
        if body:
            # Remove script and style tags
            for tag in body(['script', 'style', 'nav', 'header', 'footer']):
                tag.decompose()
            return body.get_text(strip=True, separator=' ')
        
        return soup.get_text(strip=True, separator=' ')

    def _extract_structured_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Extract structured data from the page.
        """
        structured_data = {}
        
        # JSON-LD structured data
        json_ld_tags = soup.find_all('script', type='application/ld+json')
        json_ld_data = []
        
        for tag in json_ld_tags:
            try:
                data = json.loads(tag.string)
                json_ld_data.append(data)
            except (json.JSONDecodeError, TypeError):
                continue
        
        if json_ld_data:
            structured_data['json_ld'] = json_ld_data
        
        # Meta tags
        meta_data = {}
        meta_tags = soup.find_all('meta')
        
        for tag in meta_tags:
            name = tag.get('name') or tag.get('property') or tag.get('itemprop')
            content = tag.get('content')
            if name and content:
                meta_data[name] = content
        
        if meta_data:
            structured_data['meta'] = meta_data
        
        return structured_data

    def _calculate_quality_metrics(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Calculate content quality metrics for the extracted content.
        """
        text = soup.get_text(strip=True)
        
        return {
            "text_length": len(text),
            "word_count": len(text.split()),
            "paragraph_count": len(soup.find_all('p')),
            "link_count": len(soup.find_all('a')),
            "image_count": len(soup.find_all('img')),
            "has_structured_content": bool(soup.find_all(['article', 'section', 'header'])),
            "content_density": len(text) / max(len(str(soup)), 1)
        }

    @handle_errors_gracefully
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of the modern site strategy.
        """
        try:
            # Test basic browser functionality
            browser_config = BrowserConfig(headless=True, java_script_enabled=True)
            
            async with AsyncWebCrawler(config=browser_config) as crawler:
                # Test with a simple page
                result = await crawler.arun(
                    url="https://httpbin.org/html",
                    crawler_config=CrawlerRunConfig(page_timeout=10000)
                )
                
                return {
                    "status": "healthy" if result.success else "degraded",
                    "browser_functional": result.success,
                    "javascript_enabled": True,
                    "error": result.error_message if not result.success else None
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "browser_functional": False
            }
