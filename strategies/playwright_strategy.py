"""
Playwright Extraction Strategy for SmartScrape

This strategy uses Playwright for browser-based content extraction with retry logic
and dynamic content handling. It's designed as a reliable fallback for JavaScript-heavy
sites and complex dynamic content.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from playwright.async_api import async_playwright, Page, Browser, BrowserContext
from .base_strategy import BaseExtractionStrategy

# Optional stealth plugin
try:
    from playwright_stealth import stealth_async
    STEALTH_AVAILABLE = True
except ImportError:
    STEALTH_AVAILABLE = False

logger = logging.getLogger(__name__)

class PlaywrightStrategy(BaseExtractionStrategy):
    """Browser-based extraction strategy using Playwright"""
    
    def __init__(self):
        super().__init__()
        self.name = "playwright"
        self.description = "Browser-based extraction with Playwright for dynamic content"
        self.priority = 40  # Higher priority for JavaScript-heavy sites
        self.browser = None
        self.context = None
        
    async def extract(self, url: str, html: str = None, **kwargs) -> Dict[str, Any]:
        """
        Extract content using Playwright with retry logic
        
        Args:
            url: The URL to extract content from
            html: Ignored for Playwright (uses browser)
            **kwargs: Additional options
            
        Returns:
            Dict containing extracted content and metadata
        """
        max_retries = kwargs.get('max_retries', 3)
        return await self.extract_with_retry(url, max_retries)
    
    async def extract_with_retry(self, url: str, max_retries: int = 3) -> Dict[str, Any]:
        """Extract content with retry logic"""
        for attempt in range(max_retries):
            try:
                result = await self._extract_single_attempt(url, attempt)
                if result and result.get('success'):
                    return result
                    
            except Exception as e:
                logger.warning(f"Playwright extraction attempt {attempt + 1} failed for {url}: {e}")
                if attempt == max_retries - 1:
                    return self._create_error_result(url, f"All {max_retries} extraction attempts failed: {str(e)}")
                
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
        
        return self._create_error_result(url, f"All {max_retries} extraction attempts failed")
    
    async def _extract_single_attempt(self, url: str, attempt: int) -> Dict[str, Any]:
        """Single extraction attempt"""
        page = None
        try:
            async with async_playwright() as p:
                # Launch browser with appropriate settings
                browser = await p.chromium.launch(
                    headless=True,
                    args=[
                        '--no-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-blink-features=AutomationControlled',
                        '--disable-web-security',
                        '--disable-features=VizDisplayCompositor'
                    ]
                )
                
                # Create context with stealth settings
                context = await browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                )
                
                page = await context.new_page()
                
                # Apply stealth if available
                if STEALTH_AVAILABLE:
                    await stealth_async(page)
                
                # Set aggressive timeouts for fallback mode
                page.set_default_timeout(15000 + (attempt * 5000))  # Increase timeout with retries
                
                # Navigate to page
                await page.goto(url, wait_until='domcontentloaded', timeout=30000)
                
                # Wait for dynamic content to load
                await asyncio.sleep(2 + attempt)  # Longer wait on retries
                
                # Try to wait for common content indicators
                try:
                    await page.wait_for_selector('body', timeout=5000)
                except:
                    pass  # Continue if no body selector
                
                # Extract content using multiple methods
                content_data = await self._extract_content_data(page, url)
                
                await browser.close()
                
                return content_data
                
        except Exception as e:
            if page:
                try:
                    await page.close()
                except:
                    pass
            raise e
    
    async def _extract_content_data(self, page: Page, url: str) -> Dict[str, Any]:
        """Extract comprehensive content data from page"""
        try:
            # Extract main content using multiple selectors
            content = await page.evaluate("""
                () => {
                    // Remove scripts, styles, and other non-content elements
                    const elementsToRemove = document.querySelectorAll('script, style, nav, header, footer, aside, .ad, .ads, .advertisement');
                    elementsToRemove.forEach(el => el.remove());
                    
                    // Try to find main content using various selectors
                    const contentSelectors = [
                        'main',
                        'article', 
                        '[role="main"]',
                        '.content',
                        '#content',
                        '.post-content',
                        '.entry-content',
                        '.article-content',
                        '.page-content'
                    ];
                    
                    let mainContent = null;
                    for (const selector of contentSelectors) {
                        const element = document.querySelector(selector);
                        if (element && element.innerText.trim().length > 100) {
                            mainContent = element;
                            break;
                        }
                    }
                    
                    // Fall back to body if no main content found
                    if (!mainContent) {
                        mainContent = document.body;
                    }
                    
                    return {
                        text: mainContent ? mainContent.innerText.trim() : '',
                        html: mainContent ? mainContent.innerHTML : '',
                        title: document.title || '',
                        url: window.location.href
                    };
                }
            """)
            
            # Extract metadata
            metadata = await page.evaluate("""
                () => {
                    const getMeta = (name) => {
                        const meta = document.querySelector(`meta[name="${name}"], meta[property="${name}"], meta[property="og:${name}"]`);
                        return meta ? meta.content : null;
                    };
                    
                    return {
                        title: document.title || getMeta('title'),
                        description: getMeta('description'),
                        author: getMeta('author'),
                        keywords: getMeta('keywords'),
                        og_title: getMeta('og:title'),
                        og_description: getMeta('og:description'),
                        og_image: getMeta('og:image'),
                        canonical: document.querySelector('link[rel="canonical"]')?.href
                    };
                }
            """)
            
            # Get page metrics
            metrics = await self._get_page_metrics(page)
            
            success = bool(content.get('text') and len(content.get('text', '')) > 50)
            
            result = {
                'content': content.get('text', ''),
                'html_content': content.get('html', ''),
                'url': url,
                'final_url': content.get('url', url),
                'strategy': self.name,
                'success': success,
                'metadata': metadata,
                'metrics': metrics,
                'word_count': len(content.get('text', '').split()) if content.get('text') else 0,
                'extraction_method': 'playwright'
            }
            
            if success:
                logger.info(f"Playwright extraction successful for {url}")
            else:
                logger.warning(f"Playwright extraction yielded minimal content for {url}")
                
            return result
            
        except Exception as e:
            logger.error(f"Content extraction failed: {e}")
            return self._create_error_result(url, str(e))
    
    async def _get_page_metrics(self, page: Page) -> Dict[str, Any]:
        """Get page performance and quality metrics"""
        try:
            metrics = await page.evaluate("""
                () => {
                    return {
                        dom_elements: document.querySelectorAll('*').length,
                        images: document.querySelectorAll('img').length,
                        links: document.querySelectorAll('a').length,
                        forms: document.querySelectorAll('form').length,
                        scripts: document.querySelectorAll('script').length,
                        stylesheets: document.querySelectorAll('link[rel="stylesheet"]').length,
                        viewport_width: window.innerWidth,
                        viewport_height: window.innerHeight,
                        page_height: document.body.scrollHeight,
                        load_time: performance.now()
                    };
                }
            """)
            return metrics
        except Exception as e:
            logger.warning(f"Failed to get page metrics: {e}")
            return {}
    
    def _create_error_result(self, url: str, error_message: str) -> Dict[str, Any]:
        """Create standardized error result"""
        return {
            'content': '',
            'url': url,
            'strategy': self.name,
            'success': False,
            'error': error_message,
            'metadata': {},
            'extraction_method': 'playwright'
        }
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Return configuration schema for this strategy"""
        return {
            'max_retries': {
                'type': 'integer',
                'default': 3,
                'description': 'Maximum number of retry attempts'
            },
            'headless': {
                'type': 'boolean',
                'default': True,
                'description': 'Run browser in headless mode'
            },
            'wait_time': {
                'type': 'number',
                'default': 2.0,
                'description': 'Time to wait for dynamic content (seconds)'
            },
            'timeout': {
                'type': 'integer',
                'default': 30000,
                'description': 'Page load timeout in milliseconds'
            },
            'stealth_mode': {
                'type': 'boolean',
                'default': True,
                'description': 'Use stealth mode to avoid detection'
            }
        }
