"""
Enhanced URL validation and discovery system for SmartScrape.
Provides intelligent URL validation with smart fallbacks and alternative discovery methods.
"""

import asyncio
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urljoin, urlparse, urlunparse
import aiohttp
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET


class URLValidator:
    """
    Enhanced URL validation with intelligent fallback strategies.
    """
    
    def __init__(self, timeout: int = 10, max_retries: int = 2):
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = None
        
        # Error patterns for detection
        self.captcha_patterns = [
            r'captcha',
            r'verify you are human',
            r'robot check',
            r'security check',
            r'please complete the security check',
            r'cloudflare',
            r'ddos protection',
            r'access denied'
        ]
        
        self.error_404_patterns = [
            r'404',
            r'page not found',
            r'not found',
            r'does not exist',
            r'page unavailable',
            r'content not available'
        ]
        
        self.access_denied_patterns = [
            r'access denied',
            r'forbidden',
            r'401',
            r'403',
            r'unauthorized',
            r'permission denied',
            r'login required'
        ]
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def validate_url(self, url: str) -> Dict[str, Any]:
        """
        Validate a single URL and return detailed validation results.
        
        Args:
            url: The URL to validate
            
        Returns:
            Dict containing validation results and suggested alternatives
        """
        result = {
            'url': url,
            'is_valid': False,
            'status_code': None,
            'error_type': None,
            'error_message': None,
            'content_preview': None,
            'suggested_alternatives': [],
            'validation_time': time.time()
        }
        
        try:
            # Basic URL format validation
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                result['error_type'] = 'invalid_format'
                result['error_message'] = 'Invalid URL format'
                return result
            
            # Attempt to fetch the URL
            for attempt in range(self.max_retries + 1):
                try:
                    async with self.session.get(url, allow_redirects=True) as response:
                        result['status_code'] = response.status
                        
                        # Check for HTTP errors
                        if response.status >= 400:
                            if response.status == 404:
                                result['error_type'] = '404_not_found'
                                result['error_message'] = 'Page not found'
                            elif response.status in [401, 403]:
                                result['error_type'] = 'access_denied'
                                result['error_message'] = 'Access denied or forbidden'
                            else:
                                result['error_type'] = 'http_error'
                                result['error_message'] = f'HTTP {response.status}'
                            
                            # Try to suggest alternatives for failed URLs
                            result['suggested_alternatives'] = await self._suggest_alternative_urls(url)
                            return result
                        
                        # Get content for pattern analysis
                        content = await response.text()
                        result['content_preview'] = content[:1000]  # First 1000 chars
                        
                        # Check for error patterns in content
                        content_lower = content.lower()
                        
                        # Check for CAPTCHA
                        if any(re.search(pattern, content_lower, re.IGNORECASE) for pattern in self.captcha_patterns):
                            result['error_type'] = 'captcha_detected'
                            result['error_message'] = 'CAPTCHA or bot protection detected'
                            result['suggested_alternatives'] = await self._suggest_alternative_urls(url)
                            return result
                        
                        # Check for 404 content patterns
                        if any(re.search(pattern, content_lower, re.IGNORECASE) for pattern in self.error_404_patterns):
                            result['error_type'] = 'content_404'
                            result['error_message'] = 'Page content indicates 404 error'
                            result['suggested_alternatives'] = await self._suggest_alternative_urls(url)
                            return result
                        
                        # Check for access denied patterns
                        if any(re.search(pattern, content_lower, re.IGNORECASE) for pattern in self.access_denied_patterns):
                            result['error_type'] = 'content_access_denied'
                            result['error_message'] = 'Page content indicates access denied'
                            result['suggested_alternatives'] = await self._suggest_alternative_urls(url)
                            return result
                        
                        # If we reach here, the URL is valid
                        result['is_valid'] = True
                        result['error_message'] = 'URL is accessible and valid'
                        return result
                        
                except aiohttp.ClientError as e:
                    if attempt == self.max_retries:
                        result['error_type'] = 'connection_error'
                        result['error_message'] = f'Connection failed: {str(e)}'
                        result['suggested_alternatives'] = await self._suggest_alternative_urls(url)
                    else:
                        # Wait before retry
                        await asyncio.sleep(1)
                        continue
                        
        except Exception as e:
            result['error_type'] = 'validation_error'
            result['error_message'] = f'Validation failed: {str(e)}'
            result['suggested_alternatives'] = await self._suggest_alternative_urls(url)
        
        return result
    
    async def _suggest_alternative_urls(self, failed_url: str) -> List[str]:
        """
        Suggest alternative URLs when the original fails.
        
        Args:
            failed_url: The URL that failed validation
            
        Returns:
            List of suggested alternative URLs
        """
        alternatives = []
        parsed = urlparse(failed_url)
        
        if not parsed.netloc:
            return alternatives
        
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        
        # Common alternative paths to try
        common_paths = [
            '/',  # Homepage
            '/index.html',
            '/home',
            '/main',
            '/news',
            '/blog',
            '/articles',
            '/search',
            '/sitemap.xml'
        ]
        
        # If the original had a path, try variations
        if parsed.path and parsed.path != '/':
            path_parts = parsed.path.strip('/').split('/')
            
            # Try parent directories
            for i in range(len(path_parts) - 1, 0, -1):
                parent_path = '/' + '/'.join(path_parts[:i])
                alternatives.append(urljoin(base_url, parent_path))
            
            # Try with different extensions
            if '.' in path_parts[-1]:
                base_name = path_parts[-1].split('.')[0]
                path_without_last = '/'.join(path_parts[:-1])
                for ext in ['.html', '.htm', '.php', '.asp', '.aspx']:
                    new_path = f"/{path_without_last}/{base_name}{ext}" if path_without_last else f"/{base_name}{ext}"
                    alternatives.append(urljoin(base_url, new_path))
        
        # Add common paths
        for path in common_paths:
            alt_url = urljoin(base_url, path)
            if alt_url not in alternatives and alt_url != failed_url:
                alternatives.append(alt_url)
        
        # Limit to top 5 alternatives to avoid overwhelming
        return alternatives[:5]
    
    async def discover_site_urls(self, base_domain: str, user_query: str) -> List[str]:
        """
        Discover relevant URLs from a website using multiple methods.
        
        Args:
            base_domain: The base domain to search (e.g., 'example.com')
            user_query: The user's search query to find relevant pages
            
        Returns:
            List of discovered URLs
        """
        discovered_urls = []
        
        # Ensure proper URL format
        if not base_domain.startswith('http'):
            base_url = f"https://{base_domain}"
        else:
            base_url = base_domain
        
        try:
            # Method 1: Check sitemap
            sitemap_urls = await self._check_sitemap(base_url)
            discovered_urls.extend(sitemap_urls)
            
            # Method 2: Check robots.txt
            robots_urls = await self._check_robots_txt(base_url)
            discovered_urls.extend(robots_urls)
            
            # Method 3: Crawl main page for relevant links
            main_page_urls = await self._crawl_main_page(base_url, user_query)
            discovered_urls.extend(main_page_urls)
            
        except Exception as e:
            print(f"Error discovering URLs for {base_domain}: {e}")
        
        # Remove duplicates and limit results
        unique_urls = list(dict.fromkeys(discovered_urls))  # Preserve order while removing duplicates
        return unique_urls[:10]  # Limit to top 10 discoveries
    
    async def _check_sitemap(self, base_url: str) -> List[str]:
        """Check sitemap.xml for relevant URLs."""
        sitemap_urls = []
        
        for sitemap_path in ['/sitemap.xml', '/sitemap_index.xml', '/sitemaps/sitemap.xml']:
            try:
                sitemap_url = urljoin(base_url, sitemap_path)
                async with self.session.get(sitemap_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        # Parse XML sitemap
                        try:
                            root = ET.fromstring(content)
                            # Handle different sitemap namespaces
                            namespaces = {
                                'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9',
                                'news': 'http://www.google.com/schemas/sitemap-news/0.9'
                            }
                            
                            # Look for URL entries
                            for url_elem in root.findall('.//sm:url', namespaces):
                                loc_elem = url_elem.find('sm:loc', namespaces)
                                if loc_elem is not None and loc_elem.text:
                                    sitemap_urls.append(loc_elem.text)
                            
                            # If no URLs found with namespace, try without
                            if not sitemap_urls:
                                for url_elem in root.findall('.//url'):
                                    loc_elem = url_elem.find('loc')
                                    if loc_elem is not None and loc_elem.text:
                                        sitemap_urls.append(loc_elem.text)
                                        
                        except ET.ParseError:
                            # If XML parsing fails, try to extract URLs with regex
                            url_pattern = r'<loc>(https?://[^<]+)</loc>'
                            urls = re.findall(url_pattern, content)
                            sitemap_urls.extend(urls)
                        
                        # Limit sitemap URLs
                        if len(sitemap_urls) >= 20:
                            break
                            
            except Exception as e:
                continue  # Try next sitemap path
        
        return sitemap_urls[:20]  # Limit to 20 URLs from sitemap
    
    async def _check_robots_txt(self, base_url: str) -> List[str]:
        """Check robots.txt for sitemap references and allowed paths."""
        robots_urls = []
        
        try:
            robots_url = urljoin(base_url, '/robots.txt')
            async with self.session.get(robots_url) as response:
                if response.status == 200:
                    content = await response.text()
                    
                    # Extract sitemap URLs from robots.txt
                    sitemap_pattern = r'Sitemap:\s*(https?://[^\s]+)'
                    sitemaps = re.findall(sitemap_pattern, content, re.IGNORECASE)
                    
                    for sitemap_url in sitemaps:
                        try:
                            async with self.session.get(sitemap_url) as sitemap_response:
                                if sitemap_response.status == 200:
                                    sitemap_content = await sitemap_response.text()
                                    url_pattern = r'<loc>(https?://[^<]+)</loc>'
                                    urls = re.findall(url_pattern, sitemap_content)
                                    robots_urls.extend(urls[:10])  # Limit per sitemap
                        except Exception:
                            continue
                            
        except Exception as e:
            pass  # robots.txt might not exist
        
        return robots_urls
    
    async def _crawl_main_page(self, base_url: str, user_query: str) -> List[str]:
        """Crawl the main page to find relevant links."""
        main_page_urls = []
        
        try:
            async with self.session.get(base_url) as response:
                if response.status == 200:
                    content = await response.text()
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    # Find all links
                    links = soup.find_all('a', href=True)
                    
                    # Keywords from user query for relevance scoring
                    query_keywords = user_query.lower().split()
                    
                    link_scores = []
                    for link in links:
                        href = link['href']
                        
                        # Convert relative URLs to absolute
                        if href.startswith('/'):
                            href = urljoin(base_url, href)
                        elif not href.startswith('http'):
                            continue  # Skip invalid or non-HTTP links
                        
                        # Skip external links (different domain)
                        if urlparse(href).netloc != urlparse(base_url).netloc:
                            continue
                        
                        # Calculate relevance score
                        link_text = (link.get_text() or '').lower()
                        href_lower = href.lower()
                        
                        score = 0
                        for keyword in query_keywords:
                            if keyword in link_text:
                                score += 2
                            if keyword in href_lower:
                                score += 1
                        
                        # Boost certain types of pages
                        if any(word in href_lower for word in ['news', 'article', 'blog', 'post']):
                            score += 1
                        if any(word in href_lower for word in ['search', 'category', 'tag']):
                            score += 0.5
                        
                        if score > 0:
                            link_scores.append((href, score))
                    
                    # Sort by relevance score and take top results
                    link_scores.sort(key=lambda x: x[1], reverse=True)
                    main_page_urls = [url for url, score in link_scores[:15]]
                    
        except Exception as e:
            pass  # Main page might not be accessible
        
        return main_page_urls
    
    async def _generate_search_urls(self, base_url: str, user_query: str) -> List[str]:
        """Generate potential search URLs based on user query and base URL."""
        search_urls = []
        
        try:
            # Parse the base URL to get domain info
            parsed = urlparse(base_url)
            domain = parsed.netloc
            
            # Common search path patterns
            search_patterns = [
                '/search?q={}',
                '/search?query={}',
                '/s?q={}',
                '/?s={}',
                '/find?q={}',
                '/search.php?q={}',
                '/search.html?q={}',
                '/search/{}',
                '/results?q={}',
                '/query?q={}'
            ]
            
            # Clean and encode the query
            clean_query = re.sub(r'[^\w\s-]', '', user_query).strip()
            encoded_query = clean_query.replace(' ', '+')
            
            # Generate search URLs
            for pattern in search_patterns:
                search_path = pattern.format(encoded_query)
                search_url = urljoin(base_url, search_path)
                search_urls.append(search_url)
            
            # Also try with URL-encoded spaces
            encoded_query_spaces = clean_query.replace(' ', '%20')
            for pattern in search_patterns[:3]:  # Just top 3 patterns for space variant
                search_path = pattern.format(encoded_query_spaces)
                search_url = urljoin(base_url, search_path)
                search_urls.append(search_url)
                
        except Exception as e:
            # If URL generation fails, return empty list
            pass
        
        return search_urls[:10]  # Limit to 10 search URLs


async def enhanced_url_validation_and_discovery(urls: List[str], user_query: str) -> Dict[str, Any]:
    """
    Enhanced URL validation with intelligent discovery for failed URLs.
    
    Args:
        urls: List of URLs to validate
        user_query: User's original query for context-aware discovery
        
    Returns:
        Dict containing validation results and discovered alternatives
    """
    result = {
        'original_urls': urls,
        'valid_urls': [],
        'invalid_urls': [],
        'discovered_alternatives': [],
        'validation_summary': {},
        'total_processing_time': 0
    }
    
    start_time = time.time()
    
    async with URLValidator() as validator:
        # Validate each URL
        validation_tasks = [validator.validate_url(url) for url in urls]
        validation_results = await asyncio.gather(*validation_tasks, return_exceptions=True)
        
        valid_count = 0
        invalid_count = 0
        
        for i, validation_result in enumerate(validation_results):
            if isinstance(validation_result, Exception):
                # Handle validation exceptions
                result['invalid_urls'].append({
                    'url': urls[i],
                    'error': str(validation_result),
                    'alternatives': []
                })
                invalid_count += 1
                continue
            
            if validation_result['is_valid']:
                result['valid_urls'].append(validation_result)
                valid_count += 1
            else:
                result['invalid_urls'].append(validation_result)
                invalid_count += 1
                
                # Try to discover alternatives from the same domain
                if validation_result.get('suggested_alternatives'):
                    # Validate suggested alternatives
                    alt_validation_tasks = [
                        validator.validate_url(alt_url) 
                        for alt_url in validation_result['suggested_alternatives'][:3]  # Limit to 3
                    ]
                    alt_results = await asyncio.gather(*alt_validation_tasks, return_exceptions=True)
                    
                    for alt_result in alt_results:
                        if not isinstance(alt_result, Exception) and alt_result['is_valid']:
                            result['discovered_alternatives'].append(alt_result)
        
        # Enhanced aggressive fallback when all/most LLM URLs fail
        total_failures = len(result['invalid_urls'])
        total_valid = len(result['valid_urls'])
        failure_rate = total_failures / len(urls) if urls else 0
        
        # Trigger aggressive site discovery if failure rate is high or very few valid URLs
        should_trigger_aggressive_fallback = (
            failure_rate >= 0.7 or  # 70%+ failure rate
            total_valid < 2 or      # Less than 2 valid URLs
            (total_failures > 0 and total_valid == 0)  # All URLs failed
        )
        
        if should_trigger_aggressive_fallback:
            print(f"🚨 Aggressive fallback triggered! Failure rate: {failure_rate:.2f}, Valid URLs: {total_valid}")
            
            # Collect all unique domains from both valid and invalid URLs
            all_domains = set()
            
            # Add domains from invalid URLs (primary targets for discovery)
            for invalid_url_data in result['invalid_urls']:
                parsed = urlparse(invalid_url_data['url'])
                if parsed.netloc:
                    all_domains.add(parsed.netloc)
            
            # Also add domains from valid URLs for additional content discovery
            for valid_url_data in result['valid_urls']:
                parsed = urlparse(valid_url_data['url'])
                if parsed.netloc:
                    all_domains.add(parsed.netloc)
            
            # If no domains found, try to extract from original URLs
            if not all_domains:
                for url in urls:
                    try:
                        parsed = urlparse(url)
                        if parsed.netloc:
                            all_domains.add(parsed.netloc)
                    except:
                        continue
            
            print(f"🔍 Performing aggressive site discovery on {len(all_domains)} domains: {list(all_domains)}")
            
            # Enhanced discovery with multiple strategies per domain
            discovery_tasks = []
            for domain in list(all_domains)[:3]:  # Limit to 3 domains to avoid overload
                # Create comprehensive base URL
                domain_url = f"https://{domain}"
                discovery_tasks.append(
                    validator.discover_site_urls(domain_url, user_query)
                )
            
            if discovery_tasks:
                discovery_results = await asyncio.gather(*discovery_tasks, return_exceptions=True)
                
                all_discovered = []
                for discovery_result in discovery_results:
                    if not isinstance(discovery_result, Exception) and discovery_result:
                        all_discovered.extend(discovery_result)
                
                print(f"🔍 Site discovery found {len(all_discovered)} potential URLs")
                
                # Enhanced validation of discovered URLs with prioritization
                if all_discovered:
                    # Remove duplicates and prioritize by relevance
                    unique_discovered = list(dict.fromkeys(all_discovered))  # Preserve order
                    
                    # Validate discovered URLs (increased limit for aggressive fallback)
                    max_to_validate = min(20, len(unique_discovered))  # Up to 20 URLs
                    discovered_validation_tasks = [
                        validator.validate_url(url) for url in unique_discovered[:max_to_validate]
                    ]
                    discovered_validation_results = await asyncio.gather(
                        *discovered_validation_tasks, return_exceptions=True
                    )
                    
                    validated_alternatives = []
                    for disc_result in discovered_validation_results:
                        if not isinstance(disc_result, Exception) and disc_result.get('is_valid'):
                            validated_alternatives.append(disc_result)
                    
                    print(f"✅ Aggressive fallback validated {len(validated_alternatives)} alternative URLs")
                    result['discovered_alternatives'].extend(validated_alternatives)
                    
                    # If we still have very few alternatives, try even more aggressive strategies
                    if len(validated_alternatives) < 3 and all_domains:
                        print("🔥 Trying additional aggressive discovery strategies...")
                        
                        # Try common paths and search functionality
                        additional_urls = []
                        for domain in list(all_domains)[:2]:  # Limit to 2 domains
                            base_url = f"https://{domain}"
                            
                            # Generate query-specific search URLs
                            search_attempts = await validator._generate_search_urls(base_url, user_query)
                            additional_urls.extend(search_attempts[:5])  # Top 5 search attempts
                        
                        if additional_urls:
                            # Validate additional URLs
                            additional_validation_tasks = [
                                validator.validate_url(url) for url in additional_urls[:10]
                            ]
                            additional_results = await asyncio.gather(
                                *additional_validation_tasks, return_exceptions=True
                            )
                            
                            for add_result in additional_results:
                                if not isinstance(add_result, Exception) and add_result.get('is_valid'):
                                    result['discovered_alternatives'].append(add_result)
                                    
                            print(f"🔥 Additional aggressive discovery found {len([r for r in additional_results if not isinstance(r, Exception) and r.get('is_valid')])} more URLs")
        
        # Fallback to less aggressive discovery for moderate failure rates
        elif len(result['valid_urls']) < 2 and result['invalid_urls']:
            print("⚠️ Standard fallback: Few valid URLs found, attempting site discovery...")
            
            # Try to discover from failed domains (original logic)
            discovery_tasks = []
            processed_domains = set()
            
            for invalid_url_data in result['invalid_urls'][:2]:  # Limit to 2 domains
                parsed = urlparse(invalid_url_data['url'])
                domain = parsed.netloc
                
                if domain and domain not in processed_domains:
                    processed_domains.add(domain)
                    discovery_tasks.append(
                        validator.discover_site_urls(f"https://{domain}", user_query)
                    )
            
            if discovery_tasks:
                discovery_results = await asyncio.gather(*discovery_tasks, return_exceptions=True)
                
                all_discovered = []
                for discovery_result in discovery_results:
                    if not isinstance(discovery_result, Exception):
                        all_discovered.extend(discovery_result)
                
                # Validate discovered URLs
                if all_discovered:
                    discovered_validation_tasks = [
                        validator.validate_url(url) for url in all_discovered[:10]  # Limit to 10
                    ]
                    discovered_validation_results = await asyncio.gather(
                        *discovered_validation_tasks, return_exceptions=True
                    )
                    
                    for disc_result in discovered_validation_results:
                        if not isinstance(disc_result, Exception) and disc_result['is_valid']:
                            result['discovered_alternatives'].append(disc_result)
    
    # Compile summary
    result['validation_summary'] = {
        'total_urls_checked': len(urls),
        'valid_urls_found': len(result['valid_urls']),
        'invalid_urls_found': len(result['invalid_urls']),
        'alternatives_discovered': len(result['discovered_alternatives']),
        'success_rate': len(result['valid_urls']) / len(urls) if urls else 0
    }
    
    result['total_processing_time'] = time.time() - start_time
    
    return result
