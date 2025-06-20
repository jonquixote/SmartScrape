"""
Site Structure Analyzer Module

Analyzes website structure to determine the optimal crawling approach.
"""

import re
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup

from components.pagination_handler import PaginationHandler
from utils.ai_cache import AIResponseCache

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SiteStructureAnalyzer")

class SiteStructureAnalyzer:
    """
    Analyzes the structure of a website to determine the optimal crawling approach.
    
    This class:
    - Identifies site type (e-commerce, blog, listing site, etc.)
    - Detects search functionality
    - Analyzes navigation structure
    - Determines optimal crawling paths based on user intent
    """
    
    def __init__(self, ai_cache: Optional[AIResponseCache] = None):
        """
        Initialize the SiteStructureAnalyzer.
        
        Args:
            ai_cache: Optional AIResponseCache instance for caching analysis results
        """
        self.pagination_handler = PaginationHandler()
        self.ai_cache = ai_cache or AIResponseCache(cache_dir=".structure_analysis_cache")
        
        # Common patterns for different site sections
        self.section_patterns = {
            "listings": [
                "listings", "properties", "products", "catalog", 
                "homes", "houses", "real-estate", "for-sale",
                "apartments", "condos", "search-results", "search"
            ],
            "detail": [
                "detail", "property", "product", "listing", 
                "item", "home", "house", "mls"
            ],
            "search": [
                "search", "find", "lookup", "filter", "browse"
            ],
            "sitemap": [
                "sitemap", "site-map", "site_map"
            ],
            "api": [
                "api", "graphql", "data", "json", "ajax"
            ]
        }
    
    def _make_hashable(self, obj: Any) -> Union[Tuple, str, int, float, bool, None]:
        """
        Convert an object to a hashable type recursively.
        
        Args:
            obj: Object to convert
            
        Returns:
            A hashable equivalent of the object
        """
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, dict):
            return tuple(sorted((k, self._make_hashable(v)) for k, v in obj.items()))
        elif isinstance(obj, (list, tuple, set)):
            return tuple(self._make_hashable(x) for x in obj)
        else:
            # For any other types, convert to string
            return str(obj)
        
    async def analyze(self, url: str, html_content: str, user_intent: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze the structure of a website to determine optimal crawling approach.
        
        Args:
            url: URL of the website
            html_content: HTML content of the page
            user_intent: User's intent for crawling
            
        Returns:
            Dictionary with site structure analysis results
        """
        logger.info("Analyzing site structure for %s", url)
        
        # Parse the HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Get the domain
        domain = urlparse(url).netloc
        
        # Detect the site type
        site_type = self._detect_site_type(url, soup, user_intent)
        
        # Base analysis results
        analysis = {
            "domain": domain,
            "site_type": site_type,
            "is_homepage": self._is_homepage(url),
            "has_search": False,             # Will be set below
            "search_form": None,             # Will be set if search form is found
            "has_sitemap": False,            # Will be set below
            "sitemap_links": [],             # Will be set if sitemap is found
            "has_pagination": False,         # Will be set below
            "page_type": "unknown",          # Will be set based on content analysis
            "important_links": [],           # Will be populated with important links
            "deep_crawl_paths": [],          # Will be populated with paths for deep crawling
            "recommended_approach": "deep_crawl"  # Default approach
        }
        
        # Check for search functionality (enhanced with more rigorous checks)
        search_form_info = self._find_search_form(soup, url)
        analysis["has_search"] = search_form_info["has_search"]
        analysis["search_form"] = search_form_info["form_data"] if search_form_info["has_search"] else None
        
        # PRIORITIZE SEARCH: Always look for search links even if a form wasn't found
        search_links = self._find_search_links(soup, url)
        if search_links and not analysis["has_search"]:
            analysis["has_search"] = True
            analysis["search_links"] = search_links
        
        # Check for sitemap
        sitemap_info = self._find_sitemap(soup, url)
        analysis["has_sitemap"] = sitemap_info["has_sitemap"]
        analysis["sitemap_links"] = sitemap_info["links"] if sitemap_info["has_sitemap"] else []
        
        # Check if this is a search results page
        if self._is_search_results_page(soup, url):
            analysis["page_type"] = "search_results"
            analysis["has_pagination"] = self._has_pagination(soup, url)
        
        # Check if this is a listing page
        elif self._is_listing_page(soup, url, site_type):
            analysis["page_type"] = "listing_page"
            analysis["has_pagination"] = self._has_pagination(soup, url)
        
        # Determine optimal crawling approach
        if analysis["has_search"] and "search" in user_intent.get("extract_description", "").lower():
            analysis["recommended_approach"] = "search"
            logger.info("Recommended approach: Use search functionality")
        elif analysis["has_sitemap"]:
            analysis["recommended_approach"] = "sitemap"
            logger.info("Recommended approach: Use sitemap")
        else:
            analysis["recommended_approach"] = "deep_crawl"
            logger.info("Recommended approach: Deep crawl")
        
        # Extract important links for deep crawling
        analysis["important_links"] = self._extract_important_links(soup, url, site_type, user_intent)
        
        # Identify promising paths for deeper crawling
        analysis["deep_crawl_paths"] = self._identify_deep_crawl_paths(soup, url, site_type, user_intent)
        
        return analysis
    
    def _is_homepage(self, url: str) -> bool:
        """
        Determine if the URL is likely a homepage.
        
        Args:
            url: URL to check
            
        Returns:
            Boolean indicating whether this is likely a homepage
        """
        parsed = urlparse(url)
        path = parsed.path
        
        # If path is empty or just "/"
        if not path or path == "/":
            return True
            
        # If path is very short with no additional directories
        if len(path.strip("/").split("/")) == 1 and path.strip("/") in ["home", "index", "main"]:
            return True
            
        return False
    
    def _identify_page_type(self, html_content: str, analysis: Dict[str, Any]) -> None:
        """
        Identify the type of page (listing, detail, etc.)
        
        Args:
            html_content: HTML content of the page
            analysis: Analysis dictionary to update
        """
        # Check if page contains multiple listings
        listing_patterns = [
            r'<div\s+class=["\'][^"\']*(?:listing|property|product|item|result)[^"\']*["\']',
            r'<article\s+class=["\'][^"\']*(?:listing|property|product|item)[^"\']*["\']',
            r'<li\s+class=["\'][^"\']*(?:listing|property|product|item)[^"\']*["\']'
        ]
        
        listing_count = 0
        for pattern in listing_patterns:
            matches = re.findall(pattern, html_content, re.IGNORECASE)
            listing_count += len(matches)
        
        if listing_count >= 3:  # If there are 3 or more listing elements
            analysis["page_type"] = "listing_page"
        else:
            # Check if it's a detail page
            detail_patterns = [
                r'<div\s+class=["\'][^"\']*(?:detail|property-detail|product-detail)[^"\']*["\']',
                r'<section\s+class=["\'][^"\']*(?:detail|property-detail|product-detail)[^"\']*["\']',
                r'itemprop=["\'](description|price|offer|product)["\']'
            ]
            
            for pattern in detail_patterns:
                if re.search(pattern, html_content, re.IGNORECASE):
                    analysis["page_type"] = "detail_page"
                    break
            
        # Default if no specific type was detected
        if "page_type" not in analysis:
            analysis["page_type"] = "other"
    
    def _detect_search_functionality(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Detect search functionality on the page.
        
        Args:
            soup: BeautifulSoup object of the page
            
        Returns:
            List of dictionaries with search form information
        """
        search_forms = []
        
        # Look for search forms
        forms = soup.find_all("form")
        for form in forms:
            # Check if form has search-related attributes
            action = form.get("action", "").lower()
            form_id = form.get("id", "").lower()
            form_class = form.get("class", [])
            form_class = " ".join(form_class).lower() if form_class else ""
            
            is_search = False
            
            # Check form attributes for search indicators
            if any(term in action for term in ["search", "find", "query"]):
                is_search = True
            elif any(term in form_id for term in ["search", "find", "query"]):
                is_search = True
            elif any(term in form_class for term in ["search", "find", "query", "filter"]):
                is_search = True
                
            # Check for search input fields
            search_inputs = form.find_all("input", {"type": ["search", "text"]})
            for input_field in search_inputs:
                input_name = input_field.get("name", "").lower()
                input_id = input_field.get("id", "").lower()
                placeholder = input_field.get("placeholder", "").lower()
                
                if any(term in input_name for term in ["search", "q", "query", "find", "keyword"]):
                    is_search = True
                elif any(term in input_id for term in ["search", "q", "query", "find", "keyword"]):
                    is_search = True
                elif any(term in placeholder for term in ["search", "find", "type", "enter"]):
                    is_search = True
            
            if is_search:
                # Get form action URL
                action_url = form.get("action", "")
                method = form.get("method", "get").lower()
                
                # Get input fields
                input_fields = []
                for input_field in form.find_all("input"):
                    input_fields.append({
                        "name": input_field.get("name", ""),
                        "type": input_field.get("type", ""),
                        "id": input_field.get("id", ""),
                        "placeholder": input_field.get("placeholder", "")
                    })
                
                search_forms.append({
                    "action": action_url,
                    "method": method,
                    "inputs": input_fields
                })
        
        return search_forms
    
    def _find_sitemap_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """
        Find sitemap links on the page.
        
        Args:
            soup: BeautifulSoup object of the page
            base_url: Base URL for resolving relative links
            
        Returns:
            List of sitemap URLs
        """
        sitemap_links = []
        
        # Look for links to sitemaps
        for a in soup.find_all("a"):
            href = a.get("href", "")
            text = a.get_text().lower().strip()
            
            # Check for common sitemap indicators in link text or href
            if "sitemap" in href.lower() or "sitemap" in text:
                sitemap_links.append(urljoin(base_url, href))
                
        # Also check for common sitemap paths
        common_sitemap_paths = [
            "/sitemap.xml", 
            "/sitemap_index.xml",
            "/sitemap/", 
            "/site-map", 
            "/sitemap.html"
        ]
        
        domain = urlparse(base_url).netloc
        scheme = urlparse(base_url).scheme
        
        for path in common_sitemap_paths:
            sitemap_links.append(f"{scheme}://{domain}{path}")
            
        return list(set(sitemap_links))  # Remove duplicates
    
    def _find_important_links(self, soup: BeautifulSoup, base_url: str, crawl_intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find important navigation links based on the user's crawl intent.
        
        Args:
            soup: BeautifulSoup object of the page
            base_url: Base URL for resolving relative links
            crawl_intent: User's crawl intent
            
        Returns:
            List of important links with relevance scores
        """
        important_links = []
        keywords = crawl_intent.get("keywords", [])
        location_data = crawl_intent.get("location_data", {})
        
        # Extract location information
        location_terms = []
        if isinstance(location_data, dict):
            location_terms.extend([v for k, v in location_data.items() if v and isinstance(v, str)])
        elif isinstance(location_data, str):
            location_terms.append(location_data)
            
        # Combine keywords and location terms
        search_terms = keywords + location_terms
        
        # Skip if we don't have any search terms
        if not search_terms:
            return []
            
        # Get all links
        for a in soup.find_all("a"):
            href = a.get("href", "")
            text = a.get_text().strip()
            
            # Skip if no href or text
            if not href or not text:
                continue
                
            # Skip anchors, javascript, and mailto links
            if href.startswith('#') or href.startswith('javascript:') or href.startswith('mailto:'):
                continue
                
            # Calculate relevance score based on:
            # 1. Presence of search terms in text or href
            # 2. URL structure indicators
            
            relevance = 0.0
            explanation = []
            
            # Check for search terms in text and href
            lower_text = text.lower()
            lower_href = href.lower()
            
            for term in search_terms:
                if not term or not isinstance(term, str):
                    continue
                    
                term = term.lower()
                if term in lower_text:
                    relevance += 0.3
                    explanation.append(f"Contains term '{term}' in link text")
                if term in lower_href:
                    relevance += 0.2
                    explanation.append(f"Contains term '{term}' in URL")
            
            # Check for section indicators in URL
            for section_type, patterns in self.section_patterns.items():
                for pattern in patterns:
                    if pattern in lower_href:
                        relevance += 0.2
                        explanation.append(f"URL contains {section_type} indicator '{pattern}'")
                        break
            
            # Normalize relevance score
            if relevance > 0:
                important_links.append({
                    "url": urljoin(base_url, href),
                    "text": text,
                    "relevance": min(1.0, relevance),  # Cap at 1.0
                    "explanation": " | ".join(explanation)
                })
        
        # Sort by relevance (descending)
        important_links.sort(key=lambda x: x["relevance"], reverse=True)
        
        return important_links
    
    def _find_deep_crawl_paths(self, soup: BeautifulSoup, base_url: str, crawl_intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find the best paths for deep crawling based on the user's intent.
        
        Args:
            soup: BeautifulSoup object of the page
            base_url: Base URL for resolving relative links
            crawl_intent: User's crawl intent
            
        Returns:
            List of deep crawl paths with relevance scores
        """
        # Start with important links
        deep_crawl_paths = self._find_important_links(soup, base_url, crawl_intent)
        
        # If we didn't find any important links based on keywords,
        # fall back to section patterns
        if not deep_crawl_paths:
            # Look specifically for listing sections
            for a in soup.find_all("a"):
                href = a.get("href", "")
                text = a.get_text().strip()
                
                # Skip if no href or text
                if not href or not text:
                    continue
                    
                # Skip anchors, javascript, and mailto links
                if href.startswith('#') or href.startswith('javascript:') or href.startswith('mailto:'):
                    continue
                
                lower_href = href.lower()
                lower_text = text.lower()
                
                # Check for section indicators in URL
                for pattern in self.section_patterns["listings"]:
                    if pattern in lower_href or pattern in lower_text:
                        relevance = 0.7  # High relevance for listing sections
                        explanation = f"URL contains listings indicator '{pattern}'"
                        
                        deep_crawl_paths.append({
                            "url": urljoin(base_url, href),
                            "text": text,
                            "relevance": relevance,
                            "explanation": explanation
                        })
                        break
        
        # Sort by relevance (descending)
        deep_crawl_paths.sort(key=lambda x: x["relevance"], reverse=True)
        
        return deep_crawl_paths
    
    def _determine_site_type_heuristic(self, soup: BeautifulSoup, url: str, title: str) -> str:
        """
        Determine the site type using heuristic approach.
        
        Args:
            soup: BeautifulSoup object of the page
            url: URL of the page
            title: Title of the page
            
        Returns:
            Site type
        """
        # Use cached result if available
        cache_context = {"url": url, "title": title}
        cached_type = self.ai_cache.get("site_type_determination", context=cache_context)
        if cached_type:
            logger.debug(f"Using cached site type for {url}: {cached_type}")
            return cached_type
            
        # Check URL and title for indicators
        url_lower = url.lower()
        title_lower = title.lower() if title else ""
        
        # Check for real estate indicators
        real_estate_terms = ["real estate", "property", "properties", "home", "homes", 
                             "house", "houses", "apartment", "realty", "realtor"]
        
        if any(term in url_lower for term in real_estate_terms) or \
           any(term in title_lower for term in real_estate_terms):
            site_type = "real_estate"
            self.ai_cache.put("site_type_determination", site_type, context=cache_context)
            return site_type
            
        # Check for e-commerce indicators
        ecommerce_terms = ["shop", "store", "product", "buy", "purchase", "cart", 
                           "checkout", "order", "catalog"]
                           
        if any(term in url_lower for term in ecommerce_terms) or \
           any(term in title_lower for term in ecommerce_terms):
            site_type = "ecommerce"
            self.ai_cache.put("site_type_determination", site_type, context=cache_context)
            return site_type
            
        # Check for blog indicators
        blog_terms = ["blog", "article", "news", "post", "journal"]
        
        if any(term in url_lower for term in blog_terms) or \
           any(term in title_lower for term in blog_terms):
            site_type = "blog"
            self.ai_cache.put("site_type_determination", site_type, context=cache_context)
            return site_type
            
        # Check for forum indicators
        forum_terms = ["forum", "community", "discussion", "board", "thread"]
        
        if any(term in url_lower for term in forum_terms) or \
           any(term in title_lower for term in forum_terms):
            site_type = "forum"
            self.ai_cache.put("site_type_determination", site_type, context=cache_context)
            return site_type
            
        # Check for content-heavy sites
        meta_description = ""
        meta_desc = soup.find("meta", {"name": "description"})
        if meta_desc:
            meta_description = meta_desc.get("content", "").lower()
            
        if len(meta_description) > 150 or len(soup.find_all("p")) > 10:
            site_type = "content"
            self.ai_cache.put("site_type_determination", site_type, context=cache_context)
            return site_type
            
        # Default
        site_type = "general"
        self.ai_cache.put("site_type_determination", site_type, context=cache_context)
        return site_type
    
    def _determine_crawling_approach(self, analysis: Dict[str, Any], crawl_intent: Dict[str, Any]) -> str:
        """
        Determine the recommended crawling approach.
        
        Args:
            analysis: Site structure analysis
            crawl_intent: User's crawl intent
            
        Returns:
            Recommended crawling approach
        """
        # Check if we're on a homepage and have found deep crawl paths
        if analysis.get("is_homepage", False) and "deep_crawl_paths" in analysis and analysis["deep_crawl_paths"]:
            return "targeted_deep_crawl"
            
        # Check if the site has search and search would be beneficial
        if analysis.get("has_search", False) and ("keywords" in crawl_intent or "location_data" in crawl_intent):
            return "search_first"
            
        # Check if the site has a sitemap
        if analysis.get("has_sitemap", False):
            return "sitemap_based"
            
        # Default to deep crawl
        return "deep_crawl"
        
    def get_listing_page_selectors(self, site_type: str) -> Dict[str, List[str]]:
        """
        Get CSS selectors for finding content on listing pages based on site type.
        
        Args:
            site_type: Type of website
            
        Returns:
            Dictionary of CSS selectors for different elements
        """
        # Use cached result if available
        cached_selectors = self.ai_cache.get("listing_page_selectors", context={"site_type": site_type})
        if cached_selectors:
            return cached_selectors
            
        # Common selectors across all site types
        common_selectors = {
            "container": [
                ".listing", ".list", ".result", ".results", 
                "[class*='listing']", "[class*='result']",
                "article", ".product", "[class*='product']",
                ".item", "[class*='item']"
            ],
            "link": [
                "a", ".title a", ".name a", "h2 a", "h3 a"
            ]
        }
        
        # Site-specific selectors
        if site_type == "real_estate":
            selectors = {
                "container": [
                    ".property", ".listing", ".real-estate-item", 
                    "[class*='property']", "[class*='listing']", 
                    "[itemtype*='Product']", "[itemtype*='RealEstateListing']",
                    "article", ".card", "[class*='card']"
                ],
                "link": [
                    "a[href*='property']", "a[href*='listing']", "a[href*='home']",
                    "a[href*='house']", "a.property-link", ".property-title a",
                    "h2 a", "h3 a", ".listing-title a"
                ],
                "price": [
                    ".price", "[class*='price']", "[itemprop='price']",
                    "[data-price]", ".property-price", ".listing-price"
                ],
                "image": [
                    "img", ".photo", ".image", "[class*='image']", 
                    "[class*='photo']", "[itemprop='image']"
                ]
            }
        elif site_type == "ecommerce":
            selectors = {
                "container": [
                    ".product", ".item", "[class*='product']", 
                    "[class*='item']", "[itemtype*='Product']",
                    "article", ".card", "[class*='card']"
                ],
                "link": [
                    "a[href*='product']", "a[href*='item']", "a.product-link",
                    ".product-title a", "h2 a", "h3 a", ".item-title a"
                ],
                "price": [
                    ".price", "[class*='price']", "[itemprop='price']",
                    "[data-price]", ".product-price", ".item-price"
                ],
                "image": [
                    "img", ".photo", ".image", "[class*='image']", 
                    "[class*='photo']", "[itemprop='image']"
                ]
            }
        else:
            selectors = common_selectors
            
        # Cache the result
        self.ai_cache.put("listing_page_selectors", selectors, context={"site_type": site_type})
        return selectors

    def _find_search_form(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """
        Find search forms on the page.
        
        Args:
            soup: BeautifulSoup object of the page
            url: URL of the page
            
        Returns:
            Dictionary with search form information
        """
        logger.debug(f"Looking for search forms on {url}")
        
        result = {
            "has_search": False,
            "form_data": None
        }
        
        # Look for search forms
        forms = soup.find_all("form")
        for form in forms:
            # Get form attributes
            action = form.get("action", "").lower()
            form_id = form.get("id", "").lower()
            form_name = form.get("name", "").lower()
            form_class = " ".join(form.get("class", [])).lower() if form.get("class") else ""
            method = form.get("method", "get").lower()
            
            # Check form attributes for search indicators
            is_search = False
            
            # Check form attributes
            search_indicators = ["search", "find", "query", "q=", "lookup", "filter"]
            if any(indicator in action for indicator in search_indicators):
                is_search = True
            if any(indicator in form_id for indicator in search_indicators):
                is_search = True
            if any(indicator in form_name for indicator in search_indicators):
                is_search = True
            if any(indicator in form_class for indicator in search_indicators):
                is_search = True
                
            # If no search indicators found in form attributes, check inputs
            if not is_search:
                inputs = form.find_all("input")
                for input_tag in inputs:
                    input_type = input_tag.get("type", "").lower()
                    input_name = input_tag.get("name", "").lower()
                    input_id = input_tag.get("id", "").lower()
                    input_class = " ".join(input_tag.get("class", [])).lower() if input_tag.get("class") else ""
                    placeholder = input_tag.get("placeholder", "").lower()
                    
                    # Check input attributes for search indicators
                    if input_type == "search":
                        is_search = True
                        break
                    if any(indicator in input_name for indicator in ["search", "query", "q", "find", "keyword"]):
                        is_search = True
                        break
                    if any(indicator in input_id for indicator in ["search", "query", "q", "find", "keyword"]):
                        is_search = True
                        break
                    if any(indicator in input_class for indicator in ["search", "query", "q", "find", "keyword"]):
                        is_search = True
                        break
                    if any(indicator in placeholder for indicator in ["search", "find", "keyword", "enter"]):
                        is_search = True
                        break
                
                # Check for submit buttons with search indicators
                buttons = form.find_all(["button", "input[type='submit']"])
                for button in buttons:
                    button_text = button.get_text().lower()
                    button_value = button.get("value", "").lower()
                    button_class = " ".join(button.get("class", [])).lower() if button.get("class") else ""
                    
                    if any(term in button_text for term in ["search", "find", "go"]):
                        is_search = True
                        break
                    if any(term in button_value for term in ["search", "find", "go"]):
                        is_search = True
                        break
                    if any(term in button_class for term in ["search", "find"]):
                        is_search = True
                        break
            
            if is_search:
                # Collect information about the search form
                inputs = []
                for input_tag in form.find_all("input"):
                    if input_tag.get("type") not in ["submit", "button", "reset", "hidden"]:
                        inputs.append({
                            "name": input_tag.get("name", ""),
                            "type": input_tag.get("type", "text"),
                            "id": input_tag.get("id", ""),
                            "placeholder": input_tag.get("placeholder", ""),
                            "required": input_tag.has_attr("required"),
                            "value": input_tag.get("value", "")
                        })
                
                # Special handling for select elements
                for select in form.find_all("select"):
                    options = []
                    for option in select.find_all("option"):
                        options.append({
                            "value": option.get("value", ""),
                            "text": option.get_text().strip()
                        })
                    
                    inputs.append({
                        "name": select.get("name", ""),
                        "type": "select",
                        "id": select.get("id", ""),
                        "options": options,
                        "required": select.has_attr("required")
                    })
                
                # Build form data
                action_url = form.get("action", "")
                if action_url:
                    if not action_url.startswith("http"):
                        action_url = urljoin(url, action_url)
                else:
                    # If no action, form submits to current page
                    action_url = url
                
                form_data = {
                    "action": action_url,
                    "method": method,
                    "id": form_id,
                    "class": form_class,
                    "inputs": inputs
                }
                
                result["has_search"] = True
                result["form_data"] = form_data
                
                # If we found a search form, return immediately
                return result
        
        # If we didn't find a form with clear search indicators, look for isolated search inputs
        if not result["has_search"]:
            # Look for isolated search input elements
            search_inputs = soup.select("input[type='search'], input[name='q'], input[name='query'], input[name='search']")
            if search_inputs:
                result["has_search"] = True
                
                # Try to guess the form submission details
                parent_form = search_inputs[0].find_parent("form")
                
                if parent_form:
                    action_url = parent_form.get("action", url)
                    if not action_url.startswith("http"):
                        action_url = urljoin(url, action_url)
                    
                    method = parent_form.get("method", "get").lower()
                else:
                    # If no parent form, assume GET to the current page
                    action_url = url
                    method = "get"
                
                # Create basic form data
                form_data = {
                    "action": action_url,
                    "method": method,
                    "inputs": [{
                        "name": search_inputs[0].get("name", "q"),
                        "type": search_inputs[0].get("type", "text"),
                        "id": search_inputs[0].get("id", ""),
                        "placeholder": search_inputs[0].get("placeholder", ""),
                        "required": search_inputs[0].has_attr("required"),
                        "value": search_inputs[0].get("value", "")
                    }]
                }
                
                result["form_data"] = form_data
        
        return result
    
    def _find_search_links(self, soup: BeautifulSoup, url: str) -> List[Dict[str, Any]]:
        """
        Find links to search pages if no search form is present on the current page.
        
        Args:
            soup: BeautifulSoup object of the page
            url: URL of the page
            
        Returns:
            List of search link information
        """
        search_links = []
        
        # Look for links with search indicators
        search_indicators = ["search", "find", "lookup", "browse", "filter", "listings", "properties", "homes"]
        
        for a in soup.find_all("a"):
            href = a.get("href", "")
            text = a.get_text().strip()
            
            # Skip if no href or text
            if not href or not text:
                continue
                
            # Skip anchors, javascript, and mailto links
            if href.startswith('#') or href.startswith('javascript:') or href.startswith('mailto:'):
                continue
                
            # Look for search indicators in link text or href
            href_lower = href.lower()
            text_lower = text.lower()
            
            # Check for search indicators in text or href
            if any(indicator in text_lower for indicator in search_indicators) or \
               any(indicator in href_lower for indicator in search_indicators):
                
                # Resolve URL
                full_url = urljoin(url, href)
                
                # Skip links to the current page
                if full_url == url:
                    continue
                
                # Calculate relevance score
                relevance = 0.0
                
                # Higher relevance for "search" specifically
                if "search" in text_lower or "search" in href_lower:
                    relevance += 0.5
                # Medium relevance for other search terms
                elif any(indicator in text_lower for indicator in search_indicators[1:]):
                    relevance += 0.3
                elif any(indicator in href_lower for indicator in search_indicators):
                    relevance += 0.2
                
                # Add search link
                search_links.append({
                    "url": full_url,
                    "text": text,
                    "relevance": relevance
                })
        
        # Check for common search paths if no links found
        if not search_links:
            domain = urlparse(url).netloc
            scheme = urlparse(url).scheme
            base = f"{scheme}://{domain}"
            
            common_search_paths = [
                "/search", "/find", "/lookup", "/browse",
                "/search-mls-listings", "/properties", "/homes",
                "/search-results", "/search.html", "/search.php"
            ]
            
            for path in common_search_paths:
                search_links.append({
                    "url": f"{base}{path}",
                    "text": "Search",
                    "relevance": 0.2,
                    "is_common_path": True
                })
        
        # Sort search links by relevance
        search_links.sort(key=lambda x: x["relevance"], reverse=True)
        
        return search_links
    
    def _find_sitemap(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """
        Find sitemap link on the page.
        
        Args:
            soup: BeautifulSoup object of the page
            url: URL of the page
            
        Returns:
            Dictionary with sitemap information
        """
        result = {
            "has_sitemap": False,
            "links": []
        }
        
        # Look for sitemap links in the page
        sitemap_links = []
        
        # Check for links with sitemap in text or href
        for a in soup.find_all("a"):
            href = a.get("href", "")
            text = a.get_text().strip()
            
            # Skip if no href or text
            if not href or not text:
                continue
                
            # Skip anchors, javascript, and mailto links
            if href.startswith('#') or href.startswith('javascript:') or href.startswith('mailto:'):
                continue
                
            # Look for sitemap indicators
            href_lower = href.lower()
            text_lower = text.lower()
            
            if "sitemap" in href_lower or "sitemap" in text_lower or "site map" in text_lower:
                sitemap_links.append(urljoin(url, href))
        
        # Also check for standard sitemap locations
        domain = urlparse(url).netloc
        scheme = urlparse(url).scheme
        base = f"{scheme}://{domain}"
        
        standard_sitemap_paths = [
            "/sitemap.xml",
            "/sitemap_index.xml",
            "/sitemap",
            "/sitemap.html",
            "/site-map",
            "/site-map.html"
        ]
        
        # Add the first link we found in the page
        if sitemap_links:
            result["has_sitemap"] = True
            result["links"] = sitemap_links
        # If no links found in the page, check if one of the standard paths exists
        elif any(base + path for path in standard_sitemap_paths):
            for path in standard_sitemap_paths:
                sitemap_url = base + path
                result["links"].append(sitemap_url)
            result["has_sitemap"] = True
        
        return result

    def _detect_site_type(self, url: str, soup: BeautifulSoup, user_intent: Dict[str, Any] = None) -> str:
        """
        Detect the type of site (e-commerce, real estate, blog, etc.).
        
        Args:
            url: URL of the website
            soup: BeautifulSoup object of the page
            user_intent: User's intent for crawling
            
        Returns:
            Site type as a string
        """
        # Extract page title
        title = soup.find('title')
        title_text = title.get_text() if title else ""
        
        # Try to determine site type from various signals
        return self._determine_site_type_heuristic(soup, url, title_text)

    def _is_search_results_page(self, soup: BeautifulSoup, url: str) -> bool:
        """
        Determine if the page is a search results page.
        
        Args:
            soup: BeautifulSoup object of the page
            url: URL of the page
            
        Returns:
            Boolean indicating if this is a search results page
        """
        # Check URL for search indicators
        url_lower = url.lower()
        if any(term in url_lower for term in ["search", "find", "query", "q=", "results", "listings"]):
            return True
            
        # Check for common search result indicators in the page
        # Look for search forms at the top
        search_form = soup.find("form", {"role": "search"}) or soup.find("form", {"class": lambda x: x and "search" in x.lower()})
        if search_form:
            return True
            
        # Look for search result containers
        result_containers = [
            soup.find("div", {"id": lambda x: x and "search-results" in x.lower()}),
            soup.find("div", {"class": lambda x: x and "search-results" in x.lower()}),
            soup.find("div", {"class": lambda x: x and "search_results" in x.lower()}),
            soup.find("div", {"class": lambda x: x and "listings" in x.lower()}),
            soup.find("div", {"class": lambda x: x and "properties" in x.lower()})
        ]
        
        if any(result_containers):
            return True
            
        # Look for search result item patterns (at least 3 similar items indicates a results page)
        result_items = soup.select(".result, .search-result, .listing, .property, .item")
        if len(result_items) >= 3:
            return True
            
        # Look for common search results page elements
        search_headings = soup.find_all(["h1", "h2"], string=lambda text: text and any(term in text.lower() for term in ["search", "results", "found", "listings"]))
        if search_headings:
            return True
            
        # Check for search stats that often appear on results pages
        stats_text = soup.find(string=lambda text: text and re.search(r'(\d+)\s+(results|found|properties|listings)', text.lower()))
        if stats_text:
            return True
            
        return False

    def _is_listing_page(self, soup: BeautifulSoup, url: str, site_type: str) -> bool:
        """
        Determine if the page is a listing page (contains multiple items).
        
        Args:
            soup: BeautifulSoup object of the page
            url: URL of the page
            site_type: Type of the site
            
        Returns:
            Boolean indicating if this is a listing page
        """
        # Check URL for listing indicators
        url_lower = url.lower()
        if any(term in url_lower for term in ["listings", "properties", "homes", "catalog", "category", "products"]):
            return True
            
        # Check for common listing page patterns based on site type
        if site_type == "real_estate":
            # Look for multiple property listings
            property_containers = [
                soup.select(".property, .listing, .property-item, .listing-item"),
                soup.select("[itemtype*='RealEstateListing']"),
                soup.select(".properties-grid .property, .listings-grid .listing")
            ]
            
            for container_list in property_containers:
                if len(container_list) >= 2:  # At least 2 items to be a listing page
                    return True
            
        elif site_type == "ecommerce":
            # Look for product grids or lists
            product_containers = [
                soup.select(".product, .item, .product-item"),
                soup.select("[itemtype*='Product']"),
                soup.select(".products-grid .product, .product-list .item")
            ]
            
            for container_list in product_containers:
                if len(container_list) >= 2:
                    return True
        
        else:
            # Generic listing detection for other site types
            # Look for repeating patterns of similar elements
            item_containers = [
                soup.select(".item, .list-item, .card, .grid-item"),
                soup.select("article"),
                soup.select(".grid > div, .list > div")
            ]
            
            for container_list in item_containers:
                if len(container_list) >= 3:
                    return True
        
        # Check for pagination which often indicates a listing page
        pagination = soup.find("div", {"class": lambda x: x and any(term in x.lower() for term in ["pagination", "pager"])})
        if pagination:
            return True
            
        return False
        
    def _has_pagination(self, soup: BeautifulSoup, url: str) -> bool:
        """
        Determine if the page has pagination.
        
        Args:
            soup: BeautifulSoup object of the page
            url: URL of the page
            
        Returns:
            Boolean indicating if the page has pagination
        """
        # Look for common pagination containers
        pagination_containers = [
            soup.find("div", {"class": lambda x: x and "pagination" in x.lower()}),
            soup.find("nav", {"class": lambda x: x and "pagination" in x.lower()}),
            soup.find("ul", {"class": lambda x: x and "pagination" in x.lower()}),
            soup.find("div", {"class": lambda x: x and "pager" in x.lower()}),
            soup.find("div", {"id": lambda x: x and "pagination" in x.lower()})
        ]
        
        if any(pagination_containers):
            return True
            
        # Look for next/prev links
        next_links = [
            soup.find("a", string=lambda x: x and "next" in x.lower()),
            soup.find("a", {"class": lambda x: x and "next" in x.lower()}),
            soup.find("a", {"rel": "next"}),
            soup.find("a", {"aria-label": lambda x: x and "next" in x.lower()})
        ]
        
        if any(next_links):
            return True
            
        # Look for page number links (at least 2 to indicate pagination)
        page_numbers = soup.select("a[href*='page='], a[href*='/page/']")
        if len(page_numbers) >= 2:
            return True
            
        return False

    def _extract_important_links(self, soup: BeautifulSoup, url: str, site_type: str, user_intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find important links based on user intent and site type.
        
        Args:
            soup: BeautifulSoup object of the page
            url: URL of the page
            site_type: Type of the site
            user_intent: User's intent for crawling
            
        Returns:
            List of important links with relevance scores
        """
        # Start with no important links
        important_links = []
        
        # Get keywords from user intent
        keywords = []
        if "keywords" in user_intent:
            keywords.extend(user_intent["keywords"])
        if "location_data" in user_intent:
            location_data = user_intent["location_data"]
            if isinstance(location_data, dict):
                for key, value in location_data.items():
                    if value:
                        keywords.append(value)
            elif isinstance(location_data, str) and location_data:
                keywords.append(location_data)
        
        # Extract all relevant links
        for a in soup.find_all("a"):
            href = a.get("href", "")
            text = a.get_text().strip()
            
            # Skip if no href or text
            if not href or not text:
                continue
                
            # Skip anchors, javascript, and mailto links
            if href.startswith('#') or href.startswith('javascript:') or href.startswith('mailto:'):
                continue
                
            # Full URL
            full_url = urljoin(url, href)
            
            # Calculate relevance based on keywords
            relevance = 0.0
            
            # Check for keywords in link text and href
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    relevance += 0.3
                if keyword.lower() in href.lower():
                    relevance += 0.2
                    
            # Check for listing-related terms in URL
            listing_terms = ["listing", "property", "properties", "homes", "house", 
                            "real-estate", "for-sale", "rent", "realty", "details", 
                            "product", "item"]
                            
            for term in listing_terms:
                if term in href.lower() or term in text.lower():
                    relevance += 0.1
                    break
                    
            # If the link has any relevance, add it to the list
            if relevance > 0:
                important_links.append({
                    "url": full_url,
                    "text": text,
                    "relevance": relevance
                })
        
        # Sort by relevance
        important_links.sort(key=lambda x: x["relevance"], reverse=True)
        
        return important_links
        
    def _identify_deep_crawl_paths(self, soup: BeautifulSoup, url: str, site_type: str, user_intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify the most promising paths for deep crawling.
        
        Args:
            soup: BeautifulSoup object of the page
            url: URL of the page
            site_type: Type of the site
            user_intent: User's intent for crawling
            
        Returns:
            List of promising paths with relevance scores
        """
        # Start with important links as the basis for deep crawl paths
        deep_crawl_paths = self._extract_important_links(soup, url, site_type, user_intent)
        
        # If no important links were found, look for site structure hints
        if not deep_crawl_paths:
            # Look for navigation menus
            nav_elements = soup.find_all(["nav", "div"], {"class": lambda x: x and any(term in (x.lower() if x else "") for term in ["nav", "menu", "navigation"])})
            
            for nav in nav_elements:
                links = nav.find_all("a")
                for link in links:
                    href = link.get("href", "")
                    text = link.get_text().strip()
                    
                    # Skip if no href or text
                    if not href or not text:
                        continue
                        
                    # Skip anchors, javascript, and mailto links
                    if href.startswith('#') or href.startswith('javascript:') or href.startswith('mailto:'):
                        continue
                        
                    # Full URL
                    full_url = urljoin(url, href)
                    
                    # Add as a deep crawl path with a default relevance
                    deep_crawl_paths.append({
                        "url": full_url,
                        "text": text,
                        "relevance": 0.5  # Default relevance for navigation links
                    })
        
        # Ensure we don't have duplicates
        seen_urls = set()
        unique_paths = []
        
        for path in deep_crawl_paths:
            if path["url"] not in seen_urls:
                seen_urls.add(path["url"])
                unique_paths.append(path)
        
        # Sort by relevance
        unique_paths.sort(key=lambda x: x["relevance"], reverse=True)
        
        return unique_paths