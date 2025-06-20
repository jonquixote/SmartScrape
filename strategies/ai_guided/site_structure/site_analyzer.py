"""
Site Structure Analyzer Module

Analyzes website structure to determine the optimal crawling approach.
"""

import re
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Set
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup

import google.generativeai as genai

from components.pagination_handler import PaginationHandler

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
    
    def __init__(self, response_cache=None, use_ai=True):
        """
        Initialize the SiteStructureAnalyzer.
        
        Args:
            response_cache: Cache for AI responses
            use_ai: Whether to use AI for analysis
        """
        self.response_cache = response_cache
        self.use_ai = use_ai
        self.pagination_handler = PaginationHandler()
        
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
        
    async def analyze(self, url: str, html_content: str, crawl_intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the structure of a website.
        
        Args:
            url: URL of the website
            html_content: HTML content of the page
            crawl_intent: User's crawl intent
            
        Returns:
            Dictionary with site structure analysis
        """
        # Start with basic analysis
        analysis = {
            "site_type": "unknown",
            "has_search": False,
            "has_pagination": False,
            "has_sitemap": False,
            "recommended_approach": "deep_crawl",  # Default to deep crawl
            "important_sections": [],
            "path_recommendations": []
        }
        
        try:
            # Parse the HTML using BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract domain name
            domain = urlparse(url).netloc
            
            # Try to determine if it's a homepage
            is_homepage = self._is_homepage(url)
            
            # Get page title
            title = soup.title.string if soup.title else ""
            
            # Check for listings or detail page indicators
            self._identify_page_type(html_content, analysis)
            
            # Check for search functionality
            search_forms = self._detect_search_functionality(soup)
            analysis["has_search"] = bool(search_forms)
            
            # Check for pagination
            pagination_info = await self.pagination_handler.detect_pagination_type(html_content, url)
            analysis["has_pagination"] = pagination_info["has_pagination"]
            
            # Check for sitemap links
            sitemap_links = self._find_sitemap_links(soup, url)
            analysis["has_sitemap"] = bool(sitemap_links)
            if sitemap_links:
                analysis["sitemap_links"] = sitemap_links
            
            # Identify important navigation paths
            important_links = self._find_important_links(soup, url, crawl_intent)
            analysis["important_links"] = important_links
            
            # If we're on the homepage and need to crawl deeply, find the best path
            if is_homepage:
                deep_crawl_paths = self._find_deep_crawl_paths(soup, url, crawl_intent)
                analysis["deep_crawl_paths"] = deep_crawl_paths
                
                if len(deep_crawl_paths) > 0:
                    analysis["path_recommendations"].append({
                        "type": "start_paths",
                        "paths": deep_crawl_paths[:3],  # Recommend top 3 paths
                        "reason": "These paths appear to lead to the main content sections relevant to your query."
                    })
                    
            # Determine site type and recommended approach using AI if available
            if self.use_ai:
                ai_analysis = await self._analyze_with_ai(url, html_content, crawl_intent)
                # Merge AI analysis with our basic analysis
                for key, value in ai_analysis.items():
                    if key not in analysis or not analysis[key]:
                        analysis[key] = value
            else:
                # Use heuristic approach to determine site type
                site_type = self._determine_site_type_heuristic(soup, url, title)
                analysis["site_type"] = site_type
            
            # Determine the recommended crawling approach
            analysis["recommended_approach"] = self._determine_crawling_approach(analysis, crawl_intent)
            
        except Exception as e:
            logger.error(f"Error analyzing site structure: {str(e)}")
            
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
            sitemap_url = f"{scheme}://{domain}{path}"
            if sitemap_url not in sitemap_links:
                sitemap_links.append(sitemap_url)
                
        return sitemap_links
    
    def _find_important_links(self, soup: BeautifulSoup, base_url: str, crawl_intent: Dict[str, Any]) -> List[str]:
        """
        Find important links on the page based on crawl intent.
        
        Args:
            soup: BeautifulSoup object of the page
            base_url: Base URL for resolving relative links
            crawl_intent: User's crawl intent
            
        Returns:
            List of important URLs
        """
        important_links = []
        
        # Look for links that match the crawl intent
        for a in soup.find_all("a"):
            href = a.get("href", "")
            text = a.get_text().lower().strip()
            
            # Check if link matches any of the intent keywords
            if any(keyword in text for keyword in crawl_intent.get("keywords", [])):
                important_links.append(urljoin(base_url, href))
                
        return important_links
    
    def _find_deep_crawl_paths(self, soup: BeautifulSoup, base_url: str, crawl_intent: Dict[str, Any]) -> List[str]:
        """
        Find deep crawl paths on the homepage.
        
        Args:
            soup: BeautifulSoup object of the page
            base_url: Base URL for resolving relative links
            crawl_intent: User's crawl intent
            
        Returns:
            List of deep crawl paths
        """
        deep_crawl_paths = []
        
        # Look for links that match the crawl intent
        for a in soup.find_all("a"):
            href = a.get("href", "")
            text = a.get_text().lower().strip()
            
            # Check if link matches any of the intent keywords
            if any(keyword in text for keyword in crawl_intent.get("keywords", [])):
                deep_crawl_paths.append(urljoin(base_url, href))
                
        return deep_crawl_paths
    
    async def _analyze_with_ai(self, url: str, html_content: str, crawl_intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the site structure using AI.
        
        Args:
            url: URL of the website
            html_content: HTML content of the page
            crawl_intent: User's crawl intent
            
        Returns:
            Dictionary with AI analysis
        """
        # Try to use cached analysis first
        cache_key = f"site_analysis:{url}"
        
        if self.response_cache:
            cached_analysis = self.response_cache.get(cache_key)
            if cached_analysis:
                logger.info(f"Using cached AI analysis for {url}")
                return cached_analysis
        
        # Create a condensed version of HTML to send to AI
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract key elements to reduce token usage
        title = soup.title.string if soup.title else ""
        meta_description = ""
        meta_description_tag = soup.find("meta", {"name": "description"})
        if meta_description_tag and meta_description_tag.get("content"):
            meta_description = meta_description_tag["content"]
        
        # Extract main content areas and navigation
        main_content = soup.find("main") or soup.find(id=["main", "content", "main-content"]) or \
                       soup.find(class_=["main", "content", "main-content"])
        navigation = soup.find("nav") or soup.find(id=["nav", "navigation"]) or \
                     soup.find(class_=["nav", "navigation", "menu"])
        
        # Prepare a condensed HTML representation
        main_content_html = main_content.decode() if main_content else ""
        nav_html = navigation.decode() if navigation else ""
        
        # Get link patterns
        links = soup.find_all("a", href=True)
        link_patterns = []
        for link in links[:20]:  # Limit to first 20 links
            href = link.get("href", "")
            text = link.get_text().strip()
            if href and not href.startswith(("javascript:", "#", "mailto:", "tel:")):
                link_patterns.append({"href": href, "text": text})
        
        # Create analysis prompt
        domain = urlparse(url).netloc
        intent_keywords = ", ".join(crawl_intent.get("keywords", []))
        
        prompt = f"""
        Analyze this website information and return a JSON structure describing the site:
        
        URL: {url}
        Domain: {domain}
        Title: {title}
        Meta Description: {meta_description}
        User Intent Keywords: {intent_keywords}
        
        Link Patterns (sample):
        {json.dumps(link_patterns, indent=2)}
        
        Provide the following information in your response (as JSON):
        1. "site_type": One of ["e_commerce", "blog", "news", "real_estate", "job_listings", "forum", "generic"]
        2. "recommended_approach": One of ["deep_crawl", "shallow_crawl", "search_driven", "api_based", "follow_pagination"]
        3. "important_sections": List of important site sections related to the user intent
        4. "suggested_patterns": URL patterns that likely contain relevant content
        """
        
        # Use Google's Generative AI model if available, otherwise use placeholder for now
        try:
            ai_analysis = {
                "site_type": self._detect_site_type(url, html_content, soup),
                "recommended_approach": "deep_crawl",
                "important_sections": [],
                "suggested_patterns": []
            }
            
            # Implement actual AI call here
            # For example:
            # model = genai.GenerativeModel(model_name="gemini-pro")
            # response = await model.generate_content_async(prompt)
            # ai_analysis = json.loads(response.text)
            
            # Cache the result if cache is available
            if self.response_cache:
                self.response_cache.put(cache_key, ai_analysis)
                logger.info(f"Cached AI analysis for {url}")
                
            return ai_analysis
            
        except Exception as e:
            logger.error(f"Error during AI analysis: {str(e)}")
            return {
                "site_type": "generic",
                "recommended_approach": "deep_crawl",
                "important_sections": [],
                "suggested_patterns": []
            }
    
    def _determine_site_type_heuristic(self, soup: BeautifulSoup, url: str, title: str) -> str:
        """
        Determine the site type using heuristic approach.
        
        Args:
            soup: BeautifulSoup object of the page
            url: URL of the website
            title: Page title
            
        Returns:
            Site type as a string
        """
        # Placeholder for heuristic site type determination
        # This is a placeholder implementation and should be replaced with actual heuristic analysis
        return "e_commerce"
    
    def _determine_crawling_approach(self, analysis: Dict[str, Any], crawl_intent: Dict[str, Any]) -> str:
        """
        Determine the recommended crawling approach based on analysis and crawl intent.
        
        Args:
            analysis: Site structure analysis
            crawl_intent: User's crawl intent
            
        Returns:
            Recommended crawling approach as a string
        """
        # Placeholder for determining crawling approach
        # This is a placeholder implementation and should be replaced with actual logic
        return "deep_crawl"
    
    def _extract_content_selectors(
        self, 
        site_type: str, 
        html_content: str, 
        crawl_intent: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """
        Extract CSS selectors for key content elements
        
        Args:
            site_type: The type of website
            html_content: The HTML content of the page
            crawl_intent: The crawl intent dictionary
            
        Returns:
            Dictionary of CSS selectors by element type
        """
        selectors = {}
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Get generic selectors based on site type
            if site_type == "e_commerce":
                selectors = {
                    "listing_containers": [".product", ".item", "[class*='product']", "[class*='item']"],
                    "title_selectors": ["h2", "h3", ".product-title", ".name"],
                    "price_selectors": [".price", "[class*='price']", "[data-price]"],
                    "image_selectors": ["img", ".product-image", "[class*='image']"],
                    "description_selectors": [".description", "[class*='description']", "[data-description]"]
                }
            elif site_type == "blog" or site_type == "news":
                selectors = {
                    "article_containers": ["article", ".post", ".blog-post", "[class*='article']"],
                    "title_selectors": ["h1", "h2", ".post-title", ".title"],
                    "date_selectors": [".date", "[class*='date']", "time", "[datetime]"],
                    "author_selectors": [".author", "[class*='author']", "[rel='author']"],
                    "content_selectors": [".content", ".post-content", "article"]
                }
            elif site_type == "real_estate":
                selectors = {
                    "listing_containers": [".property", ".listing", "[class*='property']", "[class*='listing']"],
                    "address_selectors": [".address", "[class*='address']", "[itemprop='address']"],
                    "price_selectors": [".price", "[class*='price']", "[data-price]"],
                    "features_selectors": [".features", ".details", "[class*='features']", "[class*='details']"],
                    "image_selectors": ["img", ".property-image", "[class*='image']"]
                }
            elif site_type == "job_listings":
                selectors = {
                    "job_containers": [".job", ".posting", "[class*='job']", "[class*='posting']"],
                    "title_selectors": ["h2", "h3", ".job-title", ".title"],
                    "company_selectors": [".company", "[class*='company']"],
                    "location_selectors": [".location", "[class*='location']"],
                    "description_selectors": [".description", "[class*='description']"]
                }
            else:
                # Generic selectors for any site
                selectors = {
                    "item_containers": [".item", "article", "[class*='item']", "[class*='card']"],
                    "title_selectors": ["h1", "h2", "h3", ".title"],
                    "link_selectors": ["a", ".more", "[class*='link']"],
                    "description_selectors": [".description", "p", "[class*='description']"],
                    "image_selectors": ["img", "[class*='image']"]
                }
            
        except Exception as e:
            print(f"Error extracting content selectors: {str(e)}")
            
        return selectors
    
    def _identify_priority_patterns(
        self, 
        site_type: str, 
        html_content: str,
        crawl_intent: Dict[str, Any]
    ) -> List[str]:
        """
        Identify URL patterns that likely contain target content
        
        Args:
            site_type: The type of website
            html_content: The HTML content of the page
            crawl_intent: The crawl intent dictionary
            
        Returns:
            List of priority URL patterns
        """
        priority_patterns = []
        
        # Base patterns on site type
        if site_type == "e_commerce":
            priority_patterns = ["product", "item", "shop", "store", "category"]
        elif site_type == "blog" or site_type == "news":
            priority_patterns = ["article", "post", "blog", "news"]
        elif site_type == "real_estate":
            priority_patterns = ["property", "home", "house", "listing", "real-estate"]
        elif site_type == "job_listings":
            priority_patterns = ["job", "career", "position", "opening", "employment"]
        
        # Add intent-specific patterns
        keywords = crawl_intent.get("keywords", [])
        for keyword in keywords:
            # Only use keywords that are at least 4 characters long to avoid false matches
            if len(keyword) >= 4:
                priority_patterns.append(keyword.lower())
        
        return priority_patterns
    
    def _detect_site_type(self, url: str, html_content: str, soup: BeautifulSoup) -> str:
        """
        Detect site type based on URL, content, and HTML structure
        
        Args:
            url: The URL being analyzed
            html_content: The HTML content of the page
            soup: BeautifulSoup object
            
        Returns:
            String indicating site type
        """
        # Check URL for clues
        url_lower = url.lower()
        domain = urlparse(url).netloc.lower()
        
        # Check meta tags and title
        title = soup.title.get_text().lower() if soup.title else ""
        
        meta_description = ""
        meta_description_tag = soup.find("meta", {"name": "description"})
        if meta_description_tag and meta_description_tag.get("content"):
            meta_description = meta_description_tag["content"].lower()
            
        # Create a combined text for pattern matching
        combined_text = f"{title} {meta_description} {domain}"
        
        # E-commerce indicators
        ecommerce_patterns = ["shop", "store", "product", "buy", "cart", "checkout"]
        if any(pattern in url_lower for pattern in ecommerce_patterns) or \
           any(pattern in combined_text for pattern in ecommerce_patterns):
            return "e_commerce"
            
        # Blog/news indicators
        blog_news_patterns = ["blog", "news", "article", "post"]
        if any(pattern in url_lower for pattern in blog_news_patterns) or \
           any(pattern in combined_text for pattern in blog_news_patterns):
            return "blog" if "blog" in combined_text else "news"
            
        # Real estate indicators
        real_estate_patterns = ["property", "realty", "real estate", "home", "house", "apartment"]
        if any(pattern in url_lower for pattern in real_estate_patterns) or \
           any(pattern in combined_text for pattern in real_estate_patterns):
            return "real_estate"
            
        # Job listings indicators
        job_patterns = ["job", "career", "employ"]
        if any(pattern in url_lower for pattern in job_patterns) or \
           any(pattern in combined_text for pattern in job_patterns):
            return "job_listings"
            
        # Check for forums/discussion
        forum_patterns = ["forum", "discussion", "community", "board"]
        if any(pattern in url_lower for pattern in forum_patterns) or \
           any(pattern in combined_text for pattern in forum_patterns):
            return "forum"
            
        # Default to generic
        return "generic"
    
    def _detect_listings_directory(
        self, 
        html_content: str, 
        soup: BeautifulSoup, 
        site_type: str
    ) -> bool:
        """
        Detect if the page contains a listings directory
        
        Args:
            html_content: The HTML content of the page
            soup: BeautifulSoup object
            site_type: The type of website
            
        Returns:
            Boolean indicating if a listings directory was detected
        """
        # Check for common directory indicators
        directory_indicators = ["directory", "listings", "browse", "catalog"]
        
        # Look for links containing directory indicators
        for indicator in directory_indicators:
            directory_links = soup.find_all("a", href=lambda href: href and indicator in href.lower())
            if directory_links:
                return True
                
        # Check for grid/list layouts which often indicate directories
        grid_layouts = soup.select(".grid, .list, .catalog, [class*='grid'], [class*='list'], [class*='catalog']")
        if grid_layouts:
            return True
            
        # Check for repeated item patterns (at least 3 similar items suggests a directory)
        item_containers = []
        
        # Use different selectors based on site type
        if site_type == "e_commerce":
            item_containers = soup.select(".product, .item, [class*='product'], [class*='item']")
        elif site_type == "real_estate":
            item_containers = soup.select(".property, .listing, [class*='property'], [class*='listing']")
        elif site_type == "job_listings":
            item_containers = soup.select(".job, .posting, [class*='job'], [class*='posting']")
        elif site_type == "blog" or site_type == "news":
            item_containers = soup.select("article, .post, [class*='article'], [class*='post']")
        else:
            # Generic checks
            item_containers = soup.select(".item, article, [class*='item'], [class*='card']")
            
        # If we found at least 3 similar containers, it's likely a directory
        return len(item_containers) >= 3