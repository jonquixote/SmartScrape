"""
Search Integration Module

Handles search form detection, submission, and result processing
for the AI-guided strategy.
"""

import re
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import urljoin, urlparse, parse_qs, urlunparse, urlencode, quote_plus

from bs4 import BeautifulSoup
from components.search_automation import SearchFormDetector, SearchAutomator, PlaywrightSearchHandler
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
# Use BrowserConfig correctly for JavaScript handling
from crawl4ai.async_configs import BrowserConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime.s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SearchIntegration")

# Helper function to create JavaScript-enabled config
def create_js_config(timeout=30):
    """Create a crawler config with JavaScript support."""
    # Create a crawler config with no parameters (ignore the timeout parameter)
    return CrawlerRunConfig()

class SearchIntegrator:
    """
    Integrates search functionality into the AI-guided strategy.
    
    This class:
    - Discovers search forms on websites
    - Generates appropriate search terms based on user intent
    - Submits search queries and processes results
    - Analyzes search result pages and extracts key information
    """
    
    def __init__(self):
        """Initialize the search integrator with a form detector."""
        self.form_detector = SearchFormDetector()
        
        # Store search history to avoid duplicate searches
        self.search_history = {}
        
        # Store discovered search forms by domain
        self.domain_search_forms = {}
        
    async def discover_search_functionality(self, url: str, html_content: str, crawler: AsyncWebCrawler) -> Dict[str, Any]:
        """
        Discover search functionality on a website.
        
        Args:
            url: URL of the website
            html_content: HTML content of the page
            crawler: AsyncWebCrawler instance
            
        Returns:
            Dictionary with search functionality information
        """
        try:
            # Parse the domain from the URL
            domain = urlparse(url).netloc
            
            # Check if we already discovered search forms for this domain
            if domain in self.domain_search_forms:
                logger.info(f"Using cached search form data for {domain}")
                return self.domain_search_forms[domain]
                
            # Create a search automator instance
            search_automator = SearchAutomator(crawler)
            
            # Detect the domain type for specialized form detection
            try:
                domain_type = search_automator._detect_domain_type(url)
            except AttributeError:
                # Fallback if method doesn\'t exist or has different name
                logger.warning("SearchAutomator._detect_domain_type method not found, using 'general' as default")
                domain_type = 'general'
            
            # Detect search forms
            forms = await self.form_detector.detect_search_forms(html_content, domain_type)
            
            # Check for search links (e.g., "/search" or "/find") if no forms found
            if not forms:
                search_links = self._find_search_links(html_content, url)
                if search_links:
                    logger.info(f"Found {len(search_links)} search links")
                    forms = [{"type": "link", "url": link, "score": score} for link, score in search_links]
            
            search_info = {
                "has_search": bool(forms),
                "search_forms": forms,
                "domain_type": domain_type
            }
            
            if forms:
                search_info["best_form"] = forms[0]
                
            # Cache the results
            self.domain_search_forms[domain] = search_info
                
            return search_info
            
        except Exception as e:
            logger.error(f"Error discovering search functionality: {str(e)}")
            return {"has_search": False, "error": str(e)}
            
    def _find_search_links(self, html_content: str, base_url: str) -> List[Tuple[str, float]]:
        """
        Find links that likely lead to search pages.
        
        Args:
            html_content: HTML content to analyze
            base_url: Base URL for resolving relative links
            
        Returns:
            List of tuples (url, relevance_score)
        """
        search_links = []
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Common text indicators of search pages
        search_terms = ['search', 'find', 'lookup', 'advanced search', 'property search', 
                       'find homes', 'find properties', 'property finder', 'home finder',
                       'search properties', 'search listings', 'mls search']
        
        # Check all links on the page
        for a_tag in soup.find_all('a', href=True):
            href = a_tag.get('href', '')
            if not href or href.startswith('#') or href.startswith('javascript:'):
                continue
                
            # Convert to absolute URL
            link_url = urljoin(base_url, href)
            
            # Don't consider URLs that are clearly not search pages
            if any(ext in link_url.lower() for ext in ['.jpg', '.png', '.pdf', '.zip']):
                continue
                
            # Get text and attributes for relevance calculation
            link_text = a_tag.get_text().strip().lower()
            link_id = a_tag.get('id', '').lower()
            link_class = ' '.join(a_tag.get('class', [])).lower()
            link_title = a_tag.get('title', '').lower()
            
            # Calculate relevance score based on various factors
            relevance = 0.0
            
            # Check URL patterns
            if 'search' in link_url.lower():
                relevance += 0.3
            if 'find' in link_url.lower():
                relevance += 0.2
            if 'advanced' in link_url.lower() and ('search' in link_url.lower() or 'find' in link_url.lower()):
                relevance += 0.3
                
            # Check link text against search terms
            for term in search_terms:
                if term in link_text:
                    relevance += 0.5
                    break
                    
            # Check attributes
            if any(term in link_id for term in ['search', 'find']):
                relevance += 0.3
            if any(term in link_class for term in ['search', 'find']):
                relevance += 0.3
            if any(term in link_title for term in ['search', 'find']):
                relevance += 0.2
                
            # Consider search icons
            if a_tag.find('i', class_=lambda c: c and ('search' in c or 'fa-search' in c)):
                relevance += 0.4
                
            # If there's reasonable relevance, include this link
            if relevance > 0.3:
                search_links.append((link_url, relevance))
        
        # Sort by relevance (highest first)
        search_links.sort(key=lambda x: x[1], reverse=True)
        
        return search_links
    
    def extract_search_terms(self, user_intent: Dict[str, Any]) -> List[str]:
        """
        Extract relevant search terms from user intent using a comprehensive approach.
        
        Args:
            user_intent: User intent dictionary containing original query, keywords, etc.
            
        Returns:
            List of prioritized search terms
        """
        search_terms = []
        
        # Step 1: Extract entity-focused terms
        # These are the most specific search terms that clearly identify what we're looking for
        entity_terms = self._extract_entity_terms(user_intent)
        if entity_terms:
            search_terms.extend(entity_terms)
        
        # Step 2: Extract explicit search phrases
        # Look for phrases like "search for X" or "find Y"
        explicit_terms = self._extract_explicit_search_phrases(user_intent)
        if explicit_terms:
            search_terms.extend(explicit_terms)
        
        # Step 3: Extract key phrases from user intent
        # Use a more sophisticated NLP-like approach to extract meaningful phrases
        key_phrases = self._extract_key_phrases(user_intent)
        if key_phrases:
            search_terms.extend(key_phrases)
        
        # Step 4: Create combinations of keywords if available
        keyword_combos = self._create_keyword_combinations(user_intent)
        if keyword_combos:
            search_terms.extend(keyword_combos)
        
        # Step 5: Add location-based terms if available
        location_terms = self._extract_location_terms(user_intent)
        if location_terms:
            search_terms.extend(location_terms)
        
        # Filter and clean terms
        cleaned_terms = []
        for term in search_terms:
            if isinstance(term, str) and term.strip():
                # Remove short or generic terms that are unlikely to be good search terms
                if len(term.strip()) > 2 and term.lower().strip() not in ['the', 'and', 'for', 'from', 'with', 'search']:
                    cleaned_terms.append(term.strip())
        
        # Remove duplicates while preserving priority order
        unique_terms = []
        for term in cleaned_terms:
            if term.lower() not in [t.lower() for t in unique_terms]:
                unique_terms.append(term)
        
        # If we still have no terms, create a fallback term from the original query
        if not unique_terms and 'original_query' in user_intent:
            # Remove common words and create a term from the first few words
            words = user_intent['original_query'].split()
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'of', 'for', 'in', 'to', 'with', 'by', 'about'}
            content_words = [w for w in words if w.lower() not in stop_words and len(w) > 2]
            if content_words:
                # Use the first 3 content words as a fallback term
                fallback_term = ' '.join(content_words[:3])
                unique_terms.append(fallback_term)
        
        return unique_terms

    def _extract_entity_terms(self, user_intent: Dict[str, Any]) -> List[str]:
        """Extract specific entity-focused terms from user intent."""
        entity_terms = []
        
        # Check product-specific fields
        if 'product_info' in user_intent:
            product_info = user_intent['product_info']
            if isinstance(product_info, dict):
                # Product name is usually the best search term
                if 'name' in product_info:
                    entity_terms.append(product_info['name'])
                elif 'model' in product_info:
                    entity_terms.append(product_info['model'])
                
                # Brand + Model/Type combinations are also good
                if 'brand' in product_info and 'model' in product_info:
                    entity_terms.append(f"{product_info['brand']} {product_info['model']}")
                elif 'brand' in product_info and 'type' in product_info:
                    entity_terms.append(f"{product_info['brand']} {product_info['type']}")
        
        # Check property-specific fields (for real estate sites)
        if 'property_info' in user_intent:
            prop_info = user_intent['property_info']
            if isinstance(prop_info, dict):
                # Location is usually the best search term for properties
                location_parts = []
                for field in ['city', 'neighborhood', 'zip_code', 'county']:
                    if field in prop_info:
                        location_parts.append(prop_info[field])
                
                if location_parts:
                    entity_terms.append(' '.join(location_parts))
                
                # Property type + location combinations
                if 'property_type' in prop_info and location_parts:
                    entity_terms.append(f"{prop_info['property_type']} in {' '.join(location_parts)}")
        
        # Check for entity mentions in extraction requirements
        if 'extraction_requirements' in user_intent:
            reqs = user_intent['extraction_requirements']
            if isinstance(reqs, list):
                for req in reqs:
                    if isinstance(req, str) and ':' in req:
                        # Parse requirements like "Product: iPhone 13"
                        parts = req.split(':', 1)
                        if len(parts) == 2 and parts[1].strip():
                            entity_terms.append(parts[1].strip())
        
        return entity_terms

    def _extract_explicit_search_phrases(self, user_intent: Dict[str, Any]) -> List[str]:
        """Extract search terms from explicit search phrases in the text."""
        explicit_terms = []
        
        # Check all text fields for explicit search patterns
        text_fields = ['original_query', 'extract_description', 'user_intent_description']
        
        for field in text_fields:
            if field in user_intent and isinstance(user_intent[field], str):
                text = user_intent[field]
                
                # More comprehensive patterns for explicit search instructions
                patterns = [
                    # Search-oriented patterns
                    r'search(?:ing)? for [""]?([^"".]+)[""]?',
                    r'search(?:ing)? (?:about|on) [""]?([^"".]+)[""]?',
                    r'(?:try to )?find [""]?([^"".]+)[""]?',
                    r'look(?:ing)? for [""]?([^"".]+)[""]?',
                    r'(?:I want|I\'d like|I need|I wish) to find [""]?([^"".]+)[""]?',
                    
                    # Retrieval patterns
                    r'get (?:information|info|details) (?:about|on) [""]?([^"".]+)[""]?',
                    r'retrieve (?:information|info|details) (?:about|on) [""]?([^"".]+)[""]?',
                    
                    # Interest patterns
                    r'(?:I\'m |I am )interested in [""]?([^"".]+)[""]?',
                    r'(?:I want|I\'d like|I need) (?:information|info|details) (?:about|on) [""]?([^"".]+)[""]?',
                    
                    # Question patterns
                    r'(?:what|where|how) (?:is|are|can I find) [""]?([^"?]+)[""]?',
                    
                    # Direct specification patterns
                    r'(?:product|item|thing): [""]?([^"".]+)[""]?',
                    r'specifically (?:looking for|want|need) [""]?([^"".]+)[""]?'
                ]
                
                for pattern in patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        term = match.group(1).strip()
                        if term and len(term) > 2:  # Avoid very short terms
                            explicit_terms.append(term)
        
        return explicit_terms

    def _extract_key_phrases(self, user_intent: Dict[str, Any]) -> List[str]:
        """Extract key phrases using NLP-like techniques."""
        key_phrases = []
        
        # Use a more sophisticated approach to extract meaningful phrases
        text_fields = ['original_query', 'extract_description', 'user_intent_description']
        combined_text = ""
        
        for field in text_fields:
            if field in user_intent and isinstance(user_intent[field], str):
                combined_text += " " + user_intent[field]
        
        if combined_text:
            # Remove punctuation and convert to lowercase
            combined_text = re.sub(r'[^\w\s]', ' ', combined_text.lower())
            
            # More comprehensive stop words list
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'of', 'for', 'in', 'to', 'with', 'by', 'about', 'against', 'between', 'into', 'through',
                'during', 'before', 'after', 'above', 'below', 'up', 'down', 'on', 'off', 'over',
                'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
                'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
                'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will',
                'just', 'should', 'now', 'want', 'get', 'like', 'make', 'need', 'would', 'could', 'search',
                'find', 'look', 'trying', 'try', 'please', 'help', 'know', 'see', 'think', 'do', 'does'
            }
            
            # Tokenize the text
            words = combined_text.split()
            content_words = [w for w in words if w not in stop_words and len(w) > 2]
            
            # Create n-grams (unigrams, bigrams, trigrams)
            unigrams = content_words[:10]  # Top 10 content words
            
            bigrams = []
            for i in range(len(content_words) - 1):
                bigrams.append(f"{content_words[i]} {content_words[i+1]}")
            
            trigrams = []
            for i in range(len(content_words) - 2):
                trigrams.append(f"{content_words[i]} {content_words[i+1]} {content_words[i+2]}")
            
            # Count word frequencies to identify important terms
            word_count = {}
            for word in content_words:
                word_count[word] = word_count.get(word, 0) + 1
            
            # Sort unigrams by frequency
            sorted_unigrams = sorted([(w, word_count.get(w, 0)) for w in unigrams], 
                                  key=lambda x: x[1], reverse=True)
            
            # Add top phrases to our list
            for gram, _ in sorted_unigrams[:5]:  # Top 5 unigrams
                if gram not in stop_words and len(gram) > 2:
                    key_phrases.append(gram)
            
            # Add top bigrams (usually most useful)
            for bigram in bigrams[:5]:  # Top 5 bigrams
                key_phrases.append(bigram)
                
            # Add top trigrams (more specific)
            for trigram in trigrams[:3]:  # Top 3 trigrams
                key_phrases.append(trigram)
        
        return key_phrases

    def _create_keyword_combinations(self, user_intent: Dict[str, Any]) -> List[str]:
        """Create meaningful combinations from keywords list."""
        combos = []
        
        if 'keywords' in user_intent and isinstance(user_intent['keywords'], list):
            keywords = user_intent['keywords']
            
            # Skip if we only have one keyword
            if len(keywords) <= 1:
                return keywords
            
            # Create combinations of adjacent keywords (often more meaningful together)
            for i in range(len(keywords) - 1):
                combos.append(f"{keywords[i]} {keywords[i+1]}")
            
            # Create combinations with modifiers in keywords
            modifiers = ['best', 'new', 'top', 'cheap', 'quality', 'affordable', 'luxury', 'latest']
            for keyword in keywords:
                if keyword.lower() in modifiers:
                    for other_kw in keywords:
                        if other_kw.lower() != keyword.lower():
                            combos.append(f"{keyword} {other_kw}")
            
            # Create combinations with type/category and specific terms
            types = ['type', 'category', 'brand', 'model', 'style']
            for i, kw1 in enumerate(keywords):
                for j, kw2 in enumerate(keywords):
                    if i != j and any(t in kw1.lower() for t in types):
                        combos.append(f"{kw1} {kw2}")
        
        return combos

    def _extract_location_terms(self, user_intent: Dict[str, Any]) -> List[str]:
        """Extract location-based search terms."""
        location_terms = []
        
        # Extract from the location_data field if available
        if 'location_data' in user_intent:
            loc_data = user_intent['location_data']
            
            if isinstance(loc_data, dict):
                # Combine various location components for better search
                location_parts = []
                
                # City and state are the most common and useful location identifiers
                if 'city' in loc_data and 'state' in loc_data:
                    location_terms.append(f"{loc_data['city']}, {loc_data['state']}")
                
                # Check for other location components
                for field in ['city', 'state', 'zip_code', 'county', 'neighborhood', 'region']:
                    if field in loc_data and loc_data[field]:
                        location_parts.append(str(loc_data[field]))
                        if field in ['city', 'state', 'zip_code']:  # These are useful on their own too
                            location_terms.append(str(loc_data[field]))
                
                # Create different location combinations
                if len(location_parts) >= 2:
                    location_terms.append(' '.join(location_parts[:2]))  # First two components
                if len(location_parts) >= 3:
                    location_terms.append(' '.join(location_parts[:3]))  # First three components
                
            elif isinstance(loc_data, str) and loc_data.strip():
                location_terms.append(loc_data.strip())
        
        # Also try to extract location information from text fields
        text_fields = ['original_query', 'extract_description']
        
        for field in text_fields:
            if field in user_intent and isinstance(user_intent[field], str):
                text = user_intent[field]
                
                # Look for location patterns like "in {location}" or "near {location}"
                location_patterns = [
                    r'in ([A-Za-z\s,]+),? ([A-Z]{2})',  # "in Columbus, OH" or "in Columbus OH"
                    r'near ([A-Za-z\s,]+),? ([A-Z]{2})',  # "near Columbus, OH"
                    r'around ([A-Za-z\s,]+),? ([A-Z]{2})',  # "around Columbus, OH"
                    r'located in ([A-Za-z\s,]+)',  # "located in Downtown"
                    r'area of ([A-Za-z\s,]+)',  # "area of Seattle"
                    r'properties in ([A-Za-z\s,]+)'  # "properties in Chicago"
                ]
                
                for pattern in location_patterns:
                    matches = re.finditer(pattern, text)
                    for match in matches:
                        if len(match.groups()) == 2:  # City, State pattern
                            location_terms.append(f"{match.group(1).strip()}, {match.group(2).strip()}")
                        else:  # Single location component
                            location_terms.append(match.group(1).strip())
        
        return location_terms
        
    async def perform_search(self, crawler: AsyncWebCrawler, url: str, html_content: str, 
                           search_terms: List[str], domain_type: str = "general") -> Dict[str, Any]:
        """
        Perform a search using the first available search term with enhanced domain intelligence.
        
        Args:
            crawler: AsyncWebCrawler instance
            url: Website URL
            html_content: HTML content
            search_terms: List of search terms to try
            domain_type: Type of domain (e.g., "real_estate", "e_commerce", "general")
            
        Returns:
            Dictionary with search results
        """
        if not search_terms:
            logger.warning("No search terms provided")
            return {"success": False, "reason": "No search terms provided", "url": url}
        
        # Create a search automator
        search_automator = SearchAutomator(crawler)
        
        # Get domain for caching purposes
        domain = urlparse(url).netloc
        
        # Check if we've already searched on this domain recently
        search_key = f"{domain}:{search_terms[0]}"
        if search_key in self.search_history:
            logger.info(f"Using cached search result for {search_key}")
            return self.search_history[search_key]
        
        # Log the domain type and search terms
        logger.info(f"Searching on domain type: {domain_type} with terms: {search_terms}")
        
        # STEP 1: First check if search form exists on current page
        logger.info(f"Looking for search form on current page: {url}")
        soup = BeautifulSoup(html_content, 'html.parser')
        search_form_info = self._find_search_form(soup, url)
        
        # If no search form on current page, look for links to search pages
        if not search_form_info["has_search"]:
            logger.info("No search form found on current page, looking for search page links")
            search_page_links = self._find_search_links(html_content, url)
            
            # Navigate to the most promising search page
            if search_page_links:
                search_page_link, relevance = search_page_links[0]
                logger.info(f"Navigating to search page: {search_page_link} (relevance: {relevance})")
                
                # Use JavaScript-enabled crawler config for modern sites
                js_config = create_js_config(timeout=30)
                search_page_result = await crawler.arun(url=search_page_link, config=js_config)
                
                if search_page_result.success:
                    logger.info(f"Successfully navigated to search page")
                    url = search_page_link
                    html_content = search_page_result.html
                    
                    # Try to find search form on the search page
                    soup = BeautifulSoup(html_content, 'html.parser')
                    search_form_info = self._find_search_form(soup, url)
                else:
                    logger.warning(f"Failed to navigate to search page: {search_page_link}")
        
        # STEP 2: Try enhanced Playwright-based search first if available
        # This works for any website type, even complex modern sites
        try:
            playwright_handler = PlaywrightSearchHandler(logger=logger)
            logger.info(f"Attempting universal Playwright search with term: '{search_terms[0]}'")
            
            # Create a domain-specific search config
            search_config = {
                "domain_type": domain_type,
                "url": url,
                "search_term": search_terms[0]
            }
            
            # Perform the search with enhanced capabilities
            playwright_result = await playwright_handler.perform_search(url, search_terms[0])
            
            if playwright_result.get("success", False):
                logger.info(f"Playwright search successful using method: {playwright_result.get('method', 'unknown')}")
                playwright_result["term"] = search_terms[0]
                
                # Cache the result
                self.search_history[search_key] = playwright_result
                return playwright_result
            else:
                logger.info(f"Playwright search failed: {playwright_result.get('reason', 'Unknown reason')}")
        except Exception as e:
            logger.error(f"Error in Playwright search: {str(e)}")
        
        # STEP 3: Try each search term with traditional search form submission
        for term in search_terms:
            logger.info(f"Attempting search with term: '{term}'")
            
            # Try using a detected search form
            if search_form_info["has_search"]:
                logger.info(f"Using detected search form")
                # Remove the config parameter which was causing the error
                search_result = await search_automator.submit_search_form(
                    search_form_info["form_data"], 
                    term
                )
                
                # Check if search was successful
                if search_result.get("success"):
                    logger.info(f"Search successful using form")
                    search_result["term"] = term
                    search_result["domain_type"] = domain_type
                    self.search_history[search_key] = search_result
                    return search_result
            
            # If form search failed or no form found, try URL-based search with domain intelligence
            logger.info(f"Trying URL-based search patterns")
            url_patterns = await self._try_url_based_search(url, term, domain_type)
            
            # Try each URL pattern
            for search_url in url_patterns:
                try:
                    logger.info(f"Trying URL pattern: {search_url}")
                    js_config = create_js_config(timeout=30)
                    result = await crawler.arun(url=search_url, config=js_config)
                    
                    if result.success:
                        # Check if search term appears on the page (simple validation)
                        if term.lower() in result.html.lower():
                            logger.info(f"URL-based search successful with pattern: {search_url}")
                            search_result = {
                                "success": True,
                                "url": search_url,
                                "html": result.html,
                                "term": term,
                                "domain_type": domain_type,
                                "method": "url_pattern"
                            }
                            self.search_history[search_key] = search_result
                            return search_result
                except Exception as e:
                    logger.warning(f"Error trying URL pattern {search_url}: {str(e)}")
                    continue
        
        # If we reach here, all search attempts failed
        logger.warning("All search attempts failed")
        return {
            "success": False, 
            "reason": "All search attempts failed", 
            "url": url,
            "search_terms_tried": search_terms,
            "domain_type": domain_type
        }
    
    def _construct_search_url(self, url: str, search_term: str) -> str:
        """
        Construct a search URL by adding the search term to the URL.
        
        Args:
            url: Base URL
            search_term: Search term to add
            
        Returns:
            Modified URL with search term
        """
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        
        # Common search parameter names in order of preference
        search_params = ['q', 'query', 'search', 'term', 'keywords', 's', 'text', 'searchTerm']
        
        # Look for common search parameter names
        search_param_found = False
        for param in search_params:
            if param in query_params:
                query_params[param] = [search_term]
                search_param_found = True
                break
        
        if not search_param_found:
            # If no common search param found, add 'q' as default
            query_params['q'] = [search_term]
            
        # Reconstruct the query string
        new_query = urlencode(query_params, doseq=True)
        
        # Reconstruct the URL
        new_url = urlunparse((
            parsed_url.scheme,
            parsed_url.netloc,
            parsed_url.path,
            parsed_url.params,
            new_query,
            parsed_url.fragment
        ))
        
        return new_url
    
    def _detect_url_change(self, original_url: str, new_url: str) -> bool:
        """
        Detect if a URL has meaningfully changed (more than just a fragment or minor query param).
        
        Args:
            original_url: Original URL before search
            new_url: URL after search
            
        Returns:
            True if URL has meaningfully changed, False otherwise
        """
        if (original_url == new_url):
            return False
            
        original_parsed = urlparse(original_url)
        new_parsed = urlparse(new_url)
        
        # If host or path changed, it's definitely a change
        if original_parsed.netloc != new_parsed.netloc or original_parsed.path != new_parsed.path:
            return True
            
        # Get query parameters for both URLs
        original_params = parse_qs(original_parsed.query)
        new_params = parse_qs(new_parsed.query)
        
        # Check if search-related parameters were added
        search_params = ['q', 'query', 'search', 'term', 'keywords', 's', 'text', 'searchTerm']
        for param in search_params:
            if param in new_params and (param not in original_params or original_params[param] != new_params[param]):
                return True
                
        # Check if there are any other significant changes to parameters
        if set(new_params.keys()) != set(original_params.keys()):
            return True
            
        # If fragment changed but nothing else, that's not a significant change
        # (often just an anchor change on the same page)
        return False
        
    async def _check_empty_results(self, html_content: str) -> bool:
        """
        Check if a search results page indicates no results were found.
        
        Args:
            html_content: HTML content of the results page
            
        Returns:
            True if the page indicates no results, False otherwise
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Common phrases indicating no search results
        no_results_phrases = [
            "no results", "no matches", "no items found", 
            "couldn't find", "couldn't be found", 
            "no products", "no listings", "nothing found",
            "0 results", "zero results", "try another search",
            "no results found", "didn't match", "not find any",
            "no search results", "no pages", "does not match",
            "did not match", "no results found", "nothing was found"
        ]
        
        # Check for no results messages in the page content
        page_text = soup.get_text().lower()
        for phrase in no_results_phrases:
            if phrase in page_text:
                return True
                
        # Look for empty results containers
        results_containers = soup.select(".results, .search-results, .product-list, .listings, [class*='result'], [class*='search-result']")
        for container in results_containers:
            # Check if the container is empty or has a "no results" message
            if not container.find_all() or any(phrase in container.get_text().lower() for phrase in no_results_phrases):
                return True
                
        return False
        
    async def analyze_search_results(self, crawler: AsyncWebCrawler, search_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze search results page to detect result patterns, pagination, etc.
        
        Args:
            crawler: AsyncWebCrawler instance
            search_result: Search result dictionary
            
        Returns:
            Dictionary with analysis results
        """
        if not search_result.get("success", False) or not search_result.get("html"):
            return {"has_results": False}
            
        html_content = search_result.get("html")
        results_url = search_result.get("url")
        search_term = search_result.get("term")
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Result data structure
        analysis = {
            "has_results": True,
            "url": results_url,
            "search_term": search_term,
            "result_count": 0,
            "result_items": [],
            "has_pagination": False,
            "next_page": None,
            "pagination_type": None,
            "total_pages": None,
            "current_page": 1,
        }
        
        try:
            # 1. Try to find the results container
            result_containers = self._identify_result_containers(soup)
            
            if not result_containers:
                logger.info("No result containers found on search results page")
                analysis["has_results"] = False
                return analysis
                
            logger.info(f"Found {len(result_containers)} potential result containers")
            
            # 2. Count and extract result items from the most likely container
            best_container = result_containers[0]  # Most likely container
            result_items = best_container.find_all(['li', 'div', 'article'], recursive=False)
            
            # If no direct children found, look one level deeper
            if len(result_items) <= 1:
                result_items = best_container.select('li, div[class*="item"], div[class*="result"], article')
            
            # Filter out likely non-result items (too small/empty)
            valid_results = []
            for item in result_items:
                # Skip items without meaningful content or very small items
                item_text = item.get_text().strip()
                if len(item_text) < 20 or not item.find_all():
                    continue
                    
                # Skip items that don't have links
                if not item.find('a'):
                    continue
                    
                valid_results.append(item)
            
            analysis["result_count"] = len(valid_results)
            
            # Extract basic info from a sample of results (up to 5)
            for i, item in enumerate(valid_results[:5]):
                result_info = self._extract_result_item_info(item, results_url)
                if result_info:
                    analysis["result_items"].append(result_info)
            
            # 3. Check for pagination
            pagination_data = self._identify_pagination(soup, results_url)
            if pagination_data:
                analysis.update(pagination_data)
            
            # 4. Try to detect total result count
            total_count = self._extract_total_result_count(soup)
            if total_count is not None:
                analysis["total_result_count"] = total_count
            
            logger.info(f"Search results analysis: found {analysis['result_count']} valid results, pagination: {analysis['has_pagination']}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing search results: {str(e)}")
            analysis["error"] = str(e)
            return analysis
    
    def _identify_result_containers(self, soup: BeautifulSoup) -> List[Any]:
        """
        Identify likely containers of search results.
        
        Args:
            soup: BeautifulSoup object of the page
            
        Returns:
            List of container elements, sorted by likelihood
        """
        containers = []
        
        # 1. Look for containers with results-related classes
        result_class_patterns = [
            'result', 'search-result', 'listing', 'product-list', 'items', 
            'search-listing', 'search-items', 'product-grid', 'product-container',
            'search-container', 'results-wrapper', 'search-wrapper'
        ]
        
        for pattern in result_class_patterns:
            elements = soup.find_all(['div', 'ul', 'section'], {'class': lambda c: c and pattern in c.lower() if c else False})
            for element in elements:
                containers.append((element, 0.9))  # High confidence for explicit result classes
        
        # 2. Look for containers with multiple similar child elements
        # This can find results even when there's no clear class name
        candidate_containers = soup.find_all(['div', 'ul', 'section'])
        for container in candidate_containers:
            # Skip small containers and ones we've already identified
            if container in [c[0] for c in containers]:
                continue
                
            # Look for repeated elements that might be search results
            repeated_elements = self._find_repeated_elements(container)
            if repeated_elements >= 3:  # At least 3 similar items
                # Score based on how many repeated elements and if they have links
                link_count = len(container.find_all('a', href=True))
                score = min(0.5 + (repeated_elements * 0.05) + (link_count * 0.02), 0.89)
                containers.append((container, score))
                
        # 3. Look for containers with pagination nearby
        for element in soup.find_all(['div', 'section', 'main']):
            pagination = element.find(['div', 'nav', 'ul'], {'class': lambda c: c and 'pag' in c.lower() if c else False})
            if pagination:
                # Container with pagination is likely to contain results
                if element not in [c[0] for c in containers]:
                    containers.append((element, 0.7))
        
        # Sort by confidence score
        containers.sort(key=lambda x: x[1], reverse=True)
        
        # Just return the container elements
        return [container for container, _ in containers]
    
    def _find_repeated_elements(self, container) -> int:
        """
        Find repeated elements in a container that might be search results.
        
        Args:
            container: BeautifulSoup element
            
        Returns:
            Count of repeated elements
        """
        # Check direct children
        children = container.find_all(['div', 'li', 'article'], recursive=False)
        
        # If very few direct children, look one level deeper
        if len(children) < 3:
            children = container.select('div > div, div > li, div > article, ul > li')
            
        # No potential results
        if len(children) < 3:
            return 0
            
        # Check for similarly structured elements
        # Count elements that have similar child structure
        similar_count = 0
        
        # Compare first element's structure to others
        if children:
            first_child = children[0]
            first_links = len(first_child.find_all('a'))
            first_imgs = len(first_child.find_all('img'))
            first_paras = len(first_child.find_all(['p', 'span', 'div']))
            
            # Count elements with similar structure
            for child in children[1:]:
                # Count same types of elements
                links = len(child.find_all('a'))
                imgs = len(child.find_all('img'))
                paras = len(child.find_all(['p', 'span', 'div']))
                
                # Check if structure is similar (allow some variation)
                if (abs(links - first_links) <= 1 and 
                    abs(imgs - first_imgs) <= 1 and 
                    abs(paras - first_paras) <= 2):
                    similar_count += 1
                    
        return similar_count + 1  # +1 for the first element
    
    def _extract_result_item_info(self, result_item, base_url: str) -> Dict[str, Any]:
        """
        Extract key information from a search result item.
        
        Args:
            result_item: BeautifulSoup element of a result item
            base_url: Base URL for resolving relative links
            
        Returns:
            Dictionary with result item information
        """
        try:
            # Initialize with empty data
            result_info = {
                "title": None,
                "url": None,
                "description": None,
                "image_url": None,
                "price": None,
            }
            
            # Extract title and URL (from the first or most prominent link)
            links = result_item.find_all('a')
            
            # Skip items without links
            if not links:
                return None
                
            # Find the most relevant link (usually the title link)
            main_link = None
            
            # First try links with prominent classes
            title_candidates = result_item.find_all('a', {'class': lambda c: c and any(x in c.lower() for x in ['title', 'name', 'heading', 'product']) if c else False})
            if title_candidates:
                main_link = title_candidates[0]
            else:
                # Then links wrapped in headings
                heading_links = result_item.find_all(['h1', 'h2', 'h3', 'h4', 'h5'])
                for heading in heading_links:
                    link = heading.find('a')
                    if link:
                        main_link = link
                        break
                        
                # If still not found, use the first link with text
                if not main_link:
                    for link in links:
                        if link.get_text().strip():
                            main_link = link
                            break
                    
                    # Last resort: just use the first link
                    if not main_link and links:
                        main_link = links[0]
            
            # Extract title and URL from the main link
            if main_link:
                result_info["title"] = main_link.get_text().strip()
                href = main_link.get('href')
                if href:
                    result_info["url"] = urljoin(base_url, href)
            
            # Extract description
            desc_candidates = [
                result_item.find(['p', 'div'], {'class': lambda c: c and any(x in c.lower() for x in ['desc', 'summary', 'text', 'info']) if c else False}),
                result_item.find('p'),
                result_item.find(['div', 'span'], {'class': lambda c: c and 'text' in c.lower() if c else False})
            ]
            
            for candidate in desc_candidates:
                if candidate and candidate.get_text().strip():
                    # Skip if this contains the title or is too similar
                    text = candidate.get_text().strip()
                    if result_info["title"] and (result_info["title"] in text or text in result_info["title"]):
                        continue
                    result_info["description"] = text
                    break
            
            # Extract image
            img = result_item.find('img')
            if img and img.get('src'):
                result_info["image_url"] = urljoin(base_url, img.get('src'))
            
            # Extract price (common in e-commerce, real estate, etc.)
            price_candidates = [
                result_item.find(['span', 'div'], {'class': lambda c: c and any(x in c.lower() for x in ['price', 'cost', 'amount']) if c else False}),
                result_item.find(text=lambda t: t and re.search(r'(\$|€|£|USD|EUR|GBP)\s*\d+[\d,.]*', t))
            ]
            
            for candidate in price_candidates:
                if candidate:
                    # For element
                    if hasattr(candidate, 'get_text'):
                        price_text = candidate.get_text().strip()
                    else:  # For direct text node
                        price_text = str(candidate).strip()
                        
                    if price_text:
                        # Clean up price text
                        result_info["price"] = price_text
                        break
            
            return result_info
            
        except Exception as e:
            logger.error(f"Error extracting result item info: {str(e)}")
            return None
    
    def _identify_pagination(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """
        Identify pagination elements and extract pagination information.
        
        Args:
            soup: BeautifulSoup object of the page
            url: Current URL
            
        Returns:
            Dictionary with pagination information
        """
        pagination_data = {
            "has_pagination": False,
            "next_page": None,
            "pagination_type": None,
            "total_pages": None,
            "current_page": 1
        }
        
        # 1. Look for pagination container
        pagination_candidates = [
            soup.find(['div', 'nav', 'ul'], {'class': lambda c: c and 'pagination' in c.lower() if c else False}),
            soup.find(['div', 'nav', 'ul'], {'class': lambda c: c and 'pager' in c.lower() if c else False}),
            soup.find(['div', 'nav', 'ul'], {'class': lambda c: c and 'pages' in c.lower() if c else False})
        ]
        
        pagination = next((p for p in pagination_candidates if p), None)
        
        if pagination:
            pagination_data["has_pagination"] = True
            
            # Determine pagination type
            if pagination.find('button', string=lambda s: s and ('load more' in s.lower() or 'show more' in s.lower())):
                pagination_data["pagination_type"] = "load_more"
            elif pagination.find_all('a', {'href': True}):
                pagination_data["pagination_type"] = "numbered"
            
            # Extract next page URL
            next_link = None
            
            # Look for explicit next buttons
            next_candidates = [
                pagination.find('a', string=lambda s: s and ('next' in s.lower() or '→' in s or '>' in s)),
                pagination.find('a', {'class': lambda c: c and 'next' in c.lower() if c else False}),
                pagination.find('a', {'rel': 'next'}),
                pagination.find('a', {'aria-label': lambda a: a and 'next' in a.lower()})
            ]
            
            next_link = next((l for l in next_candidates if l and l.get('href')), None)
            
            # If no explicit next button, infer from current page
            if not next_link:
                # Try to find current page number
                current_page_elem = pagination.find(['span', 'a'], {'class': lambda c: c and any(x in c.lower() for x in ['current', 'active', 'selected']) if c else False})
                
                if current_page_elem:
                    # Extract current page number
                    current_text = current_page_elem.get_text().strip()
                    current_match = re.search(r'\d+', current_text)
                    
                    if current_match:
                        current_page = int(current_match.group())
                        pagination_data["current_page"] = current_page
                        
                        # Look for a link to the next page number
                        next_page = current_page + 1
                        next_page_link = pagination.find('a', string=lambda s: s and str(next_page) in s)
                        
                        if next_page_link and next_page_link.get('href'):
                            next_link = next_page_link
            
            # For numbered pagination, get total pages
            if pagination_data["pagination_type"] == "numbered":
                page_links = pagination.find_all('a')
                page_numbers = []
                
                for link in page_links:
                    text = link.get_text().strip()
                    match = re.search(r'\d+', text)
                    if match:
                        page_numbers.append(int(match.group()))
                
                if page_numbers:
                    pagination_data["total_pages"] = max(page_numbers)
            
            # Set next page URL if found
            if next_link and next_link.get('href'):
                pagination_data["next_page"] = urljoin(url, next_link.get('href'))
        
        # 2. Check for infinite scroll / lazy load indicators
        infinite_scroll_indicators = [
            soup.find(['div', 'span'], {'class': lambda c: c and 'infinite' in c.lower() if c else False}),
            soup.find(['div', 'span'], {'class': lambda c: c and 'lazy' in c.lower() if c else False}),
            soup.find('script', string=lambda s: s and ('infiniteScroll' in s or 'lazyLoad' in s or 'endless' in s))
        ]
        
        if any(infinite_scroll_indicators) and not pagination_data["pagination_type"]:
            pagination_data["has_pagination"] = True
            pagination_data["pagination_type"] = "infinite_scroll"
        
        # 3. Look for URL patterns suggesting pagination
        if not pagination_data["has_pagination"]:
            parsed_url = urlparse(url)
            query_params = parse_qs(parsed_url.query)
            
            # Common pagination parameters
            page_params = ['page', 'p', 'pg', 'pagenum', 'pageNumber', 'start', 'offset']
            
            for param in page_params:
                if param in query_params:
                    pagination_data["has_pagination"] = True
                    pagination_data["pagination_type"] = "url_param"
                    
                    # Current page from URL
                    try:
                        current_page = int(query_params[param][0])
                        pagination_data["current_page"] = current_page
                        
                        # Construct next page URL
                        next_page = current_page + 1
                        new_params = query_params.copy()
                        new_params[param] = [str(next_page)]
                        
                        # Rebuild the URL with new page parameter
                        new_query = urlencode(new_params, doseq=True)
                        next_url = urlunparse((
                            parsed_url.scheme,
                            parsed_url.netloc,
                            parsed_url.path,
                            parsed_url.params,
                            new_query,
                            parsed_url.fragment
                        ))
                        
                        pagination_data["next_page"] = next_url
                    except (ValueError, IndexError):
                        pass
                        
                    break
        
        return pagination_data
    
    def _extract_total_result_count(self, soup: BeautifulSoup) -> Optional[int]:
        """
        Extract the total number of search results if available.
        
        Args:
            soup: BeautifulSoup object of the page
            
        Returns:
            Total result count if found, None otherwise
        """
        # Look for common result count patterns
        count_patterns = [
            r'(\d+,?\d*)\s*(?:results|items|products|properties|listings)',
            r'showing\s+\d+\s*-\s*\d+\s*of\s+(\d+,?\d*)',
            r'found\s+(\d+,?\d*)',
            r'(\d+,?\d*)\s*matches',
            r'(\d+,?\d*)\s*total'
        ]
        
        # Check text content for result count
        page_text = soup.get_text()
        
        for pattern in count_patterns:
            match = re.search(pattern, page_text, re.IGNORECASE)
            if match:
                count_str = match.group(1).replace(',', '')
                try:
                    return int(count_str)
                except ValueError:
                    pass
        
        # Check for result count in specific elements
        count_elements = [
            soup.find(['div', 'span', 'p'], {'class': lambda c: c and any(x in c.lower() for x in ['count', 'total', 'results']) if c else False}),
            soup.find(['h1', 'h2', 'h3'], {'class': lambda c: c and any(x in c.lower() for x in ['count', 'total', 'results']) if c else False})
        ]
        
        for element in count_elements:
            if element:
                text = element.get_text().strip()
                match = re.search(r'(\d+,?\d*)', text)
                if match:
                    count_str = match.group(1).replace(',', '')
                    try:
                        return int(count_str)
                    except ValueError:
                        pass
        
        return None
        
    def clear_cache(self):
        """Clear the search history and domain search forms cache."""
        self.search_history = {}
        self.domain_search_forms = {}

    async def extract_search_terms(self, 
                                 crawler: AsyncWebCrawler, 
                                 user_query: str, 
                                 crawl_intent: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Extract search terms from user query based on intent.
        
        Args:
            crawler: AsyncWebCrawler instance
            user_query: Original user query
            crawl_intent: Dictionary containing user intent information
            
        Returns:
            List of dictionaries with search term info
        """
        search_terms = []
        
        # Default relevance
        default_relevance = 0.8
        
        # If we have crawl intent, extract terms from it
        if crawl_intent:
            # Check for explicit search terms
            if "search_terms" in crawl_intent:
                # Direct search terms provided
                terms = crawl_intent["search_terms"]
                if isinstance(terms, list):
                    for term in terms:
                        search_terms.append({
                            "term": term,
                            "relevance": default_relevance,
                            "source": "intent",
                            "context": "Explicit search term from intent"
                        })
                elif isinstance(terms, str):
                    search_terms.append({
                        "term": terms,
                        "relevance": default_relevance,
                        "source": "intent",
                        "context": "Explicit search term from intent"
                    })
            
            # Check for keywords
            if "keywords" in crawl_intent and isinstance(crawl_intent["keywords"], list):
                # Convert keywords to search terms
                for keyword in crawl_intent["keywords"]:
                    if isinstance(keyword, str) and keyword.strip():
                        # Check if this keyword is already included
                        if not any(term["term"].lower() == keyword.lower() for term in search_terms):
                            search_terms.append({
                                "term": keyword.strip(),
                                "relevance": 0.7,  # Slightly lower than explicit search terms
                                "source": "keyword",
                                "context": "Extracted from keywords"
                            })
        
        # If no search terms found, extract from user query
        if not search_terms and user_query:
            # Direct search from user query
            search_terms.append({
                "term": user_query,
                "relevance": 0.9,  # High relevance for direct user input
                "source": "query",
                "context": "Extracted from user query"
            })
            
            # For location/real estate searches, try to extract location information
            if await self._is_location_search(user_query, crawl_intent):
                location_terms = await self._extract_location_terms(user_query)
                if location_terms:
                    for location in location_terms:
                        # Check if this location is already included
                        if not any(term["term"].lower() == location.lower() for term in search_terms):
                            search_terms.append({
                                "term": location,
                                "relevance": 0.85, 
                                "source": "location",
                                "context": "Location extracted from query"
                            })
            
            # For product searches, try to extract product information
            elif await self._is_product_search(user_query, crawl_intent):
                product_terms = await self._extract_product_terms(user_query)
                if product_terms:
                    for product in product_terms:
                        # Check if this product is already included
                        if not any(term["term"].lower() == product.lower() for term in search_terms):
                            search_terms.append({
                                "term": product,
                                "relevance": 0.85,
                                "source": "product",
                                "context": "Product term extracted from query"
                            })
        
        # Ensure we have at least one search term
        if not search_terms:
            # Fallback to generic extract of noun phrases
            fallback_terms = self._extract_noun_phrases(user_query)
            if fallback_terms:
                for term in fallback_terms[:2]:  # Limit to top 2 fallback terms
                    search_terms.append({
                        "term": term,
                        "relevance": 0.6,  # Lower relevance for fallback extraction
                        "source": "fallback",
                        "context": "Fallback extraction from query"
                    })
            else:
                # Final fallback - use the whole query
                search_terms.append({
                    "term": user_query,
                    "relevance": 0.5,
                    "source": "fallback",
                    "context": "Fallback to full query"
                })
        
        # Sort by relevance
        search_terms.sort(key=lambda x: x["relevance"], reverse=True)
        
        return search_terms
        
    def _extract_noun_phrases(self, text: str) -> List[str]:
        """
        Extract noun phrases from text as potential search terms.
        
        Args:
            text: Text to extract from
            
        Returns:
            List of extracted noun phrases
        """
        # Simple extraction using common patterns
        # This is a fallback when NLP processing isn't available
        
        # Clean and tokenize
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        
        # Filter out common stop words
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when', 
                      'at', 'from', 'in', 'on', 'for', 'with', 'by', 'about', 'against', 
                      'between', 'into', 'through', 'during', 'before', 'after', 'above', 
                      'below', 'to', 'of', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                      'have', 'has', 'had', 'do', 'does', 'did', 'can', 'could', 'will',
                      'would', 'shall', 'should', 'may', 'might', 'must', 'i', 'you', 'he',
                      'she', 'it', 'we', 'they', 'who', 'get', 'want', 'looking', 'need'}
        
        filtered_words = [w for w in words if w not in stop_words and len(w) > 1]
        
        # Simple chunk extraction (adjacent words)
        chunks = []
        current_chunk = []
        
        for word in filtered_words:
            current_chunk.append(word)
            
            # Limit chunk size to 3 words
            if len(current_chunk) >= 3:
                chunks.append(' '.join(current_chunk))
                current_chunk = current_chunk[1:]  # Sliding window
        
        # Add any remaining chunk
        if len(current_chunk) > 1:
            chunks.append(' '.join(current_chunk))
            
        # Add individual words as 1-grams
        for word in filtered_words:
            if word not in chunks:
                chunks.append(word)
        
        # Sort by length (prefer longer phrases) then alphabetically
        chunks.sort(key=lambda x: (-len(x.split()), x))
        
        return chunks
    
    async def _is_location_search(self, query: str, crawl_intent: Dict[str, Any] = None) -> bool:
        """
        Determine if the query is for a location-based search.
        
        Args:
            query: User query
            crawl_intent: Dictionary containing user intent information
            
        Returns:
            True if query is likely location-based, False otherwise
        """
        # Check intent for location signals
        if crawl_intent:
            # Explicit intent type
            if "type" in crawl_intent:
                intent_type = crawl_intent["type"].lower()
                if intent_type in ["real_estate", "property", "home", "house", "apartment", "location", "place"]:
                    return True
            
            # Location field in intent
            if "location" in crawl_intent:
                return True
        
        # Check query for location signals
        location_keywords = [
            "house", "apartment", "home", "property", "real estate", "condo", "rent", 
            "rental", "townhouse", "listing", "for sale", "near me", "in", "at", "around",
            "location", "zip code", "postal code", "city", "town", "neighborhood"
        ]
        
        query_lower = query.lower();
        
        # Check for location keywords
        if any(keyword in query_lower for keyword in location_keywords):
            return True;
            
        # Check for location patterns (e.g., zip codes, city+state)
        if re.search(r'\b\d{5}\b', query) or re.search(r'\b[A-Z][a-z]+,\s*[A-Z]{2}\b', query):
            return True;
            
        return False;
        
    async def _extract_location_terms(self, query: str) -> List[str]:
        """
        Extract location-specific search terms from a query.
        
        Args:
            query: User query
            
        Returns:
            List of location search terms
        """
        location_terms = []
        
        # Clean query
        query = query.strip()
        
        # Look for city, state patterns
        city_state_matches = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*([A-Z]{2})', query)
        for city, state in city_state_matches:
            location_terms.append(f"{city}, {state}")
        
        # Look for zip codes
        zip_matches = re.findall(r'\b(\d{5})\b', query)
        for zip_code in zip_matches:
            location_terms.append(zip_code)
        
        # Look for common location phrases
        location_phrase_patterns = [
            r'homes? in\s+([A-Za-z0-9\s]+)',
            r'properties? in\s+([A-Za-z0-9\s]+)',
            r'houses? in\s+([A-Za-z0-9\s]+)',
            r'real estate in\s+([A-Za-z0-9\s]+)',
            r'condos? in\s+([A-Za-z0-9\s]+)',
            r'apartments? in\s+([A-Za-z0-9\s]+)',
            r'living in\s+([A-Za-z0-9\s]+)',
            r'rentals? in\s+([A-Za-z0-9\s]+)'
        ]
        
        for pattern in location_phrase_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                match = match.strip()
                if match and len(match) > 2:
                    location_terms.append(match)
        
        # Simple heuristic: If query contains "in" followed by capitalized words, extract them
        in_location_matches = re.findall(r'in\s+([A-Z][A-Za-z\s]+)', query)
        for match in in_location_matches:
            match = match.strip()
            if match and len(match) > 2:
                location_terms.append(match)
        
        # Deduplicate
        return list(set(location_terms))
        
    async def _is_product_search(self, query: str, crawl_intent: Dict[str, Any] = None) -> bool:
        """
        Determine if the query is for a product search.
        
        Args:
            query: User query
            crawl_intent: Dictionary containing user intent information
            
        Returns:
            True if query is likely product-based, False otherwise
        """
        # Check intent for product signals
        if crawl_intent:
            # Explicit intent type
            if "type" in crawl_intent:
                intent_type = crawl_intent["type"].lower()
                if intent_type in ["product", "item", "shopping", "buy", "purchase", "shop"]:
                    return True
        
        # Check query for product signals
        product_keywords = [
            "buy", "purchase", "shop", "shopping", "order", "best", "top", "cheap", 
            "affordable", "expensive", "quality", "review", "price", "deal", "discount",
            "sale", "new", "used", "refurbished", "product", "item", "brand", "model"
        ]
        
        query_lower = query.lower();
        
        # Check for product keywords
        if any(keyword in query_lower for keyword in product_keywords):
            return True;
            
        return False;
        
    async def _extract_product_terms(self, query: str) -> List[str]:
        """
        Extract product-specific search terms from a query.
        
        Args:
            query: User query
            
        Returns:
            List of product search terms
        """
        product_terms = []
        
        # Clean query
        query = query.strip()
        
        # Look for product patterns
        product_phrase_patterns = [
            r'buy\s+([A-Za-z0-9\s]+)',
            r'shop\s+for\s+([A-Za-z0-9\s]+)',
            r'purchase\s+([A-Za-z0-9\s]+)',
            r'looking\s+for\s+([A-Za-z0-9\s]+)',
            r'need\s+a\s+([A-Za-z0-9\s]+)',
            r'want\s+to\s+buy\s+([A-Za-z0-9\s]+)',
            r'best\s+([A-Za-z0-9\s]+)',
            r'top\s+([A-Za-z0-9\s]+)',
            r'cheap\s+([A-Za-z0-9\s]+)',
            r'affordable\s+([A-Za-z0-9\s]+)',
            r'expensive\s+([A-Za-z0-9\s]+)',
            r'quality\s+([A-Za-z0-9\s]+)',
        ]
        
        for pattern in product_phrase_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                match = match.strip()
                if match and len(match) > 2:
                    product_terms.append(match)
        
        # Remove common ending words that aren't part of product
        cleaned_terms = []
        for term in product_terms:
            # Remove ending words like "online", "now", etc.
            cleaned = re.sub(r'\b(online|now|today|quickly|easily|here)\b$', '', term).strip()
            if cleaned:
                cleaned_terms.append(cleaned)
        
        if cleaned_terms:
            product_terms = cleaned_terms
            
        # If no patterns matched, extract noun phrases
        if not product_terms:
            # Remove question words and common phrases to get the core product
            cleaned_query = re.sub(r'^(where|what|how|who|when|why)(\s+can\s+i|\s+to|\s+do\s+i|\s+should\s+i)?\s+', '', query, flags=re.IGNORECASE)
            cleaned_query = re.sub(r'^(find|get|buy|purchase|order|shop for)\s+', '', cleaned_query, flags=re.IGNORECASE)
            
            # If cleaned query is substantial, add it
            if cleaned_query and len(cleaned_query) > 3 and cleaned_query != query:
                product_terms.append(cleaned_query.strip())
        
        # Deduplicate
        return list(set(product_terms))
    
    def _find_search_form(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """
        Find search forms on the page with enhanced detection for any site.
        
        Args:
            soup: BeautifulSoup object of the page
            url: URL of the page
            
        Returns:
            Dictionary with search form information
        """
        logger.info(f"Looking for search forms on {url}")
        
        result = {
            "has_search": False,
            "form_data": None
        }
        
        # Domain specific selectors - start with real estate
        domain_specific_selectors = {
            "real_estate": [
                'form[action*="search"], form[action*="find"], form[id*="search"], form[class*="search"]',
                'div[class*="search-form"], div[id*="search-form"]',
                'div[class*="property-search"], div[class*="listing-search"]',
                '.idx-search, .mls-search, #property-search-container',
                'form[class*="idx"], form[id*="idx"], div[class*="idx-omnibar"]',
                'div[class*="location-search"], div[class*="address-search"]'
            ],
            "general": [
                'form[role="search"]',
                'form[class*="search"], form[id*="search"]',
                'input[type="search"]',
                'div[class*="search-container"], div[id*="search-container"]',
                'div[class*="search-wrapper"], div[id*="search-wrapper"]'
            ]
        }
        
        # Determine domain type
        domain_type = self._detect_domain_type(url)
        
        # Try domain-specific selectors first, then fall back to general selectors
        selectors_to_try = domain_specific_selectors.get(domain_type, []) + domain_specific_selectors["general"]
        
        # Process each selector
        for selector in selectors_to_try:
            elements = soup.select(selector)
            if elements:
                element = elements[0]  # Take the first matching element
                
                # Handle both forms and non-form containers
                if element.name == 'form':
                    # Traditional form
                    form_data = {
                        "type": "standard_form",
                        "id": element.get('id', ''),
                        "action": element.get('action', ''),
                        "method": element.get('method', 'get').lower()
                    }
                    
                    # Find inputs
                    inputs = []
                    for input_el in element.find_all(['input', 'select', 'textarea']):
                        input_type = input_el.get('type', '')
                        input_name = input_el.get('name', '')
                        input_id = input_el.get('id', '')
                        
                        # Skip hidden and submit fields
                        if input_type in ['hidden', 'submit', 'button']:
                            continue
                            
                        # Check if this looks like a search input
                        is_search_input = (
                            input_type == 'search' or
                            any(term in input_name.lower() for term in ['search', 'q', 'query', 'keyword', 'term', 'location', 'address', 'city']) or
                            any(term in input_id.lower() for term in ['search', 'q', 'query', 'keyword', 'term', 'location', 'address', 'city'])
                        )
                        
                        if is_search_input:
                            inputs.append({
                                "type": input_type,
                                "name": input_name,
                                "id": input_id,
                                "is_search_field": True
                            })
                        
                    form_data["inputs"] = inputs
                    
                    if inputs:  # Only include if we found viable inputs
                        result["has_search"] = True
                        result["form_data"] = form_data
                        return result
                        
                else:
                    # Non-form container (modern sites often use divs with JS)
                    container_data = {
                        "type": "js_component",
                        "id": element.get('id', ''),
                        "classes": ' '.join(element.get('class', [])),
                        "inputs": []
                    }
                    
                    # Find inputs inside this container
                    for input_el in element.find_all('input'):
                        input_type = input_el.get('type', '')
                        input_name = input_el.get('name', '')
                        input_id = input_el.get('id', '')
                        
                        # Skip hidden inputs
                        if input_type == 'hidden':
                            continue
                            
                        container_data["inputs"].append({
                            "type": input_type,
                            "name": input_name,
                            "id": input_id,
                            "is_search_field": True
                        })
                    
                    if container_data["inputs"]:  # Only include if we found inputs
                        result["has_search"] = True
                        result["form_data"] = container_data
                        return result
        
        # If we haven't found any form yet, try to find standalone search inputs
        standalone_inputs = soup.find_all('input', {'type': ['text', 'search']})
        for input_el in standalone_inputs:
            # Skip inputs inside forms (we've already checked forms)
            if input_el.find_parent('form'):
                continue
                
            # Check if it looks like a search input
            input_id = input_el.get('id', '')
            input_name = input_el.get('name', '')
            input_placeholder = input_el.get('placeholder', '')
            input_class = ' '.join(input_el.get('class', []))
            
            if (input_el.get('type') == 'search' or
                any(term in input_id.lower() for term in ['search', 'q', 'query', 'keyword']) or
                any(term in input_name.lower() for term in ['search', 'q', 'query', 'keyword']) or
                any(term in input_placeholder.lower() for term in ['search', 'find', 'enter']) or
                any(term in input_class.lower() for term in ['search', 'query', 'finder'])):
                
                result["has_search"] = True
                result["form_data"] = {
                    "type": "standalone_input",
                    "inputs": [{
                        "type": input_el.get('type', 'text'),
                        "name": input_name,
                        "id": input_id,
                        "is_search_field": True
                    }]
                }
                return result
        
        # Handle real estate specific inputs for location/address search
        if domain_type == "real_estate":
            # Fix the lambda function to properly handle attribute types
            location_inputs = []
            for input_el in soup.find_all('input'):
                # Check if each required attribute exists before accessing
                if (input_el.has_attr('placeholder') and 
                    any(loc in input_el['placeholder'].lower() for loc in ['city', 'location', 'address', 'zip'])):
                    location_inputs.append(input_el)
                elif (input_el.has_attr('name') and 
                      any(loc in input_el['name'].lower() for loc in ['city', 'location', 'address', 'zip'])):
                    location_inputs.append(input_el)
                elif (input_el.has_attr('id') and 
                      any(loc in input_el['id'].lower() for loc in ['city', 'location', 'address', 'zip'])):
                    location_inputs.append(input_el)
            
            if location_inputs:
                input_el = location_inputs[0]
                result["has_search"] = True
                result["form_data"] = {
                    "type": "real_estate_location_input",
                    "inputs": [{
                        "type": input_el.get('type', 'text'),
                        "name": input_el.get('name', ''),
                        "id": input_el.get('id', ''),
                        "is_search_field": True
                    }]
                }
                return result
        
        logger.info("No search form found on page")
        return result

    def _detect_domain_type(self, url: str) -> str:
        """
        Detect the type of domain from the URL to use specialized search detection.
        
        Args:
            url: URL of the website
        
        Returns:
            Domain type (e.g., "real_estate", "general")
        """
        url_lower = url.lower()
        
        # Real estate indicators
        real_estate_indicators = [
            'real', 'estate', 'property', 'properties', 'home', 'homes', 'house', 'realty',
            'broker', 'apartment', 'rent', 'realtor', 'zillow', 'trulia', 'redfin', 'listing',
            'mls', 'idx', 'housing'
        ]
        
        # Check if the URL contains real estate indicators
        if any(indicator in url_lower for indicator in real_estate_indicators):
            return "real_estate"
        
        # Add more domain type detection as needed
        
        # Default to general
        return "general"
        
    async def _try_url_based_search(self, url: str, search_term: str, domain_type: str = "general") -> List[str]:
        """
        Try URL-based search with domain-specific patterns
        
        Args:
            url: Base URL
            search_term: Search term to use
            domain_type: Type of domain (e.g., "real_estate", "e_commerce", "general")
            
        Returns:
            List of possible search URLs to try
        """
        parsed_url = urlparse(url)
        domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        domain_path = domain + parsed_url.path.rstrip('/')
        quoted_term = quote_plus(search_term)
        
        # General URL patterns
        general_patterns = [
            f"{domain}/search?q={quoted_term}",
            f"{domain}/search?query={quoted_term}",
            f"{domain}/search?keyword={quoted_term}",
            f"{domain}/search?s={quoted_term}",
            f"{domain}/search?term={quoted_term}",
            f"{domain}/search/{quoted_term}",
            f"{domain}?s={quoted_term}",
            f"{domain}?q={quoted_term}",
            f"{domain}?search={quoted_term}",
            f"{domain_path}?q={quoted_term}",
            f"{domain_path}?search={quoted_term}",
        ]
        
        # Domain-specific patterns
        if domain_type == "real_estate":
            real_estate_patterns = [
                f"{domain}/properties/search?q={quoted_term}",
                f"{domain}/property-search?location={quoted_term}",
                f"{domain}/homes?location={quoted_term}",
                f"{domain}/listings?search={quoted_term}",
                f"{domain}/mls-listings/search?q={quoted_term}",
                f"{domain}/search-mls-listings/?location={quoted_term}",
                f"{domain}/idx/search?location={quoted_term}",
                f"{domain}/idx/?location={quoted_term}",
                f"{domain}/property-search.php?location={quoted_term}",
                f"{domain}/properties/{quoted_term}",
                f"{domain}/search-listings/{quoted_term}"
            ]
            return real_estate_patterns + general_patterns
        
        elif domain_type == "e_commerce":
            ecommerce_patterns = [
                f"{domain}/products/search?q={quoted_term}",
                f"{domain}/shop/search?q={quoted_term}",
                f"{domain}/catalog/search?q={quoted_term}",
                f"{domain}/product-search?keyword={quoted_term}",
                f"{domain}/search-results?search={quoted_term}",
                f"{domain}/products?search={quoted_term}",
                f"{domain}/shop?search={quoted_term}",
                f"{domain}/store/search?q={quoted_term}"
            ]
            return ecommerce_patterns + general_patterns
            
        elif domain_type == "job_listing":
            job_patterns = [
                f"{domain}/jobs/search?q={quoted_term}",
                f"{domain}/careers/search?q={quoted_term}",
                f"{domain}/job-search?keyword={quoted_term}",
                f"{domain}/positions?search={quoted_term}",
                f"{domain}/vacancies?q={quoted_term}",
                f"{domain}/opportunities?keyword={quoted_term}"
            ]
            return job_patterns + general_patterns
            
        elif domain_type == "news":
            news_patterns = [
                f"{domain}/news/search?q={quoted_term}",
                f"{domain}/articles/search?q={quoted_term}",
                f"{domain}/stories?keyword={quoted_term}",
                f"{domain}/search-news?q={quoted_term}",
                f"{domain}/news?s={quoted_term}"
            ]
            return news_patterns + general_patterns
        
        # Return just general patterns for other domain types
        return general_patterns