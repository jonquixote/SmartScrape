# domain_intelligence.py - Domain-specific intelligence for SmartScrape

import re
import json
import logging
import time
import datetime
from collections import Counter
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse

# Import our enhanced utilities
from utils.html_utils import parse_html, extract_text_fast, extract_meta_tags, find_by_xpath
from utils.retry_utils import with_exponential_backoff
from utils.http_utils import fetch_html
# Import enhanced content extraction
from extraction.content_extraction import ContentExtractor
from extraction.content_analysis import ContentAnalyzer
# Change this line to use the correct class name
from extraction.content_evaluation import ContentEvaluator
from core.service_interface import BaseService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DomainIntelligence")

class DomainIntelligence(BaseService):
    def __init__(self):
        # Flag to track initialization status
        self._initialized = False
        # These will be initialized in initialize()
        self.domain_patterns = None
        self.content_type_patterns = None
        self.discovered_patterns = None
        self.content_extractor = None
        self.content_analyzer = None
        self.evaluation_engine = None
        
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the service with the given configuration."""
        if self._initialized:
            return
            
        # Initialize collections for patterns
        self.domain_patterns = {}
        self.content_type_patterns = {}
        self.discovered_patterns = {}
        
        # Initialize enhanced extraction components
        self.content_extractor = ContentExtractor()
        self.content_analyzer = ContentAnalyzer()
        self.evaluation_engine = ContentEvaluator()
        
        self._initialized = True
        logger.info("DomainIntelligence service initialized")
    
    def shutdown(self) -> None:
        """Perform cleanup and shutdown operations."""
        self._initialized = False
        logger.info("DomainIntelligence service shut down")
    
    @property
    def name(self) -> str:
        """Return the name of the service."""
        return "domain_intelligence"
        
    @with_exponential_backoff(max_attempts=3)
    async def analyze_domain(self, html, url):
        """
        Analyze a page to determine its domain/industry and content type
        in a completely content-agnostic way.
        
        Args:
            html: HTML content of the page
            url: URL of the page
            
        Returns:
            Dictionary with domain analysis results
        """
        try:
            domain = urlparse(url).netloc
            
            # Parse HTML with optimized lxml parser
            soup = parse_html(html)
            
            # Extract key elements
            title = soup.title.get_text() if soup.title else ""
            headings = []
            for h in soup.find_all(['h1', 'h2', 'h3']):
                headings.append(extract_text_fast(h))
            
            # Get meta description and keywords using optimized extraction
            meta_tags = extract_meta_tags(soup)
            meta_desc = meta_tags.get("description", "")
            
            # Extract keywords from meta tags
            meta_keywords = []
            if "keywords" in meta_tags:
                meta_keywords = meta_tags["keywords"].split(",")
                meta_keywords = [k.strip() for k in meta_keywords]
            
            # Collect text from important elements
            important_text = title + " " + meta_desc + " " + " ".join(headings)
            
            # Extract common structural patterns on the page
            content_patterns = self._extract_content_patterns(soup)
            
            # Detect content type based on page structure and text clues
            content_type = self._detect_content_type(soup, important_text, content_patterns)
            
            # Enhanced content analysis using the content analyzer
            content_analysis = await self.content_analyzer.analyze_page_content(soup, url)
            
            # Learn patterns from this page for future reference
            self._learn_patterns(domain, content_type, content_patterns)
            
            return {
                "domain": domain,
                "page_type": content_type,
                "content_patterns": content_patterns,
                "meta_keywords": meta_keywords,
                "content_analysis": content_analysis
            }
            
        except Exception as e:
            logger.error(f"Error analyzing domain: {str(e)}")
            return {"error": str(e)}
    
    def _extract_content_patterns(self, soup):
        """
        Extract common structural patterns from the page using optimized lxml selectors.
        
        Args:
            soup: BeautifulSoup object with the parsed HTML
            
        Returns:
            Dictionary of content patterns
        """
        patterns = {}
        
        # Look for repeating elements that might indicate listings
        # For example, product cards, article lists, etc.
        common_containers = ['div', 'article', 'section', 'li']
        common_classes = ['card', 'item', 'product', 'listing', 'post', 'article', 'result']
        
        for container in common_containers:
            for class_hint in common_classes:
                elements = soup.select(f"{container}[class*='{class_hint}']")
                if len(elements) >= 3:  # At least 3 repeating elements indicate a pattern
                    # Extract common attributes from these elements
                    common_attrs = self._extract_common_attributes(elements)
                    pattern_key = f"{container}.{class_hint}"
                    patterns[pattern_key] = {
                        "count": len(elements),
                        "common_attributes": common_attrs
                    }
        
        # Look for important data patterns (e.g., prices, dates, metrics)
        price_pattern = re.compile(r'(\$|€|£|¥)\s*[\d,]+(\.\d{2})?|\d+\s*(\$|€|£|¥)')
        price_elements = soup.find_all(string=price_pattern)
        if price_elements:
            patterns["price_elements"] = {
                "count": len(price_elements),
                "sample": str(price_elements[0]) if price_elements else ""
            }
        
        return patterns
    
    def _extract_common_attributes(self, elements):
        """
        Extract common attributes from a set of similar elements.
        """
        attributes = {}
        
        if not elements:
            return attributes
            
        # Sample the first 5 elements
        sample_size = min(5, len(elements))
        sample_elements = elements[:sample_size]
        
        # Get all text nodes within these elements
        all_text_nodes = []
        for element in sample_elements:
            text_nodes = [text for text in element.stripped_strings]
            all_text_nodes.append(text_nodes)
        
        # If elements have similar structure (same number of text nodes)
        if all(len(nodes) == len(all_text_nodes[0]) for nodes in all_text_nodes):
            # Count similar positions
            for i in range(len(all_text_nodes[0])):
                text_samples = []
                for nodes in all_text_nodes:
                    if i < len(nodes):
                        text_samples.append(nodes[i])
                
                # Detect what this position might represent
                position_type = self._classify_text_content(text_samples)
                if position_type:
                    attributes[f"position_{i}"] = position_type
        
        return attributes
    
    def _classify_text_content(self, text_samples):
        """
        Classify what type of data a set of text samples might represent.
        """
        if not text_samples:
            return None
            
        # Check if it might be a price
        price_pattern = re.compile(r'(\$|€|£|¥)\s*[\d,]+(\.\d{2})?|\d+\s*(\$|€|£|¥)')
        if all(price_pattern.search(str(text)) for text in text_samples):
            return "price"
            
        # Check if it might be a date
        date_pattern = re.compile(r'\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}|\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{2,4}')
        if all(date_pattern.search(str(text)) for text in text_samples):
            return "date"
            
        # Check if it might be a metric/number
        number_pattern = re.compile(r'\d+\s*(?:sq ft|sqft|m²|acres|bedrooms|baths|years|miles|km)')
        if all(number_pattern.search(str(text)) for text in text_samples):
            return "metric"
        
        # If similar length and format, might be a title or name
        if all(len(text) > 5 and len(text) < 100 for text in text_samples):
            return "title"
            
        return "general_text"
    
    def _detect_content_type(self, soup, important_text, content_patterns):
        """
        Detect the general type of content on a page.
        This is completely content-agnostic.
        """
        # Look for common indicators in the page structure
        
        # Product listing indicators
        product_indicators = ['price', 'buy', 'add to cart', 'product', 'shop', 'store']
        if any(indicator in important_text.lower() for indicator in product_indicators) or 'price_elements' in content_patterns:
            # Check if it's a single product or multiple products
            if content_patterns and any("count" in v and v["count"] >= 3 for k, v in content_patterns.items()):
                return "product_listing"
            return "product_detail"
            
        # Article indicators
        article_indicators = ['article', 'post', 'blog', 'news', 'published', 'author']
        if any(indicator in important_text.lower() for indicator in article_indicators):
            # Is it a list of articles or a single article?
            if soup.find('article') or len(soup.find_all('p')) > 5:
                return "article"
            if content_patterns and any("count" in v and v["count"] >= 3 for k, v in content_patterns.items()):
                return "article_listing"
                
        # Search results indicators
        search_indicators = ['search results', 'found', 'results for', 'showing']
        if any(indicator in important_text.lower() for indicator in search_indicators):
            return "search_results"
            
        # Contact page indicators
        contact_indicators = ['contact', 'email', 'phone', 'address', 'reach us', 'get in touch']
        if any(indicator in important_text.lower() for indicator in contact_indicators):
            return "contact_page"
            
        # About page indicators
        about_indicators = ['about us', 'our story', 'mission', 'history', 'who we are']
        if any(indicator in important_text.lower() for indicator in about_indicators):
            return "about_page"
            
        # Check for home page
        if not soup.select('body > div > div > div > div') or urlparse(soup.url).path in ['/', '']:
            return "homepage"
            
        # Check for content-rich page that doesn't match other patterns
        if len(soup.find_all('p')) > 10:
            return "content_page"
            
        # Check for listing page based on repeating elements
        if content_patterns and any("count" in v and v["count"] >= 5 for k, v in content_patterns.items()):
            return "listing_page"
            
        # Default
        return "general"
    
    def _learn_patterns(self, domain, content_type, content_patterns):
        """
        Learn patterns from this page for future reference.
        This creates a self-improving system that gets better over time.
        """
        domain_key = domain.split('.')[-2] if '.' in domain else domain
        
        if domain_key not in self.discovered_patterns:
            self.discovered_patterns[domain_key] = {
                'content_types': Counter(),
                'patterns': {}
            }
            
        # Update content type counter
        self.discovered_patterns[domain_key]['content_types'][content_type] += 1
        
        # Update patterns for this domain
        for pattern_key, pattern_data in content_patterns.items():
            if pattern_key not in self.discovered_patterns[domain_key]['patterns']:
                self.discovered_patterns[domain_key]['patterns'][pattern_key] = pattern_data
            else:
                # Update existing pattern data
                existing_data = self.discovered_patterns[domain_key]['patterns'][pattern_key]
                if "count" in existing_data and "count" in pattern_data:
                    # Calculate rolling average of counts
                    existing_data["count"] = (existing_data["count"] + pattern_data["count"]) / 2
                    
                # Merge common_attributes if present in both
                if "common_attributes" in existing_data and "common_attributes" in pattern_data:
                    for attr_key, attr_val in pattern_data["common_attributes"].items():
                        if attr_key not in existing_data["common_attributes"]:
                            existing_data["common_attributes"][attr_key] = attr_val
                            
    def get_extraction_schema_for_content_type(self, content_type):
        """
        Return a generic extraction schema based on content type.
        This is completely content-agnostic and can adapt to any type of content.
        """
        # Define generic schemas for different content types
        generic_schemas = {
            "product_detail": {
                "title": ["h1", "product-title", "title"],
                "price": ["price", ".price", "[class*='price']", "span.amount"],
                "description": ["description", "product-description", ".details", "[class*='description']"],
                "features": ["features", "specifications", "specs", "[class*='feature']", "ul li"],
                "images": ["img.product-image", ".product img", "img[class*='product']", "img[class*='main']"]
            },
            "product_listing": {
                "items": [".product", "[class*='product']", ".item", "[class*='item']"],
                "item_title": ["h3", "h2", ".title", "[class*='title']"],
                "item_price": [".price", "[class*='price']", "span.amount"],
                "item_image": ["img", ".product-image", "img[class*='product']"]
            },
            "article": {
                "title": ["h1", ".article-title", "[class*='title']"],
                "date": ["[class*='date']", "[class*='published']", "time"],
                "author": ["[class*='author']", "[rel='author']"],
                "content": ["[class*='content']", "[class*='article-body']", ".post-content", "article"],
                "images": ["img", "[class*='featured-image']"]
            },
            "article_listing": {
                "items": ["article", ".post", "[class*='article']", "[class*='post']"],
                "item_title": ["h2", "h3", ".title", "[class*='title']"],
                "item_date": ["[class*='date']", "time"],
                "item_excerpt": ["[class*='excerpt']", "[class*='summary']", "p"]
            },
            "listing_page": {
                "items": [".item", "[class*='item']", ".listing", "[class*='listing']", "li.result"],
                "item_title": ["h2", "h3", ".title", "[class*='title']"],
                "item_details": ["p", ".details", "[class*='description']"],
                "item_image": ["img"]
            },
            "search_results": {
                "result_count": ["[class*='count']", "[class*='results']"],
                "items": ["[class*='result']", ".item", "li"],
                "item_title": ["h2", "h3", ".title", "[class*='title']"],
                "item_snippet": ["[class*='snippet']", "[class*='description']", "p"]
            },
            "general": {
                "title": ["h1", "h2", ".title"],
                "content": ["[class*='content']", "[class*='body']", "article", "section", ".main"],
                "images": ["img"]
            }
        }
        
        # Return the appropriate schema or a generic one if not found
        if content_type in generic_schemas:
            return generic_schemas[content_type]
        else:
            return generic_schemas["general"]
            
    @with_exponential_backoff(max_attempts=3)
    async def extract_data_from_html(self, html, url, extraction_schema=None):
        """
        Extract data from HTML based on a schema using enhanced extraction techniques.
        
        Args:
            html: HTML content
            url: URL of the page
            extraction_schema: Optional extraction schema to use
            
        Returns:
            Dictionary of extracted data
        """
        try:
            # Parse HTML with optimized lxml parser
            soup = parse_html(html)
            
            # If no schema provided, detect content type and use appropriate schema
            if not extraction_schema:
                content_type = self._detect_content_type(soup, soup.title.string if soup.title else "", {})
                extraction_schema = self.get_extraction_schema_for_content_type(content_type)
            
            # Enhanced extraction using our content extractor
            enhanced_extraction = await self.content_extractor.extract_with_schema(
                soup, 
                url, 
                extraction_schema
            )
            
            # Evaluate the extraction quality
            evaluation = await self.evaluation_engine.evaluate_extraction(
                enhanced_extraction,
                soup, 
                url
            )
            
            # If the evaluation score is low, try fallback extraction
            if evaluation.get("quality_score", 0) < 0.7:
                logger.info(f"Low quality extraction detected, using fallback extraction for {url}")
                from extraction.fallback_extraction import extract_with_fallback
                enhanced_extraction = await extract_with_fallback(soup, url, extraction_schema)
            
            # Add extraction metadata
            enhanced_extraction["_meta"] = {
                "extraction_timestamp": datetime.datetime.now().isoformat(),
                "extraction_quality": evaluation.get("quality_score", 0),
                "extraction_method": "enhanced" if evaluation.get("quality_score", 0) >= 0.7 else "fallback"
            }
            
            return enhanced_extraction
                
        except Exception as e:
            logger.error(f"Error extracting data from HTML: {str(e)}")
            return {"error": str(e)}
            
    async def get_enhanced_extraction_schema(self, content_type, url=None, sample_html=None):
        """
        Get an enhanced extraction schema that includes additional fields and extraction strategies.
        
        Args:
            content_type: Type of content to extract
            url: Optional URL to help with domain-specific customization
            sample_html: Optional sample HTML to analyze for extraction hints
            
        Returns:
            Enhanced extraction schema
        """
        # Start with the basic schema
        base_schema = self.get_extraction_schema_for_content_type(content_type)
        
        # Enhanced schema with additional extraction strategies
        enhanced_schema = {
            "base": base_schema,
            "extraction_strategies": [
                "css_selector",
                "xpath",
                "semantic",
                "visual_cues",
                "ai_assisted"
            ],
            "validation_rules": {
                "required_fields": [],
                "field_types": {}
            },
            "normalization": {
                "date_formats": ["auto_detect", "ISO"],
                "price_format": "decimal",
                "text_cleanup": True
            }
        }
        
        # Add content type specific enhancements
        if content_type == "product_detail":
            enhanced_schema["validation_rules"]["required_fields"] = ["title", "price"]
            enhanced_schema["validation_rules"]["field_types"] = {
                "price": "currency",
                "features": "list"
            }
            
        elif content_type == "article":
            enhanced_schema["validation_rules"]["required_fields"] = ["title", "content"]
            enhanced_schema["validation_rules"]["field_types"] = {
                "date": "datetime",
                "content": "html_or_text"
            }
            
        elif content_type == "listing_page":
            enhanced_schema["validation_rules"]["required_fields"] = ["items"]
            enhanced_schema["pagination"] = {
                "detect": True,
                "strategies": ["url_pattern", "next_button", "load_more"]
            }
        
        return enhanced_schema
    
    @with_exponential_backoff(max_attempts=3)
    async def detect_website_type(self, url, html_content=None):
        """
        Detect the type of website based on URL and optional HTML content.
        
        Args:
            url: Website URL
            html_content: Optional HTML content
            
        Returns:
            Website type as string (e.g., "real_estate", "e_commerce")
        """
        domain = urlparse(url).netloc.lower()
        
        # Real estate detection
        real_estate_domains = ["zillow", "realtor", "redfin", "trulia", "homes", "realty", "property", "estate"]
        if any(term in domain for term in real_estate_domains):
            return "real_estate"
            
        # E-commerce detection
        ecommerce_domains = ["amazon", "ebay", "walmart", "shop", "store", "buy", "product"]
        if any(term in domain for term in ecommerce_domains):
            return "e_commerce"
            
        # Job listings detection
        job_domains = ["indeed", "monster", "linkedin", "career", "job", "employment"]
        if any(term in domain for term in job_domains):
            return "job_listings"
            
        # News/media detection
        news_domains = ["news", "media", "times", "post", "tribune", "herald"]
        if any(term in domain for term in news_domains):
            return "news"
            
        # If we have HTML content, try more detailed detection
        if html_content:
            # Fetch HTML if not provided
            if not html_content:
                try:
                    html_content = await fetch_html(url)
                except Exception as e:
                    logger.error(f"Error fetching HTML: {str(e)}")
                    return "general"
                    
            html_lower = html_content.lower()
            
            # Check for real estate indicators
            real_estate_terms = ["real estate", "property", "home", "house", "apartment", "listing", "realty"]
            if any(term in html_lower for term in real_estate_terms):
                return "real_estate"
                
            # Check for e-commerce indicators
            ecommerce_terms = ["shop", "store", "buy", "cart", "checkout", "product", "price"]
            if any(term in html_lower for term in ecommerce_terms):
                return "e_commerce"
                
            # Check for job listings indicators
            job_terms = ["job", "career", "employ", "hiring", "resume", "cv", "position"]
            if any(term in html_lower for term in job_terms):
                return "job_listings"
                
        # Default to general if no specific type detected
        return "general"
        
    def get_specialized_extraction_config(self, website_type, config=None):
        """
        Get a specialized extraction configuration for different types of websites.
        
        Args:
            website_type: Type of website (e.g., "real_estate", "e_commerce")
            config: Optional configuration dictionary with additional parameters
            
        Returns:
            Dictionary with specialized extraction configuration
        """
        # Default configuration for general websites
        default_config = {
            "include_tables": True,
            "extract_schema": True,
            "extract_main_content": True,
            "extract_title": True,
            "extract_links": True,
            "extract_images": True,
        }
        
        # Specialized configs for different site types
        configs = {
            "real_estate": {
                **default_config,
                "css_selectors": {
                    "property_details": [".property-details", ".listing-details", "[class*='property-detail']"],
                    "price": [".price", "[class*='price']", "[data-label*='price']"],
                    "address": [".address", "[class*='address']", "[itemprop='address']"],
                    "features": [".features", ".details", ".specs", "[data-label*='feature']"],
                    "bedrooms": [".beds", "[class*='bed']", "[data-label*='bed']"],
                    "bathrooms": [".baths", "[class*='bath']", "[data-label*='bath']"],
                    "square_feet": [".sqft", "[class*='square']", "[data-label*='sqft']"],
                    "description": [".description", "[class*='description']", "[itemprop='description']"]
                },
                "extract_schema": True,
                "schema_types": ["Product", "Place", "Residence", "House", "Apartment"],
                "table_extraction_strategy": "simple"
            },
            
            "e_commerce": {
                **default_config,
                "css_selectors": {
                    "product_details": ["#product", ".product", "[itemtype*='Product']"],
                    "price": [".price", "[class*='price']", "[itemprop='price']"],
                    "title": ["h1", ".product-title", "[itemprop='name']"],
                    "description": [".description", "#description", "[itemprop='description']"],
                    "sku": ["[itemprop='sku']", ".sku"],
                    "availability": ["[itemprop='availability']", ".availability"],
                    "reviews": [".reviews", "#reviews", "[itemprop='review']"],
                    "rating": [".rating", "[itemprop='ratingValue']"]
                },
                "extract_schema": True,
                "schema_types": ["Product", "Offer", "AggregateOffer", "Review"],
                "extract_json_ld": True,
                "images_max_count": 10
            },
            
            "job_listings": {
                **default_config,
                "css_selectors": {
                    "job_title": ["h1", ".job-title", "[class*='title']"],
                    "company": [".company", "[class*='company']", "[data-test*='company']"],
                    "location": [".location", "[class*='location']", "[data-test*='location']"],
                    "salary": [".salary", "[class*='salary']", "[data-test*='salary']"],
                    "job_description": [".description", "#job-description", "[class*='description']"],
                    "requirements": [".requirements", "[class*='requirement']"],
                    "benefits": [".benefits", "[class*='benefit']"],
                    "application_info": [".apply", "[class*='apply']"]
                },
                "extract_schema": True,
                "schema_types": ["JobPosting"]
            },
            
            "news": {
                **default_config,
                "css_selectors": {
                    "headline": ["h1", ".headline", "[itemprop='headline']"],
                    "article_body": ["[itemprop='articleBody']", ".article-body", ".content"],
                    "author": ["[itemprop='author']", ".author", "[class*='author']"],
                    "published_date": ["[itemprop='datePublished']", ".published-date", "[class*='date']"],
                    "category": ["[itemprop='articleSection']", ".category"],
                    "tags": [".tags", "[class*='tag']"]
                },
                "extract_schema": True,
                "schema_types": ["NewsArticle", "Article", "WebPage"]
            },
            
            "general": default_config
        }
        
        # Return the specialized config or fall back to the general config
        return configs.get(website_type, default_config)
    
    def extract_keywords(self, text, max_keywords=10):
        """
        Extract keywords from text for relevance matching.
        
        Args:
            text: Text to extract keywords from
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List of keywords
        """
        if not text:
            return []
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and split into words
        words = re.sub(r'[^\w\s]', ' ', text).split()
        
        # Remove common stop words that don't add much value for matching
        stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
            'when', 'where', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
            'most', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should',
            'now', 'to', 'from', 'by', 'for', 'with', 'about', 'against', 'between',
            'into', 'through', 'during', 'before', 'after', 'above', 'below', 'of',
            'at', 'on', 'in', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'i', 'me',
            'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her',
            'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
            'themselves', 'this', 'that', 'these', 'those', 'am', 'get', 'got', 'would',
            'could', 'should'
        }
        
        # Filter out stop words and short words
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Count word frequencies
        word_counts = {}
        for word in filtered_words:
            word_counts[word] = word_counts.get(word, 0) + 1
            
        # Sort by frequency (descending)
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Extract the most frequent keywords
        keywords = [word for word, _ in sorted_words[:max_keywords]]
        
        return keywords
    
    def is_search_oriented_request(self, description):
        """
        Determine if a request is search-oriented.
        
        Args:
            description: The request description
            
        Returns:
            bool: Whether the request is search-oriented
        """
        if not description:
            return False
            
        # Check for search-related terms
        search_terms = ['search', 'find', 'look for', 'locate', 'query']
        return any(term in description.lower() for term in search_terms)
    
    def extract_search_terms(self, description):
        """
        Extract search terms from a description.
        
        Args:
            description: The description to extract search terms from
            
        Returns:
            The extracted search terms
        """
        if not description:
            return ""
            
        # Simple extraction - after search-related terms
        search_indicators = ['search for', 'find', 'look for', 'search', 'locate']
        
        # Find the first indicator and extract what comes after it
        description_lower = description.lower()
        for indicator in search_indicators:
            if indicator in description_lower:
                idx = description_lower.find(indicator) + len(indicator)
                search_terms = description[idx:].strip()
                # Clean up any trailing instructions
                for end_marker in ['and scrape', 'and extract', ', then', '. then']:
                    if end_marker in search_terms.lower():
                        search_terms = search_terms.lower().split(end_marker)[0].strip()
                return search_terms
                
        # Fallback: use the entire description
        return description
    
    # E-commerce specific term handling (Task 2.2.1)
    def process_ecommerce_search_terms(self, search_terms, options=None):
        """
        Process search terms specifically for e-commerce sites.
        
        Args:
            search_terms: Original search terms
            options: Optional parameters controlling processing behavior
            
        Returns:
            Dict with processed search terms and metadata
        """
        options = options or {}
        result = {
            "original_terms": search_terms,
            "processed_terms": search_terms,
            "variations": [],
            "filters": {},
            "sort_options": [],
            "category_hints": []
        }
        
        # Normalize terms
        processed = search_terms.strip()
        
        # Extract price ranges
        price_pattern = re.compile(r'(\$|€|£|¥)?\s*(\d+)(?:\s*-\s*(\$|€|£|¥)?\s*(\d+))?')
        price_match = price_pattern.search(processed)
        if price_match:
            if price_match.group(4):  # It's a range
                min_price = price_match.group(2)
                max_price = price_match.group(4)
                result["filters"]["price_range"] = {
                    "min": min_price,
                    "max": max_price
                }
            else:  # It's a single price
                result["filters"]["price"] = price_match.group(2)
                
            # Remove price info from main search term
            processed = price_pattern.sub('', processed).strip()
            
        # Extract brand information
        brand_patterns = [
            r'by (\w+)',
            r'from (\w+)',
            r'(\w+) brand'
        ]
        for pattern in brand_patterns:
            brand_match = re.search(pattern, processed, re.IGNORECASE)
            if brand_match:
                result["filters"]["brand"] = brand_match.group(1)
                processed = re.sub(pattern, '', processed, flags=re.IGNORECASE).strip()
                break
                
        # Detect product categories
        categories = {
            "electronics": ["phone", "laptop", "computer", "tv", "camera", "headphone", "speaker", "tablet"],
            "clothing": ["shirt", "pants", "dress", "shoe", "jacket", "hat", "sock", "jean"],
            "home": ["furniture", "bed", "sofa", "chair", "table", "lamp", "rug", "desk"],
            "toys": ["toy", "game", "puzzle", "doll", "action figure", "lego", "board game"],
            "beauty": ["makeup", "cosmetic", "skincare", "fragrance", "perfume", "shampoo"],
            "books": ["book", "novel", "textbook", "cookbook", "magazine"]
        }
        
        for category, terms in categories.items():
            if any(term in processed.lower() for term in terms):
                result["category_hints"].append(category)
                
        # Extract condition (new, used, refurbished)
        condition_terms = ["new", "used", "refurbished", "pre-owned", "open box"]
        for term in condition_terms:
            if term in processed.lower():
                result["filters"]["condition"] = term
                processed = processed.lower().replace(term, "").strip()
                break
                
        # Generate common shopping variations
        variations = []
        variations.append(f"best {processed}")
        variations.append(f"{processed} review")
        variations.append(f"top rated {processed}")
        variations.append(f"{processed} deal")
        
        # Add size variations if applicable
        size_pattern = re.compile(r'\b(small|medium|large|xl|xxl|xs|s|m|l)\b', re.IGNORECASE)
        if size_pattern.search(processed):
            result["filters"]["size"] = size_pattern.search(processed).group(1)
            base_term = size_pattern.sub('', processed).strip()
            # Add other sizes as variations
            sizes = ["small", "medium", "large", "xl"] 
            for size in sizes:
                if size.lower() != result["filters"]["size"].lower():
                    variations.append(f"{base_term} {size}")
        
        # Sort options common for e-commerce
        result["sort_options"] = [
            "price low to high",
            "price high to low",
            "best selling",
            "highest rated",
            "newest arrivals"
        ]
        
        # Clean up the processed terms
        result["processed_terms"] = processed
        result["variations"] = variations
        
        return result
    
    # Real estate and location-aware search (Task 2.2.2)
    def process_real_estate_search_terms(self, search_terms, options=None):
        """
        Process search terms specifically for real estate sites with location awareness.
        
        Args:
            search_terms: Original search terms
            options: Optional parameters controlling processing behavior
            
        Returns:
            Dict with processed search terms and metadata
        """
        options = options or {}
        result = {
            "original_terms": search_terms,
            "processed_terms": search_terms,
            "variations": [],
            "location_info": {},
            "property_attributes": {},
            "filters": {},
            "sort_options": []
        }
        
        # Normalize terms
        processed = search_terms.strip()
        
        # Extract location information
        # Look for city, state, or zip code patterns
        location_patterns = [
            # City, State pattern
            (r'in\s+([A-Za-z\s]+),\s+([A-Z]{2})', 'city_state'),
            # Just city
            (r'in\s+([A-Za-z\s]+)', 'city'),
            # Zip code
            (r'in\s+(\d{5}(?:-\d{4})?)', 'zip')
        ]
        
        for pattern, location_type in location_patterns:
            location_match = re.search(pattern, processed, re.IGNORECASE)
            if location_match:
                if location_type == 'city_state':
                    result["location_info"]["city"] = location_match.group(1).strip()
                    result["location_info"]["state"] = location_match.group(2)
                elif location_type == 'city':
                    result["location_info"]["city"] = location_match.group(1).strip()
                elif location_type == 'zip':
                    result["location_info"]["zip"] = location_match.group(1)
                    
                # Remove location info from main search term
                processed = re.sub(pattern, '', processed, flags=re.IGNORECASE).strip()
                break
        
        # Extract price ranges
        price_pattern = re.compile(r'(\$|€|£|¥)?\s*(\d+)(?:k|K)?\s*(?:-|to)\s*(\$|€|£|¥)?\s*(\d+)(?:k|K)?')
        price_match = price_pattern.search(processed)
        if price_match:
            min_price = price_match.group(2)
            max_price = price_match.group(4)
            # Convert k to 000
            min_price = min_price + "000" if "k" in price_match.group(0).lower() and min_price.isdigit() else min_price
            max_price = max_price + "000" if "k" in price_match.group(0).lower() and max_price.isdigit() else max_price
            result["filters"]["price_range"] = {
                "min": min_price,
                "max": max_price
            }
            # Remove price info from main search term
            processed = price_pattern.sub('', processed).strip()
        
        # Extract property type
        property_types = {
            "house": ["house", "home", "single family", "detached"],
            "apartment": ["apartment", "apt", "condo", "flat"],
            "townhouse": ["townhouse", "townhome", "town house"],
            "land": ["land", "lot", "acreage", "vacant"],
            "commercial": ["commercial", "office", "retail", "warehouse"]
        }
        
        for prop_type, terms in property_types.items():
            if any(term in processed.lower() for term in terms):
                result["property_attributes"]["type"] = prop_type
                # Don't remove the property type as it's often integral to the search
        
        # Extract bedrooms/bathrooms
        bed_bath_pattern = re.compile(r'(\d+)\s*(?:bed|bedroom|br|bd).*?(\d+)\s*(?:bath|bathroom|ba|bth)?')
        bed_bath_match = bed_bath_pattern.search(processed.lower())
        if bed_bath_match:
            result["property_attributes"]["bedrooms"] = bed_bath_match.group(1)
            if bed_bath_match.group(2):
                result["property_attributes"]["bathrooms"] = bed_bath_match.group(2)
            # Remove bed/bath info from main search term to clean it up
            processed = bed_bath_pattern.sub('', processed).strip()
        else:
            # Try just bedrooms
            bed_pattern = re.compile(r'(\d+)\s*(?:bed|bedroom|br|bd)')
            bed_match = bed_pattern.search(processed.lower())
            if bed_match:
                result["property_attributes"]["bedrooms"] = bed_match.group(1)
                processed = bed_pattern.sub('', processed).strip()
                
            # Try just bathrooms
            bath_pattern = re.compile(r'(\d+)\s*(?:bath|bathroom|ba|bth)')
            bath_match = bath_pattern.search(processed.lower())
            if bath_match:
                result["property_attributes"]["bathrooms"] = bath_match.group(1)
                processed = bath_pattern.sub('', processed).strip()
        
        # Extract square footage
        sqft_pattern = re.compile(r'(\d+)\s*(?:sq ft|sqft|square feet|sf)')
        sqft_match = sqft_pattern.search(processed.lower())
        if sqft_match:
            result["property_attributes"]["square_feet"] = sqft_match.group(1)
            processed = sqft_pattern.sub('', processed).strip()
            
        # Extract lot size
        lot_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*(?:acre|acres)')
        lot_match = lot_pattern.search(processed.lower())
        if lot_match:
            result["property_attributes"]["lot_size"] = {
                "value": lot_match.group(1),
                "unit": "acres"
            }
            processed = lot_pattern.sub('', processed).strip()
            
        # Special features extraction
        features = [
            "garage", "pool", "basement", "fireplace", "waterfront", 
            "new construction", "remodeled", "updated", "view",
            "fenced yard", "central air", "balcony", "deck"
        ]
        
        extracted_features = []
        for feature in features:
            if feature in processed.lower():
                extracted_features.append(feature)
                # Keep these in the search term as they're important
        
        if extracted_features:
            result["property_attributes"]["features"] = extracted_features
            
        # Generate location-based variations
        variations = []
        if "city" in result["location_info"]:
            city = result["location_info"]["city"]
            city_variations = [
                f"{city} real estate",
                f"homes for sale in {city}",
                f"houses in {city}"
            ]
            variations.extend(city_variations)
            
            # Add proximity searches
            variations.append(f"near {city}")
            variations.append(f"within 10 miles of {city}")
            
        # Sort options common for real estate
        result["sort_options"] = [
            "price low to high",
            "price high to low",
            "newest first",
            "most sq ft",
            "lot size",
            "closest to location"
        ]
        
        # Clean up the processed terms
        result["processed_terms"] = processed
        result["variations"] = variations
        
        return result
    
    # Job search specific term handling (Task 2.2.3)
    def process_job_search_terms(self, search_terms, options=None):
        """
        Process search terms specifically for job search sites.
        
        Args:
            search_terms: Original search terms
            options: Optional parameters controlling processing behavior
            
        Returns:
            Dict with processed search terms and metadata
        """
        options = options or {}
        result = {
            "original_terms": search_terms,
            "processed_terms": search_terms,
            "variations": [],
            "location_info": {},
            "job_attributes": {},
            "filters": {},
            "sort_options": []
        }
        
        # Normalize terms
        processed = search_terms.strip()
        
        # Extract location information similar to real estate
        location_patterns = [
            # City, State pattern
            (r'in\s+([A-Za-z\s]+),\s+([A-Z]{2})', 'city_state'),
            # Just city
            (r'in\s+([A-Za-z\s]+)', 'city'),
            # Remote work
            (r'\b(remote|work from home|wfh)\b', 'remote')
        ]
        
        for pattern, location_type in location_patterns:
            location_match = re.search(pattern, processed, re.IGNORECASE)
            if location_match:
                if location_type == 'city_state':
                    result["location_info"]["city"] = location_match.group(1).strip()
                    result["location_info"]["state"] = location_match.group(2)
                elif location_type == 'city':
                    result["location_info"]["city"] = location_match.group(1).strip()
                elif location_type == 'remote':
                    result["job_attributes"]["remote"] = True
                    
                # Remove location info from main search term
                processed = re.sub(pattern, '', processed, flags=re.IGNORECASE).strip()
                break
                
        # Extract salary information
        salary_pattern = re.compile(r'(\$|€|£|¥)?\s*(\d+)(?:k|K)?\s*(?:-|to)\s*(\$|€|£|¥)?\s*(\d+)(?:k|K)?')
        salary_match = salary_pattern.search(processed)
        if salary_match:
            min_salary = salary_match.group(2)
            max_salary = salary_match.group(4)
            # Convert k to 000
            min_salary = min_salary + "000" if "k" in salary_match.group(0).lower() and min_salary.isdigit() else min_salary
            max_salary = max_salary + "000" if "k" in salary_match.group(0).lower() and max_salary.isdigit() else max_salary
            result["job_attributes"]["salary_range"] = {
                "min": min_salary,
                "max": max_salary
            }
            # Remove salary info from main search term
            processed = salary_pattern.sub('', processed).strip()
            
        # Extract job type
        job_types = {
            "full_time": ["full time", "full-time", "ft"],
            "part_time": ["part time", "part-time", "pt"],
            "contract": ["contract", "contractor", "freelance"],
            "temporary": ["temp", "temporary"],
            "internship": ["intern", "internship"]
        }
        
        for job_type, terms in job_types.items():
            if any(term in processed.lower() for term in terms):
                result["job_attributes"]["job_type"] = job_type
                # Remove job type from search term
                for term in terms:
                    processed = re.sub(r'\b' + re.escape(term) + r'\b', '', processed, flags=re.IGNORECASE).strip()
                    
        # Extract experience level
        experience_levels = {
            "entry_level": ["entry level", "entry-level", "junior", "jr"],
            "mid_level": ["mid level", "mid-level", "intermediate"],
            "senior": ["senior", "sr", "experienced", "expert"],
            "manager": ["manager", "director", "lead"],
            "executive": ["executive", "vp", "c-level", "chief"]
        }
        
        for level, terms in experience_levels.items():
            if any(term in processed.lower() for term in terms):
                result["job_attributes"]["experience_level"] = level
                # Keep these in the search term as they're important
                
        # Education requirements
        education_levels = {
            "high_school": ["high school", "hs", "ged"],
            "associate": ["associate", "aa", "as"],
            "bachelor": ["bachelor", "bs", "ba", "bsc", "undergraduate"],
            "master": ["master", "ms", "ma", "msc", "graduate"],
            "phd": ["phd", "doctorate", "doctoral"]
        }
        
        for level, terms in education_levels.items():
            if any(term in processed.lower() for term in terms):
                result["job_attributes"]["education"] = level
                # Remove from search terms
                for term in terms:
                    processed = re.sub(r'\b' + re.escape(term) + r'\b', '', processed, flags=re.IGNORECASE).strip()
        
        # Job industry extraction
        industries = [
            "technology", "healthcare", "finance", "education", 
            "retail", "manufacturing", "hospitality", "marketing",
            "legal", "engineering", "science", "non-profit"
        ]
        
        for industry in industries:
            if industry in processed.lower():
                result["job_attributes"]["industry"] = industry
                processed = re.sub(r'\b' + re.escape(industry) + r'\b', '', processed, flags=re.IGNORECASE).strip()
                break
                
        # Skills extraction
        common_skills = [
            "python", "javascript", "java", "c++", "html", "css", "sql",
            "aws", "azure", "react", "angular", "node.js", "django",
            "excel", "powerpoint", "photoshop", "illustrator", 
            "project management", "scrum", "agile", "customer service"
        ]
        
        extracted_skills = []
        for skill in common_skills:
            if skill.lower() in processed.lower():
                extracted_skills.append(skill)
                # Keep skills in the search term as they're crucial
                
        if extracted_skills:
            result["job_attributes"]["skills"] = extracted_skills
            
        # Generate variations
        variations = []
        
        # Base job variations
        job_title = processed.strip()
        if job_title:
            variations.extend([
                f"{job_title} jobs",
                f"entry level {job_title}",
                f"senior {job_title}",
                f"{job_title} career"
            ])
            
        # With location
        if "city" in result["location_info"]:
            city = result["location_info"]["city"]
            variations.append(f"{job_title} jobs in {city}")
            variations.append(f"{job_title} {city}")
            
        # With remote
        if "remote" in result["job_attributes"] and result["job_attributes"]["remote"]:
            variations.append(f"remote {job_title}")
            variations.append(f"{job_title} work from home")
            
        # Sort options common for job search
        result["sort_options"] = [
            "date posted",
            "relevance",
            "salary high to low",
            "distance"
        ]
        
        # Clean up the processed terms
        result["processed_terms"] = processed
        result["variations"] = variations
        
        return result
    
    # Academic and research term processing (Task 2.2.4)
    def process_academic_research_terms(self, search_terms, options=None):
        """
        Process search terms specifically for academic and research queries.
        
        Args:
            search_terms: Original search terms
            options: Optional parameters controlling processing behavior
            
        Returns:
            Dict with processed search terms and metadata
        """
        options = options or {}
        result = {
            "original_terms": search_terms,
            "processed_terms": search_terms,
            "variations": [],
            "academic_attributes": {},
            "filters": {},
            "operators": [],
            "sort_options": []
        }
        
        # Normalize terms
        processed = search_terms.strip()
        
        # Extract date/year range
        year_pattern = re.compile(r'(\d{4})\s*(?:-|to)\s*(\d{4})')
        year_match = year_pattern.search(processed)
        if year_match:
            start_year = year_match.group(1)
            end_year = year_match.group(2)
            result["filters"]["year_range"] = {
                "start": start_year,
                "end": end_year
            }
            # Remove year info from main search term
            processed = year_pattern.sub('', processed).strip()
        else:
            # Look for single year
            single_year_pattern = re.compile(r'in\s+(\d{4})')
            single_year_match = single_year_pattern.search(processed)
            if single_year_match:
                result["filters"]["year"] = single_year_match.group(1)
                processed = single_year_pattern.sub('', processed).strip()
                
        # Extract publication type
        pub_types = {
            "journal": ["journal", "article"],
            "conference": ["conference", "proceedings", "symposium"],
            "book": ["book", "textbook", "monograph"],
            "thesis": ["thesis", "dissertation"],
            "report": ["report", "technical paper", "white paper"]
        }
        
        for pub_type, terms in pub_types.items():
            if any(term in processed.lower() for term in terms):
                result["academic_attributes"]["publication_type"] = pub_type
                # Remove publication type for cleaner search terms
                for term in terms:
                    processed = re.sub(r'\b' + re.escape(term) + r'\b', '', processed, flags=re.IGNORECASE).strip()
                break
                
        # Extract author information
        author_pattern = re.compile(r'by\s+([A-Za-z\s\.]+)')
        author_match = author_pattern.search(processed)
        if author_match:
            result["academic_attributes"]["author"] = author_match.group(1).strip()
            processed = author_pattern.sub('', processed).strip()
            
        # Extract field of study
        fields = [
            "computer science", "physics", "chemistry", "biology", 
            "mathematics", "psychology", "sociology", "economics",
            "medicine", "engineering", "linguistics", "history",
            "philosophy", "political science", "law", "education"
        ]
        
        for field in fields:
            field_pattern = re.compile(r'\b' + re.escape(field) + r'\b', re.IGNORECASE)
            if field_pattern.search(processed):
                result["academic_attributes"]["field"] = field
                # Keep field in the search term as it's important for context
                break
                
        # Generate academic search operators
        operators = [
            'AND', 'OR', 'NOT', 
            'intitle:', 'inauthor:', 'insubject:',
            'filetype:pdf'
        ]
        result["operators"] = operators
        
        # Create advanced query variations
        variations = []
        
        # Base topic variations
        base_term = processed.strip()
        if base_term:
            # Add quotes for exact phrase search
            if " " in base_term and not base_term.startswith('"') and not base_term.endswith('"'):
                variations.append(f'"{base_term}"')
                
            # Add field-specific variations
            if "field" in result["academic_attributes"]:
                field = result["academic_attributes"]["field"]
                variations.append(f'{base_term} in {field}')
                
            # Add publication-specific variations
            if "publication_type" in result["academic_attributes"]:
                pub_type = result["academic_attributes"]["publication_type"]
                variations.append(f'{base_term} {pub_type}')
                
            # Add review/survey variations
            variations.append(f'review of {base_term}')
            variations.append(f'survey of {base_term}')
            variations.append(f'{base_term} state of the art')
            
            # Add methodology variations
            variations.append(f'{base_term} methodology')
            variations.append(f'{base_term} framework')
            variations.append(f'{base_term} analysis')
            
        # Add specific search operators
        if "author" in result["academic_attributes"]:
            author = result["academic_attributes"]["author"]
            variations.append(f'inauthor:"{author}" {base_term}')
            
        # Add year-specific variations
        if "year" in result["filters"]:
            year = result["filters"]["year"]
            variations.append(f'{base_term} {year}')
            
        # Sort options common for academic search
        result["sort_options"] = [
            "relevance",
            "date",
            "most cited",
            "recent first"
        ]
        
        # Clean up the processed terms
        result["processed_terms"] = processed
        result["variations"] = variations
        
        return result
    
    # News and content term processing (Task 2.2.5)
    def process_news_content_terms(self, search_terms, options=None):
        """
        Process search terms specifically for news and content searches.
        
        Args:
            search_terms: Original search terms
            options: Optional parameters controlling processing behavior
            
        Returns:
            Dict with processed search terms and metadata
        """
        options = options or {}
        result = {
            "original_terms": search_terms,
            "processed_terms": search_terms,
            "variations": [],
            "content_attributes": {},
            "filters": {},
            "sort_options": []
        }
        
        # Normalize terms
        processed = search_terms.strip()
        
        # Extract date range (for news, recency is often important)
        date_patterns = [
            # Last X days/weeks/months
            (r'last\s+(\d+)\s+(day|days|week|weeks|month|months|year|years)', 'relative'),
            # From date to date
            (r'from\s+(\d{1,2}/\d{1,2}/\d{2,4})\s+to\s+(\d{1,2}/\d{1,2}/\d{2,4})', 'absolute'),
            # Since date
            (r'since\s+(\d{1,2}/\d{1,2}/\d{2,4})', 'since'),
            # Before date
            (r'before\s+(\d{1,2}/\d{1,2}/\d{2,4})', 'before')
        ]
        
        for pattern, date_type in date_patterns:
            date_match = re.search(pattern, processed, re.IGNORECASE)
            if date_match:
                if date_type == 'relative':
                    result["filters"]["date_filter"] = {
                        "type": "relative",
                        "value": date_match.group(1),
                        "unit": date_match.group(2)
                    }
                elif date_type == 'absolute':
                    result["filters"]["date_filter"] = {
                        "type": "range",
                        "start": date_match.group(1),
                        "end": date_match.group(2)
                    }
                elif date_type == 'since':
                    result["filters"]["date_filter"] = {
                        "type": "since",
                        "date": date_match.group(1)
                    }
                elif date_type == 'before':
                    result["filters"]["date_filter"] = {
                        "type": "before",
                        "date": date_match.group(1)
                    }
                
                # Remove date filter from main search term
                processed = re.sub(pattern, '', processed, flags=re.IGNORECASE).strip()
                break
                
        # Extract content type
        content_types = {
            "news": ["news", "article", "story", "report"],
            "blog": ["blog", "post", "opinion"],
            "review": ["review", "rating", "comparison"],
            "analysis": ["analysis", "deep dive", "breakdown"],
            "tutorial": ["tutorial", "how-to", "guide", "walkthrough"],
            "video": ["video", "youtube", "watch"]
        }
        
        for content_type, terms in content_types.items():
            if any(term in processed.lower() for term in terms):
                result["content_attributes"]["content_type"] = content_type
                # Leave in the main search term as it's integral
                break
                
        # Extract source preferences
        source_pattern = re.compile(r'from\s+([A-Za-z0-9\s\.]+)')
        source_match = source_pattern.search(processed)
        if source_match:
            result["content_attributes"]["source"] = source_match.group(1).strip()
            processed = source_pattern.sub('', processed).strip()
            
        # Extract topic categories
        categories = [
            "politics", "business", "technology", "science", "health",
            "sports", "entertainment", "travel", "lifestyle", "education"
        ]
        
        for category in categories:
            if category in processed.lower():
                result["content_attributes"]["category"] = category
                # Remove from search term for cleaner search
                processed = re.sub(r'\b' + re.escape(category) + r'\b', '', processed, flags=re.IGNORECASE).strip()
                break
                
        # Extract content format preferences
        formats = ["article", "video", "podcast", "infographic", "slideshow", "interactive"]
        for fmt in formats:
            if fmt in processed.lower():
                result["content_attributes"]["format"] = fmt
                processed = re.sub(r'\b' + re.escape(fmt) + r'\b', '', processed, flags=re.IGNORECASE).strip()
                break
                
        # Generate content-specific variations
        variations = []
        
        # Base term variations
        base_term = processed.strip()
        if base_term:
            # Add quotes for exact phrase
            if " " in base_term and not base_term.startswith('"') and not base_term.endswith('"'):
                variations.append(f'"{base_term}"')
                
            # Add news-specific variations
            variations.append(f'{base_term} latest')
            variations.append(f'{base_term} breaking news')
            variations.append(f'recent {base_term}')
            
            # Add content-type specific variations
            if "content_type" in result["content_attributes"]:
                content_type = result["content_attributes"]["content_type"]
                variations.append(f'{content_type} about {base_term}')
                variations.append(f'{base_term} {content_type}')
                
            # Add category-specific variations
            if "category" in result["content_attributes"]:
                category = result["content_attributes"]["category"]
                variations.append(f'{base_term} {category}')
                variations.append(f'{category} {base_term}')
                
        # Add analysis variations
        variations.append(f'{base_term} explained')
        variations.append(f'what is {base_term}')
        variations.append(f'{base_term} meaning')
        
        # Sort options common for news/content
        result["sort_options"] = [
            "most recent",
            "relevance",
            "most popular",
            "trending"
        ]
        
        # Clean up the processed terms
        result["processed_terms"] = processed
        result["variations"] = variations
        
        return result
    
    def process_domain_specific_terms(self, search_terms, domain_type, options=None):
        """
        Unified interface for domain-specific term processing.
        
        Args:
            search_terms: The original search terms to process
            domain_type: The type of domain ('e_commerce', 'real_estate', etc.)
            options: Optional parameters controlling processing behavior
            
        Returns:
            Dict with processed search terms and metadata
        """
        # Map domain types to specific processors
        domain_processors = {
            'e_commerce': self.process_ecommerce_search_terms,
            'real_estate': self.process_real_estate_search_terms,
            'job_listings': self.process_job_search_terms,
            'academic': self.process_academic_research_terms,
            'news': self.process_news_content_terms,
        }
        
        # Use the appropriate processor based on domain type
        if domain_type in domain_processors:
            return domain_processors[domain_type](search_terms, options)
        else:
            # For unknown domain types, return a basic processing
            return {
                "original_terms": search_terms,
                "processed_terms": search_terms.strip(),
                "variations": [],
                "filters": {}
            }
    
    async def detect_javascript_dependency(self, url: str, html: str) -> Dict:
        """Enhanced JavaScript dependency detection"""
        js_indicators = {
            'frameworks': {
                'react': ['react', 'reactdom', '__REACT_DEVTOOLS_GLOBAL_HOOK__', 'React.createElement'],
                'vue': ['vue', 'Vue', '__VUE__', 'v-if', 'v-for', 'v-model'],
                'angular': ['angular', 'ng-', 'Angular', '@angular', '[ng'],
                'jquery': ['jquery', '$', 'jQuery'],
                'spa': ['router', 'history.pushState', 'single-page', 'spa']
            },
            'lazy_loading': [
                'lazy', 'intersection-observer', 'loading="lazy"', 
                'lazyload', 'data-src', 'loading=lazy'
            ],
            'dynamic_content': [
                'fetch(', 'axios', 'XMLHttpRequest', 'addEventListener',
                'async ', 'await ', 'Promise', '.then(', 'ajax'
            ],
            'js_heavy_indicators': [
                '<script', 'document.createElement', 'innerHTML',
                'appendChild', 'querySelector', 'getElementById'
            ]
        }
        
        detection_result = {
            'requires_js': False,
            'frameworks': [],
            'features': [],
            'confidence': 0.0,
            'js_heavy': False,
            'estimated_js_percentage': 0.0
        }
        
        if not html:
            return detection_result
            
        # Check HTML content
        html_lower = html.lower()
        total_indicators = 0
        found_indicators = 0
        
        # Count different types of JS indicators
        framework_count = 0
        feature_count = 0
        
        for category, indicators in js_indicators.items():
            if category == 'frameworks':
                for framework, patterns in indicators.items():
                    found_patterns = []
                    for pattern in patterns:
                        if pattern.lower() in html_lower:
                            found_patterns.append(pattern)
                            found_indicators += 1
                            framework_count += 1
                        total_indicators += 1
                    if found_patterns:
                        detection_result['frameworks'].append({
                            'name': framework,
                            'patterns_found': found_patterns,
                            'confidence': len(found_patterns) / len(patterns)
                        })
            else:
                category_features = []
                for indicator in indicators:
                    if indicator.lower() in html_lower:
                        category_features.append(indicator)
                        found_indicators += 1
                        feature_count += 1
                    total_indicators += 1
                if category_features:
                    detection_result['features'].append({
                        'category': category,
                        'indicators': category_features
                    })
        
        # Calculate overall confidence
        detection_result['confidence'] = found_indicators / max(total_indicators, 1)
        
        # Determine if JS is required
        detection_result['requires_js'] = detection_result['confidence'] > 0.3
        
        # Check for JS-heavy content
        script_tags = html_lower.count('<script')
        total_html_length = len(html)
        if total_html_length > 0:
            detection_result['estimated_js_percentage'] = min(
                (script_tags * 100) / max(total_html_length / 1000, 1), 100
            )
        
        detection_result['js_heavy'] = (
            script_tags > 5 or 
            detection_result['estimated_js_percentage'] > 20 or
            framework_count > 2
        )
        
        # Additional heuristics
        if any(framework['name'] in ['react', 'vue', 'angular'] 
               for framework in detection_result['frameworks']):
            detection_result['requires_js'] = True
            detection_result['js_heavy'] = True
        
        logger.info(f"JS detection for {url}: confidence={detection_result['confidence']:.2f}, "
                   f"requires_js={detection_result['requires_js']}, "
                   f"frameworks={[f['name'] for f in detection_result['frameworks']]}")
        
        return detection_result