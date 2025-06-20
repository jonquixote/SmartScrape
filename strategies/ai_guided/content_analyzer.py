"""
Content Analysis Module

Analyzes web page content to extract meaningful information, determine relevance,
and guide the crawler's decision-making process.
"""

import re
import json
import html
from typing import Dict, List, Any, Tuple, Optional, Set
from urllib.parse import urlparse

import google.generativeai as genai
from bs4 import BeautifulSoup

# Import content extraction libraries
from readability import Document
import trafilatura
import justext
from goose3 import Goose

from strategies.ai_guided.ai_cache import AIResponseCache


class ContentAnalyzer:
    """
    Analyzes web page content to extract structured information and determine relevance.
    """
    
    def __init__(self,
                response_cache: AIResponseCache,
                crawl_intent: Dict[str, Any] = None,
                content_memory: Dict[str, Any] = None,
                performance_stats: Dict[str, Any] = None):
        """
        Initialize the content analyzer.
        
        Args:
            response_cache: Cache for AI API responses
            crawl_intent: Dict containing user intent information
            content_memory: Dict of page content summaries keyed by URL
            performance_stats: Dict tracking performance metrics
        """
        self.response_cache = response_cache
        self.crawl_intent = crawl_intent or {}
        self.content_memory = content_memory or {}
        self.performance_stats = performance_stats or {"ai_calls": 0, "cache_hits": 0}
        
        # Schema registry to store detected data schemas
        self.schema_registry = {}
        
    async def analyze_content(self,
                             url: str,
                             html_content: str,
                             page_type_hint: str = None) -> Dict[str, Any]:
        """
        Analyze page content to determine relevance, extract information, and store summary.
        
        Args:
            url: The URL of the page
            html_content: The HTML content of the page
            page_type_hint: Optional hint about the type of page
            
        Returns:
            Dictionary with analysis results including relevance and extracted data
        """
        # Default results
        results = {
            "url": url,
            "relevance": 0.5,
            "content_type": "unknown",
            "extracted_data": {},
            "summary": "",
            "evaluation": ""
        }
        
        try:
            # Skip non-text content
            content_type = self._detect_content_type(html_content)
            if content_type != "text/html":
                results["relevance"] = 0.1
                results["content_type"] = content_type
                results["evaluation"] = "Skipped non-HTML content"
                return results
            
            # Clean and parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "noscript", "iframe"]):
                script.extract()
                
            # Extract text content
            text_content = soup.get_text(separator=' ', strip=True)
            text_content = re.sub(r'\s+', ' ', text_content).strip()
            
            # If there's very little textual content, it's probably not useful
            if len(text_content) < 100:
                results["relevance"] = 0.2
                results["evaluation"] = "Minimal text content"
                return results
            
            # Truncate content for analysis if it's very large
            max_content_length = 8000
            if len(text_content) > max_content_length:
                text_content = text_content[:max_content_length] + "..."
            
            # Detect content type based on page structure and text
            detected_type, type_confidence = self._detect_content_type_by_structure(soup, url, page_type_hint)
            results["content_type"] = detected_type
            
            # Prepare AI analysis based on detected content type and user intent
            if self.crawl_intent:
                extracted_data, relevance, summary, evaluation = await self._extract_with_ai(
                    url, text_content, soup, detected_type
                )
                
                results["extracted_data"] = extracted_data
                results["relevance"] = relevance
                results["summary"] = summary
                results["evaluation"] = evaluation
                
                # Store content in memory for context
                self.content_memory[url] = {
                    "url": url,
                    "content_type": detected_type,
                    "summary": summary,
                    "extracted_data": extracted_data,
                    "relevance": relevance
                }
            else:
                # Fallback to heuristic evaluation if no crawl intent
                relevance = self._evaluate_relevance_heuristic(text_content, soup, url)
                results["relevance"] = relevance
                results["summary"] = text_content[:200] + "..."
                results["evaluation"] = "Evaluated with heuristics (no crawl intent)"
            
            return results
            
        except Exception as e:
            # Log error and return default results
            print(f"Error analyzing content: {str(e)}")
            results["evaluation"] = f"Error during analysis: {str(e)}"
            return results
    
    async def _extract_with_ai(self, 
                              url: str,
                              text_content: str,
                              soup: BeautifulSoup,
                              content_type: str) -> Tuple[Dict[str, Any], float, str, str]:
        """
        Use AI to extract information and evaluate relevance.
        
        Args:
            url: The URL of the page
            text_content: The text content of the page
            soup: BeautifulSoup object of the page
            content_type: The detected content type
            
        Returns:
            Tuple of (extracted_data, relevance, summary, evaluation)
        """
        # Prepare extraction schema based on content type and intent
        extraction_schema = self._get_extraction_schema(content_type)
        
        # Get more specific data based on page structure
        structured_data = self._extract_structured_data(soup, content_type)
        
        # Build context information for AI
        context = {
            "url": url,
            "content_type": content_type,
            "user_intent": self.crawl_intent,
            "structured_data": structured_data
        }
        
        model_name = 'gemini-2.0-flash'
        
        # Create AI prompt for extraction and evaluation
        ai_prompt = f"""
        You are analyzing a web page to extract information and determine relevance.
        
        USER INTENT:
        {json.dumps(self.crawl_intent)}
        
        PAGE INFORMATION:
        URL: {url}
        Content Type: {content_type}
        
        PAGE CONTENT:
        {text_content[:4000]}  # Limit content length for API
        
        EXTRACTION TASK:
        1. Extract information according to this schema:
        {json.dumps(extraction_schema)}
        
        2. Evaluate the relevance of this page to the user's intent on a scale of 0.0 to 1.0.
        
        3. Provide a concise summary of the page (2-3 sentences).
        
        4. Explain why this page is or is not relevant to the user's intent.
        
        Return a JSON object with these fields:
        {{
          "extracted_data": {{
            "field1": "value1",
            "field2": "value2"
          }},
          "relevance": 0.0-1.0,
          "summary": "Concise summary of the page",
          "evaluation": "Explanation of relevance"
        }}
        
        JSON response only:
        """
        
        try:
            # Try to get from cache first
            cached_response = self.response_cache.get(ai_prompt, model_name)
            if cached_response:
                self.performance_stats["cache_hits"] += 1
                result_text = cached_response
            else:
                # Make API call if not in cache
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(ai_prompt)
                result_text = response.text.strip()
                
                # Cache the response
                self.response_cache.set(ai_prompt, model_name, result_text)
                self.performance_stats["ai_calls"] += 1
            
            # Extract JSON part if needed
            json_match = re.search(r'```json\n(.*?)\n```', result_text, re.DOTALL)
            if json_match:
                result_text = json_match.group(1)
            
            # Parse JSON result
            result = json.loads(result_text)
            
            # Extract fields with fallbacks
            extracted_data = result.get("extracted_data", {})
            relevance = float(result.get("relevance", 0.5))
            summary = result.get("summary", "No summary provided")
            evaluation = result.get("evaluation", "No evaluation provided")
            
            # Register any new schema detected
            if extracted_data and content_type not in self.schema_registry:
                schema = self._infer_schema(extracted_data)
                if schema:
                    self.schema_registry[content_type] = schema
            
            return extracted_data, relevance, summary, evaluation
            
        except Exception as e:
            print(f"Error in AI extraction: {str(e)}")
            # Return default values on error
            return {}, 0.5, "Error in content analysis", f"Error: {str(e)}"
    
    def _detect_content_type(self, content: str) -> str:
        """
        Detect the MIME type of the content.
        
        Args:
            content: The raw content
            
        Returns:
            MIME type string
        """
        # Check if content is HTML
        if content.strip().startswith(('<!DOCTYPE', '<html', '<?xml')):
            return "text/html"
            
        # Check if content is JSON
        try:
            json.loads(content)
            return "application/json"
        except:
            pass
            
        # Check if content is XML
        if content.strip().startswith('<?xml') or re.search(r'<\w+[^>]*>.*?</\w+>', content):
            return "application/xml"
            
        # Default to text
        return "text/plain"
    
    def _detect_content_type_by_structure(self, 
                                        soup: BeautifulSoup, 
                                        url: str, 
                                        page_type_hint: str = None) -> Tuple[str, float]:
        """
        Detect content type based on page structure and URL.
        
        Args:
            soup: BeautifulSoup object of the page
            url: The URL of the page
            page_type_hint: Optional hint about the page type
            
        Returns:
            Tuple of (content_type, confidence)
        """
        # Use hint if provided
        if page_type_hint:
            return page_type_hint, 0.7
            
        # Confidence score
        confidence = 0.5
        
        # Check URL for clues
        path = urlparse(url).path.lower()
        
        # Check for product patterns in URL
        if re.search(r'/(product|item|detail)s?/|/p/\d+|product[-_]detail|/(dp|gp)/\w+', path):
            confidence = 0.7
            return "product", confidence
            
        # Check for category patterns in URL
        if re.search(r'/(category|categories|department|collection)s?/|/c/\d+|browse|catalog', path):
            confidence = 0.7
            return "category", confidence
            
        # Check for article patterns in URL
        if re.search(r'/(article|post|blog|news|story)s?/|/\d{4}/\d{2}/', path):
            confidence = 0.7
            return "article", confidence
            
        # Check for search results in URL
        if re.search(r'/search|/find|results|q=|query=', url.lower()):
            confidence = 0.7
            return "search_results", confidence
            
        # Check page structure for clues
        
        # Check for product page
        product_signals = [
            soup.find('div', {'id': lambda x: x and 'product' in x.lower()}),
            soup.find('div', {'class': lambda x: x and 'product' in x.lower()}),
            soup.find(['h1', 'h2'], {'class': lambda x: x and 'product' in x.lower()}),
            soup.find('div', {'itemtype': 'http://schema.org/Product'})
        ]
        
        if any(product_signals):
            confidence = 0.8
            return "product", confidence
            
        # Check for category/listing page
        listing_signals = [
            soup.find('div', {'class': lambda x: x and 'category' in x.lower()}),
            soup.find('div', {'class': lambda x: x and ('product-list' in x.lower() or 'listing' in x.lower())}),
            soup.find('ul', {'class': lambda x: x and ('product-list' in x.lower() or 'listing' in x.lower())}),
            len(soup.find_all('div', {'class': lambda x: x and 'item' in x.lower()})) > 3
        ]
        
        if any(listing_signals):
            confidence = 0.8
            return "listing", confidence
            
        # Check for article page
        article_signals = [
            soup.find('article'),
            soup.find('div', {'class': lambda x: x and 'article' in x.lower()}),
            soup.find('div', {'class': lambda x: x and 'post' in x.lower()}),
            soup.find('div', {'class': lambda x: x and 'blog' in x.lower()})
        ]
        
        if any(article_signals):
            confidence = 0.8
            return "article", confidence
            
        # Check for search results
        search_signals = [
            soup.find('div', {'class': lambda x: x and 'search-results' in x.lower()}),
            soup.find('div', {'id': lambda x: x and 'search' in x.lower()}),
            soup.find('form', {'role': 'search'})
        ]
        
        if any(search_signals):
            confidence = 0.8
            return "search_results", confidence
        
        # Default to generic page
        return "generic", 0.4
    
    def _extract_structured_data(self, soup: BeautifulSoup, content_type: str) -> Dict[str, Any]:
        """
        Extract structured data from the page based on content type.
        
        Args:
            soup: BeautifulSoup object of the page
            content_type: The detected content type
            
        Returns:
            Dictionary of extracted structured data
        """
        data = {}
        
        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            data['title'] = title_tag.get_text().strip()
            
        # Extract meta description
        meta_desc = soup.find('meta', {'name': 'description'})
        if meta_desc and 'content' in meta_desc.attrs:
            data['meta_description'] = meta_desc['content'].strip()
            
        # Extract schema.org structured data
        ld_scripts = soup.find_all('script', {'type': 'application/ld+json'})
        structured_data_list = []
        
        for script in ld_scripts:
            try:
                json_text = script.string
                if json_text:
                    structured_data = json.loads(json_text)
                    structured_data_list.append(structured_data)
            except:
                pass
                
        if structured_data_list:
            data['schema_org'] = structured_data_list
            
        # Content type specific extraction
        if content_type == "product":
            product_data = self._extract_product_data(soup)
            data.update(product_data)
            
        elif content_type == "article":
            article_data = self._extract_article_data(soup)
            data.update(article_data)
            
        elif content_type == "listing":
            listing_data = self._extract_listing_data(soup)
            data.update(listing_data)
            
        return data
    
    def _extract_product_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Extract product-specific data from a product page.
        
        Args:
            soup: BeautifulSoup object of the page
            
        Returns:
            Dictionary of product data
        """
        data = {}
        
        # Extract product name/title
        product_title_candidates = [
            soup.find('h1', {'class': lambda x: x and 'product' in x.lower()}),
            soup.find('h1', {'itemprop': 'name'}),
            soup.find('div', {'class': lambda x: x and 'product-title' in x.lower()})
        ]
        
        for candidate in product_title_candidates:
            if candidate:
                data['product_name'] = candidate.get_text().strip()
                break
                
        # Extract price
        price_candidates = [
            soup.find(['span', 'div'], {'class': lambda x: x and 'price' in x.lower()}),
            soup.find(['span', 'div'], {'itemprop': 'price'}),
            soup.find(['span', 'div'], {'class': lambda x: x and 'amount' in x.lower()})
        ]
        
        for candidate in price_candidates:
            if candidate:
                price_text = candidate.get_text().strip()
                # Try to clean up price text
                price_text = re.sub(r'[^\d.,]', '', price_text)
                data['price'] = price_text
                break
                
        # Extract description
        description_candidates = [
            soup.find(['div', 'p'], {'class': lambda x: x and 'description' in x.lower()}),
            soup.find(['div', 'p'], {'itemprop': 'description'}),
            soup.find('meta', {'property': 'og:description'})
        ]
        
        for candidate in description_candidates:
            if candidate:
                if candidate.name == 'meta':
                    data['description'] = candidate.get('content', '').strip()
                else:
                    data['description'] = candidate.get_text().strip()
                break
        
        # Extract product image
        img_candidates = [
            soup.find('img', {'class': lambda x: x and 'product' in x.lower()}),
            soup.find('img', {'itemprop': 'image'}),
            soup.find('meta', {'property': 'og:image'})
        ]
        
        for candidate in img_candidates:
            if candidate:
                if candidate.name == 'meta':
                    data['image_url'] = candidate.get('content', '')
                else:
                    data['image_url'] = candidate.get('src', '')
                break
        
        return data
    
    def _extract_article_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Extract article-specific data from an article page.
        
        Args:
            soup: BeautifulSoup object of the page
            
        Returns:
            Dictionary of article data
        """
        data = {}
        
        # Extract article title
        title_candidates = [
            soup.find('h1', {'class': lambda x: x and 'article' in x.lower()}),
            soup.find('h1', {'class': lambda x: x and 'post' in x.lower()}),
            soup.find('h1', {'class': lambda x: x and 'entry' in x.lower()})
        ]
        
        for candidate in title_candidates:
            if candidate:
                data['title'] = candidate.get_text().strip()
                break
                
        # Extract author
        author_candidates = [
            soup.find(['span', 'div', 'a'], {'class': lambda x: x and 'author' in x.lower()}),
            soup.find(['span', 'div', 'a'], {'itemprop': 'author'})
        ]
        
        for candidate in author_candidates:
            if candidate:
                data['author'] = candidate.get_text().strip()
                break
                
        # Extract date
        date_candidates = [
            soup.find(['time', 'span', 'div'], {'class': lambda x: x and 'date' in x.lower()}),
            soup.find(['time', 'span', 'div'], {'class': lambda x: x and 'published' in x.lower()}),
            soup.find(['time', 'span', 'div'], {'itemprop': 'datePublished'})
        ]
        
        for candidate in date_candidates:
            if candidate:
                if candidate.has_attr('datetime'):
                    data['date'] = candidate['datetime']
                else:
                    data['date'] = candidate.get_text().strip()
                break
                
        # Extract article content/preview
        content_candidates = [
            soup.find(['div', 'article'], {'class': lambda x: x and 'content' in x.lower()}),
            soup.find(['div', 'article'], {'class': lambda x: x and 'body' in x.lower()}),
            soup.find(['div', 'article'], {'itemprop': 'articleBody'})
        ]
        
        for candidate in content_candidates:
            if candidate:
                # Get first few paragraphs only
                paragraphs = candidate.find_all('p', limit=3)
                content = " ".join([p.get_text().strip() for p in paragraphs])
                data['content_preview'] = content[:500] + "..." if len(content) > 500 else content
                break
        
        return data
    
    def _extract_listing_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Extract data from a listing/category page.
        
        Args:
            soup: BeautifulSoup object of the page
            
        Returns:
            Dictionary of listing data
        """
        data = {}
        
        # Extract category/listing title
        title_candidates = [
            soup.find('h1', {'class': lambda x: x and 'category' in x.lower()}),
            soup.find('h1', {'class': lambda x: x and 'listing' in x.lower()}),
            soup.find('h1', {'class': lambda x: x and 'page-title' in x.lower()})
        ]
        
        for candidate in title_candidates:
            if candidate:
                data['listing_title'] = candidate.get_text().strip()
                break
        
        # Try to extract items in the listing
        items = []
        
        # Common item container patterns
        item_containers = []
        
        # Pattern 1: Product grid items
        grid_items = soup.find_all(['div', 'li'], {'class': lambda x: x and ('product' in x.lower() or 'item' in x.lower())})
        if grid_items:
            item_containers.extend(grid_items[:10])  # Limit to 10 items
            
        # Pattern 2: Search result items  
        search_items = soup.find_all(['div', 'li'], {'class': lambda x: x and 'result' in x.lower()})
        if search_items:
            item_containers.extend(search_items[:10])  # Limit to 10 items
            
        # Extract limited information from items
        for i, container in enumerate(item_containers[:10]):
            item = {}
            
            # Extract item title
            title_elem = container.find(['h2', 'h3', 'h4', 'a'])
            if title_elem:
                item['title'] = title_elem.get_text().strip()
                
            # Extract item link
            link = container.find('a')
            if link and link.has_attr('href'):
                item['link'] = link['href']
                
            # Extract item image
            img = container.find('img')
            if img and img.has_attr('src'):
                item['image'] = img['src']
                
            # Extract price if present
            price_elem = container.find(['span', 'div'], {'class': lambda x: x and 'price' in x.lower()})
            if price_elem:
                item['price'] = price_elem.get_text().strip()
                
            # Only add non-empty items
            if item:
                items.append(item)
                
        if items:
            data['items_count'] = len(items)
            data['items_sample'] = items
            
        # Look for pagination elements
        pagination = soup.find(['div', 'nav', 'ul'], {'class': lambda x: x and 'pag' in x.lower()})
        if pagination:
            data['has_pagination'] = True
            
            # Try to find total number of pages
            page_links = pagination.find_all('a')
            page_numbers = []
            
            for link in page_links:
                text = link.get_text().strip()
                if text.isdigit():
                    page_numbers.append(int(text))
                    
            if page_numbers:
                data['total_pages'] = max(page_numbers)
            
        return data
    
    def _get_extraction_schema(self, content_type: str) -> Dict[str, Any]:
        """
        Get the extraction schema based on content type and crawl intent.
        
        Args:
            content_type: The detected content type
            
        Returns:
            Dictionary describing the extraction schema
        """
        # Check if we have a registered schema for this content type
        if content_type in self.schema_registry:
            return self.schema_registry[content_type]
        
        # Default schemas based on content type
        default_schemas = {
            "product": {
                "product_name": "Product name or title",
                "price": "Product price",
                "currency": "Price currency",
                "description": "Product description",
                "specifications": "Technical specifications or features",
                "brand": "Brand name",
                "availability": "In stock status",
                "rating": "Customer rating",
                "review_count": "Number of reviews"
            },
            "article": {
                "title": "Article title",
                "author": "Author name",
                "date_published": "Publication date",
                "content_summary": "Brief summary of the article content",
                "categories": "Article categories or tags",
                "main_topic": "The main topic or subject of the article"
            },
            "listing": {
                "listing_title": "Title of the listing or category",
                "item_count": "Number of items in the listing",
                "price_range": "Range of prices (min-max)",
                "categories": "Categories or filters available",
                "sorting_options": "Available sorting options"
            },
            "search_results": {
                "query": "The search query used",
                "result_count": "Number of results found",
                "top_results": "Brief descriptions of top results",
                "categories": "Categories or filters available"
            },
            "generic": {
                "page_title": "Title of the page",
                "main_content": "Summary of the main content",
                "key_points": "Key points from the page"
            }
        }
        
        # If we have the content type, return the corresponding schema
        if content_type in default_schemas:
            return default_schemas[content_type]
            
        # Otherwise, return the generic schema
        return default_schemas["generic"]
    
    def _infer_schema(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Infer a schema from extracted data.
        
        Args:
            data: Extracted data dictionary
            
        Returns:
            Inferred schema dictionary
        """
        schema = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                schema[key] = f"String field: {key}"
            elif isinstance(value, (int, float)):
                schema[key] = f"Numeric field: {key}"
            elif isinstance(value, dict):
                schema[key] = f"Object: {list(value.keys())}"
            elif isinstance(value, list):
                if value and isinstance(value[0], dict):
                    schema[key] = f"Array of objects with fields: {list(value[0].keys())}"
                else:
                    schema[key] = f"Array of {type(value[0]).__name__ if value else 'unknown'}"
            else:
                schema[key] = f"Field: {key}"
                
        return schema
    
    def _evaluate_relevance_heuristic(self, text_content: str, soup: BeautifulSoup, url: str) -> float:
        """
        Evaluate page relevance using simple heuristics when AI is not available.
        
        Args:
            text_content: The extracted text content
            soup: BeautifulSoup object of the page
            url: The URL of the page
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        # Default relevance
        relevance = 0.5
        
        # Check if we have crawl intent keywords
        if self.crawl_intent and "keywords" in self.crawl_intent:
            keywords = self.crawl_intent["keywords"]
            
            # Count keyword matches
            match_count = 0
            for keyword in keywords:
                if keyword.lower() in text_content.lower():
                    match_count += 1
                    
            # Calculate relevance based on keyword matches
            if keywords:
                keyword_relevance = min(1.0, match_count / len(keywords))
                relevance = 0.3 + (keyword_relevance * 0.6)  # Scale to 0.3-0.9
        
        # Check for page quality indicators
        
        # Page has a proper title
        title = soup.find('title')
        if title and len(title.get_text().strip()) > 5:
            relevance += 0.05
            
        # Page has meta description
        meta_desc = soup.find('meta', {'name': 'description'})
        if meta_desc and 'content' in meta_desc.attrs:
            relevance += 0.05
            
        # Page has significant content
        if len(text_content) > 1000:
            relevance += 0.05
            
        # Page has proper heading structure
        if soup.find('h1') and soup.find_all(['h2', 'h3']):
            relevance += 0.05
            
        # Ensure relevance is within bounds
        return max(0.1, min(relevance, 1.0))
    
    def extract_clean_content_with_readability(self, html_content: str) -> Dict[str, Any]:
        """
        Extract clean article content using readability-lxml.
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            Dictionary with extracted content and metadata
        """
        try:
            # Create readability document
            doc = Document(html_content)
            
            # Extract title and content
            title = doc.title()
            content = doc.summary()
            
            # Create BeautifulSoup object for additional processing
            content_soup = BeautifulSoup(content, 'html.parser')
            
            # Extract plain text
            text_content = content_soup.get_text(separator=' ', strip=True)
            
            return {
                "title": title,
                "content_html": content,
                "content_text": text_content,
                "extraction_method": "readability-lxml",
                "success": True
            }
        except Exception as e:
            print(f"Error in readability extraction: {str(e)}")
            return {
                "error": str(e),
                "extraction_method": "readability-lxml",
                "success": False
            }
    
    def extract_content_with_trafilatura(self, html_content: str, url: str = None) -> Dict[str, Any]:
        """
        Extract structured content using trafilatura.
        
        Args:
            html_content: Raw HTML content
            url: Optional URL for metadata extraction
            
        Returns:
            Dictionary with extracted content and metadata
        """
        try:
            # Extract main content
            extracted_text = trafilatura.extract(html_content, include_comments=False, 
                                              include_tables=True, no_fallback=False)
            
            # Extract HTML content
            extracted_html = trafilatura.extract(html_content, output_format="html", 
                                              include_comments=False, include_tables=True)
            
            # Extract metadata
            metadata = {}
            if url:
                try:
                    extracted_metadata = trafilatura.metadata.extract_metadata(html_content, url=url)
                    if extracted_metadata:
                        metadata = {
                            "title": extracted_metadata.title,
                            "author": extracted_metadata.author,
                            "date": extracted_metadata.date,
                            "hostname": extracted_metadata.hostname,
                            "categories": extracted_metadata.categories,
                            "tags": extracted_metadata.tags,
                            "sitename": extracted_metadata.sitename
                        }
                except Exception as metadata_err:
                    print(f"Error extracting metadata with trafilatura: {str(metadata_err)}")
            
            return {
                "content_text": extracted_text,
                "content_html": extracted_html,
                "metadata": metadata,
                "extraction_method": "trafilatura",
                "success": bool(extracted_text)
            }
        except Exception as e:
            print(f"Error in trafilatura extraction: {str(e)}")
            return {
                "error": str(e),
                "extraction_method": "trafilatura",
                "success": False
            }
    
    def remove_boilerplate_with_justext(self, html_content: str, language: str = 'English') -> Dict[str, Any]:
        """
        Remove boilerplate content using justext.
        
        Args:
            html_content: Raw HTML content
            language: Language of the content (default: English)
            
        Returns:
            Dictionary with cleaned content
        """
        try:
            # Get available stoplist
            try:
                stoplist = getattr(justext, language.lower())
            except AttributeError:
                stoplist = justext.get_stoplist("english")  # Fallback to English
            
            # Extract paragraphs
            paragraphs = justext.justext(html_content, stoplist)
            
            # Separate good (content) from bad (boilerplate) paragraphs
            good_paragraphs = [p.text for p in paragraphs if not p.is_boilerplate]
            bad_paragraphs = [p.text for p in paragraphs if p.is_boilerplate]
            
            # Join good paragraphs into clean text
            clean_text = "\n\n".join(good_paragraphs)
            
            # Create simple HTML from good paragraphs
            clean_html = "".join([f"<p>{p}</p>" for p in good_paragraphs])
            
            return {
                "content_text": clean_text,
                "content_html": clean_html,
                "boilerplate_removed": len(bad_paragraphs),
                "paragraphs_kept": len(good_paragraphs),
                "extraction_method": "justext",
                "success": bool(clean_text)
            }
        except Exception as e:
            print(f"Error in justext extraction: {str(e)}")
            return {
                "error": str(e),
                "extraction_method": "justext",
                "success": False
            }
    
    def extract_article_with_goose(self, html_content: str, url: str = None) -> Dict[str, Any]:
        """
        Extract article content using goose3.
        
        Args:
            html_content: Raw HTML content
            url: Optional URL for additional context
            
        Returns:
            Dictionary with extracted article content and metadata
        """
        try:
            # Initialize goose
            g = Goose()
            
            # Extract article
            if url:
                article = g.extract(url=url)
            else:
                article = g.extract(raw_html=html_content)
            
            # Get metadata
            metadata = {
                "title": article.title,
                "meta_description": article.meta_description,
                "meta_keywords": article.meta_keywords,
                "tags": article.tags,
                "publish_date": article.publish_date.isoformat() if article.publish_date else None,
                "authors": article.authors,
                "top_image": article.top_image.src if article.top_image else None
            }
            
            # Clean up after extraction
            g.close()
            
            return {
                "content_text": article.cleaned_text,
                "content_html": article.cleaned_article_html,
                "metadata": metadata,
                "extraction_method": "goose3",
                "success": bool(article.cleaned_text)
            }
        except Exception as e:
            print(f"Error in goose extraction: {str(e)}")
            return {
                "error": str(e),
                "extraction_method": "goose3",
                "success": False
            }
    
    def extract_content_with_all_methods(self, url: str, html_content: str) -> Dict[str, Any]:
        """
        Extract content using all available methods and select the best result.
        
        Args:
            url: URL of the page
            html_content: Raw HTML content
            
        Returns:
            Dictionary with extracted content from all methods and the best result
        """
        results = {}
        
        # Extract with readability
        readability_result = self.extract_clean_content_with_readability(html_content)
        results["readability"] = readability_result
        
        # Extract with trafilatura
        trafilatura_result = self.extract_content_with_trafilatura(html_content, url)
        results["trafilatura"] = trafilatura_result
        
        # Extract with justext
        justext_result = self.remove_boilerplate_with_justext(html_content)
        results["justext"] = justext_result
        
        # Extract with goose
        goose_result = self.extract_article_with_goose(html_content, url)
        results["goose"] = goose_result
        
        # Determine best extraction based on content length and success
        best_method = self._select_best_extraction(results)
        
        return {
            "all_results": results,
            "best_method": best_method,
            "best_result": results.get(best_method, {}),
            "original_html": html_content
        }
    
    def _select_best_extraction(self, extraction_results: Dict[str, Any]) -> str:
        """
        Select the best extraction method based on content quality.
        
        Args:
            extraction_results: Dictionary of results from various extraction methods
            
        Returns:
            Name of the best extraction method
        """
        # Default method if all fail
        best_method = "readability"
        best_score = 0
        
        for method, result in extraction_results.items():
            # Skip failed extractions
            if not result.get("success", False):
                continue
                
            # Calculate score based on content length and structure
            score = 0
            
            # Length of content is a good indicator of extraction quality
            content_text = result.get("content_text", "")
            if content_text:
                # Longer content gets higher score (up to a point)
                length_score = min(len(content_text) / 2000, 5)
                score += length_score
                
                # More paragraphs usually indicates better structure
                paragraph_count = content_text.count('\n\n') + 1
                paragraph_score = min(paragraph_count / 5, 3)
                score += paragraph_score
            
            # Metadata presence is good
            if result.get("metadata", {}):
                score += 1
                
            # HTML content with proper structure is valuable
            content_html = result.get("content_html", "")
            if content_html:
                if '<p>' in content_html:
                    score += 1
                if '<h' in content_html:
                    score += 1
            
            # If this method has the best score so far, select it
            if score > best_score:
                best_score = score
                best_method = method
        
        return best_method
        
    async def analyze_content_enhanced(self, url: str, html_content: str, page_type_hint: str = None) -> Dict[str, Any]:
        """
        Enhanced version of analyze_content that uses multiple extraction methods.
        
        Args:
            url: The URL of the page
            html_content: The HTML content of the page
            page_type_hint: Optional hint about the type of page
            
        Returns:
            Dictionary with analysis results including extracted content
        """
        # First, extract content using all methods
        extraction_results = self.extract_content_with_all_methods(url, html_content)
        
        # Use best extraction for further analysis
        best_result = extraction_results["best_result"]
        best_method = extraction_results["best_method"]
        
        # Parse the page for structure analysis
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Detect content type
        detected_type, type_confidence = self._detect_content_type_by_structure(soup, url, page_type_hint)
        
        # Get extracted text content from the best method
        text_content = best_result.get("content_text", "")
        
        # If we have crawl intent, use AI to analyze
        if self.crawl_intent:
            extracted_data, relevance, summary, evaluation = await self._extract_with_ai(
                url, text_content, soup, detected_type
            )
            
            # Store content in memory for context
            self.content_memory[url] = {
                "url": url,
                "content_type": detected_type,
                "summary": summary,
                "extracted_data": extracted_data,
                "relevance": relevance
            }
            
            # Return combined results
            return {
                "url": url,
                "relevance": relevance,
                "content_type": detected_type,
                "extracted_data": extracted_data,
                "summary": summary,
                "evaluation": evaluation,
                "extraction": {
                    "method": best_method,
                    "content_html": best_result.get("content_html", ""),
                    "content_text": text_content,
                    "metadata": best_result.get("metadata", {})
                },
                "original_html": html_content,
                "extraction_results": extraction_results["all_results"]
            }
        else:
            # Fallback to heuristic evaluation if no crawl intent
            relevance = self._evaluate_relevance_heuristic(text_content, soup, url)
            
            return {
                "url": url,
                "relevance": relevance,
                "content_type": detected_type,
                "extraction": {
                    "method": best_method,
                    "content_html": best_result.get("content_html", ""),
                    "content_text": text_content,
                    "metadata": best_result.get("metadata", {})
                },
                "summary": text_content[:200] + "..." if len(text_content) > 200 else text_content,
                "evaluation": f"Evaluated with heuristics (no crawl intent) using {best_method}",
                "original_html": html_content
            }
