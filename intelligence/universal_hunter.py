"""
Universal Intelligent Hunter System - Refactored Open-Source Architecture

This module implements a universal, AI-powered content extraction engine using
a combination of open-source tools to ensure reliability without paid services.

The architecture is as follows:
1.  **Intent Analysis:** Understands the user's query and defines a structured output format (JSON schema).
2.  **URL Discovery:** Uses DuckDuckGo to find relevant URLs.
3.  **Multi-Stage Extraction Pipeline:**
    - Fetches content using httpx and Playwright for JavaScript rendering.
    - Cleans the HTML and extracts the main content using Trafilatura/Readability/Justext.
    - Performs Heuristic/Rule-Based Structured Extraction (BeautifulSoup/LXML).
    - Conditionally uses LLM-Guided Refinement/Extraction as a fallback.
"""

import logging
import asyncio
import json
import re
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field

from duckduckgo_search import DDGS
from trafilatura import fetch_url, extract
from playwright.async_api import async_playwright
import httpx
from bs4 import BeautifulSoup # For heuristic extraction
from jsonschema import validate, ValidationError, Draft7Validator

# Assume crawl4ai is installed and configured
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, LLMConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("UniversalHunter")

@dataclass
class HuntingIntent:
    """Structured representation of what the user is hunting for, including the desired output format."""
    query: str
    output_schema: Dict[str, Any]
    target_type: str = "information"
    content_category: str = "general"
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)

@dataclass
class HuntingResult:
    """A structured result from a hunting operation."""
    url: str
    data: Dict[str, Any]
    relevance_score: float

class UniversalHunter:
    """
    The orchestration engine that coordinates the hunting process.
    It uses a combination of open-source tools for discovery and extraction.
    """

    def __init__(self, ai_service_client, concurrency_limit: int = 5, extraction_timeout: int = 30):
        self.logger = logging.getLogger(__name__)
        self.ai_service_client = ai_service_client
        # The AsyncWebCrawler from crawl4ai will use the provided LLM client
        # We will initialize it later with a proper LLMConfig
        self.crawler = None
        # Concurrency and resource management
        self.concurrency_limit = concurrency_limit
        self.extraction_timeout = extraction_timeout
        self.semaphore = asyncio.Semaphore(concurrency_limit)

    async def hunt(self, intent: HuntingIntent, max_targets: int = 5) -> List[HuntingResult]:
        """
        Main hunting interface. It discovers URLs and extracts content based on the user's intent.
        Enhanced with concurrency control and timeout management.
        """
        self.logger.info(f"🎯 Starting hunt for: '{intent.query}'")

        # 1. Discover URLs using DuckDuckGo
        discovered_urls = self._discover_urls(intent.query, max_targets)
        if not discovered_urls:
            self.logger.warning("No URLs discovered.")
            return []

        # 2. Extract content from each URL with concurrency control
        async def limited_extract(url: str) -> Dict[str, Any]:
            async with self.semaphore:
                try:
                    return await asyncio.wait_for(
                        self._extract_content(url, intent), 
                        timeout=self.extraction_timeout
                    )
                except asyncio.TimeoutError:
                    self.logger.warning(f"⏰ Extraction timed out for: {url}")
                    return {
                        "success": False,
                        "error": "timeout",
                        "url": url,
                        "message": f"Extraction timed out after {self.extraction_timeout} seconds"
                    }
                except Exception as e:
                    self.logger.error(f"❌ Extraction failed for {url}: {e}")
                    return {
                        "success": False,
                        "error": "extraction_failed",
                        "url": url,
                        "message": str(e)
                    }

        extraction_tasks = [limited_extract(url) for url in discovered_urls]
        extracted_data = await asyncio.gather(*extraction_tasks, return_exceptions=True)

        # 3. Consolidate and rank the results
        results = self._consolidate_results(discovered_urls, extracted_data, intent)
        
        # 4. Deduplicate results
        deduplicated_results = self._deduplicate_results(results)
        
        # 5. Rank final results
        ranked_results = self._rank_results(deduplicated_results)

        self.logger.info(f"✅ Hunt complete. Found {len(ranked_results)} results.")
        return ranked_results[:max_targets]

    def _discover_urls(self, query: str, limit: int) -> List[str]:
        """Discover relevant URLs using DuckDuckGo Search."""
        self.logger.info(f"Discovering URLs for query: '{query}'")
        try:
            with DDGS() as ddgs:
                results = [r for r in ddgs.text(query, max_results=limit)]
                return [r['href'] for r in results]
        except Exception as e:
            self.logger.error(f"URL discovery with DuckDuckGo failed: {e}")
            return []

    async def _fetch_and_clean_html(self, url: str) -> Optional[Dict[str, str]]:
        """
        Fetches a URL and returns both raw HTML and extracted main article text.
        Enhanced with better error handling and content validation.
        """
        self.logger.info(f"Fetching raw HTML and extracting main content: {url}")
        
        # Try with httpx first
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
                response = await client.get(
                    url, 
                    follow_redirects=True,
                    headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                )
                response.raise_for_status()
                raw_html = response.text
                
                # Use Trafilatura to extract the main article content
                article_text = extract(
                    raw_html,
                    favor_precision=True,       # try to return the most content
                    include_comments=False,
                    include_tables=False
                )
                
                # Validate that we got meaningful content
                if article_text and len(article_text.strip()) > 50:
                    self.logger.info(f"Successfully extracted {len(article_text)} characters from {url}")
                    return {
                        "raw_html": raw_html,
                        "article_text": article_text
                    }
                else:
                    # Try without trafilatura cleaning if extraction is too short
                    self.logger.warning(f"Trafilatura extraction too short ({len(article_text or '')} chars), trying raw HTML")
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(raw_html, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    # Get text content
                    text = soup.get_text()
                    
                    # Clean up whitespace
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = ' '.join(chunk for chunk in chunks if chunk)
                    
                    if text and len(text.strip()) > 50:
                        self.logger.info(f"Successfully extracted {len(text)} characters using BeautifulSoup from {url}")
                        return {
                            "raw_html": raw_html,
                            "article_text": text
                        }
                    else:
                        self.logger.warning(f"No meaningful content extracted from {url}")
                        return None
                        
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            self.logger.warning(f"httpx fetch failed for {url}: {e}. Falling back to Playwright.")

        # Fallback to Playwright if httpx fails
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(url, wait_until='networkidle', timeout=30000)
                raw_html = await page.content()
                await browser.close()
                
                # Use Trafilatura to extract the main article content
                article_text = extract(
                    raw_html,
                    favor_precision=True,
                    include_comments=False,
                    include_tables=False
                )
                
                if article_text and len(article_text.strip()) > 50:
                    self.logger.info(f"Successfully extracted {len(article_text)} characters via Playwright from {url}")
                    return {
                        "raw_html": raw_html,
                        "article_text": article_text
                    }
                    
        except Exception as e:
            self.logger.error(f"Playwright fetch failed for {url}: {e}")
            return None

    async def _extract_structured_data_non_llm(self, raw_html: str, article_text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts structured data from raw HTML and article text using BeautifulSoup and heuristic rules.
        Uses raw HTML for metadata extraction and article text for content.
        """
        self.logger.info("Attempting non-LLM structured data extraction.")
        extracted_data = {}
        soup = BeautifulSoup(raw_html, 'lxml')

        properties = schema.get("properties", {})
        
        # Heuristics for common fields
        for field_name, field_props in properties.items():
            if field_name == "title":
                # Enhanced title extraction using multiple strategies
                title = self._extract_title_heuristics(soup)
                if title:
                    extracted_data[field_name] = title
            
            elif field_name == "url":
                # Canonical URL or current URL (if available from metadata)
                canonical_link = soup.find('link', {'rel': 'canonical'})
                if canonical_link and canonical_link.get('href'):
                    extracted_data[field_name] = canonical_link['href']
                # Note: Original URL would need to be passed from the calling method

            elif field_name == "summary" or field_name == "description":
                # Try meta description first
                meta_desc = soup.find('meta', {'name': 'description'})
                if meta_desc and meta_desc.get('content'):
                    extracted_data[field_name] = meta_desc['content'].strip()
                else:
                    # Fallback to first part of article text
                    if article_text:
                        # Take first 2-3 sentences or up to 500 characters
                        sentences = article_text.split('. ')
                        summary = '. '.join(sentences[:3])
                        if len(summary) > 500:
                            summary = summary[:500] + "..."
                        extracted_data[field_name] = summary
            
            elif field_name == "author":
                # Try common author meta tags or bylines
                author_meta = soup.find('meta', {'name': 'author'})
                if author_meta and author_meta.get('content'):
                    extracted_data[field_name] = author_meta['content'].strip()
                else:
                    # Try other common meta tags
                    author_meta = soup.find('meta', {'property': 'article:author'}) or \
                                soup.find('meta', {'name': 'article:author'})
                    if author_meta and author_meta.get('content'):
                        extracted_data[field_name] = author_meta['content'].strip()
                    else:
                        # Try common CSS selectors for author
                        author_selectors = [
                            '[data-author]', '.author', '.byline', '.by-author',
                            '[rel="author"]', '.article-author', '.post-author'
                        ]
                        for selector in author_selectors:
                            author_elem = soup.select_one(selector)
                            if author_elem:
                                author_text = author_elem.get_text(strip=True)
                                if author_text and len(author_text) < 100:  # Reasonable author name length
                                    extracted_data[field_name] = author_text
                                    break

            elif field_name == "publication_date" or field_name == "date_published":
                # Try common date meta tags or time tags
                date_meta = soup.find('meta', {'property': 'article:published_time'}) or \
                           soup.find('meta', {'name': 'date'}) or \
                           soup.find('meta', {'name': 'publish_date'}) or \
                           soup.find('meta', {'property': 'article:published'})
                           
                if date_meta and date_meta.get('content'):
                    extracted_data[field_name] = date_meta['content'].strip()
                else:
                    # Try time tags
                    time_tag = soup.find('time')
                    if time_tag and time_tag.get('datetime'):
                        extracted_data[field_name] = time_tag['datetime'].strip()
                    elif time_tag and time_tag.get_text(strip=True):
                        extracted_data[field_name] = time_tag.get_text(strip=True)

            elif field_name in ["full_content", "article_content", "content"]:
                # Use the main article content extracted by Trafilatura
                if article_text:
                    extracted_data[field_name] = article_text

            elif field_name == "image_urls":
                # Extract image URLs from within the main article content area
                images = self._extract_article_images(soup, article_text)
                if images:
                    extracted_data[field_name] = images
            
            # Generic CSS selector extraction if 'selector' is provided in schema property
            if "selector" in field_props:
                elements = soup.select(field_props["selector"])
                if elements:
                    if field_props.get("type") == "text":
                        extracted_data[field_name] = elements[0].get_text(strip=True)
                    elif field_props.get("type") == "attribute" and "attribute" in field_props:
                        extracted_data[field_name] = elements[0].get(field_props["attribute"])
                    elif field_props.get("type") == "list":
                        extracted_data[field_name] = [el.get_text(strip=True) for el in elements]
                    # Add more types as needed (e.g., nested objects)

        return {k: v for k, v in extracted_data.items() if v is not None and v != ''}

    def _extract_title_heuristics(self, soup: BeautifulSoup) -> Optional[str]:
        """Enhanced title extraction using multiple strategies."""
        # Strategy 1: <title> tag
        title_tag = soup.find('title')
        if title_tag and title_tag.string:
            title = title_tag.string.strip()
            # Clean up common title patterns
            title = re.sub(r'\s*\|\s*.*$', '', title)  # Remove "| Site Name" 
            title = re.sub(r'\s*-\s*.*$', '', title)   # Remove "- Site Name"
            if title:
                return title

        # Strategy 2: Open Graph title
        og_title = soup.find('meta', {'property': 'og:title'})
        if og_title and og_title.get('content'):
            return og_title['content'].strip()

        # Strategy 3: Twitter title
        twitter_title = soup.find('meta', {'name': 'twitter:title'})
        if twitter_title and twitter_title.get('content'):
            return twitter_title['content'].strip()

        # Strategy 4: First h1 tag
        h1_tag = soup.find('h1')
        if h1_tag and h1_tag.get_text(strip=True):
            return h1_tag.get_text(strip=True)

        # Strategy 5: Article title selector
        article_title_selectors = [
            'article h1', '.article-title', '.post-title', 
            '.entry-title', '[data-title]', '.title'
        ]
        for selector in article_title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem:
                title_text = title_elem.get_text(strip=True)
                if title_text:
                    return title_text

        return None

    def _extract_article_images(self, soup: BeautifulSoup, article_text: str) -> List[str]:
        """
        Extract image URLs from within the main article content area, 
        filtering out ads, logos, and placeholder images.
        """
        images = []
        
        # First, try to find images within article tags
        article_containers = soup.find_all(['article', 'main', '.content', '.post-content', '.article-content'])
        if not article_containers:
            # Fallback to any container that might hold the main content
            article_containers = [soup]
        
        for container in article_containers:
            for img in container.find_all('img'):
                src = img.get('src') or img.get('data-src') or img.get('data-original')
                if src and self._is_valid_article_image(src, img):
                    # Convert relative URLs to absolute if needed
                    if src.startswith('http'):
                        images.append(src)
                    elif src.startswith('//'):
                        images.append('https:' + src)
                    # Skip relative URLs for now as we don't have the base URL
                    
        # Remove duplicates while preserving order
        seen = set()
        unique_images = []
        for img in images:
            if img not in seen:
                seen.add(img)
                unique_images.append(img)
                
        return unique_images[:10]  # Limit to first 10 images

    def _is_valid_article_image(self, src: str, img_tag) -> bool:
        """
        Check if an image URL is likely to be a valid article image
        (not an ad, logo, or placeholder).
        """
        # Filter out obvious non-article images
        src_lower = src.lower()
        
        # Skip placeholder and example images
        if any(domain in src_lower for domain in [
            'example.com', 'placeholder', 'lorem', 'dummy', 'fake',
            'test.jpg', 'sample.', 'default.', 'blank.'
        ]):
            return False
            
        # Skip very small images (likely icons/logos)
        width = img_tag.get('width')
        height = img_tag.get('height')
        if width and height:
            try:
                w, h = int(width), int(height)
                if w < 100 or h < 100:  # Skip very small images
                    return False
            except (ValueError, TypeError):
                pass
                
        # Skip images with ad-related classes or attributes
        classes = img_tag.get('class', [])
        if isinstance(classes, str):
            classes = [classes]
        
        ad_indicators = ['ad', 'advertisement', 'sponsor', 'banner', 'logo', 'icon']
        if any(indicator in ' '.join(classes).lower() for indicator in ad_indicators):
            return False
            
        # Skip images with ad-related alt text
        alt_text = img_tag.get('alt', '').lower()
        if any(indicator in alt_text for indicator in ad_indicators):
            return False
            
        return True

    def _validate_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate extracted data against JSON schema and auto-fix common issues.
        Returns validation result with fixed data and error details.
        """
        validation_result = {
            "is_valid": False,
            "data": data.copy(),
            "errors": [],
            "fixes_applied": []
        }
        
        try:
            # First, try basic validation
            validate(instance=data, schema=schema)
            validation_result["is_valid"] = True
            return validation_result
            
        except ValidationError as e:
            validation_result["errors"].append({
                "path": list(e.path),
                "message": e.message,
                "field": e.path[-1] if e.path else "root"
            })
            
            # Attempt auto-fixes for common issues
            fixed_data = data.copy()
            
            # Auto-fix type mismatches
            if "properties" in schema:
                for field, field_schema in schema["properties"].items():
                    if field in fixed_data:
                        expected_type = field_schema.get("type")
                        current_value = fixed_data[field]
                        
                        # Fix string to number conversions
                        if expected_type == "number" and isinstance(current_value, str):
                            try:
                                fixed_data[field] = float(current_value.replace(',', ''))
                                validation_result["fixes_applied"].append(f"Converted {field} from string to number")
                            except ValueError:
                                pass
                                
                        # Fix string to integer conversions  
                        elif expected_type == "integer" and isinstance(current_value, str):
                            try:
                                fixed_data[field] = int(float(current_value.replace(',', '')))
                                validation_result["fixes_applied"].append(f"Converted {field} from string to integer")
                            except ValueError:
                                pass
                                
                        # Fix array to string or vice versa
                        elif expected_type == "array" and not isinstance(current_value, list):
                            if isinstance(current_value, str):
                                # Split string into array if it contains separators
                                if ',' in current_value:
                                    fixed_data[field] = [item.strip() for item in current_value.split(',')]
                                    validation_result["fixes_applied"].append(f"Split {field} string into array")
                                else:
                                    fixed_data[field] = [current_value]
                                    validation_result["fixes_applied"].append(f"Wrapped {field} string in array")
                                    
                        elif expected_type == "string" and isinstance(current_value, list):
                            fixed_data[field] = ', '.join(str(item) for item in current_value)
                            validation_result["fixes_applied"].append(f"Joined {field} array into string")
                            
                        # Fix boolean conversions
                        elif expected_type == "boolean" and isinstance(current_value, str):
                            if current_value.lower() in ["true", "1", "yes", "on"]:
                                fixed_data[field] = True
                                validation_result["fixes_applied"].append(f"Converted {field} string to boolean")
                            elif current_value.lower() in ["false", "0", "no", "off"]:
                                fixed_data[field] = False
                                validation_result["fixes_applied"].append(f"Converted {field} string to boolean")
            
            # Try validation again with fixed data
            try:
                validate(instance=fixed_data, schema=schema)
                validation_result["is_valid"] = True
                validation_result["data"] = fixed_data
            except ValidationError as retry_error:
                validation_result["errors"].append({
                    "path": list(retry_error.path),
                    "message": retry_error.message,
                    "field": retry_error.path[-1] if retry_error.path else "root",
                    "after_fixes": True
                })
                
        return validation_result

    async def _extract_content(self, url: str, intent: HuntingIntent) -> Dict[str, Any]:
        """
        Extracts structured content from a URL using a multi-stage process:
        1. Fetch and clean HTML.
        2. Attempt non-LLM heuristic extraction.
        3. If non-LLM is insufficient, use LLM for refinement/extraction.
        
        Returns structured result with success/error information and partial data.
        """
        import time
        start_time = time.time()
        extraction_metadata = {
            "extraction_time": 0,
            "method_used": None,
            "stages_attempted": []
        }
        
        self.logger.info(f"Extracting content from: {url}")
        
        try:
            # Stage 1: Fetch and clean HTML
            extraction_metadata["stages_attempted"].append("fetch_html")
            content_data = await self._fetch_and_clean_html(url)
            if not content_data:
                self.logger.warning(f"Could not retrieve main content from {url}")
                return {
                    "success": False,
                    "url": url,
                    "error": "fetch_failed",
                    "message": "Could not retrieve main content",
                    "data": {},
                    "partial_data": {},
                    "errors": [{"type": "network", "message": "Failed to fetch or clean HTML", "stage": "fetch_html"}],
                    "metadata": extraction_metadata
                }

            raw_html = content_data["raw_html"]
            article_text = content_data["article_text"]

            # Stage 2: Attempt non-LLM heuristic extraction
            extraction_metadata["stages_attempted"].append("heuristic_extraction")
            non_llm_extracted_data = await self._extract_structured_data_non_llm(raw_html, article_text, intent.output_schema)
            
            # Check if non-LLM extraction is sufficient (e.g., all required fields are present)
            required_fields = intent.output_schema.get("required", [])
            missing_fields = [field for field in required_fields if field not in non_llm_extracted_data or not non_llm_extracted_data[field]]
            
            if not missing_fields and non_llm_extracted_data:
                # Validate schema before returning
                validation_result = self._validate_schema(non_llm_extracted_data, intent.output_schema)
                extraction_metadata["method_used"] = "heuristic"
                extraction_metadata["extraction_time"] = time.time() - start_time
                extraction_metadata["validation"] = {
                    "is_valid": validation_result["is_valid"],
                    "fixes_applied": validation_result["fixes_applied"]
                }
                
                self.logger.info(f"Non-LLM extraction sufficient for {url}. Schema valid: {validation_result['is_valid']}")
                
                return {
                    "success": True,
                    "url": url,
                    "data": validation_result["data"],
                    "partial_data": {},
                    "errors": validation_result["errors"],
                    "metadata": extraction_metadata
                }
            else:
                self.logger.info(f"Non-LLM extraction insufficient for {url}. Missing fields: {missing_fields}. Falling back to LLM.")
                
                # Stage 3: Fallback to LLM-guided extraction for refinement or full extraction
                extraction_metadata["stages_attempted"].append("llm_extraction")
                try:
                    instruction = f"""
                    From the provided webpage content, extract the following information into a JSON object, strictly adhering to the provided JSON schema.
                    - 'title': The main title of the article or page.
                    - 'url': The canonical URL of the article or page.
                    - 'summary': A concise summary of the article's main points. If no explicit summary is available, generate a brief summary based on the main content.
                    - 'author': The author of the content.
                    - 'publication_date': The date the content was published.
                    - 'full_content': The complete main textual content of the page.
                    - 'image_urls': A list of relevant image URLs from the content.

                    Desired JSON Schema: {json.dumps(intent.output_schema)}

                    Prioritize filling all fields, especially required ones. If some data was already extracted by non-LLM methods,
                    use that as a starting point and fill in any missing or refine existing fields.
                    """
                    
                    # Get the default model configuration from the AI service client
                    default_model_name = self.ai_service_client.default_model_name
                    default_model_config = next(
                        (m for m in self.ai_service_client._config.get("models", []) if m["name"] == default_model_name),
                        None
                    )

                    if not default_model_config:
                        raise ValueError("Default AI model configuration not found.")

                    # Create LLMConfig from the AI service's model configuration
                    provider_string = f"{default_model_config['type']}/{default_model_config['model_id']}"
                    
                    # Ensure API key is passed if available
                    api_key_for_llm = default_model_config.get("api_key")
                    if not api_key_for_llm:
                        self.logger.warning(f"No API key found for model {default_model_name}. LLM extraction might fail.")

                    llm_config = LLMConfig(
                        provider=default_model_config['type'], # Use just the provider type here
                        api_token=api_key_for_llm # Pass the API key
                    )

                    # Initialize the crawler here with the correct LLMConfig
                    if self.crawler is None:
                        self.crawler = AsyncWebCrawler(llm_config=llm_config)

                    strategy = LLMExtractionStrategy(
                        llm_config=llm_config,
                        model=provider_string, # Pass the full model string including provider
                        schema=intent.output_schema,
                        extraction_type="schema",
                        instruction=instruction, # Use the more specific instruction
                        extra_args={"api_token": api_key_for_llm} # Pass API token as extra_args
                    )
                    config = CrawlerRunConfig(extraction_strategy=strategy)
                    
                    # Pass the original URL directly to arun, let crawl4ai handle fetching and processing
                    result = await self.crawler.arun(url=url, config=config)

                    # Parse the JSON string content into a dictionary
                    if result.extracted_content:
                        try:
                            llm_extracted_data = json.loads(result.extracted_content)
                            
                            # Validate schema for LLM data
                            validation_result = self._validate_schema(llm_extracted_data, intent.output_schema)
                            extraction_metadata["method_used"] = "llm"
                            extraction_metadata["extraction_time"] = time.time() - start_time
                            extraction_metadata["validation"] = {
                                "is_valid": validation_result["is_valid"],
                                "fixes_applied": validation_result["fixes_applied"]
                            }
                            
                            return {
                                "success": True,
                                "url": url,
                                "data": validation_result["data"],
                                "partial_data": non_llm_extracted_data,
                                "errors": validation_result["errors"],
                                "metadata": extraction_metadata
                            }
                        except json.JSONDecodeError as e:
                            self.logger.error(f"Failed to decode JSON from extracted content for {url}: {e}. Content: {result.extracted_content[:500]}...")
                            extraction_metadata["extraction_time"] = time.time() - start_time
                            return {
                                "success": False,
                                "url": url,
                                "error": "json_decode_failed",
                                "message": f"Failed to decode LLM JSON response: {str(e)}",
                                "data": {},
                                "partial_data": non_llm_extracted_data,
                                "errors": [{"type": "parsing", "message": str(e), "stage": "llm_extraction"}],
                                "metadata": extraction_metadata
                            }
                    else:
                        extraction_metadata["extraction_time"] = time.time() - start_time
                        return {
                            "success": False,
                            "url": url,
                            "error": "llm_no_content",
                            "message": "LLM extraction returned no content",
                            "data": {},
                            "partial_data": non_llm_extracted_data,
                            "errors": [{"type": "extraction", "message": "LLM returned no content", "stage": "llm_extraction"}],
                            "metadata": extraction_metadata
                        }
                        
                except Exception as llm_error:
                    self.logger.error(f"LLM extraction failed for {url}: {llm_error}")
                    extraction_metadata["extraction_time"] = time.time() - start_time
                    return {
                        "success": False,
                        "url": url,
                        "error": "llm_extraction_failed",
                        "message": str(llm_error),
                        "data": {},
                        "partial_data": non_llm_extracted_data,
                        "errors": [{"type": "llm", "message": str(llm_error), "stage": "llm_extraction"}],
                        "metadata": extraction_metadata
                    }

        except Exception as e:
            extraction_metadata["extraction_time"] = time.time() - start_time
            self.logger.error(f"Content extraction failed for {url}: {e}")
            return {
                "success": False,
                "url": url,
                "error": "extraction_failed",
                "message": str(e),
                "data": {},
                "partial_data": {},
                "errors": [{"type": "general", "message": str(e), "stage": "unknown"}],
                "metadata": extraction_metadata
            }


    def _consolidate_results(self, urls: List[str], extractions: List[Dict[str, Any]], intent: HuntingIntent) -> List[HuntingResult]:
        """
        Consolidate the extracted data into a list of HuntingResult objects.
        Now handles both successful and failed extractions with structured error information.
        """
        results = []
        for url, extraction_result in zip(urls, extractions):
            if isinstance(extraction_result, Exception):
                # Handle exceptions that weren't caught by the timeout wrapper
                self.logger.error(f"Unhandled exception for {url}: {extraction_result}")
                continue
                
            if extraction_result and extraction_result.get("success", False):
                # Successful extraction
                data = extraction_result.get("data", {})
                if data:  # Only include if we have actual data
                    relevance = self._calculate_relevance(data, intent)
                    results.append(HuntingResult(
                        url=url, 
                        data=data, 
                        relevance_score=relevance
                    ))
            elif extraction_result and extraction_result.get("partial_data"):
                # Failed extraction but has partial data - include with lower relevance
                partial_data = extraction_result.get("partial_data", {})
                if partial_data:
                    relevance = self._calculate_relevance(partial_data, intent) * 0.5  # Reduce relevance for partial data
                    # Add error information to the data
                    enhanced_data = {
                        **partial_data,
                        "_extraction_status": "partial",
                        "_errors": extraction_result.get("errors", []),
                        "_metadata": extraction_result.get("metadata", {})
                    }
                    results.append(HuntingResult(
                        url=url, 
                        data=enhanced_data, 
                        relevance_score=relevance
                    ))
            else:
                # Complete failure - log but don't include in results
                if extraction_result:
                    self.logger.warning(f"Extraction failed for {url}: {extraction_result.get('message', 'Unknown error')}")
                
        return results

    def _calculate_relevance(self, data: Union[Dict[str, Any], List[Dict[str, Any]]], intent: HuntingIntent) -> float:
        """
        Calculate advanced relevance score using multiple factors:
        1. Keyword matching
        2. Entity matching  
        3. Content quality indicators
        4. Freshness (if date available)
        5. Content completeness
        """
        scores = {
            "keyword_match": 0.0,
            "entity_match": 0.0,
            "content_quality": 0.0,
            "freshness": 0.0,
            "completeness": 0.0
        }
        
        # Extract text content for analysis
        text_content = self._extract_text_for_analysis(data)
        
        # 1. Keyword matching (enhanced with position weighting)
        scores["keyword_match"] = self._calculate_keyword_relevance(text_content, intent.keywords)
        
        # 2. Entity matching
        scores["entity_match"] = self._calculate_entity_relevance(text_content, intent.entities)
        
        # 3. Content quality assessment
        scores["content_quality"] = self._assess_content_quality(data, text_content)
        
        # 4. Freshness score (newer content gets higher score)
        scores["freshness"] = self._calculate_freshness_score(data)
        
        # 5. Content completeness (how many expected fields are filled)
        scores["completeness"] = self._calculate_completeness_score(data, intent.output_schema)
        
        # Weighted combination based on content category
        weights = self._get_scoring_weights(intent.content_category)
        final_score = sum(scores[k] * weights[k] for k in scores)
        
        return min(1.0, final_score)
    
    def _extract_text_for_analysis(self, data: Union[Dict, List]) -> str:
        """Extract all text content from data for relevance analysis."""
        if isinstance(data, list):
            all_values = []
            for item in data:
                if isinstance(item, dict):
                    all_values.extend(item.values())
            return " ".join(str(v) for v in all_values).lower()
        elif isinstance(data, dict):
            # Prioritize certain fields for relevance
            important_fields = ["title", "summary", "description", "full_content"]
            text_parts = []
            
            # Add important fields first (with higher weight)
            for field in important_fields:
                if field in data and data[field]:
                    text_parts.append(str(data[field]))
            
            # Add other text fields
            for key, value in data.items():
                if key not in important_fields and isinstance(value, str):
                    text_parts.append(value)
                    
            return " ".join(text_parts).lower()
        else:
            return str(data).lower()
    
    def _calculate_keyword_relevance(self, text: str, keywords: List[str]) -> float:
        """Calculate keyword matching score with position and frequency weighting."""
        if not keywords or not text:
            return 0.0
            
        score = 0.0
        text_words = text.split()
        total_words = len(text_words)
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # Exact phrase matching (higher weight)
            if keyword_lower in text:
                score += 0.3
                
            # Individual word matching
            keyword_words = keyword_lower.split()
            matches = sum(1 for word in keyword_words if word in text_words)
            if keyword_words:
                word_match_ratio = matches / len(keyword_words)
                score += word_match_ratio * 0.2
                
        return min(1.0, score)
    
    def _calculate_entity_relevance(self, text: str, entities: List[str]) -> float:
        """Calculate entity matching score."""
        if not entities or not text:
            return 0.0
            
        score = 0.0
        for entity in entities:
            if entity.lower() in text:
                score += 0.4  # Higher weight for entity matches
                
        return min(1.0, score)
    
    def _assess_content_quality(self, data: Union[Dict, List], text: str) -> float:
        """Assess content quality based on various indicators."""
        quality_score = 0.0
        
        # Convert list to dict if needed
        if isinstance(data, list):
            if not data:
                return 0.0
            # Use first item if it's a list of dicts
            data = data[0] if isinstance(data[0], dict) else {}
        elif not isinstance(data, dict):
            return 0.0
        
        # Text length (optimal range)
        text_length = len(text.split())
        if 50 <= text_length <= 2000:  # Sweet spot for quality content
            quality_score += 0.3
        elif text_length > 20:  # At least some content
            quality_score += 0.1
            
        # Presence of key fields
        quality_indicators = ["title", "author", "publication_date", "summary"]
        filled_indicators = sum(1 for field in quality_indicators if data.get(field))
        quality_score += (filled_indicators / len(quality_indicators)) * 0.4
        
        # Structured content indicators
        if data.get("image_urls") and isinstance(data["image_urls"], list):
            quality_score += 0.1
            
        # Language quality (simple heuristic)
        if text and self._has_readable_text(text):
            quality_score += 0.2
            
        return min(1.0, quality_score)
    
    def _has_readable_text(self, text: str) -> bool:
        """Simple heuristic to check if text appears to be readable."""
        if not text or len(text) < 20:
            return False
            
        # Check for reasonable word/character ratio
        words = text.split()
        if not words:
            return False
            
        avg_word_length = sum(len(word) for word in words) / len(words)
        return 2 <= avg_word_length <= 15  # Reasonable average word length
    
    def _calculate_freshness_score(self, data: Dict) -> float:
        """Calculate freshness score based on publication date."""
        date_fields = ["publication_date", "date_published", "date"]
        
        for field in date_fields:
            if field in data and data[field]:
                try:
                    from datetime import datetime, timedelta
                    # Try to parse the date (simplified - could be enhanced with dateutil)
                    date_str = str(data[field])
                    
                    # Try common date formats
                    for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%d/%m/%Y", "%m/%d/%Y"]:
                        try:
                            pub_date = datetime.strptime(date_str[:10], fmt[:10])
                            days_old = (datetime.now() - pub_date).days
                            
                            # Fresher content gets higher scores
                            if days_old <= 7:
                                return 1.0
                            elif days_old <= 30:
                                return 0.8
                            elif days_old <= 90:
                                return 0.6
                            elif days_old <= 365:
                                return 0.4
                            else:
                                return 0.2
                        except ValueError:
                            continue
                except Exception:
                    pass
        
        return 0.5  # Default score when no date available
    
    def _calculate_completeness_score(self, data: Union[Dict, List], schema: Dict) -> float:
        """Calculate how complete the extracted data is according to the schema."""
        if not schema.get("properties"):
            return 1.0
            
        # Convert list to dict if needed
        if isinstance(data, list):
            if not data:
                return 0.0
            data = data[0] if isinstance(data[0], dict) else {}
        elif not isinstance(data, dict):
            return 0.0
            
        total_fields = len(schema["properties"])
        filled_fields = sum(1 for field in schema["properties"] if data.get(field))
        
        completeness = filled_fields / total_fields if total_fields > 0 else 1.0
        
        # Bonus for required fields
        required_fields = schema.get("required", [])
        if required_fields:
            required_filled = sum(1 for field in required_fields if data.get(field))
            required_completeness = required_filled / len(required_fields)
            # Weight required fields more heavily
            completeness = (completeness * 0.7) + (required_completeness * 0.3)
            
        return completeness
    
    def _get_scoring_weights(self, content_category: str) -> Dict[str, float]:
        """Get scoring weights based on content category."""
        # Default weights
        default_weights = {
            "keyword_match": 0.25,
            "entity_match": 0.25,
            "content_quality": 0.25,
            "freshness": 0.15,
            "completeness": 0.10
        }
        
        # Category-specific adjustments
        if content_category == "news":
            return {
                "keyword_match": 0.20,
                "entity_match": 0.25,
                "content_quality": 0.20,
                "freshness": 0.25,  # More weight on freshness for news
                "completeness": 0.10
            }
        elif content_category == "product" or content_category == "e_commerce":
            return {
                "keyword_match": 0.30,
                "entity_match": 0.20,
                "content_quality": 0.30,
                "freshness": 0.05,  # Less weight on freshness for products
                "completeness": 0.15
            }
        elif content_category == "research" or content_category == "academic":
            return {
                "keyword_match": 0.20,
                "entity_match": 0.30,
                "content_quality": 0.35,  # Higher quality weight for research
                "freshness": 0.05,
                "completeness": 0.10
            }
        
        return default_weights

    def _rank_results(self, results: List[HuntingResult]) -> List[HuntingResult]:
        """Rank the results based on their relevance score."""
        return sorted(results, key=lambda r: r.relevance_score, reverse=True)

    def _deduplicate_results(self, results: List[HuntingResult]) -> List[HuntingResult]:
        """
        Remove duplicate results using multiple deduplication strategies:
        1. Canonical URL detection
        2. Content similarity hashing
        3. Title similarity
        """
        if not results:
            return results
            
        seen_canonical = set()
        seen_content_hashes = set()
        seen_titles = set()
        deduplicated = []
        
        for result in results:
            # Handle data that might be a list or dict
            data = result.data
            if isinstance(data, list):
                if not data:
                    continue
                data = data[0] if isinstance(data[0], dict) else {}
            elif not isinstance(data, dict):
                continue
                
            # Strategy 1: Check canonical URL
            canonical = data.get("canonical_url") or data.get("url") or result.url
            if canonical:
                # Normalize URL (remove trailing slashes, fragments, etc.)
                canonical_normalized = canonical.rstrip('/').split('#')[0].split('?')[0]
                if canonical_normalized in seen_canonical:
                    self.logger.debug(f"Skipping duplicate canonical URL: {canonical_normalized}")
                    continue
                seen_canonical.add(canonical_normalized)
            
            # Strategy 2: Content similarity check
            content_for_hash = ""
            if data.get("summary"):
                content_for_hash += data["summary"] + " "
            if data.get("title"):
                content_for_hash += data["title"] + " "
            if data.get("full_content"):
                # Use first 500 chars of content for hashing to avoid memory issues
                content_for_hash += data["full_content"][:500]
                
            if content_for_hash.strip():
                content_hash = hash(content_for_hash.strip().lower())
                if content_hash in seen_content_hashes:
                    self.logger.debug(f"Skipping duplicate content for: {result.url}")
                    continue
                seen_content_hashes.add(content_hash)
            
            # Strategy 3: Title similarity (exact match for now, could be enhanced with fuzzy matching)
            title = data.get("title", "").strip().lower()
            if title and title in seen_titles:
                self.logger.debug(f"Skipping duplicate title: {title}")
                continue
            if title:
                seen_titles.add(title)
            
            deduplicated.append(result)
        
        self.logger.info(f"Deduplication: {len(results)} -> {len(deduplicated)} results")
        return deduplicated
