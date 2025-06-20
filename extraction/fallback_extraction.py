"""
Fallback Extraction Module

Provides backup extraction methods when primary extraction strategies fail.
Uses multiple content extraction libraries and methods to ensure data extraction.
"""

import json
import asyncio
import logging
from typing import Dict, List, Any, Union, Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import google.generativeai as genai

from extraction.content_extraction import ContentExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FallbackExtraction")

# Global ContentExtractor instance (initialized on first use)
_content_extractor = None

def get_content_extractor(use_stealth_browser: bool = False) -> ContentExtractor:
    """
    Get (or initialize) the global ContentExtractor instance.
    
    Args:
        use_stealth_browser: Whether to use Playwright with stealth mode
    
    Returns:
        ContentExtractor instance
    """
    global _content_extractor
    if not _content_extractor:
        _content_extractor = ContentExtractor(use_stealth_browser=use_stealth_browser)
    return _content_extractor

async def perform_extraction_with_fallback(
    html_content: str,
    url: str,
    ai_suggestions: Optional[Dict[str, Any]] = None,
    use_stealth_browser: bool = False
) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Perform content extraction with multiple fallback methods.
    
    Args:
        html_content: HTML content to extract from
        url: URL of the content
        ai_suggestions: Optional AI suggestions to guide extraction
        use_stealth_browser: Whether to use Playwright with stealth mode
    
    Returns:
        Extracted content as list or dictionary
    """
    # Get ContentExtractor instance
    content_extractor = get_content_extractor(use_stealth_browser)
    
    # Try to determine the content type from URL pattern
    content_type = None
    if ai_suggestions and "content_type" in ai_suggestions:
        content_type = ai_suggestions["content_type"]
    
    logger.info(f"Performing fallback extraction for {url} (content_type: {content_type})")
    
    # Extract content using ContentExtractor
    extraction_result = await content_extractor.extract_content(
        html_content=html_content,
        url=url,
        content_type=content_type
    )
    
    # If extraction succeeded, convert to standard format
    if extraction_result.get("success", False):
        # Format based on content type
        if extraction_result["content_type"] == "listing":
            return format_listing_result(extraction_result, url)
        elif extraction_result["content_type"] == "article":
            return format_article_result(extraction_result, url)
        elif extraction_result["content_type"] == "data_table":
            return format_table_result(extraction_result, url)
        else:
            # Generic content
            return format_generic_result(extraction_result, url)
    
    # If all fallbacks failed, try AI-based extraction
    ai_extraction = await extract_with_ai(html_content, url, ai_suggestions)
    if ai_extraction:
        return ai_extraction
    
    # Last resort: basic extraction with BeautifulSoup
    return basic_extraction(html_content, url)

def format_listing_result(extraction_result: Dict[str, Any], url: str) -> List[Dict[str, Any]]:
    """
    Format listing extraction result to standard format.
    
    Args:
        extraction_result: Extraction result from ContentExtractor
        url: URL of the content
    
    Returns:
        Formatted listing items
    """
    results = []
    
    # Get items from extraction result
    items = extraction_result.get("items", [])
    
    for item in items:
        formatted_item = {
            "source_url": url,
            "item_url": item.get("url", ""),
            "title": item.get("title", ""),
            "description": item.get("description", ""),
            "content_type": "listing_item",
            "extraction_method": extraction_result.get("extraction_method", "unknown"),
            "tags": ["fallback_extraction", "listing_item"]
        }
        
        # Add price if available
        if "price" in item:
            formatted_item["price"] = item["price"]
        
        # Add image if available
        if "image" in item:
            formatted_item["image_url"] = item["image"]
        
        results.append(formatted_item)
    
    # If no items found, return generic extraction
    if not results:
        return [format_generic_result(extraction_result, url)]
    
    return results

def format_article_result(extraction_result: Dict[str, Any], url: str) -> Dict[str, Any]:
    """
    Format article extraction result to standard format.
    
    Args:
        extraction_result: Extraction result from ContentExtractor
        url: URL of the content
    
    Returns:
        Formatted article content
    """
    # Extract metadata
    metadata = extraction_result.get("metadata", {})
    
    return {
        "source_url": url,
        "title": extraction_result.get("title", ""),
        "content": extraction_result.get("text", ""),
        "html_content": extraction_result.get("html", ""),
        "content_type": "article",
        "extraction_method": extraction_result.get("extraction_method", "unknown"),
        "tags": ["fallback_extraction", "article"],
        "author": metadata.get("author", ""),
        "date_published": metadata.get("date", metadata.get("published_date", "")),
        "image_url": metadata.get("image", "")
    }

def format_table_result(extraction_result: Dict[str, Any], url: str) -> List[Dict[str, Any]]:
    """
    Format table extraction result to standard format.
    
    Args:
        extraction_result: Extraction result from ContentExtractor
        url: URL of the content
    
    Returns:
        Formatted table data
    """
    results = []
    
    # Get tables from extraction result
    tables = extraction_result.get("tables", [])
    
    for table in tables:
        # Convert table data to records (list of dictionaries)
        records = []
        headers = table.get("headers", [])
        rows = table.get("rows", [])
        
        for row in rows:
            # Skip empty rows
            if not any(cell for cell in row):
                continue
                
            record = {}
            # Match cells with headers
            for i, cell in enumerate(row):
                header = headers[i] if i < len(headers) else f"column_{i}"
                record[header] = cell
            
            records.append(record)
        
        # Create result for this table
        formatted_table = {
            "source_url": url,
            "table_id": table.get("id", ""),
            "content_type": "data_table",
            "extraction_method": extraction_result.get("extraction_method", "unknown"),
            "tags": ["fallback_extraction", "data_table"],
            "headers": headers,
            "row_count": table.get("row_count", 0),
            "data": records
        }
        
        results.append(formatted_table)
    
    # If no tables found, return generic extraction
    if not results:
        return [format_generic_result(extraction_result, url)]
    
    return results

def format_generic_result(extraction_result: Dict[str, Any], url: str) -> Dict[str, Any]:
    """
    Format generic extraction result to standard format.
    
    Args:
        extraction_result: Extraction result from ContentExtractor
        url: URL of the content
    
    Returns:
        Formatted generic content
    """
    # Extract metadata
    metadata = extraction_result.get("metadata", {})
    
    return {
        "source_url": url,
        "title": extraction_result.get("title", ""),
        "content": extraction_result.get("text", ""),
        "html_content": extraction_result.get("html", ""),
        "content_type": "generic",
        "extraction_method": extraction_result.get("extraction_method", "unknown"),
        "tags": ["fallback_extraction", "generic"],
        "metadata": metadata
    }

async def extract_with_ai(html_content: str, url: str, ai_suggestions: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Extract content using AI analysis.
    
    Args:
        html_content: HTML content
        url: URL of the content
        ai_suggestions: Optional AI suggestions to guide extraction
    
    Returns:
        Extracted content or None if extraction failed
    """
    if not html_content:
        return None
    
    try:
        # Clean HTML content for AI processing
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "noscript", "iframe"]):
            script.extract()
            
        # Get text content
        text_content = soup.get_text(separator=' ', strip=True)
        
        # Limit text size for AI processing
        if len(text_content) > 8000:
            text_content = text_content[:8000] + "..."
        
        # Build prompt for AI extraction
        prompt = f"""
        Extract the main content from this webpage:
        
        URL: {url}
        
        PAGE CONTENT:
        {text_content}
        
        Extract the following information and return it as JSON:
        1. "title": The title of the page
        2. "content": The main textual content
        3. "content_type": The type of content (article, product, listing, etc.)
        4. "key_elements": A list of key elements or facts from the page
        
        Return only the JSON with no explanation.
        """
        
        # Call Gemini API
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        response_text = response.text
        
        # Extract JSON from response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}')
        
        if json_start != -1 and json_end != -1:
            json_string = response_text[json_start:json_end+1]
            result = json.loads(json_string)
            
            # Format result
            return {
                "source_url": url,
                "title": result.get("title", ""),
                "content": result.get("content", ""),
                "content_type": result.get("content_type", "generic"),
                "extraction_method": "ai",
                "tags": ["fallback_extraction", "ai"],
                "key_elements": result.get("key_elements", [])
            }
        
        # If JSON extraction failed, return None
        return None
        
    except Exception as e:
        logger.error(f"AI extraction failed: {str(e)}")
        return None

def basic_extraction(html_content: str, url: str) -> Dict[str, Any]:
    """
    Perform basic extraction using BeautifulSoup as a last resort.
    
    Args:
        html_content: HTML content
        url: URL of the content
    
    Returns:
        Extracted content
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "noscript", "iframe"]):
            script.extract()
            
        # Extract title
        title = soup.title.get_text(strip=True) if soup.title else ""
        
        # Extract text content
        text_content = soup.get_text(separator=' ', strip=True)
        
        # Find main content containers (best guess)
        main_content = soup.find('main') or soup.find('article') or soup.find('div', {'id': 'content'})
        
        if main_content:
            main_text = main_content.get_text(separator=' ', strip=True)
            if len(main_text) > 200:  # If main content is substantial
                text_content = main_text
        
        # Extract images
        images = []
        for img in soup.find_all('img', src=True):
            src = img['src']
            if src.startswith('data:'):
                continue  # Skip data URIs
                
            # Resolve relative URLs
            if src.startswith('/'):
                src = urljoin(url, src)
                
            alt = img.get('alt', '')
            images.append({
                "src": src,
                "alt": alt
            })
        
        # Extract links
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            # Skip anchor links
            if href.startswith('#'):
                continue
                
            # Resolve relative URLs
            if href.startswith('/'):
                href = urljoin(url, href)
                
            text = a.get_text(strip=True)
            links.append({
                "href": href,
                "text": text
            })
        
        return {
            "source_url": url,
            "title": title,
            "content": text_content,
            "content_type": "generic",
            "extraction_method": "basic_soup",
            "tags": ["fallback_extraction", "basic"],
            "images": images[:10],  # Limit to 10 images
            "links": links[:20]     # Limit to 20 links
        }
        
    except Exception as e:
        logger.error(f"Basic extraction failed: {str(e)}")
        return {
            "source_url": url,
            "title": "",
            "content": "Extraction failed",
            "content_type": "error",
            "extraction_method": "error",
            "tags": ["extraction_error"],
            "error": str(e)
        }