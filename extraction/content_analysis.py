"""
Content Analysis Module

This module provides functionality to analyze website structure and content,
with both function-based utilities for simple analysis and class-based tools
for more advanced content processing.
"""

# ----- IMPORTS (consolidated) -----
import json
import re
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import Counter
from urllib.parse import urlparse

from bs4 import BeautifulSoup
import google.generativeai as genai
import numpy as np

# spaCy imports with fallback
try:
    import spacy
    from spacy.lang.en.stop_words import STOP_WORDS
    
    # Try to load the best available English model
    SPACY_MODELS = ['en_core_web_lg', 'en_core_web_md', 'en_core_web_sm']
    nlp = None
    SPACY_AVAILABLE = False
    
    for model_name in SPACY_MODELS:
        try:
            nlp = spacy.load(model_name)
            SPACY_AVAILABLE = True
            logger = logging.getLogger("ContentAnalysis")
            logger.info(f"Successfully loaded spaCy model: {model_name}")
            break
        except OSError:
            continue
    
    if not SPACY_AVAILABLE:
        logger = logging.getLogger("ContentAnalysis")
        logger.error("No spaCy models available. Install with: python -m spacy download en_core_web_lg")
        
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None
    nlp = None
    STOP_WORDS = set()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ContentAnalysis")

# ----- FUNCTION-BASED SITE STRUCTURE ANALYSIS -----

async def analyze_site_structure(html_content: str, url: str) -> Dict[str, Any]:
    """
    Analyze website structure to determine the best extraction approach.
    
    Args:
        html_content: HTML content of the page
        url: URL of the page
        
    Returns:
        Dictionary with site analysis results
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Initialize analysis results
        analysis = {
            "site_type": "generic",
            "content_structure": 5,  # Scale of 1-10, where 10 is highly structured
            "js_dependent": False,
            "main_content_selector": "body",
            "item_container_selector": "",
            "recommended_extraction_type": "raw",
            "recommended_selectors": [],
            "requires_browser": False,  # New flag to indicate if browser rendering is needed
            "js_rendering_level": 0,   # Scale of 0-3, indicating JS complexity
            "dynamic_content_markers": []  # Selectors for areas with dynamically loaded content
        }
        
        # Basic checks for site structure
        has_structured_data = bool(soup.find_all(['article', 'product', 'section', 'div']))
        has_lists = bool(soup.find_all(['ul', 'ol', 'dl']))
        has_tables = bool(soup.find_all('table'))
        has_forms = bool(soup.find_all('form'))
        has_pagination = bool(soup.select('[class*="pag"]') or soup.select('[id*="pag"]'))
        
        # Enhanced JavaScript dependency detection
        js_dependency_score = detect_javascript_dependency(soup)
        js_dependent = js_dependency_score > 1
        
        # Set additional JS-related properties
        analysis['js_dependent'] = js_dependent
        analysis['js_rendering_level'] = js_dependency_score
        analysis['requires_browser'] = js_dependency_score >= 2
        
        # Identify dynamic content areas
        dynamic_areas = identify_dynamic_content_areas(soup)
        analysis['dynamic_content_markers'] = dynamic_areas
        
        # Detect site type
        site_type = detect_site_type(soup, url)
        analysis['site_type'] = site_type
        
        # Assess content structure level
        structure_score = 0
        if has_structured_data:
            structure_score += 2
        if has_lists:
            structure_score += 1
        if has_tables:
            structure_score += 3
        if has_pagination:
            structure_score += 1
        if len(soup.find_all('div', {'class': True})) > 10:
            structure_score += 1
        
        # Check for repeated patterns (likely list of items)
        repeated_patterns = detect_repeated_patterns(soup)
        if repeated_patterns['found']:
            structure_score += 2
            analysis['item_container_selector'] = repeated_patterns['container_selector']
            
        # Cap structure score at 10
        analysis['content_structure'] = min(10, structure_score)
        
        # Find main content area
        main_content = find_main_content_area(soup)
        if main_content:
            analysis['main_content_selector'] = main_content
            
        # Recommend extraction type based on analysis
        if structure_score >= 8 and not js_dependent:
            analysis['recommended_extraction_type'] = 'css'
        elif js_dependency_score >= 2:
            analysis['recommended_extraction_type'] = 'browser'
        elif structure_score >= 6 or js_dependent:
            analysis['recommended_extraction_type'] = 'hybrid'
        else:
            analysis['recommended_extraction_type'] = 'raw'
            
        # Get recommended CSS selectors if appropriate
        if analysis['recommended_extraction_type'] in ['css', 'hybrid']:
            analysis['recommended_selectors'] = generate_selectors_for_site_type(soup, site_type)
            
        # Generate content schema using AI if needed
        if not analysis['recommended_selectors'] and analysis['content_structure'] >= 5:
            ai_selectors = await analyze_with_ai(html_content, url, site_type)
            if ai_selectors:
                analysis['recommended_selectors'] = ai_selectors
                
        return analysis
        
    except Exception as e:
        print(f"Error in site analysis: {str(e)}")
        return {
            "analysis_error": str(e),
            "site_type": "generic",
            "content_structure": 1,
            "js_dependent": False,
            "main_content_selector": "body",
            "recommended_extraction_type": "raw",
            "requires_browser": False,
            "js_rendering_level": 0,
            "dynamic_content_markers": []
        }


def detect_javascript_dependency(soup: BeautifulSoup) -> int:
    """
    Analyze the page for JavaScript dependency and rate it on a scale of 0-3.
    
    Args:
        soup: BeautifulSoup object of the HTML
        
    Returns:
        Integer score from 0-3, where:
        0 = No JS dependency
        1 = Light JS enhancement
        2 = Moderate JS dependency
        3 = Heavy JS dependency (SPA, dynamic loading)
    """
    js_score = 0
    
    # Check for SPA frameworks
    spa_frameworks = ['react', 'vue', 'angular', 'ember', 'backbone', 'next', 'nuxt']
    spa_detected = False
    
    # Check scripts for framework imports
    scripts = soup.find_all('script')
    script_text = ' '.join([s.get('src', '') for s in scripts if s.get('src')])
    script_content = ' '.join([s.string or '' for s in scripts])
    
    # Check for framework indicators in script tags
    for framework in spa_frameworks:
        if (f'{framework}.js' in script_text or 
            f'{framework}.min.js' in script_text or 
            f'/{framework}@' in script_text or
            f'{framework}.' in script_text):
            spa_detected = True
            js_score = 3
            break
    
    # Check for framework identifiers in HTML
    html_str = str(soup)
    framework_indicators = [
        'data-reactroot', 'ng-app', 'ng-controller', 'v-', 'v-for', 'v-if',
        'x-data', 'x-bind', 'data-vue', 'ember-view', 'svelte'
    ]
    
    if any(indicator in html_str for indicator in framework_indicators) and not spa_detected:
        spa_detected = True
        js_score = 3
    
    # Check for lazy loading and AJAX indicators
    ajax_indicators = [
        'loading="lazy"', 'load more', 'infinite scroll', 'pagination',
        'fetch(', 'axios.', 'ajax', '$http', 'XMLHttpRequest'
    ]
    
    ajax_detected = any(indicator in html_str or indicator in script_content 
                       for indicator in ajax_indicators)
    
    if ajax_detected and js_score < 2:
        js_score = 2
    
    # Check for typical JS behavior indicators
    if js_score == 0:
        behavior_indicators = [
            'onclick=', 'addEventListener', 'getElementById', 'querySelector',
            'toggle(', 'classList', 'innerText', 'innerHTML'
        ]
        
        if any(indicator in html_str or indicator in script_content 
              for indicator in behavior_indicators):
            js_score = 1
    
    # Check for empty content containers that might be filled by JS
    empty_containers = soup.select('div[id]:empty, div[class]:empty')
    if len(empty_containers) > 3:
        js_score = max(js_score, 2)
    
    return js_score


def identify_dynamic_content_areas(soup: BeautifulSoup) -> List[str]:
    """
    Identify areas in the page that are likely populated by JavaScript.
    
    Args:
        soup: BeautifulSoup object of the HTML
        
    Returns:
        List of CSS selectors targeting dynamic content areas
    """
    dynamic_areas = []
    
    # Check for loading indicators
    loading_classes = ['loading', 'spinner', 'skeleton', 'placeholder', 'shimmer']
    for cls in loading_classes:
        elements = soup.select(f'[class*="{cls}"]')
        if elements:
            for elem in elements:
                if elem.get('class'):
                    dynamic_areas.append(f'.{elem.get("class")[0]}')
                elif elem.get('id'):
                    dynamic_areas.append(f'#{elem.get("id")}')
    
    # Check for containers with data-* attributes (often used for JS hooks)
    data_containers = soup.select('[data-src], [data-url], [data-api], [data-load], [data-fetch]')
    for container in data_containers:
        if container.get('id'):
            dynamic_areas.append(f'#{container.get("id")}')
        elif container.get('class'):
            dynamic_areas.append(f'.{container.get("class")[0]}')
    
    # Check for empty containers with classes/ids that suggest content
    content_indicators = ['content', 'results', 'data', 'items', 'products', 'list']
    for indicator in content_indicators:
        selectors = [
            f'[id*="{indicator}"]:empty', 
            f'[class*="{indicator}"]:empty',
            f'[id*="{indicator}"] > :only-child.loading',
            f'[class*="{indicator}"] > :only-child.loading'
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            for elem in elements:
                if elem.get('id'):
                    dynamic_areas.append(f'#{elem.get("id")}')
                elif elem.get('class'):
                    dynamic_areas.append(f'.{elem.get("class")[0]}')
    
    # Return unique selectors
    return list(set(dynamic_areas))


def detect_site_type(soup: BeautifulSoup, url: str) -> str:
    """
    Detect the type of website from its content and URL.
    
    Args:
        soup: BeautifulSoup object of the HTML
        url: URL of the page
        
    Returns:
        Site type string
    """
    url_lower = url.lower()
    text_content = soup.get_text().lower()
    
    # Look for keywords in URL and content
    ecommerce_terms = ['product', 'shop', 'store', 'buy', 'price', 'cart', 'checkout']
    news_terms = ['news', 'article', 'story', 'report', 'headline', 'journalist', 'editor']
    blog_terms = ['blog', 'post', 'author', 'comment', 'entry', 'latest']
    forum_terms = ['forum', 'thread', 'topic', 'reply', 'post', 'comment', 'discussion']
    realestate_terms = ['property', 'house', 'home', 'apartment', 'rent', 'sale', 'real estate']
    
    # Check URL first for strong indicators
    if any(term in url_lower for term in ecommerce_terms):
        return 'ecommerce'
    if any(term in url_lower for term in news_terms):
        return 'news'
    if any(term in url_lower for term in blog_terms):
        return 'blog'
    if any(term in url_lower for term in forum_terms):
        return 'forum'
    if any(term in url_lower for term in realestate_terms):
        return 'realestate'
    
    # Check content as fallback
    ecommerce_count = sum(1 for term in ecommerce_terms if term in text_content)
    news_count = sum(1 for term in news_terms if term in text_content)
    blog_count = sum(1 for term in blog_terms if term in text_content)
    forum_count = sum(1 for term in forum_terms if term in text_content)
    realestate_count = sum(1 for term in realestate_terms if term in text_content)
    
    counts = {
        'ecommerce': ecommerce_count,
        'news': news_count,
        'blog': blog_count,
        'forum': forum_count,
        'realestate': realestate_count
    }
    
    # Get the site type with the highest count
    if max(counts.values()) > 0:
        return max(counts, key=counts.get)
        
    # Look for structural indicators
    if soup.find('article'):
        return 'news'
    if soup.find(class_=lambda c: c and ('product' in c.lower())):
        return 'ecommerce'
    if soup.find(class_=lambda c: c and ('post' in c.lower() or 'blog' in c.lower())):
        return 'blog'
    if soup.find(class_=lambda c: c and ('thread' in c.lower() or 'topic' in c.lower())):
        return 'forum'
    
    # Default to generic if nothing matches
    return 'generic'


def detect_repeated_patterns(soup: BeautifulSoup) -> Dict[str, Any]:
    """
    Detect repeated patterns in the HTML that might indicate lists of items.
    
    Args:
        soup: BeautifulSoup object of the HTML
        
    Returns:
        Dictionary with pattern detection results
    """
    result = {
        "found": False,
        "container_selector": "",
        "item_selector": "",
        "count": 0
    }
    
    # Check for common list patterns
    for container in ['ul', 'ol', 'div', 'section', 'table']:
        # Look for containers with multiple same-tag children
        for selector in [f'{container} > li', f'{container} > div', f'{container} > article']:
            items = soup.select(selector)
            if len(items) >= 3:
                # Found a repeated pattern
                result["found"] = True
                result["container_selector"] = container
                result["item_selector"] = selector.split('>')[-1].strip()
                result["count"] = len(items)
                return result
    
    # Look for divs with class patterns indicating lists
    list_classes = ['list', 'grid', 'items', 'products', 'results', 'cards', 'articles']
    for class_name in list_classes:
        containers = soup.find_all(class_=lambda c: c and class_name in c.lower())
        for container in containers:
            # Check if it has multiple similar children
            children = [c for c in container.children if c.name]
            child_tags = [c.name for c in children if c.name]
            
            # Count occurrences of each tag
            tag_counts = {}
            for tag in child_tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
                
            # If any tag appears 3+ times, it's likely a list
            for tag, count in tag_counts.items():
                if count >= 3:
                    result["found"] = True
                    container_classes = container.get('class', [])
                    container_class = container_classes[0] if container_classes else ''
                    result["container_selector"] = f"{container.name}.{container_class}" if container_class else container.name
                    result["item_selector"] = tag
                    result["count"] = count
                    return result
    
    return result


def find_main_content_area(soup: BeautifulSoup) -> str:
    """
    Find the main content area of the page.
    
    Args:
        soup: BeautifulSoup object of the HTML
        
    Returns:
        CSS selector for the main content area
    """
    # Check for semantic main tag
    if soup.find('main'):
        return 'main'
        
    # Check for common main content id/class patterns
    main_ids = ['main', 'content', 'main-content', 'primary']
    for id_name in main_ids:
        element = soup.find(id=id_name)
        if element:
            return f'#{id_name}'
    
    main_classes = ['main', 'content', 'main-content', 'container', 'page-content']
    for class_name in main_classes:
        elements = soup.find_all(class_=lambda c: c and class_name.lower() in c.lower())
        if elements:
            return f'.{elements[0].get("class")[0]}'
    
    # If we can't find a specific main content area, look for the div with most content
    divs = soup.find_all('div')
    if divs:
        # Find the div with the most text content
        div_content = [(d, len(d.get_text())) for d in divs]
        div_content.sort(key=lambda x: x[1], reverse=True)
        
        # Get the top div if it has a significant amount of content
        if div_content and div_content[0][1] > 200:
            top_div = div_content[0][0]
            div_id = top_div.get('id', '')
            div_classes = top_div.get('class', [])
            
            if div_id:
                return f'#{div_id}'
            elif div_classes:
                return f'.{div_classes[0]}'
    
    # Fallback to body
    return 'body'


def generate_selectors_for_site_type(soup: BeautifulSoup, site_type: str) -> List[Dict[str, str]]:
    """
    Generate appropriate CSS selectors based on the site type.
    
    Args:
        soup: BeautifulSoup object of the HTML
        site_type: Type of site
        
    Returns:
        List of selector dictionaries
    """
    selectors = []
    
    if site_type == 'ecommerce':
        # Product selectors - generalized to work with any product page
        product_name = find_selector(soup, ['product-title', 'product-name', 'product_title'])
        if product_name:
            selectors.append({"name": "product_name", "selector": product_name, "type": "text"})
            
        price = find_selector(soup, ['price', 'product-price', 'product_price'])
        if price:
            selectors.append({"name": "price", "selector": price, "type": "text"})
            
        image = find_selector(soup, ['product-image', 'product_image'])
        if image:
            selectors.append({"name": "image_url", "selector": f"{image} img", "type": "attribute", "attribute": "src"})
            
        description = find_selector(soup, ['description', 'product-description', 'product_description'])
        if description:
            selectors.append({"name": "description", "selector": description, "type": "text"})
            
    elif site_type == 'news' or site_type == 'blog':
        # Article selectors - generalized for any article/news content
        title = find_selector(soup, ['title', 'headline', 'article-title', 'post-title'])
        if title:
            selectors.append({"name": "title", "selector": title, "type": "text"})
            
        author = find_selector(soup, ['author', 'byline', 'meta-author'])
        if author:
            selectors.append({"name": "author", "selector": author, "type": "text"})
            
        date = find_selector(soup, ['date', 'time', 'published', 'meta-date'])
        if date:
            selectors.append({"name": "date", "selector": date, "type": "text"})
            
        content = find_selector(soup, ['content', 'article-content', 'post-content', 'entry-content'])
        if content:
            selectors.append({"name": "content", "selector": content, "type": "text"})
            
        image = find_selector(soup, ['featured-image', 'article-image', 'post-image'])
        if image:
            selectors.append({"name": "image_url", "selector": f"{image} img", "type": "attribute", "attribute": "src"})
            
    elif site_type == 'forum':
        # Forum discussion selectors - generalized for forum content
        topic = find_selector(soup, ['topic', 'thread-title', 'discussion-title'])
        if topic:
            selectors.append({"name": "topic", "selector": topic, "type": "text"})
            
        post = find_selector(soup, ['post', 'message', 'comment', 'post-content'])
        if post:
            selectors.append({"name": "post_content", "selector": post, "type": "text"})
            
        author = find_selector(soup, ['author', 'username', 'poster'])
        if author:
            selectors.append({"name": "author", "selector": author, "type": "text"})
            
        date = find_selector(soup, ['date', 'time', 'post-date', 'timestamp'])
        if date:
            selectors.append({"name": "date", "selector": date, "type": "text"})
            
    elif site_type == 'realestate':
        # Real estate property selectors - generalized for property listings
        address = find_selector(soup, ['address', 'property-address', 'listing-address'])
        if address:
            selectors.append({"name": "address", "selector": address, "type": "text"})
            
        price = find_selector(soup, ['price', 'property-price', 'listing-price'])
        if price:
            selectors.append({"name": "price", "selector": price, "type": "text"})
            
        bedrooms = find_selector(soup, ['beds', 'bedrooms', 'bed-bath'])
        if bedrooms:
            selectors.append({"name": "bedrooms", "selector": bedrooms, "type": "text"})
            
        bathrooms = find_selector(soup, ['baths', 'bathrooms', 'bed-bath'])
        if bathrooms:
            selectors.append({"name": "bathrooms", "selector": bathrooms, "type": "text"})
            
        image = find_selector(soup, ['property-image', 'listing-image'])
        if image:
            selectors.append({"name": "image_url", "selector": f"{image} img", "type": "attribute", "attribute": "src"})
            
        description = find_selector(soup, ['description', 'property-description', 'listing-description'])
        if description:
            selectors.append({"name": "description", "selector": description, "type": "text"})
    
    return selectors


def find_selector(soup: BeautifulSoup, class_patterns: List[str]) -> str:
    """
    Find the first matching selector for a set of class patterns.
    
    Args:
        soup: BeautifulSoup object
        class_patterns: List of class name patterns to look for
        
    Returns:
        CSS selector string or empty string if not found
    """
    # Check ids first
    for pattern in class_patterns:
        element = soup.find(id=lambda i: i and pattern.lower() in i.lower())
        if element:
            return f"#{element.get('id')}"
    
    # Check classes
    for pattern in class_patterns:
        elements = soup.find_all(class_=lambda c: c and pattern.lower() in c.lower())
        if elements:
            classes = elements[0].get('class', [])
            if classes:
                for cls in classes:
                    if pattern.lower() in cls.lower():
                        return f".{cls}"
    
    # Check data attributes
    for pattern in class_patterns:
        elements = soup.find_all(attrs={"data-testid": lambda v: v and pattern.lower() in v.lower()})
        if elements:
            return f"[data-testid='{elements[0].get('data-testid')}']"
            
    elements = soup.find_all(attrs={"data-id": lambda v: v and any(p.lower() in v.lower() for p in class_patterns)})
    if elements:
        return f"[data-id='{elements[0].get('data-id')}']"
    
    # Check common heading patterns
    for heading in ['h1', 'h2', 'h3', 'h4']:
        for pattern in class_patterns:
            elements = soup.find_all(heading, string=lambda s: s and pattern.lower() in s.lower())
            if elements:
                return heading
    
    # Check common HTML5 semantic elements
    semantic_tags = {
        'title': ['title', 'header h1', 'article h1'],
        'author': ['author', 'byline', 'meta'],
        'date': ['time', 'date', 'published'],
        'content': ['article', 'section', 'main'],
        'description': ['description', 'summary', 'details'],
        'price': ['price', '.price']
    }
    
    for pattern in class_patterns:
        if pattern in semantic_tags:
            for tag in semantic_tags[pattern]:
                if soup.select(tag):
                    return tag
    
    return ""


async def analyze_with_ai(html_content: str, url: str, site_type: str) -> List[Dict[str, str]]:
    """
    Use AI to analyze page structure and identify CSS selectors.
    
    Args:
        html_content: HTML content of the page
        url: URL of the page
        site_type: Type of site
        
    Returns:
        List of selector dictionaries generated by AI
    """
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Create a clean subset of HTML to avoid token limits
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Strip scripts and styles to reduce size
        for script in soup(['script', 'style', 'noscript', 'iframe', 'svg']):
            script.decompose()
        
        # Extract the main content area if possible
        main_content = soup.find('main') or soup.find(id=lambda i: i and 'content' in i.lower())
        if not main_content:
            main_content = soup.find('body')
        
        # Sample HTML structure (just show important parts)
        sample_html = str(main_content)[:5000]  # Limit to first 5000 chars
        
        instruction = f"""
        Analyze the HTML structure of this {site_type} website page to identify important selectors for data extraction.
        
        Return ONLY a JSON array of selectors with this structure:
        [
          {{
            "name": "selector_name",
            "selector": "CSS selector",
            "type": "text" or "attribute",
            "attribute": "attribute_name" (only if type is "attribute")
          }}
        ]
        
        URL: {url}
        HTML sample:
        {sample_html}
        """
        
        # Get AI response
        response = model.generate_content(instruction)
        response_text = response.text
        
        # Extract JSON
        if "```json" in response_text:
            json_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_text = response_text.split("```")[1].strip()
        else:
            json_text = response_text
        
        # Try to parse the JSON
        try:
            selectors = json.loads(json_text)
            
            # Validate the structure
            if isinstance(selectors, list):
                valid_selectors = []
                for selector in selectors:
                    if isinstance(selector, dict) and "name" in selector and "selector" in selector and "type" in selector:
                        # Only add valid selectors
                        valid_selectors.append(selector)
                return valid_selectors
        except json.JSONDecodeError:
            print("Failed to parse AI-generated selectors as JSON")
            return []
    
    except Exception as e:
        print(f"Error in AI analysis: {str(e)}")
        return []


async def generate_content_filter_instructions(user_prompt: str) -> str:
    """
    Generate content filtering instructions based on user prompt.
    
    Args:
        user_prompt: The user's original extraction prompt
        
    Returns:
        Content filtering instructions
    """
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        instruction = f"""
        Based on this user scraping request:
        "{user_prompt}"
        
        Create clear instructions for filtering scraped content to focus only on what the user wants.
        Return only the instructions with no explanation or formatting.
        """
        
        response = model.generate_content(instruction)
        
        # Return the filtering instructions
        return response.text.strip()
        
    except Exception as e:
        print(f"Error generating content filter instructions: {str(e)}")
        return f"Extract only the information that directly relates to: {user_prompt}"


def analyze_page_structure(html_content: str) -> Dict[str, Any]:
    """
    Analyze a webpage's structure to determine content organization and extraction approach.
    
    Args:
        html_content: HTML content of the page
        
    Returns:
        Dictionary with analysis results
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Initialize result structure
    result = {
        "content_type": "unknown",
        "structure_score": 0,  # 0-10 scale for structural clarity
        "main_content_area": None,
        "has_listings": False,
        "has_pagination": False,
        "has_tables": False,
        "js_dependent": False,
        "metadata": {},
        "recommended_extraction": "llm",  # Default to llm
    }
    
    # Check for article/blog content
    article_elements = soup.select('article, .article, .post, .blog-post, [itemtype*="Article"]')
    if article_elements:
        result["content_type"] = "article"
        result["structure_score"] = 7
        result["main_content_area"] = "article"
        
    # Check for product listings content
    product_elements = soup.select('.product, .item, [itemtype*="Product"], .products, .items, .listing')
    if product_elements:
        result["content_type"] = "product_listing"
        result["structure_score"] = 8
        result["main_content_area"] = ".product, .item, [itemtype*='Product']"
        result["has_listings"] = True
        
    # Check for data tables
    tables = soup.find_all('table')
    if tables:
        result["has_tables"] = True
        if len(tables) > 0 and not result["content_type"] == "product_listing":
            result["content_type"] = "data_table"
            result["structure_score"] = 9  # Tables are usually very structured
            result["main_content_area"] = "table"
        
    # Check for pagination
    pagination_elements = soup.select('.pagination, .pager, [class*="pag"], nav ul li a[href*="page"]')
    if pagination_elements:
        result["has_pagination"] = True
        
    # Check for JavaScript dependency
    script_tags = soup.find_all('script')
    js_load_indicators = [
        "loading", "spinner", "lazy", "ajax", "dynamic", "fetch", 
        "load more", "infinite scroll", "Vue", "React", "Angular"
    ]
    
    js_dependent = False
    for tag in script_tags:
        text = tag.string if tag.string else ""
        if any(indicator.lower() in text.lower() for indicator in js_load_indicators):
            js_dependent = True
            break
            
    # Also check for SPA frameworks 
    spa_indicators = ['vue', 'react', 'angular', 'ember', 'backbone']
    js_framework_detected = any(
        f'/{framework}.' in str(soup) or f'{framework}.js' in str(soup)
        for framework in spa_indicators
    )
    
    result["js_dependent"] = js_dependent or js_framework_detected
    
    # Extract page metadata
    meta_tags = soup.find_all('meta')
    metadata = {}
    
    for tag in meta_tags:
        if tag.get('name') and tag.get('content'):
            metadata[tag.get('name')] = tag.get('content')
        elif tag.get('property') and tag.get('content'):
            metadata[tag.get('property')] = tag.get('content')
    
    # Also look for JSON-LD structured data
    json_ld_tags = soup.find_all('script', type='application/ld+json')
    if json_ld_tags:
        metadata['has_json_ld'] = True
        
    result["metadata"] = metadata
    
    # Determine recommended extraction approach
    if result["structure_score"] >= 7 and not result["js_dependent"]:
        result["recommended_extraction"] = "css"
    elif result["js_dependent"] or result["structure_score"] < 5:
        result["recommended_extraction"] = "llm"
    else:
        result["recommended_extraction"] = "hybrid"
        
    return result

def identify_content_elements(html_content: str, content_description: str) -> Dict[str, Any]:
    """
    Identify elements in the page that match the content description.
    
    Args:
        html_content: HTML content of the page
        content_description: Description of the content to extract
        
    Returns:
        Dictionary with identified elements and selectors
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract keywords from the content description
    keywords = extract_keywords(content_description)
    
    # Initialize results
    result = {
        "possible_selectors": [],
        "content_containers": [],
        "confidence": 0,
        "element_counts": {}
    }
    
    # Map common content types to likely CSS selectors
    content_type_selectors = {
        "product": [".product", ".item", "[itemtype*='Product']", ".products", ".items"],
        "article": ["article", ".article", ".post", ".blog-post", "[itemtype*='Article']", ".content"],
        "table": ["table", ".table", ".data-table", ".grid"],
        "list": ["ul", "ol", ".list", ".results", ".items"],
        "profile": [".profile", ".user", ".author", "[itemtype*='Person']", ".bio"],
        "contact": [".contact", ".address", ".location", ".phone", ".email"],
        "review": [".review", ".rating", ".testimonial", ".comment", ".feedback"]
    }
    
    # Determine content type based on keywords
    content_type = "unknown"
    for ctype, terms in content_type_mapping().items():
        if any(term.lower() in content_description.lower() for term in terms):
            content_type = ctype
            break
    
    # Get potential selectors for this content type
    potential_selectors = content_type_selectors.get(content_type, [])
    
    # Count elements for each potential selector
    for selector in potential_selectors:
        try:
            elements = soup.select(selector)
            if elements:
                result["possible_selectors"].append({
                    "selector": selector,
                    "count": len(elements),
                    "first_element_text": elements[0].get_text()[:100] if elements else ""
                })
                result["element_counts"][selector] = len(elements)
        except Exception:
            continue
    
    # Look for elements containing keywords
    for keyword in keywords:
        for tag in soup.find_all(['div', 'section', 'article', 'main', 'aside']):
            text = tag.get_text().lower()
            if keyword.lower() in text:
                # Check if this element has a class or ID
                if tag.get('class') or tag.get('id'):
                    selector = build_selector_for_element(tag)
                    # Check if this selector already exists
                    if selector and not any(s["selector"] == selector for s in result["possible_selectors"]):
                        result["possible_selectors"].append({
                            "selector": selector,
                            "count": 1,
                            "first_element_text": tag.get_text()[:100],
                            "matched_keyword": keyword
                        })
    
    # Sort selectors by count (prefer selectors that match multiple elements)
    result["possible_selectors"] = sorted(
        result["possible_selectors"], 
        key=lambda x: x["count"], 
        reverse=True
    )
    
    # Calculate confidence based on match quality
    if result["possible_selectors"]:
        # Higher confidence if we have multiple matching selectors
        if len(result["possible_selectors"]) >= 3:
            result["confidence"] = 0.8
        elif len(result["possible_selectors"]) >= 1:
            result["confidence"] = 0.6
        else:
            result["confidence"] = 0.4
    else:
        result["confidence"] = 0.2
        
    return result

def build_selector_for_element(tag):
    """Build a CSS selector for a specific element"""
    selector_parts = []
    
    # Add tag type
    selector_parts.append(tag.name)
    
    # Add ID if present (highest specificity)
    if tag.get('id'):
        selector_parts.append(f'#{tag["id"]}')
        return ' '.join(selector_parts)
    
    # Add first class if present
    if tag.get('class'):
        selector_parts.append(f'.{tag["class"][0]}')
    
    return ' '.join(selector_parts)

def extract_keywords(text):
    """Extract keywords from text by removing common words"""
    # Split the text into words
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Remove common stop words
    stop_words = [
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'with', 'about', 'from', 'by', 'of', 'that', 'this', 'these', 'those',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'can', 'could', 'will', 'would', 'should', 'all'
    ]
    
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Return unique keywords
    return list(set(keywords))

def content_type_mapping():
    """Mapping between content types and their common terms"""
    return {
        "product": [
            "product", "item", "price", "buy", "purchase", "shop", "store", 
            "cart", "catalog", "ecommerce", "sale", "discount"
        ],
        "article": [
            "article", "post", "blog", "news", "story", "content", "read",
            "publication", "author", "published", "date"
        ],
        "table": [
            "table", "data", "statistics", "comparison", "stats", "spreadsheet",
            "rows", "columns", "grid", "values"
        ],
        "list": [
            "list", "items", "bullet", "numbered", "ranking", "top", "best",
            "results", "collection"
        ],
        "profile": [
            "profile", "user", "person", "author", "bio", "about", "member",
            "account", "contact", "personal"
        ],
        "contact": [
            "contact", "address", "location", "phone", "email", "map",
            "directions", "hours", "schedule"
        ],
        "review": [
            "review", "rating", "testimonial", "feedback", "comment", "opinion",
            "stars", "recommendation", "critique"
        ]
    }

# ----- CLASS-BASED CONTENT ANALYSIS -----

class ContentAnalyzer:
    """
    Analyzes content to determine quality, extract entities, and structure information.
    
    This class provides:
    - Content quality scoring
    - Entity extraction (people, organizations, locations, etc.)
    - Topic classification
    - Sentiment analysis
    - Structured information extraction
    - Key metadata extraction
    """
    
    def __init__(self, use_ai: bool = True, ai_model: str = 'gemini-2.0-flash'):
        """
        Initialize the content analyzer.
        
        Args:
            use_ai: Whether to use AI for enhanced analysis
            ai_model: AI model to use for analysis
        """
        self.use_ai = use_ai
        self.ai_model = ai_model
        
        # Initialize NLP components with spaCy first, fallback to basic methods
        if SPACY_AVAILABLE:
            self.nlp = nlp
            self.stop_words = STOP_WORDS
        else:
            self.nlp = None
            # Basic fallback stopwords
            self.stop_words = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
                "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
                'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 
                'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
                'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
                'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
                'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 
                'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
                'further', 'then', 'once'
            }
        
        # Initialize statistics
        self.stats = {
            "total_analyzed": 0,
            "ai_calls": 0,
            "analysis_by_type": {},
            "avg_quality_score": 0.0,
            "total_quality_sum": 0.0
        }
    
    async def analyze(self, 
                    content: Dict[str, Any], 
                    analysis_type: str = "full") -> Dict[str, Any]:
        """
        Analyze content and extract information.
        
        Args:
            content: Content to analyze
            analysis_type: Type of analysis to perform
                - full: Comprehensive analysis
                - basic: Basic analysis without AI
                - quality: Quality scoring only
                - entities: Entity extraction only
                - structure: Structured data extraction only
            
        Returns:
            Analysis results
        """
        self.stats["total_analyzed"] += 1
        
        # Determine content type
        content_type = content.get("content_type", "generic")
        self.stats["analysis_by_type"][content_type] = self.stats["analysis_by_type"].get(content_type, 0) + 1
        
        # Create result structure
        result = {
            "content_id": self._generate_content_id(content),
            "content_type": content_type,
            "quality_score": 0.0,
            "word_count": 0,
            "sentiment": "neutral",
            "entities": [],
            "topics": [],
            "readability": {
                "score": 0,
                "level": "unknown"
            },
            "structured_data": {},
            "summary": "",
            "key_points": []
        }
        
        # Get main text content depending on content type
        text_content = self._get_text_content(content)
        
        # Basic statistical analysis
        stats = self._analyze_text_stats(text_content)
        result.update(stats)
        
        # Perform basic quality scoring
        quality_score = self._calculate_quality_score(content, stats)
        result["quality_score"] = quality_score
        
        # Update quality statistics
        self.stats["total_quality_sum"] += quality_score
        self.stats["avg_quality_score"] = self.stats["total_quality_sum"] / self.stats["total_analyzed"]
        
        # If only quality scoring is requested, return early
        if analysis_type == "quality":
            return result
        
        # Perform entity extraction and basic content analysis
        if analysis_type in ["full", "entities", "basic"]:
            entities = self._extract_entities(text_content)
            result["entities"] = entities
            
            # Extract topics
            topics = self._classify_topics(text_content, entities)
            result["topics"] = topics
            
            # Basic readability assessment
            readability = self._assess_readability(text_content)
            result["readability"] = readability
            
            # Basic sentiment analysis
            sentiment = self._analyze_sentiment(text_content)
            result["sentiment"] = sentiment
        
        # Perform structured data extraction for specific content types
        if analysis_type in ["full", "structure"]:
            structured_data = self._extract_structured_data(content)
            result["structured_data"] = structured_data
        
        # Enhanced AI-based analysis if enabled
        if self.use_ai and analysis_type == "full":
            ai_analysis = await self._analyze_with_ai(content, text_content)
            
            # Update result with AI analysis
            if ai_analysis:
                result.update(ai_analysis)
                
                # If AI provided better quality score, use it
                if "quality_score" in ai_analysis and ai_analysis["quality_score"] > 0:
                    result["quality_score"] = (result["quality_score"] + ai_analysis["quality_score"]) / 2
        
        return result
    
    def _get_text_content(self, content: Dict[str, Any]) -> str:
        """
        Extract the main text content based on content type.
        
        Args:
            content: Content to analyze
            
        Returns:
            Main text content
        """
        content_type = content.get("content_type", "generic")
        
        if content_type == "article":
            return content.get("content", "")
        elif content_type == "product":
            return content.get("description", "")
        elif content_type == "listing_item":
            return f"{content.get('title', '')} {content.get('description', '')}"
        elif content_type == "data_table":
            # Combine header and data for text analysis
            headers = content.get("headers", [])
            data_rows = content.get("data", [])
            
            if isinstance(data_rows, list) and data_rows and isinstance(data_rows[0], dict):
                # Handle list of dictionaries
                text_parts = []
                for row in data_rows[:10]:  # Limit to first 10 rows
                    text_parts.append(" ".join(str(v) for v in row.values()))
                return " ".join(text_parts)
            return ""
        elif content_type == "listing":
            # Combine titles and descriptions from items
            items = content.get("items", [])
            if items:
                text_parts = []
                for item in items[:10]:  # Limit to first 10 items
                    if isinstance(item, dict):
                        title = item.get("title", "")
                        desc = item.get("description", "")
                        text_parts.append(f"{title} {desc}")
                return " ".join(text_parts)
            return content.get("title", "")
        else:
            # Generic or unknown content type
            return content.get("content", "")
    
    def _analyze_text_stats(self, text: str) -> Dict[str, Any]:
        """
        Analyze basic text statistics.
        
        Args:
            text: Text to analyze
            
        Returns:
            Text statistics
        """
        if not text:
            return {
                "word_count": 0,
                "sentence_count": 0,
                "avg_sentence_length": 0,
                "avg_word_length": 0,
                "unique_words": 0
            }
        
        # Tokenize text using spaCy first, fallback to basic methods
        if SPACY_AVAILABLE and self.nlp:
            doc = self.nlp(text)
            sentences = list(doc.sents)
            words = [token.text.lower() for token in doc if token.is_alpha]
            
            # Filter out punctuation and stop words
            content_words = [token.lemma_.lower() for token in doc 
                           if token.is_alpha and not token.is_stop and len(token.text) > 1]
        else:
            # Fallback to basic tokenization
            sentences = re.split(r'[.!?]+', text.strip())
            sentences = [s.strip() for s in sentences if s.strip()]
            words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
            
            # Filter out stop words
            content_words = [word for word in words if word.isalnum() and word not in self.stop_words]
        
        # Calculate statistics
        word_count = len(content_words)
        sentence_count = len(sentences)
        unique_words = len(set(content_words))
        
        # Avoid division by zero
        avg_sentence_length = word_count / max(1, sentence_count)
        avg_word_length = sum(len(word) for word in content_words) / max(1, word_count)
        
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": round(avg_sentence_length, 1),
            "avg_word_length": round(avg_word_length, 1),
            "unique_words": unique_words
        }
    
    def _calculate_quality_score(self, content: Dict[str, Any], stats: Dict[str, Any]) -> float:
        """
        Calculate a quality score for the content based on various factors.
        
        Args:
            content: Content to analyze
            stats: Text statistics
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        # Start with a base score
        score = 0.5
        
        # Adjust based on word count
        word_count = stats["word_count"]
        if word_count > 1000:
            score += 0.15
        elif word_count > 500:
            score += 0.1
        elif word_count > 200:
            score += 0.05
        elif word_count < 50:
            score -= 0.2
        elif word_count < 100:
            score -= 0.1
        
        # Adjust based on vocabulary richness (unique words ratio)
        if word_count > 0:
            unique_ratio = stats["unique_words"] / word_count
            if unique_ratio > 0.7:
                score += 0.1
            elif unique_ratio > 0.5:
                score += 0.05
            elif unique_ratio < 0.3:
                score -= 0.05
        
        # Adjust based on sentence length
        avg_sentence_length = stats["avg_sentence_length"]
        if 12 <= avg_sentence_length <= 25:
            # Ideal range for readability
            score += 0.05
        elif avg_sentence_length > 40:
            # Very long sentences reduce readability
            score -= 0.1
        
        # Adjust based on content structure and completeness
        content_type = content.get("content_type", "generic")
        
        if content_type == "article":
            if not content.get("title"):
                score -= 0.1
            if content.get("author"):
                score += 0.05
            if content.get("date_published"):
                score += 0.05
                
        elif content_type == "product":
            if not content.get("title"):
                score -= 0.15
            if not content.get("description"):
                score -= 0.15
            if content.get("price"):
                score += 0.1
            if content.get("image_url"):
                score += 0.05
                
        elif content_type == "listing":
            items = content.get("items", [])
            if items:
                score += min(0.2, len(items) / 50)  # Bonus for more items, up to 0.2
                
                # Check quality of first few items
                complete_items = 0
                for item in items[:5]:
                    if isinstance(item, dict) and item.get("title") and item.get("item_url"):
                        complete_items += 1
                
                score += (complete_items / 5) * 0.1
        
        # Bound score between 0.0 and 1.0
        return max(0.0, min(1.0, score))
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of extracted entities
        """
        if not text or len(text) < 10:
            return []
            
        # Basic entity extraction using regex patterns
        entities = []
        
        # Extract emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        for email in emails:
            entities.append({
                "type": "email",
                "value": email,
                "confidence": 0.9
            })
        
        # Extract URLs
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w\.-]*(?:\?\S+)?'
        urls = re.findall(url_pattern, text)
        for url in urls:
            entities.append({
                "type": "url",
                "value": url,
                "confidence": 0.9
            })
        
        # Extract phone numbers
        phone_pattern = r'\b(?:\+\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b'
        phones = re.findall(phone_pattern, text)
        for phone in phones:
            entities.append({
                "type": "phone",
                "value": phone,
                "confidence": 0.8
            })
        
        # Extract dates
        date_patterns = [
            r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',  # 01/01/2020
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b'  # January 1, 2020
        ]
        
        dates = []
        for pattern in date_patterns:
            dates.extend(re.findall(pattern, text, re.IGNORECASE))
            
        for date in dates:
            entities.append({
                "type": "date",
                "value": date,
                "confidence": 0.8
            })
        
        # Extract prices
        price_pattern = r'\$\d+(?:\.\d{2})?|\d+(?:\.\d{2})?\s(?:USD|EUR|GBP)'
        prices = re.findall(price_pattern, text)
        for price in prices:
            entities.append({
                "type": "price",
                "value": price,
                "confidence": 0.9
            })
        
        # Extract percentages
        percentage_pattern = r'\b\d+(?:\.\d+)?%\b'
        percentages = re.findall(percentage_pattern, text)
        for percentage in percentages:
            entities.append({
                "type": "percentage",
                "value": percentage,
                "confidence": 0.9
            })
        
        # Extract organizations (basic heuristic)
        org_indicators = [" Inc", " Corp", " LLC", " Ltd", " Company", " Organization"]
        for indicator in org_indicators:
            pattern = r'\b[A-Z][A-Za-z0-9\'\-]+(?: [A-Z][A-Za-z0-9\'\-]+)*' + indicator + r'\b'
            orgs = re.findall(pattern, text)
            for org in orgs:
                entities.append({
                    "type": "organization",
                    "value": org,
                    "confidence": 0.7
                })
        
        # Add potential locations (basic heuristic)
        location_indicators = [" City", " Street", " Avenue", " Road", " Boulevard", " Lane"]
        for indicator in location_indicators:
            pattern = r'\b[A-Z][A-Za-z0-9\'\-]+(?: [A-Z][A-Za-z0-9\'\-]+)*' + indicator + r'\b'
            locations = re.findall(pattern, text)
            for location in locations:
                entities.append({
                    "type": "location",
                    "value": location,
                    "confidence": 0.6
                })
        
        # Remove duplicates
        seen = set()
        unique_entities = []
        for entity in entities:
            key = f"{entity['type']}:{entity['value']}"
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _classify_topics(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classify the main topics of the content.
        
        Args:
            text: Text to analyze
            entities: Extracted entities
            
        Returns:
            List of topics with confidence scores
        """
        if not text or len(text) < 50:
            return []
            
        # Extract keywords using spaCy first, fallback to basic methods
        if SPACY_AVAILABLE and self.nlp:
            doc = self.nlp(text.lower())
            content_words = [token.lemma_.lower() for token in doc 
                           if token.is_alpha and not token.is_stop and len(token.text) > 3]
        else:
            # Fallback to basic tokenization
            words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
            content_words = [word for word in words if word.isalnum() and len(word) > 3 and word not in self.stop_words]
        
        # Count word frequency
        word_counts = Counter(content_words)
        
        # Get top keywords
        top_keywords = word_counts.most_common(20)
        
        # Define topic categories and related keywords
        topic_categories = {
            "technology": ["computer", "software", "hardware", "tech", "digital", "programming", "code", "app", "data", "internet", "online", "web", "website", "device", "gadget", "mobile", "smartphone", "cloud", "server", "network"],
            "business": ["business", "company", "market", "industry", "corporate", "finance", "investment", "investor", "startup", "entrepreneur", "profit", "revenue", "economic", "economy", "stock", "trade", "commercial", "management", "strategy", "growth"],
            "health": ["health", "medical", "doctor", "hospital", "patient", "disease", "treatment", "medicine", "drug", "therapy", "surgery", "clinical", "diagnosis", "symptom", "wellness", "fitness", "diet", "nutrition", "weight", "exercise"],
            "science": ["science", "research", "scientific", "study", "experiment", "laboratory", "scientist", "theory", "physics", "chemistry", "biology", "genetics", "molecule", "cell", "organism", "discovery", "innovation", "breakthrough", "analysis", "evidence"],
            "education": ["education", "school", "university", "college", "student", "teacher", "professor", "academic", "learning", "teaching", "study", "course", "degree", "curriculum", "training", "knowledge", "skill", "lecture", "classroom", "lesson"],
            "entertainment": ["entertainment", "movie", "film", "music", "song", "artist", "actor", "actress", "celebrity", "cinema", "theater", "concert", "performance", "show", "television", "tv", "series", "episode", "game", "gaming"],
            "sports": ["sport", "team", "player", "game", "match", "tournament", "championship", "league", "football", "soccer", "basketball", "baseball", "hockey", "tennis", "golf", "athlete", "coach", "training", "competition", "victory"],
            "politics": ["politics", "government", "political", "policy", "election", "candidate", "president", "parliament", "congress", "senate", "democratic", "republican", "party", "campaign", "vote", "voter", "law", "legislation", "reform", "regulation"],
            "environment": ["environment", "climate", "environmental", "green", "sustainable", "renewable", "energy", "pollution", "emission", "conservation", "ecosystem", "biodiversity", "nature", "wildlife", "planet", "earth", "forest", "ocean", "water", "recycle"],
            "travel": ["travel", "destination", "tourism", "tourist", "vacation", "holiday", "trip", "journey", "adventure", "explore", "flight", "hotel", "resort", "accommodation", "booking", "tour", "guide", "attraction", "landmark", "sightseeing"],
            "food": ["food", "restaurant", "chef", "cooking", "recipe", "ingredient", "dish", "meal", "cuisine", "menu", "dining", "breakfast", "lunch", "dinner", "flavor", "taste", "culinary", "nutritious", "delicious", "organic"],
            "fashion": ["fashion", "style", "trend", "design", "designer", "clothing", "dress", "wear", "outfit", "accessory", "collection", "brand", "luxury", "model", "beauty", "makeup", "cosmetic", "hair", "skincare", "fragrance"],
            "real_estate": ["property", "real estate", "home", "house", "apartment", "condo", "building", "residential", "commercial", "rent", "lease", "buy", "sell", "agent", "broker", "listing", "mortgage", "loan", "buyer", "seller"],
            "automotive": ["car", "vehicle", "automotive", "auto", "motor", "drive", "driver", "truck", "suv", "sedan", "engine", "wheel", "transmission", "fuel", "electric", "hybrid", "battery", "charging", "manufacturer", "dealer"]
        }
        
        # Calculate topic scores
        topic_scores = {}
        
        # Score based on keywords
        for word, count in top_keywords:
            for topic, keywords in topic_categories.items():
                if word in keywords:
                    topic_scores[topic] = topic_scores.get(topic, 0) + count
        
        # Add scores from entity names
        for entity in entities:
            entity_value = entity["value"].lower()
            for topic, keywords in topic_categories.items():
                for keyword in keywords:
                    if keyword in entity_value:
                        topic_scores[topic] = topic_scores.get(topic, 0) + 2  # Entities have higher weight
        
        # Calculate confidence scores (normalize to 0-1 range)
        total_score = sum(topic_scores.values()) or 1  # Avoid division by zero
        topics = [
            {
                "topic": topic,
                "confidence": min(0.95, score / total_score)
            }
            for topic, score in topic_scores.items()
        ]
        
        # Sort by confidence
        topics.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Return top topics (those with significant confidence)
        return [topic for topic in topics if topic["confidence"] >= 0.1][:5]
    
    def _assess_readability(self, text: str) -> Dict[str, Any]:
        """
        Assess the readability of text content.
        
        Args:
            text: Text to analyze
            
        Returns:
            Readability assessment
        """
        if not text or len(text) < 100:
            return {
                "score": 0,
                "level": "unknown"
            }
            
        # Calculate Flesch Reading Ease score using spaCy first, fallback to basic methods
        if SPACY_AVAILABLE and self.nlp:
            doc = self.nlp(text)
            sentences = list(doc.sents)
            words = [token.text for token in doc if token.is_alpha]
        else:
            # Fallback to basic tokenization
            sentences = re.split(r'[.!?]+', text.strip())
            sentences = [s.strip() for s in sentences if s.strip()]
            words = re.findall(r'\b[a-zA-Z]+\b', text)
        
        word_count = len(words)
        sentence_count = len(sentences)
        
        if word_count == 0 or sentence_count == 0:
            return {
                "score": 0,
                "level": "unknown"
            }
            
        # Count syllables (approximation)
        syllable_count = 0
        for word in words:
            word = word.lower()
            if len(word) <= 3:
                syllable_count += 1
                continue
                
            # Count vowel groups as syllables
            vowels = "aeiouy"
            prev_is_vowel = False
            count = 0
            
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_is_vowel:
                    count += 1
                prev_is_vowel = is_vowel
                
            # Add 1 if word ends with certain letters
            if word.endswith('e'):
                count -= 1
            if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
                count += 1
            if count == 0:
                count = 1
                
            syllable_count += count
        
        # Calculate the Flesch Reading Ease score
        avg_sentence_length = word_count / sentence_count
        avg_syllables_per_word = syllable_count / word_count
        
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Determine reading level
        level = "unknown"
        if score >= 90:
            level = "very_easy"
        elif score >= 80:
            level = "easy"
        elif score >= 70:
            level = "fairly_easy"
        elif score >= 60:
            level = "standard"
        elif score >= 50:
            level = "fairly_difficult"
        elif score >= 30:
            level = "difficult"
        else:
            level = "very_difficult"
            
        return {
            "score": round(score, 1),
            "level": level
        }
    
    def _analyze_sentiment(self, text: str) -> str:
        """
        Perform basic sentiment analysis on text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment label
        """
        if not text or len(text) < 50:
            return "neutral"
            
        # Define sentiment word lists
        positive_words = [
            "good", "great", "excellent", "amazing", "wonderful", "fantastic",
            "outstanding", "terrific", "positive", "awesome", "brilliant",
            "love", "best", "happy", "superb", "perfect", "pleasant", "impressive",
            "favorable", "beneficial", "recommended", "satisfied", "enjoy", "praise"
        ]
        
        negative_words = [
            "bad", "terrible", "poor", "awful", "horrible", "worst",
            "disappointing", "negative", "unfortunate", "mediocre", "failure",
            "hate", "dislike", "unhappy", "angry", "inferior", "useless",
            "broken", "faulty", "defective", "complaint", "issue", "problem",
            "expensive", "overpriced", "difficult", "hard", "impossible"
        ]
        
        # Tokenize and normalize text
        words = text.lower().split()
        
        # Count sentiment words
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        # Determine sentiment based on counts
        if positive_count > negative_count + 2:
            return "positive"
        elif negative_count > positive_count + 2:
            return "negative"
        else:
            return "neutral"
    
    def _extract_structured_data(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured data based on content type.
        
        Args:
            content: Content to analyze
            
        Returns:
            Extracted structured data
        """
        content_type = content.get("content_type", "generic")
        
        if content_type == "article":
            return self._extract_article_data(content)
        elif content_type == "product":
            return self._extract_product_data(content)
        elif content_type == "listing":
            return self._extract_listing_data(content)
        else:
            return {}
    
    def _extract_article_data(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured data from an article.
        
        Args:
            content: Article content
            
        Returns:
            Structured article data
        """
        structured_data = {}
        
        # Extract publication info
        if "author" in content:
            structured_data["author"] = content["author"]
            
        if "date_published" in content:
            structured_data["date_published"] = content["date_published"]
            
        # Extract source info
        if "source_url" in content:
            url = content["source_url"]
            parsed_url = urlparse(url)
            structured_data["domain"] = parsed_url.netloc
            
        # Extract category from tags or categories
        if "categories" in content:
            structured_data["categories"] = content["categories"]
        elif "tags" in content:
            structured_data["categories"] = content["tags"]
            
        # Extract main image
        if "image_url" in content:
            structured_data["main_image"] = content["image_url"]
            
        return structured_data
    
    def _extract_product_data(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured data from a product.
        
        Args:
            content: Product content
            
        Returns:
            Structured product data
        """
        structured_data = {}
        
        # Extract product details
        product_attributes = ["price", "currency", "brand", "availability", "sku"]
        for attr in product_attributes:
            if attr in content and content[attr]:
                structured_data[attr] = content[attr]
                
        # Extract product specifications if available
        if "specifications" in content:
            structured_data["specifications"] = content["specifications"]
            
        # Extract product variants if available
        if "variants" in content:
            structured_data["variants"] = content["variants"]
            
        # Extract related products if available
        if "related_products" in content:
            structured_data["related_products"] = content["related_products"]
            
        return structured_data
    
    def _extract_listing_data(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured data from a listing.
        
        Args:
            content: Listing content
            
        Returns:
            Structured listing data
        """
        structured_data = {}
        
        # Extract item count and pagination info
        if "count" in content:
            structured_data["total_items"] = content["count"]
            
        # Extract price range from items if available
        items = content.get("items", [])
        if items:
            prices = []
            for item in items:
                if isinstance(item, dict) and "price" in item:
                    try:
                        # Extract numeric price value
                        price_str = str(item["price"])
                        price_match = re.search(r'[\d.]+', price_str)
                        if price_match:
                            prices.append(float(price_match.group(0)))
                    except (ValueError, TypeError):
                        pass
                        
            if prices:
                structured_data["price_range"] = {
                    "min": min(prices),
                    "max": max(prices),
                    "avg": sum(prices) / len(prices)
                }
                
        # Extract categories if available
        if "categories" in content:
            structured_data["categories"] = content["categories"]
            
        # Extract filters if available
        if "filters" in content:
            structured_data["filters"] = content["filters"]
            
        return structured_data
    
    async def _analyze_with_ai(self, content: Dict[str, Any], text_content: str) -> Dict[str, Any]:
        """
        Perform enhanced analysis using AI.
        
        Args:
            content: Content to analyze
            text_content: Text content
            
        Returns:
            AI-enhanced analysis results
        """
        if not self.use_ai or not text_content or len(text_content) < 100:
            return {}
            
        self.stats["ai_calls"] += 1
        
        # Truncate text if too long
        if len(text_content) > 8000:
            text_content = text_content[:8000] + "..."
            
        # Prepare context for AI analysis
        content_type = content.get("content_type", "generic")
        title = content.get("title", "")
        source_url = content.get("source_url", "")
        
        try:
            # Build prompt for AI analysis
            prompt = f"""
            Analyze the following {content_type} content and provide structured information.
            
            TITLE: {title}
            SOURCE: {source_url}
            
            CONTENT:
            {text_content}
            
            Provide the following information in JSON format:
            1. "quality_score": A score from 0.0 to 1.0 rating the quality and value of the content
            2. "summary": A brief 1-2 sentence summary of the main content
            3. "key_points": A list of 3-5 key points from the content
            4. "topics": A list of main topics covered, with confidence scores
            5. "sentiment": The overall sentiment (positive, negative, or neutral)
            6. "entities": Important named entities mentioned (people, organizations, products, etc.)
            
            Response format:
            {{
                "quality_score": 0.0-1.0,
                "summary": "...",
                "key_points": ["...", "..."],
                "topics": [
                    {{"topic": "...", "confidence": 0.0-1.0}},
                    ...
                ],
                "sentiment": "positive/negative/neutral",
                "entities": [
                    {{"type": "...", "value": "...", "relevance": 0.0-1.0}},
                    ...
                ]
            }}
            
            Return the JSON object only, with no additional commentary.
            """
            
            # Call AI model
            model = genai.GenerativeModel(self.ai_model)
            response = model.generate_content(prompt)
            response_text = response.text
            
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}')
            
            if json_start != -1 and json_end != -1:
                json_str = response_text[json_start:json_end+1]
                result = json.loads(json_str)
                return result
            
            return {}
            
        except Exception as e:
            logger.error(f"Error in AI analysis: {str(e)}")
            return {}
    
    def _generate_content_id(self, content: Dict[str, Any]) -> str:
        """
        Generate a unique content ID based on content values.
        
        Args:
            content: Content dict
            
        Returns:
            Unique content ID
        """
        # Combine key content values
        content_type = content.get("content_type", "generic")
        title = content.get("title", "")
        url = content.get("source_url", content.get("url", ""))
        
        # Create a string to hash
        hash_str = f"{content_type}:{url}:{title}"
        
        # Generate hash
        hash_obj = hashlib.md5(hash_str.encode())
        return hash_obj.hexdigest()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        return self.stats

class ResultStructureAnalyzer:
    """
    Analyzes result structures to identify patterns, groupings, and relationships.
    
    This class provides functionality to:
    - Identify common attributes across result items
    - Group results by common features
    - Discover hierarchical relationships
    - Standardize irregular result structures
    - Detect and resolve inconsistencies
    """
    
    def __init__(self):
        """Initialize the result structure analyzer."""
        self.debug = False
        self.stats = {
            "sets_analyzed": 0,
            "groups_created": 0,
            "hierarchies_detected": 0,
            "features_identified": 0
        }
    
    async def analyze_result_structure(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the structure of a set of results to identify patterns and relationships.
        
        Args:
            results: List of result items (dictionaries)
            
        Returns:
            Dictionary with structure analysis results
        """
        if not results or not isinstance(results, list):
            return {
                "success": False,
                "error": "Invalid or empty results list",
                "attributes": {},
                "groups": {},
                "structural_consistency": 0.0
            }
            
        # Update stats
        self.stats["sets_analyzed"] += 1
        
        try:
            # Analyze attribute patterns
            attribute_analysis = self._analyze_attributes(results)
            
            # Find natural groupings
            groups = self._find_groupings(results, attribute_analysis)
            
            # Detect hierarchical relationships
            hierarchies = self._detect_hierarchies(results)
            
            # Calculate structure consistency score
            consistency_score = self._calculate_consistency(results, attribute_analysis)
            
            # Suggest standardization approach
            standardization = self._suggest_standardization(results, attribute_analysis)
            
            # Compile results
            analysis_result = {
                "success": True,
                "result_count": len(results),
                "attributes": attribute_analysis,
                "groups": groups,
                "hierarchies": hierarchies,
                "structural_consistency": consistency_score,
                "standardization": standardization
            }
            
            # Add stats
            self.stats["groups_created"] += len(groups)
            self.stats["hierarchies_detected"] += len(hierarchies)
            self.stats["features_identified"] += len(attribute_analysis["common_attributes"])
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing result structure: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "attributes": {},
                "groups": {},
                "structural_consistency": 0.0
            }
    
    def _analyze_attributes(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze attribute patterns across results.
        
        Args:
            results: List of result items
            
        Returns:
            Dictionary with attribute analysis
        """
        # Count items
        item_count = len(results)
        
        # Initialize analysis containers
        all_attributes = set()
        attribute_counts = {}
        attribute_types = {}
        attribute_samples = {}
        attribute_entropies = {}
        
        # Collect all possible attributes
        for item in results:
            for key in item.keys():
                all_attributes.add(key)
                
                # Initialize counts dictionary for this attribute if needed
                if key not in attribute_counts:
                    attribute_counts[key] = 0
                    attribute_types[key] = {}
                    attribute_samples[key] = []
                    
                # Update count
                attribute_counts[key] += 1
                
                # Record type information
                value_type = type(item[key]).__name__
                attribute_types[key][value_type] = attribute_types[key].get(value_type, 0) + 1
                
                # Add sample value (up to 5)
                if len(attribute_samples[key]) < 5 and item[key] is not None:
                    value_str = str(item[key])
                    if len(value_str) > 100:
                        value_str = value_str[:100] + "..."
                    if value_str not in attribute_samples[key]:
                        attribute_samples[key].append(value_str)
                
        # Calculate attribute presence ratios
        attribute_ratios = {attr: count / item_count for attr, count in attribute_counts.items()}
        
        # Classify attributes by presence
        common_attributes = []
        variable_attributes = []
        rare_attributes = []
        
        for attr, ratio in attribute_ratios.items():
            if ratio >= 0.8:
                common_attributes.append(attr)
            elif ratio >= 0.3:
                variable_attributes.append(attr)
            else:
                rare_attributes.append(attr)
                
        # Calculate entropy (variability) for attributes
        for attr in all_attributes:
            # Only calculate for attributes with sufficient presence
            if attribute_ratios[attr] >= 0.3:
                values = []
                valid_items = [item for item in results if attr in item and item[attr] is not None]
                
                if valid_items:
                    values = [str(item[attr]) for item in valid_items]
                    attribute_entropies[attr] = self._calculate_entropy(values)
                else:
                    attribute_entropies[attr] = 0.0
            else:
                attribute_entropies[attr] = 0.0
                
        # Determine primary key candidates
        primary_key_candidates = []
        
        for attr in common_attributes:
            # Check for uniqueness
            values = [str(item.get(attr)) for item in results if attr in item and item[attr] is not None]
            unique_values = set(values)
            
            # If all values are unique, it's a primary key candidate
            if len(values) > 0 and len(unique_values) == len(values):
                primary_key_candidates.append(attr)
                
        # Identify attribute relationships
        attribute_relationships = self._identify_relationships(results, common_attributes + variable_attributes)
                
        return {
            "all_attributes": list(all_attributes),
            "common_attributes": common_attributes,
            "variable_attributes": variable_attributes,
            "rare_attributes": rare_attributes,
            "attribute_presence": attribute_ratios,
            "attribute_types": attribute_types,
            "attribute_samples": attribute_samples,
            "attribute_entropies": attribute_entropies,
            "primary_key_candidates": primary_key_candidates,
            "attribute_relationships": attribute_relationships
        }
    
    def _calculate_entropy(self, values: List[str]) -> float:
        """
        Calculate Shannon entropy to measure variability of values.
        
        Args:
            values: List of string values
            
        Returns:
            Entropy score (higher means more diverse)
        """
        if not values:
            return 0.0
            
        # Count occurrences
        value_counts = Counter(values)
        
        # Calculate probabilities
        total_count = len(values)
        probabilities = [count / total_count for count in value_counts.values()]
        
        # Calculate entropy
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        
        return float(entropy)
    
    def _identify_relationships(self, results: List[Dict[str, Any]], attributes: List[str]) -> List[Dict[str, Any]]:
        """
        Identify relationships between attributes.
        
        Args:
            results: List of result items
            attributes: List of attributes to analyze
            
        Returns:
            List of detected relationships
        """
        relationships = []
        
        # Need at least 5 items for meaningful correlation
        if len(results) < 5:
            return relationships
            
        # For each pair of attributes, check for correlations
        for i, attr1 in enumerate(attributes):
            for attr2 in attributes[i+1:]:
                # Get pairs of values where both attributes exist
                pairs = []
                for item in results:
                    if attr1 in item and attr2 in item and item[attr1] is not None and item[attr2] is not None:
                        # Convert to strings for comparison
                        pairs.append((str(item[attr1]), str(item[attr2])))
                        
                # Skip if not enough pairs
                if len(pairs) < 5:
                    continue
                    
                # Check for 1:1 relationships (same values always appear together)
                value_map = {}
                reverse_map = {}
                
                for val1, val2 in pairs:
                    if val1 not in value_map:
                        value_map[val1] = []
                    if val2 not in reverse_map:
                        reverse_map[val2] = []
                    
                    value_map[val1].append(val2)
                    reverse_map[val2].append(val1)
                
                # Check for 1:1 mapping
                one_to_one = True
                for val, mappings in value_map.items():
                    if len(set(mappings)) > 1:
                        one_to_one = False
                        break
                        
                for val, mappings in reverse_map.items():
                    if len(set(mappings)) > 1:
                        one_to_one = False
                        break
                
                if one_to_one:
                    relationships.append({
                        "type": "one_to_one",
                        "attributes": [attr1, attr2],
                        "description": f"{attr1} and {attr2} have a one-to-one relationship"
                    })
                    continue
                
                # Check for hierarchical relationships (parent-child)
                is_hierarchical = True
                for val, mappings in value_map.items():
                    # If a value maps to multiple other values, it might be a parent
                    if len(set(mappings)) <= 1:
                        is_hierarchical = False
                        break
                
                if is_hierarchical:
                    relationships.append({
                        "type": "hierarchical",
                        "parent": attr1,
                        "child": attr2,
                        "description": f"{attr1} appears to be a parent of {attr2}"
                    })
                    continue
                
                # Check for derived relationships (one calculated from the other)
                # This is a simplified check for potential derivation
                derived = False
                for val1, val2 in pairs[:5]:  # Check first few pairs
                    if val1 in val2 or val2 in val1:
                        derived = True
                        break
                
                if derived:
                    relationships.append({
                        "type": "derived",
                        "attributes": [attr1, attr2],
                        "description": f"{attr1} may be derived from {attr2} or vice versa"
                    })
        
        return relationships
    
    def _find_groupings(self, results: List[Dict[str, Any]], attribute_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find natural groupings in the result set.
        
        Args:
            results: List of result items
            attribute_analysis: Attribute analysis results
            
        Returns:
            Dictionary of identified groups
        """
        groups = {}
        
        # Get common and variable attributes
        common_attrs = attribute_analysis["common_attributes"]
        variable_attrs = attribute_analysis["variable_attributes"]
        
        # Try to find categorical groupings based on variable attributes
        for attr in variable_attrs:
            # Skip attributes with high entropy (too many unique values)
            if attribute_analysis["attribute_entropies"].get(attr, 1.0) > 0.8:
                continue
                
            # Get all values for this attribute
            values = []
            for item in results:
                if attr in item and item[attr] is not None:
                    val = str(item[attr])
                    if val not in values:
                        values.append(val)
            
            # If there are a reasonable number of values, use as grouping
            if 2 <= len(values) <= 10:
                # Create groups
                attr_groups = {}
                for value in values:
                    # Find items matching this value
                    matching_items = [
                        i for i, item in enumerate(results)
                        if attr in item and str(item[attr]) == value
                    ]
                    
                    if matching_items:
                        attr_groups[value] = matching_items
                
                groups[attr] = {
                    "type": "categorical",
                    "values": list(attr_groups.keys()),
                    "groups": attr_groups,
                    "coverage": sum(len(g) for g in attr_groups.values()) / len(results)
                }
        
        # Try to find numeric range groupings
        for attr in common_attrs + variable_attrs:
            # Check if attribute has numeric values
            numeric_values = []
            for item in results:
                if attr in item and item[attr] is not None:
                    try:
                        num_val = float(str(item[attr]).replace('$', '').replace(',', ''))
                        numeric_values.append(num_val)
                    except (ValueError, TypeError):
                        continue
            
            # Only proceed if we have sufficient numeric values
            if len(numeric_values) >= len(results) * 0.5:
                # Create range-based groups
                if numeric_values:
                    min_val = min(numeric_values)
                    max_val = max(numeric_values)
                    
                    # Skip if no range
                    if min_val == max_val:
                        continue
                    
                    # Determine number of buckets (2-5 based on range size)
                    range_size = max_val - min_val
                    num_buckets = min(5, max(2, int(range_size / 10)))
                    
                    # Create buckets
                    bucket_size = range_size / num_buckets
                    buckets = {}
                    
                    for i in range(num_buckets):
                        lower = min_val + i * bucket_size
                        upper = min_val + (i + 1) * bucket_size
                        
                        # Adjust upper bound for last bucket to include max value
                        if i == num_buckets - 1:
                            upper = max_val
                        
                        # Create bucket label
                        if attr.endswith('price') or attr == 'price':
                            bucket_label = f"${lower:.2f} - ${upper:.2f}"
                        else:
                            bucket_label = f"{lower:.1f} - {upper:.1f}"
                            
                        # Find items in this range
                        matching_items = []
                        for idx, item in enumerate(results):
                            if attr in item and item[attr] is not None:
                                try:
                                    val = float(str(item[attr]).replace('$', '').replace(',', ''))
                                    if lower <= val <= upper:
                                        matching_items.append(idx)
                                except (ValueError, TypeError):
                                    continue
                                    
                        if matching_items:
                            buckets[bucket_label] = matching_items
                    
                    if buckets:
                        groups[f"{attr}_range"] = {
                            "type": "numeric_range",
                            "attribute": attr,
                            "ranges": list(buckets.keys()),
                            "groups": buckets,
                            "coverage": sum(len(g) for g in buckets.values()) / len(results)
                        }
        
        # Try to find text-based similarity groups
        for attr in variable_attrs:
            # Skip attributes already used in other groupings
            if attr in groups or f"{attr}_range" in groups:
                continue
                
            # Get text values
            text_values = []
            for item in results:
                if attr in item and item[attr] is not None and isinstance(item[attr], str):
                    text_values.append(str(item[attr]))
            
            # Only proceed if we have sufficient text values
            if len(text_values) >= len(results) * 0.5:
                # Extract keywords from each text value
                item_keywords = []
                for i, text in enumerate(text_values):
                    # Simple keyword extraction (split words, remove common ones)
                    words = re.findall(r'\b\w+\b', text.lower())
                    stop_words = ['a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with']
                    keywords = [w for w in words if len(w) > 3 and w not in stop_words]
                    item_keywords.append((i, keywords))
                
                # Find most common keywords across items
                all_keywords = []
                for _, keywords in item_keywords:
                    all_keywords.extend(keywords)
                
                keyword_counts = Counter(all_keywords)
                top_keywords = [kw for kw, count in keyword_counts.most_common(5) if count >= 3]
                
                # Create keyword-based groups
                if top_keywords:
                    keyword_groups = {}
                    
                    for keyword in top_keywords:
                        # Find items containing this keyword
                        matching_items = []
                        for idx, item in enumerate(results):
                            if attr in item and item[attr] is not None and isinstance(item[attr], str):
                                if keyword in item[attr].lower():
                                    matching_items.append(idx)
                                    
                        if len(matching_items) >= 2:
                            keyword_groups[keyword] = matching_items
                    
                    if keyword_groups:
                        groups[f"{attr}_keywords"] = {
                            "type": "keyword",
                            "attribute": attr,
                            "keywords": list(keyword_groups.keys()),
                            "groups": keyword_groups,
                            "coverage": len(set().union(*keyword_groups.values())) / len(results)
                        }
        
        return groups
    
    def _detect_hierarchies(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect hierarchical relationships in the results.
        
        Args:
            results: List of result items
            
        Returns:
            List of detected hierarchies
        """
        hierarchies = []
        
        # Check for parent-child relationships based on nested structures
        for i, item in enumerate(results):
            for key, value in item.items():
                # Check if value is a dictionary or list of dictionaries
                if isinstance(value, dict) and value:
                    hierarchies.append({
                        "type": "nested_object",
                        "parent_key": key,
                        "child_keys": list(value.keys()),
                        "items": [i],
                        "example": {key: value}
                    })
                elif isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
                    hierarchies.append({
                        "type": "nested_array",
                        "parent_key": key,
                        "child_keys": list(value[0].keys()) if value[0] else [],
                        "items": [i],
                        "example": {key: value[0]} if value[0] else {}
                    })
        
        # Consolidate similar hierarchies
        consolidated = []
        for hierarchy in hierarchies:
            # Check if similar hierarchy already exists
            found = False
            for existing in consolidated:
                if (existing["type"] == hierarchy["type"] and 
                    existing["parent_key"] == hierarchy["parent_key"] and
                    set(existing["child_keys"]) == set(hierarchy["child_keys"])):
                    # Consolidate
                    existing["items"].extend(hierarchy["items"])
                    found = True
                    break
            
            if not found:
                consolidated.append(hierarchy)
        
        return consolidated
    
    def _calculate_consistency(self, results: List[Dict[str, Any]], attribute_analysis: Dict[str, Any]) -> float:
        """
        Calculate structural consistency score for the result set.
        
        Args:
            results: List of result items
            attribute_analysis: Attribute analysis results
            
        Returns:
            Consistency score between 0.0 and 1.0
        """
        # An empty list is perfectly consistent
        if not results:
            return 1.0
            
        # Count items    
        item_count = len(results)
        
        # Get attribute presence ratios
        presence_ratios = attribute_analysis["attribute_presence"]
        
        # Calculate average presence ratio
        if presence_ratios:
            avg_presence = sum(presence_ratios.values()) / len(presence_ratios)
        else:
            avg_presence = 0.0
        
        # Get attribute type consistency
        type_consistency = 0.0
        for attr, type_counts in attribute_analysis["attribute_types"].items():
            # Skip if attribute is rarely present
            if presence_ratios.get(attr, 0) < 0.3:
                continue
                
            # Calculate type consistency for this attribute
            if type_counts:
                # Get most common type and its count
                most_common_type = max(type_counts.items(), key=lambda x: x[1])
                type_ratio = most_common_type[1] / sum(type_counts.values())
                type_consistency += type_ratio
            
        # Average type consistency across attributes
        if attribute_analysis["attribute_types"]:
            type_consistency /= len(attribute_analysis["attribute_types"])
        
        # Calculate structure similarity score
        structure_vectors = []
        for item in results:
            # Create binary vector indicating attribute presence
            vector = []
            for attr in sorted(attribute_analysis["all_attributes"]):
                vector.append(1 if attr in item else 0)
            structure_vectors.append(vector)
        
        # Calculate average cosine similarity between structure vectors
        similarities = []
        if structure_vectors and len(structure_vectors) > 1:
            for i in range(len(structure_vectors)):
                for j in range(i+1, len(structure_vectors)):
                    similarity = self._cosine_similarity(structure_vectors[i], structure_vectors[j])
                    similarities.append(similarity)
            
            avg_structure_similarity = sum(similarities) / len(similarities)
        else:
            avg_structure_similarity = 1.0
        
        # Combine factors into final consistency score
        # Weighted average of attribute presence, type consistency, and structure similarity
        consistency_score = (0.4 * avg_presence + 
                           0.3 * type_consistency + 
                           0.3 * avg_structure_similarity)
        
        return min(1.0, max(0.0, consistency_score))
    
    def _cosine_similarity(self, vector1: List[int], vector2: List[int]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vector1: First binary vector
            vector2: Second binary vector
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Guard against empty vectors
        if not vector1 or not vector2:
            return 0.0
            
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        
        # Calculate magnitudes
        magnitude1 = sum(a * a for a in vector1) ** 0.5
        magnitude2 = sum(b * b for b in vector2) ** 0.5
        
        # Calculate similarity
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        else:
            return dot_product / (magnitude1 * magnitude2)
    
    def _suggest_standardization(self, results: List[Dict[str, Any]], attribute_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggest standardization approach based on result structure.
        
        Args:
            results: List of result items
            attribute_analysis: Attribute analysis results
            
        Returns:
            Standardization suggestions
        """
        standardization = {
            "schema": {},
            "transformations": [],
            "normalization_rules": []
        }
        
        # Create a standardized schema based on common attributes
        for attr in attribute_analysis["common_attributes"]:
            # Determine most common type
            type_counts = attribute_analysis["attribute_types"].get(attr, {})
            if type_counts:
                most_common_type = max(type_counts.items(), key=lambda x: x[1])[0]
                standardization["schema"][attr] = most_common_type
        
        # Add variable attributes that appear frequently
        for attr in attribute_analysis["variable_attributes"]:
            if attribute_analysis["attribute_presence"].get(attr, 0) >= 0.5:
                type_counts = attribute_analysis["attribute_types"].get(attr, {})
                if type_counts:
                    most_common_type = max(type_counts.items(), key=lambda x: x[1])[0]
                    standardization["schema"][attr] = most_common_type
        
        # Suggest transformations for inconsistent data
        for attr, type_counts in attribute_analysis["attribute_types"].items():
            # Skip if attribute is rarely present
            if attribute_analysis["attribute_presence"].get(attr, 0) < 0.3:
                continue
                
            # Check for type inconsistencies
            if len(type_counts) > 1:
                # Multiple types for the same attribute
                primary_type = max(type_counts.items(), key=lambda x: x[1])[0]
                
                standardization["transformations"].append({
                    "attribute": attr,
                    "issue": "type_inconsistency",
                    "primary_type": primary_type,
                    "secondary_types": [t for t in type_counts.keys() if t != primary_type],
                    "transformation": f"convert_to_{primary_type}"
                })
        
        # Suggest normalizations for text fields
        for attr in attribute_analysis["all_attributes"]:
            # Check if attribute appears to be text
            type_counts = attribute_analysis["attribute_types"].get(attr, {})
            is_text = type_counts.get("str", 0) > 0
            
            if is_text:
                # Get samples
                samples = attribute_analysis["attribute_samples"].get(attr, [])
                
                # Check for case inconsistency
                if samples and any(s != s.lower() and s != s.upper() for s in samples):
                    standardization["normalization_rules"].append({
                        "attribute": attr,
                        "issue": "case_inconsistency",
                        "rule": "convert_to_lowercase",
                        "examples": samples[:3]
                    })
                
                # Check for potential trimming needs
                if samples and any(s != s.strip() for s in samples):
                    standardization["normalization_rules"].append({
                        "attribute": attr,
                        "issue": "whitespace_padding",
                        "rule": "trim_whitespace",
                        "examples": samples[:3]
                    })
        
        return standardization
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        return self.stats

class ResultGrouper:
    """
    Groups results based on structural similarity and content relationships.
    
    This class provides functionality to:
    - Group results by common structural features
    - Identify related items within result sets
    - Apply clustering techniques to identify natural groupings
    - Split heterogeneous result sets into coherent subgroups
    """
    
    def __init__(self):
        """Initialize the result grouper."""
        self.structure_analyzer = ResultStructureAnalyzer()
        self.stats = {
            "sets_processed": 0,
            "total_groups_created": 0
        }
    
    async def group_results(self, results: List[Dict[str, Any]], grouping_criteria: str = "auto") -> Dict[str, Any]:
        """
        Group a set of results based on structure and content.
        
        Args:
            results: List of result items
            grouping_criteria: Criteria for grouping ('auto', 'structure', 'content', 'attribute:name')
            
        Returns:
            Dictionary with grouping results
        """
        if not results:
            return {
                "success": False,
                "error": "Empty result set",
                "groups": []
            }
            
        self.stats["sets_processed"] += 1
        
        # Analyze the structure
        structure_analysis = await self.structure_analyzer.analyze_result_structure(results)
        
        if not structure_analysis["success"]:
            return {
                "success": False,
                "error": structure_analysis.get("error", "Structure analysis failed"),
                "groups": []
            }
            
        # If structure is very inconsistent, suggest splitting into multiple result sets
        consistency = structure_analysis["structural_consistency"]
        
        if consistency < 0.5:
            split_groups = self._split_heterogeneous_results(results, structure_analysis)
            
            if len(split_groups) > 1:
                return {
                    "success": True,
                    "message": "Results split due to structural inconsistency",
                    "consistency": consistency,
                    "grouping_type": "split",
                    "group_count": len(split_groups),
                    "groups": split_groups
                }
        
        # Apply grouping based on criteria
        if grouping_criteria == "auto":
            # Choose best grouping method based on data characteristics
            groups = self._auto_group_results(results, structure_analysis)
        elif grouping_criteria == "structure":
            # Group purely by structural similarity
            groups = self._group_by_structure(results, structure_analysis)
        elif grouping_criteria == "content":
            # Group by content similarity
            groups = self._group_by_content(results, structure_analysis)
        elif grouping_criteria.startswith("attribute:"):
            # Group by specific attribute
            attribute = grouping_criteria.split(":", 1)[1]
            groups = self._group_by_attribute(results, attribute)
        else:
            # Default to auto grouping
            groups = self._auto_group_results(results, structure_analysis)
            
        # Update stats
        self.stats["total_groups_created"] += len(groups)
            
        return {
            "success": True,
            "consistency": consistency,
            "grouping_type": grouping_criteria,
            "group_count": len(groups),
            "groups": groups
        }
    
    def _auto_group_results(self, results: List[Dict[str, Any]], structure_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Automatically select the best grouping method based on data characteristics.
        
        Args:
            results: List of result items
            structure_analysis: Structure analysis results
            
        Returns:
            List of result groups
        """
        # Check if there are natural groupings in the structure analysis
        if structure_analysis["groups"]:
            # Find the grouping with the best coverage
            best_grouping = None
            best_coverage = 0
            
            for attr, grouping in structure_analysis["groups"].items():
                coverage = grouping.get("coverage", 0)
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_grouping = grouping
            
            if best_grouping and best_coverage >= 0.7:
                # Use this grouping
                groups = []
                
                for value, item_indices in best_grouping["groups"].items():
                    group_items = [results[idx] for idx in item_indices]
                    
                    groups.append({
                        "name": value,
                        "count": len(group_items),
                        "attribute": best_grouping.get("attribute", ""),
                        "type": best_grouping["type"],
                        "items": group_items
                    })
                
                return groups
        
        # If no good structural grouping, try content-based clustering
        if len(results) > 5:
            content_groups = self._group_by_content(results, structure_analysis)
            
            # Only use content groups if they're meaningful
            if content_groups and max(len(g["items"]) for g in content_groups) < len(results) * 0.8:
                return content_groups
        
        # Fallback to simple structural groups
        return self._group_by_structure(results, structure_analysis)
    
    def _group_by_structure(self, results: List[Dict[str, Any]], structure_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Group results based on structural similarity.
        
        Args:
            results: List of result items
            structure_analysis: Structure analysis results
            
        Returns:
            List of structurally grouped results
        """
        # If very consistent, keep as a single group
        if structure_analysis["structural_consistency"] >= 0.8:
            return [{
                "name": "All Results",
                "count": len(results),
                "type": "structural",
                "attribute": "",
                "items": results
            }]
            
        # Create structure signatures for each item
        signatures = []
        for item in results:
            # Create signature based on present attributes
            sig = frozenset(item.keys())
            signatures.append(sig)
            
        # Group by signature
        signature_groups = {}
        for i, sig in enumerate(signatures):
            sig_key = str(sorted(list(sig)))
            if sig_key not in signature_groups:
                signature_groups[sig_key] = []
            signature_groups[sig_key].append(i)
            
        # Create result groups
        groups = []
        for sig_key, indices in signature_groups.items():
            attributes = eval(sig_key)  # Convert string back to list
            group_items = [results[idx] for idx in indices]
            
            # Create a descriptive name based on key attributes
            if len(attributes) <= 3:
                name = f"Items with {', '.join(attributes)}"
            else:
                name = f"Items with {len(attributes)} attributes"
                
            groups.append({
                "name": name,
                "count": len(group_items),
                "type": "structural",
                "attribute": "",
                "signature": attributes,
                "items": group_items
            })
            
        # Sort groups by size (largest first)
        groups.sort(key=lambda g: g["count"], reverse=True)
        
        return groups
    
    def _group_by_content(self, results: List[Dict[str, Any]], structure_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Group results based on content similarity.
        
        Args:
            results: List of result items
            structure_analysis: Structure analysis results
            
        Returns:
            List of content-similarity grouped results
        """
        # If fewer than 4 items, don't bother clustering
        if len(results) < 4:
            return [{
                "name": "All Results",
                "count": len(results),
                "type": "content",
                "attribute": "",
                "items": results
            }]
            
        # For each item, create a feature vector based on text content
        feature_vectors = []
        for item in results:
            # Combine all text fields
            text_fields = []
            for key, value in item.items():
                if isinstance(value, str):
                    text_fields.append(value)
                elif isinstance(value, (int, float)):
                    text_fields.append(str(value))
                    
            if not text_fields:
                # No text content, use a placeholder
                feature_vectors.append(None)
                continue
                
            # Create a basic bag-of-words vector
            text = " ".join(text_fields).lower()
            words = re.findall(r'\b\w+\b', text)
            
            # Remove common stop words
            stop_words = ['a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with']
            content_words = [w for w in words if len(w) > 3 and w not in stop_words]
            
            if not content_words:
                feature_vectors.append(None)
                continue
                
            # Count word frequencies
            word_counts = Counter(content_words)
            
            # Keep the most frequent words
            feature_vectors.append(word_counts)
            
        # Skip items without feature vectors
        valid_items = [i for i, v in enumerate(feature_vectors) if v is not None]
        
        if len(valid_items) < 4:
            # Not enough valid items for meaningful clustering
            return [{
                "name": "All Results",
                "count": len(results),
                "type": "content",
                "attribute": "",
                "items": results
            }]
            
        # Calculate pairwise similarities
        similarities = {}
        for i in range(len(valid_items)):
            for j in range(i+1, len(valid_items)):
                item_i = valid_items[i]
                item_j = valid_items[j]
                
                # Calculate Jaccard similarity
                vec_i = feature_vectors[item_i]
                vec_j = feature_vectors[item_j]
                
                similarity = self._calculate_text_similarity(vec_i, vec_j)
                similarities[(item_i, item_j)] = similarity
        
        # Simple clustering: group items with similarity above threshold
        groups = []
        remaining = set(valid_items)
        
        while remaining:
            # Start a new group with the first remaining item
            current = next(iter(remaining))
            cluster = {current}
            remaining.remove(current)
            
            # Find similar items
            added = True
            while added and remaining:
                added = False
                
                for item in list(remaining):
                    # Check similarity with all items in the current cluster
                    avg_similarity = 0
                    count = 0
                    
                    for cluster_item in cluster:
                        pair = (min(cluster_item, item), max(cluster_item, item))
                        if pair in similarities:
                            avg_similarity += similarities[pair]
                            count += 1
                            
                    if count > 0:
                        avg_similarity /= count
                        
                        # Add to cluster if similar enough
                        if avg_similarity >= 0.3:  # Similarity threshold
                            cluster.add(item)
                            remaining.remove(item)
                            added = True
            
            # Create a group from this cluster
            cluster_items = [results[idx] for idx in cluster]
            
            # Find common keywords
            cluster_text = ""
            for item in cluster_items:
                for key, value in item.items():
                    if isinstance(value, str):
                        cluster_text += " " + value
                        
            # Extract keywords
            words = re.findall(r'\b\w+\b', cluster_text.lower())
            stop_words = ['a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with']
            content_words = [w for w in words if len(w) > 3 and w not in stop_words]
            
            # Get top keywords
            word_counts = Counter(content_words)
            top_keywords = [word for word, _ in word_counts.most_common(3)]
            
            # Create group name
            if top_keywords:
                name = f"Group: {', '.join(top_keywords)}"
            else:
                name = f"Content Group {len(groups) + 1}"
                
            groups.append({
                "name": name,
                "count": len(cluster_items),
                "type": "content",
                "attribute": "",
                "keywords": top_keywords,
                "items": cluster_items
            })
        
        # Add items with no feature vectors to a special group
        invalid_items = [results[i] for i in range(len(results)) if i not in valid_items]
        if invalid_items:
            groups.append({
                "name": "Other Items",
                "count": len(invalid_items),
                "type": "content",
                "attribute": "",
                "items": invalid_items
            })
            
        # Sort groups by size
        groups.sort(key=lambda g: g["count"], reverse=True)
        
        return groups
    
    def _calculate_text_similarity(self, vec1: Counter, vec2: Counter) -> float:
        """
        Calculate text similarity between two word count vectors.
        
        Args:
            vec1: First word count vector
            vec2: Second word count vector
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Calculate Jaccard similarity
        words1 = set(vec1.keys())
        words2 = set(vec2.keys())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _group_by_attribute(self, results: List[Dict[str, Any]], attribute: str) -> List[Dict[str, Any]]:
        """
        Group results by a specific attribute.
        
        Args:
            results: List of result items
            attribute: Attribute name to group by
            
        Returns:
            List of attribute-based groups
        """
        # Check if attribute exists in results
        if not any(attribute in item for item in results):
            return [{
                "name": "All Results",
                "count": len(results),
                "type": "attribute",
                "attribute": attribute,
                "items": results
            }]
            
        # Group by attribute value
        value_groups = {}
        
        for item in results:
            if attribute in item and item[attribute] is not None:
                value = str(item[attribute])
                if value not in value_groups:
                    value_groups[value] = []
                value_groups[value].append(item)
        
        # Create result groups
        groups = []
        for value, items in value_groups.items():
            groups.append({
                "name": f"{attribute}: {value}",
                "count": len(items),
                "type": "attribute",
                "attribute": attribute,
                "value": value,
                "items": items
            })
        
        # Add items without the attribute to a separate group
        missing_items = [item for item in results if attribute not in item or item[attribute] is None]
        if missing_items:
            groups.append({
                "name": f"No {attribute}",
                "count": len(missing_items),
                "type": "attribute",
                "attribute": attribute,
                "value": None,
                "items": missing_items
            })
            
        # Sort groups by size
        groups.sort(key=lambda g: g["count"], reverse=True)
        
        return groups
    
    def _split_heterogeneous_results(self, results: List[Dict[str, Any]], structure_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a heterogeneous result set into more coherent subgroups.
        
        Args:
            results: List of result items
            structure_analysis: Structure analysis results
            
        Returns:
            List of split result groups
        """
        # Start with structural grouping
        structure_groups = self._group_by_structure(results, structure_analysis)
        
        # If we already have meaningful groups, return them
        if len(structure_groups) > 1 and structure_groups[0]["count"] < len(results) * 0.8:
            return structure_groups
            
        # If we couldn't split well by structure, try content-based grouping
        content_groups = self._group_by_content(results, structure_analysis)
        
        if len(content_groups) > 1 and content_groups[0]["count"] < len(results) * 0.8:
            return content_groups
            
        # If both methods failed to create good groups, try attribute-based split
        # Find attributes with good variability
        for attr in structure_analysis["attributes"]["variable_attributes"]:
            if structure_analysis["attributes"]["attribute_presence"].get(attr, 0) >= 0.5:
                attr_groups = self._group_by_attribute(results, attr)
                
                if len(attr_groups) > 1 and attr_groups[0]["count"] < len(results) * 0.7:
                    return attr_groups
        
        # If all splitting methods failed, return the original set as a single group
        return [{
            "name": "All Results",
            "count": len(results),
            "type": "original",
            "attribute": "",
            "items": results
        }]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get grouper statistics."""
        # Combine with structure analyzer stats
        stats = self.stats.copy()
        stats.update({
            f"structure_analyzer_{k}": v 
            for k, v in self.structure_analyzer.get_stats().items()
        })
        return stats