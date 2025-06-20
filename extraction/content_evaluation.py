"""
Content Evaluation Module

This module provides a centralized ContentEvaluator class that encapsulates
all functionality related to analyzing website content structure and 
determining the best extraction strategies. It also includes specialized 
content type analyzers for different types of web content (e-commerce products, 
news articles, job listings, etc.) to improve extraction accuracy.
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import google.generativeai as genai

import config
from extraction.content_analysis import (
    detect_site_type, find_main_content_area, detect_repeated_patterns,
    generate_selectors_for_site_type, find_selector, extract_keywords,
    content_type_mapping
)


class ContentEvaluator:
    """
    ContentEvaluator class that encapsulates all content analysis functionality.
    It provides methods to analyze webpage structure and content, determine the
    best extraction approaches, identify relevant content elements, and generate
    AI-assisted content filtering.
    """
    
    def __init__(self, cache_dir: Optional[str] = None, ttl_hours: int = 24):
        """
        Initialize the ContentEvaluator with optional caching settings.
        
        Args:
            cache_dir: Directory to store cache files (defaults to .cache in current directory)
            ttl_hours: Time to live for cache entries in hours (defaults to 24 hours)
        """
        self.cache_dir = cache_dir
        self.ttl_hours = ttl_hours
        self._setup_ai()
    
    def _setup_ai(self):
        """Configure the AI model for content evaluation."""
        # Use configuration settings for AI model
        if hasattr(config, 'GEMINI_API_KEY') and config.GEMINI_API_KEY:
            genai.configure(api_key=config.GEMINI_API_KEY)
        
        # Default model for analysis
        self.ai_model = 'gemini-2.0-flash'
    
    async def analyze_site_structure(self, html_content: str, url: str) -> Dict[str, Any]:
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
                "recommended_selectors": []
            }
            
            # Basic checks for site structure
            has_structured_data = bool(soup.find_all(['article', 'product', 'section', 'div']))
            has_lists = bool(soup.find_all(['ul', 'ol', 'dl']))
            has_tables = bool(soup.find_all('table'))
            has_forms = bool(soup.find_all('form'))
            has_pagination = bool(soup.select('[class*="pag"]') or soup.select('[id*="pag"]'))
            
            # Check for JavaScript dependency
            js_tags = soup.find_all('script')
            js_text = ' '.join([tag.text for tag in js_tags if tag.text])
            js_dependent = 'document.getElementById' in js_text or 'createElement' in js_text
            
            if 'react' in js_text.lower() or 'vue' in js_text.lower() or 'angular' in js_text.lower():
                js_dependent = True
                
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
            
            # Set JS dependent flag
            analysis['js_dependent'] = js_dependent
            
            # Find main content area
            main_content = find_main_content_area(soup)
            if main_content:
                analysis['main_content_selector'] = main_content
                
            # Recommend extraction type based on analysis
            if structure_score >= 8 and not js_dependent:
                analysis['recommended_extraction_type'] = 'css'
            elif structure_score >= 6 or js_dependent:
                analysis['recommended_extraction_type'] = 'hybrid'
            else:
                analysis['recommended_extraction_type'] = 'raw'
                
            # Get recommended CSS selectors if appropriate
            if analysis['recommended_extraction_type'] in ['css', 'hybrid']:
                analysis['recommended_selectors'] = generate_selectors_for_site_type(soup, site_type)
                
            # Generate content schema using AI if needed
            if not analysis['recommended_selectors'] and analysis['content_structure'] >= 5:
                ai_selectors = await self.analyze_with_ai(html_content, url, site_type)
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
                "recommended_extraction_type": "raw"
            }
    
    async def analyze_with_ai(self, html_content: str, url: str, site_type: str) -> List[Dict[str, str]]:
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
            model = genai.GenerativeModel(self.ai_model)
            
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
    
    def analyze_page_structure(self, html_content: str) -> Dict[str, Any]:
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
    
    def identify_content_elements(self, html_content: str, content_description: str) -> Dict[str, Any]:
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
                        selector = self._build_selector_for_element(tag)
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
    
    def _build_selector_for_element(self, tag):
        """
        Build a CSS selector for a specific element
        
        Args:
            tag: BeautifulSoup tag object
            
        Returns:
            CSS selector string
        """
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
    
    async def generate_content_filter_instructions(self, user_prompt: str) -> str:
        """
        Generate content filtering instructions based on user prompt.
        
        Args:
            user_prompt: The user's original extraction prompt
            
        Returns:
            Content filtering instructions
        """
        try:
            model = genai.GenerativeModel(self.ai_model)
            
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
    
    async def evaluate_content_relevance(self, text_content: str, user_prompt: str) -> Dict[str, Any]:
        """
        Evaluate the relevance of extracted content to the user's prompt.
        
        Args:
            text_content: Extracted text content to evaluate
            user_prompt: The user's original extraction prompt
            
        Returns:
            Dictionary with relevance evaluation
        """
        try:
            model = genai.GenerativeModel(self.ai_model)
            
            # Keep the text content to a reasonable length
            content_sample = text_content[:5000] if len(text_content) > 5000 else text_content
            
            instruction = f"""
            Evaluate if the following content matches what the user requested:
            
            User request: "{user_prompt}"
            
            Content sample: 
            {content_sample}
            
            Return ONLY a JSON object with these properties:
            - relevance_score: number from 0-1 indicating relevance (0 = not relevant, 1 = perfect match)
            - key_info_present: boolean indicating if key information is present
            - missing_elements: array of strings describing what information is missing (if any)
            - noise_ratio: number from 0-1 indicating how much irrelevant content is present
            """
            
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
                evaluation = json.loads(json_text)
                return evaluation
            except json.JSONDecodeError:
                print("Failed to parse AI-generated evaluation as JSON")
                return {
                    "relevance_score": 0.5,  # Neutral score on failure
                    "key_info_present": False,
                    "missing_elements": ["Failed to evaluate content"],
                    "noise_ratio": 0.5
                }
        
        except Exception as e:
            print(f"Error in content evaluation: {str(e)}")
            return {
                "relevance_score": 0.5,
                "key_info_present": False,
                "missing_elements": [f"Evaluation error: {str(e)}"],
                "noise_ratio": 0.5
            }
    
    async def suggest_extraction_improvements(self, 
                                            html_content: str, 
                                            extracted_data: Dict[str, Any], 
                                            user_prompt: str) -> Dict[str, Any]:
        """
        Analyze extraction results and suggest improvements to the extraction strategy.
        
        Args:
            html_content: Original HTML content
            extracted_data: Current extraction results
            user_prompt: The user's original extraction prompt
            
        Returns:
            Dictionary with improvement suggestions
        """
        try:
            model = genai.GenerativeModel(self.ai_model)
            
            # Sample of HTML and extraction results to avoid token limits
            html_sample = html_content[:3000] if len(html_content) > 3000 else html_content
            extraction_sample = json.dumps(extracted_data)[:2000] if len(json.dumps(extracted_data)) > 2000 else json.dumps(extracted_data)
            
            instruction = f"""
            Analyze the extraction results and suggest improvements:
            
            User request: "{user_prompt}"
            
            Extraction results: 
            {extraction_sample}
            
            HTML sample:
            {html_sample}
            
            Return ONLY a JSON object with these properties:
            - quality_assessment: string description of extraction quality
            - data_completeness: number from 0-1 indicating completeness
            - suggested_improvements: array of improvement suggestions
            - additional_fields: array of any additional fields that should be extracted
            - improved_selectors: object with improved CSS selectors (if applicable)
            """
            
            response = model.generate_content(instruction)
            response_text = response.text
            
            # Extract JSON
            if "```json" in response_text:
                json_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_text = response_text.split("```")[1].strip()
            else:
                json_text = response_text
            
            # Parse the JSON
            try:
                suggestions = json.loads(json_text)
                return suggestions
            except json.JSONDecodeError:
                print("Failed to parse AI-generated suggestions as JSON")
                return {
                    "quality_assessment": "Unable to assess extraction quality",
                    "data_completeness": 0.5,
                    "suggested_improvements": ["Try different extraction method"],
                    "additional_fields": [],
                    "improved_selectors": {}
                }
        
        except Exception as e:
            print(f"Error generating improvement suggestions: {str(e)}")
            return {
                "quality_assessment": f"Error during analysis: {str(e)}",
                "data_completeness": 0,
                "suggested_improvements": ["Check extraction code for errors"],
                "additional_fields": [],
                "improved_selectors": {}
            }


class ContentTypeAnalyzer:
    """Base class for content type-specific analyzers"""
    
    def __init__(self):
        self.confidence_score = 0.0
    
    def analyze(self, soup: BeautifulSoup, url: str = None) -> Dict[str, Any]:
        """
        Analyze content and determine if it matches this content type
        
        Args:
            soup: BeautifulSoup object of the page content
            url: URL of the page (optional)
            
        Returns:
            Dictionary with analysis results
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_confidence_score(self) -> float:
        """
        Returns confidence level (0.0-1.0) that content matches this type
        """
        return self.confidence_score
    
    def get_recommended_selectors(self) -> Dict[str, str]:
        """
        Returns recommended CSS selectors for this content type
        """
        raise NotImplementedError("Subclasses must implement this method")


class EcommerceProductAnalyzer(ContentTypeAnalyzer):
    """Analyzer for e-commerce product pages"""
    
    def analyze(self, soup: BeautifulSoup, url: str = None) -> Dict[str, Any]:
        """Analyze if content is an e-commerce product page"""
        self.confidence_score = 0.0
        evidence = []
        
        # Check product indicators in URL
        if url:
            url_indicators = ['product', 'item', 'p/', 'pd/', 'details', '-p-']
            if any(indicator in url.lower() for indicator in url_indicators):
                evidence.append("URL pattern matches product page")
                self.confidence_score += 0.15
        
        # Check for product schema markup
        schema_script = soup.find('script', {'type': 'application/ld+json'})
        if schema_script and schema_script.string:
            try:
                schema_data = json.loads(schema_script.string)
                if '@type' in schema_data and schema_data['@type'] in ['Product', 'IndividualProduct', 'SomeProducts']:
                    evidence.append("Product schema markup found")
                    self.confidence_score += 0.3
            except:
                pass
        
        # Check for product elements
        price_elements = soup.select('[class*="price"], [id*="price"], .price, #price, [itemprop="price"]')
        if price_elements:
            evidence.append(f"Price elements found: {len(price_elements)}")
            self.confidence_score += 0.15
            
        add_to_cart_btn = soup.select('[id*="add-to-cart"], [class*="add-to-cart"], [id*="buy-now"], [class*="buy-now"]')
        if add_to_cart_btn:
            evidence.append("Add to cart/Buy now button found")
            self.confidence_score += 0.2
            
        product_gallery = soup.select('[class*="gallery"], [id*="gallery"], [class*="carousel"], [id*="carousel"]')
        if product_gallery:
            evidence.append("Product gallery/carousel found")
            self.confidence_score += 0.1
            
        # Check for product variants
        variants = soup.select('[class*="variant"], [id*="variant"], [class*="option"], [id*="option"]')
        if variants:
            evidence.append(f"Product variants/options found: {len(variants)}")
            self.confidence_score += 0.1
            
        # Cap the confidence score at 1.0
        self.confidence_score = min(1.0, self.confidence_score)
        
        return {
            "content_type": "ecommerce_product",
            "confidence": self.confidence_score,
            "evidence": evidence,
            "recommended_selectors": self.get_recommended_selectors() if self.confidence_score > 0.5 else {}
        }
    
    def get_recommended_selectors(self) -> Dict[str, str]:
        """Returns recommended CSS selectors for product pages"""
        return {
            "title": "h1, .product-title, #product-title, [itemprop='name']",
            "price": ".price, #price, [itemprop='price'], .product-price, #product-price",
            "description": ".product-description, #product-description, [itemprop='description']",
            "images": ".product-image, #product-image, [itemprop='image'], .carousel img, .gallery img",
            "sku": "[itemprop='sku'], .sku, #sku, .product-sku, #product-sku",
            "rating": ".rating, #rating, [itemprop='ratingValue'], .stars, .product-rating",
            "availability": "[itemprop='availability'], .availability, #availability, .stock-status",
            "variants": ".variant, #variant, .option, #option, .product-variant, .product-option"
        }


class NewsArticleAnalyzer(ContentTypeAnalyzer):
    """Analyzer for news article pages"""
    
    def analyze(self, soup: BeautifulSoup, url: str = None) -> Dict[str, Any]:
        """Analyze if content is a news article"""
        self.confidence_score = 0.0
        evidence = []
        
        # Check URL patterns
        if url:
            url_indicators = ['/article/', '/news/', '/story/', '/post/', '/blog/']
            if any(indicator in url.lower() for indicator in url_indicators):
                evidence.append("URL pattern matches news article")
                self.confidence_score += 0.15
                
        # Check for article schema markup
        schema_script = soup.find('script', {'type': 'application/ld+json'})
        if schema_script and schema_script.string:
            try:
                schema_data = json.loads(schema_script.string)
                if '@type' in schema_data and schema_data['@type'] in ['Article', 'NewsArticle', 'BlogPosting']:
                    evidence.append("Article schema markup found")
                    self.confidence_score += 0.25
            except:
                pass
        
        # Check for article elements
        article_tag = soup.find('article')
        if article_tag:
            evidence.append("Article tag found")
            self.confidence_score += 0.15
            
        date_elements = soup.select('[itemprop="datePublished"], [class*="date"], [id*="date"], .published, .posted, time')
        if date_elements:
            evidence.append("Publication date element found")
            self.confidence_score += 0.1
            
        author_elements = soup.select('[itemprop="author"], [class*="author"], [id*="author"], .byline')
        if author_elements:
            evidence.append("Author element found")
            self.confidence_score += 0.1
            
        # Check for article structural elements
        h1_tags = soup.find_all('h1')
        if len(h1_tags) == 1:
            evidence.append("Single H1 headline found")
            self.confidence_score += 0.1
            
        paragraphs = soup.select('article p') or soup.select('.article p') or soup.select('.content p')
        if len(paragraphs) > 3:
            evidence.append(f"Article body found with {len(paragraphs)} paragraphs")
            self.confidence_score += 0.15
            
        # Cap the confidence score at 1.0
        self.confidence_score = min(1.0, self.confidence_score)
        
        return {
            "content_type": "news_article",
            "confidence": self.confidence_score,
            "evidence": evidence,
            "recommended_selectors": self.get_recommended_selectors() if self.confidence_score > 0.5 else {}
        }
    
    def get_recommended_selectors(self) -> Dict[str, str]:
        """Returns recommended CSS selectors for news articles"""
        return {
            "headline": "h1, .headline, .title, [itemprop='headline']",
            "author": "[itemprop='author'], .author, .byline",
            "date_published": "[itemprop='datePublished'], .date, .published, time", 
            "content": "article, .article-body, .content, [itemprop='articleBody']",
            "summary": ".summary, .excerpt, [itemprop='description'], .standfirst",
            "category": "[itemprop='articleSection'], .category",
            "image": "[itemprop='image'], .featured-image, .article-image",
            "tags": ".tags, .keywords, [itemprop='keywords']"
        }


class ListingPageAnalyzer(ContentTypeAnalyzer):
    """Analyzer for pages with multiple product/content listings"""
    
    def analyze(self, soup: BeautifulSoup, url: str = None) -> Dict[str, Any]:
        """Analyze if content is a listing/search results page"""
        self.confidence_score = 0.0
        evidence = []
        
        # Check URL patterns
        if url:
            url_indicators = ['/search', '/catalog', '/category', '/collection', '/list', '/results']
            if any(indicator in url.lower() for indicator in url.indicators):
                evidence.append("URL pattern matches listing page")
                self.confidence_score += 0.15
                
        # Check for sorting/filtering elements
        sort_filters = soup.select('[id*="sort"], [class*="sort"], [id*="filter"], [class*="filter"]')
        if sort_filters:
            evidence.append(f"Sorting/filtering controls found: {len(sort_filters)}")
            self.confidence_score += 0.15
            
        # Check for pagination
        pagination = soup.select('[class*="pag"], [id*="pag"], .pages, .page-nav')
        if pagination:
            evidence.append("Pagination controls found")
            self.confidence_score += 0.15
            
        # Check for common listing page elements
        grid_elements = soup.select('.grid, .products, .listings, .cards, [class*="grid"], [class*="list"]')
        if grid_elements:
            evidence.append("Grid/list container found")
            self.confidence_score += 0.1
            
        # Look for repeated similar elements (likely product cards)
        repeated_elements = self._find_repeated_elements(soup)
        if repeated_elements['count'] >= 3:
            evidence.append(f"Found {repeated_elements['count']} repeated similar elements")
            self.confidence_score += 0.3
            
        # Check for search results indicators
        search_headers = soup.select('[class*="search-result"], [id*="search-result"], [class*="results"], [id*="results"]')
        if search_headers:
            evidence.append("Search results indicators found")
            self.confidence_score += 0.15
            
        # Cap the confidence score at 1.0
        self.confidence_score = min(1.0, self.confidence_score)
        
        return {
            "content_type": "listing_page",
            "confidence": self.confidence_score,
            "evidence": evidence,
            "item_selector": repeated_elements['selector'] if repeated_elements['count'] >= 3 else "",
            "recommended_selectors": self.get_recommended_selectors(repeated_elements['selector']) if self.confidence_score > 0.5 else {}
        }
    
    def _find_repeated_elements(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Find repeated similar elements that could be product/content cards"""
        candidates = [
            ('div[class*="product"]', 'product class'),
            ('div[class*="item"]', 'item class'),
            ('div[class*="card"]', 'card class'),
            ('div[class*="result"]', 'result class'),
            ('article', 'article tag'),
            ('li.product', 'product list item'),
            ('.products > li', 'products list item'),
            ('.items > div', 'items div')
        ]
        
        best_match = {"selector": "", "count": 0}
        
        for selector, desc in candidates:
            elements = soup.select(selector)
            if len(elements) > best_match['count']:
                best_match = {"selector": selector, "count": len(elements)}
                
        return best_match
    
    def get_recommended_selectors(self, item_selector: str = "") -> Dict[str, str]:
        """Returns recommended CSS selectors for listing pages"""
        base_selector = item_selector if item_selector else ".item, .product, .card, .result"
        return {
            "item_container": base_selector,
            "title": f"{base_selector} h2, {base_selector} h3, {base_selector} .title",
            "price": f"{base_selector} .price, {base_selector} [class*='price']",
            "image": f"{base_selector} img",
            "link": f"{base_selector} a, {base_selector} > a",
            "description": f"{base_selector} .description, {base_selector} [class*='desc']",
            "rating": f"{base_selector} .rating, {base_selector} .stars"
        }


class JobListingAnalyzer(ContentTypeAnalyzer):
    """Analyzer for job listing pages"""
    
    def analyze(self, soup: BeautifulSoup, url: str = None) -> Dict[str, Any]:
        """Analyze if content is a job listing"""
        self.confidence_score = 0.0
        evidence = []
        
        # Check URL patterns
        if url:
            url_indicators = ['/job/', '/career', '/position', '/vacancy', '/employment']
            if any(indicator in url.lower() for indicator in url.indicators):
                evidence.append("URL pattern matches job listing")
                self.confidence_score += 0.15
        
        # Check for job posting schema
        schema_script = soup.find('script', {'type': 'application/ld+json'})
        if schema_script and schema_script.string:
            try:
                schema_data = json.loads(schema_script.string)
                if '@type' in schema_data and schema_data['@type'] in ['JobPosting', 'JobListing']:
                    evidence.append("Job posting schema markup found")
                    self.confidence_score += 0.25
            except:
                pass
        
        # Check for job-specific terminology
        page_text = soup.get_text().lower()
        job_terms = ['salary', 'apply now', 'job description', 'responsibilities', 
                     'qualifications', 'experience required', 'full-time', 'part-time',
                     'position', 'remote', 'hybrid', 'on-site', 'benefits']
        
        matching_terms = [term for term in job_terms if term in page_text]
        if matching_terms:
            evidence.append(f"Job terminology found: {', '.join(matching_terms)}")
            self.confidence_score += min(0.3, len(matching_terms) * 0.05)
        
        # Check for job application elements
        apply_buttons = soup.select('a[href*="apply"], button:contains("Apply")')
        if apply_buttons:
            evidence.append("Apply button found")
            self.confidence_score += 0.15
        
        # Check for company and job title structure
        company_elements = soup.select('[itemprop="hiringOrganization"], [class*="company"], [id*="company"]')
        if company_elements:
            evidence.append("Company information element found")
            self.confidence_score += 0.1
        
        # Check for job details sections
        details_sections = soup.select('[class*="details"], [id*="details"], [class*="description"], [id*="description"]')
        if details_sections:
            evidence.append("Job details/description section found")
            self.confidence_score += 0.1
        
        # Cap the confidence score at 1.0
        self.confidence_score = min(1.0, self.confidence_score)
        
        return {
            "content_type": "job_listing",
            "confidence": self.confidence_score,
            "evidence": evidence,
            "recommended_selectors": self.get_recommended_selectors() if self.confidence_score > 0.5 else {}
        }
    
    def get_recommended_selectors(self) -> Dict[str, str]:
        """Returns recommended CSS selectors for job listings"""
        return {
            "job_title": "h1, .job-title, #job-title, [itemprop='title']",
            "company": "[itemprop='hiringOrganization'], .company, #company",
            "location": "[itemprop='jobLocation'], .location, #location, [class*='location']",
            "salary": "[itemprop='baseSalary'], .salary, #salary, [class*='salary']",
            "description": "[itemprop='description'], .description, #description, .job-description",
            "requirements": ".requirements, #requirements, [class*='qualifications'], [class*='requirements']",
            "responsibilities": ".responsibilities, #responsibilities, [class*='responsibilities']",
            "apply_button": "a[href*='apply'], button:contains('Apply'), .apply-button, #apply-button"
        }


class ContentTypeFactory:
    """Factory for creating content type analyzers"""
    
    @staticmethod
    def create_analyzers() -> List[ContentTypeAnalyzer]:
        """Create all available content type analyzers"""
        return [
            EcommerceProductAnalyzer(),
            NewsArticleAnalyzer(),
            ListingPageAnalyzer(),
            JobListingAnalyzer()
        ]
    
    @staticmethod
    def analyze_content(soup: BeautifulSoup, url: str = None) -> Dict[str, Any]:
        """
        Analyze content with all available analyzers and return the best match
        
        Args:
            soup: BeautifulSoup object of the page content
            url: URL of the page (optional)
            
        Returns:
            Dictionary with the best matching content type analysis
        """
        analyzers = ContentTypeFactory.create_analyzers()
        results = []
        
        for analyzer in analyzers:
            result = analyzer.analyze(soup, url)
            results.append(result)
        
        # Sort by confidence score, highest first
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Return the best match if it meets minimum confidence threshold
        if results and results[0]['confidence'] >= 0.5:
            return {
                "detected_type": results[0]['content_type'],
                "confidence": results[0]['confidence'],
                "evidence": results[0]['evidence'],
                "recommended_selectors": results[0].get('recommended_selectors', {}),
                "item_selector": results[0].get('item_selector', ""),
                "all_results": results
            }
        else:
            return {
                "detected_type": "unknown",
                "confidence": results[0]['confidence'] if results else 0.0,
                "all_results": results
            }