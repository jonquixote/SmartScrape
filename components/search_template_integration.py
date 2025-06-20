"""
Search Template Integration module

This module integrates search functionality with template extraction to enable
more effective and context-aware data extraction from websites.
"""

import logging
import json
import re
import asyncio
import time
from urllib.parse import urlparse, urljoin
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None
    logging.warning("BeautifulSoup not available - template extraction functionality will be limited")

from components.search_automation import SearchAutomator
from components.template_storage import TemplateStorage
from components.domain_intelligence import DomainIntelligence
from components.pagination_handler import PaginationHandler
# Fix import error by importing functions instead of non-existent class
from extraction.content_analysis import analyze_site_structure, analyze_page_structure, identify_content_elements
# Fix the import error by using the actual functions that exist
from ai_helpers.prompt_generator import optimize_extraction_prompt, generate_content_filter_instructions
# Replace the non-existent parse_ai_response with actual functions
from ai_helpers.response_parser import extract_json_from_response, parse_json_safely, clean_extraction_output
from extraction.content_extraction import extract_content_with_ai
# Fix import error: use the correct function name
from extraction.fallback_extraction import perform_extraction_with_fallback

# Import crawler class
try:
    from crawl4ai import AsyncWebCrawler
except ImportError:
    # Define a minimal compatibility class if not available
    class AsyncWebCrawler:
        async def crawl(self, url):
            logging.warning("AsyncWebCrawler is not properly imported, using stub implementation")
            return {"success": False, "error": "WebCrawler not available"}
            
        async def arun(self, url):
            logging.warning("AsyncWebCrawler is not properly imported, using stub implementation")
            return type('obj', (object,), {'success': False, 'error': "WebCrawler not available"})

        async def afetch(self, url):
            logging.warning("AsyncWebCrawler is not properly imported, using stub implementation")
            return type('obj', (object,), {'success': False, 'error': "WebCrawler not available"})

class SearchTemplateIntegrator:
    """
    Integrates search functionality with template extraction to enable
    more effective and context-aware data extraction from websites.
    
    This class bridges the gap between search capabilities and template extraction,
    allowing for more dynamic and accurate data extraction based on user intent.
    """
    
    def __init__(self, crawler=None, template_path="templates"):
        """
        Initialize the SearchTemplateIntegrator.
        
        Args:
            crawler: The web crawler to use for fetching web pages
            template_path: The path to the template storage directory
        """
        self.crawler = crawler if crawler else AsyncWebCrawler()
        self.search_automator = SearchAutomator(config={"crawler": self.crawler})
        self.template_storage = TemplateStorage(template_path)
        # Using content analysis functions directly instead of non-existent class
        self.domain_intelligence = DomainIntelligence()
        self.pagination_handler = PaginationHandler()  # Initialize pagination handler
        self.extraction_success_metrics = {}  # Track extraction success metrics by template ID
        self.domain_specific_optimizations = {}  # Store domain-specific optimizations
        self.ai_extraction_enabled = True  # Enable AI-guided extraction by default
        self.fallback_extraction_threshold = 0.3  # Threshold to trigger fallback extraction
        
    async def process_search_and_extract(self, url: str, user_intent: Dict[str, Any], 
                                       search_params: Dict[str, Any] = None, 
                                       extraction_requirements: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a search and extraction request based on user intent.
        
        Args:
            url: The base URL of the website to search
            user_intent: Dictionary containing the user's intent information
                - original_query: The original user query
                - target_information: Information about what data to extract
                - entity_types: Types of entities to extract (e.g., product, person, etc.)
                - properties: Properties to extract for each entity
            search_params: Optional search parameters generated from user intent
            extraction_requirements: Optional extraction requirements for the data
                
        Returns:
            Dict containing the extraction results
        """
        try:
            
            # Use provided search_params if available, otherwise extract from user intent
            if search_params:
                search_terms = search_params
                logging.info(f"Using provided search params: {search_terms}")
            else:
                # Extract search terms from user intent
                search_terms = self._extract_search_terms(user_intent)
                logging.info(f"Extracted search terms: {search_terms}")
            
            # Apply domain-specific optimizations to search terms
            search_terms = self.apply_domain_specific_optimizations(search_terms, url)
            
            # Use provided extraction_requirements if available
            if extraction_requirements:
                logging.info(f"Using provided extraction requirements: {extraction_requirements}")
                # Store extraction requirements for use in template selection and creation
                user_intent["extraction_requirements"] = extraction_requirements
            
            # Find and analyze search forms
            search_form_info = await self.search_automator.detect_search_forms(url)
            
            if not search_form_info or "forms" not in search_form_info or not search_form_info["forms"]:
                logging.warning(f"No search forms found on {url}")
                return {
                    "success": False,
                    "error": "No search forms found on the website",
                    "url": url
                }
                
            # Choose the most appropriate search form
            search_form = self._select_best_search_form(search_form_info["forms"], search_terms, user_intent)
            
            if not search_form:
                logging.warning(f"No suitable search form found on {url}")
                return {
                    "success": False,
                    "error": "No suitable search form found on the website",
                    "url": url
                }
                
            # Execute search
            search_results = await self.search_automator.execute_search(
                url,
                search_form,
                search_terms
            )
            
            if not search_results or not search_results.get("success"):
                logging.warning(f"Search failed: {search_results.get('error', 'Unknown error')}")
                return {
                    "success": False,
                    "error": f"Search failed: {search_results.get('error', 'Unknown error')}",
                    "url": url
                }
                
            # Extract data from the search results
            extraction_results = await self.extract_from_search_results(
                search_results["result_url"],
                search_results["content"],
                user_intent
            )

            # Check for pagination and process additional pages if needed
            all_results = extraction_results.get("entities", [])
            paginated_results = []
            
            # Process pagination if needed and user intent suggests comprehensive results
            if extraction_results.get("success") and self._should_process_pagination(user_intent):
                paginated_results = await self._process_paginated_results(
                    search_results["result_url"],
                    search_results["content"],
                    user_intent,
                    extraction_results.get("template_id")
                )
                
            # Merge results from all pages
            if paginated_results:
                all_results.extend(paginated_results)
                
            # Deduplicate results based on unique identifier (URL or title)
            deduplicated_results = self._deduplicate_results(all_results)
                
            # Update the extraction results with the combined results
            extraction_results["entities"] = deduplicated_results
            extraction_results["total_results"] = len(deduplicated_results)
            extraction_results["paginated"] = True if paginated_results else False
            
            # Enhanced: Apply post-processing for improved data quality
            enhanced_results = await self._apply_post_processing(deduplicated_results, user_intent)
            extraction_results["entities"] = enhanced_results
            
            return extraction_results
            
        except Exception as e:
            logging.exception(f"Error in process_search_and_extract: {str(e)}")
            return {
                "success": False,
                "error": f"Error processing search: {str(e)}",
                "url": url
            }
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate results based on URL or title.
        
        Args:
            results: List of extracted entity dictionaries
            
        Returns:
            Deduplicated list of entities
        """
        unique_results = {}
        
        for item in results:
            # Use URL as primary identifier if available, otherwise use title
            identifier = item.get("url", "") or item.get("title", "")
            if identifier and identifier not in unique_results:
                unique_results[identifier] = item
                
        return list(unique_results.values())
        
    async def _apply_post_processing(self, results: List[Dict[str, Any]], user_intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Apply post-processing techniques to improve data quality.
        
        Args:
            results: List of extracted entity dictionaries
            user_intent: Dictionary containing the user's intent information
            
        Returns:
            List of enhanced entity dictionaries
        """
        enhanced_results = []
        
        for item in results:
            # Normalize field names and values
            normalized_item = self._normalize_item_fields(item)
            
            # Extract missing fields using AI if needed
            if self.ai_extraction_enabled and self._needs_ai_enhancement(normalized_item, user_intent):
                try:
                    # Get the item details page URL if available
                    detail_url = normalized_item.get("url")
                    if detail_url:
                        # Fetch the detail page
                        detail_page = await self.crawler.afetch(detail_url)
                        if detail_page and hasattr(detail_page, 'html'):
                            # Use AI-guided extraction to enhance the item data
                            enhanced_data = await extract_content_with_ai(
                                detail_page.html,
                                detail_url,
                                user_intent
                            )
                            
                            if enhanced_data and isinstance(enhanced_data, dict):
                                # Merge enhanced data with original data, keeping original values
                                # for fields that already had data
                                for key, value in enhanced_data.items():
                                    if key not in normalized_item or not normalized_item[key]:
                                        normalized_item[key] = value
                except Exception as e:
                    logging.warning(f"Error enhancing item with AI: {str(e)}")
            
            # Apply data type conversions (e.g., strings to numbers)
            processed_item = self._apply_data_type_conversions(normalized_item)
            
            # Add confidence score for extraction quality
            processed_item["_extraction_confidence"] = self._calculate_extraction_confidence(processed_item, user_intent)
            
            enhanced_results.append(processed_item)
            
        # Sort by confidence if we have scores
        if enhanced_results and "_extraction_confidence" in enhanced_results[0]:
            enhanced_results.sort(key=lambda x: x.get("_extraction_confidence", 0), reverse=True)
            
        return enhanced_results
    
    def _normalize_item_fields(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize field names and clean values in an extracted item.
        
        Args:
            item: Dictionary containing extracted data for an item
            
        Returns:
            Dictionary with normalized field names and values
        """
        normalized = {}
        
        # Field name normalization map
        name_map = {
            "product_title": "title",
            "name": "title",
            "product_name": "title",
            "item_title": "title",
            "product_price": "price",
            "item_price": "price",
            "product_image": "image",
            "item_image": "image",
            "product_url": "url",
            "item_url": "url",
            "product_description": "description",
            "item_description": "description",
            "product_rating": "rating",
            "item_rating": "rating"
        }
        
        # Copy the item while normalizing field names
        for key, value in item.items():
            # Normalize the key
            normalized_key = name_map.get(key.lower(), key.lower())
            
            # Clean and normalize the value
            if isinstance(value, str):
                # Remove excessive whitespace
                cleaned_value = re.sub(r'\s+', ' ', value).strip()
                
                # Specific cleaning for known fields
                if normalized_key == "price":
                    # Extract numeric price value with currency symbol
                    price_match = re.search(r'(\$|€|£|\¥)?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)', cleaned_value)
                    if price_match:
                        currency = price_match.group(1) or '$'
                        price_value = price_match.group(2)
                        cleaned_value = f"{currency}{price_value}"
                
                normalized[normalized_key] = cleaned_value
            else:
                normalized[normalized_key] = value
                
        return normalized
    
    def _apply_data_type_conversions(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply appropriate data type conversions to item fields.
        
        Args:
            item: Dictionary containing normalized data for an item
            
        Returns:
            Dictionary with fields converted to appropriate data types
        """
        converted = item.copy()
        
        # Handle numeric conversions
        for key in ["rating", "review_count"]:
            if key in converted and isinstance(converted[key], str):
                # Try to extract and convert numeric values
                try:
                    # Extract numbers from string (e.g., "4.5 stars" -> 4.5)
                    match = re.search(r'(\d+(?:\.\d+)?)', converted[key])
                    if match:
                        num_value = float(match.group(1))
                        # Convert to int if it's a whole number
                        if num_value.is_integer():
                            converted[key] = int(num_value)
                        else:
                            converted[key] = num_value
                except (ValueError, TypeError):
                    # Keep as string if conversion fails
                    pass
                    
        # Handle price conversion (store as string but also provide numeric version)
        if "price" in converted and isinstance(converted["price"], str):
            try:
                # Extract numeric price value
                price_match = re.search(r'[^\d]*([\d,]+\.?\d*)', converted["price"].replace(',', ''))
                if price_match:
                    price_value = price_match.group(1)
                    converted["price_numeric"] = float(price_value)
            except (ValueError, TypeError):
                pass
                
        # Convert date strings to ISO format
        date_fields = ["date", "published_date", "release_date"]
        for date_field in date_fields:
            if date_field in converted and isinstance(converted[date_field], str):
                try:
                    # This would need a more sophisticated date parser in a real implementation
                    # Here we just check if it looks like YYYY-MM-DD
                    if re.match(r'\d{4}-\d{2}-\d{2}', converted[date_field]):
                        # Already in ISO format, keep as is
                        pass
                    else:
                        # Would need additional date parsing logic
                        pass
                except Exception:
                    # Keep original if parsing fails
                    pass
                    
        return converted
    
    def _needs_ai_enhancement(self, item: Dict[str, Any], user_intent: Dict[str, Any]) -> bool:
        """
        Determine if an item needs AI-based enhancement.
        
        Args:
            item: Dictionary containing extracted data for an item
            user_intent: Dictionary containing the user's intent information
            
        Returns:
            True if the item needs AI enhancement, False otherwise
        """
        # Check if we have the minimum required fields
        required_fields = ["title", "url"]
        if not all(field in item for field in required_fields):
            return True
            
        # Check if we have the user-requested properties
        if "properties" in user_intent:
            requested_properties = user_intent["properties"]
            # If the user requested specific properties and we're missing any, use AI
            if any(prop not in item or not item[prop] for prop in requested_properties):
                return True
                
        # If we have very little data, use AI
        content_fields = [field for field in item if field not in ["url", "_extraction_confidence"]]
        if len(content_fields) < 3:
            return True
            
        return False
    
    def _calculate_extraction_confidence(self, item: Dict[str, Any], user_intent: Dict[str, Any]) -> float:
        """
        Calculate a confidence score for the extraction quality.
        
        Args:
            item: Dictionary containing extracted data for an item
            user_intent: Dictionary containing the user's intent information
            
        Returns:
            Confidence score between 0 and 1
        """
        score = 0.5  # Start with a moderate baseline confidence
        
        # More fields = higher confidence
        field_count = len([f for f in item if not f.startswith("_")])
        field_score = min(0.3, field_count / 10)  # Up to 0.3 for fields
        score += field_score
        
        # Having essential fields increases confidence
        essential_fields = ["title", "url"]
        if all(field in item for field in essential_fields):
            score += 0.1
            
        # Having requested properties increases confidence
        if "properties" in user_intent:
            requested = user_intent["properties"]
            found = [prop for prop in requested if prop in item and item[prop]]
            prop_score = min(0.3, len(found) / max(1, len(requested)) * 0.3)
            score += prop_score
            
        # Complete URLs (not relative) increase confidence
        if "url" in item and item["url"].startswith(("http://", "https://")):
            score += 0.05
            
        # Longer textual content usually indicates better extraction
        text_fields = ["title", "description"]
        text_content = ""
        for field in text_fields:
            if field in item and isinstance(item[field], str):
                text_content += item[field]
                
        text_score = min(0.15, len(text_content) / 500 * 0.15)  # Up to 0.15 for text content
        score += text_score
        
        return min(1.0, score)  # Cap at 1.0
    
    async def extract_from_search_results(self, url: str, content: str, user_intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured data from search results.
        
        Args:
            url: URL of the search results page
            content: HTML content of the search results page
            user_intent: User's intent information
            
        Returns:
            Dict containing extraction results
        """
        try:
            # Get domain for this URL
            domain = urlparse(url).netloc
            
            # Check if we have a template for this domain
            template = self.template_storage.get_template_for_domain(domain)
            
            if not template:
                # No template found, create a new one based on the search results
                template = await self._create_template_from_search_results(url, content, user_intent)
                template_id = f"{domain}_{int(time.time())}"
                self.template_storage.save_template(template_id, template)
            else:
                template_id = template.get("id", f"{domain}_{int(time.time())}")
                
            # Extract data using the template
            extraction_results = await self._extract_with_template(content, template, url)
            
            # Enhanced: Check if we need to apply AI-guided extraction as fallback
            if not extraction_results.get("success") or not extraction_results.get("results") or len(extraction_results.get("results", [])) < 2:
                logging.warning(f"Template extraction yielded insufficient results, trying AI-guided extraction")
                
                if self.ai_extraction_enabled:
                    # Apply AI-guided extraction as fallback
                    ai_extraction_results = await self._apply_ai_guided_extraction(url, content, user_intent)
                    
                    if ai_extraction_results and ai_extraction_results.get("success") and ai_extraction_results.get("entities"):
                        logging.info(f"AI-guided extraction succeeded with {len(ai_extraction_results['entities'])} results")
                        return {
                            "success": True,
                            "entities": ai_extraction_results.get("entities", []),
                            "meta": {
                                "url": url,
                                "timestamp": datetime.now().isoformat(),
                                "extraction_method": "ai_guided",
                                "template_id": None
                            },
                            "template_id": None
                        }
                
                # Apply fallback extraction as a last resort
                if not extraction_results.get("success") or not extraction_results.get("results"):
                    logging.warning(f"Both template and AI extraction failed, trying fallback extraction")
                    fallback_results = await perform_extraction_with_fallback(content, url, user_intent)
                    
                    if fallback_results and fallback_results.get("results"):
                        logging.info(f"Fallback extraction succeeded with {len(fallback_results['results'])} results")
                        return {
                            "success": True,
                            "entities": fallback_results.get("results", []),
                            "meta": {
                                "url": url,
                                "timestamp": datetime.now().isoformat(),
                                "extraction_method": "fallback",
                                "template_id": None
                            },
                            "template_id": None
                        }
            
            # Track extraction success metrics
            self._update_extraction_metrics(template_id, extraction_results)
            
            return {
                "success": True,
                "entities": extraction_results.get("results", []),
                "meta": {
                    "url": url,
                    "timestamp": datetime.now().isoformat(),
                    "template_id": template_id,
                    "extraction_method": "template"
                },
                "template_id": template_id
            }
            
        except Exception as e:
            logging.exception(f"Error in extract_from_search_results: {str(e)}")
            return {
                "success": False,
                "error": f"Error extracting data: {str(e)}",
                "url": url
            }
    
    async def _apply_ai_guided_extraction(self, url: str, content: str, user_intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply AI-guided extraction on search results.
        
        Args:
            url: URL of the search results page
            content: HTML content of the search results page
            user_intent: User's intent information
            
        Returns:
            Dict containing AI-guided extraction results
        """
        try:
            # Analyze the page structure to provide better context to the AI
            page_analysis = analyze_page_structure(content, url)
            
            # Generate prompt for AI extraction
            desired_properties = user_intent.get("properties", [])
            entity_type = user_intent.get("entity_types", ["item"])[0] if user_intent.get("entity_types") else "item"
            
            # Extract content with AI
            ai_results = await extract_content_with_ai(
                content,
                url,
                user_intent,
                page_structure=page_analysis,
                desired_properties=desired_properties,
                entity_type=entity_type
            )
            
            if not ai_results or not isinstance(ai_results, list):
                return {
                    "success": False,
                    "error": "AI extraction did not return valid results",
                    "entities": []
                }
                
            # Process and clean the AI extraction results
            processed_results = []
            for item in ai_results:
                if not isinstance(item, dict):
                    continue
                    
                # Clean and normalize the item data
                normalized_item = self._normalize_item_fields(item)
                
                # Make sure URLs are absolute
                if "url" in normalized_item and not normalized_item["url"].startswith(("http://", "https://")):
                    normalized_item["url"] = urljoin(url, normalized_item["url"])
                    
                processed_results.append(normalized_item)
                
            return {
                "success": True,
                "entities": processed_results,
                "meta": {
                    "url": url,
                    "timestamp": datetime.now().isoformat(),
                    "extraction_method": "ai_guided"
                }
            }
            
        except Exception as e:
            logging.error(f"Error in AI-guided extraction: {str(e)}")
            return {
                "success": False,
                "error": f"AI-guided extraction failed: {str(e)}",
                "entities": []
            }
    
    def _update_extraction_metrics(self, template_id: str, extraction_results: Dict[str, Any]) -> None:
        """
        Update extraction success metrics for a template.
        
        Args:
            template_id: ID of the template
            extraction_results: Results of the extraction
        """
        if template_id not in self.extraction_success_metrics:
            self.extraction_success_metrics[template_id] = {
                "total_attempts": 0,
                "successful_attempts": 0,
                "total_items": 0,
                "average_items_per_success": 0,
                "last_update": datetime.now().isoformat()
            }
            
        metrics = self.extraction_success_metrics[template_id]
        metrics["total_attempts"] += 1
        
        if extraction_results.get("success", False):
            metrics["successful_attempts"] += 1
            items_count = len(extraction_results.get("results", []))
            metrics["total_items"] += items_count
            
            if metrics["successful_attempts"] > 0:
                metrics["average_items_per_success"] = metrics["total_items"] / metrics["successful_attempts"]
                
        metrics["last_update"] = datetime.now().isoformat()
        
    def _identify_result_pattern(self, patterns: List[Dict[str, Any]], user_intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify the pattern that most likely represents search results.
        
        Args:
            patterns: List of content patterns identified in the page
            user_intent: User's intent information
            
        Returns:
            Dictionary containing the identified search result pattern
        """
        if not patterns:
            return None
            
        # Score patterns based on how likely they are to represent search results
        scored_patterns = []
        
        for pattern in patterns:
            score = 0
            
            # Check for repetitive patterns (search results are often repeated)
            if pattern.get("repetition_count", 0) >= 3:
                score += 30
            elif pattern.get("repetition_count", 0) >= 2:
                score += 20
                
            # Check for common search result indicators
            pattern_text = pattern.get("content", "").lower()
            common_indicators = [
                "product", "item", "result", "listing", "card", "entry",
                "title", "price", "rating", "review", "description"
            ]
            
            for indicator in common_indicators:
                if indicator in pattern_text:
                    score += 5
                    
            # Check if pattern contains fields that match user intent
            if "properties" in user_intent:
                for prop in user_intent["properties"]:
                    if prop.lower() in pattern_text:
                        score += 10
                        
            # Patterns with links are more likely to be search results
            if pattern.get("has_links", False):
                score += 15
                
            # Patterns with images are often product results
            if pattern.get("has_images", False):
                score += 10
                
            # Prefer patterns with moderate content length (not too short, not too long)
            content_length = len(pattern.get("content", ""))
            if 50 <= content_length <= 1000:
                score += 10
                
            scored_patterns.append((score, pattern))
            
        # Sort by score and return the best pattern
        scored_patterns.sort(key=lambda x: x[0], reverse=True)
        
        if scored_patterns:
            best_pattern = scored_patterns[0][1]
            logging.info(f"Selected search result pattern with score {scored_patterns[0][0]}: {best_pattern.get('selector', 'No selector')}")
            return best_pattern
            
        return None

    def _generate_selectors_from_pattern(self, pattern: Dict[str, Any], user_intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate extraction selectors based on identified pattern and user intent.
        
        Args:
            pattern: The identified search result pattern
            user_intent: User's intent information
            
        Returns:
            Dictionary containing extraction selectors
        """
        selectors = {
            "container": pattern.get("selector", ""),
            "fields": {}
        }
        
        # Generate field selectors based on user intent and common patterns
        requested_properties = user_intent.get("properties", [])
        
        # Common selector patterns for different fields
        field_selector_patterns = {
            "title": [
                "h1", "h2", "h3", "h4", ".title", ".product-title", ".item-title",
                "[class*='title']", "[class*='name']", "a[href]"
            ],
            "price": [
                ".price", ".cost", "[class*='price']", "[class*='cost']",
                "[data-price]", ".currency", "[class*='currency']"
            ],
            "description": [
                ".description", ".summary", ".excerpt", "[class*='description']",
                "[class*='summary']", "p"
            ],
            "url": [
                "a[href]", "[data-url]", "link"
            ],
            "image": [
                "img", "[data-src]", ".image", "[class*='image']", "[class*='photo']"
            ],
            "rating": [
                ".rating", ".stars", "[class*='rating']", "[class*='star']",
                "[data-rating]"
            ],
            "review_count": [
                ".reviews", ".review-count", "[class*='review']", "[class*='count']"
            ]
        }
        
        # Generate selectors for requested properties
        for prop in requested_properties:
            prop_lower = prop.lower()
            
            # Find matching patterns
            for field_type, patterns in field_selector_patterns.items():
                if field_type in prop_lower or prop_lower in field_type:
                    selectors["fields"][prop] = {
                        "selector": patterns[0],  # Start with the most common selector
                        "alternatives": patterns[1:3],  # Keep some alternatives
                        "attribute": "href" if field_type == "url" else "src" if field_type == "image" else "text"
                    }
                    break
            else:
                # Default generic selector for unknown properties
                selectors["fields"][prop] = {
                    "selector": f"[class*='{prop_lower}'], [data-{prop_lower}]",
                    "alternatives": [f".{prop_lower}", f"#{prop_lower}"],
                    "attribute": "text"
                }
        
        # Add some common fields if not already specified
        common_fields = ["title", "url"]
        for field in common_fields:
            if field not in selectors["fields"]:
                selectors["fields"][field] = {
                    "selector": field_selector_patterns[field][0],
                    "alternatives": field_selector_patterns[field][1:3],
                    "attribute": "href" if field == "url" else "text"
                }
        
        return selectors

    async def _create_template_from_search_results(self, page_url: str, html_content: str, user_intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an extraction template from search results.
        
        Args:
            page_url: URL of the search results page
            html_content: HTML content of the search results page
            user_intent: User's intent information
            
        Returns:
            Dictionary containing the extraction template
        """
        # Get domain for this URL
        domain = urlparse(page_url).netloc
        
        # Analyze the page to find patterns
        content_patterns = analyze_site_structure(html_content, page_url)
        
        if not content_patterns or "patterns" not in content_patterns:
            logging.error(f"Could not identify content patterns on {page_url}")
            return None
            
        # Identify the pattern that most likely represents search results
        result_pattern = self._identify_result_pattern(content_patterns["patterns"], user_intent)
        
        if not result_pattern:
            logging.error(f"Could not identify a search result pattern on {page_url}")
            return None
            
        # Create extraction template based on the identified pattern and user intent
        template = {
            "id": f"{domain}_search_results_{hash(str(user_intent)) % 10000:04d}",
            "name": f"Search Results Template for {domain}",
            "domain": domain,
            "created_at": self.template_storage.get_current_timestamp(),
            "updated_at": self.template_storage.get_current_timestamp(),
            "entity_type": user_intent.get("entity_types", ["item"])[0] if user_intent.get("entity_types") else "item",
            "properties": user_intent.get("properties", []),
            "extraction_pattern": result_pattern,
            "selectors": self._generate_selectors_from_pattern(result_pattern, user_intent)
        }
        
        # Save the template
        self.template_storage.save_template(template)
        
        return template
        
    async def _extract_with_template(self, content: str, template: Dict[str, Any], url: str) -> Dict[str, Any]:
        """
        Extract data using the provided template.
        
        Args:
            content: HTML content to extract from
            template: Extraction template
            url: URL of the page being extracted
            
        Returns:
            Dictionary containing extraction results
        """
        try:
            if not BeautifulSoup:
                raise ImportError("BeautifulSoup is required for template extraction")
                
            soup = BeautifulSoup(content, 'html.parser')
            selectors = template.get("selectors", {})
            container_selector = selectors.get("container", "")
            field_selectors = selectors.get("fields", {})
            
            results = []
            
            # Find all container elements
            if container_selector:
                containers = soup.select(container_selector)
            else:
                # Fallback: try to find common result containers
                common_containers = [
                    ".result", ".item", ".product", ".listing", 
                    "[class*='result']", "[class*='item']", "[class*='product']"
                ]
                containers = []
                for selector in common_containers:
                    containers = soup.select(selector)
                    if len(containers) >= 2:  # Found repeating elements
                        break
            
            # Extract data from each container
            for container in containers[:50]:  # Limit to 50 results
                item_data = {}
                
                # Extract each field
                for field_name, field_config in field_selectors.items():
                    value = self._extract_field_value(container, field_config, url)
                    if value:
                        item_data[field_name] = value
                
                # Only add items that have at least a title or URL
                if item_data.get("title") or item_data.get("url"):
                    results.append(item_data)
            
            success = len(results) > 0
            
            return {
                "success": success,
                "results": results,
                "template_id": template.get("id"),
                "extraction_method": "template"
            }
            
        except Exception as e:
            logging.exception(f"Error in template extraction: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "results": []
            }

    async def _process_paginated_results(
        self, 
        url: str, 
        content: str, 
        user_intent: Dict[str, Any],
        template_id: str = None
    ) -> List[Dict[str, Any]]:
        """
        Process paginated search results to extract data from multiple pages.
        
        Args:
            url: URL of the first search results page
            content: HTML content of the first search results page
            user_intent: User's intent information
            template_id: ID of the template to use for extraction
            
        Returns:
            List of extracted entities from all paginated results
        """
        # Detect pagination on the search results page
        pagination_info = await self.pagination_handler.detect_pagination_type(content, url)
        
        if not pagination_info["has_pagination"]:
            logging.info("No pagination detected on search results page")
            return []
        
        all_entities = []
        max_pages_to_process = self._determine_max_pages(user_intent)
        pages_processed = 0
        
        # Get domain for this URL
        domain = urlparse(url).netloc
        
        # Get the template if we have a template_id, otherwise try to get by domain
        template = None
        if template_id:
            template = self.template_storage.get_template(template_id)
        if not template:
            template = self.template_storage.get_template_for_domain(domain)
        
        # Generate pagination URLs
        pagination_urls = await self.pagination_handler.deep_pagination_urls(
            url, 
            content, 
            max_pages=max_pages_to_process
        )
        
        # Process each pagination URL
        for next_url in pagination_urls:
            if pages_processed >= max_pages_to_process:
                break
                
            try:
                # Fetch the next page
                next_page_result = await self.crawler.afetch(next_url)
                
                if not next_page_result or not hasattr(next_page_result, 'html'):
                    continue
                    
                next_content = next_page_result.html
                
                # Extract data from this page
                if template:
                    extraction_results = await self._extract_with_template(next_url, next_content, template, user_intent)
                    if extraction_results.get("entities"):
                        all_entities.extend(extraction_results["entities"])
                else:
                    # If no template, create one and extract
                    new_template = await self._create_template_from_search_results(next_url, next_content, user_intent)
                    extraction_results = await self._extract_with_template(next_url, next_content, new_template, user_intent)
                    if extraction_results.get("entities"):
                        all_entities.extend(extraction_results["entities"])
                
                pages_processed += 1
                
            except Exception as e:
                logging.exception(f"Error processing pagination page {next_url}: {str(e)}")
                continue
                
        logging.info(f"Processed {pages_processed} pagination pages, found {len(all_entities)} additional entities")
        return all_entities
        
    def _extract_field_value(self, container, field_config: Dict[str, Any], base_url: str) -> str:
        """
        Extract a field value from a container element using the field configuration.
        
        Args:
            container: BeautifulSoup element container
            field_config: Field extraction configuration
            base_url: Base URL for resolving relative URLs
            
        Returns:
            Extracted field value or None
        """
        try:
            selector = field_config.get("selector", "")
            attribute = field_config.get("attribute", "text")
            alternatives = field_config.get("alternatives", [])
            
            # Try main selector first
            element = container.select_one(selector)
            
            # Try alternatives if main selector fails
            if not element:
                for alt_selector in alternatives:
                    element = container.select_one(alt_selector)
                    if element:
                        break
            
            if not element:
                return None
            
            # Extract value based on attribute
            if attribute == "text":
                value = element.get_text(strip=True)
            elif attribute == "href":
                value = element.get("href", "")
                # Convert relative URLs to absolute
                if value and not value.startswith(("http://", "https://")):
                    from urllib.parse import urljoin
                    value = urljoin(base_url, value)
            elif attribute == "src":
                value = element.get("src", "")
                # Convert relative URLs to absolute
                if value and not value.startswith(("http://", "https://")):
                    from urllib.parse import urljoin
                    value = urljoin(base_url, value)
            else:
                value = element.get(attribute, "")
            
            return value.strip() if isinstance(value, str) else value
            
        except Exception as e:
            logging.warning(f"Error extracting field value: {str(e)}")
            return None

    def _get_best_selector(self, selectors: List[str], container, field_name: str) -> str:
        """
        Get the best selector from a list of candidate selectors.
        
        Args:
            selectors: List of candidate selectors
            container: BeautifulSoup element to test selectors against
            field_name: Name of the field being extracted
            
        Returns:
            Best selector from the list
        """
        try:
            best_selector = None
            best_score = 0
            
            for selector in selectors:
                try:
                    elements = container.select(selector)
                    
                    if not elements:
                        continue
                    
                    score = 0
                    
                    # Score based on number of matches (but not too many)
                    num_matches = len(elements)
                    if 1 <= num_matches <= 5:
                        score += 20
                    elif num_matches > 5:
                        score += 10  # Too many matches might be too generic
                    
                    # Score based on content quality
                    for element in elements[:3]:  # Check first 3 elements
                        text = element.get_text(strip=True)
                        if text:
                            score += 10
                            
                            # Field-specific scoring
                            if field_name == "price" and any(symbol in text for symbol in ["$", "€", "£", "¥"]):
                                score += 15
                            elif field_name == "rating" and any(word in text.lower() for word in ["star", "rating", "review"]):
                                score += 15
                            elif field_name == "title" and 10 <= len(text) <= 200:
                                score += 10
                    
                    if score > best_score:
                        best_score = score
                        best_selector = selector
                        
                except Exception:
                    continue
            
            return best_selector or (selectors[0] if selectors else "")
            
        except Exception as e:
            logging.warning(f"Error finding best selector: {str(e)}")
            return selectors[0] if selectors else ""

    def _should_process_pagination(self, user_intent: Dict[str, Any]) -> bool:
        """
        Determine if pagination should be processed based on user intent.
        
        Args:
            user_intent: User's intent information
            
        Returns:
            Boolean indicating whether to process pagination
        """
        # Check for explicit pagination flags in user intent
        if "pagination" in user_intent:
            return user_intent["pagination"]
            
        # Check for keywords indicating comprehensive results are desired
        query = user_intent.get("original_query", "").lower()
        comprehensive_keywords = ["all", "every", "complete", "comprehensive", "list", "full"]
        
        if any(keyword in query for keyword in comprehensive_keywords):
            return True
            
        # Check if the intent suggests a collection of items (plural)
        entity_types = user_intent.get("entity_types", [])
        if entity_types and isinstance(entity_types, list):
            for entity_type in entity_types:
                if entity_type.endswith('s'):  # Simple plural check
                    return True
                    
        # Default behavior based on the number of results from first page
        # If we got a good number of results, it might be worth checking pagination
        return True
        
    def _determine_max_pages(self, user_intent: Dict[str, Any]) -> int:
        """
        Determine the maximum number of pages to process based on user intent.
        
        Args:
            user_intent: User's intent information
            
        Returns:
            Maximum number of pages to process
        """
        # Check for explicit max_pages in user intent
        if "max_pages" in user_intent:
            try:
                return int(user_intent["max_pages"])
            except (ValueError, TypeError):
                pass
                
        # Default max pages based on query complexity
        query_complexity = self._calculate_query_complexity(user_intent)
        
        if query_complexity > 0.8:  # High complexity
            return 10
        elif query_complexity > 0.5:  # Medium complexity
            return 5
        else:  # Low complexity
            return 3
            
    def _calculate_query_complexity(self, user_intent: Dict[str, Any]) -> float:
        """
        Calculate the complexity of a query to determine pagination depth.
        
        Args:
            user_intent: User's intent information
            
        Returns:
            Complexity score between 0 and 1
        """
        complexity = 0.0
        
        # Check number of properties to extract
        properties = user_intent.get("properties", [])
        if properties:
            complexity += min(len(properties) / 10, 0.5)  # Max 0.5 from properties
            
        # Check specificity of query
        query = user_intent.get("original_query", "")
        words = query.split()
        specificity = min(len(words) / 20, 0.3)  # Max 0.3 from query length
        complexity += specificity
        
        # Check if filtering conditions are present
        filtering_keywords = ["where", "filter", "only", "specific", "exact"]
        if any(keyword in query.lower() for keyword in filtering_keywords):
            complexity += 0.2
            
        return min(complexity, 1.0)  # Ensure max is 1.0

    # ... existing methods ...
    
    def _extract_search_terms(self, user_intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract search terms from user intent with enhanced support for real estate queries.
        
        Args:
            user_intent: Dictionary containing the user's intent information
                
        Returns:
            Dictionary of search terms organized by field type
        """
        # Initialize search terms dictionary
        search_terms = {
            "general": [],
            "location": [],
            "property_type": [],
            "price_range": [],
            "bedrooms": [],
            "bathrooms": [],
            "other_features": []
        }
        
        if "original_query" in user_intent:
            # Start with the original query
            original_query = user_intent["original_query"]
            
            # Remove common prefixes like "find", "search for", etc.
            cleaned_query = re.sub(r'^(find|search for|look for|get|retrieve)\s+', '', original_query, flags=re.IGNORECASE)
            
            # Extract location information (city, state, zip, neighborhood)
            location_pattern = r'(?:in|near|around|at)\s+([\w\s,]+?)(?:\s+\d{5})?(?:$|and|\s+with|\s+has)'
            location_match = re.search(location_pattern, cleaned_query, re.IGNORECASE)
            if location_match:
                location = location_match.group(1).strip()
                search_terms["location"].append(location)
                # Remove the location from the query to avoid duplication
                cleaned_query = re.sub(location_pattern, ' ', cleaned_query, flags=re.IGNORECASE)
            
            # Extract property types
            property_types = ['house', 'apartment', 'condo', 'townhouse', 'duplex', 'studio', 
                             'loft', 'mansion', 'cottage', 'bungalow', 'single-family', 
                             'multi-family', 'mobile home', 'land', 'lot', 'commercial']
            for prop_type in property_types:
                if re.search(r'\b' + re.escape(prop_type) + r'(s|es)?\b', cleaned_query, re.IGNORECASE):
                    search_terms["property_type"].append(prop_type)
                    
            # Extract price ranges
            price_pattern = r'(?:under|below|above|over|from|between)\s+\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:-|to)?\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)?'
            price_match = re.search(price_pattern, cleaned_query, re.IGNORECASE)
            if price_match:
                if price_match.group(1) and price_match.group(2):
                    min_price = price_match.group(1).replace(',', '')
                    max_price = price_match.group(2).replace(',', '')
                    search_terms["price_range"].append(f"{min_price}-{max_price}")
                elif price_match.group(1):
                    price = price_match.group(1).replace(',', '')
                    search_terms["price_range"].append(price)
            
            # Extract bedroom requirements
            bedroom_pattern = r'(\d+)\s*(?:bed|bedroom|br)'
            bedroom_match = re.search(bedroom_pattern, cleaned_query, re.IGNORECASE)
            if bedroom_match:
                search_terms["bedrooms"].append(bedroom_match.group(1))
                
            # Extract bathroom requirements
            bathroom_pattern = r'(\d+)\s*(?:bath|bathroom|ba)'
            bathroom_match = re.search(bathroom_pattern, cleaned_query, re.IGNORECASE)
            if bathroom_match:
                search_terms["bathrooms"].append(bathroom_match.group(1))
                
            # Extract other features
            feature_keywords = ['garage', 'pool', 'basement', 'fireplace', 'backyard', 
                              'yard', 'garden', 'renovated', 'new', 'updated', 'modern', 
                              'air conditioning', 'parking', 'furnished', 'balcony', 'view']
            for feature in feature_keywords:
                if re.search(r'\b' + re.escape(feature) + r'\b', cleaned_query, re.IGNORECASE):
                    search_terms["other_features"].append(feature)
            
            # Split remaining query into words and filter out stop words
            stop_words = ["a", "an", "the", "in", "on", "at", "by", "for", "with", "about", "from"]
            remaining_words = [word for word in cleaned_query.split() if word.lower() not in stop_words]
            search_terms["general"].extend(remaining_words)
            
        # If we have specific entities mentioned, add them to the appropriate category
        if "entity_types" in user_intent:
            entity_types = user_intent["entity_types"]
            if isinstance(entity_types, list):
                for entity_type in entity_types:
                    entity_lower = entity_type.lower()
                    # Check if it's a property type
                    if entity_lower in ['house', 'apartment', 'condo', 'property', 'home', 'real estate']:
                        search_terms["property_type"].append(entity_lower)
                    else:
                        search_terms["general"].append(entity_type)
            elif isinstance(entity_types, str):
                entity_lower = entity_types.lower()
                if entity_lower in ['house', 'apartment', 'condo', 'property', 'home', 'real estate']:
                    search_terms["property_type"].append(entity_lower)
                else:
                    search_terms["general"].append(entity_types)
                
        # Add any specific properties that might help with the search
        if "properties" in user_intent:
            properties = user_intent["properties"]
            if isinstance(properties, dict):
                # Handle dictionary of properties
                for key, value in properties.items():
                    key_lower = key.lower()
                    if value:
                        if key_lower in ['city', 'state', 'zip', 'area', 'neighborhood']:
                            search_terms["location"].append(str(value))
                        elif key_lower in ['type', 'property_type', 'home_type']:
                            search_terms["property_type"].append(str(value))
                        elif key_lower in ['price', 'price_range', 'cost']:
                            search_terms["price_range"].append(str(value))
                        elif key_lower in ['bed', 'beds', 'bedroom', 'bedrooms']:
                            search_terms["bedrooms"].append(str(value))
                        elif key_lower in ['bath', 'baths', 'bathroom', 'bathrooms']:
                            search_terms["bathrooms"].append(str(value))
                        else:
                            search_terms["general"].append(str(value))
            elif isinstance(properties, list):
                # Handle list of property names
                for prop in properties:
                    search_terms["general"].append(str(prop))
        
        # If we have a key_property specified, prioritize it
        if "key_property" in user_intent and "properties" in user_intent:
            key_property = user_intent["key_property"]
            if key_property in user_intent["properties"]:
                property_value = user_intent["properties"][key_property]
                if property_value:
                    # Add to general terms with higher priority
                    search_terms["general"].insert(0, str(property_value))
                    
        # Remove duplicates from each category
        for category in search_terms:
            search_terms[category] = list(dict.fromkeys(search_terms[category]))
        
        # Special case: if there's a reference to Cleveland, OH in the real estate context,
        # explicitly add it to the location
        if "cleveland" in ' '.join(search_terms["general"]).lower() or "cleveland" in ' '.join(search_terms["location"]).lower():
            if "cleveland, oh" not in ' '.join(search_terms["location"]).lower():
                search_terms["location"] = ["Cleveland, OH"] + search_terms["location"]
        
        # If search_terms is empty or only has stop words, add a generic real estate search term
        if not any(search_terms.values()):
            search_terms["general"].append("homes for sale")
                    
        return search_terms
        
    def _select_best_search_form(self, forms: List[Dict[str, Any]], search_terms: Dict[str, Any], user_intent: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Select the most appropriate search form based on user intent and search terms.
        
        Args:
            forms: List of search forms found on the website
            search_terms: Dictionary of search terms extracted from user intent
            user_intent: Dictionary containing the user's intent information
                
        Returns:
            The selected search form or None if no suitable form is found
        """
        if not forms:
            return None
            
        # If there's only one form, use it
        if len(forms) == 1:
            return forms[0]
            
        # Score each form based on how well it matches our needs
        scored_forms = []
        for form in forms:
            score = 0
            
            # Prefer forms with a reasonable number of inputs
            input_count = len(form.get("inputs", []))
            if 1 <= input_count <= 3:
                score += 3
            elif input_count > 0:
                score += 1
                
            # Check form attributes for search-related terms
            form_action = form.get("action", "").lower()
            if "search" in form_action or "find" in form_action or "query" in form_action:
                score += 2
                
            # Check input names and placeholders for search-related terms
            for inp in form.get("inputs", []):
                input_name = inp.get("name", "").lower()
                placeholder = inp.get("placeholder", "").lower()
                input_type = inp.get("type", "").lower()
                
                if "search" in input_name or "query" in input_name or "q" == input_name:
                    score += 2
                    
                if "search" in placeholder or "find" in placeholder:
                    score += 1
                    
                # Check if the input appears to be for our specific search terms
                for category, terms in search_terms.items():
                    for term in terms:
                        term_lower = term.lower()
                        if term_lower in placeholder or term_lower in input_name:
                            score += 2
                            
                # Real estate specific scoring - favor forms with advanced search capabilities
                if category in ["location"] and ("location" in input_name or "city" in input_name or "zip" in input_name):
                    score += 3
                if category in ["price_range"] and ("price" in input_name or "cost" in input_name):
                    score += 3
                if category in ["bedrooms"] and ("bed" in input_name or "br" in input_name):
                    score += 3
                if category in ["bathrooms"] and ("bath" in input_name or "ba" in input_name):
                    score += 3
                            
                # Prefer text inputs for general search
                if input_type == "text":
                    score += 1
                    
                # Prefer selects for specific categories
                if input_type == "select":
                    if "property" in input_name or "type" in input_name:
                        score += 2
                    if "price" in input_name:
                        score += 2
                    if "bed" in input_name or "bath" in input_name:
                        score += 2
                        
            # Preferred form method
            if form.get("method", "").lower() == "get":
                score += 1  # GET is slightly preferred for easy debugging
                
            # Prefer forms with submit buttons
            if any(inp.get("type") == "submit" for inp in form.get("inputs", [])):
                score += 1
                
            # Prefer forms that are visible on the page (not hidden)
            if not form.get("hidden", False):
                score += 2
                
            # Store score with form
            scored_forms.append((score, form))
            
        # Sort by score (descending) and return the best
        if scored_forms:
            scored_forms.sort(reverse=True, key=lambda x: x[0])
            return scored_forms[0][1]
            
        # If no form has a good score, return the first one
        return forms[0] if forms else None

    async def _process_pagination(self, pagination_data: Dict[str, Any], 
                                 template: Dict[str, Any], 
                                 user_intent: Dict[str, Any], 
                                 max_pages: int = 3) -> Dict[str, Any]:
        """
        Process pagination to extract data from multiple pages of results.
        
        Args:
            pagination_data: Pagination information from the search results page
            template: The extraction template to use
            user_intent: Dictionary containing the user's intent information
            max_pages: Maximum number of pages to process
            
        Returns:
            Dictionary containing combined results from all pages
        """
        combined_results = {"results": []}
        
        # Check if pagination data is valid
        if not pagination_data or "next_page" not in pagination_data:
            return combined_results
            
        pages_processed = 0
        next_page_url = pagination_data.get("next_page")
        
        while next_page_url and pages_processed < max_pages:
            try:
                # Crawl the next page
                if self.crawler:
                    page_result = await self.crawler.crawl(next_page_url)
                    
                    if page_result and "html" in page_result:
                        # Extract data from this page
                        extraction_result = await self._extract_with_template(
                            page_result["html"],
                            template,
                            next_page_url
                        )
                        
                        if extraction_result and "results" in extraction_result:
                            combined_results["results"].extend(extraction_result["results"])
                            
                # Check for next page link
                if "pagination" in page_result and "next_page" in page_result["pagination"]:
                    next_page_url = page_result["pagination"]["next_page"]
                else:
                    next_page_url = None
                    
                pages_processed += 1
                    
            except Exception as e:
                logging.error(f"Error processing pagination: {str(e)}")
                break
                
        return combined_results
    
    async def evaluate_approach_options(self, url: str, user_intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate different approaches (search, sitemap, direct crawl) in parallel to determine the best strategy.
        
        Args:
            url: The base URL of the website to analyze
            user_intent: Dictionary containing the user's intent information
            
        Returns:
            Dictionary with approach evaluation results and recommendations
        """
        # Create tasks for parallel execution
        search_task = asyncio.create_task(self._evaluate_search_viability(url, user_intent))
        sitemap_task = asyncio.create_task(self._evaluate_sitemap_viability(url))
        
        # Wait for both tasks to complete
        search_result, sitemap_result = await asyncio.gather(search_task, sitemap_task)
        
        # Determine the best approach based on results
        approaches = []
        
        # Add viable approaches with confidence scores
        if search_result.get("viable", False):
            approaches.append({
                "name": "search",
                "confidence": search_result.get("confidence", 0.0),
                "details": search_result
            })
            
        if sitemap_result.get("viable", False):
            approaches.append({
                "name": "sitemap",
                "confidence": sitemap_result.get("confidence", 0.0),
                "details": sitemap_result
            })
        
        # Add default direct crawl approach as fallback
        approaches.append({
            "name": "direct_crawl",
            "confidence": 0.3,  # Lower base confidence for direct crawl
            "details": {
                "viable": True,
                "reason": "Fallback approach"
            }
        })
        
        # Sort approaches by confidence score
        approaches.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        return {
            "url": url,
            "approaches": approaches,
            "recommended_approach": approaches[0]["name"] if approaches else "direct_crawl",
            "search_viable": search_result.get("viable", False),
            "sitemap_viable": sitemap_result.get("viable", False)
        }

    async def _evaluate_search_viability(self, url: str, user_intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the viability of using search functionality for data extraction.
        
        Args:
            url: The website URL
            user_intent: Dictionary containing the user's intent information
            
        Returns:
            Dictionary with search viability assessment
        """
        try:
            # Extract search terms from user intent
            search_terms = self._extract_search_terms(user_intent)
            
            # Skip if we couldn't extract any useful search terms
            if not search_terms:
                return {
                    "viable": False,
                    "confidence": 0.0,
                    "reason": "No suitable search terms could be extracted"
                }
            
            # Find search forms
            search_form_info = await self.search_automator.detect_search_forms(url)
            
            if not search_form_info or "forms" not in search_form_info or not search_form_info["forms"]:
                return {
                    "viable": False,
                    "confidence": 0.0,
                    "reason": "No search forms found"
                }
            
            # Choose the best search form
            best_form = self._select_best_search_form(search_form_info["forms"], search_terms, user_intent)
            
            if not best_form:
                return {
                    "viable": False,
                    "confidence": 0.0,
                    "reason": "No suitable search form found"
                }
            
            # Calculate confidence in search approach based on form quality and search terms
            form_score = best_form.get("score", 0) / 10  # Normalize to 0-1 range
            search_term_quality = min(1.0, len(search_terms) / 3)  # More terms is better, up to 3
            
            confidence = 0.7 * form_score + 0.3 * search_term_quality
            
            return {
                "viable": True,
                "confidence": min(0.95, confidence),  # Cap at 0.95
                "form": best_form,
                "search_terms": search_terms[:3],  # Include top 3 search terms
                "reason": "Search functionality available and viable"
            }
            
        except Exception as e:
            logging.error(f"Error evaluating search viability: {str(e)}")
            return {
                "viable": False,
                "confidence": 0.0,
                "reason": f"Error: {str(e)}"
            }

    async def _evaluate_sitemap_viability(self, url: str) -> Dict[str, Any]:
        """
        Evaluate if sitemap approach is viable for extracting data from the given URL
        
        Args:
            url: The website URL to analyze
            
        Returns:
            Dictionary with viability assessment and confidence score
        """
        from urllib.parse import urljoin
        import aiohttp
        from xml.etree import ElementTree as ET
        
        # Common sitemap locations
        sitemap_locations = [
            "sitemap.xml",
            "sitemap_index.xml",
            "sitemap/sitemap.xml",
            "sitemaps/sitemap.xml",
            "robots.txt"  # Check robots.txt for sitemap references
        ]
        
        # Normalize the base URL
        base_url = url.rstrip('/')
        if not base_url.startswith('http'):
            base_url = f"https://{base_url}"
            
        sitemap_urls = []
        sitemap_found = False
        sitemap_entry_count = 0
        
        async with aiohttp.ClientSession() as session:
            # First check robots.txt for sitemap references
            try:
                robots_url = urljoin(base_url, "/robots.txt")
                async with session.get(robots_url, timeout=5) as response:
                    if response.status == 200:
                        robots_text = await response.text()
                        for line in robots_text.splitlines():
                            if line.lower().startswith("sitemap:"):
                                sitemap_url = line.split(":", 1)[1].strip()
                                sitemap_urls.append(sitemap_url)
            except Exception as e:
                logging.debug(f"Error checking robots.txt: {e}")
                
            # Check common sitemap locations if none found in robots.txt
            if not sitemap_urls:
                for location in sitemap_locations:
                    if location == "robots.txt":
                        continue  # Already checked
                        
                    try:
                        sitemap_url = urljoin(base_url, location)
                        async with session.get(sitemap_url, timeout=5) as response:
                            if response.status == 200:
                                content_type = response.headers.get('Content-Type', '')
                                if 'xml' in content_type:
                                    sitemap_urls.append(sitemap_url)
                                    break
                    except Exception as e:
                        logging.debug(f"Error checking sitemap at {location}: {e}")
            
            # Analyze the first sitemap found
            if sitemap_urls:
                try:
                    sitemap_url = sitemap_urls[0]
                    async with session.get(sitemap_url, timeout=10) as response:
                        if response.status == 200:
                            sitemap_xml = await response.text()
                            sitemap_found = True
                            
                            # Parse the sitemap
                            root = ET.fromstring(sitemap_xml)
                            
                            # Handle sitemap index files
                            if root.tag.endswith('sitemapindex'):
                                # Get the first child sitemap
                                sitemap_tags = root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap') or root.findall('.//sitemap')
                                
                                if sitemap_tags:
                                    loc_tags = sitemap_tags[0].findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc') or sitemap_tags[0].findall('.//loc')
                                    if loc_tags:
                                        child_sitemap_url = loc_tags[0].text.strip()
                                        async with session.get(child_sitemap_url, timeout=10) as child_response:
                                            if child_response.status == 200:
                                                sitemap_xml = await child_response.text()
                                                root = ET.fromstring(sitemap_xml)
                            
                            # Count URLs in the sitemap
                            url_tags = root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url') or root.findall('.//url')
                            sitemap_entry_count = len(url_tags)
                except Exception as e:
                    logging.error(f"Error analyzing sitemap: {e}")
        
        # Determine viability and confidence
        viable = sitemap_found and sitemap_entry_count > 10
        
        # Calculate confidence score (0.0-1.0)
        confidence = 0.0
        if sitemap_found:
            # Base confidence from having a sitemap
            confidence = 0.5
            
            # Adjust based on number of entries
            if sitemap_entry_count > 100:
                confidence += 0.3
            elif sitemap_entry_count > 50:
                confidence += 0.2
            elif sitemap_entry_count > 20:
                confidence += 0.1
                
            # Cap at 0.85 since search approach might still be better
            confidence = min(confidence, 0.85)
        
        return {
            "viable": viable,
            "confidence": confidence,
            "reason": f"Sitemap {'found' if sitemap_found else 'not found'} with {sitemap_entry_count} entries",
            "sitemap_urls": sitemap_urls,
            "entry_count": sitemap_entry_count
        }

    async def navigate_to_details_page(self, search_results: Dict[str, Any], 
                               crawler: AsyncWebCrawler, 
                               depth_info: Dict[str, int] = None) -> List[Dict[str, Any]]:
        """
        Navigate to detail pages from search results while managing depth properly.
        
        Args:
            search_results: Dictionary containing search results
            crawler: AsyncWebCrawler instance
            depth_info: Dictionary tracking depth information for the navigation flow
            
        Returns:
            List of details page results with proper depth tracking
        """
        if not search_results or "results" not in search_results or not search_results["results"]:
            return []
            
        # Initialize depth tracking if not provided
        if depth_info is None:
            depth_info = {
                "search_depth": 1,    # Search page is considered depth 1
                "results_depth": 2,    # Results pages are depth 2
                "details_depth": 3,    # Detail pages are depth 3
                "max_details": 10      # Maximum number of detail pages to visit
            }
        
        details_results = []
        visited_urls = set()
        
        # Get the results from the search
        items = search_results.get("results", [])
        
        # Limit the number of detail pages to visit
        items_to_process = items[:min(len(items), depth_info.get("max_details", 10))]
        
        logging.info(f"Navigating to {len(items_to_process)} detail pages from search results")
        
        # Process each item to get its detail page
        for item in items_to_process:
            # Skip if no URL
            if "url" not in item or not item["url"]:
                continue
                
            detail_url = item["url"]
            
            # Skip already visited URLs
            if detail_url in visited_urls:
                continue
                
            visited_urls.add(detail_url)
            
            try:
                # Fetch the detail page
                result = await crawler.afetch(detail_url)
                
                if result.success:
                    # Add the detail page result with proper depth tracking
                    detail_result = {
                        "url": detail_url,
                        "html": result.html,
                        "depth": depth_info.get("details_depth", 3),
                        "parent_type": "search_result",
                        "extraction_context": {
                            "parent_url": search_results.get("url", ""),
                            "search_term": search_results.get("term_used", ""),
                            "result_position": items.index(item) + 1,
                            "search_to_detail_flow": True
                        }
                    }
                    
                    # Extract data from detail page
                    extracted_data = await self._extract_detail_page(detail_result, item)
                    
                    # Add extracted data to the result
                    detail_result["data"] = extracted_data
                    details_results.append(detail_result)
                    
            except Exception as e:
                logging.error(f"Error navigating to detail page {detail_url}: {str(e)}")
        
        return details_results

    async def _extract_detail_page(self, detail_result: Dict[str, Any], 
                                  search_result_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract data from a detail page, combining with data from search result.
        
        Args:
            detail_result: Dictionary containing detail page information
            search_result_item: Dictionary containing the search result item data
            
        Returns:
            Extracted and merged data from both sources
        """
        try:
            # Initialize with data from search result
            extracted_data = dict(search_result_item)
            
            # Get the HTML and URL
            html = detail_result.get("html", "")
            url = detail_result.get("url", "")
            
            if not html:
                return extracted_data
                
            # Create or find an appropriate extraction template
            template = await self._get_or_create_detail_template(html, url, search_result_item)
            
            if not template:
                return extracted_data
                
            # Extract additional data from the detail page
            detail_data = await self._extract_with_template(template, html, url)
            
            # Merge data, preferring detail page data for overlapping fields
            if detail_data and "results" in detail_data and detail_data["results"]:
                # Get the first result from the detail page extraction
                first_result = detail_data["results"][0]
                
                # Merge with search result data, preferring detail page values
                for key, value in first_result.items():
                    if value:  # Only update if we have a value
                        extracted_data[key] = value
            
            return extracted_data
            
        except Exception as e:
            logging.error(f"Error extracting detail page data: {str(e)}")
            return search_result_item

    async def _get_or_create_detail_template(self, html: str, url: str, 
                                            search_result_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get or create a template for extracting data from a detail page.
        
        Args:
            html: HTML content of the detail page
            url: URL of the detail page
            search_result_item: Dictionary containing the search result item data
            
        Returns:
            Template for extracting data from the detail page
        """
        # Extract domain for template identification
        domain = self._extract_domain(url)
        
        # Try to find existing detail page template
        existing_templates = self.template_storage.find_templates_for_domain(domain)
        
        for template in existing_templates:
            if template.get("template_type") == "detail":
                logging.info(f"Using existing detail template for {domain}")
                return template
        
        # No existing template found, create a new one
        logging.info(f"Creating new detail template for {domain}")
        
        try:
            # Create a basic set of properties to extract based on the search result item
            properties = list(search_result_item.keys())
            
            # Add additional common detail page properties if not already included
            common_detail_properties = ["description", "specifications", "features", "details", 
                                       "price", "contact", "address", "images", "seller"]
                                       
            for prop in common_detail_properties:
                if prop not in properties:
                    properties.append(prop)
            
            # Create a new template
            detail_template = {
                "id": f"{domain}_detail_{hash(url) % 10000:04d}",
                "name": f"Detail Template for {domain}",
                "domain": domain,
                "template_type": "detail",
                "created_at": self.template_storage.get_current_timestamp(),
                "updated_at": self.template_storage.get_current_timestamp(),
                "properties": properties,
                "selectors": await self._generate_detail_selectors(html, properties)
            }
            
            # Save the template
            self.template_storage.save_template(detail_template)
            
            return detail_template
            
        except Exception as e:
            logging.error(f"Error creating detail template: {str(e)}")
            return None

    async def _generate_detail_selectors(self, html: str, properties: List[str]) -> Dict[str, str]:
        """
        Generate CSS selectors for extracting data from a detail page.
        
        Args:
            html: HTML content of the detail page
            properties: List of properties to extract
            
        Returns:
            Dictionary mapping property names to CSS selectors
        """
        selectors = {}
        
        if not html:
            return selectors
            
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # Map of property names to potential CSS selectors
            selector_patterns = {
                "title": ["h1", "h1.title", ".product-title", ".item-title", 
                         "[itemprop='name']", ".page-title", ".main-title"],
                "price": [".price", ".product-price", "[itemprop='price']", 
                         ".item-price", ".main-price", ".current-price"],
                "description": [".description", "[itemprop='description']", 
                              ".product-description", ".item-description", 
                              ".main-description", "section.description"],
                "specifications": [".specifications", ".specs", ".product-specs", 
                                 ".item-specs", ".technical-specs", ".details"],
                "features": [".features", ".product-features", ".key-features", 
                           ".item-features", ".benefits"],
                "details": [".details", ".product-details", ".item-details"],
                "images": ["img.product-image", "img.main-image", 
                          "[itemprop='image']", ".gallery img", ".carousel img"],
                "address": [".address", "[itemprop='address']", ".location", 
                           ".property-address", ".item-address"],
                "seller": [".seller", ".vendor", ".store", ".dealer", 
                          "[itemprop='seller']", ".company", ".agent"],
                "contact": [".contact", ".contact-info", ".phone", 
                           ".email", "[itemprop='telephone']"]
            }
            
            # Generic property selector patterns
            generic_patterns = [
                ".{prop}", "#{prop}", "[itemprop='{prop}']",
                ".{prop}-value", "#{prop}-value", "[data-{prop}]",
                ".{prop}-container .value", "#{prop}-container .value",
                "dt:contains('{prop}') + dd"
            ]
            
            # For each property, try to find a matching selector
            for prop in properties:
                prop_lower = prop.lower()
                
                # If we have specific patterns for this property, try those first
                if prop_lower in selector_patterns:
                    for pattern in selector_patterns[prop_lower]:
                        elements = soup.select(pattern)
                        if elements:
                            selectors[prop_lower] = pattern
                            break
                
                # If no specific selector found, try generic patterns
                if prop_lower not in selectors:
                    for pattern in generic_patterns:
                        try:
                            selector = pattern.format(prop=prop_lower)
                            elements = soup.select(selector)
                            if elements:
                                selectors[prop_lower] = selector
                                break
                        except:
                            continue
            
            return selectors
            
        except Exception as e:
            logging.error(f"Error generating detail page selectors: {str(e)}")
            return selectors

    async def _process_pagination(self, 
                              search_results_url: str, 
                              html_content: str, 
                              template: Dict[str, Any],
                              max_pages: int = 5) -> Dict[str, Any]:
        """
        Process pagination for search results to extract data from multiple pages.
        
        Args:
            search_results_url: URL of the initial search results page
            html_content: HTML content of the initial search results page
            template: Extraction template
            max_pages: Maximum number of pages to process
            
        Returns:
            Dictionary containing the combined extraction results
        """
        try:
            # Initialize pagination handler if needed
            if not hasattr(self, 'pagination_handler'):
                from components.pagination_handler import PaginationHandler
                self.pagination_handler = PaginationHandler()
                
            # Detect pagination type
            pagination_info = await self.pagination_handler.detect_pagination_type(html_content, search_results_url)
            
            # Initialize combined results container
            combined_results = {
                "success": True,
                "url": search_results_url,
                "extraction_time": 0,
                "results": [],
                "pagination": {
                    "detected": pagination_info["has_pagination"],
                    "type": pagination_info["pagination_type"] if pagination_info["has_pagination"] else None,
                    "page_count": 1,
                    "processed_pages": 1
                }
            }
            
            # Process the first page (we already have its content)
            first_page_results = await self._extract_with_template(html_content, template, search_results_url)
            start_time = time.time()
            
            # Add first page results
            if first_page_results["success"] and first_page_results.get("results"):
                combined_results["results"].extend(first_page_results["results"])
            
            # If no pagination detected or we hit the maximum pages, return current results
            if not pagination_info["has_pagination"] or not pagination_info["next_page_url"]:
                combined_results["extraction_time"] = time.time() - start_time
                combined_results["count"] = len(combined_results["results"])
                return combined_results
                
            # Continue to follow pagination for additional pages
            current_page = 1
            next_page_url = pagination_info["next_page_url"]
            
            while next_page_url and current_page < max_pages:
                current_page += 1
                logging.info(f"Processing pagination page {current_page}: {next_page_url}")
                
                # Fetch the next page
                crawler = AsyncWebCrawler()
                page_result = await crawler.arun(url=next_page_url)
                
                if not page_result.success:
                    logging.warning(f"Failed to fetch pagination page {current_page}: {next_page_url}")
                    break
                    
                # Extract data from this page
                page_content = page_result.html
                page_results = await self._extract_with_template(page_content, template, next_page_url)
                
                # Add to combined results
                if page_results["success"] and page_results.get("results"):
                    combined_results["results"].extend(page_results["results"])
                    combined_results["pagination"]["processed_pages"] += 1
                    
                # Look for next page link
                page_pagination = await self.pagination_handler.detect_pagination_type(page_content, next_page_url)
                
                if not page_pagination["has_pagination"] or not page_pagination["next_page_url"]:
                    break
                    
                next_page_url = page_pagination["next_page_url"]
                
            # Deduplicate results by URL
            unique_results = {}
            for result in combined_results["results"]:
                key = result.get("url", "") or result.get("title", "")
                if key and key not in unique_results:
                    unique_results[key] = result
                    
            combined_results["results"] = list(unique_results.values())
            combined_results["count"] = len(combined_results["results"])
            combined_results["extraction_time"] = time.time() - start_time
            combined_results["pagination"]["page_count"] = current_page
            
            return combined_results
            
        except Exception as e:
            logging.error(f"Error processing pagination: {str(e)}")
            return {
                "success": False,
                "error": f"Error processing pagination: {str(e)}",
                "extraction_time": 0,
                "results": []
            }

    def _enhance_results(self, results: List[Dict[str, Any]], base_url: str) -> List[Dict[str, Any]]:
        """
        Clean up and enhance extraction results.
        
        Args:
            results: List of extracted results
            base_url: Base URL of the search results page
            
        Returns:
            Enhanced list of results
        """
        enhanced_results = []
        
        for result in results:
            # Skip empty results
            if not result:
                continue
                
            # Create a copy to avoid modifying the original
            enhanced = result.copy()
            
            # Fix relative URLs
            if "url" in enhanced and enhanced["url"]:
                enhanced["url"] = urljoin(base_url, enhanced["url"])
                
            if "image" in enhanced and enhanced["image"]:
                enhanced["image"] = urljoin(base_url, enhanced["image"])
                
            if "thumbnail" in enhanced and enhanced["thumbnail"]:
                enhanced["thumbnail"] = urljoin(base_url, enhanced["thumbnail"])
                
            # Clean up text fields
            text_fields = ["title", "description", "price", "name", "address"]
            for field in text_fields:
                if field in enhanced and enhanced[field]:
                    # Convert to string if needed
                    value = str(enhanced[field])
                    
                    # Remove extra whitespace
                    value = re.sub(r'\s+', ' ', value).strip()
                    
                    # Remove common boilerplate text
                    value = re.sub(r'(^Read more|More info$|Click here$|View details$)', '', value).strip()
                    
                    enhanced[field] = value
            
            # Extract numeric values from price
            if "price" in enhanced and enhanced["price"]:
                price_str = enhanced["price"]
                if isinstance(price_str, str):
                    # Extract numeric part of price
                    price_match = re.search(r'[\$€£¥]?\s*([0-9,]+(?:\.\d{1,2})?)', price_str)
                    if price_match:
                        try:
                            price_value = float(price_match.group(1).replace(',', ''))
                            enhanced["price_value"] = price_value
                            
                            # Extract currency symbol
                            currency_match = re.search(r'([\$€£¥])', price_str)
                            if currency_match:
                                enhanced["currency"] = currency_match.group(1)
                        except (ValueError, TypeError):
                            pass
            
            # Add domain information
            if "url" in enhanced and enhanced["url"]:
                domain = self._extract_domain(enhanced["url"])
                enhanced["domain"] = domain
            
            # Add timestamp
            enhanced["extracted_at"] = datetime.now().isoformat()
            
            enhanced_results.append(enhanced)
            
        return enhanced_results

    def _extract_domain(self, url: str) -> str:
        """
        Extract the domain from a URL.
        
        Args:
            url: URL to extract domain from
            
        Returns:
            Domain name
        """
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            # Remove www. prefix if present
            if domain.startswith("www."):
                domain = domain[4:]
                
            return domain
        except Exception:
            return ""

    def _template_matches_intent(self, template: Dict[str, Any], user_intent: Dict[str, Any]) -> bool:
        """
        Check if a template matches the user's intent.
        
        Args:
            template: Extraction template
            user_intent: User's intent dictionary
            
        Returns:
            True if the template matches the intent, False otherwise
        """
        # If there's no intent data or template is generic, consider it a match
        if not user_intent or not template.get("entity_types", []):
            return True
            
        # Check if entity types match
        intent_entities = set(user_intent.get("entity_types", []))
        template_entities = set(template.get("entity_types", []))
        
        # If both have entity types, check for overlap
        if intent_entities and template_entities:
            # If there's at least one common entity type, it's potentially a match
            if intent_entities.intersection(template_entities):
                return True
            else:
                return False
                
        # Check properties
        intent_props = set(user_intent.get("properties", []))
        template_props = set(prop["name"] for prop in template.get("properties", []))
        
        # If both have properties, check for overlap
        if intent_props and template_props:
            # If there's a significant overlap in properties, it's a match
            common_props = intent_props.intersection(template_props)
            if len(common_props) >= min(2, len(intent_props) // 2):
                return True
        
        # When in doubt, assume it's a potential match
        return True

    async def refine_template_based_on_extraction(self, 
                                           template: Dict[str, Any], 
                                           extraction_results: Dict[str, Any],
                                           html_content: str,
                                           apply_immediately: bool = True) -> Dict[str, Any]:
        """
        Refine a template based on extraction success or failure.
        
        Args:
            template: The template to refine
            extraction_results: Results of the extraction using this template
            html_content: HTML content that was used for extraction
            apply_immediately: Whether to immediately save and apply the refined template
            
        Returns:
            Dictionary containing the refined template and improvement metrics
        """
        if not template or not extraction_results:
            return {"success": False, "reason": "Invalid template or extraction results"}
            
        template_id = template.get("id")
        if not template_id:
            return {"success": False, "reason": "Template has no ID"}
            
        # Initialize metrics for this template if not already tracking
        if template_id not in self.extraction_success_metrics:
            self.extraction_success_metrics[template_id] = {
                "total_uses": 0,
                "successful_extractions": 0,
                "empty_extractions": 0,
                "failed_extractions": 0,
                "property_success_rates": {},
                "average_extraction_time": 0.0
            }
            
        metrics = self.extraction_success_metrics[template_id]
        
        # Update metrics based on current extraction
        metrics["total_uses"] += 1
        
        if not extraction_results.get("success", False):
            metrics["failed_extractions"] += 1
            refinement_reason = "extraction_failure"
        elif not extraction_results.get("results"):
            metrics["empty_extractions"] += 1
            refinement_reason = "empty_results"
        else:
            metrics["successful_extractions"] += 1
            refinement_reason = "optimization"
            
        # Track property extraction success rates
        results = extraction_results.get("results", [])
        if results:
            properties = template.get("selectors", {}).keys()
            for prop in properties:
                # Count how many results have this property
                successful_extractions = sum(1 for r in results if prop in r and r[prop])
                success_rate = successful_extractions / len(results)
                
                # Update running average for this property
                if prop not in metrics["property_success_rates"]:
                    metrics["property_success_rates"][prop] = success_rate
                else:
                    # Weighted average giving more importance to recent extractions
                    prev_rate = metrics["property_success_rates"][prop]
                    metrics["property_success_rates"][prop] = 0.7 * success_rate + 0.3 * prev_rate
        
        # Update extraction time tracking
        extraction_time = extraction_results.get("extraction_time", 0)
        if extraction_time > 0:
            if metrics["average_extraction_time"] == 0:
                metrics["average_extraction_time"] = extraction_time
            else:
                metrics["average_extraction_time"] = (0.7 * metrics["average_extraction_time"] + 
                                                     0.3 * extraction_time)
        
        # Determine what needs improvement
        needs_improvement = []
        
        # Failed extraction or empty results indicate need for fundamental improvement
        if refinement_reason in ["extraction_failure", "empty_results"]:
            needs_improvement.append("basic_selectors")
        
        # Check property success rates for more specific improvements
        low_success_properties = []
        for prop, rate in metrics["property_success_rates"].items():
            if rate < 0.7:  # Less than 70% success rate
                low_success_properties.append(prop)
                
        if low_success_properties:
            needs_improvement.append("property_selectors")
            
        # Check if optimization is possible
        if metrics["total_uses"] >= 3 and metrics["average_extraction_time"] > 0.5:
            needs_improvement.append("performance")
            
        # No need for refinement if everything is working well
        if not needs_improvement:
            return {
                "success": True,
                "template_id": template_id,
                "refinement_needed": False,
                "metrics": metrics,
                "reason": "Template performing well, no refinement needed"
            }
            
        logging.info(f"Refining template {template_id} for reasons: {needs_improvement}")
        
        # Create a copy of the template for refinement
        refined_template = dict(template)
        selectors = dict(template.get("selectors", {}))
        refinements_made = []
        
        try:
            # Process different improvement needs
            if "basic_selectors" in needs_improvement:
                # Try to fix fundamental selectors (e.g., item container)
                improved_selectors = await self._improve_basic_selectors(
                    html_content, 
                    selectors,
                    refinement_reason
                )
                
                if improved_selectors:
                    selectors.update(improved_selectors)
                    refinements_made.append("improved_basic_selectors")
            
            if "property_selectors" in needs_improvement:
                # Try to fix specific property selectors
                improved_properties = await self._improve_property_selectors(
                    html_content,
                    selectors,
                    low_success_properties
                )
                
                if improved_properties:
                    selectors.update(improved_properties)
                    refinements_made.append("improved_property_selectors")
            
            if "performance" in needs_improvement:
                # Optimize selectors for performance
                optimized_selectors = self._optimize_selectors_for_performance(selectors)
                
                if optimized_selectors:
                    selectors = optimized_selectors
                    refinements_made.append("optimized_performance")
            
            # Update the template with refined selectors
            refined_template["selectors"] = selectors
            refined_template["updated_at"] = self.template_storage.get_current_timestamp()
            refined_template["refinement_history"] = template.get("refinement_history", []) + [{
                "date": self.template_storage.get_current_timestamp(),
                "reason": refinement_reason,
                "refinements": refinements_made
            }]
            
            # Save and apply the refined template if requested
            if apply_immediately and refinements_made:
                self.template_storage.save_template(refined_template)
                logging.info(f"Refined template {template_id} saved successfully")
            
            return {
                "success": True,
                "template_id": template_id,
                "refinement_needed": True,
                "refinements_made": refinements_made,
                "metrics": metrics,
                "template": refined_template if refinements_made else template
            }
            
        except Exception as e:
            logging.error(f"Error refining template: {str(e)}")
            return {
                "success": False,
                "template_id": template_id,
                "reason": f"Error during template refinement: {str(e)}",
                "metrics": metrics
            }
            
    async def _improve_basic_selectors(self, html_content: str, 
                                    existing_selectors: Dict[str, str],
                                    failure_reason: str) -> Dict[str, str]:
        """
        Improve fundamental selectors for successful extraction.
        
        Args:
            html_content: HTML content that was used for extraction
            existing_selectors: Current selectors in the template
            failure_reason: Reason for refinement
            
        Returns:
            Dictionary of improved selectors
        """
        improved_selectors = {}
        
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find possible result containers
            potential_containers = []
            
            # Look for common list/grid containers
            container_selectors = [
                ".products", ".items", ".search-results", ".listings", 
                ".grid", ".row", ".results", "ul.product-list", "div.products",
                ".cards", "[data-testid='search-results']", ".search-result-list",
                ".category-products", ".product-grid", ".collection-products"
            ]
            
            for selector in container_selectors:
                elements = soup.select(selector)
                potential_containers.extend(elements)
                
            # If we found containers, analyze their children for consistent structure
            best_container = None
            max_consistent_children = 0
            
            for container in potential_containers:
                children = container.find_all(recursive=False)
                if len(children) < 2:
                    continue
                    
                # Check if children have consistent structure
                # Simple approach: check if they have same tag name
                tags = [child.name for child in children]
                most_common_tag = max(set(tags), key=tags.count)
                consistent_children = sum(1 for tag in tags if tag == most_common_tag)
                
                if consistent_children > max_consistent_children:
                    max_consistent_children = consistent_children
                    best_container = container
            
            # If we found a good container, update item selectors
            if best_container and max_consistent_children >= 3:
                container_selector = self._get_best_selector(best_container)
                most_common_tag = max(set([child.name for child in best_container.find_all(recursive=False)]), 
                                    key=[child.name for child in best_container.find_all(recursive=False)].count)
                
                # Update the item selector
                improved_selectors["container"] = container_selector
                improved_selectors["item"] = f"{container_selector} > {most_common_tag}"
                
                # Find a better title selector if title extraction failed
                sample_item = best_container.find(most_common_tag)
                if sample_item:
                    # Look for heading elements within the item
                    headings = sample_item.find_all(["h1", "h2", "h3", "h4"])
                    if headings:
                        improved_selectors["title"] = f"{improved_selectors['item']} {self._get_best_selector(headings[0])}"
                    
                    # Look for links that might contain titles
                    links = sample_item.find_all("a")
                    if links:
                        improved_selectors["url"] = f"{improved_selectors['item']} a"
                        
                        # If no heading found, use the first link text as title
                        if "title" not in improved_selectors and links[0].get_text().strip():
                            improved_selectors["title"] = f"{improved_selectors['item']} a"
                    
                    # Look for images
                    images = sample_item.find_all("img")
                    if images:
                        improved_selectors["image"] = f"{improved_selectors['item']} img"
                    
                    # Look for price elements
                    price_selectors = [".price", ".product-price", "[itemprop='price']", 
                                     ".item-price", ".main-price", ".current-price"]
                    
                    for price_selector in price_selectors:
                        price_elements = sample_item.select(price_selector)
                        if price_elements:
                            improved_selectors["price"] = f"{improved_selectors['item']} {price_selector}"
                            break
            
            return improved_selectors
            
        except Exception as e:
            logging.error(f"Error improving basic selectors: {str(e)}")
            return {}
            
    async def _improve_property_selectors(self, html_content: str, 
                                      existing_selectors: Dict[str, str],
                                      low_success_properties: List[str]) -> Dict[str, str]:
        """
        Improve selectors for specific properties with low extraction success.
        
        Args:
            html_content: HTML content that was used for extraction
            existing_selectors: Current selectors in the template
            low_success_properties: List of properties with low extraction success rates
            
        Returns:
            Dictionary of improved selectors for the specified properties
        """
        improved_selectors = {}
        
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Process each property needing improvement
            for prop in low_success_properties:
                # Skip if property name is not in existing selectors
                if prop not in existing_selectors:
                    continue
                
                current_selector = existing_selectors[prop]
                
                # Try different approaches based on property type
                if prop == "title":
                    # Look for heading elements
                    for heading_tag in ["h1", "h2", "h3", "h4"]:
                        headings = soup.find_all(heading_tag)
                        if headings:
                            # Filter to those that actually contain some text
                            text_headings = [h for h in headings if h.get_text().strip()]
                            if text_headings:
                                # If inside container, scope selector to container
                                container_selector = existing_selectors.get("container", "")
                                if container_selector:
                                    improved_selectors[prop] = f"{container_selector} {heading_tag}"
                                else:
                                    improved_selectors[prop] = heading_tag
                                break
                
                elif prop == "price":
                    # Try more specific price selectors
                    price_selectors = [
                        ".product-price", ".offer-price", ".current-price", 
                        "[itemprop='price']", ".amount", ".product-price", 
                        ".sale-price", ".item-price", ".price"
                    ]
                    
                    # Try to find elements containing price patterns
                    price_pattern = r'(\$|€|£|\d+\.\d{2}|\d+\,\d{2})'
                    price_elements = []
                    
                    for element in soup.find_all(text=re.compile(price_pattern)):
                        parent = element.parent
                        if parent and parent.name != 'script' and parent.name != 'style':
                            price_elements.append(parent)
                    
                    if price_elements:
                        # Use the selector of the first found price element
                        improved_selectors[prop] = self._get_best_selector(price_elements[0])
                    else:
                        # Try standard price selectors
                        for selector in price_selectors:
                            elements = soup.select(selector)
                            if elements:
                                improved_selectors[prop] = selector
                                break
                
                elif prop == "image":
                    # Look for product images
                    image_selectors = [
                        ".product-image", ".main-image", "[itemprop='image']", 
                        ".product-img img", ".item-image", ".thumbnail"
                    ]
                    
                    for selector in image_selectors:
                        elements = soup.select(selector)
                        if elements:
                            improved_selectors[prop] = selector
                            break
                            
                    # If still not found, look for largest image in each item
                    if prop not in improved_selectors:
                        # Find all images and sort by size if dimensions available
                        all_images = soup.find_all("img")
                        
                        # Check for data-src attribute (lazy loading)
                        if all_images:
                            has_data_src = any(img.has_attr('data-src') for img in all_images)
                            if has_data_src:
                                improved_selectors[prop] = "img[data-src]"
                            else:
                                improved_selectors[prop] = "img"
                
                elif prop == "url":
                    # Look for links in combination with title
                    title_selector = existing_selectors.get("title", "")
                    if title_selector:
                        title_elements = soup.select(title_selector)
                        for title_el in title_elements:
                            # Check if title is inside a link
                            parent_link = title_el.find_parent("a")
                            if parent_link:
                                improved_selectors[prop] = self._get_best_selector(parent_link)
                                break
                                
                            # Check if there's a link nearby
                            next_link = title_el.find_next("a")
                            if next_link and next_link.get_text().strip() in ["View", "Details", "More"]:
                                improved_selectors[prop] = self._get_best_selector(next_link)
                                break
                    
                    # If still not found, try product link patterns
                    if prop not in improved_selectors:
                        link_selectors = [
                            "a.product-link", "a.title-link", "a.item-link", 
                            ".product-title a", ".item-title a", 
                            "[itemprop='url']", ".product-card a"
                        ]
                        
                        for selector in link_selectors:
                            elements = soup.select(selector)
                            if elements:
                                improved_selectors[prop] = selector
                                break
                
                else:
                    # For other properties, try property-specific selectors
                    selectors_to_try = [
                        f".{prop}", f"#{prop}", f"[itemprop='{prop}']",
                        f".{prop}-value", f"#{prop}-value", f"[data-{prop}]",
                        f".product-{prop}", f".item-{prop}"
                    ]
                    
                    for selector in selectors_to_try:
                        elements = soup.select(selector)
                        if elements:
                            improved_selectors[prop] = selector
                            break
                    
                    # If still not found, look for labels
                    if prop not in improved_selectors:
                        label_pattern = re.compile(f"{prop}[:\\s]", re.IGNORECASE)
                        labels = soup.find_all(text=label_pattern)
                        
                        for label in labels:
                            parent = label.parent
                            if parent:
                                next_sibling = parent.find_next_sibling()
                                if next_sibling:
                                    improved_selectors[prop] = f"{self._get_best_selector(parent)} + *"
                                    break
            
            return improved_selectors
            
        except Exception as e:
            logging.error(f"Error improving property selectors: {str(e)}")
            return {}

    def _optimize_selectors_for_performance(self, selectors: Dict[str, str]) -> Dict[str, str]:
        """
        Optimize selectors for better performance.
        
        Args:
            selectors: Dictionary of current selectors
            
        Returns:
            Dictionary of optimized selectors
        """
        optimized = {}
        
        try:
            # Process each selector
            for prop, selector in selectors.items():
                # Skip empty selectors
                if not selector:
                    continue
                    
                # Optimize selectors by making them more specific/efficient
                
                # 1. Prefer ID selectors when possible
                if "#" in selector:
                    # Extract just the ID part for maximum performance
                    id_match = re.search(r'#([a-zA-Z0-9_-]+)', selector)
                    if id_match:
                        optimized[prop] = f"#{id_match.group(1)}"
                        continue
                
                # 2. Simplify complex selectors when possible
                if " > " in selector:
                    # Keep the parent and direct child for structure but simplify if possible
                    parts = selector.split(" > ")
                    if len(parts) > 2:
                        # Keep first and last part for structural integrity
                        optimized[prop] = f"{parts[0]} > {parts[-1]}"
                        continue
                
                # 3. Optimize descendant selectors
                if " " in selector and ">" not in selector:
                    # If there are multiple space-separated parts, try to simplify
                    parts = selector.split(" ")
                    if len(parts) > 2:
                        # Keep first and last part
                        optimized[prop] = f"{parts[0]} {parts[-1]}"
                        continue
                
                # 4. Reduce unnecessary class combinations
                if "." in selector:
                    # If a selector has multiple classes on the same element (.class1.class2)
                    # keep only the most specific one
                    class_parts = re.findall(r'\.([a-zA-Z0-9_-]+)', selector)
                    if len(class_parts) > 1:
                        # Find the most specific class (usually the longest)
                        most_specific = max(class_parts, key=len)
                        # Rebuild the selector with just the tag and most specific class
                        tag_part = selector.split(".")[0]
                        optimized[prop] = f"{tag_part}.{most_specific}"
                        continue
                
                # If no optimizations applied, keep the original
                optimized[prop] = selector
                
        except Exception as e:
            logging.error(f"Error optimizing selectors: {str(e)}")
            # Return original selectors on error
            return selectors
            
        return optimized
        
    async def optimize_templates_for_domain(self, domain: str) -> Dict[str, Any]:
        """
        Apply domain-specific optimizations to templates.
        
        Args:
            domain: The domain to optimize templates for
            
        Returns:
            Dictionary with optimization results
        """
        try:
            # Find all templates for this domain
            templates = self.template_storage.find_templates_for_domain(domain)
            
            if not templates:
                return {
                    "success": False,
                    "domain": domain,
                    "reason": "No templates found for domain"
                }
                
            # Get domain intelligence information
            domain_info = await self.domain_intelligence.analyze_domain(domain)
            
            # Track templates that were optimized
            optimized_templates = []
            optimization_actions = []
            
            # Apply different optimizations based on domain type
            if domain_info.get("site_type"):
                site_type = domain_info.get("site_type")
                
                # Apply e-commerce specific optimizations
                if site_type == "ecommerce":
                    for template in templates:
                        was_optimized = False
                        
                        # Ensure essential e-commerce properties have selectors
                        selectors = template.get("selectors", {})
                        properties = template.get("properties", [])
                        
                        # Add essential ecommerce properties if missing
                        ecommerce_properties = ["price", "image", "title", "description", "sku"]
                        for prop in ecommerce_properties:
                            if prop not in selectors and prop not in properties:
                                properties.append(prop)
                                was_optimized = True
                        
                        # Add common ecommerce selectors if missing
                        if "price" not in selectors:
                            selectors["price"] = ".price, [itemprop='price'], .product-price, .current-price"
                            was_optimized = True
                        
                        if "add_to_cart" not in selectors:
                            selectors["add_to_cart"] = ".add-to-cart, .buy-now, [data-action='add-to-cart']"
                            was_optimized = True
                            
                        if "sku" not in selectors:
                            selectors["sku"] = "[itemprop='sku'], .sku, .product-sku, .product-code"
                            was_optimized = True
                        
                        # Add product schema extraction
                        if "extraction_schema" not in template:
                            template["extraction_schema"] = "Product"
                            was_optimized = True
                            
                        if was_optimized:
                            template["selectors"] = selectors
                            template["properties"] = properties
                            template["updated_at"] = self.template_storage.get_current_timestamp()
                            template["domain_optimized"] = True
                            self.template_storage.save_template(template)
                            optimized_templates.append(template.get("id"))
                            optimization_actions.append("ecommerce_optimization")
                
                # Apply real estate specific optimizations
                elif site_type == "real_estate":
                    for template in templates:
                        was_optimized = False
                        
                        # Ensure essential real estate properties have selectors
                        selectors = template.get("selectors", {})
                        properties = template.get("properties", [])
                        
                        # Add essential real estate properties if missing
                        realestate_properties = ["price", "address", "bedrooms", "bathrooms", "sqft", "images"]
                        for prop in realestate_properties:
                            if prop not in selectors and prop not in properties:
                                properties.append(prop)
                                was_optimized = True
                        
                        # Add common real estate selectors if missing
                        if "price" not in selectors:
                            selectors["price"] = ".price, [itemprop='price'], .listing-price, .home-price"
                            was_optimized = True
                        
                        if "address" not in selectors:
                            selectors["address"] = ".address, [itemprop='address'], .listing-address, .property-address"
                            was_optimized = True
                            
                        if "bedrooms" not in selectors:
                            selectors["bedrooms"] = ".beds, .bedrooms, [data-label='bedrooms'], [itemprop='numberOfBedrooms']"
                            was_optimized = True
                            
                        if "bathrooms" not in selectors:
                            selectors["bathrooms"] = ".baths, .bathrooms, [data-label='bathrooms'], [itemprop='numberOfBathroomsTotal']"
                            was_optimized = True
                            
                        if "sqft" not in selectors:
                            selectors["sqft"] = ".sqft, .square-feet, [data-label='sqft'], [itemprop='floorSize']"
                            was_optimized = True
                            
                        # Add real estate schema extraction
                        if "extraction_schema" not in template:
                            template["extraction_schema"] = "RealEstateListing"
                            was_optimized = True
                            
                        if was_optimized:
                            template["selectors"] = selectors
                            template["properties"] = properties
                            template["updated_at"] = self.template_storage.get_current_timestamp()
                            template["domain_optimized"] = True
                            self.template_storage.save_template(template)
                            optimized_templates.append(template.get("id"))
                            optimization_actions.append("real_estate_optimization")
                
                # Apply news/blog specific optimizations
                elif site_type in ["news", "blog"]:
                    for template in templates:
                        was_optimized = False
                        
                        # Ensure essential news/blog properties have selectors
                        selectors = template.get("selectors", {})
                        properties = template.get("properties", [])
                        
                        # Add essential news/blog properties if missing
                        news_properties = ["title", "author", "date", "content", "category"]
                        for prop in news_properties:
                            if prop not in selectors and prop not in properties:
                                properties.append(prop)
                                was_optimized = True
                        
                        # Add common news/blog selectors if missing
                        if "author" not in selectors:
                            selectors["author"] = ".author, [itemprop='author'], .byline, .post-author"
                            was_optimized = True
                        
                        if "date" not in selectors:
                            selectors["date"] = ".date, [itemprop='datePublished'], .post-date, .published-date, .timestamp"
                            was_optimized = True
                            
                        if "content" not in selectors:
                            selectors["content"] = ".content, [itemprop='articleBody'], .post-content, .article-body"
                            was_optimized = True
                            
                        if "category" not in selectors:
                            selectors["category"] = ".category, [itemprop='articleSection'], .post-category, .article-category"
                            was_optimized = True
                            
                        # Add news/article schema extraction
                        if "extraction_schema" not in template:
                            template["extraction_schema"] = "NewsArticle"
                            was_optimized = True
                            
                        if was_optimized:
                            template["selectors"] = selectors
                            template["properties"] = properties
                            template["updated_at"] = self.template_storage.get_current_timestamp()
                            template["domain_optimized"] = True
                            self.template_storage.save_template(template)
                            optimized_templates.append(template.get("id"))
                            optimization_actions.append("news_blog_optimization")
            
            # Apply JavaScript detection optimizations if site uses JS heavily
            if domain_info.get("uses_javascript", False):
                for template in templates:
                    was_optimized = False
                    
                    # Set JavaScript handling flags
                    if "requires_javascript" not in template:
                        template["requires_javascript"] = True
                        was_optimized = True
                        
                    if "wait_for_selectors" not in template:
                        # Add selectors to wait for before extraction
                        template["wait_for_selectors"] = [
                            next(iter(template.get("selectors", {}).values()), "body")
                        ]
                        was_optimized = True
                    
                    if "wait_time_ms" not in template:
                        # Add a wait time for JS-heavy sites
                        template["wait_time_ms"] = 1000
                        was_optimized = True
                    
                    # Look for lazy-loaded images
                    selectors = template.get("selectors", {})
                    if "image" in selectors and "data-src" not in selectors["image"]:
                        selectors["image"] = f"{selectors['image']}, [data-src]"
                        template["selectors"] = selectors
                        was_optimized = True
                    
                    if was_optimized:
                        template["updated_at"] = self.template_storage.get_current_timestamp()
                        template["js_optimized"] = True
                        self.template_storage.save_template(template)
                        if template.get("id") not in optimized_templates:
                            optimized_templates.append(template.get("id"))
                        optimization_actions.append("javascript_optimization")
                    
            # Apply pagination optimizations if site has pagination
            if domain_info.get("has_pagination", False):
                for template in templates:
                    was_optimized = False
                    
                    # Add pagination selectors if missing
                    if "pagination" not in template:
                        template["pagination"] = {
                            "next_page_selector": ".next-page, .pagination .next, .pagination a[rel='next']",
                            "page_count_selector": ".pagination, .pages, .page-numbers",
                            "current_page_selector": ".pagination .current, .current-page, .active-page",
                            "detect_infinite_scroll": domain_info.get("uses_infinite_scroll", False)
                        }
                        was_optimized = True
                    
                    if was_optimized:
                        template["updated_at"] = self.template_storage.get_current_timestamp()
                        template["pagination_optimized"] = True
                        self.template_storage.save_template(template)
                        if template.get("id") not in optimized_templates:
                            optimized_templates.append(template.get("id"))
                        optimization_actions.append("pagination_optimization")
            
            # Return optimization results
            if not optimized_templates:
                return {
                    "success": True,
                    "domain": domain,
                    "templates_optimized": 0,
                    "message": "Templates already optimized or no suitable optimizations available"
                }
            else:
                return {
                    "success": True,
                    "domain": domain,
                    "templates_optimized": len(optimized_templates),
                    "template_ids": optimized_templates,
                    "site_type": domain_info.get("site_type", "unknown"),
                    "optimization_actions": list(set(optimization_actions))
                }
                
        except Exception as e:
            logging.error(f"Error optimizing templates for domain {domain}: {str(e)}")
            return {
                "success": False,
                "domain": domain,
                "error": f"Error optimizing templates: {str(e)}"
            }

    async def evaluate_all_approaches(self, url: str, user_intent: str) -> Dict[str, Any]:
        """
        Evaluate all available approaches in parallel and determine the best strategy
        
        Args:
            url: The website URL to analyze
            user_intent: The user's data extraction intent
            
        Returns:
            Dictionary with the best approach and evaluation results
        """
        import asyncio
        
        # Run all evaluations in parallel
        tasks = [
            self._evaluate_search_viability(url, user_intent),
            self._evaluate_sitemap_viability(url),
            self._evaluate_direct_crawl_viability(url, user_intent)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle any exceptions
        approach_results = {}
        
        # Search approach results
        if isinstance(results[0], dict):
            approach_results["search"] = results[0]
        else:
            approach_results["search"] = {"viable": False, "confidence": 0.0, "reason": f"Error: {str(results[0])}"}
            
        # Sitemap approach results
        if isinstance(results[1], dict):
            approach_results["sitemap"] = results[1]
        else:
            approach_results["sitemap"] = {"viable": False, "confidence": 0.0, "reason": f"Error: {str(results[1])}"}
            
        # Direct crawl approach results
        if isinstance(results[2], dict):
            approach_results["direct_crawl"] = results[2]
        else:
            approach_results["direct_crawl"] = {"viable": False, "confidence": 0.0, "reason": f"Error: {str(results[2])}"}
        
        # Select the best approach based on viability and confidence
        viable_approaches = {k: v for k, v in approach_results.items() if v.get("viable", False)}
        
        if not viable_approaches:
            # If no approach is viable, fall back to direct crawl with lowest confidence
            best_approach = "direct_crawl"
            logging.warning("No viable approaches found, falling back to direct crawl")
        else:
            # Select the approach with the highest confidence
            best_approach = max(viable_approaches.items(), key=lambda x: x[1].get("confidence", 0))[0]
            
        return {
            "best_approach": best_approach,
            "approach_results": approach_results,
            "recommendation": f"Using {best_approach} approach with confidence {approach_results[best_approach].get('confidence', 0):.2f}"
        }

    def apply_domain_specific_optimizations(self, search_terms: Dict[str, Any], url: str) -> Dict[str, Any]:
        """
        Apply domain-specific optimizations to search terms.
        
        Args:
            search_terms: Dictionary of search terms organized by field type
            url: URL of the website being scraped
            
        Returns:
            Optimized search terms dictionary
        """
        try:
            # Extract the domain from the URL
            domain = urlparse(url).netloc
            
            # Get domain intelligence for this domain
            domain_info = self.domain_intelligence.get_domain_info(domain)
            
            if not domain_info:
                # No specific intelligence for this domain
                return search_terms
                
            # Clone the search terms to avoid modifying the original
            optimized_terms = {k: list(v) for k, v in search_terms.items()}
            
            # Apply domain-specific term translations if available
            if 'term_translations' in domain_info:
                for category in optimized_terms:
                    if category in domain_info['term_translations']:
                        translations = domain_info['term_translations'][category]
                        new_terms = []
                        
                        for term in optimized_terms[category]:
                            # Check if we have a direct translation for this term
                            if term in translations:
                                new_terms.append(translations[term])
                            else:
                                # Check for partial matches
                                for original, translated in translations.items():
                                    if original in term:
                                        term = term.replace(original, translated)
                                new_terms.append(term)
                                
                        optimized_terms[category] = new_terms
            
            # Apply term formatting rules if available
            if 'term_formatting' in domain_info:
                for category, format_rules in domain_info['term_formatting'].items():
                    if category in optimized_terms and optimized_terms[category]:
                        # Apply formatting rules to each term
                        formatted_terms = []
                        for term in optimized_terms[category]:
                            if 'prefix' in format_rules:
                                term = f"{format_rules['prefix']}{term}"
                            if 'suffix' in format_rules:
                                term = f"{term}{format_rules['suffix']}"
                            if 'replace_spaces' in format_rules:
                                term = term.replace(' ', format_rules['replace_spaces'])
                            if 'case' in format_rules:
                                if format_rules['case'] == 'upper':
                                    term = term.upper()
                                elif format_rules['case'] == 'lower':
                                    term = term.lower()
                                elif format_rules['case'] == 'title':
                                    term = term.title()
                            
                            formatted_terms.append(term)
                        
                        optimized_terms[category] = formatted_terms
            
            # Apply domain-specific field mappings
            if 'field_mappings' in domain_info:
                for src_field, dest_field in domain_info['field_mappings'].items():
                    if src_field in optimized_terms and optimized_terms[src_field]:
                        # Copy terms from source field to destination field
                        if dest_field not in optimized_terms:
                            optimized_terms[dest_field] = []
                        optimized_terms[dest_field].extend(optimized_terms[src_field])
            
            # Apply special case logic for real estate websites
            if domain_info.get('domain_type') == 'real_estate':
                self._apply_real_estate_optimizations(optimized_terms, domain_info)
                
            # Apply special case logic for e-commerce websites
            elif domain_info.get('domain_type') == 'ecommerce':
                self._apply_ecommerce_optimizations(optimized_terms, domain_info)
                
            # Remove empty categories
            optimized_terms = {k: v for k, v in optimized_terms.items() if v}
            
            return optimized_terms
            
        except Exception as e:
            logging.error(f"Error applying domain-specific optimizations: {str(e)}")
            # Return original search terms if optimization fails
            return search_terms
    
    def _apply_real_estate_optimizations(self, search_terms: Dict[str, Any], domain_info: Dict[str, Any]) -> None:
        """
        Apply real estate specific optimizations to search terms.
        
        Args:
            search_terms: Dictionary of search terms organized by field type
            domain_info: Domain intelligence information
        """
        # Handle location formatting for real estate sites
        if 'location' in search_terms and search_terms['location']:
            # Check if we need to separate city and state
            location_parts = []
            for location in search_terms['location']:
                if ',' in location:
                    # Split at comma and clean up each part
                    parts = [part.strip() for part in location.split(',')]
                    
                    # Some real estate sites need city and state in separate fields
                    if domain_info.get('separate_city_state', False):
                        if 'city' not in search_terms:
                            search_terms['city'] = []
                        if 'state' not in search_terms:
                            search_terms['state'] = []
                            
                        if len(parts) >= 1:
                            search_terms['city'].append(parts[0])
                        if len(parts) >= 2:
                            search_terms['state'].append(parts[1])
                    else:
                        location_parts.append(location)
                else:
                    location_parts.append(location)
                    
            search_terms['location'] = location_parts
            
        # Convert price range to min_price and max_price if needed
        if 'price_range' in search_terms and search_terms['price_range']:
            if domain_info.get('split_price_range', False):
                if 'min_price' not in search_terms:
                    search_terms['min_price'] = []
                if 'max_price' not in search_terms:
                    search_terms['max_price'] = []
                    
                for price_range in search_terms['price_range']:
                    if '-' in price_range:
                        min_price, max_price = price_range.split('-', 1)
                        search_terms['min_price'].append(min_price.strip())
                        search_terms['max_price'].append(max_price.strip())
                    else:
                        # If no range, use as minimum price
                        search_terms['min_price'].append(price_range.strip())
    
    def _apply_ecommerce_optimizations(self, search_terms: Dict[str, Any], domain_info: Dict[str, Any]) -> None:
        """
        Apply e-commerce specific optimizations to search terms.
        
        Args:
            search_terms: Dictionary of search terms organized by field type
            domain_info: Domain intelligence information
        """
        # For e-commerce sites, prioritize product names and types
        general_terms = []
        product_terms = []
        
        if 'general' in search_terms:
            for term in search_terms['general']:
                product_match = False
                
                # Check if term matches known product categories
                for category in domain_info.get('product_categories', []):
                    if category.lower() in term.lower():
                        product_terms.append(term)
                        product_match = True
                        break
                
                if not product_match:
                    general_terms.append(term)
            
            # Update general terms list, prioritizing product terms
            if product_terms:
                search_terms['general'] = product_terms + general_terms
                
        # Handle brand names specially if site supports brand filtering
        if domain_info.get('supports_brand_filter', False):
            brand_terms = []
            remaining_terms = []
            
            for term in search_terms.get('general', []):
                brand_match = False
                
                # Check if term matches known brands
                for brand in domain_info.get('known_brands', []):
                    if brand.lower() in term.lower():
                        brand_terms.append(brand)  # Use canonical brand name
                        brand_match = True
                        break
                
                if not brand_match:
                    remaining_terms.append(term)
            
            if brand_terms:
                search_terms['brand'] = brand_terms
                search_terms['general'] = remaining_terms