# search_automation.py - Search form detection and automation for SmartScrape

import asyncio
from bs4 import BeautifulSoup
import re
import json
import logging
from urllib.parse import urlparse, urljoin, quote, unquote, parse_qs
from playwright.async_api import async_playwright
from typing import Dict, List, Any, Optional, Union, Tuple
import time
from tenacity import retry, stop_after_attempt, wait_exponential

# Import enhanced utils
from utils.html_utils import parse_html, extract_text_fast, find_by_xpath, select_with_css
from utils.retry_utils import with_exponential_backoff
from utils.http_utils import fetch_html

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SearchAutomation")

# Import pagination handler (no circular dependency here)
from components.pagination_handler import PaginationHandler

# Import playwright-stealth
try:
    from playwright_stealth import stealth_async
except ImportError:
    logger.warning("playwright_stealth not installed, stealth mode will not be available")
    # Define a dummy function as a fallback
    async def stealth_async(page):
        logger.warning("Using dummy stealth_async function. Install playwright_stealth for better results.")
        return

# In components/search_automation.py - add this import
from crawl4ai import CrawlerRunConfig, AsyncWebCrawler, CacheMode

import asyncio
import re
import logging
from typing import Dict, List, Any
from urllib.parse import urljoin, urlparse, parse_qs, urlunparse, urlencode, quote_plus

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from crawl4ai import AsyncWebCrawler

# Import stealth plugin for avoiding detection
from playwright_stealth import stealth_async
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levellevel)s - %(message)s')
logger = logging.getLogger("SearchAutomation")

# Add new APIParameterAnalyzer class for task 3.3.4
class APIParameterAnalyzer:
    """
    Analyzes API parameters from network requests to identify search-related parameters
    and understand API behavior for search functionality.
    
    This class:
    - Monitors and captures API requests during search operations
    - Identifies search-related parameters in API requests
    - Analyzes parameter patterns across multiple requests
    - Builds a model of API parameter influence on search results
    - Supports both RESTful APIs and GraphQL
    """
    
    def __init__(self):
        self.logger = logging.getLogger("APIParameterAnalyzer")
        # Store parameter info by API endpoint
        self.endpoint_parameters = {}
        # Track search terms and their associated parameters
        self.search_parameter_mapping = {}
        # Common search parameter names
        self.search_param_indicators = [
            'q', 'query', 'search', 'keyword', 'keywords', 'term', 'text',
            'input', 'find', 'filter', 'where', 's', 'kw', 'searchTerm'
        ]
        # Track GraphQL operations by name
        self.graphql_operations = {}
        
    async def analyze_network_requests(self, page, search_term: str) -> Dict[str, Any]:
        """
        Monitor and analyze network requests during a search operation.
        
        Args:
            page: Playwright page object
            search_term: Search term being used
            
        Returns:
            Dictionary with analysis results
        """
        self.logger.info(f"Starting API request analysis for search term: {search_term}")
        
        # Store all captured requests
        requests = []
        
        # Setup network request monitoring
        async def on_request(request):
            # Capture API requests (XHR, Fetch, GraphQL)
            if (request.resource_type in ['xhr', 'fetch'] or 
                'graphql' in request.url.lower() or
                request.method == 'POST'):
                try:
                    request_data = {
                        'url': request.url,
                        'method': request.method,
                        'resource_type': request.resource_type,
                        'post_data': request.post_data,
                        'headers': request.headers,
                        'timestamp': time.time()
                    }
                    requests.append(request_data)
                except Exception as e:
                    self.logger.warning(f"Error capturing request: {str(e)}")
        
        # Set up the listener
        page.on('request', on_request)
        
        # Let the page run for a short time to capture requests
        await asyncio.sleep(3)
        
        # Process and analyze the captured requests
        api_endpoints = await self._process_requests(requests, search_term)
        
        # Clean up
        page.remove_listener('request', on_request)
        
        return {
            'success': True,
            'api_endpoints': api_endpoints,
            'request_count': len(requests)
        }
    
    async def _process_requests(self, requests: List[Dict[str, Any]], search_term: str) -> List[Dict[str, Any]]:
        """Process captured network requests to identify API endpoints and parameters"""
        endpoints = []
        
        for request in requests:
            try:
                url = request['url']
                method = request['method']
                post_data = request['post_data']
                
                # Parse URL
                parsed_url = urlparse(url)
                endpoint = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
                
                # Get query parameters from URL
                query_params = parse_qs(parsed_url.query)
                
                # Initialize endpoint info
                endpoint_info = {
                    'endpoint': endpoint,
                    'method': method,
                    'parameters': {},
                    'search_related': False,
                    'confidence_score': 0
                }
                
                # Process URL query parameters
                for param, values in query_params.items():
                    param_value = values[0] if values else ''
                    endpoint_info['parameters'][param] = param_value
                    
                    # Check if this parameter appears to be search-related
                    if any(indicator in param.lower() for indicator in self.search_param_indicators):
                        endpoint_info['search_related'] = True
                        endpoint_info['confidence_score'] += 30
                        
                    # Check if search term is in the parameter value
                    if search_term.lower() in param_value.lower():
                        endpoint_info['search_related'] = True
                        endpoint_info['confidence_score'] += 50
                        # Store this parameter as likely search term parameter
                        self._add_search_parameter_mapping(endpoint, param, search_term, param_value)
                
                # Process POST data if available
                if post_data:
                    post_params = self._extract_post_parameters(post_data)
                    
                    # For GraphQL, handle specially
                    if 'graphql' in endpoint.lower() or (
                            post_params and ('query' in post_params or 'operationName' in post_params)):
                        graphql_info = self._analyze_graphql_request(post_data, search_term)
                        if graphql_info:
                            # Merge GraphQL info into endpoint info
                            endpoint_info.update(graphql_info)
                            if graphql_info.get('search_related', False):
                                endpoint_info['search_related'] = True
                                endpoint_info['confidence_score'] += graphql_info.get('confidence_score', 0)
                    else:
                        # Regular POST parameters
                        for param, value in post_params.items():
                            endpoint_info['parameters'][param] = value
                            
                            # Check if this parameter appears to be search-related
                            if any(indicator in param.lower() for indicator in self.search_param_indicators):
                                endpoint_info['search_related'] = True
                                endpoint_info['confidence_score'] += 30
                                
                            # Check if search term is in the parameter value
                            if isinstance(value, str) and search_term.lower() in value.lower():
                                endpoint_info['search_related'] = True
                                endpoint_info['confidence_score'] += 50
                                # Store this parameter as likely search term parameter
                                self._add_search_parameter_mapping(endpoint, param, search_term, value)
                
                # Check content-type for API indicators
                content_type = request['headers'].get('content-type', '').lower()
                if 'json' in content_type or 'graphql' in content_type:
                    endpoint_info['confidence_score'] += 10
                
                # Only include endpoints with reasonable confidence
                if endpoint_info['confidence_score'] > 20:
                    # Store this endpoint in our collection
                    self._update_endpoint_parameters(endpoint, endpoint_info)
                    endpoints.append(endpoint_info)
            
            except Exception as e:
                self.logger.warning(f"Error processing request: {str(e)}")
        
        # Sort endpoints by confidence score
        endpoints.sort(key=lambda x: x['confidence_score'], reverse=True)
        
        return endpoints
    
    def _extract_post_parameters(self, post_data: str) -> Dict[str, Any]:
        """Extract parameters from POST data, handling different formats"""
        if not post_data:
            return {}
            
        try:
            # Try to parse as JSON
            return json.loads(post_data)
        except json.JSONDecodeError:
            try:
                # Try to parse as form data
                params = {}
                for param in post_data.split('&'):
                    if '=' in param:
                        key, value = param.split('=', 1)
                        params[key] = unquote(value)
                return params
            except Exception:
                # If all parsing fails, return empty dict
                return {}
    
    def _analyze_graphql_request(self, post_data: str, search_term: str) -> Dict[str, Any]:
        """Analyze a GraphQL request for search-related operations"""
        try:
            # Parse GraphQL request
            data = json.loads(post_data)
            
            # GraphQL specific info
            graphql_info = {
                'type': 'graphql',
                'search_related': False,
                'confidence_score': 0
            }
            
            # Extract GraphQL query/mutation
            query = data.get('query', '')
            operation_name = data.get('operationName', '')
            variables = data.get('variables', {})
            
            graphql_info['operation_name'] = operation_name
            graphql_info['variables'] = variables
            
            # Look for search-related terms in the query
            search_terms = ['search', 'query', 'find', 'filter', 'where']
            if any(term in query.lower() for term in search_terms):
                graphql_info['search_related'] = True
                graphql_info['confidence_score'] += 40
            
            # Look for search-related operation names
            if operation_name and any(term in operation_name.lower() for term in search_terms):
                graphql_info['search_related'] = True
                graphql_info['confidence_score'] += 30
            
            # Check variables for search term
            for var_name, var_value in variables.items():
                if any(term in var_name.lower() for term in self.search_param_indicators):
                    graphql_info['search_related'] = True
                    graphql_info['confidence_score'] += 20
                
                # Check if the search term is in any variable value
                if isinstance(var_value, str) and search_term.lower() in var_value.lower():
                    graphql_info['search_related'] = True
                    graphql_info['confidence_score'] += 50
                    # Track this variable as a potential search parameter
                    if operation_name:
                        self._add_graphql_operation(operation_name, var_name, search_term)
            
            return graphql_info
            
        except Exception as e:
            self.logger.warning(f"Error analyzing GraphQL request: {str(e)}")
            return {}
    
    def _update_endpoint_parameters(self, endpoint: str, endpoint_info: Dict[str, Any]):
        """Update our knowledge about an endpoint's parameters"""
        if endpoint not in self.endpoint_parameters:
            self.endpoint_parameters[endpoint] = {
                'parameters': {},
                'search_related': False,
                'confidence': 0,
                'request_count': 0
            }
        
        # Update endpoint info
        self.endpoint_parameters[endpoint]['request_count'] += 1
        self.endpoint_parameters[endpoint]['confidence'] = max(
            self.endpoint_parameters[endpoint]['confidence'],
            endpoint_info['confidence_score']
        )
        self.endpoint_parameters[endpoint]['search_related'] = (
            self.endpoint_parameters[endpoint]['search_related'] or 
            endpoint_info['search_related']
        )
        
        # Update parameter knowledge
        for param, value in endpoint_info['parameters'].items():
            if param not in self.endpoint_parameters[endpoint]['parameters']:
                self.endpoint_parameters[endpoint]['parameters'][param] = {
                    'values': [],
                    'is_search_param': False,
                    'frequency': 0
                }
            
            # Track parameter values
            param_info = self.endpoint_parameters[endpoint]['parameters'][param]
            if value not in param_info['values']:
                param_info['values'].append(value)
            param_info['frequency'] += 1
            
            # Check if this might be a search parameter
            if any(indicator in param.lower() for indicator in self.search_param_indicators):
                param_info['is_search_param'] = True
    
    def _add_search_parameter_mapping(self, endpoint: str, param: str, search_term: str, value: str):
        """Track mapping between search terms and parameter values"""
        key = f"{endpoint}:{param}"
        if key not in self.search_parameter_mapping:
            self.search_parameter_mapping[key] = []
        
        # Add this mapping
        self.search_parameter_mapping[key].append({
            'search_term': search_term,
            'param_value': value
        })
    
    def _add_graphql_operation(self, operation_name: str, var_name: str, search_term: str):
        """Track GraphQL operations and their search-related variables"""
        if operation_name not in self.graphql_operations:
            self.graphql_operations[operation_name] = {
                'search_variables': {},
                'usage_count': 0
            }
        
        # Track the variable
        if var_name not in self.graphql_operations[operation_name]['search_variables']:
            self.graphql_operations[operation_name]['search_variables'][var_name] = []
        
        self.graphql_operations[operation_name]['search_variables'][var_name].append(search_term)
        self.graphql_operations[operation_name]['usage_count'] += 1
    
    async def generate_api_request(self, endpoint_info: Dict[str, Any], search_term: str) -> Dict[str, Any]:
        """
        Generate an API request based on analyzed parameters.
        
        Args:
            endpoint_info: Information about the API endpoint
            search_term: Search term to use
            
        Returns:
            Dictionary with request details
        """
        try:
            endpoint = endpoint_info['endpoint']
            method = endpoint_info['method']
            parameters = endpoint_info['parameters'].copy()
            
            # For GraphQL, handle specially
            if endpoint_info.get('type') == 'graphql':
                return await self._generate_graphql_request(endpoint_info, search_term)
            
            # Find search parameters
            search_param = None
            for param, value in parameters.items():
                if any(indicator in param.lower() for indicator in self.search_param_indicators):
                    search_param = param
                    parameters[param] = search_term
                    break
            
            # If no obvious search parameter found but we have confidence this is a search API
            if not search_param and endpoint_info['search_related']:
                # Look at our stored mappings for this endpoint
                for param in parameters.keys():
                    key = f"{endpoint}:{param}"
                    if key in self.search_parameter_mapping:
                        search_param = param
                        parameters[param] = search_term
                        break
            
            # Build request based on method
            if method == 'GET':
                query_string = urlencode(parameters)
                url = f"{endpoint}?{query_string}"
                
                return {
                    'method': 'GET',
                    'url': url,
                    'headers': {
                        'Accept': 'application/json',
                        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'
                    }
                }
                
            elif method == 'POST':
                # Determine if this is likely JSON or form data
                if any(isinstance(v, dict) for v in parameters.values()):
                    # Probably JSON
                    content_type = 'application/json'
                    post_data = json.dumps(parameters)
                else:
                    # Probably form data
                    content_type = 'application/x-www-form-urlencoded'
                    post_data = urlencode(parameters)
                
                return {
                    'method': 'POST',
                    'url': endpoint,
                    'headers': {
                        'Content-Type': content_type,
                        'Accept': 'application/json',
                        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'
                    },
                    'data': post_data
                }
            
            return {'error': 'Unable to generate API request', 'endpoint_info': endpoint_info}
            
        except Exception as e:
            self.logger.error(f"Error generating API request: {str(e)}")
            return {'error': str(e), 'endpoint_info': endpoint_info}
    
    async def _generate_graphql_request(self, endpoint_info: Dict[str, Any], search_term: str) -> Dict[str, Any]:
        """Generate a GraphQL request for a search operation"""
        try:
            endpoint = endpoint_info['endpoint']
            operation_name = endpoint_info.get('operation_name', '')
            variables = endpoint_info.get('variables', {}).copy()
            
            # Find which variable likely contains the search term
            search_var = None
            for var_name in variables.keys():
                if any(term in var_name.lower() for term in self.search_param_indicators):
                    search_var = var_name
                    variables[var_name] = search_term
                    break
            
            # If no obvious search variable found but we have information about this operation
            if not search_var and operation_name in self.graphql_operations:
                search_vars = self.graphql_operations[operation_name]['search_variables']
                if search_vars:
                    # Use the first identified search variable
                    search_var = next(iter(search_vars.keys()))
                    variables[search_var] = search_term
            
            # If we still don't have a search variable but this is search-related, 
            # try the most common variable names
            if not search_var and endpoint_info.get('search_related', False):
                for common_var in ['query', 'search', 'term', 'keyword', 'q']:
                    if common_var in variables:
                        search_var = common_var
                        variables[common_var] = search_term
                        break
            
            # Build GraphQL request
            graphql_request = {
                'query': endpoint_info.get('query', ''),
                'variables': variables
            }
            
            if operation_name:
                graphql_request['operationName'] = operation_name
            
            return {
                'method': 'POST',
                'url': endpoint,
                'headers': {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'
                },
                'data': json.dumps(graphql_request)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating GraphQL request: {str(e)}")
            return {'error': str(e), 'endpoint_info': endpoint_info}
    
    async def analyze_api_response(self, response_content: str, search_term: str) -> Dict[str, Any]:
        """
        Analyze API response to understand result structure and extract search results.
        
        Args:
            response_content: Response content from the API
            search_term: Search term that was used
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Parse JSON response
            data = json.loads(response_content)
            
            # Analysis results
            analysis = {
                'search_term': search_term,
                'result_count': 0,
                'results_found': False,
                'result_path': [],
                'extracted_results': []
            }
            
            # Find result arrays in the response
            result_arrays = self._find_result_arrays(data)
            
            # Score each candidate result array
            scored_arrays = []
            for path, array in result_arrays:
                if not array:  # Skip empty arrays
                    continue
                    
                score = self._score_result_array(array, search_term)
                scored_arrays.append((path, array, score))
            
            # Sort by score (descending)
            scored_arrays.sort(key=lambda x: x[2], reverse=True)
            
            # If we found potential result arrays
            if scored_arrays:
                best_path, best_array, score = scored_arrays[0]
                
                # Only consider this a valid result if score is sufficient
                if score > 10:
                    analysis['results_found'] = True
                    analysis['result_count'] = len(best_array)
                    analysis['result_path'] = best_path
                    
                    # Extract results (limit to first 20)
                    for item in best_array[:20]:
                        result = self._extract_item_properties(item)
                        if result:
                            analysis['extracted_results'].append(result)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing API response: {str(e)}")
            return {
                'search_term': search_term,
                'error': str(e),
                'results_found': False
            }
    
    def _find_result_arrays(self, data, current_path=None, max_depth=5):
        """Recursively find arrays in the data that might contain results"""
        if current_path is None:
            current_path = []
            
        # Stop at max depth to prevent stack overflow
        if len(current_path) >= max_depth:
            return []
            
        results = []
        
        if isinstance(data, list) and data:
            # This is an array - might be results
            results.append((current_path, data))
            
            # Also check for nested arrays in the first item
            if data and isinstance(data[0], (dict, list)):
                child_results = self._find_result_arrays(data[0], current_path + ['[0]'], max_depth)
                results.extend(child_results)
                
        elif isinstance(data, dict):
            # Look for keys that might indicate result arrays
            result_key_indicators = [
                'results', 'items', 'data', 'hits', 'documents', 'list',
                'properties', 'listings', 'products', 'posts', 'search'
            ]
            
            for key, value in data.items():
                # Check for array values directly under keys that suggest results
                if isinstance(value, list) and any(indicator in key.lower() for indicator in result_key_indicators):
                    results.append((current_path + [key], value))
                
                # Recurse into this value
                child_results = self._find_result_arrays(value, current_path + [key], max_depth)
                results.extend(child_results)
                
        return results
    
    def _score_result_array(self, array, search_term):
        """Score an array on how likely it is to contain search results"""
        if not array or not isinstance(array, list):
            return 0
            
        score = 0
        
        # Basic indicators that this might be results
        if len(array) > 0:
            score += 5  # Non-empty array
        if len(array) > 1:
            score += 5  # Multiple items
        if len(array) > 10:
            score += 3  # Lots of items
            
        # Check first item structure
        first_item = array[0] if array else None
        if isinstance(first_item, dict):
            # Items are objects - good sign
            score += 10
            
            # Check for common result fields
            field_indicators = [
                # Generic result fields
                'id', 'title', 'name', 'description', 'url', 'link',
                # Domain-specific fields
                'price', 'address', 'date', 'image', 'thumbnail', 'category',
                'rating', 'reviews', 'author', 'location'
            ]
            
            matched_fields = [field for field in field_indicators if field in first_item]
            score += len(matched_fields) * 2
            
            # Check if search term appears in values
            search_term_lower = search_term.lower()
            for value in first_item.values():
                if isinstance(value, str) and search_term_lower in value.lower():
                    score += 15
                    break
                    
        return score
    
    def _extract_item_properties(self, item):
        """Extract relevant properties from a result item"""
        if not isinstance(item, dict):
            return None
            
        # Common field mappings for different types of results
        field_mappings = {
            'title': ['title', 'name', 'headline', 'subject', 'product_name'],
            'description': ['description', 'summary', 'content', 'text', 'snippet', 'abstract'],
            'url': ['url', 'link', 'href', 'uri', 'web_url', 'product_url'],
            'id': ['id', 'uid', 'item_id', 'product_id', 'listing_id'],
            'image': ['image', 'thumbnail', 'photo', 'picture', 'image_url'],
            'price': ['price', 'cost', 'amount', 'value', 'price_amount'],
            'rating': ['rating', 'stars', 'score', 'rank'],
            'location': ['location', 'address', 'geo', 'place', 'area']
        }
        
        result = {}
        
        # Extract fields using mappings
        for target_field, source_fields in field_mappings.items():
            for field in source_fields:
                if field in item:
                    result[target_field] = item[field]
                    break
        
        # If we didn't find any mapped fields, just return the original item
        return result if result else item

class SearchFormDetector:
    def __init__(self, domain_intelligence=None):
        # Basic search indicators that apply to any domain
        self.search_indicators = [
            # Generic search terms
            "search", "find", "lookup", "query", "filter", "browse", "explore", 
            # Common form field names
            "q", "s", "query", "keyword", "keywords", "term", "terms", "text", "input",
            # Domain-agnostic search verbs
            "check", "discover", "hunt", "locate", "look", "seek"
        ]
        
        # Use domain intelligence if provided or create default instance
        if domain_intelligence:
            self.domain_intelligence = domain_intelligence
        else:
            # Lazy import to avoid circular dependency
            from components.domain_intelligence import DomainIntelligence
            self.domain_intelligence = DomainIntelligence()
        
        # Domain-specific indicators that can be used for specialized searches
        self.domain_indicators = {
            "real_estate": ["property", "home", "house", "apartment", "listing", "realty", "real estate",
                           "bedrooms", "bathrooms", "sqft", "square feet", "location", "address"],
            "e_commerce": ["product", "shop", "buy", "cart", "checkout", "price", "purchase", "store",
                          "item", "brand", "shipping", "category", "catalog", "sale", "discount"],
            "job_listings": ["job", "career", "employment", "position", "hiring", "resume", "cv", "salary",
                           "experience", "skills", "qualification", "apply"]
        }
        
        # Add custom domain indicators
        self.domain_indicators.update({
            "news": ["news", "article", "story", "headline", "media", "press", "publication"],
            "academic": ["paper", "journal", "research", "publication", "study", "author", "citation"],
            "forum": ["forum", "thread", "post", "topic", "message", "discussion", "comment", "reply"],
            "travel": ["destination", "hotel", "flight", "accommodation", "trip", "vacation", "booking"],
            "food": ["restaurant", "recipe", "menu", "food", "dish", "cuisine", "ingredient"],
            "health": ["doctor", "health", "medical", "clinic", "symptom", "treatment", "disease", "condition"]
        })

    @with_exponential_backoff(max_attempts=3)
    async def detect_search_forms(self, html_content, domain_type="general"):
        """
        Detect search forms in the HTML content with enhanced detection for any interface type.
        
        Args:
            html_content: HTML content of the page
            domain_type: Type of domain for specialized detection
            
        Returns:
            List of detected forms with relevance scores
        """
        # Create a BeautifulSoup object with optimized lxml parser
        soup = parse_html(html_content)
        
        # List of forms with relevance score
        detected_forms = []
        
        # Choose appropriate indicators based on domain type
        indicators = self.search_indicators.copy()
        if (domain_type in self.domain_indicators):
            indicators.extend(self.domain_indicators[domain_type])
            
        # Detect all potential search interfaces
        self._detect_standard_forms(soup, indicators, detected_forms)
        self._detect_js_search_components(soup, indicators, detected_forms)
        self._detect_custom_search_interfaces(soup, indicators, detected_forms)
            
        # Sort forms by relevance score (descending)
        detected_forms.sort(key=lambda x: x["search_relevance_score"], reverse=True)
        
        return detected_forms
    
    def _detect_standard_forms(self, soup, indicators, detected_forms):
        """Detect standard HTML form-based search interfaces using lxml optimization"""
        # Find all forms using faster lxml search
        form_elements = find_by_xpath(soup, '//form')
        
        for form in form_elements:
            form_score = 0
            form_info = {
                "id": form.get('id', ''),
                "action": form.get('action', ''),
                "method": form.get('method', 'get').lower(),
                "fields": [],
                "type": "standard_form"  # Mark this as a standard HTML form
            }
            
            # Check if form has search indicators in id/class
            form_attrs = form.get('id', '') + ' ' + ' '.join(form.get('class', []))
            if any(indicator in form_attrs.lower() for indicator in indicators):
                form_score += 10
                
            # Check form action URL for indicators
            action = form.get('action', '')
            if any(indicator in action.lower() for indicator in indicators):
                form_score += 5
                
            # Analyze input fields
            fields = []
            # Use faster lxml-based selection for input fields
            input_elements = find_by_xpath(form, './/input | .//select | .//textarea')
            
            for field in input_elements:
                field_type = field.get('type', '')
                field_name = field.get('name', '')
                field_id = field.get('id', '')
                field_placeholder = field.get('placeholder', '')
                field_attrs = ' '.join([field_name, field_id, field_placeholder, 
                                      ' '.join(field.get('class', []))])
                
                # Skip hidden and submit fields for scoring but include them in fields
                if field_type in ['hidden', 'submit', 'button']:
                    field_info = {
                        "type": field_type,
                        "name": field_name,
                        "id": field_id,
                        "placeholder": field_placeholder,
                        "possible_values": [],
                        "is_search_field": False
                    }
                    form_info["fields"].append(field_info)
                    continue
                    
                field_info = {
                    "type": field_type,
                    "name": field_name,
                    "id": field_id,
                    "placeholder": field_placeholder,
                    "possible_values": [],
                    "is_search_field": False
                }
                
                # Check if field is likely a search field
                is_search_field = any(indicator in field_attrs.lower() for indicator in indicators)
                
                # For select fields, gather possible options
                if field.name == 'select':
                    options = []
                    option_elements = find_by_xpath(field, './/option')
                    for option in option_elements:
                        option_value = option.get('value', '')
                        option_text = extract_text_fast(option)
                        if option_value or option_text:
                            options.append({
                                "value": option_value,
                                "text": option_text
                            })
                    field_info["possible_values"] = options
                    
                    # If select has options related to search indicators, increase score
                    option_texts = ' '.join([opt.get('text', '') for opt in options])
                    if any(indicator in option_texts.lower() for indicator in indicators):
                        is_search_field = True
                
                # Score each search-related field
                if is_search_field:
                    form_score += 5
                    field_info["is_search_field"] = True
                    
                    # Extra points for text inputs that are likely search boxes
                    if field_type in ['text', ''] and any(term in field_attrs.lower() 
                                                       for term in ["search", "find", "query", "keyword"]):
                        form_score += 10
                
                # Add field to form info
                form_info["fields"].append(field_info)
            
            # Verify the form has at least one input field
            if form_info["fields"]:
                # Check for submit button with search text
                submit_btns = find_by_xpath(form, './/button[@type="submit"] | .//input[@type="submit"] | .//input[@type="button"] | .//input[@type="image"]')
                for btn in submit_btns:
                    btn_text = extract_text_fast(btn)
                    btn_value = btn.get('value', '').lower()
                    btn_attrs = btn.get('id', '') + ' ' + ' '.join(btn.get('class', []))
                    
                    # If submit button has search indicators, increase score
                    if any(indicator in (btn_text + ' ' + btn_value + ' ' + btn_attrs).lower() 
                           for indicator in indicators):
                        form_score += 8
                
                # Store the form if it has a minimum score
                if form_score >= 5:
                    form_info["search_relevance_score"] = form_score
                    detected_forms.append(form_info)
                    
    def _detect_js_search_components(self, soup, indicators, detected_forms):
        """Detect JavaScript-based search interfaces like React/Angular components"""
        # Look for divs that might be JS search components
        potential_components = []
        
        # Use faster XPath to find elements with search-related attributes
        search_attributes_xpath = ' or '.join([f'contains(@class, "{ind}")' for ind in indicators])
        search_attributes_xpath = f'//*[{search_attributes_xpath} or contains(@placeholder, "search") or contains(@aria-label, "search")]'
        
        elements = find_by_xpath(soup, search_attributes_xpath)
        potential_components.extend(elements)
                
        # Group elements by their parent to identify JS components
        component_groups = {}
        for elem in potential_components:
            # Look up to 3 levels up for the parent component
            parent = elem.parent
            level = 0
            found_component = False
            
            while parent and level < 3 and not found_component:
                parent_id = parent.get('id', '')
                parent_classes = ' '.join(parent.get('class', []))
                
                # Check if parent might be a search component container
                if (parent_id or parent_classes) and any(indicator in (parent_id + ' ' + parent_classes).lower() 
                                                        for indicator in indicators + ['search', 'filter']):
                    # Generate a key for grouping
                    key = f"{parent.name}#{parent_id}#{parent_classes}"
                    if key not in component_groups:
                        component_groups[key] = {
                            "element": parent,
                            "fields": [],
                            "score": 0
                        }
                    
                    component_groups[key]["fields"].append({
                        "element": elem,
                        "type": elem.name,
                        "id": elem.get('id', ''),
                        "classes": ' '.join(elem.get('class', [])),
                        "attrs": ' '.join([
                            elem.get('id', ''),
                            ' '.join(elem.get('class', [])),
                            elem.get('placeholder', ''),
                            elem.get('aria-label', ''),
                            elem.get('data-testid', ''),
                            elem.get('role', '')
                        ]).lower()
                    })
                    component_groups[key]["score"] += 5
                    found_component = True
                    
                parent = parent.parent
                level += 1
        
        # Process each potential component group
        for key, component in component_groups.items():
            element = component["element"]
            score = component["score"]
            fields = []
            
            # Analyze fields in this component
            for field_info in component["fields"]:
                field_elem = field_info["element"]
                
                # Look for input boxes in or near this element
                input_elems = find_by_xpath(field_elem, './/input | .//select') if field_elem.name != 'input' else [field_elem]
                
                for input_elem in input_elems:
                    field_type = input_elem.get('type', '')
                    field_id = input_elem.get('id', '')
                    field_name = input_elem.get('name', '')
                    field_placeholder = input_elem.get('placeholder', '')
                    
                    # Skip hidden inputs
                    if field_type == 'hidden':
                        continue
                    
                    field = {
                        "type": field_type or input_elem.name,
                        "id": field_id,
                        "name": field_name,
                        "placeholder": field_placeholder,
                        "possible_values": [],
                        "is_search_field": True  # Assume true since it's in a search component
                    }
                    
                    # For select fields, gather options
                    if input_elem.name == 'select':
                        options = []
                        option_elements = find_by_xpath(input_elem, './/option')
                        for option in option_elements:
                            option_value = option.get('value', '')
                            option_text = extract_text_fast(option)
                            if option_value or option_text:
                                options.append({
                                    "value": option_value,
                                    "text": option_text
                                })
                        field["possible_values"] = options
                    
                    fields.append(field)
                    score += 3  # Additional score for each input field
            
            # Look for buttons that might trigger search using optimal selectors
            button_xpath = './/button | .//*[@role="button"] | .//div[contains(@class, "button")] | .//span[contains(@class, "button")]'
            buttons = find_by_xpath(element, button_xpath)
            
            has_search_button = False
            for button in buttons:
                button_text = extract_text_fast(button)
                button_attrs = ' '.join([
                    button.get('id', ''),
                    ' '.join(button.get('class', [])),
                    button.get('aria-label', ''),
                    button.get('title', '')
                ]).lower()
                
                if any(indicator in (button_text + ' ' + button_attrs) 
                       for indicator in indicators + ['submit', 'go']):
                    has_search_button = True
                    score += 8
                    break
            
            # Only include component if it has fields and reasonable score
            if fields and score >= 10:
                js_component = {
                    "id": element.get('id', ''),
                    "classes": ' '.join(element.get('class', [])),
                    "type": "js_component",
                    "fields": fields,
                    "search_relevance_score": score
                }
                detected_forms.append(js_component)
    
    def _detect_custom_search_interfaces(self, soup, indicators, detected_forms):
        """Detect custom search interfaces like search boxes without forms"""
        # Look for standalone search inputs with optimal XPath
        inputs_xpath = '//input[not(ancestor::form)]'
        input_elements = find_by_xpath(soup, inputs_xpath)
        
        for input_elem in input_elements:
            input_type = input_elem.get('type', '')
            input_attrs = ' '.join([
                input_elem.get('id', ''),
                ' '.join(input_elem.get('class', [])),
                input_elem.get('name', ''),
                input_elem.get('placeholder', ''),
                input_elem.get('aria-label', '')
            ]).lower()
            
            # Check if input has search indicators
            if input_type in ['text', 'search', ''] and any(indicator in input_attrs 
                                                         for indicator in indicators):
                # Look for nearby button (search within parent and siblings)
                parent = input_elem.parent
                button = None
                
                # Check siblings with optimal tree traversal
                next_elem = input_elem.next_sibling
                while next_elem and not button:
                    if hasattr(next_elem, 'name') and next_elem.name in ['button', 'a', 'div', 'span', 'input']:
                        button_attrs = ' '.join([
                            next_elem.get('id', '') or '',
                            ' '.join(next_elem.get('class', [])) or '',
                            next_elem.get('type', '') or '',
                            extract_text_fast(next_elem) or ''
                        ])
                        if any(indicator in button_attrs for indicator in indicators + ['submit', 'go']):
                            button = next_elem
                    next_elem = next_elem.next_sibling
                
                # If no button found among siblings, check parent's children with XPath
                if not button and parent:
                    button_xpath = './button | ./a | ./div[contains(@class, "button")] | ./span[contains(@class, "button")] | ./input[@type="submit"]'
                    buttons = find_by_xpath(parent, button_xpath)
                    
                    for child in buttons:
                        if child != input_elem:
                            button_attrs = ' '.join([
                                child.get('id', '') or '',
                                ' '.join(child.get('class', [])) or '',
                                child.get('type', '') or '',
                                extract_text_fast(child) or ''
                            ])
                            if any(indicator in button_attrs for indicator in indicators + ['submit', 'go']):
                                button = child
                                break
                
                # Calculate score
                score = 15 if button else 10  # Higher score if we found a search button
                
                # Create a custom form representation
                custom_form = {
                    "id": input_elem.get('id', ''),
                    "type": "custom_interface",
                    "action": "",  # No explicit action
                    "method": "get",  # Default method
                    "fields": [{
                        "type": input_type or "text",
                        "id": input_elem.get('id', ''),
                        "name": input_elem.get('name', ''),
                        "placeholder": input_elem.get('placeholder', ''),
                        "possible_values": [],
                        "is_search_field": True
                    }],
                    "search_relevance_score": score
                }
                
                detected_forms.append(custom_form)

    async def detect_search_forms(self, html: str, url: str) -> List[Dict[str, Any]]:
        """Enhanced detection for search forms including JavaScript-based implementations"""
        soup = BeautifulSoup(html, 'lxml')
        forms = []
        
        # Standard form detection
        detected_forms = []
        self._detect_standard_forms(soup, self.search_indicators, detected_forms)
        if detected_forms:
            forms.extend(detected_forms)
        
        # Detect non-form search inputs
        standalone_inputs = self._find_standalone_search_inputs(soup)
        if standalone_inputs:
            forms.extend(standalone_inputs)
            
        # Detect search containers that might not use traditional forms
        search_containers = self._find_search_containers(soup)
        if search_containers:
            forms.extend(search_containers)
            
        return forms
        
    def _find_standalone_search_inputs(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Find standalone search inputs that might not be within a form"""
        inputs = []
        
        # Look for text/search inputs outside of forms
        for input_el in soup.find_all('input', type=['text', 'search']):
            # Skip inputs that are already inside a form
            if input_el.find_parent('form'):
                continue
                
            # Check if it looks like a search input
            attributes = input_el.attrs
            if any(term in str(attributes).lower() for term in ['search', 'find', 'query', 'keyword']):
                # Find a likely submit element nearby
                parent = input_el.parent
                submit = None
                for i in range(3):  # Look up to 3 levels up
                    if not parent:
                        break
                    submit = parent.find(['button', 'input'], type=['submit', 'button']) or \
                             parent.find(['a', 'div'], class_=lambda c: c and any(term in (c or '') for term in ['submit', 'search', 'button']))
                    if submit:
                        break
                    parent = parent.parent
                
                inputs.append({
                    'form': input_el.parent,
                    'inputs': [input_el],
                    'submit': submit,
                    'type': 'standalone_input'
                })
        
        return inputs
    
    def _find_search_containers(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Find div containers that likely contain search functionality"""
        containers = []
        
        # Common container patterns for search functionality
        search_container_selectors = [
            'div[class*="search"]',
            'div[id*="search"]',
            'div[class*="finder"]',
            'div[class*="filter"]',
            'div[class*="idx-"]',
            'div[class*="property-search"]',
            'div[data-role="search"]',
            '#IDX-quickSearch',
            '.idx-omnibar-form',
            '.idx-search-form'
        ]
        
        for selector in search_container_selectors:
            for container in soup.select(selector):
                # Skip if this is already part of a form
                if container.find_parent('form'):
                    continue
                    
                # Find all inputs in this container
                inputs = container.find_all('input')
                if inputs:
                    # Find a likely submit element
                    submit = container.find(['button', 'input'], type=['submit', 'button']) or \
                             container.find(['a', 'div'], class_=lambda c: c and any(term in (c or '') for term in ['submit', 'search', 'button']))
                    
                    containers.append({
                        'form': container,
                        'inputs': inputs,
                        'submit': submit,
                        'type': 'search_container'
                    })
        
        return containers

class SearchAutomator:
    """
    Main search automation facade for SmartScrape.
    
    This class serves as the primary entry point for search functionality throughout 
    the application. It coordinates modular search components and provides a simplified 
    interface for higher-level controllers and strategies.
    """
    
    def __init__(self, config=None):
        """
        Initialize the search automator.
        
        Args:
            config: Dictionary with configuration options
        """
        self.logger = logging.getLogger("SearchAutomation")
        self.config = config or {}
        
        # Initialize search form detector (legacy for backward compatibility)
        self.form_detector = SearchFormDetector()
        self.playwright_handler = PlaywrightSearchHandler(logger=self.logger)
        
        # Import search coordinator lazily to avoid circular imports
        from components.search.search_coordinator import SearchCoordinator
        
        # Initialize the new modular search coordinator
        self.search_coordinator = SearchCoordinator(config=self.config)
        
    async def perform_search(self, url: str, search_term: str, domain_type: str = "general",
                          max_retries: int = 2, preferred_method: str = None) -> Dict[str, Any]:
        """
        Perform a search operation using the most appropriate method.
        
        This is the main entry point for search functionality.
        
        Args:
            url: Target website URL to search
            search_term: Search term to use
            domain_type: Type of domain (e.g., "e_commerce", "real_estate")
            max_retries: Maximum number of retry attempts
            preferred_method: Optional preferred search method
            
        Returns:
            Dictionary with search results and metadata
        """
        self.logger.info(f"Performing search on {url} for term: '{search_term}'")
        
        # Use the new modular search coordinator
        result = await self.search_coordinator.search(
            url=url,
            search_term=search_term,
            domain_type=domain_type,
            max_retries=max_retries,
            preferred_method=preferred_method
        )
        
        # Return the search results
        return result
    
    async def cancel_search(self) -> Dict[str, Any]:
        """
        Cancel an ongoing search operation.
        
        Returns:
            Dictionary with cancellation result
        """
        self.logger.info("Cancelling search operation")
        
        # Use the search coordinator to cancel the search
        result = await self.search_coordinator.cancel_search()
        
        # Return the cancellation result
        return result
    
    # Legacy methods preserved for backward compatibility
    
    async def detect_search_forms(self, url: str, browser_page=None) -> List[Dict[str, Any]]:
        """Legacy method for detecting search forms"""
        self.logger.warning("detect_search_forms is deprecated, use search_coordinator instead")
        
        # Use the form detector directly
        if browser_page:
            html_content = await browser_page.content()
            domain_type = self._detect_domain_type(url)
            return await self.form_detector.detect_search_forms(html_content, domain_type)
        
        # Navigate to URL if no page provided
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.goto(url, wait_until="domcontentloaded")
            html_content = await page.content()
            domain_type = self._detect_domain_type(url)
            forms = await self.form_detector.detect_search_forms(html_content, domain_type)
            await browser.close()
            return forms
    
    async def submit_search_form(self, form_data: Dict[str, Any], search_term: str, 
                               browser_page=None) -> Dict[str, Any]:
        """Legacy method for submitting search forms"""
        self.logger.warning("submit_search_form is deprecated, use search_coordinator instead")
        
        # Use the playwright handler directly
        if browser_page:
            return await self.playwright_handler.submit_form(browser_page, form_data, search_term)
        
        # Create a browser session if none provided
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            # Navigate to the form's action URL if absolute
            form_action = form_data.get("action", "")
            if form_action and form_action.startswith(("http://", "https://")):
                await page.goto(form_action, wait_until="domcontentloaded")
            
            # Submit the form
            result = await self.playwright_handler.submit_form(page, form_data, search_term)
            
            # Add the final page HTML for further processing
            if result.get("success", False):
                result["html"] = await page.content()
                result["url"] = page.url
            
            await browser.close()
            return result
    
    def _detect_domain_type(self, url: str) -> str:
        """Detect the domain type from a URL"""
        # Re-using existing code
        url_lower = url.lower()
        
        # Real estate domain indicators
        real_estate_indicators = [
            'real', 'estate', 'property', 'properties', 'home', 'homes', 'house', 'realty',
            'broker', 'apartment', 'rent', 'realtor', 'zillow', 'trulia', 'redfin', 'listing'
        ]
        
        # E-commerce domain indicators
        ecommerce_indicators = [
            'shop', 'store', 'buy', 'cart', 'checkout', 'product', 'mall', 'market',
            'amazon', 'ebay', 'price', 'order', 'purchase', 'catalog', 'retail'
        ]
        
        # Job listing domain indicators
        job_indicators = [
            'job', 'career', 'employ', 'hire', 'recruit', 'resume', 'cv', 'indeed',
            'linkedin', 'position', 'vacancy', 'work', 'glassdoor', 'staffing'
        ]
        
        # Check URL against indicators
        if any(indicator in url_lower for indicator in real_estate_indicators):
            return 'real_estate'
        elif any(indicator in url_lower for indicator in ecommerce_indicators):
            return 'e_commerce'
        elif any(indicator in url_lower for indicator in job_indicators):
            return 'job_listings'
        
        # Default to general if no specific domain detected
        return 'general'

class PlaywrightSearchHandler:
    """
    Handles search operations using Playwright with stealth mode for enhanced automation.
    
    This class:
    - Provides reliable detection and interaction with search forms on modern sites
    - Uses stealth mode to avoid bot detection
    - Handles complex JavaScript interactions and dynamically loading content
    - Can detect and use search functionality on any website type
    - Validates search results and implements fallback patterns
    - Handles pagination to gather comprehensive results
    """
    
    def __init__(self, logger=None):
        """Initialize the Playwright search handler"""
        self.logger = logger or logging.getLogger("PlaywrightSearch")
        self.domain_registry = {}  # Store successful selectors by domain
        self.pattern_cache = {}    # Cache successful patterns
        
    async def validate_search_results(self, page, search_term: str) -> Dict[str, Any]:
        """
        Validate if the page contains search results.
        
        Args:
            page: Playwright page object
            search_term: Search term that was used
            
        Returns:
            Dictionary with validation results
        """
        try:
            # Check if search term appears in the page
            content = await page.content()
            if search_term.lower() not in content.lower():
                return {"valid": False, "reason": "Search term not found in page"}
            
            # Look for common result indicators
            result_indicators = await page.evaluate("""() => {
                // Look for result count information
                const resultCounts = [];
                
                // Common patterns for result counts
                const countMatches = document.body.innerText.match(/([0-9,]+)\\s*(results|homes|properties|items|products|listings|found)/i);
                if (countMatches && countMatches[1]) {
                    resultCounts.push(parseInt(countMatches[1].replace(/,/g, '')));
                }
                
                // Look for elements that might indicate result counts
                const countElements = document.querySelectorAll('[class*="result-count"], [class*="count"], [class*="total"]');
                for (const el of countElements) {
                    const text = el.innerText;
                    const matches = text.match(/([0-9,]+)/);
                    if (matches && matches[1]) {
                        resultCounts.push(parseInt(matches[1].replace(/,/g, '')));
                    }
                }
                
                // Count result containers
                const resultContainers = [
                    document.querySelectorAll('.search-result, .search-results-item, .result').length,
                    document.querySelectorAll('.listing, .property, .product-item').length,
                    document.querySelectorAll('[class*="result-item"], [class*="listing-item"]').length,
                    document.querySelectorAll('article').length
                ];
                
                // Find the maximum container count
                const maxContainers = Math.max(...resultContainers);
                
                return {
                    resultCounts: resultCounts,
                    containerCount: maxContainers,
                    hasResultWords: document.body.innerText.match(/result|found|match|showing|displayed/i) !== null,
                    hasListings: document.querySelectorAll('a[href*="detail"], a[href*="property"], a[href*="listing"], a[href*="product"]').length > 0
                };
            }""")
            
            # Determine if the page has valid search results
            result_count = 0
            if result_indicators.get("resultCounts") and len(result_indicators["resultCounts"]) > 0:
                # Use the largest count found
                result_count = max(result_indicators["resultCounts"])
            elif result_indicators.get("containerCount", 0) > 0:
                # Use container count if explicit count not found
                result_count = result_indicators["containerCount"]
            
            # Look for "no results" indicators
            no_results_found = await page.evaluate("""() => {
                const noResultsTexts = ['no results', 'no matches', 'found 0', '0 results', 
                                       'nothing found', 'no properties', 'no listings',
                                       'no products', 'try different', 'try another'];
                
                const pageText = document.body.innerText.toLowerCase();
                
                for (const text of noResultsTexts) {
                    if (pageText.includes(text)) {
                        return true;
                    }
                }
                
                // Look for empty result containers
                const emptyContainers = document.querySelectorAll('.no-results, .empty-results, .zero-results');
                if (emptyContainers.length > 0) {
                    return true;
                }
                
                return false;
            }""")
            
            if no_results_found:
                return {
                    "valid": True,
                    "result_count": 0,
                    "empty_results": True,
                    "message": "Search completed but no results found"
                }
            
            # Determine if results are valid based on multiple signals
            valid_results = (
                (result_count > 0) or
                result_indicators.get("hasResultWords", False) or
                result_indicators.get("hasListings", False)
            )
            
            return {
                "valid": valid_results,
                "result_count": result_count,
                "empty_results": False,
                "indicators": result_indicators
            }
            
        except Exception as e:
            self.logger.error(f"Error validating search results: {str(e)}")
            return {"valid": False, "reason": f"Error: {str(e)}"}
        
    async def try_fallback_search(self, url: str, search_term: str) -> Dict[str, Any]:
        """
        Try fallback search patterns when primary search fails.
        
        Args:
            url: URL to start from
            search_term: Search term to use
            
        Returns:
            Dictionary with search results from fallback attempt
        """
        self.logger.info(f"Trying fallback search patterns for '{search_term}' on {url}")
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={"width": 1280, "height": 800},
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"
            )
            page = await context.new_page()
            
            # Apply stealth mode
            await stealth_async(page)
            
            try:
                # Get domain for caching successful patterns
                domain = urlparse(url).netloc
                
                # Load the original URL with progressive timeout strategy
                success = False
                for attempt, (wait_condition, timeout_ms) in enumerate([
                    ("domcontentloaded", 10000),  # First attempt: fast DOM load
                    ("load", 15000),              # Second attempt: full page load
                    ("networkidle", 20000)        # Third attempt: network idle (reduced timeout)
                ], 1):
                    try:
                        self.logger.info(f"Navigation attempt {attempt}: {wait_condition} with {timeout_ms}ms timeout")
                        await page.goto(url, wait_until=wait_condition, timeout=timeout_ms)
                        self.logger.info(f"Successfully navigated to {url} using {wait_condition}")
                        success = True
                        break
                    except Exception as e:
                        self.logger.warning(f"Navigation attempt {attempt} failed with {wait_condition}: {str(e)}")
                        if attempt == 3:
                            self.logger.error(f"All navigation attempts failed for {url}")
                            continue  # Don't raise here, let the calling function handle the failure
                
                if not success:
                    self.logger.error(f"Failed to navigate to {url} with all timeout strategies")
                    return {"success": False, "error": "Navigation timeout with all strategies"}
                
                # Try these fallback approaches in order:
                
                # 1. Check for a dedicated search page
                search_page_result = await self._try_dedicated_search_page(page, search_term)
                if search_page_result.get("success", False):
                    return search_page_result
                
                # 2. Try direct URL manipulation with common patterns
                url_manipulation_result = await self._try_url_manipulation(page, search_term)
                if url_manipulation_result.get("success", False):
                    return url_manipulation_result
                
                # 3. Try advanced JavaScript injection to find hidden search forms
                js_injection_result = await self._try_js_search_injection(page, search_term)
                if js_injection_result.get("success", False):
                    return js_injection_result
                
                # 4. Try clicking on menu items that might lead to search
                menu_navigation_result = await self._try_menu_navigation(page, search_term)
                if menu_navigation_result.get("success", False):
                    return menu_navigation_result
                
                # If all fallbacks fail
                return {
                    "success": False, 
                    "reason": "All fallback search patterns failed",
                    "url": page.url
                }
                
            except Exception as e:
                self.logger.error(f"Error in fallback search: {str(e)}")
                return {"success": False, "reason": f"Fallback error: {str(e)}"}
            finally:
                await browser.close()
    
    async def _try_dedicated_search_page(self, page, search_term: str) -> Dict[str, Any]:
        """Try to find and use a dedicated search page"""
        try:
            # Common search page paths
            search_paths = [
                "/search", "/find", "/search.html", "/find.html", "/search.php",
                "/advanced-search", "/property-search", "/product-search"
            ]
            
            current_url = page.url
            parsed_url = urlparse(current_url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            
            for path in search_paths:
                try:
                    search_url = f"{base_url}{path}"
                    self.logger.info(f"Trying dedicated search page: {search_url}")
                    
                    await page.goto(search_url, timeout=10000)
                    await page.wait_for_load_state("domcontentloaded")
                    
                    # If page loaded, try to find and use a search form
                    # This is a placeholder - in the real implementation, 
                    # we would call a method to find and use search forms
                    # search_result = await self._try_universal_search_approach(page, search_term)
                    
                    # For now, let's look for a simple search input to fill
                    search_input_selectors = [
                        'input[type="search"]',
                        'input[type="text"][name*="search"]',
                        'input[type="text"][placeholder*="search"]',
                        'input[type="text"][id*="search"]',
                        'input[type="text"]'
                    ]
                    
                    for selector in search_input_selectors:
                        if await page.locator(selector).count() > 0:
                            await page.fill(selector, search_term)
                            await page.keyboard.press('Enter')
                            
                            # Wait for possible navigation
                            await page.wait_for_load_state("networkidle", timeout=5000)
                            
                            # Validate if this page contains search results
                            validation = await self.validate_search_results(page, search_term)
                            if validation.get("valid", False):
                                return {
                                    "success": True,
                                    "method": "dedicated_search_page",
                                    "url": page.url,
                                    "html": await page.content()
                                }
                except Exception as e:
                    self.logger.debug(f"Error with search path {path}: {str(e)}")
                    continue
            
            return {"success": False}
            
        except Exception as e:
            self.logger.debug(f"Error finding dedicated search page: {str(e)}")
            return {"success": False}
    
    async def _try_url_manipulation(self, page, search_term: str) -> Dict[str, Any]:
        """Try direct URL manipulation with common patterns"""
        try:
            current_url = page.url
            parsed_url = urlparse(current_url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            
            # Common URL patterns for search
            url_patterns = [
                f"{base_url}/search?q={quote(search_term)}",
                f"{base_url}/search?query={quote(search_term)}",
                f"{base_url}/search?keyword={quote(search_term)}",
                f"{base_url}/search?s={quote(search_term)}",
                f"{base_url}/search?term={quote(search_term)}",
                f"{base_url}?s={quote(search_term)}",
                f"{base_url}?q={quote(search_term)}",
                f"{base_url}/search/{quote(search_term)}",
                f"{base_url}/find?q={quote(search_term)}",
                # Real estate specific
                f"{base_url}/properties/search?q={quote(search_term)}",
                f"{base_url}/property-search?location={quote(search_term)}",
                f"{base_url}/homes?location={quote(search_term)}"
            ]
            
            for url in url_patterns:
                try:
                    self.logger.info(f"Trying search URL pattern: {url}")
                    await page.goto(url, timeout=10000)
                    await page.wait_for_load_state("domcontentloaded")
                    
                    # Validate if this page contains search results
                    validation = await self.validate_search_results(page, search_term)
                    if validation.get("valid", False):
                        return {
                            "success": True,
                            "method": "url_manipulation",
                            "url": page.url,
                            "html": await page.content()
                        }
                except Exception as e:
                    self.logger.debug(f"Error with URL pattern {url}: {str(e)}")
                    continue
            
            return {"success": False}
            
        except Exception as e:
            self.logger.debug(f"Error in URL manipulation: {str(e)}")
            return {"success": False}
    
    async def _try_js_search_injection(self, page, search_term: str) -> Dict[str, Any]:
        """Try advanced JavaScript injection to find hidden search forms"""
        try:
            self.logger.info("Trying JavaScript search injection")
            
            # Inject script to find and fill any possible search input
            search_success = await page.evaluate("""async (searchTerm) => {
                // Find all inputs in the page, even hidden ones
                const inputs = document.querySelectorAll('input');
                
                // Helper function to determine if an input could be a search input
                const isSearchInput = (input) => {
                    const attrs = (input.id || '') + ' ' + (input.name || '') + ' ' + 
                                (input.className || '') + ' ' + (input.placeholder || '');
                    const attrsLower = attrs.toLowerCase();
                    
                    return input.type === 'search' || 
                           input.type === 'text' || 
                           !input.type || 
                           attrsLower.includes('search') ||
                           attrsLower.includes('query') ||
                           attrsLower.includes('keyword');
                };
                
                // Try search inputs that are currently hidden
                for (const input of inputs) {
                    if (isSearchInput(input)) {
                        try {
                            // Temporarily make the input visible and interactable
                            const originalDisplay = input.style.display;
                            const originalVisibility = input.style.visibility;
                            const originalPosition = input.style.position;
                            
                            input.style.display = 'block';
                            input.style.visibility = 'visible';
                            input.style.position = 'static';
                            
                            // Fill the input
                            input.value = searchTerm;
                            input.dispatchEvent(new Event('input', { bubbles: true }));
                            input.dispatchEvent(new Event('change', { bubbles: true }));
                            
                            // Look for a submit button
                            const form = input.closest('form');
                            if (form) {
                                const button = form.querySelector('button[type="submit"], input[type="submit"]');
                                if (button) {
                                    // Click the button
                                    button.click();
                                    return { success: true, method: "hidden_form_button" };
                                } else {
                                    // Submit the form directly
                                    form.submit();
                                    return { success: true, method: "hidden_form_submit" };
                                }
                            }
                            
                            // Try simulating Enter key
                            input.dispatchEvent(new KeyboardEvent('keydown', {
                                key: 'Enter',
                                code: 'Enter',
                                keyCode: 13,
                                which: 13,
                                bubbles: true
                            }));
                            
                            // Restore original styles
                            input.style.display = originalDisplay;
                            input.style.visibility = originalVisibility;
                            input.style.position = originalPosition;
                            
                            return { success: true, method: "hidden_input_enter" };
                        } catch (e) {
                            console.error("Error interacting with hidden input:", e);
                        }
                    }
                }
                
                return { success: false };
            }""", search_term)
            
            if search_success.get("success", False):
                # Wait for possible navigation
                await page.wait_for_load_state("networkidle", timeout=5000)
                
                # Check if results are present
                validation = await self.validate_search_results(page, search_term)
                if validation.get("valid", False):
                    return {
                        "success": True,
                        "method": f"js_injection_{search_success.get('method', 'unknown')}",
                        "url": page.url,
                        "html": await page.content()
                    }
            
            return {"success": False}
            
        except Exception as e:
            self.logger.debug(f"Error in JS search injection: {str(e)}")
            return {"success": False}
    
    async def _try_menu_navigation(self, page, search_term: str) -> Dict[str, Any]:
        """Try clicking on menu items that might lead to search"""
        try:
            self.logger.info("Trying menu navigation to find search")
            
            # Look for navigation links that might lead to search
            search_link_selectors = [
                'a[href*="search" i]',
                'a:text("Search")',
                'a:text("Find")',
                'a:text("Browse")',
                'a:text("Advanced Search")',
                'header a',
                'nav a',
                '.nav a',
                '.menu a',
                '.main-navigation a'
            ]
            
            for selector in search_link_selectors:
                try:
                    links = page.locator(selector)
                    count = await links.count()
                    
                    for i in range(count):
                        link = links.nth(i)
                        if await link.is_visible():
                            link_text = await link.text_content()
                            
                            # Only click on links that might lead to search
                            if link_text and any(term in link_text.lower() 
                                               for term in ["search", "find", "browse", "advanced"]):
                                self.logger.info(f"Clicking menu link: {link_text}")
                                
                                current_url = page.url
                                await link.click()
                                
                                try:
                                    # Wait for navigation
                                    await page.wait_for_load_state("networkidle", timeout=5000)
                                    
                                    # If URL changed, try to find a search form
                                    if page.url != current_url:
                                        # Look for search inputs
                                        search_input_selectors = [
                                            'input[type="search"]',
                                            'input[type="text"][name*="search"]',
                                            'input[type="text"][placeholder*="search"]',
                                            'input[type="text"][id*="search"]',
                                            'input[type="text"]'
                                        ]
                                        
                                        for input_selector in search_input_selectors:
                                            if await page.locator(input_selector).count() > 0:
                                                await page.fill(input_selector, search_term)
                                                await page.keyboard.press('Enter')
                                                
                                                # Wait for possible navigation
                                                await page.wait_for_load_state("networkidle", timeout=5000)
                                                
                                                # Validate if this page contains search results
                                                validation = await self.validate_search_results(page, search_term)
                                                if validation.get("valid", False):
                                                    return {
                                                        "success": True,
                                                        "method": "menu_navigation",
                                                        "url": page.url,
                                                        "html": await page.content()
                                                    }
                                except Exception as e:
                                    self.logger.debug(f"Error after clicking menu item: {str(e)}")
                                    continue
                except Exception as e:
                    self.logger.debug(f"Error with selector {selector}: {str(e)}")
                    continue
            
            return {"success": False}
            
        except Exception as e:
            self.logger.debug(f"Error in menu navigation: {str(e)}")
            return {"success": False}
    
    async def handle_pagination(self, page, site_type: str = None) -> Dict[str, Any]:
        """
        Handle pagination to access all search results.
        
        Args:
            page: Playwright page with search results
            site_type: Type of site (optional) for specialized handling
            
        Returns:
            Dictionary with pagination results
        """
        try:
            self.logger.info("Handling pagination for search results")
            
            # Detect pagination type
            pagination_type = await self._detect_pagination_type(page)
            self.logger.info(f"Detected pagination type: {pagination_type}")
            
            # Store current page URL
            first_page_url = page.url
            
            # Store all visited URLs to avoid loops
            visited_urls = set([first_page_url])
            
            # Save first page content
            pages_content = [{
                "url": first_page_url,
                "content": await page.content(),
                "page_num": 1
            }]
            
            # Maximum pages to fetch (can be made configurable)
            max_pages = 5
            current_page = 1
            
            # Handle pagination based on detected type
            if pagination_type == "numbered":
                # Handle numbered pagination (1, 2, 3, ...)
                while current_page < max_pages:
                    next_page_available = await self._click_next_numbered_page(page, current_page + 1)
                    if not next_page_available:
                        break
                    
                    # Wait for navigation
                    await page.wait_for_load_state("networkidle", timeout=5000)
                    
                    # Check if URL is new
                    if page.url in visited_urls:
                        self.logger.info("Already visited this URL, stopping pagination")
                        break
                    
                    visited_urls.add(page.url)
                    current_page += 1
                    
                    # Save this page
                    pages_content.append({
                        "url": page.url,
                        "content": await page.content(),
                        "page_num": current_page
                    })
                    self.logger.info(f"Fetched page {current_page} at {page.url}")
                    
            elif pagination_type == "next_button":
                # Handle "Next" button pagination
                while current_page < max_pages:
                    has_next = await self._click_next_button(page)
                    if not has_next:
                        break
                    
                    # Wait for navigation or content change
                    await page.wait_for_load_state("networkidle", timeout=5000)
                    
                    # Check if this is a new URL
                    if page.url in visited_urls:
                        # Check if content changed (AJAX pagination)
                        new_content = await page.content()
                        if new_content == pages_content[-1]["content"]:
                            self.logger.info("Content did not change, stopping pagination")
                            break
                    
                    visited_urls.add(page.url)
                    current_page += 1
                    
                    # Save this page
                    pages_content.append({
                        "url": page.url,
                        "content": await page.content(),
                        "page_num": current_page
                    })
                    self.logger.info(f"Fetched page {current_page} at {page.url}")
                    
            elif pagination_type == "load_more":
                # Handle "Load More" button pagination
                while current_page < max_pages:
                    clicked = await self._click_load_more(page)
                    if not clicked:
                        break
                    
                    # Wait for content change
                    await asyncio.sleep(2)  # Wait for AJAX to load content
                    
                    current_page += 1
                    
                    # Save this page
                    pages_content.append({
                        "url": page.url,
                        "content": await page.content(),
                        "page_num": current_page
                    })
                    self.logger.info(f"Loaded more content, page {current_page}")
                    
            elif pagination_type == "infinite_scroll":
                # Handle infinite scroll pagination
                while current_page < max_pages:
                    scrolled = await self._trigger_infinite_scroll(page)
                    if not scrolled:
                        break
                    
                    # Wait for content to load
                    await asyncio.sleep(2)
                    
                    # Check if content changed
                    new_content = await page.content()
                    if new_content == pages_content[-1]["content"]:
                        self.logger.info("Content did not change after scrolling, stopping pagination")
                        break
                    
                    current_page += 1
                    
                    # Save this page
                    pages_content.append({
                        "url": page.url,
                        "content": await page.content(),
                        "page_num": current_page
                    })
                    self.logger.info(f"Scrolled to load more content, page {current_page}")
            
            return {
                "success": True,
                "page_count": len(pages_content),
                "pagination_type": pagination_type,
                "pages": pages_content
            }
            
        except Exception as e:
            self.logger.error(f"Error handling pagination: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "pages": [{"url": page.url, "content": await page.content(), "page_num": 1}]
            }
    
    async def _detect_pagination_type(self, page) -> str:
        """Detect the type of pagination used on the page"""
        try:
            pagination_type = await page.evaluate("""() => {
                // Check for numbered pagination
                const hasNumberedPagination = 
                    document.querySelector('.pagination, .pager, nav[aria-label*="pagination" i]') !== null ||
                    document.querySelectorAll('a[href*="page="], a[href*="/page/"]').length > 0;
                
                // Check for next button pagination
                const hasNextButton = 
                    document.querySelector('a:text("Next"), a[aria-label="Next"], ' +
                                         'a.next, .next > a, [class*="pagination"] a[rel="next"]') !== null;
                
                // Check for "Load More" button
                const hasLoadMore = 
                    document.querySelector('button:text("Load More"), ' +
                                         'button:text("Show More"), ' +
                                         'a:text("Load More"), ' +
                                         '.load-more, .show-more, .more-results') !== null;
                
                // Check for infinite scroll
                const hasInfiniteScroll = 
                    document.querySelectorAll('[class*="infinite"], [class*="scroll-container"]').length > 0 ||
                    document.querySelector('div[data-scroll], #infinite-scroll') !== null;
                
                // Determine pagination type based on the checks
                if (hasNumberedPagination) return "numbered";
                if (hasNextButton) return "next_button";
                if (hasLoadMore) return "load_more";
                if (hasInfiniteScroll) return "infinite_scroll";
                
                // Default to no pagination or can't determine
                return "unknown";
            }""")
            
            return pagination_type
            
        except Exception as e:
            self.logger.debug(f"Error detecting pagination type: {str(e)}")
            return "unknown"
    
    async def _click_next_numbered_page(self, page, next_page_num: int) -> bool:
        """Click on a specific numbered page in the pagination"""
        try:
            # First look for an exact page number link
            next_page_selectors = [
                f'a:text("{next_page_num}")',
                f'[class*="pagination"] a:text("{next_page_num}")',
                f'[class*="pager"] a:text("{next_page_num}")',
                f'a[aria-label="Page {next_page_num}"]'
            ]
            
            for selector in next_page_selectors:
                try:
                    links = page.locator(selector)
                    count = await links.count()
                    
                    for i in range(count):
                        link = links.nth(i)
                        if await link.is_visible():
                            # Check if this link is in a pagination container
                            is_in_pagination = await page.evaluate("""(linkElement) => {
                                const parents = [];
                                let el = linkElement;
                                for (let i = 0; i < 5; i++) {  // Check up to 5 levels up
                                    el = el.parentElement;
                                    if (!el) break;
                                    parents.push(el);
                                }
                                
                                return parents.some(p => {
                                    const classes = p.className.toLowerCase();
                                    return classes.includes('pagination') || 
                                           classes.includes('pager') || 
                                           classes.includes('pages');
                                });
                            }""", link.element_handle())
                            
                            if is_in_pagination:
                                await link.click()
                                return True
                except Exception as e:
                    self.logger.debug(f"Error with next page selector {selector}: {str(e)}")
                    continue
            
            # If specific page not found, try "Next" button
            return await self._click_next_button(page)
            
        except Exception as e:
            self.logger.debug(f"Error clicking numbered page {next_page_num}: {str(e)}")
            return False
    
    async def _click_next_button(self, page) -> bool:
        """Click on a "Next" button in the pagination"""
        try:
            next_button_selectors = [
                'a:text("Next")',
                'a[aria-label="Next"]',
                'a[rel="next"]',
                'a.next, .next > a',
                '[class*="pagination"] a.next, [class*="pagination"] a[rel="next"]',
                'button:text("Next")',
                'a svg[class*="arrow-right"], a svg[class*="next"]'
            ]
            
            for selector in next_button_selectors:
                try:
                    if await page.locator(selector).count() > 0:
                        next_button = page.locator(selector).first
                        if await next_button.is_visible() and await next_button.is_enabled():
                            await next_button.click()
                            return True
                except Exception as e:
                    self.logger.debug(f"Error with next button selector {selector}: {str(e)}")
                    continue
            
            return False
            
        except Exception as e:
            self.logger.debug(f"Error clicking next button: {str(e)}")
            return False
    
    async def _click_load_more(self, page) -> bool:
        """Click on a "Load More" button"""
        try:
            load_more_selectors = [
                'button:text("Load More")',
                'button:text("Show More")',
                'a:text("Load More")',
                'a:text("Show More")',
                '.load-more, .show-more, .more-results',
                'button[data-testid*="load-more" i]',
                'div[role="button"]:has-text("Load More")'
            ]
            
            for selector in load_more_selectors:
                try:
                    if await page.locator(selector).count() > 0:
                        load_more = page.locator(selector).first
                        if await load_more.is_visible() and await load_more.is_enabled():
                            await load_more.click()
                            return True
                except Exception as e:
                    self.logger.debug(f"Error with load more selector {selector}: {str(e)}")
                    continue
            
            return False
            
        except Exception as e:
            self.logger.debug(f"Error clicking load more: {str(e)}")
            return False
    
    async def _trigger_infinite_scroll(self, page) -> bool:
        """Trigger infinite scroll by scrolling to the bottom of the page"""
        try:
            # Get current height
            current_height = await page.evaluate("""() => document.body.scrollHeight""")
            
            # Scroll to bottom
            await page.evaluate("""() => window.scrollTo(0, document.body.scrollHeight)""")
            
            # Wait for possible loading
            await asyncio.sleep(2)
            
            # Check if height increased
            new_height = await page.evaluate("""() => document.body.scrollHeight""")
            
            return new_height > current_height
            
        except Exception as e:
            self.logger.debug(f"Error triggering infinite scroll: {str(e)}")
            return False

# Add this class for AJAX response handling (task 3.3.5)
class AJAXResponseHandler:
    """
    Handles AJAX responses for search results.
    
    This class:
    - Captures and processes AJAX responses during search operations
    - Extracts search results from AJAX responses
    - Handles different AJAX response formats (JSON, HTML, XML)
    - Manages dynamic content loading through infinite scroll or pagination
    """
    
    def __init__(self):
        self.logger = logging.getLogger("AJAXResponseHandler")
        # Track responses by URL
        self.responses = {}
        # Track result extraction patterns
        self.extraction_patterns = {}
        
    async def setup_response_monitoring(self, page):
        """Set up monitoring for AJAX responses"""
        self.logger.info("Setting up AJAX response monitoring")
        
        # Create a list to store response data
        response_data = []
        
        # Handle responses
        async def on_response(response):
            try:
                # Only process responses that might contain results
                if response.status == 200 and response.request.resource_type in ['xhr', 'fetch']:
                    url = response.url
                    
                    # Skip static resources like images and stylesheets
                    if url.endswith(('.jpg', '.png', '.gif', '.css', '.js')):
                        return
                    
                    # Get content-type
                    content_type = response.headers.get('content-type', '')
                    
                    # Process based on content type
                    if 'application/json' in content_type or url.endswith('.json'):
                        try:
                            response_body = await response.text()
                            response_data.append({
                                'url': url,
                                'content_type': 'json',
                                'body': response_body,
                                'timestamp': time.time()
                            })
                        except Exception as e:
                            self.logger.debug(f"Failed to get JSON response text: {e}")
                    
                    elif 'text/html' in content_type:
                        try:
                            response_body = await response.text()
                            # Only store HTML fragments that might contain results
                            if '<div' in response_body or '<li' in response_body:
                                response_data.append({
                                    'url': url,
                                    'content_type': 'html',
                                    'body': response_body,
                                    'timestamp': time.time()
                                })
                        except Exception as e:
                            self.logger.debug(f"Failed to get HTML response text: {e}")
                    
                    elif 'application/xml' in content_type or 'text/xml' in content_type:
                        try:
                            response_body = await response.text()
                            response_data.append({
                                'url': url,
                                'content_type': 'xml',
                                'body': response_body,
                                'timestamp': time.time()
                            })
                        except Exception as e:
                            self.logger.debug(f"Failed to get XML response text: {e}")
            except Exception as e:
                self.logger.warning(f"Error monitoring response: {str(e)}")
        
        # Set up the listener
        page.on('response', on_response)
        
        return response_data
    
    async def process_captured_responses(self, responses, search_term):
        """Process captured AJAX responses to extract search results"""
        self.logger.info(f"Processing {len(responses)} captured AJAX responses")
        
        results = []
        
        for response in responses:
            try:
                content_type = response.get('content_type', '')
                body = response.get('body', '')
                url = response.get('url', '')
                
                if not body:
                    continue
                
                # Process different content types
                extracted_items = []
                if content_type == 'json':
                    extracted_items = self._process_json_response(body, search_term)
                elif content_type == 'html':
                    extracted_items = self._process_html_response(body, search_term)
                elif content_type == 'xml':
                    extracted_items = self._process_xml_response(body, search_term)
                
                if extracted_items:
                    results.append({
                        'url': url,
                        'content_type': content_type,
                        'result_count': len(extracted_items),
                        'results': extracted_items,
                        'timestamp': response.get('timestamp', time.time())
                    })
            except Exception as e:
                self.logger.warning(f"Error processing response: {str(e)}")
        
        # Sort results by result count (descending)
        results.sort(key=lambda x: x['result_count'], reverse=True)
        
        return results
    
    def _process_json_response(self, body, search_term):
        """Process a JSON response to extract search results"""
        try:
            # Parse JSON
            data = json.loads(body)
            
            # Find potential result arrays
            candidate_arrays = self._find_candidate_arrays(data)
            
            # Score and filter candidates
            scored_candidates = []
            for path, array in candidate_arrays:
                if not array:
                    continue
                
                score = self._score_candidate_array(array, search_term)
                scored_candidates.append((path, array, score))
            
            # Sort by score (descending)
            scored_candidates.sort(key=lambda x: x[2], reverse=True)
            
            # Extract and return results from best candidate
            if scored_candidates and scored_candidates[0][2] >= 10:
                path, array, score = scored_candidates[0]
                # Remember this extraction pattern
                self._remember_extraction_pattern('json', path)
                # Return structured results
                return self._extract_structured_results(array)
            
            return []
            
        except Exception as e:
            self.logger.warning(f"Error processing JSON response: {str(e)}")
            return []
    
    def _process_html_response(self, body, search_term):
        """Process an HTML response to extract search results"""
        try:
            # Parse HTML
            soup = BeautifulSoup(body, 'html.parser')
            
            # Look for result containers (using common patterns)
            candidates = []
            
            # Common result container patterns
            selectors = [
                'div.results', 'div.search-results', 'ul.results', 'ol.results',
                'div.product-list', 'div.products', 'div.listings',
                '.search-result', '.result-item', '.product-item', '.listing-item',
                '[data-testid="search-results"]', '[aria-label="Search results"]'
            ]
            
            # Add selectors for result lists
            for selector in selectors:
                elements = soup.select(selector)
                if elements:
                    candidates.append((selector, elements))
            
            # If we didn't find any obvious result containers, try more generic approaches
            if not candidates:
                # Look for groups of similar elements that might be results
                candidates.extend(self._find_repeated_structures(soup))
            
            # Score candidates
            scored_candidates = []
            for selector, elements in candidates:
                if not elements:
                    continue
                    
                score = self._score_html_candidate(elements, search_term)
                scored_candidates.append((selector, elements, score))
            
            # Sort by score (descending)
            scored_candidates.sort(key=lambda x: x[2], reverse=True)
            
            # Extract and return results from best candidate
            if scored_candidates and scored_candidates[0][2] >= 10:
                selector, elements, score = scored_candidates[0]
                # Remember this extraction pattern
                self._remember_extraction_pattern('html', selector)
                
                # Extract results
                results = []
                for element in elements[:20]:  # Process up to 20 items
                    item = self._extract_html_result(element)
                    if item:
                        results.append(item)
                        
                return results
            
            return []
            
        except Exception as e:
            self.logger.warning(f"Error processing HTML response: {str(e)}")
            return []
    
    def _process_xml_response(self, body, search_term):
        """Process an XML response to extract search results"""
        try:
            # Parse XML
            soup = BeautifulSoup(body, 'lxml-xml')
            
            # Look for repeated elements (potential results)
            candidates = []
            
            # Common XML result patterns
            for tag in soup.find_all():
                # Skip root element
                if tag.parent is None or tag.parent.parent is None:
                    continue
                    
                # Check if this tag is repeated (more than 3 times)
                siblings = tag.find_parent().find_all(tag.name, recursive=False)
                if len(siblings) >= 3:
                    candidates.append((tag.name, siblings))
            
            # Score candidates
            scored_candidates = []
            for tag_name, elements in candidates:
                if not elements:
                    continue
                    
                score = self._score_xml_candidate(elements, search_term)
                scored_candidates.append((tag_name, elements, score))
            
            # Sort by score (descending)
            scored_candidates.sort(key=lambda x: x[2], reverse=True)
            
            # Extract and return results from best candidate
            if scored_candidates and scored_candidates[0][2] >= 10:
                tag_name, elements, score = scored_candidates[0]
                # Remember this extraction pattern
                self._remember_extraction_pattern('xml', tag_name)
                
                # Extract results
                results = []
                for element in elements[:20]:  # Process up to 20 items
                    item = self._extract_xml_result(element)
                    if item:
                        results.append(item)
                        
                return results
            
            return []
            
        except Exception as e:
            self.logger.warning(f"Error processing XML response: {str(e)}")
            return []
    
    def _find_candidate_arrays(self, data, current_path=None):
        """Find arrays in a JSON object that might contain search results"""
        if current_path is None:
            current_path = []
            
        candidates = []
        
        if isinstance(data, list) and data:
            # This is a candidate array
            candidates.append((current_path, data))
            
            # Also check the first item for nested arrays
            if data and isinstance(data[0], (dict, list)):
                nested_candidates = self._find_candidate_arrays(data[0], current_path + ["[0]"])
                candidates.extend(nested_candidates)
                
        elif isinstance(data, dict):
            # Look for result arrays in the dict
            result_indicators = [
                'results', 'items', 'data', 'content', 'products', 'listings',
                'hits', 'documents', 'search', 'list', 'collection'
            ]
            
            for key, value in data.items():
                if isinstance(value, list) and value:
                    if any(indicator in key.lower() for indicator in result_indicators):
                        # This key suggests it contains search results
                        candidates.append((current_path + [key], value))
                        
                # Recurse into this value
                if isinstance(value, (dict, list)):
                    nested_candidates = self._find_candidate_arrays(value, current_path + [key])
                    candidates.extend(nested_candidates)
                
        return candidates
    
    def _score_candidate_array(self, array, search_term):
        """Score an array on how likely it contains search results"""
        if not array:
            return 0
            
        score = 0
        
        # Size-based scoring
        if len(array) > 0:
            score += 5
        if len(array) > 3:
            score += 10  # More items is better for search results
        if len(array) > 20:
            score += 5   # Lots of results
        
        # Check first few items
        if array and isinstance(array[0], dict):
            # Items are objects - good sign
            score += 15
            
            # Check for common result fields
            item = array[0]
            result_fields = ['id', 'title', 'name', 'description', 'url', 'link', 
                            'price', 'image', 'thumbnail', 'summary']
            
            field_matches = [field for field in result_fields if field in item]
            score += len(field_matches) * 3
            
            # Check if search term appears in values
            search_term_lower = search_term.lower()
            for value in item.values():
                if isinstance(value, str) and search_term_lower in value.lower():
                    score += 15
                    break
        
        return score
    
    def _score_html_candidate(self, elements, search_term):
        """Score HTML elements on how likely they contain search results"""
        if not elements:
            return 0
            
        score = 0
        
        # Size-based scoring
        if len(elements) > 3:
            score += 10  # Multiple similar elements
        if len(elements) > 10:
            score += 5   # Lots of similar elements
        
        # Check first element
        first_element = elements[0]
        
        # Look for result indicators
        for indicator in ['title', 'price', 'description', 'image', 'link', 'product', 'result']:
            if first_element.select(f'[class*="{indicator}"]'):
                score += 8
                
        # Check if element has links
        if first_element.find('a'):
            score += 10  # Results typically have links
            
        # Check if element has images
        if first_element.find('img'):
            score += 5  # Results often have images
            
        # Check if search term appears in text
        search_term_lower = search_term.lower()
        if search_term_lower in first_element.get_text().lower():
            score += 15
            
        # Check uniform structure
        if len(elements) > 1:
            # If elements have similar structure (similar classes)
            first_classes = set(first_element.get('class', []))
            second_classes = set(elements[1].get('class', []))
            if first_classes and first_classes == second_classes:
                score += 10
                
        return score
    
    def _score_xml_candidate(self, elements, search_term):
        """Score XML elements on how likely they contain search results"""
        if not elements:
            return 0
            
        score = 0
        
        # Size-based scoring
        if len(elements) > 3:
            score += 10  # Multiple similar elements
        if len(elements) > 10:
            score += 5   # Lots of similar elements
        
        # Check structure of first element
        first_element = elements[0]
        
        # Look for result indicators in child element names
        result_indicators = ['title', 'description', 'link', 'url', 'id', 'name', 'price']
        for indicator in result_indicators:
            if first_element.find(indicator):
                score += 8
                
        # Check if search term appears in text
        search_term_lower = search_term.lower()
        if search_term_lower in first_element.get_text().lower():
            score += 15
            
        # Check uniform structure - compare child elements of first two elements
        if len(elements) > 1:
            first_children = [child.name for child in first_element.children if child.name]
            second_children = [child.name for child in elements[1].children if child.name]
            if first_children and set(first_children) == set(second_children):
                score += 10
                
        return score
    
    def _extract_structured_results(self, array):
        """Extract structured results from a JSON array"""
        results = []
        
        # Process up to 20 items
        for item in array[:20]:
            if isinstance(item, dict):
                # Extract key fields or use the entire item
                structured_result = {
                    'title': item.get('title', item.get('name', '')),
                    'description': item.get('description', item.get('summary', '')),
                    'url': item.get('url', item.get('link', item.get('href', ''))),
                    'id': item.get('id', ''),
                    'image': item.get('image', item.get('thumbnail', item.get('imageUrl', ''))),
                    'price': item.get('price', item.get('cost', '')),
                    'data': item  # Include the full item data
                }
                
                # Only include the result if it has at least one meaningful field
                if structured_result['title'] or structured_result['url'] or structured_result['id']:
                    results.append(structured_result)
                
        return results
    
    def _extract_html_result(self, element):
        """Extract a result from an HTML element"""
        result = {}
        
        # Extract title - look for heading or strong text
        title_candidates = [
            element.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']),
            element.find('strong'),
            element.find(class_=lambda c: c and any(x in str(c) for x in ['title', 'name', 'heading'])),
            element.find('a')  # Fallback to link text
        ]
        
        for candidate in title_candidates:
            if candidate and candidate.get_text().strip():
                result['title'] = candidate.get_text().strip()
                break
        
        # Extract URL - look for links
        link = element.find('a')
        if link and link.get('href'):
            result['url'] = link.get('href')
            
        # Extract image - look for image elements
        img = element.find('img')
        if img and img.get('src'):
            result['image'] = img.get('src')
            
        # Extract description - look for paragraphs or divs with description classes
        desc_candidates = [
            element.find('p'),
            element.find(class_=lambda c: c and any(x in str(c) for x in ['desc', 'summary', 'text', 'content'])),
            element.find(['div', 'span'], class_=lambda c: c and 'description' in str(c))
        ]
        
        for candidate in desc_candidates:
            if candidate and candidate.get_text().strip():
                result['description'] = candidate.get_text().strip()
                break
        
        # Extract price - look for price indicators
        price_candidates = [
            element.find(class_=lambda c: c and any(x in str(c) for x in ['price', 'cost', 'amount'])),
            element.find(string=lambda s: s and re.search(r'(\$|€|£)\s*\d+', s))
        ]
        
        for candidate in price_candidates:
            if candidate:
                if hasattr(candidate, 'get_text'):
                    price_text = candidate.get_text().strip()
                else:
                    price_text = str(candidate).strip()
                
                # Extract price with regex
                price_match = re.search(r'(\$|€|£)\s*(\d+(?:,\d+)*(?:\.\d+)?)', price_text)
                if price_match:
                    result['price'] = price_match.group(0)
                    break
        
        return result if result else None
    
    def _extract_xml_result(self, element):
        """Extract a result from an XML element"""
        result = {}
        
        # Common field mappings for XML
        field_mappings = {
            'title': ['title', 'name', 'heading'],
            'description': ['description', 'summary', 'content', 'text'],
            'url': ['url', 'link', 'href', 'uri'],
            'id': ['id', 'uid', 'identifier', 'guid'],
            'image': ['image', 'img', 'thumbnail', 'enclosure'],
            'price': ['price', 'cost', 'amount', 'value'],
        }
        
        # Extract fields based on child element names
        for target_field, source_fields in field_mappings.items():
            for field in source_fields:
                field_element = element.find(field)
                if field_element and field_element.string:
                    result[target_field] = field_element.string.strip()
                    break
                # Special case for link elements with href attribute
                elif field == 'link' and element.find(field) and element.find(field).get('href'):
                    result['url'] = element.find(field).get('href')
                    break
                elif field == 'enclosure' and element.find(field) and element.find(field).get('url'):
                    result['image'] = element.find(field).get('url')
                    break
        
        return result if result else None
    
    def _find_repeated_structures(self, soup):
        """Find elements with repeated similar structures that might be results"""
        candidates = []
        
        # Look for common parent containers
        containers = soup.select('div, ul, ol, section, main')
        
        for container in containers:
            # Skip small containers
            if len(container.contents) < 3:
                continue
                
            # Get direct children with the same tag
            for tag in ['div', 'li', 'article']:
                elements = container.find_all(tag, recursive=False)
                if len(elements) >= 3:  # At least 3 similar elements
                    candidates.append((f"{container.name} > {tag}", elements))
        
        return candidates
    
    def _remember_extraction_pattern(self, content_type, pattern):
        """Remember successful extraction patterns for future use"""
        if content_type not in self.extraction_patterns:
            self.extraction_patterns[content_type] = []
            
        if pattern not in self.extraction_patterns[content_type]:
            self.extraction_patterns[content_type].append(pattern)
    
    async def wait_for_ajax_results(self, page, timeout=5000):
        """
        Wait for AJAX results to load.
        
        Args:
            page: Playwright page
            timeout: Maximum time to wait in ms
            
        Returns:
            True if results were detected, False otherwise
        """
        try:
            # Wait for common result container selectors
            selectors = [
                'div.results', '.search-results', 'ul.results', 
                '.product-list', '.search-result', '[data-testid="search-results"]'
            ]
            
            selector = ', '.join(selectors)
            
            # Wait for either a result container or network idle
            result = await asyncio.gather(
                page.wait_for_selector(selector, timeout=timeout).catch(lambda _: None),
                page.wait_for_load_state('networkidle', timeout=timeout).catch(lambda _: None)
            )
            
            # Check if results were found
            return result[0] is not None
            
        except Exception as e:
            self.logger.warning(f"Error waiting for AJAX results: {str(e)}")
            return False
    
    async def scroll_for_more_results(self, page, max_scrolls=5):
        """
        Scroll down to load more results (for infinite scroll pages).
        
        Args:
            page: Playwright page
            max_scrolls: Maximum number of scrolls to perform
            
        Returns:
            Number of scrolls performed
        """
        scrolls_performed = 0
        
        try:
            # Get initial page height
            initial_height = await page.evaluate("document.body.scrollHeight")
            
            for i in range(max_scrolls):
                # Scroll to bottom
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                
                # Wait for possible content to load
                await page.wait_for_timeout(1000)  # 1 second wait
                
                # Check if page height increased (new content loaded)
                new_height = await page.evaluate("document.body.scrollHeight")
                
                if new_height > initial_height:
                    scrolls_performed += 1
                    initial_height = new_height
                else:
                    # No new content, stop scrolling
                    break
            
            return scrolls_performed
            
        except Exception as e:
            self.logger.warning(f"Error scrolling for more results: {str(e)}")
            return scrolls_performed

# In the existing SearchAutomation class, add reference to our new classes
class SearchAutomation:
    """
    Main search automation class for SmartScrape.
    
    This class handles detection and automation of search interfaces on websites,
    including form-based search, URL-based search patterns, and API-based search.
    """
    
    def __init__(self, config=None):
        self.logger = logging.getLogger("SearchAutomation")
        self.form_detector = SearchFormDetector()
        self.pagination_handler = PaginationHandler()
        
        # Add instances of our new classes
        self.api_analyzer = APIParameterAnalyzer()
        self.ajax_handler = AJAXResponseHandler()
        
        self.config = config or {}
        
    # ...existing methods remain the same...

    async def perform_search(self, url, search_term, browser=None, options=None):
        """
        Perform search on the target website.
        
        Args:
            url: Target website URL
            search_term: Search term to use
            browser: Optional existing browser instance
            options: Dictionary of search options
            
        Returns:
            Dictionary with search results and metadata
        """
        options = options or {}
        should_close_browser = False
        
        self.logger.info(f"Performing search on {url} for term: {search_term}")
        
        try:
            # Initialize browser if not provided
            if not browser:
                browser = await async_playwright().start()
                chromium = browser.chromium
                should_close_browser = True
                
                browser_instance = await chromium.launch(
                    headless=options.get('headless', True)
                )
            else:
                browser_instance = browser
            
            # Create context and page
            context = await browser_instance.new_context(
                viewport={'width': 1280, 'height': 800},
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
            )
            
            # Apply stealth mode to avoid detection
            page = await context.new_page()
            try:
                await stealth_async(page)
            except NameError:
                self.logger.warning("stealth_async not available, continuing without stealth mode")
            
            # Setup AJAX response monitoring
            ajax_responses = await self.ajax_handler.setup_response_monitoring(page)
            
            # Navigate to the page
            await page.goto(url, wait_until='domcontentloaded')
            
            # Wait for page to be interactive
            await page.wait_for_load_state('networkidle', timeout=10000).catch(lambda _: None)
            
            # Detect and use search form if available
            form_result = await self.form_detector.detect_and_use_search_form(page, search_term)
            
            if form_result['success']:
                self.logger.info("Used search form successfully")
                
                # Wait for AJAX results to load
                await self.ajax_handler.wait_for_ajax_results(page)
                
                # Monitor API requests for API-based search
                api_analysis = await self.api_analyzer.analyze_network_requests(page, search_term)
                
                # Scroll for more results if needed
                await self.ajax_handler.scroll_for_more_results(page, max_scrolls=3)
                
                # Process AJAX responses
                ajax_results = await self.ajax_handler.process_captured_responses(ajax_responses, search_term)
                
                # Handle pagination
                pagination_info = await self.pagination_handler.detect_pagination(page)
                
                # Extract results
                html_content = await page.content()
                current_url = page.url
                
                # Combine all search information
                search_result = {
                    'success': True,
                    'search_term': search_term,
                    'method': 'form',
                    'url': url,
                    'current_url': current_url,
                    'form_info': form_result,
                    'pagination_info': pagination_info,
                    'api_info': api_analysis,
                    'ajax_results': ajax_results,
                    'html_content': html_content
                }
                
                # Clean up
                await context.close()
                if should_close_browser:
                    await browser_instance.close()
                    await browser.stop()
                
                return search_result
            else:
                self.logger.info("No search form found, trying URL-based search")
                
                # TODO: Implement URL-based search as fallback
                
                # Clean up
                await context.close()
                if should_close_browser:
                    await browser_instance.close()
                    await browser.stop()
                
                return {
                    'success': False,
                    'error': 'Search form not found and URL-based search not implemented yet',
                    'url': url,
                    'search_term': search_term
                }
                
        except Exception as e:
            self.logger.error(f"Error performing search: {str(e)}")
            
            if 'context' in locals():
                await context.close()
            
            if should_close_browser and 'browser_instance' in locals():
                await browser_instance.close()
                await browser.stop()
            
            return {
                'success': False,
                'error': str(e),
                'url': url,
                'search_term': search_term
            }

    async def extract_api_parameters(self, url: str, network_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract and analyze API parameters from network requests.
        
        Args:
            url: The base URL of the site
            network_logs: List of network requests captured during site interaction
            
        Returns:
            Dictionary with API parameters analysis
        """
        api_params = {
            "endpoints": {},
            "common_parameters": {},
            "required_parameters": {},
            "parameter_types": {},
            "authentication": {
                "required": False,
                "type": None,
                "headers": []
            }
        }
        
        # Filter for API calls
        api_requests = []
        for request in network_logs:
            # Check if this is likely an API call
            if self._is_api_request(request):
                api_requests.append(request)
        
        if not api_requests:
            return api_params
            
        # Analyze parameters across all API requests
        param_frequency = {}
        endpoint_params = {}
        
        for request in api_requests:
            endpoint = self._extract_endpoint_path(request.get("url", ""))
            method = request.get("method", "GET")
            
            # Initialize endpoint data if not seen before
            endpoint_key = f"{method} {endpoint}"
            if endpoint_key not in api_params["endpoints"]:
                api_params["endpoints"][endpoint_key] = {
                    "url": endpoint,
                    "method": method,
                    "parameters": {},
                    "response_fields": set(),
                    "frequency": 0
                }
            
            api_params["endpoints"][endpoint_key]["frequency"] += 1
            
            # Extract parameters from URL, body, and headers
            params = {}
            
            # URL parameters
            url_params = self._extract_url_parameters(request.get("url", ""))
            params.update(url_params)
            
            # Body parameters
            body_params = self._extract_body_parameters(request.get("postData", {}))
            params.update(body_params)
            
            # Add parameters to endpoint
            for param_name, param_value in params.items():
                # Track parameter frequency
                if param_name not in param_frequency:
                    param_frequency[param_name] = 0
                param_frequency[param_name] += 1
                
                # Add parameter to endpoint
                if param_name not in api_params["endpoints"][endpoint_key]["parameters"]:
                    api_params["endpoints"][endpoint_key]["parameters"][param_name] = {
                        "values": [],
                        "type": self._infer_parameter_type(param_value),
                        "required": True  # Assume required until we see requests without it
                    }
                
                # Add value if unique and not too long
                if param_value and len(param_value) < 100:
                    values = api_params["endpoints"][endpoint_key]["parameters"][param_name]["values"]
                    if param_value not in values:
                        values.append(param_value)
                        if len(values) > 5:  # Cap at 5 example values
                            values.pop(0)
                
                # Track parameters by endpoint
                if endpoint_key not in endpoint_params:
                    endpoint_params[endpoint_key] = set()
                endpoint_params[endpoint_key].add(param_name)
            
            # Check for authentication headers
            auth_headers = self._detect_auth_headers(request.get("headers", {}))
            if auth_headers:
                api_params["authentication"]["required"] = True
                api_params["authentication"]["type"] = auth_headers.get("type")
                if auth_headers.get("header") not in api_params["authentication"]["headers"]:
                    api_params["authentication"]["headers"].append(auth_headers.get("header"))
                    
            # Extract response fields if available
            if "response" in request and "content" in request["response"]:
                try:
                    response_data = json.loads(request["response"]["content"])
                    fields = self._extract_json_fields(response_data)
                    api_params["endpoints"][endpoint_key]["response_fields"].update(fields)
                except (json.JSONDecodeError, TypeError):
                    pass
        
        # Determine common and required parameters
        all_endpoints = set(endpoint_params.keys())
        for param_name, frequency in param_frequency.items():
            # A parameter is common if it appears in more than 50% of endpoints
            if frequency / len(api_requests) > 0.5:
                api_params["common_parameters"][param_name] = frequency
                
            # Check if parameter is present in all requests for its endpoints
            for endpoint_key in all_endpoints:
                if endpoint_key in endpoint_params and param_name in endpoint_params[endpoint_key]:
                    endpoint_request_count = api_params["endpoints"][endpoint_key]["frequency"]
                    if frequency == endpoint_request_count:
                        api_params["required_parameters"][param_name] = True
        
        # Convert response_fields from set to list for JSON serialization
        for endpoint_key in api_params["endpoints"]:
            api_params["endpoints"][endpoint_key]["response_fields"] = list(
                api_params["endpoints"][endpoint_key]["response_fields"]
            )
            
        return api_params
        
    def _is_api_request(self, request: Dict[str, Any]) -> bool:
        """Check if a request is likely an API call."""
        url = request.get("url", "")
        content_type = ""
        
        # Check headers for JSON content type
        for header in request.get("headers", []):
            if header.get("name", "").lower() == "content-type":
                content_type = header.get("value", "").lower()
                break
        
        # Check response headers for JSON content type
        if "response" in request and "headers" in request["response"]:
            for header in request["response"]["headers"]:
                if header.get("name", "").lower() == "content-type":
                    content_type = header.get("value", "").lower()
                    break
        
        # Likely an API if:
        # 1. URL contains /api/, /service/, /rest/, /graphql, or /v\d+/
        # 2. Response or request is JSON
        # 3. URL has .json extension
        return (
            re.search(r'/(api|service|rest|graphql|v\d+)/', url) is not None or
            "application/json" in content_type or
            url.endswith(".json") or
            request.get("method", "") in ["POST", "PUT", "DELETE"] or
            request.get("isXHR", False)
        )
    
    def _extract_endpoint_path(self, url: str) -> str:
        """Extract the endpoint path from a URL."""
        try:
            parsed = urlparse(url)
            path = parsed.path
            
            # Remove trailing slashes
            path = path.rstrip('/')
            
            # If path is empty, use /
            if not path:
                path = "/"
                
            return path
        except:
            return url
    
    def _extract_url_parameters(self, url: str) -> Dict[str, str]:
        """Extract parameters from URL query string."""
        try:
            parsed = urlparse(url)
            return dict(parse_qsl(parsed.query))
        except:
            return {}
    
    def _extract_body_parameters(self, post_data: Dict[str, Any]) -> Dict[str, str]:
        """Extract parameters from request body."""
        params = {}
        
        if not post_data:
            return params
            
        # Check for different formats
        if "text" in post_data:
            # Try to parse as JSON
            try:
                body_data = json.loads(post_data["text"])
                if isinstance(body_data, dict):
                    # Flatten nested dictionary for simplified analysis
                    return self._flatten_dict(body_data)
            except json.JSONDecodeError:
                # Try to parse as query string
                try:
                    return dict(parse_qsl(post_data["text"]))
                except:
                    pass
                    
        # Check for form data
        elif "params" in post_data:
            for param in post_data["params"]:
                name = param.get("name", "")
                value = param.get("value", "")
                if name:
                    params[name] = value
                    
        return params
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '') -> Dict[str, str]:
        """Flatten a nested dictionary for parameter analysis."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            elif isinstance(v, list):
                # For lists, just note it's a list type
                items.append((new_key, str(v) if len(str(v)) < 50 else f"List[{len(v)}]"))
            else:
                items.append((new_key, str(v) if v is not None else ""))
                
        return dict(items)
    
    def _infer_parameter_type(self, value: str) -> str:
        """Infer the data type of a parameter value."""
        if not value:
            return "string"
            
        # Check for boolean
        if value.lower() in ("true", "false"):
            return "boolean"
            
        # Check for integer
        if value.isdigit():
            return "integer"
            
        # Check for float
        try:
            float(value)
            return "float"
        except ValueError:
            pass
            
        # Check for date
        date_patterns = [
            r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
            r'^\d{4}/\d{2}/\d{2}$',  # YYYY/MM/DD
            r'^\d{2}-\d{2}-\d{4}$',  # DD-MM-YYYY
            r'^\d{2}/\d{2}/\d{4}$'   # DD/MM/YYYY
        ]
        
        for pattern in date_patterns:
            if re.match(pattern, value):
                return "date"
                
        # Default to string
        return "string"
    
    def _detect_auth_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Detect authentication headers in a request."""
        auth_headers = {
            "Authorization": "bearer",
            "X-API-Key": "api_key",
            "Api-Key": "api_key",
            "Token": "token",
            "X-Auth-Token": "token",
            "X-Access-Token": "token"
        }
        
        for header in headers:
            name = header.get("name", "")
            value = header.get("value", "")
            
            if name in auth_headers:
                auth_type = auth_headers[name]
                
                # Refine bearer token type
                if name == "Authorization" and value.startswith("Bearer "):
                    auth_type = "bearer"
                elif name == "Authorization" and value.startswith("Basic "):
                    auth_type = "basic"
                    
                return {
                    "type": auth_type,
                    "header": name
                }
                
        return None
        
    def _extract_json_fields(self, data: Any, prefix: str = "") -> set[str]:
        """Extract field names from JSON data."""
        fields = set()
        
        if isinstance(data, dict):
            for key, value in data.items():
                field_name = f"{prefix}.{key}" if prefix else key
                fields.add(field_name)
                
                # Recursively process nested structures
                if isinstance(value, (dict, list)):
                    fields.update(self._extract_json_fields(value, field_name))
                    
        elif isinstance(data, list) and len(data) > 0:
            # For lists, examine the first item as an example
            first_item = data[0]
            if isinstance(first_item, (dict, list)):
                fields.update(self._extract_json_fields(first_item, prefix))
                
        return fields

    # ...remaining methods...