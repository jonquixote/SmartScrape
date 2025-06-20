"""
API detection module for SmartScrape search automation.

This module provides functionality for detecting and analyzing API-based search
interfaces through network request monitoring and parameter analysis.
"""

import logging
import json
import time
import re
from urllib.parse import urljoin, urlparse, parse_qs, urlunparse, urlencode, quote_plus, unquote
from typing import Dict, List, Any, Optional

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
            'q', 'query', 'search', 'keyword', 'keywords', 'term', 'terms', 'text',
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

    async def detect_api_endpoints(self, page, search_term: str) -> Dict[str, Any]:
        """
        Detect API endpoints that could be used for search by monitoring network activity.
        
        Args:
            page: Playwright page object
            search_term: Search term to use for detection
            
        Returns:
            Dict with API detection results
        """
        self.logger.info(f"Detecting API endpoints for search with term: {search_term}")
        
        try:
            # Monitor network requests
            api_analysis = await self.analyze_network_requests(page, search_term)
            
            if not api_analysis.get('success', False):
                return {"success": False, "reason": "Failed to analyze network requests"}
                
            endpoints = api_analysis.get('api_endpoints', [])
            
            if not endpoints:
                return {"success": False, "reason": "No API endpoints detected"}
                
            # Return the best endpoints
            return {
                "success": True,
                "endpoints": endpoints[:3],  # Return top 3 candidates
                "total_detected": len(endpoints)
            }
                
        except Exception as e:
            self.logger.error(f"Error detecting API endpoints: {str(e)}")
            return {"success": False, "reason": str(e)}