"""
API Request Builder for SmartScrape

This module provides functionality for constructing API-based search requests
based on search intent and API endpoint characteristics.
"""

import logging
import json
import re
import asyncio
from urllib.parse import urljoin, urlparse, parse_qs, urlunparse, urlencode, quote_plus
from typing import Dict, List, Any, Optional, Union, Tuple

from components.domain_intelligence import DomainIntelligence
from components.search.api_detection import APIParameterAnalyzer
from utils.http_utils import HttpClient, fetch_json, post
from utils.retry_utils import with_exponential_backoff

logger = logging.getLogger(__name__)

class APIRequestBuilder:
    """
    Builds API requests based on search intent and API endpoint characteristics.
    
    This class:
    - Maps search terms and filters to API request parameters
    - Handles different API styles (RESTful, GraphQL, etc.)
    - Supports various authentication methods
    - Adapts to different parameter formats and structures
    """
    
    def __init__(self, domain_intelligence: Optional[DomainIntelligence] = None):
        """
        Initialize the API request builder.
        
        Args:
            domain_intelligence: Optional domain intelligence component for domain-specific request handling
        """
        self.domain_intelligence = domain_intelligence or DomainIntelligence()
        self.api_analyzer = APIParameterAnalyzer()
        self.http_client = None
        self.known_endpoints = {}
        
        # Common parameter names for pagination
        self.pagination_params = {
            'limit': ['limit', 'size', 'per_page', 'pageSize', 'count', 'max', 'results'],
            'offset': ['offset', 'from', 'start', 'startIndex', 'skip'],
            'page': ['page', 'pageNumber', 'pageNum', 'p']
        }
        
        # Common parameter names for sorting
        self.sorting_params = {
            'sort': ['sort', 'sortBy', 'sortField', 'orderBy', 'order', 'sortOrder'],
            'direction': ['sortDirection', 'direction', 'order', 'orderDirection', 'dir']
        }
    
    async def initialize(self):
        """Initialize the HTTP client if not already done."""
        if not self.http_client:
            self.http_client = await HttpClient.get_instance()
    
    @with_exponential_backoff(max_attempts=3)
    async def build_search_request(self, 
                                  endpoint_info: Dict[str, Any],
                                  search_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a search request for a given API endpoint.
        
        Args:
            endpoint_info: Information about the API endpoint
            search_parameters: Dictionary containing search parameters
                - search_term: The main search term
                - filters: Additional filters to apply
                - pagination: Pagination parameters
                - sorting: Sorting parameters
            
        Returns:
            Dictionary with the constructed request
        """
        await self.initialize()
        
        search_term = search_parameters.get('search_term', '')
        filters = search_parameters.get('filters', {})
        pagination = search_parameters.get('pagination', {})
        sorting = search_parameters.get('sorting', {})
        
        # Base request from API analyzer
        base_request = await self.api_analyzer.generate_api_request(endpoint_info, search_term)
        
        # If there was an error, return it
        if 'error' in base_request:
            return base_request
        
        # Get the method and URL
        method = base_request.get('method', 'GET')
        url = base_request.get('url', '')
        headers = base_request.get('headers', {})
        
        # Handle based on method
        if method == 'GET':
            return await self._build_get_request(url, headers, search_term, filters, pagination, sorting, endpoint_info)
        elif method == 'POST':
            data = base_request.get('data', '')
            content_type = headers.get('Content-Type', '')
            return await self._build_post_request(url, headers, content_type, data, search_term, filters, pagination, sorting, endpoint_info)
        else:
            return {'error': f'Unsupported HTTP method: {method}'}
    
    async def _build_get_request(self, 
                               url: str,
                               headers: Dict[str, str],
                               search_term: str,
                               filters: Dict[str, Any],
                               pagination: Dict[str, Any],
                               sorting: Dict[str, Any],
                               endpoint_info: Dict[str, Any]) -> Dict[str, Any]:
        """Build a GET request with all parameters."""
        # Parse the URL to get existing parameters
        parsed_url = urlparse(url)
        query_params = dict(parse_qs(parsed_url.query))
        
        # Convert all values to single strings (not lists)
        params = {k: v[0] if isinstance(v, list) and len(v) > 0 else v 
                 for k, v in query_params.items()}
        
        # Add filters
        params = self._add_filters_to_params(params, filters, endpoint_info)
        
        # Add pagination
        params = self._add_pagination_to_params(params, pagination, endpoint_info)
        
        # Add sorting
        params = self._add_sorting_to_params(params, sorting, endpoint_info)
        
        # Reconstruct the URL with updated parameters
        new_query = urlencode(params)
        new_url = urlunparse((
            parsed_url.scheme,
            parsed_url.netloc,
            parsed_url.path,
            parsed_url.params,
            new_query,
            parsed_url.fragment
        ))
        
        return {
            'method': 'GET',
            'url': new_url,
            'headers': headers
        }
    
    async def _build_post_request(self,
                                url: str,
                                headers: Dict[str, str],
                                content_type: str,
                                data: str,
                                search_term: str,
                                filters: Dict[str, Any],
                                pagination: Dict[str, Any],
                                sorting: Dict[str, Any],
                                endpoint_info: Dict[str, Any]) -> Dict[str, Any]:
        """Build a POST request with all parameters."""
        # Handle different content types
        if 'application/json' in content_type:
            try:
                # Parse the JSON data
                json_data = json.loads(data)
                
                # Add filters
                json_data = self._add_filters_to_json(json_data, filters, endpoint_info)
                
                # Add pagination
                json_data = self._add_pagination_to_json(json_data, pagination, endpoint_info)
                
                # Add sorting
                json_data = self._add_sorting_to_json(json_data, sorting, endpoint_info)
                
                # Convert back to string
                new_data = json.dumps(json_data)
                
                return {
                    'method': 'POST',
                    'url': url,
                    'headers': headers,
                    'data': new_data
                }
                
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON data: {data}")
                return {
                    'method': 'POST',
                    'url': url,
                    'headers': headers,
                    'data': data
                }
                
        elif 'application/x-www-form-urlencoded' in content_type:
            # Parse form data
            form_params = {}
            for param in data.split('&'):
                if '=' in param:
                    key, value = param.split('=', 1)
                    form_params[key] = value
            
            # Add filters
            form_params = self._add_filters_to_params(form_params, filters, endpoint_info)
            
            # Add pagination
            form_params = self._add_pagination_to_params(form_params, pagination, endpoint_info)
            
            # Add sorting
            form_params = self._add_sorting_to_params(form_params, sorting, endpoint_info)
            
            # Convert back to form data
            new_data = urlencode(form_params)
            
            return {
                'method': 'POST',
                'url': url,
                'headers': headers,
                'data': new_data
            }
            
        elif 'application/graphql' in content_type or 'graphql' in url.lower():
            # Handle GraphQL request
            return await self._build_graphql_request(url, headers, data, search_term, filters, pagination, sorting, endpoint_info)
            
        else:
            # For other content types, return as is
            return {
                'method': 'POST',
                'url': url,
                'headers': headers,
                'data': data
            }
    
    async def _build_graphql_request(self,
                                   url: str,
                                   headers: Dict[str, str],
                                   data: str,
                                   search_term: str,
                                   filters: Dict[str, Any],
                                   pagination: Dict[str, Any],
                                   sorting: Dict[str, Any],
                                   endpoint_info: Dict[str, Any]) -> Dict[str, Any]:
        """Build a GraphQL request with all parameters."""
        try:
            # Parse the GraphQL request
            graphql_data = json.loads(data)
            
            # Get variables and update them
            variables = graphql_data.get('variables', {})
            
            # Add filters to variables
            for filter_key, filter_value in filters.items():
                # Try to map filter key to a GraphQL variable
                mapped_key = self._map_filter_to_graphql_variable(filter_key, endpoint_info)
                if mapped_key:
                    variables[mapped_key] = filter_value
            
            # Add pagination to variables
            if pagination:
                # Map pagination keys to GraphQL variables
                page_size = pagination.get('limit', pagination.get('size', 10))
                page_num = pagination.get('page', 0)
                offset = pagination.get('offset', 0)
                
                # Try to find matching variables in the GraphQL query
                if 'limit' in variables or any(p in variables for p in self.pagination_params['limit']):
                    for param in self.pagination_params['limit']:
                        if param in variables:
                            variables[param] = page_size
                            break
                    
                if 'page' in variables or any(p in variables for p in self.pagination_params['page']):
                    for param in self.pagination_params['page']:
                        if param in variables:
                            variables[param] = page_num
                            break
                
                if 'offset' in variables or any(p in variables for p in self.pagination_params['offset']):
                    for param in self.pagination_params['offset']:
                        if param in variables:
                            variables[param] = offset
                            break
            
            # Add sorting to variables
            if sorting:
                sort_field = sorting.get('field', '')
                sort_dir = sorting.get('direction', 'desc')
                
                # Try to find matching variables in the GraphQL query
                if any(p in variables for p in self.sorting_params['sort']):
                    for param in self.sorting_params['sort']:
                        if param in variables:
                            variables[param] = sort_field
                            break
                
                if any(p in variables for p in self.sorting_params['direction']):
                    for param in self.sorting_params['direction']:
                        if param in variables:
                            variables[param] = sort_dir
                            break
            
            # Update the variables in the GraphQL request
            graphql_data['variables'] = variables
            
            # Convert back to string
            new_data = json.dumps(graphql_data)
            
            return {
                'method': 'POST',
                'url': url,
                'headers': headers,
                'data': new_data
            }
            
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse GraphQL data: {data}")
            return {
                'method': 'POST',
                'url': url,
                'headers': headers,
                'data': data
            }
    
    def _add_filters_to_params(self, 
                              params: Dict[str, Any], 
                              filters: Dict[str, Any],
                              endpoint_info: Dict[str, Any]) -> Dict[str, Any]:
        """Add filters to request parameters."""
        if not filters:
            return params
            
        # Get parameter mappings for this endpoint, if available
        param_mappings = self._get_parameter_mappings(endpoint_info)
        
        # Add each filter to parameters
        for filter_key, filter_value in filters.items():
            # Try to map filter key to a parameter name
            param_name = self._map_filter_to_param(filter_key, param_mappings)
            
            if param_name:
                # Parameter mapping found
                params[param_name] = filter_value
            else:
                # No mapping, use direct name or prefix
                params[filter_key] = filter_value
        
        return params
    
    def _add_pagination_to_params(self, 
                                 params: Dict[str, Any], 
                                 pagination: Dict[str, Any],
                                 endpoint_info: Dict[str, Any]) -> Dict[str, Any]:
        """Add pagination parameters to request parameters."""
        if not pagination:
            return params
            
        # Get standard pagination parameters
        limit = pagination.get('limit', pagination.get('size', 10))
        page = pagination.get('page', 0)
        offset = pagination.get('offset', 0)
        
        # Check existing parameters to determine pagination style
        has_limit = any(param in params for param in self.pagination_params['limit'])
        has_page = any(param in params for param in self.pagination_params['page'])
        has_offset = any(param in params for param in self.pagination_params['offset'])
        
        # If no pagination parameters exist, try to infer from endpoint info
        if not (has_limit or has_page or has_offset):
            # Use domain intelligence to determine common pagination for this type of API
            pagination_style = self._infer_pagination_style(endpoint_info)
            
            if pagination_style == 'limit_offset':
                has_limit = has_offset = True
            elif pagination_style == 'page_size':
                has_page = has_limit = True
        
        # Apply pagination parameters
        if has_limit:
            # Find the right parameter name for limit
            for param in self.pagination_params['limit']:
                if param in params:
                    params[param] = limit
                    break
            else:
                # Not found, use the first one
                params[self.pagination_params['limit'][0]] = limit
        
        if has_page:
            # Find the right parameter name for page
            for param in self.pagination_params['page']:
                if param in params:
                    params[param] = page
                    break
            else:
                # Not found, use the first one
                params[self.pagination_params['page'][0]] = page
                
        if has_offset:
            # Find the right parameter name for offset
            for param in self.pagination_params['offset']:
                if param in params:
                    params[param] = offset
                    break
            else:
                # Not found, use the first one
                params[self.pagination_params['offset'][0]] = offset
        
        return params
    
    def _add_sorting_to_params(self, 
                              params: Dict[str, Any], 
                              sorting: Dict[str, Any],
                              endpoint_info: Dict[str, Any]) -> Dict[str, Any]:
        """Add sorting parameters to request parameters."""
        if not sorting:
            return params
            
        sort_field = sorting.get('field', '')
        sort_direction = sorting.get('direction', 'desc')
        
        if not sort_field:
            return params
            
        # Check existing parameters to determine sorting style
        has_sort = any(param in params for param in self.sorting_params['sort'])
        has_direction = any(param in params for param in self.sorting_params['direction'])
        
        # Apply sort field
        if has_sort:
            # Find the right parameter name for sort field
            for param in self.sorting_params['sort']:
                if param in params:
                    params[param] = sort_field
                    break
            else:
                # Not found, use the first one
                params[self.sorting_params['sort'][0]] = sort_field
        else:
            # No sort parameter found, try standard ones
            params[self.sorting_params['sort'][0]] = sort_field
        
        # Apply sort direction
        if has_direction:
            # Find the right parameter name for sort direction
            for param in self.sorting_params['direction']:
                if param in params:
                    params[param] = sort_direction
                    break
        else:
            # Check if we use a combined format like "sort=field:direction"
            for param in self.sorting_params['sort']:
                if param in params:
                    params[param] = f"{sort_field}:{sort_direction}"
                    break
        
        return params
    
    def _add_filters_to_json(self, 
                            json_data: Dict[str, Any], 
                            filters: Dict[str, Any],
                            endpoint_info: Dict[str, Any]) -> Dict[str, Any]:
        """Add filters to JSON request data."""
        if not filters:
            return json_data
            
        # Check if there's a dedicated "filters" object in the JSON
        if 'filters' in json_data:
            for filter_key, filter_value in filters.items():
                json_data['filters'][filter_key] = filter_value
            return json_data
            
        # Try to add filters directly to the JSON
        for filter_key, filter_value in filters.items():
            # Try to map filter key to a parameter name
            param_mappings = self._get_parameter_mappings(endpoint_info)
            param_name = self._map_filter_to_param(filter_key, param_mappings)
            
            if param_name:
                # Parameter mapping found
                json_data[param_name] = filter_value
            else:
                # No mapping, use direct name
                json_data[filter_key] = filter_value
        
        return json_data
    
    def _add_pagination_to_json(self, 
                               json_data: Dict[str, Any], 
                               pagination: Dict[str, Any],
                               endpoint_info: Dict[str, Any]) -> Dict[str, Any]:
        """Add pagination parameters to JSON request data."""
        if not pagination:
            return json_data
            
        # Check if there's a dedicated "pagination" object in the JSON
        if 'pagination' in json_data:
            for key, value in pagination.items():
                json_data['pagination'][key] = value
            return json_data
            
        # Get standard pagination parameters
        limit = pagination.get('limit', pagination.get('size', 10))
        page = pagination.get('page', 0)
        offset = pagination.get('offset', 0)
        
        # Apply pagination parameters to the root level
        # Find appropriate parameter names
        for param in self.pagination_params['limit']:
            if param in json_data:
                json_data[param] = limit
                break
        else:
            # Not found, use the first one
            json_data[self.pagination_params['limit'][0]] = limit
        
        if 'page' in pagination:
            for param in self.pagination_params['page']:
                if param in json_data:
                    json_data[param] = page
                    break
            else:
                # Not found, use the first one if we're using page-based pagination
                if 'offset' not in pagination:
                    json_data[self.pagination_params['page'][0]] = page
        
        if 'offset' in pagination:
            for param in self.pagination_params['offset']:
                if param in json_data:
                    json_data[param] = offset
                    break
            else:
                # Not found, use the first one if we're using offset-based pagination
                if 'page' not in pagination:
                    json_data[self.pagination_params['offset'][0]] = offset
        
        return json_data
    
    def _add_sorting_to_json(self, 
                            json_data: Dict[str, Any], 
                            sorting: Dict[str, Any],
                            endpoint_info: Dict[str, Any]) -> Dict[str, Any]:
        """Add sorting parameters to JSON request data."""
        if not sorting:
            return json_data
            
        # Check if there's a dedicated "sort" or "ordering" object in the JSON
        if 'sort' in json_data and isinstance(json_data['sort'], dict):
            for key, value in sorting.items():
                json_data['sort'][key] = value
            return json_data
            
        if 'ordering' in json_data and isinstance(json_data['ordering'], dict):
            for key, value in sorting.items():
                json_data['ordering'][key] = value
            return json_data
            
        # Apply sorting parameters to the root level
        sort_field = sorting.get('field', '')
        sort_direction = sorting.get('direction', 'desc')
        
        if not sort_field:
            return json_data
            
        # Find appropriate parameter names
        for param in self.sorting_params['sort']:
            if param in json_data:
                json_data[param] = sort_field
                break
        else:
            # Not found, use the first one
            json_data[self.sorting_params['sort'][0]] = sort_field
        
        # Apply sort direction
        for param in self.sorting_params['direction']:
            if param in json_data:
                json_data[param] = sort_direction
                break
        else:
            # Not found, see if we need to use a combined format
            combined_sort = f"{sort_field}:{sort_direction}"
            for param in self.sorting_params['sort']:
                if param in json_data:
                    json_data[param] = combined_sort
                    break
        
        return json_data
    
    def _map_filter_to_param(self, 
                            filter_key: str, 
                            param_mappings: Dict[str, str]) -> Optional[str]:
        """Map a filter key to a parameter name."""
        # Check direct mapping
        if filter_key in param_mappings:
            return param_mappings[filter_key]
            
        # Check similar names
        for param_key, param_name in param_mappings.items():
            if filter_key.lower() == param_key.lower():
                return param_name
                
        # No mapping found
        return None
    
    def _map_filter_to_graphql_variable(self, 
                                       filter_key: str, 
                                       endpoint_info: Dict[str, Any]) -> Optional[str]:
        """Map a filter key to a GraphQL variable name."""
        variables = endpoint_info.get('variables', {})
        
        # Check direct match
        if filter_key in variables:
            return filter_key
            
        # Check similar names
        for var_name in variables.keys():
            if filter_key.lower() == var_name.lower():
                return var_name
                
            # Check for common filter prefixes
            if var_name.lower().endswith(filter_key.lower()):
                return var_name
                
            # Check for filter in variable name
            if filter_key.lower() in var_name.lower():
                return var_name
                
        # No mapping found
        return None
    
    def _get_parameter_mappings(self, endpoint_info: Dict[str, Any]) -> Dict[str, str]:
        """Get parameter mappings for an endpoint."""
        endpoint = endpoint_info.get('endpoint', '')
        
        # Check if we have mappings for this endpoint
        if endpoint in self.known_endpoints:
            return self.known_endpoints[endpoint].get('parameter_mappings', {})
            
        # No mappings, return empty dict
        return {}
    
    def _infer_pagination_style(self, endpoint_info: Dict[str, Any]) -> str:
        """Infer pagination style for an endpoint."""
        endpoint = endpoint_info.get('endpoint', '')
        
        # Check if we have information about this endpoint
        if endpoint in self.known_endpoints:
            return self.known_endpoints[endpoint].get('pagination_style', 'limit_offset')
            
        # No information, guess based on known characteristics
        parameters = endpoint_info.get('parameters', {})
        
        if any(param in parameters for param in self.pagination_params['page']):
            return 'page_size'
        else:
            return 'limit_offset'
    
    @with_exponential_backoff(max_attempts=3)
    async def execute_search_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a search request and return the response.
        
        Args:
            request: The request to execute
            
        Returns:
            Dictionary with the response
        """
        await self.initialize()
        
        method = request.get('method', 'GET')
        url = request.get('url', '')
        headers = request.get('headers', {})
        data = request.get('data', '')
        
        try:
            if method == 'GET':
                response = await self.http_client.get(url, headers=headers)
                return {
                    'status_code': response.status_code,
                    'content': response.text,
                    'content_type': response.headers.get('Content-Type', ''),
                    'url': url
                }
            elif method == 'POST':
                response = await self.http_client.post(url, data=data, headers=headers)
                return {
                    'status_code': response.status_code,
                    'content': response.text,
                    'content_type': response.headers.get('Content-Type', ''),
                    'url': url
                }
            else:
                return {'error': f'Unsupported HTTP method: {method}'}
                
        except Exception as e:
            logger.error(f"Error executing search request: {str(e)}")
            return {'error': str(e)}
    
    async def learn_from_response(self, 
                                 endpoint_info: Dict[str, Any], 
                                 request: Dict[str, Any],
                                 response: Dict[str, Any],
                                 search_parameters: Dict[str, Any]) -> None:
        """
        Learn from a search response to improve future requests.
        
        Args:
            endpoint_info: Information about the API endpoint
            request: The request that was executed
            response: The response that was received
            search_parameters: The search parameters that were used
        """
        endpoint = endpoint_info.get('endpoint', '')
        
        # Skip if there was an error
        if 'error' in response:
            return
            
        # Initialize endpoint info if not already present
        if endpoint not in self.known_endpoints:
            self.known_endpoints[endpoint] = {
                'parameter_mappings': {},
                'pagination_style': 'limit_offset',
                'success_rate': 0,
                'request_count': 0
            }
            
        endpoint_data = self.known_endpoints[endpoint]
        
        # Update request count
        endpoint_data['request_count'] += 1
        
        # Check if the response was successful
        status_code = response.get('status_code', 0)
        if 200 <= status_code < 300:
            endpoint_data['success_rate'] = (
                endpoint_data['success_rate'] * (endpoint_data['request_count'] - 1) + 1
            ) / endpoint_data['request_count']
            
            # Try to parse and analyze the response
            try:
                response_content = response.get('content', '')
                search_term = search_parameters.get('search_term', '')
                
                analysis = await self.api_analyzer.analyze_api_response(response_content, search_term)
                
                if analysis.get('results_found', False):
                    # If results were found, we can learn parameter mappings
                    self._learn_parameter_mappings(
                        endpoint_data, 
                        request, 
                        search_parameters
                    )
            except Exception as e:
                logger.warning(f"Error analyzing response: {str(e)}")
        else:
            # Response was not successful
            endpoint_data['success_rate'] = (
                endpoint_data['success_rate'] * (endpoint_data['request_count'] - 1)
            ) / endpoint_data['request_count']
    
    def _learn_parameter_mappings(self,
                                 endpoint_data: Dict[str, Any],
                                 request: Dict[str, Any],
                                 search_parameters: Dict[str, Any]) -> None:
        """Learn parameter mappings from a successful request."""
        method = request.get('method', 'GET')
        url = request.get('url', '')
        data = request.get('data', '')
        
        # Get the request parameters
        params = {}
        if method == 'GET':
            # Extract parameters from URL
            parsed_url = urlparse(url)
            query_params = parse_qs(parsed_url.query)
            params = {k: v[0] if isinstance(v, list) and len(v) > 0 else v 
                     for k, v in query_params.items()}
        elif method == 'POST':
            # Extract parameters from data
            if data:
                try:
                    # Try JSON first
                    params = json.loads(data)
                except json.JSONDecodeError:
                    # Try form data
                    try:
                        form_params = {}
                        for param in data.split('&'):
                            if '=' in param:
                                key, value = param.split('=', 1)
                                form_params[key] = value
                        params = form_params
                    except Exception:
                        # Failed to parse
                        pass
        
        # Learn parameter mappings
        parameter_mappings = endpoint_data.get('parameter_mappings', {})
        
        # Learn search term parameter
        search_term = search_parameters.get('search_term', '')
        if search_term:
            for param, value in params.items():
                if isinstance(value, str) and search_term.lower() in value.lower():
                    parameter_mappings['search_term'] = param
                    break
        
        # Learn filter parameters
        filters = search_parameters.get('filters', {})
        for filter_key, filter_value in filters.items():
            for param, value in params.items():
                if value == filter_value:
                    parameter_mappings[filter_key] = param
                    break
        
        # Learn pagination parameters
        pagination = search_parameters.get('pagination', {})
        if pagination:
            # Determine pagination style
            if 'page' in pagination and 'limit' in pagination:
                endpoint_data['pagination_style'] = 'page_size'
                
                # Map page parameter
                for param, value in params.items():
                    if value == pagination.get('page', 0):
                        parameter_mappings['page'] = param
                        break
                        
                # Map limit parameter
                limit = pagination.get('limit', pagination.get('size', 10))
                for param, value in params.items():
                    if value == limit:
                        parameter_mappings['limit'] = param
                        break
                        
            elif 'offset' in pagination and 'limit' in pagination:
                endpoint_data['pagination_style'] = 'limit_offset'
                
                # Map offset parameter
                for param, value in params.items():
                    if value == pagination.get('offset', 0):
                        parameter_mappings['offset'] = param
                        break
                        
                # Map limit parameter
                limit = pagination.get('limit', pagination.get('size', 10))
                for param, value in params.items():
                    if value == limit:
                        parameter_mappings['limit'] = param
                        break
                        
        # Learn sorting parameters
        sorting = search_parameters.get('sorting', {})
        if sorting:
            sort_field = sorting.get('field', '')
            sort_direction = sorting.get('direction', 'desc')
            
            if sort_field:
                for param, value in params.items():
                    if value == sort_field:
                        parameter_mappings['sort_field'] = param
                        break
                    elif isinstance(value, str) and sort_field in value:
                        parameter_mappings['sort_field'] = param
                        break
                        
            if sort_direction:
                for param, value in params.items():
                    if value == sort_direction:
                        parameter_mappings['sort_direction'] = param
                        break
        
        # Update the parameter mappings
        endpoint_data['parameter_mappings'] = parameter_mappings