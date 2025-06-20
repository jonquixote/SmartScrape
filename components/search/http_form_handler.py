"""
HTTP-Based Form Submission Handler

This module provides specialized functionality for handling form submissions
via direct HTTP requests without requiring a browser. This approach is faster
and uses fewer resources than browser-based form handling.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from urllib.parse import urljoin, urlparse, parse_qsl, urlencode
import json
import re

import httpx
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

from utils.retry_utils import with_exponential_backoff
from utils.extraction_utils import extract_links

logger = logging.getLogger(__name__)

class HTTPFormHandler:
    """
    Handles form analysis and submission via direct HTTP requests without using a browser.
    
    This approach is more efficient than browser automation for simple forms that don't 
    require JavaScript execution. It's particularly useful for:
    
    1. Simple search forms
    2. Newsletter signups
    3. Basic contact forms
    4. Any form that doesn't rely on JS for validation or submission
    """
    
    def __init__(self, timeout: int = 30, max_retries: int = 3, follow_redirects: bool = True):
        """
        Initialize the HTTP form handler.
        
        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts for failed requests
            follow_redirects: Whether to follow HTTP redirects
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.follow_redirects = follow_redirects
        self.user_agent = UserAgent().random
        
    async def analyze_page(self, url: str) -> Dict[str, Any]:
        """
        Analyze a page to find and categorize forms.
        
        Args:
            url: URL of the page to analyze
            
        Returns:
            Dictionary containing form analysis results
        """
        try:
            html = await self._fetch_page(url)
            if not html:
                return {"success": False, "error": "Failed to fetch page", "forms": []}
            
            soup = BeautifulSoup(html, 'html.parser')
            forms = soup.find_all('form')
            
            if not forms:
                return {"success": True, "form_count": 0, "forms": []}
            
            analyzed_forms = []
            for i, form in enumerate(forms):
                form_data = self._analyze_form(form, url, i)
                analyzed_forms.append(form_data)
            
            return {
                "success": True,
                "form_count": len(forms),
                "forms": analyzed_forms
            }
            
        except Exception as e:
            logger.error(f"Error analyzing page {url}: {str(e)}")
            return {"success": False, "error": str(e), "forms": []}
    
    def _analyze_form(self, form, base_url: str, index: int) -> Dict[str, Any]:
        """
        Analyze a single form to determine its purpose and fields.
        
        Args:
            form: BeautifulSoup form element
            base_url: Base URL of the page containing the form
            index: Index of the form on the page
            
        Returns:
            Dictionary with form analysis
        """
        # Extract basic form properties
        action = form.get('action', '')
        method = form.get('method', 'get').lower()
        form_id = form.get('id', '')
        form_class = ' '.join(form.get('class', []))
        enctype = form.get('enctype', 'application/x-www-form-urlencoded')
        
        # Handle relative action URLs
        if action:
            action = urljoin(base_url, action)
        else:
            action = base_url
        
        # Analyze form purpose
        purpose = self._determine_form_purpose(form)
        
        # Analyze form fields
        fields = self._extract_form_fields(form)
        
        # Determine which field is likely the main search/query field
        search_field = None
        if purpose == "search":
            search_field = self._identify_search_field(fields)
        
        return {
            "index": index,
            "action": action,
            "method": method,
            "id": form_id,
            "class": form_class,
            "enctype": enctype,
            "purpose": purpose,
            "fields": fields,
            "search_field": search_field,
            "confidence": self._calculate_form_confidence(form, purpose)
        }
    
    def _determine_form_purpose(self, form) -> str:
        """
        Determine the likely purpose of a form.
        
        Args:
            form: BeautifulSoup form element
            
        Returns:
            String indicating the form purpose
        """
        # Examine form attributes
        form_id = form.get('id', '').lower()
        form_class = ' '.join(form.get('class', [])).lower()
        form_action = form.get('action', '').lower()
        
        # Look for search indicators
        search_indicators = ['search', 'find', 'query', 'lookup', 'seek']
        if any(term in attr for term in search_indicators 
               for attr in [form_id, form_class, form_action]):
            return "search"
        
        # Look for login indicators
        login_indicators = ['login', 'signin', 'sign-in', 'auth']
        if any(term in attr for term in login_indicators 
               for attr in [form_id, form_class, form_action]):
            return "login"
        
        # Look for registration indicators
        reg_indicators = ['register', 'signup', 'sign-up', 'create']
        if any(term in attr for term in reg_indicators 
               for attr in [form_id, form_class, form_action]):
            return "registration"
        
        # Look for contact form indicators
        contact_indicators = ['contact', 'feedback', 'message', 'support']
        if any(term in attr for term in contact_indicators 
               for attr in [form_id, form_class, form_action]):
            return "contact"
        
        # Look for newsletter indicators
        newsletter_indicators = ['newsletter', 'subscribe', 'email']
        if any(term in attr for term in newsletter_indicators 
               for attr in [form_id, form_class, form_action]):
            return "newsletter"
        
        # Analyze input fields for further clues
        inputs = form.find_all('input')
        
        # Count different types of inputs
        text_inputs = [i for i in inputs if i.get('type', 'text').lower() in ['text', 'search', '']]
        password_inputs = [i for i in inputs if i.get('type', '').lower() == 'password']
        email_inputs = [i for i in inputs if i.get('type', '').lower() == 'email']
        
        # Check for password fields (login/registration)
        if password_inputs:
            # If only one text/email field with password, likely login
            if len(text_inputs) + len(email_inputs) <= 2:
                return "login"
            # Multiple fields with password, likely registration
            return "registration"
        
        # Single text input with submit, likely search
        if len(text_inputs) == 1 and form.find('input', {'type': 'submit'}):
            input_attrs = ' '.join([
                text_inputs[0].get('name', ''), 
                text_inputs[0].get('id', ''),
                text_inputs[0].get('placeholder', '')
            ]).lower()
            
            if any(term in input_attrs for term in search_indicators):
                return "search"
        
        # If we have an email field, might be newsletter
        if email_inputs and not password_inputs:
            return "newsletter"
        
        # Default to generic form
        return "generic"
    
    def _extract_form_fields(self, form) -> List[Dict[str, Any]]:
        """
        Extract all fields from a form.
        
        Args:
            form: BeautifulSoup form element
            
        Returns:
            List of dictionaries with field information
        """
        fields = []
        
        # Process input elements
        for input_el in form.find_all('input'):
            input_type = input_el.get('type', 'text').lower()
            
            # Skip submit/reset/button inputs
            if input_type in ['submit', 'reset', 'button', 'image']:
                continue
                
            name = input_el.get('name', '')
            if not name:
                continue  # Skip inputs without name
                
            field = {
                "name": name,
                "type": input_type,
                "id": input_el.get('id', ''),
                "required": input_el.get('required') is not None,
                "placeholder": input_el.get('placeholder', ''),
                "default_value": input_el.get('value', ''),
                "element": "input"
            }
            
            # For checkboxes and radio buttons, note if checked
            if input_type in ['checkbox', 'radio']:
                field["checked"] = input_el.get('checked') is not None
                
            fields.append(field)
        
        # Process select elements
        for select in form.find_all('select'):
            name = select.get('name', '')
            if not name:
                continue
                
            options = []
            selected_value = None
            
            for option in select.find_all('option'):
                value = option.get('value', option.text.strip())
                selected = option.get('selected') is not None
                
                options.append({
                    "value": value,
                    "text": option.text.strip(),
                    "selected": selected
                })
                
                if selected:
                    selected_value = value
            
            fields.append({
                "name": name,
                "type": "select",
                "id": select.get('id', ''),
                "required": select.get('required') is not None,
                "options": options,
                "selected_value": selected_value,
                "element": "select"
            })
        
        # Process textarea elements
        for textarea in form.find_all('textarea'):
            name = textarea.get('name', '')
            if not name:
                continue
                
            fields.append({
                "name": name,
                "type": "textarea",
                "id": textarea.get('id', ''),
                "required": textarea.get('required') is not None,
                "placeholder": textarea.get('placeholder', ''),
                "default_value": textarea.text.strip(),
                "element": "textarea"
            })
        
        return fields
    
    def _identify_search_field(self, fields: List[Dict[str, Any]]) -> Optional[str]:
        """
        Identify which field is most likely the main search/query field.
        
        Args:
            fields: List of field dictionaries
            
        Returns:
            Name of the field most likely to be the search field, or None
        """
        if not fields:
            return None
            
        search_indicators = ['search', 'query', 'q', 'find', 'keyword', 'term']
        
        # First pass: look for fields with search indicators in name/id
        for field in fields:
            name = field["name"].lower()
            field_id = field["id"].lower() if field["id"] else ""
            
            if field["type"] in ['text', 'search', ''] and any(
                indicator in attr for indicator in search_indicators
                for attr in [name, field_id]):
                return field["name"]
        
        # Second pass: look for any text inputs
        for field in fields:
            if field["type"] in ['text', 'search', '']:
                return field["name"]
        
        # No suitable field found
        return None
    
    def _calculate_form_confidence(self, form, purpose: str) -> float:
        """
        Calculate confidence score for the form purpose determination.
        
        Args:
            form: BeautifulSoup form element
            purpose: Determined purpose of the form
            
        Returns:
            Confidence score from 0.0 to 1.0
        """
        score = 0.5  # Start with baseline confidence
        
        # More confident for forms with IDs
        if form.get('id'):
            score += 0.1
            
        # More confident for forms with explicit action URLs
        if form.get('action'):
            score += 0.1
            
        # Confidence based on the number and types of fields
        inputs = form.find_all('input')
        selects = form.find_all('select')
        textareas = form.find_all('textarea')
        
        # Complex forms (many fields) are more confidently classified
        if len(inputs) + len(selects) + len(textareas) > 5:
            score += 0.1
            
        # Purpose-specific confidence boosts
        if purpose == "search" and len(inputs) <= 3:
            # Simple search forms are usually very distinctive
            score += 0.2
        elif purpose == "login" and any(i.get('type') == 'password' for i in inputs):
            # Login forms with password fields are distinctive
            score += 0.2
            
        # Cap score at 1.0
        return min(score, 1.0)
        
    @with_exponential_backoff(max_retries=3, exceptions=(httpx.HTTPError,))
    async def _fetch_page(self, url: str) -> Optional[str]:
        """
        Fetch a page with retry logic.
        
        Args:
            url: URL to fetch
            
        Returns:
            HTML content of the page or None
        """
        try:
            headers = {"User-Agent": self.user_agent}
            async with httpx.AsyncClient(
                follow_redirects=self.follow_redirects, 
                timeout=self.timeout
            ) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                return response.text
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None
    
    async def submit_form(self, 
                         form_info: Dict[str, Any], 
                         data: Dict[str, str],
                         custom_headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Submit a form using the provided data.
        
        Args:
            form_info: Form information dictionary from analyze_page
            data: Data to submit with the form
            custom_headers: Optional custom headers for the request
            
        Returns:
            Dictionary with submission results
        """
        try:
            # Extract form details
            action = form_info.get("action", "")
            method = form_info.get("method", "get").lower()
            enctype = form_info.get("enctype", "application/x-www-form-urlencoded")
            
            if not action:
                return {"success": False, "error": "No form action URL provided"}
            
            # Prepare form data by combining default values with provided data
            form_data = {}
            
            # Start with default values from form fields
            for field in form_info.get("fields", []):
                field_name = field.get("name", "")
                
                if field.get("type") in ["checkbox", "radio"]:
                    if field.get("checked", False):
                        form_data[field_name] = field.get("default_value", "on")
                elif field.get("type") == "select":
                    if field.get("selected_value") is not None:
                        form_data[field_name] = field.get("selected_value")
                else:
                    default_value = field.get("default_value", "")
                    if default_value:
                        form_data[field_name] = default_value
            
            # Override with provided data
            form_data.update(data)
            
            # Prepare headers
            headers = {"User-Agent": self.user_agent}
            if custom_headers:
                headers.update(custom_headers)
                
            # Set content type based on enctype
            if method == "post":
                if enctype == "multipart/form-data":
                    # For multipart, let httpx handle it
                    pass
                elif enctype == "application/json":
                    headers["Content-Type"] = "application/json"
                    form_data = json.dumps(form_data)
                else:
                    # Default to application/x-www-form-urlencoded
                    headers["Content-Type"] = "application/x-www-form-urlencoded"
            
            # Make the request
            async with httpx.AsyncClient(
                follow_redirects=self.follow_redirects, 
                timeout=self.timeout
            ) as client:
                if method == "post":
                    if enctype == "application/json" and isinstance(form_data, str):
                        response = await client.post(action, content=form_data, headers=headers)
                    else:
                        response = await client.post(action, data=form_data, headers=headers)
                else:
                    # For GET requests, data goes in query params
                    response = await client.get(action, params=form_data, headers=headers)
                
                # Process response
                return {
                    "success": True,
                    "status_code": response.status_code,
                    "url": str(response.url),
                    "content_type": response.headers.get("content-type", ""),
                    "response_text": response.text,
                    "cookies": dict(response.cookies),
                    "headers": dict(response.headers),
                    "links": extract_links(response.text, str(response.url))
                }
                
        except Exception as e:
            logger.error(f"Error submitting form: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def search(self, url: str, query: str, form_index: int = 0) -> Dict[str, Any]:
        """
        Perform a search using a form on the specified URL.
        
        Args:
            url: URL containing the search form
            query: Search query
            form_index: Index of the form to use (if multiple forms on page)
            
        Returns:
            Dictionary with search results
        """
        try:
            # Analyze the page to find forms
            analysis = await self.analyze_page(url)
            
            if not analysis.get("success", False):
                return {"success": False, "error": analysis.get("error", "Failed to analyze page")}
                
            forms = analysis.get("forms", [])
            if not forms:
                return {"success": False, "error": "No forms found on the page"}
                
            # Make sure the requested form index exists
            if form_index >= len(forms):
                if len(forms) > 0:
                    # Default to the first form
                    form_index = 0
                else:
                    return {"success": False, "error": f"Form index {form_index} not found"}
            
            form_info = forms[form_index]
            
            # Determine which field to use for the search query
            search_field = form_info.get("search_field")
            if not search_field:
                # If no search field identified, look for the first text input
                for field in form_info.get("fields", []):
                    if field.get("type") in ["text", "search", ""]:
                        search_field = field.get("name")
                        break
                        
            if not search_field:
                return {"success": False, "error": "No suitable search field found in the form"}
                
            # Prepare data with the search query
            data = {search_field: query}
            
            # Submit the form
            result = await self.submit_form(form_info, data)
            
            if not result.get("success", False):
                return {"success": False, "error": result.get("error", "Failed to submit form")}
                
            # Process and enhance the result
            result["query"] = query
            result["form_purpose"] = form_info.get("purpose")
            
            return result
            
        except Exception as e:
            logger.error(f"Error performing search: {str(e)}")
            return {"success": False, "error": str(e)}