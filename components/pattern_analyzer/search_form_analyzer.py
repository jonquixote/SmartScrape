"""
Search Form Analyzer Module

This module provides functionality to detect and analyze search forms
on websites. It identifies search input fields, buttons, and form attributes.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from urllib.parse import urlparse
from bs4 import BeautifulSoup, Tag

from components.pattern_analyzer.base_analyzer import PatternAnalyzer, get_registry

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SearchFormAnalyzer")

class SearchFormAnalyzer(PatternAnalyzer):
    """
    Analyzer for detecting and analyzing search forms on web pages.
    It can identify different types of search interfaces including:
    - Simple search boxes with buttons
    - Advanced search forms with multiple fields
    - Filter interfaces that function as search
    """
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize the search form analyzer.
        
        Args:
            confidence_threshold: Minimum confidence level for pattern detection
        """
        super().__init__(confidence_threshold)
        # Patterns that indicate a field might be search-related
        self.search_field_patterns = {
            'id': [re.compile(r'search', re.I), re.compile(r'query', re.I), re.compile(r'find', re.I), 
                  re.compile(r'filter', re.I), re.compile(r'keyword', re.I)],
            'name': [re.compile(r'search', re.I), re.compile(r'query', re.I), re.compile(r'q\b', re.I),
                    re.compile(r'keyword', re.I), re.compile(r'term', re.I)],
            'placeholder': [re.compile(r'search', re.I), re.compile(r'find', re.I), re.compile(r'enter', re.I),
                           re.compile(r'keyword', re.I), re.compile(r'what are you looking for', re.I)],
            'class': [re.compile(r'search', re.I), re.compile(r'query', re.I), re.compile(r'find', re.I),
                     re.compile(r'filter', re.I), re.compile(r'input', re.I)]
        }
        
        # Patterns that indicate a button might be a search submission button
        self.search_button_patterns = {
            'id': [re.compile(r'search', re.I), re.compile(r'submit', re.I), re.compile(r'go\b', re.I), 
                  re.compile(r'find', re.I)],
            'class': [re.compile(r'search', re.I), re.compile(r'submit', re.I), re.compile(r'button', re.I),
                     re.compile(r'btn', re.I), re.compile(r'find', re.I)],
            'value': [re.compile(r'search', re.I), re.compile(r'go\b', re.I), re.compile(r'find', re.I),
                     re.compile(r'submit', re.I)]
        }
    
    async def analyze(self, html: str, url: str) -> Dict[str, Any]:
        """
        Analyze a page to detect search forms.
        
        Args:
            html: HTML content of the page
            url: URL of the page
            
        Returns:
            Dictionary with detected search form patterns
        """
        logger.info(f"Analyzing search forms on {url}")
        soup = self.parse_html(html)
        domain = self.get_domain(url)
        
        # Look for forms first since they're most common for search
        forms = soup.find_all('form')
        logger.info(f"Found {len(forms)} forms on page")
        
        # Results will contain all detected search patterns
        results = {
            "search_forms": [],
            "search_inputs": [],
            "search_buttons": [],
            "confidence_scores": {}
        }
        
        # Analyze each form for search patterns
        for form_idx, form in enumerate(forms):
            search_form_data = self._analyze_form(form, form_idx)
            
            # If the form has search-like properties, add it to the results
            if search_form_data["confidence_score"] >= self.confidence_threshold:
                results["search_forms"].append(search_form_data)
                form_id = search_form_data["form_id"]
                results["confidence_scores"][form_id] = search_form_data["confidence_score"]
                logger.info(f"Detected search form: {form_id} with confidence {search_form_data['confidence_score']:.2f}")
        
        # If no forms were found, look for standalone search elements
        if not results["search_forms"]:
            logger.info("No clear search forms found, looking for standalone search components")
            standalone_results = self._find_standalone_search(soup)
            results["search_inputs"].extend(standalone_results["inputs"])
            results["search_buttons"].extend(standalone_results["buttons"])
            
            # If we found search components, create a virtual form
            if standalone_results["inputs"] and standalone_results["buttons"]:
                virtual_form = {
                    "form_id": "virtual_search_form",
                    "form_type": "virtual",
                    "inputs": standalone_results["inputs"],
                    "buttons": standalone_results["buttons"],
                    "action": url,  # Default action is the current page
                    "method": "GET",  # Default method
                    "confidence_score": standalone_results["confidence_score"],
                    "selector": self._generate_field_selector(standalone_results["inputs"][0]["element"])
                }
                results["search_forms"].append(virtual_form)
                results["confidence_scores"]["virtual_search_form"] = standalone_results["confidence_score"]
                logger.info(f"Created virtual search form with confidence {standalone_results['confidence_score']:.2f}")
        
        # Register the patterns in the global registry
        if results["search_forms"]:
            best_form = max(results["search_forms"], key=lambda x: x["confidence_score"])
            get_registry().register_pattern(
                pattern_type="search_form",
                url=url,
                pattern_data=best_form,
                confidence=best_form["confidence_score"]
            )
            logger.info(f"Registered search form pattern for {domain}")
        
        return results
    
    def _analyze_form(self, form: Tag, form_idx: int) -> Dict[str, Any]:
        """
        Analyze a form element to determine if it's a search form.
        
        Args:
            form: BeautifulSoup form element
            form_idx: Index of the form on the page
            
        Returns:
            Dictionary with form analysis data
        """
        form_data = {
            "form_id": form.get('id', f"form_{form_idx}"),
            "form_type": "unknown",
            "inputs": [],
            "buttons": [],
            "action": form.get('action', ''),
            "method": form.get('method', 'GET'),
            "confidence_score": 0.0
        }
        
        # Evidence collection for confidence calculation
        evidence_points = []
        
        # Check form attributes for search indicators
        form_attrs = str(form.get('class', '')) + str(form.get('id', '')) + str(form.get('name', ''))
        if re.search(r'search|find|query', form_attrs, re.I):
            evidence_points.append(0.7)
        
        # Check form action for search indicators
        action = form.get('action', '')
        if action and re.search(r'search|find|results|query', action, re.I):
            evidence_points.append(0.8)
            form_data["form_type"] = "search"
        
        # Analyze input fields
        inputs = form.find_all(['input', 'select', 'textarea'])
        text_inputs = []
        
        for input_idx, input_field in enumerate(inputs):
            input_type = input_field.get('type', '')
            
            # Skip hidden, submit, and button inputs for search field analysis
            if input_type in ['hidden', 'submit', 'button']:
                continue
                
            input_data = self._analyze_input_field(input_field, input_idx)
            form_data["inputs"].append(input_data)
            
            # Consider text-like inputs for search confidence
            if input_type in ['text', 'search', ''] or input_field.name in ['textarea', 'select']:
                text_inputs.append(input_data)
                
                # Add to evidence if it looks like a search field
                if input_data["search_confidence"] > 0.5:
                    evidence_points.append(input_data["search_confidence"])
        
        # Find buttons that could be search buttons
        buttons = form.find_all(['button', 'input[type="submit"]', 'input[type="button"]'])
        
        for button_idx, button in enumerate(buttons):
            button_data = self._analyze_button(button, button_idx)
            form_data["buttons"].append(button_data)
            
            # Add to evidence if it looks like a search button
            if button_data["search_confidence"] > 0.5:
                evidence_points.append(button_data["search_confidence"])
        
        # Form characteristics that influence confidence
        # Simple search forms typically have few fields
        if len(form_data["inputs"]) < 3:
            evidence_points.append(0.6)
        
        # Search forms typically have a search-like input and button
        has_search_input = any(i["search_confidence"] > 0.7 for i in form_data["inputs"])
        has_search_button = any(b["search_confidence"] > 0.7 for b in form_data["buttons"])
        
        if has_search_input and has_search_button:
            evidence_points.append(0.9)
            form_data["form_type"] = "search"
        
        # Advanced search forms might have multiple fields but still be search-oriented
        if len(form_data["inputs"]) >= 3 and has_search_button:
            # Check if inputs are related to search refinements
            refinement_count = 0
            for input_data in form_data["inputs"]:
                if re.search(r'price|location|category|filter|sort|date|range', 
                             str(input_data["attributes"]), re.I):
                    refinement_count += 1
            
            if refinement_count >= 2:
                evidence_points.append(0.8)
                form_data["form_type"] = "advanced_search"
        
        # Calculate overall confidence score
        form_data["confidence_score"] = self.calculate_confidence(evidence_points)
        
        # Generate a CSS selector for the form
        form_data["selector"] = self._generate_form_selector(form)
        
        return form_data
    
    def _analyze_input_field(self, input_field: Tag, input_idx: int) -> Dict[str, Any]:
        """
        Analyze an input field to determine its role in a search form.
        
        Args:
            input_field: BeautifulSoup input element
            input_idx: Index of the input in the form
            
        Returns:
            Dictionary with input field analysis data
        """
        # Extract all relevant attributes
        attrs = {
            'id': input_field.get('id', ''),
            'name': input_field.get('name', ''),
            'type': input_field.get('type', ''),
            'class': input_field.get('class', []),
            'placeholder': input_field.get('placeholder', '')
        }
        
        # Evidence for this being a search field
        evidence_points = []
        
        # Check each attribute against search patterns
        for attr_name, patterns in self.search_field_patterns.items():
            if attr_name in attrs and attrs[attr_name]:
                attr_value = ' '.join(attrs[attr_name]) if isinstance(attrs[attr_name], list) else attrs[attr_name]
                for pattern in patterns:
                    if pattern.search(attr_value):
                        evidence_points.append(0.8)
                        break
        
        # Special check for type="search"
        if attrs['type'] == 'search':
            evidence_points.append(1.0)
        
        # If it's a prominent text input, it might be a search field
        if (attrs['type'] in ['text', ''] and 
            input_field.parent and 
            input_field.parent.name in ['div', 'span', 'form'] and
            input_field.get('placeholder')):
            evidence_points.append(0.5)
            
        # Calculate confidence score
        search_confidence = self.calculate_confidence(evidence_points)
        
        # Determine the field role based on patterns
        field_role = "unknown"
        if search_confidence > 0.7:
            field_role = "search_query"
        elif re.search(r'location|city|zip|postal|address', str(attrs), re.I):
            field_role = "location"
        elif re.search(r'category|type|filter', str(attrs), re.I):
            field_role = "category"
        elif re.search(r'price|cost|budget|min|max', str(attrs), re.I):
            field_role = "price"
        elif re.search(r'date|time|when', str(attrs), re.I):
            field_role = "date"
        
        # Create the input data
        input_data = {
            "input_id": attrs['id'] or attrs['name'] or f"input_{input_idx}",
            "element": input_field.name,
            "type": attrs['type'],
            "role": field_role,
            "attributes": attrs,
            "search_confidence": search_confidence,
            "selector": self._generate_field_selector(input_field)
        }
        
        return input_data
    
    def _analyze_button(self, button: Tag, button_idx: int) -> Dict[str, Any]:
        """
        Analyze a button to determine if it's a search submission button.
        
        Args:
            button: BeautifulSoup button element
            button_idx: Index of the button in the form
            
        Returns:
            Dictionary with button analysis data
        """
        # Extract all relevant attributes
        is_input = button.name == 'input'
        
        attrs = {
            'id': button.get('id', ''),
            'class': button.get('class', []),
            'type': button.get('type', ''),
            'value': button.get('value', '') if is_input else '',
            'name': button.get('name', '')
        }
        
        # For non-input buttons, look at text content
        if not is_input:
            button_text = button.get_text().strip()
            attrs['text'] = button_text
            
        # Evidence for this being a search button
        evidence_points = []
        
        # Check each attribute against search button patterns
        for attr_name, patterns in self.search_button_patterns.items():
            if attr_name in attrs and attrs[attr_name]:
                attr_value = ' '.join(attrs[attr_name]) if isinstance(attrs[attr_name], list) else attrs[attr_name]
                for pattern in patterns:
                    if pattern.search(attr_value):
                        evidence_points.append(0.8)
                        break
        
        # Look for search icons
        icon_classes = ['fa-search', 'icon-search', 'search-icon']
        for icon_class in icon_classes:
            icon = button.find(class_=re.compile(icon_class, re.I))
            if icon:
                evidence_points.append(0.9)
                break
        
        # Look for magnifying glass content (common for search)
        if not is_input and button.get_text().strip() in ['🔍', '🔎']:
            evidence_points.append(1.0)
        
        # If it's a button with no text and an icon, it might be a search button
        if not is_input and not button.get_text().strip() and button.find('i'):
            evidence_points.append(0.5)
            
        # Calculate confidence score
        search_confidence = self.calculate_confidence(evidence_points)
        
        # Create the button data
        button_data = {
            "button_id": attrs['id'] or attrs['name'] or f"button_{button_idx}",
            "element": button.name,
            "type": attrs['type'],
            "attributes": attrs,
            "search_confidence": search_confidence,
            "selector": self._generate_field_selector(button)
        }
        
        return button_data
    
    def _find_standalone_search(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Find standalone search components that aren't part of a form.
        
        Args:
            soup: BeautifulSoup object with parsed HTML
            
        Returns:
            Dictionary with standalone search components
        """
        results = {
            "inputs": [],
            "buttons": [],
            "confidence_score": 0.0
        }
        
        # Look for inputs that might be search inputs
        inputs = soup.find_all('input')
        text_inputs = [i for i in inputs if i.get('type', '') in ['text', 'search', '']]
        
        # Analyze each input
        for input_idx, input_field in enumerate(text_inputs):
            # Skip inputs that are inside forms
            if input_field.find_parent('form'):
                continue
                
            input_data = self._analyze_input_field(input_field, input_idx)
            
            # If it's likely to be a search input, add it
            if input_data["search_confidence"] > 0.6:
                results["inputs"].append(input_data)
        
        # If we didn't find any text inputs outside forms, check for search-like divs
        if not results["inputs"]:
            search_divs = soup.select('div[class*="search"], div[id*="search"]')
            for div_idx, div in enumerate(search_divs):
                # Skip divs that are inside forms
                if div.find_parent('form'):
                    continue
                    
                # Look for text inputs inside the div
                div_inputs = div.find_all('input')
                for input_idx, input_field in enumerate(div_inputs):
                    if input_field.get('type', '') in ['text', 'search', '']:
                        input_data = self._analyze_input_field(input_field, input_idx)
                        if input_data["search_confidence"] > 0.5:
                            results["inputs"].append(input_data)
        
        # Look for buttons near search inputs
        for input_data in results["inputs"]:
            input_elem = soup.select_one(input_data["selector"])
            if not input_elem:
                continue
                
            # Check next siblings and parent's next siblings
            siblings = list(input_elem.next_siblings)
            parent_siblings = []
            if input_elem.parent:
                parent_siblings = list(input_elem.parent.next_siblings)
                
            # Check for buttons in various locations
            potential_buttons = []
            for elem in siblings + parent_siblings:
                if not isinstance(elem, Tag):
                    continue
                    
                if elem.name == 'button':
                    potential_buttons.append(elem)
                elif elem.name == 'input' and elem.get('type', '') in ['submit', 'button']:
                    potential_buttons.append(elem)
                else:
                    # Look for buttons within this element
                    buttons = elem.find_all(['button', 'input[type="submit"]', 'input[type="button"]'])
                    potential_buttons.extend(buttons)
            
            # Analyze each potential button
            for button_idx, button in enumerate(potential_buttons):
                button_data = self._analyze_button(button, button_idx)
                if button_data["search_confidence"] > 0.6:
                    results["buttons"].append(button_data)
                    break  # We only need one button per input
        
        # Calculate overall confidence based on the components found
        evidence_points = []
        
        # If we found both inputs and buttons
        if results["inputs"] and results["buttons"]:
            evidence_points.append(0.8)
            
            # Average the confidence of the components
            input_confidences = [i["search_confidence"] for i in results["inputs"]]
            button_confidences = [b["search_confidence"] for b in results["buttons"]]
            avg_confidence = (sum(input_confidences) + sum(button_confidences)) / (len(input_confidences) + len(button_confidences))
            evidence_points.append(avg_confidence)
        
        results["confidence_score"] = self.calculate_confidence(evidence_points)
        
        return results
    
    def _generate_form_selector(self, form: Tag) -> str:
        """
        Generate a CSS selector for a form.
        
        Args:
            form: BeautifulSoup form element
            
        Returns:
            CSS selector string
        """
        # If the form has an ID, that's the most reliable selector
        if form.get('id'):
            return f"form#{form['id']}"
            
        # If the form has classes, use those
        if form.get('class'):
            class_selector = '.'.join(form['class'])
            return f"form.{class_selector}"
            
        # If the form has a name attribute
        if form.get('name'):
            return f"form[name='{form['name']}']"
            
        # If the form has an action attribute
        if form.get('action'):
            action_value = form['action']
            # Use just the last part of the action to make it more robust
            action_parts = action_value.split('/')
            if action_parts:
                return f"form[action$='{action_parts[-1]}']"
            return f"form[action='{action_value}']"
            
        # Fallback: try to generate a selector based on form position
        parent = form.parent
        if parent and parent.name != 'body':
            if parent.get('id'):
                return f"#{parent['id']} > form"
            elif parent.get('class'):
                class_selector = '.'.join(parent['class'])
                return f".{class_selector} > form"
        
        # Last resort: use nth-of-type
        forms = list(form.find_previous_siblings('form')) + [form]
        return f"form:nth-of-type({len(forms)})"
    
    def _generate_field_selector(self, field: Tag) -> str:
        """
        Generate a CSS selector for a form field.
        
        Args:
            field: BeautifulSoup form field element
            
        Returns:
            CSS selector string
        """
        # If the field has an ID, that's the most reliable selector
        if field.get('id'):
            return f"#{field['id']}"
            
        # If the field has a name, use that
        if field.get('name'):
            return f"{field.name}[name='{field['name']}']"
            
        # If the field has classes, use those
        if field.get('class'):
            class_selector = '.'.join(field['class'])
            return f"{field.name}.{class_selector}"
            
        # If the field has a placeholder attribute
        if field.get('placeholder'):
            placeholder = field['placeholder']
            # Use a shortened version of the placeholder if it's long
            if len(placeholder) > 20:
                placeholder = placeholder[:20]
            return f"{field.name}[placeholder^='{placeholder}']"
            
        # Fallback: try to generate a selector based on field position
        parent = field.parent
        if parent and parent.name != 'body' and parent.name != 'form':
            if parent.get('id'):
                return f"#{parent['id']} > {field.name}"
            elif parent.get('class'):
                class_selector = '.'.join(parent['class'])
                return f".{class_selector} > {field.name}"
        
        # Last resort: use nth-of-type
        siblings = list(field.find_previous_siblings(field.name)) + [field]
        if parent and parent.name == 'form' and parent.get('id'):
            return f"#{parent['id']} {field.name}:nth-of-type({len(siblings)})"
        return f"{field.name}:nth-of-type({len(siblings)})"