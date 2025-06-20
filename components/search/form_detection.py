"""
Form detection module for SmartScrape search automation.

This module provides functionality for detecting search forms on web pages
using various techniques, including standard forms, JavaScript components,
and custom search interfaces.
"""

import logging
import re
from bs4 import BeautifulSoup
from typing import Dict, List, Any, Optional
from urllib.parse import urljoin, urlparse

# Import enhanced utils
from utils.html_utils import parse_html, extract_text_fast, find_by_xpath, select_with_css
from utils.retry_utils import with_exponential_backoff

class SearchFormDetector:
    """
    Detects search forms on web pages using multiple detection strategies.
    
    This class can identify:
    - Standard HTML forms with search-related attributes
    - JavaScript-powered search components (React, Angular, etc.)
    - Custom search interfaces without traditional forms
    """
    
    def __init__(self, domain_intelligence=None):
        """
        Initialize the search form detector.
        
        Args:
            domain_intelligence: Optional domain intelligence component for
                                specialized form detection based on site type
        """
        self.logger = logging.getLogger("SearchFormDetector")
        
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

    async def detect_and_use_search_form(self, page, search_term):
        """
        Detect and use a search form on the current page.
        
        Args:
            page: Playwright page object
            search_term: Search term to use
            
        Returns:
            Dict with search form result information
        """
        self.logger.info(f"Attempting to detect and use search form with term: {search_term}")
        
        try:
            # Get the HTML content
            html = await page.content()
            
            # Detect search forms
            forms = await self.detect_search_forms(html)
            
            if not forms:
                return {"success": False, "reason": "No search forms detected"}
                
            # Use the highest scoring form
            best_form = forms[0]
            
            # Determine search field
            search_field = None
            for field in best_form["fields"]:
                if field["is_search_field"] and field["type"] in ["text", "search", ""]:
                    search_field = field
                    break
                    
            if not search_field:
                return {"success": False, "reason": "No suitable search field found in form"}
                
            # Fill the search field
            field_selector = None
            # Try to find the field in the form by type
            if best_form["type"] == "standard_form":
                form_selector = f"form#{best_form['id']}" if best_form["id"] else "form"
                field_selector = f"{form_selector} input[type='text'], {form_selector} input[type='search']"
            else:
                # For non-standard forms, try a more general approach
                container_selector = f"#{best_form['id']}" if best_form["id"] else \
                                    (".{0}".format(best_form['classes'].replace(' ', '.')) if best_form.get("classes") else "body")
                field_selector = f"{container_selector} input[type='text'], {container_selector} input[type='search']"
                    
            if not field_selector:
                return {"success": False, "reason": "Could not create selector for search field"}
                
            # Clear the field first
            await page.fill(field_selector, "")
            
            # Fill in the search term
            await page.fill(field_selector, search_term)
            self.logger.info(f"Filled search field with term: {search_term}")
            
            # Find and click submit button
            submit_clicked = await self._click_submit_button(page, best_form)
            
            if not submit_clicked:
                # If no submit button found, try pressing Enter
                await page.press(field_selector, "Enter")
                self.logger.info("No submit button found, pressed Enter instead")
                
            # Wait for navigation or content change
            try:
                await page.wait_for_load_state("networkidle", timeout=5000)
            except Exception as e:
                self.logger.info(f"Navigation timeout after form submission: {str(e)}")
                
            return {
                "success": True,
                "form_type": best_form["type"],
                "form_id": best_form["id"],
                "method": "enter_key" if not submit_clicked else "submit_button"
            }
                
        except Exception as e:
            self.logger.error(f"Error detecting and using search form: {str(e)}")
            return {"success": False, "reason": str(e)}
            
    async def _click_submit_button(self, page, form_info):
        """
        Attempt to click the submit button of a form.
        
        Args:
            page: Playwright page object
            form_info: Form information
            
        Returns:
            Boolean indicating whether a submit button was clicked
        """
        try:
            # Different strategies based on form type
            if form_info["type"] == "standard_form":
                # For standard forms, look for submit button in the form
                form_selector = f"form#{form_info['id']}" if form_info["id"] else "form"
                button_selectors = [
                    f"{form_selector} button[type='submit']",
                    f"{form_selector} input[type='submit']",
                    f"{form_selector} button:has-text('Search')",
                    f"{form_selector} button:has-text('Find')",
                    f"{form_selector} button"
                ]
                
                for selector in button_selectors:
                    if await page.locator(selector).count() > 0:
                        await page.click(selector)
                        return True
                        
            elif form_info["type"] == "js_component":
                # For JS components, look for buttons inside the component
                container_selector = f"#{form_info['id']}" if form_info["id"] else \
                                    (".{0}".format(form_info['classes'].replace(' ', '.')) if form_info.get("classes") else "body")
                button_selectors = [
                    f"{container_selector} button[type='submit']",
                    f"{container_selector} input[type='submit']",
                    f"{container_selector} [role='button']",
                    f"{container_selector} button",
                    f"{container_selector} [class*='search-button']",
                    f"{container_selector} [class*='submit']"
                ]
                
                for selector in button_selectors:
                    if await page.locator(selector).count() > 0:
                        await page.click(selector)
                        return True
                        
            elif form_info["type"] == "custom_interface":
                # For custom interfaces, try finding a nearby button
                field_selector = f"#{form_info['fields'][0]['id']}" if form_info["fields"][0].get("id") else \
                               (f"[name='{form_info['fields'][0]['name']}']" if form_info["fields"][0].get("name") else "input[type='text'], input[type='search']")
                                
                # Look for buttons near this field
                button_selectors = [
                    f"{field_selector} + button",
                    f"{field_selector} ~ button",
                    f"{field_selector} + input[type='submit']",
                    f"{field_selector} ~ input[type='submit']",
                    f"{field_selector} + [role='button']",
                    f"{field_selector} ~ [role='button']"
                ]
                
                for selector in button_selectors:
                    if await page.locator(selector).count() > 0:
                        await page.click(selector)
                        return True
                        
            return False
            
        except Exception as e:
            self.logger.warning(f"Error clicking submit button: {str(e)}")
            return False