"""
Search result extraction templates for SmartScrape.

This module defines extraction templates for various search result types.
Templates are used to consistently extract structured data from search results
across different websites, allowing for standardized data processing.
"""

import logging
import re
import json
from typing import Dict, List, Any, Optional, Union, Callable
from bs4 import BeautifulSoup, Tag
from urllib.parse import urljoin

from utils.extraction_utils import extract_content_by_rules
from utils.html_utils import parse_html, extract_text_fast
from extraction.content_normalization import normalize_value, standardize_date


class SearchResultTemplate:
    """
    Base class for search result extraction templates.
    
    Templates define:
    - Selectors for identifying result containers
    - Field mapping rules for extracting data
    - Normalization rules for standardizing extracted data
    - Validation rules for ensuring data quality
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize a search result template.
        
        Args:
            name: Template name
            description: Template description
        """
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"Template.{name}")
        
        # Basic template configuration
        self.result_container_selectors = []
        self.pagination_selectors = []
        self.field_mappings = {}
        self.required_fields = []
        self.normalization_rules = {}
        self.validation_rules = {}
        
    def add_result_selector(self, selector: str, weight: int = 1, is_fallback: bool = False):
        """
        Add a CSS selector for identifying result containers.
        
        Args:
            selector: CSS selector for result containers
            weight: Confidence weight for this selector (higher means more confident)
            is_fallback: Whether this is a fallback selector used only if primary selectors fail
        """
        self.result_container_selectors.append({
            "selector": selector,
            "weight": weight,
            "is_fallback": is_fallback
        })
        return self
        
    def add_pagination_selector(self, selector: str, url_pattern: str = None, 
                               next_text: str = None, is_ajax: bool = False):
        """
        Add selectors for pagination controls.
        
        Args:
            selector: CSS selector for pagination links
            url_pattern: Optional regex pattern to validate pagination URLs
            next_text: Optional text pattern to identify "next page" links
            is_ajax: Whether pagination is handled via AJAX
        """
        self.pagination_selectors.append({
            "selector": selector,
            "url_pattern": url_pattern,
            "next_text": next_text,
            "is_ajax": is_ajax
        })
        return self
        
    def add_field_mapping(self, field_name: str, selectors: Union[str, List[str]], 
                         attr: str = None, required: bool = False, 
                         is_url: bool = False, base_url: str = None,
                         preprocessor: Callable = None):
        """
        Add a field mapping for extracting data from result containers.
        
        Args:
            field_name: Name of the field in the output data
            selectors: CSS selector(s) to extract this field
            attr: HTML attribute to extract (None for element text)
            required: Whether this field is required for a valid result
            is_url: Whether this field is a URL that needs to be resolved
            base_url: Base URL for resolving relative URLs
            preprocessor: Optional function to preprocess the field value
        """
        if isinstance(selectors, str):
            selectors = [selectors]
            
        self.field_mappings[field_name] = {
            "selectors": selectors,
            "attr": attr,
            "is_url": is_url,
            "base_url": base_url,
            "preprocessor": preprocessor
        }
        
        if required:
            self.required_fields.append(field_name)
            
        return self
        
    def add_normalization_rule(self, field_name: str, normalization_type: str, 
                              params: Dict[str, Any] = None):
        """
        Add a normalization rule for a field.
        
        Args:
            field_name: Field to normalize
            normalization_type: Type of normalization (e.g., 'date', 'price', 'text')
            params: Optional parameters for normalization
        """
        self.normalization_rules[field_name] = {
            "type": normalization_type,
            "params": params or {}
        }
        return self
        
    def add_validation_rule(self, field_name: str, rule_type: str, 
                           params: Dict[str, Any] = None):
        """
        Add a validation rule for a field.
        
        Args:
            field_name: Field to validate
            rule_type: Type of validation (e.g., 'required', 'pattern', 'range')
            params: Optional parameters for validation
        """
        if field_name not in self.validation_rules:
            self.validation_rules[field_name] = []
            
        self.validation_rules[field_name].append({
            "type": rule_type,
            "params": params or {}
        })
        return self
        
    def extract_results(self, html_content: str, base_url: str = None) -> List[Dict[str, Any]]:
        """
        Extract search results using this template.
        
        Args:
            html_content: HTML content of the search results page
            base_url: Base URL for resolving relative URLs
            
        Returns:
            List of extracted result dictionaries
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        results = []
        
        # Try each result container selector in order
        primary_selectors = [s for s in self.result_container_selectors if not s.get("is_fallback")]
        fallback_selectors = [s for s in self.result_container_selectors if s.get("is_fallback")]
        
        # Start with primary selectors
        result_elements = []
        for selector_data in primary_selectors:
            selector = selector_data["selector"]
            elements = soup.select(selector)
            if elements:
                result_elements = elements
                self.logger.info(f"Found {len(elements)} results with selector: {selector}")
                break
        
        # If no results found with primary selectors, try fallbacks
        if not result_elements and fallback_selectors:
            for selector_data in fallback_selectors:
                selector = selector_data["selector"]
                elements = soup.select(selector)
                if elements:
                    result_elements = elements
                    self.logger.info(f"Used fallback selector: {selector} ({len(elements)} results)")
                    break
        
        if not result_elements:
            self.logger.warning("No result containers found on page")
            return []
            
        # Extract data from each result element
        for element in result_elements:
            result_data = self._extract_result_fields(element, base_url)
            
            # Skip results missing required fields
            if not all(field in result_data for field in self.required_fields):
                missing = [f for f in self.required_fields if f not in result_data]
                self.logger.debug(f"Skipping result missing required fields: {missing}")
                continue
                
            # Apply normalization
            for field, rule in self.normalization_rules.items():
                if field in result_data:
                    result_data[field] = self._normalize_field(
                        result_data[field], 
                        rule["type"], 
                        rule["params"]
                    )
            
            # Apply validation
            valid = True
            for field, rules in self.validation_rules.items():
                if field not in result_data:
                    continue
                    
                for rule in rules:
                    if not self._validate_field(result_data[field], rule["type"], rule["params"]):
                        valid = False
                        self.logger.debug(f"Validation failed for field {field}")
                        break
                        
                if not valid:
                    break
                    
            if valid:
                results.append(result_data)
                
        self.logger.info(f"Extracted {len(results)} valid results")
        return results
        
    def extract_pagination_links(self, html_content: str, current_url: str) -> List[Dict[str, Any]]:
        """
        Extract pagination links from search results.
        
        Args:
            html_content: HTML content of the search results page
            current_url: Current page URL
            
        Returns:
            List of pagination links
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        pagination_links = []
        
        for selector_info in self.pagination_selectors:
            selector = selector_info["selector"]
            elements = soup.select(selector)
            
            if not elements:
                continue
                
            for element in elements:
                link_data = {
                    "is_ajax": selector_info.get("is_ajax", False)
                }
                
                # Extract URL from the element
                if element.name == 'a' and element.has_attr('href'):
                    url = element['href']
                    # Resolve relative URLs
                    if url and not url.startswith(('http://', 'https://')):
                        url = urljoin(current_url, url)
                    link_data["url"] = url
                elif selector_info.get("is_ajax") and element.has_attr('data-page'):
                    link_data["page"] = element['data-page']
                else:
                    continue
                
                # Check if this is a "next" link
                next_text = selector_info.get("next_text")
                if next_text:
                    element_text = element.get_text().strip().lower()
                    if next_text.lower() not in element_text:
                        continue
                
                # Validate URL pattern if specified
                url_pattern = selector_info.get("url_pattern")
                if url_pattern and "url" in link_data:
                    if not re.search(url_pattern, link_data["url"]):
                        continue
                
                pagination_links.append(link_data)
        
        return pagination_links
    
    def _extract_result_fields(self, element: Tag, base_url: str = None) -> Dict[str, Any]:
        """
        Extract fields from a single result element.
        
        Args:
            element: BeautifulSoup Tag representing a result container
            base_url: Base URL for resolving relative URLs
            
        Returns:
            Dictionary of extracted field values
        """
        result = {}
        
        for field_name, mapping in self.field_mappings.items():
            value = None
            
            # Try each selector in order
            for selector in mapping["selectors"]:
                found_elements = element.select(selector)
                if not found_elements:
                    continue
                    
                target_element = found_elements[0]
                
                # Extract value from element
                if mapping["attr"]:
                    if target_element.has_attr(mapping["attr"]):
                        value = target_element[mapping["attr"]]
                else:
                    value = target_element.get_text().strip()
                    
                # If value found, stop trying selectors
                if value:
                    break
            
            # Skip if no value found
            if not value:
                continue
                
            # Apply preprocessing if defined
            preprocessor = mapping.get("preprocessor")
            if preprocessor and callable(preprocessor):
                value = preprocessor(value)
                
            # Handle URL resolution
            if mapping.get("is_url") and value:
                field_base_url = mapping.get("base_url") or base_url
                if field_base_url and not value.startswith(('http://', 'https://')):
                    value = urljoin(field_base_url, value)
            
            result[field_name] = value
            
        return result
    
    def _normalize_field(self, value: Any, norm_type: str, params: Dict[str, Any]) -> Any:
        """
        Normalize a field value.
        
        Args:
            value: Value to normalize
            norm_type: Type of normalization
            params: Normalization parameters
            
        Returns:
            Normalized value
        """
        if norm_type == "date":
            date_format = params.get("format")
            return standardize_date(value, date_format)
            
        elif norm_type == "price":
            currency = params.get("currency")
            decimal_point = params.get("decimal_point", ".")
            thousands_sep = params.get("thousands_sep", ",")
            
            # Extract numeric part of price
            if isinstance(value, str):
                # Remove currency symbols and non-numeric chars except decimal point and thousands separator
                clean_value = re.sub(r'[^\d' + re.escape(decimal_point) + re.escape(thousands_sep) + r']', '', value)
                
                # Handle thousands separator
                if thousands_sep:
                    clean_value = clean_value.replace(thousands_sep, '')
                    
                # Handle decimal point
                if decimal_point and decimal_point != '.':
                    clean_value = clean_value.replace(decimal_point, '.')
                    
                try:
                    value = float(clean_value)
                except ValueError:
                    return None
            
            return value
            
        elif norm_type == "text":
            strip_html = params.get("strip_html", True)
            lowercase = params.get("lowercase", False)
            
            if isinstance(value, str):
                if strip_html:
                    value = re.sub(r'<[^>]+>', '', value)
                    
                value = value.strip()
                
                if lowercase:
                    value = value.lower()
            
            return value
            
        elif norm_type == "enum":
            mapping = params.get("mapping", {})
            case_sensitive = params.get("case_sensitive", False)
            
            if not case_sensitive and isinstance(value, str):
                # Case-insensitive matching
                lowercase_mapping = {k.lower(): v for k, v in mapping.items()}
                return lowercase_mapping.get(value.lower(), value)
            else:
                # Case-sensitive matching
                return mapping.get(value, value)
                
        else:
            # Unknown normalization type, return value unchanged
            return value
    
    def _validate_field(self, value: Any, rule_type: str, params: Dict[str, Any]) -> bool:
        """
        Validate a field value.
        
        Args:
            value: Value to validate
            rule_type: Type of validation
            params: Validation parameters
            
        Returns:
            True if validation passes, False otherwise
        """
        if rule_type == "required":
            return value is not None and value != ""
            
        elif rule_type == "pattern":
            pattern = params.get("pattern")
            if pattern and isinstance(value, str):
                return bool(re.search(pattern, value))
            return False
            
        elif rule_type == "range":
            min_val = params.get("min")
            max_val = params.get("max")
            
            if value is None:
                return False
                
            if isinstance(value, (int, float)):
                if min_val is not None and value < min_val:
                    return False
                if max_val is not None and value > max_val:
                    return False
                return True
                
            elif isinstance(value, str):
                if min_val is not None and len(value) < min_val:
                    return False
                if max_val is not None and len(value) > max_val:
                    return False
                return True
                
            return False
            
        elif rule_type == "enum":
            allowed_values = params.get("values", [])
            case_sensitive = params.get("case_sensitive", True)
            
            if not case_sensitive and isinstance(value, str):
                value = value.lower()
                allowed_values = [v.lower() if isinstance(v, str) else v for v in allowed_values]
                
            return value in allowed_values
            
        else:
            # Unknown validation type, consider it failed
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert template to a dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "result_container_selectors": self.result_container_selectors,
            "pagination_selectors": self.pagination_selectors,
            "field_mappings": {
                field: {
                    k: v for k, v in mapping.items() 
                    if k != "preprocessor"  # Skip functions that can't be serialized
                }
                for field, mapping in self.field_mappings.items()
            },
            "required_fields": self.required_fields,
            "normalization_rules": self.normalization_rules,
            "validation_rules": self.validation_rules
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchResultTemplate':
        """Create a template from a dictionary."""
        template = cls(data["name"], data.get("description", ""))
        
        # Load result container selectors
        for selector_data in data.get("result_container_selectors", []):
            template.add_result_selector(
                selector_data["selector"],
                selector_data.get("weight", 1),
                selector_data.get("is_fallback", False)
            )
        
        # Load pagination selectors
        for selector_data in data.get("pagination_selectors", []):
            template.add_pagination_selector(
                selector_data["selector"],
                selector_data.get("url_pattern"),
                selector_data.get("next_text"),
                selector_data.get("is_ajax", False)
            )
        
        # Load field mappings
        for field_name, mapping in data.get("field_mappings", {}).items():
            template.add_field_mapping(
                field_name,
                mapping["selectors"],
                mapping.get("attr"),
                field_name in data.get("required_fields", []),
                mapping.get("is_url", False),
                mapping.get("base_url")
            )
        
        # Load normalization rules
        for field_name, rule in data.get("normalization_rules", {}).items():
            template.add_normalization_rule(
                field_name,
                rule["type"],
                rule.get("params", {})
            )
        
        # Load validation rules
        for field_name, rules in data.get("validation_rules", {}).items():
            for rule in rules:
                template.add_validation_rule(
                    field_name,
                    rule["type"],
                    rule.get("params", {})
                )
        
        return template


# Domain-specific template classes

class ProductResultTemplate(SearchResultTemplate):
    """Template for product search results."""
    
    def __init__(self, name: str = "product", description: str = "Product search results"):
        super().__init__(name, description)
        
        # Common result container selectors for products
        self.add_result_selector(".product", 5)
        self.add_result_selector(".product-item", 5)
        self.add_result_selector(".item.product", 4)
        self.add_result_selector("[data-role='product']", 4)
        self.add_result_selector(".search-result-item", 3)
        self.add_result_selector(".grid-item", 2, is_fallback=True)
        
        # Common product field mappings
        self.add_field_mapping("title", [
            ".product-title", 
            ".product-name", 
            "h3.title",
            "h3 a",
            ".name"
        ], required=True)
        
        self.add_field_mapping("price", [
            ".price", 
            ".product-price", 
            ".price-current",
            "[data-role='price']"
        ])
        
        self.add_field_mapping("image", [
            "img.product-image", 
            ".product-img img", 
            "img.thumb",
            "img"
        ], attr="src", is_url=True)
        
        self.add_field_mapping("url", [
            "a.product-link", 
            ".product-title a", 
            ".title a",
            "h3 a",
            "a:first-child"
        ], attr="href", is_url=True, required=True)
        
        # Add common normalization rules
        self.add_normalization_rule("price", "price", {"decimal_point": "."})
        self.add_normalization_rule("title", "text", {"strip_html": True})
        
        # Add common validation rules
        self.add_validation_rule("title", "required")
        self.add_validation_rule("url", "required")
        self.add_validation_rule("price", "range", {"min": 0})


class RealEstateResultTemplate(SearchResultTemplate):
    """Template for real estate search results."""
    
    def __init__(self, name: str = "real_estate", description: str = "Real estate search results"):
        super().__init__(name, description)
        
        # Common real estate result container selectors
        self.add_result_selector(".property-item", 5)
        self.add_result_selector(".listing", 5)
        self.add_result_selector(".real-estate-item", 4)
        self.add_result_selector("[data-role='property']", 4)
        self.add_result_selector(".home-card", 3)
        self.add_result_selector(".grid-item", 2, is_fallback=True)
        
        # Common real estate field mappings
        self.add_field_mapping("title", [
            ".property-title", 
            ".listing-title", 
            "h3.title",
            "h3 a",
            ".address"
        ], required=True)
        
        self.add_field_mapping("price", [
            ".price", 
            ".property-price", 
            ".price-display",
            "[data-role='price']"
        ])
        
        self.add_field_mapping("image", [
            "img.property-image", 
            ".property-img img", 
            "img.main-image",
            "img:first-child"
        ], attr="src", is_url=True)
        
        self.add_field_mapping("url", [
            "a.property-link", 
            ".property-title a", 
            ".title a",
            "h3 a",
            "a:first-child"
        ], attr="href", is_url=True, required=True)
        
        self.add_field_mapping("address", [
            ".address",
            ".property-address",
            ".location"
        ])
        
        self.add_field_mapping("bedrooms", [
            ".beds",
            ".bedrooms",
            "[data-label='beds']",
            ".property-features .bed"
        ])
        
        self.add_field_mapping("bathrooms", [
            ".baths",
            ".bathrooms",
            "[data-label='baths']",
            ".property-features .bath"
        ])
        
        self.add_field_mapping("area", [
            ".area",
            ".square-feet",
            ".sqft",
            "[data-label='sqft']",
            ".property-features .area"
        ])
        
        # Add common normalization rules
        self.add_normalization_rule("price", "price", {"decimal_point": "."})
        self.add_normalization_rule("title", "text", {"strip_html": True})
        self.add_normalization_rule("bedrooms", "text", {"strip_html": True})
        self.add_normalization_rule("bathrooms", "text", {"strip_html": True})
        self.add_normalization_rule("area", "text", {"strip_html": True})
        
        # Add common validation rules
        self.add_validation_rule("title", "required")
        self.add_validation_rule("url", "required")
        self.add_validation_rule("price", "range", {"min": 0})


class JobResultTemplate(SearchResultTemplate):
    """Template for job search results."""
    
    def __init__(self, name: str = "job", description: str = "Job search results"):
        super().__init__(name, description)
        
        # Common job result container selectors
        self.add_result_selector(".job-item", 5)
        self.add_result_selector(".job-card", 5)
        self.add_result_selector(".job-posting", 4)
        self.add_result_selector("[data-role='job']", 4)
        self.add_result_selector(".result-card", 3)
        self.add_result_selector(".search-result", 2, is_fallback=True)
        
        # Common job field mappings
        self.add_field_mapping("title", [
            ".job-title", 
            ".position-title", 
            "h3.title",
            "h3 a",
            ".role"
        ], required=True)
        
        self.add_field_mapping("company", [
            ".company-name", 
            ".employer", 
            ".company",
            "[data-role='company']"
        ])
        
        self.add_field_mapping("location", [
            ".location", 
            ".job-location", 
            ".region",
            "[data-role='location']"
        ])
        
        self.add_field_mapping("url", [
            "a.job-link", 
            ".job-title a", 
            ".title a",
            "h3 a",
            "a:first-child"
        ], attr="href", is_url=True, required=True)
        
        self.add_field_mapping("salary", [
            ".salary",
            ".compensation",
            ".pay-range"
        ])
        
        self.add_field_mapping("date_posted", [
            ".date-posted",
            ".posted-date",
            ".post-date",
            ".listing-date"
        ])
        
        # Add common normalization rules
        self.add_normalization_rule("title", "text", {"strip_html": True})
        self.add_normalization_rule("company", "text", {"strip_html": True})
        self.add_normalization_rule("location", "text", {"strip_html": True})
        self.add_normalization_rule("date_posted", "date")
        
        # Add common validation rules
        self.add_validation_rule("title", "required")
        self.add_validation_rule("url", "required")


class TemplateManager:
    """
    Manages search result templates for different domains.
    
    This class:
    - Stores and retrieves templates
    - Supports template serialization and deserialization
    - Provides domain-specific template selection
    - Handles custom template creation
    """
    
    def __init__(self, storage_path: str = None):
        """
        Initialize the template manager.
        
        Args:
            storage_path: Optional path to store templates
        """
        self.logger = logging.getLogger("TemplateManager")
        self.storage_path = storage_path
        self.templates = {}
        
        # Create and register default templates
        self._register_default_templates()
        
    def _register_default_templates(self):
        """Register default templates for common domains."""
        # Register product template
        product_template = ProductResultTemplate()
        self.templates[product_template.name] = product_template
        
        # Register real estate template
        real_estate_template = RealEstateResultTemplate()
        self.templates[real_estate_template.name] = real_estate_template
        
        # Register job template
        job_template = JobResultTemplate()
        self.templates[job_template.name] = job_template
        
        # Register generic template
        generic_template = SearchResultTemplate("generic", "Generic search results")
        generic_template.add_result_selector(".result", 5)
        generic_template.add_result_selector(".search-result", 5)
        generic_template.add_result_selector(".item", 3, is_fallback=True)
        generic_template.add_result_selector("article", 2, is_fallback=True)
        generic_template.add_result_selector(".col", 1, is_fallback=True)
        
        generic_template.add_field_mapping("title", [
            "h2", "h3", ".title", "a strong", "a h3", "a h2", 
            ".result-title", ".name", "a"
        ], required=True)
        
        generic_template.add_field_mapping("url", [
            "a", "h2 a", "h3 a", ".title a"
        ], attr="href", is_url=True, required=True)
        
        generic_template.add_field_mapping("description", [
            ".description", ".snippet", ".content", "p"
        ])
        
        self.templates[generic_template.name] = generic_template
    
    def get_template(self, name: str) -> Optional[SearchResultTemplate]:
        """
        Get a template by name.
        
        Args:
            name: Template name
            
        Returns:
            Template object or None if not found
        """
        return self.templates.get(name)
    
    def add_template(self, template: SearchResultTemplate):
        """
        Add a template to the manager.
        
        Args:
            template: Template to add
        """
        self.templates[template.name] = template
        self._save_templates()
    
    def create_template(self, name: str, domain_type: str = "generic", 
                       base_template: str = None) -> SearchResultTemplate:
        """
        Create a new template, optionally based on an existing one.
        
        Args:
            name: Template name
            domain_type: Domain type for specialized templates
            base_template: Optional base template name to extend
            
        Returns:
            New template instance
        """
        if base_template:
            # Clone from base template
            base = self.get_template(base_template)
            if not base:
                raise ValueError(f"Base template '{base_template}' not found")
                
            template_dict = base.to_dict()
            template_dict["name"] = name
            template = SearchResultTemplate.from_dict(template_dict)
            
        elif domain_type == "product":
            template = ProductResultTemplate(name)
        elif domain_type == "real_estate":
            template = RealEstateResultTemplate(name)
        elif domain_type == "job":
            template = JobResultTemplate(name)
        else:
            # Generic template
            template = SearchResultTemplate(name, f"{domain_type.capitalize()} search results")
            
            # Add some sensible defaults
            template.add_result_selector(".result", 5)
            template.add_result_selector(".search-result", 5)
            template.add_result_selector(".item", 3, is_fallback=True)
            
            template.add_field_mapping("title", [
                "h2", "h3", ".title", "a"
            ], required=True)
            
            template.add_field_mapping("url", [
                "a", "h2 a", "h3 a", ".title a"
            ], attr="href", is_url=True, required=True)
            
        # Add the template to the manager
        self.add_template(template)
        return template
    
    def delete_template(self, name: str) -> bool:
        """
        Delete a template.
        
        Args:
            name: Template name
            
        Returns:
            True if deleted, False if not found
        """
        if name in self.templates:
            del self.templates[name]
            self._save_templates()
            return True
        return False
    
    def select_template_for_domain(self, domain_type: str) -> SearchResultTemplate:
        """
        Select the best template for a domain type.
        
        Args:
            domain_type: Domain type (e.g., 'product', 'real_estate')
            
        Returns:
            Most appropriate template
        """
        if domain_type == "product":
            return self.get_template("product")
        elif domain_type in ["real_estate", "property", "housing"]:
            return self.get_template("real_estate")
        elif domain_type in ["job", "employment", "career"]:
            return self.get_template("job")
        else:
            return self.get_template("generic")
    
    def _save_templates(self):
        """Save templates to disk if storage path is set."""
        if not self.storage_path:
            return
            
        try:
            template_dicts = {name: template.to_dict() 
                           for name, template in self.templates.items()}
            
            with open(self.storage_path, 'w') as f:
                json.dump(template_dicts, f, indent=2)
                
            self.logger.info(f"Saved {len(template_dicts)} templates to {self.storage_path}")
        except Exception as e:
            self.logger.error(f"Error saving templates: {str(e)}")
    
    def load_templates(self) -> bool:
        """
        Load templates from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.storage_path:
            return False
            
        try:
            with open(self.storage_path, 'r') as f:
                template_dicts = json.load(f)
                
            for name, template_dict in template_dicts.items():
                self.templates[name] = SearchResultTemplate.from_dict(template_dict)
                
            self.logger.info(f"Loaded {len(template_dicts)} templates from {self.storage_path}")
            return True
        except FileNotFoundError:
            self.logger.info(f"Template file not found: {self.storage_path}")
            return False
        except Exception as e:
            self.logger.error(f"Error loading templates: {str(e)}")
            return False


# Register components in __init__.py
__all__ = [
    'SearchResultTemplate', 
    'ProductResultTemplate', 
    'RealEstateResultTemplate', 
    'JobResultTemplate',
    'TemplateManager'
]