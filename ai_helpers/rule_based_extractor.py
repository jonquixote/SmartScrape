"""
Rule-based extractor for search intent understanding.

This module implements rule-based extraction of intent, entities, and preferences
from search queries without requiring AI models.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple

class RuleBasedExtractor:
    """
    Extracts structured information from search queries using rule-based approaches.
    
    This class provides lightweight alternatives to full NLP models by using
    pattern matching, regular expressions, and heuristics to extract entities
    and constraints from search queries.
    """
    
    def __init__(self):
        """Initialize the rule-based extractor."""
        self.logger = logging.getLogger("RuleBasedExtractor")
        
        # Initialize pattern dictionaries
        self._initialize_patterns()
    
    def _initialize_patterns(self):
        """Initialize pattern dictionaries for various entity types."""
        # Price patterns (supports various currencies and formats)
        self.price_patterns = [
            r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',  # $1,234.56
            r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:dollars?|USD)',  # 1234 dollars
            r'(?:price|cost|priced at|costs?)\s*(?:is\s*)?(?:around\s*)?(?:approximately\s*)?(?:\$)?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',  # price is $100
            r'(?:under|below|less than|<)\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',  # under $100
            r'(?:over|above|more than|>)\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',  # over $100
            r'(?:between|from)\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:and|to|-)\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'  # between $100 and $200
        ]
        
        # Date patterns
        self.date_patterns = [
            r'(?:after|from|since)\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # after 12/31/2023
            r'(?:before|until|by)\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # before 12/31/2023
            r'(?:in|during)\s*(\d{4})',  # in 2023
            r'(?:last|past)\s*(\d+)\s*(days?|weeks?|months?|years?)',  # last 30 days
            r'(?:next|upcoming)\s*(\d+)\s*(days?|weeks?|months?|years?)',  # next 7 days
            r'(?:this|current)\s*(week|month|year)',  # this month
            r'(?:today|tomorrow|yesterday)'  # relative dates
        ]
        
        # Location patterns
        self.location_patterns = [
            r'(?:in|at|near|around)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:,\s*[A-Z]{2})?)',  # in New York, NY
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:area|region|vicinity)',  # New York area
            r'(?:zip|zipcode|postal code)\s*:?\s*(\d{5}(?:-\d{4})?)',  # zip: 12345
            r'(\d{5}(?:-\d{4})?)\s+(?:area|zipcode|zip)',  # 12345 area
        ]
        
        # Size/dimension patterns
        self.size_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:x|by|×)\s*(\d+(?:\.\d+)?)\s*(?:x|by|×)?\s*(\d+(?:\.\d+)?)?',  # 10x20x30
            r'(?:size|dimension|dimensions?)\s*:?\s*(\d+(?:\.\d+)?)\s*(?:x|by|×)\s*(\d+(?:\.\d+)?)',  # size: 10x20
            r'(?:length|width|height|depth)\s*:?\s*(\d+(?:\.\d+)?)\s*(inches?|in|feet|ft|cm|mm|m)',  # length: 10 inches
            r'(?:small|medium|large|xl|xxl|s|m|l)',  # size abbreviations
            r'(?:king|queen|twin|full|california king)\s*(?:size)?'  # bed sizes
        ]
        
        # Color patterns
        self.color_patterns = [
            r'(?:color|colored?)\s*:?\s*([a-z]+)',  # color: red
            r'(?:in|with)\s+([a-z]+)\s+(?:color)',  # in red color
            r'\b(red|blue|green|yellow|orange|purple|pink|black|white|gray|grey|brown|tan|beige|navy|maroon|silver|gold)\b'  # direct color words
        ]
        
        # Brand patterns
        self.brand_patterns = [
            r'(?:brand|made by|from|by)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # brand: Apple
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:brand)',  # Apple brand
            # Common brand names could be added here
        ]
        
        # Material patterns
        self.material_patterns = [
            r'(?:made of|made from|in)\s+([a-z]+(?:\s+[a-z]+)?)',
            r'([a-z]+(?:\s+[a-z]+)?)\s+material',
            r'(?:material):\s*([a-z]+(?:\s+[a-z]+)?)'
        ]
        
        # Exclusion patterns
        self.exclusion_patterns = [
            r'(?:not|no|exclude|excluding|except)\s+([A-Za-z0-9\s\'\-\.]+)',
            r'(?:without)\s+([A-Za-z0-9\s\'\-\.]+)'
        ]
        
        # Sorting/ordering patterns
        self.sort_patterns = [
            r'(?:sort by|order by|sorted by)\s+([a-z]+(?:\s+[a-z]+)?)',
            r'(?:in)\s+(ascending|descending)\s+order',
            r'(?:from)\s+(highest|lowest|newest|oldest|most\s+recent|least\s+recent)\s+(?:to)\s+(highest|lowest|newest|oldest|most\s+recent|least\s+recent)',
            r'(highest|lowest|newest|oldest|most\s+recent|least\s+recent)\s+first'
        ]
    
    def extract_entities(self, query):
        """
        Extract entities from a search query
        
        Args:
            query: The search query string or dict containing query
            
        Returns:
            Dictionary of extracted entities
        """
        self.logger.info(f"Extracting entities from query: {query}")
        
        # Handle dict input by extracting the query string
        if isinstance(query, dict):
            query_string = query.get('query', '')
            if not query_string:
                # Try other common keys
                query_string = query.get('q', query.get('search', str(query)))
        else:
            query_string = str(query) if query else ''
        
        # Normalize the query
        normalized_query = query_string.lower().strip()
        
        # Initialize results dictionary
        entities = {
            "core_terms": self._extract_core_terms(normalized_query),
            "constraints": {}
        }
        
        # Extract price constraints
        price_constraints = self._extract_price_constraints(normalized_query)
        if price_constraints:
            entities["constraints"]["price"] = price_constraints
        
        # Extract date constraints
        date_constraints = self._extract_date_constraints(normalized_query)
        if date_constraints:
            entities["constraints"]["date"] = date_constraints
        
        # Extract location
        location = self._extract_location(normalized_query)
        if location:
            entities["constraints"]["location"] = location
        
        # Extract brand/manufacturer
        brand = self._extract_brand(normalized_query)
        if brand:
            entities["constraints"]["brand"] = brand
        
        # Extract size/dimensions
        size = self._extract_size(normalized_query)
        if size:
            entities["constraints"]["size"] = size
        
        # Extract color
        color = self._extract_color(normalized_query)
        if color:
            entities["constraints"]["color"] = color
        
        # Extract material
        material = self._extract_material(normalized_query)
        if material:
            entities["constraints"]["material"] = material
        
        # Extract exclusions
        exclusions = self._extract_exclusions(normalized_query)
        if exclusions:
            entities["constraints"]["exclusions"] = exclusions
        
        # Extract sorting preferences
        sorting = self._extract_sorting(normalized_query)
        if sorting:
            entities["constraints"]["sort"] = sorting
        
        self.logger.info(f"Extracted entities: {entities}")
        return entities
    
    def _extract_core_terms(self, query: str) -> List[str]:
        """Extract core search terms."""
        # Simple word splitting for now
        words = query.split()
        return [word for word in words if len(word) > 2]
    
    def _extract_price_constraints(self, query: str) -> Dict[str, Any]:
        """Extract price constraints."""
        return {}  # Simplified for now
    
    def _extract_date_constraints(self, query: str) -> Dict[str, Any]:
        """Extract date constraints."""
        return {}  # Simplified for now
    
    def _extract_location(self, query: str) -> Dict[str, Any]:
        """Extract location information."""
        return {}  # Simplified for now
    
    def _extract_brand(self, query: str) -> str:
        """Extract brand information."""
        return None  # Simplified for now
    
    def _extract_size(self, query: str) -> Dict[str, Any]:
        """Extract size information."""
        return {}  # Simplified for now
    
    def _extract_color(self, query: str) -> str:
        """Extract color information."""
        return None  # Simplified for now
    
    def _extract_material(self, query: str) -> str:
        """Extract material information."""
        return None  # Simplified for now
    
    def _extract_exclusions(self, query: str) -> List[str]:
        """Extract exclusion terms."""
        return []  # Simplified for now
    
    def _extract_sorting(self, query: str) -> Dict[str, str]:
        """Extract sorting preferences."""
        return {}  # Simplified for now
    
    def identify_domain_preferences(self, query: str) -> Dict[str, float]:
        """Identify domain preferences based on query terms."""
        return {}  # Simplified for now

def extract_intent(query: str) -> dict:
    """
    Extract search intent from a user query.
    
    This is a simplified wrapper around RuleBasedExtractor for easier access.
    
    Args:
        query: The search query string
        
    Returns:
        Dictionary containing extracted intent information
    """
    extractor = RuleBasedExtractor()
    entities = extractor.extract_entities(query)
    domain_prefs = extractor.identify_domain_preferences(query)
    
    return {
        "query": query,
        "entities": entities,
        "domain_preferences": domain_prefs,
        "core_terms": entities.get("core_terms", []),
        "constraints": entities.get("constraints", {})
    }

def extract_intent_with_rules(query: str) -> dict:
    """
    Extract search intent using rule-based methods.
    This is an alias function for extract_intent to maintain backward compatibility.
    
    Args:
        query: The search query string
        
    Returns:
        Dictionary containing extracted intent information
    """
    return extract_intent(query)
