"""
AI Schema Generator using Pydantic for Dynamic Data Validation

This component generates Pydantic models dynamically based on content analysis,
user intent, or sample data. It provides schema validation for extracted data
and helps ensure consistency in the final unified output.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Type, Union, Tuple, get_type_hints
from dataclasses import dataclass
from enum import Enum
import json

try:
    from pydantic import BaseModel, create_model, Field, ValidationError
    from pydantic.fields import FieldInfo
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

from config import (
    AI_SCHEMA_GENERATION_ENABLED, PYDANTIC_VALIDATION_ENABLED,
    SCHEMA_VALIDATION_STRICT, ADAPTIVE_SCHEMA_REFINEMENT
)


class FieldType(Enum):
    """Enumeration of supported field types for schema generation"""
    STRING = "str"
    INTEGER = "int"
    FLOAT = "float"
    BOOLEAN = "bool"
    LIST = "List[str]"
    DICT = "Dict[str, Any]"
    OPTIONAL_STRING = "Optional[str]"
    OPTIONAL_INTEGER = "Optional[int]"
    OPTIONAL_FLOAT = "Optional[float]"
    OPTIONAL_BOOLEAN = "Optional[bool]"
    URL = "str"  # Special string type for URLs
    EMAIL = "str"  # Special string type for emails
    PHONE = "str"  # Special string type for phone numbers
    DATE = "str"  # Date string type
    CURRENCY = "float"  # Currency/price field


@dataclass
class SchemaField:
    """Represents a field in a dynamically generated schema"""
    name: str
    field_type: FieldType
    description: str
    required: bool = True
    default: Any = None
    constraints: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.constraints is None:
            self.constraints = {}


@dataclass
class SchemaGenerationResult:
    """Result of schema generation process"""
    schema_model: Optional[Type[BaseModel]]
    fields: List[SchemaField]
    confidence: float
    generation_method: str
    validation_errors: List[str]
    metadata: Dict[str, Any]


class AISchemaGenerator:
    """
    Generates Pydantic models dynamically based on content analysis or user intent.
    
    This component analyzes user queries, intent analysis results, or sample data
    to automatically generate appropriate Pydantic schemas for data validation
    and structuring.
    """
    
    def __init__(self, intent_analyzer=None):
        """
        Initialize the AI Schema Generator
        
        Args:
            intent_analyzer: UniversalIntentAnalyzer instance for semantic analysis
        """
        if not PYDANTIC_AVAILABLE:
            raise ImportError("Pydantic is not available. Please install with: pip install pydantic>=2.0.0")
        
        if not AI_SCHEMA_GENERATION_ENABLED:
            raise ValueError("AI Schema Generation is disabled in configuration")
        
        self.intent_analyzer = intent_analyzer
        self.logger = logging.getLogger(__name__)
        
        # Intent-based field mappings
        self.intent_field_mappings = self._init_intent_field_mappings()
        
        # Common field patterns for recognition
        self.field_patterns = self._init_field_patterns()
        
        # Type inference rules
        self.type_inference_rules = self._init_type_inference_rules()
        
        # Generated schemas cache
        self.schema_cache = {}
        
        self.logger.info("AISchemaGenerator initialized successfully")
    
    def _init_intent_field_mappings(self) -> Dict[str, List[str]]:
        """Initialize mappings of intent types to common field names."""
        return {
            "news": ["title", "summary", "author", "publication_date", "full_content", "image_urls", "category"],
            "product": ["name", "price", "description", "rating", "reviews_count", "availability", "image_urls"],
            "research": ["title", "authors", "abstract", "publication_date", "journal", "doi", "keywords"],
            "event": ["name", "description", "start_date", "end_date", "location", "organizer", "price"],
            "profile": ["name", "title", "biography", "company", "location", "email", "social_links"],
            "job": ["title", "company", "description", "location", "salary_range", "employment_type", "requirements"],
            "general": ["title", "summary", "url", "full_content", "author", "publication_date", "image_urls", "tags"]
        }
    
    def _init_field_patterns(self) -> Dict[str, str]:
        """Initialize patterns for recognizing field types from content."""
        return {
            "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "url": r"https?://[^\s]+",
            "phone": r"[\+]?[1-9]?[\d\s\-\(\)]{10,}",
            "date": r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}",
            "price": r"\$[\d,]+\.?\d*|[\d,]+\.?\d*\s*(USD|EUR|GBP)",
            "rating": r"[0-5](\.\d+)?\s*(stars?|/5)"
        }
    
    def _init_type_inference_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize rules for inferring field types from values."""
        return {
            "string_indicators": ["title", "name", "description", "summary", "author", "content"],
            "number_indicators": ["price", "cost", "rating", "score", "count", "quantity"],
            "boolean_indicators": ["available", "published", "active", "verified", "featured"],
            "array_indicators": ["tags", "categories", "images", "links", "authors", "keywords"],
            "date_indicators": ["date", "time", "published", "created", "updated", "modified"]
        }

    async def generate_schema(self, sample_data: Dict[str, Any], schema_name: str, 
                            strict_mode: bool = False) -> Dict[str, Any]:
        """
        Universal schema generation method - primary entry point for dynamic schema creation.
        
        This method analyzes the provided sample data/query and generates an appropriate
        JSON schema for the expected output. It serves as a fallback when no schema
        is provided by the user.
        
        Args:
            sample_data: Dictionary containing query, expected_content, or actual sample data
            schema_name: Name for the generated schema
            strict_mode: Whether to use strict validation or allow flexibility
            
        Returns:
            Dict[str, Any]: JSON schema compatible with jsonschema library
        """
        try:
            self.logger.info(f"Generating universal schema '{schema_name}' from sample data")
            
            # Check if this is a query-based generation or sample-based
            if "query" in sample_data and "expected_content" in sample_data:
                # Query-based schema generation
                return await self._generate_schema_from_query(
                    query=sample_data["query"],
                    expected_content=sample_data["expected_content"],
                    schema_name=schema_name,
                    strict_mode=strict_mode
                )
            else:
                # Sample data-based schema generation
                return await self._generate_schema_from_sample_data(
                    sample_data=sample_data,
                    schema_name=schema_name,
                    strict_mode=strict_mode
                )
                
        except Exception as e:
            self.logger.error(f"Schema generation failed: {e}")
            # Return a fallback universal schema
            return self._get_fallback_universal_schema(schema_name)
    
    async def _generate_schema_from_query(self, query: str, expected_content: str,
                                        schema_name: str, strict_mode: bool) -> Dict[str, Any]:
        """Generate schema based on user query and expected content type."""
        
        # Analyze query to determine content type and intent
        content_type = self._analyze_query_content_type(query)
        
        # Get template based on content type
        if content_type in self.schema_templates:
            base_schema = self.schema_templates[content_type].copy()
            self.logger.info(f"Using template for content type: {content_type}")
        else:
            # Use AI to generate custom schema
            base_schema = await self._ai_generate_custom_schema(query, expected_content)
            self.logger.info(f"Generated custom schema using AI for query: {query}")
        
        # Customize schema name and adjust for strict mode
        base_schema["title"] = schema_name
        
        if not strict_mode:
            # Make all fields optional in flexible mode
            base_schema = self._make_schema_flexible(base_schema)
        
        return base_schema
    
    async def _generate_schema_from_sample_data(self, sample_data: Dict[str, Any],
                                              schema_name: str, strict_mode: bool) -> Dict[str, Any]:
        """Generate schema by analyzing actual sample data structure."""
        
        schema = {
            "title": schema_name,
            "type": "object",
            "properties": {},
            "required": [] if not strict_mode else []
        }
        
        for key, value in sample_data.items():
            field_type, additional_constraints = self._infer_field_type(value)
            
            schema["properties"][key] = {
                "type": field_type,
                **additional_constraints
            }
            
            if strict_mode:
                schema["required"].append(key)
        
        return schema
    
    def _analyze_query_content_type(self, query: str) -> str:
        """Analyze query to determine the type of content being requested."""
        
        query_lower = query.lower()
        
        # News content indicators
        if any(word in query_lower for word in ["news", "article", "breaking", "report", "latest", "update"]):
            return "news_article"
        
        # Product/shopping indicators  
        if any(word in query_lower for word in ["price", "buy", "product", "review", "rating", "shop", "store"]):
            return "product"
        
        # Research/academic indicators
        if any(word in query_lower for word in ["research", "study", "paper", "academic", "journal", "publication"]):
            return "research_paper"
        
        # Event indicators
        if any(word in query_lower for word in ["event", "conference", "meeting", "schedule", "calendar"]):
            return "event"
        
        # Person/profile indicators
        if any(word in query_lower for word in ["profile", "biography", "person", "contact", "about"]):
            return "person_profile"
        
        # Job posting indicators
        if any(word in query_lower for word in ["job", "career", "hiring", "position", "employment"]):
            return "job_posting"
        
        # Default to general content
        return "general_content"
    
    async def _ai_generate_custom_schema(self, query: str, expected_content: str) -> Dict[str, Any]:
        """Use AI to generate a custom schema when no template matches."""
        
        if self.intent_analyzer and hasattr(self.intent_analyzer, 'ai_service'):
            try:
                prompt = f"""
                Generate a JSON schema for scraping web content based on this query: "{query}"
                Expected content type: {expected_content}
                
                The schema should be in JSON Schema format and include fields commonly found 
                for this type of content. Include fields like title, summary, url, and other 
                relevant fields.
                
                Return only the JSON schema object, no additional text.
                """
                
                response = await self.intent_analyzer.ai_service.generate_response(
                    prompt=prompt,
                    response_format="json"
                )
                
                if response and "content" in response:
                    import json
                    return json.loads(response["content"])
                    
            except Exception as e:
                self.logger.warning(f"AI schema generation failed: {e}")
        
        # Fallback to universal content schema
        return self.schema_templates["general_content"]
    
    def _make_schema_flexible(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Make a schema more flexible by making fields optional and allowing additional properties."""
        
        # Make all required fields optional
        if "required" in schema:
            schema["required"] = []
        
        # Allow additional properties
        schema["additionalProperties"] = True
        
        # Make nested objects flexible too
        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                if isinstance(prop_schema, dict) and prop_schema.get("type") == "object":
                    prop_schema["additionalProperties"] = True
        
        return schema
    
    def _infer_field_type(self, value: Any) -> Tuple[str, Dict[str, Any]]:
        """Infer JSON schema type and constraints from a sample value."""
        
        additional_constraints = {}
        
        if isinstance(value, str):
            # Check for specific string patterns
            if value.startswith(("http://", "https://")):
                additional_constraints["format"] = "uri"
            elif "@" in value and "." in value:
                additional_constraints["format"] = "email"
            elif value.replace("-", "").replace(" ", "").isdigit():
                additional_constraints["pattern"] = r"[\d\s\-\(\)]+"
            
            return "string", additional_constraints
            
        elif isinstance(value, int):
            return "integer", additional_constraints
            
        elif isinstance(value, float):
            return "number", additional_constraints
            
        elif isinstance(value, bool):
            return "boolean", additional_constraints
            
        elif isinstance(value, list):
            if value:
                # Infer array item type from first element
                item_type, _ = self._infer_field_type(value[0])
                additional_constraints["items"] = {"type": item_type}
            return "array", additional_constraints
            
        elif isinstance(value, dict):
            return "object", {"additionalProperties": True}
            
        else:
            return "string", additional_constraints
    
    def _get_fallback_universal_schema(self, schema_name: str) -> Dict[str, Any]:
        """Return a fallback universal schema that works for most content types."""
        
        return {
            "title": schema_name,
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Title or headline of the content"
                },
                "summary": {
                    "type": "string",
                    "description": "Summary or description of the content",
                    "maxLength": 500
                },
                "url": {
                    "type": "string",
                    "format": "uri",
                    "description": "URL of the source page"
                },
                "full_content": {
                    "type": "string",
                    "description": "Main content or article body"
                },
                "author": {
                    "type": "string",
                    "description": "Author or creator of the content"
                },
                "publication_date": {
                    "type": "string",
                    "description": "Date of publication or creation"
                },
                "image_urls": {
                    "type": "array",
                    "items": {"type": "string", "format": "uri"},
                    "description": "URLs of images in the content"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags or categories associated with content"
                }
            },
            "required": ["title"],
            "additionalProperties": True
        }
    
    @property
    def schema_templates(self) -> Dict[str, Dict[str, Any]]:
        """Pre-built schema templates for common content types."""
        
        return {
            "news_article": {
                "title": "NewsArticleSchema",
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Article headline"},
                    "summary": {"type": "string", "maxLength": 500, "description": "Article summary"},
                    "url": {"type": "string", "format": "uri"},
                    "author": {"type": "string", "description": "Article author"},
                    "publication_date": {"type": "string", "description": "Publication date"},
                    "full_content": {"type": "string", "description": "Full article text"},
                    "image_urls": {
                        "type": "array",
                        "items": {"type": "string", "format": "uri"},
                        "description": "Article images"
                    },
                    "category": {"type": "string", "description": "News category"},
                    "tags": {
                        "type": "array", 
                        "items": {"type": "string"},
                        "description": "Article tags"
                    }
                },
                "required": ["title", "summary"],
                "additionalProperties": True
            },
            
            "product": {
                "title": "ProductSchema", 
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Product name"},
                    "price": {"type": "number", "description": "Product price"},
                    "description": {"type": "string", "description": "Product description"},
                    "url": {"type": "string", "format": "uri"},
                    "image_urls": {
                        "type": "array",
                        "items": {"type": "string", "format": "uri"},
                        "description": "Product images"
                    },
                    "rating": {"type": "number", "minimum": 0, "maximum": 5},
                    "reviews_count": {"type": "integer", "minimum": 0},
                    "availability": {"type": "boolean"},
                    "brand": {"type": "string"},
                    "category": {"type": "string"},
                    "features": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["name", "price"],
                "additionalProperties": True
            },
            
            "research_paper": {
                "title": "ResearchPaperSchema",
                "type": "object", 
                "properties": {
                    "title": {"type": "string", "description": "Paper title"},
                    "authors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Paper authors"
                    },
                    "abstract": {"type": "string", "description": "Paper abstract"},
                    "url": {"type": "string", "format": "uri"},
                    "publication_date": {"type": "string"},
                    "journal": {"type": "string", "description": "Journal or conference name"},
                    "doi": {"type": "string", "description": "Digital Object Identifier"},
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "citations_count": {"type": "integer", "minimum": 0}
                },
                "required": ["title", "abstract"],
                "additionalProperties": True
            },
            
            "event": {
                "title": "EventSchema",
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Event name"},
                    "description": {"type": "string", "description": "Event description"},
                    "url": {"type": "string", "format": "uri"},
                    "start_date": {"type": "string", "description": "Event start date"},
                    "end_date": {"type": "string", "description": "Event end date"},
                    "location": {"type": "string", "description": "Event location"},
                    "organizer": {"type": "string", "description": "Event organizer"},
                    "price": {"type": "number", "description": "Ticket price"},
                    "category": {"type": "string", "description": "Event category"},
                    "image_urls": {
                        "type": "array",
                        "items": {"type": "string", "format": "uri"}
                    }
                },
                "required": ["name", "start_date"],
                "additionalProperties": True
            },
            
            "person_profile": {
                "title": "PersonProfileSchema",
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Person's name"},
                    "title": {"type": "string", "description": "Job title or position"},
                    "biography": {"type": "string", "description": "Biography or bio"},
                    "url": {"type": "string", "format": "uri"},
                    "company": {"type": "string", "description": "Company or organization"},
                    "location": {"type": "string", "description": "Location"},
                    "email": {"type": "string", "format": "email"},
                    "phone": {"type": "string"},
                    "social_links": {
                        "type": "array",
                        "items": {"type": "string", "format": "uri"}
                    },
                    "skills": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "image_url": {"type": "string", "format": "uri"}
                },
                "required": ["name"],
                "additionalProperties": True
            },
            
            "job_posting": {
                "title": "JobPostingSchema",
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Job title"},
                    "company": {"type": "string", "description": "Company name"},
                    "description": {"type": "string", "description": "Job description"},
                    "url": {"type": "string", "format": "uri"},
                    "location": {"type": "string", "description": "Job location"},
                    "salary_range": {"type": "string", "description": "Salary range"},
                    "employment_type": {"type": "string", "description": "Full-time, part-time, etc."},
                    "posted_date": {"type": "string", "description": "Date posted"},
                    "requirements": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Job requirements"
                    },
                    "benefits": {
                        "type": "array", 
                        "items": {"type": "string"},
                        "description": "Job benefits"
                    }
                },
                "required": ["title", "company"],
                "additionalProperties": True
            },
            
            "general_content": {
                "title": "GeneralContentSchema",
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Content title"},
                    "summary": {"type": "string", "maxLength": 500, "description": "Content summary"},
                    "url": {"type": "string", "format": "uri"},
                    "full_content": {"type": "string", "description": "Main content"},
                    "author": {"type": "string", "description": "Content author"},
                    "publication_date": {"type": "string", "description": "Publication date"},
                    "image_urls": {
                        "type": "array",
                        "items": {"type": "string", "format": "uri"}
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["title"],
                "additionalProperties": True
            }
        }
