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
    
    def generate_schema_from_intent(self, intent_analysis: Dict[str, Any], 
                                  query: str = "") -> SchemaGenerationResult:
        """
        Generate a Pydantic model based on identified data types and entities
        from intent analysis.
        
        Args:
            intent_analysis: Results from UniversalIntentAnalyzer
            query: Original user query for additional context
            
        Returns:
            SchemaGenerationResult with generated schema and metadata
        """
        try:
            self.logger.info(f"Generating schema from intent: {intent_analysis.get('intent_type', 'unknown')}")
            
            # Extract intent information
            intent_type = intent_analysis.get('intent_type', '')
            entities = intent_analysis.get('entities', [])
            confidence = intent_analysis.get('confidence', 0.0)
            
            # Generate fields based on intent type
            fields = self._generate_fields_from_intent(intent_type, entities, query)
            
            # Add entity-specific fields
            entity_fields = self._generate_fields_from_entities(entities)
            fields.extend(entity_fields)
            
            # Add query-specific fields
            query_fields = self._generate_fields_from_query(query)
            fields.extend(query_fields)
            
            # Remove duplicates and validate fields
            fields = self._deduplicate_fields(fields)
            
            # Create Pydantic model
            schema_model = self._create_pydantic_model(fields, f"Schema_{intent_type.title()}")
            
            result = SchemaGenerationResult(
                schema_model=schema_model,
                fields=fields,
                confidence=min(confidence + 0.1, 1.0),  # Boost confidence slightly
                generation_method="intent_analysis",
                validation_errors=[],
                metadata={
                    'intent_type': intent_type,
                    'entity_count': len(entities),
                    'field_count': len(fields),
                    'query': query
                }
            )
            
            self.logger.info(f"Generated schema with {len(fields)} fields (confidence: {result.confidence:.3f})")
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating schema from intent: {e}", exc_info=True)
            return SchemaGenerationResult(
                schema_model=None,
                fields=[],
                confidence=0.0,
                generation_method="intent_analysis",
                validation_errors=[str(e)],
                metadata={}
            )
    
    def generate_schema_from_sample(self, data_sample: List[Dict[str, Any]], 
                                  schema_name: str = "InferredSchema") -> SchemaGenerationResult:
        """
        Infer a Pydantic model from a sample of extracted data.
        
        Args:
            data_sample: List of dictionaries representing sample data
            schema_name: Name for the generated schema
            
        Returns:
            SchemaGenerationResult with inferred schema
        """
        try:
            self.logger.info(f"Generating schema from {len(data_sample)} sample items")
            
            if not data_sample:
                return SchemaGenerationResult(
                    schema_model=None,
                    fields=[],
                    confidence=0.0,
                    generation_method="sample_inference",
                    validation_errors=["No sample data provided"],
                    metadata={}
                )
            
            # Analyze sample data to infer fields
            field_analysis = self._analyze_sample_data(data_sample)
            
            # Generate fields from analysis
            fields = self._generate_fields_from_analysis(field_analysis)
            
            # Create Pydantic model
            schema_model = self._create_pydantic_model(fields, schema_name)
            
            # Calculate confidence based on data consistency
            confidence = self._calculate_sample_confidence(data_sample, field_analysis)
            
            result = SchemaGenerationResult(
                schema_model=schema_model,
                fields=fields,
                confidence=confidence,
                generation_method="sample_inference",
                validation_errors=[],
                metadata={
                    'sample_size': len(data_sample),
                    'field_count': len(fields),
                    'schema_name': schema_name,
                    'field_analysis': field_analysis
                }
            )
            
            self.logger.info(f"Inferred schema with {len(fields)} fields (confidence: {result.confidence:.3f})")
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating schema from sample: {e}", exc_info=True)
            return SchemaGenerationResult(
                schema_model=None,
                fields=[],
                confidence=0.0,
                generation_method="sample_inference",
                validation_errors=[str(e)],
                metadata={}
            )
    
    def validate_data_with_schema(self, data: Dict[str, Any], 
                                schema_model: Type[BaseModel]) -> Tuple[bool, List[str], Optional[BaseModel]]:
        """
        Validate a single data item against the provided Pydantic schema.
        
        Args:
            data: Data dictionary to validate
            schema_model: Pydantic model class for validation
            
        Returns:
            Tuple of (is_valid, error_messages, validated_instance)
        """
        if not PYDANTIC_VALIDATION_ENABLED:
            return True, [], None
        
        try:
            validated_instance = schema_model(**data)
            return True, [], validated_instance
            
        except ValidationError as e:
            error_messages = []
            for error in e.errors():
                field = error.get('loc', ['unknown'])[0]
                message = error.get('msg', 'Unknown error')
                error_messages.append(f"Field '{field}': {message}")
            
            if SCHEMA_VALIDATION_STRICT:
                return False, error_messages, None
            else:
                # In non-strict mode, try to create a partial instance
                try:
                    filtered_data = self._filter_valid_fields(data, schema_model)
                    partial_instance = schema_model(**filtered_data)
                    return True, error_messages, partial_instance
                except Exception:
                    return False, error_messages, None
        
        except Exception as e:
            return False, [f"Validation error: {str(e)}"], None
    
    def validate_batch_data(self, data_list: List[Dict[str, Any]], 
                          schema_model: Type[BaseModel]) -> Dict[str, Any]:
        """
        Validate a batch of data items against the schema.
        
        Args:
            data_list: List of data dictionaries to validate
            schema_model: Pydantic model class for validation
            
        Returns:
            Dictionary with validation results and statistics
        """
        results = {
            'valid_items': [],
            'invalid_items': [],
            'validation_errors': [],
            'total_items': len(data_list),
            'valid_count': 0,
            'invalid_count': 0,
            'validation_rate': 0.0
        }
        
        for i, data_item in enumerate(data_list):
            is_valid, errors, validated_instance = self.validate_data_with_schema(
                data_item, schema_model
            )
            
            if is_valid:
                results['valid_items'].append({
                    'index': i,
                    'data': data_item,
                    'validated_instance': validated_instance
                })
                results['valid_count'] += 1
            else:
                results['invalid_items'].append({
                    'index': i,
                    'data': data_item,
                    'errors': errors
                })
                results['invalid_count'] += 1
                results['validation_errors'].extend(errors)
        
        results['validation_rate'] = results['valid_count'] / max(results['total_items'], 1)
        
        self.logger.info(f"Batch validation: {results['valid_count']}/{results['total_items']} "
                        f"valid ({results['validation_rate']:.1%})")
        
        return results
    
    def refine_schema_from_failures(self, schema_result: SchemaGenerationResult,
                                  validation_failures: List[Dict[str, Any]]) -> SchemaGenerationResult:
        """
        Refine schema based on validation failures (adaptive schema refinement).
        
        Args:
            schema_result: Original schema generation result
            validation_failures: List of validation failure data
            
        Returns:
            Refined SchemaGenerationResult
        """
        if not ADAPTIVE_SCHEMA_REFINEMENT or not validation_failures:
            return schema_result
        
        try:
            self.logger.info(f"Refining schema based on {len(validation_failures)} validation failures")
            
            # Analyze validation failures
            failure_analysis = self._analyze_validation_failures(validation_failures)
            
            # Update field definitions based on failures
            refined_fields = self._refine_fields_from_failures(
                schema_result.fields, failure_analysis
            )
            
            # Create new refined model
            refined_model = self._create_pydantic_model(
                refined_fields, 
                f"{schema_result.schema_model.__name__}_Refined"
            )
            
            # Calculate new confidence
            refined_confidence = max(0.1, schema_result.confidence - 0.1)  # Slight penalty for needing refinement
            
            refined_result = SchemaGenerationResult(
                schema_model=refined_model,
                fields=refined_fields,
                confidence=refined_confidence,
                generation_method=f"{schema_result.generation_method}_refined",
                validation_errors=[],
                metadata={
                    **schema_result.metadata,
                    'refinement_count': schema_result.metadata.get('refinement_count', 0) + 1,
                    'failure_analysis': failure_analysis
                }
            )
            
            self.logger.info(f"Schema refined: {len(refined_fields)} fields, "
                           f"confidence: {refined_confidence:.3f}")
            
            return refined_result
            
        except Exception as e:
            self.logger.error(f"Error refining schema: {e}")
            return schema_result  # Return original on error
    
    def _generate_fields_from_intent(self, intent_type: str, entities: List[Dict], 
                                   query: str) -> List[SchemaField]:
        """Generate fields based on intent type"""
        fields = []
        
        # Get base fields for the intent type
        base_fields = self.intent_field_mappings.get(intent_type, [])
        fields.extend(base_fields)
        
        # Add common fields that apply to most intents
        common_fields = [
            SchemaField("url", FieldType.URL, "Source URL of the data", required=False),
            SchemaField("title", FieldType.OPTIONAL_STRING, "Title or headline"),
            SchemaField("description", FieldType.OPTIONAL_STRING, "Description or summary"),
            SchemaField("timestamp", FieldType.OPTIONAL_STRING, "Timestamp of data collection")
        ]
        fields.extend(common_fields)
        
        return fields
    
    def _generate_fields_from_entities(self, entities: List[Dict]) -> List[SchemaField]:
        """Generate fields based on detected entities"""
        fields = []
        
        for entity in entities:
            entity_label = entity.get('label', '').upper()
            entity_text = entity.get('text', '')
            
            # Map entity labels to field types
            if entity_label in ['PERSON', 'ORG', 'GPE']:
                field_name = entity_label.lower()
                fields.append(SchemaField(
                    name=field_name,
                    field_type=FieldType.OPTIONAL_STRING,
                    description=f"Detected {entity_label.lower()}: {entity_text}",
                    required=False
                ))
            
            elif entity_label == 'MONEY':
                fields.append(SchemaField(
                    name="price",
                    field_type=FieldType.OPTIONAL_FLOAT,
                    description="Price or monetary value",
                    required=False
                ))
            
            elif entity_label == 'DATE':
                fields.append(SchemaField(
                    name="date",
                    field_type=FieldType.DATE,
                    description="Date information",
                    required=False
                ))
        
        return fields
    
    def _generate_fields_from_query(self, query: str) -> List[SchemaField]:
        """Generate fields based on query analysis"""
        fields = []
        query_lower = query.lower()
        
        # Detect specific field needs from query
        if any(word in query_lower for word in ['price', 'cost', 'expensive', 'cheap', '$']):
            fields.append(SchemaField(
                name="price",
                field_type=FieldType.OPTIONAL_FLOAT,
                description="Price information",
                required=False
            ))
        
        if any(word in query_lower for word in ['location', 'address', 'where', 'city']):
            fields.append(SchemaField(
                name="location",
                field_type=FieldType.OPTIONAL_STRING,
                description="Location or address",
                required=False
            ))
        
        if any(word in query_lower for word in ['phone', 'contact', 'call']):
            fields.append(SchemaField(
                name="phone",
                field_type=FieldType.PHONE,
                description="Phone number",
                required=False
            ))
        
        if any(word in query_lower for word in ['email', 'contact']):
            fields.append(SchemaField(
                name="email",
                field_type=FieldType.EMAIL,
                description="Email address",
                required=False
            ))
        
        return fields
    
    def _analyze_sample_data(self, data_sample: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sample data to understand field types and patterns"""
        field_analysis = {}
        
        # Collect all field names and their values
        all_fields = set()
        for item in data_sample:
            all_fields.update(item.keys())
        
        # Analyze each field
        for field_name in all_fields:
            values = []
            for item in data_sample:
                if field_name in item and item[field_name] is not None:
                    values.append(item[field_name])
            
            if values:
                field_info = {
                    'name': field_name,
                    'total_count': len(data_sample),
                    'present_count': len(values),
                    'presence_rate': len(values) / len(data_sample),
                    'inferred_type': self._infer_field_type(values),
                    'sample_values': values[:5],  # Sample values for inspection
                    'unique_values': len(set(str(v) for v in values))
                }
                field_analysis[field_name] = field_info
        
        return field_analysis
    
    def _infer_field_type(self, values: List[Any]) -> FieldType:
        """Infer field type from sample values"""
        if not values:
            return FieldType.OPTIONAL_STRING
        
        # Check for specific patterns first
        str_values = [str(v) for v in values]
        
        # URL pattern
        if any(self._is_url(v) for v in str_values):
            return FieldType.URL
        
        # Email pattern
        if any(self._is_email(v) for v in str_values):
            return FieldType.EMAIL
        
        # Phone pattern
        if any(self._is_phone(v) for v in str_values):
            return FieldType.PHONE
        
        # Currency pattern
        if any(self._is_currency(v) for v in str_values):
            return FieldType.CURRENCY
        
        # Type-based inference
        type_counts = {}
        for value in values:
            if isinstance(value, bool):
                value_type = 'bool'
            elif isinstance(value, int):
                value_type = 'int'
            elif isinstance(value, float):
                value_type = 'float'
            elif isinstance(value, list):
                value_type = 'list'
            elif isinstance(value, dict):
                value_type = 'dict'
            else:
                value_type = 'str'
            
            type_counts[value_type] = type_counts.get(value_type, 0) + 1
        
        # Get most common type
        most_common_type = max(type_counts, key=type_counts.get)
        
        # Map to FieldType
        type_mapping = {
            'str': FieldType.OPTIONAL_STRING,
            'int': FieldType.OPTIONAL_INTEGER,
            'float': FieldType.OPTIONAL_FLOAT,
            'bool': FieldType.OPTIONAL_BOOLEAN,
            'list': FieldType.LIST,
            'dict': FieldType.DICT
        }
        
        return type_mapping.get(most_common_type, FieldType.OPTIONAL_STRING)
    
    def _generate_fields_from_analysis(self, field_analysis: Dict[str, Any]) -> List[SchemaField]:
        """Generate schema fields from sample data analysis"""
        fields = []
        
        for field_name, analysis in field_analysis.items():
            # Determine if field should be required
            presence_rate = analysis['presence_rate']
            required = presence_rate > 0.8  # Required if present in 80%+ of samples
            
            # Create field
            field = SchemaField(
                name=field_name,
                field_type=analysis['inferred_type'],
                description=f"Field inferred from sample data (presence: {presence_rate:.1%})",
                required=required
            )
            
            fields.append(field)
        
        return fields
    
    def _calculate_sample_confidence(self, data_sample: List[Dict[str, Any]], 
                                   field_analysis: Dict[str, Any]) -> float:
        """Calculate confidence based on sample data consistency"""
        if not data_sample or not field_analysis:
            return 0.0
        
        # Base confidence from sample size
        sample_size = len(data_sample)
        size_confidence = min(1.0, sample_size / 10)  # Max confidence at 10+ samples
        
        # Consistency confidence from field presence rates
        presence_rates = [analysis['presence_rate'] for analysis in field_analysis.values()]
        avg_presence_rate = sum(presence_rates) / len(presence_rates)
        
        # Combine confidences
        total_confidence = (size_confidence * 0.4) + (avg_presence_rate * 0.6)
        
        return min(1.0, total_confidence)
    
    def _create_pydantic_model(self, fields: List[SchemaField], 
                             model_name: str) -> Type[BaseModel]:
        """Create a Pydantic model from schema fields"""
        if not fields:
            # Create empty model
            return create_model(model_name)
        
        # Convert fields to Pydantic field definitions
        field_definitions = {}
        
        for field in fields:
            # Get Python type from FieldType
            python_type = self._get_python_type(field.field_type)
            
            # Create field with default and description
            if field.required:
                if field.default is not None:
                    field_def = (python_type, Field(default=field.default, description=field.description))
                else:
                    field_def = (python_type, Field(description=field.description))
            else:
                field_def = (python_type, Field(default=None, description=field.description))
            
            field_definitions[field.name] = field_def
        
        # Create the model
        return create_model(model_name, **field_definitions)
    
    def _get_python_type(self, field_type: FieldType) -> Type:
        """Convert FieldType to Python type"""
        type_mapping = {
            FieldType.STRING: str,
            FieldType.INTEGER: int,
            FieldType.FLOAT: float,
            FieldType.BOOLEAN: bool,
            FieldType.LIST: List[str],
            FieldType.DICT: Dict[str, Any],
            FieldType.OPTIONAL_STRING: Optional[str],
            FieldType.OPTIONAL_INTEGER: Optional[int],
            FieldType.OPTIONAL_FLOAT: Optional[float],
            FieldType.OPTIONAL_BOOLEAN: Optional[bool],
            FieldType.URL: str,
            FieldType.EMAIL: str,
            FieldType.PHONE: str,
            FieldType.DATE: str,
            FieldType.CURRENCY: float
        }
        
        return type_mapping.get(field_type, str)
    
    def _deduplicate_fields(self, fields: List[SchemaField]) -> List[SchemaField]:
        """Remove duplicate fields, preferring the most specific one"""
        field_dict = {}
        
        for field in fields:
            if field.name not in field_dict:
                field_dict[field.name] = field
            else:
                # Keep the more specific field (required over optional, specific type over generic)
                existing = field_dict[field.name]
                if (field.required and not existing.required) or \
                   (field.field_type != FieldType.OPTIONAL_STRING and existing.field_type == FieldType.OPTIONAL_STRING):
                    field_dict[field.name] = field
        
        return list(field_dict.values())
    
    def _filter_valid_fields(self, data: Dict[str, Any], 
                           schema_model: Type[BaseModel]) -> Dict[str, Any]:
        """Filter data to only include fields that are valid for the schema"""
        filtered_data = {}
        
        # Get model fields
        model_fields = schema_model.__fields__ if hasattr(schema_model, '__fields__') else {}
        
        for field_name, value in data.items():
            if field_name in model_fields:
                try:
                    # Try to validate just this field
                    temp_data = {field_name: value}
                    schema_model(**temp_data)
                    filtered_data[field_name] = value
                except ValidationError:
                    continue  # Skip invalid fields
        
        return filtered_data
    
    def _analyze_validation_failures(self, validation_failures: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze validation failures to understand common patterns"""
        failure_patterns = {}
        
        for failure in validation_failures:
            errors = failure.get('errors', [])
            for error in errors:
                if ':' in error:
                    field_part, error_part = error.split(':', 1)
                    field_name = field_part.replace("Field '", "").replace("'", "")
                    
                    if field_name not in failure_patterns:
                        failure_patterns[field_name] = []
                    failure_patterns[field_name].append(error_part.strip())
        
        return failure_patterns
    
    def _refine_fields_from_failures(self, original_fields: List[SchemaField],
                                   failure_analysis: Dict[str, Any]) -> List[SchemaField]:
        """Refine field definitions based on validation failure analysis"""
        refined_fields = []
        
        for field in original_fields:
            if field.name in failure_analysis:
                # Make field optional if it's causing validation errors
                if field.required:
                    refined_field = SchemaField(
                        name=field.name,
                        field_type=self._make_field_optional(field.field_type),
                        description=f"{field.description} (made optional due to validation issues)",
                        required=False,
                        default=field.default,
                        constraints=field.constraints
                    )
                    refined_fields.append(refined_field)
                else:
                    refined_fields.append(field)
            else:
                refined_fields.append(field)
        
        return refined_fields
    
    def _make_field_optional(self, field_type: FieldType) -> FieldType:
        """Convert a field type to its optional equivalent"""
        optional_mapping = {
            FieldType.STRING: FieldType.OPTIONAL_STRING,
            FieldType.INTEGER: FieldType.OPTIONAL_INTEGER,
            FieldType.FLOAT: FieldType.OPTIONAL_FLOAT,
            FieldType.BOOLEAN: FieldType.OPTIONAL_BOOLEAN,
            FieldType.URL: FieldType.OPTIONAL_STRING,
            FieldType.EMAIL: FieldType.OPTIONAL_STRING,
            FieldType.PHONE: FieldType.OPTIONAL_STRING,
            FieldType.DATE: FieldType.OPTIONAL_STRING,
            FieldType.CURRENCY: FieldType.OPTIONAL_FLOAT
        }
        
        return optional_mapping.get(field_type, field_type)
    
    def _is_url(self, value: str) -> bool:
        """Check if value looks like a URL"""
        return bool(re.match(r'^https?://', str(value)))
    
    def _is_email(self, value: str) -> bool:
        """Check if value looks like an email"""
        return bool(re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', str(value)))
    
    def _is_phone(self, value: str) -> bool:
        """Check if value looks like a phone number"""
        phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        return bool(re.search(phone_pattern, str(value)))
    
    def _is_currency(self, value: str) -> bool:
        """Check if value looks like currency"""
        currency_pattern = r'[\$£€¥]?\d+([,.]?\d{2,3})*\.?\d{0,2}'
        return bool(re.search(currency_pattern, str(value)))
    
    def _init_intent_field_mappings(self) -> Dict[str, List[SchemaField]]:
        """Initialize intent-specific field mappings"""
        return {
            'product': [
                SchemaField("name", FieldType.STRING, "Product name", required=True),
                SchemaField("price", FieldType.CURRENCY, "Product price", required=False),
                SchemaField("brand", FieldType.OPTIONAL_STRING, "Product brand"),
                SchemaField("category", FieldType.OPTIONAL_STRING, "Product category"),
                SchemaField("rating", FieldType.OPTIONAL_FLOAT, "Product rating"),
                SchemaField("reviews_count", FieldType.OPTIONAL_INTEGER, "Number of reviews"),
                SchemaField("availability", FieldType.OPTIONAL_STRING, "Availability status"),
                SchemaField("sku", FieldType.OPTIONAL_STRING, "Product SKU"),
            ],
            'restaurant': [
                SchemaField("name", FieldType.STRING, "Restaurant name", required=True),
                SchemaField("cuisine", FieldType.OPTIONAL_STRING, "Cuisine type"),
                SchemaField("rating", FieldType.OPTIONAL_FLOAT, "Restaurant rating"),
                SchemaField("price_range", FieldType.OPTIONAL_STRING, "Price range (e.g., $$)"),
                SchemaField("address", FieldType.OPTIONAL_STRING, "Restaurant address"),
                SchemaField("phone", FieldType.PHONE, "Phone number"),
                SchemaField("hours", FieldType.OPTIONAL_STRING, "Operating hours"),
                SchemaField("website", FieldType.URL, "Restaurant website"),
            ],
            'news': [
                SchemaField("headline", FieldType.STRING, "News headline", required=True),
                SchemaField("author", FieldType.OPTIONAL_STRING, "Article author"),
                SchemaField("publication_date", FieldType.DATE, "Publication date"),
                SchemaField("source", FieldType.OPTIONAL_STRING, "News source"),
                SchemaField("category", FieldType.OPTIONAL_STRING, "News category"),
                SchemaField("summary", FieldType.OPTIONAL_STRING, "Article summary"),
                SchemaField("content", FieldType.OPTIONAL_STRING, "Full article content"),
            ],
            'property': [
                SchemaField("address", FieldType.STRING, "Property address", required=True),
                SchemaField("price", FieldType.CURRENCY, "Property price"),
                SchemaField("bedrooms", FieldType.OPTIONAL_INTEGER, "Number of bedrooms"),
                SchemaField("bathrooms", FieldType.OPTIONAL_FLOAT, "Number of bathrooms"),
                SchemaField("square_feet", FieldType.OPTIONAL_INTEGER, "Square footage"),
                SchemaField("property_type", FieldType.OPTIONAL_STRING, "Type of property"),
                SchemaField("year_built", FieldType.OPTIONAL_INTEGER, "Year built"),
                SchemaField("agent_name", FieldType.OPTIONAL_STRING, "Real estate agent"),
                SchemaField("agent_phone", FieldType.PHONE, "Agent phone number"),
            ],
            'job': [
                SchemaField("title", FieldType.STRING, "Job title", required=True),
                SchemaField("company", FieldType.OPTIONAL_STRING, "Company name"),
                SchemaField("location", FieldType.OPTIONAL_STRING, "Job location"),
                SchemaField("salary", FieldType.OPTIONAL_STRING, "Salary information"),
                SchemaField("employment_type", FieldType.OPTIONAL_STRING, "Employment type"),
                SchemaField("experience_level", FieldType.OPTIONAL_STRING, "Required experience"),
                SchemaField("posted_date", FieldType.DATE, "Job posting date"),
                SchemaField("application_deadline", FieldType.DATE, "Application deadline"),
            ]
        }
    
    def _init_field_patterns(self) -> Dict[str, str]:
        """Initialize field recognition patterns"""
        return {
            'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            'phone': r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            'url': r'https?://[^\s]+',
            'price': r'[\$£€¥]\d+([,.]?\d{2,3})*\.?\d{0,2}',
            'date': r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{1,2}-\d{1,2}'
        }
    
    def _init_type_inference_rules(self) -> Dict[str, Any]:
        """Initialize type inference rules"""
        return {
            'integer_indicators': ['count', 'number', 'qty', 'quantity', 'age', 'year'],
            'float_indicators': ['price', 'cost', 'amount', 'rate', 'rating', 'score'],
            'boolean_indicators': ['is_', 'has_', 'active', 'enabled', 'available'],
            'date_indicators': ['date', 'time', 'created', 'updated', 'published'],
            'url_indicators': ['url', 'link', 'href', 'website', 'site']
        }
