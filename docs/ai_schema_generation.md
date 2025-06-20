# AI Schema Generation in SmartScrape

## Overview

SmartScrape's AI Schema Generation system automatically creates Pydantic data models and validation schemas based on scraped content and user intent. This intelligent system eliminates the need for manual schema definition and adapts to diverse data structures across different websites and content types.

## Architecture

### Components

1. **AISchemaGenerator** - Core schema generation engine
2. **Pydantic Integration** - Dynamic model creation and validation
3. **Content Analysis Engine** - Structure and pattern detection
4. **Schema Optimization** - Performance and accuracy improvements
5. **Validation Pipeline** - Quality assurance and error handling

### Data Flow

```
Scraped Content → Structure Analysis → Pattern Detection → Schema Generation → Validation → Optimized Model
```

## Core Features

### 1. Automatic Schema Detection

The system analyzes content to identify:

- **Data types**: Strings, numbers, dates, URLs, emails
- **Nested structures**: Objects, arrays, hierarchical data
- **Optional fields**: Nullable and conditional attributes
- **Relationships**: Foreign keys, references, dependencies
- **Constraints**: Min/max values, patterns, enumerations

### 2. Content-Aware Schema Generation

Creates schemas tailored to specific content types:

- **E-commerce**: Products, prices, reviews, specifications
- **Real Estate**: Properties, prices, locations, features
- **News**: Articles, authors, dates, categories
- **Social Media**: Posts, users, metrics, interactions
- **Business**: Companies, contacts, services, locations

### 3. Dynamic Model Creation

Generates Pydantic models at runtime:

```python
# Generated model example
class ProductSchema(BaseModel):
    name: str
    price: Optional[float] = None
    description: Optional[str] = None
    rating: Optional[float] = Field(ge=0, le=5)
    availability: bool = True
    categories: List[str] = []
    specifications: Dict[str, Any] = {}
```

## Implementation Details

### AISchemaGenerator Class

```python
from components.ai_schema_generator import AISchemaGenerator

# Initialize the generator
generator = AISchemaGenerator()

# Generate schema from sample data
schema = await generator.generate_schema(
    sample_data=[
        {"name": "iPhone 15", "price": 999.99, "rating": 4.5},
        {"name": "Samsung Galaxy", "price": 899.99, "rating": 4.3}
    ],
    content_type="product",
    intent_context="e-commerce search"
)

print(schema)  # Generated Pydantic model class
```

### Configuration Options

Available in `config.py`:

```python
# AI Schema Generation
AI_SCHEMA_GENERATION_ENABLED = True
AI_SCHEMA_GENERATION_MODEL = "gpt-3.5-turbo"
AI_SCHEMA_GENERATION_MAX_TOKENS = 2000

# Schema optimization
SCHEMA_CACHE_ENABLED = True
SCHEMA_VALIDATION_STRICT = True
SCHEMA_FIELD_DETECTION_THRESHOLD = 0.7

# Performance settings
SCHEMA_GENERATION_TIMEOUT = 30.0
SCHEMA_COMPLEXITY_LIMIT = 50  # Maximum fields per schema
```

## Usage Examples

### Basic Schema Generation

```python
from components.ai_schema_generator import AISchemaGenerator

generator = AISchemaGenerator()

# Generate schema from scraped data
sample_data = [
    {
        "title": "MacBook Pro 16-inch",
        "price": "$2,399.00",
        "rating": "4.5 stars",
        "reviews": 1250,
        "in_stock": True
    }
]

schema = await generator.generate_schema(
    sample_data=sample_data,
    content_type="product"
)

# Use the generated schema
validated_data = schema.validate(new_product_data)
```

### Schema with Custom Constraints

```python
# Generate schema with specific requirements
schema = await generator.generate_schema(
    sample_data=restaurant_data,
    content_type="restaurant",
    constraints={
        "price_range": {"type": "enum", "values": ["$", "$$", "$$$", "$$$$"]},
        "rating": {"type": "float", "min": 0, "max": 5},
        "phone": {"type": "string", "pattern": r"^\+?1?-?\d{3}-?\d{3}-?\d{4}$"}
    }
)
```

### Intent-Driven Schema Generation

```python
# Schema generation based on user intent
from components.universal_intent_analyzer import UniversalIntentAnalyzer

analyzer = UniversalIntentAnalyzer()
intent_result = await analyzer.analyze_intent("find luxury hotels in Paris")

schema = await generator.generate_schema(
    sample_data=hotel_data,
    intent_context=intent_result,
    focus_areas=["luxury_features", "location", "pricing"]
)
```

## Schema Types and Templates

### E-commerce Schemas

```python
# Product schema template
class ProductSchemaTemplate:
    base_fields = {
        "name": "str",
        "price": "Optional[float]",
        "description": "Optional[str]",
        "rating": "Optional[float]",
        "reviews_count": "Optional[int]",
        "availability": "bool",
        "brand": "Optional[str]",
        "category": "Optional[str]",
        "images": "List[str]",
        "specifications": "Dict[str, Any]"
    }
```

### Real Estate Schemas

```python
# Property schema template
class PropertySchemaTemplate:
    base_fields = {
        "address": "str",
        "price": "Optional[float]",
        "bedrooms": "Optional[int]",
        "bathrooms": "Optional[float]",
        "square_feet": "Optional[int]",
        "lot_size": "Optional[float]",
        "year_built": "Optional[int]",
        "property_type": "Optional[str]",
        "features": "List[str]",
        "agent_info": "Optional[Dict[str, str]]"
    }
```

### News Article Schemas

```python
# Article schema template
class ArticleSchemaTemplate:
    base_fields = {
        "title": "str",
        "author": "Optional[str]",
        "published_date": "Optional[datetime]",
        "content": "str",
        "summary": "Optional[str]",
        "category": "Optional[str]",
        "tags": "List[str]",
        "source": "Optional[str]",
        "url": "Optional[str]"
    }
```

## Advanced Features

### 1. Hierarchical Schema Generation

```python
# Generate nested schemas for complex data
nested_schema = await generator.generate_nested_schema(
    sample_data=complex_product_data,
    max_depth=3,
    include_relationships=True
)
```

### 2. Schema Evolution

```python
# Update existing schemas with new data patterns
evolved_schema = await generator.evolve_schema(
    existing_schema=current_schema,
    new_data=additional_samples,
    merge_strategy="conservative"  # or "aggressive"
)
```

### 3. Multi-Source Schema Merging

```python
# Combine schemas from different data sources
merged_schema = await generator.merge_schemas(
    schemas=[schema_a, schema_b, schema_c],
    conflict_resolution="union",  # or "intersection"
    field_priority_rules=custom_rules
)
```

## Integration with Extraction Pipeline

### Extraction Coordinator Integration

```python
from controllers.extraction_coordinator import ExtractionCoordinator

coordinator = ExtractionCoordinator()

# Schema generation is integrated into the extraction process
result = await coordinator.coordinate_extraction(
    query="electronics products",
    max_pages=5,
    generate_schema=True,  # Enable automatic schema generation
    schema_optimization=True
)

# Access the generated schema
schema = result.metadata.get("generated_schema")
```

### Content Quality Scoring Integration

```python
from components.content_quality_scorer import ContentQualityScorer

scorer = ContentQualityScorer()

# Schema compliance affects quality scores
quality_score = await scorer.score_content(
    content=scraped_data,
    schema=generated_schema,
    check_schema_compliance=True
)
```

## Performance Optimization

### Caching Strategies

1. **Schema Templates**: Cache common schema patterns
2. **Field Patterns**: Store recognized field types and constraints
3. **Generation Results**: Cache schemas for similar content types
4. **Validation Rules**: Pre-compile common validation patterns

### Batch Processing

```python
# Generate schemas for multiple content types in batch
schemas = await generator.generate_schemas_batch(
    content_samples=[
        {"type": "product", "data": product_samples},
        {"type": "review", "data": review_samples},
        {"type": "store", "data": store_samples}
    ],
    parallel_processing=True
)
```

### Memory Management

```python
# Configure memory limits for large schema generation
generator = AISchemaGenerator(
    max_memory_mb=512,
    field_limit_per_schema=100,
    enable_garbage_collection=True
)
```

## Validation and Quality Assurance

### Schema Validation Pipeline

```python
# Comprehensive schema validation
validation_result = await generator.validate_schema(
    schema=generated_schema,
    test_data=sample_data,
    quality_checks=[
        "field_coverage",
        "type_accuracy",
        "constraint_validity",
        "performance_impact"
    ]
)
```

### Quality Metrics

- **Field Coverage**: Percentage of data fields captured
- **Type Accuracy**: Correctness of inferred data types
- **Constraint Validity**: Effectiveness of validation rules
- **Performance Impact**: Schema validation overhead
- **Flexibility**: Ability to handle data variations

## Error Handling

### Common Issues and Solutions

1. **Inconsistent Data Types**
   ```python
   # Handle mixed data types gracefully
   schema = await generator.generate_schema(
       sample_data=mixed_data,
       type_inference_strategy="lenient",
       fallback_to_string=True
   )
   ```

2. **Missing Required Fields**
   ```python
   # Generate schemas with optional fields
   schema = await generator.generate_schema(
       sample_data=incomplete_data,
       required_field_threshold=0.8,
       default_to_optional=True
   )
   ```

3. **Complex Nested Structures**
   ```python
   # Simplify complex nested data
   schema = await generator.generate_schema(
       sample_data=nested_data,
       max_nesting_depth=2,
       flatten_complex_objects=True
   )
   ```

## Monitoring and Analytics

### Key Metrics

- **Schema generation success rate**
- **Validation accuracy**
- **Performance metrics** (generation time, memory usage)
- **Schema reuse rate**
- **Field detection accuracy**

### Logging Configuration

```python
import logging

# Enable detailed schema generation logging
logging.getLogger('ai_schema_generator').setLevel(logging.INFO)

# Generate schema with logging
schema = await generator.generate_schema(
    sample_data=data,
    enable_logging=True,
    log_analysis_steps=True
)
```

## Best Practices

### Data Preparation

1. **Provide Representative Samples**: Include diverse examples of target data
2. **Clean Input Data**: Remove or handle malformed data before schema generation
3. **Specify Context**: Provide content type and intent information when available

### Schema Design

1. **Balance Flexibility and Strictness**: Allow for data variations while maintaining validation
2. **Consider Performance**: Avoid overly complex schemas for high-volume processing
3. **Plan for Evolution**: Design schemas that can adapt to changing data patterns

### Validation Strategy

1. **Test with Real Data**: Validate schemas against actual scraped content
2. **Monitor Performance**: Track validation overhead and optimize as needed
3. **Handle Exceptions**: Implement fallback strategies for validation failures

## API Reference

### AISchemaGenerator Methods

```python
class AISchemaGenerator:
    async def generate_schema(
        self,
        sample_data: List[Dict[str, Any]],
        content_type: Optional[str] = None,
        intent_context: Optional[Any] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Type[BaseModel]:
        """Generate a Pydantic schema from sample data."""
        
    async def validate_schema(
        self,
        schema: Type[BaseModel],
        test_data: List[Dict[str, Any]]
    ) -> ValidationResult:
        """Validate generated schema against test data."""
        
    async def optimize_schema(
        self,
        schema: Type[BaseModel],
        performance_requirements: Dict[str, Any]
    ) -> Type[BaseModel]:
        """Optimize schema for performance and accuracy."""
```

## Future Enhancements

### Planned Features

1. **Machine Learning Integration**: Learn from successful schema patterns
2. **Custom Field Types**: Support for domain-specific data types
3. **Schema Versioning**: Track and manage schema evolution over time
4. **Visual Schema Builder**: GUI interface for schema customization

### Research Areas

1. **Zero-shot Schema Generation**: Generate schemas without sample data
2. **Cross-domain Transfer**: Apply learned patterns across different domains
3. **Federated Schema Learning**: Collaborate on schema patterns while preserving privacy
4. **Automated Schema Testing**: Generate test cases for schema validation

---

*For technical support or questions about AI schema generation, please refer to the main documentation or contact the development team.*
