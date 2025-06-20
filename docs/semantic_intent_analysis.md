# Semantic Intent Analysis in SmartScrape

## Overview

SmartScrape's Semantic Intent Analysis leverages advanced Natural Language Processing (NLP) through spaCy and sentence transformers to understand user queries, expand context, and intelligently guide the web scraping process. This document explains the implementation, capabilities, and usage of the semantic analysis system.

## Architecture

### Components

1. **UniversalIntentAnalyzer** - Core semantic analysis engine
2. **Sentence Transformers** - Embedding generation for semantic similarity
3. **spaCy Models** - Advanced NLP processing and entity recognition
4. **Contextual Query Expansion** - Intelligent query enhancement

### Data Flow

```
User Query → Intent Analysis → Semantic Embedding → Context Expansion → URL Generation → Targeted Scraping
```

## Core Features

### 1. Intent Classification

The system automatically classifies user intents into categories:

- **E-commerce**: Product searches, price comparisons, reviews
- **News**: Current events, article searches, press releases
- **Research**: Academic papers, documentation, technical resources
- **Social**: Social media content, user profiles, discussions
- **Business**: Company information, contact details, services
- **Real Estate**: Property listings, market data, agent information
- **Travel**: Hotels, flights, destinations, reviews

### 2. Semantic Query Expansion

Automatically enhances queries with:

- **Synonyms and related terms**
- **Domain-specific vocabulary**
- **Common variations and alternatives**
- **Industry-specific terminology**

Example:
```
Input: "cheap laptops"
Expanded: "cheap laptops affordable notebooks budget computers discounted portable PCs"
```

### 3. Entity Recognition

Extracts and categorizes entities from queries:

- **Organizations**: Company names, brands
- **Products**: Specific items, models, categories
- **Locations**: Cities, regions, addresses
- **Monetary values**: Prices, budgets, ranges
- **Dates**: Time periods, deadlines, ranges

## Implementation Details

### UniversalIntentAnalyzer Class

```python
from components.universal_intent_analyzer import UniversalIntentAnalyzer

# Initialize the analyzer
analyzer = UniversalIntentAnalyzer()

# Analyze a query
result = await analyzer.analyze_intent(
    query="Find affordable restaurants in downtown Seattle",
    context={"location": "Seattle", "budget": "affordable"}
)

print(result.intent_category)  # "business"
print(result.confidence)      # 0.85
print(result.entities)        # [{"text": "Seattle", "label": "GPE"}]
print(result.expanded_query)   # Enhanced with synonyms and related terms
```

### Configuration Options

Available in `config.py`:

```python
# spaCy settings
SPACY_ENABLED = True
SPACY_MODEL = "en_core_web_md"  # Options: sm, md, lg
SPACY_INTENT_ANALYSIS = True

# Semantic Search & Intent Analysis
SEMANTIC_SEARCH_ENABLED = True
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
CONTEXTUAL_QUERY_EXPANSION = True

# Performance settings
INTENT_ANALYSIS_CACHE_SIZE = 1000
INTENT_ANALYSIS_TIMEOUT = 30.0
```

## Usage Examples

### Basic Intent Analysis

```python
from components.universal_intent_analyzer import UniversalIntentAnalyzer

analyzer = UniversalIntentAnalyzer()

# Simple query analysis
result = await analyzer.analyze_intent("best pizza places near me")

print(f"Intent: {result.intent_category}")
print(f"Confidence: {result.confidence}")
print(f"Expanded Query: {result.expanded_query}")
print(f"Entities: {result.entities}")
```

### Advanced Analysis with Context

```python
# Provide additional context for better analysis
context = {
    "user_location": "New York",
    "previous_searches": ["restaurants", "food delivery"],
    "time_of_day": "evening"
}

result = await analyzer.analyze_intent(
    query="quick dinner options",
    context=context
)
```

### Domain-Specific Analysis

```python
# E-commerce focused analysis
ecommerce_result = await analyzer.analyze_intent(
    query="wireless noise canceling headphones under $200",
    domain_hint="ecommerce"
)

# Real estate focused analysis
realestate_result = await analyzer.analyze_intent(
    query="3 bedroom houses in Austin Texas",
    domain_hint="real_estate"
)
```

## Integration with Other Components

### URL Generation

The intent analysis results are used by the `IntelligentURLGenerator`:

```python
from components.intelligent_url_generator import IntelligentURLGenerator

url_generator = IntelligentURLGenerator()

# URLs are generated based on intent analysis
urls = await url_generator.generate_urls(
    query="Italian restaurants downtown",
    intent_result=analysis_result
)
```

### Content Quality Scoring

Intent analysis influences content relevance scoring:

```python
from components.content_quality_scorer import ContentQualityScorer

scorer = ContentQualityScorer()

# Content is scored based on alignment with user intent
quality_score = await scorer.score_content(
    content=scraped_content,
    intent_result=analysis_result
)
```

## Performance Considerations

### Caching Strategy

- **Intent results are cached** to avoid re-analysis of similar queries
- **Embeddings are cached** for frequently used terms
- **Model loading is optimized** with lazy initialization

### Memory Management

- **Batch processing** for multiple queries
- **Model optimization** using smaller variants when appropriate
- **Resource cleanup** for long-running processes

### Latency Optimization

- **Parallel processing** of analysis components
- **Pre-computed embeddings** for common terms
- **Efficient model selection** based on query complexity

## Monitoring and Metrics

### Key Metrics

- **Analysis latency** (target: < 500ms)
- **Intent classification accuracy**
- **Entity extraction recall**
- **Query expansion effectiveness**
- **Cache hit rates**

### Logging

The system provides detailed logging for:

- Intent classification decisions
- Entity extraction results
- Query expansion processes
- Performance metrics
- Error handling

## Best Practices

### Query Preparation

1. **Normalize input**: Remove extra whitespace, handle Unicode
2. **Context provision**: Include relevant context when available
3. **Domain hints**: Specify domain when known for better accuracy

### Performance Optimization

1. **Batch processing**: Analyze multiple queries together when possible
2. **Cache warmup**: Pre-load common embeddings
3. **Model selection**: Use appropriate model size for use case

### Error Handling

1. **Graceful degradation**: Fall back to keyword-based analysis
2. **Timeout handling**: Set appropriate timeouts for analysis
3. **Resource monitoring**: Monitor memory and CPU usage

## Advanced Features

### Custom Intent Categories

```python
# Define custom intent categories
custom_categories = {
    "medical": ["health", "medicine", "doctor", "hospital"],
    "legal": ["law", "attorney", "court", "legal"]
}

analyzer = UniversalIntentAnalyzer(custom_categories=custom_categories)
```

### Multi-language Support

```python
# Configure for different languages
analyzer = UniversalIntentAnalyzer(
    spacy_model="de_core_news_md",  # German model
    sentence_model="distiluse-base-multilingual-cased"
)
```

### Semantic Similarity Search

```python
# Find semantically similar content
similar_queries = await analyzer.find_similar_queries(
    query="budget smartphones",
    threshold=0.8,
    max_results=10
)
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure spaCy models are installed: `python -m spacy download en_core_web_md`
   - Check available disk space for model files

2. **Performance Issues**
   - Monitor memory usage during analysis
   - Consider using smaller models for high-volume scenarios
   - Implement proper caching strategies

3. **Accuracy Issues**
   - Provide more context in queries
   - Use domain hints for specialized content
   - Consider custom training data for specific domains

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging
logging.getLogger('semantic_analysis').setLevel(logging.DEBUG)

# Analyze with detailed logging
result = await analyzer.analyze_intent(query, debug=True)
```

## Future Enhancements

### Planned Features

1. **Custom model fine-tuning** for domain-specific improvements
2. **User feedback integration** for continuous learning
3. **Multi-modal analysis** including image and video content
4. **Real-time learning** from successful extraction patterns

### Research Areas

1. **Cross-lingual intent analysis** for global content
2. **Temporal intent understanding** for time-sensitive queries
3. **Personalized intent modeling** based on user history
4. **Federated learning** for privacy-preserving improvements

---

*For technical support or questions about semantic intent analysis, please refer to the main documentation or contact the development team.*
