# Pipeline Examples

This directory contains example implementations of the SmartScrape pipeline architecture. Each example demonstrates different aspects of creating, configuring, and executing data processing pipelines.

## Examples Overview

### 1. `extraction_pipeline.py`

**Purpose:** Demonstrates a complete content extraction pipeline that fetches content from a URL, processes HTML, extracts structured data, normalizes it, validates against a schema, and outputs JSON.

**Key Concepts:**
- Basic pipeline creation and execution
- Using standard input/processing/output stages
- Error handling and validation
- Two different pipeline creation methods: direct and registry-based

### 2. `custom_stage_example.py`

**Purpose:** Shows how to create custom pipeline stages by implementing the `PipelineStage` interface, focusing on text analysis functionality.

**Key Concepts:**
- Custom stage implementation
- Configuration handling
- Stage lifecycle hooks
- Input validation
- Error handling
- Context data interaction

## Running the Examples

### Prerequisites

Ensure you have installed all required dependencies:

```bash
pip install -r requirements.txt
```

### Running an Example

Each example can be run directly as a Python script:

```bash
# Run the extraction pipeline example
python examples/pipelines/extraction_pipeline.py

# Run the custom stage example
python examples/pipelines.custom_stage_example.py
```

## Expected Outputs

### Extraction Pipeline Example

The extraction pipeline example will produce output similar to:

```json
{
  "success": true,
  "url": "https://example.com/product",
  "data": {
    "name": "Product Name",
    "price": {
      "value": 99.99,
      "currency": "USD"
    },
    "description": "This is a sample product description.",
    "extracted_at": "2025-05-06T14:30:25.123456"
  },
  "metrics": {
    "pipeline_name": "extraction_pipeline",
    "total_time": 0.35,
    "stages": {
      "HttpInputStage": { "status": "success", "execution_time": 0.05 },
      "HtmlProcessingStage": { "status": "success", "execution_time": 0.07 },
      "ContentExtractionStage": { "status": "success", "execution_time": 0.08 },
      "DataNormalizationStage": { "status": "success", "execution_time": 0.06 },
      "SchemaValidationStage": { "status": "success", "execution_time": 0.04 },
      "JsonOutputStage": { "status": "success", "execution_time": 0.05 }
    },
    "successful_stages": 6,
    "total_stages": 6,
    "has_errors": false
  }
}
```

### Custom Stage Example

The custom stage example will produce output similar to:

```json
{
  "success": true,
  "original_text": "SmartScrape is an amazing tool for web extraction...",
  "output": {
    "original_text": "SmartScrape is an amazing tool for web extraction...",
    "analysis": {
      "keywords": ["extraction", "pipeline", "architecture", "flexible", "powerful"],
      "sentiment": {
        "score": 0.875,
        "label": "positive",
        "positive_words": ["amazing", "easy", "efficient", "flexible", "powerful", "happy", "clean", "recommended"],
        "negative_words": []
      },
      "word_count": 55,
      "char_count": 370
    },
    "summary": "SmartScrape is an amazing tool for web extraction. It makes gathering data from websites incredibly easy and efficient.",
    "metadata": {
      "generated_at": "2025-05-06T14:35:10.123456",
      "version": "1.0"
    }
  },
  "metrics": {
    "pipeline_name": "text_analysis_pipeline",
    "total_time": 0.45,
    "stages": {
      "KeywordExtractionStage": { "status": "success", "execution_time": 0.12 },
      "SentimentAnalysisStage": { "status": "success", "execution_time": 0.11 },
      "TextEnrichmentStage": { "status": "success", "execution_time": 0.14 },
      "JsonFormatterStage": { "status": "success", "execution_time": 0.08 }
    },
    "successful_stages": 4,
    "total_stages": 4,
    "has_errors": false
  }
}
```

## Learning Exercises

Here are some modifications you can make to the examples to learn more about pipeline architecture:

### Extraction Pipeline Modifications

1. **Add Error Handling**: Modify the URL to a non-existent one to see how errors are handled and propagated.

2. **Add a New Stage**: Create and add a new stage that enhances the extracted data with additional information (e.g., add timestamp, user agent, etc.).

3. **Enable Parallel Execution**: Modify the pipeline configuration to run compatible stages in parallel and observe performance differences.

4. **Add Conditional Branching**: Add logic to process different types of content differently based on the detected content type.

### Custom Stage Modifications

1. **Add New Analysis**: Create a new stage that performs additional text analysis (e.g., readability metrics, topic detection).

2. **Modify Stage Dependencies**: Change the pipeline to have more complex stage dependencies and experiment with parallel execution.

3. **Implement Caching**: Add a caching mechanism to avoid reprocessing identical inputs.

4. **Create a Circuit Breaker**: Implement a stage that acts as a circuit breaker to handle potential failures in external services.

## Integration with Other Components

The examples can be integrated with other SmartScrape components:

- Use the `extraction_pipeline.py` example with the `adaptive_scraper.py` controller
- Connect the pipeline with external storage systems to persist results
- Integrate with the monitoring system for advanced metrics and alerting

## Next Steps

After exploring these examples, you might want to:

1. Review the detailed documentation in the `docs/pipeline/` directory
2. Create your own custom stages for specific use cases
3. Contribute to the standard stage library by implementing new stage types