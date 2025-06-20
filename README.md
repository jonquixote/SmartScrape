# Intelligent Web Scraper (SmartScrape)

A web application that combines crawl4ai with Google's Generative AI and Beautiful Soup for intelligent web scraping.

## Features

- Simple URL input field accepting both specific URLs and general website names
- Multiple scraping strategies including:
  - Basic page scraping (single page)
  - Breadth-first crawling (explore all links at each depth level)
  - Depth-first crawling (explore deep paths before moving to siblings)
  - **NEW: AI-guided strategy** (intelligently navigate websites based on user intent)
  - **NEW: Pipeline architecture** (modular, configurable data processing workflows)
- Text input field for users to describe what data to extract
- Integration with Google Generative AI to optimize extraction requests
- Visual loading indicator showing real-time scraping progress
- Results display with export options (JSON, Excel, CSV)
- Error handling for inaccessible sites or failed scrapes
- Anti-CAPTCHA mechanisms including request throttling, rotating user agents, proxy rotation, browser fingerprint randomization, and headless browser detection avoidance

## AI-Guided Strategy

The new AI-guided strategy is the most advanced way to scrape websites with SmartScrape. It uses natural language processing to understand your intent and intelligently navigate websites to find the most relevant information.

### Key Features of AI-Guided Strategy

- **Natural Language Understanding**: Describe what you're looking for in plain English
- **Intent-Based Navigation**: Prioritizes links and pages based on relevance to your request
- **Smart Content Evaluation**: Analyzes page content to determine its relevance to your query
- **Result Consolidation**: Combines information from multiple pages into cohesive results
- **Adaptive Exploration**: Balances exploring new areas vs. exploiting known good sources
- **Site Feature Detection**: Automatically identifies and utilizes sitemaps and search forms
- **Transparent Decision Making**: Provides explanations for why pages were selected

### Configuration Options

- **Exploration Ratio**: Balance between exploring new content vs. focusing on known relevant areas
- **Include External Links**: Allow the crawler to follow links to external domains
- **Use Sitemap**: Automatically detect and utilize website sitemaps for efficient crawling
- **Use Search Forms**: Detect and utilize website search functionality
- **AI Model Quality**: Choose between standard and high-quality AI models for different needs

### Example Use Cases

- **Product Research**: "Find information about the latest smartphones under $500 with good battery life"
- **Content Aggregation**: "Collect recent articles about climate change adaptation strategies"
- **Comparative Analysis**: "Compare pricing and features of online project management tools"
- **Information Extraction**: "Extract contact information for tech companies in San Francisco"

## Pipeline Architecture

SmartScrape now includes a powerful pipeline architecture for building modular, customizable data processing workflows. This architecture makes it easy to create, configure, and execute complex data extraction and transformation processes.

### Key Features of Pipeline Architecture

- **Modular Design**: Compose pipelines from reusable, single-purpose stages
- **Configuration-Driven**: Create and modify pipelines using simple configuration files
- **Parallel Execution**: Process independent stages concurrently for better performance
- **Error Handling**: Comprehensive error handling with recovery mechanisms
- **Monitoring & Metrics**: Track execution progress and performance metrics
- **Extensibility**: Create custom stages for specialized processing needs

### Core Components

- **Pipeline**: Orchestrates the execution of multiple stages
- **Stages**: Individual processing units that perform specific operations
- **Context**: Shared state object that flows through the pipeline
- **Registry**: Catalog of available stages and pipeline templates

### Example Pipeline

```python
# Create a pipeline for product extraction
pipeline = Pipeline("product_extraction")

# Add stages to the pipeline
pipeline.add_stage(HttpInputStage({"url": "https://example.com/product"}))
pipeline.add_stage(HtmlProcessingStage({"remove_scripts": True}))
pipeline.add_stage(ContentExtractionStage())
pipeline.add_stage(DataNormalizationStage())
pipeline.add_stage(SchemaValidationStage())
pipeline.add_stage(JsonOutputStage({"pretty_print": True}))

# Execute the pipeline
result = await pipeline.execute()
```

### Quick Start

To use the pipeline architecture in your own code:

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Create a Basic Pipeline**:
   ```python
   from core.pipeline.pipeline import Pipeline
   from core.pipeline.stage import PipelineStage
   
   # Create a custom stage
   class MyCustomStage(PipelineStage):
       async def process(self, context):
           # Your processing logic here
           return True
   
   # Create and run a pipeline
   pipeline = Pipeline("my_pipeline")
   pipeline.add_stage(MyCustomStage())
   result = await pipeline.execute()
   ```

3. **Explore Examples**:
   Check out the examples in `examples/pipelines/` for more advanced use cases.

### Documentation

For more detailed information about the pipeline architecture, see:

- [Pipeline Architecture Overview](docs/pipeline/architecture.md)
- [Usage Guide](docs/pipeline/usage_guide.md)
- [Custom Stages Guide](docs/pipeline/custom_stages.md)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/intelligent-web-scraper.git
cd intelligent-web-scraper
```

## Setting Up the Environment

To ensure proper isolation of dependencies, it's recommended to use a Python virtual environment. Follow these steps:

1. Navigate to the project directory:

   ```bash
   cd /Users/johnny/Downloads/SmartScrape
   ```

2. Create a virtual environment:

   ```bash
   python3 -m venv venv
   ```

3. Activate the virtual environment:

   ```bash
   source venv/bin/activate
   ```

4. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Configure your API keys:
   
   Copy `config.py.example` to `config.py` and add your Google API key:
   ```python
   GOOGLE_API_KEY = "your_api_key_here"
   ```

6. Run the application:

   ```bash
   python app.py
   ```

7. When finished, deactivate the virtual environment:

   ```bash
   deactivate
   ```

This setup ensures that the project dependencies are isolated from your system Python installation.

## Running Tests

SmartScrape includes comprehensive unit tests to ensure all components work correctly. To run the tests:

```bash
# Navigate to the tests directory
cd tests

# Run all tests
python run_tests.py --all

# Run only strategy tests
python run_tests.py --strategies

# Run a specific test
python run_tests.py --test test_ai_guided_strategy
```

## Project Structure

```
app.py                  # Main application entry point
config.py               # Configuration settings
requirements.txt        # Project dependencies
ai_helpers/             # AI-related helper functions
├── prompt_generator.py # Generates prompts for AI models
└── response_parser.py  # Parses responses from AI models
components/             # Specialized components for web scraping
├── domain_intelligence.py
├── pagination_handler.py
├── search_automation.py  # Search form detection and automation
├── site_discovery.py     # Site structure discovery including sitemaps
└── template_storage.py
extraction/             # Content extraction modules
├── content_analysis.py   # Analyzes page content structure
├── extraction_helpers.py # Helper functions for extraction
└── fallback_extraction.py # Fallback methods when primary extraction fails
strategies/             # Crawling strategies
├── ai_guided_strategy.py # AI-guided crawling strategy
├── base_strategy.py      # Abstract base class for strategies
├── best_first.py         # Best-first search implementation
├── bfs_strategy.py       # Breadth-first search implementation
├── dfs_strategy.py       # Depth-first search implementation
└── result_consolidation.py # Consolidates results from multiple pages
tests/                  # Test suite
└── strategies/           # Tests for crawling strategies
utils/                  # Utility functions
├── export.py             # Export functionality (JSON, CSV, Excel)
└── url_filters.py        # URL filtering utilities
web/                    # Web interface components
├── models.py             # Pydantic models
├── routes.py             # FastAPI routes
└── templates.py          # HTML templates
```

## Troubleshooting

### Python 3.13 Compatibility Issue

If you encounter errors related to pandas during installation, it may be due to compatibility issues with Python 3.13. Here are some solutions:

1. **Use a compatible pandas version**:
   Ensure you have a compatible version of pandas installed:
   ```bash
   pip install pandas>=2.2.0
   ```

2. **Use an older Python version**:
   If you have Python 3.11 or 3.12 installed, create a new virtual environment with one of these versions:
   ```bash
   deactivate  # Deactivate the current virtual environment
   python3.11 -m venv venv_py311
   source venv_py311/bin/activate
   pip install -r requirements.txt
   ```

3. **Update `requirements.txt`**:
   Ensure your `requirements.txt` specifies a compatible pandas version:
   ```plaintext
   pandas>=2.2.0
   ```

These steps should resolve any compatibility issues with Python 3.13.

## Production Readiness

This application has been configured for production deployment with the following features:

### Authentication and Security
- API key authentication system for protecting endpoints
- Rate limiting to prevent abuse
- Security headers for enhanced protection
- Environment-based configuration with `.env.production`
- Secrets management through environment variables

### Monitoring and Observability
- Health check endpoint (`/health`) for system status monitoring
- Prometheus metrics endpoint (`/metrics`) for collecting performance metrics
- Structured logging with JSON format in production
- Request ID tracking for distributed tracing

### Database and Data Management
- SQLAlchemy ORM for database interactions
- Alembic migrations for database schema management
- Audit logging for tracking system events and user actions
- Data backup script in `scripts/backup.sh`

### Deployment
- Multi-stage Docker builds for optimized container images
- Docker Compose configuration for entire stack deployment
- Nginx for reverse proxy, load balancing, and TLS termination
- Redis integration for caching and session management
- Supervisord for process management
- Healthchecks for container monitoring
- CI/CD pipeline with GitHub Actions

See the `docs/production_deployment.md` for detailed deployment instructions.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.