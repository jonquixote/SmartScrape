# SmartScrape Application Guide

## Overview

SmartScrape is an intelligent web scraper that combines various scraping strategies, Google's Generative AI, and Beautiful Soup for efficient and customizable data extraction. It features an adaptive scraping controller, AI-guided strategies, a modular pipeline architecture, and advanced AI-driven capabilities including semantic intent analysis, dynamic schema generation, resilience management, intelligent caching, user feedback integration, and progressive collection strategies.

## New Advanced Features (Phases 1-7)

### Semantic Intent Analysis

SmartScrape now includes advanced semantic intent analysis capabilities through the `UniversalIntentAnalyzer` that:

- Performs intelligent query expansion using spaCy NLP models
- Provides entity recognition and semantic search
- Classifies user intent into categories (e-commerce, news, research, social media, etc.)
- Generates contextually relevant search terms and URL patterns
- Enables semantic similarity matching for better content discovery

### AI Schema Generation

The new `AISchemaGenerator` provides dynamic schema creation:

- Content-aware schema generation for different domains
- Pydantic integration for robust data validation
- Multi-source schema merging and evolution
- Hierarchical schema structures for complex data
- Automatic field type inference and validation rules

### Resilience and Proxy Management

Enhanced robustness through comprehensive resilience features:

- Advanced proxy management with health monitoring
- Anti-detection measures including browser fingerprint randomization
- CAPTCHA detection and handling strategies
- Session management with intelligent rotation
- Real-time monitoring and failure recovery

### Intelligent Caching Strategy

Multi-tier caching system for optimal performance:

- Memory, Redis, and persistent storage tiers
- Content-aware caching strategies
- Intelligent cache invalidation and refresh mechanisms
- Component-specific caching patterns
- Performance optimization and memory management

### User Feedback Integration

Continuous learning and improvement through feedback:

- Multi-modal feedback collection (explicit, implicit, comparative)
- Sentiment analysis and feedback analytics
- Adaptive parameter tuning based on user preferences
- Personalization engine with user profiling
- Quality assessment and improvement mechanisms

### Progressive Collection and Consolidated AI Processing

Two-stage approach for efficiency and quality:

- Lightweight collection phase for rapid data gathering
- Consolidated AI processing for enhanced analysis
- Advanced deduplication and cross-page analysis
- Token efficiency and performance optimization
- Quality improvements through batch processing

## Adaptive Scraper

The `AdaptiveScraper` class in `controllers/adaptive_scraper.py` is the core of the application. It orchestrates the entire scraping process, from preparing search terms to processing results and applying fallbacks.

### Key Components

*   **Strategy Factory:** Creates and manages different scraping strategies.
*   **Strategy Context:** Provides a shared context for strategies, including configuration and services.
*   **Circuit Breaker Manager:** Manages circuit breakers to prevent failures when scraping specific domains.
*   **Service Registry:** A central registry for core services like search term generation, site discovery, and domain intelligence. The `ServiceRegistry` class in `core/service_registry.py` manages the application's services, handling registration, dependency resolution, and initialization. It uses a singleton pattern to ensure a single instance of the registry.
*   **Configuration:** The application's configuration is managed through the `core/configuration.py` file. This module defines the schema and default values for resource management services like session management, rate limiting, proxy management, error classification, retry management, and circuit breaking.
*   **Pipeline Architecture:** A modular system for building customizable data processing workflows.

### Workflow

The adaptive scraper follows these steps:

1.  **Prepare Search Terms:** Generates search terms based on the user's query and intent.
2.  **Select Search Strategy:** Selects the most appropriate scraping strategy based on the target URL and required capabilities.
3.  **Execute Search:** Executes the selected strategy to retrieve raw results.
4.  **Process Results:** Processes the raw results, extracting relevant data.
5.  **Apply Fallback (if needed):** If the primary strategy fails, applies fallback strategies to ensure successful scraping.

\`\`\`mermaid
graph LR
    A[User Query] --> B(Prepare Search Terms);
    B --> C{URL Provided?};
    C -- Yes --> D(Check Circuit Breaker);
    C -- No --> E(Select Search Strategy);
    D -- Allowed --> E;
    D -- Blocked --> F[Return Error: Circuit Breaker Open];
    E --> G(Execute Search);
    G --> H{Results?};
    H -- Yes --> I(Process Results);
    H -- No --> J(Apply Fallback);
    I --> K(Return Results);
    J --> G;
    K --> L[End];
    F --> L;
\`\`\`

## Key Modules and Directories

*   **`ai_helpers/`:** Contains helper functions for AI-related tasks, such as prompt generation and response parsing.
*   **`components/`:** Contains specialized components for web scraping, such as search automation, site discovery, and pagination handling.
*   **`controllers/`:** Contains controllers that orchestrate the scraping process, including the `AdaptiveScraper`.
*   **`core/`:** Contains core functionalities and services, such as AI models, caching, and service interfaces.
*   **`docs/`:** Contains documentation for the project.
*   **`extraction/`:** Contains modules for content extraction, analysis, and normalization.
*   **`extraction_strategies/`:** Contains configurations and preferences for extraction, including extraction profiles.
*   **`strategies/`:** Contains different scraping strategies, such as AI-guided, form-based, and URL parameter strategies.
*   **`utils/`:** Contains utility functions for tasks like exporting data and filtering URLs.
*   **`web/`:** Contains web interface components, such as models and routes.

## Detailed Module Descriptions

*   **`ai_helpers/`**: Provides utilities for interacting with AI models, including parsing natural language queries (`intent_parser.py`), generating prompts (`prompt_generator.py`), and processing AI responses (`response_parser.py`). It also includes a rule-based extraction helper (`rule_based_extractor.py`).
*   **`components/`**: Houses reusable components for various scraping tasks. This includes handling pagination (`pagination_handler.py`), automating search interactions (`search_automation.py`), orchestrating search processes (`search_orchestrator.py`), integrating search templates (`search_template_integration.py`), generating search terms (`search_term_generator.py`), discovering site structures (`site_discovery.py`), and managing search templates (`template_storage.py`). It also contains subdirectories for pattern analysis and search-related components.
*   **`controllers/`**: Contains the main orchestration logic, primarily the `AdaptiveScraper` (`adaptive_scraper.py`).
*   **`core/`**: Holds fundamental functionalities and services. This includes AI model management (`ai_models.py`), a unified AI service interface (`ai_service.py`), AI caching (`ai_cache.py`), alerting mechanisms (`alerting.py`), batch processing (`batch_processor.py`), circuit breaker implementation for resilience (`circuit_breaker.py`), application configuration (`configuration.py`), content processing (`content_processor.py`), error classification (`error_classifier.py`), HTML service utilities (`html_service.py`), AI model selection (`model_selector.py`), monitoring and metrics (`monitoring.py`), proxy management (`proxy_manager.py`), rate limiting (`rate_limiter.py`), retry management (`retry_manager.py`), a rule engine (`rule_engine.py`), service interfaces (`service_interface.py`), session management (`session_manager.py`), and URL service utilities (`url_service.py`). It also includes subdirectories for circuit breaker and pipeline implementations.
*   **`extraction/`**: Dedicated to content extraction and processing. Modules here handle content analysis (`content_analysis.py`), evaluation (`content_evaluation.py`), core extraction (`content_extraction.py`), normalization (`content_normalizer.py`), quality checks (`content_quality.py`), validation (`content_validation.py`), extraction helpers (`extraction_helpers.py`), fallback mechanisms (`fallback_extraction.py`, `fallback_framework.py`), metadata extraction (`metadata_extractor.py`), pattern-based extraction (`pattern_extractor.py`), pipeline execution (`pipeline_executor.py`), pipeline registry (`pipeline_registry.py`), pipeline service (`pipeline_service.py`), quality evaluator implementation (`quality_evaluator_impl.py`), schema management (`schema_manager.py`), schema validation (`schema_validator.py`), semantic extraction (`semantic_extractor.py`), and structural analysis (`structural_analyzer.py`). It also contains subdirectories for core extraction logic, helpers, and pipeline stages.
*   **`strategies/`**: Contains the implementations for different scraping strategies.
*   **`utils/`**: Provides general utility functions, including HTML parsing (`html_utils.py`), HTTP requests and session management (`http_utils.py`), and data export (`export.py`).
*   **`web/`**: Contains components related to the web interface, including data models and route definitions. This directory likely includes subdirectories for static assets (`static/`) and HTML templates (`templates/`) if a web UI is implemented.

## Important Files

*   **`app.py`:** The main application entry point.
*   **`config.py`:** Configuration settings for the application.
*   **`requirements.txt`:** Project dependencies.
*   **`README.md`:** Project overview and instructions.
*   **`controllers/adaptive_scraper.py`:** Adaptive scraper controller.
*   **`strategies/ai_guided_strategy.py`:** AI-guided scraping strategy.
*   **`strategies/form_strategy.py`:** Form-based scraping strategy.
*   **`strategies/dom_strategy.py`:** DOM-based strategy.
*   **`core/configuration.py`:** Application configuration and resource management settings.
*   **`core/service_registry.py`:** Central registry for application services.
*   **`components/search_term_generator.py`:** Generates optimized search terms.
*   **`components/site_discovery.py`:** Utilities for discovering website structure.
*   **`components/domain_intelligence.py`:** Analyzes web pages for domain and content type.
*   **`extraction/content_extraction.py`:** Enhanced content extraction capabilities.
*   **`extraction/schema_extraction.py`:** Extracts structured data using schema definitions.
*   **`extraction/quality_evaluator.py`:** Evaluates the quality of extracted data.
*   **`core/ai_models.py`:** Manages interactions with various AI models.
*   **`core/ai_service.py`:** Provides a unified interface for AI-related tasks.
*   **`core/circuit_breaker.py`:** Implements the circuit breaker pattern for resilience.
*   **`core/error_classifier.py`:** Classifies different types of scraping errors.
*   **`core/retry_manager.py`:** Manages retry logic for failed requests.
*   **`utils/http_utils.py`:** Provides HTTP utility functions.
*   **`utils/export.py`:** Handles data export to various formats.
*   **`core/pipeline/`:** Pipeline architecture implementation.
*   **`core/ai_cache.py`:** Caching for AI interactions.
*   **`core/alerting.py`:** Alerting mechanisms.
*   **`core/batch_processor.py`:** Batch processing capabilities.
*   **`core/content_processor.py`:** Content processing.
*   **`core/html_service.py`:** HTML service utilities.
*   **`core/model_selector.py`:** Selects AI models.
*   **`core/monitoring.py`:** Monitoring and metrics.
*   **`core/proxy_manager.py`:** Proxy management.
*   **`core/rate_limiter.py`:** Rate limiting.
*   **`core/rule_engine.py`:** Rule engine.
*   **`core/service_interface.py`:** Service interfaces.
*   **`core/session_manager.py`:** Session management.
*   **`core/url_service.py`:** URL service utilities.
*   **`components/pagination_handler.py`:** Handles pagination during scraping.
*   **`components/search_automation.py`:** Automates search interactions.
*   **`components/search_orchestrator.py`:** Orchestrates search processes.
*   **`components/search_template_integration.py`:** Integrates search templates.
*   **`components/template_storage.py`:** Stores and manages search templates.
*   **`extraction/content_analysis.py`:** Analyzes extracted content.
*   **`extraction/content_evaluation.py`:** Evaluates the quality and relevance of content.
*   **`extraction/content_normalizer.py`:** Normalizes extracted content.
*   **`extraction/content_quality.py`:** Defines metrics and checks for content quality.
*   **`extraction/content_validation.py`:** Validates extracted content against rules or schemas.
*   **`extraction/fallback_extraction.py`:** Implements fallback mechanisms for extraction.
*   **`extraction/fallback_framework.py`:** Provides a framework for managing extraction fallbacks.
*   **`extraction/metadata_extractor.py`:** Extracts metadata from web pages.
*   **`extraction/pattern_extractor.py`:** Extracts data based on predefined patterns.
*   **`extraction/pipeline_executor.py`:** Executes extraction pipelines.
*   **`extraction/pipeline_registry.py`:** Registers and manages extraction pipelines.
*   **`extraction/pipeline_service.py`:** Provides a service interface for extraction pipelines.
*   **`extraction/quality_evaluator_impl.py`:** Implementation of the quality evaluator.
*   **`extraction/schema_manager.py`:** Manages extraction schemas.
*   **`extraction/schema_validator.py`:** Validates extracted data against schemas.
*   **`extraction/semantic_extractor.py`:** Extracts semantic information from content.
*   **`extraction/structural_analyzer.py`:** Analyzes the structural layout of web pages.
*   **`utils/html_utils.py`:** Provides utilities for HTML parsing and manipulation.
*   **`utils/http_utils.py`:** Provides utilities for HTTP requests and session management.
*   **`utils/export.py`:** Handles exporting extracted data.

## Scraping Strategies

SmartScrape offers multiple scraping strategies:

*   **Basic page scraping:** Scrapes a single page.
*   **Breadth-first crawling:** Explores all links at each depth level.
*   **Depth-first crawling:** Explores deep paths before moving to siblings.
*   **AI-guided strategy:** Intelligently navigates websites based on user intent. The `AIGuidedStrategy` in `strategies/ai_guided_strategy.py` uses AI models (including Google's Generative AI) to analyze website structure and develop dynamic search strategies.
*   **Form-based strategy:** Interacts with HTML forms to perform searches. The `FormSearchEngine` in `strategies/form_strategy.py` detects search forms, fills them out, and submits them. It can handle pagination and uses browser automation for JavaScript-heavy forms.
*   **DOM-based strategy:** Uses DOM manipulation and browser automation (via Playwright) to extract information from web pages. The `DOMStrategy` in `strategies/dom_strategy.py` can handle dynamic content and JavaScript-heavy sites.
*   **URL Parameter Strategy:** Scrapes data by manipulating URL parameters.
*   **API Strategy:** Interacts with website APIs to retrieve data.
*   **Search Term Generator:** The `SearchTermGenerator` in `components/search_term_generator.py` generates optimized search terms based on user intent and website characteristics. It uses rule-based approaches and AI to create effective search queries.
*   **Site Discovery:** The `SiteDiscovery` class in `components/site_discovery.py` provides utilities for discovering website structure, sitemaps, and other key features of websites to enable intelligent crawling.
*   **Domain Intelligence:** The `DomainIntelligence` class in `components/domain_intelligence.py` analyzes web pages to determine their domain and content type. This information is used to optimize scraping strategies and extraction techniques.
*   **AI Helpers:** The `ai_helpers` directory contains helper functions for AI-related tasks. The `IntentParser` in `ai_helpers/intent_parser.py` parses natural language queries into structured intents. The `PromptGenerator` in `ai_helpers/prompt_generator.py` generates optimized prompts for various AI tasks.
*   **Domain Intelligence:** The `DomainIntelligence` class in `components/domain_intelligence.py` analyzes web pages to determine their domain and content type. This information is used to optimize scraping strategies and extraction techniques.
*   **Extraction Modules:** The `extraction` directory contains modules for content extraction, analysis, and normalization. The `ContentExtractor` in `extraction/content_extraction.py` provides enhanced content extraction capabilities using multiple libraries and techniques, including Beautiful Soup for HTML parsing. The `SchemaExtractor` in `extraction/schema_extraction.py` extracts structured data from web pages using schema definitions. The `ExtractedDataQualityEvaluator` in `extraction/quality_evaluator.py` evaluates the quality of extracted data, including completeness, confidence, relevance, and anomaly detection.
*   **Utilities:** The `utils` directory contains various utility modules. The `html_utils.py` file provides optimized HTML parsing and processing tools. The `http_utils.py` file provides HTTP utilities for making requests, managing sessions, and handling cookies. The `export.py` file provides utilities for exporting extracted data to various formats.

## Resource Management

SmartScrape includes robust resource management capabilities to handle various aspects of web scraping efficiently and reliably. These services are configured via `core/configuration.py` and managed by the `ServiceRegistry`.

*   **Session Management:** Manages HTTP sessions, cookies, and connection pooling.
*   **Rate Limiting:** Controls the rate of requests to avoid overwhelming target servers.
*   **Proxy Management:** Handles proxy rotation and management for distributed scraping.
*   **Error Classification:** Categorizes different types of scraping errors for targeted handling.
*   **Retry Management:** Implements logic for retrying failed requests based on error types and policies.
*   **Circuit Breaking:** Prevents repeated requests to domains that are consistently failing.

## Pipeline Architecture

SmartScrape includes a powerful pipeline architecture for building modular, customizable data processing workflows.

### Key Features

*   **Modular Design:** Compose pipelines from reusable, single-purpose stages.
*   **Configuration-Driven:** Create and modify pipelines using simple configuration files.
*   **Parallel Execution:** Process independent stages concurrently for better performance.
*   **Error Handling:** Comprehensive error handling with recovery mechanisms.
*   **Monitoring & Metrics:** Track execution progress and performance metrics.
*   **Extensibility:** Create custom stages for specialized processing needs.

### Core Components

*   **Pipeline:** Orchestrates the execution of multiple stages.
*   **Stages:** Individual processing units that perform specific operations.
*   **Context:** Shared state object that flows through the pipeline.
*   **Registry:** Catalog of available stages and pipeline templates.

## Testing and Examples

*   **`tests/`**: Contains unit and integration tests for various components and strategies. Running these tests is crucial for verifying functionality and stability.
*   **`examples/`**: Provides example scripts demonstrating how to use different parts of the SmartScrape library, including error handling, extraction, resource management, and strategies.
*   **`test_results/`**: Directory for storing test execution results and reports.
*   **`test_env/`**: Contains files and configurations specific to the testing environment.
*   **`metrics/`**: Directory for storing metrics data collected during runs.

## Project Management and Development

*   **`CHANGELOG.md`**: Tracks changes and version history of the project.
*   **`requirements.txt`**: Lists the project's dependencies.
*   **`README.md`**: Provides a high-level overview and setup instructions.
