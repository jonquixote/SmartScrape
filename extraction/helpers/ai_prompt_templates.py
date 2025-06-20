"""
AI Prompt Templates for Extraction

This module provides optimized prompt templates for various extraction tasks.
These templates are designed to guide AI models in extracting specific types
of content from web pages efficiently and accurately.
"""

import json
from typing import Dict, Any, List, Optional, Union

# General extraction prompt for versatile content extraction
GENERAL_EXTRACTION_PROMPT = """
You are a specialized data extraction system. Extract the most relevant information from the provided HTML content.
Focus on the main content area and extract data according to the schema.

HTML CONTENT:
{content}

EXTRACTION SCHEMA:
{schema}

Extract the data in a structured JSON format matching the schema exactly. Only include fields from the schema.
For missing data, use null values. Ensure all text is properly cleaned with no HTML artifacts.
"""

# Product extraction prompt optimized for e-commerce pages
PRODUCT_EXTRACTION_PROMPT = """
You are a specialized product data extraction system. Extract product information from the provided HTML content.

HTML CONTENT:
{content}

Extract the following product details in JSON format:
- name: The product name/title
- price: The current price (include currency symbol)
- original_price: Original price if discounted (null if not applicable)
- description: Product description (limit to 1000 chars)
- features: List of product features/specifications as key-value pairs
- images: List of image URLs
- ratings: Average product rating (numerical, null if not found)
- review_count: Number of reviews (null if not found)
- availability: Stock status (in_stock, out_of_stock, etc.)
- sku: Product SKU/ID
- brand: Brand name
- category: Product category/categories
- variants: Available variants (colors, sizes, etc.)

Only include fields where data is found. Keep text fields concise. Ensure all prices are properly formatted.
"""

# Article extraction prompt for blog posts, news articles, etc.
ARTICLE_EXTRACTION_PROMPT = """
You are a specialized article content extraction system. Extract key components from the provided HTML content.

HTML CONTENT:
{content}

Extract the following article details in JSON format:
- title: The article title
- author: Author name(s)
- published_date: Publication date
- modified_date: Last modified date (if available)
- categories: Article categories or topics
- tags: Article tags
- summary: A brief summary (150 words max)
- content: The main article content, divided into sections:
  - sections: List of sections, each with:
    - heading: Section heading
    - content: Section text content
- images: List of images with URL and caption

Maintain the article's structure with proper paragraphs. Exclude navigation, ads, footers, and sidebars.
Focus only on the main article content.
"""

# List extraction prompt for product listings, search results, etc.
LIST_EXTRACTION_PROMPT = """
You are a specialized list extraction system. Extract items from the list in the provided HTML content.

HTML CONTENT:
{content}

SCHEMA FOR EACH ITEM:
{item_schema}

Identify and extract all items in this list/grid. For each item, extract data according to the provided schema.
Return a JSON object with:
- list_title: Title of the list (if available)
- list_description: Description of the list (if available)
- item_count: Number of items found
- items: Array of extracted items (limit to {max_items} items)
- pagination: Pagination information if available (current_page, total_pages, next_page_url, prev_page_url)

Focus on consistent extraction across all items. If an item is missing attributes, use null values.
"""

# Tabular data extraction prompt for HTML tables
TABULAR_DATA_PROMPT = """
You are a specialized table data extraction system. Extract data from tables in the provided HTML content.

HTML CONTENT:
{content}

Identify and extract data from all relevant tables. For each table:
1. Identify the table structure (headers and data rows)
2. Extract headers as keys
3. Extract each row as an object with values mapped to the headers

Return a JSON object with:
- table_count: Number of tables found
- tables: Array of extracted tables, each with:
  - caption: Table caption or title (if available)
  - headers: List of column headers
  - rows: List of data rows (as objects mapping headers to values)
  - raw_data: 2D array of all cells (including headers)

Ensure data types are appropriate (numbers as numbers, text as strings). 
Clean cell values by removing excess whitespace and unnecessary formatting.
"""

# Metadata extraction prompt for page-level metadata
METADATA_EXTRACTION_PROMPT = """
You are a specialized metadata extraction system. Extract metadata from the provided HTML content.

HTML CONTENT:
{content}

Extract the following metadata in JSON format:
- title: Page title
- description: Meta description
- canonical_url: Canonical URL
- language: Page language
- robots: Robots meta directives
- og_data: All Open Graph metadata (og:type, og:title, etc.)
- twitter_data: All Twitter card metadata
- json_ld: Structured data in JSON-LD format
- schema_org: Other schema.org metadata
- hreflang: Language alternatives
- favicon: Favicon URL
- viewport: Viewport settings
- author: Page author
- published_date: Publication date
- modified_date: Last modified date

Only include fields where data is found. Parse JSON-LD carefully to maintain its structure.
"""

# Schema-guided extraction prompt for flexible extraction based on provided schema
SCHEMA_GUIDED_EXTRACTION_PROMPT = """
You are a specialized data extraction system. Extract information according to the provided schema.

HTML CONTENT:
{content}

EXTRACTION SCHEMA:
{schema}

Instructions:
1. Identify content on the page matching each field in the schema
2. Extract data according to the field types and descriptions
3. Return data in a structured JSON format matching the schema exactly
4. For missing data, use null values
5. For array fields, extract all matching items
6. For nested objects, maintain the hierarchy

Follow the schema precisely. Only extract fields defined in the schema.
Ensure all text is properly cleaned with no HTML artifacts.
"""

# Detailed extraction prompt for deep content analysis
DETAILED_EXTRACTION_PROMPT = """
You are an advanced content extraction system. Perform a deep analysis of the HTML content.

HTML CONTENT:
{content}

EXTRACTION TARGETS:
{targets}

Perform a comprehensive extraction and analysis:
1. Identify all content matching the specified targets
2. Extract both explicit data (directly stated) and implicit data (implied)
3. Analyze relationships between different content elements
4. Identify the most salient information for each target
5. Extract contextual information that supports the main content

Return a detailed JSON structure with:
- main_content: The primary content matching the targets
- supporting_content: Related information that provides context
- metadata: Any metadata related to the content
- relationships: Connections between different content elements
- confidence: Confidence level for each extracted element (0-1)

Provide thorough, detailed extraction while maintaining accuracy.
"""

# New: Semantic understanding prompt for context-aware extraction
SEMANTIC_UNDERSTANDING_PROMPT = """
You are an advanced semantic extraction system with deep language understanding capabilities.
Your task is to extract meaningful information from content while preserving semantic relationships and context.

CONTENT:
{content}

EXTRACTION GOALS:
{goals}

Go beyond simple pattern matching and use your understanding of language to:
1. Identify the core message and key points in the content
2. Recognize implicit information and logical connections
3. Distinguish between facts, opinions, and speculations
4. Capture the relative importance of different information elements
5. Understand content in context of its broader topic

Return a structured JSON object containing:
- main_topics: List of primary topics discussed
- key_points: List of important assertions or claims
- entities: Named entities with their descriptions and relationships
- concepts: Abstract concepts referenced in the content
- sentiment: Overall sentiment and emotional tone
- context: Broader context that helps frame the content
- factual_claims: Statements presented as facts
- inferences: Logical conclusions that can be drawn from the content

Prioritize accuracy and contextual understanding in your extraction.
"""

# New: Document structure analysis prompt
DOCUMENT_STRUCTURE_PROMPT = """
You are a document structure analysis system. Analyze the structure and organization of the provided document.

DOCUMENT CONTENT:
{content}

Analyze the document's structure and organization to extract:
1. Document hierarchy (sections, subsections, etc.)
2. Logical flow and information architecture
3. Content organization patterns
4. Information density across different sections
5. Content relationships and cross-references

Return a JSON structure with:
- document_type: Type of document (article, report, etc.)
- structure: Hierarchical representation of document structure
- sections: List of identified sections with:
  - title: Section title
  - level: Heading level (1 for main headings, 2 for subheadings, etc.)
  - content_summary: Brief summary of section content
  - key_points: Main points in the section
  - word_count: Approximate word count
- navigation: Cross-reference and navigation elements
- key_structural_elements: Important structural components (TOC, appendices, etc.)

Focus on understanding how information is organized and presented.
"""

# New: Knowledge extraction prompt for complex content
KNOWLEDGE_EXTRACTION_PROMPT = """
You are a specialized knowledge extraction system. Extract structured knowledge from complex content.

CONTENT:
{content}

DOMAIN CONTEXT:
{domain}

Extract knowledge elements including:
1. Core concepts and their definitions
2. Relationships between concepts (hierarchical, causal, temporal, etc.)
3. Properties and attributes of identified concepts
4. Processes and workflows
5. Conditions and constraints
6. Examples and instances

Return a structured knowledge representation as JSON:
- concepts: Map of concept names to definitions and properties
- relationships: List of relationships between concepts
- processes: Sequential or conditional steps in identified processes
- attributes: Properties and their possible values
- constraints: Limitations and conditions
- examples: Illustrative instances of concepts or processes
- taxonomy: Hierarchical organization of domain concepts

Prioritize precision and semantic accuracy in knowledge representation.
"""

# New: Competitive intelligence extraction prompt
COMPETITIVE_INTELLIGENCE_PROMPT = """
You are a competitive intelligence extraction system. Extract business and market intelligence from the provided content.

CONTENT:
{content}

INDUSTRY CONTEXT:
{industry}

Extract the following competitive intelligence elements:
1. Company information (name, location, size, etc.)
2. Product/service details (features, pricing, positioning)
3. Market positioning and value propositions
4. Competitive advantages mentioned
5. Target customer segments
6. Business model indicators
7. Strategic initiatives and future plans
8. Market trends and industry insights
9. Partnerships and ecosystem relationships

Return a structured JSON with:
- company_profile: Basic company information
- offerings: Products/services with their attributes
- positioning: Market positioning and messaging
- advantages: Stated competitive advantages
- target_market: Described customer segments
- business_model: Indicators of business model
- strategy: Future plans and strategic directions
- market_insights: Industry trends and observations
- relationships: Partnerships and ecosystem connections

Focus on extracting actionable intelligence from the content.
"""

# Function to customize a prompt with specific parameters
def customize_prompt(prompt_template: str, **kwargs) -> str:
    """
    Customize a prompt template with specific parameters.
    
    Args:
        prompt_template: The template to customize
        **kwargs: Key-value pairs to insert into the template
        
    Returns:
        Customized prompt string
    """
    return prompt_template.format(**kwargs)

# Function to select the most appropriate prompt for a given content type
def select_prompt_for_content(content_type: str, custom_schema: Optional[Dict[str, Any]] = None) -> str:
    """
    Select the most appropriate prompt template for a given content type.
    
    Args:
        content_type: Type of content (product, article, list, table, etc.)
        custom_schema: Optional custom schema to use
        
    Returns:
        Selected prompt template
    """
    prompt_mapping = {
        "product": PRODUCT_EXTRACTION_PROMPT,
        "article": ARTICLE_EXTRACTION_PROMPT,
        "list": LIST_EXTRACTION_PROMPT,
        "table": TABULAR_DATA_PROMPT,
        "metadata": METADATA_EXTRACTION_PROMPT,
        "general": GENERAL_EXTRACTION_PROMPT,
        "schema": SCHEMA_GUIDED_EXTRACTION_PROMPT,
        "detailed": DETAILED_EXTRACTION_PROMPT,
        "semantic": SEMANTIC_UNDERSTANDING_PROMPT,
        "document": DOCUMENT_STRUCTURE_PROMPT,
        "knowledge": KNOWLEDGE_EXTRACTION_PROMPT,
        "competitive": COMPETITIVE_INTELLIGENCE_PROMPT
    }
    
    # Default to general extraction if content type not recognized
    return prompt_mapping.get(content_type.lower(), GENERAL_EXTRACTION_PROMPT)

# Function to generate a schema-specific prompt
def generate_schema_prompt(schema: Dict[str, Any], content: str) -> str:
    """
    Generate a prompt tailored to a specific schema.
    
    Args:
        schema: Data schema to extract
        content: HTML content to process
        
    Returns:
        Custom prompt for the schema
    """
    # Start with the schema-guided template
    prompt_template = SCHEMA_GUIDED_EXTRACTION_PROMPT
    
    # Customize based on schema complexity
    if isinstance(schema, dict):
        if schema.get("_type") == "product":
            prompt_template = PRODUCT_EXTRACTION_PROMPT
        elif schema.get("_type") == "article":
            prompt_template = ARTICLE_EXTRACTION_PROMPT
        elif schema.get("_type") == "list":
            prompt_template = LIST_EXTRACTION_PROMPT
        elif schema.get("_type") == "semantic":
            prompt_template = SEMANTIC_UNDERSTANDING_PROMPT
        
    # Format the schema as a string for inclusion in the prompt
    schema_str = json.dumps(schema, indent=2) if isinstance(schema, dict) else str(schema)
    
    return customize_prompt(prompt_template, content=content, schema=schema_str)

# Function to optimize prompts based on model capabilities
def optimize_prompt_for_model(prompt: str, model_name: str) -> str:
    """
    Optimize a prompt for a specific AI model.
    
    Args:
        prompt: The prompt to optimize
        model_name: Name of the AI model (e.g., "gemini-1.5-flash", "gpt-4")
        
    Returns:
        Optimized prompt for the specific model
    """
    # Base optimization - trim excess whitespace and normalize
    optimized = "\n".join(line.strip() for line in prompt.split("\n")).strip()
    
    # Model-specific optimizations
    if model_name.startswith("gemini-1.5-flash"):
        # Gemini models are generally robust; specific optimizations can be added if needed
        return optimized
    elif model_name.startswith("gpt-4"): # Keep gpt-4 for now, in case it's used elsewhere or for comparison
        # GPT-4 can handle more complex instructions
        return optimized
    elif model_name.startswith("gpt-3.5"): # Keep gpt-3.5 for now
        # For GPT-3.5, simplify instructions and be more direct
        optimized = optimized.replace("Perform a comprehensive extraction and analysis", 
                              "Extract the following information")
        optimized = optimized.replace("structured JSON format matching the schema exactly", 
                              "JSON format following this structure")
    
    return optimized

# Function to generate refined extraction prompt when initial results are incomplete
def generate_refinement_prompt(initial_result: Dict[str, Any], content: str, 
                             missing_fields: List[str]) -> str:
    """
    Generate a prompt to refine extraction for missing or incomplete fields.
    
    Args:
        initial_result: The initial extraction result
        content: The content to extract from
        missing_fields: List of fields that need improvement
        
    Returns:
        Refinement prompt
    """
    refinement_prompt = """
    You previously extracted information from content, but some fields need improvement.
    
    ORIGINAL CONTENT:
    {content}
    
    CURRENT EXTRACTION:
    {current_extraction}
    
    FIELDS TO IMPROVE:
    {fields_to_improve}
    
    Please re-examine the content and provide improved extraction ONLY for the specified fields.
    Focus specifically on finding accurate data for these fields. Return the results as a JSON object
    containing only the improved fields.
    """
    
    return customize_prompt(
        refinement_prompt,
        content=content,
        current_extraction=json.dumps(initial_result, indent=2),
        fields_to_improve="\n".join(f"- {field}" for field in missing_fields)
    )