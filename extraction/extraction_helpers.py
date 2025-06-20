"""
Extraction Helpers Module

Core functions for extraction strategy building and result processing.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from io import BytesIO

import pandas as pd
from bs4 import BeautifulSoup
import google.generativeai as genai

from crawl4ai import (
    AsyncWebCrawler, 
    CacheMode, 
    BrowserConfig,
    CrawlerRunConfig
)
from crawl4ai.extraction_strategy import (
    LLMExtractionStrategy,
    JsonCssExtractionStrategy
)
from crawl4ai.async_configs import LLMConfig

from components.site_discovery import SiteDiscovery
# Removing circular imports - will import inside functions where needed
# from components.search_automation import SearchFormDetector
# from components.domain_intelligence import DomainIntelligence
from components.pagination_handler import PaginationHandler
from components.template_storage import TemplateStorage

from ai_helpers.prompt_generator import (
    optimize_extraction_prompt, 
    generate_content_filter_instructions
)
from ai_helpers.response_parser import clean_extraction_output, analyze_extraction_result

# Import service registry instead of direct URL filters
from core.service_registry import ServiceRegistry

from strategies.base_strategy import get_crawl_strategy

# Remove this import to fix circular dependency
# from web.routes import jobs
# We'll get the jobs object within the functions that need it

import config

async def create_dynamic_extraction_strategy(
    site_analysis: Dict[str, Any],
    user_request: str,
    html_content: Optional[str] = None,
    ai_model_quality: str = "fast"
) -> Union[LLMExtractionStrategy, JsonCssExtractionStrategy]:
    """
    Creates the optimal extraction strategy based on site analysis results.
    
    Args:
        site_analysis: Dict with site analysis results
        user_request: The user's extraction request
        html_content: HTML content for additional analysis if needed
        ai_model_quality: AI model quality (fast, premium, basic)
        
    Returns:
        An appropriate extraction strategy object for the specific site
    """
    # Default to LLM strategy with optimized prompt if analysis failed
    if "analysis_error" in site_analysis:
        print(f"Site analysis had errors, using default LLM strategy: {site_analysis['analysis_error']}")
        return LLMExtractionStrategy(
            llm_config=LLMConfig(
                provider="gemini/gemini-2.0-flash-lite",
                api_token=config.GEMINI_API_KEY
            ),
            extraction_type="raw",
            instruction=f"Extract the following information from this webpage: {user_request}",
            chunk_size=4000,
            apply_chunking=True
        )
    
    # Generate an optimized extraction prompt based on the site analysis
    optimized_prompt = f"""
    Based on this {site_analysis['site_type']} website, extract:
    {user_request}
    
    Focus on content within {site_analysis['main_content_selector']} elements.
    """
    
    # For highly structured content, use CSS extraction when possible
    if site_analysis.get('content_structure', 0) >= 8 and site_analysis.get('recommended_extraction_type') == 'css':
        # First try JSON CSS extraction if we have good selectors
        if site_analysis.get('recommended_selectors') and len(site_analysis['recommended_selectors']) > 0:
            # Build schema for extraction
            schema = {
                "name": "Extracted Content",
                "baseSelector": site_analysis.get('item_container_selector') or site_analysis['main_content_selector'],
                "fields": []
            }
            
            # Create strongly-typed fields from the recommended selectors
            for i, selector_info in enumerate(site_analysis['recommended_selectors']):
                if isinstance(selector_info, str):
                    # Simple selector string
                    field_name = f"item_{i+1}"
                    selector = selector_info
                    field_type = "text"
                else:
                    # Object with field information
                    field_name = selector_info.get('name', f"item_{i+1}")
                    selector = selector_info.get('selector', '')
                    field_type = selector_info.get('type', 'text')
                
                # Only add the field if we have a valid selector
                if selector:
                    field = {
                        "name": field_name,
                        "selector": selector,
                        "type": field_type
                    }
                    # Add attribute property for non-text fields
                    if field_type == "attribute" and selector_info.get('attribute'):
                        field["attribute"] = selector_info['attribute']
                    
                    schema["fields"].append(field)
            
            # If we have valid fields, use JsonCssExtractionStrategy
            if len(schema["fields"]) > 0:
                print(f"Using JsonCssExtractionStrategy with {len(schema['fields'])} fields")
                return JsonCssExtractionStrategy(schema=schema)
    
    # For JavaScript-dependent sites or complex structures, use structured LLM extraction
    if site_analysis.get('js_dependent', False) or site_analysis.get('recommended_extraction_type') == 'hybrid':
        # Build a schema based on the user request
        instruction = f"""
        Based on this {site_analysis['site_type']} website and the request: 
        "{user_request}"
        
        Create a JSON schema that represents the data structure to extract.
        Return only the JSON schema with no explanation.
        """
        
        # Get schema from LLM
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(instruction)
            
            schema_text = response.text
            # Extract JSON if it's wrapped in code blocks
            if "```json" in schema_text:
                schema_text = schema_text.split("```json")[1].split("```")[0].strip()
            elif "```" in schema_text:
                schema_text = schema_text.split("```")[1].strip()
                
            # Parse the schema
            schema = json.loads(schema_text)
            
            # Use structured LLM extraction with the schema
            print("Using structured LLM extraction with AI-generated schema")
            return LLMExtractionStrategy(
                llm_config=LLMConfig(
                    provider="gemini/gemini-2.0-flash-lite",
                    api_token=config.GEMINI_API_KEY
                ),
                extraction_type="structured",
                schema=schema,
                instruction=optimized_prompt,
                chunk_size=4000,
                apply_chunking=True
            )
        except Exception as e:
            print(f"Failed to create structured extraction: {str(e)}")
            # Fall through to raw extraction
    
    # Default to raw LLM extraction for unstructured content or if other methods failed
    print("Using raw LLM extraction")
    return LLMExtractionStrategy(
        llm_config=LLMConfig(
            provider="gemini/gemini-2.0-flash-lite",
            api_token=config.GEMINI_API_KEY
        ),
        extraction_type="raw",
        instruction=optimized_prompt,
        chunk_size=4000,
        apply_chunking=True
    )


def process_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process and normalize the scraped data.
    
    Args:
        results: Raw extraction results
        
    Returns:
        Processed and normalized results
    """
    processed_data = []
    
    for result in results:
        # Handle list results directly
        if isinstance(result, list):
            for item in result:
                if isinstance(item, dict):
                    processed_data.append({
                        "source_url": item.get("source_url", "Unknown URL"),
                        "depth": item.get("depth", 0),
                        "score": item.get("score", 0),
                        "data": item  # Use the entire item as data if it doesn't have a "data" field
                               if "data" not in item else item.get("data", {})
                    })
        # Handle dictionary results
        elif isinstance(result, dict):
            if "data" in result:
                # Normal case where data is provided
                processed_data.append({
                    "source_url": result.get("source_url", "Unknown URL"),
                    "depth": result.get("depth", 0),
                    "score": result.get("score", 0),
                    "data": result.get("data", {})
                })
            else:
                # Case where the result itself is the data
                processed_data.append({
                    "source_url": result.get("source_url", "Unknown URL"),
                    "depth": result.get("depth", 0),
                    "score": result.get("score", 0),
                    "data": result  # Use the entire result as data
                })
        # Handle object-based results
        elif hasattr(result, 'data'):
            processed_data.append({
                "source_url": getattr(result, 'url', "Unknown URL"),
                "depth": getattr(result, 'depth', 0),
                "score": getattr(result, 'score', 0),
                "data": result.data
            })
    
    return processed_data


def generate_json_export(data: List[Dict[str, Any]]) -> str:
    """
    Generate a JSON file from the data.
    
    Args:
        data: The data to export
        
    Returns:
        JSON string representation
    """
    return json.dumps(data, indent=2)


def generate_csv_export(data: List[Dict[str, Any]]) -> str:
    """
    Generate a CSV file from the data.
    
    Args:
        data: The data to export
        
    Returns:
        CSV string representation
    """
    # This is a more sophisticated flattening function for nested data
    flattened_data = []
    
    for item in data:
        base_item = {
            "source_url": item["source_url"],
            "depth": item["depth"],
            "score": item["score"]
        }
        
        # Handle different data types
        if isinstance(item["data"], list):
            # For list data, create separate rows for each item
            for i, subitem in enumerate(item["data"]):
                row = base_item.copy()
                if isinstance(subitem, dict):
                    # Flatten dict items
                    for key, value in subitem.items():
                        # Handle nested values by converting to string
                        if isinstance(value, (dict, list)):
                            row[key] = json.dumps(value)
                        else:
                            row[key] = value
                else:
                    row[f"item_{i}"] = subitem
                flattened_data.append(row)
        elif isinstance(item["data"], dict):
            # For dict data, flatten keys into columns
            row = base_item.copy()
            for key, value in item["data"].items():
                # Handle nested values by converting to string
                if isinstance(value, (dict, list)):
                    row[key] = json.dumps(value)
                else:
                    row[key] = value
            flattened_data.append(row)
        else:
            # For scalar data
            row = base_item.copy()
            row["data"] = item["data"]
            flattened_data.append(row)
    
    # If no data, return an empty CSV
    if not flattened_data:
        return "source_url,depth,score,data\n"
    
    # Convert to DataFrame and then to CSV
    try:
        df = pd.DataFrame(flattened_data)
        return df.to_csv(index=False)
    except Exception as e:
        print(f"Error generating CSV: {e}")
        # Fallback to basic CSV
        return "source_url,depth,score,data\n" + "\n".join([
            f"{item['source_url']},{item['depth']},{item['score']},{json.dumps(item['data'])}"
            for item in data
        ])


def generate_excel_export(data: List[Dict[str, Any]]) -> bytes:
    """
    Generate an Excel file from the data.
    
    Args:
        data: The data to export
        
    Returns:
        Excel file as bytes
    """
    # Use the same flattened data approach as CSV
    flattened_data = []
    
    for item in data:
        base_item = {
            "source_url": item["source_url"],
            "depth": item["depth"],
            "score": item["score"]
        }
        
        # Handle different data types
        if isinstance(item["data"], list):
            for i, subitem in enumerate(item["data"]):
                row = base_item.copy()
                if isinstance(subitem, dict):
                    for key, value in subitem.items():
                        if isinstance(value, (dict, list)):
                            row[key] = json.dumps(value)
                        else:
                            row[key] = value
                else:
                    row[f"item_{i}"] = subitem
                flattened_data.append(row)
        elif isinstance(item["data"], dict):
            row = base_item.copy()
            for key, value in item["data"].items():
                if isinstance(value, (dict, list)):
                    row[key] = json.dumps(value)
                else:
                    row[key] = value
            flattened_data.append(row)
        else:
            row = base_item.copy()
            row["data"] = item["data"]
            flattened_data.append(row)
    
    # Convert to DataFrame and use BytesIO for in-memory Excel file
    try:
        df = pd.DataFrame(flattened_data)
        excel_buffer = BytesIO()
        df.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_buffer.seek(0)
        return excel_buffer.getvalue()
    except Exception as e:
        print(f"Error generating Excel: {e}")
        # Return empty Excel file in case of error
        df = pd.DataFrame({"error": ["Failed to generate Excel file"]})
        excel_buffer = BytesIO()
        df.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_buffer.seek(0)
        return excel_buffer.getvalue()


async def scrape_website(
    job_id, 
    url, 
    strategy_type, 
    max_depth, 
    include_external, 
    extract_description, 
    crawl_entire_site, 
    keywords=None,
    # AI-guided strategy parameters
    ai_exploration_ratio=0.5,
    ai_use_sitemap=True, 
    ai_use_search=True,
    ai_consolidate_results=True,
    ai_model_quality="fast",
    **kwargs
):
    """
    Enhanced scrape_website function with intelligent search capabilities and template reuse
    
    Args:
        job_id: Unique identifier for this job
        url: URL to scrape
        strategy_type: Crawling strategy (ai_guided, bfs, dfs, best_first)
        max_depth: Maximum crawling depth
        include_external: Whether to include external links
        extract_description: Description of what to extract
        crawl_entire_site: Whether to crawl the entire site
        keywords: Optional keywords for best-first strategy
        ai_exploration_ratio: Balance between exploration (1.0) and exploitation (0.0)
        ai_use_sitemap: Whether to use sitemap discovery
        ai_use_search: Whether to use search form detection
        ai_consolidate_results: Whether to consolidate results from multiple sources
        ai_model_quality: AI model quality (fast, premium, basic)
        **kwargs: Additional parameters
    """
    try:
        # Import jobs here to avoid circular import
        from web.routes import jobs
        
        # Format URL if needed (add https:// if missing)
        url = normalize_url(url)
        
        # Get all request parameters from kwargs or use defaults
        request_params = jobs[job_id].get("parameters", {})
        
        # Update job status
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["progress"] = 0.1
        
        # Configure browser based on settings
        browser_config = create_browser_config(request_params)
        
        # Initialize our intelligent components - Move SearchFormDetector import inside the function to break circular import
        from components.search_automation import SearchFormDetector
        
        site_discovery = SiteDiscovery()
        search_detector = SearchFormDetector()
        from components.domain_intelligence import DomainIntelligence
        domain_intelligence = DomainIntelligence()
        pagination_handler = PaginationHandler()
        template_storage = TemplateStorage("./extraction_strategies")
        
        # Create crawler with specified configurations
        cache_mode = get_cache_mode(request_params.get("cache_mode", "memory"))

        # Initialize the result consolidator if using AI-guided strategy
        result_consolidator = None
        if strategy_type == "ai_guided" and ai_consolidate_results:
            from strategies.result_consolidation import ResultConsolidator
            result_consolidator = ResultConsolidator()
            print("Using ResultConsolidator for AI-guided strategy")
        
        # Set the AI model based on quality parameter for AI-guided strategy
        ai_model = "gemini-2.0-flash"  # Default (standard)
        if strategy_type == "ai_guided":
            if ai_model_quality == "premium":
                ai_model = "gemini-2.0-pro"
                print("Using premium AI model (Gemini-2.0-Pro)")
            elif ai_model_quality == "basic":
                ai_model = "gemini-1.5-flash"
                print("Using basic AI model (Gemini-1.5-Flash)")
                
            # Store the model choice in job metadata
            jobs[job_id]["ai_model_used"] = ai_model
        
        async with AsyncWebCrawler(
            config=browser_config, 
            cache_mode=cache_mode,
            cache_directory="./cache" if request_params.get("cache_mode") == "disk" else None
        ) as crawler:
            # First fetch to get page sample
            sample_result = await crawler.arun(
                url=url,
                config=CrawlerRunConfig(cache_mode=cache_mode)
            )
            
            # Update progress
            jobs[job_id]["progress"] = 0.15
            
            # Get the page content
            page_sample = sample_result.html if sample_result.html else sample_result.markdown
            
            # Try to find a matching extraction template
            matching_template = await template_storage.find_matching_template(url)
            
            if matching_template:
                print(f"Found matching template for {url}, applying it")
                jobs[job_id]["using_template"] = True
                
                template_result = await template_storage.apply_template(
                    url, matching_template, crawler
                )
                
                # If template extraction succeeded, we can return early
                if template_result and template_result.get("success"):
                    print("Template-based extraction succeeded")
                    jobs[job_id]["results"] = [{
                        "source_url": url,
                        "depth": 0,
                        "score": 1.0,
                        "data": template_result.get("data"),
                    }]
                    jobs[job_id]["extraction_method"] = "template"
                    complete_job(job_id)
                    return
            
            # Check for sitemap to discover content (only if ai_use_sitemap is enabled)
            sitemap_info = None
            if strategy_type == "ai_guided" and ai_use_sitemap:
                print("Checking for sitemap to discover content...")
                sitemap_info = await site_discovery.find_sitemap(url, page_sample)
                
                if sitemap_info and sitemap_info.get("sitemap_found"):
                    handle_sitemap_discovery(job_id, sitemap_info, extract_description, domain_intelligence, site_discovery)
            
            # Check if we need to perform a search based on the request (only if ai_use_search is enabled)
            is_search_request = False
            if strategy_type == "ai_guided" and ai_use_search:
                is_search_request = domain_intelligence.is_search_oriented_request(extract_description)
                
                if is_search_request:
                    await handle_search_request(
                        job_id, url, page_sample, extract_description, 
                        domain_intelligence, search_detector
                    )
            
            # Perform dynamic site structure analysis
            print(f"Performing site structure analysis for {url}...")
            site_analysis = await analyze_site(url, page_sample, extract_description, domain_intelligence, ai_model_quality)
            
            # Store site analysis in job metadata
            jobs[job_id]["site_analysis"] = site_analysis
            
            # Update progress
            jobs[job_id]["progress"] = 0.2
            
            # Generate dynamic content filtering instructions
            filter_instructions = await generate_content_filter_instructions(
                url, extract_description, page_sample, model_name=ai_model
            )
            
            # Store filtering instructions in job metadata
            jobs[job_id]["filter_instructions"] = filter_instructions
            
            # Use optimize_extraction_prompt for AI-guided extraction
            ai_suggestions = await optimize_extraction_prompt(extract_description, page_sample, model_name=ai_model)
            
            # Store AI suggestions in job metadata
            jobs[job_id]["ai_suggestions"] = ai_suggestions
            
            # Update progress
            jobs[job_id]["progress"] = 0.3
            
            # Create the appropriate extraction strategy
            extraction_strategy = await create_dynamic_extraction_strategy(
                site_analysis, extract_description, page_sample, ai_model_quality
            )
            
            # Define target URLs for extraction based on collected info
            target_urls = []
            
            # For AI-guided strategy, use a different approach
            if strategy_type == "ai_guided":
                print("Using AI-guided strategy for crawling and extraction")
                
                # Import the AI-guided strategy
                from strategies.ai_guided_strategy import AIGuidedStrategy
                
                # Create and configure the AI-guided strategy
                ai_strategy = AIGuidedStrategy(
                    max_depth=max_depth,
                    max_pages=request_params.get("max_pages", 100),
                    include_external=include_external,
                    user_prompt=extract_description
                )
                
                # Set the exploration/exploitation balance
                ai_strategy.exploration_ratio = ai_exploration_ratio
                
                # Set up specialized crawl config for AI-guided strategy
                ai_crawl_config = CrawlerRunConfig(
                    extraction_strategy=extraction_strategy,
                    cache_mode=cache_mode,
                    verbose=True
                )
                
                # Execute the AI-guided strategy
                print(f"Starting AI-guided crawl with exploration ratio: {ai_exploration_ratio}")
                ai_results = await ai_strategy.execute_async(crawler, url, ai_crawl_config)
                
                # Store the AI crawl results
                if ai_results:
                    # If result consolidation is enabled, consolidate results
                    if ai_consolidate_results and result_consolidator:
                        print("Consolidating AI-guided strategy results")
                        consolidated_results = await result_consolidator.consolidate(
                            ai_results.get("results", []), 
                            extract_description,
                            model_name=ai_model
                        )
                        
                        jobs[job_id]["results"] = consolidated_results
                        jobs[job_id]["crawl_info"] = {
                            "total_pages_crawled": ai_results.get("crawl_stats", {}).get("total_pages_crawled", 0),
                            "max_depth_reached": ai_results.get("crawl_stats", {}).get("max_depth_reached", 0),
                            "crawl_intent": ai_results.get("crawl_intent", {}),
                            "strategy": "ai_guided",
                            "using_consolidation": True
                        }
                    else:
                        # Use results directly without consolidation
                        jobs[job_id]["results"] = ai_results.get("results", [])
                        jobs[job_id]["crawl_info"] = {
                            "total_pages_crawled": ai_results.get("crawl_stats", {}).get("total_pages_crawled", 0),
                            "max_depth_reached": ai_results.get("crawl_stats", {}).get("max_depth_reached", 0),
                            "crawl_intent": ai_results.get("crawl_intent", {}),
                            "strategy": "ai_guided",
                            "using_consolidation": False
                        }
                    
                    # Include exploration history for visualization
                    if "exploration_history" in ai_results:
                        jobs[job_id]["exploration_history"] = ai_results["exploration_history"]
                        
                    # Complete the job
                    jobs[job_id]["extraction_method"] = "ai_guided"
                    complete_job(job_id)
                    return
            
            # If we're here, we're using traditional strategies or AI-guided failed
            
            # Get target URLs for traditional extraction approach
            target_urls = get_target_urls(job_id)
            
            # If we have target URLs and not crawling the entire site, process them directly
            if target_urls and not crawl_entire_site:
                await process_target_urls(
                    job_id, target_urls, crawler, extraction_strategy, 
                    cache_mode, pagination_handler, site_analysis, template_storage, url
                )
                return
            
            # If we're here, either we want to crawl the entire site or we didn't find targeted URLs
            # Continue with the original deep crawling approach
            if crawl_entire_site:
                await process_deep_crawl(
                    job_id, url, crawler, strategy_type, max_depth, include_external,
                    keywords, request_params, extraction_strategy, site_analysis, template_storage
                )
            else:
                # For single-page extraction
                await process_single_page(
                    job_id, url, crawler, extraction_strategy, cache_mode, site_analysis, template_storage
                )
                
    except Exception as e:
        # Log the error
        print(f"Error in job {job_id}: {str(e)}")
        # Update job status
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)

# Helper functions for scrape_website

async def analyze_site(url, page_sample, extract_description, domain_intelligence, ai_model_quality):
    """Analyze site structure and content to determine extraction strategy"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        instruction = f"""
        Analyze this webpage structure and content to determine the optimal extraction approach:
        
        URL: {url}
        USER EXTRACTION REQUEST: {extract_description}
        
        PAGE SAMPLE HTML:
        {page_sample[:5000]}
        
        Return a JSON with these properties:
        1. "site_type": The type of website (e.g., "e-commerce", "blog", "listing", "social-media", "data-table")
        2. "content_structure": How structured the content is (1-10 scale, where 10 is highly structured)
        3. "js_dependent": Whether content appears to rely on JavaScript rendering (true/false)
        4. "pagination_detected": Whether the page appears to have pagination (true/false)
        5. "infinite_scroll_detected": Whether the page might use infinite scrolling (true/false)
        6. "main_content_selector": CSS selector that would target the main content area
        7. "item_container_selector": CSS selector for repeating content items/cards (if any)
        8. "recommended_extraction_type": Either "css", "llm", or "hybrid"
        9. "recommended_selectors": Array of recommended CSS selectors for the requested content
        10. "click_actions_needed": Array of elements that may need to be clicked to reveal content

        Provide this as a clean JSON object without any explanation, markdown or code blocks.
        """
        
        response = model.generate_content(instruction)
        
        try:
            # Parse the JSON response
            if "```json" in response.text:
                result = json.loads(response.text.split("```json")[1].split("```")[0])
            elif "```" in response.text:
                result = json.loads(response.text.split("```")[1])
            else:
                result = json.loads(response.text)
                
            # Add timestamp for future reference
            result["analysis_timestamp"] = datetime.now().isoformat()
            
            # Enhance with domain intelligence
            domain_type = domain_intelligence.detect_website_type(page_sample, url)
            result["detected_domain_type"] = domain_type
            result["specialized_extraction"] = domain_intelligence.get_specialized_extraction_config(domain_type, extract_description)
            
            return result
        except Exception as e:
            print(f"Error parsing site analysis response: {str(e)}")
            # Return fallback analysis
            return fallback_site_analysis(url, domain_intelligence, page_sample, extract_description)
    except Exception as e:
        print(f"Site structure analysis failed: {str(e)}")
        return fallback_site_analysis(url, domain_intelligence, page_sample, extract_description)

def fallback_site_analysis(url, domain_intelligence, page_sample, extract_description=None):
    """Generate a basic fallback site analysis when AI analysis fails"""
    domain_type = domain_intelligence.detect_website_type(page_sample, url)
    
    return {
        "site_type": "unknown",
        "content_structure": 5,
        "js_dependent": True,
        "pagination_detected": False,
        "infinite_scroll_detected": False,
        "main_content_selector": "body",
        "item_container_selector": None,
        "recommended_extraction_type": "llm",
        "recommended_selectors": [],
        "click_actions_needed": [],
        "analysis_timestamp": datetime.now().isoformat(),
        "detected_domain_type": domain_type,
        "specialized_extraction": domain_intelligence.get_specialized_extraction_config(domain_type, extract_description),
        "analysis_error": "Failed to analyze site structure"
    }

def create_browser_config(request_params):
    """Create a browser configuration based on user settings"""
    from fake_useragent import UserAgent
    
    user_agent = UserAgent().random
    
    # Configure browser based on settings
    if request_params.get("use_browser", True):
        return BrowserConfig(
            headless=True,
            user_agent=user_agent,
            java_script_enabled=not request_params.get("disable_javascript", False),
            proxy=request_params.get("proxy_url") if request_params.get("use_proxy", False) else None
        )
    else:
        return BrowserConfig(
            headless=True,
            user_agent=user_agent,
            java_script_enabled=True,
            proxy=None
        )

def get_cache_mode(cache_mode_str):
    """Convert a string cache mode to the proper CacheMode enum value"""
    if cache_mode_str == "memory":
        return CacheMode.ENABLED
    elif cache_mode_str == "none":
        return CacheMode.DISABLED
    elif cache_mode_str == "disk":
        return CacheMode.ENABLED  # For disk cache, we still use ENABLED but also set cache_directory
    else:
        return CacheMode.ENABLED  # Default

def handle_sitemap_discovery(job_id, sitemap_info, extract_description, domain_intelligence, site_discovery):
    """Handle the discovery of a sitemap and gather relevant URLs"""
    print(f"Found sitemap at {sitemap_info['sitemap_url']}")
    jobs[job_id]["sitemap_found"] = True
    jobs[job_id]["sitemap_url"] = sitemap_info["sitemap_url"]
    
    # Extract keywords for targeting specific content
    target_keywords = []
    if extract_description:
        # Extract keywords from user request
        keywords_from_desc = domain_intelligence.extract_keywords(extract_description)
        target_keywords = keywords_from_desc[:5]  # Use top 5 keywords
    
    # Process sitemap asynchronously (this should be awaited, but simplified for this example)
    asyncio.create_task(
        process_sitemap_async(job_id, sitemap_info["sitemap_url"], target_keywords, site_discovery)
    )

async def process_sitemap_async(job_id, sitemap_url, target_keywords, site_discovery):
    """Process sitemap in the background and update job with relevant URLs"""
    relevant_urls = await site_discovery.process_sitemap(
        sitemap_url,
        target_keywords=target_keywords if target_keywords else None
    )
    
    if relevant_urls:
        print(f"Found {len(relevant_urls)} relevant URLs in sitemap")
        jobs[job_id]["sitemap_urls_found"] = len(relevant_urls)
        
        # Sort URLs by priority and take top 10
        sorted_urls = sorted(relevant_urls, 
                           key=lambda x: x.get("priority", 0.5), 
                           reverse=True)
        top_urls = [item["url"] for item in sorted_urls[:10]]
        
        print(f"Using top {len(top_urls)} URLs from sitemap for targeted scraping")
        jobs[job_id]["using_sitemap_urls"] = True
        jobs[job_id]["sitemap_urls"] = top_urls

async def handle_search_request(job_id, url, page_sample, extract_description, domain_intelligence, search_detector):
    """Handle search-oriented extraction request"""
    print(f"Request seems to be search-oriented: '{extract_description}'")
    
    # Extract search terms from the request
    search_terms = domain_intelligence.extract_search_terms(extract_description)
    print(f"Extracted search terms: '{search_terms}'")
    
    # Try to discover and use search functionality on the site
    print("Discovering search functionality...")
    search_results = await search_detector.discover_search_functionality(
        url, page_sample, search_terms
    )
    
    if search_results.get("search_executed") and search_results.get("next_urls"):
        print(f"Search executed successfully, found {len(search_results['next_urls'])} result URLs")
        jobs[job_id]["search_results"] = {
            "performed": True,
            "method": search_results.get("method"),
            "search_url": search_results.get("search_url"),
            "results_found": len(search_results.get("next_urls", [])),
            "total_results": search_results.get("total_results_found", 0)
        }
        
        # Get URLs from search results for targeted extraction
        search_result_urls = search_results.get("next_urls", [])
        
        if search_result_urls:
            print(f"Using {len(search_result_urls)} search result URLs for extraction")
            jobs[job_id]["using_search_results"] = True
            jobs[job_id]["search_urls"] = search_result_urls[:10]  # Limit to first 10
            
            # Also store pagination URLs if available
            if search_results.get("pagination_urls"):
                jobs[job_id]["pagination_urls"] = search_results.get("pagination_urls")

def get_target_urls(job_id):
    """Get target URLs for extraction from job data"""
    target_urls = []
    
    # Priority 1: Use search result URLs if available
    if jobs[job_id].get("using_search_results") and jobs[job_id].get("search_urls"):
        target_urls = jobs[job_id].get("search_urls")
        print(f"Using {len(target_urls)} URLs from search results")
    
    # Priority 2: Use sitemap URLs if available and no search results
    elif jobs[job_id].get("using_sitemap_urls") and jobs[job_id].get("sitemap_urls"):
        target_urls = jobs[job_id].get("sitemap_urls")
        print(f"Using {len(target_urls)} URLs from sitemap")
        
    return target_urls

async def process_target_urls(job_id, target_urls, crawler, extraction_strategy, cache_mode,
                              pagination_handler, site_analysis, template_storage, base_url):
    """Process a list of target URLs for extraction"""
    results = []
    
    for idx, target_url in enumerate(target_urls):
        print(f"Extracting from target URL ({idx+1}/{len(target_urls)}): {target_url}")
        
        try:
            # Configure extraction
            crawl_config = CrawlerRunConfig(
                extraction_strategy=extraction_strategy,
                cache_mode=cache_mode,
                verbose=True
            )
            
            # Run extraction
            extraction_result = await crawler.arun(url=target_url, config=crawl_config)
            
            if extraction_result.success:
                # Process the result
                content = extraction_result.extracted_content
                
                # Parse JSON if it's returned as a string
                if isinstance(content, str):
                    try:
                        if content.strip():  # Only try to parse if not empty
                            content = json.loads(content)
                        else:
                            print(f"Empty extraction result from {target_url}, using fallback")
                            content = await perform_extraction_with_fallback(
                                target_url, 
                                extraction_result.html, 
                                jobs[job_id]["ai_suggestions"],
                                job_id
                            )
                    except json.JSONDecodeError:
                        # If JSON parsing fails, use our fallback mechanism
                        print(f"JSON parsing failed for {target_url}, trying fallback extraction")
                        content = await perform_extraction_with_fallback(
                            target_url, 
                            extraction_result.html, 
                            jobs[job_id]["ai_suggestions"],
                            job_id
                        )
                
                # Clean the extraction output
                content = clean_extraction_output(content)
                
                # Add to results
                results.append({
                    "source_url": target_url,
                    "depth": 0,
                    "score": 1.0 - (idx * 0.02),  # Score decreases slightly for later items
                    "data": content
                })
        except Exception as e:
            print(f"Error extracting from {target_url}: {str(e)}")
        
        # Update progress
        progress = 0.3 + ((idx + 1) / len(target_urls)) * 0.6
        jobs[job_id]["progress"] = min(progress, 0.9)
    
    # Check if we need to handle pagination
    if jobs[job_id].get("search_results") and jobs[job_id].get("pagination_urls"):
        print("Processing pagination for search results")
        
        # Get pagination URLs from search results
        pagination_urls = jobs[job_id].get("pagination_urls", [])
        
        if pagination_urls:
            # Process pagination up to 3 more pages
            paginated_results = await pagination_handler.process_pagination(
                pagination_urls[:3],  # Limit to 3 pagination pages
                crawler,
                extraction_strategy,
                cache_mode
            )
            
            if paginated_results:
                print(f"Found {len(paginated_results)} additional results from pagination")
                # Clean the paginated results
                for result in paginated_results:
                    if "data" in result:
                        result["data"] = clean_extraction_output(result["data"])
                results.extend(paginated_results)
    
    # Store results and complete the job
    if results:
        print(f"Completed extraction with {len(results)} results")
        jobs[job_id]["results"] = process_results(results)
        jobs[job_id]["extraction_method"] = "intelligent_targeted"
        
        # Save a template if extraction was successful
        if len(results) > 0:
            template_success = await template_storage.save_template(
                base_url, site_analysis, extraction_strategy, results
            )
            if template_success:
                jobs[job_id]["template_saved"] = True
        
        complete_job(job_id)

async def process_deep_crawl(job_id, url, crawler, strategy_type, max_depth, include_external,
                            keywords, request_params, extraction_strategy, site_analysis, template_storage):
    """Process a deep crawl of the entire site"""
    print(f"Setting up deep crawl with strategy type: {strategy_type}, max_depth: {max_depth}")
    
    # Parse keywords if provided
    keyword_list = []
    if keywords:
        keyword_list = [k.strip() for k in keywords.split(",") if k.strip()]
    
    # Get max pages from request params or use default
    max_pages = request_params.get("max_pages", 100)
    
    # If site has pagination detected, adjust the max_depth for better results
    if site_analysis.get("pagination_detected", False) and max_depth < 3:
        print("Pagination detected, increasing max_depth to ensure complete content extraction")
        max_depth = max(max_depth, 3)
    
    # Create the appropriate crawl strategy
    deep_crawl_strategy = get_crawl_strategy(
        strategy_type,
        max_depth=max_depth,
        include_external=include_external,
        user_prompt=extract_description,
        max_pages=max_pages,
        keywords=keyword_list,
        include_patterns=request_params.get("include_patterns"),
        exclude_patterns=request_params.get("exclude_patterns"),
        exclude_media=request_params.get("exclude_media", True)
    )
    
    # Setup crawler configuration - in 0.6.0rc1, streaming must be enabled for deep crawling
    crawl_config = CrawlerRunConfig(
        deep_crawl_strategy=deep_crawl_strategy,
        extraction_strategy=extraction_strategy,
        cache_mode=get_cache_mode(request_params.get("cache_mode", "memory")),
        verbose=True,
        stream=True  # This is critical for deep crawling in 0.6.0rc1
    )
    
    # Add JS click actions if needed based on site analysis
    if site_analysis.get("click_actions_needed"):
        js_code = []
        for action in site_analysis.get("click_actions_needed", []):
            if isinstance(action, dict) and action.get("selector") and action.get("action") == "click":
                js_code.append(f"""
                (async () => {{
                    const elem = document.querySelector('{action["selector"]}');
                    if (elem) {{
                        console.log('Clicking on element: {action["selector"]}');
                        elem.click();
                        return await new Promise(r => setTimeout(r, 1000));
                    }}
                    return false;
                }})();
                """)
        
        if js_code:
            crawl_config.js_code = js_code
    
    # Update progress
    jobs[job_id]["progress"] = 0.4
    
    # Run the deep crawl
    results = []
    
    # For deep crawling, use streaming to get results as they come
    async for result in await crawler.arun(url=url, config=crawl_config):
        if result.success:
            try:
                # Process the result
                extraction_result = result.extracted_content
                
                # Parse JSON if it's returned as a string
                if isinstance(extraction_result, str):
                    try:
                        if extraction_result.strip():  # Only try to parse if not empty
                            extraction_result = json.loads(extraction_result)
                        else:
                            print(f"Empty extraction result from {result.url}, using fallback")
                            extraction_result = await perform_extraction_with_fallback(
                                result.url, 
                                result.html, 
                                jobs[job_id]["ai_suggestions"],
                                job_id
                            )
                    except json.JSONDecodeError as e:
                        # If JSON parsing fails, use our fallback mechanism
                        print(f"JSON parsing failed for {result.url}: {str(e)}, trying fallback extraction")
                        extraction_result = await perform_extraction_with_fallback(
                            result.url, 
                            result.html, 
                            jobs[job_id]["ai_suggestions"],
                            job_id
                        )
                
                # Clean the extraction output
                extraction_result = clean_extraction_output(extraction_result)
                
                # Add to results
                results.append({
                    "source_url": result.url,
                    "depth": result.metadata.get("depth", 0),
                    "score": result.metadata.get("score", 0),
                    "data": extraction_result
                })
                
                # Update job progress based on completed items
                completed = len(results)
                total = max_pages
                progress = 0.4 + (completed / max(total, 1)) * 0.5
                jobs[job_id]["progress"] = min(progress, 0.9)
                
            except Exception as e:
                print(f"Error processing result from {result.url}: {str(e)}")
                jobs[job_id]["errors"] = jobs[job_id].get("errors", [])
                jobs[job_id]["errors"].append(f"Failed to process result from {result.url}: {str(e)}")
    
    # Process results
    processed_data = process_results(results)
    
    # Store site analysis in results for reference
    if site_analysis:
        # Add a summary of the extraction approach used
        extraction_summary = {
            "site_type": site_analysis.get("site_type", "unknown"),
            "domain_type": site_analysis.get("detected_domain_type", "unknown"),
            "extraction_approach": site_analysis.get("recommended_extraction_type", "llm"),
            "content_structure_score": site_analysis.get("content_structure", 5),
            "js_dependent": site_analysis.get("js_dependent", False),
            "pagination_detected": site_analysis.get("pagination_detected", False)
        }
        processed_data.append({
            "source_url": f"{url}#extraction_metadata",
            "depth": -1,  # Special marker for metadata
            "score": 0,
            "data": {
                "extraction_metadata": extraction_summary
            }
        })
    
    # Store results
    jobs[job_id]["results"] = processed_data
    
    # Save template if extraction was successful
    if len(processed_data) > 0 and not jobs[job_id].get("template_saved"):
        await template_storage.save_template(
            url, site_analysis, extraction_strategy, processed_data
        )
    
    complete_job(job_id)

async def process_single_page(job_id, url, crawler, extraction_strategy, cache_mode, site_analysis, template_storage):
    """Process a single page for extraction"""
    # For single-page extraction
    crawl_config = CrawlerRunConfig(
        extraction_strategy=extraction_strategy,
        cache_mode=cache_mode,
        verbose=True
    )
    
    result = await crawler.arun(url=url, config=crawl_config)
    
    if result.success:
        try:
            # Process the result
            extraction_result = result.extracted_content
            
            # Parse JSON if it's returned as a string
            if isinstance(extraction_result, str):
                try:
                    if extraction_result.strip():  # Only try to parse if not empty
                        extraction_result = json.loads(extraction_result)
                    else:
                        print(f"Empty extraction result from {url}, using fallback")
                        extraction_result = await perform_extraction_with_fallback(
                            url, 
                            result.html, 
                            jobs[job_id]["ai_suggestions"],
                            job_id
                        )
                except json.JSONDecodeError as e:
                    # If JSON parsing fails, use our fallback mechanism
                    print(f"JSON parsing failed for {url}: {str(e)}, trying fallback extraction")
                    extraction_result = await perform_extraction_with_fallback(
                        url, 
                        result.html, 
                        jobs[job_id]["ai_suggestions"],
                        job_id
                    )
            
            # Clean the extraction output
            extraction_result = clean_extraction_output(extraction_result)
            
            # If extraction_result is empty or had errors, try fallback
            if not extraction_result or (isinstance(extraction_result, list) and 
                                       any(item.get("error", False) for item in extraction_result)):
                print(f"LLM extraction failed or returned errors, using fallback for {url}")
                fallback_result = await perform_extraction_with_fallback(
                    url, 
                    result.html, 
                    jobs[job_id]["ai_suggestions"],
                    job_id
                )
                
                # If fallback got results, use those. Otherwise keep what we have
                if fallback_result and len(fallback_result) > 0:
                    extraction_result = fallback_result
            
            # Store the extraction result
            processed_data = process_results([{
                "source_url": url,
                "depth": 0,
                "score": 1.0,
                "data": extraction_result
            }])
            
            jobs[job_id]["results"] = processed_data
            
            # Save template if extraction was successful
            if len(processed_data) > 0:
                template_success = await template_storage.save_template(
                    url, site_analysis, extraction_strategy, processed_data
                )
                if template_success:
                    jobs[job_id]["template_saved"] = True
            
            complete_job(job_id)
                
        except Exception as e:
            print(f"Error processing extraction: {str(e)}")
            jobs[job_id]["errors"] = jobs[job_id].get("errors", [])
            jobs[job_id]["errors"].append(f"Failed to process extraction from {url}: {str(e)}")
            
            # Try fallback extraction in case of any exception
            try:
                fallback_result = await perform_extraction_with_fallback(
                    url, 
                    result.html, 
                    jobs[job_id]["ai_suggestions"],
                    job_id
                )
                
                if fallback_result:
                    jobs[job_id]["results"] = process_results([{
                        "source_url": url,
                        "depth": 0,
                        "score": 1.0,
                        "data": fallback_result
                    }])
                    
                    complete_job(job_id)
            except Exception as fallback_error:
                print(f"Fallback extraction also failed: {str(fallback_error)}")
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["error"] = str(fallback_error)
    else:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = "Failed to fetch the page"

async def perform_extraction_with_fallback(url, html_content, ai_suggestions, job_id):
    """
    Performs extraction with fallback mechanisms if the primary extraction fails.
    
    Args:
        url: The URL being scraped
        html_content: The HTML content to extract data from
        ai_suggestions: The AI suggestions object with schema and selectors
        job_id: The current job ID for error tracking
    
    Returns:
        List of extracted items
    """
    try:
        print("DEBUG: About to import perform_extraction_with_fallback function")
        # Try using BeautifulSoup with CSS selectors as a fallback extraction method
        from extraction.fallback_extraction import perform_extraction_with_fallback as fallback_extract
        print("DEBUG: Import successful, about to call the function")
        result = await fallback_extract(html_content, url, ai_suggestions)
        return result
    except Exception as e:
        print(f"Fallback extraction failed: {str(e)}")
        # Create a minimal result to avoid complete failure
        return [{"error": True, "source_url": url, "tags": ["error"], "content": f"Extraction failed: {str(e)}"}]

def complete_job(job_id):
    """Mark a job as completed and generate export files"""
    jobs[job_id]["status"] = "completed"
    jobs[job_id]["progress"] = 1.0
    jobs[job_id]["completed_at"] = datetime.now().isoformat()
    
    # Generate export files
    jobs[job_id]["exports"] = {
        "json": generate_json_export(jobs[job_id]["results"]),
        "csv": generate_csv_export(jobs[job_id]["results"]),
        "excel": generate_excel_export(jobs[job_id]["results"])
    }

def normalize_url(url: str, base_url: str = None) -> str:
    """
    Normalize a URL using the URL service.
    
    Args:
        url: The URL to normalize
        base_url: Optional base URL for relative URL resolution
        
    Returns:
        Normalized URL string
    """
    # Get URL service from registry
    registry = ServiceRegistry()
    url_service = registry.get_service("url_service")
    
    # Use URL service to normalize the URL
    return url_service.normalize_url(url, base_url)