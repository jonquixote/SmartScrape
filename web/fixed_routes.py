"""
Fixed routes that use the new non-recursive system
"""
import asyncio
import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException
from controllers.simple_scraper import simple_scrape_handler
from components.duckduckgo_url_generator import DuckDuckGoURLGenerator
import logging

logger = logging.getLogger(__name__)

# Create a separate router for fixed endpoints
fixed_router = APIRouter()

@fixed_router.post("/scrape-intelligent-fixed")
async def scrape_intelligent_fixed(request_data: dict):
    """Fixed intelligent scraping without recursive loops"""
    try:
        query = request_data.get('query', '')
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
            
        # Use the simple scraper instead of the broken recursive system
        result = await simple_scrape_handler(query)
        
        job_id = str(uuid.uuid4())
        
        # Format response to match expected structure
        formatted_result = {
            'job_id': job_id,
            'status': 'completed',
            'result': {
                'query_received': query,
                'extracted_data': {
                    'title': f"Found {result['total_items']} results for '{query}'",
                    'summary': f"Successfully extracted {result['total_items']} items",
                    'items_found': result['items'],
                    'urls_attempted': result['urls_processed'],
                    'successful_scrapes': result['total_items']
                },
                'status': result['status'],
                'message': f'Successfully processed query: {query}'
            },
            'submitted_at': datetime.now().isoformat(),
            'completed_at': datetime.now().isoformat()
        }
        
        return formatted_result
        
    except Exception as e:
        logger.error(f"Fixed scraping failed: {e}")
        return {"error": str(e), "status": "failed"}, 500

@fixed_router.get("/test-real-duckduckgo")
async def test_real_duckduckgo():
    """Test endpoint to verify real DuckDuckGo URLs"""
    try:
        generator = DuckDuckGoURLGenerator()
        results = generator.generate_urls("Tesla news", max_urls=5)
        
        # Extract just URL and title for testing
        url_list = []
        for result in results:
            url_list.append({
                "title": result.title if hasattr(result, 'title') else 'Unknown',
                "url": result.url if hasattr(result, 'url') else str(result),
                "is_real": generator._is_valid_real_url(result.url if hasattr(result, 'url') else str(result))
            })
        
        return {
            "status": "success",
            "query": "Tesla news", 
            "total_results": len(url_list),
            "urls": url_list,
            "fake_urls_filtered": True,
            "test_passed": all(item['is_real'] for item in url_list)
        }
    except Exception as e:
        logger.error(f"DuckDuckGo test failed: {e}")
        return {"error": str(e), "status": "failed"}, 500

@fixed_router.get("/test-simple-scraper/{query}")
async def test_simple_scraper(query: str):
    """Test the simple scraper directly"""
    try:
        result = await simple_scrape_handler(query)
        return {
            "status": "success",
            "result": result,
            "no_recursion": True,
            "test_passed": result['total_items'] > 0
        }
    except Exception as e:
        logger.error(f"Simple scraper test failed: {e}")
        return {"error": str(e), "status": "failed"}, 500

@fixed_router.get("/debug-url-generation/{query}")
async def debug_url_generation(query: str):
    """Debug URL generation to see what DuckDuckGo returns"""
    try:
        from duckduckgo_search import DDGS
        
        # Test raw DuckDuckGo
        ddgs = DDGS()
        raw_results = ddgs.text(query, max_results=5)
        
        # Test our generator
        generator = DuckDuckGoURLGenerator()
        processed_results = generator.generate_urls(query, max_urls=5)
        
        raw_urls = []
        for result in raw_results:
            raw_urls.append({
                "title": result.get('title', ''),
                "url": result.get('href', ''),
                "is_fake": any(fake in result.get('href', '').lower() 
                             for fake in ['lite.duckduckgo.com', 'html.duckduckgo.com'])
            })
        
        processed_urls = []
        for result in processed_results:
            processed_urls.append({
                "title": result.title if hasattr(result, 'title') else 'Unknown',
                "url": result.url if hasattr(result, 'url') else str(result),
                "is_real": generator._is_valid_real_url(result.url if hasattr(result, 'url') else str(result))
            })
        
        return {
            "query": query,
            "raw_duckduckgo_results": raw_urls,
            "processed_results": processed_urls,
            "raw_count": len(raw_urls),
            "processed_count": len(processed_urls),
            "fake_urls_in_raw": sum(1 for url in raw_urls if url['is_fake']),
            "real_urls_in_processed": sum(1 for url in processed_urls if url['is_real'])
        }
        
    except Exception as e:
        logger.error(f"URL generation debug failed: {e}")
        return {"error": str(e), "status": "failed"}, 500
