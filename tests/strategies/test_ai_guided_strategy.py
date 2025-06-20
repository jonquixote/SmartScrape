"""
Test suite for the AIGuidedStrategy class.

This tests the AI-guided strategy's ability to analyze websites,
generate search instructions, and execute searches using different methods.
"""

import pytest
import asyncio
import json
from unittest.mock import MagicMock, patch, AsyncMock

from strategies.ai_guided_strategy import AIGuidedStrategy
from strategies.core.strategy_context import StrategyContext
from strategies.core.strategy_types import StrategyCapability

# Sample data for tests
SAMPLE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Test Website</title>
    <meta name="description" content="A website for testing">
</head>
<body>
    <nav>
        <ul>
            <li><a href="/about">About</a></li>
            <li><a href="/products">Products</a></li>
            <li><a href="/contact">Contact</a></li>
        </ul>
    </nav>
    <div class="search-container">
        <form action="/search" method="get" class="search-form">
            <input type="text" name="q" placeholder="Search...">
            <button type="submit">Search</button>
        </form>
    </div>
    <div class="content">
        <h1>Welcome to our site</h1>
        <p>This is a sample page for testing the AI-guided strategy.</p>
    </div>
    <script>
        const apiUrl = "/api/search";
        fetch("/api/products").then(res => res.json());
    </script>
</body>
</html>
"""

SAMPLE_SEARCH_INSTRUCTIONS = {
    "method": "form",
    "form_selector": "form.search-form",
    "input_selector": "input[name='q']",
    "approach": "Use the search form",
    "url": "https://example.com"
}

# Mock AIService response
MOCK_AI_RESPONSE = """
```json
{
  "method": "form",
  "form_selector": "form.search-form",
  "input_selector": "input[name='q']",
  "approach": "Use the search form"
}
```
"""

@pytest.fixture
def mock_context():
    """Create a mock strategy context with the required services."""
    context = MagicMock(spec=StrategyContext)
    
    # Mock AI service
    ai_service = AsyncMock()
    ai_service.generate_response = AsyncMock(return_value={"content": MOCK_AI_RESPONSE})
    
    # Add the service to context
    context.get_service.return_value = ai_service
    
    return context

@pytest.fixture
def strategy(mock_context):
    """Create an AI-guided strategy with mocked services."""
    config = {
        'max_depth': 2,
        'max_pages': 10,
        'cache_instructions': True
    }
    
    with patch('strategies.ai_guided_strategy.DOMStrategy') as MockDOMStrategy:
        # Set up mock DOM strategy
        mock_dom_strategy = MockDOMStrategy.return_value
        mock_dom_strategy.search = AsyncMock(return_value={"success": True, "results": {"items": []}})
        
        # Create the strategy
        strategy = AIGuidedStrategy(context=mock_context, config=config)
        
        # Mock the fetch methods to avoid actual network requests
        strategy._fetch_url = MagicMock(return_value=SAMPLE_HTML)
        strategy._fetch_url_async = AsyncMock(return_value=SAMPLE_HTML)
        
        # Mock the search automator
        strategy.search_automator.detect_search_forms = AsyncMock(return_value=[
            {"form_selector": "form.search-form", "input_selector": "input[name='q']", "search_relevance_score": 0.9}
        ])
        
        return strategy

class TestAIGuidedStrategy:
    """Test cases for the AIGuidedStrategy class."""
    
    def test_initialization_with_context(self, mock_context):
        """Test strategy initialization with context."""
        strategy = AIGuidedStrategy(context=mock_context)
        
        # Check that it inherits properly from BaseStrategyV2
        assert hasattr(strategy, 'extract_links')
        assert hasattr(strategy, 'clean_html')
        
        # Check that it set up required internal components
        assert hasattr(strategy, 'prompt_generator')
        assert hasattr(strategy, 'search_automator')
        assert hasattr(strategy, 'dom_strategy')
        
        # Check that it has the right name
        assert strategy.name == "ai_guided"
        
        # Check that it set up AI service or model
        assert strategy.ai_service is not None or strategy.ai_model is not None
    
    def test_strategy_metadata(self):
        """Test that the strategy has the correct metadata."""
        assert hasattr(AIGuidedStrategy, '_metadata')
        metadata = AIGuidedStrategy._metadata
        
        # Check capabilities
        assert StrategyCapability.AI_ASSISTED in metadata.capabilities
        assert StrategyCapability.FORM_INTERACTION in metadata.capabilities
        assert StrategyCapability.API_INTERACTION in metadata.capabilities
        
        # Check description
        assert "AI-guided" in metadata.description
        assert "machine learning" in metadata.description
    
    @pytest.mark.asyncio
    async def test_analyze_website(self, strategy):
        """Test website analysis functionality."""
        url = "https://example.com"
        analysis = await strategy._analyze_website(url)
        
        # Verify the analysis result
        assert analysis is not None
        assert analysis["title"] == "Test Website"
        assert analysis["description"] == "A website for testing"
        assert analysis["has_search_form"] is True
        assert analysis["has_api"] is True
        assert len(analysis["api_endpoints"]) > 0
        assert "/api/search" in analysis["api_endpoints"] or "/api/products" in analysis["api_endpoints"]
    
    @pytest.mark.asyncio
    async def test_generate_search_instructions(self, strategy):
        """Test generating search instructions using AI."""
        url = "https://example.com"
        site_analysis = {
            "title": "Test Website",
            "description": "A website for testing",
            "site_type": "content",
            "has_search_form": True,
            "has_api": True,
            "is_spa": False,
            "navigation": [[{"text": "About", "url": "/about"}]],
            "search_forms": [],
            "api_endpoints": ["/api/search"],
            "url": url
        }
        
        # Patch the prompt generator to avoid calling actual implementation
        with patch.object(strategy.prompt_generator, 'generate_search_strategy_prompt', 
                         return_value="Generate search instructions for example.com"):
            
            # Test instructions generation
            search_term = "test query"
            instructions = await strategy._generate_search_instructions(url, site_analysis, search_term)
            
            # Verify the instructions
            assert instructions is not None
            assert "method" in instructions
            assert instructions["method"] == "form"
            assert "form_selector" in instructions
    
    @pytest.mark.asyncio
    async def test_execute_with_cached_instructions(self, strategy):
        """Test execute method with cached instructions."""
        # Set up a cached instruction
        url = "https://example.com"
        domain = "example.com"
        search_term = "test query"
        strategy.site_instructions_cache[domain] = SAMPLE_SEARCH_INSTRUCTIONS
        
        # Mock _execute_search_with_instructions
        strategy._execute_search_with_instructions = AsyncMock(return_value={
            "success": True,
            "engine": "ai_guided_form",
            "results": {"items": [{"title": "Test Result", "url": "https://example.com/result"}]}
        })
        
        # Execute the search
        result = await strategy.search(url, search_term)
        
        # Verify the result
        assert result["success"] is True
        assert result["engine"] == "ai_guided_form"
        assert "results" in result
        assert "items" in result["results"]
        
        # Verify cache was used
        strategy._execute_search_with_instructions.assert_called_once()
        args, kwargs = strategy._execute_search_with_instructions.call_args
        assert args[0] == url
        assert args[1] == search_term
        assert args[2] == SAMPLE_SEARCH_INSTRUCTIONS
    
    @pytest.mark.asyncio
    async def test_execute_method(self, strategy):
        """Test the main execute method."""
        # Mock internal methods
        strategy._extract_with_ai_guidance = AsyncMock(return_value={
            "url": "https://example.com",
            "title": "Example Page",
            "content": "Sample content",
            "extraction_method": "ai_guided_schema"
        })
        
        # Execute the strategy
        url = "https://example.com"
        result = await strategy.execute(url, search_term="test")
        
        # Verify the results
        assert result is not None
        assert result["extraction_method"] == "ai_guided_schema"
        assert result["url"] == url
        
        # Verify internal methods were called
        strategy._extract_with_ai_guidance.assert_called_once()
        
        # Verify result was added to results list
        assert len(strategy._results) == 1
        assert strategy._results[0] == result
    
    @pytest.mark.asyncio
    async def test_extract_with_ai_guidance(self, strategy):
        """Test the _extract_with_ai_guidance method with mocked AI service."""
        # Mock the _determine_extraction_approach method
        strategy._determine_extraction_approach = AsyncMock(return_value={"method": "standard"})
        
        # Mock super().extract
        with patch.object(strategy.__class__.__bases__[0], 'extract', 
                         return_value={"url": "https://example.com", "title": "Example", "content": "Test content"}):
            
            # Call the method
            url = "https://example.com"
            html_content = SAMPLE_HTML
            user_intent = "find information"
            result = await strategy._extract_with_ai_guidance(html_content, url, user_intent, {})
            
            # Verify the result
            assert result is not None
            assert result["url"] == url
            assert result["title"] == "Example"
            
            # Verify internal methods were called
            strategy._determine_extraction_approach.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_crawl_method(self, strategy):
        """Test the crawl method with AI scoring."""
        # Mock internal methods
        strategy._extract_with_ai_guidance = AsyncMock(return_value={
            "url": "https://example.com",
            "title": "Example Page",
            "content": "Sample content"
        })
        
        strategy._extract_and_score_links = AsyncMock(return_value=[
            {"url": "https://example.com/page1", "score": 0.9, "text": "Page 1"},
            {"url": "https://example.com/page2", "score": 0.7, "text": "Page 2"}
        ])
        
        # Execute crawl
        url = "https://example.com"
        result = await strategy.crawl(url, user_intent="find information")
        
        # Verify the results
        assert result is not None
        assert "results" in result
        assert len(result["results"]) > 0
        assert "metrics" in result
        assert "visited_urls" in result
        
        # Verify internal methods were called
        strategy._extract_with_ai_guidance.assert_called()
        strategy._extract_and_score_links.assert_called()
        
        # Verify results added to results list
        assert len(strategy._results) > 0
    
    @pytest.mark.asyncio
    async def test_extract_and_score_links(self, strategy):
        """Test the link extraction and scoring."""
        # Create a mock HTML with links
        html_with_links = """
        <html>
            <body>
                <a href="https://example.com/relevant">Very Relevant Link</a>
                <a href="https://example.com/somewhat">Somewhat Relevant</a>
                <a href="https://example.com/not">Not Relevant</a>
            </body>
        </html>
        """
        
        # Mock response from AI service
        mock_response = {"content": """
        ```json
        {
          "0": 0.9,
          "1": 0.6,
          "2": 0.2
        }
        ```
        """}
        
        # If we have AI service in context, mock its response
        if hasattr(strategy, 'ai_service') and strategy.ai_service:
            strategy.ai_service.generate_response = AsyncMock(return_value=mock_response)
        
        # Mock extract_links to return predefined links
        with patch.object(strategy, 'extract_links', return_value=[
            {"url": "https://example.com/relevant", "text": "Very Relevant Link"},
            {"url": "https://example.com/somewhat", "text": "Somewhat Relevant"},
            {"url": "https://example.com/not", "text": "Not Relevant"}
        ]):
            # Call the method
            url = "https://example.com"
            user_intent = "find relevant information"
            result = await strategy._extract_and_score_links(html_with_links, url, user_intent, {})
            
            # Verify the results
            assert len(result) == 3
            assert result[0]["url"] == "https://example.com/relevant"
            assert result[0]["score"] >= 0.5  # Should have a high score
    
    @pytest.mark.asyncio
    async def test_parse_ai_response(self, strategy):
        """Test parsing AI response into structured instructions."""
        # Test JSON response
        json_response = """
        ```json
        {
          "method": "form",
          "form_selector": "form.search-form",
          "input_selector": "input[name='q']",
          "approach": "Use the search form"
        }
        ```
        """
        
        url = "https://example.com"
        result = strategy._parse_ai_response(json_response, url)
        
        # Verify the parsed result
        assert result["method"] == "form"
        assert result["form_selector"] == "form.search-form"
        assert result["input_selector"] == "input[name='q']"
        assert result["approach"] == "Use the search form"
        assert result["url"] == url
        
        # Test text response
        text_response = """
        Approach: Use the API endpoint
        
        Method: api
        
        Endpoint: /api/search
        
        Parameters: q={search_term}&limit=10
        
        Additional considerations: This site has a rate limit of 10 requests per minute.
        """
        
        result = strategy._parse_ai_response(text_response, url)
        
        # Verify the parsed result
        assert result["method"] == "api"
        assert "api_endpoint" in result
        assert "api_parameters" in result
        assert "special_considerations" in result