"""
Test sitemap integration in AdaptiveScraper.

These tests verify that the sitemap processing methods work correctly
and integrate properly with the existing pipeline.
"""

import asyncio
import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, Any, List

from controllers.adaptive_scraper import AdaptiveScraper


class TestSitemapIntegration:
    """
    Tests for AdaptiveScraper sitemap integration.
    """

    @pytest.fixture
    def adaptive_scraper(self):
        """Create an instance of AdaptiveScraper for testing."""
        scraper = AdaptiveScraper(config={
            'use_ai': True,
            'max_pages': 5,
            'max_depth': 2,
            'use_pipelines': True
        })
        return scraper

    @pytest.fixture
    def sample_sitemap_xml(self):
        """Sample sitemap XML for testing."""
        return """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url>
                <loc>https://example.com/page1</loc>
                <lastmod>2024-01-01</lastmod>
                <changefreq>weekly</changefreq>
                <priority>0.8</priority>
            </url>
            <url>
                <loc>https://example.com/page2</loc>
                <lastmod>2024-01-02</lastmod>
                <changefreq>monthly</changefreq>
                <priority>0.6</priority>
            </url>
            <url>
                <loc>https://example.com/page3</loc>
                <lastmod>2024-01-03</lastmod>
                <changefreq>daily</changefreq>
                <priority>0.9</priority>
            </url>
        </urlset>"""

    @pytest.fixture
    def sample_html_content(self):
        """Sample HTML content for testing extraction."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Page</title>
            <meta name="description" content="This is a test page">
        </head>
        <body>
            <h1>Main Title</h1>
            <div class="content">
                <p>This is test content paragraph 1.</p>
                <p>This is test content paragraph 2.</p>
            </div>
            <div class="sidebar">
                <h2>Related Links</h2>
                <ul>
                    <li><a href="/link1">Link 1</a></li>
                    <li><a href="/link2">Link 2</a></li>
                </ul>
            </div>
        </body>
        </html>
        """

    @pytest.mark.asyncio
    async def test_extract_urls_from_sitemap(self, adaptive_scraper, sample_sitemap_xml):
        """Test extracting URLs from a sitemap."""
        # Mock the site_discovery.process_sitemap method
        with patch.object(adaptive_scraper.site_discovery, 'process_sitemap') as mock_process:
            mock_process.return_value = [
                {'url': 'https://example.com/page1', 'lastmod': '2024-01-01', 'priority': 0.8},
                {'url': 'https://example.com/page2', 'lastmod': '2024-01-02', 'priority': 0.6},
                {'url': 'https://example.com/page3', 'lastmod': '2024-01-03', 'priority': 0.9}
            ]
            
            urls = await adaptive_scraper.extract_urls_from_sitemap('https://example.com/sitemap.xml')
            
            assert len(urls) == 3
            assert 'https://example.com/page1' in urls
            assert 'https://example.com/page2' in urls
            assert 'https://example.com/page3' in urls
            
            mock_process.assert_called_once_with('https://example.com/sitemap.xml', max_urls=100)

    @pytest.mark.asyncio
    async def test_extract_urls_from_sitemap_with_limit(self, adaptive_scraper, sample_sitemap_xml):
        """Test extracting URLs from a sitemap with a limit."""
        # Mock the site_discovery.process_sitemap method
        with patch.object(adaptive_scraper.site_discovery, 'process_sitemap') as mock_process:
            mock_process.return_value = [
                {'url': 'https://example.com/page1', 'lastmod': '2024-01-01', 'priority': 0.8},
                {'url': 'https://example.com/page2', 'lastmod': '2024-01-02', 'priority': 0.6},
                {'url': 'https://example.com/page3', 'lastmod': '2024-01-03', 'priority': 0.9}
            ]
            
            urls = await adaptive_scraper.extract_urls_from_sitemap('https://example.com/sitemap.xml', limit=2)
            
            assert len(urls) == 2
            # Should prioritize higher priority URLs
            assert 'https://example.com/page3' in urls  # priority 0.9
            assert 'https://example.com/page1' in urls  # priority 0.8

    @pytest.mark.asyncio
    async def test_extract_urls_from_sitemap_error_handling(self, adaptive_scraper):
        """Test error handling when sitemap processing fails."""
        # Mock the site_discovery.process_sitemap method to raise an exception
        with patch.object(adaptive_scraper.site_discovery, 'process_sitemap') as mock_process:
            mock_process.side_effect = Exception("Failed to process sitemap")
            
            urls = await adaptive_scraper.extract_urls_from_sitemap('https://example.com/sitemap.xml')
            
            assert urls == []

    @pytest.mark.asyncio
    async def test_extract_content_with_crawl4ai_success(self, adaptive_scraper, sample_html_content):
        """Test successful content extraction with Crawl4AI."""
        # Mock Crawl4AI components
        with patch('controllers.adaptive_scraper.AsyncWebCrawler') as mock_crawler_class:
            mock_crawler = AsyncMock()
            mock_crawler_class.return_value.__aenter__.return_value = mock_crawler
            
            # Mock successful crawl result
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.extracted_content = json.dumps({
                'title': 'Test Page',
                'content': 'This is test content paragraph 1. This is test content paragraph 2.',
                'metadata': {'description': 'This is a test page'}
            })
            mock_crawler.arun.return_value = mock_result
            
            content = await adaptive_scraper._extract_content_with_crawl4ai('https://example.com/page1')
            
            assert content is not None
            assert content['title'] == 'Test Page'
            assert 'test content' in content['content']
            assert content['metadata']['description'] == 'This is a test page'

    @pytest.mark.asyncio
    async def test_extract_content_with_crawl4ai_fallback(self, adaptive_scraper, sample_html_content):
        """Test fallback when Crawl4AI extraction fails."""
        # Mock Crawl4AI to fail
        with patch('controllers.adaptive_scraper.AsyncWebCrawler') as mock_crawler_class:
            mock_crawler = AsyncMock()
            mock_crawler_class.return_value.__aenter__.return_value = mock_crawler
            
            mock_result = MagicMock()
            mock_result.success = False
            mock_result.extracted_content = None
            mock_crawler.arun.return_value = mock_result
            
            # Mock the fallback extraction
            with patch.object(adaptive_scraper, '_extract_content_fallback') as mock_fallback:
                mock_fallback.return_value = {
                    'title': 'Fallback Title',
                    'content': 'Fallback content',
                    'metadata': {}
                }
                
                content = await adaptive_scraper._extract_content_with_crawl4ai('https://example.com/page1')
                
                assert content is not None
                assert content['title'] == 'Fallback Title'
                assert content['content'] == 'Fallback content'
                mock_fallback.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_sitemap_extraction_integration(self, adaptive_scraper):
        """Test the full sitemap extraction pipeline."""
        # Mock site discovery to return sitemap URLs
        with patch.object(adaptive_scraper.site_discovery, 'find_sitemap') as mock_find:
            mock_find.return_value = ['https://example.com/sitemap.xml']
            
            # Mock URL extraction
            with patch.object(adaptive_scraper, 'extract_urls_from_sitemap') as mock_extract:
                mock_extract.return_value = [
                    'https://example.com/page1',
                    'https://example.com/page2'
                ]
                
                # Mock content extraction
                with patch.object(adaptive_scraper, '_extract_content_with_crawl4ai') as mock_content:
                    mock_content.side_effect = [
                        {
                            'title': 'Page 1 Title',
                            'content': 'Page 1 content',
                            'metadata': {'url': 'https://example.com/page1'}
                        },
                        {
                            'title': 'Page 2 Title', 
                            'content': 'Page 2 content',
                            'metadata': {'url': 'https://example.com/page2'}
                        }
                    ]
                    
                    results = await adaptive_scraper._process_sitemap_extraction(
                        'https://example.com',
                        url_limit=2
                    )
                    
                    assert len(results) == 2
                    assert results[0]['title'] == 'Page 1 Title'
                    assert results[1]['title'] == 'Page 2 Title'
                    assert all('content' in result for result in results)

    @pytest.mark.asyncio
    async def test_process_sitemap_extraction_no_sitemaps(self, adaptive_scraper):
        """Test sitemap extraction when no sitemaps are found."""
        # Mock site discovery to return empty list
        with patch.object(adaptive_scraper.site_discovery, 'find_sitemap') as mock_find:
            mock_find.return_value = []
            
            results = await adaptive_scraper._process_sitemap_extraction('https://example.com')
            
            assert results == []

    @pytest.mark.asyncio
    async def test_fallback_extraction(self, adaptive_scraper, sample_html_content):
        """Test the fallback extraction method."""
        # Mock HTTP fetch
        with patch('controllers.adaptive_scraper.fetch_html') as mock_fetch:
            mock_fetch.return_value = sample_html_content
            
            content = await adaptive_scraper._extract_content_fallback('https://example.com/page1')
            
            assert content is not None
            assert content['title'] == 'Test Page'
            assert 'test content' in content['content']
            assert content['metadata']['description'] == 'This is a test page'

    @pytest.mark.asyncio
    async def test_fallback_extraction_error_handling(self, adaptive_scraper):
        """Test error handling in fallback extraction."""
        # Mock HTTP fetch to raise an exception
        with patch('controllers.adaptive_scraper.fetch_html') as mock_fetch:
            mock_fetch.side_effect = Exception("Failed to fetch HTML")
            
            content = await adaptive_scraper._extract_content_fallback('https://example.com/page1')
            
            assert content is None


if __name__ == "__main__":
    pytest.main([__file__])
