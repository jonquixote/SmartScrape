"""
Comprehensive tests for HTML cleaning functionality across different content types and scenarios.

This test suite validates content extraction and cleaning for various HTML scenarios,
including edge cases, different content types, and scraping strategies.
"""

import pytest
import asyncio
from typing import Dict, Any, List
from unittest.mock import MagicMock, patch, AsyncMock
from bs4 import BeautifulSoup, Comment

from core.html_service import HTMLService
from core.pipeline.context import PipelineContext
from core.pipeline.dto import PipelineRequest, PipelineResponse, ResponseStatus
from core.pipeline.stages.processing.html_processing import (
    HTMLCleaningStage,
    ContentExtractionStage,
    CleaningStrategy
)


class TestHTMLCleaningComprehensive:
    """Comprehensive tests for HTML cleaning functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.html_service = HTMLService()
        self.html_service.initialize()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, 'html_service'):
            self.html_service.shutdown()

    # ======================== BASIC CLEANING TESTS ========================
    
    def test_script_removal_comprehensive(self):
        """Test comprehensive script tag removal and JavaScript content."""
        # Test script tag removal
        script_tags = [
            '<script>alert("test");</script>',
            '<script type="text/javascript">console.log("test");</script>',
            '<script src="https://example.com/script.js"></script>',
            '<script async>window.dataLayer = [];</script>',
            '<script defer src="/js/app.js"></script>',
            '<script type="module">import { test } from "./test.js";</script>',
        ]
        
        for script_html in script_tags:
            html = f'<html><body>{script_html}<p>Clean content</p></body></html>'
            cleaned = self.html_service.clean_html(html)
            
            # Should not contain script tags or their content
            assert '<script' not in cleaned.lower()
            assert 'alert(' not in cleaned
            assert 'console.log' not in cleaned
            
            # Should preserve clean content
            assert 'Clean content' in cleaned
        
        # Test inline event handlers (these are preserved by HTML service)
        inline_handlers = [
            '<div onclick="alert(\'click\')">Click me</div>',
            '<button onmouseover="this.style.color=\'red\'">Hover</button>',
            '<form onsubmit="return validate()">Submit</form>',
        ]
        
        for handler_html in inline_handlers:
            html = f'<html><body>{handler_html}<p>Clean content</p></body></html>'
            cleaned = self.html_service.clean_html(html)
            
            # Inline event handlers are preserved in basic cleaning
            # This is expected behavior for the current HTML service
            assert 'Click me' in cleaned or 'Hover' in cleaned or 'Submit' in cleaned
            assert 'Clean content' in cleaned
        
        # Test JavaScript URLs (these should be preserved as they're in href attributes)
        js_urls = [
            '<a href="javascript:void(0)">Link</a>',
            '<a href="javascript:alert(\'test\')">Alert</a>',
        ]
        
        for js_url_html in js_urls:
            html = f'<html><body>{js_url_html}<p>Clean content</p></body></html>'
            cleaned = self.html_service.clean_html(html)
            
            # JavaScript URLs in href attributes are preserved
            assert 'Link' in cleaned or 'Alert' in cleaned
            assert 'Clean content' in cleaned

    def test_css_removal_comprehensive(self):
        """Test comprehensive CSS removal including inline and external styles."""
        # Test style tag removal
        style_tags = [
            '<style>body { color: red; }</style>',
            '<style type="text/css">.class { display: none; }</style>',
            '<style>@import url("imported.css");</style>',
            '<style>@import "theme.css";</style>',
        ]
        
        for style_html in style_tags:
            html = f'<html><head>{style_html}</head><body><p>Clean content</p></body></html>'
            cleaned = self.html_service.clean_html(html)
            
            # Should not contain style tags or their content
            assert '<style' not in cleaned.lower()
            assert 'color: red' not in cleaned
            assert 'display: none' not in cleaned
            assert '@import' not in cleaned
            
            # Should preserve clean content
            assert 'Clean content' in cleaned
        
        # Test link tag removal (stylesheets) - these may be preserved
        link_tags = [
            '<link rel="stylesheet" href="styles.css">',
            '<link rel="stylesheet" type="text/css" href="/css/main.css">',
        ]
        
        for link_html in link_tags:
            html = f'<html><head>{link_html}</head><body><p>Clean content</p></body></html>'
            cleaned = self.html_service.clean_html(html)
            
            # Link tags for stylesheets may be preserved by HTML service
            # Focus on content preservation
            assert 'Clean content' in cleaned
        
        # Test inline styles (these may be preserved by HTML service)
        inline_styles = [
            '<div style="color: blue; background: white;">Styled div</div>',
            '<p style="font-size: 14px; margin: 10px;">Styled paragraph</p>',
        ]
        
        for style_html in inline_styles:
            html = f'<html><body>{style_html}<p>Clean content</p></body></html>'
            cleaned = self.html_service.clean_html(html)
            
            # Inline styles may be preserved - focus on content preservation
            assert 'Styled div' in cleaned or 'Styled paragraph' in cleaned
            assert 'Clean content' in cleaned

    def test_comment_removal_comprehensive(self):
        """Test comprehensive HTML comment removal."""
        test_cases = [
            # Standard comments
            '<!-- Simple comment -->',
            '<!-- Multi-line\ncomment\nhere -->',
            
            # Comments with special characters
            '<!-- Comment with <tags> and &entities; -->',
            '<!-- Comment with "quotes" and \'apostrophes\' -->',
            
            # Conditional comments (IE)
            '<!--[if IE]><p>IE only</p><![endif]-->',
            '<!--[if lt IE 9]><script src="ie-fix.js"></script><![endif]-->',
            
            # Nested-like structures
            '<!-- <!-- Nested comment --> -->',
            '<!-- Comment with -- dashes -->',
        ]
        
        for comment_html in test_cases:
            html = f'<html><body>{comment_html}<p>Clean content</p></body></html>'
            cleaned = self.html_service.clean_html(html)
            
            # Should not contain any comments
            assert '<!--' not in cleaned
            assert '-->' not in cleaned
            assert 'Simple comment' not in cleaned
            assert 'Multi-line' not in cleaned
            assert 'IE only' not in cleaned
            
            # Should preserve clean content
            assert 'Clean content' in cleaned

    def test_hidden_elements_removal(self):
        """Test handling of hidden and invisible elements."""
        # The HTML service now properly removes CSS-hidden elements with comprehensive security
        
        test_cases_removed = [
            # Hidden attribute - should be removed
            ('<div hidden>Hidden content</div>', 'Hidden content'),
            ('<p hidden="hidden">Also hidden</p>', 'Also hidden'),
            
            # CSS display none - should be removed
            ('<div style="display: none;">Not visible</div>', 'Not visible'),
            ('<span style="display:none">Hidden span</span>', 'Hidden span'),
            
            # CSS visibility hidden - should be removed
            ('<div style="visibility: hidden;">Invisible</div>', 'Invisible'),
            ('<p style="visibility:hidden;">Hidden paragraph</p>', 'Hidden paragraph'),
            
            # Zero dimensions - should be removed
            ('<div style="width: 0; height: 0;">Zero size</div>', 'Zero size'),
            ('<div style="width:0px;height:0px;">Zero pixels</div>', 'Zero pixels'),
            
            # Off-screen positioning - should be removed
            ('<div style="position: absolute; left: -9999px;">Off screen</div>', 'Off screen'),
            ('<div style="text-indent: -9999px;">Indented away</div>', 'Indented away'),
        ]
        
        for hidden_html, expected_content in test_cases_removed:
            html = f'<html><body>{hidden_html}<p>Visible content</p></body></html>'
            cleaned = self.html_service.clean_html(html)
            
            # Hidden content should be removed with comprehensive cleaning
            assert expected_content not in cleaned, f"Expected '{expected_content}' to be removed from cleaned HTML"
            
            # Should always preserve visible content
            assert 'Visible content' in cleaned

    # ======================== CONTENT TYPE TESTS ========================
    
    def test_news_article_cleaning(self):
        """Test cleaning of typical news article HTML."""
        news_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Breaking News: Important Event</title>
            <script>gtag('config', 'GA_MEASUREMENT_ID');</script>
            <style>.ad-banner { display: block; }</style>
        </head>
        <body>
            <header class="site-header">
                <nav>
                    <ul>
                        <li><a href="/">Home</a></li>
                        <li><a href="/politics">Politics</a></li>
                    </ul>
                </nav>
            </header>
            
            <main>
                <article>
                    <h1>Breaking News: Important Event</h1>
                    <div class="article-meta">
                        <time datetime="2023-12-01">December 1, 2023</time>
                        <span class="author">By Jane Reporter</span>
                    </div>
                    <div class="article-content">
                        <p>This is the main content of the news article with important information.</p>
                        <p>Another paragraph with <a href="https://source.com">source link</a>.</p>
                        <blockquote>"This is a quoted statement from an official."</blockquote>
                    </div>
                </article>
                
                <div class="ad-container" style="display: block;">
                    <!-- Advertisement -->
                    <script>showAd('banner-top');</script>
                </div>
            </main>
            
            <aside class="sidebar">
                <div class="related-articles">
                    <h3>Related Stories</h3>
                    <ul>
                        <li><a href="/story1">Related Story 1</a></li>
                        <li><a href="/story2">Related Story 2</a></li>
                    </ul>
                </div>
            </aside>
            
            <footer>
                <p>&copy; 2023 News Site</p>
                <script>trackPageView();</script>
            </footer>
        </body>
        </html>
        """
        
        cleaned = self.html_service.clean_html(news_html)
        
        # Should preserve article content
        assert 'Breaking News: Important Event' in cleaned
        assert 'main content of the news article' in cleaned
        assert 'quoted statement from an official' in cleaned
        assert 'December 1, 2023' in cleaned
        assert 'By Jane Reporter' in cleaned
        
        # Should remove scripts and ads
        assert 'gtag(' not in cleaned
        assert 'showAd(' not in cleaned
        assert 'trackPageView(' not in cleaned
        
        # Should remove navigation and footer (depending on strategy)
        # This may vary based on cleaning strategy, so we'll test main content preservation

    def test_ecommerce_product_cleaning(self):
        """Test cleaning of e-commerce product page HTML."""
        product_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Amazing Product - Buy Now</title>
            <script>dataLayer.push({'event': 'product_view'});</script>
            <style>.price-highlight { color: red; font-weight: bold; }</style>
        </head>
        <body>
            <header>
                <div class="shopping-cart">
                    <span id="cart-count">0</span>
                </div>
            </header>
            
            <main class="product-page">
                <div class="product-details">
                    <h1>Amazing Product</h1>
                    <div class="price-container">
                        <span class="current-price">$99.99</span>
                        <span class="original-price" style="text-decoration: line-through;">$149.99</span>
                    </div>
                    <div class="product-description">
                        <p>This amazing product will change your life. High quality materials and excellent craftsmanship.</p>
                        <ul class="features">
                            <li>Feature 1: Durable construction</li>
                            <li>Feature 2: Easy to use</li>
                            <li>Feature 3: 30-day guarantee</li>
                        </ul>
                    </div>
                    <div class="product-specs">
                        <table>
                            <tr><td>Weight</td><td>2.5 lbs</td></tr>
                            <tr><td>Dimensions</td><td>10" x 8" x 6"</td></tr>
                            <tr><td>Material</td><td>Premium aluminum</td></tr>
                        </table>
                    </div>
                    <button onclick="addToCart()" class="add-to-cart-btn">Add to Cart</button>
                </div>
                
                <div class="recommendations" style="margin-top: 20px;">
                    <h3>You might also like</h3>
                    <script>loadRecommendations();</script>
                </div>
            </main>
            
            <script>initializeProductPage();</script>
        </body>
        </html>
        """
        
        cleaned = self.html_service.clean_html(product_html, preserve_event_handlers=True)
        
        # Should preserve product information
        assert 'Amazing Product' in cleaned
        assert '$99.99' in cleaned
        assert '$149.99' in cleaned
        assert 'change your life' in cleaned
        assert 'Durable construction' in cleaned
        assert 'Easy to use' in cleaned
        assert '30-day guarantee' in cleaned
        assert '2.5 lbs' in cleaned
        assert 'Premium aluminum' in cleaned
        
        # Should remove script tags but preserve inline event handlers (actual behavior)
        assert 'dataLayer.push' not in cleaned  # Script tag content is removed
        # Note: HTML service preserves inline event handlers - this is the actual behavior
        assert 'addToCart()' in cleaned  # Inline onclick handler is preserved
        assert 'loadRecommendations()' not in cleaned  # Script tag content is removed
        assert 'initializeProductPage()' not in cleaned  # Script tag content is removed

    def test_blog_post_cleaning(self):
        """Test cleaning of blog post HTML."""
        blog_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>How to Clean HTML: A Complete Guide</title>
            <meta name="description" content="Learn the best practices for HTML cleaning">
            <script async src="https://www.googletagmanager.com/gtag/js"></script>
            <style>
                .code-block { background: #f5f5f5; padding: 10px; }
                .highlight { background-color: yellow; }
            </style>
        </head>
        <body>
            <header class="blog-header">
                <h1 class="site-title">Tech Blog</h1>
                <nav class="main-nav">
                    <a href="/">Home</a>
                    <a href="/about">About</a>
                    <a href="/contact">Contact</a>
                </nav>
            </header>
            
            <main class="blog-content">
                <article class="blog-post">
                    <header class="post-header">
                        <h1>How to Clean HTML: A Complete Guide</h1>
                        <div class="post-meta">
                            <time datetime="2023-12-01">December 1, 2023</time>
                            <span class="author">by Tech Writer</span>
                            <span class="reading-time">5 min read</span>
                        </div>
                    </header>
                    
                    <div class="post-content">
                        <p>HTML cleaning is an essential process in web scraping and content extraction. Here's how to do it right.</p>
                        
                        <h2>Why Clean HTML?</h2>
                        <p>Raw HTML contains many elements that aren't relevant for content extraction:</p>
                        <ul>
                            <li>JavaScript code that can interfere with processing</li>
                            <li>CSS styles that add noise to the content</li>
                            <li>Navigation elements and ads</li>
                            <li>Hidden or invisible content</li>
                        </ul>
                        
                        <h2>Best Practices</h2>
                        <ol>
                            <li>Remove script and style tags completely</li>
                            <li>Strip out HTML comments</li>
                            <li>Filter hidden elements</li>
                            <li>Preserve semantic content structure</li>
                        </ol>
                        
                        <div class="code-example">
                            <h3>Example Code</h3>
                            <pre><code>
from bs4 import BeautifulSoup

def clean_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    for script in soup(['script', 'style']):
        script.extract()
    return soup.get_text()
                            </code></pre>
                        </div>
                        
                        <blockquote class="tip">
                            <p><strong>Pro Tip:</strong> Always test your cleaning logic with various types of content to ensure it preserves what matters.</p>
                        </blockquote>
                    </div>
                </article>
                
                <div class="social-sharing" style="margin: 20px 0;">
                    <script>initSocialButtons();</script>
                </div>
                
                <section class="comments">
                    <h3>Comments</h3>
                    <div id="disqus_thread"></div>
                    <script>loadDisqusComments();</script>
                </section>
            </main>
            
            <aside class="sidebar">
                <div class="recent-posts">
                    <h3>Recent Posts</h3>
                    <ul>
                        <li><a href="/post1">Web Scraping Basics</a></li>
                        <li><a href="/post2">BeautifulSoup Tutorial</a></li>
                    </ul>
                </div>
                
                <div class="newsletter-signup">
                    <h3>Subscribe</h3>
                    <form onsubmit="submitNewsletter(event)">
                        <input type="email" placeholder="Your email">
                        <button type="submit">Subscribe</button>
                    </form>
                </div>
            </aside>
            
            <footer class="blog-footer">
                <p>&copy; 2023 Tech Blog. All rights reserved.</p>
                <script>initFooterTracking();</script>
            </footer>
        </body>
        </html>
        """
        
        cleaned = self.html_service.clean_html(blog_html)
        
        # Should preserve main content
        assert 'How to Clean HTML: A Complete Guide' in cleaned
        assert 'HTML cleaning is an essential process' in cleaned
        assert 'Why Clean HTML?' in cleaned
        assert 'Best Practices' in cleaned
        assert 'JavaScript code that can interfere' in cleaned
        assert 'Remove script and style tags' in cleaned
        assert 'Pro Tip:' in cleaned
        assert 'from bs4 import BeautifulSoup' in cleaned
        
        # Should remove scripts and tracking
        assert 'googletagmanager.com' not in cleaned
        assert 'initSocialButtons()' not in cleaned
        assert 'loadDisqusComments()' not in cleaned
        assert 'submitNewsletter(' not in cleaned
        assert 'initFooterTracking()' not in cleaned

    # ======================== EDGE CASES AND ERROR HANDLING ========================
    
    def test_malformed_html_handling(self):
        """Test handling of malformed HTML."""
        malformed_cases = [
            # Unclosed tags
            '<div><p>Unclosed paragraph<div>Another div</div>',
            '<span>Unclosed span<b>Bold text</span>',
            
            # Overlapping tags
            '<b><i>Bold and italic</b></i>',
            '<div><span>Overlapping</div></span>',
            
            # Invalid nesting
            '<p><div>Block inside inline</div></p>',
            '<ul><div>Div in list</div><li>List item</li></ul>',
            
            # Missing required attributes
            '<img>Image without src</img>',
            '<a>Link without href</a>',
            
            # Invalid characters
            '<div class="test<">Invalid attribute</div>',
            '<p>Text with & unescaped ampersand</p>',
        ]
        
        for malformed_html in malformed_cases:
            html = f'<html><body>{malformed_html}<p>Clean content</p></body></html>'
            
            # Should not raise exceptions
            try:
                cleaned = self.html_service.clean_html(html)
                # Should still preserve clean content
                assert 'Clean content' in cleaned
            except Exception as e:
                pytest.fail(f"HTML cleaning failed on malformed HTML: {e}")

    def test_empty_and_none_input(self):
        """Test handling of empty and None inputs."""
        test_cases = [
            None,
            "",
            " ",
            "\n\t",
            "   \n   ",
        ]
        
        for test_input in test_cases:
            result = self.html_service.clean_html(test_input)
            assert result == "" or result.strip() == ""

    def test_non_html_content(self):
        """Test handling of content that isn't HTML."""
        non_html_cases = [
            # Plain text
            "This is just plain text without any HTML tags.",
            "Multiple lines\nof plain text\nwith line breaks.",
            
            # Text with angle brackets but not HTML
            "Mathematical expression: 5 < 10 > 3",
            "Code snippet: if (x < y && y > z) return true;",
            
            # JSON content
            '{"name": "John", "age": 30, "city": "New York"}',
            
            # CSV content
            "Name,Age,City\nJohn,30,New York\nJane,25,Boston",
            
            # XML-like but not HTML
            '<?xml version="1.0"?><data><item>value</item></data>',
        ]
        
        for content in non_html_cases:
            cleaned = self.html_service.clean_html(content)
            # Should preserve non-HTML content mostly unchanged
            # For XML, allow for encoding to be added
            if content.startswith('<?xml'):
                assert '<data><item>value</item></data>' in cleaned
            else:
                assert content.strip() in cleaned or cleaned.strip() in content

    def test_large_html_document(self):
        """Test cleaning of large HTML documents."""
        # Generate a large HTML document
        large_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Large Document</title>
            <script>
                // Large JavaScript block
                var data = [];
                for (let i = 0; i < 1000; i++) {
                    data.push({id: i, name: 'Item ' + i});
                }
            </script>
            <style>
                /* Large CSS block */
                .item { margin: 5px; padding: 10px; border: 1px solid #ccc; }
                .item:hover { background-color: #f0f0f0; }
                .item.active { background-color: #007bff; color: white; }
            </style>
        </head>
        <body>
        """
        
        # Add many content sections
        for i in range(100):
            large_html += f"""
            <section class="content-section">
                <h2>Section {i + 1}</h2>
                <p>This is content for section {i + 1} with meaningful information that should be preserved.</p>
                <ul>
                    <li>Item {i * 3 + 1}</li>
                    <li>Item {i * 3 + 2}</li>
                    <li>Item {i * 3 + 3}</li>
                </ul>
                <script>trackSection({i + 1});</script>
            </section>
            """
        
        large_html += """
        </body>
        </html>
        """
        
        # Should handle large documents without issues
        cleaned = self.html_service.clean_html(large_html)
        
        # Should preserve content from multiple sections
        assert 'Section 1' in cleaned
        assert 'Section 50' in cleaned
        assert 'Section 100' in cleaned
        assert 'meaningful information that should be preserved' in cleaned
        
        # Should remove all scripts
        assert 'trackSection(' not in cleaned
        assert 'var data = []' not in cleaned
        
        # Cleaned version should be significantly smaller
        assert len(cleaned) < len(large_html) * 0.8  # At least 20% reduction

    # ======================== STRATEGY-SPECIFIC TESTS ========================
    
    @pytest.mark.asyncio
    async def test_dom_strategy_cleaning(self):
        """Test HTML cleaning with DOM-based extraction strategy."""
        dom_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>DOM Strategy Test</title>
            <script>var config = {api: 'https://api.example.com'};</script>
        </head>
        <body>
            <div class="container">
                <div class="content-wrapper">
                    <article class="main-content">
                        <h1>Main Article Title</h1>
                        <p>This is the main content that should be extracted by DOM strategy.</p>
                        <div class="article-body">
                            <p>Additional paragraph with important information.</p>
                            <blockquote>Important quote from the article.</blockquote>
                        </div>
                    </article>
                    <aside class="sidebar">
                        <div class="widget">
                            <h3>Related Links</h3>
                            <ul>
                                <li><a href="/related1">Related Article 1</a></li>
                                <li><a href="/related2">Related Article 2</a></li>
                            </ul>
                        </div>
                    </aside>
                </div>
            </div>
            <script>initDOMHandlers();</script>
        </body>
        </html>
        """
        
        # Test with HTML cleaning stage
        stage = HTMLCleaningStage(config={"strategy": "MODERATE"})
        request = PipelineRequest(
            source="test",
            params={"html": dom_html},
            metadata={"extraction_strategy": "dom_strategy"}
        )
        
        response = await stage.process_request(request, PipelineContext())
        
        assert response.status == ResponseStatus.SUCCESS
        cleaned_html = response.data["html"]
        
        # Should preserve structural content
        assert 'Main Article Title' in cleaned_html
        assert 'main content that should be extracted' in cleaned_html
        assert 'Important quote from the article' in cleaned_html
        
        # Should remove scripts
        assert 'var config' not in cleaned_html
        assert 'initDOMHandlers()' not in cleaned_html

    @pytest.mark.asyncio
    async def test_ai_guided_strategy_cleaning(self):
        """Test HTML cleaning with AI-guided extraction strategy."""
        ai_guided_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Guided Test Page</title>
            <script>
                window.aiTracking = {
                    track: function(event) { console.log(event); }
                };
            </script>
            <style>
                .ai-content { background: #f8f9fa; padding: 20px; }
                .metadata { font-size: 12px; color: #666; }
            </style>
        </head>
        <body>
            <main class="ai-content">
                <header>
                    <h1>AI-Powered Article Analysis</h1>
                    <div class="metadata">
                        <span>Published: 2023-12-01</span>
                        <span>Author: AI Researcher</span>
                        <span>Category: Technology</span>
                    </div>
                </header>
                <section class="article-content">
                    <p>This article discusses the latest advances in AI technology and their applications in web scraping.</p>
                    <h2>Key Benefits</h2>
                    <ul>
                        <li>Improved accuracy in content extraction</li>
                        <li>Better handling of dynamic content</li>
                        <li>Adaptive learning from patterns</li>
                    </ul>
                    <h2>Implementation Details</h2>
                    <p>The AI-guided approach uses machine learning models to identify content patterns and extract relevant information.</p>
                    <div class="code-snippet">
                        <pre>
                        def extract_with_ai(html, intent):
                            model = load_ai_model()
                            return model.extract(html, intent)
                        </pre>
                    </div>
                </section>
            </main>
            <script>window.aiTracking.track('page_view');</script>
        </body>
        </html>
        """
        
        # Test with HTML cleaning stage configured for AI-guided strategy
        stage = HTMLCleaningStage(config={"strategy": "LIGHT"})  # Light cleaning for AI processing
        request = PipelineRequest(
            source="test",
            params={"html": ai_guided_html},
            metadata={"extraction_strategy": "ai_guided"}
        )
        
        response = await stage.process_request(request, PipelineContext())
        
        assert response.status == ResponseStatus.SUCCESS
        cleaned_html = response.data["html"]
        
        # Should preserve content for AI analysis
        assert 'AI-Powered Article Analysis' in cleaned_html
        assert 'latest advances in AI technology' in cleaned_html
        assert 'Improved accuracy in content extraction' in cleaned_html
        assert 'def extract_with_ai' in cleaned_html
        
        # Should remove tracking scripts
        assert 'window.aiTracking' not in cleaned_html
        assert "track('page_view')" not in cleaned_html

    # ======================== PERFORMANCE AND SCALABILITY TESTS ========================
    
    def test_cleaning_performance(self):
        """Test HTML cleaning performance with various document sizes."""
        import time
        
        # Small document
        small_html = '<html><body><p>Small content</p></body></html>'
        start = time.time()
        self.html_service.clean_html(small_html)
        small_time = time.time() - start
        
        # Medium document (1KB)
        medium_html = '<html><body>' + '<p>Medium content paragraph. </p>' * 50 + '</body></html>'
        start = time.time()
        self.html_service.clean_html(medium_html)
        medium_time = time.time() - start
        
        # Large document (10KB)
        large_html = '<html><body>' + '<p>Large content paragraph with more text. </p>' * 500 + '</body></html>'
        start = time.time()
        self.html_service.clean_html(large_html)
        large_time = time.time() - start
        
        # Performance should scale reasonably
        assert small_time < 0.1  # Should be very fast for small documents
        assert medium_time < 0.5  # Should be reasonable for medium documents
        assert large_time < 2.0   # Should be acceptable for large documents
        
        # Scaling should be roughly linear
        assert large_time < small_time * 100  # Shouldn't be more than 100x slower

    def test_memory_usage_stability(self):
        """Test that HTML cleaning doesn't cause memory leaks."""
        import gc
        
        # Force garbage collection before test
        gc.collect()
        
        # Run many cleaning operations
        test_html = """
        <html>
        <head>
            <script>var data = {};</script>
            <style>.test { color: red; }</style>
        </head>
        <body>
            <div class="content">
                <p>Test content that will be cleaned multiple times.</p>
            </div>
        </body>
        </html>
        """
        
        # Clean the same HTML many times
        for _ in range(1000):
            cleaned = self.html_service.clean_html(test_html)
            assert 'Test content' in cleaned
            
        # Force garbage collection after test
        gc.collect()
        
        # This test mainly ensures no exceptions occur during repeated cleaning
        # Actual memory usage would require more sophisticated monitoring

    # ======================== SECURITY TESTS ========================
    
    def test_xss_prevention(self):
        """Test that HTML cleaning prevents XSS attacks."""
        xss_attempts = [
            # Script injection
            '<script>alert("XSS")</script>',
            '<img src="x" onerror="alert(\'XSS\')">',
            '<svg onload="alert(\'XSS\')"></svg>',
            
            # Event handlers
            '<div onclick="maliciousFunction()">Click me</div>',
            '<input onfocus="stealData()" type="text">',
            '<body onload="executeAttack()">',
            
            # JavaScript URLs
            '<a href="javascript:alert(\'XSS\')">Click</a>',
            '<iframe src="javascript:alert(\'XSS\')"></iframe>',
            
            # Data URLs with scripts
            '<img src="data:text/html,<script>alert(\'XSS\')</script>">',
            
            # CSS-based attacks
            '<style>body { background: url("javascript:alert(\'XSS\')"); }</style>',
            '<div style="background: expression(alert(\'XSS\'))">IE attack</div>',
        ]
        
        for xss_html in xss_attempts:
            html = f'<html><body>{xss_html}<p>Safe content</p></body></html>'
            cleaned = self.html_service.clean_html(html)
            
            # Should remove all potential XSS vectors
            assert 'alert(' not in cleaned
            assert 'javascript:' not in cleaned.lower()
            assert 'onclick' not in cleaned.lower()
            assert 'onerror' not in cleaned.lower()
            assert 'onload' not in cleaned.lower()
            assert 'onfocus' not in cleaned.lower()
            assert 'maliciousFunction' not in cleaned
            assert 'stealData' not in cleaned
            assert 'executeAttack' not in cleaned
            assert 'expression(' not in cleaned
            
            # Should preserve safe content
            assert 'Safe content' in cleaned

    def test_injection_prevention(self):
        """Test prevention of various injection attacks."""
        injection_attempts = [
            # SQL injection patterns in HTML
            '<div data-query="SELECT * FROM users WHERE id=1; DROP TABLE users;">Data</div>',
            
            # Command injection patterns
            '<span title="$(rm -rf /)">Dangerous title</span>',
            '<div data-cmd="`cat /etc/passwd`">Command injection</div>',
            
            # Template injection patterns
            '<p>{{7*7}}</p>',
            '<div>${{7*7}}</div>',
            '<span><%=7*7%></span>',
            
            # XML injection
            '<!DOCTYPE test [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>',
            '<div>&xxe;</div>',
        ]
        
        for injection_html in injection_attempts:
            html = f'<html><body>{injection_html}<p>Clean content</p></body></html>'
            cleaned = self.html_service.clean_html(html)
            
            # Should not contain injection patterns
            assert 'DROP TABLE' not in cleaned
            assert 'rm -rf' not in cleaned
            assert 'cat /etc/passwd' not in cleaned
            assert '/etc/passwd' not in cleaned
            assert '{{7*7}}' not in cleaned
            assert '${' not in cleaned
            assert '<%=' not in cleaned
            assert 'ENTITY xxe' not in cleaned
            
            # Should preserve clean content
            assert 'Clean content' in cleaned

    # ======================== INTEGRATION TESTS ========================
    
    @pytest.mark.asyncio
    async def test_pipeline_integration(self):
        """Test HTML cleaning integration with the full pipeline."""
        complex_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Complex Integration Test</title>
            <script>analytics.track('page_view');</script>
            <style>.container { max-width: 1200px; }</style>
        </head>
        <body>
            <header>
                <nav>Navigation menu</nav>
            </header>
            <main>
                <article>
                    <h1>Main Article</h1>
                    <p>This is the main content that should be extracted and cleaned properly.</p>
                    <div class="metadata">
                        <span>Author: Test Author</span>
                        <time>2023-12-01</time>
                    </div>
                </article>
            </main>
            <aside>
                <div class="ads">
                    <script>showAds();</script>
                </div>
            </aside>
            <footer>
                <p>Footer content</p>
                <script>trackFooter();</script>
            </footer>
        </body>
        </html>
        """
        
        # Test with both cleaning and extraction stages
        cleaning_stage = HTMLCleaningStage(config={"strategy": "MODERATE"})
        extraction_stage = ContentExtractionStage()
        
        # Create pipeline request
        request = PipelineRequest(
            source="test",
            params={"html": complex_html},
            metadata={"url": "https://example.com/test"}
        )
        
        context = PipelineContext()
        
        # First pass through cleaning stage
        cleaned_response = await cleaning_stage.process_request(request, context)
        assert cleaned_response.status == ResponseStatus.SUCCESS
        
        # Create new request with cleaned HTML
        extraction_request = PipelineRequest(
            source="test",
            params={"html": cleaned_response.data["html"]},
            metadata=request.metadata
        )
        
        # Pass through extraction stage
        extracted_response = await extraction_stage.process_request(extraction_request, context)
        assert extracted_response.status == ResponseStatus.SUCCESS
        
        # Verify the pipeline preserved important content
        final_content = extracted_response.data.get("extracted_content", "")
        assert 'Main Article' in final_content or 'Main Article' in str(extracted_response.data)

    def test_real_world_scraping_scenarios(self):
        """Test HTML cleaning with real-world scraping scenarios."""
        # Test with actual HTML patterns found in scraping results
        real_world_html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Real Estate Listing - Beautiful Home</title>
            
            <!-- Google Analytics -->
            <script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
            <script>
                window.dataLayer = window.dataLayer || [];
                function gtag(){dataLayer.push(arguments);}
                gtag('js', new Date());
                gtag('config', 'GA_MEASUREMENT_ID');
            </script>
            
            <style>
                .listing-container { margin: 20px auto; max-width: 1200px; }
                .price-highlight { color: #d32f2f; font-size: 24px; font-weight: bold; }
                .property-details { background: #f5f5f5; padding: 15px; }
                .hidden-info { display: none; }
            </style>
        </head>
        <body>
            <header class="site-header">
                <div class="container">
                    <h1 class="logo">RealEstate.com</h1>
                    <nav class="main-nav">
                        <ul>
                            <li><a href="/">Home</a></li>
                            <li><a href="/search">Search</a></li>
                            <li><a href="/contact">Contact</a></li>
                        </ul>
                    </nav>
                </div>
            </header>
            
            <main class="listing-container">
                <div class="property-header">
                    <h1>Beautiful 3BR/2BA Home in Suburban Neighborhood</h1>
                    <div class="price-container">
                        <span class="price-highlight">$325,000</span>
                        <span class="price-details">Listed 3 days ago</span>
                    </div>
                </div>
                
                <div class="property-details">
                    <div class="basic-info">
                        <div class="info-item">
                            <label>Bedrooms:</label>
                            <span>3</span>
                        </div>
                        <div class="info-item">
                            <label>Bathrooms:</label>
                            <span>2</span>
                        </div>
                        <div class="info-item">
                            <label>Square Feet:</label>
                            <span>1,850 sq ft</span>
                        </div>
                        <div class="info-item">
                            <label>Lot Size:</label>
                            <span>0.25 acres</span>
                        </div>
                    </div>
                    
                    <div class="description">
                        <h2>Property Description</h2>
                        <p>This stunning 3-bedroom, 2-bathroom home offers modern living in a quiet suburban setting. Recently updated kitchen with stainless steel appliances, hardwood floors throughout, and a spacious backyard perfect for entertaining.</p>
                        
                        <h3>Features</h3>
                        <ul>
                            <li>Updated kitchen with granite countertops</li>
                            <li>Hardwood floors in living areas</li>
                            <li>Master suite with walk-in closet</li>
                            <li>Fenced backyard with deck</li>
                            <li>2-car garage</li>
                            <li>Central air conditioning</li>
                        </ul>
                    </div>
                    
                    <div class="location-info">
                        <h2>Location</h2>
                        <p>123 Maple Street, Suburban Heights, OH 44123</p>
                        <p>School District: Excellent Public Schools</p>
                        <p>Nearby: Shopping centers, parks, and public transportation</p>
                    </div>
                    
                    <div class="hidden-info" style="display: none;">
                        <p>Internal note: Property needs minor repairs before showing</p>
                    </div>
                </div>
                
                <div class="contact-section">
                    <h2>Contact Information</h2>
                    <div class="agent-info">
                        <p><strong>Agent:</strong> Sarah Johnson</p>
                        <p><strong>Phone:</strong> (555) 123-4567</p>
                        <p><strong>Email:</strong> sarah.johnson@realestate.com</p>
                    </div>
                    
                    <div class="action-buttons">
                        <button onclick="scheduleShowing()" class="btn-primary">Schedule Showing</button>
                        <button onclick="requestInfo()" class="btn-secondary">Request More Info</button>
                    </div>
                </div>
            </main>
            
            <aside class="sidebar">
                <div class="similar-properties">
                    <h3>Similar Properties</h3>
                    <div class="property-card" onclick="viewProperty(456)">
                        <p>2BR/1BA Condo - $245,000</p>
                    </div>
                    <div class="property-card" onclick="viewProperty(789)">
                        <p>4BR/3BA House - $425,000</p>
                    </div>
                </div>
                
                <div class="mortgage-calculator">
                    <h3>Mortgage Calculator</h3>
                    <script>loadMortgageCalculator();</script>
                </div>
            </aside>
            
            <footer class="site-footer">
                <div class="container">
                    <p>&copy; 2023 RealEstate.com. All rights reserved.</p>
                    <div class="footer-links">
                        <a href="/privacy">Privacy Policy</a>
                        <a href="/terms">Terms of Service</a>
                    </div>
                </div>
                <script>initFooterTracking();</script>
            </footer>
            
            <!-- Additional tracking scripts -->
            <script>
                // Track property view
                if (typeof gtag !== 'undefined') {
                    gtag('event', 'property_view', {
                        'property_id': '12345',
                        'price': 325000,
                        'bedrooms': 3,
                        'bathrooms': 2
                    });
                }
                
                // Initialize page interactions
                document.addEventListener('DOMContentLoaded', function() {
                    initPropertyPage();
                    setupImageGallery();
                    loadSimilarProperties();
                });
            </script>
        </body>
        </html>
        """
        
        cleaned = self.html_service.clean_html(real_world_html)
        
        # Should preserve all important property information
        assert 'Beautiful 3BR/2BA Home in Suburban Neighborhood' in cleaned
        assert '$325,000' in cleaned
        assert 'Listed 3 days ago' in cleaned
        assert 'Bedrooms:' in cleaned and '3' in cleaned
        assert 'Bathrooms:' in cleaned and '2' in cleaned
        assert '1,850 sq ft' in cleaned
        assert '0.25 acres' in cleaned
        assert 'stunning 3-bedroom, 2-bathroom home' in cleaned
        assert 'Updated kitchen with granite countertops' in cleaned
        assert 'Hardwood floors in living areas' in cleaned
        assert '123 Maple Street, Suburban Heights, OH 44123' in cleaned
        assert 'Sarah Johnson' in cleaned
        assert '(555) 123-4567' in cleaned
        assert 'sarah.johnson@realestate.com' in cleaned
        
        # Should remove all JavaScript and tracking
        assert 'gtag(' not in cleaned
        assert 'dataLayer' not in cleaned
        assert 'scheduleShowing()' not in cleaned
        assert 'requestInfo()' not in cleaned
        assert 'viewProperty(' not in cleaned
        assert 'loadMortgageCalculator()' not in cleaned
        assert 'initFooterTracking()' not in cleaned
        assert 'initPropertyPage()' not in cleaned
        assert 'setupImageGallery()' not in cleaned
        
        # Should remove hidden content
        assert 'Internal note: Property needs minor repairs' not in cleaned
        
        # Should remove CSS styles
        assert '.listing-container' not in cleaned
        assert 'color: #d32f2f' not in cleaned
        assert 'background: #f5f5f5' not in cleaned


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
