"""
Tests for the metadata extraction functionality.

This module contains tests for the various metadata extractors including:
- HTML meta tag extraction
- OpenGraph and Twitter Card extraction
- JSON-LD structured data extraction 
- Microdata extraction
- JavaScript data extraction
- Metadata consolidation and normalization
"""

import unittest
from unittest.mock import MagicMock, patch
import json
import os
from bs4 import BeautifulSoup

from extraction.metadata_extractor import (
    CompositeMetadataExtractor,
    HTMLMetaExtractor,
    OpenGraphExtractor,
    JSONLDExtractor,
    MicrodataExtractor
)
from extraction.helpers.javascript_extractor import JavaScriptExtractor
from strategies.core.strategy_context import StrategyContext

class TestHTMLMetaExtractor(unittest.TestCase):
    """Tests for the HTML meta tag extractor."""
    
    def setUp(self):
        """Set up test environment."""
        self.extractor = HTMLMetaExtractor()
        self.extractor.initialize()
        
        # Sample HTML with meta tags
        self.html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Page Title</title>
            <meta name="description" content="Test page description">
            <meta name="keywords" content="test, page, keywords">
            <meta name="author" content="Test Author">
            <meta http-equiv="content-language" content="en">
            <meta charset="UTF-8">
            <link rel="canonical" href="https://example.com/test">
            <link rel="icon" href="/favicon.ico">
        </head>
        <body>
            <h1>Test Page</h1>
            <p>This is a test page.</p>
        </body>
        </html>
        """
    
    def test_extract_title_and_description(self):
        """Test extraction of title and description."""
        result = self.extractor.extract_title_and_description(self.html)
        self.assertEqual(result.get("title"), "Test Page Title")
        self.assertEqual(result.get("description"), "Test page description")
    
    def test_extract_meta_tags(self):
        """Test extraction of meta tags."""
        result = self.extractor.extract_meta_tags(self.html)
        self.assertEqual(result.get("description"), "Test page description")
        self.assertEqual(result.get("keywords"), "test, page, keywords")
        self.assertEqual(result.get("author"), "Test Author")
        self.assertEqual(result.get("http-equiv:content-language"), "en")
        self.assertEqual(result.get("charset"), "UTF-8")
    
    def test_extract_meta_keywords(self):
        """Test extraction of meta keywords."""
        result = self.extractor.extract_meta_keywords(self.html)
        self.assertEqual(result, "test, page, keywords")
    
    def test_extract_canonical_url(self):
        """Test extraction of canonical URL."""
        result = self.extractor.extract_canonical_url(self.html)
        self.assertEqual(result, "https://example.com/test")
    
    def test_extract_favicon(self):
        """Test extraction of favicon."""
        result = self.extractor.extract_favicon(self.html)
        self.assertEqual(result, "/favicon.ico")
    
    def test_extract_all(self):
        """Test complete extraction of HTML meta information."""
        result = self.extractor.extract(self.html)
        self.assertEqual(result.get("title"), "Test Page Title")
        self.assertEqual(result.get("description"), "Test page description")
        self.assertEqual(result.get("keywords"), "test, page, keywords")
        self.assertEqual(result.get("canonical_url"), "https://example.com/test")
        self.assertEqual(result.get("favicon"), "/favicon.ico")
        self.assertTrue("_metadata" in result)
    
    def test_empty_content(self):
        """Test handling of empty content."""
        result = self.extractor.extract("")
        self.assertTrue("_metadata" in result)
        self.assertFalse(result.get("_metadata").get("success", True))
    
    def tearDown(self):
        """Clean up test environment."""
        self.extractor.shutdown()


class TestOpenGraphExtractor(unittest.TestCase):
    """Tests for the Open Graph and Twitter Card extractor."""
    
    def setUp(self):
        """Set up test environment."""
        self.extractor = OpenGraphExtractor()
        self.extractor.initialize()
        
        # Sample HTML with Open Graph and Twitter Card meta tags
        self.html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Regular Title</title>
            <meta property="og:title" content="OG Title">
            <meta property="og:description" content="OG description">
            <meta property="og:url" content="https://example.com/og">
            <meta property="og:image" content="https://example.com/image.jpg">
            <meta property="og:image:width" content="800">
            <meta property="og:image:height" content="600">
            <meta property="og:type" content="website">
            <meta property="og:site_name" content="Example Site">
            <meta property="og:locale" content="en_US">
            
            <meta name="twitter:card" content="summary_large_image">
            <meta name="twitter:title" content="Twitter Title">
            <meta name="twitter:description" content="Twitter description">
            <meta name="twitter:image" content="https://example.com/twitter-image.jpg">
            <meta name="twitter:site" content="@example">
        </head>
        <body>
            <h1>Test Page</h1>
        </body>
        </html>
        """
    
    def test_extract_og_metadata(self):
        """Test extraction of Open Graph metadata."""
        result = self.extractor.extract_og_metadata(self.html)
        self.assertEqual(result.get("title"), "OG Title")
        self.assertEqual(result.get("description"), "OG description")
        self.assertEqual(result.get("url"), "https://example.com/og")
        self.assertEqual(result.get("image"), "https://example.com/image.jpg")
        self.assertEqual(result.get("type"), "website")
        self.assertEqual(result.get("site_name"), "Example Site")
        self.assertEqual(result.get("locale"), "en_US")
        
        # Check image data structure
        self.assertTrue("image_data" in result)
        self.assertEqual(result["image_data"]["url"], "https://example.com/image.jpg")
        self.assertEqual(result["image_data"]["width"], "800")
        self.assertEqual(result["image_data"]["height"], "600")
    
    def test_extract_twitter_cards(self):
        """Test extraction of Twitter Card metadata."""
        result = self.extractor.extract_twitter_cards(self.html)
        self.assertEqual(result.get("twitter_card"), "summary_large_image")
        self.assertEqual(result.get("twitter_title"), "Twitter Title")
        self.assertEqual(result.get("twitter_description"), "Twitter description")
        self.assertEqual(result.get("twitter_image"), "https://example.com/twitter-image.jpg")
        self.assertEqual(result.get("twitter_site"), "@example")
    
    def test_extract_sharing_image(self):
        """Test extraction of sharing image."""
        result = self.extractor.extract_sharing_image(self.html)
        self.assertEqual(result, "https://example.com/image.jpg")
        
        # Test fallback to Twitter image
        html_without_og = self.html.replace('og:image', 'og:img')
        result = self.extractor.extract_sharing_image(html_without_og)
        self.assertEqual(result, "https://example.com/twitter-image.jpg")
    
    def test_map_social_metadata(self):
        """Test mapping of social metadata to standard format."""
        og_data = {
            "title": "OG Title",
            "twitter_title": "Twitter Title",
            "image": "https://example.com/image.jpg",
            "image_data": {"url": "https://example.com/image.jpg", "width": "800"}
        }
        
        result = self.extractor.map_social_metadata(og_data)
        self.assertEqual(result.get("title"), "OG Title")
        self.assertEqual(result.get("image"), "https://example.com/image.jpg")
        self.assertTrue("image_data" in result)
    
    def test_extract_all(self):
        """Test complete extraction of social metadata."""
        result = self.extractor.extract(self.html)
        self.assertEqual(result.get("title"), "OG Title")
        self.assertEqual(result.get("description"), "OG description")
        self.assertEqual(result.get("image"), "https://example.com/image.jpg")
        self.assertEqual(result.get("type"), "website")
        self.assertEqual(result.get("site_name"), "Example Site")
        self.assertTrue("twitter_title" in result)
        self.assertTrue("_metadata" in result)
    
    def tearDown(self):
        """Clean up test environment."""
        self.extractor.shutdown()


class TestJSONLDExtractor(unittest.TestCase):
    """Tests for the JSON-LD structured data extractor."""
    
    def setUp(self):
        """Set up test environment."""
        self.extractor = JSONLDExtractor()
        self.extractor.initialize()
        
        # Sample HTML with JSON-LD data
        self.html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>JSON-LD Test</title>
            <script type="application/ld+json">
            {
                "@context": "https://schema.org",
                "@type": "Article",
                "headline": "JSON-LD Article Title",
                "description": "This is a JSON-LD article description",
                "image": "https://example.com/article-image.jpg",
                "datePublished": "2023-01-15T08:00:00+08:00",
                "dateModified": "2023-01-16T10:30:00+08:00",
                "author": {
                    "@type": "Person",
                    "name": "John Doe",
                    "url": "https://example.com/john"
                },
                "publisher": {
                    "@type": "Organization",
                    "name": "Example Organization",
                    "logo": {
                        "@type": "ImageObject",
                        "url": "https://example.com/logo.png"
                    }
                },
                "mainEntityOfPage": {
                    "@type": "WebPage",
                    "@id": "https://example.com/article"
                }
            }
            </script>
            <script type="application/ld+json">
            {
                "@context": "https://schema.org",
                "@type": "BreadcrumbList",
                "itemListElement": [
                    {
                        "@type": "ListItem",
                        "position": 1,
                        "name": "Home",
                        "item": "https://example.com/"
                    },
                    {
                        "@type": "ListItem",
                        "position": 2,
                        "name": "Articles",
                        "item": "https://example.com/articles/"
                    }
                ]
            }
            </script>
        </head>
        <body>
            <h1>JSON-LD Test</h1>
        </body>
        </html>
        """
    
    def test_extract_jsonld_blocks(self):
        """Test extraction of JSON-LD script blocks."""
        result = self.extractor.extract_jsonld_blocks(self.html)
        self.assertEqual(len(result), 2)
        self.assertIn('"@type": "Article"', result[0])
        self.assertIn('"@type": "BreadcrumbList"', result[1])
    
    def test_parse_jsonld_data(self):
        """Test parsing of JSON-LD data."""
        blocks = self.extractor.extract_jsonld_blocks(self.html)
        
        article = self.extractor.parse_jsonld_data(blocks[0])
        self.assertEqual(article.get("@type"), "Article")
        self.assertEqual(article.get("headline"), "JSON-LD Article Title")
        self.assertEqual(article.get("author").get("name"), "John Doe")
        
        breadcrumbs = self.extractor.parse_jsonld_data(blocks[1])
        self.assertEqual(breadcrumbs.get("@type"), "BreadcrumbList")
        self.assertEqual(len(breadcrumbs.get("itemListElement")), 2)
    
    def test_map_schema_types(self):
        """Test mapping of schema types."""
        blocks = self.extractor.extract_jsonld_blocks(self.html)
        items = [self.extractor.parse_jsonld_data(block) for block in blocks]
        
        result = self.extractor.map_schema_types(items)
        self.assertTrue("Article" in result)
        self.assertTrue("BreadcrumbList" in result)
        self.assertEqual(len(result["Article"]), 1)
        self.assertEqual(len(result["BreadcrumbList"]), 1)
    
    def test_extract_entity_data(self):
        """Test extraction of entity data."""
        blocks = self.extractor.extract_jsonld_blocks(self.html)
        article = self.extractor.parse_jsonld_data(blocks[0])
        
        result = self.extractor.extract_entity_data(article)
        self.assertEqual(result.get("@type"), "Article")
        self.assertEqual(result.get("headline"), "JSON-LD Article Title")
        self.assertEqual(result.get("author"), "John Doe")  # Should be simplified from object to string
    
    def test_extract_all(self):
        """Test complete extraction of JSON-LD data."""
        result = self.extractor.extract(self.html)
        
        self.assertEqual(len(result.get("items")), 2)
        self.assertTrue("Article" in result.get("types"))
        self.assertTrue("BreadcrumbList" in result.get("types"))
        
        # Check primary item fields
        self.assertEqual(result.get("title"), "JSON-LD Article Title")
        self.assertEqual(result.get("description"), "This is a JSON-LD article description")
        self.assertEqual(result.get("image"), "https://example.com/article-image.jpg")
        self.assertEqual(result.get("datePublished"), "2023-01-15T08:00:00+08:00")
        self.assertEqual(result.get("author"), "John Doe")
        
        self.assertTrue("_metadata" in result)
        self.assertEqual(result["_metadata"]["item_count"], 2)
    
    def test_malformed_jsonld(self):
        """Test handling of malformed JSON-LD data."""
        malformed_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <script type="application/ld+json">
            {
                "@context": "https://schema.org",
                "@type": "Article",
                "name": "Malformed JSON-LD Test",
                "description": "This is a malformed JSON-LD example,
            }
            </script>
        </head>
        <body>
            <h1>Malformed JSON-LD Test</h1>
        </body>
        </html>
        """
        
        result = self.extractor.extract(malformed_html)
        self.assertEqual(len(result.get("items", [])), 0)
        self.assertTrue("_metadata" in result)
        self.assertEqual(result["_metadata"]["item_count"], 0)
    
    def tearDown(self):
        """Clean up test environment."""
        self.extractor.shutdown()


class TestMicrodataExtractor(unittest.TestCase):
    """Tests for the Microdata extractor."""
    
    def setUp(self):
        """Set up test environment."""
        self.extractor = MicrodataExtractor()
        self.extractor.initialize()
        
        # Sample HTML with Microdata
        self.html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Microdata Test</title>
        </head>
        <body>
            <div itemscope itemtype="https://schema.org/Product">
                <h1 itemprop="name">Microdata Product</h1>
                <img itemprop="image" src="https://example.com/product.jpg" alt="Product Image">
                <div itemprop="description">This is a product description.</div>
                <div itemprop="offers" itemscope itemtype="https://schema.org/Offer">
                    <span itemprop="price" content="49.99">$49.99</span>
                    <meta itemprop="priceCurrency" content="USD">
                    <link itemprop="availability" href="https://schema.org/InStock">In Stock
                </div>
                <div itemprop="brand" itemscope itemtype="https://schema.org/Brand">
                    <span itemprop="name">Example Brand</span>
                </div>
            </div>
            
            <div itemscope itemtype="https://schema.org/Person">
                <span itemprop="name">Jane Doe</span>
                <a itemprop="url" href="https://example.com/jane">Profile</a>
                <img itemprop="image" src="https://example.com/jane.jpg">
            </div>
        </body>
        </html>
        """
    
    def test_extract_microdata(self):
        """Test extraction of Microdata."""
        result = self.extractor.extract_microdata(self.html)
        self.assertEqual(len(result), 2)
        
        # Check product
        product = result[0]
        self.assertEqual(product.get("itemtype"), "https://schema.org/Product")
        self.assertEqual(product["properties"].get("name"), "Microdata Product")
        self.assertEqual(product["properties"].get("image"), "https://example.com/product.jpg")
        self.assertTrue(isinstance(product["properties"].get("offers"), dict))
        
        # Check person
        person = result[1]
        self.assertEqual(person.get("itemtype"), "https://schema.org/Person")
        self.assertEqual(person["properties"].get("name"), "Jane Doe")
        self.assertEqual(person["properties"].get("url"), "https://example.com/jane")
    
    def test_extract_itemscope_elements(self):
        """Test extraction of itemscope elements."""
        result = self.extractor.extract_itemscope_elements(self.html)
        self.assertEqual(len(result), 4)  # Product, Offer, Brand, Person
        
        # Check top-level elements
        top_level = [item for item in result if not item["is_nested"]]
        self.assertEqual(len(top_level), 2)  # Product, Person
        
        # Check nested elements
        nested = [item for item in result if item["is_nested"]]
        self.assertEqual(len(nested), 2)  # Offer, Brand
    
    def test_map_microdata_to_schema(self):
        """Test mapping of Microdata to schema types."""
        microdata = self.extractor.extract_microdata(self.html)
        result = self.extractor.map_microdata_to_schema(microdata)
        
        self.assertTrue("Product" in result)
        self.assertTrue("Person" in result)
        self.assertEqual(len(result["Product"]), 1)
        self.assertEqual(len(result["Person"]), 1)
    
    def test_extract_all(self):
        """Test complete extraction of Microdata."""
        result = self.extractor.extract(self.html)
        
        self.assertEqual(len(result.get("items")), 2)
        self.assertEqual(result.get("itemscope_count"), 4)
        self.assertTrue("Product" in result.get("schema_data"))
        self.assertTrue("Person" in result.get("schema_data"))
        
        self.assertTrue("_metadata" in result)
        self.assertEqual(result["_metadata"]["item_count"], 2)
    
    def test_empty_microdata(self):
        """Test handling of HTML without Microdata."""
        html_without_microdata = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>No Microdata</title>
        </head>
        <body>
            <h1>No Microdata Here</h1>
            <p>This page has no Microdata attributes.</p>
        </body>
        </html>
        """
        
        result = self.extractor.extract(html_without_microdata)
        self.assertEqual(len(result.get("items", [])), 0)
        self.assertEqual(result.get("itemscope_count"), 0)
        self.assertEqual(len(result.get("schema_data", {})), 0)
    
    def tearDown(self):
        """Clean up test environment."""
        self.extractor.shutdown()


class TestJavaScriptExtractor(unittest.TestCase):
    """Tests for the JavaScript variable extractor."""
    
    def setUp(self):
        """Set up test environment."""
        self.extractor = JavaScriptExtractor()
        
        # Sample HTML with JavaScript variables
        self.html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>JavaScript Variables Test</title>
            <script type="text/javascript">
                var pageData = {
                    "title": "JS Page Title",
                    "id": 12345,
                    "metadata": {
                        "author": "JS Author",
                        "date": "2023-01-20"
                    }
                };
                
                const productInfo = {
                    "name": "JS Product",
                    "price": 99.99,
                    "currency": "USD",
                    "inStock": true,
                    "images": [
                        "https://example.com/product1.jpg",
                        "https://example.com/product2.jpg"
                    ]
                };
                
                window.siteConfig = {
                    "apiUrl": "https://api.example.com",
                    "language": "en-US",
                    "features": {
                        "comments": true,
                        "sharing": true
                    }
                };
                
                // Configuration object
                var CONFIG = {
                    "debug": false,
                    "version": "1.2.3",
                    "theme": "light"
                };
            </script>
        </head>
        <body>
            <h1>JavaScript Variables Test</h1>
            <script>
                // JSON object directly in script
                {
                    "type": "analytics",
                    "enabled": true,
                    "trackingId": "UA-12345-6"
                }
            </script>
        </body>
        </html>
        """
    
    def test_extract_js_variables(self):
        """Test extraction of JavaScript variables."""
        result = self.extractor.extract_js_variables(self.html)
        
        self.assertTrue("pageData" in result)
        self.assertTrue("productInfo" in result)
        self.assertTrue("window.siteConfig" in result)
        
        self.assertEqual(result["pageData"]["title"], "JS Page Title")
        self.assertEqual(result["pageData"]["id"], 12345)
        self.assertEqual(result["productInfo"]["name"], "JS Product")
        self.assertEqual(result["productInfo"]["price"], 99.99)
        self.assertEqual(len(result["productInfo"]["images"]), 2)
        self.assertEqual(result["window.siteConfig"]["language"], "en-US")
    
    def test_extract_json_objects(self):
        """Test extraction of JSON objects from JavaScript."""
        # Extract from a script tag
        script_content = """
        {
            "type": "analytics",
            "enabled": true,
            "trackingId": "UA-12345-6"
        }
        """
        
        result = self.extractor.extract_json_objects(script_content)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "analytics")
        self.assertEqual(result[0]["trackingId"], "UA-12345-6")
    
    def test_extract_config_objects(self):
        """Test extraction of configuration objects."""
        script_content = """
        var CONFIG = {
            "debug": false,
            "version": "1.2.3",
            "theme": "light"
        };
        """
        
        result = self.extractor.extract_config_objects(script_content)
        self.assertEqual(len(result), 1)
        self.assertTrue("config_0" in result)
        self.assertEqual(result["config_0"]["version"], "1.2.3")
    
    def test_map_js_data_to_schema(self):
        """Test mapping of JavaScript data to schema."""
        js_data = {
            "pageData": {
                "title": "JS Page Title",
                "id": 12345
            },
            "productData": {
                "name": "JS Product",
                "price": 99.99
            },
            "config": {
                "debug": False
            }
        }
        
        result = self.extractor.map_js_data_to_schema(js_data)
        self.assertTrue("metadata" in result)
        self.assertTrue("product" in result)
        self.assertTrue("config" in result)
        self.assertEqual(result["metadata"]["title"], "JS Page Title")
        self.assertEqual(result["product"]["name"], "JS Product")
    
    def test_extract_all_data(self):
        """Test complete extraction of JavaScript data."""
        result = self.extractor.extract_all_data(self.html)
        
        self.assertTrue("variables" in result)
        self.assertTrue("configs" in result)
        self.assertTrue("json_objects" in result)
        self.assertTrue("mapped_data" in result)
        
        self.assertTrue("pageData" in result["variables"])
        self.assertTrue("productInfo" in result["variables"])
        self.assertTrue("window.siteConfig" in result["variables"])
        
        # Check configs
        self.assertTrue(len(result["configs"]) > 0)
        # Check JSON objects
        self.assertTrue(len(result["json_objects"]) > 0)
    
    def test_safe_parse_json(self):
        """Test safe parsing of JSON strings."""
        # Test valid JSON
        valid_json = '{"name": "Test", "value": 123}'
        result = self.extractor._safe_parse_json(valid_json)
        self.assertEqual(result["name"], "Test")
        self.assertEqual(result["value"], 123)
        
        # Test JavaScript-style JSON with single quotes
        js_json = "{'name': 'Test', 'value': 123}"
        result = self.extractor._safe_parse_json(js_json)
        self.assertEqual(result["name"], "Test")
        self.assertEqual(result["value"], 123)
        
        # Test JSON with trailing commas
        trailing_comma_json = '{"name": "Test", "values": [1, 2, 3,], }'
        result = self.extractor._safe_parse_json(trailing_comma_json)
        self.assertEqual(result["name"], "Test")
        self.assertEqual(result["values"], [1, 2, 3])
        
        # Test JSON with unquoted keys
        unquoted_json = '{name: "Test", value: 123}'
        result = self.extractor._safe_parse_json(unquoted_json)
        self.assertEqual(result["name"], "Test")
        self.assertEqual(result["value"], 123)


class TestCompositeMetadataExtractor(unittest.TestCase):
    """Tests for the composite metadata extractor."""
    
    def setUp(self):
        """Set up test environment."""
        self.context = MagicMock(spec=StrategyContext)
        self.html_service = MagicMock()
        self.context.get_service.return_value = self.html_service
        self.html_service.clean_html.return_value = "cleaned html"
        
        self.extractor = CompositeMetadataExtractor(self.context)
        self.extractor.initialize()
        
        # Sample HTML with various metadata formats
        self.html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Meta Test Page</title>
            <meta name="description" content="Meta description">
            <meta name="keywords" content="meta, test, keywords">
            <link rel="canonical" href="https://example.com/meta-test">
            
            <!-- Open Graph -->
            <meta property="og:title" content="OG Title">
            <meta property="og:description" content="OG description">
            <meta property="og:image" content="https://example.com/og-image.jpg">
            <meta property="og:url" content="https://example.com/og-url">
            <meta property="og:type" content="article">
            <meta property="og:site_name" content="Example Site">
            
            <!-- Twitter Cards -->
            <meta name="twitter:card" content="summary">
            <meta name="twitter:title" content="Twitter Title">
            
            <!-- JSON-LD -->
            <script type="application/ld+json">
            {
                "@context": "https://schema.org",
                "@type": "Article",
                "headline": "JSON-LD Title",
                "description": "JSON-LD description",
                "image": "https://example.com/jsonld-image.jpg",
                "datePublished": "2023-01-10T12:00:00+00:00",
                "author": {
                    "@type": "Person",
                    "name": "John Author"
                }
            }
            </script>
            
            <!-- Microdata -->
            <div itemscope itemtype="https://schema.org/WebPage">
                <h1 itemprop="name">Microdata Title</h1>
                <p itemprop="description">Microdata description</p>
            </div>
        </head>
        <body>
            <h1>Content Title</h1>
            <p>Content paragraph</p>
        </body>
        </html>
        """
        
        # Mock the sub-extractors to return known data
        self.extractor.html_extractor.extract = MagicMock(return_value={
            "title": "Meta Test Page",
            "description": "Meta description",
            "keywords": "meta, test, keywords",
            "canonical_url": "https://example.com/meta-test"
        })
        
        self.extractor.open_graph_extractor.extract = MagicMock(return_value={
            "title": "OG Title",
            "description": "OG description",
            "image": "https://example.com/og-image.jpg",
            "url": "https://example.com/og-url",
            "type": "article",
            "site_name": "Example Site",
            "twitter_title": "Twitter Title"
        })
        
        self.extractor.jsonld_extractor.extract = MagicMock(return_value={
            "title": "JSON-LD Title",
            "description": "JSON-LD description",
            "image": "https://example.com/jsonld-image.jpg",
            "datePublished": "2023-01-10T12:00:00+00:00",
            "author": "John Author",
            "items": [{"@type": "Article"}]
        })
        
        self.extractor.microdata_extractor.extract = MagicMock(return_value={
            "items": [
                {
                    "itemtype": "https://schema.org/WebPage",
                    "properties": {
                        "name": "Microdata Title",
                        "description": "Microdata description"
                    }
                }
            ]
        })
    
    def test_extract_all_metadata(self):
        """Test extraction of metadata from all sources."""
        result = self.extractor.extract_all_metadata(self.html)
        
        self.assertTrue("html_meta" in result)
        self.assertTrue("open_graph" in result)
        self.assertTrue("json_ld" in result)
        self.assertTrue("microdata" in result)
        
        # Check HTML meta data
        self.assertEqual(result["html_meta"]["title"], "Meta Test Page")
        self.assertEqual(result["html_meta"]["description"], "Meta description")
        
        # Check Open Graph data
        self.assertEqual(result["open_graph"]["title"], "OG Title")
        self.assertEqual(result["open_graph"]["image"], "https://example.com/og-image.jpg")
        
        # Check JSON-LD data
        self.assertEqual(result["json_ld"]["title"], "JSON-LD Title")
        self.assertEqual(result["json_ld"]["author"], "John Author")
    
    def test_consolidate_metadata(self):
        """Test consolidation of metadata from different sources."""
        metadata_sources = {
            "html_meta": {
                "title": "Meta Test Page",
                "description": "Meta description",
                "keywords": "meta, test, keywords",
                "canonical_url": "https://example.com/meta-test"
            },
            "open_graph": {
                "title": "OG Title",
                "description": "OG description",
                "image": "https://example.com/og-image.jpg",
                "site_name": "Example Site"
            },
            "json_ld": {
                "title": "JSON-LD Title",
                "description": "JSON-LD description",
                "datePublished": "2023-01-10T12:00:00+00:00"
            }
        }
        
        result = self.extractor.consolidate_metadata(metadata_sources)
        
        # Check top-level fields
        self.assertEqual(result["title"], "JSON-LD Title")  # JSON-LD has priority
        self.assertEqual(result["description"], "JSON-LD description")
        self.assertEqual(result["keywords"], "meta, test, keywords")
        self.assertEqual(result["canonical_url"], "https://example.com/meta-test")
        self.assertEqual(result["image"], "https://example.com/og-image.jpg")
        self.assertEqual(result["site_name"], "Example Site")
        self.assertEqual(result["published_date"], "2023-01-10T12:00:00+00:00")
        
        # Check source-specific sections
        self.assertTrue("html_meta" in result)
        self.assertTrue("open_graph" in result)
        self.assertTrue("json_ld" in result)
    
    def test_get_priority_metadata(self):
        """Test prioritization of metadata."""
        consolidated = {
            "title": "JSON-LD Title",
            "html_meta": {"title": "Meta Title"},
            "open_graph": {"title": "OG Title"},
            "json_ld": {"title": "JSON-LD Title", "datePublished": "2023-01-10T12:00:00+00:00"},
            "image": "https://example.com/og-image.jpg",
            "keywords": "meta, test, keywords",
            "site_name": "Example Site"
        }
        
        result = self.extractor.get_priority_metadata(consolidated)
        
        # Check that prioritization worked correctly
        self.assertEqual(result.get("title"), "JSON-LD Title")
        self.assertEqual(result.get("image"), "https://example.com/og-image.jpg")
        self.assertEqual(result.get("keywords"), "meta, test, keywords")
        self.assertEqual(result.get("site_name"), "Example Site")
        
        # Check that nested fields are correctly extracted
        self.assertEqual(result.get("published_date"), "2023-01-10T12:00:00+00:00")
    
    def test_normalize_metadata(self):
        """Test normalization of metadata."""
        metadata = {
            "title": "example title | Example Site",
            "description": "  This is a   description with  extra  whitespace.  ",
            "keywords": "keyword1,keyword2, keyword3",
            "site_name": "Example Site",
            "published_date": "2023-01-10 12:00:00",
            "image": "/relative-path.jpg",
            "canonical_url": "https://example.com/page"
        }
        
        result = self.extractor.normalize_metadata(metadata)
        
        # Check title normalization (site name removal, capitalization)
        self.assertEqual(result["title"], "Example Title")
        
        # Check description normalization (whitespace)
        self.assertEqual(result["description"], "This is a description with extra whitespace.")
        
        # Check keywords normalization (array conversion)
        self.assertIsInstance(result["keywords"], list)
        self.assertEqual(len(result["keywords"]), 3)
        self.assertEqual(result["keywords"][0], "keyword1")
        
        # Check image URL normalization (relative to absolute)
        self.assertEqual(result["image"], "https://example.com/relative-path.jpg")
    
    def test_extract_complete(self):
        """Test complete extraction with all steps."""
        result = self.extractor.extract(self.html, {
            "prioritize_metadata": True,
            "normalize": True
        })
        
        # Check that we have data
        self.assertTrue("title" in result)
        self.assertTrue("description" in result)
        
        # Check metadata about the extraction
        self.assertTrue("_metadata" in result)
        self.assertTrue("sources" in result["_metadata"])
        self.assertTrue("source_count" in result["_metadata"])
        self.assertEqual(result["_metadata"]["extractor"], "CompositeMetadataExtractor")
    
    def tearDown(self):
        """Clean up test environment."""
        self.extractor.shutdown()


if __name__ == "__main__":
    unittest.main()