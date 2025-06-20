import sys
import os
import unittest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import json

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.search_template_integration import SearchTemplateIntegrator
from components.template_storage import TemplateStorage
from components.domain_intelligence import DomainIntelligence

class TestSearchTemplateIntegrator(unittest.TestCase):
    """Test case for the SearchTemplateIntegrator class."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary template storage directory
        self.temp_template_dir = os.path.join(os.path.dirname(__file__), 'temp_templates')
        os.makedirs(self.temp_template_dir, exist_ok=True)
        
        # Mock crawler
        self.mock_crawler = AsyncMock()
        
        # Create the integrator with mocked dependencies
        self.integrator = SearchTemplateIntegrator(crawler=self.mock_crawler, template_path=self.temp_template_dir)
        
        # Replace template_storage with a mock
        self.mock_template_storage = MagicMock(spec=TemplateStorage)
        self.integrator.template_storage = self.mock_template_storage
        
        # Set up a sample template and extraction results
        self.sample_template = {
            "id": "test-template-1",
            "domain": "example.com",
            "selectors": {
                "item": ".product-item",
                "title": ".product-title",
                "price": ".product-price",
                "url": ".product-link"
            },
            "properties": ["title", "price", "url"]
        }
        
        self.sample_extraction_results = {
            "success": True,
            "results": [
                {"title": "Test Product 1", "price": "$19.99", "url": "https://example.com/p1"},
                {"title": "Test Product 2", "price": "$29.99", "url": "https://example.com/p2"}
            ],
            "extraction_time": 0.45
        }
        
        self.sample_html = """
        <div class="product-listing">
            <div class="product-item">
                <h2 class="product-title">Test Product 1</h2>
                <div class="product-price">$19.99</div>
                <a class="product-link" href="https://example.com/p1">View Details</a>
            </div>
            <div class="product-item">
                <h2 class="product-title">Test Product 2</h2>
                <div class="product-price">$29.99</div>
                <a class="product-link" href="https://example.com/p2">View Details</a>
            </div>
        </div>
        """
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove the temporary template directory if it exists
        if os.path.exists(self.temp_template_dir):
            import shutil
            shutil.rmtree(self.temp_template_dir)
    
    def test_initialization(self):
        """Test that the SearchTemplateIntegrator initializes correctly."""
        integrator = SearchTemplateIntegrator(crawler=self.mock_crawler, template_path=self.temp_template_dir)
        
        self.assertIsNotNone(integrator.crawler)
        self.assertIsNotNone(integrator.search_automator)
        self.assertIsNotNone(integrator.template_storage)
        self.assertEqual(integrator.extraction_success_metrics, {})
        
    @patch('components.search_template_integration.SearchAutomator')
    @patch('components.search_template_integration.ContentAnalysis')
    async def test_find_suitable_templates(self, mock_content_analysis, mock_search_automator):
        """Test finding suitable templates for a query."""
        # Setup
        self.mock_template_storage.find_templates_for_domain.return_value = [self.sample_template]
        self.mock_template_storage.find_templates_by_keywords.return_value = [self.sample_template]
        
        # Mock search results
        search_result = {
            "url": "https://example.com/search",
            "domain": "example.com",
            "title": "Search Results",
            "snippet": "Sample search results"
        }
        
        # Mock search automator to return results
        mock_search_instance = mock_search_automator.return_value
        mock_search_instance.search.return_value = [search_result]
        
        # Execute
        result = await self.integrator.find_suitable_templates("test product")
        
        # Assert
        self.assertTrue(result["success"])
        self.assertEqual(len(result["templates"]), 1)
        self.assertEqual(result["templates"][0]["id"], "test-template-1")
        
        # Verify method calls
        mock_search_instance.search.assert_called_once_with("test product")
        self.mock_template_storage.find_templates_for_domain.assert_called_once_with("example.com")
    
    @patch('components.search_template_integration.SearchAutomator')
    async def test_find_suitable_templates_no_results(self, mock_search_automator):
        """Test finding suitable templates when search returns no results."""
        # Setup
        mock_search_instance = mock_search_automator.return_value
        mock_search_instance.search.return_value = []
        
        # Execute
        result = await self.integrator.find_suitable_templates("test product")
        
        # Assert
        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "No search results found")
        
    async def test_refine_template_based_on_extraction_success(self):
        """Test refining a template based on successful extraction."""
        # Setup
        self.mock_template_storage.get_current_timestamp.return_value = "2023-01-01T12:00:00Z"
        
        # Execute
        result = await self.integrator.refine_template_based_on_extraction(
            self.sample_template, 
            self.sample_extraction_results,
            self.sample_html,
            apply_immediately=False
        )
        
        # Assert basic result structure
        self.assertTrue(result["success"])
        self.assertEqual(result["template_id"], "test-template-1")
        
        # Assert metrics were updated
        metrics = self.integrator.extraction_success_metrics["test-template-1"]
        self.assertEqual(metrics["total_uses"], 1)
        self.assertEqual(metrics["successful_extractions"], 1)
        self.assertEqual(metrics["empty_extractions"], 0)
        self.assertEqual(metrics["failed_extractions"], 0)
        
        # Property success rates should be tracked
        self.assertIn("property_success_rates", metrics)
        self.assertIn("title", metrics["property_success_rates"])
        self.assertIn("price", metrics["property_success_rates"])
        self.assertIn("url", metrics["property_success_rates"])
        
        # All properties were successfully extracted
        for prop in ["title", "price", "url"]:
            self.assertEqual(metrics["property_success_rates"][prop], 1.0)
            
        # Extraction time should be recorded
        self.assertGreater(metrics["average_extraction_time"], 0)
        
    async def test_refine_template_based_on_extraction_failure(self):
        """Test refining a template based on failed extraction."""
        # Setup
        self.mock_template_storage.get_current_timestamp.return_value = "2023-01-01T12:00:00Z"
        
        # Create failed extraction results
        failed_extraction = {
            "success": False,
            "error": "Extraction failed",
            "extraction_time": 0.2
        }
        
        # Execute
        result = await self.integrator.refine_template_based_on_extraction(
            self.sample_template, 
            failed_extraction,
            self.sample_html,
            apply_immediately=False
        )
        
        # Assert
        self.assertTrue(result["success"])
        self.assertEqual(result["template_id"], "test-template-1")
        self.assertTrue(result["refinement_needed"])
        
        # Assert metrics were updated
        metrics = self.integrator.extraction_success_metrics["test-template-1"]
        self.assertEqual(metrics["total_uses"], 1)
        self.assertEqual(metrics["successful_extractions"], 0)
        self.assertEqual(metrics["empty_extractions"], 0)
        self.assertEqual(metrics["failed_extractions"], 1)
        
        # Check refinements were attempted
        self.assertIn("refinements_made", result)
        
    async def test_refine_template_empty_results(self):
        """Test refining a template with empty extraction results."""
        # Setup
        self.mock_template_storage.get_current_timestamp.return_value = "2023-01-01T12:00:00Z"
        
        # Create empty extraction results
        empty_extraction = {
            "success": True,
            "results": [],
            "extraction_time": 0.15
        }
        
        # Execute
        result = await self.integrator.refine_template_based_on_extraction(
            self.sample_template, 
            empty_extraction,
            self.sample_html,
            apply_immediately=False
        )
        
        # Assert
        self.assertTrue(result["success"])
        self.assertEqual(result["template_id"], "test-template-1")
        self.assertTrue(result["refinement_needed"])
        
        # Assert metrics were updated
        metrics = self.integrator.extraction_success_metrics["test-template-1"]
        self.assertEqual(metrics["total_uses"], 1)
        self.assertEqual(metrics["successful_extractions"], 0)
        self.assertEqual(metrics["empty_extractions"], 1)
        self.assertEqual(metrics["failed_extractions"], 0)
        
    @patch('components.search_template_integration.BeautifulSoup')
    async def test_improve_basic_selectors(self, mock_bs):
        """Test improving basic selectors when extraction fails."""
        # Create mock soup and elements
        mock_soup = MagicMock()
        mock_bs.return_value = mock_soup
        
        # Mock container selection
        mock_container = MagicMock()
        mock_container.find_all.return_value = [
            MagicMock(name='div'),
            MagicMock(name='div')
        ]
        mock_soup.select.return_value = [mock_container]
        
        # Mock child elements
        mock_child = MagicMock()
        mock_child.name = 'div'
        mock_child.find_all.return_value = [MagicMock()]
        mock_container.find_all.return_value = [mock_child, mock_child]
        
        # Execute
        result = await self.integrator._improve_basic_selectors(
            self.sample_html,
            {"item": ".product-item"},
            "extraction_failure"
        )
        
        # Assert
        self.assertIsInstance(result, dict)
        
    @patch('components.domain_intelligence.DomainIntelligence.analyze_domain')
    async def test_optimize_templates_for_domain_ecommerce(self, mock_analyze_domain):
        """Test optimizing templates for an ecommerce domain."""
        # Setup
        domain = "example-shop.com"
        
        # Mock domain intelligence analysis
        mock_analyze_domain.return_value = {
            "domain": domain,
            "site_type": "ecommerce",
            "uses_javascript": True,
            "has_pagination": True
        }
        
        # Mock finding templates for domain
        self.mock_template_storage.find_templates_for_domain.return_value = [
            {
                "id": "template-1",
                "domain": domain,
                "selectors": {
                    "item": ".product",
                    "title": ".product-name"
                },
                "properties": ["title"]
            }
        ]
        
        # Execute
        result = await self.integrator.optimize_templates_for_domain(domain)
        
        # Assert
        self.assertTrue(result["success"])
        self.assertEqual(result["domain"], domain)
        self.assertEqual(result["templates_optimized"], 1)
        self.assertEqual(result["template_ids"], ["template-1"])
        self.assertEqual(result["site_type"], "ecommerce")
        
        # Check that the template was saved with optimizations
        self.mock_template_storage.save_template.assert_called()
        
    @patch('components.domain_intelligence.DomainIntelligence.analyze_domain')
    async def test_optimize_templates_for_domain_real_estate(self, mock_analyze_domain):
        """Test optimizing templates for a real estate domain."""
        # Setup
        domain = "example-homes.com"
        
        # Mock domain intelligence analysis
        mock_analyze_domain.return_value = {
            "domain": domain,
            "site_type": "real_estate",
            "uses_javascript": False,
            "has_pagination": True
        }
        
        # Mock finding templates for domain
        self.mock_template_storage.find_templates_for_domain.return_value = [
            {
                "id": "template-2",
                "domain": domain,
                "selectors": {
                    "item": ".listing",
                    "title": ".listing-title"
                },
                "properties": ["title"]
            }
        ]
        
        # Execute
        result = await self.integrator.optimize_templates_for_domain(domain)
        
        # Assert
        self.assertTrue(result["success"])
        self.assertEqual(result["domain"], domain)
        self.assertEqual(result["templates_optimized"], 1)
        self.assertIn("real_estate_optimization", result["optimization_actions"])
        
    @patch('components.domain_intelligence.DomainIntelligence.analyze_domain')
    async def test_optimize_templates_for_domain_news(self, mock_analyze_domain):
        """Test optimizing templates for a news domain."""
        # Setup
        domain = "example-news.com"
        
        # Mock domain intelligence analysis
        mock_analyze_domain.return_value = {
            "domain": domain,
            "site_type": "news",
            "uses_javascript": False,
            "has_pagination": True
        }
        
        # Mock finding templates for domain
        self.mock_template_storage.find_templates_for_domain.return_value = [
            {
                "id": "template-3",
                "domain": domain,
                "selectors": {
                    "item": ".article",
                    "title": ".article-title"
                },
                "properties": ["title"]
            }
        ]
        
        # Execute
        result = await self.integrator.optimize_templates_for_domain(domain)
        
        # Assert
        self.assertTrue(result["success"])
        self.assertEqual(result["domain"], domain)
        self.assertIn("template-3", result["template_ids"])
        self.assertIn("news_blog_optimization", result["optimization_actions"])
        
    def test_optimize_selectors_for_performance(self):
        """Test optimizing selectors for better performance."""
        # Setup
        selectors = {
            "item": ".products .product-grid .product-item",
            "title": ".product-item .product-title h2",
            "id": "div#product-12345.product-item",
            "multiple_classes": ".class1.class2.most-specific"
        }
        
        # Execute
        result = self.integrator._optimize_selectors_for_performance(selectors)
        
        # Assert
        self.assertIsInstance(result, dict)
        
        # Check selector optimizations
        self.assertEqual(result["item"], ".products .product-item")  # Simplified descendant
        self.assertEqual(result["title"], ".product-item h2")  # Simplified chain
        self.assertEqual(result["id"], "#product-12345")  # ID optimization
        self.assertIn("most-specific", result["multiple_classes"])  # Class optimization

    @patch('components.search_template_integration.BeautifulSoup')
    async def test_improve_property_selectors(self, mock_bs):
        """Test improving property selectors for low success properties."""
        # Setup
        existing_selectors = {
            "item": ".product",
            "title": ".title",
            "price": ".price",
            "image": ".image"
        }
        
        low_success_properties = ["price", "image"]
        
        # Mock BeautifulSoup and element finding
        mock_soup = MagicMock()
        mock_bs.return_value = mock_soup
        
        # Mock price elements
        price_elements = [MagicMock()]
        mock_soup.find_all.return_value = price_elements
        
        # Mock price selector
        price_element = MagicMock()
        mock_soup.select.return_value = [price_element]
        
        # Execute
        result = await self.integrator._improve_property_selectors(
            self.sample_html,
            existing_selectors,
            low_success_properties
        )
        
        # Assert
        self.assertIsInstance(result, dict)

def run_async_test(test_function):
    """Helper function to run async tests."""
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_function())

if __name__ == "__main__":
    unittest.main()