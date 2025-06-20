import unittest
from bs4 import BeautifulSoup
from unittest.mock import MagicMock

from extraction.structural_analyzer import DOMStructuralAnalyzer

class TestDOMStructuralAnalyzer(unittest.TestCase):
    """Test cases for DOMStructuralAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock context
        self.mock_context = MagicMock()
        self.html_service = MagicMock()
        self.html_service.generate_selector.return_value = "#test-selector"
        self.mock_context.get_service.return_value = self.html_service
        
        # Initialize analyzer
        self.analyzer = DOMStructuralAnalyzer(self.mock_context)
        self.analyzer.initialize()
        
        # Sample HTML with various structures
        self.sample_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Page</title>
        </head>
        <body>
            <header>
                <nav>
                    <ul>
                        <li><a href="/">Home</a></li>
                        <li><a href="/about">About</a></li>
                        <li><a href="/products">Products</a></li>
                        <li><a href="/contact">Contact</a></li>
                    </ul>
                </nav>
            </header>
            
            <main>
                <section class="hero">
                    <h1>Welcome to Our Website</h1>
                    <p>This is a test page for structural analysis.</p>
                </section>
                
                <section class="featured-products">
                    <h2>Featured Products</h2>
                    <div class="product-grid">
                        <div class="product">
                            <img src="product1.jpg" alt="Product 1">
                            <h3>Product 1</h3>
                            <p class="price">$19.99</p>
                            <button>Add to Cart</button>
                        </div>
                        <div class="product">
                            <img src="product2.jpg" alt="Product 2">
                            <h3>Product 2</h3>
                            <p class="price">$29.99</p>
                            <button>Add to Cart</button>
                        </div>
                        <div class="product">
                            <img src="product3.jpg" alt="Product 3">
                            <h3>Product 3</h3>
                            <p class="price">$39.99</p>
                            <button>Add to Cart</button>
                        </div>
                    </div>
                </section>
                
                <article>
                    <h2>Latest News</h2>
                    <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
                    Nullam eget felis eget nunc lobortis mattis aliquam faucibus.</p>
                    <p>Pellentesque habitant morbi tristique senectus et netus et 
                    malesuada fames ac turpis egestas.</p>
                </article>
            </main>
            
            <aside>
                <h3>Related Links</h3>
                <ul>
                    <li><a href="/related1">Related 1</a></li>
                    <li><a href="/related2">Related 2</a></li>
                </ul>
            </aside>
            
            <footer>
                <p>&copy; 2025 Test Company</p>
                <div class="social-links">
                    <a href="https://facebook.com">Facebook</a>
                    <a href="https://twitter.com">Twitter</a>
                    <a href="https://instagram.com">Instagram</a>
                </div>
            </footer>
        </body>
        </html>
        """
        
        # Parse HTML
        self.soup = BeautifulSoup(self.sample_html, 'html.parser')
    
    def test_can_handle(self):
        """Test content type detection."""
        self.assertTrue(self.analyzer.can_handle(self.sample_html))
        self.assertTrue(self.analyzer.can_handle(self.soup))
        self.assertTrue(self.analyzer.can_handle("<div>Test</div>", "html"))
        self.assertFalse(self.analyzer.can_handle("Plain text content", "text"))
    
    def test_analyze_basic(self):
        """Test basic analysis functionality."""
        result = self.analyzer.analyze(self.sample_html)
        
        # Check that analysis was successful
        self.assertTrue(result["success"])
        self.assertEqual(result["page_title"], "Test Page")
        
        # Check that core analysis sections are present
        self.assertIn("hierarchy", result)
        self.assertIn("content_sections", result)
        self.assertIn("content_boundaries", result)
        self.assertIn("navigation_elements", result)
        self.assertIn("result_groups", result)
        self.assertIn("content_type", result)
        self.assertIn("structure_map", result)
    
    def test_analyze_dom_hierarchy(self):
        """Test DOM hierarchy analysis."""
        hierarchy = self.analyzer.analyze_dom_hierarchy(self.soup)
        
        # Check structure metrics are calculated
        self.assertIn("element_count", hierarchy)
        self.assertIn("max_depth", hierarchy)
        self.assertIn("complexity", hierarchy)
        self.assertIn("dominant_patterns", hierarchy)
        
        # Validate metrics
        self.assertGreater(hierarchy["element_count"], 10)
        self.assertGreater(hierarchy["max_depth"], 3)
        self.assertGreater(hierarchy["complexity"], 0)
        self.assertGreaterEqual(len(hierarchy["dominant_patterns"]), 1)
    
    def test_identify_content_sections(self):
        """Test content section identification."""
        sections = self.analyzer.identify_content_sections(self.soup)
        
        # Check sections were found
        self.assertGreater(len(sections), 0)
        
        # Verify section properties
        first_section = sections[0]
        self.assertIn("section_type", first_section)
        self.assertIn("text_density", first_section)
        self.assertIn("is_boilerplate", first_section)
        self.assertIn("importance_score", first_section)
    
    def test_detect_result_groups(self):
        """Test result group detection."""
        groups = self.analyzer.detect_result_groups(self.soup)
        
        # Check if product grid was detected
        self.assertGreaterEqual(len(groups), 1)
        
        # Verify it found at least 3 products
        if groups:
            product_group = groups[0]
            self.assertGreaterEqual(product_group["item_count"], 3)
            self.assertGreater(product_group["structural_similarity"], 0.5)
    
    def test_detect_content_type(self):
        """Test content type detection."""
        content_type = self.analyzer.detect_content_type(self.soup)
        
        # Check structure
        self.assertIn("primary_type", content_type)
        self.assertIn("confidence", content_type)
        self.assertIn("content_signals", content_type)
        
        # Should detect this as a product/listing page
        self.assertIn(content_type["primary_type"], ["product", "listing"])
        self.assertGreater(content_type["confidence"], 0.5)
    
    def test_navigation_detection(self):
        """Test navigation element detection."""
        nav_elements = self.analyzer.identify_navigation_elements(self.soup)
        
        # Should find primary navigation
        self.assertGreaterEqual(len(nav_elements["primary_nav"]), 1)
        
        # Check if it found the correct number of menu items
        if nav_elements["primary_nav"]:
            primary_nav = nav_elements["primary_nav"][0]
            self.assertEqual(primary_nav["item_count"], 4)  # 4 links in our test HTML
    
    def test_semantic_sectioning(self):
        """Test semantic section recognition."""
        # Create a mock with just semantic sections
        semantic_html = """
        <body>
            <header>Header</header>
            <main>Main Content</main>
            <article>Article</article>
            <section>Section</section>
            <aside>Sidebar</aside>
            <footer>Footer</footer>
        </body>
        """
        
        soup = BeautifulSoup(semantic_html, 'html.parser')
        sections = self.analyzer._find_semantic_sections(soup)
        
        # Should find all 6 semantic sections
        self.assertEqual(len(sections), 6)
        
        # Check section types
        section_types = [s["section_type"] for s in sections]
        self.assertIn("header", section_types)
        self.assertIn("main", section_types)
        self.assertIn("article", section_types)
        self.assertIn("section", section_types)
        self.assertIn("aside", section_types)
        self.assertIn("footer", section_types)
    
    def test_element_relationships(self):
        """Test element relationship detection."""
        relationships = self.analyzer.detect_element_relationships(self.soup)
        
        # Check for parent-child relationships
        self.assertIn("parent_child", relationships)
        self.assertGreater(len(relationships["parent_child"]), 0)
        
        # Check for container-item relationships (for products)
        self.assertIn("container_item", relationships)
        self.assertGreater(len(relationships["container_item"]), 0)
        
        # Check for heading-content relationships
        self.assertIn("heading_content", relationships)
        self.assertGreater(len(relationships["heading_content"]), 0)
    
    def test_content_density(self):
        """Test content density analysis."""
        density = self.analyzer.analyze_content_density(self.soup)
        
        # Check basic structure
        self.assertIn("overall_density", density)
        self.assertIn("section_densities", density)
        self.assertIn("text_length", density)
        
        # Density should be between 0 and 1
        self.assertGreaterEqual(density["overall_density"], 0.0)
        self.assertLessEqual(density["overall_density"], 1.0)
        
        # Should find some content-rich areas
        self.assertGreaterEqual(len(density["section_densities"]), 1)

if __name__ == '__main__':
    unittest.main()