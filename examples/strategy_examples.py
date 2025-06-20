"""
Strategy Pattern Examples

This module demonstrates various examples of creating and using strategies
with the SmartScrape strategy pattern implementation.

Examples include:
- Simple strategy implementation
- Composite strategy usage
- Custom strategy creation
- Integration with controllers
"""

import logging
from typing import Dict, Any, Optional, List, Set
import time
import random

from strategies.core.strategy_context import StrategyContext
from strategies.core.strategy_factory import StrategyFactory
from strategies.core.strategy_interface import BaseStrategy
from strategies.core.strategy_types import (
    StrategyType, StrategyCapability, strategy_metadata
)
from strategies.base_strategy_v2 import BaseStrategyV2
from strategies.core.composite_strategy import (
    SequentialStrategy, FallbackStrategy, PipelineStrategy
)
from strategies.core.strategy_error_handler import (
    StrategyErrorCategory, StrategyErrorSeverity
)
from controllers.adaptive_scraper import AdaptiveScraper

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


#######################
# 1. Simple Strategy Implementation Example
#######################

@strategy_metadata(
    strategy_type=StrategyType.EXTRACTION,
    capabilities={
        StrategyCapability.SCHEMA_EXTRACTION,
        StrategyCapability.CONTENT_NORMALIZATION
    },
    description="Example basic extraction strategy for demonstration."
)
class SimpleExtractionStrategy(BaseStrategyV2):
    """
    A simple strategy that extracts basic information from a webpage.
    Used for demonstration purposes.
    """
    
    @property
    def name(self) -> str:
        return "simple_extraction"
    
    def execute(self, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Execute the extraction strategy."""
        logger.info(f"Executing SimpleExtractionStrategy on {url}")
        
        # Simulate fetching URL
        content = self._simulate_fetch_url(url)
        if not content:
            logger.warning(f"Failed to fetch content from {url}")
            return None
        
        # Extract data from the content
        result = self.extract(content, url, **kwargs)
        if result:
            self._results.append(result)
        
        return result
    
    def extract(self, html_content: str, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Extract information from HTML content."""
        # Simulate extraction with basic pattern matching
        title = self._extract_title(html_content)
        description = self._extract_description(html_content)
        links = self._extract_links(html_content, url)
        
        return {
            "url": url,
            "title": title,
            "description": description,
            "link_count": len(links),
            "links": links[:5],  # Include first 5 links
            "extraction_time": time.time()
        }
    
    def _simulate_fetch_url(self, url: str) -> Optional[str]:
        """Simulate fetching content from a URL."""
        # In a real implementation, this would use proper URL service and requests
        logger.info(f"Simulating fetch of {url}")
        
        # Simulate failure for certain URLs
        if "error" in url or "fail" in url:
            self.handle_error(
                message=f"Failed to fetch {url}",
                category=StrategyErrorCategory.NETWORK,
                severity=StrategyErrorSeverity.WARNING
            )
            return None
        
        # Create a simulated HTML response based on the URL
        if "product" in url:
            return f"""
            <html>
                <head>
                    <title>Product: {url.split('/')[-1]}</title>
                    <meta name="description" content="This is a product page for {url.split('/')[-1]}">
                </head>
                <body>
                    <h1>Product: {url.split('/')[-1]}</h1>
                    <p>Product description goes here.</p>
                    <div class="price">$99.99</div>
                    <div class="related">
                        <a href="/product/related1">Related 1</a>
                        <a href="/product/related2">Related 2</a>
                    </div>
                </body>
            </html>
            """
        elif "category" in url:
            return f"""
            <html>
                <head>
                    <title>Category: {url.split('/')[-1]}</title>
                    <meta name="description" content="Browse products in {url.split('/')[-1]}">
                </head>
                <body>
                    <h1>Category: {url.split('/')[-1]}</h1>
                    <div class="products">
                        <div class="product">
                            <a href="/product/item1">Item 1</a>
                        </div>
                        <div class="product">
                            <a href="/product/item2">Item 2</a>
                        </div>
                        <div class="product">
                            <a href="/product/item3">Item 3</a>
                        </div>
                    </div>
                </body>
            </html>
            """
        else:
            return f"""
            <html>
                <head>
                    <title>Example Page: {url}</title>
                    <meta name="description" content="This is an example page for {url}">
                </head>
                <body>
                    <h1>Welcome to the Example Page</h1>
                    <p>This is a simulated page for demonstration purposes.</p>
                    <nav>
                        <a href="/category/electronics">Electronics</a>
                        <a href="/category/books">Books</a>
                        <a href="/about">About</a>
                        <a href="/contact">Contact</a>
                    </nav>
                </body>
            </html>
            """
    
    def _extract_title(self, html_content: str) -> str:
        """Extract title from HTML content."""
        # Simple regex-like extraction (for demonstration)
        start_marker = "<title>"
        end_marker = "</title>"
        if start_marker in html_content and end_marker in html_content:
            start_pos = html_content.find(start_marker) + len(start_marker)
            end_pos = html_content.find(end_marker, start_pos)
            return html_content[start_pos:end_pos].strip()
        return "Unknown Title"
    
    def _extract_description(self, html_content: str) -> str:
        """Extract description from HTML content."""
        # Simple regex-like extraction (for demonstration)
        start_marker = 'content="'
        end_marker = '">'
        if 'meta name="description"' in html_content and start_marker in html_content:
            meta_pos = html_content.find('meta name="description"')
            start_pos = html_content.find(start_marker, meta_pos) + len(start_marker)
            end_pos = html_content.find(end_marker, start_pos)
            return html_content[start_pos:end_pos].strip()
        return "No description available"
    
    def _extract_links(self, html_content: str, base_url: str) -> List[Dict[str, str]]:
        """Extract links from HTML content."""
        # Simple regex-like extraction (for demonstration)
        links = []
        current_pos = 0
        
        while True:
            href_marker = 'href="'
            current_pos = html_content.find(href_marker, current_pos)
            
            if current_pos == -1:
                break
                
            start_pos = current_pos + len(href_marker)
            end_pos = html_content.find('"', start_pos)
            
            if end_pos == -1:
                break
                
            href_value = html_content[start_pos:end_pos]
            
            # Find link text
            link_text_start = html_content.find('>', end_pos) + 1
            link_text_end = html_content.find('</a>', link_text_start)
            link_text = html_content[link_text_start:link_text_end].strip() if link_text_end > link_text_start else ""
            
            # Normalize URL
            if href_value.startswith('/'):
                domain = '/'.join(base_url.split('/')[:3])  # Get domain from base URL
                full_url = f"{domain}{href_value}"
            elif not href_value.startswith(('http://', 'https://')):
                full_url = f"{base_url.rstrip('/')}/{href_value.lstrip('/')}"
            else:
                full_url = href_value
            
            links.append({
                "url": full_url,
                "text": link_text
            })
            
            current_pos = end_pos
        
        return links


#######################
# 2. Specialized Strategy Example
#######################

@strategy_metadata(
    strategy_type=StrategyType.TRAVERSAL,
    capabilities={
        StrategyCapability.PAGINATION_HANDLING,
        StrategyCapability.RATE_LIMITING
    },
    description="Example pagination strategy that handles multi-page content."
)
class PaginationStrategy(BaseStrategyV2):
    """
    A strategy that traverses paginated content.
    """
    
    @property
    def name(self) -> str:
        return "pagination_strategy"
    
    def __init__(self, context: StrategyContext):
        super().__init__(context)
        self.page_count = 0
        self.max_pages = 3  # Default limit for demonstration
    
    def execute(self, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Execute the pagination strategy."""
        logger.info(f"Executing PaginationStrategy on {url}")
        
        # Override max_pages if provided in kwargs
        self.max_pages = kwargs.get('max_pages', self.max_pages)
        
        # Start with the first page
        current_url = url
        self.page_count = 0
        all_items = []
        
        while current_url and self.page_count < self.max_pages:
            # Fetch current page
            logger.info(f"Fetching page {self.page_count + 1}: {current_url}")
            content = self._simulate_fetch_url(current_url)
            
            if not content:
                break
                
            # Extract items from current page
            items, next_url = self._extract_items_and_next_page(content, current_url)
            all_items.extend(items)
            
            # Update for next iteration
            self.page_count += 1
            current_url = next_url
            
            # Simulate rate limiting
            if current_url:
                logger.info(f"Rate limiting - sleeping before next page...")
                time.sleep(0.2)  # Small delay for demonstration
        
        # Store the results
        result = {
            "start_url": url,
            "pages_visited": self.page_count,
            "total_items": len(all_items),
            "items": all_items
        }
        self._results.append(result)
        
        return result
    
    def _simulate_fetch_url(self, url: str) -> Optional[str]:
        """Simulate fetching content from a URL."""
        # In a real implementation, this would use proper URL service and requests
        # Similar to the method in SimpleExtractionStrategy but with pagination markers
        
        # Extract page number from URL if present
        page_param = "page="
        page_number = 1
        if page_param in url:
            try:
                page_number = int(url.split(page_param)[1].split('&')[0])
            except (ValueError, IndexError):
                pass
        
        # Generate simulated paginated content
        return f"""
        <html>
            <head>
                <title>Page {page_number} - Paginated Results</title>
            </head>
            <body>
                <h1>Products - Page {page_number}</h1>
                <div class="products">
                    <div class="product">Product {page_number}-1</div>
                    <div class="product">Product {page_number}-2</div>
                    <div class="product">Product {page_number}-3</div>
                </div>
                <div class="pagination">
                    {"<a href=\"" + url.split('?')[0] + f"?page={page_number+1}\">Next Page</a>" 
                     if page_number < 5 else "Last Page"}
                </div>
            </body>
        </html>
        """
    
    def _extract_items_and_next_page(self, content: str, current_url: str) -> tuple:
        """Extract items and next page URL from the content."""
        # Extract items (simple version for demonstration)
        items = []
        
        # Find products
        start_pos = 0
        while True:
            prod_start = content.find('<div class="product">', start_pos)
            if prod_start == -1:
                break
                
            prod_end = content.find('</div>', prod_start)
            if prod_end == -1:
                break
                
            product_text = content[prod_start + len('<div class="product">'):prod_end].strip()
            items.append({"title": product_text})
            
            start_pos = prod_end
        
        # Extract next page URL
        next_url = None
        next_marker = 'Next Page'
        if next_marker in content:
            href_pos = content.rfind('href="', 0, content.find(next_marker))
            if href_pos != -1:
                url_start = href_pos + len('href="')
                url_end = content.find('"', url_start)
                next_url = content[url_start:url_end]
                
                # Handle relative URLs
                if next_url.startswith('/'):
                    base_parts = current_url.split('/')
                    domain = '/'.join(base_parts[:3])
                    next_url = f"{domain}{next_url}"
                elif not next_url.startswith(('http://', 'https://')):
                    base_url = '/'.join(current_url.split('/')[:-1])
                    next_url = f"{base_url}/{next_url}"
        
        return items, next_url


#######################
# 3. Composite Strategy Examples
#######################

def create_sequential_strategy(context: StrategyContext) -> SequentialStrategy:
    """Create an example sequential strategy."""
    # Create individual strategies
    extraction_strategy = SimpleExtractionStrategy(context)
    pagination_strategy = PaginationStrategy(context)
    
    # Create sequential strategy
    sequential = SequentialStrategy(context)
    sequential.add_strategy(extraction_strategy)
    sequential.add_strategy(pagination_strategy)
    
    return sequential

def create_fallback_strategy(context: StrategyContext) -> FallbackStrategy:
    """Create an example fallback strategy."""
    # Create a strategy that will fail
    @strategy_metadata(
        strategy_type=StrategyType.EXTRACTION,
        capabilities={StrategyCapability.API_INTERACTION},
        description="A strategy that always fails for demonstration."
    )
    class FailingStrategy(BaseStrategyV2):
        @property
        def name(self) -> str:
            return "failing_strategy"
        
        def execute(self, url: str, **kwargs) -> Optional[Dict[str, Any]]:
            logger.info(f"FailingStrategy attempting to execute on {url}")
            self.handle_error(
                message="This strategy always fails",
                category=StrategyErrorCategory.UNKNOWN,
                severity=StrategyErrorSeverity.ERROR
            )
            return None
    
    # Create strategies
    failing = FailingStrategy(context)
    extraction = SimpleExtractionStrategy(context)
    
    # Create fallback strategy
    fallback = FallbackStrategy(context)
    fallback.add_strategy(failing)
    fallback.add_strategy(extraction)
    
    return fallback

def create_pipeline_strategy(context: StrategyContext) -> PipelineStrategy:
    """Create an example pipeline strategy."""
    # Create first stage: URL discovery
    @strategy_metadata(
        strategy_type=StrategyType.TRAVERSAL,
        capabilities={StrategyCapability.SITEMAP_DISCOVERY},
        description="First stage: discovers URLs"
    )
    class DiscoveryStrategy(BaseStrategyV2):
        @property
        def name(self) -> str:
            return "discovery_strategy"
        
        def execute(self, url: str, **kwargs) -> Optional[Dict[str, Any]]:
            logger.info(f"DiscoveryStrategy discovering URLs from {url}")
            # Simulate discovering URLs
            discovered_urls = [
                f"{url}/product/item1",
                f"{url}/product/item2",
                f"{url}/category/electronics"
            ]
            result = {
                "discovered_urls": discovered_urls
            }
            self._results.append(result)
            return result
    
    # Create second stage: processing discovered URLs
    @strategy_metadata(
        strategy_type=StrategyType.EXTRACTION,
        capabilities={StrategyCapability.CONTENT_NORMALIZATION},
        description="Second stage: processes discovered URLs"
    )
    class ProcessingStrategy(BaseStrategyV2):
        @property
        def name(self) -> str:
            return "processing_strategy"
        
        def execute(self, url: str, **kwargs) -> Optional[Dict[str, Any]]:
            logger.info(f"ProcessingStrategy processing data")
            # Get discovered URLs from previous stage
            discovered_urls = kwargs.get("discovered_urls", [])
            if not discovered_urls:
                logger.warning("No URLs to process")
                return None
            
            # Process each URL (simplified for demo)
            processed_data = []
            for url in discovered_urls:
                processed_data.append({
                    "url": url,
                    "processed": True,
                    "title": f"Processed {url.split('/')[-1]}"
                })
            
            result = {
                "processed_count": len(processed_data),
                "data": processed_data
            }
            self._results.append(result)
            return result
    
    # Create the strategies
    discovery = DiscoveryStrategy(context)
    processing = ProcessingStrategy(context)
    
    # Create pipeline strategy
    pipeline = PipelineStrategy(context)
    pipeline.add_strategy(discovery)
    pipeline.add_strategy(processing)
    
    return pipeline


#######################
# 4. Custom Strategy Example
#######################

@strategy_metadata(
    strategy_type=StrategyType.SPECIAL_PURPOSE,
    capabilities={
        StrategyCapability.SITE_SPECIFIC,
        StrategyCapability.ERROR_HANDLING
    },
    description="Site-specific strategy for e-commerce sites."
)
class ECommerceStrategy(BaseStrategyV2):
    """
    A specialized strategy for e-commerce sites that can handle
    product listings, detail pages, and price extraction.
    """
    
    @property
    def name(self) -> str:
        return "ecommerce_strategy"
    
    def execute(self, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Execute the e-commerce strategy based on URL type."""
        logger.info(f"Executing ECommerceStrategy on {url}")
        
        # Determine URL type
        url_type = self._determine_url_type(url)
        
        # Execute appropriate method based on URL type
        if url_type == "product":
            result = self._handle_product_page(url)
        elif url_type == "category":
            result = self._handle_category_page(url)
        elif url_type == "cart":
            result = self._handle_cart_page(url)
        else:
            result = self._handle_generic_page(url)
        
        if result:
            self._results.append(result)
        
        return result
    
    def _determine_url_type(self, url: str) -> str:
        """Determine the type of URL (product, category, etc.)."""
        if "/product/" in url or "/item/" in url or "/p/" in url:
            return "product"
        elif "/category/" in url or "/c/" in url or "/department/" in url:
            return "category"
        elif "/cart" in url or "/basket" in url:
            return "cart"
        else:
            return "generic"
    
    def _handle_product_page(self, url: str) -> Dict[str, Any]:
        """Handle a product detail page."""
        logger.info(f"Handling product page: {url}")
        
        # Simulate fetching and extracting product data
        content = self._simulate_fetch_url(url)
        
        # Extract product details
        title = self._extract_title(content)
        price = self._extract_price(content)
        description = self._extract_description(content)
        
        return {
            "url": url,
            "type": "product",
            "title": title,
            "price": price,
            "description": description
        }
    
    def _handle_category_page(self, url: str) -> Dict[str, Any]:
        """Handle a category listing page."""
        logger.info(f"Handling category page: {url}")
        
        # Simulate fetching and extracting category data
        content = self._simulate_fetch_url(url)
        
        # Extract category details
        title = self._extract_title(content)
        products = self._extract_products(content, url)
        
        return {
            "url": url,
            "type": "category",
            "title": title,
            "product_count": len(products),
            "products": products
        }
    
    def _handle_cart_page(self, url: str) -> Dict[str, Any]:
        """Handle a shopping cart page."""
        logger.info(f"Handling cart page: {url}")
        
        # Simulate fetching and extracting cart data
        content = self._simulate_fetch_url(url)
        
        # Extract cart details
        items = self._extract_cart_items(content)
        total = self._calculate_cart_total(items)
        
        return {
            "url": url,
            "type": "cart",
            "item_count": len(items),
            "items": items,
            "total": total
        }
    
    def _handle_generic_page(self, url: str) -> Dict[str, Any]:
        """Handle a generic e-commerce page."""
        logger.info(f"Handling generic page: {url}")
        
        # Simulate fetching and basic extraction
        content = self._simulate_fetch_url(url)
        
        # Basic extraction
        title = self._extract_title(content)
        links = self._extract_links(content, url)
        
        return {
            "url": url,
            "type": "generic",
            "title": title,
            "link_count": len(links)
        }
    
    def _simulate_fetch_url(self, url: str) -> str:
        """Simulate fetching content for various e-commerce page types."""
        # URL-based content simulation for demonstration
        if "product" in url:
            product_name = url.split("/")[-1].replace("-", " ").title()
            price = f"${random.randint(10, 200)}.{random.randint(0, 99):02d}"
            return f"""
            <html>
                <head>
                    <title>{product_name} | Example Store</title>
                    <meta name="description" content="Buy {product_name} at the best price!">
                </head>
                <body>
                    <h1>{product_name}</h1>
                    <div class="price">{price}</div>
                    <div class="description">
                        This is an amazing {product_name}. Features include...
                    </div>
                </body>
            </html>
            """
        elif "category" in url:
            category_name = url.split("/")[-1].replace("-", " ").title()
            return f"""
            <html>
                <head>
                    <title>{category_name} | Example Store</title>
                </head>
                <body>
                    <h1>{category_name}</h1>
                    <div class="products">
                        <div class="product">
                            <a href="/product/item1">Product 1</a>
                            <div class="price">$19.99</div>
                        </div>
                        <div class="product">
                            <a href="/product/item2">Product 2</a>
                            <div class="price">$29.99</div>
                        </div>
                        <div class="product">
                            <a href="/product/item3">Product 3</a>
                            <div class="price">$39.99</div>
                        </div>
                    </div>
                </body>
            </html>
            """
        elif "cart" in url:
            return """
            <html>
                <head>
                    <title>Shopping Cart | Example Store</title>
                </head>
                <body>
                    <h1>Your Shopping Cart</h1>
                    <div class="cart-items">
                        <div class="item">
                            <span class="name">Product 1</span>
                            <span class="price">$19.99</span>
                            <span class="quantity">2</span>
                        </div>
                        <div class="item">
                            <span class="name">Product 2</span>
                            <span class="price">$29.99</span>
                            <span class="quantity">1</span>
                        </div>
                    </div>
                    <div class="total">$69.97</div>
                </body>
            </html>
            """
        else:
            return f"""
            <html>
                <head>
                    <title>Example Store - Online Shopping</title>
                </head>
                <body>
                    <h1>Welcome to Example Store</h1>
                    <nav>
                        <a href="/category/electronics">Electronics</a>
                        <a href="/category/clothing">Clothing</a>
                        <a href="/category/home">Home & Garden</a>
                        <a href="/cart">View Cart</a>
                    </nav>
                    <div class="featured">
                        <h2>Featured Products</h2>
                        <div class="product">
                            <a href="/product/featured1">Featured Product 1</a>
                            <div class="price">$49.99</div>
                        </div>
                    </div>
                </body>
            </html>
            """
    
    # Various extraction helpers
    def _extract_title(self, content: str) -> str:
        """Extract title from content."""
        # Implementation similar to previous examples
        title_pattern = "<title>"
        if title_pattern in content:
            start = content.find(title_pattern) + len(title_pattern)
            end = content.find("</title>", start)
            return content[start:end].strip()
        return "Unknown Title"
    
    def _extract_price(self, content: str) -> Optional[str]:
        """Extract price from content."""
        price_pattern = '<div class="price">'
        if price_pattern in content:
            start = content.find(price_pattern) + len(price_pattern)
            end = content.find("</div>", start)
            return content[start:end].strip()
        return None
    
    def _extract_products(self, content: str, base_url: str) -> List[Dict[str, Any]]:
        """Extract products from category page."""
        products = []
        start_pos = 0
        
        while True:
            prod_start = content.find('<div class="product">', start_pos)
            if prod_start == -1:
                break
                
            prod_end = content.find('</div>', prod_start)
            if prod_end == -1:
                break
                
            # Extract product details
            product_html = content[prod_start:prod_end]
            
            # Extract product link
            url = None
            link_start = product_html.find('href="') + 6
            if link_start > 5:  # If 'href="' was found
                link_end = product_html.find('"', link_start)
                url_path = product_html[link_start:link_end]
                
                # Handle relative URLs
                if url_path.startswith('/'):
                    domain = '/'.join(base_url.split('/')[:3])
                    url = f"{domain}{url_path}"
                else:
                    url = url_path
            
            # Extract product name
            name = "Unknown Product"
            name_start = product_html.find('>', link_end) + 1
            name_end = product_html.find('</a>', name_start)
            if name_end > name_start:
                name = product_html[name_start:name_end].strip()
            
            # Extract price
            price = None
            price_start = product_html.find('<div class="price">') + 18
            if price_start > 17:  # If price div was found
                price_end = product_html.find('</div>', price_start)
                price = product_html[price_start:price_end].strip()
            
            products.append({
                "name": name,
                "url": url,
                "price": price
            })
            
            start_pos = prod_end
        
        return products
    
    def _extract_cart_items(self, content: str) -> List[Dict[str, Any]]:
        """Extract items from cart page."""
        items = []
        start_pos = 0
        
        while True:
            item_start = content.find('<div class="item">', start_pos)
            if item_start == -1:
                break
                
            item_end = content.find('</div>', item_start)
            if item_end == -1:
                break
                
            item_html = content[item_start:item_end]
            
            # Extract name
            name = None
            name_start = item_html.find('<span class="name">') + 19
            if name_start > 18:
                name_end = item_html.find('</span>', name_start)
                name = item_html[name_start:name_end].strip()
            
            # Extract price
            price = None
            price_start = item_html.find('<span class="price">') + 20
            if price_start > 19:
                price_end = item_html.find('</span>', price_start)
                price = item_html[price_start:price_end].strip()
            
            # Extract quantity
            quantity = 1
            qty_start = item_html.find('<span class="quantity">') + 23
            if qty_start > 22:
                qty_end = item_html.find('</span>', qty_start)
                try:
                    quantity = int(item_html[qty_start:qty_end].strip())
                except ValueError:
                    pass
            
            items.append({
                "name": name or "Unknown Item",
                "price": price or "$0.00",
                "quantity": quantity
            })
            
            start_pos = item_end
        
        return items
    
    def _calculate_cart_total(self, items: List[Dict[str, Any]]) -> str:
        """Calculate total from cart items."""
        total = 0.0
        
        for item in items:
            price_str = item.get("price", "$0.00")
            quantity = item.get("quantity", 1)
            
            try:
                # Extract numeric price value
                price_val = float(price_str.replace('$', '').replace(',', ''))
                total += price_val * quantity
            except ValueError:
                pass
        
        return f"${total:.2f}"
    
    def _extract_links(self, content: str, base_url: str) -> List[Dict]:
        """Extract links from content."""
        # Implementation similar to previous examples
        links = []
        current_pos = 0
        
        while True:
            href_marker = 'href="'
            current_pos = content.find(href_marker, current_pos)
            
            if current_pos == -1:
                break
                
            start_pos = current_pos + len(href_marker)
            end_pos = content.find('"', start_pos)
            
            if end_pos == -1:
                break
                
            href_value = content[start_pos:end_pos]
            
            # Find link text
            link_text_start = content.find('>', end_pos) + 1
            link_text_end = content.find('</a>', link_text_start)
            link_text = content[link_text_start:link_text_end].strip() if link_text_end > link_text_start else ""
            
            # Normalize URL
            if href_value.startswith('/'):
                domain = '/'.join(base_url.split('/')[:3])  # Get domain from base URL
                full_url = f"{domain}{href_value}"
            elif not href_value.startswith(('http://', 'https://')):
                full_url = f"{base_url.rstrip('/')}/{href_value.lstrip('/')}"
            else:
                full_url = href_value
            
            links.append({
                "url": full_url,
                "text": link_text
            })
            
            current_pos = end_pos
        
        return links


#######################
# 5. Controller Integration Example
#######################

def run_with_adaptive_scraper():
    """Demonstrate using the adaptive scraper with strategies."""
    # Create and initialize the scraper
    scraper = AdaptiveScraper()
    
    # Register our custom strategies
    scraper._register_strategy(SimpleExtractionStrategy)
    scraper._register_strategy(PaginationStrategy)
    scraper._register_strategy(ECommerceStrategy)
    
    # Example 1: Use a specific strategy
    logger.info("\n\n===== ADAPTIVE SCRAPER EXAMPLE 1: SPECIFIC STRATEGY =====")
    result1 = scraper.scrape(
        "https://example.com/product/sample-item",
        strategy_name="ecommerce_strategy"
    )
    logger.info(f"Result from ecommerce_strategy: {result1}")
    
    # Example 2: Use capabilities to select a strategy
    logger.info("\n\n===== ADAPTIVE SCRAPER EXAMPLE 2: CAPABILITY-BASED SELECTION =====")
    result2 = scraper.scrape(
        "https://example.com/category/electronics?page=1",
        required_capabilities={StrategyCapability.PAGINATION_HANDLING}
    )
    logger.info(f"Result from pagination capability: {result2}")
    
    # Example 3: Handle error case gracefully
    logger.info("\n\n===== ADAPTIVE SCRAPER EXAMPLE 3: ERROR HANDLING =====")
    result3 = scraper.scrape(
        "https://example.com/error-page",
        strategy_name="failing_strategy"  # This strategy doesn't exist
    )
    logger.info(f"Result from non-existent strategy: {result3}")
    
    # Cleanup
    scraper.shutdown()


#######################
# Running the Examples
#######################

def main():
    """Run all strategy examples."""
    logger.info("=== STRATEGY PATTERN EXAMPLES ===")
    
    # Set up context and factory
    context = StrategyContext({
        "max_retries": 3,
        "timeout": 30
    })
    factory = StrategyFactory(context)
    
    # Register strategies
    factory.register_strategy(SimpleExtractionStrategy)
    factory.register_strategy(PaginationStrategy)
    factory.register_strategy(ECommerceStrategy)
    
    # Example 1: Simple strategy execution
    logger.info("\n\n===== EXAMPLE 1: SIMPLE STRATEGY =====")
    simple = factory.get_strategy("simple_extraction")
    result1 = simple.execute("https://example.com/product/item123")
    logger.info(f"Simple strategy result: {result1}")
    
    # Example 2: Pagination strategy execution
    logger.info("\n\n===== EXAMPLE 2: PAGINATION STRATEGY =====")
    pagination = factory.get_strategy("pagination_strategy")
    result2 = pagination.execute("https://example.com/listing?page=1", max_pages=2)
    logger.info(f"Pagination strategy result: {result2}")
    
    # Example 3: E-commerce strategy on different URL types
    logger.info("\n\n===== EXAMPLE 3: E-COMMERCE STRATEGY =====")
    ecommerce = factory.get_strategy("ecommerce_strategy")
    
    # Product page
    logger.info("\nE-commerce strategy on product page:")
    product_result = ecommerce.execute("https://example.com/product/laptop")
    logger.info(f"Product page result: {product_result}")
    
    # Category page
    logger.info("\nE-commerce strategy on category page:")
    category_result = ecommerce.execute("https://example.com/category/electronics")
    logger.info(f"Category page result: {category_result}")
    
    # Cart page
    logger.info("\nE-commerce strategy on cart page:")
    cart_result = ecommerce.execute("https://example.com/cart")
    logger.info(f"Cart page result: {cart_result}")
    
    # Example 4: Sequential composite strategy
    logger.info("\n\n===== EXAMPLE 4: SEQUENTIAL COMPOSITE STRATEGY =====")
    sequential = create_sequential_strategy(context)
    seq_result = sequential.execute("https://example.com")
    logger.info(f"Sequential strategy result: {seq_result}")
    
    # Example 5: Fallback composite strategy
    logger.info("\n\n===== EXAMPLE 5: FALLBACK COMPOSITE STRATEGY =====")
    fallback = create_fallback_strategy(context)
    fb_result = fallback.execute("https://example.com")
    logger.info(f"Fallback strategy result: {fb_result}")
    
    # Example 6: Pipeline composite strategy
    logger.info("\n\n===== EXAMPLE 6: PIPELINE COMPOSITE STRATEGY =====")
    pipeline = create_pipeline_strategy(context)
    pipe_result = pipeline.execute("https://example.com")
    logger.info(f"Pipeline strategy result: {pipe_result}")
    
    # Example 7: Using with AdaptiveScraper
    logger.info("\n\n===== EXAMPLE 7: ADAPTIVE SCRAPER INTEGRATION =====")
    run_with_adaptive_scraper()
    
    logger.info("\n\n=== ALL EXAMPLES COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()