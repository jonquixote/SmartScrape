"""
Listing Analyzer Module

This module provides functionality to detect and analyze listing patterns
on websites, such as product grids, article lists, and other repeating elements.
"""

import re
import logging
import itertools
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from collections import Counter
from bs4 import BeautifulSoup, Tag

from components.pattern_analyzer.base_analyzer import PatternAnalyzer, get_registry

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ListingAnalyzer")

class ListingAnalyzer(PatternAnalyzer):
    """
    Analyzer for detecting and analyzing listing patterns on web pages.
    This includes product grids, search results, article lists, and any
    repeating content structures.
    """
    
    def __init__(self, confidence_threshold: float = 0.7, min_items: int = 3):
        """
        Initialize the listing analyzer.
        
        Args:
            confidence_threshold: Minimum confidence level for pattern detection
            min_items: Minimum number of items to consider as a listing
        """
        super().__init__(confidence_threshold)
        self.min_items = min_items
        
        # List candidate selectors that commonly represent listing containers
        self.listing_container_candidates = [
            # Common product grid selectors
            'div.products', '.product-grid', '.product-list', 'ul.products',
            # Common search results selectors
            '.search-results', '.results', '.result-list', 
            # Common article list selectors
            '.article-list', '.news-list', '.blog-posts', 'div.articles',
            # Common generic list selectors
            '.items', '.listing', '.grid', '.card-container', '.tiles'
        ]
        
        # Patterns that indicate an element might be a list item
        self.item_patterns = {
            'class': [re.compile(r'product', re.I), re.compile(r'item', re.I), re.compile(r'result', re.I), 
                    re.compile(r'card', re.I), re.compile(r'article', re.I), re.compile(r'post', re.I),
                    re.compile(r'listing', re.I), re.compile(r'tile', re.I), re.compile(r'entry', re.I)]
        }
    
    async def analyze_page(self, html: str, url: str) -> Dict[str, Any]:
        """
        Analyze a page to detect patterns and structure with graceful degradation.
        This method provides robust analysis even when parts of the process fail.
        
        Args:
            html: HTML content of the page
            url: URL of the page
            
        Returns:
            Dictionary with page analysis results including detected patterns
        """
        logger.info(f"Analyzing page structure for {url}")
        
        # Default response structure with fallback values
        page_analysis = {
            "content_type": "unknown",
            "site_type": "unknown",
            "has_search": False,
            "has_pagination": False,
            "structure": {
                "has_header": False,
                "has_footer": False,
                "has_sidebar": False,
                "has_main_content": True,
                "has_navigation": False,
                "element_counts": {}
            },
            "patterns": {"listings": []}
        }
        
        try:
            # Parse HTML - this is a critical step, so handle separately
            soup = self.parse_html(html)
            if not soup:
                logger.warning(f"Failed to parse HTML for {url}")
                return page_analysis
                
            # Try to perform listing analysis, but continue if it fails
            try:
                listing_results = await self.analyze(html, url)
                page_analysis["patterns"] = listing_results
                
                # Try to determine content type based on listings
                if listing_results.get("listings"):
                    try:
                        # Get the most confident listing
                        best_listing = max(listing_results["listings"], key=lambda x: x["confidence_score"])
                        listing_type = best_listing.get("listing_type", "generic_listing")
                        
                        # Map listing type to content type
                        if listing_type == "product_listing":
                            page_analysis["content_type"] = "listing_page"
                            page_analysis["site_type"] = "ecommerce"
                        elif listing_type == "search_results":
                            page_analysis["content_type"] = "search_results"
                            page_analysis["site_type"] = "search"
                        elif listing_type == "article_listing":
                            page_analysis["content_type"] = "article_list"
                            page_analysis["site_type"] = "content"
                        elif listing_type == "media_gallery":
                            page_analysis["content_type"] = "gallery"
                            page_analysis["site_type"] = "media"
                    except Exception as content_error:
                        logger.warning(f"Error determining content type: {str(content_error)}")
                        # Continue with default values
            except Exception as listing_error:
                logger.warning(f"Listing analysis failed: {str(listing_error)}", exc_info=True)
                # We continue with other analyses even if this one fails
            
            # Even if listing analysis fails, try to detect search functionality
            try:
                search_forms = soup.find_all('form', action=True)
                search_inputs = soup.find_all('input', {'type': 'search'})
                search_buttons = soup.find_all('button', string=re.compile(r'search', re.I))
                
                page_analysis["has_search"] = bool(search_forms or search_inputs or search_buttons)
            except Exception as search_error:
                logger.warning(f"Search detection failed: {str(search_error)}")
            
            # Try to detect pagination
            try:
                pagination_classes = ['pagination', 'pager', 'pages', 'page-numbers']
                pagination_elements = []
                
                for cls in pagination_classes:
                    pagination_elements.extend(soup.find_all(class_=re.compile(cls, re.I)))
                
                # Look for numbered links
                page_links = soup.find_all('a', string=re.compile(r'^\d+$'))
                
                # Look for next/prev navigation
                next_links = soup.find_all('a', string=re.compile(r'next|>>', re.I))
                prev_links = soup.find_all('a', string=re.compile(r'prev|<<|previous', re.I))
                
                page_analysis["has_pagination"] = bool(pagination_elements or page_links or next_links or prev_links)
            except Exception as pagination_error:
                logger.warning(f"Pagination detection failed: {str(pagination_error)}")
            
            # Try to analyze page structure
            try:
                page_analysis["structure"] = self._analyze_page_structure(soup)
            except Exception as structure_error:
                logger.warning(f"Structure analysis failed: {str(structure_error)}")
                # Keep default structure
            
        except Exception as e:
            logger.error(f"Overall page analysis failed: {str(e)}", exc_info=True)
            # Return default structure with basic information
        
        return page_analysis
    
    def _analyze_page_structure(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Analyze the general structure of a page.
        
        Args:
            soup: BeautifulSoup object with parsed HTML
            
        Returns:
            Dictionary with page structure information
        """
        structure = {
            "has_header": bool(soup.find('header')),
            "has_footer": bool(soup.find('footer')),
            "has_sidebar": bool(soup.find(['aside', 'div'], class_=re.compile(r'sidebar', re.I))),
            "has_main_content": bool(soup.find('main') or soup.find(['div', 'section'], id=re.compile(r'content|main', re.I))),
            "has_navigation": bool(soup.find('nav') or soup.find(['div', 'ul'], class_=re.compile(r'nav|menu', re.I))),
            "element_counts": {
                "images": len(soup.find_all('img')),
                "links": len(soup.find_all('a')),
                "buttons": len(soup.find_all('button')),
                "forms": len(soup.find_all('form')),
                "tables": len(soup.find_all('table')),
                "headings": len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']))
            }
        }
        
        return structure
        
    async def analyze(self, html: str, url: str) -> Dict[str, Any]:
        """
        Analyze a page to detect listing patterns.
        
        Args:
            html: HTML content of the page
            url: URL of the page
            
        Returns:
            Dictionary with detected listing patterns
        """
        logger.info(f"Analyzing listing patterns on {url}")
        soup = self.parse_html(html)
        domain = self.get_domain(url)
        
        # Results will contain all detected listing patterns
        results = {
            "listings": [],
            "confidence_scores": {}
        }
        
        # First try to find listings using common selectors
        common_listings = self._find_common_listings(soup)
        
        # Then try a more general approach to find repeating structures
        if not common_listings:
            logger.info("No common listings found, looking for repeating structures")
            common_listings = self._find_repeating_structures(soup)
        
        # Add any found listings to results
        for listing_idx, listing_data in enumerate(common_listings):
            if listing_data["confidence_score"] >= self.confidence_threshold:
                listing_id = listing_data.get("listing_id", f"listing_{listing_idx}")
                results["listings"].append(listing_data)
                results["confidence_scores"][listing_id] = listing_data["confidence_score"]
                logger.info(f"Detected listing: {listing_id} with confidence {listing_data['confidence_score']:.2f}")
        
        # Register the best listing pattern in the global registry
        if results["listings"]:
            best_listing = max(results["listings"], key=lambda x: x["confidence_score"])
            get_registry().register_pattern(
                pattern_type="listing",
                url=url,
                pattern_data=best_listing,
                confidence=best_listing["confidence_score"]
            )
            logger.info(f"Registered listing pattern for {domain}")
        
        return results
    
    def _find_common_listings(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Find listings using common container selectors.
        
        Args:
            soup: BeautifulSoup object with parsed HTML
            
        Returns:
            List of listing data dictionaries
        """
        listings = []
        
        # Try each common listing container selector
        for selector in self.listing_container_candidates:
            containers = soup.select(selector)
            
            for container_idx, container in enumerate(containers):
                # Skip small containers
                if len(container.find_all()) < self.min_items * 2:  # Each item likely has at least 2 elements
                    continue
                
                # Analyze the container to determine if it's a listing
                listing_data = self._analyze_container(container, container_idx)
                
                # If it's likely to be a listing, add it
                if listing_data["confidence_score"] > self.confidence_threshold:
                    listings.append(listing_data)
                    logger.info(f"Found listing with selector: {selector}")
        
        return listings
    
    def _find_repeating_structures(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Find repeating structures that may represent listings.
        
        Args:
            soup: BeautifulSoup object with parsed HTML
            
        Returns:
            List of listing data dictionaries
        """
        listings = []
        
        # Look for containers that have multiple children with the same tag
        potential_containers = []
        
        # Common container tags
        container_tags = ['div', 'ul', 'ol', 'section', 'article']
        
        for tag in container_tags:
            containers = soup.find_all(tag)
            
            for container in containers:
                # Count the number of immediate children by tag
                child_tags = [child.name for child in container.children if isinstance(child, Tag)]
                tag_counter = Counter(child_tags)
                
                # If any tag appears multiple times, this might be a listing
                for child_tag, count in tag_counter.items():
                    if count >= self.min_items and child_tag in ['div', 'li', 'article', 'section', 'a']:
                        potential_containers.append((container, child_tag, count))
        
        # Sort by count descending to prioritize containers with more items
        potential_containers.sort(key=lambda x: x[2], reverse=True)
        
        # Analyze each potential container
        for container_idx, (container, child_tag, count) in enumerate(potential_containers):
            # Get the immediate children of the specified tag
            items = [child for child in container.children if isinstance(child, Tag) and child.name == child_tag]
            
            # Skip if too few items
            if len(items) < self.min_items:
                continue
            
            # Check for similar class attributes
            class_similarities = self._check_class_similarities(items)
            
            # If items share similar classes, this is likely a listing
            if class_similarities > 0.7:
                listing_data = self._analyze_with_items(container, items, container_idx)
                
                # Add to results if confidence is high enough
                if listing_data["confidence_score"] > self.confidence_threshold:
                    listings.append(listing_data)
                    logger.info(f"Found repeating structure: {count} x {child_tag} in {container.name}")
        
        return listings
    
    def _check_class_similarities(self, items: List[Tag]) -> float:
        """
        Check how similar the class attributes are across a list of elements.
        
        Args:
            items: List of BeautifulSoup tag elements
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        if not items:
            return 0.0
            
        # Extract classes from each item
        item_classes = []
        for item in items:
            if item.get('class'):
                item_classes.append(set(item['class']))
            else:
                item_classes.append(set())
        
        # If no classes were found, return low similarity
        if not any(item_classes):
            return 0.0
            
        # Calculate similarity based on common classes
        similarity_scores = []
        
        for i, j in itertools.combinations(range(len(item_classes)), 2):
            set_i = item_classes[i]
            set_j = item_classes[j]
            
            # Skip empty sets
            if not set_i or not set_j:
                continue
                
            # Calculate Jaccard similarity (intersection over union)
            intersection = len(set_i.intersection(set_j))
            union = len(set_i.union(set_j))
            
            if union > 0:
                similarity_scores.append(intersection / union)
        
        # Return average similarity
        return sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
    
    def _analyze_container(self, container: Tag, container_idx: int) -> Dict[str, Any]:
        """
        Analyze a container to determine if it contains a listing.
        
        Args:
            container: BeautifulSoup container element
            container_idx: Index of the container
            
        Returns:
            Dictionary with container analysis data
        """
        # Look for potential list items within the container
        potential_items = []
        
        # First try specific item selectors
        item_selectors = [
            # Generic item selectors
            'li', '.item', '[class*="item"]', '.product', '[class*="product"]',
            '.result', '[class*="result"]', '.card', '[class*="card"]',
            '.article', '[class*="article"]', '.post', '[class*="post"]',
            '.entry', '[class*="entry"]', '.tile', '[class*="tile"]'
        ]
        
        for selector in item_selectors:
            items = container.select(selector)
            if len(items) >= self.min_items:
                potential_items = items
                break
        
        # If no specific items found, look for repeating child structures
        if not potential_items:
            # Get immediate children
            children = [child for child in container.children if isinstance(child, Tag)]
            
            # Count tag occurrences
            tag_counter = Counter([child.name for child in children])
            
            # Find the most common tag with enough occurrences
            most_common = tag_counter.most_common(1)
            if most_common and most_common[0][1] >= self.min_items:
                most_common_tag = most_common[0][0]
                potential_items = [child for child in children if child.name == most_common_tag]
        
        # If we've found enough potential items, analyze them
        if len(potential_items) >= self.min_items:
            return self._analyze_with_items(container, potential_items, container_idx)
        
        # Return low confidence if no suitable items found
        return {
            "listing_id": f"container_{container_idx}",
            "container_type": container.name,
            "items": [],
            "item_count": 0,
            "confidence_score": 0.0,
            "selector": self._generate_container_selector(container)
        }
    
    def _analyze_with_items(self, container: Tag, items: List[Tag], container_idx: int) -> Dict[str, Any]:
        """
        Analyze a container with its identified list items.
        
        Args:
            container: BeautifulSoup container element
            items: List of potential list item elements
            container_idx: Index of the container
            
        Returns:
            Dictionary with listing analysis data
        """
        container_attrs = {
            'id': container.get('id', ''),
            'class': container.get('class', []),
            'role': container.get('role', '')
        }
        
        # Evidence for this being a listing
        evidence_points = []
        
        # Check container attributes for listing indicators
        container_class_str = ' '.join(container_attrs['class']) if isinstance(container_attrs['class'], list) else container_attrs['class']
        container_id = container_attrs['id']
        
        if re.search(r'list|grid|products|results|items|cards|articles|posts|entries|tiles', 
                   container_class_str + ' ' + container_id, re.I):
            evidence_points.append(0.8)
        
        # Check ARIA role
        if container_attrs['role'] in ['list', 'grid', 'feed', 'listbox']:
            evidence_points.append(0.9)
        
        # If container is a ul or ol, it's likely a list
        if container.name in ['ul', 'ol']:
            evidence_points.append(0.9)
        
        # More items suggest a stronger listing pattern
        item_count = len(items)
        item_confidence = min(1.0, item_count / 10)  # Cap at 1.0
        evidence_points.append(item_confidence)
        
        # Check for structural similarities among items
        # Similar structure strongly indicates a listing
        structure_similarity = self._check_structural_similarity(items)
        evidence_points.append(structure_similarity)
        
        # Check content type similarity
        content_similarity = self._check_content_similarity(items)
        evidence_points.append(content_similarity)
        
        # Calculate confidence score
        confidence_score = self.calculate_confidence(evidence_points)
        
        # Analyze a sample of items to determine the listing type and content structure
        item_samples = self._analyze_item_samples(items[:min(5, len(items))])
        
        # Determine listing type based on container and item characteristics
        listing_type = self._determine_listing_type(container, items, item_samples)
        
        # Generate selectors for the items
        item_selector = self._generate_item_selector(container, items[0])
        
        listing_data = {
            "listing_id": container_attrs['id'] or f"listing_{container_idx}",
            "listing_type": listing_type,
            "container_type": container.name,
            "container_attrs": container_attrs,
            "item_count": item_count,
            "items": item_samples,
            "confidence_score": confidence_score,
            "selector": self._generate_container_selector(container),
            "item_selector": item_selector,
            "structure_similarity": structure_similarity,
            "content_similarity": content_similarity
        }
        
        return listing_data
    
    def _check_structural_similarity(self, items: List[Tag]) -> float:
        """
        Check how similar the structure is across list items.
        
        Args:
            items: List of BeautifulSoup tag elements
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        if not items or len(items) < 2:
            return 0.0
            
        # Get a structural fingerprint for each item
        fingerprints = []
        
        for item in items:
            # Create a simple structural representation
            fingerprint = self._get_structure_fingerprint(item)
            fingerprints.append(fingerprint)
        
        # Count how many items have similar fingerprints
        fingerprint_counter = Counter(fingerprints)
        most_common = fingerprint_counter.most_common(1)
        
        if most_common:
            # Calculate what percentage of items share the most common structure
            return most_common[0][1] / len(items)
        
        return 0.0
    
    def _get_structure_fingerprint(self, element: Tag) -> str:
        """
        Generate a simple structural fingerprint for an element.
        
        Args:
            element: BeautifulSoup tag element
            
        Returns:
            String representing the element's structure
        """
        fingerprint = []
        
        # Add element's tag
        fingerprint.append(element.name)
        
        # Add immediate children's tags
        children = [child for child in element.children if isinstance(child, Tag)]
        child_tags = [child.name for child in children]
        fingerprint.append('-'.join(child_tags))
        
        # Add key descendant elements that are common in list items
        has_img = 1 if element.find('img') else 0
        has_a = 1 if element.find('a') else 0
        has_heading = 1 if element.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']) else 0
        has_price = 1 if element.find(string=re.compile(r'(\$|€|£|¥)\s*[\d,.]+')) else 0
        
        fingerprint.append(f"{has_img}-{has_a}-{has_heading}-{has_price}")
        
        return '|'.join(fingerprint)
    
    def _check_content_similarity(self, items: List[Tag]) -> float:
        """
        Check how similar the content is across list items.
        
        Args:
            items: List of BeautifulSoup tag elements
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        if not items or len(items) < 2:
            return 0.0
            
        # Check for similar content patterns
        similarities = []
        
        # Check for similar images
        has_images = [bool(item.find('img')) for item in items]
        if has_images:
            image_similarity = sum(has_images) / len(has_images)
            similarities.append(image_similarity)
        
        # Check for similar links
        has_links = [bool(item.find('a')) for item in items]
        if has_links:
            link_similarity = sum(has_links) / len(has_links)
            similarities.append(link_similarity)
        
        # Check for similar text length patterns
        text_lengths = [len(item.get_text().strip()) for item in items]
        if text_lengths:
            avg_length = sum(text_lengths) / len(text_lengths)
            # Calculate coefficient of variation (lower means more similar)
            std_dev = (sum((length - avg_length) ** 2 for length in text_lengths) / len(text_lengths)) ** 0.5
            length_similarity = 1.0 - min(1.0, std_dev / avg_length if avg_length > 0 else 0)
            similarities.append(length_similarity)
        
        # Return average similarity
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _analyze_item_samples(self, items: List[Tag]) -> List[Dict[str, Any]]:
        """
        Analyze a sample of list items to determine their structure.
        
        Args:
            items: List of BeautifulSoup tag elements
            
        Returns:
            List of item analysis data
        """
        item_samples = []
        
        for item_idx, item in enumerate(items):
            # Extract key elements from the item
            images = item.find_all('img')
            links = item.find_all('a')
            headings = item.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            paragraphs = item.find_all('p')
            
            # Try to detect price
            price_text = None
            price_pattern = re.compile(r'(\$|€|£|¥)\s*[\d,.]+')
            price_elem = item.find(string=price_pattern)
            if price_elem:
                price_match = price_pattern.search(price_elem)
                if price_match:
                    price_text = price_match.group(0)
            
            # Extract item content
            item_data = {
                "item_idx": item_idx,
                "element": item.name,
                "selector": self._generate_field_selector(item),
                "has_image": bool(images),
                "has_link": bool(links),
                "has_heading": bool(headings),
                "has_price": bool(price_text),
                "components": {
                    "images": [{"src": img.get('src', ''), "alt": img.get('alt', '')} for img in images[:2]],
                    "links": [{"href": a.get('href', ''), "text": a.get_text().strip()} for a in links[:2]],
                    "headings": [h.get_text().strip() for h in headings[:2]],
                    "paragraphs": [p.get_text().strip() for p in paragraphs[:2]],
                    "price": price_text
                }
            }
            
            item_samples.append(item_data)
        
        return item_samples
    
    def _determine_listing_type(self, container: Tag, items: List[Tag], 
                              item_samples: List[Dict[str, Any]]) -> str:
        """
        Determine the type of listing based on container and item characteristics.
        
        Args:
            container: BeautifulSoup container element
            items: List of list item elements
            item_samples: Analyzed item samples
            
        Returns:
            String indicating the listing type
        """
        # Default type
        listing_type = "generic_listing"
        
        # Check container attributes for clues
        container_text = container.get('class', [])
        container_text = ' '.join(container_text) if isinstance(container_text, list) else container_text
        container_text += ' ' + container.get('id', '')
        
        # Check for product indicators
        if (re.search(r'product|item|shop|store', container_text, re.I) or
            any(sample.get('has_price', False) for sample in item_samples)):
            listing_type = "product_listing"
            
        # Check for search result indicators
        elif re.search(r'result|search', container_text, re.I):
            listing_type = "search_results"
            
        # Check for article/post indicators
        elif re.search(r'article|post|blog|news', container_text, re.I):
            listing_type = "article_listing"
            
        # Try to infer from content
        else:
            # Count item characteristics
            has_price_count = sum(1 for sample in item_samples if sample.get('has_price', False))
            has_image_count = sum(1 for sample in item_samples if sample.get('has_image', False))
            has_heading_count = sum(1 for sample in item_samples if sample.get('has_heading', False))
            
            # Product listings often have prices
            if has_price_count > len(item_samples) * 0.5:
                listing_type = "product_listing"
                
            # Article listings often have headings and maybe images
            elif (has_heading_count > len(item_samples) * 0.7 and 
                 has_image_count > len(item_samples) * 0.3):
                listing_type = "article_listing"
                
            # Listings with just images might be gallery/media
            elif has_image_count > len(item_samples) * 0.8 and has_heading_count < len(item_samples) * 0.3:
                listing_type = "media_gallery"
        
        return listing_type
    
    def _generate_container_selector(self, container: Tag) -> str:
        """
        Generate a CSS selector for a container.
        
        Args:
            container: BeautifulSoup container element
            
        Returns:
            CSS selector string
        """
        # If the container has an ID, that's the most reliable selector
        if container.get('id'):
            return f"#{container['id']}"
            
        # If the container has classes, use those
        if container.get('class'):
            class_selector = '.'.join(container['class'])
            return f"{container.name}.{class_selector}"
            
        # If the container has a role attribute
        if container.get('role'):
            return f"{container.name}[role='{container['role']}']"
            
        # Fallback: try to generate a selector based on container position
        parent = container.parent
        if parent and parent.name != 'body':
            if parent.get('id'):
                return f"#{parent['id']} > {container.name}"
            elif parent.get('class'):
                class_selector = '.'.join(parent['class'])
                return f".{class_selector} > {container.name}"
        
        # Last resort: use nth-of-type
        siblings = list(container.find_previous_siblings(container.name)) + [container]
        return f"{container.name}:nth-of-type({len(siblings)})"
    
    def _generate_item_selector(self, container: Tag, item: Tag) -> str:
        """
        Generate a CSS selector for list items relative to their container.
        
        Args:
            container: BeautifulSoup container element
            item: BeautifulSoup item element
            
        Returns:
            CSS selector string
        """
        container_selector = self._generate_container_selector(container)
        
        # If the item has an ID, use that
        if item.get('id'):
            return f"#{item['id']}"
            
        # If the item has classes, use those
        if item.get('class'):
            class_selector = '.'.join(item['class'])
            return f"{container_selector} {item.name}.{class_selector}"
            
        # If all items are direct children
        if item.parent == container:
            return f"{container_selector} > {item.name}"
            
        # Generic child selector
        return f"{container_selector} {item.name}"
    
    def _generate_field_selector(self, field: Tag) -> str:
        """
        Generate a CSS selector for a field.
        
        Args:
            field: BeautifulSoup field element
            
        Returns:
            CSS selector string
        """
        # If the field has an ID, that's the most reliable selector
        if field.get('id'):
            return f"#{field['id']}"
            
        # If the field has classes, use those
        if field.get('class'):
            class_selector = '.'.join(field['class'])
            return f"{field.name}.{class_selector}"
            
        # Fallback: try to generate a selector based on field position
        parent = field.parent
        if parent and parent.name != 'body':
            if parent.get('id'):
                return f"#{parent['id']} > {field.name}"
            elif parent.get('class'):
                class_selector = '.'.join(parent['class'])
                return f".{class_selector} > {field.name}"
        
        # Last resort: use nth-of-type
        siblings = list(field.find_previous_siblings(field.name)) + [field]
        return f"{field.name}:nth-of-type({len(siblings)})"

    # Phase 6 implementation - Result Identification methods
    
    def identify_result_containers(self, html: str) -> List[Dict[str, Any]]:
        """
        Find HTML elements likely to be result containers based on common patterns.
        
        Args:
            html: HTML content to analyze
            
        Returns:
            List of dictionaries containing container information and selectors
        """
        soup = self.parse_html(html)
        if not soup:
            logger.warning("Failed to parse HTML")
            return []
            
        containers = []
        
        # Check for common result container patterns
        result_container_patterns = [
            # Common search result containers
            ".search-results", "#search-results", ".results", "#results", "[data-component='search-results']",
            # E-commerce result containers
            ".products", ".product-grid", ".product-list", ".items", ".item-grid", 
            # Content/article result containers
            ".articles", ".posts", ".blog-posts", ".news-items",
            # Generic grid/listing containers
            ".listing", ".grid", ".cards", ".tiles", ".results-list", "[role='list']", "[role='grid']"
        ]
        
        for pattern in result_container_patterns:
            for element in soup.select(pattern):
                # Skip small containers or those with less than 2 children
                if len(element.find_all()) < 4:
                    continue
                    
                # Analyze container
                confidence = self._calculate_result_container_confidence(element)
                if confidence > 0.6:  # Threshold for confidence
                    container_info = {
                        "element": element.name,
                        "selector": self._generate_container_selector(element),
                        "confidence": confidence,
                        "child_count": len([c for c in element.children if isinstance(c, Tag)]),
                        "likely_item_tags": self._identify_likely_item_tags(element),
                        "content_type": self._infer_container_content_type(element)
                    }
                    containers.append(container_info)
                    
        # Also look for generic repeating structures
        repeating_structures = self._find_repeating_element_structures(soup)
        for structure in repeating_structures:
            if structure["confidence"] > 0.6 and not any(c["selector"] == structure["selector"] for c in containers):
                containers.append(structure)
                
        # Sort by confidence
        containers.sort(key=lambda x: x["confidence"], reverse=True)
        
        return containers
    
    def _calculate_result_container_confidence(self, element: Tag) -> float:
        """
        Calculate confidence that an element is a result container.
        
        Args:
            element: BeautifulSoup element
            
        Returns:
            Confidence score between 0 and 1
        """
        evidence_points = []
        
        # Check element attributes
        element_classes = ' '.join(element.get('class', []))
        element_id = element.get('id', '')
        combined_text = f"{element_classes} {element_id}"
        
        # Check for result-related terms in class/id
        if re.search(r'result|search|product|list|grid|item', combined_text, re.I):
            evidence_points.append(0.8)
        
        # Check if element has ARIA role
        if element.get('role') in ['list', 'grid', 'feed', 'listbox']:
            evidence_points.append(0.9)
            
        # Check if element is a list (ul/ol)
        if element.name in ['ul', 'ol']:
            evidence_points.append(0.7)
            
        # Check for repeating child structure
        child_tags = [child.name for child in element.children if isinstance(child, Tag)]
        tag_counter = Counter(child_tags)
        
        most_common = tag_counter.most_common(1)
        if most_common and most_common[0][1] >= 3:  # At least 3 of the same tag
            repeat_ratio = most_common[0][1] / len(child_tags) if child_tags else 0
            evidence_points.append(min(0.9, repeat_ratio))
            
        # Check for similar child classes
        child_elements = [child for child in element.children if isinstance(child, Tag)]
        if len(child_elements) >= 3:
            class_similarity = self._check_class_similarities(child_elements)
            evidence_points.append(class_similarity)
            
        # Check for key result features (images, links, headings) in children
        features_present = 0
        for child in child_elements[:5]:  # Check first 5 children
            if child.find('img'):
                features_present += 1
            if child.find('a'):
                features_present += 1
            if child.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                features_present += 1
                
        feature_score = min(0.8, features_present / 5)
        evidence_points.append(feature_score)
        
        # Calculate overall confidence
        return self.calculate_confidence(evidence_points)
    
    def _identify_likely_item_tags(self, container: Tag) -> List[str]:
        """
        Identify tags that are likely to represent result items within a container.
        
        Args:
            container: Container element
            
        Returns:
            List of likely item tag names with counts
        """
        # Get direct children
        child_tags = [child.name for child in container.children if isinstance(child, Tag)]
        tag_counter = Counter(child_tags)
        
        # Find tags that appear multiple times
        recurring_tags = []
        for tag, count in tag_counter.most_common():
            if count >= 2:  # At least 2 occurrences
                recurring_tags.append({"tag": tag, "count": count})
            
        return recurring_tags
    
    def _infer_container_content_type(self, container: Tag) -> str:
        """
        Infer the content type of a container based on its contents.
        
        Args:
            container: Container element
            
        Returns:
            String indicating the content type
        """
        # Default type
        content_type = "generic"
        
        # Check container attributes
        container_classes = ' '.join(container.get('class', []))
        container_id = container.get('id', '')
        combined_text = f"{container_classes} {container_id}"
        
        # Check for product indicators
        if re.search(r'product|item|shop|store', combined_text, re.I):
            # Confirm by checking for prices
            price_pattern = re.compile(r'(\$|€|£|¥)\s*[\d,.]+')
            if container.find(string=price_pattern):
                return "products"
                
        # Check for search result indicators
        if re.search(r'result|search', combined_text, re.I):
            return "search_results"
            
        # Check for article indicators
        if re.search(r'article|post|blog|news', combined_text, re.I):
            return "articles"
            
        # Analyze content to determine type
        children = [child for child in container.children if isinstance(child, Tag)]
        
        # Check first few children for characteristic elements
        price_count = 0
        image_count = 0
        heading_count = 0
        
        for child in children[:5]:
            if child.find(string=re.compile(r'(\$|€|£|¥)\s*[\d,.]+')) or child.find(text=re.compile(r'price|cost', re.I)):
                price_count += 1
            if child.find('img'):
                image_count += 1
            if child.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                heading_count += 1
                
        # Make determination based on content
        if price_count >= 2:
            content_type = "products"
        elif heading_count >= 3:
            content_type = "articles"
        elif image_count >= 4:
            content_type = "gallery"
            
        return content_type
    
    def _find_repeating_element_structures(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Find repeating element structures that may represent result listings.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            List of repeating structure information
        """
        structures = []
        
        # Common container tags
        container_tags = ['div', 'section', 'main', 'ul', 'ol', 'table']
        for tag in container_tags:
            containers = soup.find_all(tag)
            
            for container in containers:
                # Skip small containers
                if len(container.find_all()) < 10:
                    continue
                    
                # Count children by tag
                direct_children = [child for child in container.children if isinstance(child, Tag)]
                child_tag_counter = Counter([child.name for child in direct_children])
                
                # Look for repeated child tags
                for child_tag, count in child_tag_counter.items():
                    if count >= 3:  # At least 3 occurrences
                        items = [child for child in direct_children if child.name == child_tag]
                        
                        # Check for structural similarity among children
                        structure_similarity = self._check_structural_similarity(items)
                        
                        if structure_similarity > 0.6:
                            container_info = {
                                "element": container.name,
                                "selector": self._generate_container_selector(container),
                                "confidence": structure_similarity * 0.9,  # Scale by 0.9 as this is heuristic-based
                                "child_count": count,
                                "item_tag": child_tag,
                                "likely_item_tags": [{"tag": child_tag, "count": count}],
                                "content_type": self._infer_container_content_type(container)
                            }
                            structures.append(container_info)
        
        return structures
        
    def analyze_result_structure(self, container_html: str) -> Dict[str, Any]:
        """
        Analyze the structure of results to understand common patterns.
        
        Args:
            container_html: HTML of the result container
            
        Returns:
            Dictionary with structural analysis results
        """
        soup = BeautifulSoup(container_html, 'lxml')
        
        # Get all child elements
        children = [child for child in soup.children if isinstance(child, Tag)]
        
        # If there are no meaningful children, return empty analysis
        if not children:
            return {"success": False, "message": "No valid children found in container"}
            
        # Count element types and their occurrences
        element_types = Counter([child.name for child in children])
        most_common_element = element_types.most_common(1)[0][0] if element_types else None
        
        # Identify elements that are likely result items
        potential_items = [child for child in children if child.name == most_common_element]
        
        if len(potential_items) < 2:
            # Try to find another way to identify items
            for child in children:
                if len(list(child.children)) > 1:  # Check if child has its own children
                    potential_items = children
                    break
        
        # Analyze the structure of result items
        structure_analysis = self._analyze_result_items_structure(potential_items)
        
        # Extract common fields across items
        common_fields = self._identify_common_fields(potential_items)
        
        # Create field mapping suggestions for extraction
        field_mappings = self._generate_field_mappings(potential_items, common_fields)
        
        result = {
            "success": True,
            "item_count": len(potential_items),
            "common_element_type": most_common_element,
            "element_distribution": {elem: count for elem, count in element_types.items()},
            "structure_consistency": structure_analysis["consistency"],
            "common_fields": common_fields,
            "field_mappings": field_mappings,
            "example_item_structure": structure_analysis["example_structure"]
        }
        
        return result
    
    def _analyze_result_items_structure(self, items: List[Tag]) -> Dict[str, Any]:
        """
        Analyze the structure of result items to determine consistency.
        
        Args:
            items: List of possible result item elements
            
        Returns:
            Dictionary with analysis results
        """
        if not items:
            return {"consistency": 0, "example_structure": None}
            
        # Generate structural signature for each item
        signatures = []
        for item in items:
            sig = self._generate_structural_signature(item)
            signatures.append(sig)
            
        # Calculate consistency by comparing signatures
        matching_signatures = Counter(signatures)
        most_common_sig = matching_signatures.most_common(1)[0] if matching_signatures else (None, 0)
        
        if most_common_sig[0] is None:
            return {"consistency": 0, "example_structure": None}
            
        consistency = most_common_sig[1] / len(items) if items else 0
        
        # Find an item with the most common structure
        example_item = next((item for item, sig in zip(items, signatures) if sig == most_common_sig[0]), None)
        example_structure = self._document_element_structure(example_item) if example_item else None
        
        return {
            "consistency": consistency,
            "example_structure": example_structure
        }
    
    def _generate_structural_signature(self, element: Tag) -> str:
        """
        Generate a structural signature for an element based on its content.
        
        Args:
            element: BeautifulSoup element
            
        Returns:
            String representing the element's structure
        """
        if not element:
            return ""
            
        # Create a structure ID based on descendant elements
        structure_parts = []
        
        # Add element tag
        structure_parts.append(element.name)
        
        # Add direct children's tags in order
        child_tags = [child.name for child in element.children if isinstance(child, Tag)]
        structure_parts.append("-".join(child_tags) if child_tags else "")
        
        # Add key features for common elements
        features = []
        if element.find('img'):
            features.append("img")
        if element.find('a'):
            features.append("link")
        if element.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            features.append("heading")
        if element.find('p'):
            features.append("paragraph")
        if element.find(string=re.compile(r'(\$|€|£|¥)\s*[\d,.]+|price|cost', re.I)):
            features.append("price")
            
        structure_parts.append("-".join(sorted(features)))
        
        return "|".join(structure_parts)
    
    def _document_element_structure(self, element: Tag) -> Dict[str, Any]:
        """
        Create a simplified document of an element's structure.
        
        Args:
            element: BeautifulSoup element
            
        Returns:
            Dictionary documenting the element structure
        """
        if not element:
            return {}
            
        structure = {
            "tag": element.name,
            "attributes": {k: v for k, v in element.attrs.items()},
            "direct_children": []
        }
        
        # Add direct children
        for child in element.children:
            if isinstance(child, Tag):
                child_info = {
                    "tag": child.name,
                    "classes": child.get('class', []),
                    "contains_image": bool(child.find('img')),
                    "contains_link": bool(child.find('a')),
                    "contains_heading": bool(child.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])),
                    "has_text": bool(child.get_text(strip=True))
                }
                structure["direct_children"].append(child_info)
        
        return structure
    
    def _identify_common_fields(self, items: List[Tag]) -> List[Dict[str, Any]]:
        """
        Identify common fields across result items.
        
        Args:
            items: List of result item elements
            
        Returns:
            List of common fields with their properties
        """
        if not items:
            return []
            
        # Common field patterns to check
        field_patterns = {
            "title": [
                {"tags": ["h1", "h2", "h3", "h4", "h5", "h6"], "classes": ["title", "name", "heading"]},
                {"tags": ["a"], "classes": ["title", "name", "product-title", "item-title"]},
                {"tags": ["div", "span"], "classes": ["title", "name", "product-title"]}
            ],
            "price": [
                {"tags": ["span", "div"], "classes": ["price", "cost", "amount"]},
                {"tags": ["p", "div", "span"], "text_pattern": r'(\$|€|£|¥)\s*[\d,.]+'}
            ],
            "image": [
                {"tags": ["img"], "classes": []},
                {"tags": ["div"], "classes": ["image", "thumbnail", "product-image", "img"]}
            ],
            "description": [
                {"tags": ["p"], "classes": ["description", "snippet", "summary"]},
                {"tags": ["div"], "classes": ["description", "text", "content"]}
            ],
            "link": [
                {"tags": ["a"], "classes": []}
            ]
        }
        
        # Track which fields were found in each item
        field_occurences = {field: [] for field in field_patterns.keys()}
        
        # Check each item for each field type
        for item_idx, item in enumerate(items):
            for field_name, patterns in field_patterns.items():
                for pattern in patterns:
                    # Check for elements matching the pattern
                    elements = []
                    
                    # Find by tag and class if specified
                    if pattern["tags"] and pattern.get("classes"):
                        for tag in pattern["tags"]:
                            for class_name in pattern["classes"]:
                                elements.extend(item.find_all(tag, class_=re.compile(class_name, re.I)))
                    # Find just by tag
                    elif pattern["tags"] and not pattern.get("classes"):
                        for tag in pattern["tags"]:
                            elements.extend(item.find_all(tag))
                    
                    # Check for text pattern if specified
                    if pattern.get("text_pattern") and not elements:
                        text_pattern = re.compile(pattern["text_pattern"], re.I)
                        elements = item.find_all(string=text_pattern)
                        
                    # If field was found, record its occurrence and selector
                    if elements:
                        element = elements[0]  # Use the first match
                        selector = self._generate_field_selector(element) if isinstance(element, Tag) else ""
                        
                        field_occurences[field_name].append({
                            "item_idx": item_idx,
                            "element": element.name if isinstance(element, Tag) else "text",
                            "selector": selector,
                            "text": element.get_text(strip=True) if isinstance(element, Tag) else str(element).strip()
                        })
                        break  # Found a match for this field in this item, move to next field
        
        # Determine which fields are common across items
        common_fields = []
        for field_name, occurrences in field_occurences.items():
            if occurrences:
                presence_ratio = len(occurrences) / len(items)
                
                if presence_ratio >= 0.5:  # Field appears in at least 50% of items
                    common_fields.append({
                        "name": field_name,
                        "presence_ratio": presence_ratio,
                        "common_tag": Counter([o.get("element") for o in occurrences]).most_common(1)[0][0],
                        "sample_text": occurrences[0].get("text", "") if occurrences else ""
                    })
        
        # Sort by frequency
        common_fields.sort(key=lambda x: x["presence_ratio"], reverse=True)
        
        return common_fields
    
    def _generate_field_mappings(self, items: List[Tag], common_fields: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Generate CSS selector mappings for common fields.
        
        Args:
            items: List of result item elements
            common_fields: List of identified common fields
            
        Returns:
            Dictionary mapping field names to lists of potential CSS selectors
        """
        if not items or not common_fields:
            return {}
            
        mappings = {}
        
        # Get a sample item
        sample_item = items[0]
        
        for field in common_fields:
            field_name = field["name"]
            selectors = []
            
            if field_name == "title":
                # Try heading selectors
                headings = sample_item.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                if headings:
                    selectors.append(self._generate_field_selector(headings[0]))
                
                # Try elements with title-like classes
                title_elements = sample_item.find_all(class_=re.compile(r'title|name|heading', re.I))
                if title_elements:
                    selectors.append(self._generate_field_selector(title_elements[0]))
                    
                # Try the first link
                links = sample_item.find_all('a')
                if links:
                    selectors.append(self._generate_field_selector(links[0]))
                    
            elif field_name == "price":
                # Try price class elements
                price_elements = sample_item.find_all(class_=re.compile(r'price|cost|amount', re.I))
                if price_elements:
                    selectors.append(self._generate_field_selector(price_elements[0]))
                
                # Try finding elements with price patterns
                price_pattern = re.compile(r'(\$|€|£|¥)\s*[\d,.]+')
                price_texts = sample_item.find_all(string=price_pattern)
                if price_texts:
                    parent = price_texts[0].parent
                    if isinstance(parent, Tag):
                        selectors.append(self._generate_field_selector(parent))
                        
            elif field_name == "image":
                # Direct img tags
                images = sample_item.find_all('img')
                if images:
                    selectors.append(self._generate_field_selector(images[0]))
                
                # Image containers
                image_containers = sample_item.find_all(class_=re.compile(r'image|thumbnail|photo', re.I))
                if image_containers:
                    container_imgs = image_containers[0].find_all('img')
                    if container_imgs:
                        selectors.append(f"{self._generate_field_selector(image_containers[0])} img")
                    else:
                        selectors.append(f"{self._generate_field_selector(image_containers[0])}")
                        
            elif field_name == "description":
                # Find paragraphs
                paragraphs = sample_item.find_all('p')
                if paragraphs:
                    selectors.append(self._generate_field_selector(paragraphs[0]))
                
                # Try description classes
                desc_elements = sample_item.find_all(class_=re.compile(r'desc|summary|text|content', re.I))
                if desc_elements:
                    selectors.append(self._generate_field_selector(desc_elements[0]))
                    
            elif field_name == "link":
                # Find all links
                links = sample_item.find_all('a')
                if links:
                    selectors.append(self._generate_field_selector(links[0]))
            
            # Add the field mapping if we found selectors
            if selectors:
                mappings[field_name] = selectors
        
        return mappings
    
    def generate_item_selectors(self, html: str, container_selector: str) -> Dict[str, Any]:
        """
        Create reliable CSS selectors for result items.
        
        Args:
            html: HTML content of the page
            container_selector: CSS selector for the container
            
        Returns:
            Dictionary with generated selectors
        """
        soup = self.parse_html(html)
        if not soup:
            return {"success": False, "message": "Failed to parse HTML"}
            
        # Find the container
        container = soup.select_one(container_selector)
        if not container:
            return {"success": False, "message": f"Container not found with selector: {container_selector}"}
            
        # Identify potential item tags
        child_tags = [child.name for child in container.children if isinstance(child, Tag)]
        tag_counter = Counter(child_tags)
        
        item_selectors = []
        
        # Try to identify item selectors based on repeating elements
        if tag_counter:
            # Get the most common tag
            most_common_tag = tag_counter.most_common(1)[0][0]
            elements = [child for child in container.children if isinstance(child, Tag) and child.name == most_common_tag]
            
            if len(elements) >= 2:
                # Check for common classes
                class_counter = Counter()
                
                for element in elements:
                    if element.get('class'):
                        for class_name in element.get('class'):
                            class_counter[class_name] += 1
                
                # Find classes that appear in multiple elements
                common_classes = [cls for cls, count in class_counter.items() if count >= len(elements) * 0.5]
                
                if common_classes:
                    class_selector = '.'.join(common_classes)
                    item_selectors.append(f"{container_selector} {most_common_tag}.{class_selector}")
                
                # Add simple selector as fallback
                item_selectors.append(f"{container_selector} > {most_common_tag}")
                
                # Try direct selector with nth-child for the first element as example
                first_element = elements[0]
                item_selectors.append(self._generate_item_selector(container, first_element))
                
        # Try looking for semantic item elements
        semantic_selectors = [
            'li', '.item', '.product', '.result', '.card', '.article', '.post', 
            '[role="listitem"]', '[itemprop="item"]'
        ]
        
        for selector in semantic_selectors:
            elements = container.select(selector)
            if len(elements) >= 2:
                item_selectors.append(f"{container_selector} {selector}")
        
        # Try to identify field selectors for common fields
        sample_items = None
        if item_selectors:
            # Use the first selector to find items
            try:
                sample_items = soup.select(item_selectors[0])[:3]  # Get up to 3 items
            except Exception:
                pass
        
        field_selectors = {}
        
        if sample_items:
            # Find common fields
            common_fields = self._identify_common_fields(sample_items)
            
            # Generate selectors for each field
            field_selectors = self._generate_field_mappings(sample_items, common_fields)
            
            # Convert to relative selectors
            for field, selectors in field_selectors.items():
                relative_selectors = []
                for selector in selectors:
                    # Make selectors relative to item
                    if selector.startswith(container_selector):
                        # Try to extract just the part after the container selector
                        relative_part = selector[len(container_selector):].lstrip()
                        if relative_part:
                            relative_selectors.append(relative_part)
                    else:
                        relative_selectors.append(selector)
                        
                field_selectors[field] = relative_selectors
        
        # Create extraction schema
        extraction_schema = {
            "items": item_selectors
        }
        
        # Add field selectors to schema
        for field, selectors in field_selectors.items():
            extraction_schema[field] = selectors
        
        return {
            "success": True,
            "item_selectors": item_selectors,
            "field_selectors": field_selectors,
            "extraction_schema": extraction_schema,
            "item_count": len(container.select(item_selectors[0])) if item_selectors else 0
        }