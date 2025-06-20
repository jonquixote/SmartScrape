"""
Detail Analyzer Module

This module provides functionality to detect and analyze content detail pages
such as product details, article pages, profile pages, and other single-item content.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from bs4 import BeautifulSoup, Tag

from components.pattern_analyzer.base_analyzer import PatternAnalyzer, get_registry

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DetailAnalyzer")


class DetailAnalyzer(PatternAnalyzer):
    """
    Analyzer for detecting and analyzing detail page patterns.
    This includes product details, article content, profile pages, and other
    single-item focused content.
    """
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize the detail analyzer.
        
        Args:
            confidence_threshold: Minimum confidence level for pattern detection
        """
        super().__init__(confidence_threshold)
        
        # Common patterns that indicate different types of detail pages
        self.detail_patterns = {
            'product': {
                'url': [re.compile(r'product|item|detail|view', re.I)],
                'id': [re.compile(r'product|item|detail|single', re.I)],
                'class': [re.compile(r'product|item|detail|single', re.I)],
                'meta': [re.compile(r'product', re.I)]
            },
            'article': {
                'url': [re.compile(r'article|post|story|news|blog', re.I)],
                'id': [re.compile(r'article|post|story|content|main', re.I)],
                'class': [re.compile(r'article|post|story|content|main', re.I)],
                'meta': [re.compile(r'article', re.I)]
            },
            'profile': {
                'url': [re.compile(r'profile|user|member|account', re.I)],
                'id': [re.compile(r'profile|user|member|account', re.I)],
                'class': [re.compile(r'profile|user|member|account', re.I)],
                'meta': [re.compile(r'profile', re.I)]
            },
            'property': {
                'url': [re.compile(r'property|house|home|apartment|listing', re.I)],
                'id': [re.compile(r'property|house|home|apartment|listing', re.I)],
                'class': [re.compile(r'property|house|home|apartment|listing', re.I)],
                'meta': [re.compile(r'property|real estate', re.I)]
            }
        }
        
        # Common section patterns for different detail page components
        self.section_patterns = {
            'title': [
                'h1', '.title', '.product-title', '.article-title', 
                '[itemprop="name"]', '[class*="title"]', '[class*="heading"]'
            ],
            'description': [
                '.description', '.product-description', '.content', '[itemprop="description"]',
                '[class*="description"]', '[class*="content"]', 'article'
            ],
            'price': [
                '.price', '[itemprop="price"]', '[class*="price"]', '[data-price]',
                '.product-price', '.item-price'
            ],
            'images': [
                '.product-image', '.gallery', '.carousel', '[itemprop="image"]',
                '[class*="image"]', '[class*="gallery"]', '[class*="carousel"]'
            ],
            'attributes': [
                '.attributes', '.specs', '.details', '.features', '.properties',
                '[class*="attribute"]', '[class*="spec"]', '[class*="detail"]',
                '[class*="feature"]', '[class*="property"]'
            ],
            'metadata': [
                '.meta', '.metadata', '.info', '[class*="meta"]', '[class*="info"]',
                'time', '[datetime]', '.author', '.date', '.published', '.byline'
            ]
        }
    
    async def analyze(self, html: str, url: str) -> Dict[str, Any]:
        """
        Analyze a page to detect detail page patterns.
        
        Args:
            html: HTML content of the page
            url: URL of the page
            
        Returns:
            Dictionary with detected detail patterns
        """
        logger.info(f"Analyzing detail page patterns on {url}")
        soup = self.parse_html(html)
        domain = self.get_domain(url)
        
        # Results will contain all detected patterns
        results = {
            "confidence_score": 0.0,
            "is_detail_page": False,
            "page_type": "unknown",
            "content_sections": {},
            "sections": {},
            "metadata": {},
            "patterns": []
        }
        
        # Check URL for detail page indicators
        url_indicators = self._check_url_for_indicators(url)
        if url_indicators["is_detail"]:
            results["is_detail_page"] = True
            results["page_type"] = url_indicators["type"]
            evidence_points = [url_indicators["confidence"]]
        else:
            evidence_points = [0.1]  # Low starting confidence
        
        # Check title and meta for detail page indicators
        meta_indicators = self._check_meta_for_indicators(soup)
        if meta_indicators["is_detail"]:
            results["is_detail_page"] = True
            if not results["page_type"] or results["page_type"] == "unknown":
                results["page_type"] = meta_indicators["type"]
            evidence_points.append(meta_indicators["confidence"])
        
        # Check main content area
        main_content = self._identify_main_content(soup)
        if main_content:
            content_analysis = self._analyze_content(main_content, results["page_type"])
            results["content_sections"] = content_analysis["sections"]
            results["is_detail_page"] = content_analysis["is_detail"]
            
            if content_analysis["is_detail"] and (not results["page_type"] or results["page_type"] == "unknown"):
                results["page_type"] = content_analysis["type"]
                
            evidence_points.append(content_analysis["confidence"])
        
        # Find important content sections
        section_results = self._find_content_sections(soup, results["page_type"])
        results["sections"] = section_results["sections"]
        evidence_points.append(section_results["confidence"])
        
        # Extract metadata
        results["metadata"] = self._extract_metadata(soup, results["page_type"])
        
        # Calculate overall confidence
        results["confidence_score"] = self.calculate_confidence(evidence_points)
        
        # Check if we're confident this is a detail page
        if results["confidence_score"] >= self.confidence_threshold:
            results["is_detail_page"] = True
            
            # Register the detail page pattern in the global registry
            get_registry().register_pattern(
                pattern_type="detail_page",
                url=url,
                pattern_data={
                    "page_type": results["page_type"],
                    "sections": results["sections"],
                    "selectors": self._generate_selectors(results["sections"])
                },
                confidence=results["confidence_score"]
            )
            logger.info(f"Registered detail page pattern for {domain}")
        else:
            results["is_detail_page"] = False
        
        return results
    
    def _check_url_for_indicators(self, url: str) -> Dict[str, Any]:
        """
        Check if URL contains indicators of a detail page.
        
        Args:
            url: URL to check
            
        Returns:
            Dictionary with detection results
        """
        result = {
            "is_detail": False,
            "type": "unknown",
            "confidence": 0.0
        }
        
        url_lower = url.lower()
        
        # Check each detail page type
        for page_type, patterns in self.detail_patterns.items():
            for pattern in patterns['url']:
                if pattern.search(url_lower):
                    result["is_detail"] = True
                    result["type"] = page_type
                    result["confidence"] = 0.7  # Good confidence from URL
                    return result
        
        # Common detail page patterns in URLs
        detail_indicators = [
            r'/view/', r'/details?/', r'/show/', r'/single/', r'\.html$', 
            r'/p/', r'/[0-9]+$', r'id=', r'product[-_]?id='
        ]
        
        for pattern in detail_indicators:
            if re.search(pattern, url_lower):
                result["is_detail"] = True
                result["type"] = "unknown"
                result["confidence"] = 0.6  # Moderate confidence
                return result
        
        return result
    
    def _check_meta_for_indicators(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Check page title and meta tags for detail page indicators.
        
        Args:
            soup: BeautifulSoup object representing the HTML
            
        Returns:
            Dictionary with detection results
        """
        result = {
            "is_detail": False,
            "type": "unknown",
            "confidence": 0.0
        }
        
        # Check page title
        title = soup.title.string.lower() if soup.title else ""
        
        # Check each detail page type for title indicators
        for page_type, patterns in self.detail_patterns.items():
            for pattern in patterns.get('meta', []):
                if pattern.search(title):
                    result["is_detail"] = True
                    result["type"] = page_type
                    result["confidence"] = 0.6  # Moderate confidence from title
                    break
        
        # Check meta tags, especially og:type
        og_type = soup.find('meta', property='og:type')
        if og_type:
            content = og_type.get('content', '').lower()
            
            if content == 'product':
                result["is_detail"] = True
                result["type"] = 'product'
                result["confidence"] = 0.9  # High confidence from explicit meta
            elif content == 'article':
                result["is_detail"] = True
                result["type"] = 'article'
                result["confidence"] = 0.9
            elif content == 'profile':
                result["is_detail"] = True
                result["type"] = 'profile'
                result["confidence"] = 0.9
        
        # Check structured data
        if soup.find(attrs={"itemtype": re.compile(r'Product|Article|Person|RealEstate', re.I)}):
            result["is_detail"] = True
            result["confidence"] = 0.9  # High confidence from structured data
            
            # Determine type from itemtype
            itemtype = soup.find(attrs={"itemtype": True})
            if itemtype:
                itemtype_str = itemtype['itemtype'].lower()
                if 'product' in itemtype_str:
                    result["type"] = 'product'
                elif 'article' in itemtype_str:
                    result["type"] = 'article'
                elif 'person' in itemtype_str:
                    result["type"] = 'profile'
                elif 'realestate' in itemtype_str:
                    result["type"] = 'property'
        
        return result
    
    def _identify_main_content(self, soup: BeautifulSoup) -> Optional[Tag]:
        """
        Identify the main content area of the page.
        
        Args:
            soup: BeautifulSoup object representing the HTML
            
        Returns:
            Tag object for the main content area, or None if not found
        """
        # Common main content selectors in order of preference
        main_content_selectors = [
            'main',
            '#main',
            '#content',
            '.content',
            'article',
            '.main',
            '.main-content',
            '.product-detail',
            '.article-content',
            '.product',
            '.detail',
            '[role="main"]',
            '[itemtype*="Product"]',
            '[itemtype*="Article"]'
        ]
        
        for selector in main_content_selectors:
            content = soup.select_one(selector)
            if content:
                return content
        
        # If no main content found with selectors, try a heuristic approach
        # (e.g., largest content div that's not the header or footer)
        content_candidates = []
        
        for div in soup.find_all('div'):
            # Skip header, footer, nav, etc.
            if div.find_parent(['header', 'footer', 'nav']):
                continue
                
            # Skip certain classes/ids that are unlikely to be main content
            class_id = ' '.join(div.get('class', [])) + ' ' + div.get('id', '')
            if re.search(r'sidebar|menu|nav|header|footer|comment', class_id, re.I):
                continue
                
            # Calculate "content score" based on elements that suggest detailed content
            score = 0
            
            # Content with headers is good
            score += 2 * len(div.find_all(['h1', 'h2']))
            
            # Content with paragraphs is good
            score += len(div.find_all('p'))
            
            # Content with images might be product details
            score += len(div.find_all('img'))
            
            # Content with lists might be features or specs
            score += len(div.find_all(['ul', 'ol']))
            
            content_candidates.append((div, score))
        
        # Return the highest scoring candidate if any
        if content_candidates:
            sorted_candidates = sorted(content_candidates, key=lambda x: x[1], reverse=True)
            if sorted_candidates[0][1] > 0:  # Only return if score > 0
                return sorted_candidates[0][0]
        
        # If nothing found, return the body as a fallback
        return soup.body
    
    def _analyze_content(self, content: Tag, page_type_hint: str) -> Dict[str, Any]:
        """
        Analyze the content of a potential detail page.
        
        Args:
            content: Tag object for the content area
            page_type_hint: Hint about the page type from previous analysis
            
        Returns:
            Dictionary with analysis results
        """
        result = {
            "is_detail": False,
            "type": page_type_hint if page_type_hint != "unknown" else None,
            "confidence": 0.0,
            "sections": {}
        }
        
        evidence_points = []
        
        # Check for common detail page elements
        # 1. Look for a prominent title (usually h1)
        title_elem = content.find('h1')
        if title_elem:
            result["sections"]["title"] = {
                "text": title_elem.get_text().strip(),
                "element": title_elem.name,
                "selector": self._generate_element_selector(title_elem)
            }
            evidence_points.append(0.8)  # Good evidence of a detail page
        
        # 2. Look for large text blocks (descriptions, articles)
        desc_candidates = content.find_all(['p', 'div'], class_=re.compile(r'desc|content|detail', re.I))
        if desc_candidates:
            largest_desc = max(desc_candidates, key=lambda x: len(x.get_text()))
            desc_text = largest_desc.get_text().strip()
            
            if len(desc_text) > 100:  # Substantial description
                result["sections"]["description"] = {
                    "text": desc_text[:200] + "..." if len(desc_text) > 200 else desc_text,
                    "element": largest_desc.name,
                    "selector": self._generate_element_selector(largest_desc)
                }
                evidence_points.append(0.7)
        
        # 3. Look for pricing info (product pages)
        price_elem = content.find(class_=re.compile(r'price', re.I)) or content.find(attrs={"itemprop": "price"})
        if price_elem:
            result["sections"]["price"] = {
                "text": price_elem.get_text().strip(),
                "element": price_elem.name,
                "selector": self._generate_element_selector(price_elem)
            }
            evidence_points.append(0.9)  # Strong indicator of a product page
            result["type"] = "product"
        
        # 4. Look for attributes/specifications (table or list)
        attr_container = content.find(class_=re.compile(r'attribute|spec|detail|feature', re.I))
        if attr_container:
            # Extract attributes as key-value pairs where possible
            attr_data = self._extract_attributes(attr_container)
            if attr_data:
                result["sections"]["attributes"] = {
                    "data": attr_data,
                    "element": attr_container.name,
                    "selector": self._generate_element_selector(attr_container)
                }
                evidence_points.append(0.8)
                
                # Try to determine page type from attributes
                if not result["type"] or result["type"] == "unknown":
                    if any(k for k in attr_data.keys() if re.search(r'size|weight|dimension|color', k, re.I)):
                        result["type"] = "product"
                    elif any(k for k in attr_data.keys() if re.search(r'bedroom|bathroom|area|sqft', k, re.I)):
                        result["type"] = "property"
        
        # 5. Look for image gallery (product, property pages)
        gallery = content.find(class_=re.compile(r'gallery|carousel|slider', re.I))
        if gallery:
            images = gallery.find_all('img')
            if images:
                result["sections"]["gallery"] = {
                    "count": len(images),
                    "element": gallery.name,
                    "selector": self._generate_element_selector(gallery)
                }
                evidence_points.append(0.7)
        elif content.find_all('img', limit=3):  # Multiple images even without a gallery
            images = content.find_all('img')
            result["sections"]["images"] = {
                "count": len(images),
                "element": "img",
                "selector": "img"
            }
            evidence_points.append(0.5)
        
        # 6. Look for author info (article pages)
        author = content.find(class_=re.compile(r'author|byline', re.I)) or content.find(attrs={"itemprop": "author"})
        if author:
            result["sections"]["author"] = {
                "text": author.get_text().strip(),
                "element": author.name,
                "selector": self._generate_element_selector(author)
            }
            evidence_points.append(0.8)
            result["type"] = result["type"] or "article"
        
        # 7. Look for date info (article pages)
        date = content.find(class_=re.compile(r'date|publish|posted', re.I)) or content.find('time')
        if date:
            result["sections"]["date"] = {
                "text": date.get_text().strip(),
                "element": date.name,
                "selector": self._generate_element_selector(date)
            }
            evidence_points.append(0.7)
            result["type"] = result["type"] or "article"
        
        # Calculate confidence score
        if evidence_points:
            result["confidence"] = self.calculate_confidence(evidence_points)
            
            # If we have high confidence and multiple sections, it's a detail page
            if result["confidence"] >= 0.7 and len(result["sections"]) >= 2:
                result["is_detail"] = True
        
        # If we still don't know the page type but have high confidence it's a detail page
        # make a guess based on the sections we found
        if result["is_detail"] and (not result["type"] or result["type"] == "unknown"):
            if "price" in result["sections"]:
                result["type"] = "product"
            elif "author" in result["sections"] or "date" in result["sections"]:
                result["type"] = "article"
            else:
                result["type"] = "generic"
        
        return result
    
    def _find_content_sections(self, soup: BeautifulSoup, page_type_hint: str = None) -> Dict[str, Any]:
        """
        Find and analyze important content sections in the page.
        
        Args:
            soup: BeautifulSoup object representing the HTML
            page_type_hint: Optional hint about the page type
            
        Returns:
            Dictionary with detection results
        """
        results = {
            "sections": {},
            "confidence": 0.0
        }
        
        evidence_points = []
        
        # Select which section patterns to look for based on page type hint
        section_types_to_check = {}
        
        if not page_type_hint or page_type_hint == "unknown":
            # Check all section types if page type is unknown
            section_types_to_check = self.section_patterns
        elif page_type_hint == "product":
            # For product pages, look for title, description, price, images, attributes
            section_types_to_check = {k: v for k, v in self.section_patterns.items() 
                                    if k in ['title', 'description', 'price', 'images', 'attributes']}
        elif page_type_hint == "article":
            # For article pages, look for title, description (content), images, metadata
            section_types_to_check = {k: v for k, v in self.section_patterns.items() 
                                    if k in ['title', 'description', 'images', 'metadata']}
        elif page_type_hint == "property":
            # For property pages, look for title, description, price, images, attributes
            section_types_to_check = {k: v for k, v in self.section_patterns.items() 
                                    if k in ['title', 'description', 'price', 'images', 'attributes']}
        elif page_type_hint == "profile":
            # For profile pages, look for title, description, images, metadata
            section_types_to_check = {k: v for k, v in self.section_patterns.items() 
                                    if k in ['title', 'description', 'images', 'metadata']}
        
        # Look for each section type
        for section_type, selectors in section_types_to_check.items():
            for selector in selectors:
                elements = soup.select(selector)
                if elements:
                    # For most sections, we want the first matching element
                    # (title, price, etc.) - except for images where we might want multiple
                    if section_type == 'images' and len(elements) > 1:
                        results["sections"][section_type] = {
                            "count": len(elements),
                            "elements": [self._get_element_info(elem) for elem in elements[:5]],  # Limit to 5 images
                            "selector": selector
                        }
                    else:
                        results["sections"][section_type] = self._get_element_info(elements[0])
                        results["sections"][section_type]["selector"] = selector
                    
                    evidence_points.append(0.7)
                    break  # Stop checking other selectors for this section type
        
        # Calculate confidence based on how many sections we found
        results["confidence"] = self.calculate_confidence(evidence_points)
        
        return results
    
    def _extract_metadata(self, soup: BeautifulSoup, page_type_hint: str = None) -> Dict[str, Any]:
        """
        Extract metadata from the page.
        
        Args:
            soup: BeautifulSoup object representing the HTML
            page_type_hint: Optional hint about the page type
            
        Returns:
            Dictionary with metadata
        """
        metadata = {}
        
        # Extract basic metadata from meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name', '')
            property_attr = meta.get('property', '')
            content = meta.get('content', '')
            
            if name and content:
                metadata[name] = content
            elif property_attr and content:
                if property_attr.startswith('og:'):
                    # OpenGraph metadata
                    key = property_attr[3:]  # Remove 'og:' prefix
                    metadata[key] = content
                else:
                    metadata[property_attr] = content
        
        # Extract structured data if available
        structured_data = {}
        for item in soup.find_all(attrs={"itemtype": True}):
            itemtype = item.get('itemtype', '')
            if not itemtype:
                continue
                
            # Extract type from URL (e.g. http://schema.org/Product -> Product)
            type_match = re.search(r'/([^/]+)$', itemtype)
            if type_match:
                item_type = type_match.group(1)
                structured_data["type"] = item_type
                
                # Extract properties
                for prop in item.find_all(attrs={"itemprop": True}):
                    prop_name = prop.get('itemprop', '')
                    if prop_name:
                        if prop.name == 'meta':
                            prop_value = prop.get('content', '')
                        elif prop.name == 'link':
                            prop_value = prop.get('href', '')
                        elif prop.name == 'time':
                            prop_value = prop.get('datetime', prop.get_text().strip())
                        else:
                            prop_value = prop.get_text().strip()
                        
                        structured_data[prop_name] = prop_value
        
        if structured_data:
            metadata["structured_data"] = structured_data
        
        return metadata
    
    def _extract_attributes(self, container: Tag) -> Dict[str, str]:
        """
        Extract attributes or specifications from a container.
        
        Args:
            container: Tag object for the container element
            
        Returns:
            Dictionary of attribute name-value pairs
        """
        attributes = {}
        
        # Check for definition lists
        dl = container.find('dl')
        if dl:
            # DL typically contains alternating DT (term) and DD (definition)
            dts = dl.find_all('dt')
            dds = dl.find_all('dd')
            
            # Pair them up as much as possible
            for i in range(min(len(dts), len(dds))):
                key = dts[i].get_text().strip()
                value = dds[i].get_text().strip()
                if key and value:
                    attributes[key] = value
            
            return attributes
        
        # Check for tables
        table = container.find('table')
        if table:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['th', 'td'])
                if len(cells) >= 2:
                    key = cells[0].get_text().strip()
                    value = cells[1].get_text().strip()
                    if key and value:
                        attributes[key] = value
            
            return attributes
        
        # Check for lists with key-value patterns
        for li in container.find_all('li'):
            text = li.get_text().strip()
            # Look for colon or similar separators
            for separator in [':', '-', '–']:
                if separator in text and text.index(separator) > 0:
                    parts = text.split(separator, 1)
                    key = parts[0].strip()
                    value = parts[1].strip()
                    if key and value:
                        attributes[key] = value
                        break
        
        # Check for div patterns
        label_pattern = re.compile(r'label|name|key', re.I)
        value_pattern = re.compile(r'value|data', re.I)
        
        # Look for label-value div pairs
        labels = container.find_all(class_=label_pattern)
        for label in labels:
            # Try to find associated value
            value = label.find_next_sibling(class_=value_pattern)
            if value:
                key = label.get_text().strip()
                val = value.get_text().strip()
                if key and val:
                    attributes[key] = val
        
        return attributes
    
    def _get_element_info(self, element: Tag) -> Dict[str, Any]:
        """
        Get basic information about an element.
        
        Args:
            element: Tag object
            
        Returns:
            Dictionary with element information
        """
        info = {
            "element": element.name,
            "text": element.get_text().strip()
        }
        
        # Add href for links
        if element.name == 'a' and element.get('href'):
            info["href"] = element['href']
        
        # Add src for images
        if element.name == 'img':
            info["src"] = element.get('src', '')
            info["alt"] = element.get('alt', '')
            info["text"] = info["alt"]  # Use alt text as the text for images
        
        return info
    
    def _generate_element_selector(self, element: Tag) -> str:
        """
        Generate a CSS selector for an element.
        
        Args:
            element: Tag object
            
        Returns:
            CSS selector string
        """
        # If it has an ID, use that (most specific)
        if element.get('id'):
            return f"#{element['id']}"
            
        # If it has classes, use the most specific class
        if element.get('class'):
            classes = element['class']
            if len(classes) == 1:
                return f"{element.name}.{classes[0]}"
            elif len(classes) > 1:
                # Try to find the most specific class (avoid common classes like 'container')
                common_classes = ['container', 'wrapper', 'box', 'content']
                specific_classes = [c for c in classes if c.lower() not in common_classes]
                
                if specific_classes:
                    return f"{element.name}.{specific_classes[0]}"
                else:
                    # Use first two classes for better specificity
                    return f"{element.name}.{classes[0]}.{classes[1]}"
        
        # Try to use a parent with an ID or class for context
        parent = element.parent
        if parent and parent.name != 'body':
            if parent.get('id'):
                return f"#{parent['id']} > {element.name}"
            elif parent.get('class'):
                parent_class = parent['class'][0] if isinstance(parent['class'], list) else parent['class']
                return f".{parent_class} > {element.name}"
        
        # Last resort: use tag name with nth-of-type
        siblings = list(element.find_previous_siblings(element.name))
        return f"{element.name}:nth-of-type({len(siblings) + 1})"
    
    def _generate_selectors(self, sections: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """
        Generate CSS selectors for each section.
        
        Args:
            sections: Dictionary of page sections
            
        Returns:
            Dictionary mapping section types to selectors
        """
        selectors = {}
        
        for section_type, section_data in sections.items():
            if "selector" in section_data:
                selectors[section_type] = section_data["selector"]
        
        return selectors