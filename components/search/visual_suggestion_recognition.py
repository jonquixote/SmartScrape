"""
Visual suggestion recognition for SmartScrape search automation.

This module provides visual recognition capabilities for detecting and extracting
suggestions from non-standard autocomplete interfaces that cannot be easily
detected through DOM analysis alone.
"""

import logging
import asyncio
import base64
import json
import re
from typing import Dict, List, Any, Optional, Tuple, Union

from playwright.async_api import Page, ElementHandle
from bs4 import BeautifulSoup

# Optional AI detection
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


class VisualSuggestionRecognizer:
    """
    Recognizes search suggestions using visual analysis techniques.
    
    This class:
    - Captures screenshots of suggestion dropdowns
    - Uses visual pattern recognition to identify suggestion items
    - Extracts text from visually identified suggestions
    - Handles non-standard UIs where DOM analysis might fail
    """
    
    def __init__(self, ai_enabled: bool = True):
        """
        Initialize the visual suggestion recognizer.
        
        Args:
            ai_enabled: Whether to use AI for suggestion recognition
        """
        self.logger = logging.getLogger("VisualSuggestionRecognizer")
        self.ai_enabled = ai_enabled and GENAI_AVAILABLE
        
        # Initialize AI model if available and enabled
        if self.ai_enabled:
            try:
                self._initialize_ai()
            except Exception as e:
                self.logger.warning(f"Failed to initialize AI model: {str(e)}")
                self.ai_enabled = False
    
    def _initialize_ai(self):
        """Initialize the Google Generative AI model for visual analysis."""
        # Load API key from config
        try:
            from config import GEMINI_API_KEY
            genai.configure(api_key=GEMINI_API_KEY)
            
            # Configure the generative model for visual recognition
            self.generation_config = {
                "temperature": 0.4,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
            
            self.safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
            
            # Initialize the model
            self.model = genai.GenerativeModel(
                model_name="gemini-pro-vision",
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            
            self.logger.info("AI vision model initialized successfully")
            
        except (ImportError, AttributeError) as e:
            self.logger.warning(f"Failed to load API key or initialize model: {str(e)}")
            self.ai_enabled = False
    
    async def detect_visual_suggestions(self, page: Page, 
                                      input_selector: str) -> Dict[str, Any]:
        """
        Detect visual suggestions after typing in an input field.
        
        Args:
            page: Playwright page object
            input_selector: CSS selector for the input field
            
        Returns:
            Dictionary with visual suggestion information
        """
        self.logger.info(f"Detecting visual suggestions for input: {input_selector}")
        
        try:
            # First check if the input exists
            input_handle = await page.query_selector(input_selector)
            if not input_handle:
                return {"has_suggestions": False, "reason": "Input not found"}
                
            # Capture initial state
            initial_screenshot = await self._capture_viewport(page)
            
            # Type something to trigger suggestions
            await input_handle.click()
            await input_handle.fill("")
            await input_handle.type("a", delay=100)
            
            # Wait for suggestions to appear
            await asyncio.sleep(1)
            
            # Capture state after typing
            after_typing_screenshot = await self._capture_viewport(page)
            
            # Find visual differences
            diff_regions = await self._find_visual_differences(
                page, 
                initial_screenshot,
                after_typing_screenshot
            )
            
            if not diff_regions:
                # Try typing more and waiting longer
                await input_handle.type("p", delay=100)
                await asyncio.sleep(1.5)
                
                # Capture updated state
                after_more_typing_screenshot = await self._capture_viewport(page)
                
                # Try again to find differences
                diff_regions = await self._find_visual_differences(
                    page, 
                    initial_screenshot,
                    after_more_typing_screenshot
                )
                
                if not diff_regions:
                    return {"has_suggestions": False, "reason": "No visual changes detected"}
            
            # Analyze the different regions to find suggestions
            suggestion_regions = await self._analyze_diff_regions(page, diff_regions)
            
            if not suggestion_regions:
                return {"has_suggestions": False, "reason": "No suggestion regions detected"}
                
            # Extract text from suggestion regions
            suggestions = await self._extract_suggestions_from_regions(page, suggestion_regions)
            
            # Clean up
            await input_handle.fill("")
            
            return {
                "has_suggestions": len(suggestions) > 0,
                "suggestion_count": len(suggestions),
                "suggestions": suggestions,
                "visual_regions": suggestion_regions
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting visual suggestions: {str(e)}")
            return {"has_suggestions": False, "reason": f"Error: {str(e)}"}
    
    async def _capture_viewport(self, page: Page) -> str:
        """
        Capture screenshot of the current viewport.
        
        Args:
            page: Playwright page object
            
        Returns:
            Base64 encoded screenshot
        """
        screenshot_bytes = await page.screenshot(
            type="jpeg",
            quality=80,
            full_page=False
        )
        
        return base64.b64encode(screenshot_bytes).decode('utf-8')
    
    async def _find_visual_differences(self, page: Page, 
                                      before_img: str, 
                                      after_img: str) -> List[Dict[str, Any]]:
        """
        Find visual differences between two screenshots.
        
        Args:
            page: Playwright page object
            before_img: Base64 encoded before screenshot
            after_img: Base64 encoded after screenshot
            
        Returns:
            List of region dictionaries with x, y, width, height
        """
        # For comprehensive visual diff we would use a computer vision library
        # Here we'll use a simple DOM-based approach for bounding rectangles
        
        # First identify all elements that might be visible now
        new_elements = await page.evaluate('''
            () => {
                // Look for elements that are likely part of suggestion UIs
                const possibleSuggestions = [
                    // Common suggestion container selectors
                    '.autocomplete-suggestions',
                    '.autocomplete-results',
                    '.typeahead-results',
                    '.search-suggestions',
                    '.search-results',
                    '.suggestions',
                    '.results',
                    '.ui-autocomplete',
                    '.ui-menu',
                    '.tt-menu',
                    '[role="listbox"]',
                    '.dropdown-menu:not(.hidden):not([style*="display: none"])'
                ];
                
                let visibleElements = [];
                
                // Check each potential container
                for (const selector of possibleSuggestions) {
                    const elements = document.querySelectorAll(selector);
                    
                    for (const el of elements) {
                        // Check if visible
                        const rect = el.getBoundingClientRect();
                        const isVisible = rect.width > 0 && 
                                         rect.height > 0 && 
                                         window.getComputedStyle(el).display !== 'none' && 
                                         window.getComputedStyle(el).visibility !== 'hidden';
                        
                        if (isVisible) {
                            visibleElements.push({
                                x: rect.left,
                                y: rect.top,
                                width: rect.width,
                                height: rect.height,
                                selector: selector
                            });
                            
                            // Also get all visible children that look like suggestions
                            const items = el.querySelectorAll('li, .item, .suggestion, .result, [role="option"]');
                            for (const item of items) {
                                const itemRect = item.getBoundingClientRect();
                                const isItemVisible = itemRect.width > 0 && 
                                                     itemRect.height > 0;
                                
                                if (isItemVisible) {
                                    visibleElements.push({
                                        x: itemRect.left,
                                        y: itemRect.top,
                                        width: itemRect.width,
                                        height: itemRect.height,
                                        isItem: true,
                                        text: item.innerText.trim()
                                    });
                                }
                            }
                        }
                    }
                }
                
                // If no elements found with selectors, look for any newly visible elements
                if (visibleElements.length === 0) {
                    // Look for elements that appeared after viewport midpoint (likely dropdown)
                    const viewportHeight = window.innerHeight;
                    const viewportWidth = window.innerWidth;
                    
                    // Check divs and lists that might be dropdown containers
                    document.querySelectorAll('div, ul, ol').forEach(el => {
                        const rect = el.getBoundingClientRect();
                        
                        // Only consider elements below input fields and sized like dropdowns
                        const isVisible = rect.width > 0 && 
                                         rect.height > 0 && 
                                         rect.y > 100 &&  // Roughly below where inputs might be
                                         rect.height > 50 && // Tall enough to be a dropdown
                                         rect.width > 200 && // Wide enough to be a dropdown
                                         window.getComputedStyle(el).display !== 'none' && 
                                         window.getComputedStyle(el).visibility !== 'hidden' &&
                                         window.getComputedStyle(el).position === 'absolute'; // Often positioned
                                         
                        if (isVisible) {
                            // Check if it has multiple similar children (likely suggestion items)
                            const children = el.children;
                            if (children.length > 1) {
                                // Check if children are similar (likely suggestions)
                                const firstChild = children[0];
                                const childTag = firstChild.tagName;
                                let similarChildren = 0;
                                
                                for (let i = 0; i < children.length; i++) {
                                    if (children[i].tagName === childTag) {
                                        similarChildren++;
                                    }
                                }
                                
                                // If most children are similar, this is likely a suggestion container
                                if (similarChildren > children.length * 0.7) {
                                    visibleElements.push({
                                        x: rect.left,
                                        y: rect.top,
                                        width: rect.width,
                                        height: rect.height,
                                        childCount: children.length,
                                        heuristic: true
                                    });
                                    
                                    // Add children as potential items
                                    for (let i = 0; i < children.length; i++) {
                                        const child = children[i];
                                        const childRect = child.getBoundingClientRect();
                                        
                                        if (childRect.width > 0 && childRect.height > 0) {
                                            visibleElements.push({
                                                x: childRect.left,
                                                y: childRect.top,
                                                width: childRect.width,
                                                height: childRect.height,
                                                isItem: true,
                                                text: child.innerText.trim(),
                                                heuristic: true
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    });
                }
                
                return visibleElements;
            }
        ''')
        
        # If we have AI enabled, we can use it to analyze screenshots for more accuracy
        if self.ai_enabled and len(new_elements) < 2:
            try:
                ai_regions = await self._use_ai_for_diff_detection(before_img, after_img)
                if ai_regions:
                    return ai_regions
            except Exception as e:
                self.logger.warning(f"AI diff detection failed: {str(e)}")
        
        return new_elements
    
    async def _use_ai_for_diff_detection(self, before_img: str, after_img: str) -> List[Dict[str, Any]]:
        """
        Use AI vision model to detect differences between screenshots.
        
        Args:
            before_img: Base64 encoded before screenshot
            after_img: Base64 encoded after screenshot
            
        Returns:
            List of region dictionaries with x, y, width, height
        """
        if not self.ai_enabled:
            return []
            
        # Convert base64 to proper format for the model
        before_img_content = {
            "mime_type": "image/jpeg",
            "data": before_img
        }
        
        after_img_content = {
            "mime_type": "image/jpeg",
            "data": after_img
        }
        
        # Create prompt for the model
        prompt = """
        I'm going to show you two screenshots. The first is a webpage before a user typed in a search box.
        The second is after typing, which might show autocomplete or search suggestions.
        
        Please identify the following:
        1. Is there a dropdown or suggestion list visible in the second image that wasn't in the first?
        2. If yes, describe the approximate coordinates (x, y) and size (width, height) of this dropdown.
        3. List any individual suggestion items you can see, with their text content if readable.
        
        Format your response as JSON with this structure:
        {
          "has_suggestions": true/false,
          "suggestion_box": {"x": float, "y": float, "width": float, "height": float},
          "items": [
            {"text": "suggestion text", "y": float, "height": float}
          ]
        }
        
        Just provide the JSON, nothing else.
        """
        
        # Generate content with the model
        response = await asyncio.to_thread(
            self.model.generate_content,
            [prompt, before_img_content, after_img_content]
        )
        
        # Extract JSON from response
        try:
            response_text = response.text
            # Find JSON in the response (it might be embedded in other text)
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                
                # Convert to our format
                regions = []
                
                if result.get("has_suggestions", False):
                    # Add the suggestion box
                    if "suggestion_box" in result:
                        regions.append(result["suggestion_box"])
                    
                    # Add individual items
                    for item in result.get("items", []):
                        if "text" in item:
                            # Create a region for this item
                            item_region = {
                                "x": result.get("suggestion_box", {}).get("x", 0),
                                "width": result.get("suggestion_box", {}).get("width", 0),
                                "y": item.get("y", 0),
                                "height": item.get("height", 30),  # Default height if not provided
                                "text": item.get("text", ""),
                                "isItem": True,
                                "ai_detected": True
                            }
                            regions.append(item_region)
                
                return regions
            
            return []
            
        except (json.JSONDecodeError, AttributeError) as e:
            self.logger.warning(f"Failed to parse AI response: {str(e)}")
            return []
    
    async def _analyze_diff_regions(self, page: Page, 
                                   diff_regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze the diff regions to identify actual suggestion containers.
        
        Args:
            page: Playwright page object
            diff_regions: List of diff regions
            
        Returns:
            List of suggestion region dictionaries
        """
        if not diff_regions:
            return []
            
        # Filter regions that are likely to be suggestion containers
        suggestion_regions = []
        
        for region in diff_regions:
            # If it's already identified as an item, add it
            if region.get("isItem", False):
                suggestion_regions.append(region)
                continue
                
            # Check if this looks like a suggestion container
            is_container = False
            
            # If it has a selector that matches typical suggestion containers
            if "selector" in region:
                is_container = True
                
            # If it was detected via heuristics
            elif region.get("heuristic", False) and region.get("childCount", 0) > 1:
                is_container = True
                
            # If it's positioned like a dropdown (below other elements)
            elif region.get("y", 0) > 100 and region.get("width", 0) > 200:
                is_container = True
            
            if is_container:
                suggestion_regions.append(region)
        
        return suggestion_regions
    
    async def _extract_suggestions_from_regions(self, page: Page, 
                                              regions: List[Dict[str, Any]]) -> List[str]:
        """
        Extract text suggestions from the identified regions.
        
        Args:
            page: Playwright page object
            regions: List of suggestion regions
            
        Returns:
            List of suggestion strings
        """
        suggestions = []
        
        # Get text from regions that already have it
        for region in regions:
            if "text" in region and region.get("isItem", False):
                text = region["text"].strip()
                if text and text not in suggestions:
                    suggestions.append(text)
        
        # If we don't have suggestions yet, try to extract them from containers
        if not suggestions:
            for region in regions:
                # Skip items, we already processed those
                if region.get("isItem", False):
                    continue
                    
                # Try to find elements within this region and extract their text
                region_text = await self._extract_text_from_region(page, region)
                
                for text in region_text:
                    if text and text not in suggestions:
                        suggestions.append(text)
        
        # If we still don't have suggestions and AI is enabled, try OCR
        if not suggestions and self.ai_enabled:
            for region in regions:
                # Skip items, we already processed those
                if region.get("isItem", False) or "text" in region:
                    continue
                    
                # Capture screenshot of just this region
                region_img = await self._capture_region_screenshot(page, region)
                
                # Use AI for OCR
                ocr_text = await self._use_ai_for_ocr(region_img)
                
                for text in ocr_text:
                    if text and text not in suggestions:
                        suggestions.append(text)
        
        return suggestions
    
    async def _extract_text_from_region(self, page: Page, 
                                       region: Dict[str, Any]) -> List[str]:
        """
        Extract text from elements within a region.
        
        Args:
            page: Playwright page object
            region: Region dictionary
            
        Returns:
            List of text strings
        """
        try:
            # Find all elements within this region's coordinates
            elements_in_region = await page.evaluate(f'''
                () => {{
                    const region = {json.dumps(region)};
                    
                    // Function to check if an element is within the region
                    function isInRegion(element) {{
                        const rect = element.getBoundingClientRect();
                        return (
                            rect.left >= region.x && 
                            rect.right <= region.x + region.width &&
                            rect.top >= region.y && 
                            rect.bottom <= region.y + region.height
                        );
                    }}
                    
                    // Find all text-containing elements in the region
                    const textElements = [];
                    const possibleItems = document.querySelectorAll('li, .item, div, span, a, p');
                    
                    for (const el of possibleItems) {{
                        if (isInRegion(el) && el.innerText && el.innerText.trim()) {{
                            // Check if this element's parent is also in the region
                            // If so, we might be getting duplicate content
                            let shouldAdd = true;
                            let parent = el.parentElement;
                            
                            while (parent) {{
                                if (isInRegion(parent) && textElements.includes(parent)) {{
                                    shouldAdd = false;
                                    break;
                                }}
                                parent = parent.parentElement;
                            }}
                            
                            if (shouldAdd) {{
                                textElements.push(el);
                            }}
                        }}
                    }}
                    
                    // Extract text from the elements
                    return textElements.map(el => el.innerText.trim());
                }}
            ''')
            
            return elements_in_region
            
        except Exception as e:
            self.logger.warning(f"Error extracting text from region: {str(e)}")
            return []
    
    async def _capture_region_screenshot(self, page: Page, 
                                        region: Dict[str, Any]) -> str:
        """
        Capture screenshot of a specific region.
        
        Args:
            page: Playwright page object
            region: Region dictionary
            
        Returns:
            Base64 encoded screenshot
        """
        try:
            # Convert region to clip for screenshot
            clip = {
                "x": region["x"],
                "y": region["y"],
                "width": region["width"],
                "height": region["height"]
            }
            
            # Capture the screenshot
            screenshot_bytes = await page.screenshot(
                type="jpeg",
                quality=90,
                clip=clip
            )
            
            return base64.b64encode(screenshot_bytes).decode('utf-8')
            
        except Exception as e:
            self.logger.warning(f"Error capturing region screenshot: {str(e)}")
            return ""
    
    async def _use_ai_for_ocr(self, image_base64: str) -> List[str]:
        """
        Use AI vision model for OCR to extract text from an image.
        
        Args:
            image_base64: Base64 encoded image
            
        Returns:
            List of extracted text strings
        """
        if not self.ai_enabled or not image_base64:
            return []
            
        try:
            # Convert base64 to proper format for the model
            img_content = {
                "mime_type": "image/jpeg",
                "data": image_base64
            }
            
            # Create prompt for the model
            prompt = """
            This image shows a part of a webpage that might contain search suggestions or autocomplete results.
            
            Please extract all text that appears to be search suggestions or results.
            Format your response as a JSON array of strings, with each suggestion as a separate item.
            
            For example:
            ["First suggestion", "Second suggestion", "Third suggestion"]
            
            Just provide the JSON array, nothing else.
            """
            
            # Generate content with the model
            response = await asyncio.to_thread(
                self.model.generate_content,
                [prompt, img_content]
            )
            
            # Extract JSON from response
            response_text = response.text
            
            # Find JSON array in the response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                
                # Ensure result is a list of strings
                return [str(item).strip() for item in result if item]
            
            return []
            
        except Exception as e:
            self.logger.warning(f"AI OCR failed: {str(e)}")
            return []
    
    async def extract_suggestions_with_semantics(self, 
                                              suggestions: List[str], 
                                              search_term: str) -> List[Dict[str, Any]]:
        """
        Extract suggestions with semantic information about relevance to search term.
        
        Args:
            suggestions: List of suggestion strings
            search_term: Original search term
            
        Returns:
            List of suggestion dictionaries with semantic information
        """
        if not suggestions:
            return []
            
        # Simple relevance calculation without AI
        result = []
        for suggestion in suggestions:
            # Calculate simple text similarity
            relevance = self._calculate_text_similarity(suggestion, search_term)
            
            # Determine if this is a common autocomplete pattern
            is_completion = suggestion.lower().startswith(search_term.lower())
            
            # Check if it's a query refinement
            is_refinement = (search_term.lower() in suggestion.lower() and 
                             len(suggestion) > len(search_term) + 5)
            
            # Check if it's a category suggestion
            is_category = any(category in suggestion.lower() for category in 
                             ["category", "department", "section", "in", "under"])
            
            result.append({
                "text": suggestion,
                "relevance": relevance,
                "is_completion": is_completion,
                "is_refinement": is_refinement,
                "is_category": is_category
            })
        
        # Sort by relevance
        result.sort(key=lambda x: x["relevance"], reverse=True)
        
        return result
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate a simple text similarity score.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score between 0 and 1
        """
        # Convert to lowercase
        text1 = text1.lower()
        text2 = text2.lower()
        
        # Check if one text contains the other
        if text1 in text2:
            return len(text1) / len(text2)
        
        if text2 in text1:
            return len(text2) / len(text1)
        
        # Count common words
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        common_words = words1.intersection(words2)
        
        if not words1 or not words2:
            return 0
            
        # Jaccard similarity
        return len(common_words) / len(words1.union(words2))


# Register components in __init__.py
__all__ = [
    'VisualSuggestionRecognizer'
]