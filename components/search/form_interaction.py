"""
Form interaction module for SmartScrape search automation.

This module provides intelligent form field identification and interaction
capabilities for search forms, including field type detection, value inference,
and optimized form submissions.
"""

import logging
import re
import asyncio
import random
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from bs4 import BeautifulSoup, Tag
from urllib.parse import urlparse, urljoin

# Import enhanced utilities
from utils.html_utils import parse_html, extract_text_fast, find_by_xpath
from utils.retry_utils import with_exponential_backoff

# Lazy imports to avoid circular dependencies
try:
    from playwright.async_api import Page
except ImportError:
    Page = Any  # Type hint for when playwright is not available


class HumanLikeInteraction:
    """
    Provides human-like interaction with web forms to avoid detection.
    
    This class:
    - Implements random typing speeds and patterns
    - Simulates natural mouse movements and clicks
    - Adds realistic pauses between actions
    - Creates natural interaction patterns for different input types
    """
    
    def __init__(self, min_typing_speed=50, max_typing_speed=150):
        """
        Initialize human-like interaction simulator.
        
        Args:
            min_typing_speed: Minimum typing speed in characters per minute
            max_typing_speed: Maximum typing speed in characters per minute
        """
        self.logger = logging.getLogger("HumanLikeInteraction")
        self.min_typing_speed = min_typing_speed
        self.max_typing_speed = max_typing_speed
        
        # Define typing pattern parameters
        self.burst_probability = 0.3  # Probability of typing in bursts
        self.pause_probability = 0.1  # Probability of pausing while typing
        self.mistake_probability = 0.02  # Probability of making typing mistakes
        self.correction_delay = 0.3  # Delay before correcting mistakes (seconds)
        
        # Define mouse movement parameters
        self.movement_jitter = 0.2  # Amount of randomness in mouse movements
        
    async def human_type(self, page: Page, selector: str, text: str) -> bool:
        """
        Type text into an element in a human-like manner.
        
        Args:
            page: Playwright page object
            selector: CSS selector for the element
            text: Text to type
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # First check if the element exists and is visible
            element_visible = await page.evaluate(f'''
                () => {{
                    const element = document.querySelector("{selector}");
                    if (!element) return false;
                    
                    const style = window.getComputedStyle(element);
                    return element.type !== 'hidden' && 
                           style.display !== 'none' && 
                           style.visibility !== 'hidden' &&
                           element.offsetWidth > 0;
                }}
            ''')
            
            if not element_visible:
                self.logger.warning(f"Element not visible: {selector}")
                return False
                
            # Click the element to focus it (with a small natural delay)
            await self._natural_click(page, selector)
            
            # Clear the field if it already has content
            await page.evaluate(f'''
                () => {{
                    const element = document.querySelector("{selector}");
                    if (element) element.value = "";
                }}
            ''')
            
            # Type the text in a human-like manner
            await self._type_with_human_pattern(page, selector, text)
            
            # Verify text was entered
            value = await page.evaluate(f'''
                () => {{
                    const element = document.querySelector("{selector}");
                    return element ? element.value : null;
                }}
            ''')
            
            # Sometimes the typing doesn't work properly, so we'll use a fallback approach
            if not value or value != text:
                self.logger.info(f"Using fallback typing method for {selector}")
                await page.fill(selector, text)
                
                # Trigger input event to ensure any watchers are notified
                await page.evaluate(f'''
                    () => {{
                        const element = document.querySelector("{selector}");
                        if (element) {{
                            element.dispatchEvent(new Event('input', {{ bubbles: true }}));
                            element.dispatchEvent(new Event('change', {{ bubbles: true }}));
                        }}
                    }}
                ''')
            
            return True
        except Exception as e:
            self.logger.error(f"Error in human typing: {str(e)}")
            return False
            
    async def _type_with_human_pattern(self, page: Page, selector: str, text: str):
        """
        Type text with human-like patterns including bursts, pauses and occasional mistakes.
        
        Args:
            page: Playwright page object
            selector: CSS selector for the element
            text: Text to type
        """
        i = 0
        while i < len(text):
            # Decide if we should type in a burst
            if random.random() < self.burst_probability and i + 3 < len(text):
                # Type a burst of 2-5 characters
                burst_length = random.randint(2, min(5, len(text) - i))
                burst_text = text[i:i+burst_length]
                await page.type(selector, burst_text, delay=self._random_typing_delay() / 2)
                i += burst_length
            else:
                # Decide if we should make a typo
                if random.random() < self.mistake_probability:
                    # Type a wrong character
                    wrong_char = self._get_typo_character(text[i])
                    await page.type(selector, wrong_char, delay=self._random_typing_delay())
                    await asyncio.sleep(self.correction_delay)
                    
                    # Backspace to correct the mistake
                    await page.press(selector, "Backspace")
                    await asyncio.sleep(0.1)
                
                # Type the correct character
                await page.type(selector, text[i], delay=self._random_typing_delay())
                i += 1
                
            # Occasionally pause while typing
            if random.random() < self.pause_probability:
                await asyncio.sleep(random.uniform(0.5, 2.0))
    
    async def _natural_click(self, page: Page, selector: str):
        """
        Perform a natural-looking click with mouse movement.
        
        Args:
            page: Playwright page object
            selector: CSS selector for the element
        """
        # Add a small random delay before clicking
        await asyncio.sleep(random.uniform(0.2, 0.8))
        
        try:
            # Get the element's position and size
            element_info = await page.evaluate(f'''
                () => {{
                    const element = document.querySelector("{selector}");
                    if (!element) return null;
                    
                    const rect = element.getBoundingClientRect();
                    return {{
                        x: rect.left,
                        y: rect.top,
                        width: rect.width,
                        height: rect.height
                    }};
                }}
            ''')
            
            if not element_info:
                # Just perform a regular click if we can't get position info
                await page.click(selector)
                return
                
            # Calculate a random point within the element to click
            click_x = element_info['x'] + random.uniform(0.2, 0.8) * element_info['width']
            click_y = element_info['y'] + random.uniform(0.2, 0.8) * element_info['height']
            
            # Move mouse to element with a natural motion
            await page.mouse.move(
                click_x + random.uniform(-5, 5) * self.movement_jitter,
                click_y + random.uniform(-5, 5) * self.movement_jitter
            )
            
            # Small delay before clicking
            await asyncio.sleep(random.uniform(0.05, 0.2))
            
            # Click the element
            await page.mouse.click(click_x, click_y)
            
        except Exception as e:
            self.logger.warning(f"Error during natural click, falling back to regular click: {str(e)}")
            # Fall back to regular click
            await page.click(selector)
    
    async def human_select(self, page: Page, selector: str, value: str) -> bool:
        """
        Select an option from a dropdown in a human-like manner.
        
        Args:
            page: Playwright page object
            selector: CSS selector for the select element
            value: Value to select
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Click to open the dropdown
            await self._natural_click(page, selector)
            
            # Add a small delay to simulate looking at options
            await asyncio.sleep(random.uniform(0.5, 1.5))
            
            # Select the option
            selected = await page.evaluate(f'''
                (() => {{
                    const select = document.querySelector("{selector}");
                    if (!select) return false;
                    
                    // Try to find option by value
                    let option = Array.from(select.options).find(opt => opt.value === "{value}");
                    
                    // If not found, try to find by text
                    if (!option) {{
                        option = Array.from(select.options).find(opt => 
                            opt.text.toLowerCase().includes("{value.lower()}"));
                    }}
                    
                    if (option) {{
                        select.value = option.value;
                        select.dispatchEvent(new Event('change', {{ bubbles: true }}));
                        return true;
                    }}
                    
                    return false;
                }})()
            ''')
            
            # If JavaScript selection failed, try using Playwright's select method
            if not selected:
                await page.select_option(selector, value=value)
                
            return True
        except Exception as e:
            self.logger.error(f"Error in human select: {str(e)}")
            return False
    
    async def human_checkbox_click(self, page: Page, selector: str, check: bool) -> bool:
        """
        Click a checkbox in a human-like manner.
        
        Args:
            page: Playwright page object
            selector: CSS selector for the checkbox
            check: Whether to check or uncheck
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check current state
            is_checked = await page.evaluate(f'''
                () => {{
                    const checkbox = document.querySelector("{selector}");
                    return checkbox ? checkbox.checked : null;
                }}
            ''')
            
            # Only click if the state needs to change
            if is_checked is None:
                return False
                
            if (check and not is_checked) or (not check and is_checked):
                await self._natural_click(page, selector)
                
            # Verify the checkbox was toggled correctly
            new_state = await page.evaluate(f'''
                () => {{
                    const checkbox = document.querySelector("{selector}");
                    return checkbox ? checkbox.checked : null;
                }}
            ''')
            
            return new_state == check
        except Exception as e:
            self.logger.error(f"Error in human checkbox click: {str(e)}")
            return False
    
    def _random_typing_delay(self) -> float:
        """
        Generate a random typing delay in milliseconds based on human typing speed.
        
        Returns:
            Typing delay in milliseconds
        """
        # Convert characters per minute to milliseconds per character
        min_delay = 60000 / self.max_typing_speed
        max_delay = 60000 / self.min_typing_speed
        
        # Add some randomness to make typing more natural
        return random.uniform(min_delay, max_delay)
    
    def _get_typo_character(self, char: str) -> str:
        """
        Get a realistic typo for a character based on keyboard layout.
        
        Args:
            char: The intended character
            
        Returns:
            A character that would be a likely typo
        """
        # Map of adjacent keys on QWERTY keyboard
        keyboard_map = {
            'a': 'sqwz',
            'b': 'vghn',
            'c': 'xdfv',
            'd': 'serfcx',
            'e': 'wrsdf',
            'f': 'drtgvc',
            'g': 'ftyhbv',
            'h': 'gyujnb',
            'i': 'uojkl',
            'j': 'huikmn',
            'k': 'jiolm',
            'l': 'kop;',
            'm': 'njk,',
            'n': 'bhjm',
            'o': 'iklp',
            'p': 'ol;[',
            'q': 'asw',
            'r': 'edft',
            's': 'qazxdcw',
            't': 'rfgy',
            'u': 'yhji',
            'v': 'cfgb',
            'w': 'qase',
            'x': 'zasdc',
            'y': 'tghu',
            'z': 'asx',
            '0': '9-p',
            '1': '2q',
            '2': '1qw3',
            '3': '2we4',
            '4': '3er5',
            '5': '4rt6',
            '6': '5ty7',
            '7': '6yu8',
            '8': '7ui9',
            '9': '8io0',
            ' ': 'zxcvbnm'
        }
        
        # If character isn't in our map, return it unchanged
        char = char.lower()
        if char not in keyboard_map:
            return char
            
        # Return a random adjacent key
        adjacent_keys = keyboard_map[char]
        return random.choice(adjacent_keys)


class DOMManipulator:
    """
    Provides direct DOM manipulation for form interactions.
    
    This class:
    - Offers low-level DOM manipulation for form elements
    - Handles complex form structures that may be difficult to interact with
    - Provides alternatives when standard form interaction fails
    - Bypasses client-side validation when necessary
    """
    
    def __init__(self):
        """Initialize the DOM manipulator."""
        self.logger = logging.getLogger("DOMManipulator")
        
    async def set_input_value(self, page: Page, selector: str, value: str) -> bool:
        """
        Directly set an input field's value via JavaScript.
        
        Args:
            page: Playwright page object
            selector: CSS selector for the input element
            value: Value to set
            
        Returns:
            True if successful, False otherwise
        """
        try:
            success = await page.evaluate(f'''
                (() => {{
                    const element = document.querySelector("{selector}");
                    if (!element) return false;
                    
                    // Store original properties
                    const descriptor = Object.getOwnPropertyDescriptor(element, 'value');
                    const originalValue = element.value;
                    const originalSelectionStart = element.selectionStart;
                    const originalSelectionEnd = element.selectionEnd;
                    const originalSelectionDirection = element.selectionDirection;
                    
                    // Override value setter
                    Object.defineProperty(element, 'value', {{
                        configurable: true,
                        value: "{value}"
                    }});
                    
                    // Dispatch events
                    element.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    element.dispatchEvent(new Event('change', {{ bubbles: true }}));
                    
                    // Restore original property descriptor
                    if (descriptor) {{
                        Object.defineProperty(element, 'value', descriptor);
                    }}
                    
                    return true;
                }})()
            ''')
            
            return success
        except Exception as e:
            self.logger.error(f"Error setting input value: {str(e)}")
            return False
    
    async def select_option_by_text(self, page: Page, selector: str, option_text: str) -> bool:
        """
        Select a dropdown option by text rather than value.
        
        Args:
            page: Playwright page object
            selector: CSS selector for the select element
            option_text: Text of the option to select
            
        Returns:
            True if successful, False otherwise
        """
        try:
            success = await page.evaluate(f'''
                (() => {{
                    const select = document.querySelector("{selector}");
                    if (!select) return false;
                    
                    const options = Array.from(select.options);
                    const option = options.find(opt => 
                        opt.text.toLowerCase().includes("{option_text.lower()}"));
                    
                    if (!option) return false;
                    
                    select.value = option.value;
                    select.dispatchEvent(new Event('change', {{ bubbles: true }}));
                    return true;
                }})()
            ''')
            
            return success
        except Exception as e:
            self.logger.error(f"Error selecting option by text: {str(e)}")
            return False
    
    async def set_checkbox(self, page: Page, selector: str, checked: bool) -> bool:
        """
        Set a checkbox state via JavaScript.
        
        Args:
            page: Playwright page object
            selector: CSS selector for the checkbox
            checked: Whether to check or uncheck
            
        Returns:
            True if successful, False otherwise
        """
        try:
            success = await page.evaluate(f'''
                (() => {{
                    const checkbox = document.querySelector("{selector}");
                    if (!checkbox) return false;
                    
                    checkbox.checked = {str(checked).lower()};
                    checkbox.dispatchEvent(new Event('change', {{ bubbles: true }}));
                    return true;
                }})()
            ''')
            
            return success
        except Exception as e:
            self.logger.error(f"Error setting checkbox: {str(e)}")
            return False
    
    async def trigger_submit(self, page: Page, form_selector: str) -> bool:
        """
        Trigger form submission via JavaScript.
        
        Args:
            page: Playwright page object
            form_selector: CSS selector for the form
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Try multiple submission methods
            success = await page.evaluate(f'''
                (() => {{
                    const form = document.querySelector("{form_selector}");
                    if (!form) return false;
                    
                    // Method 1: Use the form's submit method
                    try {{
                        form.submit();
                        return true;
                    }} catch (e) {{
                        // Continue to next method if this fails
                        console.log("Method 1 failed, trying method 2");
                    }}
                    
                    // Method 2: Find and click the submit button
                    const submitButton = form.querySelector('input[type="submit"], button[type="submit"], button:not([type])');
                    if (submitButton) {{
                        submitButton.click();
                        return true;
                    }}
                    
                    // Method 3: Dispatch submit event
                    try {{
                        const event = new Event('submit', {{ bubbles: true, cancelable: true }});
                        const submitted = form.dispatchEvent(event);
                        return submitted;
                    }} catch (e) {{
                        return false;
                    }}
                }})()
            ''')
            
            return success
        except Exception as e:
            self.logger.error(f"Error triggering form submission: {str(e)}")
            return False
    
    async def bypass_validation(self, page: Page, selector: str, attribute_name: str) -> bool:
        """
        Bypass client-side validation by removing validation attributes.
        
        Args:
            page: Playwright page object
            selector: CSS selector for the element
            attribute_name: Validation attribute to remove (e.g., 'required', 'pattern')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            success = await page.evaluate(f'''
                (() => {{
                    const element = document.querySelector("{selector}");
                    if (!element) return false;
                    
                    // Remove the validation attribute
                    element.removeAttribute("{attribute_name}");
                    return true;
                }})()
            ''')
            
            return success
        except Exception as e:
            self.logger.error(f"Error bypassing validation: {str(e)}")
            return False
    
    async def fix_date_picker(self, page: Page, selector: str, date_value: str) -> bool:
        """
        Fix date input fields that might have custom date pickers.
        
        Args:
            page: Playwright page object
            selector: CSS selector for the date input
            date_value: Date value in 'YYYY-MM-DD' format
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # First try the standard approach
            await page.fill(selector, date_value)
            
            # Check if the value was set correctly
            value_set = await page.evaluate(f'''
                (() => {{
                    const element = document.querySelector("{selector}");
                    return element ? element.value === "{date_value}" : false;
                }})()
            ''')
            
            if value_set:
                return True
                
            # If not, try more aggressive methods
            success = await page.evaluate(f'''
                (() => {{
                    const input = document.querySelector("{selector}");
                    if (!input) return false;
                    
                    // Force the input value
                    input.value = "{date_value}";
                    
                    // Dispatch both input and change events
                    input.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    input.dispatchEvent(new Event('change', {{ bubbles: true }}));
                    
                    // If this is a custom date picker, it might have hidden inputs
                    // Look for hidden inputs near this one
                    const form = input.form || input.closest('form');
                    if (form) {{
                        const dateParts = "{date_value}".split('-');
                        if (dateParts.length === 3) {{
                            const year = dateParts[0];
                            const month = dateParts[1];
                            const day = dateParts[2];
                            
                            // Try to find and set related date inputs
                            const allInputs = form.querySelectorAll('input[type="hidden"]');
                            for (const hiddenInput of allInputs) {{
                                const name = (hiddenInput.name || '').toLowerCase();
                                if (name.includes('year')) hiddenInput.value = year;
                                else if (name.includes('month')) hiddenInput.value = month;
                                else if (name.includes('day')) hiddenInput.value = day;
                                else if (name.includes('date')) hiddenInput.value = "{date_value}";
                                
                                hiddenInput.dispatchEvent(new Event('change', {{ bubbles: true }}));
                            }}
                        }}
                    }}
                    
                    return true;
                }})()
            ''')
            
            return success
        except Exception as e:
            self.logger.error(f"Error fixing date picker: {str(e)}")
            return False


class AJAXWaitStrategies:
    """
    Implements intelligent waiting strategies for AJAX updates.
    
    This class:
    - Detects and waits for AJAX requests to complete
    - Handles dynamic content updates after form interaction
    - Implements progressive waiting patterns for better performance
    - Provides verification of content updates
    """
    
    def __init__(self):
        """Initialize AJAX wait strategies."""
        self.logger = logging.getLogger("AJAXWaitStrategies")
        
        # Configurable timeouts
        self.network_idle_timeout = 500  # ms
        self.animation_timeout = 300  # ms
        self.dynamic_content_timeout = 5000  # ms
        self.max_wait_time = 30000  # ms
        
    async def wait_for_network_idle(self, page: Page, timeout: int = None) -> bool:
        """
        Wait for network to be idle (no requests for a period).
        
        Args:
            page: Playwright page object
            timeout: Custom timeout in ms (uses default if None)
            
        Returns:
            True if network became idle, False if timed out
        """
        if timeout is None:
            timeout = self.network_idle_timeout
            
        try:
            # Add script to track XHR/fetch requests
            await page.evaluate('''
                () => {
                    if (window._activeRequests === undefined) {
                        window._activeRequests = 0;
                        window._requestsChanged = false;
                        
                        // Track XMLHttpRequest
                        const oldXHROpen = XMLHttpRequest.prototype.open;
                        XMLHttpRequest.prototype.open = function() {
                            window._activeRequests++;
                            window._requestsChanged = true;
                            
                            this.addEventListener('load', function() {
                                window._activeRequests--;
                                window._requestsChanged = true;
                            });
                            
                            this.addEventListener('error', function() {
                                window._activeRequests--;
                                window._requestsChanged = true;
                            });
                            
                            this.addEventListener('abort', function() {
                                window._activeRequests--;
                                window._requestsChanged = true;
                            });
                            
                            return oldXHROpen.apply(this, arguments);
                        };
                        
                        // Track fetch
                        const oldFetch = window.fetch;
                        window.fetch = function() {
                            window._activeRequests++;
                            window._requestsChanged = true;
                            
                            return oldFetch.apply(this, arguments)
                                .then(response => {
                                    window._activeRequests--;
                                    window._requestsChanged = true;
                                    return response;
                                })
                                .catch(error => {
                                    window._activeRequests--;
                                    window._requestsChanged = true;
                                    throw error;
                                });
                        };
                    }
                    
                    // Reset for this check
                    window._requestsChanged = false;
                }
            ''')
            
            # Wait for network to be idle
            start_time = time.time()
            max_wait_time_sec = self.max_wait_time / 1000
            
            while time.time() - start_time < max_wait_time_sec:
                # Check active requests
                requests_info = await page.evaluate('''
                    () => {
                        return {
                            active: window._activeRequests || 0,
                            changed: window._requestsChanged
                        };
                    }
                ''')
                
                active_requests = requests_info.get('active', 0)
                requests_changed = requests_info.get('changed', False)
                
                # If no active requests and none completed recently, we're idle
                if active_requests == 0 and not requests_changed:
                    return True
                    
                # Reset changed flag
                if requests_changed:
                    await page.evaluate('() => { window._requestsChanged = false; }')
                
                # Wait a bit before checking again
                await asyncio.sleep(timeout / 1000)
            
            # If we got here, we timed out
            self.logger.warning("Timed out waiting for network idle")
            return False
            
        except Exception as e:
            self.logger.error(f"Error waiting for network idle: {str(e)}")
            return False
    
    async def wait_for_animations(self, page: Page, timeout: int = None) -> bool:
        """
        Wait for CSS animations and transitions to complete.
        
        Args:
            page: Playwright page object
            timeout: Custom timeout in ms (uses default if None)
            
        Returns:
            True when animations complete or timeout reached
        """
        if timeout is None:
            timeout = self.animation_timeout
            
        try:
            # Check for any active animations
            animations_complete = await page.evaluate(f'''
                () => new Promise(resolve => {{
                    const animating = document.getAnimations ? 
                        document.getAnimations().filter(a => a.playState !== 'finished').length > 0 : false;
                    
                    if (!animating) {{
                        // Check for any CSS transitions
                        const transitioning = Array.from(document.querySelectorAll('*')).some(el => {{
                            const style = window.getComputedStyle(el);
                            return style.transitionDuration !== '0s' && style.transitionProperty !== 'none';
                        }});
                        
                        if (!transitioning) {{
                            resolve(true);
                            return;
                        }}
                    }}
                    
                    // If still animating, wait a bit and resolve anyway
                    setTimeout(() => resolve(false), {timeout});
                }});
            ''')
            
            return animations_complete
        except Exception as e:
            self.logger.error(f"Error waiting for animations: {str(e)}")
            return True  # Continue anyway on error
    
    async def wait_for_dynamic_content(self, page: Page, selector: str, timeout: int = None) -> bool:
        """
        Wait for dynamic content to appear after an AJAX request.
        
        Args:
            page: Playwright page object
            selector: CSS selector for the content to wait for
            timeout: Custom timeout in ms (uses default if None)
            
        Returns:
            True if content appeared, False if timed out
        """
        if timeout is None:
            timeout = self.dynamic_content_timeout
            
        try:
            # Use Playwright's built-in waitForSelector
            await page.wait_for_selector(selector, timeout=timeout)
            return True
        except Exception as e:
            self.logger.error(f"Error waiting for dynamic content: {str(e)}")
            return False
    
    async def wait_with_progressive_strategy(self, page: Page, content_selector: str = None) -> bool:
        """
        Use a progressive waiting strategy:
        1. First wait for network idle (short timeout)
        2. Then wait for animations to complete
        3. Then wait for specific content if provided
        4. Finally verify DOM is stable
        
        Args:
            page: Playwright page object
            content_selector: Optional CSS selector for expected content
            
        Returns:
            True if waiting completed successfully
        """
        # Wait for network to become idle
        await self.wait_for_network_idle(page, 1000)
        
        # Wait for animations
        await self.wait_for_animations(page)
        
        # Wait for specific content if selector provided
        if content_selector:
            content_appeared = await self.wait_for_dynamic_content(page, content_selector)
            if not content_appeared:
                self.logger.warning(f"Dynamic content did not appear: {content_selector}")
        
        # Verify DOM is stable (not changing rapidly)
        dom_stable = await self._verify_stable_dom(page)
        
        return dom_stable
        
    async def _verify_stable_dom(self, page: Page) -> bool:
        """
        Verify the DOM is stable (not changing rapidly).
        
        Args:
            page: Playwright page object
            
        Returns:
            True if DOM is stable
        """
        try:
            # Take DOM snapshots a second apart and compare them
            snapshot1 = await page.evaluate('''
                () => document.documentElement.outerHTML
            ''')
            
            # Wait a moment
            await asyncio.sleep(0.5)
            
            snapshot2 = await page.evaluate('''
                () => document.documentElement.outerHTML
            ''')
            
            # If the snapshots are identical or very similar, DOM is stable
            # Use a quick length comparison as a heuristic
            length_diff = abs(len(snapshot1) - len(snapshot2))
            is_stable = length_diff < 50  # Allow small differences
            
            if not is_stable:
                self.logger.info(f"DOM still changing (diff: {length_diff} chars)")
                
                # Wait a bit longer for stability
                await asyncio.sleep(1)
            
            return True  # Continue regardless of stability
        except Exception as e:
            self.logger.error(f"Error verifying DOM stability: {str(e)}")
            return True  # Continue anyway on error


class SubmissionVerification:
    """
    Verifies successful form submission and result quality.
    
    This class:
    - Confirms successful form submission
    - Detects common submission issues and error messages
    - Verifies results match expected criteria
    - Checks for captchas or anti-bot mechanisms
    """
    
    def __init__(self):
        """Initialize submission verification."""
        self.logger = logging.getLogger("SubmissionVerification")
        
    async def verify_submission(self, page: Page, expected_indicators: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Verify if a form submission was successful.
        
        Args:
            page: Playwright page object
            expected_indicators: Optional dictionary of expected indicators
            
        Returns:
            Dictionary with verification results
        """
        result = {
            "success": False,
            "url_changed": False,
            "has_results": False,
            "has_errors": False,
            "has_captcha": False,
            "message": None
        }
        
        try:
            # Check for common success indicators
            indicators = await page.evaluate('''
                () => {
                    const pageText = document.body.innerText.toLowerCase();
                    
                    // Check for result indicators
                    const resultTerms = ['result', 'found', 'match', 'listing', 'product'];
                    const hasResults = resultTerms.some(term => pageText.includes(term)) ||
                                     document.querySelectorAll('.result, .search-result, .listing, .product, [data-role="result"]').length > 0;
                    
                    // Check for error messages
                    const errorTerms = ['no result', 'not found', 'no match', 'error', 'invalid', 'failed'];
                    const hasExplicitErrors = errorTerms.some(term => pageText.includes(term)) ||
                                           document.querySelectorAll('.error, .alert-error, .alert-danger, .validation-error').length > 0;
                    
                    // Check for captcha
                    const captchaTerms = ['captcha', 'robot', 'human verification', 'verify you are human'];
                    const hasCaptcha = captchaTerms.some(term => pageText.includes(term)) ||
                                    document.querySelectorAll('.captcha, .recaptcha, iframe[src*="captcha"], iframe[src*="recaptcha"]').length > 0;
                    
                    // Extract any error messages
                    let errorMessage = null;
                    const errorElements = document.querySelectorAll('.error, .alert-error, .alert-danger, .validation-error, .form-error');
                    if (errorElements.length > 0) {
                        errorMessage = errorElements[0].innerText.trim();
                    }
                    
                    return {
                        hasResults,
                        hasExplicitErrors,
                        hasCaptcha,
                        errorMessage,
                        url: window.location.href
                    };
                }
            ''')
            
            # Update result with findings
            result["url"] = indicators.get("url", page.url)
            result["url_changed"] = page.url != page.url
            result["has_results"] = indicators.get("hasResults", False)
            result["has_errors"] = indicators.get("hasExplicitErrors", False)
            result["has_captcha"] = indicators.get("hasCaptcha", False)
            result["message"] = indicators.get("errorMessage")
            
            # Determine overall success
            result["success"] = (
                (result["url_changed"] or result["has_results"]) and 
                not result["has_errors"] and 
                not result["has_captcha"]
            )
            
            # If expected indicators provided, check against them
            if expected_indicators:
                if "expected_url_pattern" in expected_indicators:
                    pattern = expected_indicators["expected_url_pattern"]
                    result["url_matches_expected"] = pattern in result["url"]
                    result["success"] = result["success"] and result["url_matches_expected"]
                    
                if "expected_result_count" in expected_indicators:
                    # Try to detect result count
                    result_count = await self._extract_result_count(page)
                    result["result_count"] = result_count
                    expected_count = expected_indicators["expected_result_count"]
                    
                    # Verify against expectations
                    if expected_count == 0:
                        result["count_matches_expected"] = result_count == 0
                    else:
                        result["count_matches_expected"] = result_count > 0
                    
                    result["success"] = result["success"] and result["count_matches_expected"]
            
        except Exception as e:
            self.logger.error(f"Error during submission verification: {str(e)}")
            result["error"] = str(e)
            
        return result
    
    async def _extract_result_count(self, page: Page) -> int:
        """
        Extract the number of results from the page.
        
        Args:
            page: Playwright page object
            
        Returns:
            Number of results, or -1 if could not determine
        """
        try:
            count_info = await page.evaluate('''
                () => {
                    // Try to find an explicit count element
                    const countElements = Array.from(document.querySelectorAll(
                        '.result-count, .search-count, .count, [data-role="result-count"]'
                    ));
                    
                    if (countElements.length > 0) {
                        const countText = countElements[0].innerText;
                        const matches = countText.match(/\\d+/);
                        if (matches) return parseInt(matches[0], 10);
                    }
                    
                    // Try to parse from text
                    const bodyText = document.body.innerText;
                    const resultPatterns = [
                        /found (\\d+) results/i,
                        /(\\d+) results found/i,
                        /(\\d+) matches/i,
                        /(\\d+) products/i,
                        /(\\d+) listings/i,
                        /showing (\\d+) of/i
                    ];
                    
                    for (const pattern of resultPatterns) {
                        const match = bodyText.match(pattern);
                        if (match) return parseInt(match[1], 10);
                    }
                    
                    // Count result elements
                    const resultElements = document.querySelectorAll(
                        '.result, .search-result, .product, .listing, [data-role="result"]'
                    );
                    
                    if (resultElements.length > 0) {
                        return resultElements.length;
                    }
                    
                    // Could not determine
                    return -1;
                }
            ''')
            
            return count_info if count_info is not None else -1
        except Exception:
            return -1
    
    async def check_for_blocked_access(self, page: Page) -> Dict[str, Any]:
        """
        Check if access is blocked by anti-bot measures.
        
        Args:
            page: Playwright page object
            
        Returns:
            Dictionary with block detection results
        """
        try:
            block_info = await page.evaluate('''
                () => {
                    const pageText = document.body.innerText.toLowerCase();
                    
                    // Common block indicators in text
                    const blockTerms = [
                        'access denied', 'blocked', 'suspicious activity',
                        'unusual traffic', 'detected automated', 'bot detected',
                        'confirm you are human', 'please verify', 'too many requests'
                    ];
                    
                    const hasBlockText = blockTerms.some(term => pageText.includes(term));
                    
                    // Check for typical block response codes in HTTP headers
                    // Note: This won't be reliable as we can only see what's in the DOM
                    const hasErrorPage = document.title.toLowerCase().includes('error') || 
                                      document.title.toLowerCase().includes('denied') ||
                                      document.title.toLowerCase().includes('blocked');
                    
                    // Check for IP block messages
                    const hasIPBlock = pageText.includes('ip address') && 
                                     (pageText.includes('blocked') || pageText.includes('banned'));
                    
                    return {
                        blocked: hasBlockText || hasErrorPage || hasIPBlock,
                        hasBlockText,
                        hasErrorPage,
                        hasIPBlock
                    };
                }
            ''')
            
            if block_info.get("blocked", False):
                self.logger.warning("Detected blocked access or anti-bot measures")
                
            return block_info
        except Exception as e:
            self.logger.error(f"Error checking for blocked access: {str(e)}")
            return {"blocked": False, "error": str(e)}


class FormFieldIdentifier:
    """
    Identifies and analyzes form fields to determine their purpose and appropriate values.
    
    This class:
    - Intelligently identifies different form field types
    - Determines appropriate values for different field types
    - Maps user search intents to appropriate form fields
    - Handles complex form structures including multi-step forms
    """
    
    def __init__(self, domain_intelligence=None):
        """
        Initialize the form field identifier.
        
        Args:
            domain_intelligence: Optional domain intelligence component
        """
        self.logger = logging.getLogger("FormFieldIdentifier")
        
        # Use domain intelligence if provided or import on demand
        if domain_intelligence:
            self.domain_intelligence = domain_intelligence
        else:
            # Lazy import to avoid circular dependency
            from components.domain_intelligence import DomainIntelligence
            self.domain_intelligence = DomainIntelligence()
            
        # Common field type indicators
        self.field_type_indicators = {
            "search": [
                "search", "query", "q", "keyword", "find", "term", "s", "k", "looking for"
            ],
            "location": [
                "location", "city", "state", "zip", "postal", "address", "where", "near"
            ],
            "price": [
                "price", "cost", "budget", "min", "max", "from", "to", "range"
            ],
            "date": [
                "date", "when", "day", "month", "year", "start", "end", "checkin", "checkout"
            ],
            "category": [
                "category", "type", "cat", "department", "section", "group", "class"
            ],
            "bedrooms": [
                "bed", "bedroom", "beds", "br", "bedrooms"
            ],
            "bathrooms": [
                "bath", "bathroom", "baths", "ba", "bathrooms"
            ],
            "property_type": [
                "property", "home", "house", "apartment", "condo", "property type"
            ],
            "sort": [
                "sort", "order", "ordering", "sortby", "sort_by", "orderby"
            ],
            "quantity": [
                "quantity", "count", "number", "amount", "many"
            ],
            "size": [
                "size", "dimension", "length", "width", "height", "area", "sq", "square"
            ]
        }
        
        # Patterns for recognizing field types by format requirements
        self.field_format_patterns = {
            "email": re.compile(r'email|e-mail|e_mail', re.I),
            "phone": re.compile(r'phone|telephone|mobile|cell', re.I),
            "zip": re.compile(r'zip|postal|postcode', re.I),
            "date": re.compile(r'date|day|month|year|calendar', re.I),
            "time": re.compile(r'time|hour|minute', re.I),
            "number": re.compile(r'number|amount|quantity|count', re.I),
            "password": re.compile(r'password|pwd|pass', re.I),
            "username": re.compile(r'username|user|login|account', re.I),
            "url": re.compile(r'url|website|site|link', re.I),
            "file": re.compile(r'file|upload|attachment', re.I)
        }
        
        # Domain-specific field recognition patterns
        self.domain_field_patterns = {
            "real_estate": {
                "bedrooms": re.compile(r'bed|bedroom|br', re.I),
                "bathrooms": re.compile(r'bath|bathroom|ba', re.I),
                "sqft": re.compile(r'sq\s*ft|square\s*feet|size|area', re.I),
                "lot_size": re.compile(r'lot\s*size|land\s*size|land\s*area', re.I),
                "year_built": re.compile(r'year\s*built|built\s*in|construction\s*year', re.I),
                "property_type": re.compile(r'property\s*type|home\s*type|housing\s*type', re.I)
            },
            "ecommerce": {
                "brand": re.compile(r'brand|make|manufacturer', re.I),
                "color": re.compile(r'color|colour|finish', re.I),
                "size": re.compile(r'size|dimension', re.I),
                "price_range": re.compile(r'price\s*range|cost\s*range|budget', re.I),
                "rating": re.compile(r'rating|stars|review', re.I),
                "condition": re.compile(r'condition|state|quality', re.I)
            },
            "travel": {
                "destination": re.compile(r'destination|place|city|country|location', re.I),
                "check_in": re.compile(r'check\s*in|arrival|start\s*date', re.I),
                "check_out": re.compile(r'check\s*out|departure|end\s*date', re.I),
                "guests": re.compile(r'guest|person|people|adult|child', re.I),
                "rooms": re.compile(r'room|suite|accommodation', re.I),
                "airline": re.compile(r'airline|carrier|flight', re.I)
            },
            "job_search": {
                "job_title": re.compile(r'job\s*title|position|role', re.I),
                "company": re.compile(r'company|employer|organization', re.I),
                "location": re.compile(r'location|city|area|remote', re.I),
                "salary": re.compile(r'salary|wage|compensation|pay', re.I),
                "experience": re.compile(r'experience|years|seniority', re.I),
                "job_type": re.compile(r'job\s*type|employment\s*type|full\s*time|part\s*time', re.I)
            }
        }
    
    def identify_form_fields(self, form_data: Dict[str, Any], domain_type: str = "general") -> Dict[str, Any]:
        """
        Identify the purpose and type of each field in a form.
        
        Args:
            form_data: Form data including fields
            domain_type: Type of domain for specialized field identification
            
        Returns:
            Form data with enhanced field type information
        """
        form_fields = form_data.get("fields", [])
        enhanced_fields = []
        
        # Get domain-specific patterns if available
        domain_patterns = self.domain_field_patterns.get(domain_type, {})
        
        # Track field types we've seen to avoid duplicates
        identified_field_types = set()
        
        for field in form_fields:
            # Clone the field and enhance it
            enhanced_field = field.copy()
            
            # Skip hidden and submit fields
            if field.get("type") in ["hidden", "submit", "button", "image", "reset"]:
                enhanced_field["field_type"] = field.get("type")
                enhanced_fields.append(enhanced_field)
                continue
                
            # Extract field attributes for analysis
            field_attributes = ' '.join([
                field.get("name", ""),
                field.get("id", ""),
                field.get("placeholder", ""),
                ' '.join(field.get("class_list", [])),
                field.get("aria-label", ""),
                field.get("title", "")
            ]).lower()
            
            # Default field type based on HTML type
            html_type = field.get("type", "")
            if html_type in ["text", "search", "email", "tel", "url", "number", "date", "time", "file", "password"]:
                enhanced_field["field_type"] = html_type
            elif html_type == "checkbox":
                enhanced_field["field_type"] = "boolean"
            elif html_type == "radio":
                enhanced_field["field_type"] = "single_choice"
            elif field.get("element_type") == "select":
                enhanced_field["field_type"] = "dropdown"
            elif field.get("element_type") == "textarea":
                enhanced_field["field_type"] = "text_area"
            else:
                enhanced_field["field_type"] = "text"  # Default to text
                
            # Try to determine semantic field type
            semantic_type = self._determine_semantic_field_type(
                field_attributes, 
                field.get("possible_values", []),
                domain_patterns,
                identified_field_types
            )
            
            if semantic_type:
                enhanced_field["semantic_type"] = semantic_type
                identified_field_types.add(semantic_type)
                
            # Identify if this is a required field
            enhanced_field["required"] = field.get("required", False) or "required" in field_attributes
            
            # Identify if this is a primary search field
            is_search_field = (semantic_type == "search" or
                              field.get("is_search_field", False) or
                              any(term in field_attributes for term in self.field_type_indicators["search"]))
            
            enhanced_field["is_search_field"] = is_search_field
            
            # Add validation requirements if detected
            enhanced_field["validation"] = self._determine_validation_requirements(
                field_attributes, 
                field.get("type", ""),
                field.get("pattern", ""),
                field.get("min", ""),
                field.get("max", ""),
                field.get("maxlength", "")
            )
            
            # Record possible values for select, radio and checkbox groups
            if "possible_values" not in enhanced_field and enhanced_field["field_type"] in ["dropdown", "single_choice"]:
                enhanced_field["possible_values"] = []
                
            enhanced_fields.append(enhanced_field)
            
        # Check for form groups (logically related fields)
        grouped_fields = self._identify_field_groups(enhanced_fields)
        
        # Update the form data with enhanced fields
        enhanced_form = form_data.copy()
        enhanced_form["fields"] = enhanced_fields
        enhanced_form["field_groups"] = grouped_fields
        
        return enhanced_form
    
    def _determine_semantic_field_type(self, field_attributes: str, 
                                      possible_values: List[Dict[str, str]],
                                      domain_patterns: Dict[str, re.Pattern],
                                      identified_types: set) -> Optional[str]:
        """
        Determine the semantic type of a field based on its attributes and possible values.
        
        Args:
            field_attributes: Concatenated field attributes
            possible_values: List of possible values for select/radio fields
            domain_patterns: Domain-specific field patterns
            identified_types: Set of already identified field types
            
        Returns:
            Semantic field type or None if no clear type identified
        """
        # First check field attributes against the format patterns
        for format_type, pattern in self.field_format_patterns.items():
            if pattern.search(field_attributes):
                return format_type
                
        # Check against domain-specific patterns
        for field_type, pattern in domain_patterns.items():
            if field_type not in identified_types and pattern.search(field_attributes):
                return field_type
                
        # Check against general field type indicators
        for field_type, indicators in self.field_type_indicators.items():
            # Skip if we already found this type of field
            if field_type in identified_types and field_type != "search":  # Allow multiple search fields
                continue
                
            for indicator in indicators:
                if indicator in field_attributes:
                    return field_type
                    
        # For select/radio fields, analyze options to determine type
        if possible_values:
            option_text = ' '.join([
                opt.get("text", "").lower() for opt in possible_values
            ])
            
            # Check if options indicate a specific field type
            for field_type, indicators in self.field_type_indicators.items():
                for indicator in indicators:
                    if indicator in option_text:
                        return field_type
                        
        return None
    
    def _determine_validation_requirements(self, field_attributes: str, 
                                         input_type: str,
                                         pattern: str = "",
                                         min_val: str = "",
                                         max_val: str = "",
                                         maxlength: str = "") -> Dict[str, Any]:
        """
        Determine validation requirements for a field.
        
        Args:
            field_attributes: Concatenated field attributes
            input_type: HTML input type
            pattern: HTML pattern attribute
            min_val: HTML min attribute
            max_val: HTML max attribute
            maxlength: HTML maxlength attribute
            
        Returns:
            Dictionary of validation requirements
        """
        validation = {}
        
        # Determine based on HTML input type
        if input_type == "email":
            validation["format"] = "email"
        elif input_type == "tel":
            validation["format"] = "phone"
        elif input_type == "url":
            validation["format"] = "url"
        elif input_type == "number":
            validation["format"] = "number"
            if min_val:
                validation["min"] = min_val
            if max_val:
                validation["max"] = max_val
        elif input_type == "date":
            validation["format"] = "date"
        elif input_type == "time":
            validation["format"] = "time"
            
        # Pattern validation
        if pattern:
            validation["pattern"] = pattern
            
        # Length validation
        if maxlength:
            validation["maxlength"] = maxlength
            
        # Inferring validation from field attributes
        if "email" in field_attributes:
            validation["format"] = "email"
        elif "phone" in field_attributes or "tel" in field_attributes:
            validation["format"] = "phone"
        elif "zip" in field_attributes or "postal" in field_attributes:
            validation["format"] = "zip"
        elif "url" in field_attributes or "website" in field_attributes:
            validation["format"] = "url"
            
        return validation
    
    def _identify_field_groups(self, fields: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify logical groups of related fields.
        
        Args:
            fields: List of form fields
            
        Returns:
            List of field groups
        """
        groups = []
        
        # Price range group (min and max price)
        price_fields = [f for f in fields if f.get("semantic_type") == "price"]
        if len(price_fields) >= 2:
            min_price = None
            max_price = None
            
            for field in price_fields:
                field_attrs = ' '.join([
                    field.get("name", ""),
                    field.get("id", ""),
                    field.get("placeholder", "")
                ]).lower()
                
                if "min" in field_attrs or "from" in field_attrs:
                    min_price = field
                elif "max" in field_attrs or "to" in field_attrs:
                    max_price = field
            
            if min_price and max_price:
                groups.append({
                    "type": "price_range",
                    "fields": [min_price, max_price]
                })
                
        # Date range group (start and end date)
        date_fields = [f for f in fields if f.get("semantic_type") == "date"]
        if len(date_fields) >= 2:
            start_date = None
            end_date = None
            
            for field in date_fields:
                field_attrs = ' '.join([
                    field.get("name", ""),
                    field.get("id", ""),
                    field.get("placeholder", "")
                ]).lower()
                
                if any(term in field_attrs for term in ["checkin", "start", "from", "arrival"]):
                    start_date = field
                elif any(term in field_attrs for term in ["checkout", "end", "to", "departure"]):
                    end_date = field
            
            if start_date and end_date:
                groups.append({
                    "type": "date_range",
                    "fields": [start_date, end_date]
                })
                
        # Location fields group (city, state, zip)
        location_fields = [f for f in fields if f.get("semantic_type") == "location"]
        if len(location_fields) >= 2:
            groups.append({
                "type": "location_composite",
                "fields": location_fields
            })
            
        return groups


class FormInteraction:
    """
    Main class for intelligent form interaction and submission.
    
    This class:
    - Processes search forms using semantic field understanding
    - Maps search terms to appropriate form fields
    - Handles multi-step form workflows
    - Employs human-like interaction patterns
    - Validates form submission success
    """
    
    def __init__(self, page: Page = None, human_like: bool = True, domain_type: str = "general"):
        """
        Initialize form interaction handler.
        
        Args:
            page: Optional Playwright page object for browser interaction
            human_like: Whether to use human-like interaction patterns
            domain_type: Domain type for specialized form field handling
        """
        self.logger = logging.getLogger("FormInteraction")
        self.page = page
        self.human_like = human_like
        self.domain_type = domain_type
        
        # Initialize component classes
        self.field_identifier = FormFieldIdentifier()
        self.dom_manipulator = DOMManipulator()
        self.ajax_handler = AJAXWaitStrategies()
        self.submission_verifier = SubmissionVerification()
        
        # Initialize human-like interaction if enabled
        if human_like:
            self.human_interaction = HumanLikeInteraction()
        
        # Track form state for multi-step forms
        self.current_step = 0
        self.form_steps = []
        self.form_history = []
        
    async def set_page(self, page: Page):
        """
        Set the Playwright page object for browser interaction.
        
        Args:
            page: Playwright page object
        """
        self.page = page
        
    async def analyze_form(self, form_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a form and its fields to determine proper interaction strategy.
        
        Args:
            form_data: Form data including fields
            
        Returns:
            Enhanced form data with field type information
        """
        # First use field identifier to determine field types and purposes
        enhanced_form = self.field_identifier.identify_form_fields(form_data, self.domain_type)
        
        # Check if this might be a multi-step form
        is_multi_step = self._detect_multi_step_form(enhanced_form)
        enhanced_form["is_multi_step"] = is_multi_step
        
        # If multi-step, attempt to identify the workflow
        if is_multi_step:
            enhanced_form["expected_steps"] = self._identify_form_steps(enhanced_form)
            
        # Analyze form submission mechanism
        enhanced_form["submission_info"] = self._analyze_submission_mechanism(enhanced_form)
        
        return enhanced_form
    
    async def fill_form(self, form_data: Dict[str, Any], search_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fill a form with search data intelligently matching fields to values.
        
        Args:
            form_data: Enhanced form data from analyze_form
            search_data: Search terms and parameters to fill the form with
            
        Returns:
            Dictionary with results of the form filling operation
        """
        if not self.page:
            raise ValueError("Page object must be set before filling forms")
            
        result = {
            "success": False,
            "filled_fields": [],
            "skipped_fields": [],
            "errors": []
        }
        
        # Get form fields
        fields = form_data.get("fields", [])
        
        # Match search data to form fields
        field_mappings = await self._map_search_data_to_fields(search_data, fields)
        
        # Keep track of fields we've filled
        filled_fields = []
        
        # Fill each field according to its type
        for field_name, field_value in field_mappings.items():
            field_info = next((f for f in fields if f.get("name") == field_name), None)
            
            if not field_info:
                continue
                
            # Skip hidden fields for now (they'll be handled during submission)
            if field_info.get("type") == "hidden":
                continue
                
            # Get selector for this field
            selector = self._get_field_selector(field_info)
            
            if not selector:
                result["skipped_fields"].append({"name": field_name, "reason": "No selector found"})
                continue
                
            # Fill field based on its type
            try:
                filled = await self._fill_field(selector, field_info, field_value)
                
                if filled:
                    filled_fields.append({"name": field_name, "value": field_value})
                    result["filled_fields"].append({"name": field_name, "value": field_value})
                else:
                    result["skipped_fields"].append({"name": field_name, "reason": "Fill operation failed"})
                    
            except Exception as e:
                self.logger.error(f"Error filling field {field_name}: {str(e)}")
                result["errors"].append({"name": field_name, "error": str(e)})
                
        # Check if we successfully filled any fields
        result["success"] = len(filled_fields) > 0
        
        # Record the filled form state for multi-step forms
        if result["success"]:
            self.form_history.append({
                "step": self.current_step,
                "filled_fields": filled_fields,
                "timestamp": time.time()
            })
            
        return result
    
    async def submit_form(self, form_data: Dict[str, Any], 
                        expect_navigation: bool = True,
                        wait_for_selector: str = None) -> Dict[str, Any]:
        """
        Submit a form and handle post-submission behavior.
        
        Args:
            form_data: Enhanced form data from analyze_form
            expect_navigation: Whether to expect page navigation after submission
            wait_for_selector: Optional selector to wait for after submission
            
        Returns:
            Dictionary with results of the submission operation
        """
        if not self.page:
            raise ValueError("Page object must be set before submitting forms")
            
        result = {
            "success": False,
            "navigation_occurred": False,
            "errors": []
        }
        
        # Get form selector
        form_selector = form_data.get("selector", "")
        
        # Get submit button if available
        submit_button = self._get_submit_button(form_data)
        
        # Prepare for navigation if expected
        navigation_promise = None
        if expect_navigation:
            navigation_promise = self.page.wait_for_navigation(wait_until="domcontentloaded", timeout=15000)  # Reduced from 30000ms to 15000ms
        
        try:
            # Try to submit the form
            submitted = False
            
            # Method 1: Click submit button if available
            if submit_button:
                selector = submit_button.get("selector")
                
                if selector:
                    # Use human-like click if enabled
                    if self.human_like:
                        await self.human_interaction._natural_click(self.page, selector)
                    else:
                        await self.page.click(selector)
                        
                    submitted = True
            
            # Method 2: Use JavaScript to submit form if no button or click failed
            if not submitted and form_selector:
                submitted = await self.dom_manipulator.trigger_submit(self.page, form_selector)
                
            # Method 3: Press Enter key as last resort
            if not submitted:
                await self.page.keyboard.press("Enter")
                submitted = True
                
            # Wait for navigation if expected
            if expect_navigation and navigation_promise:
                try:
                    await navigation_promise
                    result["navigation_occurred"] = True
                except:
                    self.logger.warning("No navigation occurred after form submission")
            
            # Wait for AJAX updates if no navigation
            if not expect_navigation or not result["navigation_occurred"]:
                await self.ajax_handler.wait_for_network_idle(self.page)
                
                if wait_for_selector:
                    await self.ajax_handler.wait_for_dynamic_content(self.page, wait_for_selector)
                
                # Wait a bit for any animations or transitions
                await self.ajax_handler.wait_for_animations(self.page)
                
            # Verify submission success
            verification_result = await self.submission_verifier.verify_submission(self.page)
            result.update(verification_result)
            
            # Check for blocked access
            block_check = await self.submission_verifier.check_for_blocked_access(self.page)
            result["blocked"] = block_check.get("blocked", False)
            
            # For multi-step forms, check if we need to move to the next step
            if form_data.get("is_multi_step", False):
                next_step_result = await self._handle_multi_step_transition(form_data)
                result.update(next_step_result)
                
        except Exception as e:
            self.logger.error(f"Error submitting form: {str(e)}")
            result["errors"].append(str(e))
            
        return result
    
    async def handle_multi_step_form(self, form_steps: List[Dict[str, Any]], 
                                   search_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a multi-step form submission process.
        
        Args:
            form_steps: List of form data for each step
            search_data: Search terms and parameters for all steps
            
        Returns:
            Dictionary with results of the multi-step form submission
        """
        if not self.page:
            raise ValueError("Page object must be set before handling multi-step forms")
            
        result = {
            "success": False,
            "completed_steps": 0,
            "total_steps": len(form_steps),
            "step_results": []
        }
        
        # Reset step tracking
        self.current_step = 0
        self.form_steps = form_steps
        self.form_history = []
        
        # Process each form step
        for i, step_form in enumerate(form_steps):
            self.current_step = i
            
            self.logger.info(f"Processing form step {i+1} of {len(form_steps)}")
            
            # Fill the current step form
            fill_result = await self.fill_form(step_form, search_data)
            
            if not fill_result.get("success", False):
                self.logger.warning(f"Failed to fill form at step {i+1}")
                result["step_results"].append({
                    "step": i+1,
                    "action": "fill",
                    "success": False,
                    "details": fill_result
                })
                break
                
            # Submit the current step form
            expect_final_navigation = (i == len(form_steps) - 1)
            submit_result = await self.submit_form(
                step_form, 
                expect_navigation=expect_final_navigation
            )
            
            result["step_results"].append({
                "step": i+1,
                "action": "submit",
                "success": submit_result.get("success", False),
                "details": submit_result
            })
            
            # Check if submission was successful
            if not submit_result.get("success", False):
                self.logger.warning(f"Failed to submit form at step {i+1}")
                break
                
            # If blocked, stop processing
            if submit_result.get("blocked", False):
                self.logger.warning("Detected blocked access - stopping form processing")
                result["blocked"] = True
                break
                
            # Update completed steps count
            result["completed_steps"] = i + 1
            
            # For the last step, use its result as the overall result
            if i == len(form_steps) - 1:
                result["success"] = submit_result.get("success", False)
                result["final_result"] = submit_result
                
            # Wait before proceeding to next step
            await asyncio.sleep(random.uniform(1.0, 2.0))
            
        return result
    
    async def _map_search_data_to_fields(self, search_data: Dict[str, Any], 
                                        fields: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Map search data to form fields based on semantic understanding.
        
        Args:
            search_data: Search terms and parameters
            fields: Form fields with enhanced type information
            
        Returns:
            Dictionary mapping field names to values
        """
        field_mappings = {}
        
        # First, look for direct matches between search_data keys and field names
        for field in fields:
            field_name = field.get("name")
            if not field_name:
                continue
                
            # Direct match with field name
            if field_name in search_data:
                field_mappings[field_name] = search_data[field_name]
                continue
                
            # Semantic field type matching
            semantic_type = field.get("semantic_type")
            if semantic_type and semantic_type in search_data:
                field_mappings[field_name] = search_data[semantic_type]
                continue
                
            # Special case: main search query usually goes to search field
            if field.get("is_search_field", False) and "query" in search_data:
                field_mappings[field_name] = search_data["query"]
                continue
                
            # Handle common field names for search
            if field.get("is_search_field", False) and "search_term" in search_data:
                field_mappings[field_name] = search_data["search_term"]
                continue
                
        # For fields that still have no value, try more advanced matching
        for field in fields:
            field_name = field.get("name")
            
            # Skip fields already mapped
            if field_name in field_mappings:
                continue
                
            # Skip hidden and submit fields
            if field.get("type") in ["hidden", "submit", "button", "image", "reset"]:
                continue
                
            # Try to match based on field attributes and search data
            for search_key, search_value in search_data.items():
                # Skip already used search keys
                if search_key in field_mappings.values():
                    continue
                    
                field_attributes = ' '.join([
                    field.get("name", ""),
                    field.get("id", ""),
                    field.get("placeholder", ""),
                    ' '.join(field.get("class_list", [])),
                    field.get("aria-label", ""),
                    field.get("title", "")
                ]).lower()
                
                # Check if search key appears in field attributes
                if search_key.lower() in field_attributes:
                    field_mappings[field_name] = search_value
                    break
                    
        return field_mappings
    
    async def _fill_field(self, selector: str, field_info: Dict[str, Any], value: Any) -> bool:
        """
        Fill a specific field with the provided value.
        
        Args:
            selector: CSS selector for the field
            field_info: Field information with type and validation
            value: Value to fill in the field
            
        Returns:
            True if successful, False otherwise
        """
        field_type = field_info.get("field_type", "text")
        html_type = field_info.get("type", "text")
        
        try:
            # Handle different field types
            if field_type in ["text", "search", "email", "tel", "url", "number"]:
                if self.human_like:
                    return await self.human_interaction.human_type(self.page, selector, str(value))
                else:
                    await self.page.fill(selector, str(value))
                    return True
                    
            elif field_type == "dropdown" or html_type == "select":
                if self.human_like:
                    return await self.human_interaction.human_select(self.page, selector, str(value))
                else:
                    await self.page.select_option(selector, value=str(value))
                    return True
                    
            elif field_type == "single_choice" or html_type == "radio":
                # For radio buttons, we need to find the specific radio with the right value
                radio_selector = f"{selector}[value='{value}']"
                
                try:
                    if self.human_like:
                        await self.human_interaction._natural_click(self.page, radio_selector)
                    else:
                        await self.page.click(radio_selector)
                        
                    return True
                except:
                    # Fallback for when direct value selection fails: try all radios in the group
                    radio_name = field_info.get("name", "")
                    if radio_name:
                        radios = await self.page.query_selector_all(f"input[type='radio'][name='{radio_name}']")
                        
                        for i, radio in enumerate(radios):
                            radio_value = await radio.get_attribute("value")
                            
                            if str(radio_value).lower() == str(value).lower():
                                await radio.click()
                                return True
                        
                    return False
                    
            elif field_type == "boolean" or html_type == "checkbox":
                # Convert value to boolean
                checked = value if isinstance(value, bool) else (str(value).lower() in ["true", "yes", "1", "on"])
                
                if self.human_like:
                    return await self.human_interaction.human_checkbox_click(self.page, selector, checked)
                else:
                    if checked:
                        await self.page.check(selector)
                    else:
                        await self.page.uncheck(selector)
                    return True
                    
            elif field_type == "date" or html_type == "date":
                # Handle date picker
                return await self.dom_manipulator.fix_date_picker(self.page, selector, str(value))
                
            elif field_type == "text_area" or html_type == "textarea":
                if self.human_like:
                    return await self.human_interaction.human_type(self.page, selector, str(value))
                else:
                    await self.page.fill(selector, str(value))
                    return True
                    
            else:
                # For unknown types, try standard fill
                await self.page.fill(selector, str(value))
                return True
                
        except Exception as e:
            self.logger.error(f"Error filling field {selector}: {str(e)}")
            
            # Try direct DOM manipulation as fallback
            try:
                return await self.dom_manipulator.set_input_value(self.page, selector, str(value))
            except:
                return False
    
    def _get_field_selector(self, field_info: Dict[str, Any]) -> Optional[str]:
        """
        Get the best CSS selector for a field.
        
        Args:
            field_info: Field information dictionary
            
        Returns:
            CSS selector string or None if not found
        """
        # Try ID selector first (most reliable)
        field_id = field_info.get("id")
        if field_id:
            return f"#{field_id}"
            
        # Try name selector
        field_name = field_info.get("name")
        field_type = field_info.get("type", "text")
        element_type = field_info.get("element_type", "input")
        
        if field_name and element_type:
            if element_type == "select":
                return f"select[name='{field_name}']"
            else:
                return f"{element_type}[name='{field_name}']"
                
        # If there's a specific selector provided, use it
        if "selector" in field_info:
            return field_info["selector"]
            
        # As a last resort, try to build a selector from available attributes
        if field_name and field_type:
            return f"input[type='{field_type}'][name='{field_name}']"
            
        return None
    
    def _get_submit_button(self, form_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Find the submit button in a form.
        
        Args:
            form_data: Enhanced form data
            
        Returns:
            Submit button info or None if not found
        """
        fields = form_data.get("fields", [])
        
        # Look for explicit submit buttons
        for field in fields:
            if field.get("type") == "submit" or field.get("is_submit", False):
                return field
                
        # Look for buttons with submit-like attributes
        for field in fields:
            if field.get("element_type") == "button":
                button_text = field.get("text", "").lower()
                
                # Check for common submit button text
                if any(term in button_text for term in ["search", "submit", "find", "go", "send"]):
                    return field
                    
        return None
    
    def _detect_multi_step_form(self, form_data: Dict[str, Any]) -> bool:
        """
        Detect if a form is likely part of a multi-step workflow.
        
        Args:
            form_data: Enhanced form data
            
        Returns:
            True if likely a multi-step form, False otherwise
        """
        # Check for explicit multi-step indicators
        form_classes = form_data.get("class_list", [])
        form_id = form_data.get("id", "")
        
        # Check class names and ID for multi-step indicators
        multi_step_indicators = ["step", "wizard", "multi", "stage", "phase"]
        
        for indicator in multi_step_indicators:
            if any(indicator in cls.lower() for cls in form_classes):
                return True
                
            if indicator in form_id.lower():
                return True
                
        # Check if form has very few fields (might be part of a wizard)
        fields = form_data.get("fields", [])
        visible_fields = [f for f in fields if f.get("type") not in ["hidden", "submit", "button"]]
        
        # Forms with only 1-3 visible fields might be part of a multi-step process
        if 1 <= len(visible_fields) <= 3:
            return True
            
        # Check for step indicators in button text
        for field in fields:
            if field.get("element_type") == "button":
                button_text = field.get("text", "").lower()
                
                # Check for "next" or "continue" buttons
                if any(term in button_text for term in ["next", "continue", "proceed", "step"]):
                    return True
                    
        return False
    
    def _identify_form_steps(self, form_data: Dict[str, Any]) -> int:
        """
        Try to identify the number of steps in a multi-step form.
        
        Args:
            form_data: Enhanced form data
            
        Returns:
            Estimated number of steps, or 0 if unknown
        """
        # Look for step indicators in form
        html = form_data.get("html", "")
        
        if not html:
            return 0
            
        # Look for progress indicators (e.g., "Step 2 of 4")
        step_pattern = re.compile(r'step\s+\d+\s+of\s+(\d+)', re.I)
        match = step_pattern.search(html)
        
        if match:
            return int(match.group(1))
            
        # Look for numbered list items or steps
        soup = BeautifulSoup(html, 'lxml')
        
        # Check for progress indicators, step counters, etc.
        progress_elements = soup.select('.progress-step, .step, .wizard-step, .step-indicator')
        
        if progress_elements:
            return len(progress_elements)
            
        # If all else fails, estimate 2 steps
        return 2
    
    def _analyze_submission_mechanism(self, form_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze how a form is likely to be submitted.
        
        Args:
            form_data: Enhanced form data
            
        Returns:
            Dictionary with submission mechanism details
        """
        result = {
            "has_submit_button": False,
            "has_ajax_submission": False,
            "expected_navigation": True,
            "submit_button": None
        }
        
        # Look for submit button
        submit_button = self._get_submit_button(form_data)
        
        if submit_button:
            result["has_submit_button"] = True
            result["submit_button"] = submit_button
            
        # Check for AJAX submission indicators
        form_attributes = ' '.join([
            form_data.get("id", ""),
            ' '.join(form_data.get("class_list", [])),
            form_data.get("action", "")
        ]).lower()
        
        # Check for AJAX form indicators
        ajax_indicators = ["ajax", "xhr", "async", "noreload", "no-reload"]
        
        if any(indicator in form_attributes for indicator in ajax_indicators):
            result["has_ajax_submission"] = True
            result["expected_navigation"] = False
            
        # No submit button often indicates AJAX form
        if not result["has_submit_button"]:
            result["has_ajax_submission"] = True
            
        # Check the form's action attribute
        action = form_data.get("action", "")
        
        if not action or action == "#" or action == "javascript:void(0)":
            result["has_ajax_submission"] = True
            result["expected_navigation"] = False
            
        return result
    
    async def _handle_multi_step_transition(self, form_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle transition between steps in a multi-step form.
        
        Args:
            form_data: Enhanced form data for the current step
            
        Returns:
            Dictionary with multi-step transition results
        """
        result = {
            "is_final_step": False,
            "next_step_available": False,
            "step_transition_success": False
        }
        
        # Detect if we reached the final step
        # Often indicated by submit button text or lack of "next" buttons
        submit_button = self._get_submit_button(form_data)
        
        if submit_button:
            button_text = submit_button.get("text", "").lower()
            
            # Check if button text indicates final step
            final_step_indicators = ["submit", "finish", "complete", "done", "send"]
            result["is_final_step"] = any(indicator in button_text for indicator in final_step_indicators)
            
        # Check for "next" buttons indicating more steps
        fields = form_data.get("fields", [])
        
        for field in fields:
            if field.get("element_type") == "button":
                button_text = field.get("text", "").lower()
                
                # Check for "next" or "continue" buttons
                if any(term in button_text for term in ["next", "continue", "proceed"]):
                    result["next_step_available"] = True
                    
                    # Try to click this button to move to next step
                    selector = self._get_field_selector(field)
                    
                    if selector:
                        try:
                            await self.page.click(selector)
                            result["step_transition_success"] = True
                            self.current_step += 1
                        except Exception as e:
                            self.logger.error(f"Error transitioning to next step: {str(e)}")
                            
                    break
                    
        return result