"""
Form interaction strategies for SmartScrape search automation.

This module extends the form_interaction.py base functionality with specific
interaction strategies including direct DOM manipulation, human-like typing,
and multi-step form handling.
"""

import logging
import random
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
import re

from bs4 import BeautifulSoup, Tag
from playwright.async_api import Page, ElementHandle, TimeoutError

# Import related components
from components.search.form_interaction import FormInteraction, FormFieldIdentifier
from utils.retry_utils import with_exponential_backoff


class HumanLikeInteraction:
    """
    Implements human-like interaction patterns for form filling.
    
    This class:
    - Simulates human typing with realistic speed variations
    - Adds natural pauses between actions
    - Implements realistic mouse movements
    - Handles typos and corrections
    """
    
    def __init__(self, min_delay: float = 0.05, max_delay: float = 0.15):
        """
        Initialize the human-like interaction simulator.
        
        Args:
            min_delay: Minimum delay between keystrokes in seconds
            max_delay: Maximum delay between keystrokes in seconds
        """
        self.logger = logging.getLogger("HumanLikeInteraction")
        self.min_delay = min_delay
        self.max_delay = max_delay
        
        # Configure typing patterns
        self.typing_burst_probability = 0.7  # Probability of typing multiple chars in a burst
        self.max_burst_size = 4  # Maximum characters to type in a burst
        self.typo_probability = 0.01  # Probability of making a typo
        self.correction_delay = 0.2  # Seconds to wait before correcting a typo
        
        # Character clusters for realistic typos (QWERTY keyboard layout)
        self.adjacent_keys = {
            'a': ['q', 'w', 's', 'z'],
            'b': ['v', 'g', 'h', 'n'],
            'c': ['x', 'd', 'f', 'v'],
            'd': ['s', 'e', 'r', 'f', 'c', 'x'],
            'e': ['w', 's', 'd', 'r'],
            'f': ['d', 'r', 't', 'g', 'v', 'c'],
            'g': ['f', 't', 'y', 'h', 'b', 'v'],
            'h': ['g', 'y', 'u', 'j', 'n', 'b'],
            'i': ['u', 'j', 'k', 'o'],
            'j': ['h', 'u', 'i', 'k', 'm', 'n'],
            'k': ['j', 'i', 'o', 'l', 'm'],
            'l': ['k', 'o', 'p', ';'],
            'm': ['n', 'j', 'k', ','],
            'n': ['b', 'h', 'j', 'm'],
            'o': ['i', 'k', 'l', 'p'],
            'p': ['o', 'l', ';', '['],
            'q': ['1', '2', 'w', 'a'],
            'r': ['e', 'd', 'f', 't'],
            's': ['a', 'w', 'e', 'd', 'x', 'z'],
            't': ['r', 'f', 'g', 'y'],
            'u': ['y', 'h', 'j', 'i'],
            'v': ['c', 'f', 'g', 'b'],
            'w': ['q', 'a', 's', 'e'],
            'x': ['z', 's', 'd', 'c'],
            'y': ['t', 'g', 'h', 'u'],
            'z': ['a', 's', 'x']
        }
    
    async def type_text(self, page: Page, selector: str, text: str) -> bool:
        """
        Type text with human-like patterns.
        
        Args:
            page: Page object for browser interaction
            selector: Element selector to type into
            text: Text to type
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # First click on the field
            await page.click(selector)
            
            # Small pause after clicking before typing
            await asyncio.sleep(random.uniform(0.2, 0.5))
            
            # Type the text with human-like patterns
            i = 0
            while i < len(text):
                # Decide if we should type a burst or single character
                if random.random() < self.typing_burst_probability:
                    # Type a burst of characters
                    burst_size = min(random.randint(1, self.max_burst_size), len(text) - i)
                    burst_text = text[i:i+burst_size]
                    
                    # Type the burst
                    await page.type(selector, burst_text, delay=random.uniform(self.min_delay, self.max_delay))
                    i += burst_size
                else:
                    # Type a single character
                    char = text[i]
                    
                    # Decide if we should make a typo
                    if random.random() < self.typo_probability:
                        # Make a typo
                        if char.lower() in self.adjacent_keys:
                            typo_char = random.choice(self.adjacent_keys[char.lower()])
                            # Preserve case
                            if char.isupper():
                                typo_char = typo_char.upper()
                            
                            # Type the typo
                            await page.type(selector, typo_char)
                            
                            # Wait before correcting
                            await asyncio.sleep(self.correction_delay)
                            
                            # Delete the typo
                            await page.press(selector, "Backspace")
                            
                            # Small pause before typing correct character
                            await asyncio.sleep(random.uniform(0.1, 0.2))
                    
                    # Type the correct character
                    await page.type(selector, char, delay=random.uniform(self.min_delay, self.max_delay))
                    i += 1
                
                # Occasional pause while typing
                if random.random() < 0.1:
                    await asyncio.sleep(random.uniform(0.2, 0.7))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in human-like typing: {str(e)}")
            return False
    
    async def click_element(self, page: Page, selector: str) -> bool:
        """
        Click an element with human-like movement.
        
        Args:
            page: Page object for browser interaction
            selector: Element selector to click
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # First move to the element
            element = await page.query_selector(selector)
            if not element:
                self.logger.warning(f"Element not found: {selector}")
                return False
                
            # Get element bounding box
            box = await element.bounding_box()
            if not box:
                self.logger.warning(f"Could not get bounding box for: {selector}")
                return False
                
            # Calculate a random point to click within the element
            x = box["x"] + random.uniform(5, box["width"] - 5)
            y = box["y"] + random.uniform(5, box["height"] - 5)
            
            # Move to the element with a human-like curve
            await page.mouse.move(x, y, steps=random.randint(5, 10))
            
            # Small pause before clicking
            await asyncio.sleep(random.uniform(0.1, 0.3))
            
            # Click the element
            await page.mouse.click(x, y)
            
            # Small pause after clicking
            await asyncio.sleep(random.uniform(0.1, 0.3))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in human-like clicking: {str(e)}")
            return False
    
    async def select_option(self, page: Page, selector: str, value: Union[str, List[str]]) -> bool:
        """
        Select an option from a dropdown with human-like behavior.
        
        Args:
            page: Page object for browser interaction
            selector: Element selector for the dropdown
            value: Value or list of values to select
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # First click on the dropdown
            await self.click_element(page, selector)
            
            # Small pause before selecting
            await asyncio.sleep(random.uniform(0.2, 0.5))
            
            # If multiple values are provided, use an array
            if isinstance(value, list):
                for val in value:
                    await page.select_option(selector, val)
                    await asyncio.sleep(random.uniform(0.1, 0.3))
            else:
                # Select a single value
                await page.select_option(selector, value)
            
            # Small pause after selecting
            await asyncio.sleep(random.uniform(0.1, 0.3))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in human-like option selection: {str(e)}")
            return False
    
    async def fill_form_field(self, page: Page, field_selector: str, value: Any, field_type: str = "text") -> bool:
        """
        Fill a form field with human-like interaction based on field type.
        
        Args:
            page: Page object for browser interaction
            field_selector: Element selector for the field
            value: Value to fill
            field_type: Type of field (text, select, checkbox, radio, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if field_type in ["text", "search", "email", "password", "tel", "url", "number"]:
                # Text-like fields
                return await self.type_text(page, field_selector, str(value))
                
            elif field_type == "select" or field_type == "dropdown":
                # Dropdown fields
                return await self.select_option(page, field_selector, value)
                
            elif field_type in ["checkbox", "radio", "boolean"]:
                # Boolean fields
                if str(value).lower() in ["true", "yes", "on", "1"]:
                    return await self.click_element(page, field_selector)
                return True  # If false, do nothing (assumed unchecked by default)
                
            elif field_type == "date":
                # Date fields - handle both text input and date pickers
                try:
                    # Try typing directly
                    return await self.type_text(page, field_selector, str(value))
                except Exception:
                    # If that fails, try clicking to open date picker and handling it
                    self.logger.info("Direct typing failed for date field, trying date picker")
                    await self.click_element(page, field_selector)
                    # Complex date picker handling would go here
                    # For simplicity, we'll just try to set the value directly as fallback
                    await page.evaluate(f'document.querySelector("{field_selector}").value = "{value}"')
                    return True
                    
            else:
                # Default for unknown types
                self.logger.warning(f"Unknown field type: {field_type}, using default text interaction")
                return await self.type_text(page, field_selector, str(value))
                
        except Exception as e:
            self.logger.error(f"Error filling form field with human-like interaction: {str(e)}")
            return False


class FormInteractionStrategy:
    """
    Base class for form interaction strategies.
    
    This abstract class defines the interface for all form interaction strategies
    and provides common functionality.
    """
    
    def __init__(self, form_interaction: Optional[FormInteraction] = None):
        """
        Initialize the form interaction strategy.
        
        Args:
            form_interaction: Optional form interaction component
        """
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Use form interaction if provided or create a new one
        if form_interaction:
            self.form_interaction = form_interaction
        else:
            from components.search.form_interaction import FormInteraction
            self.form_interaction = FormInteraction()
    
    async def interact_with_form(self, page: Page, form_data: Dict[str, Any], 
                               field_values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interact with a form using this strategy.
        
        Args:
            page: Page object for browser interaction
            form_data: Form data including fields and selectors
            field_values: Values to fill in the form
            
        Returns:
            Result of the form interaction
        """
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement interact_with_form method")


class DirectDomStrategy(FormInteractionStrategy):
    """
    Form interaction strategy using direct DOM manipulation.
    
    This strategy:
    - Uses direct DOM manipulation for fast form filling
    - Optimizes for speed rather than human-like behavior
    - Provides efficient batch filling of form fields
    """
    
    async def interact_with_form(self, page: Page, form_data: Dict[str, Any], 
                               field_values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fill a form using direct DOM manipulation.
        
        Args:
            page: Page object for browser interaction
            form_data: Form data including fields and selectors
            field_values: Values to fill in the form
            
        Returns:
            Result of the form interaction
        """
        self.logger.info("Using direct DOM strategy for form interaction")
        
        # Analyze form to enhance field data if not already done
        if "field_types" not in form_data:
            enhanced_form = await self.form_interaction.analyze_form(form_data)
        else:
            enhanced_form = form_data
            
        # Track success for each field
        field_success = {}
        
        # Process all field values
        for field_id, value in field_values.items():
            # Get field info
            field_info = next((f for f in enhanced_form.get("fields", []) 
                             if f.get("name") == field_id or f.get("id") == field_id), None)
            
            if not field_info:
                self.logger.warning(f"Field not found in form data: {field_id}")
                field_success[field_id] = False
                continue
                
            # Build field selector
            field_selector = self._build_field_selector(field_info, enhanced_form.get("selector", ""))
            
            if not field_selector:
                self.logger.warning(f"Could not build selector for field: {field_id}")
                field_success[field_id] = False
                continue
                
            # Fill the field using direct DOM manipulation
            field_success[field_id] = await self._fill_field_direct(
                page, field_selector, value, field_info.get("field_type", "text")
            )
        
        # Submit the form
        submit_result = await self._submit_form_direct(page, enhanced_form.get("selector", ""))
        
        # Build and return result
        return {
            "success": all(field_success.values()) and submit_result.get("success", False),
            "field_status": field_success,
            "submit_result": submit_result
        }
    
    def _build_field_selector(self, field_info: Dict[str, Any], form_selector: str) -> Optional[str]:
        """
        Build a selector for a field.
        
        Args:
            field_info: Field information
            form_selector: Parent form selector
            
        Returns:
            Field selector or None if not possible
        """
        if field_info.get("id"):
            return f"#{field_info['id']}"
        elif field_info.get("name"):
            if form_selector:
                return f"{form_selector} [name='{field_info['name']}']"
            else:
                return f"[name='{field_info['name']}']"
        elif field_info.get("selector"):
            return field_info["selector"]
        else:
            return None
    
    @with_exponential_backoff(max_attempts=2, min_wait=0.1, max_wait=0.4)
    async def _fill_field_direct(self, page: Page, field_selector: str, value: Any, field_type: str) -> bool:
        """
        Fill a field using direct DOM manipulation.
        
        Args:
            page: Page object for browser interaction
            field_selector: Selector for the field
            value: Value to fill
            field_type: Type of field
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if field_type in ["text", "search", "email", "password", "tel", "url", "number", "textarea"]:
                # Clear field first
                await page.evaluate(f'''
                    (() => {{
                        const el = document.querySelector("{field_selector}");
                        if (el) el.value = "";
                    }})()
                ''')
                
                # Fill directly
                await page.fill(field_selector, str(value))
                return True
                
            elif field_type in ["select", "dropdown"]:
                # Select option
                success = await page.evaluate(f'''
                    (() => {{
                        const select = document.querySelector("{field_selector}");
                        if (!select) return false;
                        
                        // Try to find option by value
                        let option = Array.from(select.options).find(opt => 
                            opt.value === "{value}" || opt.text.toLowerCase().includes("{str(value).lower()}"));
                            
                        if (option) {{
                            select.value = option.value;
                            select.dispatchEvent(new Event('change', {{ bubbles: true }}));
                            return true;
                        }}
                        
                        return false;
                    }})()
                ''')
                
                if not success:
                    # Try using page.select_option as fallback
                    await page.select_option(field_selector, value)
                    
                return True
                
            elif field_type in ["checkbox", "radio", "boolean"]:
                # Set checked state
                checked = str(value).lower() in ["true", "yes", "on", "1"]
                await page.evaluate(f'''
                    (() => {{
                        const el = document.querySelector("{field_selector}");
                        if (el) {{
                            el.checked = {str(checked).lower()};
                            el.dispatchEvent(new Event('change', {{ bubbles: true }}));
                        }}
                    }})()
                ''')
                return True
                
            elif field_type == "date":
                # Set date value
                await page.evaluate(f'''
                    (() => {{
                        const el = document.querySelector("{field_selector}");
                        if (el) {{
                            el.value = "{value}";
                            el.dispatchEvent(new Event('input', {{ bubbles: true }}));
                            el.dispatchEvent(new Event('change', {{ bubbles: true }}));
                        }}
                    }})()
                ''')
                return True
                
            else:
                # Fallback for unknown types
                await page.fill(field_selector, str(value))
                return True
                
        except Exception as e:
            self.logger.error(f"Error filling field with direct DOM: {str(e)}")
            return False
    
    async def _submit_form_direct(self, page: Page, form_selector: str) -> Dict[str, Any]:
        """
        Submit a form using direct DOM manipulation.
        
        Args:
            page: Page object for browser interaction
            form_selector: Selector for the form
            
        Returns:
            Result of the form submission
        """
        try:
            # Try to find the submit button
            submit_button = await page.evaluate(f'''
                (() => {{
                    const form = document.querySelector("{form_selector}");
                    if (!form) return null;
                    
                    // Look for submit button
                    let button = form.querySelector('button[type="submit"], input[type="submit"]');
                    
                    // Fallback to any button-like element
                    if (!button) {{
                        button = form.querySelector('button, input[type="button"], .btn, .button');
                    }}
                    
                    return button ? {{
                        id: button.id || null,
                        selector: button.id ? `#${{button.id}}` : null,
                        exists: true
                    }} : {{ exists: false }};
                }})()
            ''')
            
            if submit_button and submit_button.get("exists"):
                # Try clicking the submit button
                if submit_button.get("selector"):
                    await page.click(submit_button["selector"])
                else:
                    # Fallback: use form.submit()
                    await page.evaluate(f'''
                        (() => {{
                            const form = document.querySelector("{form_selector}");
                            if (form) form.submit();
                        }})()
                    ''')
            else:
                # Fallback: use form.submit()
                await page.evaluate(f'''
                    (() => {{
                        const form = document.querySelector("{form_selector}");
                        if (form) form.submit();
                    }})()
                ''')
            
            # Wait for navigation
            await page.wait_for_load_state("networkidle")
            
            return {"success": True}
            
        except Exception as e:
            self.logger.error(f"Error submitting form with direct DOM: {str(e)}")
            return {"success": False, "error": str(e)}


class HumanLikeStrategy(FormInteractionStrategy):
    """
    Form interaction strategy using human-like behavior.
    
    This strategy:
    - Simulates human-like typing and interaction
    - Adds natural delays and randomness to actions
    - Handles fields based on their types in a human-like manner
    """
    
    def __init__(self, form_interaction: Optional[FormInteraction] = None):
        """
        Initialize the human-like interaction strategy.
        
        Args:
            form_interaction: Optional form interaction component
        """
        super().__init__(form_interaction)
        self.human_interaction = HumanLikeInteraction()
    
    async def interact_with_form(self, page: Page, form_data: Dict[str, Any], 
                               field_values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fill a form using human-like interaction.
        
        Args:
            page: Page object for browser interaction
            form_data: Form data including fields and selectors
            field_values: Values to fill in the form
            
        Returns:
            Result of the form interaction
        """
        self.logger.info("Using human-like strategy for form interaction")
        
        # Analyze form to enhance field data if not already done
        if "field_types" not in form_data:
            enhanced_form = await self.form_interaction.analyze_form(form_data)
        else:
            enhanced_form = form_data
            
        # Track success for each field
        field_success = {}
        
        # Process fields in a human-like order (typically top to bottom)
        field_list = enhanced_form.get("fields", [])
        
        # Sort fields by their likely visual position
        if all("position" in field for field in field_list):
            field_list.sort(key=lambda x: (x.get("position", {}).get("y", 0), x.get("position", {}).get("x", 0)))
        
        # Add a natural pause before starting to fill the form
        await asyncio.sleep(random.uniform(0.5, 1.5))
        
        # Fill fields one by one in a human-like manner
        for field_info in field_list:
            field_id = field_info.get("name") or field_info.get("id")
            
            if not field_id or field_id not in field_values:
                continue
                
            value = field_values[field_id]
            
            # Build field selector
            field_selector = self._build_field_selector(field_info, enhanced_form.get("selector", ""))
            
            if not field_selector:
                self.logger.warning(f"Could not build selector for field: {field_id}")
                field_success[field_id] = False
                continue
                
            # Fill the field using human-like interaction
            field_success[field_id] = await self.human_interaction.fill_form_field(
                page, field_selector, value, field_info.get("field_type", "text")
            )
            
            # Add a natural pause between filling fields
            await asyncio.sleep(random.uniform(0.3, 1.0))
        
        # Natural pause before submitting
        await asyncio.sleep(random.uniform(0.8, 1.5))
        
        # Submit the form
        submit_result = await self._submit_form_humanlike(page, enhanced_form.get("selector", ""))
        
        # Build and return result
        return {
            "success": all(field_success.values()) and submit_result.get("success", False),
            "field_status": field_success,
            "submit_result": submit_result
        }
    
    def _build_field_selector(self, field_info: Dict[str, Any], form_selector: str) -> Optional[str]:
        """
        Build a selector for a field.
        
        Args:
            field_info: Field information
            form_selector: Parent form selector
            
        Returns:
            Field selector or None if not possible
        """
        if field_info.get("id"):
            return f"#{field_info['id']}"
        elif field_info.get("name"):
            if form_selector:
                return f"{form_selector} [name='{field_info['name']}']"
            else:
                return f"[name='{field_info['name']}']"
        elif field_info.get("selector"):
            return field_info["selector"]
        else:
            return None
    
    async def _submit_form_humanlike(self, page: Page, form_selector: str) -> Dict[str, Any]:
        """
        Submit a form using human-like interaction.
        
        Args:
            page: Page object for browser interaction
            form_selector: Selector for the form
            
        Returns:
            Result of the form submission
        """
        try:
            # Try to find the submit button
            submit_button = await page.evaluate(f'''
                (() => {{
                    const form = document.querySelector("{form_selector}");
                    if (!form) return null;
                    
                    // Look for submit button
                    let button = form.querySelector('button[type="submit"], input[type="submit"]');
                    
                    // Fallback to any button-like element
                    if (!button) {{
                        button = form.querySelector('button, input[type="button"], .btn, .button');
                    }}
                    
                    return button ? {{
                        id: button.id || null,
                        selector: button.id ? `#${{button.id}}` : 
                                   (button.className ? `.${button.className.split(' ')[0]}` : null),
                        exists: true
                    }} : {{ exists: false }};
                }})()
            ''')
            
            if submit_button and submit_button.get("exists"):
                # Try clicking the submit button
                if submit_button.get("selector"):
                    await self.human_interaction.click_element(page, submit_button["selector"])
                else:
                    # Fallback: click within the form and press Enter
                    await self.human_interaction.click_element(page, form_selector)
                    await asyncio.sleep(random.uniform(0.3, 0.7))
                    await page.keyboard.press("Enter")
            else:
                # Fallback: click within the form and press Enter
                await self.human_interaction.click_element(page, form_selector)
                await asyncio.sleep(random.uniform(0.3, 0.7))
                await page.keyboard.press("Enter")
            
            # Wait for navigation
            await page.wait_for_load_state("networkidle")
            
            return {"success": True}
            
        except Exception as e:
            self.logger.error(f"Error submitting form with human-like interaction: {str(e)}")
            return {"success": False, "error": str(e)}


class MultiStepFormStrategy(FormInteractionStrategy):
    """
    Strategy for handling multi-step forms.
    
    This strategy:
    - Sequences through multiple form steps
    - Tracks state across form transitions
    - Handles conditional paths in multi-step forms
    - Provides verification of form step progression
    """
    
    def __init__(self, form_interaction: Optional[FormInteraction] = None, 
                use_human_like: bool = True):
        """
        Initialize the multi-step form strategy.
        
        Args:
            form_interaction: Optional form interaction component
            use_human_like: Whether to use human-like interaction
        """
        super().__init__(form_interaction)
        
        # Set up human-like or direct strategies for the steps
        if use_human_like:
            self.step_strategy = HumanLikeStrategy(form_interaction)
        else:
            self.step_strategy = DirectDomStrategy(form_interaction)
            
        # Track form state
        self.current_step = 0
        self.total_steps = 0
        self.steps_completed = []
    
    async def interact_with_form(self, page: Page, form_data: Dict[str, Any], 
                               field_values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a multi-step form.
        
        Args:
            page: Page object for browser interaction
            form_data: Form data including fields and selectors
            field_values: Values to fill in the form
            
        Returns:
            Result of the form interaction
        """
        self.logger.info("Using multi-step strategy for form interaction")
        
        # Reset state
        self.current_step = 0
        self.steps_completed = []
        
        # If form_steps is provided, use those steps
        if "form_steps" in form_data:
            steps = form_data["form_steps"]
            self.total_steps = len(steps)
            
            # Process each step
            for step_data in steps:
                self.current_step += 1
                
                self.logger.info(f"Processing form step {self.current_step}/{self.total_steps}")
                
                # Get field values for this step
                step_field_values = {}
                for field_id in step_data.get("fields", []):
                    if field_id in field_values:
                        step_field_values[field_id] = field_values[field_id]
                
                # Use the step strategy to fill the form
                step_result = await self.step_strategy.interact_with_form(
                    page, step_data, step_field_values
                )
                
                # Check if step was successful
                if not step_result.get("success", False):
                    self.logger.error(f"Failed at step {self.current_step}/{self.total_steps}")
                    return {
                        "success": False,
                        "step_failed": self.current_step,
                        "total_steps": self.total_steps,
                        "steps_completed": self.steps_completed,
                        "error": f"Failed at step {self.current_step}"
                    }
                
                self.steps_completed.append(self.current_step)
                
                # Wait for the next step to load, if this isn't the last step
                if self.current_step < self.total_steps:
                    try:
                        # Wait for navigation and new form to appear
                        await page.wait_for_load_state("networkidle", timeout=10000)
                        
                        # Look for next form
                        next_form_exists = await page.evaluate('''
                            () => {
                                return document.querySelector('form') !== null;
                            }
                        ''')
                        
                        if not next_form_exists:
                            self.logger.warning(f"No form found after step {self.current_step}")
                            # Try to detect if we're already at results page
                            is_results_page = await self._check_if_results_page(page)
                            if is_results_page:
                                self.logger.info("Detected results page, form sequence completed early")
                                break
                    except TimeoutError:
                        self.logger.warning(f"Timeout waiting for next form after step {self.current_step}")
                        # Try to detect if we're already at results page
                        is_results_page = await self._check_if_results_page(page)
                        if is_results_page:
                            self.logger.info("Detected results page, form sequence completed early")
                            break
            
            # Final result
            return {
                "success": True,
                "steps_completed": self.steps_completed,
                "total_steps": self.total_steps
            }
            
        else:
            # No predefined steps, try to handle it dynamically
            return await self._handle_dynamic_multistep_form(page, form_data, field_values)
    
    async def _handle_dynamic_multistep_form(self, page: Page, form_data: Dict[str, Any], 
                                           field_values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a multi-step form without predefined steps.
        
        Args:
            page: Page object for browser interaction
            form_data: Form data including fields and selectors
            field_values: Values to fill in the form
            
        Returns:
            Result of the form interaction
        """
        self.logger.info("Dynamically handling multi-step form")
        
        # Initialize state
        current_form_data = form_data
        remaining_fields = dict(field_values)
        max_steps = 5  # Safety limit
        
        while remaining_fields and self.current_step < max_steps:
            self.current_step += 1
            self.logger.info(f"Processing dynamic form step {self.current_step}")
            
            # Analyze current form
            current_form_data = await self.form_interaction.analyze_form(current_form_data)
            
            # Map fields to current form
            current_step_values = {}
            fields_to_remove = []
            
            for field_id, value in remaining_fields.items():
                # Check if field exists in current form
                field_exists = any(f.get("name") == field_id or f.get("id") == field_id 
                                 for f in current_form_data.get("fields", []))
                
                if field_exists:
                    current_step_values[field_id] = value
                    fields_to_remove.append(field_id)
            
            # Remove used fields
            for field_id in fields_to_remove:
                del remaining_fields[field_id]
            
            # If no fields to fill on this page, just try to proceed
            if not current_step_values:
                self.logger.warning(f"No matching fields found for step {self.current_step}")
                
                # Try to submit the form anyway to move to next step
                await self.step_strategy._submit_form_humanlike(page, current_form_data.get("selector", ""))
            else:
                # Fill and submit this step
                step_result = await self.step_strategy.interact_with_form(
                    page, current_form_data, current_step_values
                )
                
                # Check if step was successful
                if not step_result.get("success", False):
                    self.logger.error(f"Failed at dynamic step {self.current_step}")
                    return {
                        "success": False,
                        "step_failed": self.current_step,
                        "steps_completed": self.steps_completed,
                        "error": f"Failed at dynamic step {self.current_step}"
                    }
            
            self.steps_completed.append(self.current_step)
            
            # Check if we've reached results or another form
            try:
                # Wait for navigation
                await page.wait_for_load_state("networkidle", timeout=10000)
                
                # Check if we're at results page
                is_results_page = await self._check_if_results_page(page)
                if is_results_page:
                    self.logger.info(f"Detected results page after step {self.current_step}")
                    break
                
                # Check for next form
                has_form = await page.evaluate('''
                    () => {
                        return document.querySelector('form') !== null;
                    }
                ''')
                
                if not has_form:
                    self.logger.info(f"No more forms detected after step {self.current_step}")
                    break
                
                # Update form data for next iteration
                html = await page.content()
                soup = BeautifulSoup(html, 'html.parser')
                form_element = soup.select_one('form')
                
                if form_element:
                    current_form_data = self._extract_form_data(form_element)
                else:
                    self.logger.warning("Could not extract form data for next step")
                    break
                
            except TimeoutError:
                self.logger.warning(f"Timeout waiting after dynamic step {self.current_step}")
                break
        
        # Final result
        return {
            "success": True,
            "steps_completed": self.steps_completed,
            "fields_filled": list(field_values.keys())
        }
    
    async def _check_if_results_page(self, page: Page) -> bool:
        """
        Check if we're on a results page.
        
        Args:
            page: Page object for browser interaction
            
        Returns:
            True if this appears to be a results page
        """
        try:
            # Common indicators of results pages
            results_indicators = await page.evaluate('''
                () => {
                    const body = document.body.innerText.toLowerCase();
                    
                    // Check for common result indicators
                    const hasResults = body.includes('result') || 
                                      body.includes('found') || 
                                      document.querySelectorAll('.result, .search-result, .listing, .product').length > 0;
                                      
                    // Check URL for result indicators
                    const url = window.location.href.toLowerCase();
                    const urlHasResults = url.includes('result') || 
                                        url.includes('search') || 
                                        url.includes('listing');
                                        
                    return {
                        hasResultsContent: hasResults,
                        hasResultsUrl: urlHasResults
                    };
                }
            ''')
            
            return results_indicators.get("hasResultsContent", False) or results_indicators.get("hasResultsUrl", False)
            
        except Exception as e:
            self.logger.error(f"Error checking for results page: {str(e)}")
            return False
    
    def _extract_form_data(self, form_element: Tag) -> Dict[str, Any]:
        """
        Extract form data from a BeautifulSoup form element.
        
        Args:
            form_element: BeautifulSoup Tag for a form
            
        Returns:
            Form data dictionary
        """
        form_id = form_element.get('id', '')
        form_class = ' '.join(form_element.get('class', []))
        form_action = form_element.get('action', '')
        form_method = form_element.get('method', 'get').lower()
        
        # Build form selector
        form_selector = ""
        if form_id:
            form_selector = f"form#{form_id}"
        elif form_class:
            form_selector = f"form.{form_class.replace(' ', '.')}"
        else:
            form_selector = "form"
            
        # Extract fields
        fields = []
        
        # Process input fields
        for input_elem in form_element.find_all('input'):
            input_type = input_elem.get('type', 'text')
            
            # Skip hidden fields
            if input_type == 'hidden':
                continue
                
            field_data = {
                "element_type": "input",
                "type": input_type,
                "name": input_elem.get('name', ''),
                "id": input_elem.get('id', ''),
                "required": input_elem.get('required') is not None,
                "placeholder": input_elem.get('placeholder', '')
            }
            
            fields.append(field_data)
            
        # Process select fields
        for select_elem in form_element.find_all('select'):
            field_data = {
                "element_type": "select",
                "name": select_elem.get('name', ''),
                "id": select_elem.get('id', ''),
                "required": select_elem.get('required') is not None
            }
            
            fields.append(field_data)
            
        # Process textarea fields
        for textarea_elem in form_element.find_all('textarea'):
            field_data = {
                "element_type": "textarea",
                "name": textarea_elem.get('name', ''),
                "id": textarea_elem.get('id', ''),
                "required": textarea_elem.get('required') is not None,
                "placeholder": textarea_elem.get('placeholder', '')
            }
            
            fields.append(field_data)
            
        return {
            "id": form_id,
            "class": form_class,
            "action": form_action,
            "method": form_method,
            "selector": form_selector,
            "fields": fields
        }


class SubmissionVerification:
    """
    Verifies successful form submission and result page detection.
    
    This class:
    - Checks for successful form submission
    - Detects result pages vs error pages
    - Extracts submission status information
    - Provides feedback on submission issues
    """
    
    def __init__(self):
        """Initialize the submission verification component."""
        self.logger = logging.getLogger("SubmissionVerification")
    
    async def verify_submission(self, page: Page, form_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Verify if a form submission was successful.
        
        Args:
            page: Page object for browser interaction
            form_data: Optional original form data
            
        Returns:
            Verification result
        """
        # Wait for any navigation to complete
        try:
            await page.wait_for_load_state("networkidle", timeout=10000)
        except Exception as e:
            self.logger.warning(f"Timeout waiting for navigation: {str(e)}")
        
        # Check for success indicators
        success_indicators = await self._check_success_indicators(page)
        
        # Check for error indicators
        error_indicators = await self._check_error_indicators(page)
        
        # Check for validation errors on the form
        validation_errors = await self._check_validation_errors(page, form_data)
        
        # Determine overall success
        is_success = (
            success_indicators.get("has_results", False) or 
            success_indicators.get("url_indicates_results", False)
        ) and not (
            error_indicators.get("has_error_messages", False) or
            error_indicators.get("has_validation_errors", False) or
            validation_errors
        )
        
        # Compile result
        result = {
            "success": is_success,
            "url": page.url,
            "success_indicators": success_indicators,
            "error_indicators": error_indicators
        }
        
        if validation_errors:
            result["validation_errors"] = validation_errors
        
        return result
    
    async def _check_success_indicators(self, page: Page) -> Dict[str, Any]:
        """
        Check for success indicators on the page.
        
        Args:
            page: Page object for browser interaction
            
        Returns:
            Dictionary of success indicators
        """
        try:
            return await page.evaluate('''
                () => {
                    // Check URL for result indicators
                    const url = window.location.href.toLowerCase();
                    const urlIndicatesResults = url.includes('result') || 
                                              url.includes('search') || 
                                              url.includes('listing');
                    
                    // Check for common result containers
                    const resultContainers = document.querySelectorAll(
                        '.result, .search-result, .listing, .product, .item, [data-role="result"]'
                    );
                    
                    // Check content for result indicators
                    const bodyText = document.body.innerText.toLowerCase();
                    const contentIndicatesResults = bodyText.includes('result') || 
                                                  bodyText.includes('found') || 
                                                  bodyText.includes('showing') ||
                                                  bodyText.match(/\\d+\\s+(of|out of|results)/);
                    
                    // Count likely result items
                    let resultCount = resultContainers.length;
                    
                    // If no specific containers found, count potential generic ones
                    if (resultCount === 0) {
                        // Look for repeated patterns of elements that might be results
                        const candidates = [
                            'article', '.card', '.item', '.col', '.grid-item', '.product-item',
                            '.list-item', 'li.row', '.row'
                        ];
                        
                        for (const candidate of candidates) {
                            const items = document.querySelectorAll(candidate);
                            if (items.length > 1) {
                                resultCount = items.length;
                                break;
                            }
                        }
                    }
                    
                    return {
                        url_indicates_results: urlIndicatesResults,
                        has_result_containers: resultContainers.length > 0,
                        content_indicates_results: contentIndicatesResults,
                        has_results: (resultContainers.length > 0 || contentIndicatesResults),
                        estimated_result_count: resultCount
                    };
                }
            ''')
        except Exception as e:
            self.logger.error(f"Error checking success indicators: {str(e)}")
            return {
                "has_results": False,
                "error": str(e)
            }
    
    async def _check_error_indicators(self, page: Page) -> Dict[str, Any]:
        """
        Check for error indicators on the page.
        
        Args:
            page: Page object for browser interaction
            
        Returns:
            Dictionary of error indicators
        """
        try:
            return await page.evaluate('''
                () => {
                    // Check for error messages
                    const errorElements = document.querySelectorAll(
                        '.error, .alert, .alert-danger, .alert-error, .error-message, 
                        [role="alert"], .validation-error'
                    );
                    
                    // Extract error messages
                    const errorMessages = Array.from(errorElements).map(el => el.innerText.trim());
                    
                    // Check content for error indicators
                    const bodyText = document.body.innerText.toLowerCase();
                    const contentIndicatesError = bodyText.includes('error') || 
                                                bodyText.includes('invalid') || 
                                                bodyText.includes('failed') ||
                                                bodyText.includes('not found') ||
                                                bodyText.includes('no results') ||
                                                bodyText.includes('try again');
                    
                    // Check for "no results" indicators specifically
                    const hasNoResults = bodyText.includes('no results') || 
                                       bodyText.includes('0 result') ||
                                       bodyText.includes('nothing found') ||
                                       bodyText.includes('we couldn\\'t find');
                    
                    // Check for HTTP error status in page title
                    const titleHasError = document.title.match(/(404|500|403|401|error)/i) !== null;
                    
                    return {
                        has_error_elements: errorElements.length > 0,
                        error_messages: errorMessages,
                        content_indicates_error: contentIndicatesError,
                        has_no_results: hasNoResults,
                        title_indicates_error: titleHasError,
                        has_error_messages: (errorElements.length > 0 || contentIndicatesError || titleHasError)
                    };
                }
            ''')
        except Exception as e:
            self.logger.error(f"Error checking error indicators: {str(e)}")
            return {
                "has_error_messages": True,
                "error": str(e)
            }
    
    async def _check_validation_errors(self, page: Page, form_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Check for form validation errors.
        
        Args:
            page: Page object for browser interaction
            form_data: Optional original form data
            
        Returns:
            Dictionary of validation errors or None if no errors
        """
        if not form_data:
            return None
            
        try:
            form_selector = form_data.get("selector", "form")
            
            # Check if the form still exists (might indicate submission failure)
            form_exists = await page.evaluate(f'''
                () => {{
                    return document.querySelector("{form_selector}") !== null;
                }}
            ''')
            
            if not form_exists:
                return None
                
            # Check for validation errors on form fields
            validation_report = await page.evaluate(f'''
                () => {{
                    const form = document.querySelector("{form_selector}");
                    if (!form) return null;
                    
                    const invalidFields = [];
                    const fields = form.querySelectorAll('input, select, textarea');
                    
                    for (const field of fields) {{
                        // Check for invalid state
                        if (field.validity && !field.validity.valid) {{
                            invalidFields.push({{
                                name: field.name || null,
                                id: field.id || null,
                                type: field.type || null,
                                validationMessage: field.validationMessage || null
                            }});
                            continue;
                        }}
                        
                        // Check for error messages near the field
                        const fieldId = field.id;
                        if (fieldId) {{
                            // Look for associated error messages by aria-describedby
                            const describedBy = field.getAttribute('aria-describedby');
                            if (describedBy) {{
                                const errorElement = document.getElementById(describedBy);
                                if (errorElement && 
                                    (errorElement.classList.contains('error') || 
                                     errorElement.classList.contains('invalid') ||
                                     errorElement.innerText.toLowerCase().includes('error'))) {{
                                    invalidFields.push({{
                                        name: field.name || null,
                                        id: fieldId,
                                        type: field.type || null,
                                        errorMessage: errorElement.innerText.trim()
                                    }});
                                    continue;
                                }}
                            }}
                            
                            // Look for error element with pattern field-error, field-invalid, etc.
                            const errorSelectors = [
                                `#${{fieldId}}-error`, 
                                `#${{fieldId}}_error`,
                                `#${{fieldId}}-invalid`, 
                                `label[for="${{fieldId}}"] + .error`,
                                `[data-error-for="${{fieldId}}"]`
                            ];
                            
                            for (const selector of errorSelectors) {{
                                const errorElement = document.querySelector(selector);
                                if (errorElement && errorElement.innerText.trim()) {{
                                    invalidFields.push({{
                                        name: field.name || null,
                                        id: fieldId,
                                        type: field.type || null,
                                        errorMessage: errorElement.innerText.trim()
                                    }});
                                    break;
                                }}
                            }}
                        }}
                    }}
                    
                    return invalidFields.length > 0 ? {{ 
                        invalid_fields: invalidFields,
                        form_still_exists: true
                    }} : null;
                }}
            ''')
            
            return validation_report
            
        except Exception as e:
            self.logger.error(f"Error checking validation errors: {str(e)}")
            return None


class FormInteractionHandler:
    """
    Main handler for form interaction that selects and applies appropriate strategies.
    
    This class:
    - Provides a facade for all form interaction strategies
    - Selects the optimal strategy based on form type and requirements
    - Manages complex form interactions through the appropriate strategy
    - Verifies submission results
    """
    
    def __init__(self, form_interaction: Optional[FormInteraction] = None):
        """
        Initialize the form interaction handler.
        
        Args:
            form_interaction: Optional form interaction component
        """
        self.logger = logging.getLogger("FormInteractionHandler")
        
        # Create form interaction component if not provided
        if form_interaction:
            self.form_interaction = form_interaction
        else:
            from components.search.form_interaction import FormInteraction
            self.form_interaction = FormInteraction()
            
        # Create strategies
        self.direct_strategy = DirectDomStrategy(self.form_interaction)
        self.human_strategy = HumanLikeStrategy(self.form_interaction)
        self.multistep_strategy = MultiStepFormStrategy(self.form_interaction)
        
        # Create verification component
        self.verification = SubmissionVerification()
        
        # Set default strategy
        self.default_strategy = "human"  # Can be "direct", "human", or "auto"
    
    def set_default_strategy(self, strategy_type: str):
        """
        Set the default interaction strategy.
        
        Args:
            strategy_type: Strategy type ("direct", "human", or "auto")
        """
        if strategy_type in ["direct", "human", "auto"]:
            self.default_strategy = strategy_type
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    async def interact_with_form(self, page: Page, form_data: Dict[str, Any], 
                               field_values: Dict[str, Any],
                               strategy: str = None,
                               verify_submission: bool = True) -> Dict[str, Any]:
        """
        Interact with a form using the appropriate strategy.
        
        Args:
            page: Page object for browser interaction
            form_data: Form data including fields and selectors
            field_values: Values to fill in the form
            strategy: Strategy to use ("direct", "human", "multistep", or "auto")
            verify_submission: Whether to verify submission success
            
        Returns:
            Result of the form interaction
        """
        strategy = strategy or self.default_strategy
        
        # If strategy is "auto", try to determine the best strategy
        if strategy == "auto":
            strategy = await self._determine_best_strategy(page, form_data)
            self.logger.info(f"Auto-selected strategy: {strategy}")
        
        # Use the appropriate strategy
        if strategy == "direct":
            result = await self.direct_strategy.interact_with_form(page, form_data, field_values)
        elif strategy == "human":
            result = await self.human_strategy.interact_with_form(page, form_data, field_values)
        elif strategy == "multistep":
            result = await self.multistep_strategy.interact_with_form(page, form_data, field_values)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Verify submission if requested
        if verify_submission:
            verification_result = await self.verification.verify_submission(page, form_data)
            result["verification"] = verification_result
            
            # Update success based on verification
            if "success" in verification_result:
                result["success"] = verification_result["success"]
        
        return result
    
    async def _determine_best_strategy(self, page: Page, form_data: Dict[str, Any]) -> str:
        """
        Determine the best strategy for a form.
        
        Args:
            page: Page object for browser interaction
            form_data: Form data including fields and selectors
            
        Returns:
            Name of the best strategy
        """
        # Check if this is explicitly a multi-step form
        if "form_steps" in form_data or form_data.get("is_multistep", False):
            return "multistep"
            
        # Check if the form has anti-bot measures
        has_anti_bot = await self._check_for_anti_bot_measures(page, form_data)
        if has_anti_bot:
            return "human"
            
        # Check form complexity
        form_complexity = self._assess_form_complexity(form_data)
        if form_complexity > 0.7:  # Arbitrary threshold
            return "human"
            
        # Default to direct for simple forms
        return "direct"
    
    async def _check_for_anti_bot_measures(self, page: Page, form_data: Dict[str, Any]) -> bool:
        """
        Check if a form has anti-bot measures.
        
        Args:
            page: Page object for browser interaction
            form_data: Form data including fields and selectors
            
        Returns:
            True if anti-bot measures detected
        """
        try:
            form_selector = form_data.get("selector", "form")
            
            anti_bot_indicators = await page.evaluate(f'''
                () => {{
                    const form = document.querySelector("{form_selector}");
                    if (!form) return {{ has_anti_bot: false }};
                    
                    // Check for captcha
                    const hasCaptcha = document.body.innerText.toLowerCase().includes('captcha') ||
                                     document.querySelector('.captcha, .recaptcha, #captcha, [name*="captcha"]') !== null;
                    
                    // Check for honeypot fields
                    const honeypotSelectors = [
                        'input[name*="bot"], input[id*="bot"]',
                        'input[name*="honey"], input[id*="honey"]',
                        'input[name*="trap"], input[id*="trap"]',
                        'input[style*="display: none"], input[style*="visibility: hidden"]'
                    ];
                    
                    let hasHoneypot = false;
                    for (const selector of honeypotSelectors) {{
                        if (form.querySelector(selector) !== null) {{
                            hasHoneypot = true;
                            break;
                        }}
                    }}
                    
                    // Check for complex event listeners
                    let hasComplexValidation = false;
                    const inputs = form.querySelectorAll('input[type="text"], input[type="email"]');
                    if (inputs.length > 0) {{
                        const sampleInput = inputs[0];
                        const eventsScript = `
                            (() => {{
                                const events = [];
                                const original = EventTarget.prototype.addEventListener;
                                EventTarget.prototype.addEventListener = function(type, listener, options) {{
                                    events.push(type);
                                    return original.apply(this, arguments);
                                }};
                                // Trigger some events
                                const input = document.querySelector("${sampleInput}");
                                if (input) {{
                                    input.focus();
                                    input.blur();
                                    input.dispatchEvent(new Event('input'));
                                    input.dispatchEvent(new Event('change'));
                                }}
                                return events;
                            }})();
                        `;
                        // This is simplified - in reality this would need to be a more complex check
                        hasComplexValidation = form.hasAttribute('onsubmit') ||
                                             document.querySelector('script[src*="validate"]') !== null;
                    }}
                    
                    return {{
                        has_captcha: hasCaptcha,
                        has_honeypot: hasHoneypot,
                        has_complex_validation: hasComplexValidation,
                        has_anti_bot: (hasCaptcha || hasHoneypot || hasComplexValidation)
                    }};
                }}
            ''')
            
            return anti_bot_indicators.get("has_anti_bot", False)
            
        except Exception as e:
            self.logger.error(f"Error checking for anti-bot measures: {str(e)}")
            return True  # Assume anti-bot measures if we can't check
    
    def _assess_form_complexity(self, form_data: Dict[str, Any]) -> float:
        """
        Assess the complexity of a form on a scale of 0 to 1.
        
        Args:
            form_data: Form data including fields and selectors
            
        Returns:
            Complexity score from 0 (simple) to 1 (complex)
        """
        fields = form_data.get("fields", [])
        
        # Base complexity
        base_complexity = 0.0
        
        # More fields = more complex
        field_count = len(fields)
        if field_count > 10:
            base_complexity += 0.4
        elif field_count > 5:
            base_complexity += 0.2
            
        # Check for complex field types
        complex_field_types = [
            "file", "date", "time", "color", "range",
            "multiple", "tags", "autocomplete", "wysiwyg"
        ]
        
        complex_fields = 0
        for field in fields:
            field_type = field.get("field_type", field.get("type", ""))
            if field_type in complex_field_types:
                complex_fields += 1
                
        # More complex fields = more complex form
        if complex_fields > 0:
            base_complexity += min(0.4, complex_fields * 0.1)
            
        # Check if form has any custom attributes or JS validation
        if "onsubmit" in form_data.get("attributes", {}):
            base_complexity += 0.2
            
        return min(1.0, base_complexity)


# Register components in __init__.py
__all__ = [
    'HumanLikeInteraction',
    'FormInteractionStrategy',
    'DirectDomStrategy',
    'HumanLikeStrategy',
    'MultiStepFormStrategy',
    'SubmissionVerification',
    'FormInteractionHandler'
]