"""
Multi-step form handling for SmartScrape search automation.

This module provides capabilities for interacting with complex multi-step search forms
that require navigating through multiple pages or form steps before submission.
"""

import logging
import asyncio
import json
import re
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from urllib.parse import urlparse, parse_qs

from playwright.async_api import Page, ElementHandle, TimeoutError as PlaywrightTimeoutError

from components.search.form_interaction import FormInteraction, HumanLikeInteraction
from utils.extraction_utils import extract_form_data


class MultiStepFormHandler:
    """
    Handles multi-step search forms and wizards.
    
    This class:
    - Detects multi-step form patterns on websites
    - Manages state through form progression
    - Handles navigation between steps
    - Supports both client-side and server-side multi-step forms
    """
    
    def __init__(self):
        """Initialize the multi-step form handler."""
        self.logger = logging.getLogger("MultiStepFormHandler")
        self.form_interaction = FormInteraction()
        self.human_like = HumanLikeInteraction()
        
        # Patterns for detecting multi-step forms
        self.step_indicators = [
            # Visual indicator patterns
            r'step\s+(\d+)\s+of\s+(\d+)',
            r'(\d+)\s*/\s*(\d+)',
            r'page\s+(\d+)\s+of\s+(\d+)',
            
            # Common button patterns
            r'next|continue|proceed|forward',
            r'previous|back|return',
            r'finish|complete|submit'
        ]
        
        # Progress tracking
        self.current_step = 1
        self.total_steps = None
        self.step_history = []
        self.current_form_data = {}
    
    async def detect_multi_step_form(self, page: Page) -> Dict[str, Any]:
        """
        Detect if a page contains a multi-step form.
        
        Args:
            page: Playwright page object
            
        Returns:
            Dictionary with multi-step form information
        """
        self.logger.info("Detecting multi-step form")
        
        # Reset state
        self.current_step = 1
        self.total_steps = None
        self.step_history = []
        self.current_form_data = {}
        
        # Check for step indicators
        step_info = await self._detect_step_indicators(page)
        
        # Check for navigation buttons
        nav_buttons = await self._detect_navigation_buttons(page)
        
        # Look for breadcrumb patterns
        breadcrumbs = await self._detect_breadcrumb_steps(page)
        
        # Analyze URL for step parameters
        url_step_info = self._analyze_url_for_steps(page.url)
        
        # Combine all detection methods
        is_multi_step = (
            step_info.get("is_multi_step", False) or
            len(nav_buttons.get("next_buttons", [])) > 0 or
            breadcrumbs.get("has_breadcrumbs", False) or
            url_step_info.get("has_step_param", False)
        )
        
        # Determine current and total steps
        if step_info.get("is_multi_step", False):
            self.current_step = step_info.get("current_step", 1)
            self.total_steps = step_info.get("total_steps")
        elif breadcrumbs.get("has_breadcrumbs", False):
            self.current_step = breadcrumbs.get("current_step", 1)
            self.total_steps = breadcrumbs.get("total_steps")
        elif url_step_info.get("has_step_param", False):
            self.current_step = url_step_info.get("current_step", 1)
            
        result = {
            "is_multi_step": is_multi_step,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "next_buttons": nav_buttons.get("next_buttons", []),
            "prev_buttons": nav_buttons.get("prev_buttons", []),
            "finish_buttons": nav_buttons.get("finish_buttons", []),
            "step_indicators": step_info.get("indicators", []),
            "breadcrumbs": breadcrumbs.get("breadcrumb_items", []),
            "url_step_info": url_step_info
        }
        
        if is_multi_step:
            self.logger.info(f"Detected multi-step form - Current step: {self.current_step}, Total steps: {self.total_steps or 'unknown'}")
        
        return result
    
    async def _detect_step_indicators(self, page: Page) -> Dict[str, Any]:
        """
        Detect step indicators on the page.
        
        Args:
            page: Playwright page object
            
        Returns:
            Dictionary with step indicator information
        """
        indicators = await page.evaluate('''
            () => {
                const stepIndicators = [];
                
                // Text-based indicators (like "Step 2 of 5")
                const stepTexts = Array.from(document.querySelectorAll('*'))
                    .filter(el => {
                        const text = el.innerText;
                        return text && (
                            /step\\s+\\d+\\s+of\\s+\\d+/i.test(text) ||
                            /\\d+\\s*\\/\\s*\\d+/.test(text) ||
                            /page\\s+\\d+\\s+of\\s+\\d+/i.test(text)
                        );
                    })
                    .map(el => ({
                        text: el.innerText.trim(),
                        rect: el.getBoundingClientRect()
                    }));
                    
                stepIndicators.push(...stepTexts);
                
                // Visual indicators (progress bars, step circles)
                const progressBars = Array.from(document.querySelectorAll(
                    'progress, .progress, .progress-bar, [role="progressbar"]'
                )).map(el => {
                    let value = el.value || el.getAttribute('value') || 0;
                    let max = el.max || el.getAttribute('max') || 100;
                    
                    // If we have aria values
                    if (el.getAttribute('aria-valuenow')) {
                        value = el.getAttribute('aria-valuenow');
                    }
                    
                    if (el.getAttribute('aria-valuemax')) {
                        max = el.getAttribute('aria-valuemax');
                    }
                    
                    // Try to convert to numbers
                    value = Number(value);
                    max = Number(max);
                    
                    return {
                        type: 'progress',
                        value: value,
                        max: max,
                        percent: max > 0 ? (value / max) : 0,
                        rect: el.getBoundingClientRect()
                    };
                });
                
                stepIndicators.push(...progressBars);
                
                // Step circles/dots (common in multi-step UIs)
                const stepLists = Array.from(document.querySelectorAll(
                    '.steps, .step-indicator, .wizard-steps, .stepper, .step-progress'
                ));
                
                for (const stepList of stepLists) {
                    const steps = Array.from(stepList.querySelectorAll('li, .step, .step-item'));
                    
                    if (steps.length > 1) {
                        // Find the active step
                        const activeStep = steps.findIndex(
                            step => step.classList.contains('active') || 
                                  step.classList.contains('current') ||
                                  step.getAttribute('aria-current') === 'step'
                        );
                        
                        stepIndicators.push({
                            type: 'step-list',
                            total_steps: steps.length,
                            current_step: activeStep >= 0 ? activeStep + 1 : null,
                            rect: stepList.getBoundingClientRect()
                        });
                    }
                }
                
                return stepIndicators;
            }
        ''')
        
        # Process the indicators to extract current and total steps
        current_step = 1
        total_steps = None
        is_multi_step = False
        
        for indicator in indicators:
            if indicator.get("type") == "progress":
                is_multi_step = True
                # For progress bars, we estimate the current step
                percent = indicator.get("percent", 0)
                if indicator.get("max"):
                    total_steps = indicator.get("max")
                    current_step = indicator.get("value", 0)
                    
            elif indicator.get("type") == "step-list":
                is_multi_step = True
                total_steps = indicator.get("total_steps")
                if indicator.get("current_step"):
                    current_step = indicator.get("current_step")
                    
            elif indicator.get("text"):
                # Parse text-based indicators like "Step 2 of 5"
                text = indicator.get("text", "")
                
                # Try different regex patterns
                step_match = re.search(r'step\s+(\d+)\s+of\s+(\d+)', text, re.IGNORECASE)
                if not step_match:
                    step_match = re.search(r'(\d+)\s*/\s*(\d+)', text)
                if not step_match:
                    step_match = re.search(r'page\s+(\d+)\s+of\s+(\d+)', text, re.IGNORECASE)
                    
                if step_match:
                    is_multi_step = True
                    current_step = int(step_match.group(1))
                    total_steps = int(step_match.group(2))
        
        return {
            "is_multi_step": is_multi_step,
            "current_step": current_step,
            "total_steps": total_steps,
            "indicators": indicators
        }
    
    async def _detect_navigation_buttons(self, page: Page) -> Dict[str, Any]:
        """
        Detect navigation buttons for multi-step forms.
        
        Args:
            page: Playwright page object
            
        Returns:
            Dictionary with navigation button information
        """
        nav_buttons = await page.evaluate('''
            () => {
                const result = {
                    next_buttons: [],
                    prev_buttons: [],
                    finish_buttons: []
                };
                
                // Common button selectors
                const buttonElements = Array.from(document.querySelectorAll(
                    'button, input[type="button"], input[type="submit"], a.button, .btn, [role="button"]'
                ));
                
                for (const button of buttonElements) {
                    // Skip hidden buttons
                    if (button.offsetParent === null ||
                        window.getComputedStyle(button).display === 'none' ||
                        window.getComputedStyle(button).visibility === 'hidden') {
                        continue;
                    }
                    
                    const buttonText = (button.innerText || button.value || '').toLowerCase();
                    const buttonId = (button.id || '').toLowerCase();
                    const buttonName = (button.name || '').toLowerCase();
                    const buttonClass = (button.className || '').toLowerCase();
                    
                    // Determine button type by content
                    let buttonInfo = {
                        text: button.innerText || button.value,
                        id: button.id,
                        name: button.name,
                        classes: button.className,
                        tag: button.tagName.toLowerCase(),
                        type: button.type,
                        is_disabled: button.disabled || button.classList.contains('disabled') || button.hasAttribute('aria-disabled')
                    };
                    
                    // Try to get a selector
                    if (button.id) {
                        buttonInfo.selector = `#${button.id}`;
                    } else if (button.name) {
                        buttonInfo.selector = `${button.tagName.toLowerCase()}[name="${button.name}"]`;
                    } else {
                        // Generate a more complex selector
                        buttonInfo.selector = button.tagName.toLowerCase();
                        
                        // Add classes if available
                        if (button.className) {
                            const classes = button.className.split(' ')
                                .filter(c => c.trim())
                                .map(c => '.' + c.trim())
                                .join('');
                            if (classes) {
                                buttonInfo.selector += classes;
                            }
                        }
                        
                        // Add index relative to parent
                        const parent = button.parentElement;
                        if (parent) {
                            const siblings = Array.from(parent.children).filter(
                                el => el.tagName === button.tagName
                            );
                            const index = siblings.indexOf(button);
                            if (index > 0) {
                                buttonInfo.selector += `:nth-of-type(${index + 1})`;
                            }
                        }
                    }
                    
                    // Add position information
                    const rect = button.getBoundingClientRect();
                    buttonInfo.position = {
                        x: rect.left,
                        y: rect.top,
                        width: rect.width,
                        height: rect.height
                    };
                    
                    // Next button detection
                    if (/next|continue|proceed|forward/i.test(buttonText) ||
                        /next|continue|proceed|forward/i.test(buttonId) ||
                        /next|continue|proceed|forward/i.test(buttonName) ||
                        /next|continue|proceed|forward/i.test(buttonClass) ||
                        button.getAttribute('aria-label')?.toLowerCase().includes('next')) {
                        result.next_buttons.push(buttonInfo);
                    }
                    
                    // Previous button detection
                    else if (/prev|previous|back|return/i.test(buttonText) ||
                             /prev|previous|back|return/i.test(buttonId) ||
                             /prev|previous|back|return/i.test(buttonName) ||
                             /prev|previous|back|return/i.test(buttonClass) ||
                             button.getAttribute('aria-label')?.toLowerCase().includes('previous')) {
                        result.prev_buttons.push(buttonInfo);
                    }
                    
                    // Finish button detection
                    else if (/finish|complete|submit|search/i.test(buttonText) ||
                             /finish|complete|submit|search/i.test(buttonId) ||
                             /finish|complete|submit|search/i.test(buttonName) ||
                             /finish|complete|submit|search/i.test(buttonClass) ||
                             button.getAttribute('aria-label')?.toLowerCase().includes('finish') ||
                             (button.type === 'submit' && button.form)) {
                        result.finish_buttons.push(buttonInfo);
                    }
                }
                
                return result;
            }
        ''')
        
        return nav_buttons
    
    async def _detect_breadcrumb_steps(self, page: Page) -> Dict[str, Any]:
        """
        Detect breadcrumb-style step indicators.
        
        Args:
            page: Playwright page object
            
        Returns:
            Dictionary with breadcrumb information
        """
        breadcrumbs = await page.evaluate('''
            () => {
                // Look for breadcrumb containers
                const breadcrumbContainers = Array.from(document.querySelectorAll(
                    '.breadcrumb, .breadcrumbs, .wizard-progress, [role="navigation"][aria-label="breadcrumb"], nav ol'
                ));
                
                if (breadcrumbContainers.length === 0) {
                    return { has_breadcrumbs: false };
                }
                
                // Find the most likely breadcrumb container
                let bestContainer = null;
                let maxItems = 0;
                
                for (const container of breadcrumbContainers) {
                    const items = container.querySelectorAll('li, .breadcrumb-item, .step');
                    
                    if (items.length > maxItems) {
                        maxItems = items.length;
                        bestContainer = container;
                    }
                }
                
                if (!bestContainer || maxItems < 2) {
                    return { has_breadcrumbs: false };
                }
                
                // Extract the breadcrumb items
                const items = Array.from(bestContainer.querySelectorAll('li, .breadcrumb-item, .step'));
                const itemData = items.map((item, index) => {
                    const isActive = item.classList.contains('active') || 
                                    item.getAttribute('aria-current') === 'page' ||
                                    item.getAttribute('aria-current') === 'step' ||
                                    item.classList.contains('current');
                                    
                    return {
                        text: item.innerText.trim(),
                        index: index + 1,
                        is_active: isActive
                    };
                });
                
                // Determine the current step
                const activeItem = itemData.find(item => item.is_active);
                const currentStep = activeItem ? activeItem.index : itemData.length;
                
                return {
                    has_breadcrumbs: true,
                    breadcrumb_items: itemData,
                    current_step: currentStep,
                    total_steps: itemData.length
                };
            }
        ''')
        
        return breadcrumbs
    
    def _analyze_url_for_steps(self, url: str) -> Dict[str, Any]:
        """
        Analyze URL for step parameters.
        
        Args:
            url: Current page URL
            
        Returns:
            Dictionary with URL step information
        """
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        
        # Check for common step parameters
        step_param_names = ['step', 'page', 'wizard', 'stage']
        
        for param in step_param_names:
            if param in query_params:
                try:
                    step_value = int(query_params[param][0])
                    return {
                        "has_step_param": True,
                        "param_name": param,
                        "current_step": step_value
                    }
                except (ValueError, IndexError):
                    pass
        
        # Check if the path contains step information
        path_parts = parsed_url.path.strip('/').split('/')
        
        for i, part in enumerate(path_parts):
            if part.lower() in ['step', 'page', 'wizard', 'stage'] and i + 1 < len(path_parts):
                try:
                    step_value = int(path_parts[i + 1])
                    return {
                        "has_step_param": True,
                        "param_type": "path",
                        "current_step": step_value
                    }
                except ValueError:
                    pass
            
            # Check for step-N pattern
            step_match = re.match(r'^(step|page|wizard|stage)-(\d+)$', part.lower())
            if step_match:
                try:
                    step_value = int(step_match.group(2))
                    return {
                        "has_step_param": True,
                        "param_type": "path-pattern",
                        "current_step": step_value
                    }
                except ValueError:
                    pass
        
        return {"has_step_param": False}
    
    async def navigate_to_next_step(self, page: Page, form_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Navigate to the next step in a multi-step form.
        
        Args:
            page: Playwright page object
            form_data: Data to fill in the current step's form
            
        Returns:
            Dictionary with navigation result
        """
        self.logger.info(f"Navigating from step {self.current_step} to next step")
        
        # Save the current URL for comparison
        start_url = page.url
        
        # Fill the form with provided data
        if form_data:
            fill_result = await self.form_interaction.fill_form(page, form_data)
            
            if not fill_result.get("success", False):
                return {
                    "success": False,
                    "reason": f"Failed to fill form: {fill_result.get('reason')}",
                    "step_changed": False,
                    "from_step": self.current_step,
                    "to_step": self.current_step
                }
                
            # Add this data to our form data collection
            self.current_form_data.update(form_data)
        
        # Detect navigation buttons
        nav_info = await self._detect_navigation_buttons(page)
        next_buttons = nav_info.get("next_buttons", [])
        
        if not next_buttons:
            return {
                "success": False,
                "reason": "No next button found",
                "step_changed": False,
                "from_step": self.current_step,
                "to_step": self.current_step
            }
        
        # Find the best next button (first non-disabled one)
        next_button = next((button for button in next_buttons if not button.get("is_disabled", False)), 
                          next_buttons[0])
        
        # Save current step info
        self.step_history.append({
            "step": self.current_step,
            "url": page.url,
            "form_data": dict(self.current_form_data)
        })
        
        # Click the next button
        try:
            selector = next_button.get("selector")
            
            # Scroll to the button
            await page.evaluate(f'document.querySelector("{selector}").scrollIntoView({{ behavior: "smooth", block: "center" }})')
            await asyncio.sleep(0.5)
            
            # Click with human-like behavior
            await self.human_like.click_element(page, selector)
            
            # Wait for navigation or DOM changes
            changed = await self._wait_for_step_change(page, start_url)
            
            if changed:
                # Update current step
                prev_step = self.current_step
                self.current_step += 1
                
                # Check for confirmation
                step_info = await self._detect_step_indicators(page)
                if step_info.get("is_multi_step", False) and step_info.get("current_step"):
                    self.current_step = step_info.get("current_step")
                    
                if step_info.get("total_steps"):
                    self.total_steps = step_info.get("total_steps")
                
                return {
                    "success": True,
                    "step_changed": True,
                    "from_step": prev_step,
                    "to_step": self.current_step,
                    "total_steps": self.total_steps
                }
            else:
                # Check if any error messages appeared
                error_messages = await self._detect_form_errors(page)
                
                if error_messages:
                    return {
                        "success": False,
                        "reason": f"Form errors: {', '.join(error_messages)}",
                        "step_changed": False,
                        "from_step": self.current_step,
                        "to_step": self.current_step,
                        "errors": error_messages
                    }
                
                return {
                    "success": False,
                    "reason": "Clicked next but no navigation or DOM change detected",
                    "step_changed": False,
                    "from_step": self.current_step,
                    "to_step": self.current_step
                }
                
        except Exception as e:
            self.logger.error(f"Error navigating to next step: {str(e)}")
            return {
                "success": False,
                "reason": f"Error: {str(e)}",
                "step_changed": False,
                "from_step": self.current_step,
                "to_step": self.current_step
            }
    
    async def navigate_to_previous_step(self, page: Page) -> Dict[str, Any]:
        """
        Navigate to the previous step in a multi-step form.
        
        Args:
            page: Playwright page object
            
        Returns:
            Dictionary with navigation result
        """
        self.logger.info(f"Navigating from step {self.current_step} to previous step")
        
        if self.current_step <= 1:
            return {
                "success": False,
                "reason": "Already at first step",
                "step_changed": False,
                "from_step": self.current_step,
                "to_step": self.current_step
            }
        
        # Save the current URL for comparison
        start_url = page.url
        
        # Detect navigation buttons
        nav_info = await self._detect_navigation_buttons(page)
        prev_buttons = nav_info.get("prev_buttons", [])
        
        if not prev_buttons:
            return {
                "success": False,
                "reason": "No previous button found",
                "step_changed": False,
                "from_step": self.current_step,
                "to_step": self.current_step
            }
        
        # Use the first previous button
        prev_button = prev_buttons[0]
        
        try:
            selector = prev_button.get("selector")
            
            # Scroll to the button
            await page.evaluate(f'document.querySelector("{selector}").scrollIntoView({{ behavior: "smooth", block: "center" }})')
            await asyncio.sleep(0.5)
            
            # Click with human-like behavior
            await self.human_like.click_element(page, selector)
            
            # Wait for navigation or DOM changes
            changed = await self._wait_for_step_change(page, start_url)
            
            if changed:
                # Update current step
                prev_step = self.current_step
                self.current_step -= 1
                
                # Check for confirmation
                step_info = await self._detect_step_indicators(page)
                if step_info.get("is_multi_step", False) and step_info.get("current_step"):
                    self.current_step = step_info.get("current_step")
                
                # Get previously used form data for this step
                prev_step_data = None
                for step_data in reversed(self.step_history):
                    if step_data.get("step") == self.current_step:
                        prev_step_data = step_data.get("form_data")
                        break
                
                return {
                    "success": True,
                    "step_changed": True,
                    "from_step": prev_step,
                    "to_step": self.current_step,
                    "previous_data": prev_step_data
                }
            else:
                return {
                    "success": False,
                    "reason": "Clicked previous but no navigation or DOM change detected",
                    "step_changed": False,
                    "from_step": self.current_step,
                    "to_step": self.current_step
                }
                
        except Exception as e:
            self.logger.error(f"Error navigating to previous step: {str(e)}")
            return {
                "success": False,
                "reason": f"Error: {str(e)}",
                "step_changed": False,
                "from_step": self.current_step,
                "to_step": self.current_step
            }
    
    async def submit_final_step(self, page: Page, 
                              form_data: Dict[str, Any] = None,
                              verify_success: bool = True) -> Dict[str, Any]:
        """
        Submit the final step of a multi-step form.
        
        Args:
            page: Playwright page object
            form_data: Data to fill in the final step's form
            verify_success: Whether to verify successful submission
            
        Returns:
            Dictionary with submission result
        """
        self.logger.info(f"Submitting final step (step {self.current_step})")
        
        # Save the current URL for comparison
        start_url = page.url
        
        # Fill the form with provided data
        if form_data:
            fill_result = await self.form_interaction.fill_form(page, form_data)
            
            if not fill_result.get("success", False):
                return {
                    "success": False,
                    "reason": f"Failed to fill form: {fill_result.get('reason')}",
                    "submitted": False
                }
                
            # Add this data to our form data collection
            self.current_form_data.update(form_data)
        
        # Detect submit/finish buttons
        nav_info = await self._detect_navigation_buttons(page)
        finish_buttons = nav_info.get("finish_buttons", [])
        
        # If no finish buttons, try next buttons as a fallback
        if not finish_buttons:
            finish_buttons = nav_info.get("next_buttons", [])
            
        if not finish_buttons:
            return {
                "success": False,
                "reason": "No submit button found",
                "submitted": False
            }
        
        # Find the best finish button (first non-disabled one)
        finish_button = next((button for button in finish_buttons if not button.get("is_disabled", False)), 
                            finish_buttons[0])
        
        # Click the finish button
        try:
            selector = finish_button.get("selector")
            
            # Scroll to the button
            await page.evaluate(f'document.querySelector("{selector}").scrollIntoView({{ behavior: "smooth", block: "center" }})')
            await asyncio.sleep(0.5)
            
            # Click with human-like behavior
            await self.human_like.click_element(page, selector)
            
            # Wait for navigation or DOM changes
            changed = await self._wait_for_step_change(page, start_url, timeout=10000)
            
            if not changed:
                # Check if any error messages appeared
                error_messages = await self._detect_form_errors(page)
                
                if error_messages:
                    return {
                        "success": False,
                        "reason": f"Form errors: {', '.join(error_messages)}",
                        "submitted": False,
                        "errors": error_messages
                    }
                    
                # If no errors but no change, try clicking again
                await self.human_like.click_element(page, selector)
                changed = await self._wait_for_step_change(page, start_url, timeout=10000)
                
                if not changed:
                    return {
                        "success": False,
                        "reason": "Clicked submit but no navigation or DOM change detected",
                        "submitted": False
                    }
            
            # Verify successful submission if requested
            if verify_success:
                success_verification = await self._verify_submission_success(page)
                
                if not success_verification.get("success", False):
                    return {
                        "success": False,
                        "reason": success_verification.get("reason", "Unknown verification failure"),
                        "submitted": True,
                        "verified": False
                    }
                    
                return {
                    "success": True,
                    "submitted": True,
                    "verified": True,
                    "verification_method": success_verification.get("method"),
                    "all_form_data": self.current_form_data
                }
            else:
                return {
                    "success": True,
                    "submitted": True,
                    "verified": False,
                    "all_form_data": self.current_form_data
                }
                
        except Exception as e:
            self.logger.error(f"Error submitting final step: {str(e)}")
            return {
                "success": False,
                "reason": f"Error: {str(e)}",
                "submitted": False
            }
    
    async def _wait_for_step_change(self, page: Page, start_url: str, 
                                   timeout: int = 5000) -> bool:
        """
        Wait for a step change to occur after clicking a navigation button.
        
        Args:
            page: Playwright page object
            start_url: URL before the navigation attempt
            timeout: Maximum time to wait in milliseconds
            
        Returns:
            True if a step change was detected, False otherwise
        """
        try:
            # First approach: Wait for navigation (URL change)
            nav_promise = page.wait_for_navigation(timeout=timeout, wait_until="domcontentloaded")
            
            try:
                await nav_promise
                # If we got here, navigation occurred
                return True
            except PlaywrightTimeoutError:
                # No navigation, check for other changes
                pass
            
            # Second approach: Check for URL change
            current_url = page.url
            if current_url != start_url:
                return True
            
            # Third approach: Check for DOM changes that might indicate step change
            # Wait a moment for any AJAX/JS changes
            await asyncio.sleep(2)
            
            # Check if step indicators have changed
            step_info = await self._detect_step_indicators(page)
            if step_info.get("is_multi_step", False) and step_info.get("current_step", 0) != self.current_step:
                return True
                
            # Check for form changes by looking at visible form elements
            form_elements_before = await self._count_visible_form_elements(page)
            form_elements_after = await self._count_visible_form_elements(page)
            
            if abs(form_elements_after - form_elements_before) > 2:  # More than a couple elements changed
                return True
                
            # No detectable change
            return False
            
        except Exception as e:
            self.logger.warning(f"Error while waiting for step change: {str(e)}")
            return False
    
    async def _count_visible_form_elements(self, page: Page) -> int:
        """
        Count the number of visible form elements on the page.
        
        Args:
            page: Playwright page object
            
        Returns:
            Number of visible form elements
        """
        return await page.evaluate('''
            () => {
                const elements = document.querySelectorAll('input, select, textarea, button');
                return Array.from(elements).filter(el => {
                    const style = window.getComputedStyle(el);
                    return style.display !== 'none' && style.visibility !== 'hidden' && el.offsetParent !== null;
                }).length;
            }
        ''')
    
    async def _detect_form_errors(self, page: Page) -> List[str]:
        """
        Detect error messages on a form after submission attempt.
        
        Args:
            page: Playwright page object
            
        Returns:
            List of error messages
        """
        error_messages = await page.evaluate('''
            () => {
                const errors = [];
                
                // Common error selectors
                const errorSelectors = [
                    '.error',
                    '.errors',
                    '.form-error',
                    '.validation-error',
                    '.alert-danger',
                    '.alert-error',
                    '[role="alert"]',
                    'input:invalid',
                    '.invalid-feedback:not([style*="display: none"])',
                    '.is-invalid + .invalid-feedback',
                    'label.error',
                    '.field-validation-error'
                ];
                
                // Check each selector
                for (const selector of errorSelectors) {
                    const elements = document.querySelectorAll(selector);
                    
                    for (const el of elements) {
                        // Skip hidden elements
                        if (el.offsetParent === null ||
                            window.getComputedStyle(el).display === 'none' ||
                            window.getComputedStyle(el).visibility === 'hidden') {
                            continue;
                        }
                        
                        const text = el.innerText || el.textContent;
                        if (text && text.trim()) {
                            errors.push(text.trim());
                        }
                    }
                }
                
                // Look for aria-invalid attributes
                const invalidFields = document.querySelectorAll('[aria-invalid="true"]');
                for (const field of invalidFields) {
                    const errorId = field.getAttribute('aria-errormessage') || field.getAttribute('aria-describedby');
                    
                    if (errorId) {
                        const errorEl = document.getElementById(errorId);
                        if (errorEl) {
                            const text = errorEl.innerText || errorEl.textContent;
                            if (text && text.trim()) {
                                errors.push(text.trim());
                            }
                        }
                    }
                    
                    // Check for error in label
                    const label = document.querySelector(`label[for="${field.id}"]`);
                    if (label) {
                        const text = label.innerText || label.textContent;
                        if (text && text.trim()) {
                            errors.push(text.trim());
                        }
                    }
                }
                
                return errors;
            }
        ''')
        
        return error_messages
    
    async def _verify_submission_success(self, page: Page) -> Dict[str, Any]:
        """
        Verify if a form submission was successful.
        
        Args:
            page: Playwright page object
            
        Returns:
            Dictionary with verification result
        """
        # Check URL for success indicators
        current_url = page.url
        
        # Common success URL patterns
        success_url_patterns = [
            r'success',
            r'thank[-_]?you',
            r'confirmation',
            r'complete',
            r'receipt',
            r'results?',
            r'summary'
        ]
        
        for pattern in success_url_patterns:
            if re.search(pattern, current_url, re.IGNORECASE):
                return {
                    "success": True,
                    "method": "url_pattern",
                    "match": pattern
                }
        
        # Check for success messages on the page
        success_messages = await page.evaluate('''
            () => {
                const successSelectors = [
                    '.success',
                    '.alert-success',
                    '.confirmation',
                    '.thank-you',
                    '[role="status"]',
                    '.status-message'
                ];
                
                // Check for presence of success elements
                for (const selector of successSelectors) {
                    const elements = document.querySelectorAll(selector);
                    
                    for (const el of elements) {
                        // Skip hidden elements
                        if (el.offsetParent === null ||
                            window.getComputedStyle(el).display === 'none' ||
                            window.getComputedStyle(el).visibility === 'hidden') {
                            continue;
                        }
                        
                        const text = el.innerText || el.textContent;
                        if (text && text.trim()) {
                            return [text.trim()];
                        }
                    }
                }
                
                // Look for success keywords in visible text
                const bodyText = document.body.innerText;
                
                const successKeywords = [
                    "success",
                    "thank you",
                    "confirmation",
                    "confirmed",
                    "submitted",
                    "complete",
                    "completed",
                    "received"
                ];
                
                for (const keyword of successKeywords) {
                    if (bodyText.toLowerCase().includes(keyword.toLowerCase())) {
                        // Extract the sentence containing the keyword
                        const sentences = bodyText.split(/[.!?]+/);
                        for (const sentence of sentences) {
                            if (sentence.toLowerCase().includes(keyword.toLowerCase())) {
                                return [sentence.trim()];
                            }
                        }
                    }
                }
                
                return [];
            }
        ''')
        
        if success_messages and len(success_messages) > 0:
            return {
                "success": True,
                "method": "success_message",
                "messages": success_messages
            }
            
        # Check if we're seeing search results
        # This is common for search forms where success = showing results
        search_results = await page.evaluate('''
            () => {
                // Common result container selectors
                const resultSelectors = [
                    '.results',
                    '.search-results',
                    '#results',
                    '[role="main"] > div',
                    '.product-list',
                    '.product-grid',
                    '.listing',
                    '.items'
                ];
                
                for (const selector of resultSelectors) {
                    const elements = document.querySelectorAll(selector);
                    
                    if (elements.length > 0) {
                        // Check if this contains multiple children that look like results
                        const firstEl = elements[0];
                        const children = firstEl.children;
                        
                        if (children.length >= 3) {  // At least a few results
                            return {
                                found: true,
                                count: children.length,
                                selector: selector
                            };
                        }
                    }
                }
                
                // Check for result count indicators
                const countRegex = /(\d+)\s+(results|items|products|listings|properties|matches)/i;
                const bodyText = document.body.innerText;
                const countMatch = bodyText.match(countRegex);
                
                if (countMatch) {
                    return {
                        found: true,
                        count: parseInt(countMatch[1], 10),
                        text: countMatch[0]
                    };
                }
                
                return { found: false };
            }
        ''')
        
        if search_results and search_results.get("found", False):
            return {
                "success": True,
                "method": "search_results",
                "result_info": search_results
            }
            
        # Check if there are still form elements
        # If there are very few, we might have reached a confirmation/success page
        form_elements_count = await self._count_visible_form_elements(page)
        step_info = await self._detect_step_indicators(page)
        
        if (form_elements_count < 3 and not step_info.get("is_multi_step", False)) or \
           (step_info.get("is_multi_step", False) and step_info.get("current_step", 0) > self.current_step):
            return {
                "success": True,
                "method": "form_transition",
                "form_elements": form_elements_count,
                "step_info": step_info
            }
        
        # If all else fails, check if there are no visible errors
        error_messages = await self._detect_form_errors(page)
        
        if not error_messages:
            return {
                "success": True,
                "method": "no_errors",
                "confidence": "low"
            }
            
        # Could not verify success
        return {
            "success": False,
            "reason": "Could not verify successful submission",
            "errors": error_messages if error_messages else []
        }
    
    async def process_multi_step_form(self, page: Page, 
                                    search_data: Dict[str, Any],
                                    step_data_map: Dict[int, Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a multi-step form from start to finish.
        
        Args:
            page: Playwright page object
            search_data: Complete search data
            step_data_map: Mapping of steps to specific data fields
            
        Returns:
            Dictionary with processing result
        """
        self.logger.info("Processing multi-step form")
        
        # Detect if this is a multi-step form
        form_info = await self.detect_multi_step_form(page)
        
        if not form_info.get("is_multi_step", False):
            self.logger.warning("Not a multi-step form, falling back to single-step processing")
            
            # Process as a single form
            submission_result = await self.form_interaction.submit_form(
                page, 
                search_data, 
                verify_success=True
            )
            
            return {
                "success": submission_result.get("success", False),
                "is_multi_step": False,
                "steps_completed": 1,
                "result": submission_result
            }
        
        # Initialize tracking
        steps_completed = 0
        current_url = page.url
        start_time = time.time()
        
        # Prepare step data
        if not step_data_map:
            # If no mapping provided, we'll create a simple one
            # that puts all data in the first step
            step_data_map = {1: search_data}
        
        # Process each step
        max_steps = form_info.get("total_steps", 10)  # Default to 10 if unknown
        
        try:
            while self.current_step <= max_steps:
                self.logger.info(f"Processing step {self.current_step} of {max_steps or 'unknown'}")
                
                # Get data for this step
                step_data = step_data_map.get(self.current_step, {})
                
                # Check if this is the final step
                is_final_step = (
                    self.current_step == max_steps or  # Known final step
                    self.current_step >= 5 or          # Arbitrary limit for safety
                    not form_info.get("next_buttons")  # No next button
                )
                
                if is_final_step:
                    # Submit final step
                    result = await self.submit_final_step(page, step_data, verify_success=True)
                    steps_completed += 1
                    
                    return {
                        "success": result.get("success", False),
                        "is_multi_step": True,
                        "steps_completed": steps_completed,
                        "total_steps": max_steps,
                        "final_step": self.current_step,
                        "result": result,
                        "form_data": self.current_form_data,
                        "time_taken": time.time() - start_time
                    }
                else:
                    # Navigate to next step
                    result = await self.navigate_to_next_step(page, step_data)
                    
                    if not result.get("success", False):
                        return {
                            "success": False,
                            "is_multi_step": True,
                            "steps_completed": steps_completed,
                            "total_steps": max_steps,
                            "failed_at_step": self.current_step,
                            "reason": result.get("reason", "Failed to navigate to next step"),
                            "form_data": self.current_form_data,
                            "time_taken": time.time() - start_time
                        }
                    
                    steps_completed += 1
                    
                    # Update form info after navigation
                    form_info = await self.detect_multi_step_form(page)
                    
                    # If total steps was unknown, update it
                    if form_info.get("total_steps") and not max_steps:
                        max_steps = form_info.get("total_steps")
        
        except Exception as e:
            self.logger.error(f"Error processing multi-step form: {str(e)}")
            return {
                "success": False,
                "is_multi_step": True,
                "steps_completed": steps_completed,
                "error": str(e),
                "form_data": self.current_form_data,
                "time_taken": time.time() - start_time
            }
        
        # If we somehow get here, it's an abnormal situation
        return {
            "success": False,
            "is_multi_step": True,
            "steps_completed": steps_completed,
            "reason": "Form processing did not complete normally",
            "form_data": self.current_form_data,
            "time_taken": time.time() - start_time
        }


# Register components in __init__.py
__all__ = [
    'MultiStepFormHandler'
]