"""
Browser interaction module for SmartScrape search automation.

This module provides robust browser interaction capabilities for search automation,
handling various website structures and search interfaces.
"""

import logging
import asyncio
import json
import random
import time
from typing import Dict, List, Any, Optional, Union
from urllib.parse import urlparse, quote, urljoin

from playwright.async_api import async_playwright, Page, Browser, BrowserContext
from bs4 import BeautifulSoup

# Import stealth plugin for avoiding detection
try:
    from playwright_stealth import stealth_async
except ImportError:
    # Define a dummy function as a fallback
    async def stealth_async(page):
        logging.warning("Using dummy stealth_async function. Install playwright_stealth for better results.")
        return

# Browser fingerprint presets for different user profiles
BROWSER_FINGERPRINTS = {
    "generic_windows": {
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36",
        "viewport": {"width": 1920, "height": 1080},
        "locale": "en-US",
        "timezone_id": "America/New_York",
        "platform": "Win32",
        "webgl_vendor": "Google Inc. (Intel)",
        "webgl_renderer": "ANGLE (Intel, Intel(R) UHD Graphics Direct3D11 vs_5_0 ps_5_0, D3D11)"
    },
    "generic_mac": {
        "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36",
        "viewport": {"width": 1680, "height": 1050},
        "locale": "en-US",
        "timezone_id": "America/Los_Angeles",
        "platform": "MacIntel",
        "webgl_vendor": "Apple Inc.",
        "webgl_renderer": "Apple M1"
    },
    "generic_linux": {
        "user_agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36",
        "viewport": {"width": 1920, "height": 1080},
        "locale": "en-US",
        "timezone_id": "America/Chicago",
        "platform": "Linux x86_64",
        "webgl_vendor": "Google Inc.",
        "webgl_renderer": "ANGLE (Intel, Mesa Intel(R) UHD Graphics 620 (KBL GT2), OpenGL 4.6 core)"
    },
    "mobile_android": {
        "user_agent": "Mozilla/5.0 (Linux; Android 12; Pixel 6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Mobile Safari/537.36",
        "viewport": {"width": 390, "height": 844, "isMobile": True, "hasTouch": True},
        "locale": "en-US",
        "timezone_id": "America/Los_Angeles",
        "platform": "Linux armv8l",
        "webgl_vendor": "Google Inc.",
        "webgl_renderer": "ANGLE (Google, Adreno (TM) 660, OpenGL ES 3.2 V@0502.0 (GIT@31a9e86914)"
    },
    "mobile_ios": {
        "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 15_4_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.4 Mobile/15E148 Safari/604.1",
        "viewport": {"width": 390, "height": 844, "isMobile": True, "hasTouch": True},
        "locale": "en-US",
        "timezone_id": "America/New_York",
        "platform": "iPhone",
        "webgl_vendor": "Apple Inc.",
        "webgl_renderer": "Apple GPU"
    },
    # Adding more diverse fingerprint profiles
    "windows_edge": {
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36 Edg/100.0.1185.50",
        "viewport": {"width": 1920, "height": 1080},
        "locale": "en-US",
        "timezone_id": "America/Chicago",
        "platform": "Win32",
        "webgl_vendor": "Google Inc. (Microsoft)",
        "webgl_renderer": "ANGLE (Microsoft, Direct3D11 on NVIDIA GeForce GTX 1660 Ti Direct3D11 vs_5_0 ps_5_0, D3D11)"
    },
    "mac_safari": {
        "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.4 Safari/605.1.15",
        "viewport": {"width": 2560, "height": 1440},
        "locale": "en-US",
        "timezone_id": "America/New_York",
        "platform": "MacIntel",
        "webgl_vendor": "Apple Inc.",
        "webgl_renderer": "Apple M1 Pro"
    },
    "windows_firefox": {
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:100.0) Gecko/20100101 Firefox/100.0",
        "viewport": {"width": 1920, "height": 1080},
        "locale": "en-US",
        "timezone_id": "America/New_York",
        "platform": "Win32",
        "webgl_vendor": "Mozilla",
        "webgl_renderer": "Mozilla"
    },
    "linux_firefox": {
        "user_agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:100.0) Gecko/20100101 Firefox/100.0",
        "viewport": {"width": 1920, "height": 1080},
        "locale": "en-US",
        "timezone_id": "Europe/Berlin", 
        "platform": "Linux x86_64",
        "webgl_vendor": "Mesa",
        "webgl_renderer": "AMD RENOIR (DRM 3.42.0, 5.13.0-39-generic, LLVM 12.0.0)"
    },
    "samsung_android": {
        "user_agent": "Mozilla/5.0 (Linux; Android 12; SM-S908B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Mobile Safari/537.36",
        "viewport": {"width": 412, "height": 915, "isMobile": True, "hasTouch": True},
        "locale": "en-GB",
        "timezone_id": "Europe/London",
        "platform": "Linux armv8l",
        "webgl_vendor": "Google Inc.",
        "webgl_renderer": "ANGLE (Qualcomm, Adreno (TM) 730, OpenGL ES 3.2 V@0502.0 (GIT@51216ae9b2)"
    },
    "ipad_ios": {
        "user_agent": "Mozilla/5.0 (iPad; CPU OS 15_4_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.4 Mobile/15E148 Safari/604.1",
        "viewport": {"width": 1024, "height": 1366, "isMobile": True, "hasTouch": True},
        "locale": "en-US",
        "timezone_id": "America/Los_Angeles",
        "platform": "iPad",
        "webgl_vendor": "Apple Inc.",
        "webgl_renderer": "Apple M1"
    },
    # Adding new more modern fingerprints
    "windows11_edge": {
        "user_agent": "Mozilla/5.0 (Windows NT 11.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36 Edg/114.0.1823.58",
        "viewport": {"width": 1920, "height": 1080},
        "locale": "en-US",
        "timezone_id": "America/New_York",
        "platform": "Win32",
        "webgl_vendor": "Google Inc. (Microsoft)",
        "webgl_renderer": "ANGLE (Microsoft, Direct3D11 on NVIDIA GeForce RTX 3080 Direct3D11 vs_5_0 ps_5_0, D3D11)"
    },
    "macOS_monterey_chrome": {
        "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "viewport": {"width": 1792, "height": 1120},
        "locale": "en-US",
        "timezone_id": "America/Los_Angeles",
        "platform": "MacIntel",
        "webgl_vendor": "Google Inc.",
        "webgl_renderer": "ANGLE (Apple, Apple M1 Pro, OpenGL 4.1)"
    },
    "android_samsung": {
        "user_agent": "Mozilla/5.0 (Linux; Android 13; SM-S918B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Mobile Safari/537.36",
        "viewport": {"width": 412, "height": 915, "isMobile": True, "hasTouch": True},
        "locale": "en-GB",
        "timezone_id": "Europe/London",
        "platform": "Android",
        "webgl_vendor": "Google Inc.",
        "webgl_renderer": "ANGLE (Qualcomm, Adreno (TM) 740, OpenGL ES 3.2 V@0500.0 (GIT@31f9e86dab)"
    },
    "firefox_linux": {
        "user_agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/114.0",
        "viewport": {"width": 1920, "height": 1080},
        "locale": "en-US",
        "timezone_id": "America/Denver",
        "platform": "Linux x86_64",
        "webgl_vendor": "Mesa/X.org",
        "webgl_renderer": "AMD RENOIR (DRM 3.49.0, 6.2.0-26-generic, LLVM 15.0.7)"
    }
}


class BrowserInteraction:
    """
    Handles browser-based interactions for search automation.
    
    This class provides methods for:
    - Browser session management with stealth mode
    - Consistent interaction with web elements
    - Handling of complex JavaScript-driven sites
    - Human-like browsing behavior
    - Waiting strategies for dynamic content
    - Browser fingerprint simulation to avoid detection
    """
    
    def __init__(self, config=None):
        """
        Initialize the browser interaction handler.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger("BrowserInteraction")
        self.config = config or {}
        
        # Configure default browser settings
        self.browser_type = self.config.get("browser_type", "chromium")
        self.headless = self.config.get("headless", True)
        self.slow_mo = self.config.get("slow_mo", 100)  # Slow down operations for more reliable automation
        self.default_timeout = self.config.get("timeout", 30000)
        
        # Fingerprint configuration
        self.fingerprint_profile = self.config.get("fingerprint_profile", "random")
        self.use_stealth = self.config.get("use_stealth", True)
        self.enable_webgl = self.config.get("enable_webgl", True)
        self.enable_js_features = self.config.get("enable_js_features", True)
        self.enable_canvas_fingerprint = self.config.get("enable_canvas_fingerprint", True)
        self.rotate_fingerprint = self.config.get("rotate_fingerprint", False)
        self.fingerprint_rotation_interval = self.config.get("fingerprint_rotation_interval", 30) # minutes
        
        # Track active sessions
        self.active_browser = None
        self.active_context = None
        self.active_page = None
        
        # Store the currently used fingerprint
        self.current_fingerprint = None
        self.last_fingerprint_rotation = None
        
    def _get_fingerprint(self):
        """
        Get a browser fingerprint based on the configured profile.
        
        Returns:
            Dictionary with fingerprint configuration
        """
        # If a specific profile is requested and exists, use it
        if self.fingerprint_profile != "random" and self.fingerprint_profile in BROWSER_FINGERPRINTS:
            return BROWSER_FINGERPRINTS[self.fingerprint_profile]
        
        # Otherwise, select a random fingerprint
        profile_name = random.choice(list(BROWSER_FINGERPRINTS.keys()))
        return BROWSER_FINGERPRINTS[profile_name]
        
    async def initialize_browser(self) -> Browser:
        """
        Initialize a browser instance with configured settings.
        
        Returns:
            Playwright browser instance
        """
        self.logger.info(f"Initializing {self.browser_type} browser (headless: {self.headless})")
        
        playwright = await async_playwright().start()
        
        # Select browser type
        if self.browser_type == "firefox":
            browser_instance = await playwright.firefox.launch(
                headless=self.headless,
                slow_mo=self.slow_mo
            )
        elif self.browser_type == "webkit":
            browser_instance = await playwright.webkit.launch(
                headless=self.headless,
                slow_mo=self.slow_mo
            )
        else:  # Default to chromium
            browser_instance = await playwright.chromium.launch(
                headless=self.headless,
                slow_mo=self.slow_mo,
                args=[
                    '--disable-http2',  # Disable HTTP/2 to avoid protocol errors
                    '--disable-web-security',
                    '--disable-features=VizDisplayCompositor',
                    '--disable-dev-shm-usage',
                    '--no-sandbox'
                ]
            )
        
        self.active_browser = browser_instance
        return browser_instance
    
    async def create_context(self, browser: Browser = None) -> BrowserContext:
        """
        Create a browser context with appropriate fingerprint settings.
        
        Args:
            browser: Optional browser instance, uses active_browser if None
            
        Returns:
            Browser context object
        """
        browser_to_use = browser or self.active_browser
        
        if not browser_to_use:
            raise ValueError("No browser instance available. Call initialize_browser first.")
        
        # Get fingerprint configuration
        fingerprint = self._get_fingerprint()
        self.current_fingerprint = fingerprint
        
        # Log fingerprint information
        self.logger.info(f"Using fingerprint profile with user agent: {fingerprint['user_agent'][:40]}...")
        
        # Create context with fingerprint settings
        context = await browser_to_use.new_context(
            viewport=fingerprint['viewport'],
            user_agent=fingerprint['user_agent'],
            locale=fingerprint['locale'],
            timezone_id=fingerprint['timezone_id'],
            has_touch=fingerprint['viewport'].get('hasTouch', False),
            is_mobile=fingerprint['viewport'].get('isMobile', False)
        )
        
        self.active_context = context
        return context
    
    async def new_page(self, context: BrowserContext = None) -> Page:
        """
        Create a new page with stealth mode enabled.
        
        Args:
            context: Optional browser context, uses active_context if None
            
        Returns:
            Page object with stealth mode enabled
        """
        context_to_use = context or self.active_context
        
        if not context_to_use:
            raise ValueError("No browser context available. Call create_context first.")
        
        page = await context_to_use.new_page()
        
        # Apply stealth mode to avoid detection
        if self.use_stealth:
            try:
                await stealth_async(page)
                self.logger.info("Applied stealth mode to avoid detection")
            except Exception as e:
                self.logger.warning(f"Failed to apply stealth mode: {str(e)}")
        
        # Set default timeout
        page.set_default_timeout(self.default_timeout)
        
        # Apply additional fingerprint modifications using JavaScript
        if self.enable_js_features and self.current_fingerprint:
            await self._apply_advanced_fingerprint(page)
        
        self.active_page = page
        return page
    
    async def _apply_advanced_fingerprint(self, page: Page):
        """
        Apply advanced fingerprint modifications using JavaScript.
        
        Args:
            page: Page object to modify
        """
        fingerprint = self.current_fingerprint
        
        # Apply WebGL fingerprint modifications if enabled
        if self.enable_webgl:
            await page.evaluate(f"""
                () => {{
                    // Override WebGL fingerprint
                    const getParameterProxyHandler = {{
                        apply: function(target, ctx, args) {{
                            const param = args[0];
                            
                            // Return UNMASKED_VENDOR_WEBGL
                            if (param === 37445) {{
                                return "{fingerprint['webgl_vendor']}";
                            }}
                            
                            // Return UNMASKED_RENDERER_WEBGL
                            if (param === 37446) {{
                                return "{fingerprint['webgl_renderer']}";
                            }}
                            
                            return Reflect.apply(target, ctx, args);
                        }}
                    }};
                    
                    // Try to modify WebGL if it exists and has a context
                    try {{
                        const webgl = WebGLRenderingContext.prototype;
                        webgl.getParameter = new Proxy(webgl.getParameter, getParameterProxyHandler);
                    }} catch (e) {{}}
                    
                    try {{
                        const webgl2 = WebGL2RenderingContext.prototype;
                        webgl2.getParameter = new Proxy(webgl2.getParameter, getParameterProxyHandler);
                    }} catch (e) {{}}
                }}
            """)
            
        # Apply canvas fingerprint randomization if enabled
        if self.enable_canvas_fingerprint:
            # Add canvas fingerprint protection
            await page.evaluate("""
                () => {
                    // Get a random slight modification value between 0.95 and 1.05
                    const getRandomMod = () => 0.95 + (Math.random() * 0.1);
                    
                    // Create consistent but unique noise patterns for this session
                    const noiseR = getRandomMod();
                    const noiseG = getRandomMod();
                    const noiseB = getRandomMod();
                    const noiseA = getRandomMod();
                    
                    // Override canvas methods to add subtle noise
                    try {
                        const canvasProto = CanvasRenderingContext2D.prototype;
                        const originalGetImageData = canvasProto.getImageData;
                        
                        canvasProto.getImageData = function() {
                            // Call original method to get image data
                            const imageData = originalGetImageData.apply(this, arguments);
                            
                            // Skip modifications for very small areas (likely not fingerprinting)
                            const area = imageData.width * imageData.height;
                            if (area < 100) return imageData;
                            
                            // Add subtle noise that's imperceptible to humans but changes the hash
                            const data = imageData.data;
                            for (let i = 0; i < data.length; i += 4) {
                                // Very subtle modifications to prevent breaking legitimate canvas uses
                                // But enough to change fingerprint hashes
                                if (Math.random() < 0.35) {  // Only modify some pixels
                                    data[i] = Math.min(255, Math.max(0, Math.floor(data[i] * noiseR)));
                                    data[i+1] = Math.min(255, Math.max(0, Math.floor(data[i+1] * noiseG)));
                                    data[i+2] = Math.min(255, Math.max(0, Math.floor(data[i+2] * noiseB)));
                                    data[i+3] = Math.min(255, Math.max(0, Math.floor(data[i+3] * noiseA)));
                                }
                            }
                            
                            return imageData;
                        };
                        
                        // Also override toDataURL and toBlob methods as they're used for fingerprinting
                        const originalToDataURL = HTMLCanvasElement.prototype.toDataURL;
                        HTMLCanvasElement.prototype.toDataURL = function() {
                            // Only modify if canvas is likely used for fingerprinting
                            // (small canvases with specific dimensions are often used for fingerprinting)
                            const isLikelyFingerprinting = 
                                (this.width === 16 && this.height === 16) || 
                                (this.width <= 200 && this.height <= 200 && this.width === this.height);
                                
                            if (isLikelyFingerprinting) {
                                // Get context and add some unique noise
                                const ctx = this.getContext('2d');
                                if (ctx) {
                                    // Add a single transparent pixel that changes the hash
                                    ctx.fillStyle = `rgba(${Math.floor(Math.random()*10)}, ${Math.floor(Math.random()*10)}, ${Math.floor(Math.random()*10)}, 0.01)`;
                                    ctx.fillRect(this.width - 1, this.height - 1, 1, 1);
                                }
                            }
                            
                            return originalToDataURL.apply(this, arguments);
                        };
                        
                        // Override toBlob similarly
                        const originalToBlob = HTMLCanvasElement.prototype.toBlob;
                        HTMLCanvasElement.prototype.toBlob = function() {
                            const isLikelyFingerprinting = 
                                (this.width === 16 && this.height === 16) || 
                                (this.width <= 200 && this.height <= 200 && this.width === this.height);
                                
                            if (isLikelyFingerprinting) {
                                const ctx = this.getContext('2d');
                                if (ctx) {
                                    ctx.fillStyle = `rgba(${Math.floor(Math.random()*10)}, ${Math.floor(Math.random()*10)}, ${Math.floor(Math.random()*10)}, 0.01)`;
                                    ctx.fillRect(this.width - 1, this.height - 1, 1, 1);
                                }
                            }
                            
                            return originalToBlob.apply(this, arguments);
                        };
                    } catch (e) {
                        console.error("Canvas fingerprint protection error:", e);
                    }
                }
            """)
        
        # Apply platform fingerprint
        await page.evaluate(f"""
            () => {{
                // Override navigator platform
                Object.defineProperty(navigator, 'platform', {{
                    get: () => '{fingerprint['platform']}'
                }});
                
                // Add small deviations to make fingerprint less uniform
                const hardwareConcurrency = Math.min(16, Math.max(2, (navigator.hardwareConcurrency || 4) + 
                    (Math.random() > 0.5 ? 1 : -1)));
                
                Object.defineProperty(navigator, 'hardwareConcurrency', {{
                    get: () => hardwareConcurrency
                }});
                
                // Random device memory value based on platform
                let memory = 8;
                if ('{fingerprint['platform']}'.includes('Linux')) memory = 4;
                if ('{fingerprint['platform']}'.includes('Win')) memory = 8;
                if ('{fingerprint['platform']}'.includes('Mac')) memory = 16;
                
                // Add slight randomization to memory value
                memory = memory * (0.9 + Math.random() * 0.2);
                
                // Only set deviceMemory if it exists in the original navigator
                if ('deviceMemory' in navigator) {{
                    Object.defineProperty(navigator, 'deviceMemory', {{
                        get: () => memory
                    }});
                }}
                
                // Override screen size to match viewport
                if (window.screen) {{
                    Object.defineProperty(window.screen, 'width', {{
                        get: () => {fingerprint['viewport']['width']}
                    }});
                    Object.defineProperty(window.screen, 'height', {{
                        get: () => {fingerprint['viewport']['height']}
                    }});
                    
                    // Also adjust availWidth/availHeight with slight differences
                    const availWidthDiff = Math.floor(Math.random() * 20);
                    const availHeightDiff = Math.floor(Math.random() * 40);
                    
                    Object.defineProperty(window.screen, 'availWidth', {{
                        get: () => {fingerprint['viewport']['width']} - availWidthDiff
                    }});
                    Object.defineProperty(window.screen, 'availHeight', {{
                        get: () => {fingerprint['viewport']['height']} - availHeightDiff
                    }});
                }}
                
                // Advanced audio fingerprinting protection is handled separately
            }}
        """)
        
        # Handle audio fingerprinting separately
        await page.evaluate("""
            () => {
                try {
                    const audioContext = window.AudioContext || window.webkitAudioContext;
                    AudioBuffer.prototype.getChannelData = new Proxy(AudioBuffer.prototype.getChannelData, {
                        apply(target, thisArg, args) {
                            const data = Reflect.apply(target, thisArg, args);
                            
                            // Skip modification on large audio buffers used for media playback
                            if (thisArg.length > 200000) return data;
                            
                            // Create a copy of the data to avoid modifying the original
                            const copy = new Float32Array(data);
                            
                            // Add very subtle noise to a few samples
                            // This changes the fingerprint hash without affecting actual audio
                            for (let i = 0; i < copy.length; i += 500) {
                                if (Math.random() < 0.1) {
                                    const noise = (Math.random() * 0.0001) - 0.00005;
                                    copy[i] = Math.max(-1, Math.min(1, copy[i] + noise));
                                }
                            }
                            
                            return copy;
                        }
                    });
                } catch (e) {
                    // Ignore errors, fallback to default audio context
                }
            }
        """)
        
        # Apply font and language fingerprinting
        await page.evaluate("""
            () => {
                // Override language and languages arrays
                const languages = ['en-US', 'en'];
                Object.defineProperty(navigator, 'language', {
                    get: () => languages[0]
                });
                Object.defineProperty(navigator, 'languages', {
                    get: () => [...languages]
                });
                
                // Override permissions behavior if present
                if (navigator.permissions) {
                    const originalQuery = navigator.permissions.query;
                    navigator.permissions.query = (parameters) => {
                        if (parameters.name === 'notifications') {
                            return Promise.resolve({ state: "prompt" });
                        }
                        if (parameters.name === 'geolocation') {
                            return Promise.resolve({ state: "prompt" });
                        }
                        return originalQuery(parameters);
                    };
                }
                
                // Override new date to be slightly randomized
                const originalDate = Date;
                const dateOffset = (Math.random() < 0.5 ? 1 : -1) * Math.floor(Math.random() * 10000);
                
                Date = function(...args) {
                    if (args.length === 0) {
                        const date = new originalDate();
                        date.setTime(date.getTime() + dateOffset);
                        return date;
                    }
                    return new originalDate(...args);
                };
                
                Date.now = function() {
                    return originalDate.now() + dateOffset;
                };
                
                Date.parse = originalDate.parse;
                Date.UTC = originalDate.UTC;
                Date.prototype = originalDate.prototype;
                
                // Add protection against font fingerprinting
                try {
                    // Override font enumeration techniques
                    const fontSizeThreshold = 0.2;
                    const originalMeasureText = CanvasRenderingContext2D.prototype.measureText;
                    
                    CanvasRenderingContext2D.prototype.measureText = function (text) {
                        let result = originalMeasureText.apply(this, arguments);
                      
                        // Identify likely font fingerprinting
                        if (text.length === 1 && this.font) {
                            // Add a small random variation to width measurement
                            // This affects fingerprinting but not normal text rendering
                            const originalWidth = result.width;
                            const variation = (Math.random() * 0.02) - 0.01; // ±1% variation
                            
                            Object.defineProperty(result, 'width', {
                                get: function() {
                                    return originalWidth * (1 + variation);
                                }
                            });
                        }
                        
                        return result;
                    };
                } catch (e) {
                    // Ignore errors, fallback to default font behavior
                }
            }
        """)
        
        # Add fingerprint-specific cookies to better simulate a real user
        await page.evaluate("""
            () => {
                // Add some basic cookies for a more realistic profile
                document.cookie = "user_session=true; path=/";
                document.cookie = "timezone_offset=" + new Date().getTimezoneOffset() + "; path=/";
                document.cookie = "screen_res=" + screen.width + "x" + screen.height + "; path=/";
                document.cookie = "revisit=true; path=/";
            }
        """)
        
        self.logger.info("Applied advanced fingerprint modifications")
        
    async def check_and_rotate_fingerprint(self, page: Page = None):
        """
        Check if fingerprint rotation is needed and rotate if necessary.
        
        Args:
            page: Optional page object, uses active_page if None
            
        Returns:
            True if fingerprint was rotated, False otherwise
        """
        if not self.rotate_fingerprint:
            return False
            
        page_to_use = page or self.active_page
        
        if not page_to_use:
            return False
            
        current_time = time.time()
        
        # If we haven't yet set a last rotation time or if enough time has passed
        if not self.last_fingerprint_rotation or \
           (current_time - self.last_fingerprint_rotation) > (self.fingerprint_rotation_interval * 60):
            
            # Get a new fingerprint
            old_fingerprint = self.current_fingerprint
            self.current_fingerprint = self._get_fingerprint()
            
            # Make sure we don't get the same fingerprint
            attempts = 0
            while self.current_fingerprint == old_fingerprint and attempts < 5:
                self.current_fingerprint = self._get_fingerprint()
                attempts += 1
                
            # Apply the new fingerprint
            await self._apply_advanced_fingerprint(page_to_use)
            self.last_fingerprint_rotation = current_time
            
            self.logger.info(f"Rotated browser fingerprint to {self.current_fingerprint['user_agent'][:40]}...")
            return True
            
        return False
    
    async def navigate(self, url: str, page: Page = None, wait_until: str = 'domcontentloaded') -> Dict[str, Any]:
        """
        Navigate to a URL with enhanced error handling and waiting.
        
        Args:
            url: URL to navigate to
            page: Optional page object, uses active_page if None
            wait_until: Navigation wait condition
            
        Returns:
            Dictionary with navigation result
        """
        page_to_use = page or self.active_page
        
        if not page_to_use:
            raise ValueError("No page object available. Call new_page first.")
        
        self.logger.info(f"Navigating to {url}")
        
        try:
            # Capture response and page load timing
            response = await page_to_use.goto(url, wait_until=wait_until, timeout=self.default_timeout)
            
            # Wait for network to become idle
            await page_to_use.wait_for_load_state('networkidle', timeout=self.default_timeout).catch(lambda e: None)
            
            # Additional wait to ensure JavaScript execution
            await asyncio.sleep(1)
            
            return {
                'success': True,
                'url': page_to_use.url,
                'status': response.status if response else None,
                'response': response
            }
            
        except Exception as e:
            self.logger.error(f"Navigation error: {str(e)}")
            return {
                'success': False,
                'url': url,
                'error': str(e)
            }
    
    async def fill_form(self, selector: str, value: str, page: Page = None, delay: int = 50) -> bool:
        """
        Fill a form field with human-like typing behavior.
        
        Args:
            selector: CSS selector for the form field
            value: Value to type into the field
            page: Optional page object, uses active_page if None
            delay: Delay between keystrokes in milliseconds
            
        Returns:
            True if successful, False otherwise
        """
        page_to_use = page or self.active_page
        
        if not page_to_use:
            raise ValueError("No page object available.")
        
        try:
            # Check if element exists and is visible
            is_visible = await page_to_use.is_visible(selector)
            if not is_visible:
                self.logger.warning(f"Element {selector} is not visible")
                return False
            
            # Focus on the element first
            await page_to_use.focus(selector)
            
            # Clear any existing value
            await page_to_use.evaluate(f'document.querySelector("{selector}").value = ""')
            
            # Type the value with realistic delay
            await page_to_use.type(selector, value, delay=delay)
            
            # Dispatch input and change events to trigger any listeners
            await page_to_use.evaluate(f'''
                const el = document.querySelector("{selector}");
                el.dispatchEvent(new Event('input', {{ bubbles: true }}));
                el.dispatchEvent(new Event('change', {{ bubbles: true }}));
            ''')
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error filling form field {selector}: {str(e)}")
            return False
    
    async def click_element(self, selector: str, page: Page = None, 
                          force: bool = False, timeout: int = None) -> bool:
        """
        Click an element with enhanced error handling and retry logic.
        
        Args:
            selector: CSS selector for the element to click
            page: Optional page object, uses active_page if None
            force: Whether to force the click even if element is not visible
            timeout: Custom timeout in milliseconds
            
        Returns:
            True if successful, False otherwise
        """
        page_to_use = page or self.active_page
        
        if not page_to_use:
            raise ValueError("No page object available.")
        
        actual_timeout = timeout or self.default_timeout
        
        try:
            # Wait for element to be visible and enabled
            if not force:
                await page_to_use.wait_for_selector(
                    selector, 
                    state='visible', 
                    timeout=actual_timeout
                )
            
            # Try regular click first
            try:
                await page_to_use.click(selector, timeout=actual_timeout)
                return True
                
            except Exception as click_error:
                self.logger.warning(f"Standard click failed for {selector}: {str(click_error)}")
                
                if force:
                    # Try with force option
                    await page_to_use.click(selector, force=True, timeout=actual_timeout)
                    return True
                    
                # Try JavaScript click as fallback
                clicked = await page_to_use.evaluate(f'''
                    (() => {{
                        const element = document.querySelector("{selector}");
                        if(element) {{
                            element.click();
                            return true;
                        }}
                        return false;
                    }})()
                ''')
                
                if clicked:
                    return True
                    
                raise Exception(f"All click methods failed for {selector}")
                
        except Exception as e:
            self.logger.error(f"Error clicking element {selector}: {str(e)}")
            return False
    
    async def submit_form(self, form_selector: str, page: Page = None) -> bool:
        """
        Submit a form with proper event handling.
        
        Args:
            form_selector: CSS selector for the form
            page: Optional page object, uses active_page if None
            
        Returns:
            True if successful, False otherwise
        """
        page_to_use = page or self.active_page
        
        if not page_to_use:
            raise ValueError("No page object available.")
        
        try:
            # Try to find a submit button first
            submit_button_selectors = [
                f"{form_selector} button[type='submit']",
                f"{form_selector} input[type='submit']",
                f"{form_selector} .submit",
                f"{form_selector} [class*='submit']",
                f"{form_selector} button:has-text('Search')",
                f"{form_selector} button:has-text('Find')",
                f"{form_selector} button:has-text('Go')",
                f"{form_selector} button:not([type='reset'])"
            ]
            
            for button_selector in submit_button_selectors:
                if await page_to_use.is_visible(button_selector):
                    await self.click_element(button_selector, page_to_use)
                    return True
            
            # If no button found, try JavaScript form submission
            submitted = await page_to_use.evaluate(f'''
                (() => {{
                    const form = document.querySelector("{form_selector}");
                    if(form) {{
                        form.submit();
                        return true;
                    }}
                    return false;
                }})()
            ''')
            
            return submitted
            
        except Exception as e:
            self.logger.error(f"Error submitting form {form_selector}: {str(e)}")
            return False
    
    async def wait_for_navigation(self, page: Page = None, timeout: int = None) -> bool:
        """
        Wait for navigation to complete with multiple signals.
        
        Args:
            page: Optional page object, uses active_page if None
            timeout: Custom timeout in milliseconds
            
        Returns:
            True if navigation completed, False otherwise
        """
        page_to_use = page or self.active_page
        
        if not page_to_use:
            raise ValueError("No page object available.")
        
        actual_timeout = timeout or self.default_timeout
        
        try:
            # Wait for navigation to complete
            await page_to_use.wait_for_load_state('domcontentloaded', timeout=actual_timeout)
            
            # Also wait for network to become relatively idle
            await page_to_use.wait_for_load_state('networkidle', timeout=actual_timeout).catch(lambda e: None)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error waiting for navigation: {str(e)}")
            return False
    
    async def wait_for_element(self, selector: str, page: Page = None, 
                             state: str = 'visible', timeout: int = None) -> bool:
        """
        Wait for an element to appear in a specific state.
        
        Args:
            selector: CSS selector for the element
            page: Optional page object, uses active_page if None
            state: Element state to wait for ('attached', 'detached', 'visible', or 'hidden')
            timeout: Custom timeout in milliseconds
            
        Returns:
            True if element reached desired state, False otherwise
        """
        page_to_use = page or self.active_page
        
        if not page_to_use:
            raise ValueError("No page object available.")
        
        actual_timeout = timeout or self.default_timeout
        
        try:
            await page_to_use.wait_for_selector(selector, state=state, timeout=actual_timeout)
            return True
            
        except Exception as e:
            self.logger.debug(f"Element {selector} did not reach state '{state}': {str(e)}")
            return False
    
    async def extract_page_content(self, page: Page = None) -> Dict[str, Any]:
        """
        Extract various content types from the current page.
        
        Args:
            page: Optional page object, uses active_page if None
            
        Returns:
            Dictionary with extracted content
        """
        page_to_use = page or self.active_page
        
        if not page_to_use:
            raise ValueError("No page object available.")
        
        try:
            # Get full page content
            html_content = await page_to_use.content()
            
            # Get page title
            title = await page_to_use.title()
            
            # Get meta description
            meta_description = await page_to_use.evaluate('''
                () => {
                    const meta = document.querySelector('meta[name="description"]');
                    return meta ? meta.getAttribute('content') : '';
                }
            ''')
            
            # Extract visible text using Playwright's evaluateHandle
            visible_text = await page_to_use.evaluate('''
                () => {
                    // Function to get visible text, excluding scripts, styles, etc.
                    const getVisibleText = (node) => {
                        let text = '';
                        
                        // Skip hidden elements
                        try {
                            const style = window.getComputedStyle(node);
                            if (style.display === 'none' || style.visibility === 'hidden') {
                                return '';
                            }
                        } catch (e) {
                            // Element might not be an HTMLElement, ignore
                        }
                        
                        // Process node based on its type
                        switch (node.nodeType) {
                            case Node.TEXT_NODE:
                                text = node.textContent.trim();
                                break;
                            case Node.ELEMENT_NODE:
                                // Skip script, style, noscript elements
                                if (['SCRIPT', 'STYLE', 'NOSCRIPT'].includes(node.tagName)) {
                                    break;
                                }
                                
                                // Process child nodes
                                for (const child of node.childNodes) {
                                    text += ' ' + getVisibleText(child);
                                }
                                break;
                        }
                        
                        return text.trim();
                    };
                    
                    return getVisibleText(document.body);
                }
            ''')
            
            # Create a BeautifulSoup object for easier HTML analysis
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract all links
            links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href and not href.startswith(('#', 'javascript:')):
                    # Convert to absolute URL if needed
                    if not href.startswith(('http://', 'https://')):
                        href = urljoin(page_to_use.url, href)
                    
                    link_text = link.get_text().strip()
                    links.append({
                        'url': href,
                        'text': link_text if link_text else '[No Text]'
                    })
            
            # Extract images
            images = []
            for img in soup.find_all('img', src=True):
                src = img['src']
                if src and not src.startswith('data:'):
                    # Convert to absolute URL if needed
                    if not src.startswith(('http://', 'https://')):
                        src = urljoin(page_to_use.url, src)
                    
                    alt_text = img.get('alt', '')
                    images.append({
                        'src': src,
                        'alt': alt_text
                    })
            
            return {
                'url': page_to_use.url,
                'title': title,
                'meta_description': meta_description,
                'html': html_content,
                'text': visible_text,
                'links': links,
                'images': images
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting page content: {str(e)}")
            return {
                'url': page_to_use.url if page_to_use else None,
                'error': str(e)
            }
    
    async def find_search_interface(self, page: Page = None) -> Dict[str, Any]:
        """
        Find search interfaces on the current page.
        
        Args:
            page: Optional page object, uses active_page if None
            
        Returns:
            Dictionary with search interface information
        """
        page_to_use = page or self.active_page
        
        if not page_to_use:
            raise ValueError("No page object available.")
        
        try:
            # Use JavaScript to find search forms and inputs
            search_interfaces = await page_to_use.evaluate('''
                () => {
                    const results = {
                        search_forms: [],
                        standalone_inputs: [],
                        search_buttons: []
                    };
                    
                    // Find forms that look like search forms
                    document.querySelectorAll('form').forEach(form => {
                        const formData = {
                            id: form.id || '',
                            class_list: Array.from(form.classList),
                            action: form.action || '',
                            method: form.method || 'get',
                            inputs: [],
                            has_search_input: false,
                            search_relevance: 0
                        };
                        
                        // Analyze form attributes for search indicators
                        const formText = (form.id + ' ' + Array.from(form.classList).join(' ') + ' ' + form.action).toLowerCase();
                        if (formText.includes('search') || formText.includes('find') || formText.includes('query')) {
                            formData.search_relevance += 20;
                        }
                        
                        // Check input fields
                        form.querySelectorAll('input').forEach(input => {
                            const inputData = {
                                type: input.type || 'text',
                                name: input.name || '',
                                id: input.id || '',
                                placeholder: input.placeholder || '',
                                class_list: Array.from(input.classList)
                            };
                            
                            formData.inputs.push(inputData);
                            
                            // Check if this is a search input
                            const inputText = (input.name + ' ' + input.id + ' ' + input.placeholder + ' ' + Array.from(input.classList).join(' ')).toLowerCase();
                            if (input.type === 'search' || inputText.includes('search') || inputText.includes('query') || inputText.includes('keyword')) {
                                formData.has_search_input = true;
                                formData.search_relevance += 30;
                            }
                        });
                        
                        // Only include forms with reasonable search relevance
                        if (formData.has_search_input || formData.search_relevance >= 20) {
                            // Find the selector for this form
                            let selector = '';
                            if (form.id) {
                                selector = `#${form.id}`;
                            } else if (form.classList.length > 0) {
                                selector = `form.${Array.from(form.classList).join('.')}`;
                            } else {
                                // Create a more complex selector using child elements
                                const inputs = Array.from(form.querySelectorAll('input')). filter(i => i.name || i.id);
                                if (inputs.length > 0) {
                                    const firstInput = inputs[0];
                                    if (firstInput.id) {
                                        selector = `form:has(#${firstInput.id})`;
                                    } else if (firstInput.name) {
                                        selector = `form:has(input[name="${firstInput.name}"])`;
                                    }
                                }
                                
                                // Fallback selector
                                if (!selector) {
                                    const formIndex = Array.from(document.querySelectorAll('form')).indexOf(form);
                                    selector = `form:nth-of-type(${formIndex + 1})`;
                                }
                            }
                            
                            formData.selector = selector;
                            results.search_forms.push(formData);
                        }
                    });
                    
                    // Find standalone search inputs (not in forms)
                    document.querySelectorAll('input[type="search"], input[type="text"]').forEach(input => {
                        // Skip inputs already in a form
                        if (input.closest('form')) {
                            return;
                        }
                        
                        const inputText = (input.name + ' ' + input.id + ' ' + input.placeholder + ' ' + Array.from(input.classList).join(' ')).toLowerCase();
                        if (inputText.includes('search') || inputText.includes('query') || inputText.includes('keyword')) {
                            let selector = '';
                            if (input.id) {
                                selector = `#${input.id}`;
                            } else if (input.name) {
                                selector = `input[name="${input.name}"]`;
                            } else if (input.classList.length > 0) {
                                selector = `input.${Array.from(input.classList).join('.')}`;
                            } else {
                                const inputIndex = Array.from(document.querySelectorAll('input')).indexOf(input);
                                selector = `input:nth-of-type(${inputIndex + 1})`;
                            }
                            
                            results.standalone_inputs.push({
                                type: input.type || 'text',
                                name: input.name || '',
                                id: input.id || '',
                                placeholder: input.placeholder || '',
                                class_list: Array.from(input.classList),
                                selector: selector
                            });
                        }
                    });
                    
                    // Find dedicated search buttons (like magnifying glass icons)
                    const buttonSelectors = [
                        'button[aria-label*="search" i]',
                        'button.search-button',
                        'button.search',
                        'a.search-button',
                        'a.search',
                        'div.search-button',
                        'div[role="button"][aria-label*="search" i]'
                    ];
                    
                    buttonSelectors.forEach(selector => {
                        document.querySelectorAll(selector).forEach(button => {
                            let buttonSelector = selector;
                            if (button.id) {
                                buttonSelector = `#${button.id}`;
                            } else if (button.classList.length > 0) {
                                buttonSelector = `${button.tagName.toLowerCase()}.${Array.from(button.classList).join('.')}`;
                            }
                            
                            results.search_buttons.push({
                                tag_name: button.tagName.toLowerCase(),
                                id: button.id || '',
                                class_list: Array.from(button.classList),
                                text: button.textContent.trim(),
                                selector: buttonSelector
                            });
                        });
                    });
                    
                    return results;
                }
            ''')
            
            # Analyze and score the results
            scored_interfaces = []
            
            # Process search forms
            for form in search_interfaces.get('search_forms', []):
                scored_interfaces.append({
                    'type': 'form',
                    'selector': form['selector'],
                    'search_relevance': form['search_relevance'],
                    'has_search_input': form['has_search_input'],
                    'method': form['method'],
                    'action': form['action'],
                    'inputs': form['inputs']
                })
            
            # Process standalone inputs
            for input_field in search_interfaces.get('standalone_inputs', []):
                scored_interfaces.append({
                    'type': 'input',
                    'selector': input_field['selector'],
                    'search_relevance': 40,  # Standalone search inputs are highly relevant
                    'input_type': input_field['type'],
                    'placeholder': input_field['placeholder']
                })
            
            # Process search buttons
            for button in search_interfaces.get('search_buttons', []):
                scored_interfaces.append({
                    'type': 'button',
                    'selector': button['selector'],
                    'search_relevance': 30,
                    'text': button['text']
                })
            
            # Sort by search relevance
            scored_interfaces.sort(key=lambda x: x['search_relevance'], reverse=True)
            
            return {
                'success': True,
                'interfaces': scored_interfaces,
                'count': len(scored_interfaces)
            }
            
        except Exception as e:
            self.logger.error(f"Error finding search interfaces: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'interfaces': [],
                'count': 0
            }
    
    async def perform_search(self, search_term: str, page: Page = None) -> Dict[str, Any]:
        """
        Perform search using the best available search interface.
        
        Args:
            search_term: Search term to use
            page: Optional page object, uses active_page if None
            
        Returns:
            Dictionary with search results information
        """
        page_to_use = page or self.active_page
        
        if not page_to_use:
            raise ValueError("No page object available.")
        
        try:
            self.logger.info(f"Performing search for term: {search_term}")
            
            # Find search interfaces
            interfaces = await self.find_search_interface(page_to_use)
            
            if not interfaces['success'] or interfaces['count'] == 0:
                self.logger.warning("No search interfaces found")
                return {
                    'success': False,
                    'error': 'No search interfaces found',
                    'search_term': search_term
                }
            
            # Attempt to use the best interface
            for interface in interfaces['interfaces']:
                interface_type = interface['type']
                selector = interface['selector']
                
                if (interface_type == 'form'):
                    # Look for the search input within the form
                    input_selector = None
                    for input_field in interface.get('inputs', []):
                        input_type = input_field.get('type', '')
                        if (input_type in ['search', 'text', '']):
                            # Try to construct a selector for this input
                            if (input_field.get('id')):
                                input_selector = f"#{input_field['id']}"
                            elif (input_field.get('name')):
                                input_selector = f"{selector} input[name='{input_field['name']}']"
                            else:
                                input_selector = f"{selector} input[type='{input_type}']"
                            break
                    
                    if (input_selector):
                        # Fill the input
                        fill_success = await self.fill_form(input_selector, search_term, page_to_use)
                        if (fill_success):
                            # Submit the form
                            submit_success = await self.submit_form(selector, page_to_use)
                            if (submit_success):
                                # Wait for navigation or new content
                                await self.wait_for_navigation(page_to_use)
                                return {
                                    'success': True,
                                    'method': 'form',
                                    'search_term': search_term,
                                    'url': page_to_use.url
                                }
                
                elif (interface_type == 'input'):
                    # Fill the standalone input
                    fill_success = await self.fill_form(selector, search_term, page_to_use)
                    if (fill_success):
                        # Press Enter to submit
                        await page_to_use.press(selector, 'Enter')
                        # Wait for navigation or new content
                        await self.wait_for_navigation(page_to_use)
                        return {
                            'success': True,
                            'method': 'standalone_input',
                            'search_term': search_term,
                            'url': page_to_use.url
                        }
                
                elif (interface_type == 'button'):
                    # Look for nearby input
                    input_found = await page_to_use.evaluate(f'''
                        (() => {{
                            const button = document.querySelector("{selector}");
                            if (!button) return false;
                            
                            // Look for inputs near this button
                            let container = button.parentElement;
                            for (let i = 0; i < 3; i++) {{  // Check up to 3 levels up
                                if (!container) break;
                                
                                const inputs = container.querySelectorAll('input[type="search"], input[type="text"]');
                                if (inputs.length > 0) {{
                                    // Found an input - focus and fill it
                                    const input = inputs[0];
                                    input.focus();
                                    input.value = "{search_term}";
                                    input.dispatchEvent(new Event('input', {{ bubbles: true }}));
                                    input.dispatchEvent(new Event('change', {{ bubbles: true }}));
                                    return true;
                                }}
                                
                                container = container.parentElement;
                            }}
                            
                            return false;
                        }})()
                    ''')
                    
                    if (input_found):
                        # Now click the button
                        click_success = await self.click_element(selector, page_to_use)
                        if (click_success):
                            # Wait for navigation or new content
                            await self.wait_for_navigation(page_to_use)
                            return {
                                'success': True,
                                'method': 'button_with_input',
                                'search_term': search_term,
                                'url': page_to_use.url
                            }
            
            # If we've tried all interfaces without success, try more aggressive approach
            self.logger.info("Trying advanced search techniques")
            
            # Try direct JavaScript injection
            search_success = await page_to_use.evaluate(f'''
                (() => {{
                    // Try to find any possible search input
                    const searchInputs = Array.from(document.querySelectorAll('input')).filter(input => {{
                        const attrs = (input.id || '') + ' ' + (input.name || '') + ' ' + 
                                    (input.className || '') + ' ' + (input.placeholder || '');
                        const attrsLower = attrs.toLowerCase();
                        
                        return input.type === 'search' || 
                               input.type === 'text' || 
                               !input.type || 
                               attrsLower.includes('search') ||
                               attrsLower.includes('query') ||
                               attrsLower.includes('keyword');
                    }});
                    
                    if (searchInputs.length > 0) {{
                        const input = searchInputs[0];
                        input.value = "{search_term}";
                        input.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        input.dispatchEvent(new Event('change', {{ bubbles: true }}));
                        
                        // Try to submit by pressing Enter
                        const enterEvent = new KeyboardEvent('keydown', {{
                            bubbles: true, 
                            cancelable: true, 
                            key: 'Enter',
                            code: 'Enter',
                            keyCode: 13
                        }});
                        
                        input.dispatchEvent(enterEvent);
                        return true;
                    }}
                    
                    return false;
                }})()
            ''')
            
            if (search_success):
                # Wait for navigation or new content
                await self.wait_for_navigation(page_to_use)
                return {
                    'success': True,
                    'method': 'javascript_injection',
                    'search_term': search_term,
                    'url': page_to_use.url
                }
            
            # If all else fails, try URL manipulation
            current_url = page_to_use.url
            parsed_url = urlparse(current_url)
            domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
            
            # Common URL patterns for search
            search_urls = [
                f"{domain}/search?q={quote(search_term)}",
                f"{domain}/search?query={quote(search_term)}",
                f"{domain}/search?keyword={quote(search_term)}",
                f"{domain}/search?term={quote(search_term)}",
                f"{domain}/find?q={quote(search_term)}",
                f"{domain}?s={quote(search_term)}"
            ]
            
            for url in search_urls:
                nav_result = await self.navigate(url, page_to_use)
                if (nav_result['success']):
                    # Check if the page seems like a search results page
                    has_results = await page_to_use.evaluate(f'''
                        (() => {{
                            const pageText = document.body.textContent.toLowerCase();
                            return pageText.includes("{search_term.lower()}") && 
                                   (pageText.includes("result") || 
                                    pageText.includes("found") || 
                                    document.querySelectorAll(".result, .search-result, .product, .listing").length > 0);
                        }})()
                    ''')
                    
                    if (has_results):
                        return {
                            'success': True,
                            'method': 'url_manipulation',
                            'search_term': search_term,
                            'url': page_to_use.url
                        }
            
            return {
                'success': False,
                'error': 'All search methods failed',
                'search_term': search_term,
                'url': page_to_use.url
            }
            
        except Exception as e:
            self.logger.error(f"Error performing search: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'search_term': search_term
            }
    
    async def close(self):
        """Close all active browser sessions and clean up resources."""
        try:
            if self.active_page:
                await self.active_page.close()
                self.active_page = None
                
            if self.active_context:
                await self.active_context.close()
                self.active_context = None
                
            if self.active_browser:
                await self.active_browser.close()
                self.active_browser = None
                
        except Exception as e:
            self.logger.error(f"Error closing browser resources: {str(e)}")