# template_storage.py - Template storage and matching for SmartScrape

import json
import os
import re
import logging
import datetime
from urllib.parse import urlparse
from typing import Dict, List, Any, Optional, Union
import asyncio

# Import our enhanced utilities
from utils.html_utils import parse_html, find_by_xpath, select_with_css, extract_text_fast
from utils.retry_utils import with_exponential_backoff
from utils.file_utils import read_json, write_json, file_exists
# Import enhanced extraction utilities
from extraction.content_extraction import ContentExtractor
# Import the merge_extraction_configs function from the new utility file instead
from utils.extraction_utils import merge_extraction_configs

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TemplateStorage")

class TemplateStorage:
    """
    Manages storage, retrieval, and application of extraction templates.
    
    Templates allow for consistent extraction from known sites without 
    requiring repeated AI extraction.
    """
    
    def __init__(self, templates_dir: str = "templates"):
        """
        Initialize the template storage.
        
        Args:
            templates_dir: Directory where templates are stored
        """
        self.templates_dir = templates_dir
        self.templates = {}
        self.domain_patterns = {}
        self.extraction_preferences = {}
        
        # Create templates directory if it doesn't exist
        os.makedirs(templates_dir, exist_ok=True)
        
        # Create preferences directory
        self.preferences_dir = os.path.join(templates_dir, "preferences")
        os.makedirs(self.preferences_dir, exist_ok=True)
        
        # Create enhanced extraction profiles directory
        self.extraction_profiles_dir = os.path.join(templates_dir, "extraction_profiles")
        os.makedirs(self.extraction_profiles_dir, exist_ok=True)
        
        # Initialize content extractor
        self.content_extractor = ContentExtractor()
        
        # Load templates synchronously instead of using asyncio tasks
        self._load_templates_sync()
        self._load_extraction_preferences_sync()
        self._load_extraction_profiles_sync()
    
    def _load_templates_sync(self):
        """Load all templates from the templates directory synchronously"""
        if not os.path.exists(self.templates_dir):
            logger.warning(f"Templates directory {self.templates_dir} does not exist")
            return
            
        try:
            template_files = [f for f in os.listdir(self.templates_dir) if f.endswith('.json')]
            
            for filename in template_files:
                try:
                    domain = filename.replace('.json', '')
                    template_path = os.path.join(self.templates_dir, filename)
                    
                    # Read template data synchronously
                    with open(template_path, 'r', encoding='utf-8') as f:
                        template_data = json.load(f)
                    
                    if template_data:
                        self.templates[domain] = template_data
                        logger.info(f"Loaded template for {domain}")
                except Exception as e:
                    logger.error(f"Error loading template {filename}: {str(e)}")
        except Exception as e:
            logger.error(f"Error scanning templates directory: {str(e)}")
    
    def _load_extraction_preferences_sync(self):
        """Load all extraction preferences from the preferences directory synchronously"""
        if not os.path.exists(self.preferences_dir):
            logger.warning(f"Preferences directory {self.preferences_dir} does not exist")
            return
            
        try:
            preference_files = [f for f in os.listdir(self.preferences_dir) if f.endswith('.json')]
            
            for filename in preference_files:
                try:
                    domain = filename.replace('.json', '')
                    preference_path = os.path.join(self.preferences_dir, filename)
                    
                    # Read preference data synchronously
                    with open(preference_path, 'r', encoding='utf-8') as f:
                        preference_data = json.load(f)
                    
                    if preference_data:
                        self.extraction_preferences[domain] = preference_data
                        logger.info(f"Loaded extraction preference for {domain}")
                except Exception as e:
                    logger.error(f"Error loading extraction preference {filename}: {str(e)}")
        except Exception as e:
            logger.error(f"Error scanning preferences directory: {str(e)}")
    
    def _load_extraction_profiles_sync(self):
        """Load all enhanced extraction profiles from the profiles directory synchronously"""
        if not os.path.exists(self.extraction_profiles_dir):
            logger.warning(f"Extraction profiles directory {self.extraction_profiles_dir} does not exist")
            return
            
        try:
            profile_files = [f for f in os.listdir(self.extraction_profiles_dir) if f.endswith('.json')]
            
            for filename in profile_files:
                try:
                    domain = filename.replace('.json', '')
                    profile_path = os.path.join(self.extraction_profiles_dir, filename)
                    
                    # Read profile data synchronously
                    with open(profile_path, 'r', encoding='utf-8') as f:
                        profile_data = json.load(f)
                    
                    if profile_data:
                        self.extraction_preferences[domain] = profile_data
                        logger.info(f"Loaded extraction profile for {domain}")
                except Exception as e:
                    logger.error(f"Error loading extraction profile {filename}: {str(e)}")
        except Exception as e:
            logger.error(f"Error scanning extraction profiles directory: {str(e)}")
    
    # Keep the async versions for when they're called in async contexts
    async def _load_templates(self):
        """Load all templates from the templates directory asynchronously"""
        if not os.path.exists(self.templates_dir):
            logger.warning(f"Templates directory {self.templates_dir} does not exist")
            return
            
        try:
            template_files = [f for f in os.listdir(self.templates_dir) if f.endswith('.json')]
            
            for filename in template_files:
                try:
                    domain = filename.replace('.json', '')
                    template_path = os.path.join(self.templates_dir, filename)
                    
                    # Read template data
                    template_data = await read_json(template_path)
                    if template_data:
                        self.templates[domain] = template_data
                        logger.info(f"Loaded template for {domain}")
                except Exception as e:
                    logger.error(f"Error loading template {filename}: {str(e)}")
        except Exception as e:
            logger.error(f"Error scanning templates directory: {str(e)}")
    
    async def _load_extraction_preferences(self):
        """Load all extraction preferences from the preferences directory asynchronously"""
        if not os.path.exists(self.preferences_dir):
            logger.warning(f"Preferences directory {self.preferences_dir} does not exist")
            return
            
        try:
            preference_files = [f for f in os.listdir(self.preferences_dir) if f.endswith('.json')]
            
            for filename in preference_files:
                try:
                    domain = filename.replace('.json', '')
                    preference_path = os.path.join(self.preferences_dir, filename)
                    
                    # Read preference data
                    preference_data = await read_json(preference_path)
                    if preference_data:
                        self.extraction_preferences[domain] = preference_data
                        logger.info(f"Loaded extraction preference for {domain}")
                except Exception as e:
                    logger.error(f"Error loading extraction preference {filename}: {str(e)}")
        except Exception as e:
            logger.error(f"Error scanning preferences directory: {str(e)}")
    
    async def _load_extraction_profiles(self):
        """Load all enhanced extraction profiles from the profiles directory"""
        if not os.path.exists(self.extraction_profiles_dir):
            logger.warning(f"Extraction profiles directory {self.extraction_profiles_dir} does not exist")
            return
            
        try:
            profile_files = [f for f in os.listdir(self.extraction_profiles_dir) if f.endswith('.json')]
            
            for filename in profile_files:
                try:
                    domain = filename.replace('.json', '')
                    profile_path = os.path.join(self.extraction_profiles_dir, filename)
                    
                    # Read profile data
                    profile_data = await read_json(profile_path)
                    if profile_data:
                        self.extraction_preferences[domain] = profile_data
                        logger.info(f"Loaded extraction profile for {domain}")
                except Exception as e:
                    logger.error(f"Error loading extraction profile {filename}: {str(e)}")
        except Exception as e:
            logger.error(f"Error scanning extraction profiles directory: {str(e)}")
    
    async def save_extraction_profile(self, domain: str, profile_data: Dict[str, Any]):
        """
        Save an enhanced extraction profile for a domain.
        
        Args:
            domain: Domain to save profile for
            profile_data: Extraction profile data including strategies, preferences, selectors
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Normalize domain
            domain = domain.replace("www.", "").lower()
            
            # Create the profile file path
            profile_path = os.path.join(self.extraction_profiles_dir, f"{domain}.json")
            
            # Add timestamp for auditing
            profile_data["updated_at"] = datetime.datetime.now().isoformat()
            
            # Save to disk
            success = await write_json(profile_path, profile_data)
            
            if success:
                # Update in-memory cache
                self.extraction_preferences[domain] = profile_data
                logger.info(f"Saved extraction profile for {domain}")
                return True
            else:
                logger.error(f"Failed to save extraction profile for {domain}")
                return False
                
        except Exception as e:
            logger.error(f"Error saving extraction profile for {domain}: {str(e)}")
            return False
    
    def get_template_count(self) -> int:
        """
        Get the total number of loaded templates.
        
        Returns:
            Number of templates currently loaded
        """
        return len(self.templates)
    
    async def get_extraction_profile(self, url: str) -> Dict[str, Any]:
        """
        Get the extraction profile for a URL's domain.
        
        Args:
            url: URL to get profile for
            
        Returns:
            Extraction profile dictionary or empty dict if none exists
        """
        try:
            domain = urlparse(url).netloc.replace("www.", "").lower()
            
            # Check if we have a profile for this exact domain
            if domain in self.extraction_preferences:
                return self.extraction_preferences[domain]
            
            # Check if we have a profile for a parent domain
            parent_domain = ".".join(domain.split(".")[-2:])
            if parent_domain in self.extraction_preferences:
                return self.extraction_preferences[parent_domain]
            
            # No profile found
            return {}
            
        except Exception as e:
            logger.error(f"Error getting extraction profile for {url}: {str(e)}")
            return {}
    
    async def create_extraction_template_with_profile(self, url: str, html: str, 
                                                    extraction_results: Dict[str, Any],
                                                    extraction_quality: float = 0.0) -> Dict[str, Any]:
        """
        Create a template that includes extraction profile data for enhanced extraction.
        
        Args:
            url: URL the template applies to
            html: HTML content used for extraction
            extraction_results: Results from extraction
            extraction_quality: Quality score of extraction (0.0-1.0)
            
        Returns:
            Template dictionary with embedded extraction profile
        """
        try:
            # Parse URL to get domain
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.replace("www.", "").lower()
            
            # Parse the HTML
            soup = parse_html(html)
            
            # Extract key selectors used for content extraction
            selectors = await self.content_extractor.extract_key_selectors(soup, extraction_results)
            
            # Create basic template
            template = {
                "domain": domain,
                "url_pattern": self._generate_url_pattern(url),
                "created_at": datetime.datetime.now().isoformat(),
                "updated_at": datetime.datetime.now().isoformat(),
                "extraction_quality": extraction_quality,
                "extraction_selectors": selectors,
                "extraction_strategies": {
                    "css_selector": True,
                    "xpath": True,
                    "semantic": True,
                    "ai_assisted": extraction_quality < 0.8
                },
                "field_mappings": {},
                "sample_data": extraction_results
            }
            
            # Add field mappings based on extraction results
            for field, value in extraction_results.items():
                if field.startswith("_"):  # Skip metadata fields
                    continue
                    
                # Add basic field mapping
                template["field_mappings"][field] = {
                    "selector_type": "auto",
                    "selector": "",
                    "attribute": "text",
                    "post_process": []
                }
            
            # Save the template
            await self.save_template(domain, template)
            
            # Also save as an extraction profile
            profile = {
                "extraction_strategies": template["extraction_strategies"],
                "selectors": selectors,
                "field_mappings": template["field_mappings"],
                "content_types": self._infer_content_types(extraction_results)
            }
            await self.save_extraction_profile(domain, profile)
            
            return template
            
        except Exception as e:
            logger.error(f"Error creating extraction template with profile: {str(e)}")
            return {}
    
    def _infer_content_types(self, extraction_results: Dict[str, Any]) -> List[str]:
        """
        Infer content types from extraction results
        
        Args:
            extraction_results: The results from extraction
            
        Returns:
            List of inferred content types
        """
        content_types = []
        
        # Check for product fields
        if any(f in extraction_results for f in ["price", "sku", "product_id", "inventory"]):
            content_types.append("product")
            
        # Check for article fields
        if any(f in extraction_results for f in ["author", "published_date", "content"]):
            content_types.append("article")
            
        # Check for listing page
        if "items" in extraction_results and isinstance(extraction_results["items"], list):
            content_types.append("listing")
            
        # Default to generic if no specific type detected
        if not content_types:
            content_types.append("generic")
            
        return content_types
    
    async def apply_extraction_profile(self, url: str, html: str) -> Dict[str, Any]:
        """
        Apply an extraction profile to extract data from HTML
        
        Args:
            url: URL of the page
            html: HTML content
            
        Returns:
            Extracted data according to the profile
        """
        try:
            # Get the extraction profile
            profile = await self.get_extraction_profile(url)
            
            if not profile:
                logger.info(f"No extraction profile found for {url}")
                return {}
                
            # Parse the HTML
            soup = parse_html(html)
            
            # Extract data using the content extractor and profile
            extracted_data = await self.content_extractor.extract_with_profile(
                soup, 
                url, 
                profile
            )
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error applying extraction profile: {str(e)}")
            return {}
