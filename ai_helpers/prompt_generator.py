import json
import re
from typing import Dict, Any, List, Optional
from core.ai_service import AIService
from .rule_based_extractor import extract_intent

class PromptGenerator:
    """
    Generates optimized prompts for various AI tasks using the centralized AI service.
    """
    
    def __init__(self, ai_service: Optional[AIService] = None):
        """
        Initialize the PromptGenerator with an AIService instance.
        
        Args:
            ai_service: Optional AIService instance. If None, will attempt to get from service registry.
        """
        self._ai_service = ai_service
        
    @property
    def ai_service(self) -> AIService:
        """Get the AI service, retrieving from registry if needed."""
        if not self._ai_service:
            # Import here to avoid circular imports
            from core.service_registry import ServiceRegistry
            self._ai_service = ServiceRegistry.get_service("ai_service")
            if not self._ai_service:
                raise ValueError("AIService not found in ServiceRegistry")
        return self._ai_service
    
    async def optimize_extraction_prompt(self, user_prompt: str, page_sample: str) -> Dict[str, Any]:
        """
        Enhanced version of the optimize_extraction_prompt function to generate better extraction prompts
        
        Args:
            user_prompt: The user's original extraction request
            page_sample: A sample of the HTML content from the page
            
        Returns:
            A dictionary with optimized extraction parameters
        """
        instruction = f"""
        I need to extract data from a webpage using web scraping. Help me optimize this extraction task.
        
        USER REQUEST: {user_prompt}
        
        SAMPLE PAGE CONTENT:
        {page_sample[:5000]}  # Limit to first 5000 chars to stay within token limits
        
        Please analyze this request and provide:
        1. An optimized extraction instruction that will help an LLM extract exactly what's needed
        2. A list of CSS selectors that might be useful for targeting this data
        3. Key elements to look for in the page structure
        4. A JSON schema that represents the data structure to extract
        
        Format your response as JSON with these keys: 
        "optimized_prompt", "css_selectors", "key_elements", "json_schema"
        """
        
        response = await self.ai_service.generate_response(
            prompt=instruction,
            context={
                "use_cache": True,
                "task_type": "extraction_optimization",
                "quality_priority": 9,  # High priority as this affects all downstream extraction
                "options": {
                    "temperature": 0.3  # Lower temperature for more deterministic results
                }
            }
        )
        
        try:
            # Extract the JSON response
            content = response.get("content", "")
            json_str = self._extract_json_from_text(content)
            
            if not json_str:
                # Fallback to a basic structure if parsing fails
                return {
                    "optimized_prompt": user_prompt,
                    "css_selectors": [],
                    "key_elements": [],
                    "json_schema": {}
                }
                
            result = json.loads(json_str)
            
            # Ensure schema is properly formatted
            if "json_schema" in result and isinstance(result["json_schema"], str):
                try:
                    result["json_schema"] = json.loads(result["json_schema"])
                except:
                    # If schema is not valid JSON, keep as string
                    pass
                
            return result
        except Exception as e:
            print(f"Error parsing AI response: {e}")
            # Fallback to a basic structure if parsing fails
            return {
                "optimized_prompt": user_prompt,
                "css_selectors": [],
                "key_elements": [],
                "json_schema": {}
            }
    
    async def generate_content_filter_instructions(self, url: str, user_request: str, page_sample: str) -> str:
        """
        Dynamically generate content filtering instructions tailored to the specific website
        and extraction request.
        
        Args:
            url: The URL being scraped
            user_request: What the user wants to extract
            page_sample: Sample HTML content from the page
            
        Returns:
            A string containing LLM-friendly filtering instructions
        """
        try:
            instruction = f"""
            Create specific content filtering instructions for extracting data from this webpage:
            
            URL: {url}
            USER REQUEST: {user_request}
            
            PAGE SAMPLE:
            {page_sample[:3000]}
            
            Generate detailed content filtering instructions that will:
            1. Define which page elements contain relevant content for this request
            2. Specify elements that should be excluded (navigation, ads, footers, etc.)
            3. Identify any patterns that might distinguish primary content from boilerplate
            4. Suggest how to handle pagination or "load more" buttons if present
            
            Format your response as concrete, focused instructions for filtering page content.
            Do not include explanations or introductions - just the filtering instructions.
            """
            
            response = await self.ai_service.generate_response(
                prompt=instruction,
                context={
                    "use_cache": True,
                    "task_type": "content_filtering",
                    "quality_priority": 7,
                    "options": {
                        "temperature": 0.3
                    }
                }
            )
            
            return response.get("content", "").strip()
        except Exception as e:
            print(f"Failed to generate content filter instructions: {str(e)}")
            # Return basic fallback instructions
            return f"""
            Focus on elements that might contain {user_request}.
            Ignore navigation menus, headers, footers, sidebars, and advertisements.
            Extract only the main content relevant to the user's request.
            """
    
    async def parse_user_prompt(self, user_prompt: str, current_intent: Dict[str, Any] = None, 
                               pre_processed_intent: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Parses a natural language user prompt using hybrid approach:
        1. First applies rule-based extraction for high-confidence entities
        2. Then uses generative AI to extract remaining structured intent.

        Args:
            user_prompt: The natural language query from the user.
            current_intent: The existing user_intent dictionary (can be empty or partially filled).
            pre_processed_intent: Optional pre-processed intent from rule-based extraction.

        Returns:
            An updated user_intent dictionary enriched with information extracted from the prompt.
        """
        if current_intent is None:
            current_intent = {}
        
        # Use the pre-processed intent if provided, otherwise generate it
        if pre_processed_intent is None:
            pre_processed_intent = extract_intent(user_prompt)
        
        # Merge pre-processed intent with current intent as the starting point
        working_intent = current_intent.copy()
        
        # Prioritize high-confidence rule-based extractions
        if "target_item" in pre_processed_intent and pre_processed_intent["target_item"]:
            working_intent["target_item"] = pre_processed_intent["target_item"]
        
        if "location" in pre_processed_intent and pre_processed_intent["location"]:
            working_intent["location"] = pre_processed_intent["location"]
        
        if "specific_criteria" in pre_processed_intent and pre_processed_intent["specific_criteria"]:
            if "specific_criteria" not in working_intent:
                working_intent["specific_criteria"] = {}
            working_intent["specific_criteria"].update(pre_processed_intent["specific_criteria"])
        
        if "entity_type" in pre_processed_intent and pre_processed_intent["entity_type"]:
            working_intent["entity_type"] = pre_processed_intent["entity_type"]
        
        if "properties" in pre_processed_intent and pre_processed_intent["properties"]:
            if "properties" not in working_intent:
                working_intent["properties"] = []
            
            # Add properties without duplicates
            existing_props = set(working_intent["properties"])
            for prop in pre_processed_intent["properties"]:
                if prop not in existing_props:
                    working_intent["properties"].append(prop)
                    existing_props.add(prop)
        
        if "keywords" in pre_processed_intent and pre_processed_intent["keywords"]:
            if "keywords" not in working_intent:
                working_intent["keywords"] = []
            
            # Add keywords without duplicates
            existing_keywords = set(working_intent["keywords"])
            for keyword in pre_processed_intent["keywords"]:
                if keyword not in existing_keywords:
                    working_intent["keywords"].append(keyword)
                    existing_keywords.add(keyword)

        # Define the expected structure for user_intent based on observed usage
        intent_schema = {
            "type": "object",
            "properties": {
                "original_query": {"type": "string", "description": "The original user prompt."},
                "target_item": {"type": "string", "description": "The main item or entity the user is searching for (e.g., 'homes', 'laptops', 'jobs')."},
                "location": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "state": {"type": "string"},
                        "zip_code": {"type": "string"}
                    },
                    "description": "Location information in structured format."
                },
                "specific_criteria": {
                    "type": "object",
                    "description": "Specific criteria as key-value pairs (e.g., price ranges, features, filters)."
                },
                "entity_type": {"type": "string", "description": "The type of entity to extract (e.g., 'property', 'product', 'job')."},
                "properties": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific properties or attributes to extract for each entity (e.g., 'price', 'address', 'company_name')."
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Additional keywords relevant to the search."
                }
            },
            "required": ["original_query", "target_item"]
        }

        instruction = f"""
        Analyze the following user prompt and extract structured information to populate a user intent dictionary.
        The goal is to understand what the user wants to find or extract from a website.

        USER PROMPT: "{user_prompt}"

        RULE-BASED EXTRACTION RESULTS:
        {json.dumps(pre_processed_intent, indent=2)}

        CURRENT INTENT (may be empty or partially filled):
        {json.dumps(working_intent, indent=2)}

        Desired JSON Schema for the output:
        {json.dumps(intent_schema, indent=2)}

        Instructions:
        1. The rule-based extraction has already identified high-confidence entities. PRIORITIZE these rule-extracted elements but refine or add to them based on your understanding.
        2. Focus especially on elements that the rule-based extraction might have missed or misinterpreted.
        3. For the target_item, check if the rule-based extraction is correct. If it's missing or seems incorrect, provide a better one.
        4. For location information, use the structured format from the rule-based extraction if available, otherwise extract it.
        5. For specific_criteria, maintain the structured key-value format from rule-based extraction, and add any missing criteria.
        6. The entity_type should be a single string value (not an array of entity_types).
        7. For properties, enhance the list with additional relevant properties that should be extracted.
        8. Keep the original_query field as the exact user prompt.
        9. Return ONLY the final JSON object representing the enriched user intent, adhering strictly to the provided schema. Do not include explanations or markdown formatting.
        """

        try:
            response = await self.ai_service.generate_response(
                prompt=instruction,
                context={
                    "use_cache": True,
                    "task_type": "intent_parsing",
                    "quality_priority": 8,
                    "options": {
                        "temperature": 0.2
                    }
                }
            )
            
            content = response.get("content", "")
            json_str = self._extract_json_from_text(content)
            
            if not json_str:
                return self._create_fallback_intent(user_prompt, working_intent)
                
            extracted_intent = json.loads(json_str)

            # Basic validation against schema keys
            final_intent = working_intent.copy()  # Start with working intent
            final_intent.update(extracted_intent)  # Update with extracted values, overwriting if keys conflict
            final_intent["original_query"] = user_prompt  # Ensure original query is preserved

            # Ensure required fields are present, provide defaults if necessary
            if "target_item" not in final_intent or not final_intent["target_item"]:
                final_intent["target_item"] = "information"  # Default if AI fails to extract
            if "specific_criteria" not in final_intent:
                final_intent["specific_criteria"] = {}
            if "properties" not in final_intent:
                final_intent["properties"] = []
            if "keywords" not in final_intent:
                final_intent["keywords"] = []

            return final_intent

        except Exception as e:
            print(f"Error parsing user prompt with AI: {e}")
            return self._create_fallback_intent(user_prompt, working_intent)
    
    async def generate_context_aware_variations(
        self,
        user_prompt: str, 
        intent: Dict[str, Any] = None,
        site_content: str = None,
        site_url: str = None,
        site_type: str = None,
        num_variations: int = 5
    ) -> Dict[str, Any]:
        """
        Generate context-aware search term variations based on website content and user intent.
        
        Args:
            user_prompt: Original user search query
            intent: The parsed user intent dictionary
            site_content: A sample of the site's content (HTML or text)
            site_url: The URL of the site being searched
            site_type: The type of website (e.g., 'ecommerce', 'realestate')
            num_variations: Number of variations to generate
            
        Returns:
            A dictionary with various search term variations optimized for the context
        """
        # If no intent is provided, generate one
        if intent is None:
            parser = PromptGenerator(self._ai_service)
            intent = await parser.parse_user_prompt(user_prompt)
        
        try:
            # Extract domain from URL if available
            domain_info = ""
            if site_url:
                from urllib.parse import urlparse
                parsed_url = urlparse(site_url)
                domain = parsed_url.netloc
                domain_info = f"Website domain: {domain}\n"
            
            # Add site type info if available
            site_type_info = ""
            if site_type:
                site_type_info = f"Website type: {site_type}\n"
            
            # Add content sample if available (limited to prevent token overload)
            content_sample = ""
            if site_content:
                # Extract visible text content since HTML might be too verbose
                from bs4 import BeautifulSoup
                try:
                    soup = BeautifulSoup(site_content, 'html.parser')
                    # Get text and filter out empty lines
                    text_content = [line.strip() for line in soup.get_text().split('\n') if line.strip()]
                    # Join limited number of lines to avoid token limits
                    content_sample = "Website content sample:\n" + "\n".join(text_content[:20]) + "\n"
                except Exception as e:
                    # Fallback to raw content if BeautifulSoup fails
                    content_sample = "Website content sample:\n" + site_content[:500] + "\n"
            
            # Construct the prompt for the AI
            instruction = f"""
            I need to generate context-aware search term variations for a specific website's search function.
            
            Original user query: "{user_prompt}"
            {domain_info}{site_type_info}{content_sample}
            User intent details: {json.dumps(intent, indent=2)}
            
            Based on this context, please generate:
            
            1. A primary optimized search term that would work best specifically for this website
            2. {num_variations} alternative variations tailored to this specific website's search functionality
            3. A set of keywords that are likely to be recognized by this specific website's search engine
            4. A formal/advanced search query using site-specific operators or syntax if applicable
            5. A relaxed/broad version of the query that would still work on this site but capture more results
            
            For each variation, consider:
            - The specific terminology and language used on this website
            - Any domain-specific jargon or naming conventions
            - How this website's search algorithm likely works based on its type
            - Common search patterns for this type of website
            - The actual content and structure of the site
            
            Format your response as a JSON object with these keys:
            - primary_term: The best search term for this specific site
            - variations: Array of alternative search terms tailored to this site
            - site_specific_keywords: Array of keywords likely recognized by this site
            - formal_query: A more formal/precise query using site syntax if applicable
            - broad_query: A more generalized version for capturing more results
            - search_tips: Tips specific to searching on this type of site
            """
            
            response = await self.ai_service.generate_response(
                prompt=instruction,
                context={
                    "use_cache": True,
                    "task_type": "search_optimization",
                    "quality_priority": 7,
                    "options": {
                        "temperature": 0.7  # Higher temperature for creative variations
                    }
                }
            )
            
            content = response.get("content", "")
            json_str = self._extract_json_from_text(content)
            
            if not json_str:
                # Fallback to a basic structure if AI fails
                return {
                    "primary_term": user_prompt,
                    "variations": [user_prompt],
                    "site_specific_keywords": intent.get("keywords", []),
                    "formal_query": user_prompt,
                    "broad_query": user_prompt,
                    "search_tips": ["Use simple and clear search terms"],
                    "source": "fallback",
                    "original_query": user_prompt
                }
                
            result = json.loads(json_str)
            
            # Add metadata about the generation
            result["source"] = "ai-generated"
            result["original_query"] = user_prompt
            if site_url:
                result["site_url"] = site_url
            if site_type:
                result["site_type"] = site_type
                
            return result
                    
        except Exception as e:
            print(f"Error generating context-aware variations: {str(e)}")
            # Fallback to a basic structure if AI fails
            return {
                "primary_term": user_prompt,
                "variations": [user_prompt],
                "site_specific_keywords": intent.get("keywords", []),
                "formal_query": user_prompt,
                "broad_query": user_prompt,
                "search_tips": ["Use simple and clear search terms"],
                "source": "fallback",
                "original_query": user_prompt
            }
    
    async def generate_iterative_search_refinement(
        self,
        initial_query: str,
        search_results_sample: str,
        intent: Dict[str, Any] = None,
        site_url: str = None,
        iteration: int = 1,
        max_iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Iteratively refine search terms based on previous search results.
        
        Args:
            initial_query: The original search query
            search_results_sample: A sample of the search results from the previous query
            intent: The parsed user intent dictionary
            site_url: The URL of the site being searched
            iteration: Current iteration number (starting from 1)
            max_iterations: Maximum number of refinement iterations
            
        Returns:
            A dictionary with refined search terms and analysis
        """
        # If no intent is provided, generate one
        if intent is None:
            parser = PromptGenerator(self._ai_service)
            intent = await parser.parse_user_prompt(initial_query)
        
        try:
            # Extract domain from URL if available
            domain_info = ""
            if site_url:
                from urllib.parse import urlparse
                parsed_url = urlparse(site_url)
                domain = parsed_url.netloc
                domain_info = f"Website: {domain}\n"
            
            # Process search results sample (extract text if HTML)
            processed_results = search_results_sample
            if "<html" in search_results_sample.lower():
                from bs4 import BeautifulSoup
                try:
                    soup = BeautifulSoup(search_results_sample, 'html.parser')
                    # Get text and filter out empty lines
                    text_content = [line.strip() for line in soup.get_text().split('\n') if line.strip()]
                    # Join limited number of lines to avoid token limits
                    processed_results = "\n".join(text_content[:50])
                except Exception:
                    # Limit raw HTML if BeautifulSoup fails
                    processed_results = search_results_sample[:2000]
            
            # Adjust prompt based on iteration
            iteration_context = ""
            if iteration > 1:
                iteration_context = f"This is refinement iteration {iteration} of {max_iterations}. "
                if iteration == max_iterations:
                    iteration_context += "This is the final refinement, so focus on precision."
            
            # Construct the prompt for the AI
            instruction = f"""
            I need to refine a search query based on the results it returned.
            
            {domain_info}
            Original search query: "{initial_query}"
            User intent details: {json.dumps(intent, indent=2)}
            
            {iteration_context}
            
            Current search results:
            ```
            {processed_results[:2000]}
            ```
            
            Based on the original intent and these search results, please analyze:
            
            1. Whether the current results match the user's intent
            2. What aspects of the search query need refinement
            3. How to modify the query to better match the intent
            
            Then, please provide:
            
            1. A refined search query that would better match the intent
            2. 2-3 alternative query variations to try
            3. An analysis of what's working/not working in the current results
            4. Strategy adjustments for this specific search
            
            If the results are already excellent and match the intent perfectly, you can indicate that refinement isn't needed.
            
            Format your response as a JSON object with these keys:
            - analysis: Your assessment of current results vs. intent
            - needs_refinement: Boolean indicating if refinement is needed
            - refined_query: The improved search query
            - alternative_queries: Array of alternative queries to try
            - strategy_adjustments: Array of strategy changes to improve results
            """
            
            response = await self.ai_service.generate_response(
                prompt=instruction,
                context={
                    "use_cache": True,
                    "task_type": "search_refinement",
                    "quality_priority": 7,
                    "options": {
                        "temperature": 0.5
                    }
                }
            )
            
            content = response.get("content", "")
            json_str = self._extract_json_from_text(content)
            
            if not json_str:
                # Fallback if JSON extraction fails
                return {
                    "analysis": "Unable to analyze results due to an error.",
                    "needs_refinement": True,
                    "refined_query": initial_query,
                    "alternative_queries": [initial_query],
                    "strategy_adjustments": ["Simplify the search terms", "Try more specific keywords"],
                    "iteration": iteration,
                    "max_iterations": max_iterations,
                    "original_query": initial_query
                }
                
            result = json.loads(json_str)
            
            # Add metadata about the refinement process
            result["iteration"] = iteration
            result["max_iterations"] = max_iterations
            result["original_query"] = initial_query
            if site_url:
                result["site_url"] = site_url
                
            return result
                    
        except Exception as e:
            print(f"Error generating search refinement: {str(e)}")
            # Fallback to a basic structure if AI fails
            return {
                "analysis": f"Unable to analyze results due to an error: {str(e)}",
                "needs_refinement": True,
                "refined_query": initial_query,
                "alternative_queries": [initial_query],
                "strategy_adjustments": ["Simplify the search terms", "Try more specific keywords"],
                "iteration": iteration,
                "max_iterations": max_iterations,
                "original_query": initial_query
            }
    
    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """Extract JSON from response text."""
        if not text:
            return None
        
        # Try to find JSON block in markdown
        if "```json" in text:
            pattern = r"```json\s*(.+?)\s*```"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1)
        elif "```" in text:
            pattern = r"```\s*(.+?)\s*```"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1)
        
        # Try to find JSON-like structure
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            return text[start_idx:end_idx+1]
        
        return None
    
    def _create_fallback_intent(self, query: str, working_intent: Dict[str, Any]) -> Dict[str, Any]:
        """Create a fallback intent when AI extraction fails."""
        fallback_intent = working_intent.copy()
        fallback_intent["original_query"] = query
        if "target_item" not in fallback_intent or not fallback_intent["target_item"]:
            fallback_intent["target_item"] = "information"  # Default if AI fails to extract
        if "specific_criteria" not in fallback_intent:
            fallback_intent["specific_criteria"] = {}
        if "properties" not in fallback_intent:
            fallback_intent["properties"] = []
        if "keywords" not in fallback_intent:
            fallback_intent["keywords"] = []
        return fallback_intent

    def generate_url_discovery_prompt(self, user_query: str, base_url: Optional[str] = None) -> str:
        """
        Generates a realistic prompt for the LLM to suggest relevant URLs based on known patterns.
        This version uses curated, known-working URL patterns to avoid 404 errors.

        Args:
            user_query: The user's natural language query (e.g., "latest news on AI").
            base_url: Optional. A base URL if the user provided a generic site (e.g., "https://example.com").

        Returns:
            A string prompt for the LLM to generate likely relevant URLs.
        """
        
        # Define known working URL patterns for different query types
        tech_patterns = [
            "https://techcrunch.com/",
            "https://www.theverge.com/",
            "https://arstechnica.com/",
            "https://www.wired.com/",
            "https://news.ycombinator.com/"
        ]
        
        news_patterns = [
            "https://www.reuters.com/",
            "https://www.bbc.com/news",
            "https://www.cnn.com/",
            "https://www.npr.org/",
            "https://apnews.com/"
        ]
        
        business_patterns = [
            "https://www.bloomberg.com/",
            "https://www.wsj.com/",
            "https://www.ft.com/",
            "https://fortune.com/",
            "https://www.cnbc.com/"
        ]
        
        if base_url:
            instruction = f"""
            TASK: Suggest likely URLs on {base_url} for "{user_query}"
            
            APPROACH: Use the main/homepage URL since we cannot verify subpages exist.
            
            STRATEGY:
            1. Use the provided base URL as the primary target
            2. If it's a major site, suggest the homepage for broad content coverage
            3. For e-commerce sites, use main category pages
            4. For news sites, use the main news section
            
            SAFETY APPROACH: Focus on main sections that are likely to exist.
            """
        else:
            # Categorize the query to suggest appropriate working URLs
            query_lower = user_query.lower()
            
            if any(tech_word in query_lower for tech_word in ['tech', 'ai', 'software', 'computer', 'startup', 'innovation']):
                suggested_urls = tech_patterns[:3]
                category = "technology"
            elif any(business_word in query_lower for business_word in ['business', 'market', 'finance', 'stock', 'economy']):
                suggested_urls = business_patterns[:3]
                category = "business/finance"
            else:
                suggested_urls = news_patterns[:3]
                category = "general news"
            
            instruction = f"""
            TASK: Suggest authoritative URLs for "{user_query}"
            
            QUERY CATEGORY: {category}
            
            RECOMMENDED WORKING URLs (known to be accessible):
            {chr(10).join(f"- {url}" for url in suggested_urls)}
            
            STRATEGY:
            1. Use these proven, accessible URLs as your primary suggestions
            2. These sites are major, reliable sources that rarely have 404 errors
            3. Focus on homepage/main section URLs rather than deep links
            4. These URLs are selected based on query relevance and site reliability
            
            URL SELECTION LOGIC:
            - Technology queries → Tech news sites (TechCrunch, The Verge, etc.)
            - Business queries → Financial news sites (Bloomberg, WSJ, etc.)  
            - General queries → Major news outlets (Reuters, BBC, etc.)
            """

        return f"""
        {instruction.strip()}

        RESPONSE FORMAT REQUIREMENTS:
        - Return URLs that are KNOWN TO WORK and accessible
        - Format as JSON object with "urls" key containing array of URL strings
        - Include "reasoning" key explaining the URL selection strategy
        - Prioritize site reliability over deep content matching
        - Use main sections/homepages when in doubt about specific pages

        Example response structure:
        {{
            "urls": ["https://techcrunch.com/", "https://www.theverge.com/", "https://arstechnica.com/"],
            "reasoning": "Selected major technology news sites known for reliable access and comprehensive coverage"
        }}

        CRITICAL: Only suggest URLs from major, well-established websites to avoid 404 errors.
        """
        
    def generate_url_discovery_prompt_with_content_verification(self, user_query: str, base_url: Optional[str] = None, content_requirements: List[str] = None) -> str:
        """
        Advanced prompt generation with specific content verification requirements.
        
        Args:
            user_query: The user's natural language query
            base_url: Optional base URL to constrain search
            content_requirements: Specific content elements that must be present on each URL
            
        Returns:
            Enhanced prompt with detailed content verification requirements
        """
        content_checks = ""
        if content_requirements:
            content_checks = f"""
            SPECIFIC CONTENT REQUIREMENTS:
            Each URL must contain the following elements:
            {chr(10).join(f"- {req}" for req in content_requirements)}
            
            CONTENT VERIFICATION PROCESS:
            1. Access each candidate URL
            2. Scan the page content for each required element
            3. Verify all required elements are present and substantive
            4. Confirm content is not just metadata or peripheral mentions
            5. Ensure content is in the main body of the page, not just headers/footers
            """
        
        base_prompt = self.generate_url_discovery_prompt(user_query, base_url)
        
        if content_checks:
            # Insert content requirements before the response format section
            parts = base_prompt.split("RESPONSE FORMAT REQUIREMENTS:")
            enhanced_prompt = f"""
            {parts[0]}
            
            {content_checks}
            
            RESPONSE FORMAT REQUIREMENTS:
            {parts[1] if len(parts) > 1 else ""}
            """
            return enhanced_prompt
        
        return base_prompt

# Legacy functions for backward compatibility (can be removed once all calls are updated)
async def optimize_extraction_prompt(user_prompt, page_sample):
    """Legacy function that uses the PromptGenerator class."""
    generator = PromptGenerator()
    return await generator.optimize_extraction_prompt(user_prompt, page_sample)

async def generate_content_filter_instructions(url, user_request, page_sample):
    """Legacy function that uses the PromptGenerator class."""
    generator = PromptGenerator()
    return await generator.generate_content_filter_instructions(url, user_request, page_sample)

async def parse_user_prompt(user_prompt, current_intent=None, pre_processed_intent=None):
    """Legacy function that uses the PromptGenerator class."""
    generator = PromptGenerator()
    return await generator.parse_user_prompt(user_prompt, current_intent, pre_processed_intent)
    
async def generate_context_aware_variations(
    user_prompt, 
    intent=None,
    site_content=None,
    site_url=None,
    site_type=None,
    num_variations=5
):
    """Legacy function that uses the PromptGenerator class."""
    generator = PromptGenerator()
    return await generator.generate_context_aware_variations(
        user_prompt, intent, site_content, site_url, site_type, num_variations
    )
    
async def generate_iterative_search_refinement(
    initial_query,
    search_results_sample,
    intent=None,
    site_url=None,
    iteration=1,
    max_iterations=3
):
    """Legacy function that uses the PromptGenerator class."""
    generator = PromptGenerator()
    return await generator.generate_iterative_search_refinement(
        initial_query, search_results_sample, intent, site_url, iteration, max_iterations
    )
