"""
Response Parser Module

This module contains utilities for parsing and extracting structured data from AI responses.
"""

import json
import re
from typing import Any, Dict, List, Optional, Union
from bs4 import BeautifulSoup
from core.ai_service import AIService

class ResponseParser:
    """
    Parses and extracts structured data from various types of AI responses.
    Uses the centralized AI service for additional processing when needed.
    """
    
    def __init__(self, ai_service: Optional[AIService] = None):
        """
        Initialize the ResponseParser with an AIService instance.
        
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
    
    async def extract_structured_data(
        self, 
        ai_response: str, 
        expected_schema: Dict[str, Any], 
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Extract structured data from AI response based on expected schema.
        If the initial response doesn't match the schema, retry with explicit schema guidance.
        
        Args:
            ai_response: The raw response from the AI
            expected_schema: JSON schema describing the expected data structure
            context: Additional context that might help with extraction
            
        Returns:
            Structured data extracted from the response
        """
        # First attempt: Try direct JSON extraction
        extracted_json = self._extract_json_from_text(ai_response)
        
        if extracted_json:
            try:
                parsed_data = json.loads(extracted_json)
                # Validate against schema keys (basic validation)
                if self._validate_against_schema(parsed_data, expected_schema):
                    return parsed_data
            except json.JSONDecodeError:
                # If JSON parsing fails, continue to AI-assisted extraction
                pass
                
        # Second attempt: Use AI to help extract structured data
        instruction = f"""
        I need to extract structured data from this response according to a specific schema.
        
        Original AI Response:
        ```
        {ai_response}
        ```
        
        Expected JSON Schema:
        ```
        {json.dumps(expected_schema, indent=2)}
        ```
        
        Please extract the structured data from the response and format it according to the provided schema.
        If some schema fields aren't present in the response, omit them or use null values.
        Return ONLY the parsed JSON object without any explanation, markdown formatting, or additional text.
        """
        
        if context:
            instruction += f"\n\nAdditional context that might help with extraction:\n{json.dumps(context, indent=2)}"
        
        response = await self.ai_service.generate_response(
            prompt=instruction,
            context={
                "use_cache": True,
                "task_type": "response_parsing",
                "quality_priority": 8,
                "options": {
                    "temperature": 0.1  # Low temperature for deterministic response
                }
            }
        )
        
        content = response.get("content", "")
        extracted_json = self._extract_json_from_text(content)
        
        if extracted_json:
            try:
                parsed_data = json.loads(extracted_json)
                return parsed_data
            except json.JSONDecodeError:
                pass
        
        # If all extraction attempts fail, return an empty object
        return {}
    
    async def normalize_extraction_results(
        self, 
        raw_results: Union[str, Dict[str, Any], List[Dict[str, Any]]], 
        target_schema: Dict[str, Any] = None,
        source_url: str = None,
        extraction_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Normalize extraction results to a consistent format based on a target schema.
        Can handle various input formats and enhance data quality.
        
        Args:
            raw_results: Raw extraction results (text, JSON object, or list of objects)
            target_schema: Target schema to normalize to (optional)
            source_url: URL source of the data (for context)
            extraction_context: Additional context about the extraction
            
        Returns:
            Normalized data in a consistent format
        """
        # Convert string results to dictionary if possible
        if isinstance(raw_results, str):
            extracted_json = self._extract_json_from_text(raw_results)
            if extracted_json:
                try:
                    raw_results = json.loads(extracted_json)
                except json.JSONDecodeError:
                    # If parsing fails, keep as string
                    pass
        
        # If still a string, use AI to try to extract structured data
        if isinstance(raw_results, str):
            schema_instruction = ""
            if target_schema:
                schema_instruction = f"""
                Target Schema:
                ```
                {json.dumps(target_schema, indent=2)}
                ```
                """
                
            instruction = f"""
            I need to normalize and convert this raw extraction result into a structured format.
            
            Raw Extraction Text:
            ```
            {raw_results[:5000]}  # Limit to prevent token overflow
            ```
            
            {schema_instruction}
            
            Please extract structured data from this raw text and format it as a proper JSON object.
            If there appears to be multiple items/records, organize them as an array of objects.
            Include all relevant data points found in the text.
            
            Return ONLY the normalized JSON without any explanation or markdown formatting.
            """
            
            if source_url or extraction_context:
                context_info = "Additional Context:\n"
                if source_url:
                    context_info += f"Source URL: {source_url}\n"
                if extraction_context:
                    context_info += f"Extraction Context: {json.dumps(extraction_context, indent=2)}\n"
                instruction += f"\n{context_info}"
            
            response = await self.ai_service.generate_response(
                prompt=instruction,
                context={
                    "use_cache": True,
                    "task_type": "data_normalization",
                    "quality_priority": 7,
                    "options": {
                        "temperature": 0.2
                    }
                }
            )
            
            content = response.get("content", "")
            extracted_json = self._extract_json_from_text(content)
            
            if extracted_json:
                try:
                    normalized_results = json.loads(extracted_json)
                    return normalized_results
                except json.JSONDecodeError:
                    # If parsing fails, create a simple wrapper structure
                    return {"raw_text": raw_results, "normalized": False}
            else:
                # If no JSON found, create a simple wrapper structure
                return {"raw_text": raw_results, "normalized": False}
        
        # Handle list of results
        if isinstance(raw_results, list) and target_schema:
            # If we have a list but the target schema is for a single object,
            # wrap the list in a container object
            if not target_schema.get("type") == "array":
                return {"items": raw_results, "count": len(raw_results)}
        
        # If raw_results is already a dict or list and no target schema,
        # just return it as is
        return raw_results
    
    async def improve_extraction_quality(
        self, 
        extracted_data: Dict[str, Any], 
        raw_html: str = None, 
        intent: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Improves the quality of extraction results by fixing issues and enhancing data.
        
        Args:
            extracted_data: The data extracted so far
            raw_html: Optional raw HTML that was the source of extraction
            intent: User intent dictionary with extraction goals
            
        Returns:
            Improved extraction results
        """
        if not extracted_data:
            return extracted_data
        
        # Try rule-based improvements first
        improved_data = self._apply_rule_based_improvements(extracted_data)
        
        # Check if we need AI-based improvements (complex issues)
        needs_ai_improvement = self._check_needs_ai_improvement(improved_data)
        
        if needs_ai_improvement and raw_html:
            # Prepare context for AI improvement
            html_sample = raw_html[:5000] if raw_html else ""  # Limit HTML size
            intent_json = json.dumps(intent, indent=2) if intent else "{}"
            
            instruction = f"""
            I need to improve the quality of this extraction result. Please analyze the current extracted data 
            and the original source HTML, then produce an enhanced version with these improvements:
            
            1. Fix any incorrectly extracted fields
            2. Normalize inconsistent formatting (e.g. prices, dates)
            3. Merge split fields that should be together
            4. Add missing data that's present in the HTML but missing in the extraction
            5. Remove duplicate or redundant information
            6. Ensure all field values use appropriate data types
            
            Current Extraction:
            ```
            {json.dumps(improved_data, indent=2)}
            ```
            
            Original Source HTML (sample):
            ```
            {html_sample}
            ```
            
            User Intent/Extraction Goals:
            ```
            {intent_json}
            ```
            
            Return an improved version of the extraction that maintains the same structure but with better quality data.
            Return ONLY the improved JSON without any explanation.
            """
            
            response = await self.ai_service.generate_response(
                prompt=instruction,
                context={
                    "use_cache": True,
                    "task_type": "extraction_improvement",
                    "quality_priority": 8,
                    "options": {
                        "temperature": 0.3
                    }
                }
            )
            
            content = response.get("content", "")
            extracted_json = self._extract_json_from_text(content)
            
            if extracted_json:
                try:
                    ai_improved_data = json.loads(extracted_json)
                    return ai_improved_data
                except json.JSONDecodeError:
                    # If parsing fails, return the rule-based improvements
                    return improved_data
        
        return improved_data
    
    async def reconcile_extraction_results(
        self, 
        extractions: List[Dict[str, Any]], 
        strategy: str = "merge"
    ) -> Dict[str, Any]:
        """
        Reconcile multiple extraction results into a single coherent result.
        
        Args:
            extractions: List of extraction results to reconcile
            strategy: Reconciliation strategy ('merge', 'best', 'consensus')
            
        Returns:
            A single reconciled extraction result
        """
        if not extractions:
            return {}
        
        if len(extractions) == 1:
            return extractions[0]
        
        # For simple merging strategy, use rule-based approach
        if strategy == "merge":
            merged_result = {}
            for extraction in extractions:
                for key, value in extraction.items():
                    if key not in merged_result or not merged_result[key]:
                        merged_result[key] = value
            return merged_result
        
        # For more complex strategies, use AI assistance
        instruction = f"""
        I need to reconcile multiple extraction results into a single coherent result.
        
        Extraction Results:
        ```
        {json.dumps(extractions, indent=2)}
        ```
        
        Reconciliation Strategy: {strategy}
        
        If strategy is 'best', choose the most complete and accurate extraction.
        If strategy is 'consensus', use the most commonly extracted values across results.
        If strategy is 'complementary', take the best aspects of each extraction to create a more complete result.
        
        Please analyze the extraction results and produce a single reconciled result that:
        1. Resolves any conflicts between the extractions
        2. Combines complementary information
        3. Chooses the most reliable data when there are discrepancies
        4. Has a consistent structure matching the general pattern of the input extractions
        
        Return ONLY the reconciled JSON without any explanation.
        """
        
        response = await self.ai_service.generate_response(
            prompt=instruction,
            context={
                "use_cache": True,
                "task_type": "extraction_reconciliation",
                "quality_priority": 7,
                "options": {
                    "temperature": 0.2
                }
            }
        )
        
        content = response.get("content", "")
        extracted_json = self._extract_json_from_text(content)
        
        if extracted_json:
            try:
                reconciled_data = json.loads(extracted_json)
                return reconciled_data
            except json.JSONDecodeError:
                # If AI parsing fails, fallback to simple merge
                return self.reconcile_extraction_results([extractions[0]], strategy="merge")
        
        # If AI fails completely, fallback to first extraction
        return extractions[0]
    
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
        
        # Try to find JSON-like structure with balanced braces
        # This handles cases where the whole response is JSON without markdown formatting
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            potential_json = text[start_idx:end_idx+1]
            try:
                # Validate that it's actually valid JSON
                json.loads(potential_json)
                return potential_json
            except json.JSONDecodeError:
                pass
        
        # Try to find array JSON structure
        start_idx = text.find('[')
        end_idx = text.rfind(']')
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            potential_json = text[start_idx:end_idx+1]
            try:
                # Validate that it's actually valid JSON
                json.loads(potential_json)
                return potential_json
            except json.JSONDecodeError:
                pass
        
        return None

    def parse_urls_from_llm_response(self, llm_response: str) -> List[str]:
        """
        Enhanced URL parsing that supports both legacy and new verification-enhanced formats.
        Handles both simple {"urls": [...]} format and enhanced {"urls": [...], "verification_notes": "..."} format.

        Args:
            llm_response: The raw text response from the LLM.

        Returns:
            A list of extracted URLs.

        Raises:
            ValueError: If no valid JSON is found or if the "urls" key is missing/invalid.
        """
        extracted_json_str = self._extract_json_from_text(llm_response)

        if not extracted_json_str:
            raise ValueError("No JSON object found in LLM response.")

        try:
            parsed_data = json.loads(extracted_json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from LLM response: {e}")

        urls = parsed_data.get("urls")
        if not isinstance(urls, list) or not all(isinstance(url, str) for url in urls):
            raise ValueError("JSON response does not contain a valid list of 'urls'.")

        return urls
    
    def parse_enhanced_url_response(self, llm_response: str) -> Dict[str, Any]:
        """
        Parse enhanced URL discovery response that includes verification notes and metadata.
        
        Args:
            llm_response: The raw text response from the LLM with enhanced format
            
        Returns:
            Dictionary containing URLs, verification notes, and other metadata
            
        Raises:
            ValueError: If parsing fails
        """
        extracted_json_str = self._extract_json_from_text(llm_response)

        if not extracted_json_str:
            raise ValueError("No JSON object found in enhanced LLM response.")

        try:
            parsed_data = json.loads(extracted_json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from enhanced LLM response: {e}")

        # Validate that we have URLs
        urls = parsed_data.get("urls")
        if not isinstance(urls, list) or not all(isinstance(url, str) for url in urls):
            raise ValueError("Enhanced response does not contain a valid list of 'urls'.")

        # Extract additional metadata
        result = {
            "urls": urls,
            "verification_notes": parsed_data.get("verification_notes", parsed_data.get("reasoning", "")),
            "total_urls": len(urls),
            "response_type": "enhanced_verified" if ("verification_notes" in parsed_data or "reasoning" in parsed_data) else "standard"
        }
        
        # Include any additional fields that might be present
        for key, value in parsed_data.items():
            if key not in result:
                result[key] = value
                
        return result
    
    def _validate_against_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """
        Perform basic validation of data against a schema.
        This is a simplified validator that checks required fields and types.
        
        Args:
            data: The data to validate
            schema: The schema to validate against
            
        Returns:
            True if data matches schema, False otherwise
        """
        # Get required fields from schema
        required_fields = schema.get("required", [])
        
        # Check if all required fields are present
        for field in required_fields:
            if field not in data:
                return False
        
        # Check property types if specified
        properties = schema.get("properties", {})
        for prop_name, prop_spec in properties.items():
            if prop_name in data and "type" in prop_spec:
                # Skip null values
                if data[prop_name] is None:
                    continue
                
                # Check type
                expected_type = prop_spec["type"]
                if expected_type == "string" and not isinstance(data[prop_name], str):
                    return False
                elif expected_type == "number" and not isinstance(data[prop_name], (int, float)):
                    return False
                elif expected_type == "integer" and not isinstance(data[prop_name], int):
                    return False
                elif expected_type == "boolean" and not isinstance(data[prop_name], bool):
                    return False
                elif expected_type == "array" and not isinstance(data[prop_name], list):
                    return False
                elif expected_type == "object" and not isinstance(data[prop_name], dict):
                    return False
        
        return True
    
    def _apply_rule_based_improvements(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply rule-based improvements to extracted data.
        
        Args:
            data: The data to improve
            
        Returns:
            Improved data
        """
        result = data.copy()
        
        # Process each field
        for key, value in result.items():
            # Handle nested dictionaries
            if isinstance(value, dict):
                result[key] = self._apply_rule_based_improvements(value)
            
            # Handle lists of dictionaries
            elif isinstance(value, list) and all(isinstance(item, dict) for item in value):
                result[key] = [self._apply_rule_based_improvements(item) for item in value]
            
            # Handle price formatting
            elif isinstance(value, str) and "price" in key.lower():
                result[key] = self._normalize_price(value)
            
            # Handle date formatting
            elif isinstance(value, str) and any(date_key in key.lower() for date_key in ["date", "time", "published", "created"]):
                result[key] = self._normalize_date(value)
        
        return result
    
    def _normalize_price(self, price_str: str) -> str:
        """Normalize price string to consistent format."""
        if not price_str:
            return price_str
        
        # Remove non-digit characters except for decimal point and commas
        # Keep dollar/euro/pound signs
        currency_symbol = ""
        if "$" in price_str:
            currency_symbol = "$"
        elif "€" in price_str:
            currency_symbol = "€"
        elif "£" in price_str:
            currency_symbol = "£"
        
        # Extract digits and decimal point
        digits_only = ''.join(char for char in price_str if char.isdigit() or char in ['.', ','])
        
        # Handle comma as decimal separator in some locales
        if ',' in digits_only and '.' not in digits_only:
            digits_only = digits_only.replace(',', '.')
        else:
            # Remove commas used as thousand separators
            digits_only = digits_only.replace(',', '')
        
        # Ensure proper decimal format
        try:
            price_value = float(digits_only)
            formatted_price = f"{price_value:.2f}"
            return f"{currency_symbol}{formatted_price}"
        except ValueError:
            return price_str
    
    def _normalize_date(self, date_str: str) -> str:
        """Normalize date string to ISO format when possible."""
        if not date_str:
            return date_str
        
        # Try to parse common date formats
        import dateutil.parser
        try:
            parsed_date = dateutil.parser.parse(date_str)
            return parsed_date.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            return date_str
    
    def _check_needs_ai_improvement(self, data: Dict[str, Any]) -> bool:
        """
        Check if the data needs AI-based improvement.
        
        Args:
            data: The data to check
            
        Returns:
            True if AI improvement is needed, False otherwise
        """
        # Check for empty or null values
        empty_values = 0
        total_values = 0
        
        def count_empty_values(d):
            nonlocal empty_values, total_values
            for k, v in d.items():
                if isinstance(v, dict):
                    count_empty_values(v)
                elif isinstance(v, list) and v and all(isinstance(item, dict) for item in v):
                    for item in v:
                        count_empty_values(item)
                else:
                    total_values += 1
                    if v is None or v == "" or (isinstance(v, list) and len(v) == 0):
                        empty_values += 1
        
        count_empty_values(data)
        
        # If more than 20% of values are empty, or less than 5 total fields, suggest AI improvement
        if total_values > 0:
            empty_ratio = empty_values / total_values
            return empty_ratio > 0.2 or total_values < 5
        
        return False

# Legacy function for backward compatibility
async def extract_structured_data(ai_response, expected_schema, context=None):
    """Legacy function that uses the ResponseParser class."""
    parser = ResponseParser()
    return await parser.extract_structured_data(ai_response, expected_schema, context)

# Legacy function for backward compatibility
async def normalize_extraction_results(
    raw_results, 
    target_schema=None,
    source_url=None,
    extraction_context=None
):
    """Legacy function that uses the ResponseParser class."""
    parser = ResponseParser()
    return await parser.normalize_extraction_results(raw_results, target_schema, source_url, extraction_context)

# Legacy function for backward compatibility
async def improve_extraction_quality(
    extracted_data, 
    raw_html=None, 
    intent=None
):
    """Legacy function that uses the ResponseParser class."""
    parser = ResponseParser()
    return await parser.improve_extraction_quality(extracted_data, raw_html, intent)

# Legacy function for backward compatibility
async def reconcile_extraction_results(
    extractions, 
    strategy="merge"
):
    """Legacy function that uses the ResponseParser class."""
    parser = ResponseParser()
    return await parser.reconcile_extraction_results(extractions, strategy)

def clean_extraction_output(extraction_output: str) -> str:
    """
    Clean and standardize extraction output text.
    
    Args:
        extraction_output: Raw extraction output from AI model
        
    Returns:
        Cleaned extraction output
    """
    if not extraction_output:
        return ""
    
    # Remove markdown code blocks
    cleaned = re.sub(r'```json\s*', '', extraction_output)
    cleaned = re.sub(r'```\s*', '', cleaned)
    
    # Remove excessive whitespace
    cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
    cleaned = cleaned.strip()
    
    # Fix common JSON syntax issues
    cleaned = re.sub(r'(\w+)(?=\s*:)', r'"\1"', cleaned)  # Add quotes to unquoted keys
    cleaned = re.sub(r':\s*([^{\[\s"][^,\s}]*)', r': "\1"', cleaned)  # Add quotes to unquoted values
    
    # Fix trailing commas
    cleaned = re.sub(r',\s*}', '}', cleaned)
    cleaned = re.sub(r',\s*]', ']', cleaned)
    
    return cleaned

def analyze_extraction_result(
    extraction_result: Dict[str, Any], 
    expected_fields: List[str] = None
) -> Dict[str, Any]:
    """
    Analyze extraction results for completeness and quality.
    
    Args:
        extraction_result: The extraction result to analyze
        expected_fields: List of expected fields
        
    Returns:
        Analysis results
    """
    analysis = {
        "is_empty": False,
        "completeness": 1.0,
        "missing_fields": [],
        "has_errors": False,
        "error_fields": [],
        "quality_score": 1.0  # Default high score
    }
    
    # Check if result is empty
    if not extraction_result:
        analysis["is_empty"] = True
        analysis["completeness"] = 0.0
        analysis["quality_score"] = 0.0
        return analysis
    
    # Check completeness against expected fields
    if expected_fields:
        found_fields = []
        missing_fields = []
        
        for field in expected_fields:
            if field in extraction_result and extraction_result[field]:
                found_fields.append(field)
            else:
                missing_fields.append(field)
        
        if expected_fields:
            analysis["completeness"] = len(found_fields) / len(expected_fields)
            analysis["missing_fields"] = missing_fields
    
    # Check for potential errors
    error_fields = []
    for key, value in extraction_result.items():
        # Check for very short string values (might be truncated)
        if isinstance(value, str) and 0 < len(value) < 3 and key not in ["id", "code"]:
            error_fields.append(key)
        
        # Check for malformed values
        if (isinstance(value, str) and 
            (value.startswith('{') or value.startswith('[')) and 
            not (value.endswith('}') or value.endswith(']'))):
            error_fields.append(key)
    
    analysis["has_errors"] = len(error_fields) > 0
    analysis["error_fields"] = error_fields
    
    # Calculate quality score
    # Quality is reduced by incompleteness and errors
    error_penalty = 0.2 * (len(error_fields) / max(len(extraction_result), 1)) if error_fields else 0
    analysis["quality_score"] = max(0, analysis["completeness"] - error_penalty)
    
    return analysis

def extract_json_from_response(response_text: str) -> Optional[str]:
    """
    Extract JSON from a text response.
    
    Args:
        response_text: Text containing JSON
        
    Returns:
        Extracted JSON string or None if no JSON found
    """
    # Reuse the existing extraction logic from ResponseParser
    parser = ResponseParser()
    return parser._extract_json_from_text(response_text)

def parse_json_safely(json_str: str) -> Optional[Dict[str, Any]]:
    """
    Parse JSON string with error handling.
    
    Args:
        json_str: JSON string to parse
        
    Returns:
        Parsed JSON object or None if parsing fails
    """
    if not json_str:
        return None
        
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Try cleaning the JSON string first
        cleaned_json = clean_extraction_output(json_str)
        try:
            return json.loads(cleaned_json)
        except json.JSONDecodeError:
            return None
