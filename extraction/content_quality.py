"""
Content Quality Analysis Module

Provides capabilities to analyze the quality of extracted content and suggest
improvements to extraction strategies based on content completeness, relevance,
and structure.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ContentQualityAnalysis")

class ContentQualityAnalyzer:
    """
    Analyzes the quality of extracted content and provides suggestions for improvement.
    
    This class evaluates:
    - Completeness of extraction results
    - Relevance of content to user intent
    - Structure and quality of extracted data
    - Potential extraction failures or issues
    """
    
    def __init__(self, content_types_config: Dict[str, Dict[str, List[str]]] = None):
        """
        Initialize the content quality analyzer.
        
        Args:
            content_types_config: Optional configuration for expected fields by content type
        """
        # Default expected fields by content type
        self.expected_fields = {
            "product": ["title", "price", "description", "image", "brand"],
            "article": ["title", "author", "date", "content", "category"],
            "listing": ["items", "pagination", "count", "filters"],
            "profile": ["name", "bio", "contact", "image", "location"],
            "job": ["title", "company", "location", "description", "requirements", "salary"],
            "event": ["title", "date", "location", "description", "organizer"],
            "recipe": ["title", "ingredients", "instructions", "prep_time", "cook_time", "servings"],
            "review": ["title", "rating", "author", "date", "content", "pros", "cons"],
            "generic": ["title", "content", "date"]
        }
        
        # Override with custom configuration if provided
        if content_types_config:
            self.expected_fields.update(content_types_config)
            
        # Common issues patterns
        self.potential_issues = {
            "placeholder_content": [
                r"lorem ipsum", 
                r"placeholder", 
                r"sample text",
                r"\[.*?\]"  # [example text]
            ],
            "javascript_required": [
                r"enable javascript", 
                r"javascript is required", 
                r"please enable javascript"
            ],
            "access_blocked": [
                r"access denied", 
                r"forbidden", 
                r"403", 
                r"captcha",
                r"bot detected",
                r"security check"
            ],
            "login_required": [
                r"login required", 
                r"please sign in", 
                r"create an account",
                r"please log in"
            ],
            "error_page": [
                r"404", 
                r"not found", 
                r"error",
                r"page cannot be displayed"
            ]
        }
        
    def analyze_content_quality(self, 
                              content: Dict[str, Any], 
                              content_type: str = "generic",
                              expected_fields: List[str] = None) -> Dict[str, Any]:
        """
        Analyze the quality of extracted content.
        
        Args:
            content: The extracted content to analyze
            content_type: Type of content for determining expected fields
            expected_fields: Optional custom list of expected fields
            
        Returns:
            Dictionary with quality analysis results
        """
        if not content:
            return {
                "quality_score": 0,
                "completeness": 0,
                "missing_fields": [],
                "issues_detected": ["empty_content"],
                "recommendations": ["Check extraction configuration and selectors"]
            }
            
        # Get expected fields for this content type
        fields_to_check = expected_fields or self.expected_fields.get(content_type, self.expected_fields["generic"])
        
        # Initialize results
        results = {
            "quality_score": 0,
            "completeness": 0,
            "field_scores": {},
            "missing_fields": [],
            "empty_fields": [],
            "issues_detected": [],
            "successful_fields": [],
            "recommendations": []
        }
        
        # Check for items collection in listings
        if content_type == "listing" and "items" in content and isinstance(content["items"], list):
            if not content["items"]:
                results["issues_detected"].append("empty_items_list")
                results["recommendations"].append("Check item selectors, pagination might be required")
            else:
                # Analyze the first few items
                sample_items = content["items"][:5]
                item_fields = set()
                for item in sample_items:
                    if isinstance(item, dict):
                        item_fields.update(item.keys())
                
                results["item_fields_found"] = list(item_fields)
                
                # Check for minimum expected item fields
                min_item_fields = ["title", "url"]
                missing_item_fields = [f for f in min_item_fields if f not in item_fields]
                
                if missing_item_fields:
                    results["issues_detected"].append("incomplete_items")
                    results["recommendations"].append(
                        f"Items are missing key fields: {', '.join(missing_item_fields)}"
                    )
        
        # Check each expected field
        present_fields = 0
        field_scores = {}
        
        for field in fields_to_check:
            if field in content:
                field_value = content[field]
                field_score, field_issues = self._evaluate_field(field, field_value)
                
                # Record field score
                field_scores[field] = field_score
                
                if field_score > 0:
                    present_fields += 1
                    results["successful_fields"].append(field)
                    
                    # Add field-specific issues
                    if field_issues:
                        for issue in field_issues:
                            if issue not in results["issues_detected"]:
                                results["issues_detected"].append(issue)
                else:
                    results["empty_fields"].append(field)
            else:
                results["missing_fields"].append(field)
                field_scores[field] = 0
        
        # Calculate completeness
        if fields_to_check:
            results["completeness"] = present_fields / len(fields_to_check)
            
        # Store field scores
        results["field_scores"] = field_scores
        
        # Check for common issues in text content
        if "text" in content or "content" in content or "description" in content:
            text_content = content.get("text", content.get("content", content.get("description", "")))
            if isinstance(text_content, str):
                issues = self._detect_content_issues(text_content)
                for issue in issues:
                    if issue not in results["issues_detected"]:
                        results["issues_detected"].append(issue)
        
        # Calculate overall quality score (weighted)
        # 60% completeness, 40% average field quality
        avg_field_score = sum(field_scores.values()) / len(field_scores) if field_scores else 0
        results["quality_score"] = (0.6 * results["completeness"]) + (0.4 * avg_field_score)
        
        # Generate recommendations
        results["recommendations"].extend(self._generate_recommendations(results, content_type))
        
        return results
    
    def analyze_content_relevance(self, 
                                content: Dict[str, Any], 
                                user_query: str) -> Dict[str, Any]:
        """
        Analyze how relevant the extracted content is to the user's query.
        
        Args:
            content: The extracted content to analyze
            user_query: The user's original query or intent
            
        Returns:
            Dictionary with relevance analysis results
        """
        if not content or not user_query:
            return {
                "relevance_score": 0,
                "relevant_fields": [],
                "recommendations": ["No content or query provided"]
            }
            
        # Extract keywords from user query
        query_keywords = self._extract_keywords(user_query)
        
        # Initialize results
        results = {
            "relevance_score": 0,
            "query_keywords": query_keywords,
            "relevant_fields": [],
            "keyword_matches": {},
            "recommendations": []
        }
        
        # Count keyword matches in each field
        total_matches = 0
        max_possible_matches = len(query_keywords) * len(content)
        
        for field, value in content.items():
            field_matches = 0
            field_text = ""
            
            # Convert value to searchable text
            if isinstance(value, str):
                field_text = value.lower()
            elif isinstance(value, (int, float)):
                field_text = str(value).lower()
            elif isinstance(value, dict):
                # If it's a dictionary, join all string values
                field_text = " ".join(str(v).lower() for v in value.values() 
                                     if isinstance(v, (str, int, float)))
            elif isinstance(value, list):
                # If it's a list, join all string values
                field_text = " ".join(str(v).lower() for v in value 
                                     if isinstance(v, (str, int, float)))
            
            # Count keyword matches
            for keyword in query_keywords:
                if keyword.lower() in field_text:
                    field_matches += 1
                    results["keyword_matches"].setdefault(keyword, []).append(field)
                    total_matches += 1
            
            # If field contains any matches, add to relevant fields
            if field_matches > 0:
                results["relevant_fields"].append({
                    "field": field,
                    "matches": field_matches,
                    "keywords": [k for k in query_keywords if k.lower() in field_text]
                })
        
        # Calculate overall relevance score
        if max_possible_matches > 0:
            results["relevance_score"] = total_matches / max_possible_matches
            # Adjust score to give higher weight to multiple matched fields
            results["relevance_score"] *= (1 + (len(results["relevant_fields"]) / len(content)) / 2)
            # Cap at 1.0
            results["relevance_score"] = min(1.0, results["relevance_score"])
        
        # Generate recommendations
        if results["relevance_score"] < 0.3:
            results["recommendations"].append(
                "Content has low relevance to query. Consider refining extraction criteria."
            )
            
            # Suggest which fields might be missing
            missing_concepts = [k for k in query_keywords if k not in results["keyword_matches"]]
            if missing_concepts:
                results["recommendations"].append(
                    f"Query concepts not found in content: {', '.join(missing_concepts)}"
                )
        
        return results
    
    def compare_extraction_methods(self, 
                                 results_by_method: Dict[str, Dict[str, Any]], 
                                 content_type: str = "generic") -> Dict[str, Any]:
        """
        Compare results from different extraction methods to determine best approach.
        
        Args:
            results_by_method: Dictionary mapping method names to their extraction results
            content_type: Type of content for determining expected fields
            
        Returns:
            Dictionary with comparison results and recommendations
        """
        if not results_by_method:
            return {
                "best_method": None,
                "comparison": {},
                "recommendations": ["No extraction methods provided for comparison"]
            }
            
        # Initialize results
        results = {
            "best_method": None,
            "method_scores": {},
            "comparison": {},
            "strengths": {},
            "weaknesses": {},
            "recommendations": []
        }
        
        # Get expected fields for this content type
        fields_to_check = self.expected_fields.get(content_type, self.expected_fields["generic"])
        
        # Analyze each method individually
        for method_name, content in results_by_method.items():
            quality_analysis = self.analyze_content_quality(content, content_type)
            results["method_scores"][method_name] = quality_analysis["quality_score"]
            
            # Record strengths (successful fields unique to this method)
            successful_fields = set(quality_analysis.get("successful_fields", []))
            for other_method, other_content in results_by_method.items():
                if other_method != method_name:
                    other_analysis = self.analyze_content_quality(other_content, content_type)
                    other_successful = set(other_analysis.get("successful_fields", []))
                    unique_fields = successful_fields - other_successful
                    if unique_fields:
                        results["strengths"].setdefault(method_name, []).extend(unique_fields)
            
            # Record weaknesses (missing or empty fields)
            results["weaknesses"][method_name] = (
                quality_analysis.get("missing_fields", []) + 
                quality_analysis.get("empty_fields", [])
            )
        
        # Determine best method based on quality score
        if results["method_scores"]:
            results["best_method"] = max(results["method_scores"], key=results["method_scores"].get)
        
        # Build detailed comparison
        found_fields_by_method = {}
        for method_name, content in results_by_method.items():
            found_fields = {}
            for field in fields_to_check:
                if field in content and content[field]:
                    # Simplified quality check
                    if isinstance(content[field], str):
                        value = content[field].strip()
                        found_fields[field] = bool(value)
                    elif isinstance(content[field], (list, dict)):
                        found_fields[field] = bool(content[field])
                    else:
                        found_fields[field] = True
                else:
                    found_fields[field] = False
            found_fields_by_method[method_name] = found_fields
            
        # Create field by field comparison
        for field in fields_to_check:
            comparison = {}
            for method_name in results_by_method:
                comparison[method_name] = found_fields_by_method[method_name][field]
            results["comparison"][field] = comparison
        
        # Generate recommendations for hybrid approach
        if len(results_by_method) > 1:
            hybrid_recommendation = "Consider a hybrid approach using:"
            for method_name, strengths in results["strengths"].items():
                if strengths:
                    hybrid_recommendation += f"\n- {method_name} for {', '.join(strengths)}"
            results["recommendations"].append(hybrid_recommendation)
        
        return results
    
    def suggest_extraction_improvements(self, 
                                      content: Dict[str, Any], 
                                      content_type: str = "generic",
                                      extraction_method: str = None) -> Dict[str, Any]:
        """
        Suggest specific improvements to extraction configuration.
        
        Args:
            content: The extracted content to analyze
            content_type: Type of content for determining expected fields
            extraction_method: Optional method used for extraction
            
        Returns:
            Dictionary with suggested improvements
        """
        # Start with quality analysis
        quality = self.analyze_content_quality(content, content_type)
        
        # Initialize results
        results = {
            "current_quality": quality["quality_score"],
            "suggestions": {},
            "selectors_to_fix": [],
            "configuration_changes": [],
            "general_suggestions": []
        }
        
        # Check missing fields
        if quality["missing_fields"]:
            results["suggestions"]["missing_fields"] = {
                "description": f"Add selectors for: {', '.join(quality['missing_fields'])}",
                "importance": "high",
                "fields": quality["missing_fields"]
            }
            results["selectors_to_fix"].extend(quality["missing_fields"])
        
        # Check empty fields
        if quality["empty_fields"]:
            results["suggestions"]["empty_fields"] = {
                "description": f"Fix selectors for: {', '.join(quality['empty_fields'])}",
                "importance": "high",
                "fields": quality["empty_fields"]
            }
            results["selectors_to_fix"].extend(quality["empty_fields"])
        
        # Check for pagination issues
        if content_type == "listing" and "items" in content:
            items = content["items"] if isinstance(content["items"], list) else []
            if len(items) < 5:  # Arbitrary threshold
                results["suggestions"]["pagination"] = {
                    "description": "Few items found. Check pagination handling.",
                    "importance": "medium"
                }
                results["configuration_changes"].append("Add pagination handling")
        
        # Check for JavaScript issues
        js_issues = [issue for issue in quality["issues_detected"] 
                    if issue in ["javascript_required", "lazy_loaded_content"]]
        if js_issues:
            results["suggestions"]["javascript"] = {
                "description": "Content requires JavaScript. Use headless browser extraction.",
                "importance": "high"
            }
            results["configuration_changes"].append("Enable JavaScript rendering")
        
        # Check for access issues
        access_issues = [issue for issue in quality["issues_detected"] 
                        if issue in ["access_blocked", "login_required", "captcha_detected"]]
        if access_issues:
            results["suggestions"]["access"] = {
                "description": f"Access issues detected: {', '.join(access_issues)}",
                "importance": "high"
            }
            results["configuration_changes"].append("Implement access handling strategy")
        
        # Method-specific recommendations
        if extraction_method == "css":
            if quality["quality_score"] < 0.5:
                results["suggestions"]["extraction_method"] = {
                    "description": "CSS extraction performing poorly. Try hybrid or AI-guided extraction.",
                    "importance": "high"
                }
                results["configuration_changes"].append("Switch to hybrid extraction method")
                
        elif extraction_method == "raw":
            results["suggestions"]["extraction_method"] = {
                "description": "Raw extraction provides limited structure. Use CSS or AI-guided extraction.",
                "importance": "medium"
            }
            results["configuration_changes"].append("Implement CSS selectors for structured extraction")
        
        # General suggestions based on quality score
        if quality["quality_score"] < 0.3:
            results["general_suggestions"].append(
                "Extraction quality is poor. Consider a complete revision of extraction strategy."
            )
        elif quality["quality_score"] < 0.6:
            results["general_suggestions"].append(
                "Extraction quality is moderate. Focus on missing and empty fields."
            )
        else:
            results["general_suggestions"].append(
                "Extraction quality is good. Consider fine-tuning for specific fields."
            )
        
        return results
    
    def _evaluate_field(self, field_name: str, field_value: Any) -> Tuple[float, List[str]]:
        """
        Evaluate the quality of a specific field value.
        
        Args:
            field_name: Name of the field
            field_value: Value of the field
            
        Returns:
            Tuple of (quality score, list of issues)
        """
        issues = []
        
        # Handle empty values
        if field_value is None:
            return 0, ["empty_value"]
            
        if isinstance(field_value, str) and field_value.strip() == "":
            return 0, ["empty_string"]
            
        if isinstance(field_value, (list, dict)) and not field_value:
            return 0, ["empty_collection"]
        
        # Type-specific evaluation
        if isinstance(field_value, str):
            return self._evaluate_text_field(field_name, field_value)
            
        elif isinstance(field_value, (int, float)):
            # Numeric fields are usually good if present
            return 1.0, []
            
        elif isinstance(field_value, bool):
            # Boolean fields are usually good if present
            return 1.0, []
            
        elif isinstance(field_value, list):
            # For lists, check if items are non-empty
            if not field_value:
                return 0, ["empty_list"]
                
            if field_name == "items":
                # For items collection, check if they have enough structure
                if all(isinstance(item, dict) for item in field_value):
                    # Check if items have at least basic fields
                    if any('title' in item or 'name' in item for item in field_value):
                        return 1.0, []
                    else:
                        return 0.5, ["items_missing_title"]
                else:
                    return 0.3, ["unstructured_items"]
                    
            # For other lists, just check they're non-empty
            return 1.0, []
            
        elif isinstance(field_value, dict):
            # For dictionaries, check if they have reasonable keys
            if not field_value:
                return 0, ["empty_dict"]
                
            return 1.0, []
            
        # Default for unknown types
        return 0.5, ["unknown_value_type"]
    
    def _evaluate_text_field(self, field_name: str, text: str) -> Tuple[float, List[str]]:
        """
        Evaluate the quality of a text field.
        
        Args:
            field_name: Name of the field
            text: Text content to evaluate
            
        Returns:
            Tuple of (quality score, list of issues)
        """
        issues = []
        text = text.strip()
        
        # Check for empty text
        if not text:
            return 0, ["empty_text"]
            
        # Check for placeholder content
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in self.potential_issues["placeholder_content"]):
            issues.append("placeholder_content")
            
        # Check for access issues
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in self.potential_issues["access_blocked"]):
            issues.append("access_blocked")
            
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in self.potential_issues["login_required"]):
            issues.append("login_required")
            
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in self.potential_issues["error_page"]):
            issues.append("error_page")
            
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in self.potential_issues["javascript_required"]):
            issues.append("javascript_required")
            
        # Check for very short text (field-specific)
        min_expected_length = {
            "title": 3,
            "description": 20,
            "content": 50,
            "summary": 10
        }
        
        expected_length = min_expected_length.get(field_name, 1)
        if len(text) < expected_length:
            issues.append("too_short")
        
        # Field-specific checks
        if field_name == "url" and not re.match(r'^https?://', text):
            issues.append("invalid_url")
            
        if field_name == "email" and not re.match(r'^[^@]+@[^@]+\.[^@]+$', text):
            issues.append("invalid_email")
            
        if field_name in ["image", "image_url"] and not re.match(r'^(https?://|/|\.\.?/)', text):
            issues.append("invalid_image_url")
        
        # Calculate score based on issues
        if issues:
            # More issues = lower score
            return max(0, 1 - (len(issues) * 0.25)), issues
        
        # Score based on content length relative to expectations
        if field_name in min_expected_length:
            expected = min_expected_length[field_name]
            ratio = min(len(text) / (expected * 2), 1)  # Cap at 1.0
            return 0.5 + (ratio * 0.5), []  # Score between 0.5 and 1.0
            
        return 1.0, []
    
    def _detect_content_issues(self, text: str) -> List[str]:
        """
        Detect common issues in content text.
        
        Args:
            text: Text content to analyze
            
        Returns:
            List of detected issues
        """
        issues = []
        
        # Check each issue pattern
        for issue_type, patterns in self.potential_issues.items():
            if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
                issues.append(issue_type)
        
        # Check for potentially lazy-loaded content
        if re.search(r'loading(\.\.\.|…)', text, re.IGNORECASE):
            issues.append("lazy_loaded_content")
            
        # Check for truncated content
        if re.search(r'(\.\.\.|…)$', text) or text.endswith('...'):
            issues.append("truncated_content")
            
        # Check for captcha indicators
        if re.search(r'captcha|robot|human verification', text, re.IGNORECASE):
            issues.append("captcha_detected")
            
        return issues
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any], content_type: str) -> List[str]:
        """
        Generate recommendations based on analysis results.
        
        Args:
            analysis_results: Results from analyze_content_quality
            content_type: Type of content analyzed
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Low completeness recommendations
        if analysis_results["completeness"] < 0.5:
            missing_fields = analysis_results.get("missing_fields", [])
            empty_fields = analysis_results.get("empty_fields", [])
            problem_fields = missing_fields + empty_fields
            
            if problem_fields:
                field_list = ", ".join(problem_fields)
                recommendations.append(f"Focus on extracting these key fields: {field_list}")
                
                # Suggest JavaScript rendering if appropriate
                js_issues = [i for i in analysis_results.get("issues_detected", []) 
                           if i in ["javascript_required", "lazy_loaded_content"]]
                if js_issues:
                    recommendations.append("Enable JavaScript rendering for content extraction")
        
        # Issue-specific recommendations
        for issue in analysis_results.get("issues_detected", []):
            if issue == "placeholder_content":
                recommendations.append("Detected placeholder text. Site might be serving different content to scrapers.")
            elif issue == "access_blocked":
                recommendations.append("Access appears to be blocked. Consider implementing request rotation or proxies.")
            elif issue == "login_required":
                recommendations.append("Content requires login. Implement authentication handling.")
            elif issue == "error_page":
                recommendations.append("Error page detected. Verify URL and accessibility.")
            elif issue == "javascript_required":
                recommendations.append("Page requires JavaScript. Use headless browser extraction.")
            elif issue == "lazy_loaded_content":
                recommendations.append("Content appears to be lazy-loaded. Enable scrolling in headless browser.")
            elif issue == "truncated_content":
                recommendations.append("Content appears to be truncated. Check for 'read more' buttons or pagination.")
            elif issue == "captcha_detected":
                recommendations.append("CAPTCHA detected. Implement CAPTCHA handling or use proxy rotation.")
            elif issue == "empty_items_list" and content_type == "listing":
                recommendations.append("No items found in listing. Verify selectors and pagination handling.")
            elif issue == "unstructured_items" and content_type == "listing":
                recommendations.append("Items lack proper structure. Improve item selectors for better extraction.")
        
        # Content type specific recommendations
        if content_type == "article" and analysis_results["completeness"] > 0 and "content" in analysis_results.get("successful_fields", []):
            # If we got the content but missed metadata
            missed_metadata = [f for f in ["author", "date", "category"] if f in analysis_results.get("missing_fields", [])]
            if missed_metadata:
                metadata_list = ", ".join(missed_metadata)
                recommendations.append(f"Article content extracted, but missing metadata: {metadata_list}")
                
        elif content_type == "product" and "price" not in analysis_results.get("successful_fields", []):
            recommendations.append("Product price not extracted correctly. This is often in a special format, try a more specific selector.")
        
        return recommendations
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract important keywords from text.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of extracted keywords
        """
        # Remove stopwords
        stopwords = {
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", 
            "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", 
            "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", 
            "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", 
            "its", "itself", "just", "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", 
            "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", 
            "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", 
            "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", 
            "when", "where", "which", "while", "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself", 
            "yourselves", "get", "find", "look", "search", "need", "want", "like"
        }
        
        # Tokenize and clean
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        keywords = [word for word in words if word not in stopwords]
        
        # Count occurrences to find important terms
        counter = Counter(keywords)
        
        # Return most common keywords, prioritizing longer phrases
        return [word for word, _ in counter.most_common(10)]