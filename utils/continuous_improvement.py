"""
Continuous Improvement System for SmartScrape

This module implements a system that learns from test results
and scraping metrics to automatically improve extraction
strategies and patterns over time.
"""

import os
import sys
import json
import logging
import datetime
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict, Counter

from core.service_interface import BaseService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ContinuousImprovement")

class ContinuousImprovementSystem(BaseService):
    """
    A system that analyzes scraping results and metrics to
    automatically suggest and implement improvements.
    """
    
    def __init__(self, 
                results_dir: str = None, 
                patterns_dir: str = None,
                strategies_dir: str = None):
        """
        Initialize the continuous improvement system.
        
        Args:
            results_dir: Directory with test results and metrics
            patterns_dir: Directory to store improved patterns
            strategies_dir: Directory to store improved strategies
        """
        self._initialized = False
        self.results_dir = results_dir or os.path.join(os.getcwd(), 'test_results')
        self.patterns_dir = patterns_dir or os.path.join(os.getcwd(), 'extraction_strategies', 'extraction_profiles')
        self.strategies_dir = strategies_dir or os.path.join(os.getcwd(), 'strategies')
        self.history_file = None
        self.history = None
        
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the service with the given configuration."""
        if self._initialized:
            return
            
        # Apply configuration if provided
        if config:
            self.results_dir = config.get('results_dir', self.results_dir)
            self.patterns_dir = config.get('patterns_dir', self.patterns_dir)
            self.strategies_dir = config.get('strategies_dir', self.strategies_dir)
        
        # Create directories if they don't exist
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.patterns_dir, exist_ok=True)
        os.makedirs(os.path.join(self.patterns_dir, 'improvements'), exist_ok=True)
        
        # Track learning history
        self.history_file = os.path.join(self.results_dir, 'improvement_history.json')
        self.history = self._load_history()
        
        self._initialized = True
        logger.info("ContinuousImprovement service initialized")
    
    def shutdown(self) -> None:
        """Perform cleanup and shutdown operations."""
        self._save_history()
        self._initialized = False
        logger.info("ContinuousImprovement service shut down")
    
    @property
    def name(self) -> str:
        """Return the name of the service."""
        return "continuous_improvement"
    
    def learn_from_execution(self, context: Dict[str, Any], success: bool, results: List[Dict[str, Any]], metrics: Dict[str, Any]) -> None:
        """
        Learn from execution results to improve future performance.
        
        Args:
            context: Execution context with strategy and other metadata
            success: Whether the execution was successful
            results: Results from the execution
            metrics: Performance metrics from the execution
        """
        try:
            if not self._initialized:
                logger.warning("ContinuousImprovementSystem not initialized, skipping learning")
                return
            
            # Store execution data for analysis
            execution_record = {
                'timestamp': datetime.datetime.now().isoformat(),
                'success': success,
                'strategy': context.get('strategy_name', 'unknown'),
                'url': context.get('url', 'unknown'),
                'metrics': metrics,
                'result_count': len(results) if results else 0
            }
            
            # Add to history
            if self.history is None:
                self.history = {'executions': []}
            
            if 'executions' not in self.history:
                self.history['executions'] = []
                
            self.history['executions'].append(execution_record)
            
            # Keep only the last 1000 executions to prevent memory issues
            if len(self.history['executions']) > 1000:
                self.history['executions'] = self.history['executions'][-1000:]
            
            # Save updated history
            self._save_history()
            
            logger.debug(f"Recorded execution result: success={success}, strategy={context.get('strategy_name')}")
            
        except Exception as e:
            logger.warning(f"Error in learn_from_execution: {e}")

    def _load_history(self) -> Dict[str, Any]:
        """Load improvement history from file"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading history file: {str(e)}")
        
        # Create new history if file doesn't exist or has errors
        return {
            "version": "1.0",
            "created_at": datetime.datetime.now().isoformat(),
            "updated_at": datetime.datetime.now().isoformat(),
            "improvement_runs": [],
            "patterns_improved": 0,
            "strategies_improved": 0,
            "learning_events": []
        }
    
    def _save_history(self):
        """Save improvement history to file"""
        if self.history is None:
            logger.warning("History is None, cannot save")
            return
            
        self.history["updated_at"] = datetime.datetime.now().isoformat()
        
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2, default=str)
            logger.info(f"Saved improvement history to {self.history_file}")
        except Exception as e:
            logger.error(f"Error saving history file: {str(e)}")
    
    def analyze_test_results(self) -> Dict[str, Any]:
        """
        Analyze test results to identify improvement opportunities.
        
        Returns:
            Dictionary with improvement suggestions
        """
        logger.info("Analyzing test results for improvement opportunities")
        
        try:
            # Look for the most recent test results/metrics report
            report_files = [
                f for f in os.listdir(self.results_dir) 
                if f.endswith('_metrics_report.json')
            ]
            
            if not report_files:
                logger.warning("No metrics report files found")
                return {"error": "No metrics report files found"}
            
            # Sort by modification time (newest first)
            report_files.sort(key=lambda f: os.path.getmtime(os.path.join(self.results_dir, f)), reverse=True)
            
            # Load the most recent report
            latest_report_file = os.path.join(self.results_dir, report_files[0])
            logger.info(f"Analyzing report file: {latest_report_file}")
            
            with open(latest_report_file, 'r') as f:
                report = json.load(f)
            
            # Extract improvement opportunities
            improvement_areas = {
                "field_extraction": [],
                "website_categories": [],
                "error_handling": [],
                "performance": [],
                "patterns": []
            }
            
            # Extract fields with low coverage
            field_coverage = report.get("test_metrics", {}).get("field_coverage", {})
            if field_coverage:
                for field, count in field_coverage.items():
                    if isinstance(count, (int, float)) and count < 5:  # Arbitrary threshold
                        improvement_areas["field_extraction"].append({
                            "field": field,
                            "coverage": count,
                            "suggestion": f"Improve extraction patterns for field '{field}'"
                        })
            
            # Extract website categories with low success rates
            category_performance = report.get("test_metrics", {}).get("by_category", {})
            if category_performance:
                for category, stats in category_performance.items():
                    success_rate = 0
                    if "success_rate" in stats:
                        success_rate_str = stats["success_rate"]
                        if isinstance(success_rate_str, str) and "%" in success_rate_str:
                            success_rate = float(success_rate_str.strip("%")) / 100
                        elif isinstance(success_rate_str, (int, float)):
                            success_rate = float(success_rate_str)
                    
                    if success_rate < 0.7:  # 70% success threshold
                        improvement_areas["website_categories"].append({
                            "category": category,
                            "success_rate": success_rate,
                            "suggestion": f"Add specialized handling for {category} websites"
                        })
            
            # Extract common errors
            error_counts = report.get("log_metrics", {}).get("error_counts", {})
            if error_counts:
                for error_type, count in error_counts.items():
                    if count > 3:  # Arbitrary threshold
                        improvement_areas["error_handling"].append({
                            "error_type": error_type,
                            "count": count,
                            "suggestion": f"Improve handling for '{error_type}' errors"
                        })
            
            return {
                "timestamp": datetime.datetime.now().isoformat(),
                "report_analyzed": latest_report_file,
                "improvement_areas": improvement_areas
            }
            
        except Exception as e:
            logger.error(f"Error analyzing test results: {str(e)}")
            return {"error": str(e)}
    
    def generate_improvements(self) -> Dict[str, Any]:
        """
        Generate concrete improvement suggestions based on analysis.
        
        Returns:
            Dictionary with concrete improvements
        """
        logger.info("Generating improvement suggestions")
        
        # First analyze test results
        analysis = self.analyze_test_results()
        if "error" in analysis:
            return analysis
        
        improvements = {
            "timestamp": datetime.datetime.now().isoformat(),
            "pattern_improvements": [],
            "strategy_improvements": [],
            "error_handling_improvements": []
        }
        
        # Generate pattern improvements
        field_extraction_issues = analysis.get("improvement_areas", {}).get("field_extraction", [])
        for issue in field_extraction_issues:
            field = issue.get("field")
            if field:
                # Create improvement suggestion
                improvements["pattern_improvements"].append({
                    "field": field,
                    "current_coverage": issue.get("coverage", 0),
                    "improvement": {
                        "selector_alternatives": self._generate_selector_alternatives(field),
                        "attribute_alternatives": self._generate_attribute_alternatives(field),
                        "extraction_rules": self._generate_extraction_rules(field)
                    }
                })
        
        # Generate strategy improvements
        category_issues = analysis.get("improvement_areas", {}).get("website_categories", [])
        for issue in category_issues:
            category = issue.get("category")
            if category:
                # Create improvement suggestion
                improvements["strategy_improvements"].append({
                    "category": category,
                    "current_success_rate": issue.get("success_rate", 0),
                    "improvement": {
                        "strategy_adjustments": self._generate_strategy_adjustments(category),
                        "specialized_handling": self._generate_specialized_handling(category)
                    }
                })
        
        # Generate error handling improvements
        error_issues = analysis.get("improvement_areas", {}).get("error_handling", [])
        for issue in error_issues:
            error_type = issue.get("error_type")
            if error_type:
                # Create improvement suggestion
                improvements["error_handling_improvements"].append({
                    "error_type": error_type,
                    "occurrence_count": issue.get("count", 0),
                    "improvement": {
                        "fallback_mechanism": self._generate_fallback_mechanism(error_type),
                        "retry_strategy": self._generate_retry_strategy(error_type)
                    }
                })
        
        return improvements
    
    def apply_improvements(self, improvements: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Apply generated improvements to patterns and strategies.
        
        Args:
            improvements: Dictionary with improvements (if None, will generate)
            
        Returns:
            Dictionary with results of applied improvements
        """
        logger.info("Applying improvements")
        
        # Generate improvements if not provided
        if improvements is None:
            improvements = self.generate_improvements()
            if "error" in improvements:
                return improvements
        
        results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "applied_pattern_improvements": [],
            "applied_strategy_improvements": [],
            "applied_error_handling_improvements": []
        }
        
        # Apply pattern improvements
        for improvement in improvements.get("pattern_improvements", []):
            field = improvement.get("field")
            if field:
                pattern_file = os.path.join(
                    self.patterns_dir, 
                    "improvements", 
                    f"improved_{field.lower()}_extraction.json"
                )
                
                # Create improved pattern file
                pattern_data = {
                    "field": field,
                    "generated_at": datetime.datetime.now().isoformat(),
                    "selector_alternatives": improvement.get("improvement", {}).get("selector_alternatives", []),
                    "attribute_alternatives": improvement.get("improvement", {}).get("attribute_alternatives", []),
                    "extraction_rules": improvement.get("improvement", {}).get("extraction_rules", [])
                }
                
                try:
                    with open(pattern_file, 'w') as f:
                        json.dump(pattern_data, f, indent=2)
                    
                    results["applied_pattern_improvements"].append({
                        "field": field,
                        "file_created": pattern_file,
                        "success": True
                    })
                    
                    self.history["patterns_improved"] += 1
                    
                except Exception as e:
                    logger.error(f"Error creating pattern file for {field}: {str(e)}")
                    results["applied_pattern_improvements"].append({
                        "field": field,
                        "error": str(e),
                        "success": False
                    })
        
        # Track this improvement run in history
        self.history["improvement_runs"].append({
            "timestamp": datetime.datetime.now().isoformat(),
            "pattern_improvements": len(results["applied_pattern_improvements"]),
            "strategy_improvements": len(results["applied_strategy_improvements"]),
            "error_handling_improvements": len(results["applied_error_handling_improvements"])
        })
        
        # Add learning event
        self.history["learning_events"].append({
            "timestamp": datetime.datetime.now().isoformat(),
            "type": "improvement_application",
            "details": {
                "patterns_improved": len(results["applied_pattern_improvements"]),
                "strategies_improved": len(results["applied_strategy_improvements"]),
                "error_handling_improved": len(results["applied_error_handling_improvements"])
            }
        })
        
        # Save history
        self._save_history()
        
        return results
    
    def _generate_selector_alternatives(self, field: str) -> List[str]:
        """Generate alternative CSS selectors for a field"""
        # This would be more sophisticated in production, but here's a simple example
        field_lower = field.lower()
        
        common_selectors = {
            "price": [
                ".price", 
                "[data-price]", 
                ".product-price", 
                ".listing-price", 
                "span:contains('$')", 
                "span.price-value"
            ],
            "title": [
                "h1", 
                ".title", 
                ".product-title", 
                ".listing-title", 
                "[data-title]", 
                "h2.title"
            ],
            "description": [
                ".description", 
                ".product-description", 
                ".listing-description", 
                "[data-description]", 
                "div.description-text"
            ],
            "image": [
                "img.product-image", 
                ".listing-image img", 
                ".product-img", 
                "[data-image-url]", 
                ".gallery img"
            ],
            "address": [
                ".address", 
                ".property-address", 
                "[data-address]", 
                ".listing-address", 
                "span.address-line"
            ],
            "bedrooms": [
                ".beds", 
                ".bedrooms", 
                "[data-beds]", 
                "span:contains('bed')", 
                ".property-beds"
            ],
            "bathrooms": [
                ".baths", 
                ".bathrooms", 
                "[data-baths]", 
                "span:contains('bath')", 
                ".property-baths"
            ],
            "rating": [
                ".rating", 
                ".stars", 
                "[data-rating]", 
                ".product-rating", 
                ".review-score"
            ]
        }
        
        # Return common selectors for this field, or generic ones if field not recognized
        for known_field, selectors in common_selectors.items():
            if known_field in field_lower or field_lower in known_field:
                return selectors
        
        # Generic selectors for unknown fields
        return [
            f".{field_lower}",
            f"[data-{field_lower}]",
            f"#{field_lower}",
            f"[data-field='{field_lower}']",
            f"[itemprop='{field_lower}']"
        ]
    
    def _generate_attribute_alternatives(self, field: str) -> List[str]:
        """Generate alternative attributes to extract for a field"""
        field_lower = field.lower()
        
        common_attributes = {
            "price": ["content", "data-price", "value"],
            "title": ["content", "data-title", "alt", "title"],
            "description": ["content", "data-description", "alt", "title"],
            "image": ["src", "data-src", "data-original", "data-lazy-src"],
            "address": ["content", "data-address", "title"],
            "bedrooms": ["content", "data-beds", "value"],
            "bathrooms": ["content", "data-baths", "value"],
            "rating": ["content", "data-rating", "value"],
            "url": ["href", "data-url", "src"]
        }
        
        # Return common attributes for this field, or generic ones if field not recognized
        for known_field, attributes in common_attributes.items():
            if known_field in field_lower or field_lower in known_field:
                return attributes
        
        # Generic attributes for unknown fields
        return ["content", f"data-{field_lower}", "title", "value", "innerText"]
    
    def _generate_extraction_rules(self, field: str) -> List[Dict[str, Any]]:
        """Generate extraction rules for a field"""
        field_lower = field.lower()
        
        # Define common extraction patterns for different fields
        common_rules = {
            "price": [
                {
                    "type": "regex",
                    "pattern": r"\$\s*([\d,]+\.?\d*)",
                    "description": "Extract dollar amount"
                },
                {
                    "type": "regex",
                    "pattern": r"([\d,]+\.?\d*)\s*USD",
                    "description": "Extract USD amount"
                },
                {
                    "type": "format",
                    "remove_chars": ",$",
                    "description": "Clean price format"
                }
            ],
            "bedrooms": [
                {
                    "type": "regex",
                    "pattern": r"(\d+)\s*(?:bed|bedroom|bd|br)",
                    "description": "Extract bedroom count"
                },
                {
                    "type": "conversion",
                    "convert_to": "number",
                    "description": "Convert to number"
                }
            ],
            "bathrooms": [
                {
                    "type": "regex",
                    "pattern": r"(\d+(?:\.\d+)?)\s*(?:bath|bathroom|ba)",
                    "description": "Extract bathroom count"
                },
                {
                    "type": "conversion",
                    "convert_to": "number",
                    "description": "Convert to number"
                }
            ],
            "sqft": [
                {
                    "type": "regex",
                    "pattern": r"(\d+[,\d]*)\s*(?:sq\.?\s*ft|sqft|square feet|sf)",
                    "description": "Extract square footage"
                },
                {
                    "type": "format",
                    "remove_chars": ",",
                    "description": "Clean number format"
                },
                {
                    "type": "conversion",
                    "convert_to": "number",
                    "description": "Convert to number"
                }
            ],
            "rating": [
                {
                    "type": "regex",
                    "pattern": r"(\d+(?:\.\d+)?)\s*(?:star|out of|\/\s*\d+)",
                    "description": "Extract rating value"
                },
                {
                    "type": "conversion",
                    "convert_to": "number",
                    "description": "Convert to number"
                }
            ]
        }
        
        # Return common rules for this field, or generic ones if field not recognized
        for known_field, rules in common_rules.items():
            if known_field in field_lower or field_lower in known_field:
                return rules
        
        # Generic rules for unknown fields
        return [
            {
                "type": "text_normalization",
                "trim": True,
                "remove_extra_spaces": True,
                "description": "Normalize text"
            }
        ]
    
    def _generate_strategy_adjustments(self, category: str) -> List[Dict[str, Any]]:
        """Generate strategy adjustments for a website category"""
        category_lower = category.lower()
        
        # Define strategy adjustments for different categories
        category_strategies = {
            "real_estate": [
                {
                    "strategy": "BestFirstStrategy",
                    "weights": {
                        "listing_page_probability": 2.0,
                        "property_details_probability": 1.5
                    },
                    "description": "Prioritize listing pages and property details"
                },
                {
                    "strategy": "PaginationHandling",
                    "max_pages": 5,
                    "continue_if_similar_content": True,
                    "description": "Handle pagination for real estate listings"
                }
            ],
            "e_commerce": [
                {
                    "strategy": "CategoryFirstStrategy",
                    "prioritize_search": True,
                    "description": "Prioritize category pages and search results"
                },
                {
                    "strategy": "ProductDetailExtraction",
                    "follow_product_links": True,
                    "description": "Extract from product detail pages"
                }
            ],
            "content_sites": [
                {
                    "strategy": "ContentFocusedStrategy",
                    "extract_full_article": True,
                    "description": "Focus on extracting full article content"
                },
                {
                    "strategy": "AuthorMetadataExtraction",
                    "extract_publish_date": True,
                    "description": "Extract author and publishing metadata"
                }
            ],
            "directory_sites": [
                {
                    "strategy": "ListingPaginationStrategy",
                    "max_pages": 3,
                    "description": "Handle pagination in directory listings"
                },
                {
                    "strategy": "ContactInfoExtraction",
                    "prioritize_phone_email": True,
                    "description": "Prioritize contact information extraction"
                }
            ]
        }
        
        # Return strategies for this category, or generic ones if not recognized
        for known_category, strategies in category_strategies.items():
            if known_category in category_lower or category_lower in known_category:
                return strategies
        
        # Generic strategies for unknown categories
        return [
            {
                "strategy": "AdaptiveStrategy",
                "analyze_first_page": True,
                "description": "Analyze first page and adapt strategy"
            },
            {
                "strategy": "PaginationHandling",
                "max_pages": 3,
                "description": "Generic pagination handling"
            }
        ]
    
    def _generate_specialized_handling(self, category: str) -> Dict[str, Any]:
        """Generate specialized handling for a website category"""
        category_lower = category.lower()
        
        # Define specialized handling for different categories
        specialized_handling = {
            "real_estate": {
                "search_terms": ["homes for sale", "properties", "real estate listings"],
                "required_fields": ["price", "address", "bedrooms", "bathrooms", "sqft"],
                "follow_detail_pages": True,
                "handle_map_views": True,
                "description": "Specialized handling for real estate sites"
            },
            "e_commerce": {
                "search_terms": ["products", "shop", "buy online"],
                "required_fields": ["title", "price", "image", "description"],
                "handle_variants": True,
                "extract_reviews": True,
                "description": "Specialized handling for e-commerce sites"
            },
            "content_sites": {
                "search_terms": ["articles", "news", "blog"],
                "required_fields": ["title", "author", "date", "content"],
                "extract_comments": False,
                "handle_paywalls": True,
                "description": "Specialized handling for content sites"
            },
            "directory_sites": {
                "search_terms": ["listings", "directory", "business"],
                "required_fields": ["name", "address", "phone", "category"],
                "handle_filters": True,
                "extract_hours": True,
                "description": "Specialized handling for directory sites"
            }
        }
        
        # Return specialized handling for this category, or generic if not recognized
        for known_category, handling in specialized_handling.items():
            if known_category in category_lower or category_lower in known_category:
                return handling
        
        # Generic handling for unknown categories
        return {
            "search_terms": ["listings", "results", "main"],
            "required_fields": ["title", "description", "url"],
            "follow_detail_pages": True,
            "description": "Generic handling for unknown site category"
        }
    
    def _generate_fallback_mechanism(self, error_type: str) -> Dict[str, Any]:
        """Generate fallback mechanism for an error type"""
        error_lower = error_type.lower()
        
        # Define fallback mechanisms for different error types
        fallback_mechanisms = {
            "connection_error": {
                "retry": True,
                "max_retries": 3,
                "backoff_factor": 2.0,
                "use_proxy": True,
                "description": "Fallback for connection errors"
            },
            "timeout_error": {
                "retry": True,
                "max_retries": 2,
                "increase_timeout": True,
                "use_simplified_request": True,
                "description": "Fallback for timeout errors"
            },
            "extraction_error": {
                "try_alternative_selectors": True,
                "use_ai_extraction": True,
                "use_html_structure": True,
                "description": "Fallback for extraction errors"
            },
            "schema_validation_error": {
                "use_partial_results": True,
                "relax_validation": True,
                "normalize_data": True,
                "description": "Fallback for schema validation errors"
            }
        }
        
        # Return fallback mechanism for this error type, or generic if not recognized
        for known_error, mechanism in fallback_mechanisms.items():
            if known_error in error_lower or error_lower in known_error:
                return mechanism
        
        # Generic fallback for unknown error types
        return {
            "retry": True,
            "max_retries": 2,
            "log_detailed_error": True,
            "use_alternative_approach": True,
            "description": "Generic fallback for unknown errors"
        }
    
    def _generate_retry_strategy(self, error_type: str) -> Dict[str, Any]:
        """Generate retry strategy for an error type"""
        error_lower = error_type.lower()
        
        # Define retry strategies for different error types
        retry_strategies = {
            "connection_error": {
                "max_retries": 3,
                "initial_wait": 1.0,
                "backoff_factor": 2.0,
                "jitter": True,
                "description": "Exponential backoff with jitter for connection errors"
            },
            "timeout_error": {
                "max_retries": 2,
                "initial_wait": 2.0,
                "increase_timeout": True,
                "description": "Increased timeout for timeout errors"
            },
            "extraction_error": {
                "max_retries": 2,
                "try_different_approach": True,
                "approaches": ["css", "xpath", "regex", "ai"],
                "description": "Try different extraction approaches"
            },
            "url_processing_error": {
                "max_retries": 2,
                "initial_wait": 1.0,
                "normalize_url": True,
                "description": "Normalize URL and retry"
            }
        }
        
        # Return retry strategy for this error type, or generic if not recognized
        for known_error, strategy in retry_strategies.items():
            if known_error in error_lower or error_lower in known_error:
                return strategy
        
        # Generic retry strategy for unknown error types
        return {
            "max_retries": 2,
            "initial_wait": 1.0,
            "backoff_factor": 1.5,
            "description": "Generic retry strategy for unknown errors"
        }

def main():
    """Run the continuous improvement system"""
    improvement_system = ContinuousImprovementSystem()
    
    # Analyze test results
    analysis = improvement_system.analyze_test_results()
    print("\nImprovement Analysis:")
    for area, issues in analysis.get("improvement_areas", {}).items():
        print(f"\n{area.replace('_', ' ').title()}:")
        for issue in issues:
            print(f"- {issue.get('suggestion', 'No suggestion')}")
    
    # Generate improvements
    improvements = improvement_system.generate_improvements()
    
    # Apply improvements
    results = improvement_system.apply_improvements(improvements)
    
    print("\nImprovements Applied:")
    for improvement in results.get("applied_pattern_improvements", []):
        status = "Success" if improvement.get("success", False) else "Failed"
        print(f"- {improvement.get('field', 'Unknown')}: {status}")
    
    print(f"\nTotal patterns improved: {improvement_system.history.get('patterns_improved', 0)}")
    print(f"Total strategies improved: {improvement_system.history.get('strategies_improved', 0)}")
    print(f"Total improvement runs: {len(improvement_system.history.get('improvement_runs', []))}")

if __name__ == "__main__":
    main()