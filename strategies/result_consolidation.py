"""
Result Consolidation Module

This module handles combining and deduplicating results from multiple pages,
prioritizing higher quality data, and building a consolidated view for the user.
"""

import json
import re
import logging
import html
import markdown
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
from datetime import datetime
from collections import defaultdict
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ResultConsolidation")

class ResultConsolidator:
    """
    Class for consolidating, deduplicating and enhancing results from multiple pages.
    
    This class provides methods to:
    - Detect and remove duplicate results across multiple pages
    - Merge complementary information from different sources
    - Organize results based on structure and content type
    - Apply intelligent sorting and prioritization based on relevance and quality
    - Format output for different use cases
    """
    
    def __init__(self):
        """Initialize the result consolidator."""
        pass
        
    def consolidate_results(self, results: List[Dict[str, Any]], extraction_request: str = None) -> List[Dict[str, Any]]:
        """
        Consolidate results from multiple pages into a unified output.
        
        This method applies several stages of processing:
        1. Normalizes data formats across different sources
        2. Determines the structure of the results (list, table, detail page, etc.)
        3. Applies the appropriate consolidation strategy based on structure
        4. Sorts and prioritizes the most relevant and complete results
        5. Adds metadata about the consolidation process
        
        Args:
            results: List of extraction results, where each result is a dictionary containing
                    at minimum a "data" field with the extracted content
            extraction_request: Original extraction request to guide consolidation. This helps
                               determine intent and improve result organization.
            
        Returns:
            Consolidated list of results with duplicates removed and complementary information merged
        """
        if not results:
            return []
        
        # Step 1: Normalize data format
        normalized_results = self._normalize_results(results)
        
        # Step 2: Determine result structure and extraction type
        result_structure = self._determine_result_structure(normalized_results)
        
        # Step 3: Apply appropriate consolidation strategy based on structure
        if result_structure == "list_of_items":
            consolidated = self._consolidate_list_items(normalized_results)
        elif result_structure == "single_item_details":
            consolidated = self._consolidate_item_details(normalized_results)
        elif result_structure == "tabular_data":
            consolidated = self._consolidate_tabular_data(normalized_results)
        elif result_structure == "mixed_content":
            consolidated = self._consolidate_mixed_content(normalized_results, extraction_request)
        else:
            # Default consolidation
            consolidated = normalized_results
            
        # Step 4: Sort and prioritize results
        sorted_results = self._sort_and_prioritize(consolidated)
        
        # Step 5: Add metadata about consolidation
        consolidation_metadata = {
            "consolidated_from": len(results),
            "consolidation_timestamp": datetime.now().isoformat(),
            "structure_type": result_structure,
            "result_count": len(sorted_results)
        }
        
        # Add metadata as an additional non-visible property
        for result in sorted_results:
            if "_metadata" not in result:
                result["_metadata"] = {}
            result["_metadata"]["consolidation"] = consolidation_metadata
        
        return sorted_results
    
    def _normalize_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize results to a consistent format for easier processing.
        
        This method handles different input structures such as:
        - Dictionaries with a 'data' field containing the actual results
        - Lists of items
        - Single scalar values
        
        It ensures all results have consistent metadata fields (_source_url, _depth, _score)
        
        Args:
            results: List of raw extraction results in various formats
            
        Returns:
            List of normalized results with consistent structure and metadata
        """
        normalized = []
        
        for result in results:
            # Extract core data
            data = result.get("data", {})
            source_url = result.get("source_url", "unknown")
            depth = result.get("depth", 0)
            score = result.get("score", 0)
            
            # Check different data structures
            if isinstance(data, list):
                # For list results, process each item
                for item in data:
                    if isinstance(item, dict):
                        item_copy = item.copy()
                        item_copy["_source_url"] = source_url
                        item_copy["_depth"] = depth
                        item_copy["_score"] = score
                        normalized.append(item_copy)
            elif isinstance(data, dict):
                # For dictionary results, add source metadata
                data_copy = data.copy()
                data_copy["_source_url"] = source_url
                data_copy["_depth"] = depth
                data_copy["_score"] = score
                normalized.append(data_copy)
            else:
                # For scalar results, create a basic dict
                normalized.append({
                    "content": data,
                    "_source_url": source_url,
                    "_depth": depth,
                    "_score": score
                })
        
        return normalized
    
    def _determine_result_structure(self, results: List[Dict[str, Any]]) -> str:
        """
        Analyze results to determine their structure for appropriate consolidation.
        
        This method examines the structure and relationships between results to categorize them
        into one of several structural patterns, which guides the consolidation process.
        
        The structure types are:
        - "list_of_items": Collection of similar items (e.g., product listings, search results)
        - "single_item_details": Different aspects of the same entity (e.g., product details)
        - "tabular_data": Data that should be presented in table format
        - "mixed_content": Heterogeneous content requiring separate processing
        
        Args:
            results: List of normalized result dictionaries
            
        Returns:
            Structure type as a string: "list_of_items", "single_item_details", 
            "tabular_data", or "mixed_content"
        """
        if not results:
            return "unknown"
        
        # Check if results look like a list of similar items (e.g., product listings)
        if all(self._similar_keys(results[0], item, similarity_threshold=0.7) for item in results):
            # Check if items have multiple child objects or arrays (indicating tabular data)
            if self._contains_tabular_data(results[0]):
                return "tabular_data"
            else:
                return "list_of_items"
        
        # Check if results might be different aspects of the same item
        source_urls = set(item.get("_source_url", "") for item in results)
        if len(source_urls) <= 3:
            return "single_item_details"
        
        # Default to mixed content
        return "mixed_content"
    
    def _similar_keys(self, dict1: Dict[str, Any], dict2: Dict[str, Any], similarity_threshold: float = 0.7) -> bool:
        """Check if two dictionaries have similar keys."""
        if not isinstance(dict1, dict) or not isinstance(dict2, dict):
            return False
            
        keys1 = set(k for k in dict1.keys() if not k.startswith('_'))
        keys2 = set(k for k in dict2.keys() if not k.startswith('_'))
        
        if not keys1 or not keys2:
            return False
            
        # Calculate Jaccard similarity
        intersection = len(keys1.intersection(keys2))
        union = len(keys1.union(keys2))
        
        return (intersection / union) >= similarity_threshold
    
    def _contains_tabular_data(self, item: Dict[str, Any]) -> bool:
        """Check if an item contains nested lists or objects that might indicate tabular data."""
        if not isinstance(item, dict):
            return False
            
        # Look for arrays or objects in values
        for key, value in item.items():
            if key.startswith('_'):
                continue
                
            if isinstance(value, list) and len(value) > 0:
                # Check if all elements in the list are similar (indicating tabular data)
                if all(isinstance(x, dict) for x in value):
                    return True
        
        return False
    
    def _consolidate_list_items(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Consolidate a list of similar items, removing duplicates and merging complementary information.
        
        This is useful for product listings, search results, etc.
        
        Args:
            results: List of normalized result dictionaries representing similar items
            
        Returns:
            List of consolidated items with duplicates removed and complementary information merged
        """
        # Use a dictionary to track unique items and merge complementary information
        unique_items = {}
        
        for item in results:
            # Create an identifier for this item to detect duplicates
            identifier = self._create_item_identifier(item)
            
            if identifier in unique_items:
                # Compare items and keep the better one, or merge them
                existing_item = unique_items[identifier]
                
                # Calculate completeness scores
                new_score = self._calculate_completeness(item)
                existing_score = self._calculate_completeness(existing_item)
                
                # If new item is better, replace the existing one
                if new_score > existing_score:
                    unique_items[identifier] = item
                elif new_score == existing_score:
                    # If scores are equal, merge complementary fields
                    merged = self._merge_complementary_items(existing_item, item)
                    unique_items[identifier] = merged
            else:
                # New unique item
                unique_items[identifier] = item
        
        # Convert back to a list
        return list(unique_items.values())
    
    def _create_item_identifier(self, item: Dict[str, Any]) -> str:
        """
        Create a unique identifier for an item to detect duplicates.
        
        Uses multiple approaches for robustness.
        
        Args:
            item: Dictionary representing an item
            
        Returns:
            Unique identifier string for the item
        """
        # Try standard ID fields first
        id_fields = ["id", "uid", "guid", "uuid"]
        for field in id_fields:
            if field in item and item[field]:
                return f"{field}:{item[field]}"
        
        # Try URL or link fields
        url_fields = ["url", "link", "href", "source", "page", "product_url"]
        for field in url_fields:
            if field in item and item[field]:
                return f"url:{item[field]}"
        
        # Try name/title fields
        name_fields = ["name", "title", "heading"]
        for field in name_fields:
            if field in item and item[field]:
                # For names, use case-insensitive comparison and clean whitespace
                clean_name = re.sub(r'\s+', ' ', str(item[field]).lower()).strip()
                return f"name:{clean_name}"
        
        # Fall back to creating a composite key from multiple fields
        composite_parts = []
        
        # Common fields that might help identify an item
        potential_fields = ["address", "location", "price", "date", "brand", "author"]
        
        for field in potential_fields:
            if field in item and item[field]:
                value = str(item[field]).lower().strip()
                composite_parts.append(f"{field[:3]}:{value[:20]}")
        
        if composite_parts:
            return "composite:" + "|".join(composite_parts)
            
        # Last resort: hash all non-metadata values
        values_str = "|".join(str(v) for k, v in item.items() if not k.startswith("_") and v)
        return f"hash:{hash(values_str)}"
    
    def _calculate_completeness(self, item: Dict[str, Any]) -> float:
        """
        Calculate a completeness score for an item.
        
        Higher scores indicate more complete information.
        
        Args:
            item: Dictionary representing an item
            
        Returns:
            Completeness score as a float between 0 and 1
        """
        if not isinstance(item, dict):
            return 0.0
            
        # Count non-empty fields that aren't metadata
        total_fields = 0
        non_empty_fields = 0
        
        for key, value in item.items():
            if key.startswith('_'):
                continue
                
            total_fields += 1
            if value is not None and value != "" and value != [] and value != {}:
                non_empty_fields += 1
        
        if total_fields == 0:
            return 0.0
            
        # Base score on field completeness
        completeness_score = non_empty_fields / total_fields
        
        # Bonus for original source score
        source_score = item.get("_score", 0)
        
        # Combined score with completeness having more weight
        return (0.8 * completeness_score) + (0.2 * min(source_score, 1.0))
    
    def _merge_complementary_items(self, item1: Dict[str, Any], item2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two items, keeping the best information from each.
        
        This handles cases where different sources have complementary information.
        
        Args:
            item1: First item dictionary
            item2: Second item dictionary
            
        Returns:
            Merged item dictionary with the best information from both items
        """
        merged = item1.copy()
        
        for key, value2 in item2.items():
            # Skip metadata fields
            if key.startswith('_'):
                continue
                
            # If key doesn't exist in item1, add it
            if key not in merged or merged[key] is None or merged[key] == "":
                merged[key] = value2
            elif isinstance(merged[key], (list, dict)) and isinstance(value2, (list, dict)):
                # Try to merge lists/dicts
                if isinstance(merged[key], list) and isinstance(value2, list):
                    # For lists, append unique items
                    merged[key].extend([x for x in value2 if x not in merged[key]])
                elif isinstance(merged[key], dict) and isinstance(value2, dict):
                    # For dicts, recursively merge
                    for inner_key, inner_value in value2.items():
                        if inner_key not in merged[key] or merged[key][inner_key] is None or merged[key][inner_key] == "":
                            merged[key][inner_key] = inner_value
            elif self._value_is_better(value2, merged[key]):
                # Replace with value2 if it seems better
                merged[key] = value2
        
        # Keep track of all sources
        sources = set()
        if "_source_url" in item1:
            sources.add(item1["_source_url"])
        if "_source_url" in item2:
            sources.add(item2["_source_url"])
        
        merged["_source_urls"] = list(sources)
        
        # Use the better score
        merged["_score"] = max(item1.get("_score", 0), item2.get("_score", 0))
        
        return merged
    
    def _value_is_better(self, value1: Any, value2: Any) -> bool:
        """Determine if value1 is better than value2."""
        # If one is empty and the other isn't, the non-empty one is better
        if not value2 and value1:
            return True
            
        # For strings, longer might be better (more detailed)
        if isinstance(value1, str) and isinstance(value2, str):
            if len(value1) > len(value2) * 1.5:  # Significantly longer
                return True
        
        return False
    
    def _consolidate_item_details(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Consolidate details about a single item from multiple sources.
        
        This is useful for detailed product pages, entity details, etc.
        
        Args:
            results: List of normalized result dictionaries representing different aspects of the same item
            
        Returns:
            List of consolidated item details with complementary information merged
        """
        # Handle cases where we might have results about a single entity
        
        # Group by URL pattern to identify if they're from the same entity
        url_groups = defaultdict(list)
        
        for item in results:
            source_url = item.get("_source_url", "")
            # Extract base URL pattern for grouping
            base_pattern = self._extract_url_pattern(source_url)
            url_groups[base_pattern].append(item)
        
        # If most items come from the same base URL pattern, they likely describe one entity
        main_entities = []
        
        for base_pattern, items in url_groups.items():
            if not items:
                continue
                
            # Merge items from the same group into a single comprehensive entity
            entity = items[0].copy()
            
            for item in items[1:]:
                entity = self._merge_complementary_items(entity, item)
            
            # Add to entities list
            main_entities.append(entity)
        
        # Sort entities by score
        main_entities.sort(key=lambda x: x.get("_score", 0), reverse=True)
        
        return main_entities
    
    def _extract_url_pattern(self, url: str) -> str:
        """Extract a pattern from a URL for grouping similar URLs."""
        if not url or url == "unknown":
            return "unknown"
            
        try:
            # Remove protocol and query parameters
            clean_url = re.sub(r'https?://', '', url)
            clean_url = clean_url.split('?')[0].split('#')[0]
            
            # Extract domain and first part of path
            parts = clean_url.split('/')
            if len(parts) >= 2:
                return f"{parts[0]}/{parts[1]}"
            return parts[0]
        except Exception:
            return "unknown"
    
    def _consolidate_tabular_data(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Consolidate tabular data (e.g., tables, lists of records).
        
        This handles cases like table extraction where each record is a row.
        
        Args:
            results: List of normalized result dictionaries representing tabular data
            
        Returns:
            List of consolidated tables with duplicates removed and complementary information merged
        """
        # Check what kind of tabular data we're dealing with
        all_tables = []
        
        # First, identify common fields to determine tabular structure
        for result in results:
            # Look for list fields that contain tabular data
            for key, value in result.items():
                if key.startswith('_'):
                    continue
                
                if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                    # This looks like a table - a list of records
                    all_tables.append({
                        "field_name": key,
                        "data": value,
                        "source_url": result.get("_source_url", "unknown"),
                        "score": result.get("_score", 0)
                    })
        
        if not all_tables:
            # No tabular data found, fall back to list item consolidation
            return self._consolidate_list_items(results)
        
        # Group tables by field name
        table_groups = defaultdict(list)
        for table in all_tables:
            table_groups[table["field_name"]].append(table)
        
        # Merge each group of tables
        merged_results = []
        
        for field_name, tables in table_groups.items():
            # Combine all rows from all tables with this field name
            all_rows = []
            sources = set()
            
            for table in tables:
                all_rows.extend(table["data"])
                sources.add(table["source_url"])
            
            # Deduplicate the rows
            deduplicated_rows = self._consolidate_list_items(all_rows)
            
            # Create a result object with the deduplicated table
            result = {
                field_name: deduplicated_rows,
                "_source_urls": list(sources),
                "_score": max(table["score"] for table in tables),
                "_table_name": field_name,
                "_table_row_count": len(deduplicated_rows)
            }
            
            merged_results.append(result)
        
        return merged_results
    
    def _consolidate_mixed_content(self, results: List[Dict[str, Any]], extraction_request: str = None) -> List[Dict[str, Any]]:
        """
        Consolidate mixed content types.
        
        This handles cases where extraction results contain different types of data.
        
        Args:
            results: List of normalized result dictionaries with mixed content types
            extraction_request: Original extraction request to guide consolidation
            
        Returns:
            List of consolidated results with appropriate strategies applied to each content type
        """
        # Try to group results by content type
        content_groups = defaultdict(list)
        
        # First, identify content types
        for item in results:
            content_type = self._identify_content_type(item, extraction_request)
            content_groups[content_type].append(item)
        
        # Process each group with the appropriate consolidation strategy
        consolidated = []
        
        for content_type, items in content_groups.items():
            if content_type == "tabular":
                consolidated.extend(self._consolidate_tabular_data(items))
            elif content_type == "list":
                consolidated.extend(self._consolidate_list_items(items))
            elif content_type == "detail":
                consolidated.extend(self._consolidate_item_details(items))
            else:
                # For unknown types, just include as is
                consolidated.extend(items)
        
        return consolidated
    
    def _identify_content_type(self, item: Dict[str, Any], extraction_request: str = None) -> str:
        """
        Identify the content type of an item.
        
        This method uses heuristics and the original extraction request to categorize the content.
        
        The content types are:
        - "tabular": Data that should be presented in table format
        - "list": Collection of similar items
        - "detail": Detailed information about a single entity
        - "unknown": Unrecognized content type
        
        Args:
            item: Dictionary representing a single result item
            extraction_request: Original extraction request to guide content type identification
            
        Returns:
            Content type as a string: "tabular", "list", "detail", or "unknown"
        """
        # Check for tabular data
        for key, value in item.items():
            if key.startswith('_'):
                continue
                
            if isinstance(value, list) and len(value) > 0 and all(isinstance(x, dict) for x in value):
                return "tabular"
        
        # Try to make an educated guess based on structure and extraction request
        if extraction_request:
            extraction_lower = extraction_request.lower()
            
            # Look for keywords suggesting tabular data
            if any(term in extraction_lower for term in ["table", "rows", "columns", "list of", "all", "every"]):
                return "list"
                
            # Look for keywords suggesting detailed item
            if any(term in extraction_lower for term in ["details", "information about", "full", "comprehensive"]):
                return "detail"
        
        # Count non-metadata fields
        field_count = sum(1 for k in item.keys() if not k.startswith('_'))
        
        # Items with many fields are likely detail pages
        if field_count >= 8:
            return "detail"
        elif field_count <= 3:
            return "list"
            
        # Default
        return "unknown"
    
    def _sort_and_prioritize(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort and prioritize consolidated results.
        
        This ensures the most relevant results appear first.
        
        Args:
            results: List of consolidated result dictionaries
            
        Returns:
            List of sorted and prioritized results
        """
        if not results:
            return []
            
        # Sort by score (descending)
        sorted_results = sorted(results, key=lambda x: x.get("_score", 0), reverse=True)
        
        return sorted_results
    
    def generate_structured_output(self, consolidated_results: List[Dict[str, Any]], 
                                  output_format: str = "default") -> Dict[str, Any]:
        """
        Generate a well-structured output from consolidated results.
        
        This method formats the consolidated results into a structured output dictionary
        suitable for different use cases, such as simplified views or detailed reports.
        
        Args:
            consolidated_results: The consolidated results
            output_format: Desired format ("default", "simplified", "detailed")
            
        Returns:
            Structured output dictionary
        """
        if not consolidated_results:
            return {"results": [], "count": 0, "timestamp": datetime.now().isoformat()}
            
        # Determine output type based on results
        result_structure = self._determine_result_structure(consolidated_results)
        
        # Initialize output structure
        output = {
            "metadata": {
                "count": len(consolidated_results),
                "timestamp": datetime.now().isoformat(),
                "result_type": result_structure
            }
        }
        
        # Process based on output format
        if output_format == "simplified":
            # Simple format with minimal metadata
            cleaned_results = []
            
            for item in consolidated_results:
                # Create a clean copy without metadata fields
                clean_item = {k: v for k, v in item.items() if not k.startswith('_')}
                cleaned_results.append(clean_item)
                
            output["results"] = cleaned_results
            
        elif output_format == "detailed":
            # Include all metadata and extra information
            output["results"] = consolidated_results
            
            # Add summary statistics
            sources = set()
            total_fields = 0
            
            for item in consolidated_results:
                if "_source_url" in item:
                    sources.add(item["_source_url"])
                elif "_source_urls" in item:
                    sources.update(item["_source_urls"])
                    
                # Count fields
                total_fields += sum(1 for k in item.keys() if not k.startswith('_'))
            
            output["metadata"]["sources"] = len(sources)
            output["metadata"]["total_fields"] = total_fields
            
        else:
            # Default format - balanced approach
            output["results"] = []
            
            for item in consolidated_results:
                # Keep most metadata but remove internal processing fields
                result_item = {k: v for k, v in item.items() if k != "_metadata"}
                output["results"].append(result_item)
                
        return output
    