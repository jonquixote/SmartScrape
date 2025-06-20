import json
import hashlib
from typing import List, Dict, Any

def combine_and_deduplicate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Combines and deduplicates a list of structured scraping results.

    This function attempts to merge results from different sources,
    prioritizing more complete data and removing duplicates based on content hashes.

    Args:
        results: A list of dictionaries, where each dictionary represents
                 extracted data from a single page or item.

    Returns:
        A dictionary containing:
        - "combined_results": A list of combined and deduplicated dictionaries.
        - "deduplicated_count": The number of unique items after deduplication.
    """
    if not results:
        return {"combined_results": [], "deduplicated_count": 0}

    unique_results = {} # Use a dictionary to store unique items, keyed by a hash
    
    for item in results:
        # Create a hash for deduplication.
        # For simplicity, we'll hash a JSON representation of the item.
        # In a real-world scenario, you might hash specific key fields
        # that uniquely identify an item (e.g., product ID, article URL + title).
        item_hash = _generate_item_hash(item)
        
        if item_hash not in unique_results:
            unique_results[item_hash] = item
        else:
            # If a duplicate is found, attempt to merge it with the existing one
            # This simple merge prioritizes the new item's values if the existing one is empty
            # A more advanced merge would involve deep merging or AI-assisted reconciliation
            existing_item = unique_results[item_hash]
            for key, value in item.items():
                if key not in existing_item or not existing_item[key]:
                    existing_item[key] = value
            unique_results[item_hash] = existing_item # Update with potentially merged item

    combined_list = list(unique_results.values())
    return {
        "combined_results": combined_list,
        "deduplicated_count": len(combined_list)
    }

def _generate_item_hash(item: Dict[str, Any]) -> str:
    """
    Generates a SHA256 hash for a dictionary item for deduplication.
    Converts the dictionary to a sorted JSON string to ensure consistent hashing.
    """
    # Convert to a canonical JSON string for consistent hashing
    # Sort keys to ensure the same content always produces the same hash
    item_string = json.dumps(item, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(item_string.encode('utf-8')).hexdigest()

# Example of a more advanced reconciliation (could be integrated with AI)
# from ai_helpers.response_parser import ResponseParser
# async def combine_and_deduplicate_results_ai_assisted(results: List[Dict[str, Any]]) -> Dict[str, Any]:
#     parser = ResponseParser()
#     reconciled_data = await parser.reconcile_extraction_results(results, strategy="complementary")
#     return {"combined_results": [reconciled_data], "deduplicated_count": 1}
