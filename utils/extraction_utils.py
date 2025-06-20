"""
Extraction utility functions to be used across different modules
"""

from typing import Dict, Any, Optional, Union
import re
import html

# Define the list of functions to be exported
__all__ = [
    'merge_extraction_configs', 
    'clean_extracted_text', 
    'normalize_whitespace',
    'normalize_search_term',
    'standardize_search_operators',
    'generate_search_variants',
    'clean_search_results_text',
    'extract_keywords_from_text',
    'normalize_for_url'
]

def merge_extraction_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two extraction configurations, with the override config taking precedence.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Configuration to override base values
        
    Returns:
        Merged configuration dictionary
    """
    if not override_config:
        return base_config.copy()
        
    if not base_config:
        return override_config.copy()
        
    # Create a deep copy of the base config
    result = base_config.copy()
    
    # Merge dictionaries recursively
    for key, value in override_config.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            # Recursively merge nested dictionaries
            result[key] = merge_extraction_configs(result[key], value)
        else:
            # Override or add the value
            result[key] = value
            
    return result

def clean_extracted_text(text: Optional[Union[str, list]]) -> str:
    """
    Clean and normalize extracted text by removing extra whitespace, 
    decoding HTML entities, and handling common formatting issues.
    
    Args:
        text: The text to clean, can be a string or a list of strings
        
    Returns:
        Cleaned text as a string
    """
    if text is None:
        return ""
        
    # Convert list to string if necessary
    if isinstance(text, list):
        text = " ".join(str(item) for item in text if item)
    
    # Convert to string if not already
    text = str(text)
    
    # Decode HTML entities
    text = html.unescape(text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Remove common noise patterns (like "Read more", etc.)
    noise_patterns = [
        r'Read more',
        r'Show more',
        r'Click here',
        r'Learn more',
        r'Cookie Policy',
        r'Privacy Policy',
        r'Terms of Service'
    ]
    
    for pattern in noise_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return text

def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text by replacing consecutive whitespace characters
    with a single space and trimming leading/trailing whitespace.
    
    Args:
        text: The text to normalize
        
    Returns:
        Text with normalized whitespace
    """
    if not text:
        return ""
    
    # Replace all whitespace sequences (spaces, tabs, newlines) with a single space
    normalized = re.sub(r'\s+', ' ', text)
    
    # Remove leading and trailing whitespace
    normalized = normalized.strip()
    
    return normalized

def normalize_search_term(query: str, remove_stopwords: bool = True, stopwords_list: list = None) -> str:
    """
    Normalize a search term by removing stopwords (optionally), standardizing special characters,
    and applying consistent formatting for search engines.
    
    Args:
        query: The search query to normalize
        remove_stopwords: Whether to remove common stopwords
        stopwords_list: Optional custom list of stopwords
        
    Returns:
        Normalized search term
    """
    if not query:
        return ""
        
    # Default stopwords list
    default_stopwords = [
        'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
        'be', 'been', 'being', 'in', 'on', 'at', 'to', 'for', 'with',
        'by', 'about', 'against', 'between', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'from', 'up', 'down',
        'of', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
        'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
        'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
        'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
        'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
    ]
    
    # Use custom stopwords if provided, otherwise use default
    stopwords = stopwords_list if stopwords_list is not None else default_stopwords
    
    # Convert to lowercase
    query = query.lower()
    
    # Remove certain special characters
    query = re.sub(r'[^\w\s\-+&\'"]', ' ', query)
    
    # Split into tokens
    tokens = query.split()
    
    # Filter out stopwords if enabled
    if remove_stopwords:
        # Preserve some important search stopwords that add value
        important_stopwords = {'with', 'without', 'near', 'vs', 'versus', 'not', 'no'}
        tokens = [token for token in tokens if token not in stopwords or token in important_stopwords]
    
    # Rejoin tokens
    normalized = ' '.join(tokens)
    
    # Normalize whitespace
    normalized = normalize_whitespace(normalized)
    
    return normalized

def standardize_search_operators(query: str) -> str:
    """
    Standardize search operators in the query to ensure they work across different search engines.
    Handles quotes, plus, minus, AND, OR, etc.
    
    Args:
        query: The search query to standardize
        
    Returns:
        Query with standardized operators
    """
    if not query:
        return ""
    
    # Fix mismatched quote marks
    quotes_count = query.count('"')
    if quotes_count % 2 == 1:  # Odd number of quotes
        query = query + '"'  # Add closing quote
    
    # Ensure space around OR operator
    query = re.sub(r'(\w)OR(\w)', r'\1 OR \2', query)
    
    # Ensure space around AND operator
    query = re.sub(r'(\w)AND(\w)', r'\1 AND \2', query)
    
    # Standardize plus operator (ensure it's attached to the word)
    query = re.sub(r'\+ (\w)', r'+\1', query)
    
    # Standardize minus operator (ensure it's attached to the word)
    query = re.sub(r'- (\w)', r'-\1', query)
    
    # Remove spaces inside quotes (e.g., " term " -> "term")
    query = re.sub(r'"(\s*)([^"]+)(\s*)"', r'"\2"', query)
    
    # Normalize excessive whitespace
    query = normalize_whitespace(query)
    
    return query

def generate_search_variants(term: str) -> list:
    """
    Generate common variants of a search term to help with different search engines.
    
    Args:
        term: The search term to generate variants for
        
    Returns:
        List of variant search terms
    """
    if not term:
        return []
    
    variants = [term]  # Start with the original term
    
    # Add lowercase variant
    lowercase = term.lower()
    if lowercase != term:
        variants.append(lowercase)
    
    # Add variant with quotes for exact matching
    if '"' not in term and ' ' in term:
        variants.append(f'"{term}"')
    
    # Add variant with words in reverse order (if multiple words)
    words = term.split()
    if len(words) > 1:
        reversed_term = ' '.join(reversed(words))
        if reversed_term != term:
            variants.append(reversed_term)
    
    # Add variant with common pluralization (very simple approach)
    if not term.endswith('s') and not term.endswith('es'):
        variants.append(f"{term}s")
    
    # Add variant with hyphen replaced by space and vice versa
    if '-' in term:
        variants.append(term.replace('-', ' '))
    elif ' ' in term:
        variants.append(term.replace(' ', '-'))
    
    # Remove duplicates while preserving order
    unique_variants = []
    for variant in variants:
        if variant not in unique_variants:
            unique_variants.append(variant)
    
    return unique_variants

def clean_search_results_text(text: str) -> str:
    """
    Clean text specifically from search results, handling common patterns found in 
    search engine results.
    
    Args:
        text: The text from search results to clean
        
    Returns:
        Cleaned search results text
    """
    if not text:
        return ""
    
    # First apply general text cleaning
    text = clean_extracted_text(text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove common search result artifacts
    artifacts = [
        r'Cached',
        r'Similar',
        r'\d+ results',
        r'Results \d+-\d+ of',
        r'Page \d+ of \d+',
        r'Showing \d+-\d+ of \d+',
        r'Sponsored',
        r'Ad·',
        r'Sort by:',
        r'Filter:',
        r'Results for'
    ]
    
    for pattern in artifacts:
        text = re.sub(pattern, '', text)
    
    # Remove date patterns often found in search results
    text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', '', text)
    text = re.sub(r'\d{1,2}-\d{1,2}-\d{2,4}', '', text)
    text = re.sub(r'[A-Z][a-z]{2} \d{1,2}, \d{4}', '', text)
    
    # Normalize whitespace again after all the replacements
    text = normalize_whitespace(text)
    
    return text

def extract_keywords_from_text(text: str, max_keywords: int = 10) -> list:
    """
    Extract potential keywords from text using simple frequency analysis.
    
    Args:
        text: The text to extract keywords from
        max_keywords: Maximum number of keywords to return
        
    Returns:
        List of potential keywords
    """
    if not text:
        return []
    
    # Remove common stopwords from consideration
    stopwords = [
        'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
        'be', 'been', 'being', 'in', 'on', 'at', 'to', 'for', 'with',
        'by', 'about', 'against', 'between', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'from', 'up', 'down'
    ]
    
    # Clean and normalize the text
    text = clean_extracted_text(text)
    
    # Tokenize the text
    words = re.findall(r'\b[a-z0-9][a-z0-9\-]{2,}\b', text.lower())
    
    # Remove stopwords
    words = [word for word in words if word not in stopwords]
    
    # Count word frequencies
    word_counts = {}
    for word in words:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
    
    # Sort by frequency (highest first)
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Extract just the words (not counts) and limit to max_keywords
    keywords = [word for word, count in sorted_words[:max_keywords]]
    
    return keywords

def normalize_for_url(term: str) -> str:
    """
    Normalize a search term specifically for inclusion in a URL parameter.
    
    Args:
        term: The search term to normalize for URL usage
        
    Returns:
        URL-safe normalized term
    """
    if not term:
        return ""
    
    # Replace spaces with plus signs (common in search URLs)
    term = term.replace(' ', '+')
    
    # Replace other problematic characters
    term = term.replace('"', '%22')
    term = term.replace('\'', '%27')
    term = term.replace('&', '%26')
    term = term.replace('=', '%3D')
    term = term.replace(':', '%3A')
    term = term.replace(',', '%2C')
    term = term.replace('?', '%3F')
    term = term.replace('/', '%2F')
    term = term.replace('\\', '%5C')
    
    return term