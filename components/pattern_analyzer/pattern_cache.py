"""
Pattern Cache Module

This module provides functionality for storing, retrieving, and managing
pattern data. It implements pattern similarity scoring, cross-site pattern reuse,
incremental learning, and export/import functionality.
"""

import os
import json
import logging
import datetime
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from urllib.parse import urlparse
from difflib import SequenceMatcher

from components.pattern_analyzer.base_analyzer import get_registry, PatternRegistry

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PatternCache")


class PatternCache:
    """
    Manages persistent storage, analysis, and improvement of detected patterns.
    Provides cross-site pattern reuse and incremental learning capabilities.
    """
    
    def __init__(self, cache_dir: str = ".pattern_cache", 
               similarity_threshold: float = 0.7,
               enable_incremental_learning: bool = True):
        """
        Initialize the pattern cache.
        
        Args:
            cache_dir: Directory to store pattern cache files
            similarity_threshold: Threshold for pattern similarity (0.0 to 1.0)
            enable_incremental_learning: Whether to enable pattern improvement over time
        """
        self.cache_dir = cache_dir
        self.similarity_threshold = similarity_threshold
        self.enable_incremental_learning = enable_incremental_learning
        
        # Dictionary for domain patterns and metadata
        self.domain_patterns = {}
        self.pattern_metadata = {}
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Track pattern usage for incremental learning
        self.pattern_usage_stats = {}
        
        # Load existing patterns if available
        self._initialize_cache()
    
    def save_pattern(self, pattern_type: str, url: str, pattern_data: Dict[str, Any],
                    confidence: float) -> bool:
        """
        Save a pattern to the cache.
        
        Args:
            pattern_type: Type of pattern (e.g., "listing", "pagination")
            url: URL where pattern was detected
            pattern_data: Pattern data
            confidence: Confidence score for pattern detection
            
        Returns:
            True if pattern was saved, False otherwise
        """
        domain = self._get_domain(url)
        pattern_key = f"{pattern_type}:{url}"
        
        # If this exact pattern exists with higher confidence, don't overwrite
        if pattern_key in self.pattern_metadata:
            existing_conf = self.pattern_metadata[pattern_key].get('confidence', 0)
            if existing_conf >= confidence:
                # Just update usage stats
                self._update_pattern_usage(pattern_key)
                return False
        
        # Save to in-memory cache
        if domain not in self.domain_patterns:
            self.domain_patterns[domain] = {}
        
        # Add domain patterns by type
        if pattern_type not in self.domain_patterns[domain]:
            self.domain_patterns[domain][pattern_type] = []
            
        # Check if pattern already exists
        pattern_exists = False
        for i, existing_pattern in enumerate(self.domain_patterns[domain].get(pattern_type, [])):
            # If pattern exists with same URL, replace it
            if self._get_pattern_url(existing_pattern) == url:
                self.domain_patterns[domain][pattern_type][i] = pattern_data
                pattern_exists = True
                break
        
        # Add new pattern if it doesn't exist
        if not pattern_exists:
            self.domain_patterns[domain][pattern_type].append(pattern_data)
        
        # Add metadata
        self.pattern_metadata[pattern_key] = {
            'confidence': confidence,
            'domain': domain,
            'url': url,
            'type': pattern_type,
            'last_updated': datetime.datetime.now().isoformat(),
            'usage_count': 1
        }
        
        # Save to disk
        self._save_domain_patterns(domain)
        
        return True
    
    def find_pattern(self, pattern_type: str, url: str) -> Optional[Dict[str, Any]]:
        """
        Find a pattern in the cache.
        
        Args:
            pattern_type: Type of pattern to find
            url: URL to find pattern for
            
        Returns:
            Pattern data dictionary or None if not found
        """
        domain = self._get_domain(url)
        pattern_key = f"{pattern_type}:{url}"
        
        # First check for exact URL match
        for pattern in self.domain_patterns.get(domain, {}).get(pattern_type, []):
            pattern_url = self._get_pattern_url(pattern)
            if pattern_url == url:
                # Update usage stats
                self._update_pattern_usage(pattern_key)
                return pattern
        
        # If no exact match, try to find similar patterns in the same domain
        similar_pattern = self._find_similar_pattern(pattern_type, domain, url)
        if similar_pattern:
            return similar_pattern
        
        # If still no match, try other domains with similar URLs or patterns
        return self._find_cross_domain_pattern(pattern_type, url)
    
    def get_domain_patterns(self, domain: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all patterns for a domain.
        
        Args:
            domain: Domain to get patterns for
            
        Returns:
            Dictionary of pattern types to pattern lists
        """
        return self.domain_patterns.get(domain, {})
    
    def remove_pattern(self, pattern_type: str, url: str) -> bool:
        """
        Remove a pattern from the cache.
        
        Args:
            pattern_type: Type of pattern to remove
            url: URL of the pattern to remove
            
        Returns:
            True if pattern was removed, False if not found
        """
        domain = self._get_domain(url)
        pattern_key = f"{pattern_type}:{url}"
        
        if domain in self.domain_patterns and pattern_type in self.domain_patterns[domain]:
            patterns = self.domain_patterns[domain][pattern_type]
            for i, pattern in enumerate(patterns):
                if self._get_pattern_url(pattern) == url:
                    # Remove pattern
                    del patterns[i]
                    
                    # Remove metadata
                    if pattern_key in self.pattern_metadata:
                        del self.pattern_metadata[pattern_key]
                    
                    # Remove usage stats
                    if pattern_key in self.pattern_usage_stats:
                        del self.pattern_usage_stats[pattern_key]
                    
                    # Save changes
                    self._save_domain_patterns(domain)
                    return True
        
        return False
    
    def sync_with_registry(self) -> int:
        """
        Synchronize cache with the global pattern registry.
        
        Returns:
            Number of patterns synchronized
        """
        registry = get_registry()
        sync_count = 0
        
        # Get all patterns in registry
        for pattern_key, pattern in registry.patterns.items():
            # Parse the pattern key (format: "{pattern_type}:{url}")
            if ":" in pattern_key:
                pattern_type, url = pattern_key.split(":", 1)
                
                # Save pattern to cache
                if self.save_pattern(
                    pattern_type=pattern_type,
                    url=url,
                    pattern_data=pattern,
                    confidence=registry.patterns.get(f"{pattern_type}:{url}_confidence", 0.7)
                ):
                    sync_count += 1
        
        return sync_count
    
    def update_registry_from_cache(self) -> int:
        """
        Update global registry with patterns from cache.
        
        Returns:
            Number of patterns updated
        """
        registry = get_registry()
        update_count = 0
        
        # Iterate through all domains and pattern types
        for domain, patterns_by_type in self.domain_patterns.items():
            for pattern_type, patterns in patterns_by_type.items():
                for pattern in patterns:
                    url = self._get_pattern_url(pattern)
                    if url:
                        pattern_key = f"{pattern_type}:{url}"
                        metadata = self.pattern_metadata.get(pattern_key, {})
                        confidence = metadata.get('confidence', 0.7)
                        
                        # Update registry
                        registry.register_pattern(
                            pattern_type=pattern_type,
                            url=url,
                            pattern_data=pattern,
                            confidence=confidence
                        )
                        update_count += 1
        
        return update_count
    
    def export_patterns(self, output_path: str) -> bool:
        """
        Export all patterns to a file.
        
        Args:
            output_path: Path to export file
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            export_data = {
                "domain_patterns": self.domain_patterns,
                "pattern_metadata": self.pattern_metadata,
                "pattern_usage_stats": self.pattern_usage_stats,
                "export_date": datetime.datetime.now().isoformat(),
                "version": 1.0
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
                
            logger.info(f"Exported pattern data to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting pattern data: {str(e)}")
            return False
    
    def import_patterns(self, import_path: str, merge: bool = True) -> int:
        """
        Import patterns from a file.
        
        Args:
            import_path: Path to import file
            merge: Whether to merge with existing patterns (True) or replace (False)
            
        Returns:
            Number of patterns imported
        """
        try:
            with open(import_path, 'r') as f:
                import_data = json.load(f)
            
            imported_domain_patterns = import_data.get("domain_patterns", {})
            imported_metadata = import_data.get("pattern_metadata", {})
            imported_usage_stats = import_data.get("pattern_usage_stats", {})
            
            if not merge:
                # Replace existing data
                self.domain_patterns = imported_domain_patterns
                self.pattern_metadata = imported_metadata
                self.pattern_usage_stats = imported_usage_stats
                
                # Save all domains
                for domain in self.domain_patterns.keys():
                    self._save_domain_patterns(domain)
                    
                return sum(len(patterns) for domain_patterns in imported_domain_patterns.values() 
                         for patterns in domain_patterns.values())
            else:
                # Merge with existing data
                imported_count = 0
                
                # Merge domain patterns
                for domain, patterns_by_type in imported_domain_patterns.items():
                    if domain not in self.domain_patterns:
                        self.domain_patterns[domain] = {}
                        
                    for pattern_type, patterns in patterns_by_type.items():
                        if pattern_type not in self.domain_patterns[domain]:
                            self.domain_patterns[domain][pattern_type] = []
                            
                        # Find existing patterns to avoid duplicates
                        existing_urls = {self._get_pattern_url(p) for p in 
                                        self.domain_patterns[domain].get(pattern_type, [])}
                        
                        for pattern in patterns:
                            url = self._get_pattern_url(pattern)
                            if url and url not in existing_urls:
                                self.domain_patterns[domain][pattern_type].append(pattern)
                                imported_count += 1
                                existing_urls.add(url)
                    
                    # Save domain after merge
                    self._save_domain_patterns(domain)
                
                # Merge metadata
                for key, metadata in imported_metadata.items():
                    if key not in self.pattern_metadata:
                        self.pattern_metadata[key] = metadata
                
                # Merge usage stats (keep higher usage count)
                for key, stats in imported_usage_stats.items():
                    if key not in self.pattern_usage_stats:
                        self.pattern_usage_stats[key] = stats
                    else:
                        # Keep the higher usage count
                        self.pattern_usage_stats[key]['count'] = max(
                            self.pattern_usage_stats[key].get('count', 0),
                            stats.get('count', 0)
                        )
                
                return imported_count
                
        except Exception as e:
            logger.error(f"Error importing pattern data: {str(e)}")
            return 0
    
    def learn_from_new_data(self, pattern_type: str, url: str, 
                          new_data: Dict[str, Any], confidence: float) -> bool:
        """
        Improve existing patterns using new data (incremental learning).
        
        Args:
            pattern_type: Type of pattern
            url: URL of the pattern
            new_data: New pattern data
            confidence: Confidence score for new data
            
        Returns:
            True if pattern was improved, False otherwise
        """
        if not self.enable_incremental_learning:
            return False
            
        domain = self._get_domain(url)
        pattern_key = f"{pattern_type}:{url}"
        
        # Check if pattern exists
        existing_pattern = self.find_pattern(pattern_type, url)
        if not existing_pattern:
            # Just save as new pattern
            return self.save_pattern(pattern_type, url, new_data, confidence)
        
        # Get existing confidence
        existing_confidence = self.pattern_metadata.get(pattern_key, {}).get('confidence', 0.5)
        
        # If new data has higher confidence, replace pattern
        if confidence > existing_confidence:
            return self.save_pattern(pattern_type, url, new_data, confidence)
        
        # Otherwise, try to merge patterns to improve
        merged_pattern = self._merge_patterns(existing_pattern, new_data)
        
        if merged_pattern:
            # Calculate new confidence (weighted average favoring higher confidence)
            if confidence > existing_confidence:
                new_confidence = 0.7 * confidence + 0.3 * existing_confidence
            else:
                new_confidence = 0.7 * existing_confidence + 0.3 * confidence
                
            # Cap at 0.95 to avoid overconfidence
            new_confidence = min(0.95, new_confidence)
            
            # Save merged pattern
            return self.save_pattern(pattern_type, url, merged_pattern, new_confidence)
        
        return False
    
    def calculate_pattern_similarity(self, pattern1: Dict[str, Any], 
                                   pattern2: Dict[str, Any]) -> float:
        """
        Calculate similarity between two patterns.
        
        Args:
            pattern1: First pattern
            pattern2: Second pattern
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Extract key fields for comparison
        pattern1_str = self._pattern_to_string(pattern1)
        pattern2_str = self._pattern_to_string(pattern2)
        
        # Calculate string similarity using sequence matcher
        matcher = SequenceMatcher(None, pattern1_str, pattern2_str)
        return matcher.ratio()
    
    def _initialize_cache(self) -> None:
        """Initialize the cache by loading existing pattern files."""
        if not os.path.exists(self.cache_dir):
            return
            
        # Load domain pattern files
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.json'):
                domain = filename[:-5]  # Remove .json extension
                self._load_domain_patterns(domain)
    
    def _load_domain_patterns(self, domain: str) -> bool:
        """
        Load patterns for a specific domain.
        
        Args:
            domain: Domain to load patterns for
            
        Returns:
            True if patterns were loaded, False otherwise
        """
        filename = os.path.join(self.cache_dir, f"{domain}.json")
        
        if not os.path.exists(filename):
            return False
            
        try:
            with open(filename, 'r') as f:
                domain_data = json.load(f)
                
            # Load patterns by type
            self.domain_patterns[domain] = domain_data.get('patterns', {})
            
            # Load metadata for all patterns in this domain
            for pattern_type, patterns in self.domain_patterns[domain].items():
                for pattern in patterns:
                    url = self._get_pattern_url(pattern)
                    if url:
                        pattern_key = f"{pattern_type}:{url}"
                        self.pattern_metadata[pattern_key] = domain_data.get('metadata', {}).get(pattern_key, {
                            'confidence': 0.7,
                            'domain': domain,
                            'url': url,
                            'type': pattern_type,
                            'last_updated': domain_data.get('last_updated', datetime.datetime.now().isoformat())
                        })
            
            return True
        except Exception as e:
            logger.error(f"Error loading patterns for domain {domain}: {str(e)}")
            return False
    
    def _save_domain_patterns(self, domain: str) -> bool:
        """
        Save patterns for a specific domain.
        
        Args:
            domain: Domain to save patterns for
            
        Returns:
            True if patterns were saved, False otherwise
        """
        if domain not in self.domain_patterns:
            return False
            
        filename = os.path.join(self.cache_dir, f"{domain}.json")
        
        try:
            # Collect metadata for this domain
            domain_metadata = {}
            for key, metadata in self.pattern_metadata.items():
                if metadata.get('domain') == domain:
                    domain_metadata[key] = metadata
            
            domain_data = {
                'domain': domain,
                'patterns': self.domain_patterns[domain],
                'metadata': domain_metadata,
                'last_updated': datetime.datetime.now().isoformat()
            }
            
            with open(filename, 'w') as f:
                json.dump(domain_data, f, indent=2)
                
            return True
        except Exception as e:
            logger.error(f"Error saving patterns for domain {domain}: {str(e)}")
            return False
    
    def _get_domain(self, url: str) -> str:
        """
        Extract domain from URL.
        
        Args:
            url: URL to extract domain from
            
        Returns:
            Domain string
        """
        parsed_url = urlparse(url)
        return parsed_url.netloc
    
    def _get_pattern_url(self, pattern: Dict[str, Any]) -> Optional[str]:
        """
        Extract URL from pattern data.
        
        Args:
            pattern: Pattern data dictionary
            
        Returns:
            URL string or None if not found
        """
        # Check various places where URL might be stored
        return (pattern.get('url') or 
                pattern.get('source_url') or 
                pattern.get('metadata', {}).get('url'))
    
    def _update_pattern_usage(self, pattern_key: str) -> None:
        """
        Update usage statistics for a pattern.
        
        Args:
            pattern_key: Pattern key to update
        """
        if pattern_key not in self.pattern_usage_stats:
            self.pattern_usage_stats[pattern_key] = {
                'count': 0,
                'last_used': None,
                'success_rate': 0,
                'total_attempts': 0
            }
            
        self.pattern_usage_stats[pattern_key]['count'] += 1
        self.pattern_usage_stats[pattern_key]['last_used'] = datetime.datetime.now().isoformat()
    
    def _find_similar_pattern(self, pattern_type: str, domain: str, 
                            url: str) -> Optional[Dict[str, Any]]:
        """
        Find a pattern similar to the requested URL in the same domain.
        
        Args:
            pattern_type: Type of pattern to find
            domain: Domain to search in
            url: Target URL
            
        Returns:
            Similar pattern or None if not found
        """
        if domain not in self.domain_patterns or pattern_type not in self.domain_patterns[domain]:
            return None
            
        # Look for patterns with similar URL structure
        best_match = None
        best_similarity = 0
        
        for pattern in self.domain_patterns[domain][pattern_type]:
            pattern_url = self._get_pattern_url(pattern)
            if not pattern_url:
                continue
                
            # Calculate URL similarity
            similarity = self._calculate_url_similarity(url, pattern_url)
            
            if similarity > self.similarity_threshold and similarity > best_similarity:
                best_match = pattern
                best_similarity = similarity
        
        return best_match
    
    def _find_cross_domain_pattern(self, pattern_type: str, 
                                 url: str) -> Optional[Dict[str, Any]]:
        """
        Find a similar pattern across domains (pattern reuse).
        
        Args:
            pattern_type: Type of pattern to find
            url: Target URL
            
        Returns:
            Similar pattern from another domain or None if not found
        """
        target_domain = self._get_domain(url)
        best_match = None
        best_similarity = 0
        
        # Check patterns in all domains
        for domain, patterns_by_type in self.domain_patterns.items():
            if domain == target_domain or pattern_type not in patterns_by_type:
                continue
                
            for pattern in patterns_by_type[pattern_type]:
                pattern_url = self._get_pattern_url(pattern)
                if not pattern_url:
                    continue
                    
                # For cross-domain, we care about URL path similarity, not full URL
                similarity = self._calculate_path_similarity(url, pattern_url)
                
                if similarity > self.similarity_threshold and similarity > best_similarity:
                    best_match = pattern
                    best_similarity = similarity
        
        return best_match
    
    def _calculate_url_similarity(self, url1: str, url2: str) -> float:
        """
        Calculate similarity between two URLs.
        
        Args:
            url1: First URL
            url2: Second URL
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Simple sequence matching
        matcher = SequenceMatcher(None, url1, url2)
        return matcher.ratio()
    
    def _calculate_path_similarity(self, url1: str, url2: str) -> float:
        """
        Calculate similarity between URL paths (ignoring domain and query).
        
        Args:
            url1: First URL
            url2: Second URL
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        parsed1 = urlparse(url1)
        parsed2 = urlparse(url2)
        
        # Get path components
        path1 = parsed1.path.strip('/')
        path2 = parsed2.path.strip('/')
        
        # If both are empty, they're similar
        if not path1 and not path2:
            return 1.0
            
        # If one is empty, they're not similar
        if not path1 or not path2:
            return 0.0
            
        # Compare path structures
        matcher = SequenceMatcher(None, path1, path2)
        return matcher.ratio()
    
    def _pattern_to_string(self, pattern: Dict[str, Any]) -> str:
        """
        Convert a pattern to a string representation for comparison.
        
        Args:
            pattern: Pattern dictionary
            
        Returns:
            String representation
        """
        # Extract key fields that define the pattern structure
        # The specific fields depend on pattern type
        key_fields = []
        
        # Check for common pattern fields
        if 'selector' in pattern:
            key_fields.append(f"selector:{pattern['selector']}")
            
        if 'selectors' in pattern and isinstance(pattern['selectors'], dict):
            for k, v in pattern['selectors'].items():
                key_fields.append(f"selector:{k}:{v}")
                
        if 'pattern_type' in pattern:
            key_fields.append(f"type:{pattern['pattern_type']}")
        
        # Add any other pattern-specific fields
        for key in ['table_type', 'pagination_type', 'listing_type', 'page_type']:
            if key in pattern and pattern[key]:
                key_fields.append(f"{key}:{pattern[key]}")
        
        # Join all field strings
        return '|'.join(key_fields)
    
    def _merge_patterns(self, pattern1: Dict[str, Any], 
                       pattern2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Merge two patterns to create an improved pattern.
        
        Args:
            pattern1: First pattern
            pattern2: Second pattern
            
        Returns:
            Merged pattern or None if merge failed
        """
        # Check if patterns are of the same type
        if ('pattern_type' in pattern1 and 'pattern_type' in pattern2 and
            pattern1['pattern_type'] != pattern2['pattern_type']):
            return None
            
        # Create a copy of the base pattern
        merged = pattern1.copy()
        
        # Handle selector merging
        if 'selector' in pattern1 and 'selector' in pattern2:
            # If selectors are different, keep the shorter one as it's usually more reliable
            if pattern1['selector'] != pattern2['selector']:
                if len(pattern1['selector']) <= len(pattern2['selector']):
                    merged['selector'] = pattern1['selector']
                else:
                    merged['selector'] = pattern2['selector']
        
        # Handle selectors dictionary
        if 'selectors' in pattern1 and 'selectors' in pattern2:
            selectors1 = pattern1['selectors']
            selectors2 = pattern2['selectors']
            
            # Merge dictionaries
            for key, value in selectors2.items():
                if key not in selectors1:
                    # Add missing keys
                    merged['selectors'][key] = value
                elif selectors1[key] != value:
                    # For differing values, keep the shorter selector
                    if len(selectors1[key]) <= len(value):
                        # Keep existing value
                        pass
                    else:
                        merged['selectors'][key] = value
        
        # Copy any fields from pattern2 that don't exist in pattern1
        for key, value in pattern2.items():
            if key not in merged:
                merged[key] = value
        
        return merged
    
    def record_pattern_success(self, pattern_type: str, url: str, success: bool) -> None:
        """
        Record success or failure of a pattern.
        
        Args:
            pattern_type: Type of pattern
            url: URL of the pattern
            success: Whether the pattern worked successfully
        """
        pattern_key = f"{pattern_type}:{url}"
        
        if pattern_key not in self.pattern_usage_stats:
            self._update_pattern_usage(pattern_key)
        
        stats = self.pattern_usage_stats[pattern_key]
        stats['total_attempts'] = stats.get('total_attempts', 0) + 1
        
        # Calculate new success rate
        success_count = stats.get('success_count', 0)
        if success:
            success_count += 1
            
        stats['success_count'] = success_count
        stats['success_rate'] = success_count / stats['total_attempts']