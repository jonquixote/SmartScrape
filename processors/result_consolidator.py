"""
Result Consolidation and Enhancement Module

This module provides comprehensive result consolidation capabilities including:
- Duplicate detection and merging
- Content ranking by relevance and quality
- Schema enforcement and validation
- Missing data detection and gap analysis
"""

import logging
import hashlib
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime
import difflib

logger = logging.getLogger(__name__)

@dataclass
class ContentSimilarity:
    """Represents similarity between two content items"""
    item1_id: str
    item2_id: str
    similarity_score: float
    similarity_type: str  # exact, near_exact, semantic, partial
    matched_fields: List[str]
    confidence: float

@dataclass
class RankingCriteria:
    """Criteria for ranking content results"""
    relevance_weight: float = 0.4
    quality_weight: float = 0.3
    freshness_weight: float = 0.2
    completeness_weight: float = 0.1
    content_type: str = "GENERAL"

@dataclass
class ConsolidationResult:
    """Result of content consolidation process"""
    deduplicated_items: List[Dict[str, Any]]
    merged_items: List[Dict[str, Any]]
    duplicate_groups: List[List[str]]
    ranking_scores: Dict[str, float]
    quality_metrics: Dict[str, Any]
    missing_data_analysis: Dict[str, Any]

class ResultConsolidator:
    """
    Comprehensive result consolidation and enhancement system.
    
    Provides intelligent duplicate detection, content ranking, schema enforcement,
    and missing data analysis for extracted content.
    """
    
    def __init__(self, nlp_model=None):
        self.logger = logging.getLogger(__name__)
        self.nlp = nlp_model
        
        # Similarity thresholds for different content types
        self.similarity_thresholds = {
            'NEWS_ARTICLES': {
                'exact': 0.95,
                'near_exact': 0.85,
                'semantic': 0.75,
                'partial': 0.6
            },
            'PRODUCT_INFORMATION': {
                'exact': 0.9,
                'near_exact': 0.8,
                'semantic': 0.7,
                'partial': 0.5
            },
            'JOB_LISTINGS': {
                'exact': 0.9,
                'near_exact': 0.8,
                'semantic': 0.7,
                'partial': 0.6
            },
            'CONTACT_INFORMATION': {
                'exact': 0.95,
                'near_exact': 0.85,
                'semantic': 0.7,
                'partial': 0.5
            },
            'GENERAL': {
                'exact': 0.9,
                'near_exact': 0.8,
                'semantic': 0.7,
                'partial': 0.6
            }
        }
        
        # Content type specific field importance for ranking
        self.field_importance = {
            'NEWS_ARTICLES': {
                'title': 0.3,
                'content': 0.4,
                'date': 0.15,
                'author': 0.1,
                'source': 0.05
            },
            'PRODUCT_INFORMATION': {
                'name': 0.25,
                'description': 0.3,
                'price': 0.2,
                'availability': 0.15,
                'reviews': 0.1
            },
            'JOB_LISTINGS': {
                'title': 0.3,
                'company': 0.25,
                'description': 0.25,
                'location': 0.1,
                'salary': 0.1
            },
            'CONTACT_INFORMATION': {
                'name': 0.25,
                'phone': 0.25,
                'email': 0.25,
                'address': 0.2,
                'hours': 0.05
            }
        }
    
    async def consolidate_results(self, results: List[Dict[str, Any]], 
                                content_type: str = "GENERAL",
                                ranking_criteria: RankingCriteria = None,
                                schema_definition: Any = None) -> ConsolidationResult:
        """
        Consolidate extraction results with comprehensive enhancement.
        
        Args:
            results: List of extracted content items
            content_type: Type of content for specialized processing
            ranking_criteria: Custom ranking criteria
            schema_definition: Schema to enforce (optional)
            
        Returns:
            ConsolidationResult with enhanced and consolidated content
        """
        self.logger.info(f"🔄 Starting result consolidation for {len(results)} items of type: {content_type}")
        
        if not results:
            return ConsolidationResult([], [], [], {}, {}, {})
        
        # Step 1: Detect duplicates
        duplicate_groups = await self._detect_duplicates(results, content_type)
        
        # Step 2: Merge duplicates intelligently
        merged_items = await self._merge_duplicates(results, duplicate_groups, content_type)
        
        # Step 3: Enforce schema if provided
        if schema_definition:
            merged_items = await self._enforce_schema(merged_items, schema_definition)
        
        # Step 4: Rank content by relevance and quality
        ranking_scores = await self._rank_content(merged_items, content_type, ranking_criteria)
        
        # Step 5: Detect missing data and gaps
        missing_data_analysis = await self._analyze_missing_data(merged_items, content_type, schema_definition)
        
        # Step 6: Calculate quality metrics
        quality_metrics = await self._calculate_quality_metrics(merged_items, content_type)
        
        # Sort items by ranking score
        sorted_items = sorted(merged_items, key=lambda x: ranking_scores.get(x.get('id', ''), 0.0), reverse=True)
        
        result = ConsolidationResult(
            deduplicated_items=sorted_items,
            merged_items=merged_items,
            duplicate_groups=duplicate_groups,
            ranking_scores=ranking_scores,
            quality_metrics=quality_metrics,
            missing_data_analysis=missing_data_analysis
        )
        
        self.logger.info(f"✅ Consolidation complete: {len(sorted_items)} unique items, "
                        f"{len(duplicate_groups)} duplicate groups merged")
        
        return result
    
    async def _detect_duplicates(self, results: List[Dict[str, Any]], 
                               content_type: str) -> List[List[str]]:
        """
        Detect duplicate content using multiple similarity methods.
        
        Args:
            results: List of content items
            content_type: Type of content for specialized thresholds
            
        Returns:
            List of duplicate groups (each group is list of item IDs)
        """
        self.logger.info(f"🔍 Detecting duplicates among {len(results)} items")
        
        # Assign IDs to items if not present
        for i, item in enumerate(results):
            if 'id' not in item:
                item['id'] = f"item_{i}_{hashlib.md5(str(item).encode()).hexdigest()[:8]}"
        
        # Calculate similarities between all pairs
        similarities = []
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                similarity = await self._calculate_similarity(results[i], results[j], content_type)
                if similarity.similarity_score > self.similarity_thresholds[content_type]['partial']:
                    similarities.append(similarity)
        
        # Group similar items
        duplicate_groups = []
        processed_items = set()
        
        for similarity in sorted(similarities, key=lambda x: x.similarity_score, reverse=True):
            if similarity.item1_id in processed_items and similarity.item2_id in processed_items:
                continue
            
            # Check if this creates a new group or extends existing group
            group_found = False
            for group in duplicate_groups:
                if similarity.item1_id in group or similarity.item2_id in group:
                    if similarity.item1_id not in group:
                        group.append(similarity.item1_id)
                    if similarity.item2_id not in group:
                        group.append(similarity.item2_id)
                    group_found = True
                    break
            
            if not group_found and similarity.similarity_score > self.similarity_thresholds[content_type]['semantic']:
                duplicate_groups.append([similarity.item1_id, similarity.item2_id])
            
            processed_items.add(similarity.item1_id)
            processed_items.add(similarity.item2_id)
        
        self.logger.info(f"📊 Found {len(duplicate_groups)} duplicate groups")
        return duplicate_groups
    
    async def _calculate_similarity(self, item1: Dict[str, Any], item2: Dict[str, Any], 
                                  content_type: str) -> ContentSimilarity:
        """
        Calculate similarity between two content items using multiple methods.
        
        Args:
            item1: First content item
            item2: Second content item
            content_type: Type of content for field-specific comparison
            
        Returns:
            ContentSimilarity object with detailed similarity analysis
        """
        item1_id = item1.get('id', 'unknown1')
        item2_id = item2.get('id', 'unknown2')
        
        # Get important fields for this content type
        important_fields = self.field_importance.get(content_type, {})
        if not important_fields:
            important_fields = {'title': 0.4, 'content': 0.6}
        
        field_similarities = {}
        matched_fields = []
        total_weighted_similarity = 0.0
        total_weight = 0.0
        
        for field, weight in important_fields.items():
            value1 = str(item1.get(field, '')).strip()
            value2 = str(item2.get(field, '')).strip()
            
            if not value1 or not value2:
                continue
            
            # Calculate field similarity
            field_sim = self._calculate_text_similarity(value1, value2)
            field_similarities[field] = field_sim
            
            if field_sim > 0.7:  # Field is considered matched
                matched_fields.append(field)
            
            total_weighted_similarity += field_sim * weight
            total_weight += weight
        
        # Calculate overall similarity
        if total_weight > 0:
            overall_similarity = total_weighted_similarity / total_weight
        else:
            overall_similarity = 0.0
        
        # Determine similarity type
        if overall_similarity >= 0.95:
            similarity_type = "exact"
        elif overall_similarity >= 0.85:
            similarity_type = "near_exact"
        elif overall_similarity >= 0.7:
            similarity_type = "semantic"
        else:
            similarity_type = "partial"
        
        # Calculate confidence based on number of matched fields
        confidence = min(1.0, len(matched_fields) / max(1, len(important_fields)))
        
        return ContentSimilarity(
            item1_id=item1_id,
            item2_id=item2_id,
            similarity_score=overall_similarity,
            similarity_type=similarity_type,
            matched_fields=matched_fields,
            confidence=confidence
        )
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        if not text1 or not text2:
            return 0.0
        
        # Normalize texts
        text1_norm = re.sub(r'\s+', ' ', text1.lower().strip())
        text2_norm = re.sub(r'\s+', ' ', text2.lower().strip())
        
        # Exact match
        if text1_norm == text2_norm:
            return 1.0
        
        # Use difflib for sequence matching
        similarity = difflib.SequenceMatcher(None, text1_norm, text2_norm).ratio()
        
        # Enhance with semantic similarity if NLP model available
        if self.nlp and len(text1_norm) > 10 and len(text2_norm) > 10:
            try:
                doc1 = self.nlp(text1_norm)
                doc2 = self.nlp(text2_norm)
                semantic_sim = doc1.similarity(doc2)
                # Combine sequence and semantic similarity
                similarity = (similarity + semantic_sim) / 2
            except Exception as e:
                self.logger.debug(f"Semantic similarity calculation failed: {e}")
        
        return similarity
    
    async def _merge_duplicates(self, results: List[Dict[str, Any]], 
                              duplicate_groups: List[List[str]], 
                              content_type: str) -> List[Dict[str, Any]]:
        """
        Merge duplicate items intelligently, preserving best information.
        
        Args:
            results: Original list of content items
            duplicate_groups: Groups of duplicate item IDs
            content_type: Type of content for specialized merging
            
        Returns:
            List of merged content items
        """
        self.logger.info(f"🔗 Merging {len(duplicate_groups)} duplicate groups")
        
        # Create ID to item mapping
        items_by_id = {item.get('id'): item for item in results}
        
        # Track which items have been merged
        merged_item_ids = set()
        merged_items = []
        
        for group in duplicate_groups:
            if len(group) < 2:
                continue
            
            # Get items in this duplicate group
            group_items = [items_by_id[item_id] for item_id in group if item_id in items_by_id]
            
            if len(group_items) < 2:
                continue
            
            # Merge the group
            merged_item = await self._merge_item_group(group_items, content_type)
            merged_items.append(merged_item)
            
            # Track merged IDs
            merged_item_ids.update(group)
        
        # Add non-duplicate items
        for item in results:
            if item.get('id') not in merged_item_ids:
                merged_items.append(item)
        
        self.logger.info(f"📦 Merged {len(results)} items into {len(merged_items)} unique items")
        return merged_items
    
    async def _merge_item_group(self, items: List[Dict[str, Any]], 
                              content_type: str) -> Dict[str, Any]:
        """
        Merge a group of duplicate items into a single best item.
        
        Args:
            items: List of duplicate items to merge
            content_type: Type of content for specialized merging
            
        Returns:
            Merged item with best information from all duplicates
        """
        if not items:
            return {}
        
        if len(items) == 1:
            return items[0]
        
        # Start with the item that has the highest quality score
        base_item = max(items, key=lambda x: x.get('quality_score', 0.0))
        merged_item = base_item.copy()
        
        # Get field importance for this content type
        important_fields = self.field_importance.get(content_type, {})
        
        # Merge each field intelligently
        for field in set().union(*(item.keys() for item in items)):
            if field in ['id']:  # Skip system fields
                continue
            
            field_values = []
            for item in items:
                value = item.get(field)
                if value and str(value).strip():
                    field_values.append((value, item.get('quality_score', 0.0)))
            
            if not field_values:
                continue
            
            # Choose best value based on quality and completeness
            if len(field_values) == 1:
                merged_item[field] = field_values[0][0]
            else:
                # For important fields, choose the most complete value
                if field in important_fields:
                    best_value = max(field_values, key=lambda x: (len(str(x[0])), x[1]))
                    merged_item[field] = best_value[0]
                else:
                    # For other fields, choose highest quality
                    best_value = max(field_values, key=lambda x: x[1])
                    merged_item[field] = best_value[0]
        
        # Update metadata to reflect merge
        merged_item['merged_from'] = [item.get('id') for item in items]
        merged_item['merge_count'] = len(items)
        merged_item['merge_timestamp'] = datetime.now().isoformat()
        
        # Recalculate quality scores based on merged content
        if 'quality_score' in merged_item:
            avg_quality = sum(item.get('quality_score', 0.0) for item in items) / len(items)
            completeness_bonus = len([f for f in important_fields if merged_item.get(f)]) / len(important_fields) * 0.1
            merged_item['quality_score'] = min(1.0, avg_quality + completeness_bonus)
        
        return merged_item
    
    async def _rank_content(self, items: List[Dict[str, Any]], 
                          content_type: str,
                          ranking_criteria: RankingCriteria = None) -> Dict[str, float]:
        """
        Rank content items by relevance, quality, freshness, and completeness.
        
        Args:
            items: List of content items to rank
            content_type: Type of content for specialized ranking
            ranking_criteria: Custom ranking criteria
            
        Returns:
            Dictionary mapping item IDs to ranking scores
        """
        if not ranking_criteria:
            ranking_criteria = RankingCriteria(content_type=content_type)
        
        self.logger.info(f"📊 Ranking {len(items)} items with criteria: "
                        f"relevance={ranking_criteria.relevance_weight}, "
                        f"quality={ranking_criteria.quality_weight}")
        
        ranking_scores = {}
        
        for item in items:
            item_id = item.get('id', 'unknown')
            
            # Calculate component scores
            relevance_score = item.get('relevance_score', 0.5)
            quality_score = item.get('quality_score', 0.5)
            freshness_score = self._calculate_freshness_score(item)
            completeness_score = self._calculate_completeness_score(item, content_type)
            
            # Calculate weighted overall score
            overall_score = (
                relevance_score * ranking_criteria.relevance_weight +
                quality_score * ranking_criteria.quality_weight +
                freshness_score * ranking_criteria.freshness_weight +
                completeness_score * ranking_criteria.completeness_weight
            )
            
            ranking_scores[item_id] = overall_score
            
            # Update item with ranking components for transparency
            item['ranking_components'] = {
                'relevance': relevance_score,
                'quality': quality_score,
                'freshness': freshness_score,
                'completeness': completeness_score,
                'overall': overall_score
            }
        
        return ranking_scores
    
    def _calculate_freshness_score(self, item: Dict[str, Any]) -> float:
        """Calculate freshness score based on content date."""
        from datetime import datetime, timedelta
        
        # Look for date fields
        date_fields = ['date', 'published_date', 'created_at', 'timestamp']
        content_date = None
        
        for field in date_fields:
            if field in item and item[field]:
                content_date = item[field]
                break
        
        if not content_date:
            return 0.5  # Neutral score for unknown dates
        
        try:
            # Parse date (simplified - could be enhanced)
            if isinstance(content_date, str):
                # Try common date formats
                for fmt in ['%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S']:
                    try:
                        parsed_date = datetime.strptime(content_date[:19], fmt)
                        break
                    except ValueError:
                        continue
                else:
                    return 0.5
            else:
                parsed_date = content_date
            
            # Calculate age in days
            age_days = (datetime.now() - parsed_date).days
            
            # Fresher content gets higher scores
            if age_days <= 1:
                return 1.0
            elif age_days <= 7:
                return 0.9
            elif age_days <= 30:
                return 0.7
            elif age_days <= 90:
                return 0.5
            else:
                return 0.3
                
        except Exception:
            return 0.5
    
    def _calculate_completeness_score(self, item: Dict[str, Any], content_type: str) -> float:
        """Calculate completeness score based on presence of important fields."""
        important_fields = self.field_importance.get(content_type, {})
        if not important_fields:
            # Generic completeness check
            important_fields = {'title': 0.4, 'content': 0.6}
        
        present_fields = 0
        total_importance = 0
        
        for field, importance in important_fields.items():
            total_importance += importance
            if field in item and item[field] and str(item[field]).strip():
                present_fields += importance
        
        return present_fields / total_importance if total_importance > 0 else 0.0
    
    async def _enforce_schema(self, items: List[Dict[str, Any]], 
                            schema_definition: Any) -> List[Dict[str, Any]]:
        """
        Enforce schema validation and fix common issues.
        
        Args:
            items: List of content items
            schema_definition: Schema to enforce
            
        Returns:
            List of schema-compliant items
        """
        self.logger.info(f"✅ Enforcing schema on {len(items)} items")
        
        validated_items = []
        
        for item in items:
            try:
                # Attempt schema validation
                if hasattr(schema_definition, 'validate'):
                    schema_definition.validate(item)
                    validated_items.append(item)
                elif hasattr(schema_definition, '__call__'):
                    validated_item = schema_definition(**item)
                    validated_items.append(validated_item.dict() if hasattr(validated_item, 'dict') else item)
                else:
                    validated_items.append(item)  # No validation possible
                    
            except Exception as e:
                self.logger.warning(f"Schema validation failed for item {item.get('id', 'unknown')}: {e}")
                # Try to fix common issues and re-validate
                fixed_item = self._attempt_schema_fix(item, schema_definition)
                if fixed_item:
                    validated_items.append(fixed_item)
        
        self.logger.info(f"📋 Schema enforcement: {len(validated_items)}/{len(items)} items passed validation")
        return validated_items
    
    def _attempt_schema_fix(self, item: Dict[str, Any], schema_definition: Any) -> Optional[Dict[str, Any]]:
        """Attempt to fix common schema validation issues."""
        # This is a simplified fix attempt - could be enhanced based on specific schema types
        fixed_item = item.copy()
        
        # Common fixes: ensure required string fields are strings
        for key, value in fixed_item.items():
            if value is None:
                continue
            if not isinstance(value, str) and not isinstance(value, (int, float, bool, list, dict)):
                fixed_item[key] = str(value)
        
        try:
            if hasattr(schema_definition, 'validate'):
                schema_definition.validate(fixed_item)
                return fixed_item
            elif hasattr(schema_definition, '__call__'):
                validated = schema_definition(**fixed_item)
                return validated.dict() if hasattr(validated, 'dict') else fixed_item
        except Exception:
            pass
        
        return None
    
    async def _analyze_missing_data(self, items: List[Dict[str, Any]], 
                                  content_type: str,
                                  schema_definition: Any = None) -> Dict[str, Any]:
        """
        Analyze missing data and identify gaps in extracted content.
        
        Args:
            items: List of content items
            content_type: Type of content
            schema_definition: Optional schema for required field analysis
            
        Returns:
            Analysis of missing data and recommendations
        """
        self.logger.info(f"🔍 Analyzing missing data for {len(items)} items")
        
        if not items:
            return {'total_items': 0, 'missing_fields': {}, 'recommendations': []}
        
        # Get expected fields for this content type
        expected_fields = set(self.field_importance.get(content_type, {}).keys())
        
        # Add schema fields if available
        if schema_definition and hasattr(schema_definition, '__annotations__'):
            expected_fields.update(schema_definition.__annotations__.keys())
        
        # Analyze field presence
        field_presence = {}
        for field in expected_fields:
            present_count = sum(1 for item in items if field in item and item[field] and str(item[field]).strip())
            field_presence[field] = {
                'present_count': present_count,
                'missing_count': len(items) - present_count,
                'presence_rate': present_count / len(items) if items else 0.0
            }
        
        # Identify critical gaps
        critical_gaps = []
        for field, stats in field_presence.items():
            if stats['presence_rate'] < 0.5:  # Less than 50% presence
                critical_gaps.append(field)
        
        # Generate recommendations
        recommendations = []
        for field in critical_gaps:
            recommendations.append({
                'type': 'missing_field',
                'field': field,
                'severity': 'high' if field_presence[field]['presence_rate'] < 0.2 else 'medium',
                'suggestion': f"Improve extraction methods for '{field}' field (only {field_presence[field]['presence_rate']:.1%} coverage)"
            })
        
        # Analyze content quality gaps
        quality_issues = []
        for item in items:
            item_quality = item.get('quality_score', 0.0)
            if item_quality < 0.5:
                missing_fields = [f for f in expected_fields if f not in item or not item[f]]
                quality_issues.append({
                    'item_id': item.get('id', 'unknown'),
                    'quality_score': item_quality,
                    'missing_fields': missing_fields
                })
        
        return {
            'total_items': len(items),
            'expected_fields': list(expected_fields),
            'field_presence': field_presence,
            'critical_gaps': critical_gaps,
            'quality_issues': quality_issues[:10],  # Limit to first 10
            'recommendations': recommendations,
            'overall_completeness': sum(stats['presence_rate'] for stats in field_presence.values()) / len(field_presence) if field_presence else 0.0
        }
    
    async def _calculate_quality_metrics(self, items: List[Dict[str, Any]], 
                                       content_type: str) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics for the consolidated results."""
        if not items:
            return {'total_items': 0}
        
        quality_scores = [item.get('quality_score', 0.0) for item in items]
        relevance_scores = [item.get('relevance_score', 0.0) for item in items]
        
        metrics = {
            'total_items': len(items),
            'average_quality': sum(quality_scores) / len(quality_scores),
            'average_relevance': sum(relevance_scores) / len(relevance_scores),
            'high_quality_items': sum(1 for score in quality_scores if score >= 0.8),
            'low_quality_items': sum(1 for score in quality_scores if score < 0.5),
            'content_type': content_type,
            'completeness_distribution': {
                'complete': sum(1 for item in items if self._calculate_completeness_score(item, content_type) >= 0.8),
                'partial': sum(1 for item in items if 0.5 <= self._calculate_completeness_score(item, content_type) < 0.8),
                'incomplete': sum(1 for item in items if self._calculate_completeness_score(item, content_type) < 0.5)
            }
        }
        
        return metrics
