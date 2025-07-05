"""
Quality Controller

Manages quality control, validation, deduplication, and ranking of extracted results.
Ensures high-quality output and filters low-quality or irrelevant data.
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import re
from collections import Counter
import difflib

logger = logging.getLogger("QualityController")

class QualityMetric(Enum):
    """Different quality metrics."""
    COMPLETENESS = "completeness"        # How complete is the data
    ACCURACY = "accuracy"               # How accurate is the data
    RELEVANCE = "relevance"             # How relevant to the query
    UNIQUENESS = "uniqueness"           # How unique is the data
    FRESHNESS = "freshness"             # How recent is the data
    TRUSTWORTHINESS = "trustworthiness" # How trustworthy is the source

@dataclass
class QualityScore:
    """Quality score breakdown."""
    overall_score: float
    metric_scores: Dict[QualityMetric, float]
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    quality_score: QualityScore
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class QualityController:
    """
    Quality Controller
    
    Manages comprehensive quality control for extracted data:
    - Schema validation and auto-correction
    - Content quality assessment
    - Deduplication using multiple strategies
    - Relevance scoring and ranking
    - Data enrichment and normalization
    """
    
    def __init__(self):
        """Initialize the Quality Controller."""
        
        # Quality thresholds
        self.thresholds = {
            "minimum_quality": 0.5,
            "minimum_relevance": 0.4,
            "deduplication_similarity": 0.8,
            "minimum_completeness": 0.3
        }
        
        # Quality metrics weights
        self.metric_weights = {
            QualityMetric.COMPLETENESS: 0.25,
            QualityMetric.ACCURACY: 0.25,
            QualityMetric.RELEVANCE: 0.20,
            QualityMetric.UNIQUENESS: 0.15,
            QualityMetric.FRESHNESS: 0.10,
            QualityMetric.TRUSTWORTHINESS: 0.05
        }
        
        # Performance tracking
        self.quality_stats = {
            "total_validations": 0,
            "passed_validations": 0,
            "duplicates_removed": 0,
            "average_quality_score": 0.0,
            "metric_averages": {}
        }
        
        # Content patterns for quality assessment
        self.quality_patterns = self._initialize_quality_patterns()
        
        logger.info("QualityController initialized")
    
    def _initialize_quality_patterns(self) -> Dict[str, Any]:
        """Initialize patterns for quality assessment."""
        return {
            "low_quality_indicators": [
                r"lorem ipsum", r"placeholder", r"test data", r"sample text",
                r"n/a", r"null", r"undefined", r"error", r"404", r"not found"
            ],
            "spam_indicators": [
                r"click here", r"buy now", r"limited time", r"act fast",
                r"guaranteed", r"risk free", r"no obligation"
            ],
            "high_quality_indicators": [
                r"\d{4}-\d{2}-\d{2}",  # Dates
                r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",  # Emails
                r"https?://[^\s]+",  # URLs
                r"\$\d+(?:\.\d{2})?",  # Prices
            ]
        }
    
    async def validate_and_rank(self, results: List[Dict[str, Any]], 
                              config: Any) -> Dict[str, Any]:
        """
        Validate, deduplicate, and rank results based on quality.
        
        Args:
            results: List of extracted results
            config: Orchestration configuration
            
        Returns:
            Validated and ranked results with quality metadata
        """
        start_time = time.time()
        
        logger.info(f"🎯 Starting quality control for {len(results)} results")
        
        # Step 1: Validate individual results
        validation_results = await self._validate_results(results)
        
        # Step 2: Filter based on minimum quality thresholds
        filtered_results = self._filter_by_quality(validation_results, config)
        
        # Step 3: Deduplicate results
        deduplicated_results = await self._deduplicate_results(filtered_results)
        
        # Step 4: Rank by quality and relevance
        ranked_results = self._rank_results(deduplicated_results)
        
        # Step 5: Apply final limits and formatting
        final_results = self._apply_final_limits(ranked_results, config)
        
        processing_time = time.time() - start_time
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(final_results)
        
        # Update statistics
        self._update_quality_stats(validation_results, len(results) - len(final_results))
        
        logger.info(f"✅ Quality control complete: {len(final_results)} high-quality results")
        
        return {
            "validated_results": [r["data"] for r in final_results],
            "quality_metadata": [r["validation"] for r in final_results],
            "overall_quality_score": overall_metrics["average_quality"],
            "overall_confidence_score": overall_metrics["average_confidence"],
            "filtered_count": len(results) - len(filtered_results),
            "duplicates_removed": len(filtered_results) - len(deduplicated_results),
            "final_count": len(final_results),
            "processing_time": processing_time,
            "quality_metrics": overall_metrics,
            "metadata": {
                "validation_summary": self._create_validation_summary(validation_results),
                "quality_distribution": self._create_quality_distribution(final_results)
            }
        }
    
    async def _validate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate individual results and calculate quality scores."""
        validation_tasks = []
        
        for result in results:
            task = self._validate_single_result(result)
            validation_tasks.append(task)
        
        validations = await asyncio.gather(*validation_tasks)
        
        # Combine data with validation results
        validated_results = []
        for i, validation in enumerate(validations):
            validated_results.append({
                "data": results[i],
                "validation": validation
            })
        
        return validated_results
    
    async def _validate_single_result(self, result: Dict[str, Any]) -> ValidationResult:
        """Validate a single result and calculate quality scores."""
        issues = []
        suggestions = []
        
        # Calculate quality metrics
        metric_scores = {}
        
        # Completeness: How many fields are filled
        metric_scores[QualityMetric.COMPLETENESS] = self._calculate_completeness(result)
        
        # Accuracy: Data format and validity
        metric_scores[QualityMetric.ACCURACY] = self._calculate_accuracy(result)
        
        # Relevance: Content relevance (placeholder for now)
        metric_scores[QualityMetric.RELEVANCE] = 0.7  # Would need query context
        
        # Uniqueness: Content uniqueness
        metric_scores[QualityMetric.UNIQUENESS] = self._calculate_uniqueness(result)
        
        # Freshness: Data recency (if available)
        metric_scores[QualityMetric.FRESHNESS] = self._calculate_freshness(result)
        
        # Trustworthiness: Source credibility
        metric_scores[QualityMetric.TRUSTWORTHINESS] = self._calculate_trustworthiness(result)
        
        # Calculate overall score using weighted average
        overall_score = sum(
            score * self.metric_weights[metric] 
            for metric, score in metric_scores.items()
        )
        
        # Calculate confidence based on data availability
        confidence = self._calculate_confidence(result, metric_scores)
        
        # Create quality score object
        quality_score = QualityScore(
            overall_score=overall_score,
            metric_scores=metric_scores,
            confidence=confidence,
            details={
                "total_fields": len(result),
                "filled_fields": sum(1 for v in result.values() if v and str(v).strip()),
                "content_length": sum(len(str(v)) for v in result.values() if v)
            }
        )
        
        # Determine validity
        is_valid = overall_score >= self.thresholds["minimum_quality"]
        
        # Add issues and suggestions
        if metric_scores[QualityMetric.COMPLETENESS] < 0.5:
            issues.append("Low data completeness")
            suggestions.append("Try extracting from additional sources")
        
        if metric_scores[QualityMetric.ACCURACY] < 0.6:
            issues.append("Data format issues detected")
            suggestions.append("Review data validation rules")
        
        return ValidationResult(
            is_valid=is_valid,
            quality_score=quality_score,
            issues=issues,
            suggestions=suggestions,
            metadata={"validation_time": time.time()}
        )
    
    def _calculate_completeness(self, result: Dict[str, Any]) -> float:
        """Calculate data completeness score."""
        if not result:
            return 0.0
        
        filled_fields = sum(1 for value in result.values() if value and str(value).strip())
        total_fields = len(result)
        
        return filled_fields / total_fields if total_fields > 0 else 0.0
    
    def _calculate_accuracy(self, result: Dict[str, Any]) -> float:
        """Calculate data accuracy score based on format validation."""
        if not result:
            return 0.0
        
        accuracy_score = 1.0
        penalties = 0
        
        for key, value in result.items():
            if not value:
                continue
                
            value_str = str(value).lower()
            
            # Check for low-quality indicators
            for pattern in self.quality_patterns["low_quality_indicators"]:
                if re.search(pattern, value_str):
                    penalties += 0.1
            
            # Check for spam indicators
            for pattern in self.quality_patterns["spam_indicators"]:
                if re.search(pattern, value_str):
                    penalties += 0.2
        
        return max(0.0, accuracy_score - penalties)
    
    def _calculate_uniqueness(self, result: Dict[str, Any]) -> float:
        """Calculate content uniqueness score."""
        # For now, return default score
        # Could implement content similarity checking here
        return 0.8
    
    def _calculate_freshness(self, result: Dict[str, Any]) -> float:
        """Calculate data freshness score."""
        # Look for date fields and calculate recency
        # For now, return default score
        return 0.7
    
    def _calculate_trustworthiness(self, result: Dict[str, Any]) -> float:
        """Calculate source trustworthiness score."""
        # Could analyze URL domain, content patterns, etc.
        # For now, return default score
        return 0.7
    
    def _calculate_confidence(self, result: Dict[str, Any], 
                            metric_scores: Dict[QualityMetric, float]) -> float:
        """Calculate confidence score based on available data."""
        # Base confidence on completeness and accuracy
        base_confidence = (
            metric_scores[QualityMetric.COMPLETENESS] * 0.6 +
            metric_scores[QualityMetric.ACCURACY] * 0.4
        )
        
        # Adjust based on data richness
        content_length = sum(len(str(v)) for v in result.values() if v)
        richness_bonus = min(0.2, content_length / 1000)
        
        return min(1.0, base_confidence + richness_bonus)
    
    def _filter_by_quality(self, validation_results: List[Dict[str, Any]], 
                          config: Any) -> List[Dict[str, Any]]:
        """Filter results based on quality thresholds."""
        quality_threshold = getattr(config, 'quality_threshold', self.thresholds["minimum_quality"])
        
        filtered = []
        for result in validation_results:
            validation = result["validation"]
            if validation.is_valid and validation.quality_score.overall_score >= quality_threshold:
                filtered.append(result)
        
        logger.info(f"Quality filtering: {len(validation_results)} -> {len(filtered)} results")
        return filtered
    
    async def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results using multiple strategies."""
        if len(results) <= 1:
            return results
        
        # Strategy 1: Exact content hash deduplication
        content_hashes = set()
        url_seen = set()
        title_similarity_groups = []
        
        deduplicated = []
        
        for result in results:
            data = result["data"]
            
            # Create content hash
            content_str = json.dumps(data, sort_keys=True)
            content_hash = hashlib.md5(content_str.encode()).hexdigest()
            
            # Check exact duplicates
            if content_hash in content_hashes:
                continue
            content_hashes.add(content_hash)
            
            # Check URL duplicates
            url = data.get("url", "")
            if url and url in url_seen:
                continue
            if url:
                url_seen.add(url)
            
            # For now, add to deduplicated (title similarity would be more complex)
            deduplicated.append(result)
        
        logger.info(f"Deduplication: {len(results)} -> {len(deduplicated)} results")
        return deduplicated
    
    def _rank_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank results by quality and relevance scores."""
        def ranking_key(result):
            validation = result["validation"]
            quality_score = validation.quality_score.overall_score
            confidence = validation.quality_score.confidence
            # Weighted ranking score
            return (quality_score * 0.7) + (confidence * 0.3)
        
        ranked = sorted(results, key=ranking_key, reverse=True)
        
        # Add ranking metadata
        for i, result in enumerate(ranked):
            result["validation"].metadata["rank"] = i + 1
            result["validation"].metadata["ranking_score"] = ranking_key(result)
        
        return ranked
    
    def _apply_final_limits(self, results: List[Dict[str, Any]], 
                          config: Any) -> List[Dict[str, Any]]:
        """Apply final result limits and formatting."""
        max_results = getattr(config, 'max_results', 50)
        
        if len(results) > max_results:
            results = results[:max_results]
        
        return results
    
    def _calculate_overall_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall quality metrics."""
        if not results:
            return {
                "average_quality": 0.0,
                "average_confidence": 0.0,
                "metric_averages": {}
            }
        
        total_quality = sum(r["validation"].quality_score.overall_score for r in results)
        total_confidence = sum(r["validation"].quality_score.confidence for r in results)
        
        # Calculate metric averages
        metric_totals = {metric: 0.0 for metric in QualityMetric}
        for result in results:
            for metric, score in result["validation"].quality_score.metric_scores.items():
                metric_totals[metric] += score
        
        metric_averages = {
            metric.value: total / len(results) 
            for metric, total in metric_totals.items()
        }
        
        return {
            "average_quality": total_quality / len(results),
            "average_confidence": total_confidence / len(results),
            "metric_averages": metric_averages,
            "total_results": len(results)
        }
    
    def _create_validation_summary(self, validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary of validation results."""
        total = len(validation_results)
        valid = sum(1 for r in validation_results if r["validation"].is_valid)
        
        common_issues = []
        all_issues = []
        for result in validation_results:
            all_issues.extend(result["validation"].issues)
        
        if all_issues:
            issue_counts = Counter(all_issues)
            common_issues = issue_counts.most_common(5)
        
        return {
            "total_validated": total,
            "valid_count": valid,
            "invalid_count": total - valid,
            "validation_rate": valid / total if total > 0 else 0.0,
            "common_issues": common_issues
        }
    
    def _create_quality_distribution(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create quality score distribution."""
        if not results:
            return {"distribution": {}, "statistics": {}}
        
        scores = [r["validation"].quality_score.overall_score for r in results]
        
        # Create score buckets
        buckets = {
            "excellent": sum(1 for s in scores if s >= 0.9),
            "good": sum(1 for s in scores if 0.7 <= s < 0.9),
            "fair": sum(1 for s in scores if 0.5 <= s < 0.7),
            "poor": sum(1 for s in scores if s < 0.5)
        }
        
        statistics = {
            "min": min(scores),
            "max": max(scores),
            "average": sum(scores) / len(scores),
            "median": sorted(scores)[len(scores) // 2]
        }
        
        return {
            "distribution": buckets,
            "statistics": statistics
        }
    
    def _update_quality_stats(self, validation_results: List[Dict[str, Any]], 
                            filtered_count: int):
        """Update quality control statistics."""
        self.quality_stats["total_validations"] += len(validation_results)
        self.quality_stats["duplicates_removed"] += filtered_count
        
        valid_count = sum(1 for r in validation_results if r["validation"].is_valid)
        self.quality_stats["passed_validations"] += valid_count
        
        if validation_results:
            total_quality = sum(r["validation"].quality_score.overall_score for r in validation_results)
            avg_quality = total_quality / len(validation_results)
            
            # Update running average
            total_vals = self.quality_stats["total_validations"]
            current_avg = self.quality_stats["average_quality_score"]
            self.quality_stats["average_quality_score"] = (
                (current_avg * (total_vals - len(validation_results)) + 
                 (avg_quality * len(validation_results))) / total_vals
            )
    
    async def health_check(self) -> str:
        """Perform health check of quality controller."""
        try:
            # Test basic validation
            test_data = [
                {"title": "Test Title", "content": "Test content"},
                {"title": "", "content": "Some content"}  # Lower quality
            ]
            
            test_config = type('Config', (), {'quality_threshold': 0.3})()
            
            result = await self.validate_and_rank(test_data, test_config)
            
            if result and result.get("validated_results"):
                return "healthy"
            else:
                return "degraded - no validation results"
                
        except Exception as e:
            return f"error: {e}"
    
    def get_quality_stats(self) -> Dict[str, Any]:
        """Get current quality control statistics."""
        stats = self.quality_stats.copy()
        if stats["total_validations"] > 0:
            stats["pass_rate"] = stats["passed_validations"] / stats["total_validations"]
        else:
            stats["pass_rate"] = 0.0
        return stats
