"""
Result Enhancement and User Feedback Module

This module provides comprehensive result enhancement capabilities including:
- Schema validation status and metadata enhancement
- Semantic relevance scoring
- Confidence score analysis and reporting
- User feedback collection and processing
- Suggestion generation for result improvement

Features:
- Integration with AISchemaGenerator for Pydantic validation
- Semantic similarity scoring using UniversalIntentAnalyzer
- Quality assessment and confidence metrics
- User feedback mechanism for adaptive learning
- Comprehensive result metadata enrichment
"""

import json
import uuid
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Try to import optional dependencies
try:
    from pydantic import BaseModel, ValidationError
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    redis = None

logger = logging.getLogger("ResultEnhancer")

if not HAS_PYDANTIC:
    logger.info("Pydantic not available. Schema validation features will be limited.")
if not HAS_REDIS:
    logger.info("Redis not available. User feedback storage might be in-memory or disabled if Redis was configured.")

class FeedbackType(Enum):
    """Types of user feedback"""
    RELEVANCE = "relevance"
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    QUALITY = "quality"
    GENERAL = "general"

class FeedbackRating(Enum):
    """Feedback rating scale"""
    EXCELLENT = 5
    GOOD = 4
    SATISFACTORY = 3
    POOR = 2
    VERY_POOR = 1

@dataclass
class UserFeedback:
    """Structure for user feedback"""
    feedback_id: str
    query: str
    result_data: Dict[str, Any]
    feedback_type: FeedbackType
    rating: FeedbackRating
    comments: Optional[str]
    field_specific_feedback: Optional[Dict[str, Dict[str, Any]]]
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert feedback to dictionary for storage"""
        data = asdict(self)
        data['feedback_type'] = self.feedback_type.value
        data['rating'] = self.rating.value
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserFeedback':
        """Create feedback from dictionary"""
        data['feedback_type'] = FeedbackType(data['feedback_type'])
        data['rating'] = FeedbackRating(data['rating'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

class ResultEnhancer:
    """
    Enhanced result processor that adds metadata, validation status, 
    confidence scores, and collects user feedback for continuous improvement.
    """
    
    def __init__(self, 
                 intent_analyzer=None, 
                 schema_generator=None,
                 config=None):
        """
        Initialize the ResultEnhancer.
        
        Args:
            intent_analyzer: UniversalIntentAnalyzer instance for semantic analysis
            schema_generator: AISchemaGenerator instance for schema validation
            config: Configuration object with settings
        """
        self.intent_analyzer = intent_analyzer
        self.schema_generator = schema_generator
        self.config = config or {}
        
        # Import components if available
        self._initialize_components()
        
        # Feedback storage
        self.feedback_storage = self._initialize_feedback_storage()
        
        # Enhancement weights for scoring
        self.enhancement_weights = {
            'schema_validation': 0.25,
            'semantic_relevance': 0.30,
            'confidence_score': 0.25,
            'completeness': 0.20
        }
        
        logger.info("ResultEnhancer initialized successfully")

    def _initialize_components(self):
        """Initialize optional components"""
        try:
            if not self.intent_analyzer:
                from components.universal_intent_analyzer import UniversalIntentAnalyzer
                self.intent_analyzer = UniversalIntentAnalyzer()
                logger.info("UniversalIntentAnalyzer initialized")
        except ImportError:
            logger.warning("UniversalIntentAnalyzer not available")
            self.intent_analyzer = None
            
        try:
            if not self.schema_generator:
                from components.ai_schema_generator import AISchemaGenerator
                self.schema_generator = AISchemaGenerator(self.intent_analyzer)
                logger.info("AISchemaGenerator initialized")
        except ImportError:
            logger.warning("AISchemaGenerator not available")
            self.schema_generator = None

    def _initialize_feedback_storage(self):
        """Initialize feedback storage mechanism"""
        storage_type = getattr(self.config, 'FEEDBACK_STORAGE_TYPE', 'memory')
        
        if storage_type == 'redis' and HAS_REDIS and redis:
            try:
                redis_host = getattr(self.config, 'REDIS_HOST', 'localhost')
                redis_port = getattr(self.config, 'REDIS_PORT', 6379)
                redis_db = getattr(self.config, 'FEEDBACK_REDIS_DB', 2)
                
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    decode_responses=True
                )
                self.redis_client.ping()  # Verify connection
                logger.info("Successfully connected to Redis for feedback storage.")
                
                return self.redis_client
            except redis.exceptions.ConnectionError as e:
                logger.error(f"Failed to connect to Redis for feedback storage: {e}")
                return {}
            except Exception as e:
                logger.warning(f"Failed to initialize Redis for feedback storage: {e}")
                return {}
        else:
            logger.info("Feedback storage type not Redis or not configured. Using default (e.g., in-memory or none).")
            # Fall back to in-memory storage
            return {}

    def enhance_results(self, 
                       results: Dict[str, Any], 
                       query: str, 
                       pydantic_schema: Optional[BaseModel] = None,
                       context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enhance results with metadata, confidence scores, validation status, and suggestions.
        
        Args:
            results: Raw extraction results
            query: Original user query
            pydantic_schema: Optional Pydantic schema for validation
            context: Additional context information
            
        Returns:
            Enhanced results dictionary
        """
        try:
            enhancement_start_time = time.time()
            
            # Initialize enhanced results structure
            enhanced_results = {
                "original_results": results,
                "enhanced_metadata": {
                    "enhancement_timestamp": datetime.now().isoformat(),
                    "enhancement_version": "1.0",
                    "query": query,
                    "context": context or {}
                },
                "validation": {},
                "quality_metrics": {},
                "suggestions": [],
                "feedback_summary": {}
            }
            
            # 1. Schema Validation
            validation_results = self._perform_schema_validation(results, pydantic_schema)
            enhanced_results["validation"] = validation_results
            
            # 2. Semantic Relevance Scoring
            relevance_scores = self._calculate_semantic_relevance(results, query, context)
            enhanced_results["quality_metrics"]["semantic_relevance"] = relevance_scores
            
            # 3. Confidence Score Analysis
            confidence_analysis = self._analyze_confidence_scores(results)
            enhanced_results["quality_metrics"]["confidence_analysis"] = confidence_analysis
            
            # 4. Completeness Assessment
            completeness_metrics = self._assess_completeness(results, pydantic_schema)
            enhanced_results["quality_metrics"]["completeness"] = completeness_metrics
            
            # 5. Overall Quality Score
            overall_quality = self._calculate_overall_quality_score(enhanced_results)
            enhanced_results["quality_metrics"]["overall_quality_score"] = overall_quality
            
            # 6. Generate Suggestions
            suggestions = self._generate_improvement_suggestions(enhanced_results)
            enhanced_results["suggestions"] = suggestions
            
            # 7. Add Feedback Summary
            feedback_summary = self._get_feedback_summary(query, results)
            enhanced_results["feedback_summary"] = feedback_summary
            
            # 8. Performance Metrics
            enhancement_time = time.time() - enhancement_start_time
            enhanced_results["enhanced_metadata"]["enhancement_time_seconds"] = enhancement_time
            
            logger.info(f"Results enhanced in {enhancement_time:.3f} seconds")
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Error enhancing results: {str(e)}")
            return {
                "original_results": results,
                "enhancement_error": str(e),
                "enhanced_metadata": {
                    "enhancement_timestamp": datetime.now().isoformat(),
                    "enhancement_failed": True
                }
            }

    def _perform_schema_validation(self, 
                                 results: Dict[str, Any], 
                                 pydantic_schema: Optional[BaseModel]) -> Dict[str, Any]:
        """Perform Pydantic schema validation if schema is provided"""
        validation_result = {
            "schema_provided": pydantic_schema is not None,
            "validation_performed": False,
            "is_valid": None,
            "validation_errors": [],
            "field_validations": {}
        }
        
        if not pydantic_schema or not HAS_PYDANTIC:
            validation_result["validation_skipped_reason"] = "No schema provided or Pydantic not available"
            return validation_result
        
        try:
            # Validate the entire results structure
            validated_data = pydantic_schema(**results)
            validation_result["validation_performed"] = True
            validation_result["is_valid"] = True
            validation_result["validated_data"] = validated_data.dict()
            
        except ValidationError as e:
            validation_result["validation_performed"] = True
            validation_result["is_valid"] = False
            validation_result["validation_errors"] = [
                {
                    "field": ".".join(str(x) for x in error["loc"]),
                    "message": error["msg"],
                    "type": error["type"]
                }
                for error in e.errors()
            ]
            
            # Add field-specific validation results
            for error in e.errors():
                field_path = ".".join(str(x) for x in error["loc"])
                validation_result["field_validations"][field_path] = {
                    "valid": False,
                    "error_message": error["msg"],
                    "error_type": error["type"]
                }
        
        except Exception as e:
            validation_result["validation_error"] = str(e)
            
        return validation_result

    def _calculate_semantic_relevance(self, 
                                    results: Dict[str, Any], 
                                    query: str,
                                    context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate semantic relevance scores using intent analyzer"""
        relevance_result = {
            "semantic_analysis_available": self.intent_analyzer is not None,
            "query_relevance_score": 0.0,
            "field_relevance_scores": {},
            "semantic_matches": []
        }
        
        if not self.intent_analyzer:
            relevance_result["relevance_skipped_reason"] = "UniversalIntentAnalyzer not available"
            return relevance_result
        
        try:
            # Analyze query intent for context
            intent_analysis = self.intent_analyzer.analyze_intent(query)
            
            # Score overall relevance
            if hasattr(self.intent_analyzer, 'calculate_semantic_similarity'):
                # Convert results to text for similarity comparison
                results_text = self._extract_text_from_results(results)
                overall_similarity = self.intent_analyzer.calculate_semantic_similarity(
                    query, results_text
                )
                relevance_result["query_relevance_score"] = overall_similarity
            
            # Score individual fields
            field_scores = {}
            for field, value in results.items():
                if isinstance(value, str) and len(value) > 10:
                    if hasattr(self.intent_analyzer, 'calculate_semantic_similarity'):
                        field_score = self.intent_analyzer.calculate_semantic_similarity(
                            query, value
                        )
                        field_scores[field] = field_score
            
            relevance_result["field_relevance_scores"] = field_scores
            relevance_result["intent_analysis"] = intent_analysis
            
        except Exception as e:
            relevance_result["semantic_analysis_error"] = str(e)
            logger.warning(f"Error in semantic relevance calculation: {e}")
        
        return relevance_result

    def _analyze_confidence_scores(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze existing confidence scores and generate insights"""
        confidence_analysis = {
            "has_confidence_metadata": False,
            "overall_confidence": 0.0,
            "field_confidence_scores": {},
            "confidence_distribution": {},
            "low_confidence_fields": [],
            "high_confidence_fields": []
        }
        
        try:
            # Check for existing confidence metadata
            metadata = results.get("_metadata", {})
            if "confidence" in metadata:
                confidence_analysis["has_confidence_metadata"] = True
                confidence_data = metadata["confidence"]
                
                if isinstance(confidence_data, dict):
                    confidence_analysis["field_confidence_scores"] = confidence_data
                    
                    # Calculate overall confidence
                    if confidence_data:
                        overall_conf = sum(confidence_data.values()) / len(confidence_data)
                        confidence_analysis["overall_confidence"] = overall_conf
                    
                    # Categorize fields by confidence
                    for field, score in confidence_data.items():
                        if score < 0.6:
                            confidence_analysis["low_confidence_fields"].append({
                                "field": field,
                                "score": score
                            })
                        elif score > 0.8:
                            confidence_analysis["high_confidence_fields"].append({
                                "field": field,
                                "score": score
                            })
                
                elif isinstance(confidence_data, (int, float)):
                    confidence_analysis["overall_confidence"] = confidence_data
            
            # Generate confidence distribution
            if confidence_analysis["field_confidence_scores"]:
                scores = list(confidence_analysis["field_confidence_scores"].values())
                confidence_analysis["confidence_distribution"] = {
                    "mean": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "std_dev": self._calculate_std_dev(scores)
                }
        
        except Exception as e:
            confidence_analysis["analysis_error"] = str(e)
            logger.warning(f"Error analyzing confidence scores: {e}")
        
        return confidence_analysis

    def _assess_completeness(self, 
                           results: Dict[str, Any], 
                           pydantic_schema: Optional[BaseModel]) -> Dict[str, Any]:
        """Assess completeness of extraction results"""
        completeness_assessment = {
            "total_fields": 0,
            "populated_fields": 0,
            "empty_fields": [],
            "completeness_score": 0.0,
            "missing_required_fields": [],
            "schema_based_completeness": None
        }
        
        try:
            # Basic completeness analysis
            total_fields = 0
            populated_fields = 0
            empty_fields = []
            
            for field, value in results.items():
                if not field.startswith("_"):  # Skip metadata fields
                    total_fields += 1
                    if value is None or value == "" or value == [] or value == {}:
                        empty_fields.append(field)
                    else:
                        populated_fields += 1
            
            completeness_assessment["total_fields"] = total_fields
            completeness_assessment["populated_fields"] = populated_fields
            completeness_assessment["empty_fields"] = empty_fields
            
            if total_fields > 0:
                completeness_assessment["completeness_score"] = populated_fields / total_fields
            
            # Schema-based completeness if available
            if pydantic_schema and HAS_PYDANTIC:
                schema_completeness = self._calculate_schema_completeness(results, pydantic_schema)
                completeness_assessment["schema_based_completeness"] = schema_completeness
        
        except Exception as e:
            completeness_assessment["assessment_error"] = str(e)
            logger.warning(f"Error assessing completeness: {e}")
        
        return completeness_assessment

    def _calculate_schema_completeness(self, 
                                     results: Dict[str, Any], 
                                     pydantic_schema: BaseModel) -> Dict[str, Any]:
        """Calculate completeness based on Pydantic schema requirements"""
        try:
            # Get schema fields information
            schema_fields = pydantic_schema.__fields__ if hasattr(pydantic_schema, '__fields__') else {}
            
            required_fields = []
            optional_fields = []
            
            for field_name, field_info in schema_fields.items():
                if field_info.is_required():
                    required_fields.append(field_name)
                else:
                    optional_fields.append(field_name)
            
            # Check presence of required fields
            missing_required = [field for field in required_fields if field not in results or not results[field]]
            present_required = [field for field in required_fields if field in results and results[field]]
            
            # Check presence of optional fields
            present_optional = [field for field in optional_fields if field in results and results[field]]
            
            # Calculate scores
            required_completeness = len(present_required) / len(required_fields) if required_fields else 1.0
            optional_completeness = len(present_optional) / len(optional_fields) if optional_fields else 1.0
            
            return {
                "required_fields": required_fields,
                "optional_fields": optional_fields,
                "missing_required_fields": missing_required,
                "present_required_fields": present_required,
                "present_optional_fields": present_optional,
                "required_completeness_score": required_completeness,
                "optional_completeness_score": optional_completeness,
                "overall_schema_completeness": (required_completeness * 0.8) + (optional_completeness * 0.2)
            }
            
        except Exception as e:
            return {"error": str(e)}

    def _calculate_overall_quality_score(self, enhanced_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall quality score based on all metrics"""
        try:
            scores = {}
            weights = self.enhancement_weights
            
            # Schema validation score
            validation = enhanced_results.get("validation", {})
            if validation.get("validation_performed"):
                scores["schema_validation"] = 1.0 if validation.get("is_valid") else 0.0
            
            # Semantic relevance score
            relevance = enhanced_results.get("quality_metrics", {}).get("semantic_relevance", {})
            if relevance.get("semantic_analysis_available"):
                scores["semantic_relevance"] = relevance.get("query_relevance_score", 0.0)
            
            # Confidence score
            confidence = enhanced_results.get("quality_metrics", {}).get("confidence_analysis", {})
            scores["confidence_score"] = confidence.get("overall_confidence", 0.0)
            
            # Completeness score
            completeness = enhanced_results.get("quality_metrics", {}).get("completeness", {})
            scores["completeness"] = completeness.get("completeness_score", 0.0)
            
            # Calculate weighted overall score
            weighted_sum = 0.0
            total_weight = 0.0
            
            for metric, score in scores.items():
                if metric in weights:
                    weighted_sum += score * weights[metric]
                    total_weight += weights[metric]
            
            overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
            
            return {
                "individual_scores": scores,
                "weights_used": weights,
                "overall_score": overall_score,
                "quality_rating": self._get_quality_rating(overall_score)
            }
            
        except Exception as e:
            return {"error": str(e)}

    def _generate_improvement_suggestions(self, enhanced_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate suggestions for improving extraction quality"""
        suggestions = []
        
        try:
            # Validation-based suggestions
            validation = enhanced_results.get("validation", {})
            if validation.get("validation_performed") and not validation.get("is_valid"):
                for error in validation.get("validation_errors", []):
                    suggestions.append({
                        "type": "schema_validation",
                        "priority": "high",
                        "field": error.get("field"),
                        "suggestion": f"Fix validation error: {error.get('message')}",
                        "category": "data_quality"
                    })
            
            # Confidence-based suggestions
            confidence = enhanced_results.get("quality_metrics", {}).get("confidence_analysis", {})
            for low_conf_field in confidence.get("low_confidence_fields", []):
                suggestions.append({
                    "type": "confidence_improvement",
                    "priority": "medium",
                    "field": low_conf_field.get("field"),
                    "suggestion": f"Low confidence field ({low_conf_field.get('score'):.2f}) - consider manual review or re-extraction",
                    "category": "confidence"
                })
            
            # Completeness-based suggestions
            completeness = enhanced_results.get("quality_metrics", {}).get("completeness", {})
            for empty_field in completeness.get("empty_fields", []):
                suggestions.append({
                    "type": "completeness_improvement",
                    "priority": "medium",
                    "field": empty_field,
                    "suggestion": f"Field '{empty_field}' is empty - consider using additional extraction strategies",
                    "category": "completeness"
                })
            
            # Semantic relevance suggestions
            relevance = enhanced_results.get("quality_metrics", {}).get("semantic_relevance", {})
            if relevance.get("query_relevance_score", 0) < 0.5:
                suggestions.append({
                    "type": "relevance_improvement",
                    "priority": "high",
                    "suggestion": "Low semantic relevance to query - consider refining extraction strategy or URL targeting",
                    "category": "relevance"
                })
        
        except Exception as e:
            logger.warning(f"Error generating suggestions: {e}")
        
        return suggestions

    def _get_feedback_summary(self, query: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Get aggregated feedback summary for similar queries/results"""
        feedback_summary = {
            "historical_feedback_available": False,
            "average_rating": None,
            "feedback_count": 0,
            "common_issues": [],
            "recommendations_based_on_feedback": []
        }
        
        try:
            # Search for historical feedback on similar queries
            historical_feedback = self._search_historical_feedback(query, results)
            
            if historical_feedback:
                feedback_summary["historical_feedback_available"] = True
                feedback_summary["feedback_count"] = len(historical_feedback)
                
                # Calculate average rating
                ratings = [f.rating.value for f in historical_feedback]
                feedback_summary["average_rating"] = sum(ratings) / len(ratings)
                
                # Extract common issues
                issues = []
                for feedback in historical_feedback:
                    if feedback.comments:
                        issues.append(feedback.comments)
                
                feedback_summary["common_issues"] = list(set(issues))
        
        except Exception as e:
            logger.warning(f"Error getting feedback summary: {e}")
        
        return feedback_summary

    def collect_user_feedback(self, 
                            query: str,
                            results: Dict[str, Any],
                            feedback_type: FeedbackType,
                            rating: FeedbackRating,
                            comments: Optional[str] = None,
                            field_specific_feedback: Optional[Dict[str, Dict[str, Any]]] = None,
                            user_id: Optional[str] = None,
                            session_id: Optional[str] = None) -> str:
        """
        Collect user feedback for a specific extraction result.
        
        Args:
            query: Original search query
            results: Extraction results that user is providing feedback on
            feedback_type: Type of feedback being provided
            rating: User's rating of the results
            comments: Optional textual feedback
            field_specific_feedback: Optional field-specific feedback
            user_id: Optional user identifier
            session_id: Optional session identifier
            
        Returns:
            Feedback ID for tracking
        """
        try:
            feedback_id = str(uuid.uuid4())
            
            feedback = UserFeedback(
                feedback_id=feedback_id,
                query=query,
                result_data=results,
                feedback_type=feedback_type,
                rating=rating,
                comments=comments,
                field_specific_feedback=field_specific_feedback,
                timestamp=datetime.now(),
                user_id=user_id,
                session_id=session_id
            )
            
            # Store feedback
            self._store_feedback(feedback)
            
            logger.info(f"User feedback collected: {feedback_id}")
            return feedback_id
            
        except Exception as e:
            logger.error(f"Error collecting user feedback: {e}")
            return ""

    def _store_feedback(self, feedback: UserFeedback):
        """Store feedback in the configured storage system"""
        try:
            feedback_data = feedback.to_dict()
            
            if isinstance(self.feedback_storage, dict):
                # In-memory storage
                self.feedback_storage[feedback.feedback_id] = feedback_data
            elif HAS_REDIS and hasattr(self.feedback_storage, 'hset'):
                # Redis storage
                feedback_key = f"feedback:{feedback.feedback_id}"
                self.feedback_storage.hset(feedback_key, mapping=feedback_data)
                
                # Also store in query-based index for searching
                query_key = f"query_feedback:{feedback.query}"
                self.feedback_storage.sadd(query_key, feedback.feedback_id)
            
            logger.debug(f"Feedback stored: {feedback.feedback_id}")
            
        except Exception as e:
            logger.error(f"Error storing feedback: {e}")

    def _search_historical_feedback(self, 
                                  query: str, 
                                  results: Dict[str, Any]) -> List[UserFeedback]:
        """Search for historical feedback on similar queries"""
        historical_feedback = []
        
        try:
            if isinstance(self.feedback_storage, dict):
                # In-memory search
                for feedback_data in self.feedback_storage.values():
                    if self._is_similar_query(query, feedback_data.get("query", "")):
                        historical_feedback.append(UserFeedback.from_dict(feedback_data))
            
            elif HAS_REDIS and hasattr(self.feedback_storage, 'smembers'):
                # Redis search
                query_key = f"query_feedback:{query}"
                feedback_ids = self.feedback_storage.smembers(query_key)
                
                for feedback_id in feedback_ids:
                    feedback_key = f"feedback:{feedback_id}"
                    feedback_data = self.feedback_storage.hgetall(feedback_key)
                    if feedback_data:
                        historical_feedback.append(UserFeedback.from_dict(feedback_data))
        
        except Exception as e:
            logger.warning(f"Error searching historical feedback: {e}")
        
        return historical_feedback

    def get_feedback_analytics(self, 
                             time_range_days: int = 30) -> Dict[str, Any]:
        """
        Get analytics on collected feedback.
        
        Args:
            time_range_days: Number of days to analyze
            
        Returns:
            Analytics dictionary
        """
        analytics = {
            "total_feedback_count": 0,
            "average_rating": 0.0,
            "rating_distribution": {},
            "feedback_type_distribution": {},
            "common_issues": [],
            "improvement_trends": {}
        }
        
        try:
            all_feedback = self._get_all_feedback(time_range_days)
            
            if not all_feedback:
                return analytics
            
            analytics["total_feedback_count"] = len(all_feedback)
            
            # Rating analytics
            ratings = [f.rating.value for f in all_feedback]
            analytics["average_rating"] = sum(ratings) / len(ratings)
            
            rating_counts = {}
            for rating in ratings:
                rating_counts[rating] = rating_counts.get(rating, 0) + 1
            analytics["rating_distribution"] = rating_counts
            
            # Feedback type distribution
            type_counts = {}
            for feedback in all_feedback:
                type_name = feedback.feedback_type.value
                type_counts[type_name] = type_counts.get(type_name, 0) + 1
            analytics["feedback_type_distribution"] = type_counts
            
            # Common issues from comments
            comments = [f.comments for f in all_feedback if f.comments]
            analytics["common_issues"] = self._extract_common_issues(comments)
        
        except Exception as e:
            logger.error(f"Error generating feedback analytics: {e}")
            analytics["error"] = str(e)
        
        return analytics

    # Helper methods
    
    def _extract_text_from_results(self, results: Dict[str, Any]) -> str:
        """Extract all text content from results for similarity comparison"""
        text_parts = []
        
        def extract_text_recursive(obj):
            if isinstance(obj, str):
                text_parts.append(obj)
            elif isinstance(obj, dict):
                for value in obj.values():
                    extract_text_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    extract_text_recursive(item)
        
        extract_text_recursive(results)
        return " ".join(text_parts)

    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5

    def _get_quality_rating(self, score: float) -> str:
        """Convert numeric quality score to descriptive rating"""
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.8:
            return "Good"
        elif score >= 0.7:
            return "Satisfactory"
        elif score >= 0.6:
            return "Fair"
        else:
            return "Poor"

    def _is_similar_query(self, query1: str, query2: str) -> bool:
        """Check if two queries are similar (simple implementation)"""
        # Simple similarity check - could be enhanced with semantic similarity
        query1_lower = query1.lower()
        query2_lower = query2.lower()
        
        # Exact match
        if query1_lower == query2_lower:
            return True
        
        # Check for significant word overlap
        words1 = set(query1_lower.split())
        words2 = set(query2_lower.split())
        
        if len(words1) == 0 or len(words2) == 0:
            return False
        
        overlap = len(words1.intersection(words2))
        total_unique = len(words1.union(words2))
        
        similarity = overlap / total_unique if total_unique > 0 else 0
        return similarity > 0.6  # 60% similarity threshold

    def _get_all_feedback(self, time_range_days: int) -> List[UserFeedback]:
        """Get all feedback within time range"""
        cutoff_date = datetime.now() - timedelta(days=time_range_days)
        all_feedback = []
        
        try:
            if isinstance(self.feedback_storage, dict):
                for feedback_data in self.feedback_storage.values():
                    feedback = UserFeedback.from_dict(feedback_data)
                    if feedback.timestamp >= cutoff_date:
                        all_feedback.append(feedback)
            
            # Redis implementation would go here if needed
            
        except Exception as e:
            logger.warning(f"Error retrieving feedback: {e}")
        
        return all_feedback

    def _extract_common_issues(self, comments: List[str]) -> List[str]:
        """Extract common issues from user comments"""
        # Simple keyword-based extraction - could be enhanced with NLP
        issue_keywords = {
            "incomplete": "Missing or incomplete data",
            "wrong": "Incorrect information extracted",
            "irrelevant": "Irrelevant results",
            "slow": "Performance issues",
            "format": "Data formatting problems"
        }
        
        common_issues = []
        for keyword, description in issue_keywords.items():
            if any(keyword in comment.lower() for comment in comments):
                common_issues.append(description)
        
        return common_issues

# Import datetime for the helper method
from datetime import timedelta
