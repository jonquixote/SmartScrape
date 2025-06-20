from typing import Optional
# Support both pydantic v1 and v2
try:
    # Pydantic v2
    from pydantic import BaseModel, HttpUrl, field_validator
except ImportError:
    # Pydantic v1 fallback
    from pydantic import BaseModel, HttpUrl, validator as field_validator

from enum import Enum

# Feedback-related models
class FeedbackType(str, Enum):
    """Types of user feedback"""
    RELEVANCE = "relevance"
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    QUALITY = "quality"
    GENERAL = "general"

class FeedbackRating(str, Enum):
    """User feedback ratings"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

class UserFeedbackRequest(BaseModel):
    """Request model for user feedback submission"""
    result_id: str
    feedback_type: FeedbackType
    rating: FeedbackRating
    comment: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class FeedbackAnalyticsRequest(BaseModel):
    """Request model for feedback analytics"""
    result_id: Optional[str] = None
    feedback_type: Optional[FeedbackType] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    limit: int = 100

class ScrapeRequest(BaseModel):
    url: str
    strategy_type: str = "best_first"  # ai_guided, bfs, dfs, or best_first
    max_depth: int = 2
    include_external: bool = False
    extract_description: str
    crawl_entire_site: bool = False
    keywords: Optional[str] = None
    
    # AI-guided strategy settings
    ai_exploration_ratio: float = 0.5  # Balance between exploration (1.0) and exploitation (0.0)
    ai_use_sitemap: bool = True
    ai_use_search: bool = True
    ai_consolidate_results: bool = True
    ai_model_quality: str = "fast"  # fast, premium, or basic
    
    # Hard limits & optimization settings
    max_pages: int = 100
    timeout_seconds: int = 300
    max_concurrent_requests: int = 5
    min_delay: float = 0.5
    max_delay: float = 2.0
    
    # URL filtering settings
    include_patterns: Optional[str] = None
    exclude_patterns: Optional[str] = None
    exclude_media: bool = True
    
    # Cache settings
    cache_mode: str = "memory"  # none, memory, disk
    
    # Content extraction options
    extract_headers: bool = True
    extract_lists: bool = True
    extract_tables: bool = True
    extract_links: bool = False
    extract_images: bool = False
    
    # Proxy settings
    use_proxy: bool = False
    proxy_url: Optional[str] = None
    
    # Browser settings
    use_browser: bool = True
    disable_images: bool = True
    disable_css: bool = True
    disable_javascript: bool = False