"""
Adaptive Strategy Selector for SmartScrape

This component intelligently selects the optimal extraction strategy based on
domain analysis, content type, JavaScript requirements, and historical performance.
"""

import logging
import json
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse
from datetime import datetime, timedelta
import statistics

logger = logging.getLogger(__name__)

class AdaptiveStrategySelector:
    """Intelligent strategy selection based on domain analysis and performance"""
    
    def __init__(self):
        self.strategy_performance = {}
        self.domain_cache = {}
        self.performance_history = []
        
        # Strategy characteristics
        self.strategy_profiles = {
            'universal_crawl4ai': {
                'js_support': 'medium',
                'speed': 'fast',
                'reliability': 'high',
                'best_for': ['general', 'news', 'blogs']
            },
            'trafilatura': {
                'js_support': 'none',
                'speed': 'very_fast',
                'reliability': 'high',
                'best_for': ['news', 'articles', 'blogs', 'static_content']
            },
            'playwright': {
                'js_support': 'excellent',
                'speed': 'slow',
                'reliability': 'medium',
                'best_for': ['spa', 'dynamic_content', 'js_heavy']
            }
        }
    
    async def select_optimal_strategy(self, url: str, domain_info: Dict = None, content_hints: Dict = None) -> str:
        """
        Select the best extraction strategy based on domain analysis
        
        Args:
            url: The URL to be processed
            domain_info: Domain intelligence analysis results
            content_hints: Additional content type hints
            
        Returns:
            Strategy name to use
        """
        try:
            domain = self._extract_domain(url)
            
            # Check if we have cached domain analysis
            if domain in self.domain_cache:
                cached_info = self.domain_cache[domain]
                # Use cached info if it's less than 24 hours old
                if (datetime.now() - cached_info['timestamp']).total_seconds() < 86400:  # 24 hours
                    domain_info = cached_info.get('analysis', domain_info)
            
            # Strategy selection logic
            selected_strategy = await self._analyze_and_select(url, domain_info, content_hints)
            
            logger.info(f"Selected strategy '{selected_strategy}' for {url}")
            return selected_strategy
            
        except Exception as e:
            logger.error(f"Strategy selection failed for {url}: {e}")
            return 'universal_crawl4ai'  # Safe fallback
    
    async def _analyze_and_select(self, url: str, domain_info: Dict, content_hints: Dict) -> str:
        """Core strategy selection logic"""
        
        # Priority 1: JavaScript dependency check
        if domain_info and domain_info.get('requires_js', False):
            js_confidence = domain_info.get('confidence', 0)
            js_heavy = domain_info.get('js_heavy', False)
            
            if js_heavy or js_confidence > 0.7:
                logger.info(f"High JS dependency detected, selecting Playwright")
                return 'playwright'
            elif js_confidence > 0.4:
                logger.info(f"Medium JS dependency detected, selecting crawl4ai")
                return 'universal_crawl4ai'
        
        # Priority 2: Content type optimization
        content_type = self._determine_content_type(domain_info, content_hints, url)
        
        if content_type in ['news', 'article', 'blog']:
            # Trafilatura excels at news/article content
            logger.info(f"News/article content detected, selecting Trafilatura")
            return 'trafilatura'
        elif content_type in ['ecommerce', 'listings', 'directory']:
            # These often have some JS but crawl4ai handles them well
            return 'universal_crawl4ai'
        elif content_type in ['spa', 'app', 'dashboard']:
            # Single-page apps definitely need browser rendering
            return 'playwright'
        
        # Priority 3: Historical performance
        domain = self._extract_domain(url)
        if domain in self.strategy_performance:
            best_strategy = self._get_best_performing_strategy(domain)
            if best_strategy:
                logger.info(f"Using historically best strategy '{best_strategy}' for domain {domain}")
                return best_strategy
        
        # Priority 4: URL pattern analysis
        url_strategy = self._analyze_url_patterns(url)
        if url_strategy:
            return url_strategy
        
        # Default fallback
        logger.info(f"Using default strategy for {url}")
        return 'universal_crawl4ai'
    
    def _determine_content_type(self, domain_info: Dict, content_hints: Dict, url: str) -> str:
        """Determine content type from various signals"""
        
        # Check explicit content hints first
        if content_hints and content_hints.get('content_type'):
            return content_hints['content_type']
        
        # Check domain info
        if domain_info:
            detected_type = domain_info.get('content_type', '')
            if detected_type:
                return detected_type
                
            # Check for SPA indicators
            frameworks = domain_info.get('frameworks', [])
            if any(f.get('name') in ['react', 'vue', 'angular'] for f in frameworks):
                return 'spa'
        
        # URL pattern analysis
        url_lower = url.lower()
        
        # News/blog patterns
        if any(pattern in url_lower for pattern in [
            '/news/', '/article/', '/blog/', '/post/', '/story/',
            'news.', 'blog.', 'article', 'medium.com', 'substack.com'
        ]):
            return 'news'
        
        # E-commerce patterns
        if any(pattern in url_lower for pattern in [
            '/product/', '/shop/', '/store/', '/buy/', '/cart/',
            'shop.', 'store.', 'amazon.', 'ebay.', 'etsy.'
        ]):
            return 'ecommerce'
        
        # App/dashboard patterns
        if any(pattern in url_lower for pattern in [
            '/app/', '/dashboard/', '/admin/', '/panel/',
            'app.', 'dashboard.', 'admin.'
        ]):
            return 'app'
        
        return 'general'
    
    def _analyze_url_patterns(self, url: str) -> Optional[str]:
        """Analyze URL patterns for strategy hints"""
        
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()
        
        # Known domains that work better with specific strategies
        domain_preferences = {
            # News sites that work well with Trafilatura
            'news': ['cnn.com', 'bbc.com', 'reuters.com', 'ap.org', 'nytimes.com'],
            # Sites that typically need browser rendering
            'spa': ['app.', 'dashboard.', 'admin.', 'panel.'],
            # API endpoints
            'api': ['/api/', '/v1/', '/v2/', 'api.']
        }
        
        for strategy_hint, patterns in domain_preferences.items():
            for pattern in patterns:
                if pattern in domain or pattern in path:
                    if strategy_hint == 'news':
                        return 'trafilatura'
                    elif strategy_hint == 'spa':
                        return 'playwright'
                    elif strategy_hint == 'api':
                        return 'universal_crawl4ai'
        
        return None
    
    def _get_best_performing_strategy(self, domain: str) -> Optional[str]:
        """Get the best performing strategy for a domain"""
        
        if domain not in self.strategy_performance:
            return None
        
        domain_stats = self.strategy_performance[domain]
        
        # Find strategy with best success rate and reasonable attempt count
        best_strategy = None
        best_score = 0
        
        for strategy, stats in domain_stats.items():
            if stats['total_attempts'] < 3:  # Need at least 3 attempts for reliability
                continue
                
            # Calculate composite score (success rate weighted by response time)
            success_rate = stats['success_rate']
            avg_time = stats['avg_response_time']
            
            # Penalize very slow strategies
            time_penalty = 1.0
            if avg_time > 10:  # Over 10 seconds
                time_penalty = 0.8
            elif avg_time > 5:  # Over 5 seconds
                time_penalty = 0.9
            
            score = success_rate * time_penalty
            
            if score > best_score:
                best_score = score
                best_strategy = strategy
        
        return best_strategy
    
    async def update_performance(self, url: str, strategy: str, success: bool, 
                               response_time: float, content_quality: float = None):
        """Update strategy performance metrics"""
        
        domain = self._extract_domain(url)
        
        # Initialize domain tracking if needed
        if domain not in self.strategy_performance:
            self.strategy_performance[domain] = {}
        
        if strategy not in self.strategy_performance[domain]:
            self.strategy_performance[domain][strategy] = {
                'success_count': 0,
                'total_attempts': 0,
                'avg_response_time': 0.0,
                'response_times': [],
                'quality_scores': []
            }
        
        stats = self.strategy_performance[domain][strategy]
        stats['total_attempts'] += 1
        
        if success:
            stats['success_count'] += 1
        
        # Update response time tracking
        stats['response_times'].append(response_time)
        # Keep only last 10 response times
        if len(stats['response_times']) > 10:
            stats['response_times'] = stats['response_times'][-10:]
        
        stats['avg_response_time'] = statistics.mean(stats['response_times'])
        stats['success_rate'] = stats['success_count'] / stats['total_attempts']
        
        # Track content quality if provided
        if content_quality is not None:
            stats['quality_scores'].append(content_quality)
            if len(stats['quality_scores']) > 10:
                stats['quality_scores'] = stats['quality_scores'][-10:]
            stats['avg_quality'] = statistics.mean(stats['quality_scores'])
        
        # Add to global performance history
        self.performance_history.append({
            'timestamp': datetime.now(),
            'domain': domain,
            'strategy': strategy,
            'success': success,
            'response_time': response_time,
            'quality': content_quality
        })
        
        # Keep only last 1000 entries
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
        
        logger.debug(f"Updated performance for {domain}/{strategy}: "
                    f"success_rate={stats['success_rate']:.2f}, "
                    f"avg_time={stats['avg_response_time']:.2f}s")
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            return urlparse(url).netloc.lower()
        except:
            return 'unknown'
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary statistics"""
        
        summary = {
            'total_domains': len(self.strategy_performance),
            'strategies': {},
            'top_domains': {},
            'recent_performance': []
        }
        
        # Strategy-level statistics
        for domain, strategies in self.strategy_performance.items():
            for strategy, stats in strategies.items():
                if strategy not in summary['strategies']:
                    summary['strategies'][strategy] = {
                        'total_attempts': 0,
                        'total_successes': 0,
                        'domains_used': 0,
                        'avg_response_time': 0.0
                    }
                
                strat_summary = summary['strategies'][strategy]
                strat_summary['total_attempts'] += stats['total_attempts']
                strat_summary['total_successes'] += stats['success_count']
                strat_summary['domains_used'] += 1
                
                # Calculate weighted average response time
                total_time = strat_summary['avg_response_time'] * (strat_summary['domains_used'] - 1)
                total_time += stats['avg_response_time']
                strat_summary['avg_response_time'] = total_time / strat_summary['domains_used']
        
        # Calculate success rates
        for strategy in summary['strategies']:
            stats = summary['strategies'][strategy]
            stats['success_rate'] = stats['total_successes'] / max(stats['total_attempts'], 1)
        
        # Recent performance (last 24 hours)
        cutoff = datetime.now() - timedelta(hours=24)
        recent = [entry for entry in self.performance_history if entry['timestamp'] > cutoff]
        summary['recent_performance'] = {
            'total_requests': len(recent),
            'successful_requests': sum(1 for entry in recent if entry['success']),
            'avg_response_time': statistics.mean([entry['response_time'] for entry in recent]) if recent else 0
        }
        
        return summary
    
    def cache_domain_analysis(self, domain: str, analysis: Dict):
        """Cache domain analysis results"""
        self.domain_cache[domain] = {
            'analysis': analysis,
            'timestamp': datetime.now()
        }
        
        # Clean old cache entries (keep only last 100 domains)
        if len(self.domain_cache) > 100:
            # Remove oldest entries
            sorted_domains = sorted(
                self.domain_cache.items(),
                key=lambda x: x[1]['timestamp']
            )
            # Keep newest 100
            self.domain_cache = dict(sorted_domains[-100:])
    
    def export_performance_data(self) -> str:
        """Export performance data as JSON string"""
        export_data = {
            'strategy_performance': self.strategy_performance,
            'performance_history': [
                {
                    **entry,
                    'timestamp': entry['timestamp'].isoformat()
                }
                for entry in self.performance_history
            ],
            'export_timestamp': datetime.now().isoformat()
        }
        return json.dumps(export_data, indent=2)
