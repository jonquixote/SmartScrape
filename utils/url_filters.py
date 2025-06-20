from urllib.parse import urlparse
from crawl4ai.deep_crawling.filters import (
    FilterChain, 
    URLPatternFilter, 
    DomainFilter,
    ContentTypeFilter
)

def create_url_filter_chain(
    url: str, 
    include_external: bool = False,
    include_patterns: str = None,
    exclude_patterns: str = None,
    exclude_media: bool = True
):
    """
    Create a filter chain for crawling based on URL patterns and domain settings
    
    Args:
        url: Base URL for crawling
        include_external: Whether to include links to external domains
        include_patterns: Comma-separated glob patterns to include (e.g. */blog/*,*/product/*)
        exclude_patterns: Comma-separated glob patterns to exclude
        exclude_media: Whether to exclude media files
        
    Returns:
        A FilterChain object for use with crawl4ai
    """
    # Start with an empty list of filters
    filters = []
    
    # Add content type filter to focus on HTML content
    if exclude_media:
        filters.append(
            ContentTypeFilter(allowed_types=["text/html", "application/xhtml+xml"])
        )
    
    # Add URL pattern filters based on include/exclude patterns
    if include_patterns:
        patterns = [p.strip() for p in include_patterns.split(',') if p.strip()]
        if patterns:
            filters.append(
                URLPatternFilter(patterns=patterns, use_glob=True)
            )
            
    if exclude_patterns:
        exclude_list = [p.strip() for p in exclude_patterns.split(',') if p.strip()]
        if exclude_list:
            filters.append(
                URLPatternFilter(patterns=exclude_list, use_glob=True, exclude=True)
            )
            
    # Configure domain filtering based on settings
    if not include_external:
        domain = urlparse(url).netloc
        # Get base domain for flexible matching (e.g., blog.example.com -> example.com)
        parts = domain.split('.')
        if len(parts) >= 2:
            base_domain = '.'.join(parts[-2:])
            filters.append(
                DomainFilter(
                    allow_same_domain=True, 
                    allow_subdomains=True, 
                    allow_other_domains=False
                )
            )
    
    # Create and return the filter chain
    return FilterChain(filters)

def is_media_url(url: str) -> bool:
    """
    Check if a URL points to media content (images, videos, etc.)
    
    Args:
        url: URL to check
        
    Returns:
        True if the URL likely points to media content
    """
    media_extensions = [
        # Images
        '.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp', '.ico', '.bmp',
        # Videos
        '.mp4', '.webm', '.ogg', '.mov', '.avi', '.wmv', '.flv',
        # Audio
        '.mp3', '.wav', '.ogg', '.m4a', '.aac',
        # Documents
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        # Archives
        '.zip', '.rar', '.gz', '.tar', '.7z',
        # Other media
        '.swf'
    ]
    
    url_lower = url.lower()
    return any(url_lower.endswith(ext) for ext in media_extensions)

def normalize_url(url: str) -> str:
    """
    Normalize a URL by ensuring it has a proper scheme and handling edge cases
    
    Args:
        url: URL to normalize
        
    Returns:
        Normalized URL with scheme
    """
    url = url.strip()
    if not url.startswith('http://') and not url.startswith('https://'):
        url = 'https://' + url
    return url

def is_same_domain(url1: str, url2: str, allow_subdomains: bool = True) -> bool:
    """
    Check if two URLs belong to the same domain.
    
    Args:
        url1: First URL to compare
        url2: Second URL to compare
        allow_subdomains: Whether to consider subdomains of the same domain as matching
        
    Returns:
        True if both URLs belong to the same domain
    """
    domain1 = urlparse(url1).netloc.lower()
    domain2 = urlparse(url2).netloc.lower()
    
    if domain1 == domain2:
        return True
        
    if allow_subdomains:
        # Extract base domains for comparison
        parts1 = domain1.split('.')
        parts2 = domain2.split('.')
        
        # Get the last two parts (e.g., example.com)
        if len(parts1) >= 2 and len(parts2) >= 2:
            base_domain1 = '.'.join(parts1[-2:])
            base_domain2 = '.'.join(parts2[-2:])
            
            return base_domain1 == base_domain2
            
    return False

def get_domain(url: str, include_subdomain: bool = True) -> str:
    """
    Extract the domain from a URL.
    
    Args:
        url: URL to extract domain from
        include_subdomain: Whether to include the subdomain
        
    Returns:
        Domain string
    """
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    
    if not include_subdomain and domain.count('.') > 1:
        # Extract just the main domain (e.g., example.com from blog.example.com)
        parts = domain.split('.')
        if len(parts) >= 2:
            domain = '.'.join(parts[-2:])
            
    return domain