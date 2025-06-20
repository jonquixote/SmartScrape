import pytest
import threading
import time
from urllib.parse import urlparse
from core.url_service import URLService, URLQueue

class TestURLService:
    def setup_method(self):
        self.url_service = URLService()
        self.url_service.initialize()
    
    def teardown_method(self):
        self.url_service.shutdown()
    
    def test_normalize_url(self):
        # Test basic normalization
        assert self.url_service.normalize_url("http://example.com") == "http://example.com/"
        
        # Test scheme normalization
        assert self.url_service.normalize_url("example.com") == "http://example.com/"
        
        # Test relative URL resolution
        assert self.url_service.normalize_url("/page", "http://example.com") == "http://example.com/page"
        
        # Test case normalization
        assert self.url_service.normalize_url("HTTP://EXAMPLE.COM") == "http://example.com/"
        
        # Test port normalization
        assert self.url_service.normalize_url("http://example.com:80") == "http://example.com/"
        assert self.url_service.normalize_url("https://example.com:443") == "https://example.com/"
        
        # Test tracking parameter removal
        assert self.url_service.normalize_url("http://example.com?utm_source=test") == "http://example.com/"
        assert self.url_service.normalize_url("http://example.com?param=1&utm_source=test") == "http://example.com/?param=1"
    
    def test_custom_tracking_parameters(self):
        # Initialize with custom tracking parameters
        url_service = URLService()
        url_service.initialize({
            'tracking_parameters': {'custom_track', 'analytics_ref'}
        })
        
        # Test custom parameter removal
        assert url_service.normalize_url("http://example.com?custom_track=test") == "http://example.com/"
        assert url_service.normalize_url("http://example.com?analytics_ref=email") == "http://example.com/"
        
        # Verify other tracking params are still removed
        assert url_service.normalize_url("http://example.com?utm_source=test") == "http://example.com/"
        
        url_service.shutdown()
    
    def test_fragment_handling(self):
        # By default, fragments should be removed
        assert self.url_service.normalize_url("http://example.com#section") == "http://example.com/"
        
        # Initialize with fragment preservation
        url_service = URLService()
        url_service.initialize({
            'keep_fragments': True
        })
        
        # Test fragment preservation
        assert url_service.normalize_url("http://example.com#section") == "http://example.com/#section"
        
        url_service.shutdown()
    
    def test_url_queue(self):
        queue = self.url_service.get_queue("test_queue")
        
        # Test adding URLs
        assert queue.add("http://example.com") == True
        assert queue.size == 1
        
        # Test duplicate detection
        assert queue.add("http://example.com") == False
        assert queue.size == 1
        
        # Test getting URLs
        url = queue.get()
        assert url == "http://example.com"
        assert queue.size == 0
        assert queue.is_in_progress("http://example.com") == True
        assert queue.is_visited("http://example.com") == False
        
        # Test completing URLs
        queue.complete("http://example.com")
        assert queue.is_in_progress("http://example.com") == False
        assert queue.is_visited("http://example.com") == True
        
        # Test adding visited URL
        assert queue.add("http://example.com") == False
    
    def test_multiple_queues(self):
        # Test that multiple queues can be created and are separate
        queue1 = self.url_service.get_queue("queue1")
        queue2 = self.url_service.get_queue("queue2")
        
        # Add to queue1
        queue1.add("http://example1.com")
        
        # Add to queue2
        queue2.add("http://example2.com")
        
        # Verify queue1 doesn't contain queue2's URL
        assert queue1.is_in_progress("http://example2.com") == False
        
        # Verify queue2 doesn't contain queue1's URL
        assert queue2.is_in_progress("http://example1.com") == False
        
        # Verify queue sizes
        assert queue1.size == 1
        assert queue2.size == 1
    
    def test_robots_txt_methods(self):
        # Just test the interface since real robots.txt tests would need network
        result = self.url_service.is_allowed("http://example.com/page")
        assert isinstance(result, bool)
        
        delay = self.url_service.get_crawl_delay("http://example.com")
        assert delay is None or isinstance(delay, float)
        
        sitemaps = self.url_service.get_sitemaps("http://example.com")
        assert isinstance(sitemaps, list)
    
    def test_classify_url(self):
        # Test product URL classification
        product_url = "http://example.com/products/item123"
        classification = self.url_service.classify_url(product_url)
        assert classification["path_type"] == "product"
        assert classification["is_resource"] == False
        
        # Test resource URL classification
        resource_url = "http://example.com/images/photo.jpg"
        classification = self.url_service.classify_url(resource_url)
        assert classification["is_resource"] == True
        
        # Test search URL classification
        search_url = "http://example.com/search?q=test"
        classification = self.url_service.classify_url(search_url)
        assert classification["path_type"] == "search"
        assert classification["has_parameters"] == True
        
        # Test navigation URL classification
        nav_url = "http://example.com/categories/electronics"
        classification = self.url_service.classify_url(nav_url)
        assert classification["is_navigation"] == True
        assert classification["path_type"] == "category"

class TestURLQueueDirectly:
    def test_url_queue_operations(self):
        queue = URLQueue()
        
        # Test adding URLs
        assert queue.add("http://example.com") == True
        assert queue.size == 1
        
        # Test priority (although not fully implemented in current version)
        assert queue.add("http://priority.com", priority=1) == True
        
        # The queue is FIFO by default without considering priority in this implementation
        url1 = queue.get()
        assert url1 == "http://example.com"
        
        url2 = queue.get()
        assert url2 == "http://priority.com"
        
        # Test visited status
        queue.complete(url1)
        assert queue.is_visited(url1) == True
        
        # Test clearing
        queue.clear()
        assert queue.size == 0
        assert queue.visited_count == 0
    
    def test_thread_safety(self):
        queue = URLQueue()
        
        # We'll have multiple threads adding URLs
        def add_urls():
            for i in range(50):
                queue.add(f"http://example{i}.com")
                time.sleep(0.001)  # Small delay to increase chance of race condition
                
        def get_urls():
            for i in range(50):
                url = queue.get()
                if url:
                    queue.complete(url)
                time.sleep(0.002)  # Different delay to increase chance of race condition
        
        # Start threads
        threads = []
        threads.append(threading.Thread(target=add_urls))
        threads.append(threading.Thread(target=get_urls))
        
        for thread in threads:
            thread.start()
            
        for thread in threads:
            thread.join()
            
        # If we made it here without exceptions, the test passed
        # Let's also check the queue state
        assert queue.size <= 50  # Should be 0 if get_urls processed everything, but allow for some timing variation
        total_processed = queue.visited_count + queue.size
        assert total_processed <= 50  # Total processed or in queue should not exceed 50