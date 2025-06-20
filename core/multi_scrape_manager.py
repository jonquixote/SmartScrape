import asyncio
import logging
import uuid # Added import for uuid
from typing import List, Dict, Any, Optional

from controllers.adaptive_scraper import AdaptiveScraper # Assuming adaptive_scraper is the default
from utils.result_combiner import combine_and_deduplicate_results # Will create this next

class MultiScrapeManager:
    """
    Manages asynchronous scraping of multiple URLs and combines their results.
    """

    def __init__(self, config=None):
        self.adaptive_scraper = AdaptiveScraper() # Get the singleton instance

    async def scrape_multiple_urls(self, urls: List[str], user_intent: str, options: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """
        Initiates and manages concurrent scraping tasks for a list of URLs.

        Args:
            urls: A list of URLs to scrape.
            user_intent: The user's original prompt or intent for the scraping task.
            options: Optional dictionary of settings to pass to the adaptive scraper.

        Returns:
            A list of dictionaries, where each dictionary contains the results from a single URL scrape.
        """
        if not urls:
            logging.warning("No URLs provided for multi-scrape.")
            return []

        logging.info(f"Starting multi-scrape for {len(urls)} URLs with intent: '{user_intent}'")

        scrape_tasks = []
        for url in urls:
            # Each scrape will be treated as a sub-job or a direct call to the adaptive scraper.
            # For simplicity, we'll call process_user_request directly, but in a more complex
            # system, you might want to create separate job IDs for each sub-scrape.
            task = self._run_single_scrape_task(url, user_intent, options)
            scrape_tasks.append(task)

        # Run all scrape tasks concurrently
        # return_exceptions=True allows other tasks to complete even if one fails
        raw_results = await asyncio.gather(*scrape_tasks, return_exceptions=True)

        successful_results = []
        for i, result in enumerate(raw_results):
            if isinstance(result, Exception):
                logging.error(f"Scrape failed for URL {urls[i]}: {result}")
            else:
                successful_results.append(result)
        
        logging.info(f"Completed multi-scrape. Successfully scraped {len(successful_results)} out of {len(urls)} URLs.")
        return successful_results

    async def _run_single_scrape_task(self, url: str, user_intent: str, options: Optional[Dict[str, Any]]) -> Dict:
        """
        Runs a single scraping task using the adaptive scraper.
        This is a simplified wrapper; in a real scenario, you might pass a sub-job_id
        and track its progress more granularly.
        """
        logging.info(f"Initiating single scrape for URL: {url} with intent: '{user_intent}'")
        
        # The adaptive scraper's process_user_request is designed to handle a single URL or a list.
        # Here, we pass a single URL in a list.
        job_info = await self.adaptive_scraper.process_user_request(
            user_prompt=user_intent,
            start_urls=[url],
            job_id=str(uuid.uuid4()), # Generate a temporary job ID for internal tracking by adaptive scraper
            options=options
        )
        
        # Wait for the adaptive scraper to complete its internal job
        # This is a simplified polling mechanism. A more robust solution would use callbacks
        # or a message queue for status updates.
        while True:
            current_status = self.adaptive_scraper.get_job_status(job_info["job_id"])
            if current_status["status"] in ["completed", "failed", "error"]:
                if current_status["status"] == "completed":
                    logging.info(f"Single scrape completed for URL: {url}")
                    # Extract the actual 'results' list from the job output
                    job_output = self.adaptive_scraper.get_job_results(job_info["job_id"])
                    return job_output.get("results", []) # Return list of data items
                else:
                    raise Exception(f"Single scrape failed for URL {url}: {current_status.get('error', 'Unknown error')}")
            await asyncio.sleep(0.5) # Poll every 0.5 seconds

    async def scrape_and_combine(self, urls: List[str], user_intent: str, options: Optional[Dict[str, Any]] = None) -> Dict:
        """
        Scrapes multiple URLs concurrently and then combines and deduplicates the results.

        Args:
            urls: A list of URLs to scrape.
            user_intent: The user's original prompt or intent.
            options: Optional dictionary of settings for the adaptive scraper.

        Returns:
            A single, combined, and deduplicated dictionary of results.
        """
        raw_results_list = await self.scrape_multiple_urls(urls, user_intent, options)
        # raw_results_list is now a List[List[Dict]] (list of lists of data items)
        
        if not raw_results_list:
            return {"combined_results": [], "deduplicated_count": 0}

        # Flatten the list of lists into a single list of data items
        all_data_items = []
        for sublist_of_items in raw_results_list:
            if isinstance(sublist_of_items, list):
                all_data_items.extend(sublist_of_items)
            else:
                logging.warning(f"Expected a list of items from a scrape task, but got: {type(sublist_of_items)}")

        # The adaptive scraper's get_job_results returns a dict like {"results": [...], "result_count": ...}
        # combine_and_deduplicate_results expects a flat list of data items.
        combined_data = combine_and_deduplicate_results(all_data_items)
        
        logging.info(f"Combined and deduplicated results. Final count: {combined_data['deduplicated_count']}")
        return combined_data

# Singleton instance
_multi_scrape_manager = MultiScrapeManager()

def get_multi_scrape_manager() -> MultiScrapeManager:
    return _multi_scrape_manager
