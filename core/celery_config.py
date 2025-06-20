"""
Celery Configuration for SmartScrape

This module configures Celery for asynchronous task processing with Redis as
the message broker and result backend.
"""

from celery import Celery
from config import REDIS_CONFIG
import logging

logger = logging.getLogger(__name__)

# Create Celery app instance
celery_app = Celery(
    'smartscrape',
    broker=f"redis://{REDIS_CONFIG['host']}:{REDIS_CONFIG['port']}/{REDIS_CONFIG['db']}",
    backend=f"redis://{REDIS_CONFIG['host']}:{REDIS_CONFIG['port']}/{REDIS_CONFIG['db']}",
    include=['core.tasks']
)

# Configure Celery settings
celery_app.conf.update(
    # Serialization
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    
    # Timezone
    timezone='UTC',
    enable_utc=True,
    
    # Task execution
    task_track_started=True,
    task_time_limit=300,  # 5 minutes
    task_soft_time_limit=240,  # 4 minutes soft limit
    
    # Worker configuration
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    worker_disable_rate_limits=False,
    
    # Results
    result_expires=3600,  # 1 hour
    result_backend_transport_options={'retry_on_timeout': True},
    
    # Broker settings
    broker_transport_options={
        'retry_on_timeout': True,
        'max_connections': 20,
    },
    
    # Task routing
    task_routes={
        'core.tasks.extract_url_task': {'queue': 'extraction'},
        'core.tasks.batch_extract_task': {'queue': 'batch'},
        'core.tasks.cache_warm_task': {'queue': 'maintenance'},
    },
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
)

# Configure logging for Celery
celery_app.conf.update(
    worker_log_level='INFO',
    worker_log_format='[%(asctime)s: %(levelname)s/%(processName)s] %(message)s',
    worker_task_log_format='[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s',
)

logger.info("Celery app configured successfully")
