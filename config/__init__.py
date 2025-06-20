# Config package for SmartScrape
# Import everything from the main config.py module to maintain compatibility

import sys
import os

# Add parent directory to path to import config.py from parent
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import everything from the main config.py file
try:
    # Import the config module from parent directory
    import importlib.util
    config_path = os.path.join(parent_dir, 'config.py')
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    # Export all attributes from config.py
    for attr_name in dir(config_module):
        if not attr_name.startswith('_'):
            globals()[attr_name] = getattr(config_module, attr_name)
            
except Exception as e:
    # Fallback: if there's any issue, we can still access environments
    pass
