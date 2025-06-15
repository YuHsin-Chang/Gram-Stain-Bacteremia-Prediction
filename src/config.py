# ================================
# Path settings
# ================================
import os

# Automatically detect project root based on config.py location
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Path to cache, output and data files
cache_dir = os.path.join(project_root, 'cache')
output_dir = os.path.join(project_root,  'outputs')
data_dir = os.path.join(project_root, 'data')

for path in [output_dir, data_dir, cache_dir]:
    os.makedirs(path, exist_ok=True)