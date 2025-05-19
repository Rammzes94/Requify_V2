#!/usr/bin/env python3
"""
fix_broken_loggers.py

This script fixes syntax errors in logger setup statements after the update_loggers.py
script was run. It looks for and corrects patterns like:
- logger = _00_utils.get_logger("Prefix")

And converts them to the correct form:
- logger = _00_utils.get_logger("Prefix")
"""

import os
import sys
import re
import logging

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import _02_src._00_utils as _00_utils
_00_utils.setup_project_directory()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('fix_loggers')

# Project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Regex patterns to find problematic logger setups
BROKEN_LOGGER_PATTERN = re.compile(r'logger\s*=\s*_00_utils\.get_logger\("([^"]+)"\),\s*"([^"]+)"\s*\)')

def find_python_files():
    """Find all Python files in the project directory."""
    python_files = []
    
    for root, _, files in os.walk(PROJECT_ROOT):
        # Skip certain directories
        if '.venv' in root or 'backup' in root:
            continue
            
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                python_files.append(file_path)
    
    return python_files

def fix_file(file_path):
    """Fix logger setup issues in the given file."""
    try:
        # Read the file
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Look for broken logger setup patterns
        match = BROKEN_LOGGER_PATTERN.search(content)
        if not match:
            return False, "No broken logger pattern found"
        
        # Extract the prefix from the second parameter
        prefix = match.group(2).strip().strip('[]').strip()
        
        # Replace the broken pattern with the correct one
        fixed_content = BROKEN_LOGGER_PATTERN.sub(f'logger = _00_utils.get_logger("{prefix}")', content)
        
        # Write the fixed content back to the file
        with open(file_path, 'w') as f:
            f.write(fixed_content)
        
        # Return success
        return True, f"Fixed logger setup in {os.path.relpath(file_path, PROJECT_ROOT)}"
        
    except Exception as e:
        return False, f"Error fixing file {file_path}: {str(e)}"

def main():
    """Main entry point."""
    logger.info("Fixing broken logger setups...")
    
    python_files = find_python_files()
    logger.info(f"Found {len(python_files)} Python files to check")
    
    fixed_count = 0
    
    for file_path in python_files:
        success, message = fix_file(file_path)
        if success:
            fixed_count += 1
            logger.info(f"✅ {message}")
    
    logger.info(f"Fixed {fixed_count} files with broken logger setups")

if __name__ == "__main__":
    main() 