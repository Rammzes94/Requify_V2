#!/usr/bin/env python3
"""
update_loggers.py

This script updates all Python files in the codebase that have a custom ScriptLogger
class definition to use the centralized version from _00_utils instead.
"""

import os
import sys
import re
import logging
from typing import List, Tuple

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
logger = logging.getLogger('update_loggers')

# Project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Files to skip (like backup files or the utils file itself)
SKIP_PATHS = [
    'backup',
    '.venv',
    '_00_utils.py',
    'update_loggers.py'
]

# Regex patterns
SCRIPT_LOGGER_CLASS_PATTERN = re.compile(r'class\s+ScriptLogger.*?def\s+process.*?return.*?kwargs', re.DOTALL)
LOGGER_SETUP_PATTERN = re.compile(r'logger\s*=\s*ScriptLogger\s*\(.*?\)', re.MULTILINE)

def find_python_files() -> List[str]:
    """Find all Python files in the project directory."""
    python_files = []
    
    for root, _, files in os.walk(PROJECT_ROOT):
        # Skip directories that should be ignored
        if any(skip in root for skip in SKIP_PATHS):
            continue
            
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                python_files.append(file_path)
    
    return python_files

def should_update_file(file_path: str, content: str) -> bool:
    """Check if the file contains a ScriptLogger class definition."""
    # Skip if path contains any of the skip paths
    if any(skip in file_path for skip in SKIP_PATHS):
        return False
        
    # Check if file has ScriptLogger class definition
    return bool(SCRIPT_LOGGER_CLASS_PATTERN.search(content))

def update_file(file_path: str) -> Tuple[bool, str]:
    """
    Update the file to use the centralized ScriptLogger.
    
    Returns:
        Tuple of (success, message)
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        if not should_update_file(file_path, content):
            return False, "No ScriptLogger class found"
            
        # Get module name from file path
        rel_path = os.path.relpath(file_path, PROJECT_ROOT)
        module_name = os.path.splitext(os.path.basename(file_path))[0].replace('_', ' ').title().replace(' ', '_')
        
        # Replace ScriptLogger class definition
        new_content = SCRIPT_LOGGER_CLASS_PATTERN.sub('', content)
        
        # Try to detect how the logger is initialized
        logger_setup_match = LOGGER_SETUP_PATTERN.search(new_content)
        if logger_setup_match:
            old_logger_setup = logger_setup_match.group(0)
            # Extract existing prefix if possible
            prefix_match = re.search(r'\[\s*["\']([^"\']+)["\']', old_logger_setup)
            module_prefix = prefix_match.group(1) if prefix_match else module_name
            
            # Replace logger setup
            new_logger_setup = f'logger = _00_utils.get_logger("{module_prefix}")'
            new_content = new_content.replace(old_logger_setup, new_logger_setup)
        
        # Write updated content back to file
        with open(file_path, 'w') as f:
            f.write(new_content)
            
        return True, f"Updated logger in {rel_path}"
    except Exception as e:
        return False, f"Error updating {file_path}: {str(e)}"

def main():
    """Main entry point."""
    logger.info("Updating ScriptLogger usage across the codebase...")
    
    python_files = find_python_files()
    logger.info(f"Found {len(python_files)} Python files to check")
    
    updated_count = 0
    
    for file_path in python_files:
        success, message = update_file(file_path)
        if success:
            updated_count += 1
            logger.info(f"✅ {message}")
    
    logger.info(f"Updated {updated_count} files to use the centralized ScriptLogger")

if __name__ == "__main__":
    main() 