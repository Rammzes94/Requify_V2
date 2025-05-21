"""
Directory utilities for the Requify_V2 project.

Provides functions for consistent directory handling and path setup.
"""

import os
import sys
import logging

def setup_project_directory():
    """
    Smart directory handling to ensure consistent working directory.
    
    In interactive mode (Jupyter/IPython), changes to the project root folder 
    (identified by presence of .env, venv, myenv, .gitignore, or requirements.txt).
    In script mode, maintains normal execution behavior.
    
    Returns:
        str: The current working directory after any necessary adjustments
    """
    
    # Check if we're running in interactive mode (like IPython or Jupyter)
    is_interactive = (not hasattr(sys, 'ps1') and sys.argv[0] == '') or 'ipykernel' in sys.modules

    # Handle interactive mode 
    if is_interactive:
        current_dir = os.getcwd()
        
        # List of files/folders that indicate project root
        root_indicators = ['.env', 'venv', 'myenv', '.gitignore', 'requirements.txt']
        
        # Go up directory levels until we find root indicators
        test_dir = current_dir
        while test_dir != os.path.dirname(test_dir):  # Stop at filesystem root
            # Check if any root indicators exist in this directory
            if any(os.path.exists(os.path.join(test_dir, indicator)) for indicator in root_indicators):
                # Found the project root
                if test_dir != current_dir:
                    os.chdir(test_dir)
                    logger = logging.getLogger()
                    if logger.handlers:
                        logger.info(f"Interactive mode: Changed working directory to: {test_dir}")
                    else:
                        print(f"Interactive mode: Changed working directory to: {test_dir}")
                else:
                    logger = logging.getLogger()
                    if logger.handlers:
                        logger.info(f"Interactive mode: Already in project root: {test_dir}")
                    else:
                        print(f"Interactive mode: Already in project root: {test_dir}")
                break
            
            # Move up one directory level
            test_dir = os.path.dirname(test_dir)
            
            # If we've reached the filesystem root without finding indicators
            if test_dir == os.path.dirname(test_dir):
                logger = logging.getLogger()
                if logger.handlers:
                    logger.warning(f"Interactive mode: Could not find project root. Staying in: {current_dir}")
                else:
                    print(f"Interactive mode: Could not find project root. Staying in: {current_dir}")
                break
    
    return os.getcwd()