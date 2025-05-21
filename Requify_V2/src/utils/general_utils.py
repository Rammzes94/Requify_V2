"""
General utilities for the Requify_V2 project.

Provides miscellaneous utility functions like timestamp generation.
"""

from datetime import datetime

def generate_timestamp():
    """
    Generate a timestamp string in the format YYYY-MM-DD HH:MM:SS.
    
    Returns:
        str: Formatted timestamp
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")