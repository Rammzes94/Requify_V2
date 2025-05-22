"""
py - DEPRECATED

This module is deprecated and will be removed in a future version.
Please use the new utils module structure instead:

from src.utils import [function_name]

This file now simply re-exports all utilities from the new src.utils module structure.
"""

import warnings
import sys
import os

# Display deprecation warning
warnings.warn(
    "py is deprecated and will be removed in a future version. "
    "Please use the new utils module structure: from src.utils import [function_name]",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from the new utils modules
from src.utils import (
    # Logging utils
    setup_logging,
    get_logger,
    
    # Token tracking utils
    update_token_counters, 
    get_token_usage, 
    print_token_usage, 
    reset_token_counters,
    display_token_usage_status,
    generate_token_usage_report,
    check_token_limits,
    load_token_tracking_data,
    save_token_usage,
    
    # Directory utils
    setup_project_directory,
    
    # General utils
    generate_timestamp
)

# Export all the re-exported symbols
__all__ = [
    'setup_logging',
    'get_logger',
    'update_token_counters',
    'get_token_usage',
    'print_token_usage',
    'reset_token_counters',
    'setup_project_directory',
    'generate_timestamp',
    'display_token_usage_status',
    'generate_token_usage_report',
    'check_token_limits',
    'load_token_tracking_data',
    'save_token_usage'
]

