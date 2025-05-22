# Utils Module Structure

This directory contains utility functions that are used throughout the Requify project. The utilities are organized by functionality to make it easier to find and maintain the code.

## Directory Structure

```
utils/
├── __init__.py          # Exports all utility functions for easy importing
├── logging_utils.py     # Logging setup and formatting
├── token_tracking.py    # Token usage tracking and reporting
├── directory_utils.py   # Directory handling and path utilities
└── general_utils.py     # Miscellaneous utility functions
```

## Usage

Import utilities from the `src.utils` module:

```python
from src.utils import setup_logging, get_logger, update_token_counters
```

Alternatively, import specific utility modules:

```python
from src.utils.logging_utils import setup_logging
from src.utils.token_tracking import update_token_counters
```

## Module Descriptions

### logging_utils.py

Provides functions for setting up logging with custom emoji icon formatting. The module automatically determines appropriate icons based on log message content.

Key functions:
- `setup_logging()`: Configures logging for the project
- `get_logger(script_name)`: Creates a script-specific logger with consistent prefix

### token_tracking.py

Tracks token usage, calculates costs, and generates reports for LLM API calls.

Key functions:
- `update_token_counters(response, model_id)`: Updates token usage counters
- `get_token_usage()`: Returns current token usage statistics
- `print_token_usage(model_id)`: Prints a summary of token usage and costs
- `generate_token_usage_report()`: Generates an HTML report of token usage

### directory_utils.py

Handles directory paths and project structure.

Key functions:
- `setup_project_directory()`: Ensures consistent working directory for both scripts and interactive sessions

### general_utils.py

Contains miscellaneous utility functions that don't fit in other categories.

Key functions:
- `generate_timestamp()`: Creates a formatted timestamp string

## Migrating from Legacy Utilities

The `_00_utils.py` file in the src directory is now deprecated and will be removed in a future version. It currently re-exports all functions from the new utils module structure for backward compatibility.

If you're still using `_00_utils.py`, update your imports to use the new structure:

```python
# Old way (deprecated)
from _00_utils import get_logger

# New way
from src.utils import get_logger
``` 