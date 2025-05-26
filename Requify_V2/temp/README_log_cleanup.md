# Log Cleanup Tools

This directory contains tools to clean up redundant logging in the Requify codebase.

## Overview

The logging system in the Requify codebase includes detailed logs with icons and structured information, but some patterns lead to redundant or duplicate messages. These tools help identify and fix these issues.

## Files

- **log_cleanup.py** - Main script that identifies and fixes redundant logging
- **test_log_cleanup.py** - Test script to verify regex patterns
- **log_cleanup_summary.md** - Summary of issues and fixes

## How to Use

### 1. Run the cleanup script

```bash
# From the project root
python temp/log_cleanup.py
```

This script will:
1. Find the modules that contain redundant logging
2. Create backups of the files before modifying them
3. Apply the fixes to each file
4. Report on the changes made

### 2. Run the test script

```bash
# From the project root
python temp/test_log_cleanup.py
```

This script tests the regex patterns to ensure they match the expected log issues.

## Issues Fixed

### Stable_PDF_Parsing
- Duplicate "Generated images" messages
- Too many separator lines
- Redundant PDF conversion messages
- Duplicate document title generation messages

### Stable_Save_To_LanceDB & Embedding Loaders
- Duplicate SentenceTransformer loading messages
- Repetitive LanceDB connection messages
- Repeated warnings about small tables

### Agentic_Chunking
- Duplicate document processing messages
- Duplicate context-aware chunking messages
- Duplicate section processing messages

## Implementation Details

The fixes use various approaches:
- Flag variables to track if a message has been logged
- Conditional logging based on log level
- Combining multiple messages into one
- Memoization to avoid repeated warnings
- Tracking processed items in sets

All changes are focused on reducing log noise while preserving important information.

## Example Fix

Before:
```python
logger.info(f"Generated {len(image_list)} images from PDF", extra={"icon": "✅"})
logger.info(f"Generated {len(image_list)} images from PDF", extra={"icon": "✅"})
```

After:
```python
# Avoid duplicate "Generated images" messages
_generated_images_logged = False
logger.info(f"Generated {len(image_list)} images from PDF", extra={"icon": "✅"})
_generated_images_logged = True
``` 