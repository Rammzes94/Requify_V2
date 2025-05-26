#!/usr/bin/env python3
"""
log_cleanup.py

This script cleans up redundant logging in the Requify codebase.
It identifies and fixes logging patterns that lead to duplicate or redundant messages.
"""

import os
import sys
import re
from typing import List, Dict, Tuple, Set, Optional
import logging

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import setup_logging, get_logger

# Setup logging
logger = get_logger("Log_Cleanup")

# Constants
MODULES_TO_FIX = {
    "stable_pdf_parsing.py": [
        "pdf-gen-dup-images",
        "pdf-sep-lines",
        "pdf-convert-to-images",
        "pdf-title-gen-twice"
    ],
    "stable_save_to_lancedb.py": [
        "sentence-transformer-dup",
        "lancedb-connect-spam",
        "lancedb-no-index-repeat"
    ],
    "agentic_chunking.py": [
        "chunk-proc-dup-doc", 
        "chunk-context-dup",
        "chunk-section-dup"
    ]
}

# Regex patterns for identifying issues - all with proper capture groups
PATTERNS = {
    "pdf-gen-dup-images": r'(logger\.info\(f"Generated \{len\([^)]+\)\} images from PDF",\s*extra=\{"icon": "✅"\}\))',
    "pdf-sep-lines": r'(logger\.info\("[-]{80}", extra=\{"icon": "[^"]+"\}\))',
    "pdf-convert-to-images": [
        r'(logger\.info\(f"Processing \'{[^}]+}\' to images\.\.\.", extra=\{"icon": "🔄"\}\))',
        r'(logger\.info\(f"Converting PDF to images: \{[^}]+}\", extra=\{"icon": "📄"\}\))'
    ],
    "pdf-title-gen-twice": [
        r'(logger\.info\(f"Generating document title for \{[^}]+\} from page summaries\.\.\.", extra=\{"icon": "🔤"\}\))',
        r'(logger\.info\(f"Generating document title for \{[^}]+\}\.\.\.", extra=\{"icon": "🔤"\}\))'
    ],
    "sentence-transformer-dup": [
        r'(print\("Use pytorch device_name: [^"]+"\))',
        r'(logger\.info\(f"Use pytorch device_name: [^"]+", extra=\{"icon": "ℹ️"\}\))',
        r'(print\("Load pretrained SentenceTransformer: [^"]+"\))',
        r'(logger\.info\(f"Load pretrained SentenceTransformer: [^"]+", extra=\{"icon": "🧠"\}\))'
    ],
    "lancedb-connect-spam": [
        r'(logger\.info\(f"Connecting to LanceDB at: \{[^}]+\}", extra=\{"icon": "🔌"\}\))',
        r'(logger\.info\("Connected to LanceDB", extra=\{"icon": "✅"\}\))'
    ],
    "lancedb-no-index-repeat": r'(logger\.info\(f"Not creating index: \{[^}]+\} has only \{[^}]+\} rows, minimum \d+ required", extra=\{"icon": "⚠️"\}\))',
    "chunk-proc-dup-doc": [
        r'(logger\.info\(f"Processing document: \{document_id\}", extra=\{"icon": "🚀"\}\))',
        r'(logger\.info\(f"Processing document: \{document_id\}", extra=\{"icon": "🔄"\}\))'
    ],
    "chunk-context-dup": [
        r'(logger\.info\(f"Performing context-aware chunking with \{len\([^)]+\)\} reference chunks", extra=\{"icon": "🧩"\}\))',
        r'(logger\.info\(f"🔄 Performing context-aware chunking with \{len\([^)]+\)\} reference chunks", extra=\{"icon": "🔄"\}\))'
    ],
    "chunk-section-dup": r'(logger\.info\(f"🔄 Processing section \{i\+1\}/\{len\([^)]+\)\}", extra=\{"icon": "🔄"\}\))'
}

# Replacement patterns
REPLACEMENTS = {
    "pdf-gen-dup-images": {
        "pattern": PATTERNS["pdf-gen-dup-images"],
        "replacement": """# Avoid duplicate "Generated images" messages
_generated_images_logged = False
\\1
_generated_images_logged = True"""
    },
    "pdf-sep-lines": {
        "pattern": PATTERNS["pdf-sep-lines"],
        "replacement": """# Reduce separator line spam
if logger.isEnabledFor(logging.DEBUG):
    \\1"""
    },
    "pdf-convert-to-images": {
        "pattern": "|".join(PATTERNS["pdf-convert-to-images"]),
        "replacement": """# Combine PDF conversion messages
logger.info(f"Converting '{pdf_path}' to images...", extra={"icon": "🔄"})"""
    },
    "pdf-title-gen-twice": {
        "pattern": PATTERNS["pdf-title-gen-twice"][1],
        "replacement": """# Skip redundant title generation message
if not VERBOSE_PDF_PARSING_OUTPUT:
    \\1"""
    },
    "sentence-transformer-dup": {
        "pattern": "|".join(PATTERNS["sentence-transformer-dup"]),
        "replacement": """# Use only logger.info for embedding model loading
logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME} on {EMBEDDING_DEVICE}", extra={"icon": "🧠"})"""
    },
    "lancedb-connect-spam": {
        "pattern": "|".join(PATTERNS["lancedb-connect-spam"]),
        "replacement": """# Reduce LanceDB connection logging
if logger.isEnabledFor(logging.DEBUG):
    \\1"""
    },
    "lancedb-no-index-repeat": {
        "pattern": PATTERNS["lancedb-no-index-repeat"],
        "replacement": """# Memoize index warnings to avoid repetition
_index_warnings = set()
def _log_index_warning(table_name, row_count):
    key = f"{table_name}_{row_count}"
    if key not in _index_warnings:
        logger.info(f"Not creating index: {table_name} has only {row_count} rows, minimum 256 required", extra={"icon": "⚠️"})
        _index_warnings.add(key)
# Use function for warning
_log_index_warning(LANCEDB_TABLE_NAME, record_count)"""
    },
    "chunk-proc-dup-doc": {
        "pattern": "|".join(PATTERNS["chunk-proc-dup-doc"]),
        "replacement": """# Single document processing message
logger.info(f"Processing document: {document_id}", extra={"icon": "🚀"})"""
    },
    "chunk-context-dup": {
        "pattern": "|".join(PATTERNS["chunk-context-dup"]),
        "replacement": """# Single context-aware chunking message
logger.info(f"Performing context-aware chunking with {len(similar_chunks if similar_chunks else [])} reference chunks", extra={"icon": "🧩"})"""
    },
    "chunk-section-dup": {
        "pattern": PATTERNS["chunk-section-dup"],
        "replacement": """# Use a section tracking set to avoid duplicate section processing messages
_processed_sections = set()
if f"{i+1}_{len(document_sections)}" not in _processed_sections:
    \\1
    _processed_sections.add(f"{i+1}_{len(document_sections)}")"""
    }
}

def find_file_paths(modules: Dict[str, List[str]]) -> Dict[str, str]:
    """Find the actual paths to the module files."""
    result = {}
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
    
    for module_name in modules:
        matches = []
        # Walk through the src directory to find the module
        for root, _, files in os.walk(src_dir):
            if module_name in files:
                matches.append(os.path.join(root, module_name))
        
        if matches:
            # Use the first match if multiple are found
            result[module_name] = matches[0]
            logger.info(f"Found module {module_name} at {matches[0]}", extra={"icon": "✅"})
        else:
            logger.warning(f"Could not find module {module_name}", extra={"icon": "⚠️"})
    
    return result

def backup_file(file_path: str) -> str:
    """Create a backup of the file before modifying it."""
    backup_path = f"{file_path}.bak"
    try:
        with open(file_path, 'r', encoding='utf-8') as src, open(backup_path, 'w', encoding='utf-8') as dst:
            dst.write(src.read())
        logger.info(f"Created backup of {file_path} at {backup_path}", extra={"icon": "📂"})
        return backup_path
    except Exception as e:
        logger.error(f"Error creating backup of {file_path}: {str(e)}", extra={"icon": "❌"})
        return ""

def fix_module(file_path: str, issues_to_fix: List[str]) -> bool:
    """Apply fixes to a module file."""
    if not os.path.exists(file_path):
        logger.error(f"File {file_path} does not exist", extra={"icon": "❌"})
        return False
    
    # Create backup
    backup_path = backup_file(file_path)
    if not backup_path:
        return False
    
    # Read file content
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        logger.error(f"Error reading {file_path}: {str(e)}", extra={"icon": "❌"})
        return False
    
    # Apply fixes
    changes_made = False
    for issue in issues_to_fix:
        if issue in REPLACEMENTS:
            pattern = REPLACEMENTS[issue]["pattern"]
            replacement = REPLACEMENTS[issue]["replacement"]
            
            # Check if issue pattern exists in the file
            if re.search(pattern, content):
                new_content = re.sub(pattern, replacement, content)
                if new_content != content:
                    content = new_content
                    changes_made = True
                    logger.info(f"Fixed issue {issue} in {os.path.basename(file_path)}", extra={"icon": "🔧"})
                else:
                    logger.info(f"No changes needed for issue {issue} in {os.path.basename(file_path)}", extra={"icon": "ℹ️"})
            else:
                logger.info(f"Issue {issue} not found in {os.path.basename(file_path)}", extra={"icon": "ℹ️"})
    
    # Write updated content
    if changes_made:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Successfully updated {file_path}", extra={"icon": "✅"})
            return True
        except Exception as e:
            logger.error(f"Error writing to {file_path}: {str(e)}", extra={"icon": "❌"})
            # Restore from backup
            try:
                with open(backup_path, 'r', encoding='utf-8') as src, open(file_path, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())
                logger.info(f"Restored {file_path} from backup", extra={"icon": "🔄"})
            except Exception as e2:
                logger.error(f"Error restoring from backup: {str(e2)}", extra={"icon": "❌"})
            return False
    else:
        logger.info(f"No changes made to {file_path}", extra={"icon": "ℹ️"})
        return True

def main():
    """Main function to run the log cleanup script."""
    logger.info("Starting log cleanup script", extra={"icon": "🚀"})
    
    # Find module paths
    module_paths = find_file_paths(MODULES_TO_FIX)
    
    # Fix each module
    success_count = 0
    for module_name, issues in MODULES_TO_FIX.items():
        if module_name in module_paths:
            logger.info(f"Processing module {module_name}", extra={"icon": "🔄"})
            if fix_module(module_paths[module_name], issues):
                success_count += 1
    
    logger.info(f"Log cleanup completed. Successfully processed {success_count}/{len(MODULES_TO_FIX)} modules", 
                extra={"icon": "✅"})

if __name__ == "__main__":
    main() 