#!/usr/bin/env python3
"""
cleanup_codebase.py

This script reorganizes the Requify codebase according to the following principles:
1. Only pipeline-related files should be in _02_src
2. Tools should be in the tools directory
3. Tests should be in the tests directory
4. Only e2e pipeline tests should be kept, all other tests can be deleted
5. Duplicate files should be removed

The script will:
1. Move tool files from _02_src to tools
2. Move test files from _02_src to tests
3. Delete duplicate files across directories
4. Delete non-e2e tests
"""

import os
import sys
import shutil
import logging
import filecmp
from typing import List, Dict, Tuple

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import _02_src._00_utils as _00_utils
_00_utils.setup_project_directory()

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Constants
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, '_02_src')
TOOLS_DIR = os.path.join(PROJECT_ROOT, 'tools')
TESTS_DIR = os.path.join(PROJECT_ROOT, 'tests')
BACKUP_DIR = os.path.join(PROJECT_ROOT, 'backup', 'cleanup_backup')

# Setup logging
logger = logging.getLogger('cleanup_codebase')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - [%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Files to move from _02_src to tools
TOOLS_TO_MOVE = [
    {'src': os.path.join(SRC_DIR, 'reset_lancedb.py'), 'dest': os.path.join(TOOLS_DIR, 'reset_lancedb.py')},
    {'src': os.path.join(SRC_DIR, 'clean_lancedb.py'), 'dest': os.path.join(TOOLS_DIR, 'clean_lancedb.py')},
    {'src': os.path.join(SRC_DIR, 'visualize_db_relationships.py'), 'dest': os.path.join(TOOLS_DIR, 'visualization', 'visualize_db_relationships.py')},
    {'src': os.path.join(SRC_DIR, 'analyze_test_results.py'), 'dest': os.path.join(TOOLS_DIR, 'test_utils', 'analyze_test_results.py')},
    {'src': os.path.join(SRC_DIR, 'test_results_reporter.py'), 'dest': os.path.join(TOOLS_DIR, 'test_utils', 'test_results_reporter.py')},
]

# Files to move from _02_src to tests
TESTS_TO_MOVE = [
    {'src': os.path.join(SRC_DIR, 'test_document_diff.py'), 'dest': os.path.join(TESTS_DIR, 'test_document_diff.py')},
    {'src': os.path.join(SRC_DIR, 'check_test_files.py'), 'dest': os.path.join(TESTS_DIR, 'utils', 'check_test_files.py')},
    {'src': os.path.join(SRC_DIR, 'run_test_sequence.py'), 'dest': os.path.join(TESTS_DIR, 'utils', 'run_test_sequence.py')},
    {'src': os.path.join(SRC_DIR, 'test_token_tracking.py'), 'dest': os.path.join(TESTS_DIR, 'utils', 'test_token_tracking.py')},
    {'src': os.path.join(SRC_DIR, 'simple_token_test.py'), 'dest': os.path.join(TESTS_DIR, 'utils', 'simple_token_test.py')},
]

# Files to delete (duplicates or non-e2e tests)
FILES_TO_DELETE = [
    os.path.join(PROJECT_ROOT, 'clean_lancedb.py'),  # Duplicate in root
    os.path.join(PROJECT_ROOT, 'check_lancedb.py'),  # Not needed
    os.path.join(PROJECT_ROOT, 'detailed_db_check.py'),  # Not needed
    os.path.join(PROJECT_ROOT, 'check_chunk_relationships.py'),  # Not needed
    os.path.join(PROJECT_ROOT, 'test_chunk_fields.py'),  # Not needed
    os.path.join(PROJECT_ROOT, 'test_direct_lancedb.py'),  # Not needed
    os.path.join(PROJECT_ROOT, 'test_document_diff.py'),  # Duplicate in root
    os.path.join(PROJECT_ROOT, 'util_lancedb_viewer.py'),  # Not needed
    os.path.join(TESTS_DIR, 'check_chunk_relationships.py'),  # Not needed
    os.path.join(TESTS_DIR, 'test_chunk_fields.py'),  # Not needed
    os.path.join(TESTS_DIR, 'test_direct_lancedb.py'),  # Not needed
]

def create_backup_dir() -> None:
    """Create a backup directory for files before moving/deleting."""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    logger.info(f"🔄 Created backup directory at {BACKUP_DIR}")

def backup_file(file_path: str) -> None:
    """Back up a file before modifying/deleting it."""
    if os.path.exists(file_path):
        rel_path = os.path.relpath(file_path, PROJECT_ROOT)
        backup_path = os.path.join(BACKUP_DIR, rel_path)
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        shutil.copy2(file_path, backup_path)
        logger.info(f"📥 Backed up {rel_path}")

def resolve_file_conflict(src: str, dest: str) -> bool:
    """
    Resolve a conflict when the destination file already exists.
    
    Returns True if the source should replace destination, False otherwise.
    """
    if not os.path.exists(dest):
        return True  # No conflict, can proceed
    
    # Check if files are identical
    if filecmp.cmp(src, dest, shallow=False):
        logger.info(f"⚠️ Files are identical: {os.path.basename(src)}. Skipping.")
        return False  # Don't replace identical files
    
    # Get file sizes and modification times
    src_size = os.path.getsize(src)
    dest_size = os.path.getsize(dest)
    src_mtime = os.path.getmtime(src)
    dest_mtime = os.path.getmtime(dest)
    
    # If source is newer and larger, prefer it
    if src_mtime > dest_mtime and src_size >= dest_size:
        logger.info(f"🔄 Source file is newer and same/larger size: {os.path.basename(src)}. Replacing.")
        return True
    
    # Get user input for resolution
    print(f"\nConflict for file: {os.path.basename(src)}")
    print(f"  Source: {os.path.relpath(src, PROJECT_ROOT)} ({src_size} bytes, modified {src_mtime})")
    print(f"  Destination: {os.path.relpath(dest, PROJECT_ROOT)} ({dest_size} bytes, modified {dest_mtime})")
    choice = input("Replace destination with source? (y/n): ").lower().strip()
    return choice == 'y'

def move_file(src: str, dest: str) -> bool:
    """Move a file from source to destination."""
    if not os.path.exists(src):
        logger.warning(f"❌ Source file not found: {src}")
        return False
    
    # Create destination directory if it doesn't exist
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    
    # Check if destination already exists and handle conflict
    if os.path.exists(dest):
        if not resolve_file_conflict(src, dest):
            logger.info(f"⏩ Skipping {os.path.basename(src)} (keeping destination)")
            return False
    
    # Backup file before moving
    backup_file(src)
    
    # Backup destination if it exists
    if os.path.exists(dest):
        backup_file(dest)
    
    # Copy file to new location
    shutil.copy2(src, dest)
    
    # Remove original file
    os.remove(src)
    
    logger.info(f"✅ Moved {os.path.basename(src)} to {os.path.relpath(dest, PROJECT_ROOT)}")
    return True

def delete_file(file_path: str) -> bool:
    """Delete a file with backup."""
    if not os.path.exists(file_path):
        logger.warning(f"⚠️ File not found for deletion: {file_path}")
        return False
    
    # Backup file before deleting
    backup_file(file_path)
    
    # Delete file
    os.remove(file_path)
    
    logger.info(f"🗑️ Deleted {os.path.relpath(file_path, PROJECT_ROOT)}")
    return True

def move_tools_to_tools_dir() -> None:
    """Move tool files from _02_src to tools directory."""
    logger.info("🔄 Moving tools from _02_src to tools directory...")
    
    success_count = 0
    for file_info in TOOLS_TO_MOVE:
        if move_file(file_info['src'], file_info['dest']):
            success_count += 1
    
    logger.info(f"✅ Moved {success_count}/{len(TOOLS_TO_MOVE)} tools to tools directory")

def move_tests_to_tests_dir() -> None:
    """Move test files from _02_src to tests directory."""
    logger.info("🔄 Moving tests from _02_src to tests directory...")
    
    success_count = 0
    for file_info in TESTS_TO_MOVE:
        if move_file(file_info['src'], file_info['dest']):
            success_count += 1
    
    logger.info(f"✅ Moved {success_count}/{len(TESTS_TO_MOVE)} tests to tests directory")

def delete_duplicate_files() -> None:
    """Delete duplicate files and non-e2e tests."""
    logger.info("🔄 Deleting duplicate files and non-e2e tests...")
    
    success_count = 0
    for file_path in FILES_TO_DELETE:
        if delete_file(file_path):
            success_count += 1
    
    logger.info(f"✅ Deleted {success_count}/{len(FILES_TO_DELETE)} duplicate/unnecessary files")

def ensure_directories() -> None:
    """Ensure all necessary directories exist."""
    directories = [
        os.path.join(TOOLS_DIR, 'visualization'),
        os.path.join(TOOLS_DIR, 'test_utils'),
        os.path.join(TESTS_DIR, 'utils')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"📁 Ensured directory exists: {os.path.relpath(directory, PROJECT_ROOT)}")

def run_cleanup() -> None:
    """Run the codebase cleanup process."""
    logger.info("🚀 Starting codebase cleanup...")
    
    # Create backup directory
    create_backup_dir()
    
    # Ensure all required directories exist
    ensure_directories()
    
    # Move tools to tools directory
    move_tools_to_tools_dir()
    
    # Move tests to tests directory
    move_tests_to_tests_dir()
    
    # Delete duplicate files
    delete_duplicate_files()
    
    logger.info("✅ Codebase cleanup complete!")
    logger.info(f"📁 All original files were backed up to {BACKUP_DIR}")

if __name__ == "__main__":
    run_cleanup() 