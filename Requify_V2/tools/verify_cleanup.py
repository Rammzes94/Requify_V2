#!/usr/bin/env python3
"""
verify_cleanup.py

This script verifies that the codebase reorganization was successful.
It checks that:
1. Tools are in the tools directory
2. Tests are in the tests directory
3. Duplicate files have been removed
4. The e2e pipeline can still run
"""

import os
import sys
import subprocess
import logging
from typing import List, Dict, Tuple, Any

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

# Setup logging
logger = logging.getLogger('verify_cleanup')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - [%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Expected files in their new locations
EXPECTED_TOOLS = [
    os.path.join(TOOLS_DIR, 'reset_lancedb.py'),
    os.path.join(TOOLS_DIR, 'clean_lancedb.py'),
    os.path.join(TOOLS_DIR, 'visualization', 'visualize_db_relationships.py'),
    os.path.join(TOOLS_DIR, 'test_utils', 'analyze_test_results.py'),
    os.path.join(TOOLS_DIR, 'test_utils', 'test_results_reporter.py'),
]

EXPECTED_TESTS = [
    os.path.join(TESTS_DIR, 'test_document_diff.py'),
    os.path.join(TESTS_DIR, 'utils', 'check_test_files.py'),
    os.path.join(TESTS_DIR, 'utils', 'run_test_sequence.py'),
    os.path.join(TESTS_DIR, 'utils', 'test_token_tracking.py'),
    os.path.join(TESTS_DIR, 'utils', 'simple_token_test.py'),
    os.path.join(TESTS_DIR, 'e2e', 'test_scenarios.py'),
]

# Files that should be removed
REMOVED_FILES = [
    os.path.join(PROJECT_ROOT, 'clean_lancedb.py'),
    os.path.join(PROJECT_ROOT, 'check_lancedb.py'),
    os.path.join(PROJECT_ROOT, 'detailed_db_check.py'),
    os.path.join(PROJECT_ROOT, 'check_chunk_relationships.py'),
    os.path.join(PROJECT_ROOT, 'test_chunk_fields.py'),
    os.path.join(PROJECT_ROOT, 'test_direct_lancedb.py'),
    os.path.join(PROJECT_ROOT, 'test_document_diff.py'),
    os.path.join(PROJECT_ROOT, 'util_lancedb_viewer.py'),
    os.path.join(TESTS_DIR, 'check_chunk_relationships.py'),
    os.path.join(TESTS_DIR, 'test_chunk_fields.py'),
    os.path.join(TESTS_DIR, 'test_direct_lancedb.py'),
]

def check_file_exists(file_path: str) -> bool:
    """Check if a file exists at the specified path."""
    return os.path.exists(file_path) and os.path.isfile(file_path)

def check_expected_files(file_list: List[str], category: str) -> Tuple[int, int]:
    """Check that expected files exist and return count of found/expected."""
    found_count = 0
    for file_path in file_list:
        if check_file_exists(file_path):
            found_count += 1
            logger.info(f"✅ Found {category}: {os.path.relpath(file_path, PROJECT_ROOT)}")
        else:
            logger.warning(f"❌ Missing {category}: {os.path.relpath(file_path, PROJECT_ROOT)}")
    
    return found_count, len(file_list)

def check_removed_files(file_list: List[str]) -> Tuple[int, int]:
    """Check that files have been removed and return count of removed/expected."""
    removed_count = 0
    for file_path in file_list:
        if not check_file_exists(file_path):
            removed_count += 1
            logger.info(f"✅ Removed: {os.path.relpath(file_path, PROJECT_ROOT)}")
        else:
            logger.warning(f"❌ Not removed: {os.path.relpath(file_path, PROJECT_ROOT)}")
    
    return removed_count, len(file_list)

def run_e2e_test() -> bool:
    """Run the e2e test to verify the pipeline works."""
    logger.info("🔄 Running e2e pipeline test...")
    
    e2e_test_script = os.path.join(TESTS_DIR, 'e2e', 'test_scenarios.py')
    
    if not check_file_exists(e2e_test_script):
        logger.error(f"❌ E2E test script not found: {os.path.relpath(e2e_test_script, PROJECT_ROOT)}")
        return False
    
    try:
        # Run only the first test scenario to save time
        cmd = [sys.executable, e2e_test_script, "--scenario", "1"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ E2E test passed!")
            return True
        else:
            logger.error(f"❌ E2E test failed with return code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"❌ Error running E2E test: {e}")
        return False

def verify_cleanup() -> bool:
    """Verify that the codebase reorganization was successful."""
    logger.info("🚀 Starting cleanup verification...")
    
    # Check expected tools
    found_tools, total_tools = check_expected_files(EXPECTED_TOOLS, "tool")
    tools_success = found_tools == total_tools
    logger.info(f"Tool files: {found_tools}/{total_tools} found")
    
    # Check expected tests
    found_tests, total_tests = check_expected_files(EXPECTED_TESTS, "test")
    tests_success = found_tests == total_tests
    logger.info(f"Test files: {found_tests}/{total_tests} found")
    
    # Check removed files
    removed_files, total_removed = check_removed_files(REMOVED_FILES)
    removed_success = removed_files == total_removed
    logger.info(f"Removed files: {removed_files}/{total_removed} removed")
    
    # Check if pipeline still works
    pipeline_success = run_e2e_test()
    
    # Overall success
    overall_success = tools_success and tests_success and removed_success and pipeline_success
    
    if overall_success:
        logger.info("✅ Cleanup verification complete: SUCCESS")
    else:
        logger.warning("⚠️ Cleanup verification complete: SOME ISSUES FOUND")
        
        if not tools_success:
            logger.warning("   - Some tool files are missing")
        if not tests_success:
            logger.warning("   - Some test files are missing")
        if not removed_success:
            logger.warning("   - Some files were not removed")
        if not pipeline_success:
            logger.warning("   - E2E pipeline test failed")
    
    return overall_success

if __name__ == "__main__":
    success = verify_cleanup()
    sys.exit(0 if success else 1) 