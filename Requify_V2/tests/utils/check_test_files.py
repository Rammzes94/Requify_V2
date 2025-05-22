#!/usr/bin/env python3
"""
check_test_files.py

This script verifies that all test files needed for the E2E test scenarios 
are present in the input/raw directory.

It ensures that all required files for the defined test scenarios exist, logs missing or extra files, and provides file size information. This helps maintain test coverage and input hygiene for E2E tests.
"""

import os
import sys
import logging

# --- IMPORTANT PATH SETUP ---
# Add the project root and src to the system path so that _00_utils can be imported regardless of where the script is run from.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_DIR)
# --- END PATH SETUP ---

from src.utils import setup_logging, get_logger, update_token_counters, get_token_usage, print_token_usage, reset_token_counters, setup_project_directory, generate_timestamp
setup_project_directory()

# Set up logging
logger = get_logger("Test_Files_Checker")

# Import test scenarios definition 
try:
    from e2e_test_scenarios import SCENARIOS
except ImportError:
    logger.error("Could not import SCENARIOS from e2e_test_scenarios.py", extra={"icon": "❌"})
    # Define a minimal version for standalone use
    SCENARIOS = [
        {
            "id": 1,
            "name": "Baseline Document Ingest & Rerun (Identical Duplicate)",
            "steps": [
                {"file": "fighter_jet_rocket_launcher_spec.pdf"},
                {"file": "fighter_jet_rocket_launcher_spec.pdf"},
            ]
        },
        {
            "id": 2,
            "name": "Value Change Variant",
            "steps": [
                {"file": "fighter_jet_rocket_launcher_spec.pdf"},
                {"file": "fighter_jet_rocket_launcher_spec_2_changed_values.pdf"},
            ]
        },
        {
            "id": 3,
            "name": "Extra End Page",
            "steps": [
                {"file": "fighter_jet_rocket_launcher_spec.pdf"},
                {"file": "fighter_jet_rocket_launcher_spec_2_extra_end.pdf"},
            ]
        },
        {
            "id": 4,
            "name": "Reordered Specification",
            "steps": [
                {"file": "fighter_jet_rocket_launcher_spec.pdf"},
                {"file": "fighter_jet_rocket_launcher_spec_5_reordered.pdf"},
            ]
        },
        {
            "id": 5,
            "name": "Unique Original vs. Unique Reordered",
            "steps": [
                {"file": "fighter_jet_unique_original.pdf"},
                {"file": "fighter_jet_unique_reordered.pdf"},
            ]
        },
        {
            "id": 6,
            "name": "Language Variant",
            "steps": [
                {"file": "fighter_jet_rocket_launcher_spec.pdf"},
                {"file": "fighter_jet_rocket_launcher_spec_3_language_variant.pdf"},
            ]
        },
        {
            "id": 7,
            "name": "Changed vs. Extra vs. Reordered",
            "steps": [
                {"file": "fighter_jet_rocket_launcher_spec.pdf"},
                {"file": "fighter_jet_rocket_launcher_spec_2_changed_values.pdf"},
                {"file": "fighter_jet_rocket_launcher_spec_2_extra_end.pdf"},
                {"file": "fighter_jet_rocket_launcher_spec_5_reordered.pdf"},
            ]
        }
    ]

def check_test_files():
    """Check if all test files needed for the scenarios are present"""
    input_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "input", "raw")
    
    if not os.path.exists(input_dir):
        logger.error(f"Input directory not found: {input_dir}", extra={"icon": "❌"})
        return False
    
    logger.info(f"Checking test files in: {input_dir}", extra={"icon": "🔍"})
    
    # Get a list of all required files
    required_files = set()
    for scenario in SCENARIOS:
        for step in scenario["steps"]:
            required_files.add(step["file"])
    
    # Get a list of all available files
    available_files = set(os.listdir(input_dir))
    
    # Check if all required files are available
    missing_files = required_files - available_files
    
    if missing_files:
        logger.error(f"Missing {len(missing_files)} test files:", extra={"icon": "❌"})
        for file in sorted(missing_files):
            logger.error(f"  - {file}", extra={"icon": "❌"})
        return False
    
    # List the test files that are available
    logger.info(f"All {len(required_files)} required test files are available:", extra={"icon": "✅"})
    for file in sorted(required_files):
        file_path = os.path.join(input_dir, file)
        file_size = os.path.getsize(file_path) / 1024  # in KB
        logger.info(f"  - {file} ({file_size:.1f} KB)", extra={"icon": "📄"})
    
    # List any extra files that are not used in tests
    extra_files = available_files - required_files
    if extra_files:
        logger.info(f"Extra files found (not used in tests):", extra={"icon": "ℹ️"})
        for file in sorted(extra_files):
            if file.endswith(".pdf"):  # Only show PDF files
                file_path = os.path.join(input_dir, file)
                file_size = os.path.getsize(file_path) / 1024  # in KB
                logger.info(f"  - {file} ({file_size:.1f} KB)", extra={"icon": "📄"})
    
    return True

def main():
    """Main entry point"""
    success = check_test_files()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 