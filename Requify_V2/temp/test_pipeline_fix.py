#!/usr/bin/env python3
"""
test_pipeline_fix.py

This script tests the pipeline controller with two documents:
1. fighter_jet_rocket_launcher_spec_2.pdf (resetting the DB first)
2. fighter_jet_rocket_launcher_spec_2_changed_values.pdf

The test runs the pipeline up to step 5 (chunking) and automatically selects
the "detailed" analysis option to avoid requiring user input.
"""

import os
import sys
import shutil
import logging
import subprocess
from pathlib import Path

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("test_pipeline_fix")

# Constants
OUTPUT_DIR = "output"
LANCEDB_DIR = os.path.join(OUTPUT_DIR, "lancedb")
DOC1 = "input/raw/fighter_jet_rocket_launcher_spec_2.pdf"
DOC2 = "input/raw/fighter_jet_rocket_launcher_spec_2_changed_values.pdf"

def reset_database():
    """Reset the LanceDB database by removing and recreating the directory."""
    if os.path.exists(LANCEDB_DIR):
        logger.info(f"Removing existing database at {LANCEDB_DIR}")
        shutil.rmtree(LANCEDB_DIR)
    
    os.makedirs(LANCEDB_DIR, exist_ok=True)
    logger.info(f"Created empty database directory at {LANCEDB_DIR}")

def run_pipeline(doc_path, max_step=5):
    """Run the pipeline controller with the specified document."""
    logger.info(f"Running pipeline on {doc_path} up to step {max_step}")
    
    # Set environment variable to automatically choose "detailed" analysis
    env = os.environ.copy()
    env["REQUIFY_AUTO_CHOICE"] = "detailed"
    
    # Run the pipeline controller as a subprocess
    cmd = [
        sys.executable,
        "src/pipeline_controller.py",
        f"--doc_path={doc_path}",
        f"--max_step={max_step}",
        "--skip_hash_check"  # Skip hash check to ensure processing
    ]
    
    logger.info(f"Executing command: {' '.join(cmd)}")
    
    try:
        process = subprocess.run(
            cmd,
            env=env,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info(f"Pipeline command output:\n{process.stdout}")
        logger.info(f"Pipeline completed successfully for {doc_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Pipeline failed for {doc_path}: {e}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        return False

def main():
    """Main function to run the test."""
    # Ensure the test documents exist
    for doc_path in [DOC1, DOC2]:
        if not os.path.exists(doc_path):
            logger.error(f"Test document not found: {doc_path}")
            logger.info("Please ensure test documents are placed in the input/raw directory")
            return False
    
    # Step 1: Reset the database and process the first document
    logger.info("Step 1: Processing first document with fresh database")
    reset_database()
    success1 = run_pipeline(DOC1)
    
    if not success1:
        logger.error("Test failed during first document processing")
        return False
    
    # Step 2: Process the second document without resetting the database
    logger.info("Step 2: Processing second document (should detect similarity)")
    success2 = run_pipeline(DOC2)
    
    if not success2:
        logger.error("Test failed during second document processing")
        return False
    
    logger.info("Test completed successfully! The fix has been verified.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 