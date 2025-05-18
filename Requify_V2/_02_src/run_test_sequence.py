"""
run_test_sequence.py

This script runs a test sequence to demonstrate the context-aware chunking pipeline:
1. First, it cleans the LanceDB database to start fresh
2. Then it processes the first document (fighter_jet_rocket_launcher_spec_2.pdf) 
   with standard chunking
3. Finally, it processes a second document (fighter_jet_rocket_launcher_spec_2_changed_values.pdf)
   which should trigger context-aware chunking
   
The goal is to show the pipeline correctly detects document similarity and uses
context-aware chunking to identify specific changes between document versions.
"""

import os
import sys
import logging
import argparse
from dotenv import load_dotenv

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import _00_utils
_00_utils.setup_project_directory()

# Import our pipeline components
from clean_lancedb import clean_lancedb
from pipeline_controller import process_document, STEP_EXTRACT_REQS

# Load environment variables
load_dotenv()

# Setup logging
logger = _00_utils.setup_logging()

# Test document paths
DOC1_PATH = os.path.join("_01_input", "raw", "fighter_jet_rocket_launcher_spec_2.pdf")
DOC2_PATH = os.path.join("_01_input", "raw", "fighter_jet_rocket_launcher_spec_2_changed_values.pdf")

def run_test_sequence(skip_clean=False, max_step=STEP_EXTRACT_REQS):
    """
    Run the full test sequence with our test documents.
    
    Args:
        skip_clean: If True, skips the database cleaning step
        max_step: Maximum pipeline step to execute
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    doc1_full_path = os.path.join(project_root, DOC1_PATH)
    doc2_full_path = os.path.join(project_root, DOC2_PATH)
    
    # Verify documents exist
    if not os.path.exists(doc1_full_path):
        logger.error(f"❌ First test document not found at {doc1_full_path}")
        return False
        
    if not os.path.exists(doc2_full_path):
        logger.error(f"❌ Second test document not found at {doc2_full_path}")
        return False
    
    # Clean the database if requested
    if not skip_clean:
        logger.info("🧹 Cleaning LanceDB database before tests")
        if not clean_lancedb(confirm=True, backup=True):
            logger.error("❌ Failed to clean LanceDB database")
            return False
    
    # Process the first document (this should use standard chunking)
    logger.info(f"🚀 Processing first document: {DOC1_PATH}")
    if not process_document(doc1_full_path, max_step, dry_run=False, skip_hash_check=True):
        logger.error("❌ Failed to process first document")
        return False
    
    # Wait for a moment to ensure the first document is fully processed
    import time
    time.sleep(2)
    
    # Process the second document (this should trigger context-aware chunking)
    logger.info(f"🚀 Processing second document: {DOC2_PATH}")
    if not process_document(doc2_full_path, max_step, dry_run=False, skip_hash_check=True):
        logger.error("❌ Failed to process second document")
        return False
    
    logger.info("✅ Test sequence completed successfully!")
    return True

def main():
    parser = argparse.ArgumentParser(description='Run test sequence for context-aware chunking.')
    parser.add_argument('--skip-clean', action='store_true', help='Skip cleaning the database')
    parser.add_argument('--max-step', type=int, default=STEP_EXTRACT_REQS, 
                        help='Maximum pipeline step to execute')
    
    args = parser.parse_args()
    
    return run_test_sequence(skip_clean=args.skip_clean, max_step=args.max_step)

if __name__ == "__main__":
    sys.exit(0 if main() else 1) 