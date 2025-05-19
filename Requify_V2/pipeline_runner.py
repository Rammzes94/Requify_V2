#!/usr/bin/env python3
"""
pipeline_runner.py

This script provides a user-friendly interface for running the document processing pipeline.
It allows users to:
1. Select which steps of the pipeline to run
2. Set up the database before running the pipeline
3. See real-time progress and results
4. Automatically display deduplication results after processing

The script is designed for ease of use while providing control over the pipeline's
execution, making it suitable for both testing and production use.
"""

import os
import sys
import argparse
import logging
import subprocess
from typing import Dict, List, Optional

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '_02_src')))
import _00_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '_02_src', '_00_utils')))
import config

# Import main pipeline controller
from pipeline_controller import (
    process_document,
    STEP_HASH_CHECK,
    STEP_PARSE,
    STEP_DEDUP_ONLY,
    STEP_SAVE_TO_DB,
    STEP_CHUNKING,
    STEP_EXTRACT_REQS,
)

# Setup centralized logging with script prefix
logger = _00_utils.get_logger("Pipeline_Runner")

# Pipeline step constants
STEP_EXTENSION_FILTER = 1
STEP_DEDUPLICATION = 2
STEP_PARSE_PDF = 3
STEP_CHUNKING = 4
STEP_EXTRACT_REQS = 5

# Descriptions for each pipeline step
STEP_DESCRIPTIONS = {
    STEP_EXTENSION_FILTER: "File Extension Filtering",
    STEP_DEDUPLICATION: "Document Duplication Check",
    STEP_PARSE_PDF: "PDF Parsing & Content Extraction",
    STEP_CHUNKING: "Content Chunking",
    STEP_EXTRACT_REQS: "Requirements Extraction"
}

def show_deduplication_results(input_file: str) -> None:
    """Show deduplication results for a processed document."""
    doc_name = os.path.basename(input_file)
    logger.info("=== Deduplication Results ===", extra={"icon": "📊"})
    
    # Config is already loaded, no need to set environment variables
    
    try:
        # Display document info from LanceDB
        script_dir = os.path.dirname(os.path.abspath(__file__))
        validation_script = os.path.join(script_dir, "tools", "validation", "show_deduplication_results.py")
        
        if os.path.exists(validation_script):
            cmd = [sys.executable, validation_script, "--relationships", "--text", "--doc", doc_name]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.stdout:
                logger.info(result.stdout, extra={"icon": "ℹ️"})
            else:
                logger.info(f"No deduplication results found for document: {doc_name}", extra={"icon": "⚠️"})
            
            if result.stderr:
                logger.error(f"Errors from validation script: {result.stderr}", extra={"icon": "❌"})
        else:
            logger.error(f"Validation script not found at: {validation_script}", extra={"icon": "❌"})
            
    except Exception as e:
        logger.error(f"Error showing deduplication results: {e}", extra={"icon": "❌"})
    
    logger.info("=" * 80, extra={"icon": "📊"})

def setup_database() -> bool:
    """Set up the LanceDB database tables."""
    try:
        from _02_src._00_lancedb_admin.init_lancedb import main as setup_db_main
        logger.info("=== Database Setup ===", extra={"icon": "🔧"})
        logger.info("Setting up database tables...", extra={"icon": "🔄"})
        setup_db_main()
        logger.info("Database setup completed.", extra={"icon": "✅"})
        return True
    except ImportError as e:
        logger.error(f"Could not import database setup module: {e}", extra={"icon": "❌"})
        return False
    except Exception as e:
        logger.error(f"Error setting up database: {e}", extra={"icon": "❌"})
        return False

def list_input_files() -> List[str]:
    """List available input files."""
    input_dir = os.path.join("_01_input", "raw")
    if not os.path.exists(input_dir):
        logger.error(f"Input directory not found: {input_dir}", extra={"icon": "❌"})
        return []
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
             if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('.pdf', '.docx', '.txt'))]
    return sorted(files)

def print_available_steps() -> None:
    """Print available pipeline steps."""
    logger.info("=== Available Pipeline Steps ===", extra={"icon": "📋"})
    for step_num, description in STEP_DESCRIPTIONS.items():
        logger.info(f"{step_num}. {description}", extra={"icon": "ℹ️"})

def select_max_step() -> int:
    """Prompt the user to select the maximum pipeline step to run."""
    print_available_steps()
    while True:
        try:
            step = int(input("\nEnter the step number to run up to: "))
            if step in STEP_DESCRIPTIONS:
                return step
            else:
                logger.warning(f"Invalid step number. Please enter a number between {min(STEP_DESCRIPTIONS.keys())} and {max(STEP_DESCRIPTIONS.keys())}.", extra={"icon": "⚠️"})
        except ValueError:
            logger.warning("Please enter a valid number.", extra={"icon": "⚠️"})

def select_input_file() -> Optional[str]:
    """Prompt the user to select an input file."""
    files = list_input_files()
    if not files:
        logger.warning("No input files found. Please add files to the _01_input/raw directory.", extra={"icon": "⚠️"})
        return None
    logger.info("=== Available Input Files ===", extra={"icon": "📋"})
    for i, file_path in enumerate(files):
        logger.info(f"{i+1}. {os.path.basename(file_path)}", extra={"icon": "📄"})
    while True:
        try:
            selection = int(input("\nEnter the file number to process (or 0 to provide a custom path): "))
            if selection == 0:
                custom_path = input("Enter the full path to the input file: ")
                if os.path.exists(custom_path):
                    return custom_path
                else:
                    logger.warning(f"File not found: {custom_path}", extra={"icon": "⚠️"})
            elif 1 <= selection <= len(files):
                return files[selection-1]
            else:
                logger.warning(f"Invalid selection. Please enter a number between 0 and {len(files)}.", extra={"icon": "⚠️"})
        except ValueError:
            logger.warning("Please enter a valid number.", extra={"icon": "⚠️"})

def run_pipeline_interactive() -> None:
    """Run the pipeline with interactive prompts."""
    logger.info("=== Document Processing Pipeline Runner ===", extra={"icon": "🚀"})
    
    # Config is already loaded, no need to set environment variables
    
    if input("Set up database before running? (y/n): ").lower() == 'y':
        if not setup_database():
            if input("Continue anyway? (y/n): ").lower() != 'y':
                logger.info("Exiting.", extra={"icon": "🚪"})
                return
    input_file = select_input_file()
    if not input_file:
        logger.warning("No input file selected. Exiting.", extra={"icon": "🚪"})
        return
    max_step = select_max_step()
    logger.info("=== Pipeline Run Settings ===", extra={"icon": "⚙️"})
    logger.info(f"Input file: {input_file}", extra={"icon": "📄"})
    logger.info(f"Maximum step: {max_step} - {STEP_DESCRIPTIONS[max_step]}", extra={"icon": "🔢"})
    logger.info("=== Starting Pipeline ===", extra={"icon": "🚀"})
    try:
        success = process_document(input_file, max_step=max_step, dry_run=False)
        if success:
            logger.info("=== Pipeline Completed Successfully ===", extra={"icon": "✅"})
            
            # Show deduplication results if the chunking step was run
            if max_step >= STEP_CHUNKING:
                show_deduplication_results(input_file)
        else:
            logger.error("=== Pipeline Failed ===", extra={"icon": "❌"})
            logger.error("Check the logs for more information.", extra={"icon": "🔍"})
    except Exception as e:
        logger.error(f"=== Pipeline Error ===\n{e}", extra={"icon": "💥"})
        raise

def main() -> None:
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Document Processing Pipeline Runner")
    parser.add_argument("--input", type=str, help="Path to the input file")
    parser.add_argument("--max-step", type=int, choices=list(STEP_DESCRIPTIONS.keys()),
                        help="Maximum pipeline step to run")
    parser.add_argument("--setup-db", action="store_true", help="Set up database before running")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--show-dedup", action="store_true", help="Show deduplication results after processing")
    parser.add_argument("--verbose", action="store_true", default=True, 
                       help="Enable verbose output (default: True)")
    args = parser.parse_args()
    
    # If verbose argument is provided, we could update the config values if needed
    # But we'll rely on the values already loaded from config
    
    if args.interactive or not (args.input and args.max_step is not None):
        run_pipeline_interactive()
        return
    if args.setup_db:
        if not setup_database():
            logger.error("Database setup failed. Exiting.", extra={"icon": "❌"})
            return
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}", extra={"icon": "❌"})
        return
    logger.info(f"Running pipeline on {args.input} up to step {args.max_step} ({STEP_DESCRIPTIONS[args.max_step]})", extra={"icon": "🚀"})
    try:
        success = process_document(args.input, max_step=args.max_step, dry_run=False)
        if success:
            logger.info("Pipeline completed successfully.", extra={"icon": "✅"})
            
            # Show deduplication results after successful run
            if args.show_dedup or args.max_step >= STEP_CHUNKING:
                show_deduplication_results(args.input)
                
        else:
            logger.error("Pipeline failed. Check the logs for more information.", extra={"icon": "❌"})
    except Exception as e:
        logger.error(f"Pipeline error: {e}", extra={"icon": "💥"})
        raise

def test_with_hardcoded_file():
    """Test the pipeline with a hardcoded file."""
    logger.info("=== Running Pipeline Test with Hardcoded File ===", extra={"icon": "🧪"})
    
    # Config is already loaded, no need to set environment variables
    
    test_file = "_01_input/raw/fighter_jet_rocket_launcher_spec_2.pdf"
    if not os.path.exists(test_file):
        logger.error(f"Test file not found: {test_file}", extra={"icon": "❌"})
        return False
    logger.info(f"Running pipeline on test file: {test_file}", extra={"icon": "🚀"})
    try:
        max_step = STEP_EXTRACT_REQS  # Run the complete pipeline
        return process_document(test_file, max_step=max_step, dry_run=False)
    except Exception as e:
        logger.error(f"Pipeline test error: {e}", extra={"icon": "💥"})
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_with_hardcoded_file()
    else:
        main()
