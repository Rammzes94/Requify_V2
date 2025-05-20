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
import subprocess
from typing import Dict, List, Optional

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
import _00_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
import config

# Import main pipeline controller and its step constants
# These are the constants pipeline_controller.process_document expects.
from pipeline_controller import (
    process_document,
    STEP_HASH_CHECK as CTRL_STEP_HASH_CHECK,
    STEP_PARSE as CTRL_STEP_PARSE,
    STEP_DEDUP_ONLY as CTRL_STEP_DEDUP_ONLY,
    STEP_SAVE_TO_DB as CTRL_STEP_SAVE_TO_DB,
    STEP_CHUNKING as CTRL_STEP_CHUNKING,
    STEP_EXTRACT_REQS as CTRL_STEP_EXTRACT_REQS,
)

# Setup centralized logging with script prefix
logger = _00_utils.get_logger("Pipeline_Runner")

# UI step constants for pipeline_runner.py interface
# These are distinct from the controller's step constants to avoid conflicts.
UI_STEP_EXTENSION_FILTER = 1
UI_STEP_DEDUPLICATION = 2
UI_STEP_PARSE_PDF = 3
UI_STEP_CHUNKING = 4
UI_STEP_EXTRACT_REQS = 5

# Descriptions for each UI pipeline step
STEP_DESCRIPTIONS = {
    UI_STEP_EXTENSION_FILTER: "File Extension Filtering",
    UI_STEP_DEDUPLICATION: "Document Duplication Check",
    UI_STEP_PARSE_PDF: "PDF Parsing & Content Extraction",
    UI_STEP_CHUNKING: "Content Chunking",
    UI_STEP_EXTRACT_REQS: "Requirements Extraction"
}

# Mapping from UI step numbers to the actual controller step constants
# This ensures that process_document receives the constants it expects.
UI_TO_CONTROLLER_STEP_MAP = {
    UI_STEP_EXTENSION_FILTER: CTRL_STEP_HASH_CHECK,  # Assuming extension filter implies running up to hash check at least
                                                  # Or this could be a very early specific step if controller supports it.
                                                  # If extension filtering is done *before* process_document, this might map to None
                                                  # or the first relevant controller step. For now, mapping to HASH_CHECK.
    UI_STEP_DEDUPLICATION: CTRL_STEP_DEDUP_ONLY,    # Or CTRL_STEP_SAVE_TO_DB depending on controller's logic
    UI_STEP_PARSE_PDF: CTRL_STEP_PARSE,
    UI_STEP_CHUNKING: CTRL_STEP_CHUNKING,
    UI_STEP_EXTRACT_REQS: CTRL_STEP_EXTRACT_REQS
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
        from src._00_lancedb_admin.init_lancedb import main as setup_db_main
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
            if step in STEP_DESCRIPTIONS: # Check against keys of STEP_DESCRIPTIONS (UI_STEP_... constants)
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
    max_step_ui = select_max_step()
    
    # Map the UI-selected step to the actual controller step constant
    controller_max_step = UI_TO_CONTROLLER_STEP_MAP.get(max_step_ui)

    if controller_max_step is None:
        logger.error(f"Could not map UI step {max_step_ui} ('{STEP_DESCRIPTIONS.get(max_step_ui)}') to a valid controller step. Exiting.", extra={"icon": "❌"})
        return

    logger.info("=== Pipeline Run Settings ===", extra={"icon": "⚙️"})
    logger.info(f"Input file: {input_file}", extra={"icon": "📄"})
    logger.info(f"Maximum UI step: {max_step_ui} - {STEP_DESCRIPTIONS[max_step_ui]} (Controller step: {controller_max_step})", extra={"icon": "🔢"})
    logger.info("=== Starting Pipeline ===", extra={"icon": "🚀"})
    try:
        success = process_document(input_file, max_step=controller_max_step, dry_run=False)
        if success:
            logger.info("=== Pipeline Completed Successfully ===", extra={"icon": "✅"})
            
            # Show deduplication results if the chunking step (UI perspective) was run or surpassed
            if max_step_ui >= UI_STEP_CHUNKING:
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
                        help="Maximum pipeline step to run (selects UI step number)")
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
    max_step_ui = args.max_step # This is a UI step number (1-5)
    
    # Map the UI-selected step from CLI to the actual controller step constant
    controller_max_step = UI_TO_CONTROLLER_STEP_MAP.get(max_step_ui)

    if controller_max_step is None:
        logger.error(f"Could not map UI step {max_step_ui} ('{STEP_DESCRIPTIONS.get(max_step_ui)}') to a valid controller step. Exiting.", extra={"icon": "❌"})
        return

    logger.info(f"Running pipeline on {args.input} up to UI step {max_step_ui} ({STEP_DESCRIPTIONS[max_step_ui]}), mapped to controller step {controller_max_step}", extra={"icon": "🚀"})
    try:
        success = process_document(args.input, max_step=controller_max_step, dry_run=False)
        if success:
            logger.info("Pipeline completed successfully.", extra={"icon": "✅"})
            
            # Show deduplication results after successful run
            if args.show_dedup or max_step_ui >= UI_STEP_CHUNKING:
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
        # For the test, we run the complete pipeline, so we use the controller's STEP_EXTRACT_REQS directly.
        controller_max_step_for_test = CTRL_STEP_EXTRACT_REQS
        logger.info(f"Test will run up to controller step: {controller_max_step_for_test} (Requirements Extraction)")
        return process_document(test_file, max_step=controller_max_step_for_test, dry_run=False)
    except Exception as e:
        logger.error(f"Pipeline test error: {e}", extra={"icon": "💥"})
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_with_hardcoded_file()
    else:
        main()
