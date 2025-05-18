#!/usr/bin/env python3
"""
pipeline_runner.py

This script provides a user-friendly interface for running the document processing pipeline.
It allows users to:
1. Select which steps of the pipeline to run
2. Set up the database before running the pipeline
3. See real-time progress and results

The script is designed for ease of use while providing control over the pipeline's
execution, making it suitable for both testing and production use.
"""

import os
import sys
import argparse
import logging
from typing import List, Optional

# Add the parent directory and src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # Project root
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "_02_src"))

# Import utility functions
from _02_src._00_utils import setup_logging
import _02_src._00_utils as _00_utils
_00_utils.setup_project_directory()

# Import main pipeline controller
from _02_src.pipeline_controller import (
    process_document,
    STEP_HASH_CHECK,
    STEP_PARSE,
    STEP_DEDUP_ONLY,
    STEP_SAVE_TO_DB,
    STEP_CHUNKING,
    STEP_EXTRACT_REQS,
)

# Set up logging
logger = setup_logging()

# Pipeline step descriptions
STEP_DESCRIPTIONS = {
    STEP_HASH_CHECK: "Hash-based duplicate check only",
    STEP_PARSE: "Parse document only",
    STEP_DEDUP_ONLY: "Parse + Check for duplicates (no DB save)",
    STEP_SAVE_TO_DB: "Parse + Save to LanceDB (no chunking)",
    STEP_CHUNKING: "Parse + Save to LanceDB + Chunk document",
    STEP_EXTRACT_REQS: "Complete pipeline (extract requirements)"
}

def setup_database() -> bool:
    """Set up the LanceDB database tables."""
    try:
        from _02_src._00_lancedb_admin.init_lancedb import main as setup_db_main
        print("\n=== Database Setup ===")
        print("Setting up database tables...")
        setup_db_main()
        print("Database setup completed.")
        return True
    except ImportError as e:
        print(f"Error: Could not import database setup module: {e}")
        return False
    except Exception as e:
        print(f"Error setting up database: {e}")
        return False

def list_input_files() -> List[str]:
    """List available input files."""
    input_dir = os.path.join("_01_input", "raw")
    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        return []
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
             if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('.pdf', '.docx', '.txt'))]
    return sorted(files)

def print_available_steps() -> None:
    """Print available pipeline steps."""
    print("\n=== Available Pipeline Steps ===")
    for step_num, description in STEP_DESCRIPTIONS.items():
        print(f"{step_num}. {description}")

def select_max_step() -> int:
    """Prompt the user to select the maximum pipeline step to run."""
    print_available_steps()
    while True:
        try:
            step = int(input("\nEnter the step number to run up to: "))
            if step in STEP_DESCRIPTIONS:
                return step
            else:
                print(f"Invalid step number. Please enter a number between {min(STEP_DESCRIPTIONS.keys())} and {max(STEP_DESCRIPTIONS.keys())}.")
        except ValueError:
            print("Please enter a valid number.")

def select_input_file() -> Optional[str]:
    """Prompt the user to select an input file."""
    files = list_input_files()
    if not files:
        print("No input files found. Please add files to the _01_input/raw directory.")
        return None
    print("\n=== Available Input Files ===")
    for i, file_path in enumerate(files):
        print(f"{i+1}. {os.path.basename(file_path)}")
    while True:
        try:
            selection = int(input("\nEnter the file number to process (or 0 to provide a custom path): "))
            if selection == 0:
                custom_path = input("Enter the full path to the input file: ")
                if os.path.exists(custom_path):
                    return custom_path
                else:
                    print(f"File not found: {custom_path}")
            elif 1 <= selection <= len(files):
                return files[selection-1]
            else:
                print(f"Invalid selection. Please enter a number between 0 and {len(files)}.")
        except ValueError:
            print("Please enter a valid number.")

def run_pipeline_interactive() -> None:
    """Run the pipeline with interactive prompts."""
    print("\n=== Document Processing Pipeline Runner ===")
    if input("Set up database before running? (y/n): ").lower() == 'y':
        if not setup_database():
            if input("Continue anyway? (y/n): ").lower() != 'y':
                print("Exiting.")
                return
    input_file = select_input_file()
    if not input_file:
        print("No input file selected. Exiting.")
        return
    max_step = select_max_step()
    print("\n=== Pipeline Run Settings ===")
    print(f"Input file: {input_file}")
    print(f"Maximum step: {max_step} - {STEP_DESCRIPTIONS[max_step]}")
    print("\n=== Starting Pipeline ===")
    try:
        success = process_document(input_file, max_step=max_step, dry_run=False)
        if success:
            print("\n=== Pipeline Completed Successfully ===")
        else:
            print("\n=== Pipeline Failed ===")
            print("Check the logs for more information.")
    except Exception as e:
        print(f"\n=== Pipeline Error ===\n{e}")
        raise

def main() -> None:
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Document Processing Pipeline Runner")
    parser.add_argument("--input", type=str, help="Path to the input file")
    parser.add_argument("--max-step", type=int, choices=list(STEP_DESCRIPTIONS.keys()),
                        help="Maximum pipeline step to run")
    parser.add_argument("--setup-db", action="store_true", help="Set up database before running")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    args = parser.parse_args()
    if args.interactive or not (args.input and args.max_step is not None):
        run_pipeline_interactive()
        return
    if args.setup_db:
        if not setup_database():
            print("Database setup failed. Exiting.")
            return
    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        return
    print(f"Running pipeline on {args.input} up to step {args.max_step} ({STEP_DESCRIPTIONS[args.max_step]})")
    try:
        success = process_document(args.input, max_step=args.max_step, dry_run=False)
        if success:
            print("Pipeline completed successfully.")
        else:
            print("Pipeline failed. Check the logs for more information.")
    except Exception as e:
        print(f"Pipeline error: {e}")
        raise

def test_with_hardcoded_file():
    """Test the pipeline with a hardcoded file."""
    print("\n=== Running Pipeline Test with Hardcoded File ===")
    test_file = "_01_input/raw/fighter_jet_rocket_launcher_spec_2.pdf"
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return False
    print(f"Running pipeline on test file: {test_file}")
    try:
        max_step = STEP_EXTRACT_REQS  # Run the complete pipeline
        success = process_document(test_file, max_step=max_step, dry_run=False)
        if success:
            print("\n=== Pipeline Test Completed Successfully ===")
        else:
            print("\n=== Pipeline Test Failed ===")
        return success
    except Exception as e:
        print(f"\n=== Pipeline Test Error ===\n{e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_with_hardcoded_file()
    else:
        main()
