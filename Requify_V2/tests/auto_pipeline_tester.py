"""
auto_pipeline_tester.py

This script automates running the document processing pipeline with pre-defined responses
to interactive prompts. It allows for testing the full pipeline flow with different
document inputs and configuration options.

The script can:
1. Run the pipeline with a specific document
2. Automatically handle user prompts with predefined responses
3. Reset the database before running
4. Set the pipeline to run up to a specific step
5. Generate test reports and display results
6. Automatically respond to interactive prompts about document similarity

Usage:
    python temp/auto_pipeline_tester.py --doc [document_name] --step [max_step] --mode [test_mode]
"""

import os
import sys
import time
import argparse
import subprocess
import threading
import queue
from typing import Dict, List, Optional

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import setup_logging, get_logger, setup_project_directory, generate_timestamp
from src.pipeline_controller import (
    process_document,
    STEP_HASH_CHECK,
    STEP_PARSE,
    STEP_DEDUP_ONLY,
    STEP_SAVE_TO_DB,
    STEP_CHUNKING,
    STEP_EXTRACT_REQS,
)

# Setup project directory
setup_project_directory()

# Setup logging
logger = get_logger("Auto_Pipeline_Tester")

# Constants
DEFAULT_INPUT_DIR = os.path.join("input", "raw")
TEST_MODES = {
    "default": {
        "description": "Default mode - skip interactive prompts and select first option",
        "auto_choice": "keep_old",
        "auto_update_choice": "replace"
    },
    "detailed": {
        "description": "Detailed analysis mode - perform chunk-level analysis",
        "auto_choice": "detailed",
        "auto_update_choice": "merge"
    },
    "keep_both": {
        "description": "Keep both mode - keep both documents separately",
        "auto_choice": "keep_both",
        "auto_update_choice": "keep"
    },
    "replace": {
        "description": "Replace mode - always replace old with new document",
        "auto_choice": "keep_new",
        "auto_update_choice": "replace"
    }
}

STEP_NAMES = {
    STEP_HASH_CHECK: "Hash check only",
    STEP_PARSE: "Parse PDF",
    STEP_DEDUP_ONLY: "Deduplication check",
    STEP_SAVE_TO_DB: "Save to database",
    STEP_CHUNKING: "Content chunking",
    STEP_EXTRACT_REQS: "Requirements extraction"
}

# Automatically respond to specific interactive prompts
AUTO_RESPONSES = {
    "Enter your choice (1-4):": "4",  # Detailed chunk-level analysis
    "Enter your choice (1-3):": "1",  # Replace old document
    "Confirm replacing": "y",         # Confirm replacement
    "How would you like to handle this update?": "1", # Replace old document
    "Set up database before running?": "y",
    "Continue anyway?": "y",
}

def reset_database() -> bool:
    """Reset the LanceDB database by running the reset_lancedb.py script."""
    logger.info("Resetting database...", extra={"icon": "🔄"})
    try:
        reset_script = os.path.join("tools", "reset_lancedb.py")
        if not os.path.exists(reset_script):
            logger.error(f"Reset script not found at {reset_script}", extra={"icon": "❌"})
            return False
            
        result = subprocess.run([sys.executable, reset_script], capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Database reset failed with code {result.returncode}", extra={"icon": "❌"})
            if result.stderr:
                logger.error(f"Error: {result.stderr}", extra={"icon": "❌"})
            return False
            
        logger.info("Database reset successfully", extra={"icon": "✅"})
        return True
    except Exception as e:
        logger.error(f"Error resetting database: {e}", extra={"icon": "❌"})
        return False

def list_available_documents() -> List[str]:
    """List all available documents in the input directory."""
    if not os.path.exists(DEFAULT_INPUT_DIR):
        logger.error(f"Input directory not found: {DEFAULT_INPUT_DIR}", extra={"icon": "❌"})
        return []
        
    return [f for f in os.listdir(DEFAULT_INPUT_DIR) 
            if os.path.isfile(os.path.join(DEFAULT_INPUT_DIR, f)) 
            and f.lower().endswith(('.pdf', '.docx', '.txt'))]

def print_available_options():
    """Print available documents and test modes."""
    logger.info("=== Available Documents ===", extra={"icon": "📋"})
    docs = list_available_documents()
    for i, doc in enumerate(docs):
        logger.info(f"{i+1}. {doc}", extra={"icon": "📄"})
    
    logger.info("\n=== Available Test Modes ===", extra={"icon": "📋"})
    for mode, details in TEST_MODES.items():
        logger.info(f"{mode}: {details['description']}", extra={"icon": "ℹ️"})
    
    logger.info("\n=== Available Pipeline Steps ===", extra={"icon": "📋"})
    for step, name in STEP_NAMES.items():
        logger.info(f"{step}. {name}", extra={"icon": "🔢"})

def show_deduplication_results(doc_path: str):
    """Show deduplication results for a document."""
    logger.info("Showing deduplication results...", extra={"icon": "📊"})
    try:
        doc_name = os.path.basename(doc_path)
        validation_script = os.path.join("tools", "validation", "show_deduplication_results.py")
        
        if not os.path.exists(validation_script):
            logger.error(f"Validation script not found at {validation_script}", extra={"icon": "❌"})
            return
            
        result = subprocess.run(
            [sys.executable, validation_script, "--relationships", "--text", "--doc", doc_name],
            capture_output=True, text=True
        )
        
        if result.stdout:
            logger.info(result.stdout, extra={"icon": "📊"})
        else:
            logger.info("No deduplication results found", extra={"icon": "ℹ️"})
            
        if result.stderr:
            logger.error(f"Error showing results: {result.stderr}", extra={"icon": "❌"})
    except Exception as e:
        logger.error(f"Error showing deduplication results: {e}", extra={"icon": "❌"})

def reader_thread(proc, q):
    """Thread to read process output and put it in the queue."""
    for line in iter(proc.stdout.readline, b''):
        line_str = line.decode('utf-8', errors='replace').rstrip()
        print(line_str)  # Echo to console
        q.put(line_str)
    proc.stdout.close()

def writer_thread(proc, q, test_mode):
    """Thread to detect prompts and send automatic responses."""
    mode_settings = TEST_MODES.get(test_mode, TEST_MODES["default"])
    
    # Create custom responses based on the test mode
    custom_responses = AUTO_RESPONSES.copy()
    if test_mode == "detailed":
        custom_responses["Enter your choice (1-4):"] = "4"  # Detailed chunk-level analysis
    elif test_mode == "keep_both":
        custom_responses["Enter your choice (1-4):"] = "3"  # Keep both documents
    elif test_mode == "replace":
        custom_responses["Enter your choice (1-4):"] = "2"  # Replace with new document
    
    while True:
        try:
            line = q.get(timeout=0.1)
            
            # Check if this line contains a prompt we should respond to
            for prompt, response in custom_responses.items():
                if prompt in line:
                    logger.info(f"Auto-responding to prompt: '{prompt}' with '{response}'", extra={"icon": "🤖"})
                    proc.stdin.write(f"{response}\n".encode('utf-8'))
                    proc.stdin.flush()
                    break
            
            q.task_done()
        except queue.Empty:
            # Check if process is still running
            if proc.poll() is not None:
                break
        except Exception as e:
            logger.error(f"Error in writer thread: {e}", extra={"icon": "❌"})
            break

def run_pipeline_with_auto_input(doc_path: str, max_step: int, test_mode: str, skip_hash_check: bool = False) -> bool:
    """
    Run the pipeline with automatic responses to interactive prompts.
    
    Args:
        doc_path: Path to the document to process
        max_step: Maximum pipeline step to run
        test_mode: The test mode to use for automatic responses
        skip_hash_check: Whether to skip the hash-based duplication check
        
    Returns:
        True if the pipeline ran successfully, False otherwise
    """
    if not os.path.exists(doc_path):
        logger.error(f"Document not found: {doc_path}", extra={"icon": "❌"})
        return False
    
    # Set environment variables for automatic responses
    mode_settings = TEST_MODES.get(test_mode, TEST_MODES["default"])
    os.environ["REQUIFY_TEST_MODE"] = "true"
    os.environ["REQUIFY_AUTO_CHOICE"] = mode_settings["auto_choice"]
    os.environ["REQUIFY_AUTO_UPDATE_CHOICE"] = mode_settings["auto_update_choice"]
    
    logger.info("=== Pipeline Run Settings ===", extra={"icon": "⚙️"})
    logger.info(f"Document: {os.path.basename(doc_path)}", extra={"icon": "📄"})
    logger.info(f"Maximum step: {max_step} - {STEP_NAMES.get(max_step, 'Unknown')}", extra={"icon": "🔢"})
    logger.info(f"Test mode: {test_mode} - {mode_settings['description']}", extra={"icon": "🔧"})
    logger.info(f"Skip hash check: {skip_hash_check}", extra={"icon": "🔍"})
    
    # Build the command to run the pipeline
    cmd = [
        sys.executable, 
        "pipeline_runner.py",
        "--input", doc_path,
        "--max-step", str(max_step)
    ]
    
    if skip_hash_check:
        cmd.append("--skip_hash_check")
    
    # Run the command with automatic interactive responses
    logger.info("=== Starting Pipeline with Auto-Input ===", extra={"icon": "🚀"})
    start_time = time.time()
    
    try:
        # Start the process with pipes for stdin/stdout/stderr
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=False
        )
        
        # Set up queue and threads for interactive communication
        q = queue.Queue()
        rt = threading.Thread(target=reader_thread, args=(process, q))
        wt = threading.Thread(target=writer_thread, args=(process, q, test_mode))
        
        rt.daemon = True
        wt.daemon = True
        rt.start()
        wt.start()
        
        # Wait for the process to complete
        ret_code = process.wait()
        
        # Wait for threads to finish
        rt.join()
        wt.join()
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        if ret_code == 0:
            logger.info(f"Pipeline completed successfully in {elapsed:.2f} seconds", extra={"icon": "✅"})
            
            # Show deduplication results if we went through chunking or beyond
            if max_step >= STEP_CHUNKING:
                show_deduplication_results(doc_path)
                
            return True
        else:
            logger.error(f"Pipeline failed with code {ret_code} after {elapsed:.2f} seconds", extra={"icon": "❌"})
            return False
    except Exception as e:
        end_time = time.time()
        elapsed = end_time - start_time
        logger.error(f"Pipeline error after {elapsed:.2f} seconds: {e}", extra={"icon": "💥"})
        return False
    finally:
        # Clean up environment variables
        for var in ["REQUIFY_TEST_MODE", "REQUIFY_AUTO_CHOICE", "REQUIFY_AUTO_UPDATE_CHOICE", 
                   "REQUIFY_DETAILED_ANALYSIS", "REQUIFY_SIMILAR_DOC_ID", "REQUIFY_SIMILARITY_SCORE"]:
            if var in os.environ:
                del os.environ[var]

def run_pipeline_with_document(doc_path: str, max_step: int, test_mode: str, skip_hash_check: bool = False) -> bool:
    """
    Run the pipeline with a specific document and test mode.
    
    Args:
        doc_path: Path to the document to process
        max_step: Maximum pipeline step to run
        test_mode: The test mode to use for automatic responses
        skip_hash_check: Whether to skip the hash-based duplication check
        
    Returns:
        True if the pipeline ran successfully, False otherwise
    """
    if not os.path.exists(doc_path):
        logger.error(f"Document not found: {doc_path}", extra={"icon": "❌"})
        return False
    
    # Choose between the automatic input method and the environment variable method
    if max_step >= STEP_SAVE_TO_DB:
        logger.info("Using interactive auto-input method for pipeline with potential prompts", extra={"icon": "🤖"})
        return run_pipeline_with_auto_input(doc_path, max_step, test_mode, skip_hash_check)
    else:
        # Use the simpler environment variable method for non-interactive steps
        # Set environment variables for automatic responses
        mode_settings = TEST_MODES.get(test_mode, TEST_MODES["default"])
        os.environ["REQUIFY_TEST_MODE"] = "true"
        os.environ["REQUIFY_AUTO_CHOICE"] = mode_settings["auto_choice"]
        os.environ["REQUIFY_AUTO_UPDATE_CHOICE"] = mode_settings["auto_update_choice"]
        
        logger.info("=== Pipeline Run Settings ===", extra={"icon": "⚙️"})
        logger.info(f"Document: {os.path.basename(doc_path)}", extra={"icon": "📄"})
        logger.info(f"Maximum step: {max_step} - {STEP_NAMES.get(max_step, 'Unknown')}", extra={"icon": "🔢"})
        logger.info(f"Test mode: {test_mode} - {mode_settings['description']}", extra={"icon": "🔧"})
        logger.info(f"Skip hash check: {skip_hash_check}", extra={"icon": "🔍"})
        
        # Run the pipeline
        logger.info("=== Starting Pipeline ===", extra={"icon": "🚀"})
        start_time = time.time()
        
        try:
            success = process_document(doc_path, max_step=max_step, dry_run=False, skip_hash_check=skip_hash_check)
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            if success:
                logger.info(f"Pipeline completed successfully in {elapsed:.2f} seconds", extra={"icon": "✅"})
                
                # Show deduplication results if we went through chunking or beyond
                if max_step >= STEP_CHUNKING:
                    show_deduplication_results(doc_path)
                    
                return True
            else:
                logger.error(f"Pipeline failed after {elapsed:.2f} seconds", extra={"icon": "❌"})
                return False
        except Exception as e:
            end_time = time.time()
            elapsed = end_time - start_time
            logger.error(f"Pipeline error after {elapsed:.2f} seconds: {e}", extra={"icon": "💥"})
            return False
        finally:
            # Clean up environment variables
            for var in ["REQUIFY_TEST_MODE", "REQUIFY_AUTO_CHOICE", "REQUIFY_AUTO_UPDATE_CHOICE", 
                       "REQUIFY_DETAILED_ANALYSIS", "REQUIFY_SIMILAR_DOC_ID", "REQUIFY_SIMILARITY_SCORE"]:
                if var in os.environ:
                    del os.environ[var]

def run_test_scenario(scenario: Dict) -> bool:
    """
    Run a predefined test scenario with multiple documents.
    
    Args:
        scenario: Dictionary with scenario details including documents to process
        
    Returns:
        True if all steps in the scenario succeeded, False otherwise
    """
    scenario_name = scenario.get("name", "Unnamed scenario")
    steps = scenario.get("steps", [])
    
    logger.info(f"=== Running Test Scenario: {scenario_name} ===", extra={"icon": "🧪"})
    logger.info(f"Steps: {len(steps)}", extra={"icon": "🔢"})
    
    # Reset database before starting
    if not reset_database():
        logger.error("Failed to reset database before scenario", extra={"icon": "❌"})
        return False
    
    success = True
    for i, step in enumerate(steps):
        file_name = step.get("file")
        doc_path = os.path.join(DEFAULT_INPUT_DIR, file_name)
        
        logger.info(f"=== Step {i+1}/{len(steps)}: Processing {file_name} ===", extra={"icon": "🔄"})
        
        # Get expected results
        expected = step.get("expected", {})
        if expected:
            description = expected.get("description", "No description")
            logger.info(f"Expected: {description}", extra={"icon": "🔍"})
        
        # Set up auto-choice based on expectations
        auto_choice = "detailed"  # Default to detailed analysis
        if expected.get("is_duplicate", False):
            auto_choice = "keep_old"
        elif expected.get("is_hash_duplicate", False):
            # For hash duplicates, we don't need to set auto_choice
            pass
        elif expected.get("updated_chunks", False):
            auto_choice = "detailed"
        
        # For first step, we need to skip hash check to force processing
        skip_hash = i == 0
        
        # Run pipeline with this document
        step_success = run_pipeline_with_document(
            doc_path=doc_path,
            max_step=STEP_EXTRACT_REQS,
            test_mode="detailed" if auto_choice == "detailed" else "default",
            skip_hash_check=skip_hash
        )
        
        if not step_success:
            logger.error(f"Step {i+1} failed", extra={"icon": "❌"})
            success = False
            break
            
        logger.info(f"Step {i+1} completed successfully", extra={"icon": "✅"})
    
    if success:
        logger.info(f"Scenario '{scenario_name}' completed successfully", extra={"icon": "✅"})
    else:
        logger.error(f"Scenario '{scenario_name}' failed", extra={"icon": "❌"})
    
    return success

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Automated pipeline tester")
    parser.add_argument("--doc", type=str, help="Document name to process (from input/raw)")
    parser.add_argument("--step", type=int, default=STEP_EXTRACT_REQS,
                       choices=[STEP_HASH_CHECK, STEP_PARSE, STEP_DEDUP_ONLY, 
                               STEP_SAVE_TO_DB, STEP_CHUNKING, STEP_EXTRACT_REQS],
                       help="Maximum pipeline step to run")
    parser.add_argument("--mode", type=str, default="default",
                       choices=list(TEST_MODES.keys()),
                       help="Test mode for handling interactive prompts")
    parser.add_argument("--skip-hash", action="store_true",
                       help="Skip hash-based duplicate check")
    parser.add_argument("--reset-db", action="store_true",
                       help="Reset database before running")
    parser.add_argument("--scenario", type=int, help="Run a predefined test scenario")
    parser.add_argument("--list", action="store_true",
                       help="List available documents and options")
    parser.add_argument("--auto-input", action="store_true",
                       help="Always use auto-input method for interactive prompts")
    
    args = parser.parse_args()
    
    # Show available options if requested
    if args.list:
        print_available_options()
        return 0
    
    # Run a predefined scenario if requested
    if args.scenario is not None:
        # Import test scenarios from the E2E tests
        try:
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tests', 'e2e')))
            from test_scenarios import TEST_SCENARIOS
            
            # Find the requested scenario
            scenario = None
            for s in TEST_SCENARIOS:
                if s.get("id") == args.scenario:
                    scenario = s
                    break
            
            if not scenario:
                logger.error(f"Scenario {args.scenario} not found", extra={"icon": "❌"})
                return 1
                
            # Run the scenario
            success = run_test_scenario(scenario)
            return 0 if success else 1
            
        except ImportError:
            logger.error("Failed to import test scenarios", extra={"icon": "❌"})
            return 1
    
    # Reset database if requested
    if args.reset_db:
        if not reset_database():
            return 1
    
    # If no document specified, show usage
    if not args.doc:
        logger.error("No document specified. Use --doc to specify a document name or --list to see available options.", extra={"icon": "❌"})
        return 1
    
    # Construct document path
    doc_path = args.doc
    if not os.path.exists(doc_path):
        # Try looking in the input/raw directory
        alt_path = os.path.join(DEFAULT_INPUT_DIR, args.doc)
        if os.path.exists(alt_path):
            doc_path = alt_path
        else:
            logger.error(f"Document not found: {args.doc}", extra={"icon": "❌"})
            return 1
    
    # Run the pipeline with the specified document
    if args.auto_input:
        success = run_pipeline_with_auto_input(
            doc_path=doc_path,
            max_step=args.step,
            test_mode=args.mode,
            skip_hash_check=args.skip_hash
        )
    else:
        success = run_pipeline_with_document(
            doc_path=doc_path,
            max_step=args.step,
            test_mode=args.mode,
            skip_hash_check=args.skip_hash
        )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 