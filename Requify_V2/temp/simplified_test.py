#!/usr/bin/env python3
"""
simplified_test.py

A simple script to test the modified pipeline controller using subprocess calls.
"""

import os
import sys
import shutil
import subprocess
import time

# Constants
OUTPUT_DIR = "output"
LANCEDB_DIR = os.path.join(OUTPUT_DIR, "lancedb")
DOC1 = "input/raw/fighter_jet_rocket_launcher_spec_2.pdf"
DOC2 = "input/raw/fighter_jet_rocket_launcher_spec_2_changed_values.pdf"

def reset_database():
    """Reset the LanceDB database by removing and recreating the directory."""
    print(f"Removing existing database at {LANCEDB_DIR}")
    if os.path.exists(LANCEDB_DIR):
        shutil.rmtree(LANCEDB_DIR)
    
    os.makedirs(LANCEDB_DIR, exist_ok=True)
    print(f"Created empty database directory at {LANCEDB_DIR}")

def run_command(cmd, env=None):
    """Run a command and print its output."""
    print(f"Running command: {cmd}")
    
    # Use provided environment or current environment
    if env is None:
        env = os.environ.copy()
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    
    print("OUTPUT:")
    print(result.stdout)
    
    if result.stderr:
        print("ERROR:")
        print(result.stderr)
    
    return result.returncode == 0

def main():
    # 1. Check if documents exist
    for doc in [DOC1, DOC2]:
        if not os.path.exists(doc):
            print(f"ERROR: Test document not found: {doc}")
            return False
    
    # 2. Reset database and process first document
    print("\n===== TEST PART 1: FIRST DOCUMENT =====")
    reset_database()
    
    # Run with modified PYTHONPATH to avoid import issues
    cmd1 = f"PYTHONPATH=. python src/pipeline_controller.py --doc_path={DOC1} --max_step=5 --skip_hash_check"
    if not run_command(cmd1):
        print("First document processing failed")
        return False
    
    # 3. Process second document without resetting
    print("\n===== TEST PART 2: SECOND DOCUMENT =====")
    time.sleep(2)  # Brief pause between runs
    
    # Set environment variables for automatic selection
    env = os.environ.copy()
    env["REQUIFY_AUTO_CHOICE"] = "detailed"
    env["REQUIFY_AUTO_SELECT_NEW"] = "true"  # Auto-select "keep_new" for chunks
    
    cmd2 = f"PYTHONPATH=. python src/pipeline_controller.py --doc_path={DOC2} --max_step=5 --skip_hash_check"
    if not run_command(cmd2, env):
        print("Second document processing failed")
        return False
    
    print("\n===== TEST COMPLETED SUCCESSFULLY =====")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 