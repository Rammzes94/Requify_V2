#!/usr/bin/env python3
"""
run_tests.py

A simple driver script for running end-to-end test scenarios for the document processing pipeline.
This script makes it easy to run specific scenarios or groups of scenarios.
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import utilities if needed (commenting out as we don't appear to use it directly in this file)
# from src.utils import setup_logging, get_logger, update_token_counters, get_token_usage, print_token_usage, reset_token_counters, setup_project_directory, generate_timestamp

# Constants
TEST_FILES_DIR = os.path.join("input", "raw")
RESULTS_DIR = os.path.join("output", "test_results")

def check_test_files():
    """Check if test files exist before running tests"""
    # Check if required test files exist
    files_to_check = [
        "fighter_jet_rocket_launcher_spec.pdf",
        "fighter_jet_rocket_launcher_spec_2.pdf",
        "fighter_jet_rocket_launcher_spec_2_changed_values.pdf",
        "fighter_jet_rocket_launcher_spec_2_extra_end.pdf",
        "fighter_jet_unique_original.pdf",
        "fighter_jet_unique_reordered.pdf",
        "fighter_jet_rocket_launcher_spec_3_language_variant.pdf",
        "fighter_jet_rocket_launcher_spec_5_reordered.pdf"
    ]
    
    missing_files = []
    for file_name in files_to_check:
        if not os.path.exists(os.path.join(TEST_FILES_DIR, file_name)):
            missing_files.append(file_name)
    
    if missing_files:
        print(f"Missing test files: {', '.join(missing_files)}")
        return False
    
    return True

def print_scenarios():
    """Print the available test scenarios"""
    print("\nAvailable Test Scenarios:")
    print("========================")
    print("1: Baseline Document Ingest & Rerun (Identical Duplicate)")
    print("2: Value Change Variant")
    print("3: Extra End Content")
    print("4: Unique Original vs. Unique Reordered")
    print("5: Language Variant")
    print("6: Changed vs. Reordered (Combined Test)")
    print("all: Run all scenarios")
    print("subset: Run a predefined subset of tests (quick test)")

def run_scenario(scenario_id, generate_report=True):
    """Run a specific test scenario"""
    print(f"\nRunning scenario {scenario_id}...")
    cmd = [sys.executable, "tests/e2e/test_scenarios.py", "--scenario", str(scenario_id)]
    result = subprocess.run(cmd)
    success = result.returncode == 0
    
    if generate_report and success:
        generate_html_report()
    
    return success

def run_all_scenarios(generate_report=True):
    """Run all test scenarios"""
    print("\nRunning all scenarios...")
    cmd = [sys.executable, "tests/e2e/test_scenarios.py"]
    result = subprocess.run(cmd)
    success = result.returncode == 0
    
    if generate_report and success:
        generate_html_report()
    
    return success

def run_subset(generate_report=True):
    """Run a predefined subset of scenarios (quick test)"""
    print("\nRunning subset of scenarios (quick test)...")
    success = True
    
    # Define the subset to run (basic scenarios that run faster)
    subset = [1, 2, 4]  # Basic duplicate, changed values, reordered content
    
    for scenario_id in subset:
        print(f"\nRunning scenario {scenario_id}...")
        cmd = [sys.executable, "tests/e2e/test_scenarios.py", "--scenario", str(scenario_id)]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            success = False
    
    if generate_report and success:
        generate_html_report()
    
    return success

def generate_html_report():
    """Generate HTML report from the latest test results"""
    print("\nGenerating HTML report...")
    try:
        # Check if reporter exists
        if os.path.exists("tools/test_utils/results_reporter.py"):
            cmd = [sys.executable, "tools/test_utils/results_reporter.py"]
            subprocess.run(cmd)
            print("✅ HTML report generated")
            return True
        else:
            print("❌ HTML report generator not found")
            return False
    except Exception as e:
        print(f"❌ Error generating HTML report: {e}")
        return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Run E2E test scenarios for document processing pipeline')
    parser.add_argument('scenario', nargs='?', default='list', 
                        help='Scenario ID to run, "all" for all scenarios, "subset" for a quick test, or "list" to list scenarios')
    parser.add_argument('--no-report', action='store_true', help='Skip HTML report generation')
    
    args = parser.parse_args()
    
    if args.scenario == 'list':
        print_scenarios()
        return 0
    
    # Check if test files exist
    if not check_test_files():
        print("\n❌ Missing test files. Please add the required files before running tests.")
        return 1
    
    start_time = time.time()
    
    if args.scenario == 'all':
        success = run_all_scenarios(not args.no_report)
    elif args.scenario == 'subset':
        success = run_subset(not args.no_report)
    elif args.scenario == 'report':
        # Just generate a report from the latest run
        success = generate_html_report()
    else:
        try:
            scenario_id = int(args.scenario)
            success = run_scenario(scenario_id, not args.no_report)
        except ValueError:
            print(f"Error: Invalid scenario ID: {args.scenario}")
            print_scenarios()
            return 1
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"\nTest run completed in {elapsed:.2f} seconds")
    
    if success:
        print("\n✅ All tests PASSED")
        return 0
    else:
        print("\n❌ Some tests FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 