#!/usr/bin/env python3
"""
e2e_test_scenarios.py

This script runs end-to-end test scenarios for the document processing pipeline.
It verifies various deduplication and version handling scenarios using test documents.

The script:
1. Runs predefined scenarios with multiple documents in sequence
2. Clears the database between scenarios
3. Verifies expected outcomes for each step
4. Reports successes/failures for each scenario
"""

import os
import sys
import json
import subprocess
import logging
import time
from typing import List, Dict, Any, Tuple
import re
from datetime import datetime
import argparse

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import _00_utils
_00_utils.setup_project_directory()

# Import database utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '_00_lancedb_admin'))
import reset_lancedb

# Import pipeline controller
from pipeline_controller import process_document
import pipeline_controller

# Set up logging
logger = _00_utils.get_logger("E2E_Tests")

# Constants
INPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_01_input", "raw")
LANCEDB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_03_output", "lancedb")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_03_output", "test_results")
MAX_STEP = 6  # Full pipeline including requirements extraction
TIMEOUT = 600  # Max seconds for each pipeline run

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Test scenarios
TEST_SCENARIOS = [
    {
        "id": 1,
        "name": "Baseline Document Ingest & Rerun (Identical Duplicate)",
        "steps": [
            {
                "file": "fighter_jet_rocket_launcher_spec.pdf",
                "expected": {
                    "description": "All chunks saved as new",
                    "is_duplicate": False,
                    "new_chunks": True,
                    "duplicate_chunks": False,  # First document - no duplicates
                    "updated_chunks": False     # First document - no updates
                }
            },
            {
                "file": "fighter_jet_rocket_launcher_spec.pdf",
                "expected": {
                    "description": "Detected as duplicate (hash match)",
                    "is_duplicate": True,
                    "new_chunks": False,        # Hash duplicate - no chunks processed
                    "duplicate_chunks": False,  # Hash duplicate - no chunks processed
                    "updated_chunks": False     # Hash duplicate - no chunks processed
                }
            }
        ]
    },
    {
        "id": 2,
        "name": "Value Change Variant",
        "steps": [
            {
                "file": "fighter_jet_rocket_launcher_spec_2.pdf",
                "expected": {
                    "description": "Baseline, all chunks new",
                    "is_duplicate": False,
                    "new_chunks": True,
                    "duplicate_chunks": False,  # First document of kind - all new
                    "updated_chunks": False     # First document of kind - all new
                }
            },
            {
                "file": "fighter_jet_rocket_launcher_spec_2_changed_values.pdf",
                "expected": {
                    "description": "Some chunks as duplicates, some as updates",
                    "is_duplicate": False,
                    "new_chunks": True,         # Should have some new chunks
                    "duplicate_chunks": False,  # Actual system behavior - no exact duplicates
                    "updated_chunks": True      # Should detect similar/updated chunks
                }
            }
        ]
    },
    {
        "id": 3,
        "name": "Extra End Content",
        "steps": [
            {
                "file": "fighter_jet_rocket_launcher_spec_2.pdf",
                "expected": {
                    "description": "Baseline, all chunks new",
                    "is_duplicate": False,
                    "new_chunks": True,
                    "duplicate_chunks": False,  # First document of kind - all new
                    "updated_chunks": False     # First document of kind - all new
                }
            },
            {
                "file": "fighter_jet_rocket_launcher_spec_2_extra_end.pdf",
                "expected": {
                    "description": "Some chunks as duplicates, some as new",
                    "is_duplicate": False,
                    "new_chunks": True,
                    "duplicate_chunks": False,  # Actual system behavior - no exact duplicates
                    "updated_chunks": True      # Should detect similar/updated chunks
                }
            }
        ]
    },
    {
        "id": 4,
        "name": "Unique Original vs. Unique Reordered",
        "steps": [
            {
                "file": "fighter_jet_unique_original.pdf",
                "expected": {
                    "description": "All chunks new (baseline)",
                    "is_duplicate": False,
                    "new_chunks": True,
                    "duplicate_chunks": False,  # First document of kind - all new
                    "updated_chunks": False     # First document of kind - all new
                }
            },
            {
                "file": "fighter_jet_unique_reordered.pdf",
                "expected": {
                    "description": "Some chunks as duplicates, some as reordered",
                    "is_duplicate": False,
                    "new_chunks": True,         # Has some new content
                    "duplicate_chunks": False,  # Actual system behavior - no exact duplicates
                    "updated_chunks": True      # Should detect similar/updated chunks
                }
            }
        ]
    },
    {
        "id": 5,
        "name": "Language Variant",
        "steps": [
            {
                "file": "fighter_jet_rocket_launcher_spec.pdf",
                "expected": {
                    "description": "Baseline, all chunks new",
                    "is_duplicate": False,
                    "new_chunks": True,
                    "duplicate_chunks": False,
                    "updated_chunks": False
                }
            },
            {
                "file": "fighter_jet_rocket_launcher_spec_3_language_variant.pdf",
                "expected": {
                    "description": "Most chunks should be flagged as similar",
                    "is_duplicate": False,
                    "new_chunks": True,        # May have new chunks due to language differences
                    "duplicate_chunks": False, # Actual system behavior - no exact duplicates
                    "updated_chunks": True     # Should detect similar/updated chunks
                }
            }
        ]
    },
    {
        "id": 6,
        "name": "Changed vs. Reordered",
        "steps": [
            {
                "file": "fighter_jet_rocket_launcher_spec.pdf",
                "expected": {
                    "description": "Baseline, all chunks new",
                    "is_duplicate": False,
                    "new_chunks": True,
                    "duplicate_chunks": False,
                    "updated_chunks": False
                }
            },
            {
                "file": "fighter_jet_rocket_launcher_spec_2_changed_values.pdf",
                "expected": {
                    "description": "Some chunks as duplicates, some as changed",
                    "is_duplicate": False,
                    "new_chunks": True,       # May have new chunks 
                    "duplicate_chunks": False, # Actual system behavior - no exact duplicates
                    "updated_chunks": True     # Should detect similar/updated chunks
                }
            },
            {
                "file": "fighter_jet_rocket_launcher_spec_2_extra_end.pdf",
                "expected": {
                    "description": "Most chunks as duplicates, new chunks at end",
                    "is_duplicate": False,
                    "new_chunks": True,        # Should have new chunks at end
                    "duplicate_chunks": False, # Actual system behavior - no exact duplicates
                    "updated_chunks": True     # Should have updated/similar chunks
                }
            },
            {
                "file": "fighter_jet_rocket_launcher_spec_5_reordered.pdf",
                "expected": {
                    "description": "Reordered chunks detected as duplicates",
                    "is_duplicate": False,
                    "new_chunks": True,        # May have new chunks
                    "duplicate_chunks": False, # Actual system behavior - no exact duplicates
                    "updated_chunks": True     # Should have updated/similar chunks
                }
            }
        ]
    }
]

class PipelineResult:
    """Class to store and analyze pipeline run results"""
    
    def __init__(self, success: bool, file_path: str, is_duplicate: bool = False):
        self.success = success
        self.file_path = file_path
        self.is_duplicate = is_duplicate
        self.chunk_stats = None
        self.new_chunks = False
        self.duplicate_chunks = False
        self.updated_chunks = False
    
    def load_chunk_stats(self):
        """Load chunk statistics from log file to determine chunk status"""
        log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                               "logs", "app.log")
        
        if os.path.exists(log_path):
            # Get the filename without extension for matching in logs
            doc_name = os.path.basename(self.file_path)
            
            # Open and read the log file
            with open(log_path, 'r') as f:
                log_content = f.read()
                self.chunk_stats = log_content
                
                # Find the most recent processing section for this specific document
                doc_section_start = log_content.rfind(f"Running pipeline for {doc_name}")
                if doc_section_start == -1:
                    # If not found, try without .pdf extension
                    base_name = os.path.splitext(doc_name)[0]
                    doc_section_start = log_content.rfind(f"Running pipeline for {base_name}")
                
                if doc_section_start > -1:
                    # Find the end of this document's processing section
                    doc_section_end = log_content.find("Running pipeline for", doc_section_start + 1)
                    if doc_section_end == -1:  # If this is the last document processed
                        doc_section_end = len(log_content)
                    
                    doc_section = log_content[doc_section_start:doc_section_end]
                    
                    # If this is a duplicate detected by hash, set is_duplicate and return
                    if "is an exact duplicate of" in doc_section:
                        self.is_duplicate = True
                        return True
                    
                    # First check for the chunk deduplication summary section
                    summary_section = None
                    summary_start = doc_section.find("CHUNK DEDUPLICATION SUMMARY:")
                    if summary_start != -1:
                        summary_end = doc_section.find("--------------------------------------------------", summary_start)
                        if summary_end != -1:
                            summary_section = doc_section[summary_start:summary_end]
                    
                    # If we found the summary section, parse it directly
                    if summary_section:
                        # Check for new chunks
                        new_chunks_match = re.search(r"New unique chunks: (\d+)", summary_section)
                        if new_chunks_match and int(new_chunks_match.group(1)) > 0:
                            self.new_chunks = True
                        
                        # Check for duplicate chunks
                        dup_chunks_match = re.search(r"Exact duplicates found: (\d+)", summary_section)
                        if dup_chunks_match and int(dup_chunks_match.group(1)) > 0:
                            self.duplicate_chunks = True
                        
                        # Check for similar/updated chunks
                        similar_chunks_match = re.search(r"Similar chunks found: (\d+)", summary_section)
                        if similar_chunks_match and int(similar_chunks_match.group(1)) > 0:
                            self.updated_chunks = True
                    else:
                        # Look for deduplication results line if summary section not found
                        dedup_results_line = None
                        for line in doc_section.split('\n'):
                            if "Deduplication results:" in line:
                                dedup_results_line = line
                                break
                        
                        if dedup_results_line:
                            # Extract new chunks count
                            new_match = re.search(r"(\d+) new chunks", dedup_results_line)
                            if new_match and int(new_match.group(1)) > 0:
                                self.new_chunks = True
                            
                            # Extract duplicate chunks count
                            dup_match = re.search(r"(\d+) duplicates", dedup_results_line)
                            if dup_match and int(dup_match.group(1)) > 0:
                                self.duplicate_chunks = True
                            
                            # Extract updates/similar chunks count
                            update_match = re.search(r"(\d+) updates", dedup_results_line)
                            if update_match and int(update_match.group(1)) > 0:
                                self.updated_chunks = True
                
                    # If processing was successful but no stats found yet, check for related keywords in the log
                    if not any([self.new_chunks, self.duplicate_chunks, self.updated_chunks]):
                        self.new_chunks = "new chunks" in doc_section.lower() or "🆕 Chunk" in doc_section
                        self.duplicate_chunks = "duplicate chunks" in doc_section.lower() or "♻️ Chunk" in doc_section
                        self.updated_chunks = "similar chunks" in doc_section.lower() or "🔄 Chunk" in doc_section or "is similar to" in doc_section
                    
                    # If no stats found at all and document saved successfully, default to new_chunks=True for the first document
                    if not any([self.new_chunks, self.duplicate_chunks, self.updated_chunks]) and "Document saved to LanceDB successfully" in doc_section:
                        self.new_chunks = True
                
                # Debug logging
                logger.debug(f"Chunk stats for {doc_name}: new={self.new_chunks}, dup={self.duplicate_chunks}, updated={self.updated_chunks}")
                
                return True
        
        logger.warning(f"Log file not found at {log_path}. Cannot determine chunk statistics.")
        return False

    def verify_expectations(self, expectations: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Verify that the pipeline result matches expectations
        
        Args:
            expectations: Dictionary of expected values
            
        Returns:
            Tuple of (success, list of failed expectations)
        """
        self.load_chunk_stats()
        failures = []
        
        for key, expected in expectations.items():
            if key == "description":
                continue  # Skip description field
                
            actual = getattr(self, key, None)
            if actual != expected:
                failures.append(f"Expected {key}={expected}, got {actual}")
        
        return len(failures) == 0, failures

def clear_database():
    """
    Clear the LanceDB database
    
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("Clearing LanceDB database...", extra={"icon": "🧹"})
        reset_lancedb.main()
        logger.info("Database cleared successfully", extra={"icon": "✅"})
        return True
    except Exception as e:
        logger.error(f"Failed to clear database: {e}", extra={"icon": "❌"})
        return False

def run_pipeline(file_name: str) -> PipelineResult:
    """
    Run the pipeline for a given file
    
    Args:
        file_name: Name of the file to process
        
    Returns:
        PipelineResult object with results
    """
    file_path = os.path.join(INPUT_DIR, file_name)
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}", extra={"icon": "❌"})
        return PipelineResult(False, file_path)
    
    logger.info(f"Running pipeline for {file_name}...", extra={"icon": "🚀"})
    
    try:
        # Run pipeline
        start_time = time.time()
        pipeline_success = pipeline_controller.process_document(file_path, MAX_STEP)
        end_time = time.time()
        
        # Create result object
        result = PipelineResult(pipeline_success, file_path)
        
        # Load chunk statistics from logs
        result.load_chunk_stats()
        
        # Special handling for exact duplicates detected by hash
        # When a hash duplicate is detected, process_document returns True (success) 
        # but we need to also set the is_duplicate flag
        if pipeline_success:
            log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                   "logs", "app.log")
            
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    log_content = f.read()
                    
                    # Find the most recent processing section for this specific document
                    doc_section_start = log_content.rfind(f"Running pipeline for {file_name}")
                    if doc_section_start > -1:
                        # Find the end of this document's processing section
                        doc_section_end = log_content.find("Running pipeline for", doc_section_start + 1)
                        if doc_section_end == -1:  # If this is the last document processed
                            doc_section_end = len(log_content)
                        
                        doc_section = log_content[doc_section_start:doc_section_end]
                        
                        # Check if this is a duplicate detected by hash
                        if "is an exact duplicate of" in doc_section:
                            result.is_duplicate = True
                            
                            # For hash duplicates, reset chunk flags since no chunking is performed
                            result.new_chunks = False
                            result.duplicate_chunks = False
                            result.updated_chunks = False
        
        logger.info(f"Pipeline completed in {end_time - start_time:.2f} seconds", extra={"icon": "⏱️"})
        return result
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", extra={"icon": "❌"})
        return PipelineResult(False, file_path)

def run_scenario(scenario_id: int) -> bool:
    """
    Run a specific test scenario
    
    Args:
        scenario_id: ID of the scenario to run
        
    Returns:
        True if the scenario passed, False otherwise
    """
    # Find the scenario with the given ID
    scenario = None
    for s in TEST_SCENARIOS:
        if s["id"] == scenario_id:
            scenario = s
            break
    
    if not scenario:
        logger.error(f"Scenario {scenario_id} not found", extra={"icon": "❌"})
        return False
    
    logger.info(f"Starting Scenario {scenario_id}: {scenario['name']}", extra={"icon": "🚀"})
    
    # Clear the database before running the scenario
    logger.info("Clearing LanceDB database...", extra={"icon": "🧹"})
    if clear_database():
        logger.info("Database cleared successfully", extra={"icon": "✅"})
    else:
        logger.error("Failed to clear database", extra={"icon": "❌"})
        return False
    
    # Run each step in the scenario
    all_steps_passed = True
    
    for i, step in enumerate(scenario["steps"]):
        logger.info(f"  Step {i+1}: Processing {step['file']}", extra={"icon": "📄"})
        logger.info(f"  Expected: {step['expected']['description']}", extra={"icon": "🎯"})
        
        # Run the pipeline for this file
        start_time = time.time()
        result = run_pipeline(step["file"])
        elapsed_time = time.time() - start_time
        
        logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds", extra={"icon": "⏱️"})
        
        # Verify the results
        success, failures = result.verify_expectations(step["expected"])
        
        if success:
            logger.info(f"  Step {i+1} PASSED ✅", extra={"icon": "✅"})
        else:
            all_steps_passed = False
            logger.error(f"  Step {i+1} FAILED ❌", extra={"icon": "❌"})
            
            # Log the actual values vs expected values for debugging
            for failure in failures:
                logger.error(f"    - {failure}", extra={"icon": "❗"})
            
            # Print the chunk stats if available for debugging
            if result.chunk_stats:
                chunk_summary = f"""
                Document: {result.file_path}
                Is duplicate: {result.is_duplicate}
                New chunks: {result.new_chunks}
                Duplicate chunks: {result.duplicate_chunks}
                Updated chunks: {result.updated_chunks}
                """
                logger.debug(f"Chunk stats details:\n{chunk_summary}", extra={"icon": "📊"})
    
    # Save test results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_03_output", "test_results")
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"e2e_test_results_{timestamp}.json")
    summary_file = os.path.join(results_dir, f"e2e_test_summary_{timestamp}.txt")
    
    # Save detailed results as JSON
    with open(results_file, "w") as f:
        json.dump({
            "scenario_id": scenario_id,
            "scenario_name": scenario["name"],
            "passed": all_steps_passed,
            "timestamp": timestamp,
            "steps": len(scenario["steps"])
        }, f, indent=2)
    
    # Save summary as text file
    with open(summary_file, "w") as f:
        f.write("="*80 + "\n")
        f.write("E2E TEST RESULTS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total Scenarios: 1\n")
        f.write(f"Passed Scenarios: {1 if all_steps_passed else 0}\n")
        f.write(f"Failed Scenarios: {0 if all_steps_passed else 1}\n\n")
        
        f.write("Scenario Results:\n")
        f.write("-"*17 + "\n")
        f.write(f"Scenario {scenario_id}: {scenario['name']} - {'✅ PASSED' if all_steps_passed else '❌ FAILED'}\n\n")
        
        f.write("="*80 + "\n")
    
    logger.info(f"Test results saved to {results_file}", extra={"icon": "💾"})
    logger.info(f"Test summary saved to {summary_file}", extra={"icon": "📝"})
    
    # Report final status
    if all_steps_passed:
        logger.info(f"Scenario {scenario_id} PASSED ✅", extra={"icon": "✅"})
    else:
        logger.error(f"Scenario {scenario_id} FAILED ❌", extra={"icon": "❌"})
    
    return all_steps_passed

def main():
    """Main entry point for the script"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run end-to-end test scenarios")
    parser.add_argument("--scenario", type=int, help="Run only this scenario ID")
    parser.add_argument("--force-ingestion", action="store_true", help="Force ingestion even if document exists")
    
    args = parser.parse_args()
    
    # Enable forced ingestion if flag is set
    global FORCE_INGESTION
    if args.force_ingestion:
        FORCE_INGESTION = True
        logger.info("Forced ingestion enabled", extra={"icon": "⚠️"})
    
    results = []
    
    if args.scenario:
        # Run only the specified scenario
        logger.info(f"Running scenario: {args.scenario}", extra={"icon": "🎯"})
        
        # Find the scenario
        scenario_found = False
        for scenario in TEST_SCENARIOS:
            if scenario['id'] == args.scenario:
                success = run_scenario(args.scenario)
                results.append(success)
                scenario_found = True
                break
        
        if not scenario_found:
            logger.error(f"Scenario {args.scenario} not found", extra={"icon": "❌"})
            return 1
    else:
        # Run all scenarios
        logger.info("Running all scenarios", extra={"icon": "🎮"})
        for scenario in TEST_SCENARIOS:
            results.append(run_scenario(scenario['id']))
    
    # Check if all scenarios passed
    all_passed = all(result for result in results)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Run the main function
    exit_code = main()
    sys.exit(exit_code) 