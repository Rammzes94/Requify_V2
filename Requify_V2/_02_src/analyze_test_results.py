#!/usr/bin/env python3
"""
analyze_test_results.py

This script analyzes test results from the logs and provides a detailed summary
of what happened during each test run. It helps diagnose issues with the test
framework by extracting information about chunk processing and decisions.
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Set

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import _00_utils
_00_utils.setup_project_directory()

# Set up logging
logger = _00_utils.get_logger("Test_Analyzer")

# Constants
LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "app.log")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_03_output", "test_results")


def extract_document_sections(log_content: str) -> Dict[str, str]:
    """
    Extract sections for each document processing from the log file
    
    Args:
        log_content: Content of the log file
        
    Returns:
        Dictionary mapping document names to their processing sections
    """
    # Find all document processing sections
    doc_pattern = r"Running pipeline for ([^\.]+\.pdf)"
    doc_starts = [(m.group(1), m.start()) for m in re.finditer(doc_pattern, log_content)]
    
    sections = {}
    
    # Extract each document's section
    for i, (doc_name, start_pos) in enumerate(doc_starts):
        if i < len(doc_starts) - 1:
            end_pos = doc_starts[i+1][1]
        else:
            end_pos = len(log_content)
        
        sections[doc_name] = log_content[start_pos:end_pos]
    
    return sections


def check_document_properties(doc_name: str, section: str) -> Dict[str, bool]:
    """
    Check for specific document properties in the log section
    
    Args:
        doc_name: Name of the document
        section: Log section for this document
        
    Returns:
        Dictionary with boolean flags for various document properties
    """
    properties = {
        "doc_name": doc_name,
        "is_duplicate": False,
        "duplicate_detected_by_hash": False,
        "has_new_chunks": False,
        "has_duplicate_chunks": False,
        "has_updated_chunks": False,
        "is_version_update": False,
        "pipeline_completed": False,
        "has_errors": False
    }
    
    # Check for hash-based duplicate
    if "is an exact duplicate of" in section:
        properties["is_duplicate"] = True
        properties["duplicate_detected_by_hash"] = True
    
    # Check for pipeline completion
    if "Pipeline completed in" in section:
        properties["pipeline_completed"] = True
    
    # Check for errors
    if "[ERROR]" in section:
        properties["has_errors"] = True
    
    # Check for chunk status
    new_chunks_patterns = [
        r"new chunks.*?[1-9]\d*",
        r"[1-9]\d* new chunks",
        r"New unique chunks: [1-9]\d*",
        r"all new chunks"
    ]
    
    duplicate_chunks_patterns = [
        r"duplicate chunks.*?[1-9]\d*",
        r"[1-9]\d* duplicate",
        r"duplicates.*?[1-9]\d*",
        r"Exact duplicates found: [1-9]\d*"
    ]
    
    updated_chunks_patterns = [
        r"similar chunks.*?[1-9]\d*",
        r"[1-9]\d* similar chunks",
        r"update.*?chunks.*?[1-9]\d*",
        r"[1-9]\d* updates",
        r"Similar chunks found: [1-9]\d*"
    ]
    
    version_update_patterns = [
        r"Using context-aware chunking with reference document",
        r"Using chunks from similar document",
        r"update of [^ ]+"
    ]
    
    # Check for new chunks
    for pattern in new_chunks_patterns:
        if re.search(pattern, section, re.IGNORECASE):
            properties["has_new_chunks"] = True
            break
    
    # Check for duplicate chunks
    for pattern in duplicate_chunks_patterns:
        if re.search(pattern, section, re.IGNORECASE):
            properties["has_duplicate_chunks"] = True
            break
    
    # Check for updated chunks
    for pattern in updated_chunks_patterns:
        if re.search(pattern, section, re.IGNORECASE):
            properties["has_updated_chunks"] = True
            break
    
    # Check for version updates
    for pattern in version_update_patterns:
        if re.search(pattern, section, re.IGNORECASE):
            properties["is_version_update"] = True
            break
    
    return properties


def analyze_document_processing(doc_name: str, section: str) -> Dict[str, Any]:
    """
    Analyze a document processing section from the logs
    
    Args:
        doc_name: Name of the document
        section: Log section for this document
        
    Returns:
        Dictionary with analysis results
    """
    # Get document properties
    properties = check_document_properties(doc_name, section)
    
    result = {
        "document": doc_name,
        "is_duplicate": properties["is_duplicate"],
        "duplicate_detected_by_hash": properties["duplicate_detected_by_hash"],
        "new_chunks": properties["has_new_chunks"],
        "duplicate_chunks": properties["has_duplicate_chunks"],
        "updated_chunks": properties["has_updated_chunks"],
        "is_version_update": properties["is_version_update"],
        "pipeline_completed": properties["pipeline_completed"],
        "duplicate_of": None,
        "chunk_stats": {
            "total": 0,
            "new": 0,
            "duplicate": 0,
            "updated": 0
        },
        "errors": [],
    }
    
    # Check if this is a duplicate detected by hash
    hash_duplicate_match = re.search(r"is an exact duplicate of ([^\(]+)", section)
    if hash_duplicate_match:
        result["duplicate_of"] = hash_duplicate_match.group(1).strip()
    
    # Check for errors
    error_matches = re.findall(r"\[ERROR\].*?(.*?)$", section, re.MULTILINE)
    if error_matches:
        result["errors"] = [error.strip() for error in error_matches]
    
    # Extract chunk statistics
    # Look for summary patterns
    summary_patterns = [
        r"Deduplication results: (\d+) new chunks, (\d+) duplicates, (\d+) updates",
        r"Generated (\d+) chunks",
        r"Total chunks processed: (\d+).*?Exact duplicates found: (\d+).*?Similar chunks found: (\d+).*?New unique chunks: (\d+)",
        r"Processed (\d+) chunks .* \((\d+) duplicates, (\d+) updated, (\d+) new\)"
    ]
    
    for pattern in summary_patterns:
        match = re.search(pattern, section, re.DOTALL)
        if match:
            groups = match.groups()
            if len(groups) == 3:  # Deduplication results format
                result["chunk_stats"]["new"] = int(groups[0])
                result["chunk_stats"]["duplicate"] = int(groups[1])
                result["chunk_stats"]["updated"] = int(groups[2])
                result["chunk_stats"]["total"] = sum(int(g) for g in groups)
                break
            elif len(groups) == 1:  # Generated chunks format
                result["chunk_stats"]["new"] = int(groups[0])
                result["chunk_stats"]["total"] = int(groups[0])
                break
            elif len(groups) == 4:  # Detailed summary format or processed chunks format
                if "Total chunks processed" in match.group(0):
                    # Total, exact dups, similar, new
                    result["chunk_stats"]["total"] = int(groups[0])
                    result["chunk_stats"]["duplicate"] = int(groups[1])
                    result["chunk_stats"]["updated"] = int(groups[2])
                    result["chunk_stats"]["new"] = int(groups[3])
                else:
                    # Processed, dups, updated, new
                    result["chunk_stats"]["total"] = int(groups[0])
                    result["chunk_stats"]["duplicate"] = int(groups[1])
                    result["chunk_stats"]["updated"] = int(groups[2])
                    result["chunk_stats"]["new"] = int(groups[3])
                break
    
    # Fill in defaults if needed
    if result["chunk_stats"]["total"] == 0 and (result["new_chunks"] or result["duplicate_chunks"] or result["updated_chunks"]):
        # Try to infer from regex matches in the logs
        chunk_counts = re.findall(r"(\d+) (new|duplicate|similar|updated) chunks", section, re.IGNORECASE)
        for count, chunk_type in chunk_counts:
            count = int(count)
            if chunk_type.lower() == "new":
                result["chunk_stats"]["new"] = count
            elif chunk_type.lower() in ["duplicate", "duplicates"]:
                result["chunk_stats"]["duplicate"] = count
            elif chunk_type.lower() in ["similar", "updated", "updates"]:
                result["chunk_stats"]["updated"] = count
        
        result["chunk_stats"]["total"] = (
            result["chunk_stats"]["new"] + 
            result["chunk_stats"]["duplicate"] + 
            result["chunk_stats"]["updated"]
        )
    
    return result


def analyze_test_results():
    """Analyze test results from the log file"""
    if not os.path.exists(LOG_PATH):
        logger.error(f"Log file not found at {LOG_PATH}")
        return
    
    # Read the log file
    with open(LOG_PATH, 'r') as f:
        log_content = f.read()
    
    # Extract document sections
    doc_sections = extract_document_sections(log_content)
    
    # Analyze each document section
    results = {}
    for doc_name, section in doc_sections.items():
        results[doc_name] = analyze_document_processing(doc_name, section)
    
    # Find test scenario sections
    scenario_pattern = r"Starting Scenario (\d+): ([^\n]+)"
    scenario_matches = re.findall(scenario_pattern, log_content)
    
    scenarios = []
    for scenario_id, scenario_name in scenario_matches:
        scenario_id = int(scenario_id)
        
        # Find the documents that were part of this scenario
        scenario_start = log_content.find(f"Starting Scenario {scenario_id}: {scenario_name}")
        next_scenario_match = re.search(r"Starting Scenario \d+:", log_content[scenario_start+1:])
        
        if next_scenario_match:
            scenario_end = scenario_start + 1 + next_scenario_match.start()
        else:
            scenario_end = len(log_content)
        
        scenario_section = log_content[scenario_start:scenario_end]
        
        # Extract documents mentioned in this scenario
        doc_refs = re.findall(r"Processing ([^\.]+\.pdf)", scenario_section)
        scenario_docs = list(dict.fromkeys(doc_refs))  # Keep unique values in order
        
        passed = "PASSED" in scenario_section and "FAILED" not in scenario_section
        
        scenarios.append({
            "id": scenario_id,
            "name": scenario_name,
            "documents": scenario_docs,
            "passed": passed,
            "errors": re.findall(r"\[ERROR\].*?(Scenario \d+ FAILED.*?)$", scenario_section, re.MULTILINE)
        })
    
    # Generate report
    report_path = os.path.join(RESULTS_DIR, f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("=============================================\n")
        f.write("=== DETAILED TEST SCENARIOS ANALYSIS REPORT ===\n")
        f.write("=============================================\n\n")
        
        if scenarios:
            f.write("Test Scenarios Summary:\n")
            f.write("-----------------------\n")
            for scenario in scenarios:
                status = "✅ PASSED" if scenario["passed"] else "❌ FAILED"
                f.write(f"Scenario {scenario['id']}: {scenario['name']} - {status}\n")
                for doc in scenario["documents"]:
                    if doc in results:
                        doc_result = results[doc]
                        if doc_result["is_duplicate"]:
                            f.write(f"  - {doc}: Hash duplicate ♻️\n")
                        elif doc_result["new_chunks"] and not doc_result["duplicate_chunks"] and not doc_result["updated_chunks"]:
                            f.write(f"  - {doc}: All new chunks 🆕\n")
                        elif doc_result["duplicate_chunks"] and doc_result["updated_chunks"]:
                            f.write(f"  - {doc}: Mix of duplicate and updated chunks 🔄\n")
                        elif doc_result["duplicate_chunks"]:
                            f.write(f"  - {doc}: Has duplicate chunks ♻️\n")
                        elif doc_result["updated_chunks"]:
                            f.write(f"  - {doc}: Has updated chunks 🔄\n")
                        else:
                            f.write(f"  - {doc}: Processed\n")
                f.write("\n")
            
            f.write("\n")
        
        f.write("Document Processing Analysis:\n")
        f.write("----------------------------\n")
        
        for doc_name, result in results.items():
            f.write(f"Document: {doc_name}\n")
            f.write("-" * 60 + "\n")
            
            if result["is_duplicate"]:
                f.write(f"  This document is a hash-based duplicate of {result['duplicate_of']}\n")
                f.write(f"  No chunk processing performed\n")
            else:
                f.write(f"  Pipeline completed: {result['pipeline_completed']}\n")
                if result["is_version_update"]:
                    f.write(f"  This is a version update of a previous document ♻️\n")
                f.write(f"  New chunks: {result['new_chunks']}\n")
                f.write(f"  Duplicate chunks: {result['duplicate_chunks']}\n")
                f.write(f"  Updated chunks: {result['updated_chunks']}\n")
                
                f.write("  Chunk stats:\n")
                f.write(f"    Total: {result['chunk_stats']['total']}\n")
                f.write(f"    New: {result['chunk_stats']['new']}\n")
                f.write(f"    Duplicate: {result['chunk_stats']['duplicate']}\n")
                f.write(f"    Updated: {result['chunk_stats']['updated']}\n")
            
            if result["errors"]:
                f.write("  Errors:\n")
                for error in result["errors"]:
                    f.write(f"    - {error}\n")
            
            f.write("\n")
        
        f.write("\n")
        f.write("Advanced Pattern Analysis:\n")
        f.write("------------------------\n")
        
        # Check if tests show correct patterns for duplicate detection
        hash_dedup_works = any(r["is_duplicate"] and r["duplicate_detected_by_hash"] for r in results.values())
        version_detection_works = any(r["is_version_update"] for r in results.values())
        duplicate_chunk_detection_works = any(r["duplicate_chunks"] and r["chunk_stats"]["duplicate"] > 0 for r in results.values())
        updated_chunk_detection_works = any(r["updated_chunks"] and r["chunk_stats"]["updated"] > 0 for r in results.values())
        
        f.write("System capabilities detected from test runs:\n")
        f.write(f"  - Hash-based deduplication working: {'✅' if hash_dedup_works else '❌'}\n")
        f.write(f"  - Document version detection working: {'✅' if version_detection_works else '❌'}\n")
        f.write(f"  - Duplicate chunk detection working: {'✅' if duplicate_chunk_detection_works else '❌'}\n")
        f.write(f"  - Updated chunk detection working: {'✅' if updated_chunk_detection_works else '❌'}\n")
    
    logger.info(f"Analysis report written to {report_path}")
    print(f"Analysis report written to {report_path}")
    
    return results, scenarios


def main():
    parser = argparse.ArgumentParser(description='Analyze test results from logs')
    args = parser.parse_args()
    
    analyze_test_results()
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 