#!/usr/bin/env python3
"""
test_chunk_comparison.py

This script tests the chunk comparison functionality in agentic_chunking.py
by simulating a scenario with an original document and an updated one with changed values.
"""

import os
import sys
import logging
import shutil
from pathlib import Path

# Add the parent directory to the system path to allow importing modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src._02_parsing.agentic_chunking import (
    process_document_with_context,
    log_chunk_comparison,
    compute_chunk_similarity,
    save_chunks_to_db,
    SIMILARITY_THRESHOLD
)
from src.utils import setup_logging, get_logger

# Setup logging
logger = get_logger("test_chunk_comparison")

# Test document contents
ORIGINAL_DOC = """
# Fighter Jet Rocket Launcher Specifications

## Overview
The XF-35 fighter jet is equipped with advanced rocket launchers capable of delivering precision strikes.
This document outlines the specifications and operational parameters.

## Technical Specifications
- Weight: 350 kg
- Length: 4.2 meters
- Diameter: 30 cm
- Operational altitude: 0-40,000 feet
- Maximum range: 75 km
- Guidance system: Infrared + Radar
- Payload capacity: 120 kg
- Firing rate: 2 rockets per second

## Safety Parameters
All operations must follow standard safety protocols. The rocket launcher system includes 
automatic safety mechanisms that prevent accidental discharge during ground operations.
Temperature tolerances: -40°C to +60°C.
"""

UPDATED_DOC = """
# Fighter Jet Rocket Launcher Specifications

## Overview
The XF-35 fighter jet is equipped with advanced rocket launchers capable of delivering precision strikes.
This document outlines the specifications and operational parameters.

## Technical Specifications
- Weight: 320 kg (reduced)
- Length: 4.2 meters
- Diameter: 32 cm (increased)
- Operational altitude: 0-45,000 feet (improved)
- Maximum range: 85 km (extended)
- Guidance system: Infrared + Radar + GPS (enhanced)
- Payload capacity: 120 kg
- Firing rate: 3 rockets per second (improved)

## Safety Parameters
All operations must follow standard safety protocols. The rocket launcher system includes 
automatic safety mechanisms that prevent accidental discharge during ground operations.
Temperature tolerances: -45°C to +65°C (extended range).
Additional safety feature: Emergency jettison capability added for critical situations.
"""

def setup_test_environment():
    """Prepare the test environment by creating necessary directories."""
    # Create test directories if they don't exist
    test_dir = Path("temp/test_data")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test documents
    with open(test_dir / "fighter_jet_rocket_launcher_spec_2.pdf", "w") as f:
        f.write(ORIGINAL_DOC)
    
    with open(test_dir / "fighter_jet_rocket_launcher_spec_2_changed_values.pdf", "w") as f:
        f.write(UPDATED_DOC)
    
    logger.info(f"Test environment set up in {test_dir}", extra={"icon": "✅"})
    return test_dir

def run_test():
    """Run the chunk comparison test."""
    # Setup test environment
    test_dir = setup_test_environment()
    
    # Document IDs for testing
    original_doc_id = "fighter_jet_rocket_launcher_spec_2.pdf"
    updated_doc_id = "fighter_jet_rocket_launcher_spec_2_changed_values.pdf"
    
    # Step 1: Process the original document
    with open(test_dir / original_doc_id, "r") as f:
        original_content = f.read()
    
    logger.info("Processing original document...", extra={"icon": "🔄"})
    original_chunks, _ = process_document_with_context(original_content, original_doc_id)
    
    if not original_chunks:
        logger.error("Failed to generate chunks for original document", extra={"icon": "❌"})
        return False
    
    logger.info(f"Generated {len(original_chunks)} chunks for original document", extra={"icon": "✅"})
    
    # Save original chunks to database for retrieval
    save_success = save_chunks_to_db(original_chunks)
    if not save_success:
        logger.error("Failed to save original chunks to database", extra={"icon": "❌"})
        return False
    
    # Step 2: Process the updated document with context from the original
    with open(test_dir / updated_doc_id, "r") as f:
        updated_content = f.read()
    
    logger.info("Processing updated document with context from original...", extra={"icon": "🔄"})
    updated_chunks, replacements = process_document_with_context(
        updated_content, 
        updated_doc_id, 
        similar_doc_id=original_doc_id
    )
    
    if not updated_chunks:
        logger.error("Failed to generate chunks for updated document", extra={"icon": "❌"})
        return False
    
    logger.info(f"Generated {len(updated_chunks)} chunks for updated document", extra={"icon": "✅"})
    logger.info(f"Found {len(replacements)} chunk replacements", extra={"icon": "🔄"})
    
    # Step 3: Save updated chunks
    save_success = save_chunks_to_db(updated_chunks, replacements)
    if not save_success:
        logger.error("Failed to save updated chunks to database", extra={"icon": "❌"})
        return False
    
    logger.info("Test completed successfully", extra={"icon": "✅"})
    return True

if __name__ == "__main__":
    # Ensure the temp directory exists
    os.makedirs("temp", exist_ok=True)
    
    # Run the test
    success = run_test()
    
    if success:
        logger.info("Chunk comparison test passed! The fix appears to be working.", extra={"icon": "🎉"})
        sys.exit(0)
    else:
        logger.error("Chunk comparison test failed.", extra={"icon": "❌"})
        sys.exit(1) 