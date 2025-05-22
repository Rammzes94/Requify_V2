#!/usr/bin/env python3
"""
test_chunking.py

This script tests the consolidated chunking implementation with different test scenarios:

Test Scenarios:

| #  | Scenario Name             | 📝 Description                          | 🎯 Expected Outcome                  |
| -- | ------------------------- | --------------------------------------- | -----------------------------------  |
| 1  | value_changes            | 🔢 Value changes (e.g., range/capacity) | ⚠️ Prompt user for change            |
| 2  | minor_formatting         | ✨ Minor formatting/whitespace          | ✅ Detect as duplicate               |
| 3  | text_changes             | 📝 Requirement text changes             | ⚠️ Prompt user for change            |
| 4  | critical_additions       | ➕ Important info added                 | ✅ Keep new chunk                    |
| 5  | mixed_changes            | 🔀 Mix of major/minor edits             | ⚠️ Prompt user                       |
| 6  | chunk_split_merge       | 🍰 Paragraph split/merge, same meaning  | ✅ Detect as duplicate               |
| 7  | section_reordering       | 🔄 Sections moved/reordered             | ✅ Recognize as duplicate            |
| 8  | removed_content          | ❌ Key requirement deleted              | 🟫 Keep old version                  |
| 9  | table_format_change     | 🗒️ List ↔ Table, data same              | ✅ Detect as duplicate               |
| 10 | spurious_cosmetic_edit  | 🧹 Cosmetic/bullet/whitespace only      | ✅ Detect as duplicate               |
| 11 | duplicate_section        | 📄 Section duplicated verbatim          | 🟫 Keep old, non-duplicated version  |
| 12 | fuzzy_duplicate_synonym | 🤔 Same meaning, different words        | ✅ Detect as duplicate               |

The script also tests:
- Basic chunking with a small document
- Standard chunking with a larger document
- Context-aware chunking with document versions
"""

import os
import sys
import logging
import argparse
import unittest
import gc
import torch
import time
from dotenv import load_dotenv

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import utilities
from src._00_utils import setup_project_directory, get_logger

# Set up the project directory
setup_project_directory()

# Load environment variables
load_dotenv()

# Import the consolidated chunking module
from src._02_parsing import agentic_chunking

# Setup logging with script prefix
logger = get_logger("Test_Chunking")

# Define test scenarios
TEST_SCENARIOS = {
    "value_changes": {
        "description": "Test how the system handles numerical value changes",
        "expected_outcome": "Should prompt user for changes to key specifications like range and capacity",
        "original": """# Fighter Jet Rocket Launcher Specifications

## Performance Parameters
- Maximum Range: 30 km
- Guidance System: Infrared homing
- Reload Time: 45 seconds
- Operating Temperature: -40°C to +60°C 
- Maximum Altitude: 15,000 meters

## Physical Specifications
- Weight: 250 kg
- Length: 3.5 meters
- Width: 0.8 meters
- Height: 0.9 meters
- Capacity: 8 rockets
""",
        "modified": """# Fighter Jet Rocket Launcher Specifications

## Performance Parameters
- Maximum Range: 45 km
- Guidance System: Infrared homing
- Reload Time: 30 seconds
- Operating Temperature: -40°C to +60°C 
- Maximum Altitude: 15,000 meters

## Physical Specifications
- Weight: 250 kg
- Length: 3.5 meters
- Width: 0.8 meters
- Height: 0.9 meters
- Capacity: 12 rockets
"""
    },
    "minor_formatting": {
        "description": "Test how the system handles minor formatting changes",
        "expected_outcome": "Should detect these as duplicates and keep the original",
        "original": """# Project Timeline

## Phase 1: Planning
- Requirements gathering
- Architecture design
- Resource allocation

## Phase 2: Development
- Frontend implementation
- Backend services
- Database setup
""",
        "modified": """# Project Timeline

## Phase 1: Planning
- Requirements gathering
- Architecture design 
- Resource allocation

## Phase 2: Development
- Frontend implementation
- Backend services
- Database setup
"""
    },
    "text_changes": {
        "description": "Test how the system handles textual description changes",
        "expected_outcome": "Should prompt user when descriptions change substantively",
        "original": """# Software Requirements

## Security Requirements
The system must implement AES-256 encryption for all data at rest.
User authentication should use multi-factor authentication.
Password policies must enforce complexity rules.

## Performance Requirements
The system must respond to user queries within 200ms.
Database transactions should complete within 50ms.
""",
        "modified": """# Software Requirements

## Security Requirements
The system must implement ChaCha20-Poly1305 encryption for all data at rest.
User authentication should use multi-factor authentication.
Password policies must enforce complexity rules.

## Performance Requirements
The system must respond to user queries within 150ms.
Database transactions should complete within 30ms.
"""
    },
    "critical_additions": {
        "description": "Test how the system handles additions of critical information",
        "expected_outcome": "Should keep the new chunk with additions",
        "original": """# Safety Protocols

## Emergency Procedures
1. Evacuate the building using marked exit routes
2. Assemble at designated meeting points
3. Wait for further instructions

## Contact Information
- Security Office: 555-1234
- Facilities Management: 555-5678
""",
        "modified": """# Safety Protocols

## Emergency Procedures
1. Evacuate the building using marked exit routes
2. Assemble at designated meeting points
3. Wait for further instructions
4. Report any missing personnel to safety officers
5. Do not re-enter until authorized

## Contact Information
- Security Office: 555-1234
- Facilities Management: 555-5678
- Emergency Response Team: 555-9111
"""
    },
    "mixed_changes": {
        "description": "Test how the system handles mixed types of changes",
        "expected_outcome": "Should prompt for significant changes while ignoring minor formatting",
        "original": """# Product Specifications

## Power Supply
- Input Voltage: 100-240V AC
- Output: 12V DC, 5A
- Efficiency: 85%
- Protection: Over-voltage, short-circuit

## Environmental
- Operating Temperature: 0°C to 40°C
- Storage Temperature: -20°C to 60°C
- Humidity: 10% to 80% non-condensing
""",
        "modified": """# Product Specifications

## Power Supply
- Input Voltage: 100-240V AC
- Output: 24V DC, 2.5A
- Efficiency: 90%
- Protection: Over-voltage, short-circuit, thermal

## Environmental
- Operating Temperature: -10°C to 50°C
- Storage Temperature: -20°C to 60°C
- Humidity: 10% to 80% non-condensing
"""
    },
    "chunk_split_merge": {
        "description": "Test handling of paragraph splits/merges where content boundaries shift but meaning stays the same.",
        "expected_outcome": "Should detect as duplicate and keep only one version.",
        "original": """# System Description

The system must operate reliably in extreme temperatures (-40°C to +85°C). It must withstand shocks up to 100g and vibrations from 5 to 500 Hz.

All internal modules are protected against humidity and dust ingress.

# Maintenance

Regular inspections are required every 6 months. Replace all worn parts immediately.
""",
        "modified": """# System Description

The system must operate reliably in extreme temperatures (-40°C to +85°C).
It must withstand shocks up to 100g and vibrations from 5 to 500 Hz.

All internal modules are protected against humidity and dust ingress.

# Maintenance

Regular inspections are required every 6 months.
Replace all worn parts immediately.
"""
    },
    "section_reordering": {
        "description": "Test how the system handles documents where whole sections are reordered but not changed.",
        "expected_outcome": "Should recognize all unchanged sections as duplicates regardless of order.",
        "original": """# Electrical Properties

- Input Voltage: 24V DC
- Standby Current: 2A
- Peak Power: 100W

# Mechanical Properties

- Weight: 2.2 kg
- Dimensions: 200 x 100 x 75 mm

# Communication

- Protocol: CAN 2.0B
- Baudrate: 500 kbps
""",
        "modified": """# Communication

- Protocol: CAN 2.0B
- Baudrate: 500 kbps

# Mechanical Properties

- Weight: 2.2 kg
- Dimensions: 200 x 100 x 75 mm

# Electrical Properties

- Input Voltage: 24V DC
- Standby Current: 2A
- Peak Power: 100W
"""
    },
    "removed_content": {
        "description": "Test detection when a critical requirement is deleted in the new version.",
        "expected_outcome": "Should keep the old version (with all content).",
        "original": """# System Features

- Secure boot enabled
- Encrypted storage (AES-256)
- Remote wipe supported
- Dual authentication (PIN + biometric)
- Over-the-air firmware updates
""",
        "modified": """# System Features

- Secure boot enabled
- Remote wipe supported
- Dual authentication (PIN + biometric)
- Over-the-air firmware updates
"""
    },
    "table_format_change": {
        "description": "Test robustness to the same data in list vs table format.",
        "expected_outcome": "Should recognize content as the same even with different formatting.",
        "original": """# Physical Parameters

- Height: 180 cm
- Width: 90 cm
- Depth: 40 cm
- Mass: 145 kg
- Color: Olive drab
""",
        "modified": """# Physical Parameters

| Parameter | Value      |
|-----------|-----------|
| Height    | 180 cm    |
| Width     | 90 cm     |
| Depth     | 40 cm     |
| Mass      | 145 kg    |
| Color     | Olive drab|
"""
    },
    "spurious_cosmetic_edit": {
        "description": "Test detection of changes that are only whitespace, bullet style, or other cosmetic edits.",
        "expected_outcome": "Should ignore purely cosmetic differences and consider documents as duplicates.",
        "original": """# Battery Information

- Battery type: Li-ion
- Rated capacity: 3000 mAh
- Typical runtime: 10 hours

# Charger

- Input: 100-240V AC
- Output: 5V DC, 2A
""",
        "modified": """# Battery Information

* Battery type: Li-ion  
* Rated capacity: 3000 mAh  
* Typical runtime: 10 hours  

# Charger

* Input: 100-240V AC  
* Output: 5V DC, 2A  
"""
    },
    "duplicate_section": {
        "description": "Test when a section is duplicated verbatim in the new version.",
        "expected_outcome": "Should keep the old, non-duplicated version.",
        "original": """# Summary

All subsystems meet the stated technical requirements as of 2024-01-01.
No outstanding critical defects remain.
""",
        "modified": """# Summary

All subsystems meet the stated technical requirements as of 2024-01-01.
No outstanding critical defects remain.

# Summary

All subsystems meet the stated technical requirements as of 2024-01-01.
No outstanding critical defects remain.
"""
    },
    "fuzzy_duplicate_synonym": {
        "description": "Test sensitivity to synonym/reworded content that is semantically the same.",
        "expected_outcome": "Should detect as duplicate.",
        "original": """# Password Policy

All user passwords must be at least 12 characters long, contain both upper and lower case letters, at least one number, and one special character. Passwords must be changed every 180 days.
""",
        "modified": """# Password Policy

Passwords are required to be a minimum of 12 characters in length, with at least one uppercase letter, one lowercase letter, one numeric digit, and one special character. Users should update their passwords every six months.
"""
    }
}

class TestChunking(unittest.TestCase):
    """Tests for the consolidated chunking implementation."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test files path and other resources."""
        # Find the path to the test files
        cls.base_dir = os.path.dirname(os.path.abspath(__file__))
        cls.test_files_dir = os.path.join(cls.base_dir, 'test_files')
        
        # Test file paths
        cls.test_doc_path = os.path.join(cls.test_files_dir, 'test_document.txt')
        cls.test_doc_updated_path = os.path.join(cls.test_files_dir, 'test_document_updated.txt')
        cls.small_test_path = os.path.join(cls.test_files_dir, 'small_test.txt')
        
        # Verify files exist
        for file_path in [cls.test_doc_path, cls.test_doc_updated_path, cls.small_test_path]:
            if not os.path.exists(file_path):
                logger.error(f"Test file not found: {file_path}", extra={"icon": "❌"})
                raise FileNotFoundError(f"Test file not found: {file_path}")
        
        logger.info("Test files found and verified", extra={"icon": "✅"})
    
    def test_basic_chunking(self):
        """Test basic chunking with a small document."""
        logger.info("Testing basic chunking with small document", extra={"icon": "🧪"})
        
        with open(self.small_test_path, 'r', encoding='utf-8') as f:
            document_text = f.read()
        
        # Perform chunking
        chunks = agentic_chunking.chunk_markdown(document_text)
        
        # Verify results
        self.assertIsNotNone(chunks, "Chunks should not be None")
        self.assertGreater(len(chunks), 0, "Should generate at least one chunk")
        
        # Report stats
        stats = agentic_chunking.analyze_chunks(chunks)
        logger.info(f"Generated {len(chunks)} chunks from small document", extra={"icon": "📊"})
        logger.info(f"Avg chunk size: {stats['char_sizes']['avg']:.1f} chars / {stats['token_sizes']['avg']:.1f} tokens", extra={"icon": "📏"})
    
    def test_standard_chunking(self):
        """Test standard chunking with a larger document."""
        logger.info("Testing standard chunking with larger document", extra={"icon": "🧪"})
        
        with open(self.test_doc_path, 'r', encoding='utf-8') as f:
            document_text = f.read()
        
        # Perform chunking
        chunks = agentic_chunking.chunk_markdown(document_text)
        
        # Verify results
        self.assertIsNotNone(chunks, "Chunks should not be None")
        self.assertGreater(len(chunks), 1, "Should generate multiple chunks for larger document")
        
        # Report stats
        stats = agentic_chunking.analyze_chunks(chunks)
        logger.info(f"Generated {len(chunks)} chunks from standard document", extra={"icon": "📊"})
        logger.info(f"Avg chunk size: {stats['char_sizes']['avg']:.1f} chars / {stats['token_sizes']['avg']:.1f} tokens", extra={"icon": "📏"})
        
        # Verify chunk sizes
        for i, chunk in enumerate(chunks):
            self.assertLessEqual(len(chunk), agentic_chunking.MAX_CHAR_SIZE, 
                                f"Chunk {i} exceeds maximum size")
    
    def test_context_aware_chunking(self):
        """Test context-aware chunking with original and updated documents."""
        logger.info("Testing context-aware chunking with document versions", extra={"icon": "🧪"})
        
        # Load original document
        with open(self.test_doc_path, 'r', encoding='utf-8') as f:
            original_text = f.read()
            
        # Load updated document
        with open(self.test_doc_updated_path, 'r', encoding='utf-8') as f:
            updated_text = f.read()
            
        # Generate chunks for original document
        original_chunks = agentic_chunking.chunk_markdown(original_text)
        
        # Convert to format expected by context-aware chunking
        context_chunks = []
        for i, chunk_text in enumerate(original_chunks):
            context_chunks.append({
                'chunk_id': f'test_doc_{i:04d}',
                'chunk_text': chunk_text,
                'document_id': 'test_document.txt',
                'chunk_index': i
            })
        
        # Generate chunks for updated document with context
        updated_chunks = agentic_chunking.chunk_markdown(updated_text, context_chunks)
        
        # Verify results
        self.assertIsNotNone(updated_chunks, "Updated chunks should not be None")
        self.assertGreater(len(updated_chunks), 1, "Should generate multiple chunks for updated document")
        
        # Report stats
        logger.info(f"Original document: {len(original_chunks)} chunks", extra={"icon": "📊"})
        logger.info(f"Updated document: {len(updated_chunks)} chunks", extra={"icon": "📊"})
        
        # The number of chunks should be similar (but not necessarily identical)
        chunk_difference = abs(len(updated_chunks) - len(original_chunks))
        self.assertLessEqual(chunk_difference, 2, 
                            f"Chunk count difference ({chunk_difference}) should be minimal with context-aware chunking")
    
    def test_scenario_specific_cases(self):
        """Run tests for all the specific change scenarios."""
        # Set environment variable for auto-selecting new in user prompts
        os.environ["REQUIFY_AUTO_SELECT_NEW"] = "true"
        
        for scenario_name, scenario_data in TEST_SCENARIOS.items():
            logger.info(f"=== Testing Scenario: {scenario_name} ===", extra={"icon": "🧪"})
            logger.info(f"Description: {scenario_data['description']}", extra={"icon": "ℹ️"})
            logger.info(f"Expected: {scenario_data['expected_outcome']}", extra={"icon": "🎯"})
            
            # Process the original document first
            original_doc_id = f"test_{scenario_name}_original"
            
            # Generate chunks for original document
            original_chunks = agentic_chunking.chunk_markdown(scenario_data['original'])
            
            # Convert to format expected by context-aware chunking
            context_chunks = []
            for i, chunk_text in enumerate(original_chunks):
                context_chunks.append({
                    'chunk_id': f'{original_doc_id}_chunk_{i:04d}',
                    'chunk_text': chunk_text,
                    'document_id': original_doc_id,
                    'chunk_index': i
                })
            
            # Generate chunks for modified document with context
            modified_doc_id = f"test_{scenario_name}_modified"
            modified_chunks = agentic_chunking.chunk_markdown(scenario_data['modified'], context_chunks)
            
            # Verify results
            self.assertIsNotNone(modified_chunks, f"Chunks should not be None for scenario: {scenario_name}")
            
            # Report stats
            logger.info(f"Original: {len(original_chunks)} chunks, Modified: {len(modified_chunks)} chunks", extra={"icon": "📊"})
            
            # Clean up memory after each scenario
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # If using MPS (Apple Silicon), clear that cache too
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            
            time.sleep(1)  # Small delay to ensure memory is freed
        
        # Reset environment variable
        os.environ["REQUIFY_AUTO_SELECT_NEW"] = "false"

def run_tests(batch_size=None):
    """Run the tests."""
    # Override batch size if specified
    if batch_size:
        agentic_chunking.EMBEDDING_BATCH_SIZE = batch_size
        logger.info(f"Setting embedding batch size to {batch_size}", extra={"icon": "🔧"})
    
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test consolidated chunking with various scenarios')
    parser.add_argument('--batch-size', type=int, default=None, 
                        help='Batch size for processing chunks (smaller = less memory)')
    args = parser.parse_args()
    
    run_tests(args.batch_size) 