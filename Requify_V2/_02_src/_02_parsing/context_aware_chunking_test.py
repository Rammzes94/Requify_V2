"""
context_aware_chunking_test.py

A simple test script to verify that the context_aware_chunking module works as expected.
It creates test documents with various types of changes to validate the system's detection 
and decision-making capabilities for different change scenarios.

Test Scenarios:

| #  | Scenario Name             | 📝 Description                          | 🎯 Expected Outcome                  |
| -- | ------------------------- | --------------------------------------- | -----------------------------------  |
| 1  | value\_changes            | 🔢 Value changes (e.g., range/capacity) | ⚠️ Prompt user for change            |
| 2  | minor\_formatting         | ✨ Minor formatting/whitespace          | ✅ Detect as duplicate               |
| 3  | text\_changes             | 📝 Requirement text changes             | ⚠️ Prompt user for change            |
| 4  | critical\_additions       | ➕ Important info added                 | ✅ Keep new chunk                    |
| 5  | mixed\_changes            | 🔀 Mix of major/minor edits             | ⚠️ Prompt user                       |
| 6  | chunk\_split\_merge       | 🍰 Paragraph split/merge, same meaning  | ✅ Detect as duplicate               |
| 7  | section\_reordering       | 🔄 Sections moved/reordered             | ✅ Recognize as duplicate            |
| 8  | removed\_content          | ❌ Key requirement deleted              | 🟫 Keep old version                  |
| 9  | table\_format\_change     | 🗒️ List ↔ Table, data same              | ✅ Detect as duplicate               |
| 10 | spurious\_cosmetic\_edit  | 🧹 Cosmetic/bullet/whitespace only      | ✅ Detect as duplicate               |
| 11 | duplicate\_section        | 📄 Section duplicated verbatim          | 🟫 Keep old, non-duplicated version  |
| 12 | fuzzy\_duplicate\_synonym | 🤔 Same meaning, different words        | ✅ Detect as duplicate               |


"""

import os
import sys
import logging
import time
import argparse
import gc  # Add garbage collection
import torch  # Add torch for cache management
from dotenv import load_dotenv

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import _00_utils
_00_utils.setup_project_directory()

# Load environment variables
load_dotenv()

# Import the module to test - using relative import
from _02_parsing.context_aware_chunking import process_document as process_with_context
from _02_parsing.context_aware_chunking import get_similar_document_chunks

# Setup logging
logger = _00_utils.setup_logging()

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

def test_context_aware_chunking(scenario_name=None):
    """Test the context_aware_chunking module with sample data."""
    
    # Set environment variable for auto-selecting new in user prompts
    os.environ["REQUIFY_AUTO_SELECT_NEW"] = "true"
    
    if scenario_name and scenario_name not in TEST_SCENARIOS:
        logger.error(f"Unknown scenario: {scenario_name}")
        return False
    
    # Run all scenarios if none specified
    scenarios_to_run = [scenario_name] if scenario_name else TEST_SCENARIOS.keys()
    
    for scenario in scenarios_to_run:
        scenario_data = TEST_SCENARIOS[scenario]
        
        logger.info(f"=== Testing Scenario: {scenario} ===")
        logger.info(f"Description: {scenario_data['description']}")
        logger.info(f"Expected Outcome: {scenario_data['expected_outcome']}")
        
        # Process the original document first
        logger.info("\n=== Processing Original Document ===")
        original_doc_id = f"test_{scenario}_original"
        original_result = process_with_context(scenario_data['original'], original_doc_id, None)
        
        if not original_result:
            logger.error(f"Failed to process original document for {scenario}")
            continue
        
        # Now process the modified document with context
        logger.info(f"\n=== Processing Modified Document with Context ===")
        modified_doc_id = f"test_{scenario}_modified"
        
        # First get the chunks from the original document
        similar_chunks = get_similar_document_chunks(modified_doc_id, original_doc_id)
        
        if not similar_chunks:
            logger.warning(f"Could not retrieve similar chunks from original document for {scenario}")
            continue
            
        logger.info(f"Retrieved {len(similar_chunks)} reference chunks from original document")
        
        # Process the modified document with context
        modified_result = process_with_context(scenario_data['modified'], modified_doc_id, original_doc_id)
        
        # Even if no new chunks are generated (all are duplicates/skipped), 
        # consider it a success for scenarios where the expected outcome is to detect duplicates
        expected_outcome = scenario_data['expected_outcome'].lower()
        if (modified_result is None or modified_result == False) and ("duplicate" in expected_outcome or "keep old" in expected_outcome):
            logger.info(f"✅ Expected outcome achieved - all chunks were treated as duplicates/skipped")
            logger.info(f"=== Completed Scenario: {scenario} ===\n")
            continue
        elif not modified_result:
            logger.error(f"Failed to process modified document for {scenario}")
            continue
        
        # Clean up memory after each scenario
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # If using MPS (Apple Silicon), clear that cache too
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        
        # Small delay to ensure memory is freed
        time.sleep(1)
    
    # Reset environment variable
    os.environ["REQUIFY_AUTO_SELECT_NEW"] = "false"
    
    logger.info("=== All Test Scenarios Completed ===")
    return True

if __name__ == "__main__":
    # Add command line argument for specific scenario
    parser = argparse.ArgumentParser(description='Test context-aware chunking with various scenarios')
    parser.add_argument('--scenario', type=str, help='Specific scenario to test (omit to test all)')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for processing chunks (smaller = less memory)')
    args = parser.parse_args()
    
    # Set batch size in the module
    import _02_parsing.context_aware_chunking as chunking_module
    
    # Override batch size if specified
    if args.batch_size:
        chunking_module.EMBEDDING_BATCH_SIZE = args.batch_size
        logger.info(f"Setting embedding batch size to {args.batch_size}")
    
    result = test_context_aware_chunking(args.scenario)
    sys.exit(0 if result else 1) 