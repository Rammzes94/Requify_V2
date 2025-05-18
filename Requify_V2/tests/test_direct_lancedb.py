"""
test_direct_lancedb.py

This script directly tests the LanceDB connection and our save_chunks_to_db function
with minimal dependencies.
"""

import os
import sys
import pandas as pd
import lancedb
import time
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Add the parent directory to the system path to allow importing modules from it
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
import _02_src._00_utils as _00_utils
_00_utils.setup_project_directory()

# Load environment variables
load_dotenv()

# Setup logging
logger = _00_utils.setup_logging()

# Constants
OUTPUT_DIR = "_03_output"
LANCEDB_SUBDIR = "lancedb"
CHUNKS_TABLE_NAME = "document_chunks"

def ensure_db_path():
    """Ensure the database directory exists."""
    lancedb_path = os.path.join(project_root, OUTPUT_DIR, LANCEDB_SUBDIR)
    logger.info(f"📁 Creating database directory: {lancedb_path}")
    os.makedirs(lancedb_path, exist_ok=True)
    return lancedb_path

def save_chunks_to_db(
    chunks: List[Dict[str, Any]], replaced_chunks: Optional[Dict[str, str]] = None
) -> bool:
    """Simplified version of the save_chunks_to_db function."""
    logger.info("🚀 Starting save_chunks_to_db")
    
    # Connect to LanceDB
    lancedb_path = ensure_db_path()
    db = lancedb.connect(lancedb_path)
    logger.info(f"🔌 Connected to DB with tables: {db.table_names()}")
    
    # Check if table exists
    table_exists = CHUNKS_TABLE_NAME in db.table_names()
    
    # Convert chunks to DataFrame
    chunks_df = pd.DataFrame(chunks)
    
    # Ensure required columns exist
    for col in ['is_duplicate_marker', 'is_replaced', 'replaced_by']:
        if col not in chunks_df.columns:
            if col in ['is_duplicate_marker', 'is_replaced']:
                chunks_df[col] = False
            else:
                chunks_df[col] = ""
    
    if table_exists:
        logger.info(f"📊 Table {CHUNKS_TABLE_NAME} exists. Opening...")
        chunks_table = db.open_table(CHUNKS_TABLE_NAME)
        existing_df = chunks_table.to_pandas()
        logger.info(f"📋 Retrieved {len(existing_df)} existing rows")
        
        # Apply replacements
        if replaced_chunks and not existing_df.empty:
            logger.info(f"🔄 Processing {len(replaced_chunks)} replacements")
            replacement_updates = []
            for old_chunk_id, new_chunk_id in replaced_chunks.items():
                mask = existing_df['chunk_id'] == old_chunk_id
                if any(mask):
                    for idx in existing_df[mask].index:
                        updated_row = existing_df.loc[idx].copy()
                        updated_row['is_replaced'] = True
                        updated_row['replaced_by'] = new_chunk_id
                        replacement_updates.append(updated_row)
                    logger.info(f"✅ Marked chunk {old_chunk_id} as replaced by {new_chunk_id}")
                else:
                    logger.warning(f"❌ Could not find chunk {old_chunk_id} to mark as replaced")
            
            if replacement_updates:
                updates_df = pd.DataFrame(replacement_updates)
                # Remove rows that will be updated
                existing_df = existing_df[~existing_df['chunk_id'].isin(replaced_chunks.keys())]
                # Combine all data
                combined_data = pd.concat([existing_df, updates_df, chunks_df], ignore_index=True)
                # Recreate table
                logger.info("🔄 Dropping and recreating table with updated data")
                db.drop_table(CHUNKS_TABLE_NAME)
                chunks_table = db.create_table(CHUNKS_TABLE_NAME, data=combined_data)
                logger.info(f"✅ Table recreated with {len(combined_data)} rows")
                return True
            else:
                # Just add new chunks
                logger.info(f"📝 No replacements applied. Adding {len(chunks_df)} new chunks")
                chunks_table.add(chunks_df)
                return True
        else:
            # No replacements, just add new chunks
            logger.info(f"📝 Adding {len(chunks_df)} new chunks")
            chunks_table.add(chunks_df)
            return True
    else:
        # Create new table
        logger.info(f"📝 Creating new table {CHUNKS_TABLE_NAME} with {len(chunks_df)} rows")
        db.create_table(CHUNKS_TABLE_NAME, data=chunks_df)
        return True

def test_direct():
    """Direct test of chunk replacement."""
    # Ensure DB path exists
    lancedb_path = ensure_db_path()
    
    # Connect to DB
    db = lancedb.connect(lancedb_path)
    logger.info(f"🔌 Connected to database at {lancedb_path}")
    
    # Clean slate - drop table if exists
    if CHUNKS_TABLE_NAME in db.table_names():
        logger.info(f"🗑️ Dropping existing table {CHUNKS_TABLE_NAME}")
        db.drop_table(CHUNKS_TABLE_NAME)
    
    # Create original chunks
    logger.info("🏗️ Creating test chunks")
    original_chunks = [
        {
            "chunk_id": "doc1_chunk_1",
            "document_id": "doc1.pdf",
            "chunk_index": 0,
            "start_offset": 0,
            "end_offset": 100,
            "chunk_text": "This is the first test chunk",
            "token_count": 25,
            "embedding": [0.1] * 10,
            "chunk_hash": "hash1",
            "is_duplicate": False,
            "duplicate_of": "",
            "is_updated": False,
            "previous_chunk_id": "",
            "timestamp": "2023-01-01T00:00:00.000Z",
            "aligned_with_chunk_id": "",
            "aligned_with_document_id": ""
        },
        {
            "chunk_id": "doc1_chunk_2",
            "document_id": "doc1.pdf",
            "chunk_index": 1,
            "start_offset": 101,
            "end_offset": 200,
            "chunk_text": "This is the second test chunk",
            "token_count": 25,
            "embedding": [0.2] * 10,
            "chunk_hash": "hash2",
            "is_duplicate": False,
            "duplicate_of": "",
            "is_updated": False,
            "previous_chunk_id": "",
            "timestamp": "2023-01-01T00:00:00.000Z",
            "aligned_with_chunk_id": "",
            "aligned_with_document_id": ""
        }
    ]
    
    # Step 1: Save original chunks
    logger.info("💾 Step 1: Saving original chunks")
    success = save_chunks_to_db(original_chunks)
    if not success:
        logger.error("❌ Failed to save original chunks")
        return False
    
    # Verify chunks were saved properly
    chunks_table = db.open_table(CHUNKS_TABLE_NAME)
    df = chunks_table.to_pandas()
    logger.info(f"✅ Original chunks saved: {len(df)} rows")
    
    required_columns = ['chunk_id', 'is_replaced', 'replaced_by']
    available_columns = [col for col in required_columns if col in df.columns]
    logger.info(f"📋 Original chunks state:\n{df[available_columns]}")
    
    # Create replacement chunk
    logger.info("🏗️ Creating replacement chunk")
    replacement_chunks = [
        {
            "chunk_id": "doc2_chunk_1",
            "document_id": "doc2.pdf",
            "chunk_index": 0,
            "start_offset": 0,
            "end_offset": 120,
            "chunk_text": "This is the first test chunk with updates",
            "token_count": 30,
            "embedding": [0.15] * 10,
            "chunk_hash": "hash3",
            "is_duplicate": False,
            "duplicate_of": "",
            "is_updated": True,
            "previous_chunk_id": "doc1_chunk_1",
            "timestamp": "2023-01-02T00:00:00.000Z",
            "aligned_with_chunk_id": "doc1_chunk_1",
            "aligned_with_document_id": "doc1.pdf"
        }
    ]
    
    replaced_chunks = {
        "doc1_chunk_1": "doc2_chunk_1"
    }
    
    logger.info(f"🔄 Replacement map: {replaced_chunks}")
    
    # Step 2: Save replacement chunk
    logger.info("💾 Step 2: Saving replacement chunk")
    success = save_chunks_to_db(replacement_chunks, replaced_chunks)
    if not success:
        logger.error("❌ Failed to save replacement chunk")
        return False
    
    # Verify the replacement was applied correctly
    logger.info("🔍 Verifying replacement")
    chunks_table = db.open_table(CHUNKS_TABLE_NAME)
    df = chunks_table.to_pandas()
    logger.info(f"📊 Final state: {len(df)} rows")
    
    available_columns = [col for col in required_columns if col in df.columns]
    logger.info(f"📋 All chunks after replacement:\n{df[available_columns]}")
    
    # Check if original chunk is marked as replaced
    original_in_df = "doc1_chunk_1" in df["chunk_id"].values
    replaced_status = False
    if original_in_df:
        replaced_row = df[df['chunk_id'] == 'doc1_chunk_1']
        if 'is_replaced' in replaced_row.columns:
            replaced_status = replaced_row.iloc[0]['is_replaced']
            replaced_by = replaced_row.iloc[0]['replaced_by']
            logger.info(f"🔄 Replacement status - is_replaced: {replaced_status}, replaced_by: {replaced_by}")
    else:
        logger.warning("⚠️ Original chunk not found in results")
    
    if original_in_df and replaced_status and 'is_replaced' in df.columns:
        logger.info("✅ Test successful: Original chunk marked as replaced!")
        return True
    else:
        logger.error("❌ Test failed: Chunk replacement not properly applied")
        return False

if __name__ == "__main__":
    result = test_direct()
    logger.info(f"🏁 Direct test {'succeeded' if result else 'failed'}") 