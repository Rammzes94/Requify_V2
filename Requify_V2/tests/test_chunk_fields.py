"""
test_chunk_fields.py

This script tests the save_chunks_to_db function from context_aware_chunking.py
to verify that the is_replaced and replaced_by fields are properly added to the database.
"""

import os
import sys
import pandas as pd
import lancedb
import time
import logging
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

# Import the function we want to test
from _02_src._02_parsing.context_aware_chunking import save_chunks_to_db

# Constants
OUTPUT_DIR = "_03_output"
LANCEDB_SUBDIR = "lancedb"
CHUNKS_TABLE_NAME = "document_chunks"

def clean_test_db():
    """Clean the test database by removing and recreating the chunks table."""
    # Get the full path to the LanceDB directory
    lancedb_path = os.path.join(project_root, OUTPUT_DIR, LANCEDB_SUBDIR)
    
    logger.info(f"🧹 Cleaning test database at: {lancedb_path}")
    
    try:
        # Make sure the database directory exists
        logger.info(f"📁 Creating directory: {lancedb_path}")
        os.makedirs(lancedb_path, exist_ok=True)
        
        # Connect to database
        logger.info(f"🔌 Connecting to database at: {lancedb_path}")
        db = lancedb.connect(lancedb_path)
        
        # Drop the table if it exists
        table_names = db.table_names()
        logger.info(f"📋 Existing tables: {table_names}")
        
        if CHUNKS_TABLE_NAME in table_names:
            logger.info(f"🗑️ Dropping existing {CHUNKS_TABLE_NAME} table")
            db.drop_table(CHUNKS_TABLE_NAME)
        
        logger.info(f"✅ Test database prepared")
        return db
    except Exception as e:
        logger.error(f"❌ Error preparing test database: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_test_chunks():
    """Create test chunks to demonstrate replacement functionality."""
    # Original chunks from first document
    original_chunks = [
        {
            "chunk_id": "doc1_chunk_1",
            "document_id": "doc1.pdf",
            "chunk_index": 0,
            "start_offset": 0,
            "end_offset": 100,
            "chunk_text": "This is the first test chunk",
            "token_count": 25,
            "embedding": [0.1] * 10,  # Simplified embedding
            "chunk_hash": "hash1",
            "is_duplicate": False,
            "duplicate_of": "",
            "is_updated": False,
            "previous_chunk_id": "",
            "timestamp": "2023-01-01T00:00:00.000Z",
            "aligned_with_chunk_id": "",
            "aligned_with_document_id": "",
            "is_duplicate_marker": False,
            "is_replaced": False,
            "replaced_by": ""
        },
        {
            "chunk_id": "doc1_chunk_2",
            "document_id": "doc1.pdf",
            "chunk_index": 1,
            "start_offset": 101,
            "end_offset": 200,
            "chunk_text": "This is the second test chunk",
            "token_count": 25,
            "embedding": [0.2] * 10,  # Simplified embedding
            "chunk_hash": "hash2",
            "is_duplicate": False,
            "duplicate_of": "",
            "is_updated": False,
            "previous_chunk_id": "",
            "timestamp": "2023-01-01T00:00:00.000Z",
            "aligned_with_chunk_id": "",
            "aligned_with_document_id": "",
            "is_duplicate_marker": False,
            "is_replaced": False,
            "replaced_by": ""
        }
    ]
    
    # Replacement chunks from second document
    replacement_chunks = [
        {
            "chunk_id": "doc2_chunk_1",
            "document_id": "doc2.pdf",
            "chunk_index": 0,
            "start_offset": 0,
            "end_offset": 120,
            "chunk_text": "This is the first test chunk with updates",
            "token_count": 30,
            "embedding": [0.15] * 10,  # Similar but different embedding
            "chunk_hash": "hash3",
            "is_duplicate": False,
            "duplicate_of": "",
            "is_updated": True,
            "previous_chunk_id": "doc1_chunk_1",  # Reference to original
            "timestamp": "2023-01-02T00:00:00.000Z",
            "aligned_with_chunk_id": "doc1_chunk_1",
            "aligned_with_document_id": "doc1.pdf",
            "is_duplicate_marker": False,
            "is_replaced": False,
            "replaced_by": ""
        }
    ]
    
    replaced_chunks = {
        "doc1_chunk_1": "doc2_chunk_1"  # Map original -> replacement
    }
    
    logger.info(f"🧩 Created test chunks: {len(original_chunks)} original, {len(replacement_chunks)} replacement chunks")
    logger.info(f"🔄 Chunk replacement map: {replaced_chunks}")
    
    return original_chunks, replacement_chunks, replaced_chunks

def test_chunk_replacement():
    """Test the chunk replacement functionality."""
    # Clean database
    logger.info("🚀 Starting test_chunk_replacement")
    db = clean_test_db()
    if not db:
        logger.error("❌ Failed to prepare database")
        return False
    
    # Create test chunks
    logger.info("🏗️ Creating test chunks")
    original_chunks, replacement_chunks, replaced_chunks = create_test_chunks()
    
    # First save original chunks
    try:
        logger.info("💾 Saving original chunks")
        success = save_chunks_to_db(original_chunks)
        if not success:
            logger.error("❌ Failed to save original chunks")
            return False
        
        # Display current state
        logger.info("📊 Retrieving chunks from database")
        chunks_table = db.open_table(CHUNKS_TABLE_NAME)
        df = chunks_table.to_pandas()
        logger.info(f"✅ Original chunks saved: {len(df)} rows")
        logger.info(f"📋 Original chunks state:\n{df[['chunk_id', 'is_replaced', 'replaced_by']] if 'is_replaced' in df.columns else df[['chunk_id']]}")
        
        # Save replacement chunks with replacement mapping
        logger.info("🔄 Saving replacement chunks with replacement mapping")
        success = save_chunks_to_db(replacement_chunks, replaced_chunks)
        if not success:
            logger.error("❌ Failed to save replacement chunks")
            return False
        
        # Display final state
        logger.info("📊 Retrieving final state from database")
        chunks_table = db.open_table(CHUNKS_TABLE_NAME)
        df = chunks_table.to_pandas()
        logger.info(f"✅ Final state: {len(df)} rows")
        logger.info(f"📋 After replacement:\n{df[['chunk_id', 'is_replaced', 'replaced_by']] if 'is_replaced' in df.columns else df[['chunk_id']]}")
        
        # Verify the replacement was applied correctly
        if 'is_replaced' in df.columns:
            # Check if original chunk is marked as replaced
            replaced_row = df[df['chunk_id'] == 'doc1_chunk_1']
            if not replaced_row.empty and replaced_row.iloc[0]['is_replaced'] == True:
                logger.info("✅ Original chunk correctly marked as replaced")
                logger.info(f"🔄 Replacement correctly applied: {replaced_row.iloc[0]['chunk_id']} is_replaced={replaced_row.iloc[0]['is_replaced']} replaced_by={replaced_row.iloc[0]['replaced_by']}")
                return True
            else:
                logger.error("❌ Original chunk not properly marked as replaced")
                return False
        else:
            logger.error("❌ is_replaced column not found in database")
            return False
    except Exception as e:
        logger.error(f"❌ Error during test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = test_chunk_replacement()
    logger.info(f"🏁 Test {'succeeded' if result else 'failed'}") 