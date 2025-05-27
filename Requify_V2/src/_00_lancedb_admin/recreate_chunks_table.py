"""
recreate_chunks_table.py

This script recreates the document_chunks table with the correct schema to match
what is being used in integrated_chunking.py. It helps fix schema mismatches
when the code has evolved with additional fields that aren't yet in the table.
"""

import os
import sys
import lancedb
import logging
from lancedb.pydantic import LanceModel, Vector
from dotenv import load_dotenv
from src import config
from src.utils import setup_logging, get_logger, update_token_counters, get_token_usage, print_token_usage, reset_token_counters, setup_project_directory, generate_timestamp
setup_project_directory()

# Load environment variables
load_dotenv()

# Setup logging with script prefix


logger = get_logger("LanceDB_Admin")

# Constants
OUTPUT_DIR_BASE = "output"
LANCEDB_SUBDIR_NAME = "lancedb"
LANCEDB_PATH = os.path.join(OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME)
EMBEDDING_DIMENSION = config.EMBEDDING_DIMENSION  # Use from config

# Define the schema that matches what integrated_chunking.py is trying to use
class DocumentChunk(LanceModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    start_offset: int
    end_offset: int
    chunk_text: str
    token_count: int
    embedding: Vector(EMBEDDING_DIMENSION)
    chunk_hash: str  # Added field for deduplication
    is_duplicate: bool
    duplicate_of: str
    is_updated: bool
    previous_chunk_id: str
    timestamp: str
    aligned_with_chunk_id: str
    aligned_with_document_id: str

def main():
    """
    Backup the existing table data if possible, then recreate the table with the correct schema.
    """
    logger.info(f"🔄 Starting script to recreate document_chunks table with updated schema...", extra={"icon": "🔄"})
    
    # Connect to LanceDB
    try:
        db = lancedb.connect(LANCEDB_PATH)
        logger.info(f"✅ Connected to LanceDB at {LANCEDB_PATH}", extra={"icon": "✅"})
    except Exception as e:
        logger.error(f"❌ Failed to connect to LanceDB: {e}", extra={"icon": "❌"})
        return
    
    # Check if table exists
    if config.DOCUMENT_CHUNKS_TABLE not in db.table_names():
        logger.info(f"ℹ️ Table {config.DOCUMENT_CHUNKS_TABLE} doesn't exist yet. Creating with correct schema.", extra={"icon": "ℹ️"})
        try:
            db.create_table(config.DOCUMENT_CHUNKS_TABLE, schema=DocumentChunk)
            logger.info(f"✅ Created new table {config.DOCUMENT_CHUNKS_TABLE} with correct schema", extra={"icon": "✅"})
            return
        except Exception as e:
            logger.error(f"❌ Failed to create table: {e}", extra={"icon": "❌"})
            return
    
    # Table exists, so we need to backup, drop, and recreate
    logger.info(f"🔄 Table {config.DOCUMENT_CHUNKS_TABLE} exists. Preparing to recreate with updated schema...", extra={"icon": "🔄"})
    
    # Attempt to backup existing data
    try:
        existing_table = db.open_table(config.DOCUMENT_CHUNKS_TABLE)
        existing_data = existing_table.to_pandas()
        
        if len(existing_data) > 0:
            logger.info(f"📊 Backed up {len(existing_data)} rows from existing table", extra={"icon": "📊"})
        else:
            logger.info(f"ℹ️ Existing table is empty, no data to backup", extra={"icon": "ℹ️"})
            
    except Exception as e:
        logger.warning(f"⚠️ Could not backup existing data: {e}. Proceeding anyway.", extra={"icon": "⚠️"})
        existing_data = None
    
    # Drop the existing table
    try:
        db.drop_table(config.DOCUMENT_CHUNKS_TABLE)
        logger.info(f"✅ Dropped existing table {config.DOCUMENT_CHUNKS_TABLE}", extra={"icon": "✅"})
    except Exception as e:
        logger.error(f"❌ Failed to drop existing table: {e}", extra={"icon": "❌"})
        return
    
    # Create new table with correct schema
    try:
        new_table = db.create_table(config.DOCUMENT_CHUNKS_TABLE, schema=DocumentChunk)
        logger.info(f"✅ Created new table {config.DOCUMENT_CHUNKS_TABLE} with updated schema", extra={"icon": "✅"})
    except Exception as e:
        logger.error(f"❌ Failed to create new table: {e}", extra={"icon": "❌"})
        return
    
    # Try to restore data if we have a backup and it's compatible
    if existing_data is not None and len(existing_data) > 0:
        try:
            # Check what columns we have and what we're missing
            existing_columns = set(existing_data.columns)
            required_columns = set(DocumentChunk.model_fields.keys())
            
            missing_columns = required_columns - existing_columns
            
            # Add missing columns with default values
            for col in missing_columns:
                if col == 'chunk_hash':
                    # Generate hash from chunk_text
                    import hashlib
                    existing_data['chunk_hash'] = existing_data['chunk_text'].apply(
                        lambda x: hashlib.md5(str(x).encode('utf-8')).hexdigest() if x else ''
                    )
                else:
                    # Use empty string for string columns, False for booleans, 0 for numbers
                    if 'id' in col or col in ['timestamp', 'duplicate_of', 'previous_chunk_id', 'aligned_with_chunk_id', 'aligned_with_document_id']:
                        existing_data[col] = ''
                    elif col in ['is_duplicate', 'is_updated']:
                        existing_data[col] = False
                    else:
                        existing_data[col] = 0
            
            # Add data back to the new table
            new_table.add(existing_data)
            logger.info(f"✅ Restored {len(existing_data)} rows to the new table", extra={"icon": "✅"})
            
        except Exception as e:
            logger.error(f"❌ Failed to restore data: {e}", extra={"icon": "❌"})
            logger.warning(f"⚠️ New table was created but without the old data", extra={"icon": "⚠️"})
    
    logger.info(f"✅ Document chunks table recreation completed successfully", extra={"icon": "✅"})

if __name__ == "__main__":
    main() 