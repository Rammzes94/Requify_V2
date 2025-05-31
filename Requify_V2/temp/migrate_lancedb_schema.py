"""
migrate_lancedb_schema.py

This script migrates the LanceDB tables to update their schema to include the document_embedding field.

It performs the following operations:
1. Backs up the existing documents table
2. Creates a new documents table with the updated schema
3. Copies data from the backup to the new table
4. Deletes the backup table when complete

This is a temporary script for one-time schema migration.
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
import lancedb
from lancedb.pydantic import LanceModel, Vector

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import setup_logging, get_logger, update_token_counters, get_token_usage, print_token_usage, reset_token_counters, setup_project_directory, generate_timestamp
from src import config

# Setup logging
logger = get_logger("Migrate_LanceDB_Schema")

# Constants
OUTPUT_DIR_BASE = config.OUTPUT_DIR_BASE
LANCEDB_SUBDIR_NAME = config.LANCEDB_SUBDIR_NAME
EMBEDDING_DIMENSION = config.EMBEDDING_DIMENSION
DOC_EMBEDDING_DIMENSION = config.DOC_EMBEDDING_DIMENSION

# Define the updated schema
class PDFPage(LanceModel):
    pdf_identifier: str
    page_number: Optional[int]
    document_title: Optional[str]
    summary: Optional[str]
    hashtags: Optional[List[str]]
    md_content: Optional[str]
    content: Optional[str]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    processing_duration: Optional[float]
    error_flag: Optional[bool]
    timestamp: Optional[str]
    embedding: Vector(EMBEDDING_DIMENSION)
    document_embedding: Optional[Vector(DOC_EMBEDDING_DIMENSION)]
    image_b64: Optional[str]
    image: Optional[str]

def migrate_documents_table():
    """Migrate the documents table to include the document_embedding field"""
    # Connect to LanceDB
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    lancedb_path = os.path.join(project_root, OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME)
    
    if not os.path.exists(lancedb_path):
        logger.error(f"LanceDB directory not found at {lancedb_path}", extra={"icon": "❌"})
        return False
    
    logger.info(f"Connecting to LanceDB at {lancedb_path}", extra={"icon": "🔄"})
    db = lancedb.connect(lancedb_path)
    
    # Check if documents table exists
    if config.DOCUMENTS_TABLE not in db.table_names():
        logger.info(f"Documents table not found, no migration needed", extra={"icon": "ℹ️"})
        return True
    
    # Create backup table name
    backup_table_name = f"{config.DOCUMENTS_TABLE}_backup_{int(time.time())}"
    
    # Step 1: Create backup of existing table
    logger.info(f"Creating backup of documents table as {backup_table_name}", extra={"icon": "📦"})
    try:
        documents_table = db.open_table(config.DOCUMENTS_TABLE)
        df = documents_table.to_pandas()
        
        if df.empty:
            logger.info("Documents table is empty, no need to back up", extra={"icon": "ℹ️"})
        else:
            # Create backup table
            db.create_table(backup_table_name, data=df)
            logger.info(f"Backup created successfully with {len(df)} records", extra={"icon": "✅"})
    except Exception as e:
        logger.error(f"Failed to create backup: {str(e)}", extra={"icon": "❌"})
        return False
    
    # Step 2: Delete existing table
    try:
        if config.DOCUMENTS_TABLE in db.table_names():
            logger.info(f"Deleting existing documents table", extra={"icon": "🗑️"})
            db.drop_table(config.DOCUMENTS_TABLE)
    except Exception as e:
        logger.error(f"Failed to delete existing table: {str(e)}", extra={"icon": "❌"})
        return False
    
    # Step 3: Create new table with updated schema
    try:
        logger.info(f"Creating new documents table with updated schema", extra={"icon": "🆕"})
        new_table = db.create_table(config.DOCUMENTS_TABLE, schema=PDFPage)
        logger.info(f"Created new documents table with updated schema", extra={"icon": "✅"})
    except Exception as e:
        logger.error(f"Failed to create new table: {str(e)}", extra={"icon": "❌"})
        # Try to restore from backup
        if backup_table_name in db.table_names():
            logger.info(f"Attempting to restore from backup", extra={"icon": "🔄"})
            backup_df = db.open_table(backup_table_name).to_pandas()
            db.create_table(config.DOCUMENTS_TABLE, data=backup_df)
            logger.info(f"Restored from backup", extra={"icon": "✅"})
        return False
    
    # Step 4: Copy data from backup to new table
    if df is not None and not df.empty:
        try:
            logger.info(f"Copying data from backup to new table", extra={"icon": "📋"})
            
            # Add empty document_embedding column if it doesn't exist
            if 'document_embedding' not in df.columns:
                logger.info(f"Adding empty document_embedding column to dataframe", extra={"icon": "➕"})
                # Create empty document embedding arrays with correct dimension
                empty_embedding = np.zeros(DOC_EMBEDDING_DIMENSION)
                df['document_embedding'] = [empty_embedding] * len(df)
            
            # Add the data to the new table
            new_table = db.open_table(config.DOCUMENTS_TABLE)
            new_table.add(df)
            logger.info(f"Copied {len(df)} records to new table", extra={"icon": "✅"})
        except Exception as e:
            logger.error(f"Failed to copy data to new table: {str(e)}", extra={"icon": "❌"})
            return False
    
    # Step 5: Clean up backup (optional)
    keep_backup = True  # Set to False to delete the backup
    if not keep_backup and backup_table_name in db.table_names():
        try:
            logger.info(f"Deleting backup table", extra={"icon": "🗑️"})
            db.drop_table(backup_table_name)
            logger.info(f"Backup table deleted", extra={"icon": "✅"})
        except Exception as e:
            logger.error(f"Failed to delete backup table: {str(e)}", extra={"icon": "⚠️"})
            # Not critical, continue
    
    logger.info(f"Migration completed successfully", extra={"icon": "🎉"})
    return True

def main():
    """Main function to run the migration"""
    start_time = time.time()
    
    # Setup project directory
    setup_project_directory()
    
    # Run migration
    success = migrate_documents_table()
    
    end_time = time.time()
    logger.info(f"Migration script completed in {end_time - start_time:.2f} seconds", extra={"icon": "⏱️"})
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 