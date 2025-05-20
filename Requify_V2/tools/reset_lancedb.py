"""
reset_lancedb.py

This script performs a complete reset of all LanceDB tables to ensure schema consistency across the pipeline.
It removes all existing tables and reinitializes them with current schema definitions.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import _00_utils
_00_utils.setup_project_directory()

# Load environment variables
load_dotenv()

import lancedb

# Set up logging with script prefix
logger = _00_utils.get_logger("Reset_LanceDB")

# Constants
OUTPUT_DIR_BASE = "_03_output"
LANCEDB_SUBDIR_NAME = "lancedb"
ALL_TABLES = ["document_chunks", "documents", "file_hashes", "requirements"]

def main():
    # Connect to LanceDB
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    lancedb_path = os.path.join(project_root, OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME)
    
    try:
        logger.info(f"Connecting to LanceDB at {lancedb_path}", extra={"icon": "🔄"})
        db = lancedb.connect(lancedb_path)
        logger.info(f"Connected to LanceDB. Current tables: {db.table_names()}", extra={"icon": "✅"})
        
        # Drop all existing tables
        existing_tables = db.table_names()
        for table_name in existing_tables:
            logger.info(f"Dropping table: {table_name}", extra={"icon": "🗑️"})
            db.drop_table(table_name)
        
        logger.info(f"All tables dropped. Running initialization to create fresh tables.", extra={"icon": "✅"})
        
        # Import and run the initialization script
        sys.path.append(os.path.join(project_root, 'src', '_00_lancedb_admin'))
        from src._00_lancedb_admin import init_lancedb
        init_lancedb.main()
        
        logger.info(f"LanceDB tables reinitialized with consistent schemas.", extra={"icon": "✅"})
        return True
    except Exception as e:
        logger.error(f"Error resetting LanceDB: {e}", extra={"icon": "❌"})
        return False

if __name__ == "__main__":
    main()