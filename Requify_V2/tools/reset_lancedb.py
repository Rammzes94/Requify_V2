"""
reset_lancedb.py

This script performs a complete reset of all LanceDB tables to ensure schema consistency across the pipeline.
It removes all existing tables and reinitializes them with current schema definitions.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Ensure the project root is the first entry in sys.path so 'config' and other modules can be found
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if sys.path[0] != project_root:
    sys.path.insert(0, project_root)

from src.utils import setup_logging, get_logger, update_token_counters, get_token_usage, print_token_usage, reset_token_counters, setup_project_directory, generate_timestamp
setup_project_directory()

# Load environment variables
load_dotenv()

import lancedb

# Set up logging with script prefix
logger = get_logger("Reset_LanceDB")

# Constants
OUTPUT_DIR_BASE = "output"
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