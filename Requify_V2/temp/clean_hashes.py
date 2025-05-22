#!/usr/bin/env python3
"""
Utility script to clean/delete specific file hashes from the database.
"""

import os
import sys
import lancedb
import pandas as pd

# Add the parent directory to the system path to allow importing modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import setup_logging, get_logger, update_token_counters, get_token_usage, print_token_usage, reset_token_counters, setup_project_directory, generate_timestamp
setup_project_directory()

# Setup logging with script prefix
logger = get_logger("CleanHashes")

# Constants
OUTPUT_DIR_BASE = "output"
LANCEDB_SUBDIR_NAME = "lancedb"
FILE_HASHES_TABLE = "file_hashes"

def connect_to_lancedb():
    """Connect to LanceDB and return the connection."""
    # Construct path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    lancedb_path = os.path.join(project_root, OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME)
    
    try:
        logger.info(f"Connecting to LanceDB database at {lancedb_path}", extra={"icon": "🔄"})
        os.makedirs(lancedb_path, exist_ok=True)
        db = lancedb.connect(lancedb_path)
        logger.info(f"Connected to LanceDB at {lancedb_path}", extra={"icon": "✅"})
        return db
    except Exception as e:
        logger.error(f"Failed to connect to LanceDB: {e}", extra={"icon": "❌"})
        return None

def list_file_hashes():
    """List all file hashes in the database."""
    db = connect_to_lancedb()
    if not db:
        logger.error("Failed to connect to LanceDB", extra={"icon": "❌"})
        return
    
    if FILE_HASHES_TABLE not in db.table_names():
        logger.warning(f"File hashes table {FILE_HASHES_TABLE} does not exist", extra={"icon": "⚠️"})
        return
    
    try:
        table = db.open_table(FILE_HASHES_TABLE)
        df = table.to_pandas()
        
        if df.empty:
            logger.info("File hashes table is empty", extra={"icon": "ℹ️"})
            return
        
        logger.info(f"Found {len(df)} file hash records:", extra={"icon": "📊"})
        
        for idx, row in df.iterrows():
            logger.info(f"{idx+1}. {row['file_path']} (MD5: {row['md5_hash'][:8]}...)", extra={"icon": "📄"})
    
    except Exception as e:
        logger.error(f"Error listing file hashes: {e}", extra={"icon": "❌"})

def delete_file_hash(file_path=None, md5_hash=None, sha256_hash=None):
    """Delete a file hash record from the database."""
    if not file_path and not md5_hash and not sha256_hash:
        logger.error("Must provide at least one of: file_path, md5_hash, or sha256_hash", extra={"icon": "❌"})
        return False
    
    db = connect_to_lancedb()
    if not db:
        logger.error("Failed to connect to LanceDB", extra={"icon": "❌"})
        return False
    
    if FILE_HASHES_TABLE not in db.table_names():
        logger.warning(f"File hashes table {FILE_HASHES_TABLE} does not exist", extra={"icon": "⚠️"})
        return False
    
    try:
        table = db.open_table(FILE_HASHES_TABLE)
        df = table.to_pandas()
        
        if df.empty:
            logger.info("File hashes table is empty, nothing to delete", extra={"icon": "ℹ️"})
            return False
        
        # Apply filters based on provided parameters
        if file_path:
            df_filtered = df[df['file_path'].str.contains(file_path)]
        elif md5_hash:
            df_filtered = df[df['md5_hash'] == md5_hash]
        elif sha256_hash:
            df_filtered = df[df['sha256_hash'] == sha256_hash]
        
        if df_filtered.empty:
            logger.warning(f"No matching file hash records found", extra={"icon": "⚠️"})
            return False
        
        # Get the indices to delete
        ids_to_delete = df_filtered.index.tolist()
        
        # Display the records to be deleted
        for idx in ids_to_delete:
            row = df.loc[idx]
            logger.info(f"Found record to delete: {row['file_path']} (MD5: {row['md5_hash'][:8]}...)", extra={"icon": "🗑️"})
        
        # Delete records from the table
        table.delete(f"id in {ids_to_delete}")
        
        logger.info(f"Successfully deleted {len(ids_to_delete)} file hash records", extra={"icon": "✅"})
        return True
    
    except Exception as e:
        logger.error(f"Error deleting file hash: {e}", extra={"icon": "❌"})
        return False

def clean_specific_test_files():
    """Clean file hashes for specific test files."""
    test_files = [
        "fighter_jet_rocket_launcher_spec_2_extra_end.pdf",
        "fighter_jet_rocket_launcher_spec_2_changed_values.pdf",
        "fighter_jet_rocket_launcher_spec_3_language_variant.pdf",
        "fighter_jet_rocket_launcher_spec_5_reordered.pdf",
        "fighter_jet_unique_original.pdf",
        "fighter_jet_unique_reordered.pdf"
    ]
    
    for test_file in test_files:
        result = delete_file_hash(file_path=test_file)
        if result:
            logger.info(f"Successfully cleaned hash for {test_file}", extra={"icon": "✅"})
        else:
            logger.warning(f"Failed to clean hash for {test_file}", extra={"icon": "⚠️"})

def main():
    """Main function."""
    if len(sys.argv) == 1:
        # No arguments, just list the hashes
        list_file_hashes()
    elif len(sys.argv) == 2:
        if sys.argv[1] == "--clean":
            # Clean all test files
            clean_specific_test_files()
        else:
            # Delete hash for specific file
            file_path = sys.argv[1]
            delete_file_hash(file_path=file_path)
    elif len(sys.argv) == 3:
        if sys.argv[1] == "--md5":
            # Delete hash for specific MD5
            md5_hash = sys.argv[2]
            delete_file_hash(md5_hash=md5_hash)
        elif sys.argv[1] == "--sha256":
            # Delete hash for specific SHA256
            sha256_hash = sys.argv[2]
            delete_file_hash(sha256_hash=sha256_hash)
        else:
            logger.error("Invalid arguments. Usage: python clean_hashes.py [--md5 HASH | --sha256 HASH | --clean | file_path]", extra={"icon": "❌"})
    else:
        logger.error("Invalid number of arguments. Usage: python clean_hashes.py [--md5 HASH | --sha256 HASH | --clean | file_path]", extra={"icon": "❌"})

if __name__ == "__main__":
    main() 