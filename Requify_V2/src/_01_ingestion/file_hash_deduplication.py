"""
file_hash_deduplication.py

This script provides content-based file deduplication for the document processing pipeline.
It performs the following operations:
1. Calculates cryptographic hashes (MD5 and SHA256) of input files
2. Checks if files with identical hashes already exist in the database
3. Stores file hash information in the LanceDB file_hashes table
4. Prevents duplicate document processing at the earliest pipeline stage
5. Provides detailed information about duplicate files including their original paths

The script serves as the first line of defense against redundant processing,
identifying exact duplicates before any costly parsing or embedding operations.
"""

import os
import sys
import hashlib
import logging
import time
from datetime import datetime
from typing import Tuple, Optional, Dict, Any
import lancedb
from dotenv import load_dotenv

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import _00_utils
_00_utils.setup_project_directory()

# Load environment variables
load_dotenv()

# Setup logging with script prefix
logger = _00_utils.get_logger("Hash_Deduplication")

# -------------------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------------------
OUTPUT_DIR_BASE = "output"
LANCEDB_SUBDIR_NAME = "lancedb"
FILE_HASHES_TABLE = "file_hashes"
BUFFER_SIZE = 65536  # Read file in 64kb chunks for hashing

# -------------------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------------------
def connect_to_lancedb():
    """Connect to LanceDB and return the connection."""
    # Construct path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
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

def get_file_hashes(file_path: str) -> Tuple[str, str]:
    """
    Calculate MD5 and SHA256 hashes for a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Tuple of (md5_hash, sha256_hash)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    
    try:
        with open(file_path, 'rb') as f:
            while True:
                data = f.read(BUFFER_SIZE)
                if not data:
                    break
                md5.update(data)
                sha256.update(data)
        
        return md5.hexdigest(), sha256.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating file hashes for {file_path}: {e}", extra={"icon": "❌"})
        raise

def check_file_hash_in_db(db, md5_hash: str, sha256_hash: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a file with the given hashes already exists in the database.
    
    Args:
        db: LanceDB connection
        md5_hash: MD5 hash of the file
        sha256_hash: SHA256 hash of the file
        
    Returns:
        Tuple of (is_duplicate, existing_file_path)
    """
    # Check if the table exists
    if FILE_HASHES_TABLE not in db.table_names():
        logger.info(f"File hashes table {FILE_HASHES_TABLE} does not exist yet", extra={"icon": "ℹ️"})
        return False, None
    
    try:
        # Open the table
        table = db.open_table(FILE_HASHES_TABLE)
        
        # For LanceDB 0.22.0, use pandas filtering
        df = table.to_pandas()
        
        if df.empty:
            logger.info("File hashes table is empty", extra={"icon": "ℹ️"})
            return False, None
        
        # First check SHA256 hash (more reliable)
        sha_matches = df[df['sha256_hash'] == sha256_hash]
        if not sha_matches.empty:
            existing_file = sha_matches.iloc[0]['file_path']
            logger.info(f"Found SHA256 hash match: {existing_file}", extra={"icon": "🔍"})
            return True, existing_file
        
        # Then check MD5 hash (potential for collisions but unlikely)
        md5_matches = df[df['md5_hash'] == md5_hash]
        if not md5_matches.empty:
            existing_file = md5_matches.iloc[0]['file_path']
            logger.info(f"Found MD5 hash match: {existing_file}", extra={"icon": "🔍"})
            return True, existing_file
        
        return False, None
    except Exception as e:
        logger.error(f"Error checking file hash in database: {e}", extra={"icon": "❌"})
        # In case of error, conservatively return no duplicate
        return False, None

def save_file_hash_to_db(db, file_path: str, md5_hash: str, sha256_hash: str) -> bool:
    """
    Save file hash information to the database.
    
    Args:
        db: LanceDB connection
        file_path: Path to the file
        md5_hash: MD5 hash of the file
        sha256_hash: SHA256 hash of the file
        
    Returns:
        True if successful, False otherwise
    """
    # Check if the table exists
    is_new_table = FILE_HASHES_TABLE not in db.table_names()
    
    try:
        if is_new_table:
            logger.info(f"Creating new file hashes table: {FILE_HASHES_TABLE}", extra={"icon": "🆕"})
            
            # Import the FileHash model for the schema
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '_00_lancedb_admin')))
            from init_lancedb import FileHash
            
            # Create the table
            table = db.create_table(FILE_HASHES_TABLE, schema=FileHash)
        else:
            # Open existing table
            table = db.open_table(FILE_HASHES_TABLE)
        
        # Prepare record
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        record = {
            "file_path": file_path,
            "file_name": file_name,
            "file_size": file_size,
            "md5_hash": md5_hash,
            "sha256_hash": sha256_hash,
            "timestamp": timestamp
        }
        
        # Add record to table
        table.add([record])
        
        logger.info(f"Successfully saved file hash information for {file_path}", extra={"icon": "✅"})
        return True
    except Exception as e:
        logger.error(f"Error saving file hash to database: {e}", extra={"icon": "❌"})
        return False

# -------------------------------------------------------------------------------------
# Main Functions
# -------------------------------------------------------------------------------------
def check_file_duplicate(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a file is a duplicate based on its content hash.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        Tuple of (is_duplicate, existing_file_path)
    """
    start_time = time.time()
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}", extra={"icon": "❌"})
        return False, None
    
    logger.info(f"Checking for duplicate of file: {file_path}", extra={"icon": "🔍"})
    
    try:
        # Calculate file hashes
        md5_hash, sha256_hash = get_file_hashes(file_path)
        logger.info(f"Calculated hashes: MD5={md5_hash}, SHA256={sha256_hash}", extra={"icon": "🔢"})
        
        # Connect to LanceDB
        db = connect_to_lancedb()
        if not db:
            logger.error("Failed to connect to LanceDB database", extra={"icon": "❌"})
            return False, None
        
        # Check if file is a duplicate
        is_duplicate, existing_file = check_file_hash_in_db(db, md5_hash, sha256_hash)
        
        if is_duplicate:
            logger.info(f"File {file_path} is a duplicate of {existing_file}", extra={"icon": "⏭️"})
            return True, existing_file
        
        # If not a duplicate, save the hash to the database
        save_file_hash_to_db(db, file_path, md5_hash, sha256_hash)
        
        end_time = time.time()
        logger.info(f"File hash check completed in {end_time - start_time:.2f} seconds. File is unique.", extra={"icon": "✅"})
        return False, None
    except Exception as e:
        logger.error(f"Error in file duplication check: {e}", extra={"icon": "❌"})
        # In case of error, conservatively continue processing
        return False, None

if __name__ == "__main__":
    # Simple CLI for testing
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        logger.info(f"Checking file: {file_path}", extra={"icon": "🔍"})
        is_duplicate, existing_file = check_file_duplicate(file_path)
        
        if is_duplicate:
            logger.info(f"RESULT: File is a duplicate of {existing_file}", extra={"icon": "⏭️"})
        else:
            logger.info(f"RESULT: File is unique, hash information saved to database", extra={"icon": "✅"})
    else:
        logger.info("Usage: python file_hash_deduplication.py <file_path>", extra={"icon": "ℹ️"}) 