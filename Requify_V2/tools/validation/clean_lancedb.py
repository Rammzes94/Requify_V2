"""
clean_lancedb.py

This script cleans the LanceDB database by removing all tables
to start with a fresh database for testing purposes.
"""

import os
import sys
import shutil
import logging
import time
from dotenv import load_dotenv

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import _02_src._00_utils as _00_utils
_00_utils.setup_project_directory()

# Load environment variables
load_dotenv()

# Setup logging
logger = _00_utils.setup_logging()

# Constants
OUTPUT_DIR = "_03_output"
LANCEDB_SUBDIR = "lancedb"

def clean_lancedb():
    """
    Clean the LanceDB database by removing the database directory.
    Creates a backup before deletion.
    """
    # Get the full path to the LanceDB directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lancedb_path = os.path.join(script_dir, OUTPUT_DIR, LANCEDB_SUBDIR)
    
    logger.info(f"🧹 Cleaning LanceDB database at: {lancedb_path}")
    
    # Check if the database exists
    if not os.path.exists(lancedb_path):
        logger.info(f"✅ No existing database found at {lancedb_path}. Nothing to clean.")
        return True
    
    # Create a backup with timestamp
    backup_name = f"{LANCEDB_SUBDIR}_backup_{int(time.time())}"
    backup_path = os.path.join(script_dir, OUTPUT_DIR, backup_name)
    
    try:
        # Create backup
        logger.info(f"📦 Creating backup at: {backup_path}")
        shutil.copytree(lancedb_path, backup_path)
        
        # Remove the database directory
        logger.info(f"🗑️ Removing LanceDB directory: {lancedb_path}")
        shutil.rmtree(lancedb_path)
        
        # Create empty directory
        os.makedirs(lancedb_path, exist_ok=True)
        
        logger.info(f"✅ Successfully cleaned LanceDB database. Backup created at: {backup_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Error cleaning LanceDB database: {str(e)}")
        return False

if __name__ == "__main__":
    result = clean_lancedb()
    print(f"Database cleanup {'successful' if result else 'failed'}") 