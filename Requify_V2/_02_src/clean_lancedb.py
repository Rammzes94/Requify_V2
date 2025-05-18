"""
clean_lancedb.py

This script cleans the LanceDB database by removing all tables and data.
Use this script to reset the database before running tests with a fresh environment.
"""

import os
import sys
import shutil
import logging
import argparse
from dotenv import load_dotenv

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import _00_utils
_00_utils.setup_project_directory()

# Load environment variables
load_dotenv()

# Setup logging
logger = _00_utils.setup_logging()

# Constants
OUTPUT_DIR_BASE = "_03_output"
LANCEDB_SUBDIR_NAME = "lancedb"
LANCEDB_TABLES = ["documents", "document_chunks", "requirements", "file_hashes"]

def clean_lancedb(confirm=False, backup=True):
    """
    Clean the LanceDB database by removing all tables and data.
    
    Args:
        confirm: If True, does not prompt for confirmation
        backup: If True, makes a backup before cleaning
    """
    # Get LanceDB path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    lancedb_path = os.path.join(project_root, OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME)
    
    if not os.path.exists(lancedb_path):
        logger.info(f"🔄 LanceDB directory doesn't exist at {lancedb_path}. Nothing to clean.")
        return True
    
    # Confirm with the user
    if not confirm:
        user_input = input(f"⚠️ This will delete all data in the LanceDB database at {lancedb_path}. Are you sure? (y/n): ")
        if user_input.lower() != 'y':
            logger.info("❌ Cleanup aborted by user.")
            return False
    
    # Create backup if requested
    if backup:
        backup_path = f"{lancedb_path}_backup_{int(os.path.getmtime(lancedb_path))}"
        logger.info(f"🔄 Creating backup at {backup_path}")
        try:
            shutil.copytree(lancedb_path, backup_path)
            logger.info(f"✅ Backup created successfully")
        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")
            if not confirm:
                user_input = input("Continue with cleanup without backup? (y/n): ")
                if user_input.lower() != 'y':
                    logger.info("❌ Cleanup aborted by user.")
                    return False
    
    # Clean LanceDB
    try:
        logger.info(f"🔄 Removing LanceDB directory at {lancedb_path}")
        shutil.rmtree(lancedb_path)
        logger.info(f"✅ LanceDB directory removed successfully")
        
        # Create empty directory to ensure the path exists for future operations
        os.makedirs(lancedb_path, exist_ok=True)
        logger.info(f"✅ Created empty LanceDB directory")
        
        return True
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Clean the LanceDB database.')
    parser.add_argument('--force', action='store_true', help='Force cleanup without confirmation')
    parser.add_argument('--no-backup', action='store_true', help='Skip backup creation')
    
    args = parser.parse_args()
    
    return clean_lancedb(confirm=args.force, backup=not args.no_backup)

if __name__ == "__main__":
    sys.exit(0 if main() else 1) 