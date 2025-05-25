"""
clear_lancedb.py

This script clears all tables from the LanceDB database located in the
project's output directory. It's used to ensure a clean state for testing
the document processing pipeline.
"""
import os
import sys
import shutil
from dotenv import load_dotenv

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Adjusted path
from src.utils import get_logger, setup_project_directory

# Set up project directory and logging
setup_project_directory()
logger = get_logger("Clear_LanceDB")

# Load environment variables
load_dotenv()

# Constants
OUTPUT_DIR_BASE = "output"
LANCEDB_SUBDIR_NAME = "lancedb"

def clear_lancedb():
    """
    Deletes the LanceDB data directory to clear all tables.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..')) # Adjusted path
    lancedb_path = os.path.join(project_root, OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME)

    logger.info(f"Attempting to clear LanceDB data at: {lancedb_path}", extra={"icon": "🧹"})

    if os.path.exists(lancedb_path):
        try:
            shutil.rmtree(lancedb_path)
            logger.info(f"✅ Successfully deleted LanceDB directory: {lancedb_path}", extra={"icon": "✅"})
            # Recreate the directory structure so lancedb.connect doesn't fail
            os.makedirs(lancedb_path, exist_ok=True)
            logger.info(f"✅ Recreated empty LanceDB directory: {lancedb_path}", extra={"icon": "✅"})

        except Exception as e:
            logger.error(f"❌ Failed to delete LanceDB directory: {e}", extra={"icon": "❌"})
            return False
    else:
        logger.info(f"ℹ️ LanceDB directory not found at {lancedb_path}. No action needed.", extra={"icon": "ℹ️"})
        # Ensure the directory exists for subsequent operations
        os.makedirs(lancedb_path, exist_ok=True)
        logger.info(f"✅ Created empty LanceDB directory: {lancedb_path}", extra={"icon": "✅"})


    return True

if __name__ == "__main__":
    if clear_lancedb():
        logger.info("LanceDB cleared successfully.", extra={"icon": "🏁"})
    else:
        logger.error("Failed to clear LanceDB.", extra={"icon": "❌"}) 