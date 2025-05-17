"""
clear_tables.py

This script allows clearing specific tables in the LanceDB database to start fresh
with testing. It provides options to clear individual tables or all tables.
"""

import os
import sys
import lancedb
import logging
import argparse
from dotenv import load_dotenv

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import _00_utils
_00_utils.setup_project_directory()

# Load environment variables
load_dotenv()

# Setup logging with script prefix
class ScriptLogger(logging.LoggerAdapter):
    def __init__(self, logger, prefix):
        super().__init__(logger, {})
        self.prefix = prefix
        
    def process(self, msg, kwargs):
        return f"{self.prefix}{msg}", kwargs

logger = ScriptLogger(_00_utils.setup_logging(), "[LanceDB_Admin] ")

# Constants
OUTPUT_DIR_BASE = "_03_output"
LANCEDB_SUBDIR_NAME = "lancedb"
LANCEDB_PATH = os.path.join(OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME)

def clear_table(db, table_name):
    """
    Clear a specific table by dropping and recreating it.
    
    Args:
        db: LanceDB connection
        table_name: Name of the table to clear
    """
    if table_name not in db.table_names():
        logger.warning(f"⚠️ Table {table_name} does not exist. Nothing to clear.", extra={"icon": "⚠️"})
        return False
    
    try:
        # Get the schema before dropping
        table = db.open_table(table_name)
        schema = table.schema
        
        # Drop the table
        db.drop_table(table_name)
        logger.info(f"✅ Dropped table {table_name}", extra={"icon": "✅"})
        
        # Recreate with same schema but no data
        db.create_table(table_name, schema=schema)
        logger.info(f"✅ Recreated empty table {table_name} with the same schema", extra={"icon": "✅"})
        
        return True
    except Exception as e:
        logger.error(f"❌ Error clearing table {table_name}: {e}", extra={"icon": "❌"})
        return False

def clear_all_tables(db):
    """
    Clear all tables in the database.
    
    Args:
        db: LanceDB connection
    """
    table_names = db.table_names()
    logger.info(f"🔄 Clearing all tables: {table_names}", extra={"icon": "🔄"})
    
    success = True
    for table_name in table_names:
        if not clear_table(db, table_name):
            success = False
    
    return success

def main():
    parser = argparse.ArgumentParser(description="Clear LanceDB tables for fresh testing")
    parser.add_argument("--table", type=str, help="Specific table to clear (leave empty to clear all tables)")
    parser.add_argument("--all", action="store_true", help="Clear all tables")
    
    args = parser.parse_args()
    
    # Connect to LanceDB
    try:
        db = lancedb.connect(LANCEDB_PATH)
        logger.info(f"✅ Connected to LanceDB at {LANCEDB_PATH}", extra={"icon": "✅"})
    except Exception as e:
        logger.error(f"❌ Failed to connect to LanceDB: {e}", extra={"icon": "❌"})
        return 1
    
    # Check if database has any tables
    table_names = db.table_names()
    if not table_names:
        logger.info("ℹ️ Database has no tables. Nothing to clear.", extra={"icon": "ℹ️"})
        return 0
    
    # Clear tables based on arguments
    if args.all:
        success = clear_all_tables(db)
    elif args.table:
        success = clear_table(db, args.table)
    else:
        # Interactive mode if no arguments provided
        print("\nAvailable tables:")
        for i, name in enumerate(table_names, 1):
            print(f"{i}. {name}")
        print(f"{len(table_names) + 1}. ALL TABLES")
        
        while True:
            try:
                choice = int(input("\nSelect table to clear (or 0 to cancel): "))
                if choice == 0:
                    logger.info("ℹ️ Operation cancelled by user", extra={"icon": "ℹ️"})
                    return 0
                elif 1 <= choice <= len(table_names):
                    selected_table = table_names[choice - 1]
                    logger.info(f"🔄 Selected table: {selected_table}", extra={"icon": "🔄"})
                    success = clear_table(db, selected_table)
                    break
                elif choice == len(table_names) + 1:
                    logger.info("🔄 Clearing ALL tables", extra={"icon": "🔄"})
                    success = clear_all_tables(db)
                    break
                else:
                    print(f"Invalid choice. Please enter a number between 0 and {len(table_names) + 1}")
            except ValueError:
                print("Please enter a valid number")
    
    if success:
        logger.info("✅ Table clearing operation completed successfully", extra={"icon": "✅"})
        return 0
    else:
        logger.warning("⚠️ Some operations failed during table clearing", extra={"icon": "⚠️"})
        return 1

if __name__ == "__main__":
    sys.exit(main()) 