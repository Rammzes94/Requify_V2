import lancedb
import os
import pandas as pd
import logging
import sys
from dotenv import load_dotenv

"""
LanceDB Viewer Utility

This script connects to a LanceDB database, retrieves all available tables,
and exports each table to a separate sheet in an Excel file for easy viewing.
"""

# Add the parent directory to the system path to allow importing modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Load environment variables
load_dotenv()

# Define important constants
OUTPUT_DIR_BASE = "_03_output"
LANCEDB_SUBDIR_NAME = "lancedb"
EXCEL_FILENAME = "lancedb_tables.xlsx"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Construct the absolute path to the LanceDB directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
lancedb_path = os.path.join(project_root, OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME)

# Ensure the base output directory exists for Excel files
excel_output_dir = os.path.join(project_root, OUTPUT_DIR_BASE)
os.makedirs(excel_output_dir, exist_ok=True)

# Open LanceDB connection
logging.info(f"🔄 Connecting to LanceDB at: {lancedb_path}")
db = lancedb.connect(lancedb_path)

# Get all available tables
table_names = db.table_names()
logging.info(f"✅ Found {len(table_names)} tables: {table_names}")

# Create Excel writer for saving all tables
excel_path = os.path.join(excel_output_dir, EXCEL_FILENAME)
with pd.ExcelWriter(excel_path) as writer:
    if not table_names:
        logging.warning("❌ No tables found in the database")
    else:
        # Save each table to a separate sheet in the Excel file
        for table_name in table_names:
            try:
                table = db.open_table(table_name)
                df = table.to_pandas()
                df.to_excel(writer, sheet_name=table_name, index=False)
                logging.info(f"✅ Saved '{table_name}' table ({len(df)} rows) to Excel sheet")
            except Exception as e:
                logging.error(f"❌ Error saving table '{table_name}': {str(e)}")

logging.info(f"✅ Saved all tables to Excel file: {excel_path}")
