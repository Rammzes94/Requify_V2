import lancedb
import os
import pandas as pd
import logging
import sys
from dotenv import load_dotenv
from pathlib import Path

"""
LanceDB Viewer Utility

This script connects to LanceDB databases (both main and test),
retrieves all available tables, and exports each table to a 
separate sheet in an Excel file for easy viewing.
"""

# Add the parent directory to the system path to allow importing modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Load environment variables
load_dotenv()

# Define important constants
OUTPUT_DIR_BASE = "_03_output"
LANCEDB_SUBDIR_NAME = "lancedb"
EXCEL_FILENAME = "lancedb_tables.xlsx"
TEST_DB_PATH = os.path.join("tests", "e2e", OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Construct the absolute path to the main LanceDB directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
lancedb_path = os.path.join(project_root, OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME)
test_lancedb_path = os.path.join(project_root, TEST_DB_PATH)

# Ensure the base output directory exists for Excel files
excel_output_dir = os.path.join(project_root, OUTPUT_DIR_BASE)
os.makedirs(excel_output_dir, exist_ok=True)

def process_lancedb(db_path, sheet_prefix=""):
    """Process a LanceDB database and return dataframes for all tables
    
    Args:
        db_path: Path to the LanceDB database
        sheet_prefix: Prefix to add to sheet names (e.g., "test_" for test database)
    
    Returns:
        Dictionary of table_name -> dataframe
    """
    # Check if the database exists
    if not Path(db_path).exists():
        logging.warning(f"⚠️ LanceDB directory not found at: {db_path}")
        return {}
        
    # Open LanceDB connection
    logging.info(f"🔄 Connecting to LanceDB at: {db_path}")
    try:
        db = lancedb.connect(db_path)
    except Exception as e:
        logging.error(f"❌ Failed to connect to LanceDB at {db_path}: {e}")
        return {}

    # Get all available tables
    table_names = db.table_names()
    logging.info(f"✅ Found {len(table_names)} tables in {db_path}: {table_names}")
    
    table_dfs = {}
    
    if not table_names:
        logging.warning(f"❌ No tables found in the database at {db_path}")
    else:
        # Process each table
        for table_name in table_names:
            try:
                table = db.open_table(table_name)
                df = table.to_pandas()
                
                # Check for empty columns
                empty_cols = [col for col in df.columns if df[col].isna().all()]
                if empty_cols:
                    logging.warning(f"⚠️ Table '{table_name}' has empty columns: {empty_cols}")
                
                # Check for NaN values and replace with empty string for better Excel display
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].fillna("")
                
                # Output column info for debugging
                logging.info(f"ℹ️ Table '{table_name}' column info:")
                for col in df.columns:
                    non_null = df[col].count()
                    logging.info(f"  - {col}: {non_null}/{len(df)} non-null values, type: {df[col].dtype}")
                
                # For boolean columns, ensure they're properly formatted for Excel
                bool_cols = df.select_dtypes(include=['bool']).columns
                for col in bool_cols:
                    # Excel can sometimes have issues with boolean values, convert to string
                    df[col] = df[col].map({True: 'True', False: 'False'})
                
                # Add to dictionary with prefix
                sheet_name = f"{sheet_prefix}{table_name}"
                table_dfs[sheet_name] = df
                
                # Save a text summary for each table if it's document_chunks
                if table_name == "document_chunks":
                    text_path = os.path.join(excel_output_dir, f"{sheet_prefix}{table_name}_summary.txt")
                    with open(text_path, 'w') as f:
                        f.write(f"=== {sheet_prefix}{table_name} Summary ===\n\n")
                        f.write(f"Total rows: {len(df)}\n\n")
                        
                        for i, row in df.iterrows():
                            f.write(f"Chunk {i+1}: {row.get('chunk_id', 'Unknown')}\n")
                            f.write(f"  Document: {row.get('document_id', 'Unknown')}\n")
                            
                            # Show relationship information if available
                            if "is_updated" in row and row["is_updated"]:
                                f.write(f"  Updated: {row['is_updated']}\n")
                            if "previous_chunk_id" in row and row["previous_chunk_id"]:
                                f.write(f"  Previous Chunk: {row['previous_chunk_id']}\n")
                            if "is_replaced" in row and row["is_replaced"]:
                                f.write(f"  Is Replaced: {row['is_replaced']}\n")
                            if "replaced_by" in row and row["replaced_by"]:
                                f.write(f"  Replaced By: {row['replaced_by']}\n")
                            
                            # Show first 150 characters of text
                            text = row.get('chunk_text', '')
                            if text:
                                f.write(f"  Text: {text[:150]}...\n")
                            f.write("-" * 80 + "\n")
                    
                    logging.info(f"✅ Saved '{table_name}' summary to text file: {text_path}")
                
            except Exception as e:
                logging.error(f"❌ Error processing table '{table_name}': {str(e)}")
    
    return table_dfs

# Process main database
main_tables = process_lancedb(lancedb_path)

# Process test database
test_tables = process_lancedb(test_lancedb_path, "test_")

# Create Excel writer for saving all tables
excel_path = os.path.join(excel_output_dir, EXCEL_FILENAME)
with pd.ExcelWriter(excel_path) as writer:
    # Write main tables
    for sheet_name, df in main_tables.items():
        try:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            logging.info(f"✅ Saved '{sheet_name}' table ({len(df)} rows) to Excel sheet")
        except Exception as e:
            logging.error(f"❌ Error saving '{sheet_name}' to Excel: {str(e)}")
            
    # Write test tables
    for sheet_name, df in test_tables.items():
        try:
            # Excel sheet names are limited to 31 characters
            if len(sheet_name) > 31:
                short_name = sheet_name[:31]
                logging.warning(f"⚠️ Sheet name '{sheet_name}' truncated to '{short_name}'")
                sheet_name = short_name
                
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            logging.info(f"✅ Saved '{sheet_name}' table ({len(df)} rows) to Excel sheet")
        except Exception as e:
            logging.error(f"❌ Error saving '{sheet_name}' to Excel: {str(e)}")

logging.info(f"✅ Saved all tables to Excel file: {excel_path}")
print(f"To view detailed chunk info, check the *_document_chunks_summary.txt files in {excel_output_dir}")
