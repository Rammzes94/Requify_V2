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
                
                # Save to Excel
                df.to_excel(writer, sheet_name=table_name, index=False)
                logging.info(f"✅ Saved '{table_name}' table ({len(df)} rows) to Excel sheet")
                
                # Save a text summary for each table
                if table_name == "document_chunks":
                    text_path = os.path.join(excel_output_dir, f"{table_name}_summary.txt")
                    with open(text_path, 'w') as f:
                        f.write(f"=== {table_name} Summary ===\n\n")
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
                    
                    logging.info(f"✅ Saved '{table_name}' summary to text file")
                
            except Exception as e:
                logging.error(f"❌ Error saving table '{table_name}': {str(e)}")

logging.info(f"✅ Saved all tables to Excel file: {excel_path}")
print(f"To view detailed chunk info, check: {os.path.join(excel_output_dir, 'document_chunks_summary.txt')}")
