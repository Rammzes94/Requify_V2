import lancedb
import os # Import os module
import pandas as pd
# Define the output directory
OUTPUT_DIR_BASE = "03_output"
LANCEDB_SUBDIR_NAME = "lancedb"

# Construct the absolute path to the LanceDB directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".")) # Assuming util is in project root
lancedb_path = os.path.join(project_root, OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME)

# Ensure the base output directory exists for Excel files
excel_output_dir = os.path.join(project_root, OUTPUT_DIR_BASE)
os.makedirs(excel_output_dir, exist_ok=True)


# Open your LanceDB
print(f"Attempting to connect to LanceDB at: {lancedb_path}")
db = lancedb.connect(lancedb_path)

# List all tables
print("Tables:", db.table_names())

# Load tables if they exist
table_names = db.table_names()
# Create Excel writer object for single file with multiple sheets
excel_path = os.path.join(excel_output_dir, "lancedb_tables.xlsx")
with pd.ExcelWriter(excel_path) as writer:
    
    if "all_pdf_pages" in table_names:
        table = db.open_table("all_pdf_pages")
        df1 = table.to_pandas()
        df1.to_excel(writer, sheet_name='all_pdf_pages', index=False)
        print(f"Saved all_pdf_pages table to Excel sheet")
    else:
        print("Table 'all_pdf_pages' not found")

    if "requirements" in table_names:
        table2 = db.open_table("requirements")
        df2 = table2.to_pandas()
        df2.to_excel(writer, sheet_name='requirements', index=False)
        print(f"Saved requirements table to Excel sheet")
    else:
        print("Table 'requirements' not found")

print(f"Saved all tables to Excel file: {excel_path}")
