#!/usr/bin/env python
"""
04_collect_and_save_lancedb_single_table.py

This script traverses the specified parsed content directory,
locates combined JSON files (with corresponding Markdown files),
computes embeddings for each page using a specified Hugging Face embedder,
prepends the "passage: " instruction to the text before embedding,
and aggregates all records into a single table in Lance DB.

Dependencies:
    pip install sentence-transformers lancedb python-dotenv
"""

import os
import sys
import glob
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import lancedb
from lancedb.pydantic import LanceModel, Vector
from typing import List, Optional

# -------------------------------------------------------------------------------------
# Project Setup
# -------------------------------------------------------------------------------------

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import _00_utils
_00_utils.setup_project_directory()

# Load environment variables
load_dotenv()

# -------------------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------------------
PARSED_CONTENT_DIR = os.path.join("01_input", "processed", "parsed_content")
LANCEDB_DIR_NAME = "lancedb" # In project root folder
LANCEDB_TABLE_NAME = "all_pdf_pages"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
EMBEDDING_DIMENSION = 1024 # Dimension for e5-large models

# -------------------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------------------

def load_markdown(folder, page_key):
    """Loads markdown content for a given page key from a folder."""
    md_path = os.path.join(folder, f"{page_key}.md")
    if os.path.isfile(md_path):
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error reading markdown file {md_path}: {e}")
    return ""

def process_folder(folder_path, embedder):
    """Processes a single folder containing parsed content."""
    records = []
    for combined_file in glob.glob(os.path.join(folder_path, "*_combined.json")):
        print(f"Processing combined JSON: {combined_file}")
        try:
            with open(combined_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {combined_file}: {e}")
            continue
        except Exception as e:
            print(f"Error opening or reading {combined_file}: {e}")
            continue

        pdf_id = data.get("pdf_identifier", os.path.basename(folder_path)) # Use folder name as fallback ID
        
        # Get the first page to extract document_title, as it should be the same for all pages
        first_page_key = next(iter(data.get("pages", {})), None)
        document_title = None
        if first_page_key:
            document_title = data["pages"][first_page_key].get("document_title", "")

        for page_key, info in data.get("pages", {}).items():
            md = load_markdown(folder_path, page_key)
            # Use markdown content if available, otherwise fallback to summary
            text_to_embed = md if md else info.get("summary", "")
            if not text_to_embed:
                print(f"Warning: No text content found for {pdf_id}, page {page_key}. Skipping embedding.")
                embedding = [0.0] * EMBEDDING_DIMENSION # Placeholder for missing text
            else:
                # Prepend "passage: " as required by the e5-instruct model for document embeddings
                instruction_text = f"passage: {text_to_embed}"
                try:
                    embedding = embedder.encode(instruction_text).tolist()
                except Exception as e:
                    print(f"Error encoding text for {pdf_id}, page {page_key}: {e}")
                    embedding = [0.0] * EMBEDDING_DIMENSION # Placeholder on error


            records.append({
                "pdf_identifier":   pdf_id,
                "page_number":      info.get("page_number"),
                "document_title":   document_title or info.get("document_title", ""),
                "summary":          info.get("summary"),
                "hashtags":         info.get("hashtags"),
                "md_content":       md,
                "input_tokens":     info.get("input_tokens"),
                "output_tokens":    info.get("output_tokens"),
                "processing_duration": info.get("processing_duration"),
                "error_flag":       info.get("error_flag"),
                "timestamp":        info.get("timestamp"),
                "embedding":        embedding
            })
    return records

# -------------------------------------------------------------------------------------
# LanceDB Schema
# -------------------------------------------------------------------------------------

class PDFPage(LanceModel):
    pdf_identifier: str
    page_number: Optional[int] # Allow None if page number isn't reliably extracted
    document_title: Optional[str]
    summary: Optional[str]
    hashtags: Optional[List[str]]
    md_content: Optional[str]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    processing_duration: Optional[float]
    error_flag: Optional[bool]
    timestamp: Optional[str]
    embedding: Vector(EMBEDDING_DIMENSION) # Updated dimension

# -------------------------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------------------------


print(f"Using parsed content directory: {PARSED_CONTENT_DIR}")
if not os.path.isdir(PARSED_CONTENT_DIR):
    print(f"Error: Parsed content directory not found: {PARSED_CONTENT_DIR}")
    sys.exit(1)

print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
try:
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
except Exception as e:
    print(f"Error loading sentence transformer model '{EMBEDDING_MODEL_NAME}': {e}")
    print("Please ensure the model is available or install dependencies.")
    sys.exit(1)

all_records = []
# Use os.walk to recursively find subdirectories
for root, dirs, _ in os.walk(PARSED_CONTENT_DIR):
    # Process only immediate subdirectories of PARSED_CONTENT_DIR
    # Assuming each direct subfolder represents one document's parsed output
    if root == PARSED_CONTENT_DIR:
        for dir_name in dirs:
            folder_path = os.path.join(root, dir_name)
            print(f"Scanning folder: {folder_path}")
            recs = process_folder(folder_path, embedder)
            if recs:
                print(f"Found {len(recs)} records in folder: {folder_path}")
                all_records.extend(recs)



if not all_records:
    print("No records found in any subfolder. Exiting.")
    

print(f"Total records collected: {len(all_records)}")

# Connect to LanceDB in project root folder
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
lancedb_path = os.path.join(project_root, LANCEDB_DIR_NAME)
os.makedirs(lancedb_path, exist_ok=True)
print(f"Connecting to LanceDB at: {lancedb_path}")
try:
    db = lancedb.connect(lancedb_path)
except Exception as e:
    print(f"Error connecting to LanceDB at {lancedb_path}: {e}")
    sys.exit(1)



# Check if table exists, then open or create
try:
    existing_tables = db.table_names()
    if LANCEDB_TABLE_NAME in existing_tables:
        print(f"Opening existing table: {LANCEDB_TABLE_NAME}")
        tbl = db.open_table(LANCEDB_TABLE_NAME)
        print(f"Appending {len(all_records)} records...")
        tbl.add(all_records)
        print(f"Successfully appended records to table '{LANCEDB_TABLE_NAME}'.")
    else:
        print(f"Table '{LANCEDB_TABLE_NAME}' not found. Creating new table.")
        tbl = db.create_table(LANCEDB_TABLE_NAME, schema=PDFPage, mode="overwrite") # mode="overwrite" ensures it starts fresh if somehow name exists but wasn't listed?
        print(f"Adding {len(all_records)} records...")
        tbl.add(all_records)
        print(f"Successfully created table '{LANCEDB_TABLE_NAME}' with {len(all_records)} records.")
except Exception as e:
     print(f"An unexpected error occurred during LanceDB table operations for '{LANCEDB_TABLE_NAME}': {e}")



# Optional: Build index if needed (consider performance impact)
try:
    print(f"Checking record count before creating index...")
    # Get current table record count
    record_count = len(tbl.to_pandas())
    MIN_RECORDS_FOR_PQ = 256  # Minimum records needed for Product Quantization index
    
    if record_count >= MIN_RECORDS_FOR_PQ:
        print(f"Creating index on 'embedding' column for table '{LANCEDB_TABLE_NAME}' with {record_count} records...")
        tbl.create_index(vector_column_name="embedding", metric="cosine", replace=True)
        print("Index successfully created.")
    else:
        print(f"Skipping index creation: Not enough records ({record_count}/{MIN_RECORDS_FOR_PQ} needed) to build PQ index.")
        print("The table will still work for vector searches but might be slower.")
        print("Add more documents to exceed the threshold and then run this script again to create the index.")
except Exception as e:
    print(f"Failed to check record count or create index: {e}")


