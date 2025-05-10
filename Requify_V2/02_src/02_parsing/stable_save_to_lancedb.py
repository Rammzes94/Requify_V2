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
OUTPUT_DIR_BASE = "03_output" # Define base output directory
PARSED_CONTENT_DIR = os.path.join(OUTPUT_DIR_BASE, "parsed_content") # Read from 03_output
LANCEDB_SUBDIR_NAME = "lancedb" # Subdirectory for LanceDB within 03_output
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

def process_folder(folder_path, text_embedder):
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
                print(f"Warning: No text content found for {pdf_id}, page {page_key}. Skipping text embedding.")
                text_embedding_vector = [0.0] * EMBEDDING_DIMENSION # Placeholder for missing text
            else:
                # Prepend "passage: " as required by the e5-instruct model for document embeddings
                instruction_text = f"passage: {text_to_embed}"
                try:
                    text_embedding_vector = text_embedder.encode(instruction_text).tolist()
                except Exception as e:
                    print(f"Error encoding text for {pdf_id}, page {page_key}: {e}")
                    text_embedding_vector = [0.0] * EMBEDDING_DIMENSION # Placeholder on error

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
                "embedding":        text_embedding_vector, # This is the text embedding
                "image_b64":        info.get("image_b64") # Add image_b64 back
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
    embedding: Vector(EMBEDDING_DIMENSION) # Text embedding
    image_b64: Optional[str] # Add image_b64 back

# -------------------------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------------------------

# Setup centralized logging
logger = _00_utils.setup_logging()

logger.info(f"Using parsed content directory: {PARSED_CONTENT_DIR}", extra={'icon': '📂'})
if not os.path.isdir(PARSED_CONTENT_DIR):
    logger.error(f"Error: Parsed content directory not found: {PARSED_CONTENT_DIR}", extra={'icon': '❌'})
    sys.exit(1)

logger.info(f"Loading text embedding model: {EMBEDDING_MODEL_NAME}", extra={'icon': '🔄'})
try:
    text_embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
except Exception as e:
    logger.error(f"Error loading sentence transformer model '{EMBEDDING_MODEL_NAME}': {e}", extra={'icon': '❌'})
    logger.error("Please ensure the model is available or install dependencies.", extra={'icon': '❌'})
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
            recs = process_folder(folder_path, text_embedder)
            if recs:
                print(f"Found {len(recs)} records in folder: {folder_path}")
                all_records.extend(recs)

logger.info(f"Total records collected: {len(all_records)}", extra={'icon': '✅'})

# Connect to LanceDB in the 03_output folder
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
lancedb_output_path = os.path.join(project_root, OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME) # Path for lancedb within 03_output
os.makedirs(lancedb_output_path, exist_ok=True)
logger.info(f"Connecting to LanceDB at: {lancedb_output_path}", extra={'icon': '🔗'})
try:
    db = lancedb.connect(lancedb_output_path)
except Exception as e:
    logger.error(f"Error connecting to LanceDB at {lancedb_output_path}: {e}", extra={'icon': '❌'})
    sys.exit(1)

# Check if table exists, then open or create
try:
    existing_tables = db.table_names()
    if LANCEDB_TABLE_NAME in existing_tables:
        print(f"Opening existing table: {LANCEDB_TABLE_NAME}")
        tbl = db.open_table(LANCEDB_TABLE_NAME)
        print(f"Appending {len(all_records)} records...")
        tbl.add(all_records)
        logger.info(f"Successfully appended records to table '{LANCEDB_TABLE_NAME}'.", extra={'icon': '✅'})
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
        # Check if an index already exists for the 'embedding' column
        # LanceDB Python API v0.6.0+ has tbl.list_indexes()
        # For older versions, this might be different or not available.
        # Assuming we want to create/replace text embedding index if it does not fit or is old.
        print(f"Creating/replacing index on TEXT 'embedding' column for table '{LANCEDB_TABLE_NAME}' with {record_count} records...")
        tbl.create_index(vector_column_name="embedding", metric="cosine", replace=True)
        logger.info("Text embedding index successfully created/re-created.", extra={'icon': '✅'})
        print("Text embedding index successfully created/re-created.")
    else:
        print(f"Skipping index creation: Not enough records ({record_count}/{MIN_RECORDS_FOR_PQ} needed) to build PQ index.")
        print("The table will still work for vector searches but might be slower.")
        print("Add more documents to exceed the threshold and then run this script again to create the index.")
        logger.warning("The table will still work for vector searches but might be slower.", extra={'icon': '⚠️'})
        logger.warning("Add more documents to exceed the threshold and then run this script again to create the index.", extra={'icon': '⚠️'})
except Exception as e:
    print(f"Failed to check record count or create index: {e}")


