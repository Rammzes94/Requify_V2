#!/usr/bin/env python
"""
stable_save_to_lancedb.py

This script processes parsed document data and saves it to LanceDB for efficient retrieval.
It performs the following operations:
1. Takes a combined JSON file containing parsed PDF content
2. Performs deduplication checks against existing content in the database
3. Computes text embeddings for each page using a multilingual embedding model
4. Prepends necessary instruction contexts for proper embedding generation
5. Identifies and handles both new pages and updates to existing pages
6. Creates vector search indices for similarity-based retrieval
7. Saves both text content and image data (as base64) to LanceDB

The script uses the pre_save_deduplication module to avoid storing duplicate content
and organizes data for efficient semantic search in downstream processing steps.

Dependencies:
    pip install sentence-transformers lancedb python-dotenv
"""

import os
import sys
import glob
import json
import time
import logging
from datetime import datetime
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import lancedb
from lancedb.pydantic import LanceModel, Vector
from typing import List, Optional, Dict, Any

# -------------------------------------------------------------------------------------
# Project Setup
# -------------------------------------------------------------------------------------

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import _00_utils
_00_utils.setup_project_directory()

# Import the deduplication module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '_03_docs_deduplication')))
try:
    import pre_save_deduplication as dedup
except ImportError:
    print("Warning: Could not import pre_save_deduplication module. Duplicate detection will be disabled.")
    dedup = None

# Load environment variables
load_dotenv()

# Setup logging with script prefix
class ScriptLogger(logging.LoggerAdapter):
    def __init__(self, logger, prefix):
        super().__init__(logger, {})
        self.prefix = prefix
        
    def process(self, msg, kwargs):
        return f"{self.prefix}{msg}", kwargs

logger = ScriptLogger(_00_utils.setup_logging(), "[Save_To_LanceDB] ")

# -------------------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------------------
OUTPUT_DIR_BASE = "_03_output" # Define base output directory
PARSED_CONTENT_DIR = os.path.join(OUTPUT_DIR_BASE, "parsed_content") # Read from _03_output
LANCEDB_SUBDIR_NAME = "lancedb" # Subdirectory for LanceDB within _03_output
LANCEDB_TABLE_NAME = "documents"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
EMBEDDING_DIMENSION = 1024 # Dimension for e5-large models
DUPLICATE_SIMILARITY_THRESHOLD = 0.99 # Similarity threshold for considering pages duplicates

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
            logger.error(f"Error reading markdown file {md_path}: {e}", extra={"icon": "❌"})
    return ""

def process_document(document_path, text_embedder):
    """
    Processes a single document's combined JSON file and prepares it for LanceDB.
    Performs deduplication checks before preparing the records.
    
    Args:
        document_path: Path to the combined JSON file
        text_embedder: SentenceTransformer model for creating embeddings
        
    Returns:
        Tuple of (all_records, new_records, update_records) where:
            all_records: All records from the document
            new_records: Records that are new (not duplicates)
            update_records: Records that should update existing records
    """
    logger.info(f"Processing document: {document_path}", extra={"icon": "🔄"})
    
    try:
        # Load the combined JSON file
        with open(document_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading combined JSON file {document_path}: {e}", extra={"icon": "❌"})
        return [], [], []
    
    folder_path = os.path.dirname(document_path)
    pdf_id = data.get("pdf_identifier", os.path.basename(folder_path))
    
    # Get the first page to extract document_title, as it should be the same for all pages
    first_page_key = next(iter(data.get("pages", {})), None)
    document_title = None
    if first_page_key:
        document_title = data["pages"][first_page_key].get("document_title", "")
    
    # Process each page and prepare for embedding
    all_records = []
    for page_key, info in data.get("pages", {}).items():
        md = load_markdown(folder_path, page_key)
        
        # Use markdown content if available, otherwise fallback to summary
        text_to_embed = md if md else info.get("summary", "")
        if not text_to_embed:
            logger.warning(f"No text content found for {pdf_id}, page {page_key}. Using placeholder.", extra={"icon": "⚠️"})
            text_embedding_vector = [0.0] * EMBEDDING_DIMENSION # Placeholder for missing text
        else:
            # Prepend "passage: " as required by the e5-instruct model for document embeddings
            instruction_text = f"passage: {text_to_embed}"
            try:
                text_embedding_vector = text_embedder.encode(instruction_text).tolist()
            except Exception as e:
                logger.error(f"Error encoding text for {pdf_id}, page {page_key}: {e}", extra={"icon": "❌"})
                text_embedding_vector = [0.0] * EMBEDDING_DIMENSION # Placeholder on error
        
        # Create the record with all necessary fields
        record = {
            "pdf_identifier": pdf_id,
            "page_number": info.get("page_number"),
            "document_title": document_title or info.get("document_title", ""),
            "summary": info.get("summary"),
            "hashtags": info.get("hashtags"),
            "md_content": md,
            "input_tokens": info.get("input_tokens"),
            "output_tokens": info.get("output_tokens"),
            "processing_duration": info.get("processing_duration"),
            "error_flag": info.get("error_flag"),
            "timestamp": info.get("timestamp", datetime.now().isoformat()),
            "embedding": text_embedding_vector,
            "image_b64": info.get("image_b64")
        }
        all_records.append(record)
    
    if not all_records:
        logger.warning(f"No valid records found in {document_path}", extra={"icon": "⚠️"})
        return [], [], []
    
    # Perform deduplication check if the module is available
    if dedup:
        logger.info(f"Checking for duplicates in {len(all_records)} pages...", extra={"icon": "🔍"})
        dedup_results = dedup.check_new_document(all_records)
        
        # Extract results
        duplicate_pages = dedup_results.get('duplicate_pages', {})
        new_page_indices = dedup_results.get('new_pages', [])
        update_page_indices = dedup_results.get('update_pages', {})
        is_new_version = dedup_results.get('is_new_version', False)
        old_version_id = dedup_results.get('old_version_id', None)
        
        # Create lists of records to add or update
        new_records = [all_records[i] for i in new_page_indices]
        update_records = {i: all_records[i] for i in update_page_indices}
        
        # Log the results
        logger.info(f"Deduplication results: {len(new_records)} new pages, {len(duplicate_pages)} duplicates, {len(update_records)} updates", extra={"icon": "📊"})
        
        if is_new_version:
            logger.info(f"Document appears to be a new version of {old_version_id}", extra={"icon": "🔄"})
        
        return all_records, new_records, update_records
    else:
        # If deduplication module not available, treat all as new
        logger.warning("Deduplication module not available. Treating all pages as new.", extra={"icon": "⚠️"})
        return all_records, all_records, {}

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
# Main Execution Functions
# -------------------------------------------------------------------------------------

def create_or_open_table(db, table_name):
    """Create or open a LanceDB table with the PDFPage schema."""
    if table_name in db.table_names():
        logger.info(f"Opening existing table: {table_name}", extra={"icon": "📂"})
        return db.open_table(table_name)
    else:
        logger.info(f"Creating new table: {table_name}", extra={"icon": "🆕"})
        return db.create_table(table_name, schema=PDFPage)

def save_document_to_lancedb(document_path):
    """
    Process a single document and save it to LanceDB, handling duplicates.
    
    Args:
        document_path: Path to the combined JSON file
        
    Returns:
        True if successful, False otherwise
    """
    # Initialize SentenceTransformer for text embeddings
    try:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}", extra={"icon": "🧠"})
        
        # Set up sentence-transformers logging to include our script prefix
        st_logger = logging.getLogger('sentence_transformers')
        for handler in st_logger.handlers:
            st_logger.removeHandler(handler)
        st_logger.addHandler(logging.StreamHandler())
        st_logger.setLevel(logging.INFO)
        # Wrap the sentence-transformers logger with our ScriptLogger
        st_logger = ScriptLogger(st_logger, "[Save_To_LanceDB] ")
        
        text_embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("Model loaded successfully", extra={"icon": "✅"})
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}", extra={"icon": "❌"})
        return False

    # Process the document and get records
    all_records, new_records, update_records = process_document(document_path, text_embedder)
    
    if not all_records:
        logger.warning("No records to save", extra={"icon": "⚠️"})
        return False
    
    # Create path to LanceDB
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    lancedb_path = os.path.join(project_root, OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME)
    os.makedirs(lancedb_path, exist_ok=True)
    
    # Connect to LanceDB
    try:
        logger.info(f"Connecting to LanceDB at: {lancedb_path}", extra={"icon": "🔌"})
        db = lancedb.connect(lancedb_path)
        logger.info("Connected to LanceDB", extra={"icon": "✅"})
    except Exception as e:
        logger.error(f"Error connecting to LanceDB: {e}", extra={"icon": "❌"})
        return False
    
    # Create or open the table
    table = create_or_open_table(db, LANCEDB_TABLE_NAME)
    
    # Add new records
    if new_records:
        try:
            logger.info(f"Adding {len(new_records)} new pages to LanceDB", extra={"icon": "➕"})
            table.add(new_records)
            logger.info("New pages added successfully", extra={"icon": "✅"})
        except Exception as e:
            logger.error(f"Error adding new pages: {e}", extra={"icon": "❌"})
            return False
    
    # Update existing records
    if update_records:
        try:
            logger.info(f"Updating {len(update_records)} existing pages in LanceDB", extra={"icon": "🔄"})
            
            # For LanceDB 0.22.0 we're having issues with delete operations
            # Just add the new versions as additional records
            # When querying later, we'll sort by timestamp and get the latest
            update_list = [record for idx, record in update_records.items()]
            
            if update_list:
                logger.info(f"Adding {len(update_list)} updated pages to LanceDB", extra={"icon": "➕"})
                table.add(update_list)
                logger.info("Updated pages added successfully", extra={"icon": "✅"})
            
            logger.info("Updates completed successfully", extra={"icon": "✅"})
        except Exception as e:
            logger.error(f"Error updating pages: {e}", extra={"icon": "❌"})
            # Continue even if updates fail
    
    # Create an index if we have enough data
    try:
        record_count = len(table.to_pandas())
        if record_count > 100:  # Only create index with sufficient data
            logger.info("Creating vector search index", extra={"icon": "🔍"})
            table.create_index(vector_column_name="embedding", replace=True)
            logger.info("Index created successfully", extra={"icon": "✅"})
    except Exception as e:
        logger.error(f"Error creating index: {e}", extra={"icon": "❌"})
        # Index failure is not critical, continue
    
    return True

def main(document_path=None):
    """
    Main function to save documents to LanceDB.
    
    Args:
        document_path: Path to a specific document to process. This is now a required argument.
        
    Returns:
        True if successful, False otherwise
    """
    start_time = time.time()
    
    if not document_path:
        logger.error("❌ No document path provided to stable_save_to_lancedb.py. This is a required argument.")
        return False

    # Process the specified document
    logger.info(f"Processing specified document: {document_path}", extra={"icon": "📄"})
    success = save_document_to_lancedb(document_path)
    
    end_time = time.time()
    logger.info(f"Processing completed in {end_time - start_time:.2f} seconds", extra={"icon": "⏱️"})
    
    return success

if __name__ == "__main__":
    logger.info("Starting document processing and LanceDB saving", extra={"icon": "🚀"})
    
    # Check if a specific document path is provided as an argument
    if len(sys.argv) > 1:
        specific_document = sys.argv[1]
        logger.info(f"Using document path from command line: {specific_document}", extra={"icon": "📝"})
        success = main(specific_document)
    else:
        # Process the latest document
        # success = main() # This mode is deprecated
        logger.error("❌ This script now requires a document path as a command line argument or to be called with main(document_path).")
        success = False
    
    if success:
        logger.info("Document processing and saving completed successfully", extra={"icon": "✅"})
    else:
        logger.error("Document processing and saving failed", extra={"icon": "❌"})


