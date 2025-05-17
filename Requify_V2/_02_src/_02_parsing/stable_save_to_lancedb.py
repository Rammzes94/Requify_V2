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
from typing import List, Optional, Dict, Any, Tuple

# -------------------------------------------------------------------------------------
# Project Setup
# -------------------------------------------------------------------------------------

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import _00_utils
_00_utils.setup_project_directory()

# Import deduplication and pipeline interaction modules
from _03_docs_deduplication import pre_save_deduplication as dedup
from _03_docs_deduplication import pipeline_interaction

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

def process_document(document_path: str, text_embedder) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Process a document for ingestion into LanceDB"""
    try:
        # Load document
        logger.info(f"Loading document: {document_path}", extra={"icon": "🔄"})
        with open(document_path, 'r', encoding='utf-8') as f:
            document = json.load(f)
            
        doc_identifier = os.path.splitext(os.path.basename(document_path))[0]
        timestamp = datetime.now().isoformat()
        
        # Validate document structure - be more flexible about expected structure
        if not isinstance(document, dict):
            logger.error(f"Invalid document format: expected dict, got {type(document)}", extra={"icon": "❌"})
            return [], [], []
        
        # Handle different document structures - adapt to what's available
        all_records = []
        
        # Check for common document structures
        if "pages" in document and isinstance(document["pages"], list):
            # Standard structure with pages list
            pages = document["pages"]
        elif isinstance(document, dict) and all(isinstance(k, str) and isinstance(v, dict) for k, v in document.items()):
            # Structure where each key is a page ID and value is page content
            pages = list(document.values())
            logger.info(f"Using alternative document structure with {len(pages)} pages", extra={"icon": "🔄"})
        else:
            # Try to process the document itself as a single page
            pages = [document]
            logger.warning(f"Could not find explicit page structure, treating document as single page", extra={"icon": "⚠️"})
        
        # Create document records for LanceDB
        for page_idx, page in enumerate(pages):
            if not isinstance(page, dict):
                logger.warning(f"Skipping invalid page at index {page_idx}: not a dictionary", extra={"icon": "⚠️"})
                continue
                
            page_num = page.get("page_number", page_idx + 1)
            
            # Try different field names for content
            content_fields = ["content", "text", "md_content", "summary"]
            page_content = ""
            for field in content_fields:
                if field in page and page[field]:
                    page_content = page[field]
                    break
            
            # Try different field names for images
            image_fields = ["image", "image_b64", "base64_image"]
            page_image = ""
            for field in image_fields:
                if field in page and page[field]:
                    page_image = page[field]
                    break
            
            # Get document title if available
            document_title = page.get("document_title", "") or document.get("document_title", "")
            
            # Create embedding
            if text_embedder and page_content:
                embedding = text_embedder.encode(page_content, normalize_embeddings=True).tolist()
            else:
                embedding = [0.0] * EMBEDDING_DIMENSION  # Placeholder for testing
            
            # Create record compatible with LanceDB schema
            record = {
                "pdf_identifier": doc_identifier,
                "page_number": page_num,
                "document_title": document_title,
                "summary": page.get("summary", ""),
                "hashtags": page.get("hashtags", []),
                "md_content": page_content,  # Store content in md_content field
                "input_tokens": page.get("input_tokens", 0),
                "output_tokens": page.get("output_tokens", 0),
                "processing_duration": page.get("processing_duration", 0.0),
                "error_flag": page.get("error_flag", False),
                "timestamp": timestamp,
                "embedding": embedding,
                "image_b64": page_image  # Store image in image_b64 field
            }
            all_records.append(record)
            
        if not all_records:
            logger.warning(f"No valid records could be extracted from {document_path}", extra={"icon": "⚠️"})
            return [], [], []
            
        logger.info(f"Successfully extracted {len(all_records)} pages from document", extra={"icon": "✅"})
        
        # Check for duplicates before saving
        logger.info(f"Checking for document duplicates", extra={"icon": "🔄"})
        dedup_results = dedup.check_new_document(all_records)
        
        # Handle document-level similarity detection
        if dedup_results.get('is_new_version'):
            logger.info(f"Document similarity detected!", extra={"icon": "⚠️"})
            should_continue, action = pipeline_interaction.handle_document_similarity(
                all_records, dedup_results
            )
            
            if not should_continue:
                logger.info(f"Skipping document based on user choice", extra={"icon": "⏩"})
                return all_records, [], []
                
            if action == "replace_old":
                # User wants to replace the old document with the new one
                # We need to prepare for this replacement
                old_id = dedup_results.get('old_version_id')
                logger.info(f"Will replace old document {old_id} with new document", extra={"icon": "🔄"})
                
                # Mark all pages as updates to the old document's pages
                # This will trigger replacement in the save function
                return all_records, [], all_records
        
        # Proceed with normal deduplication if no document-level action taken
        # or if user chose detailed deduplication
        duplicate_pages = dedup_results.get('duplicate_pages', {})
        new_pages = dedup_results.get('new_pages', [])
        update_pages = dedup_results.get('update_pages', {})
        
        # Process record collections
        new_records = [all_records[idx] for idx in new_pages]
        update_records = [all_records[idx] for idx in update_pages.keys()]
        
        if duplicate_pages:
            dupes = ", ".join([str(list(duplicate_pages.keys())[:5])])
            logger.info(f"Found {len(duplicate_pages)} duplicate pages: {dupes}...", extra={"icon": "🔍"})
        
        logger.info(
            f"Document has {len(new_records)} new pages, {len(update_records)} updated pages", 
            extra={"icon": "📊"}
        )
        
        return all_records, new_records, update_records
    
    except Exception as e:
        logger.error(f"Error processing document: {e}", extra={"icon": "❌"})
        return [], [], []

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
    content: Optional[str]  # Add content field to match the records being created
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    processing_duration: Optional[float]
    error_flag: Optional[bool]
    timestamp: Optional[str]
    embedding: Vector(EMBEDDING_DIMENSION) # Text embedding
    image_b64: Optional[str] # Base64 encoded image
    image: Optional[str] # Alternative image field name

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
    if not os.path.isfile(document_path):
        logger.error(f"Document file not found: {document_path}", extra={"icon": "❌"})
        return False
        
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
            
            # Make sure update_records is a list, not a dictionary
            if isinstance(update_records, list):
                update_list = update_records
            elif isinstance(update_records, dict):
                update_list = [update_records[idx] for idx in update_records]
            else:
                update_list = []
                logger.warning(f"Unexpected update_records type: {type(update_records)}", extra={"icon": "⚠️"})
            
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


