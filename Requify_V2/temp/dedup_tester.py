#!/usr/bin/env python3
"""
Script to test document deduplication directly without going through the pipeline.
"""

import os
import sys
import json
import logging
import time
import lancedb
import numpy as np

# Add the parent directory to the system path to allow importing modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import _00_utils
_00_utils.setup_project_directory()

# Setup logging with script prefix
logger = _00_utils.get_logger("DedupTester")

# Load the deduplication module
from src._03_docs_deduplication import pre_save_deduplication

# Constants
OUTPUT_DIR_BASE = "output"
LANCEDB_SUBDIR_NAME = "lancedb"
DOCUMENTS_TABLE = "documents"

def load_document(json_path):
    """Load document data from JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading document {json_path}: {e}", extra={"icon": "❌"})
        return None

def prepare_document_data(doc_data):
    """Extract page data from document in the required format."""
    pages_data = []
    pdf_id = doc_data.get('pdf_identifier', 'unknown')
    title = doc_data.get('document_title', '')
    
    # Handle multi-page documents
    if 'pages' in doc_data:
        for page_key, page_info in doc_data.get('pages', {}).items():
            page_data = {
                'pdf_identifier': pdf_id,
                'page_number': page_info.get('page_number', 1),
                'document_title': title,
                'summary': page_info.get('summary', ''),
                'hashtags': page_info.get('hashtags', []),
                'md_content': page_info.get('md_content', ''),
                'embedding': page_info.get('embedding', []),
                'timestamp': page_info.get('timestamp', '')
            }
            pages_data.append(page_data)
    # Handle single-page documents
    else:
        page_data = {
            'pdf_identifier': pdf_id,
            'page_number': 1,
            'document_title': title,
            'summary': doc_data.get('summary', ''),
            'hashtags': doc_data.get('hashtags', []),
            'md_content': doc_data.get('md_content', ''),
            'embedding': doc_data.get('embedding', []),
            'timestamp': doc_data.get('timestamp', '')
        }
        pages_data.append(page_data)
    
    return pages_data

def connect_to_lancedb():
    """Connect to LanceDB and return the connection."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    lancedb_path = os.path.join(project_root, OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME)
    
    try:
        logger.info(f"Connecting to LanceDB database at {lancedb_path}", extra={"icon": "🔄"})
        os.makedirs(lancedb_path, exist_ok=True)
        db = lancedb.connect(lancedb_path)
        logger.info(f"Connected to LanceDB at {lancedb_path}", extra={"icon": "✅"})
        return db
    except Exception as e:
        logger.error(f"Failed to connect to LanceDB: {e}", extra={"icon": "❌"})
        return None

def add_embeddings_to_document(doc_path, db_connection=None):
    """
    Add random embeddings to document pages.
    In a real scenario, these would be generated from the document content.
    """
    logger.info(f"Adding embeddings to document: {doc_path}", extra={"icon": "🔄"})
    
    # Load the document
    doc_data = load_document(doc_path)
    if not doc_data:
        return False
    
    # Get embedding dimension from existing documents if available
    embedding_dim = 768  # Default dimension
    
    db = db_connection or connect_to_lancedb()
    if db and DOCUMENTS_TABLE in db.table_names():
        table = db.open_table(DOCUMENTS_TABLE)
        existing_docs = table.to_pandas()
        if not existing_docs.empty and 'embedding' in existing_docs.columns:
            # Get the first non-empty embedding's length
            for _, row in existing_docs.iterrows():
                embedding = row.get('embedding')
                if embedding is not None and (
                    (isinstance(embedding, list) and len(embedding) > 0) or
                    (isinstance(embedding, np.ndarray) and embedding.size > 0)
                ):
                    embedding_dim = len(embedding) if isinstance(embedding, list) else embedding.size
                    break
    
    # Generate embeddings for pages
    if 'pages' in doc_data:
        for page_key, page_info in doc_data.get('pages', {}).items():
            # Create a non-zero embedding that will be similar to others with the same pattern
            # This is just for testing - in real use, these would be semantic embeddings
            base = np.ones(embedding_dim) * 0.1
            noise = np.random.rand(embedding_dim) * 0.05
            embedding = base + noise
            
            # Normalize to unit length (cosine similarity calculation assumes this)
            embedding = embedding / np.linalg.norm(embedding)
            
            # Add to document
            doc_data['pages'][page_key]['embedding'] = embedding.tolist()
    else:
        # Single page document
        base = np.ones(embedding_dim) * 0.1
        noise = np.random.rand(embedding_dim) * 0.05
        embedding = base + noise
        embedding = embedding / np.linalg.norm(embedding)
        doc_data['embedding'] = embedding.tolist()
    
    # Save the updated document
    with open(doc_path, 'w', encoding='utf-8') as f:
        json.dump(doc_data, f, indent=2)
    
    logger.info(f"Added embeddings to document: {doc_path}", extra={"icon": "✅"})
    return True

def test_deduplication(doc1_path, doc2_path=None, verbose=False):
    """Test deduplication between two documents."""
    logger.info(f"Testing deduplication", extra={"icon": "🔍"})
    
    # Load and prepare documents
    doc1_data = load_document(doc1_path)
    if not doc1_data:
        return False
    
    doc1_pages = prepare_document_data(doc1_data)
    doc1_id = doc1_data.get('pdf_identifier', os.path.basename(doc1_path))
    
    # Add embeddings if needed
    if not doc1_pages[0].get('embedding') or len(doc1_pages[0].get('embedding', [])) == 0:
        logger.info(f"Document {doc1_id} has no embeddings. Adding them now.", extra={"icon": "🔄"})
        add_embeddings_to_document(doc1_path)
        doc1_data = load_document(doc1_path)
        doc1_pages = prepare_document_data(doc1_data)
    
    # Connect to LanceDB
    db = connect_to_lancedb()
    if not db:
        logger.error("Failed to connect to LanceDB. Aborting test.", extra={"icon": "❌"})
        return False
    
    # If second document is provided, load and prepare it
    if doc2_path:
        doc2_data = load_document(doc2_path)
        if not doc2_data:
            return False
        
        doc2_pages = prepare_document_data(doc2_data)
        doc2_id = doc2_data.get('pdf_identifier', os.path.basename(doc2_path))
        
        # Add embeddings if needed
        if not doc2_pages[0].get('embedding') or len(doc2_pages[0].get('embedding', [])) == 0:
            logger.info(f"Document {doc2_id} has no embeddings. Adding them now.", extra={"icon": "🔄"})
            add_embeddings_to_document(doc2_path)
            doc2_data = load_document(doc2_path)
            doc2_pages = prepare_document_data(doc2_data)
        
        # First, save the first document to the database
        # This function should check for duplicates before saving
        logger.info(f"Saving first document {doc1_id} to database", extra={"icon": "💾"})
        
        # Make sure the documents table exists
        if DOCUMENTS_TABLE not in db.table_names():
            from src._00_lancedb_admin.init_lancedb import PDFPage
            db.create_table(DOCUMENTS_TABLE, schema=PDFPage)
            
        # Save doc1 to database
        table = db.open_table(DOCUMENTS_TABLE)
        table.add(doc1_pages)
        
        logger.info(f"Now testing deduplication for second document {doc2_id}", extra={"icon": "🔍"})
        
        # Set verbose flag in pre_save_deduplication if needed
        if verbose:
            pre_save_deduplication.VERBOSE_DEDUPLICATION_OUTPUT = True
        
        # Check if doc2 is a duplicate of doc1
        dedup_results = pre_save_deduplication.check_new_document(doc2_pages)
        
        # Log results
        duplicate_pages = dedup_results.get('duplicate_pages', {})
        new_pages = dedup_results.get('new_pages', [])
        update_pages = dedup_results.get('update_pages', {})
        is_new_version = dedup_results.get('is_new_version', False)
        old_version_id = dedup_results.get('old_version_id', None)
        version_similarity = dedup_results.get('version_similarity', 0.0)
        
        logger.info(f"Deduplication results for {doc2_id}:", extra={"icon": "📊"})
        logger.info(f"- New pages: {len(new_pages)}", extra={"icon": "🆕"})
        logger.info(f"- Duplicate pages: {len(duplicate_pages)}", extra={"icon": "♻️"})
        logger.info(f"- Pages to update: {len(update_pages)}", extra={"icon": "🔄"})
        
        if is_new_version:
            logger.info(f"- Document is a new version of {old_version_id} with similarity {version_similarity:.4f}", extra={"icon": "🔄"})
        elif old_version_id:
            logger.info(f"- Document is related to {old_version_id} with similarity {version_similarity:.4f}", extra={"icon": "ℹ️"})
        
        return True
    else:
        # If only one document is provided, check if it's a duplicate of anything in the database
        logger.info(f"Checking if document {doc1_id} is a duplicate of anything in the database", extra={"icon": "🔍"})
        
        # Set verbose flag in pre_save_deduplication if needed
        if verbose:
            pre_save_deduplication.VERBOSE_DEDUPLICATION_OUTPUT = True
        
        # Check for duplicates
        dedup_results = pre_save_deduplication.check_new_document(doc1_pages)
        
        # Log results
        duplicate_pages = dedup_results.get('duplicate_pages', {})
        new_pages = dedup_results.get('new_pages', [])
        update_pages = dedup_results.get('update_pages', {})
        is_new_version = dedup_results.get('is_new_version', False)
        old_version_id = dedup_results.get('old_version_id', None)
        version_similarity = dedup_results.get('version_similarity', 0.0)
        
        logger.info(f"Deduplication results for {doc1_id}:", extra={"icon": "📊"})
        logger.info(f"- New pages: {len(new_pages)}", extra={"icon": "🆕"})
        logger.info(f"- Duplicate pages: {len(duplicate_pages)}", extra={"icon": "♻️"})
        logger.info(f"- Pages to update: {len(update_pages)}", extra={"icon": "🔄"})
        
        if is_new_version:
            logger.info(f"- Document is a new version of {old_version_id} with similarity {version_similarity:.4f}", extra={"icon": "🔄"})
        elif old_version_id:
            logger.info(f"- Document is related to {old_version_id} with similarity {version_similarity:.4f}", extra={"icon": "ℹ️"})
        
        return True

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test document deduplication')
    parser.add_argument('--doc1', type=str, required=True, help='Path to first document JSON')
    parser.add_argument('--doc2', type=str, help='Path to second document JSON (optional)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    test_deduplication(args.doc1, args.doc2, args.verbose)

if __name__ == "__main__":
    main() 