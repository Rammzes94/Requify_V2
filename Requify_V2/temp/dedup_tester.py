#!/usr/bin/env python3
"""
Script to test document deduplication directly without going through the pipeline.
"""

import os
import sys
import json
import numpy as np
import lancedb

# Add the parent directory to the system path to allow importing modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import setup_logging, get_logger, update_token_counters, get_token_usage, print_token_usage, reset_token_counters, setup_project_directory, generate_timestamp
setup_project_directory()

# Setup logging with script prefix
logger = get_logger("DedupTester")

# Load the deduplication module
from src._03_docs_deduplication import pre_save_deduplication

# Constants
EMBEDDING_DIM = 768  # Default dimension

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
            'embedding': doc_data.get('embedding', []),
            'timestamp': doc_data.get('timestamp', '')
        }
        pages_data.append(page_data)
    
    return pages_data

def connect_to_lancedb():
    """Connect to LanceDB and return the connection."""
    # Construct the absolute path to the LanceDB directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    lancedb_path = os.path.join(project_root, "output", "lancedb")
    
    try:
        logger.info(f"Connecting to LanceDB database at {lancedb_path}", extra={"icon": "🔄"})
        os.makedirs(lancedb_path, exist_ok=True)
        db = lancedb.connect(lancedb_path)
        logger.info(f"Connected to LanceDB at {lancedb_path}", extra={"icon": "✅"})
        return db
    except Exception as e:
        logger.error(f"Failed to connect to LanceDB: {e}", extra={"icon": "❌"})
        return None

def add_embeddings_to_document(doc_path):
    """Add random embeddings to document pages."""
    logger.info(f"Adding embeddings to document: {doc_path}", extra={"icon": "🔄"})
    
    # Load the document
    doc_data = load_document(doc_path)
    if not doc_data:
        return False
    
    # Generate embeddings for pages
    if 'pages' in doc_data:
        for page_key in doc_data.get('pages', {}):
            # Create a non-zero embedding with randomness for testing
            base = np.ones(EMBEDDING_DIM) * 0.1
            noise = np.random.rand(EMBEDDING_DIM) * 0.05
            embedding = base + noise
            
            # Normalize to unit length (for cosine similarity)
            embedding = embedding / np.linalg.norm(embedding)
            
            # Add to document
            doc_data['pages'][page_key]['embedding'] = embedding.tolist()
    else:
        # Single page document
        base = np.ones(EMBEDDING_DIM) * 0.1
        noise = np.random.rand(EMBEDDING_DIM) * 0.05
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
    
    # Connect to LanceDB once and reuse the connection
    db = connect_to_lancedb()
    if not db:
        logger.error("Failed to connect to LanceDB. Aborting test.", extra={"icon": "❌"})
        return False
    
    # Load and prepare first document
    doc1_data = load_document(doc1_path)
    if not doc1_data:
        return False
    
    doc1_pages = prepare_document_data(doc1_data)
    doc1_id = doc1_data.get('pdf_identifier', os.path.basename(doc1_path))
    
    # Check if embeddings exist, add if needed
    has_embeddings = False
    if doc1_pages and 'embedding' in doc1_pages[0]:
        embedding = doc1_pages[0]['embedding']
        if isinstance(embedding, list) and len(embedding) > 0:
            has_embeddings = True
    
    if not has_embeddings:
        logger.info(f"Document {doc1_id} has no embeddings. Adding them now.", extra={"icon": "🔄"})
        add_embeddings_to_document(doc1_path)
        doc1_data = load_document(doc1_path)
        doc1_pages = prepare_document_data(doc1_data)
    
    # If testing with a second document
    if doc2_path:
        # Load and prepare second document
        doc2_data = load_document(doc2_path)
        if not doc2_data:
            return False
        
        doc2_pages = prepare_document_data(doc2_data)
        doc2_id = doc2_data.get('pdf_identifier', os.path.basename(doc2_path))
        
        # Check if second document has embeddings
        has_embeddings = False
        if doc2_pages and 'embedding' in doc2_pages[0]:
            embedding = doc2_pages[0]['embedding']
            if isinstance(embedding, list) and len(embedding) > 0:
                has_embeddings = True
        
        if not has_embeddings:
            logger.info(f"Document {doc2_id} has no embeddings. Adding them now.", extra={"icon": "🔄"})
            add_embeddings_to_document(doc2_path)
            doc2_data = load_document(doc2_path)
            doc2_pages = prepare_document_data(doc2_data)
        
        # Save first document to database
        logger.info(f"Saving first document {doc1_id} to database", extra={"icon": "💾"})
        
        # Make sure the documents table exists
        if "documents" not in db.table_names():
            from src._00_lancedb_admin.init_lancedb import PDFPage
            db.create_table("documents", schema=PDFPage)
            
        # Save doc1 to database
        table = db.open_table("documents")
        table.add(doc1_pages)
        
        # Now test deduplication for the second document
        logger.info(f"Now testing deduplication for second document {doc2_id}", extra={"icon": "🔍"})
        
        # Set verbose flag if needed
        if verbose:
            pre_save_deduplication.VERBOSE_DEDUPLICATION_OUTPUT = True
        
        # Check for duplicates
        dedup_results = pre_save_deduplication.check_new_document(doc2_pages, db_connection=db)
        
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
        # If only one document, check against existing database
        logger.info(f"Checking if document {doc1_id} is a duplicate of anything in the database", extra={"icon": "🔍"})
        
        if verbose:
            pre_save_deduplication.VERBOSE_DEDUPLICATION_OUTPUT = True
        
        dedup_results = pre_save_deduplication.check_new_document(doc1_pages, db_connection=db)
        
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