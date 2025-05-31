"""
test_doc_embedding.py

This script tests the document-level embedding implementation by:
1. Loading the Alibaba-NLP/gte-Qwen2-1.5B-instruct model
2. Processing a sample document from the input directory
3. Generating document-level embedding
4. Saving the document to LanceDB
5. Running a similarity test to verify deduplication

This script can be run with CLI arguments to automate testing without user interaction.
"""

import os
import sys
import time
import json
import logging
import argparse
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils import setup_logging, get_logger, update_token_counters, get_token_usage, print_token_usage, reset_token_counters, setup_project_directory, generate_timestamp
from src.utils.doc_embedding_utils import prepare_document_text, generate_document_embedding, get_document_embedder, count_tokens

# Initialize project
setup_project_directory()
load_dotenv()

# Configure logging
logger = get_logger("Test_Doc_Embedding")

# Constants
INPUT_DIR = os.path.join("input", "raw")
DEFAULT_TEST_DOC = "fighter_jet_rocket_launcher_spec.pdf"

def run_pipeline_with_doc_embedding(doc_name=DEFAULT_TEST_DOC, skip_hash_check=True):
    """
    Run the pipeline with document-level embedding for the specified document.
    
    Args:
        doc_name: Name of the document to process
        skip_hash_check: Whether to skip the hash-based duplicate check
    """
    from src.pipeline_controller import process_document
    
    # Construct the full document path
    doc_path = os.path.join(INPUT_DIR, doc_name)
    
    if not os.path.exists(doc_path):
        logger.error(f"Document not found: {doc_path}", extra={"icon": "❌"})
        return False
    
    logger.info(f"Processing document: {doc_path}", extra={"icon": "🚀"})
    
    # Process the document with the full pipeline
    # This will now include document embedding
    success = process_document(doc_path, skip_hash_check=skip_hash_check)
    
    if success:
        logger.info(f"Document processed successfully with document-level embedding", extra={"icon": "✅"})
    else:
        logger.error(f"Failed to process document", extra={"icon": "❌"})
    
    return success

def test_document_similarity(doc1_name=DEFAULT_TEST_DOC, doc2_name=None):
    """
    Test document similarity using document-level embeddings.
    
    If doc2_name is not provided, a copy of doc1_name will be used.
    """
    import lancedb
    import numpy as np
    from src._03_docs_deduplication.pre_save_deduplication import calculate_cosine_similarity
    from src import config
    
    # Initialize document embedder
    doc_embedder = get_document_embedder()
    if not doc_embedder:
        logger.error("Failed to initialize document embedder", extra={"icon": "❌"})
        return False
    
    # Construct document paths
    doc1_path = os.path.join(INPUT_DIR, doc1_name)
    
    # If doc2 not specified, use the same document
    if doc2_name is None:
        doc2_path = doc1_path
        doc2_name = doc1_name
    else:
        doc2_path = os.path.join(INPUT_DIR, doc2_name)
    
    # Check if documents exist
    if not os.path.exists(doc1_path):
        logger.error(f"Document 1 not found: {doc1_path}", extra={"icon": "❌"})
        return False
    
    if not os.path.exists(doc2_path):
        logger.error(f"Document 2 not found: {doc2_path}", extra={"icon": "❌"})
        return False
    
    # Connect to LanceDB
    lancedb_path = os.path.join("output", config.LANCEDB_SUBDIR_NAME)
    if not os.path.exists(lancedb_path):
        logger.error(f"LanceDB directory not found: {lancedb_path}", extra={"icon": "❌"})
        return False
    
    db = lancedb.connect(lancedb_path)
    
    # Check if documents table exists
    if config.DOCUMENTS_TABLE not in db.table_names():
        logger.error(f"Documents table not found in LanceDB", extra={"icon": "❌"})
        return False
    
    # Open the documents table
    table = db.open_table(config.DOCUMENTS_TABLE)
    
    # Get records for both documents
    doc1_records = table.to_pandas().query(f"pdf_identifier == '{doc1_name}'")
    doc2_records = table.to_pandas().query(f"pdf_identifier == '{doc2_name}'")
    
    if doc1_records.empty:
        logger.error(f"No records found for document 1: {doc1_name}", extra={"icon": "❌"})
        return False
    
    if doc2_records.empty:
        logger.error(f"No records found for document 2: {doc2_name}", extra={"icon": "❌"})
        return False
    
    logger.info(f"Found {len(doc1_records)} records for document 1 and {len(doc2_records)} records for document 2", 
               extra={"icon": "✅"})
    
    # Check if document_embedding field exists
    if 'document_embedding' not in doc1_records.columns:
        logger.error("document_embedding field not found in records", extra={"icon": "❌"})
        return False
    
    # Extract document embeddings
    doc1_embedding = doc1_records.iloc[0]['document_embedding']
    doc2_embedding = doc2_records.iloc[0]['document_embedding']
    
    # Calculate similarity
    similarity = calculate_cosine_similarity(
        np.array(doc1_embedding), 
        np.array(doc2_embedding)
    )
    
    logger.info(f"Document similarity between {doc1_name} and {doc2_name}: {similarity:.4f}", 
               extra={"icon": "📊"})
    
    # Report duplication status based on thresholds
    if similarity >= config.DEDUPLICATION_DUPLICATE_THRESHOLD:
        logger.info(f"Documents are DUPLICATES (similarity >= {config.DEDUPLICATION_DUPLICATE_THRESHOLD})", 
                  extra={"icon": "♻️"})
    elif similarity >= config.DEDUPLICATION_SIMILAR_THRESHOLD:
        logger.info(f"Documents are SIMILAR (similarity >= {config.DEDUPLICATION_SIMILAR_THRESHOLD})", 
                  extra={"icon": "🔄"})
    else:
        logger.info(f"Documents are DIFFERENT (similarity < {config.DEDUPLICATION_SIMILAR_THRESHOLD})", 
                  extra={"icon": "🆕"})
    
    return True

def run_all_tests(doc_name=DEFAULT_TEST_DOC, doc2_name=None, skip_hash_check=True):
    """
    Run all tests in sequence.
    """
    start_time = time.time()
    
    # Process the document
    success = run_pipeline_with_doc_embedding(doc_name, skip_hash_check=skip_hash_check)
    if not success:
        logger.error("Failed during pipeline processing, aborting further tests", extra={"icon": "❌"})
        return False
    
    # Test similarity with itself
    success = test_document_similarity(doc_name, doc_name)
    if not success:
        logger.warning("Self-similarity test failed", extra={"icon": "⚠️"})
    
    # Test similarity with another document if provided
    if doc2_name and doc2_name != doc_name:
        success = test_document_similarity(doc_name, doc2_name)
        if not success:
            logger.warning(f"Similarity test with {doc2_name} failed", extra={"icon": "⚠️"})
    
    end_time = time.time()
    logger.info(f"All tests completed in {end_time - start_time:.2f} seconds", extra={"icon": "⏱️"})
    return True

def list_available_documents():
    """
    List all available PDF files in the input directory.
    """
    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.pdf')]
    logger.info(f"Available documents in {INPUT_DIR}:", extra={"icon": "📋"})
    for i, pdf in enumerate(pdf_files):
        logger.info(f"  {i+1}. {pdf}", extra={"icon": "📄"})
    return pdf_files

def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Test document-level embedding implementation')
    parser.add_argument('--mode', type=str, choices=['process', 'similarity', 'all'], default='all',
                        help='Test mode: process (only process document), similarity (only test similarity), all (run both)')
    parser.add_argument('--doc', type=str, default=DEFAULT_TEST_DOC,
                        help=f'Document to process (default: {DEFAULT_TEST_DOC})')
    parser.add_argument('--doc2', type=str, default=None,
                        help='Second document for similarity testing (default: use same as --doc)')
    parser.add_argument('--list', action='store_true', 
                        help='List available documents and exit')
    parser.add_argument('--skip-hash-check', action='store_true', default=True,
                        help='Skip hash-based duplicate check (default: True)')
    return parser.parse_args()

def main():
    """
    Main function to run tests.
    """
    args = parse_arguments()
    
    # List documents if requested
    if args.list:
        list_available_documents()
        return 0
    
    start_time = time.time()
    
    logger.info(f"Running tests in {args.mode} mode with document: {args.doc}", 
               extra={"icon": "🚀"})
    
    if args.mode == 'process':
        success = run_pipeline_with_doc_embedding(args.doc, skip_hash_check=args.skip_hash_check)
    elif args.mode == 'similarity':
        success = test_document_similarity(args.doc, args.doc2 or args.doc)
    else:  # mode == 'all'
        success = run_all_tests(args.doc, args.doc2, args.skip_hash_check)
    
    end_time = time.time()
    logger.info(f"Tests completed in {end_time - start_time:.2f} seconds", 
               extra={"icon": "⏱️"})
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 