"""
doc_embedding_debug.py

This script analyzes document embeddings in the database to help identify and fix issues
with document similarity detection.

It performs the following steps:
1. Connects to LanceDB
2. Examines document records to see if document_embedding field exists and has valid data
3. Directly compares embeddings for specific documents 
4. Outputs detailed information about embedding shapes and similarity scores

This is useful for debugging issues with document-level embeddings.
"""

import os
import sys
import time
import json
import logging
import numpy as np
import pandas as pd
import lancedb
import argparse
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import setup_logging, get_logger, update_token_counters, get_token_usage, print_token_usage, reset_token_counters, setup_project_directory, generate_timestamp
from src import config
from src._03_docs_deduplication.pre_save_deduplication import calculate_cosine_similarity

# Setup logging
logger = get_logger("Doc_Embedding_Debug")

# Initialize project
setup_project_directory()
load_dotenv()

def analyze_document_embeddings():
    """
    Analyze document embeddings in the database
    """
    # Connect to LanceDB
    lancedb_path = os.path.join("output", config.LANCEDB_SUBDIR_NAME)
    if not os.path.exists(lancedb_path):
        logger.error(f"LanceDB directory not found: {lancedb_path}", extra={"icon": "❌"})
        return False
    
    logger.info(f"Connecting to LanceDB at {lancedb_path}", extra={"icon": "🔄"})
    db = lancedb.connect(lancedb_path)
    
    # Check if documents table exists
    if config.DOCUMENTS_TABLE not in db.table_names():
        logger.error(f"Documents table not found in LanceDB", extra={"icon": "❌"})
        return False
    
    # Open the documents table
    table = db.open_table(config.DOCUMENTS_TABLE)
    
    # Get schema to check if document_embedding exists
    has_doc_embedding = False
    for field in table.schema:
        if field.name == 'document_embedding':
            has_doc_embedding = True
            logger.info(f"document_embedding field found in schema with type {field.type}", 
                       extra={"icon": "✅"})
            break
    
    if not has_doc_embedding:
        logger.error("document_embedding field not found in schema", extra={"icon": "❌"})
        return False
    
    # Get all records
    df = table.to_pandas()
    
    # List all documents
    doc_ids = df['pdf_identifier'].unique()
    logger.info(f"Found {len(doc_ids)} unique documents in database", extra={"icon": "📊"})
    for i, doc_id in enumerate(doc_ids):
        logger.info(f"  {i+1}. {doc_id}", extra={"icon": "📄"})
    
    # Check if document_embedding exists in records
    if 'document_embedding' not in df.columns:
        logger.error("document_embedding column not found in records", extra={"icon": "❌"})
        return False
    
    # Count records with non-null document_embedding
    records_with_embedding = df[df['document_embedding'].notna()].shape[0]
    total_records = df.shape[0]
    logger.info(f"{records_with_embedding} out of {total_records} records have document_embedding", 
               extra={"icon": "📊"})
    
    # Sample first few records to examine embedding structure
    for i, (idx, row) in enumerate(df.iterrows()):
        if i >= 3:  # Only show first 3 records
            break
            
        doc_id = row['pdf_identifier']
        embedding = row.get('document_embedding')
        
        if embedding is None:
            logger.info(f"Record {i+1} ({doc_id}): No document_embedding", extra={"icon": "⚠️"})
            continue
            
        if isinstance(embedding, list):
            embedding_type = "list"
            embedding_len = len(embedding)
            embedding_np = np.array(embedding)
        else:
            embedding_type = type(embedding).__name__
            embedding_len = getattr(embedding, "shape", ["unknown"])[0] if hasattr(embedding, "shape") else "unknown"
            embedding_np = embedding if isinstance(embedding, np.ndarray) else np.array(embedding)
        
        # Check for all zeros or NaNs
        is_zeros = np.all(embedding_np == 0)
        has_nans = np.isnan(embedding_np).any() if isinstance(embedding_np, np.ndarray) else False
        
        logger.info(f"Record {i+1} ({doc_id}): document_embedding is {embedding_type} with length {embedding_len}. "
                   f"All zeros: {is_zeros}, Has NaNs: {has_nans}", 
                  extra={"icon": "🧬"})
    
    return True

def compare_specific_documents(doc1_id, doc2_id):
    """
    Compare document embeddings for two specific documents
    """
    # Connect to LanceDB
    lancedb_path = os.path.join("output", config.LANCEDB_SUBDIR_NAME)
    if not os.path.exists(lancedb_path):
        logger.error(f"LanceDB directory not found: {lancedb_path}", extra={"icon": "❌"})
        return False
    
    logger.info(f"Connecting to LanceDB at {lancedb_path}", extra={"icon": "🔄"})
    db = lancedb.connect(lancedb_path)
    
    # Check if documents table exists
    if config.DOCUMENTS_TABLE not in db.table_names():
        logger.error(f"Documents table not found in LanceDB", extra={"icon": "❌"})
        return False
    
    # Open the documents table
    table = db.open_table(config.DOCUMENTS_TABLE)
    
    # Get all records
    df = table.to_pandas()
    
    # Get records for both documents
    doc1_records = df[df['pdf_identifier'] == doc1_id]
    doc2_records = df[df['pdf_identifier'] == doc2_id]
    
    if doc1_records.empty:
        logger.error(f"No records found for document 1: {doc1_id}", extra={"icon": "❌"})
        return False
    
    if doc2_records.empty:
        logger.error(f"No records found for document 2: {doc2_id}", extra={"icon": "❌"})
        return False
    
    logger.info(f"Found {len(doc1_records)} records for document 1 and {len(doc2_records)} records for document 2", 
               extra={"icon": "✅"})
    
    # Check if document_embedding field exists
    if 'document_embedding' not in doc1_records.columns:
        logger.error("document_embedding field not found in records", extra={"icon": "❌"})
        return False
    
    # Extract document embeddings
    doc1_embedding = doc1_records.iloc[0].get('document_embedding')
    doc2_embedding = doc2_records.iloc[0].get('document_embedding')
    
    # Analyze embeddings
    logger.info(f"Document 1 embedding: type={type(doc1_embedding).__name__}", extra={"icon": "🧬"})
    logger.info(f"Document 2 embedding: type={type(doc2_embedding).__name__}", extra={"icon": "🧬"})
    
    # Convert to numpy arrays if they're lists
    if doc1_embedding is not None and isinstance(doc1_embedding, list):
        doc1_embedding = np.array(doc1_embedding)
    
    if doc2_embedding is not None and isinstance(doc2_embedding, list):
        doc2_embedding = np.array(doc2_embedding)
    
    # Check if embeddings are valid
    if doc1_embedding is None:
        logger.error(f"Document 1 ({doc1_id}) has no embedding", extra={"icon": "❌"})
        return False
    
    if doc2_embedding is None:
        logger.error(f"Document 2 ({doc2_id}) has no embedding", extra={"icon": "❌"})
        return False
    
    # Check for all zeros or NaNs
    doc1_is_zeros = np.all(doc1_embedding == 0)
    doc1_has_nans = np.isnan(doc1_embedding).any() if isinstance(doc1_embedding, np.ndarray) else False
    
    doc2_is_zeros = np.all(doc2_embedding == 0)
    doc2_has_nans = np.isnan(doc2_embedding).any() if isinstance(doc2_embedding, np.ndarray) else False
    
    logger.info(f"Document 1 embedding: All zeros: {doc1_is_zeros}, Has NaNs: {doc1_has_nans}", 
               extra={"icon": "🧬"})
    logger.info(f"Document 2 embedding: All zeros: {doc2_is_zeros}, Has NaNs: {doc2_has_nans}", 
               extra={"icon": "🧬"})
    
    # Check shapes
    doc1_shape = doc1_embedding.shape if hasattr(doc1_embedding, 'shape') else len(doc1_embedding)
    doc2_shape = doc2_embedding.shape if hasattr(doc2_embedding, 'shape') else len(doc2_embedding)
    
    logger.info(f"Document 1 embedding shape: {doc1_shape}", extra={"icon": "🧬"})
    logger.info(f"Document 2 embedding shape: {doc2_shape}", extra={"icon": "🧬"})
    
    # Calculate similarity if possible
    try:
        similarity = calculate_cosine_similarity(doc1_embedding, doc2_embedding)
        logger.info(f"Document similarity between {doc1_id} and {doc2_id}: {similarity:.4f}", 
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
    except Exception as e:
        logger.error(f"Error calculating similarity: {str(e)}", extra={"icon": "❌"})
    
    return True

def fix_document_embeddings():
    """
    Fix document embeddings in the database
    """
    # Connect to LanceDB
    lancedb_path = os.path.join("output", config.LANCEDB_SUBDIR_NAME)
    if not os.path.exists(lancedb_path):
        logger.error(f"LanceDB directory not found: {lancedb_path}", extra={"icon": "❌"})
        return False
    
    logger.info(f"Connecting to LanceDB at {lancedb_path}", extra={"icon": "🔄"})
    db = lancedb.connect(lancedb_path)
    
    # Import the document embedder to regenerate embeddings
    from src.utils.doc_embedding_utils import prepare_document_text, generate_document_embedding, get_document_embedder
    
    # Initialize document embedder
    doc_embedder = get_document_embedder()
    if not doc_embedder:
        logger.error("Failed to initialize document embedder", extra={"icon": "❌"})
        return False
    
    # Check if documents table exists
    if config.DOCUMENTS_TABLE not in db.table_names():
        logger.error(f"Documents table not found in LanceDB", extra={"icon": "❌"})
        return False
    
    # Open the documents table
    table = db.open_table(config.DOCUMENTS_TABLE)
    
    # Get all records
    df = table.to_pandas()
    
    # Group records by document ID
    doc_groups = df.groupby('pdf_identifier')
    total_docs = len(doc_groups)
    
    logger.info(f"Processing {total_docs} documents to fix embeddings", extra={"icon": "🔄"})
    
    # Create backup before modifying
    backup_table_name = f"{config.DOCUMENTS_TABLE}_backup_{int(time.time())}"
    logger.info(f"Creating backup of documents table as {backup_table_name}", extra={"icon": "📦"})
    db.create_table(backup_table_name, data=df)
    
    # Process each document group
    for i, (doc_id, group) in enumerate(doc_groups):
        logger.info(f"Processing document {i+1}/{total_docs}: {doc_id}", extra={"icon": "🔄"})
        
        # Check if document has md_content
        has_content = 'md_content' in group.columns and not group['md_content'].isna().all()
        
        if not has_content:
            logger.warning(f"Document {doc_id} has no content, skipping", extra={"icon": "⚠️"})
            continue
        
        # Prepare document text from all pages
        records = group.to_dict('records')
        doc_text = prepare_document_text(records)
        
        if not doc_text:
            logger.warning(f"Empty document text for {doc_id}, skipping", extra={"icon": "⚠️"})
            continue
        
        # Generate document embedding
        try:
            document_embedding = generate_document_embedding(doc_text, doc_embedder)
            
            if document_embedding is None:
                logger.error(f"Failed to generate embedding for {doc_id}", extra={"icon": "❌"})
                continue
                
            # Convert to list for storage
            embedding_list = document_embedding.tolist()
            
            # Update all records for this document with the new embedding
            for idx in group.index:
                df.at[idx, 'document_embedding'] = embedding_list
                
            logger.info(f"Successfully updated embedding for {doc_id}", extra={"icon": "✅"})
        except Exception as e:
            logger.error(f"Error generating embedding for {doc_id}: {str(e)}", extra={"icon": "❌"})
    
    # Save updated data back to database
    try:
        # Drop original table
        db.drop_table(config.DOCUMENTS_TABLE)
        
        # Create new table with updated data
        db.create_table(config.DOCUMENTS_TABLE, data=df)
        
        logger.info(f"Successfully updated document embeddings for {total_docs} documents", 
                   extra={"icon": "✅"})
    except Exception as e:
        logger.error(f"Error saving updated embeddings: {str(e)}", extra={"icon": "❌"})
        
        # Try to restore from backup
        logger.info(f"Attempting to restore from backup", extra={"icon": "🔄"})
        db.drop_table(config.DOCUMENTS_TABLE)
        backup_df = db.open_table(backup_table_name).to_pandas()
        db.create_table(config.DOCUMENTS_TABLE, data=backup_df)
        logger.info(f"Restored from backup", extra={"icon": "✅"})
        
        return False
    
    return True

def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Debug and fix document embeddings')
    parser.add_argument('--analyze', action='store_true', help='Analyze document embeddings')
    parser.add_argument('--compare', nargs=2, metavar=('DOC1', 'DOC2'), 
                        help='Compare embeddings for two specific documents')
    parser.add_argument('--fix', action='store_true', 
                        help='Fix document embeddings by regenerating them')
    return parser.parse_args()

def main():
    """
    Main function
    """
    args = parse_arguments()
    
    if args.analyze:
        analyze_document_embeddings()
    elif args.compare:
        compare_specific_documents(args.compare[0], args.compare[1])
    elif args.fix:
        fix_document_embeddings()
    else:
        # Default to analyze if no arguments provided
        analyze_document_embeddings()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 