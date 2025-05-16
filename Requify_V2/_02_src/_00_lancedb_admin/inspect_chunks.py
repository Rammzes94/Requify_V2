#!/usr/bin/env python
"""
inspect_chunks.py

This script inspects the document chunks in the database to verify alignment between document versions.
It shows how many chunks have alignment relationships with other documents and prints examples
of aligned chunks to verify content similarity.
"""

import os
import sys
import logging
from dotenv import load_dotenv
import lancedb
import pandas as pd

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import _00_utils
_00_utils.setup_project_directory()

# Load environment variables
load_dotenv()

# Setup logging
logger = _00_utils.setup_logging()

# Constants
OUTPUT_DIR_BASE = "_03_output"
LANCEDB_SUBDIR_NAME = "lancedb"
DOCUMENT_CHUNKS_TABLE = "document_chunks"
DOCUMENTS_TABLE = "documents"

def connect_to_lancedb():
    """Connect to LanceDB and return the connection."""
    # Construct path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
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

def list_documents():
    """List all documents in the database."""
    # Connect to LanceDB
    db = connect_to_lancedb()
    if not db:
        logger.error("Failed to connect to LanceDB database", extra={"icon": "❌"})
        return []
    
    # Check if the documents table exists
    if DOCUMENTS_TABLE not in db.table_names():
        logger.error(f"Documents table {DOCUMENTS_TABLE} does not exist", extra={"icon": "❌"})
        return []
    
    try:
        # Open the table
        table = db.open_table(DOCUMENTS_TABLE)
        
        # Get all documents
        docs_df = table.to_pandas()
        
        if docs_df.empty:
            logger.warning("Documents table is empty", extra={"icon": "⚠️"})
            return []
        
        # Get unique document IDs
        unique_docs = docs_df[['pdf_identifier', 'document_title']].drop_duplicates()
        
        return unique_docs.to_dict('records')
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}", extra={"icon": "❌"})
        return []

def get_document_chunks(document_id):
    """Get all chunks for a specific document."""
    # Connect to LanceDB
    db = connect_to_lancedb()
    if not db:
        logger.error("Failed to connect to LanceDB database", extra={"icon": "❌"})
        return None
    
    # Check if the chunks table exists
    if DOCUMENT_CHUNKS_TABLE not in db.table_names():
        logger.error(f"Document chunks table {DOCUMENT_CHUNKS_TABLE} does not exist", extra={"icon": "❌"})
        return None
    
    try:
        # Open the table
        table = db.open_table(DOCUMENT_CHUNKS_TABLE)
        
        # Get chunks for the document
        all_chunks = table.to_pandas()
        doc_chunks = all_chunks[all_chunks['document_id'] == document_id]
        
        if doc_chunks.empty:
            logger.warning(f"No chunks found for document {document_id}", extra={"icon": "⚠️"})
            return None
        
        # Sort by chunk index
        doc_chunks = doc_chunks.sort_values('chunk_index')
        
        return doc_chunks
        
    except Exception as e:
        logger.error(f"Error getting document chunks: {e}", extra={"icon": "❌"})
        return None

def list_aligned_chunks():
    """List chunks that have alignment relationships with other documents."""
    # Connect to LanceDB
    db = connect_to_lancedb()
    if not db:
        logger.error("Failed to connect to LanceDB database", extra={"icon": "❌"})
        return
    
    # Check if the chunks table exists
    if DOCUMENT_CHUNKS_TABLE not in db.table_names():
        logger.error(f"Document chunks table {DOCUMENT_CHUNKS_TABLE} does not exist", extra={"icon": "❌"})
        return
    
    try:
        # Open the table
        table = db.open_table(DOCUMENT_CHUNKS_TABLE)
        
        # Get all chunks
        all_chunks = table.to_pandas()
        
        if all_chunks.empty:
            logger.warning("Document chunks table is empty", extra={"icon": "⚠️"})
            return
        
        # Filter for chunks that have alignment info
        aligned_chunks = all_chunks[all_chunks['aligned_with_chunk_id'].notna()]
        
        if aligned_chunks.empty:
            print("No aligned chunks found in the database.")
            return
        
        print(f"\n=== Aligned Chunks (Total: {len(aligned_chunks)}) ===")
        
        # Group by document pairs
        for doc_id in aligned_chunks['document_id'].unique():
            doc_aligned_chunks = aligned_chunks[aligned_chunks['document_id'] == doc_id]
            
            # Get unique alignment target documents
            target_docs = doc_aligned_chunks['aligned_with_document_id'].unique()
            
            for target_doc in target_docs:
                if pd.isna(target_doc):
                    continue
                    
                chunks = doc_aligned_chunks[doc_aligned_chunks['aligned_with_document_id'] == target_doc]
                print(f"\nDocument '{doc_id}' has {len(chunks)} chunks aligned with '{target_doc}'")
                
                # Show an example of aligned chunks
                if not chunks.empty:
                    example = chunks.iloc[0]
                    example_chunk_id = example['chunk_id']
                    aligned_with_id = example['aligned_with_chunk_id']
                    
                    print(f"\nExample alignment: {example_chunk_id} -> {aligned_with_id}")
                    
                    # Get the target chunk to compare content
                    target_chunk = all_chunks[all_chunks['chunk_id'] == aligned_with_id]
                    
                    if not target_chunk.empty:
                        print("\nOriginal chunk:")
                        print(f"{target_chunk.iloc[0]['chunk_text'][:200]}...")
                        
                        print("\nAligned chunk:")
                        print(f"{example['chunk_text'][:200]}...")
                    else:
                        print(f"Target chunk {aligned_with_id} not found")
        
    except Exception as e:
        logger.error(f"Error listing aligned chunks: {e}", extra={"icon": "❌"})

def main():
    """Main entry point."""
    print("\n=== Document Chunks Inspector ===")
    
    # List all documents
    documents = list_documents()
    
    if not documents:
        print("No documents found in the database.")
        return
    
    print(f"\n=== Documents (Total: {len(documents)}) ===")
    for i, doc in enumerate(documents):
        print(f"{i+1}. {doc['pdf_identifier']} - {doc['document_title']}")
    
    # List aligned chunks
    list_aligned_chunks()
    
    print("\nInspection complete.")

if __name__ == "__main__":
    main() 