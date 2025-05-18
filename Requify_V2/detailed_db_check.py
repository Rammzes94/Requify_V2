"""
detailed_db_check.py

This script provides a detailed view of the database content with a focus on
chunk relationships and content. It helps diagnose issues with chunk processing.
"""

import os
import sys
import pandas as pd
import lancedb
from dotenv import load_dotenv

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import _02_src._00_utils as _00_utils
_00_utils.setup_project_directory()

# Load environment variables
load_dotenv()

# Define important constants
OUTPUT_DIR = "_03_output"
LANCEDB_SUBDIR = "lancedb"
CHUNKS_TABLE_NAME = "document_chunks"

def main():
    """Examine database in detail."""
    # Connect to LanceDB
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lancedb_path = os.path.join(script_dir, OUTPUT_DIR, LANCEDB_SUBDIR)
    print(f"Connecting to LanceDB at: {lancedb_path}")
    
    db = lancedb.connect(lancedb_path)
    
    # Get all chunks
    chunks_table = db.open_table(CHUNKS_TABLE_NAME)
    chunks_df = chunks_table.to_pandas()
    
    print(f"\n=== Database Summary ===")
    print(f"Total chunks: {len(chunks_df)}")
    
    # Get unique document IDs
    docs = chunks_df['document_id'].unique()
    print(f"Unique documents: {len(docs)}")
    for doc in docs:
        doc_chunks = chunks_df[chunks_df['document_id'] == doc]
        print(f"  - {doc}: {len(doc_chunks)} chunks")
    
    # Print chunk details
    print("\n=== Chunk Details ===")
    for i, row in chunks_df.iterrows():
        chunk_id = row['chunk_id']
        document_id = row['document_id']
        chunk_index = row.get('chunk_index', 'N/A')
        is_updated = row.get('is_updated', False)
        is_replaced = row.get('is_replaced', False)
        previous_chunk_id = row.get('previous_chunk_id', '')
        replaced_by = row.get('replaced_by', '')
        
        # Get first 100 chars of text for preview
        text_preview = row.get('chunk_text', '')[:100].replace('\n', ' ') + '...'
        
        print(f"\nChunk #{i+1}: {chunk_id}")
        print(f"  Document: {document_id}")
        print(f"  Index: {chunk_index}")
        print(f"  Is Updated: {is_updated}")
        print(f"  Is Replaced: {is_replaced}")
        
        if previous_chunk_id:
            print(f"  Previous Chunk: {previous_chunk_id}")
        
        if replaced_by:
            print(f"  Replaced By: {replaced_by}")
        
        print(f"  Text Preview: {text_preview}")
    
    # Look for relationships
    print("\n=== Relationships Analysis ===")
    
    # Check for chunks with is_updated=True
    updated_chunks = chunks_df[chunks_df['is_updated'] == True]
    if len(updated_chunks) > 0:
        print(f"\nChunks marked as updates: {len(updated_chunks)}")
        for i, row in updated_chunks.iterrows():
            print(f"  - {row['chunk_id']} (previous: {row.get('previous_chunk_id', 'None')})")
    else:
        print("No chunks marked as updates.")
    
    # Check for chunks with is_replaced=True
    replaced_chunks = chunks_df[chunks_df['is_replaced'] == True]
    if len(replaced_chunks) > 0:
        print(f"\nChunks marked as replaced: {len(replaced_chunks)}")
        for i, row in replaced_chunks.iterrows():
            print(f"  - {row['chunk_id']} (replaced by: {row.get('replaced_by', 'None')})")
    else:
        print("No chunks marked as replaced.")
    
    # Compare chunks with similar positions in different documents
    print("\n=== Cross-Document Chunk Comparison ===")
    
    # Get chunk counts per document
    for doc1 in docs:
        for doc2 in docs:
            if doc1 >= doc2:  # Skip comparing document with itself or repeating comparisons
                continue
                
            print(f"\nComparing {doc1} with {doc2}:")
            
            doc1_chunks = chunks_df[chunks_df['document_id'] == doc1]
            doc2_chunks = chunks_df[chunks_df['document_id'] == doc2]
            
            doc1_indices = sorted(doc1_chunks['chunk_index'].unique())
            doc2_indices = sorted(doc2_chunks['chunk_index'].unique())
            
            print(f"  {doc1} has chunks with indices: {doc1_indices}")
            print(f"  {doc2} has chunks with indices: {doc2_indices}")
            
            # Compare chunks with the same index across documents
            common_indices = set(doc1_indices).intersection(set(doc2_indices))
            if common_indices:
                print(f"  Common chunk indices: {common_indices}")
                for idx in common_indices:
                    doc1_chunk = doc1_chunks[doc1_chunks['chunk_index'] == idx].iloc[0]
                    doc2_chunk = doc2_chunks[doc2_chunks['chunk_index'] == idx].iloc[0]
                    
                    print(f"  Chunk index {idx}:")
                    print(f"    {doc1} chunk: {doc1_chunk['chunk_id']} (replaced: {doc1_chunk.get('is_replaced', False)})")
                    print(f"    {doc2} chunk: {doc2_chunk['chunk_id']} (updated: {doc2_chunk.get('is_updated', False)})")
            else:
                print("  No common chunk indices found.")
    
    # Print the actual LanceDB schema
    print("\n=== LanceDB Schema ===")
    schema = chunks_table.schema
    for field in schema:
        print(f"  {field.name}: {field.type}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 