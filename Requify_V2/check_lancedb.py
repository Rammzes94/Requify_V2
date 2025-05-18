"""
check_lancedb.py

This script checks the contents of the LanceDB database
to verify the results of document processing.
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

# Setup logging
logger = _00_utils.setup_logging()

def check_lancedb():
    """
    Check the contents of the LanceDB database.
    """
    # Connect to LanceDB
    lancedb_path = os.path.join("_03_output", "lancedb")
    db = lancedb.connect(lancedb_path)
    
    # Print table names
    logger.info(f"📊 Tables in database: {db.table_names()}")
    
    # Check documents table
    if "documents" in db.table_names():
        docs_table = db.open_table("documents")
        docs_df = docs_table.to_pandas()
        logger.info(f"📚 Document count: {len(docs_df)}")
        
        # Print all columns and their non-null counts
        print("\nDocuments Table Column Info:")
        print(docs_df.info())
        
        # Show a sample of the data
        print("\nDocuments Table Sample:")
        # Select only a few key columns to display
        display_cols = ["pdf_identifier", "page_number", "document_title", "timestamp"]
        display_cols = [col for col in display_cols if col in docs_df.columns]
        print(docs_df[display_cols])
    
    # Check chunks table
    if "document_chunks" in db.table_names():
        chunks_table = db.open_table("document_chunks")
        chunks_df = chunks_table.to_pandas()
        logger.info(f"🧩 Chunk count: {len(chunks_df)}")
        
        # Print column info
        print("\nChunks Table Column Info:")
        print(chunks_df.info())
        
        # Print a sample with key columns
        print("\nChunks Table Relationships:")
        # Focus on showing the relationships between chunks
        relationship_cols = [
            "chunk_id", "document_id", "is_updated", "previous_chunk_id", 
            "is_replaced", "replaced_by"
        ]
        relationship_cols = [col for col in relationship_cols if col in chunks_df.columns]
        print(chunks_df[relationship_cols])
        
        # Print detailed chunk info for all chunks
        print("\nAll Chunks Content:")
        for idx, row in chunks_df.iterrows():
            print(f"Chunk {idx+1}: {row['chunk_id']}")
            print(f"  Document: {row['document_id']}")
            
            # Show relationship information if available
            if "is_updated" in row:
                print(f"  Updated: {row['is_updated']}")
            if "previous_chunk_id" in row and row["previous_chunk_id"]:
                print(f"  Previous Chunk: {row['previous_chunk_id']}")
            if "is_replaced" in row:
                print(f"  Is Replaced: {row['is_replaced']}")
            if "replaced_by" in row and row["replaced_by"]:
                print(f"  Replaced By: {row['replaced_by']}")
                
            # Show the first 150 characters of content
            print(f"  Text: {row['chunk_text'][:150]}...")
            print("-" * 80)

if __name__ == "__main__":
    check_lancedb() 