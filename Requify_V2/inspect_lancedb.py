"""
inspect_lancedb.py

This script inspects the document_chunks table in LanceDB to check for empty chunk_text fields
and provide diagnostics that help troubleshoot data integrity issues in the chunking pipeline.
"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
import _02_src._00_utils as _00_utils
_00_utils.setup_project_directory()

# Load environment variables
load_dotenv()

# Setup logging
logger = _00_utils.setup_logging()

def inspect_document_chunks():
    """Inspect the document_chunks table in LanceDB for empty chunk_text fields."""
    try:
        import lancedb
        
        # Connect to LanceDB
        lancedb_path = os.path.join("_03_output", "lancedb")
        logger.info(f"🔄 Connecting to LanceDB at {lancedb_path}")
        
        if not os.path.exists(lancedb_path):
            logger.error(f"❌ LanceDB directory does not exist at {lancedb_path}")
            return
        
        db = lancedb.connect(lancedb_path)
        logger.info(f"✅ Connected to LanceDB")
        
        # Get table names
        table_names = db.table_names()
        logger.info(f"📋 Available tables: {', '.join(table_names)}")
        
        if "document_chunks" not in table_names:
            logger.error(f"❌ document_chunks table does not exist in the database")
            return
        
        # Open document_chunks table
        table = db.open_table("document_chunks")
        logger.info(f"📂 Opened document_chunks table")
        
        # Get table schema
        schema = table.schema
        logger.info(f"📋 Table schema:\n{schema}")
        
        # Load data into pandas DataFrame
        df = table.to_pandas()
        logger.info(f"📊 Total chunks in database: {len(df)}")
        
        # Check for empty chunk_text
        empty_chunks = df[df["chunk_text"] == ""]
        logger.info(f"📊 Chunks with empty chunk_text: {len(empty_chunks)}")
        
        # Sample data analysis
        if len(df) > 0:
            logger.info(f"📊 First 5 chunks:")
            for i in range(min(5, len(df))):
                chunk = df.iloc[i]
                logger.info(f"  - Chunk ID: {chunk['chunk_id']}")
                logger.info(f"    Document ID: {chunk['document_id']}")
                logger.info(f"    Text length: {len(chunk['chunk_text'])}")
                logger.info(f"    Preview: {chunk['chunk_text'][:100]}..." if len(chunk['chunk_text']) > 100 else chunk['chunk_text'])
                logger.info(f"    Token count: {chunk['token_count']}")
                logger.info(f"    Is duplicate: {chunk['is_duplicate']}")
                logger.info(f"    Is updated: {chunk['is_updated']}")
                logger.info(f"    Timestamp: {chunk['timestamp']}")
                logger.info(f"    ------------------")
        
        # Summarize document counts
        document_counts = df['document_id'].value_counts()
        logger.info(f"📊 Documents in chunks table: {len(document_counts)}")
        logger.info(f"📊 Top 5 documents by chunk count:")
        for doc_id, count in document_counts.head(5).items():
            logger.info(f"  - {doc_id}: {count} chunks")
            
        # Check for documents with mostly empty chunks
        if len(document_counts) > 0:
            logger.info(f"📊 Checking for documents with empty chunks:")
            for doc_id in document_counts.index[:5]:  # Check top 5 documents
                doc_chunks = df[df['document_id'] == doc_id]
                empty_doc_chunks = doc_chunks[doc_chunks['chunk_text'] == ""]
                if len(empty_doc_chunks) > 0:
                    logger.info(f"  - {doc_id}: {len(empty_doc_chunks)}/{len(doc_chunks)} empty chunks")
                
        return df
    except Exception as e:
        logger.error(f"❌ Error inspecting document_chunks: {e}")
        return None

if __name__ == "__main__":
    inspect_document_chunks() 