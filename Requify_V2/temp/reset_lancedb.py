"""
reset_lancedb.py

This script resets the LanceDB database by deleting all tables and recreating 
the necessary table structure for a fresh start.

It performs the following operations:
1. Connects to the LanceDB database
2. Drops all existing tables
3. Creates empty tables with the correct schema
4. Cleans up backup tables

This is useful for testing the pipeline from scratch.
"""

import os
import sys
import logging
import lancedb
from typing import List, Optional
from lancedb.pydantic import LanceModel, Vector

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import setup_logging, get_logger, update_token_counters, get_token_usage, print_token_usage, reset_token_counters, setup_project_directory, generate_timestamp
from src import config

# Setup logging
logger = get_logger("Reset_LanceDB")

# Constants
OUTPUT_DIR = "output"
LANCEDB_DIR = os.path.join(OUTPUT_DIR, "lancedb")
DOCUMENTS_TABLE = "documents"
DOCUMENT_CHUNKS_TABLE = "document_chunks"
FILE_HASHES_TABLE = "file_hashes"
REQUIREMENTS_TABLE = "requirements"

# Define table schemas
class PDFPage(LanceModel):
    pdf_identifier: str
    page_number: Optional[int]
    document_title: Optional[str]
    summary: Optional[str]
    hashtags: Optional[List[str]]
    md_content: Optional[str]
    content: Optional[str]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    processing_duration: Optional[float]
    error_flag: Optional[bool]
    timestamp: Optional[str]
    embedding: Vector(config.EMBEDDING_DIMENSION)
    document_embedding: Optional[Vector(config.DOC_EMBEDDING_DIMENSION)]
    image_b64: Optional[str]
    image: Optional[str]

class DocumentChunk(LanceModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    start_offset: int
    end_offset: int
    chunk_text: str
    token_count: int
    embedding: Vector(config.EMBEDDING_DIMENSION)
    chunk_hash: str
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    is_updated: bool = False
    previous_chunk_id: Optional[str] = None
    timestamp: str
    aligned_with_chunk_id: Optional[str] = None
    aligned_with_document_id: Optional[str] = None
    replaced_by: Optional[str] = None
    is_replaced: bool = False

class Requirement(LanceModel):
    requirement_id: str
    document_id: str
    page_number: Optional[int]
    section: str
    title: str
    description: str
    requirement_rationale: str
    is_software_related: bool
    software_confidence: float
    software_rationale: str
    source_text: str
    source_chunk_id: str
    created_timestamp: str
    embedding: Vector(config.EMBEDDING_DIMENSION)
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None

class FileHash(LanceModel):
    file_path: str
    file_name: str
    file_size: int
    md5_hash: str
    sha256_hash: str
    timestamp: str

def main():
    """Reset the LanceDB database"""
    try:
        # Connect to LanceDB
        if not os.path.exists(LANCEDB_DIR):
            os.makedirs(LANCEDB_DIR, exist_ok=True)
            
        logger.info(f"Connecting to LanceDB at {LANCEDB_DIR}", extra={"icon": "🔄"})
        db = lancedb.connect(LANCEDB_DIR)
        
        # Get list of existing tables
        existing_tables = db.table_names()
        logger.info(f"Found {len(existing_tables)} existing tables: {existing_tables}", extra={"icon": "🔍"})
        
        # Drop all existing tables
        for table in existing_tables:
            logger.info(f"Dropping table: {table}", extra={"icon": "🗑️"})
            db.drop_table(table)
            
        # Create new tables with proper schemas
        logger.info(f"Creating new documents table", extra={"icon": "🔨"})
        db.create_table(DOCUMENTS_TABLE, schema=PDFPage)
        
        logger.info(f"Creating new document_chunks table", extra={"icon": "🔨"})
        db.create_table(DOCUMENT_CHUNKS_TABLE, schema=DocumentChunk)
        
        logger.info(f"Creating new requirements table", extra={"icon": "🔨"})
        db.create_table(REQUIREMENTS_TABLE, schema=Requirement)
        
        logger.info(f"Creating new file_hashes table", extra={"icon": "🔨"})
        db.create_table(FILE_HASHES_TABLE, schema=FileHash)
        
        logger.info(f"Database reset complete", extra={"icon": "✅"})
        return True
    except Exception as e:
        logger.error(f"Error resetting database: {str(e)}", extra={"icon": "❌"})
        return False
        
if __name__ == "__main__":
    main() 