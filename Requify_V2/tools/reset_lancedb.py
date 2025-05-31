"""
reset_lancedb.py

This script performs a complete reset of all LanceDB tables to ensure schema consistency across the pipeline.
It removes all existing tables and reinitializes them with current schema definitions.
"""

import os
import sys
import logging
from dotenv import load_dotenv
import lancedb
from lancedb.pydantic import LanceModel, Vector
from typing import List, Optional

# Ensure the project root is the first entry in sys.path so 'config' and other modules can be found
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if sys.path[0] != project_root:
    sys.path.insert(0, project_root)

from src.utils import setup_logging, get_logger, update_token_counters, get_token_usage, print_token_usage, reset_token_counters, setup_project_directory, generate_timestamp
from src import config
setup_project_directory()

# Load environment variables
load_dotenv()

# Set up logging with script prefix
logger = get_logger("Reset_LanceDB")

# Constants
OUTPUT_DIR_BASE = "output"
LANCEDB_SUBDIR_NAME = "lancedb"
ALL_TABLES = ["document_chunks", "documents", "file_hashes", "requirements"]

# Define schema classes directly here to ensure they're up-to-date
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
    # Connect to LanceDB
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    lancedb_path = os.path.join(project_root, OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME)
    
    # Ensure directory exists
    os.makedirs(lancedb_path, exist_ok=True)
    
    try:
        logger.info(f"Connecting to LanceDB at {lancedb_path}", extra={"icon": "🔄"})
        db = lancedb.connect(lancedb_path)
        logger.info(f"Connected to LanceDB. Current tables: {db.table_names()}", extra={"icon": "✅"})
        
        # Drop all existing tables
        existing_tables = db.table_names()
        for table_name in existing_tables:
            logger.info(f"Dropping table: {table_name}", extra={"icon": "🗑️"})
            db.drop_table(table_name)
        
        logger.info(f"All tables dropped. Creating new tables with updated schemas.", extra={"icon": "✅"})
        
        # Create tables with updated schemas
        logger.info(f"Creating documents table", extra={"icon": "🔨"})
        db.create_table("documents", schema=PDFPage)
        
        logger.info(f"Creating document_chunks table", extra={"icon": "🔨"})
        db.create_table("document_chunks", schema=DocumentChunk)
        
        logger.info(f"Creating requirements table", extra={"icon": "🔨"})
        db.create_table("requirements", schema=Requirement)
        
        logger.info(f"Creating file_hashes table", extra={"icon": "🔨"})
        db.create_table("file_hashes", schema=FileHash)
        
        logger.info(f"LanceDB tables reinitialized with consistent schemas.", extra={"icon": "✅"})
        return True
    except Exception as e:
        logger.error(f"Error resetting LanceDB: {e}", extra={"icon": "❌"})
        return False

if __name__ == "__main__":
    main()