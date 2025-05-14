"""
This script initializes and verifies the LanceDB tables used in the requirements extraction pipeline:
1. 'documents' - stores parsed PDF documents with their content and embeddings
2. 'document_chunks' - stores document chunks with their content and embeddings  
3. 'requirements' - stores extracted requirements with metadata and embeddings
4. 'file_hashes' - stores file hashes for duplicate detection

It ensures all tables are set up with the correct schemas before any data processing takes place.
"""
import os
import sys
import lancedb
import logging
from lancedb.pydantic import LanceModel, Vector
from typing import Optional, List
from pydantic import Field
from dotenv import load_dotenv

# Add the parent directory (_02_src) to the system path to allow importing _00_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import _00_utils
_00_utils.setup_project_directory() # Ensures working directory is project root

# Load environment variables
load_dotenv()

# Setup logging with script prefix
class ScriptLogger(logging.LoggerAdapter):
    def __init__(self, logger, prefix):
        super().__init__(logger, {})
        self.prefix = prefix
        
    def process(self, msg, kwargs):
        return f"{self.prefix}{msg}", kwargs

logger = ScriptLogger(_00_utils.setup_logging(), "[LanceDB_Admin] ")

# --- Constants ---
# These should align with constants used in other scripts that access these tables
OUTPUT_DIR_BASE = "_03_output"
LANCEDB_SUBDIR_NAME = "lancedb"
LANCEDB_DIR_PATH = os.path.join(OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME) # Relative to project root
DOCUMENTS_TABLE_NAME = "documents"
DOCUMENT_CHUNKS_TABLE_NAME = "document_chunks"
REQUIREMENTS_TABLE_NAME = "requirements"
FILE_HASHES_TABLE_NAME = "file_hashes"
EMBEDDING_DIMENSION = 1024  # This must be consistent with the embedding model used

# --- LanceDB Schema Definition for the 'documents' table ---
class PDFPage(LanceModel):
    pdf_identifier: str
    page_number: Optional[int] # Allow None if page number isn't reliably extracted
    document_title: Optional[str]
    summary: Optional[str]
    hashtags: Optional[List[str]]
    md_content: Optional[str]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    processing_duration: Optional[float]
    error_flag: Optional[bool]
    timestamp: Optional[str]
    embedding: Vector(EMBEDDING_DIMENSION) # Text embedding
    image_b64: Optional[str] # Base64 encoded image of the page

# --- LanceDB Schema Definition for the 'document_chunks' table ---
class DocumentChunk(LanceModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    start_offset: int
    end_offset: int
    chunk_text: str
    token_count: int
    embedding: Vector(EMBEDDING_DIMENSION)
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    is_updated: bool = False
    previous_chunk_id: Optional[str] = None
    timestamp: str

# --- LanceDB Schema Definition for the 'requirements' table ---
class Requirement(LanceModel):
    # Fields based on _02_src/_04_extract_reqs/extract_requirements.py
    requirement_id: str = Field(..., description="Unique identifier for this requirement")
    document_id: str = Field(..., description="ID of the source document")
    page_number: Optional[int] = Field(default=None, description="Page number in the source document, if applicable")
    section: str = Field(..., description="Section identifier from the source document (e.g., '3.1.2')")
    title: str = Field(..., description="Short title summarizing the requirement")
    description: str = Field(..., description="Full text of the requirement's description")
    requirement_rationale: str = Field(..., description="Rationale explaining why this is considered a technical product requirement")
    is_software_related: bool = Field(..., description="Boolean indicating if the requirement is primarily software-related")
    software_confidence: float = Field(..., description="Confidence level (0.0 to 1.0) of the software-related analysis")
    software_rationale: str = Field(..., description="Explanation of why this requirement is or is not software-related")
    source_text: str = Field(..., description="The exact text snippet from the source document that directly led to this requirement's extraction")
    source_chunk_id: str = Field(..., description="ID of the document chunk this requirement was extracted from")
    created_timestamp: str = Field(..., description="Timestamp (YYYY-MM-DD HH:MM:SS) of when the requirement was extracted or created")
    embedding: Vector(EMBEDDING_DIMENSION) = Field(..., description=f"Embedding vector of dimension {EMBEDDING_DIMENSION}")

    # New fields for deduplication tracking
    is_duplicate: bool = Field(default=False, description="Flag indicating if this requirement is considered a duplicate of another (canonical) requirement")
    duplicate_of: Optional[str] = Field(default=None, description="If is_duplicate is True, this field stores the requirement_id of the canonical version")

# --- LanceDB Schema Definition for the 'file_hashes' table ---
class FileHash(LanceModel):
    file_path: str = Field(..., description="Full path to the file")
    file_name: str = Field(..., description="Name of the file")
    file_size: int = Field(..., description="Size of the file in bytes") 
    md5_hash: str = Field(..., description="MD5 hash of the file content")
    sha256_hash: str = Field(..., description="SHA256 hash of the file content")
    timestamp: str = Field(..., description="Timestamp when the file was processed")

def create_or_verify_table(db, table_name, schema_model):
    """
    Creates a table if it doesn't exist, or verifies the schema if it does.
    
    Args:
        db: LanceDB connection
        table_name: Name of the table to create or verify
        schema_model: Pydantic model defining the table schema
        
    Returns:
        True if the table exists with correct schema, False otherwise
    """
    # Check if the table already exists
    if table_name in db.table_names():
        logger.info(f"Table '{table_name}' already exists in LanceDB. Verifying schema...")
        try:
            table = db.open_table(table_name)
            existing_schema_fields = set(table.schema.names)
            defined_schema_fields = set(schema_model.model_fields.keys()) # pydantic v2 way
            
            missing_in_db = defined_schema_fields - existing_schema_fields
            extra_in_db = existing_schema_fields - defined_schema_fields

            if not missing_in_db and not extra_in_db:
                logger.info(f"✅ Schema for table '{table_name}' matches defined schema. No action needed.")
                return True
            else:
                if missing_in_db:
                    logger.warning(f"⚠️ Schema mismatch for table '{table_name}'. Fields missing in DB: {missing_in_db}. Consider running a migration or schema update script.")
                if extra_in_db:
                    logger.warning(f"⚠️ Schema mismatch for table '{table_name}'. Extra fields in DB not in schema: {extra_in_db}.")
                logger.warning("⚠️ This script does not alter existing table schemas. Manual review or a dedicated migration script is advised.")
                return False
        except Exception as e:
            logger.error(f"❌ Could not open or verify schema for existing table '{table_name}': {e}", exc_info=True)
            return False
    else:
        # Table does not exist, so create it
        logger.info(f"Table '{table_name}' does not exist. Attempting to create it now...")
        try:
            table = db.create_table(table_name, schema=schema_model)
            logger.info(f"✅ Successfully created table '{table_name}'.")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create table '{table_name}': {e}", exc_info=True)
            return False

def main():
    """
    Main function to connect to LanceDB and create/verify all tables.
    """
    logger.info(f"🔄 Script started: Initializing LanceDB tables...")

    # setup_project_directory() should have set cwd to project root.
    project_root = os.getcwd() 
    absolute_lancedb_path = os.path.join(project_root, LANCEDB_DIR_PATH)

    # Ensure the LanceDB directory itself exists
    try:
        os.makedirs(absolute_lancedb_path, exist_ok=True)
        logger.info(f"✅ LanceDB directory ensured at: {absolute_lancedb_path}")
    except Exception as e:
        logger.error(f"❌ Could not create LanceDB directory at {absolute_lancedb_path}: {e}", exc_info=True)
        return

    logger.info(f"🔄 Attempting to connect to LanceDB at: {absolute_lancedb_path}")
    try:
        db = lancedb.connect(absolute_lancedb_path)
        logger.info(f"✅ Successfully connected to LanceDB. Current tables: {db.table_names()}")
    except Exception as e:
        logger.error(f"❌ Failed to connect to LanceDB at {absolute_lancedb_path}: {e}", exc_info=True)
        return

    # Create or verify all tables
    tables = {
        DOCUMENTS_TABLE_NAME: PDFPage,
        DOCUMENT_CHUNKS_TABLE_NAME: DocumentChunk,
        REQUIREMENTS_TABLE_NAME: Requirement,
        FILE_HASHES_TABLE_NAME: FileHash
    }
    
    all_success = True
    for table_name, schema_model in tables.items():
        success = create_or_verify_table(db, table_name, schema_model)
        if success:
            logger.info(f"✅ '{table_name}' table is ready for use.")
        else:
            logger.warning(f"⚠️ Issues detected with '{table_name}' table.")
            all_success = False
        
    # Final status
    if all_success:
        logger.info(f"✅ All LanceDB tables initialized successfully.")
    else:
        logger.warning(f"⚠️ Some issues were detected during table initialization. Review logs above.")

if __name__ == "__main__":
    main() 