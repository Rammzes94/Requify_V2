"""
This script is responsible for creating the 'requirements' table in LanceDB
with a predefined schema. It ensures that the table is set up correctly
before any data ingestion or processing takes place.

The schema includes fields for requirement details, embedding vectors,
and new fields for handling deduplication (is_duplicate, canonical_id).
"""
import os
import sys
import lancedb
from lancedb.pydantic import LanceModel, Vector
from typing import Optional
from pydantic import Field # For default values
from dotenv import load_dotenv

# Add the parent directory (02_src) to the system path to allow importing _00_utils
# This assumes the script is in a subdirectory of 02_src, like 02_src/00_lancedb_admin/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import _00_utils
_00_utils.setup_project_directory() # Ensures working directory is project root

# Load environment variables
load_dotenv()

# Setup logging
logger = _00_utils.setup_logging()

# --- Constants ---
# These should align with constants used in other scripts that access this DB/table
OUTPUT_DIR_BASE = "03_output"
LANCEDB_SUBDIR_NAME = "lancedb"
LANCEDB_DIR_PATH = os.path.join(OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME) # Relative to project root
REQUIREMENTS_TABLE_NAME = "requirements"
EMBEDDING_DIMENSION = 1024  # This must be consistent with the embedding model used

# --- LanceDB Schema Definition for the 'requirements' table ---
class Requirement(LanceModel):
    # Fields based on 02_src/04_extract_reqs/extract_requirements.py
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
    source_text: str = Field(..., description="The larger block of original text (e.g., page content) from which the requirement was extracted")
    requirement_source_snippet: str = Field(..., description="The exact text snippet from the source document that directly led to this requirement's extraction")
    created_timestamp: str = Field(..., description="Timestamp (YYYY-MM-DD HH:MM:SS) of when the requirement was extracted or created")
    embedding: Vector(EMBEDDING_DIMENSION) = Field(..., description=f"Embedding vector of dimension {EMBEDDING_DIMENSION}")

    # New fields for deduplication tracking
    is_duplicate: bool = Field(default=False, description="Flag indicating if this requirement is considered a duplicate of another (canonical) requirement")
    canonical_id: Optional[str] = Field(default=None, description="If is_duplicate is True, this field stores the requirement_id of the canonical version")

def main():
    """
    Main function to connect to LanceDB and create the 'requirements' table
    if it doesn't already exist.
    """
    logger.info(f"Script started: Attempting to ensure LanceDB table '{REQUIREMENTS_TABLE_NAME}' exists.")

    # setup_project_directory() should have set cwd to project root.
    # LANCEDB_DIR_PATH is relative to project root.
    project_root = os.getcwd() 
    absolute_lancedb_path = os.path.join(project_root, LANCEDB_DIR_PATH)

    # Ensure the LanceDB directory itself exists
    try:
        os.makedirs(absolute_lancedb_path, exist_ok=True)
        logger.info(f"LanceDB directory ensured at: {absolute_lancedb_path}")
    except Exception as e:
        logger.error(f"Could not create LanceDB directory at {absolute_lancedb_path}: {e}", exc_info=True)
        return

    logger.info(f"Attempting to connect to LanceDB at: {absolute_lancedb_path}")
    try:
        db = lancedb.connect(absolute_lancedb_path)
        logger.info(f"Successfully connected to LanceDB. Current tables: {db.table_names()}")
    except Exception as e:
        logger.error(f"Failed to connect to LanceDB at {absolute_lancedb_path}: {e}", exc_info=True)
        return

    # Check if the table already exists
    if REQUIREMENTS_TABLE_NAME in db.table_names():
        logger.info(f"Table '{REQUIREMENTS_TABLE_NAME}' already exists in LanceDB. Verifying schema...")
        try:
            table = db.open_table(REQUIREMENTS_TABLE_NAME)
            existing_schema_fields = set(table.schema.names)
            defined_schema_fields = set(Requirement.model_fields.keys()) # pydantic v2 way
            
            missing_in_db = defined_schema_fields - existing_schema_fields
            extra_in_db = existing_schema_fields - defined_schema_fields

            if not missing_in_db and not extra_in_db:
                logger.info(f"Schema for table '{REQUIREMENTS_TABLE_NAME}' matches defined schema. No action needed.")
            else:
                if missing_in_db:
                    logger.warning(f"Schema mismatch for table '{REQUIREMENTS_TABLE_NAME}'. Fields missing in DB: {missing_in_db}. Consider running a migration or schema update script.")
                if extra_in_db:
                    logger.warning(f"Schema mismatch for table '{REQUIREMENTS_TABLE_NAME}'. Extra fields in DB not in schema: {extra_in_db}.")
                logger.warning("This script does not alter existing table schemas. Manual review or a dedicated migration script is advised.")

        except Exception as e:
            logger.error(f"Could not open or verify schema for existing table '{REQUIREMENTS_TABLE_NAME}': {e}", exc_info=True)
    else:
        # Table does not exist, so create it
        logger.info(f"Table '{REQUIREMENTS_TABLE_NAME}' does not exist. Attempting to create it now...")
        try:
            table = db.create_table(REQUIREMENTS_TABLE_NAME, schema=Requirement)
            logger.info(f"Successfully created table '{REQUIREMENTS_TABLE_NAME}'. Schema: {table.schema}")
        except Exception as e:
            logger.error(f"Failed to create table '{REQUIREMENTS_TABLE_NAME}': {e}", exc_info=True)

if __name__ == "__main__":
    main() 