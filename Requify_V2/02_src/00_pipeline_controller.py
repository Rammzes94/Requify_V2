"""
00_pipeline_controller.py

This script controls the document processing pipeline, orchestrating the end-to-end flow:
1. Parse a PDF document to extract text and images
2. Check for duplicates at the page level before saving to LanceDB
3. Extract requirements from the document
4. Check for duplicate requirements before saving to the requirements table

The pipeline processes one document at a time, ensuring proper deduplication
at both the document/page level and the requirements level.
"""

import os
import sys
import time
import argparse
import logging
from typing import Optional
from dotenv import load_dotenv

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import _00_utils
_00_utils.setup_project_directory()

# Load environment variables
load_dotenv()

# Setup logging with script prefix
logger = _00_utils.setup_logging()

class ScriptLogger(logging.LoggerAdapter):
    def __init__(self, logger, prefix):
        super().__init__(logger, {})
        self.prefix = prefix
        
    def process(self, msg, kwargs):
        return f"{self.prefix}{msg}", kwargs

logger = ScriptLogger(_00_utils.setup_logging(), "[Pipeline_Controller] ")

# Import the necessary modules from our pipeline
from _00_utils import setup_logging
# Import PDF parsing module
sys.path.append(os.path.join(os.path.dirname(__file__), '02_parsing'))
import stable_pdf_parsing

# Import LanceDB saving module
import stable_save_to_lancedb

# Import requirements extraction module
sys.path.append(os.path.join(os.path.dirname(__file__), '04_extract_reqs'))
import extract_requirements

# Constants
DEFAULT_INPUT_DIR = os.path.join("01_input", "raw")
DEFAULT_OUTPUT_DIR = "03_output"

def process_document(doc_path: str) -> bool:
    """
    Process a single document through the entire pipeline.
    
    Args:
        doc_path: Path to the PDF document to process
        
    Returns:
        True if processing was successful, False otherwise
    """
    start_time = time.time()
    doc_name = os.path.basename(doc_path)
    
    logger.info(f"🚀 Starting pipeline for document: {doc_name}")
    
    # Step 1: Parse the PDF document
    logger.info(f"Step 1: Parsing document {doc_name}")
    processor = stable_pdf_parsing.PDFProcessor(
        stable_pdf_parsing.plain_agent, 
        stable_pdf_parsing.structured_agent,
        stable_pdf_parsing.document_title_agent
    )
    
    combined_json_path = processor.pdf_to_structured_json(doc_path)
    
    if not combined_json_path or not os.path.exists(combined_json_path):
        logger.error(f"❌ Document parsing failed for {doc_name} (no combined JSON output)")
        return False
    
    logger.info(f"✅ Document parsing completed: {combined_json_path}")
    
    # Step 2: Save to LanceDB with deduplication
    logger.info(f"Step 2: Saving document to LanceDB with deduplication")
    save_success = stable_save_to_lancedb.main(combined_json_path)
    
    if not save_success:
        logger.error(f"❌ Saving to LanceDB failed for {doc_name} (deduplication or DB error)")
        return False
    
    logger.info(f"✅ Document saved to LanceDB successfully")
    
    # Step 3: Extract requirements
    logger.info(f"Step 3: Extracting requirements from document")
    
    # Extract document ID from filename for requirements extraction
    doc_id = os.path.basename(doc_path)
    
    # Process requirements
    try:
        extract_requirements.process_single_document(doc_id)
        logger.info(f"✅ Requirements extraction completed for {doc_name}")
    except Exception as e:
        logger.error(f"❌ Requirements extraction failed: {str(e)}")
        return False
    
    end_time = time.time()
    logger.info(f"🏁 Pipeline completed in {end_time - start_time:.2f} seconds for {doc_name}")
    return True

def main():
    """Parse arguments and run the pipeline"""
    parser = argparse.ArgumentParser(description="Run the document processing pipeline")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to the input PDF document")
    args = parser.parse_args()
    
    # Process the specific document
    doc_path = args.input
    if not os.path.exists(doc_path):
        logger.error(f"❌ Document not found: {doc_path}")
        return 1
    
    success = process_document(doc_path)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 