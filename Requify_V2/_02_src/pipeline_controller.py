"""
pipeline_controller.py

This script controls the document processing pipeline, orchestrating the end-to-end flow:
1. Perform initial hash-based deduplication to filter out exact duplicates
2. Parse PDF documents to extract text and images
3. Check for document-level duplicates before saving to LanceDB
4. Perform agentic chunking of document content
5. Check for chunk-level duplicates and document updates
6. Extract requirements from unique document chunks
7. Check for duplicate requirements before saving

The pipeline processes one document at a time, ensuring proper deduplication
at the hash level, document/page level, chunk level, and requirements level.

The pipeline can be configured to run only up to a specific step, allowing
for selective processing of documents without completing the entire pipeline.
"""

import os
import sys
import time
import logging
import argparse
from typing import Optional, Dict, List, Tuple, Any
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

# Import Hash-based deduplication
sys.path.append(os.path.join(os.path.dirname(__file__), '_01_ingestion'))
import file_hash_deduplication

# Import PDF parsing module
sys.path.append(os.path.join(os.path.dirname(__file__), '_02_parsing'))
import stable_pdf_parsing
import agentic_chunking

# Import document deduplication modules
sys.path.append(os.path.join(os.path.dirname(__file__), '_03_docs_deduplication'))
import pre_save_deduplication
import user_interaction

# Import LanceDB saving module
import stable_save_to_lancedb

# Import requirements extraction module
sys.path.append(os.path.join(os.path.dirname(__file__), '_04_extract_reqs'))
import extract_requirements

# Constants
DEFAULT_DOC_PATH = os.path.join("_01_input", "raw", "fighter_jet_rocket_launcher_spec_2.pdf")
DEFAULT_OUTPUT_DIR = "_03_output"

# Pipeline step constants
STEP_HASH_CHECK = 1    # Just do hash-based duplicate check
STEP_PARSE = 2         # Parse PDF to JSON
STEP_DEDUP_ONLY = 3    # Parse + Check for duplicates (no DB save)
STEP_SAVE_TO_DB = 4    # Parse + Save to LanceDB (no chunking)
STEP_CHUNKING = 5      # Parse + Save to LanceDB + Chunk document
STEP_EXTRACT_REQS = 6  # Complete pipeline with requirements

MAX_STEP = 6  # Change this value to control how far the pipeline runs

# Set to True to check without saving to database
DRY_RUN = False

def process_document(doc_path: str, max_step: int = STEP_EXTRACT_REQS, dry_run: bool = False) -> bool:
    """
    Process a single document through the pipeline up to a specified step.
    
    Args:
        doc_path: Path to the PDF document to process
        max_step: Maximum step to run in the pipeline:
                  1 = Hash-based duplicate check only
                  2 = Parse only
                  3 = Parse + Deduplication check only (no DB save)
                  4 = Parse + Save to LanceDB (no chunking)
                  5 = Parse + Save to LanceDB + Chunk document
                  6 = Complete pipeline (default)
        dry_run: If True, performs checks but doesn't modify the database
        
    Returns:
        True if processing was successful up to the specified step, False otherwise
    """
    start_time = time.time()
    doc_name = os.path.basename(doc_path)

    logger.info(f"🚀 Starting pipeline for document: {doc_name} (max_step={max_step}, dry_run={dry_run})")

    # Step 1: Hash-based deduplication
    logger.info(f"Step 1: Performing hash-based duplicate check for {doc_name}")
    is_duplicate, existing_file = file_hash_deduplication.check_file_duplicate(doc_path)

    if is_duplicate:
        logger.info(f"❌ Document {doc_name} is an exact duplicate of {existing_file} (by hash). Aborting further processing.")
        return True  # Successfully determined it's a duplicate, so pipeline "succeeded"
    
    logger.info(f"✅ Document {doc_name} passed hash-based duplicate check")
    
    # If we only want to do the hash check, we're done
    if max_step == STEP_HASH_CHECK:
        logger.info(f"🏁 Pipeline completed at step {max_step} (Hash check only) for {doc_name}")
        return True
    
    # Step 2: Parse the PDF document
    logger.info(f"Step 2: Parsing document {doc_name}")
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
    
    # If we only want to parse, we're done
    if max_step == STEP_PARSE:
        logger.info(f"🏁 Pipeline completed at step {max_step} (Parse only) for {doc_name}")
        return True

    # Step 3: Check for document-level duplicates
    logger.info(f"Step 3: Checking document for duplicates")
    
    # For deduplication-only mode, we need to import and call the deduplication module directly
    if max_step == STEP_DEDUP_ONLY:
        try:
            # Load document data from the combined JSON
            import json
            with open(combined_json_path, 'r') as f:
                doc_data = json.load(f)
            
            # Convert to format expected by check_new_document
            pages_data = []
            for page_key, page_info in doc_data.get('pages', {}).items():
                # Only include minimal fields needed for deduplication
                page_data = {
                    'pdf_identifier': doc_data.get('pdf_identifier', os.path.basename(doc_path)),
                    'page_number': page_info.get('page_number'),
                    'document_title': page_info.get('document_title', ''),
                    'summary': page_info.get('summary', ''),
                    'embedding': page_info.get('embedding', []),
                    'timestamp': page_info.get('timestamp', '')
                }
                pages_data.append(page_data)
            
            # Run deduplication check
            dedup_results = pre_save_deduplication.check_new_document(pages_data)
            
            # Log results
            duplicate_pages = dedup_results.get('duplicate_pages', {})
            new_pages = dedup_results.get('new_pages', [])
            update_pages = dedup_results.get('update_pages', {})
            is_new_version = dedup_results.get('is_new_version', False)
            old_version_id = dedup_results.get('old_version_id', None)
            
            logger.info(f"✅ Deduplication check results for {doc_name}:")
            logger.info(f"   - New pages: {len(new_pages)}")
            logger.info(f"   - Duplicate pages: {len(duplicate_pages)}")
            logger.info(f"   - Pages to update: {len(update_pages)}")
            
            if is_new_version:
                logger.info(f"   - Document appears to be a new version of: {old_version_id}")
            
            logger.info(f"🏁 Pipeline completed at step {max_step} (Deduplication check only) for {doc_name}")
            return True
        except Exception as e:
            logger.error(f"❌ Deduplication check failed: {str(e)}")
            return False

    # Step 4: Save to LanceDB with deduplication
    logger.info(f"Step 4: Saving document to LanceDB with deduplication")
    
    if dry_run:
        logger.info(f"🔍 Dry run mode - checking document without saving to database")
        logger.info(f"✅ Dry run check completed (would have saved to LanceDB)")
        save_success = True
    else:
        save_success = stable_save_to_lancedb.main(combined_json_path)

    if not save_success:
        logger.error(f"❌ Saving to LanceDB failed for {doc_name} (deduplication or DB error)")
        return False

    logger.info(f"✅ Document saved to LanceDB successfully")
    
    # If we only want to save to LanceDB (no chunking), we're done
    if max_step == STEP_SAVE_TO_DB:
        logger.info(f"🏁 Pipeline completed at step {max_step} (Save to LanceDB only) for {doc_name}")
        return True

    # Step 5: Perform agentic chunking and check for chunk-level duplicates
    logger.info(f"Step 5: Chunking document and checking for duplicate chunks")
    
    # Load document text from combined JSON
    try:
        import json
        with open(combined_json_path, 'r', encoding='utf-8') as f:
            doc_data = json.load(f)
        
        # Extract document text (combine all page content)
        document_text = ""
        for page_key, page_info in sorted(doc_data.get('pages', {}).items()):
            md_content = page_info.get('md_content', '')
            if md_content:
                document_text += md_content + "\n\n"
        
        document_id = doc_data.get('pdf_identifier', os.path.basename(doc_path))
        
        if not document_text.strip():
            logger.error(f"❌ Document has no text content to chunk")
            return False
            
        # Process document for chunking
        if dry_run:
            logger.info(f"🔍 Dry run mode - checking document chunks without saving")
            # Just get the chunks to analyze without saving
            chunks, duplicate_info, is_update = agentic_chunking.process_document_chunks(document_text, document_id)
            
            # Check if document is a complete duplicate at chunk level
            unique_chunks = [c for c in chunks if c.chunk_id not in duplicate_info or not duplicate_info[c.chunk_id].get("is_duplicate", False)]
            is_duplicate = len(unique_chunks) == 0 and len(chunks) > 0
            
            # Log results
            if is_duplicate:
                logger.info(f"📋 Document appears to be a complete duplicate at chunk level")
            else:
                logger.info(f"📊 Document chunking results:")
                logger.info(f"   - Total chunks: {len(chunks)}")
                logger.info(f"   - Duplicate chunks: {sum(1 for c in chunks if duplicate_info.get(c.chunk_id, {}).get('is_duplicate', False))}")
                logger.info(f"   - Updated chunks: {sum(1 for c in chunks if duplicate_info.get(c.chunk_id, {}).get('is_updated', False))}")
                logger.info(f"   - New chunks: {sum(1 for c in chunks if c.chunk_id not in duplicate_info)}")
            
            logger.info(f"✅ Dry run chunk check completed")
            chunk_success = True
        else:
            # Actually process and save the document chunks
            chunk_success, is_duplicate, document_info = agentic_chunking.process_and_save_document(document_text, document_id)
            
            if is_duplicate:
                logger.info(f"⏭️ Document {doc_name} is a complete duplicate at chunk level. No further processing needed.")
                return True
                
            logger.info(f"✅ Document chunking completed: {document_info}")
            
    except Exception as e:
        logger.error(f"❌ Error during document chunking: {str(e)}")
        return False
    
    # If we only want to do chunking, we're done
    if max_step == STEP_CHUNKING:
        logger.info(f"🏁 Pipeline completed at step {max_step} (Chunking only) for {doc_name}")
        return True

    # Step 6: Extract requirements from document chunks
    logger.info(f"Step 6: Extracting requirements from document chunks")

    # Extract document ID from filename for requirements extraction
    doc_id = os.path.basename(doc_path)

    # Process requirements
    try:
        if dry_run:
            logger.info(f"🔍 Dry run mode - skipping requirements extraction")
            logger.info(f"✅ Dry run requirements check completed (would have extracted requirements)")
        else:
            extract_requirements.process_single_document(doc_id)
            logger.info(f"✅ Requirements extraction completed for {doc_name}")
    except Exception as e:
        logger.error(f"❌ Requirements extraction failed: {str(e)}")
        return False

    end_time = time.time()
    logger.info(f"🏁 Pipeline completed in {end_time - start_time:.2f} seconds for {doc_name}")
    return True


def main():
    """Run the pipeline with configurable options"""
    parser = argparse.ArgumentParser(description="Document processing pipeline controller")
    parser.add_argument("--doc_path", type=str, default=DEFAULT_DOC_PATH,
                        help=f"Path to the document to process (default: {DEFAULT_DOC_PATH})")
    parser.add_argument("--max_step", type=int, default=MAX_STEP, choices=[1, 2, 3, 4, 5, 6],
                        help="Maximum step to run in the pipeline (default from MAX_STEP constant)")
    parser.add_argument("--dry_run", action="store_true", default=DRY_RUN,
                        help="Check but don't save to database (default from DRY_RUN constant)")
    
    args = parser.parse_args()
    
    doc_path = args.doc_path
    if not os.path.exists(doc_path):
        logger.error(f"❌ Document not found: {doc_path}")
        return 1

    success = process_document(doc_path, max_step=args.max_step, dry_run=args.dry_run)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
