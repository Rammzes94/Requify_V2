"""
pipeline_controller.py

This script controls the document processing pipeline, orchestrating the end-to-end flow:
1. Perform initial hash-based deduplication to filter out exact duplicates
2. Parse PDF documents to extract text and images
3. Check for document-level duplicates before saving to LanceDB
4. Perform content-aligned chunking of document content
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
import integrated_chunking

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

def process_document(doc_path: str, max_step: int = STEP_EXTRACT_REQS, dry_run: bool = False, skip_hash_check: bool = False) -> bool:
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
        skip_hash_check: If True, skips the hash-based duplicate check
        
    Returns:
        True if processing was successful up to the specified step, False otherwise
    """
    start_time = time.time()
    doc_name = os.path.basename(doc_path)

    logger.info(f"🚀 Starting pipeline for document: {doc_name} (max_step={max_step}, dry_run={dry_run}, skip_hash_check={skip_hash_check})")

    # Step 1: Hash-based deduplication
    if not skip_hash_check:
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

    # Step 5: Perform content-aligned chunking with page-level deduplication
    logger.info(f"Step 5: Chunking document and checking for duplicate chunks")
    
    # Load document text and pages from combined JSON
    try:
        import json
        with open(combined_json_path, 'r', encoding='utf-8') as f:
            doc_data = json.load(f)
        
        # Extract document text (combine all page content)
        document_text = ""
        
        # Prepare page data for integrated_chunking
        document_pages = []
        
        for page_key, page_info in sorted(doc_data.get('pages', {}).items()):
            md_content = page_info.get('md_content', '')
            if md_content:
                document_text += md_content + "\n\n"
                # Create page dictionary
                document_pages.append({
                    'md_content': md_content,
                    'page_number': page_info.get('page_number', 0),
                    'embedding': page_info.get('embedding', []),
                    'pdf_identifier': doc_data.get('pdf_identifier', os.path.basename(doc_path))
                })
        
        document_id = doc_data.get('pdf_identifier', os.path.basename(doc_path))
        
        if not document_text.strip():
            logger.error(f"❌ Document has no text content to chunk")
            return False
            
        # Process document for chunking using integrated_chunking
        if dry_run:
            logger.info(f"🔍 Dry run mode - checking document chunks without saving")
            # TODO: Add dry run logic for integrated_chunking if needed
            chunk_success = True
        else:
            # Process document with integrated chunking, passing both text and page data
            chunk_success = integrated_chunking.process_document(document_text, document_id, document_pages)
            
            if not chunk_success:
                logger.error(f"❌ Agentic chunking failed for {doc_name}")
                return False
                
            logger.info(f"✅ Agentic chunking completed successfully")
            
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
    """Parse command-line arguments and run the pipeline."""
    parser = argparse.ArgumentParser(description='Run the document processing pipeline.')
    parser.add_argument('--doc_path', type=str, help='Path to the document to process')
    parser.add_argument('--dir_path', type=str, help='Directory containing documents to process')
    parser.add_argument('--max_step', type=int, default=STEP_EXTRACT_REQS, 
                        help=f'Maximum step to run (1-{STEP_EXTRACT_REQS})')
    parser.add_argument('--dry_run', action='store_true', 
                        help='Perform checks without modifying the database')
    parser.add_argument('--skip_hash_check', action='store_true',
                        help='Skip the hash-based duplicate check (useful for testing)')
    
    args = parser.parse_args()
    
    if not args.doc_path and not args.dir_path:
        parser.error("Either --doc_path or --dir_path must be specified")
    
    if args.doc_path:
        process_document(args.doc_path, args.max_step, args.dry_run, args.skip_hash_check)
    elif args.dir_path:
        process_directory(args.dir_path, args.max_step, args.dry_run)
        
    return True

if __name__ == "__main__":
    sys.exit(main())
