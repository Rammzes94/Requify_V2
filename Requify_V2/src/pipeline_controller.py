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
import numpy as np

# _00_utils will be imported using an absolute path assuming project root is in sys.path
# The setup_project_directory call should ideally be done once at the entry point (e.g. test_scenarios.py or main script)
# For now, we keep it, but use absolute import for _00_utils.
from src import _00_utils
_00_utils.setup_project_directory() # This might still be problematic if it assumes CWD.

# Load environment variables
load_dotenv()

# Setup logging with script prefix
logger = _00_utils.get_logger("Pipeline_Controller")

# Import Hash-based deduplication
from src._01_ingestion import file_hash_deduplication

# Import PDF parsing module
from src._02_parsing import stable_pdf_parsing
from src._02_parsing import consolidated_chunking # Assuming this is also in _02_parsing

# Import document deduplication modules
from src._03_docs_deduplication import pre_save_deduplication
from src._03_docs_deduplication import user_interaction

# Import LanceDB saving module
from src._02_parsing import stable_save_to_lancedb

# Import requirements extraction module
from src._04_extract_reqs import extract_requirements

# Constants
DEFAULT_DOC_PATH = os.path.join("input", "raw", "fighter_jet_rocket_launcher_spec_2.pdf")
DEFAULT_OUTPUT_DIR = "output"
LANCEDB_SUBDIR_NAME = "lancedb"  # Added to ensure consistency

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

    logger.info(f"Starting pipeline for document: {doc_name} (max_step={max_step}, dry_run={dry_run}, skip_hash_check={skip_hash_check})", extra={"icon": "🚀"})

    # Step 1: Hash-based deduplication
    if not skip_hash_check:
        logger.info(f"Step 1: Performing hash-based duplicate check for {doc_name}", extra={"icon": "🔐"})
        is_duplicate, existing_file = file_hash_deduplication.check_file_duplicate(doc_path)

        if is_duplicate:
            logger.info(f"Document {doc_name} is an exact duplicate of {existing_file} (by hash). Aborting further processing.", extra={"icon": "❌"})
            return True  # Successfully determined it's a duplicate, so pipeline "succeeded"
        
        logger.info(f"Document {doc_name} passed hash-based duplicate check", extra={"icon": "✅"})
    
    # If we only want to do the hash check, we're done
    if max_step == STEP_HASH_CHECK:
        logger.info(f"Pipeline completed at step {max_step} (Hash check only) for {doc_name}", extra={"icon": "🏁"})
        return True
    
    # Step 2: Parse the PDF document
    logger.info(f"Step 2: Parsing document {doc_name}", extra={"icon": "📑"})
    processor = stable_pdf_parsing.PDFProcessor(
        stable_pdf_parsing.plain_agent,
        stable_pdf_parsing.structured_agent,
        stable_pdf_parsing.document_title_agent
    )

    combined_json_path = processor.pdf_to_structured_json(doc_path)

    if not combined_json_path or not os.path.exists(combined_json_path):
        logger.error(f"Document parsing failed for {doc_name} (no combined JSON output)", extra={"icon": "❌"})
        return False

    logger.info(f"Document parsing completed: {combined_json_path}", extra={"icon": "✅"})
    
    # If we only want to parse, we're done
    if max_step == STEP_PARSE:
        logger.info(f"Pipeline completed at step {max_step} (Parse only) for {doc_name}", extra={"icon": "🏁"})
        return True

    # Step 3: Check for document-level duplicates
    logger.info(f"Step 3: Checking document for duplicates", extra={"icon": "🔍"})
    
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
            version_similarity = dedup_results.get('version_similarity', 0.0)
            
            logger.info(f"Deduplication check results for {doc_name}:", extra={"icon": "✅"})
            logger.info(f"   - New pages: {len(new_pages)}", extra={"icon": "🆕"})
            logger.info(f"   - Duplicate pages: {len(duplicate_pages)}", extra={"icon": "♻️"})
            logger.info(f"   - Pages to update: {len(update_pages)}", extra={"icon": "🔄"})
            
            if is_new_version and old_version_id:
                logger.info(f"   - Document appears to be a new version of {old_version_id} (similarity: {version_similarity:.4f})", extra={"icon": "🔄"})
            
            # Also check if ANY of the pages have high similarity to existing pages
            # This might trigger context-aware chunking even if the whole document isn't detected as a new version
            similar_document_detected = False
            similar_document_id = None
            
            # First check if documents have similar content based on embedding similarity
            # Calculate similarity between document and all other documents
            if not is_new_version and not old_version_id:
                from src._03_docs_deduplication.pre_save_deduplication import calculate_cosine_similarity
                from src._03_docs_deduplication.pre_save_deduplication import SIMILAR_THRESHOLD
                import numpy as np  # Add numpy import here to ensure it's available in this scope
                
                # Connect to database
                from src._03_docs_deduplication.pre_save_deduplication import connect_to_lancedb
                script_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.abspath(os.path.join(script_dir, '..'))
                lancedb_path = os.path.join(project_root, DEFAULT_OUTPUT_DIR, LANCEDB_SUBDIR_NAME)
                db = connect_to_lancedb(lancedb_path)
                
                if db and "documents" in db.table_names():
                    doc_table = db.open_table("documents")
                    doc_df = doc_table.to_pandas()
                    
                    # Calculate similarity between current document's embedding and all others
                    current_embedding = np.array(pages_data[0].get('embedding', []))
                    if len(current_embedding) > 0:
                        highest_similarity = 0.0
                        most_similar_doc_id = None
                        
                        # Get unique document IDs from the database
                        unique_doc_ids = doc_df['pdf_identifier'].unique()
                        
                        for other_doc_id in unique_doc_ids:
                            # Skip current document
                            if other_doc_id == doc_data.get('pdf_identifier', os.path.basename(doc_path)):
                                continue
                                
                            # Get all pages for this document
                            doc_pages = doc_df[doc_df['pdf_identifier'] == other_doc_id]
                            
                            # Calculate similarity with each page
                            for _, page in doc_pages.iterrows():
                                other_embedding = np.array(page.get('embedding', []))
                                if len(other_embedding) > 0:
                                    sim = calculate_cosine_similarity(current_embedding, other_embedding)
                                    if sim > highest_similarity:
                                        highest_similarity = sim
                                        most_similar_doc_id = other_doc_id
                        
                        # If we found a similar document, use it
                        if highest_similarity >= SIMILAR_THRESHOLD and most_similar_doc_id:
                            logger.info(f"📊 Document {doc_data.get('pdf_identifier', os.path.basename(doc_path))} has high content similarity ({highest_similarity:.4f}) with document {most_similar_doc_id}", extra={"icon": "🔍"})
                            similar_document_detected = True
                            similar_document_id = most_similar_doc_id
            
            # If we haven't found similarity at document level, try page level
            if not similar_document_detected and not similar_document_id:
                # Check if ANY page has a match with high similarity
                for idx in duplicate_pages:
                    dup_info = duplicate_pages[idx]
                    similarity = dup_info.get('similarity', 0)
                    if similarity >= SIMILAR_THRESHOLD:  # Using similarity threshold from deduplication module
                        similar_id = dup_info.get('similar_id', '')
                        if similar_id:
                            doc_id = similar_id.split('_')[0]  # Extract document ID from page ID
                            logger.info(f"📊 Page {idx+1} has high similarity ({similarity:.4f}) with document {doc_id}", extra={"icon": "🔍"})
                            similar_document_detected = True
                            similar_document_id = doc_id
                            break
            
            logger.info(f"📊 Using page-level deduplication info: {len(duplicate_pages)} duplicates")
            
            logger.info(f"🏁 Pipeline completed at step {max_step} (Deduplication check only) for {doc_name}", extra={"icon": "🏁"})
            return True
        except Exception as e:
            logger.error(f"❌ Deduplication check failed: {str(e)}", extra={"icon": "❌"})
            return False

    # Step 4: Save to LanceDB with deduplication
    logger.info(f"Step 4: Saving document to LanceDB with deduplication", extra={"icon": "💾"})
    
    if dry_run:
        logger.info(f"🔍 Dry run mode - checking document without saving to database", extra={"icon": "🔍"})
        logger.info(f"✅ Dry run check completed (would have saved to LanceDB)", extra={"icon": "✅"})
        save_success = True
    else:
        save_success = stable_save_to_lancedb.main(combined_json_path)

    if not save_success:
        logger.error(f"❌ Saving to LanceDB failed for {doc_name} (deduplication or DB error)", extra={"icon": "❌"})
        return False

    logger.info(f"✅ Document saved to LanceDB successfully", extra={"icon": "✅"})
    
    # If we only want to save to LanceDB (no chunking), we're done
    if max_step == STEP_SAVE_TO_DB:
        logger.info(f"🏁 Pipeline completed at step {max_step} (Save to LanceDB only) for {doc_name}", extra={"icon": "🏁"})
        return True

    # Step 5: Perform content-aligned chunking with page-level deduplication
    logger.info(f"Step 5: Chunking document and checking for duplicate chunks", extra={"icon": "🔄"})
    
    # Load document text and pages from combined JSON
    try:
        import json
        with open(combined_json_path, 'r', encoding='utf-8') as f:
            doc_data = json.load(f)
        
        # Extract document text (combine all page content)
        document_text = ""
        
        # Prepare page data for consolidated chunking
        document_pages = []
        dedup_info = {}
        old_version_id = None
        document_id = doc_data.get('pdf_identifier', os.path.basename(doc_path))
        
        # Check if there's a previous document with similar content
        # We do this explicitly before chunking to decide on context-aware chunking
        try:
            # Import necessary functions from deduplication module
            from src._03_docs_deduplication import pre_save_deduplication as dedup
            from src._03_docs_deduplication.pre_save_deduplication import (
                calculate_cosine_similarity, SIMILAR_THRESHOLD, connect_to_lancedb
            )
            import numpy as np
            
            # Connect to database
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(script_dir, '..'))
            lancedb_path = os.path.join(project_root, DEFAULT_OUTPUT_DIR, LANCEDB_SUBDIR_NAME)
            db = connect_to_lancedb(lancedb_path)
            
            # Determine if there's a similar document already in the database
            similar_document_detected = False
            similar_document_id = None
            similarity_score = 0.0
            
            if db and "documents" in db.table_names():
                doc_table = db.open_table("documents")
                doc_df = doc_table.to_pandas()
                
                if not doc_df.empty:
                    # Get document contents for comparison
                    # For single-page docs with embedded markdown content
                    doc_content = ""
                    if 'md_content' in doc_data:
                        doc_content = doc_data.get('md_content', '')
                    # For multi-page documents
                    elif 'pages' in doc_data:
                        for _, page_info in sorted(doc_data.get('pages', {}).items()):
                            doc_content += page_info.get('md_content', '') + "\n\n"
                    
                    current_embedding = None
                    
                    # Try to get embedding from the document itself
                    if 'embedding' in doc_data:
                        current_embedding = np.array(doc_data.get('embedding', []))
                    # Or from the first page for multi-page docs
                    elif 'pages' in doc_data and len(doc_data.get('pages', {})) > 0:
                        first_page_key = list(sorted(doc_data.get('pages', {}).keys()))[0]
                        first_page = doc_data.get('pages', {}).get(first_page_key, {})
                        current_embedding = np.array(first_page.get('embedding', []))
                    
                    if current_embedding is not None and len(current_embedding) > 0:
                        # Get unique document IDs from the database
                        unique_doc_ids = doc_df['pdf_identifier'].unique()
                        highest_similarity = 0.0
                        most_similar_doc_id = None
                        
                        for other_doc_id in unique_doc_ids:
                            # Skip current document
                            if other_doc_id == document_id:
                                continue
                                
                            # Get all pages for this document
                            doc_pages = doc_df[doc_df['pdf_identifier'] == other_doc_id]
                            
                            # Calculate similarity with each page
                            doc_similarity = 0.0
                            for _, page in doc_pages.iterrows():
                                if 'embedding' in page:
                                    other_embedding = np.array(page.get('embedding', []))
                                    if len(other_embedding) > 0:
                                        sim = calculate_cosine_similarity(current_embedding, other_embedding)
                                        doc_similarity = max(doc_similarity, sim)
                            
                            if doc_similarity > highest_similarity:
                                highest_similarity = doc_similarity
                                most_similar_doc_id = other_doc_id
                        
                        # If we found a similar document with high similarity score
                        if highest_similarity >= SIMILAR_THRESHOLD and most_similar_doc_id:
                            logger.info(f"📊 Document {document_id} has high similarity ({highest_similarity:.4f}) with document {most_similar_doc_id}", extra={"icon": "🔍"})
                            similar_document_detected = True
                            similar_document_id = most_similar_doc_id
                            similarity_score = highest_similarity
                            
                            # Explain the decision to use context-aware chunking
                            if similarity_score >= 0.99:
                                logger.info(f"💯 Documents are nearly identical! Using context-aware chunking.", extra={"icon": "🔍"})
                            elif similarity_score >= 0.95:
                                logger.info(f"🔄 Documents are very similar! Using context-aware chunking to detect changes.", extra={"icon": "🔍"})
                            else:
                                logger.info(f"🔄 Documents have significant similarity. Using context-aware chunking.", extra={"icon": "🔍"})
            
            # Additional check at the chunk level - this helps detect similar documents
            # even if page embeddings don't match closely enough
            if not similar_document_detected and db and "document_chunks" in db.table_names():
                chunks_table = db.open_table("document_chunks")
                chunks_df = chunks_table.to_pandas()
                
                if not chunks_df.empty:
                    # Get unique document IDs from chunks
                    unique_doc_ids = chunks_df['document_id'].unique()
                    doc_similarity_map = {}
                    
                    # For each document ID, count how many chunks are similar
                    for other_doc_id in unique_doc_ids:
                        # Skip current document
                        if other_doc_id == document_id:
                            continue
                        
                        doc_chunks = chunks_df[chunks_df['document_id'] == other_doc_id]
                        
                        # Use this document's content to compare against chunks
                        content_text = doc_content if doc_content else document_text
                        
                        # If we have content to compare and there are chunks
                        if content_text and not doc_chunks.empty:
                            # Count how many chunks have similar content
                            similar_chunks = 0
                            total_chunks = len(doc_chunks)
                            
                            for _, chunk in doc_chunks.iterrows():
                                chunk_text = chunk.get('chunk_text', '')
                                if chunk_text:
                                    # Use a simple text similarity measure first - faster than embeddings
                                    from difflib import SequenceMatcher
                                    similarity = SequenceMatcher(None, content_text, chunk_text).ratio()
                                    
                                    if similarity >= 0.7:  # Direct text similarity threshold
                                        similar_chunks += 1
                            
                            # Calculate percentage of similar chunks
                            if total_chunks > 0:
                                similarity_percentage = similar_chunks / total_chunks
                                doc_similarity_map[other_doc_id] = similarity_percentage
                    
                    # Find the document with highest chunk similarity
                    if doc_similarity_map:
                        most_similar_doc_id, highest_similarity = max(doc_similarity_map.items(), key=lambda x: x[1])
                        
                        # If we have significant similarity in chunks
                        if highest_similarity >= 0.5:  # At least 50% of chunks are similar
                            logger.info(f"📊 Document {document_id} has chunk-level similarity with document {most_similar_doc_id} ({highest_similarity:.2%} of chunks similar)", extra={"icon": "🔍"})
                            similar_document_detected = True
                            similar_document_id = most_similar_doc_id
                            similarity_score = highest_similarity
            
            
        except Exception as e:
            logger.warning(f"⚠️ Error checking document similarity: {str(e)}. Will process as new document.", extra={"icon": "⚠️"})
            similar_document_detected = False
            similar_document_id = None
            similarity_score = 0.0
        
        # Now process the content for chunking
        for page_key, page_info in sorted(doc_data.get('pages', {}).items()):
            page_num = page_info.get('page_number', 0)
            page_idx = page_num - 1  # Convert to 0-based index
            
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
        
        if not document_text.strip():
            logger.error(f"❌ Document has no text content to chunk", extra={"icon": "❌"})
            return False
        
        logger.info(f"🔄 Processing document with {len(document_pages)} pages for chunking", extra={"icon": "🔄"})
        
        # Process document for chunking using consolidated chunking
        if dry_run:
            logger.info(f"🔍 Dry run mode - checking document chunks without saving", extra={"icon": "🔍"})
            # Add dry run logic if needed
            chunk_success = True
        else:
            # If we detected this is a similar document, use context-aware chunking
            if similar_document_detected and similar_document_id:
                logger.info(f"🔄 Using context-aware chunking with reference document: {similar_document_id} (similarity: {similarity_score:.4f})", extra={"icon": "🔍"})
                
                # Process document with consolidated chunking using similar document ID
                chunk_success = consolidated_chunking.process_document(
                    document_text, 
                    document_id, 
                    similar_document_id
                )
            else:
                # Use standard chunking for new documents
                logger.info(f"🔄 Using standard chunking for new document", extra={"icon": "🔄"})
                # Process document with consolidated chunking, no similar document needed
                chunk_success = consolidated_chunking.process_document(document_text, document_id)
            
            if not chunk_success:
                logger.error(f"❌ Chunking failed for {doc_name}", extra={"icon": "❌"})
                return False
                
            logger.info(f"✅ Chunking completed successfully", extra={"icon": "✅"})
            
    except Exception as e:
        logger.error(f"❌ Error during document chunking: {str(e)}", extra={"icon": "❌"})
        return False
    
    # If we only want to do chunking, we're done
    if max_step == STEP_CHUNKING:
        logger.info(f"🏁 Pipeline completed at step {max_step} (Chunking only) for {doc_name}", extra={"icon": "🏁"})
        return True

    # Step 6: Extract requirements from document chunks
    logger.info(f"Step 6: Extracting requirements from document chunks", extra={"icon": "📚"})

    # Extract document ID from filename for requirements extraction
    doc_id = os.path.basename(doc_path)

    # Process requirements
    try:
        if dry_run:
            logger.info(f"🔍 Dry run mode - skipping requirements extraction", extra={"icon": "🔍"})
            logger.info(f"✅ Dry run requirements check completed (would have extracted requirements)", extra={"icon": "✅"})
        else:
            extract_requirements.process_single_document(doc_id)
            logger.info(f"✅ Requirements extraction completed for {doc_name}", extra={"icon": "✅"})
    except Exception as e:
        logger.error(f"❌ Requirements extraction failed: {str(e)}", extra={"icon": "❌"})
        return False

    end_time = time.time()
    logger.info(f"🏁 Pipeline completed in {end_time - start_time:.2f} seconds for {doc_name}", extra={"icon": "🏁"})
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
