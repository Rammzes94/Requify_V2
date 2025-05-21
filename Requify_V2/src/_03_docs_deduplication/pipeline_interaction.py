"""
pipeline_interaction.py

This script serves as an interface between the pipeline controller and user interaction components.
It provides functions to:
1. Handle duplicate document detection with configurable thresholds
2. Present options to the user when duplicates are found
3. Manage the document merging/updating process
4. Track document lineage for version history

The module streamlines how the pipeline responds to duplicate or similar documents,
providing a unified interface for both automated and interactive decisions.
"""

import os
import sys
import logging
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
import lancedb
from dotenv import load_dotenv

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import _00_utils
_00_utils.setup_project_directory()

# Import user interaction utilities
from _03_docs_deduplication import user_interaction

# Setup logging with script prefix


logger = _00_utils.get_logger("Pipeline_Interaction")

# Load environment variables
load_dotenv()

# Constants
OUTPUT_DIR_BASE = "output"
LANCEDB_SUBDIR_NAME = "lancedb"
LANCEDB_DIR_PATH = os.path.join(OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME)
DOCUMENTS_TABLE_NAME = "documents"
DOCUMENT_CHUNKS_TABLE_NAME = "document_chunks"
FILE_HASHES_TABLE_NAME = "file_hashes"

# Threshold constants
EXACT_DUPLICATE_THRESHOLD = 0.995  # For considering as exact duplicate
SIMILAR_DOCUMENT_THRESHOLD = 0.82  # For considering as a similar document (lowered to match pre_save_deduplication.py)

def connect_to_lancedb():
    """Connect to LanceDB and return the connection."""
    logger.info(f"Connecting to LanceDB at: {LANCEDB_DIR_PATH}", extra={"icon": "🔄"})
    if not os.path.exists(LANCEDB_DIR_PATH):
        logger.warning(
            f"LanceDB directory does not exist at {LANCEDB_DIR_PATH}. Will be created when saving.",
            extra={"icon": "⚠️"}
        )
        return None

    try:
        db = lancedb.connect(LANCEDB_DIR_PATH)
        logger.info(f"Successfully connected to LanceDB", extra={"icon": "✅"})
        return db
    except Exception as e:
        logger.error(f"Failed to connect to LanceDB: {e}", extra={"icon": "❌"})
        return None

def check_document_similarity(new_doc: Dict, existing_doc: Dict) -> float:
    """
    Check the similarity between two documents based on their page embeddings.
    
    Args:
        new_doc: Dictionary with information about the new document including page embeddings
        existing_doc: Dictionary with information about an existing document
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    # Extract embeddings from both documents
    new_embeddings = []
    existing_embeddings = []
    
    # Collect page embeddings
    for page in new_doc.get('pages', []):
        if 'embedding' in page:
            new_embeddings.append(np.array(page['embedding']))
    
    for page in existing_doc.get('pages', []):
        if 'embedding' in page:
            existing_embeddings.append(np.array(page['embedding']))
    
    if not new_embeddings or not existing_embeddings:
        logger.warning("Missing embeddings for similarity comparison", extra={"icon": "⚠️"})
        return 0.0
    
    # Calculate cosine similarity between all pairs
    similarities = []
    for new_emb in new_embeddings:
        for exist_emb in existing_embeddings:
            sim = cosine_similarity(new_emb, exist_emb)
            similarities.append(sim)
    
    # Return average similarity
    return sum(similarities) / len(similarities) if similarities else 0.0

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    # Normalize vectors
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return float(np.dot(a_norm, b_norm))

def find_duplicate_documents(new_doc: Dict, db_connection=None) -> Tuple[bool, Optional[Dict], float]:
    """
    Find potential duplicate documents in the database.
    
    Args:
        new_doc: Dictionary with the new document's information
        db_connection: Optional existing LanceDB connection
        
    Returns:
        Tuple of (is_duplicate, duplicate_doc, similarity_score)
    """
    db = db_connection or connect_to_lancedb()
    if not db or DOCUMENTS_TABLE_NAME not in db.table_names():
        logger.info("No documents table found or empty database.", extra={"icon": "ℹ️"})
        return False, None, 0.0
    
    docs_table = db.open_table(DOCUMENTS_TABLE_NAME)
    existing_docs = docs_table.to_pandas()
    
    # Group by document_id to get all pages for each document
    grouped_docs = {}
    for _, row in existing_docs.iterrows():
        doc_id = row.get('pdf_identifier', row.get('document_id', ''))
        if doc_id:
            if doc_id not in grouped_docs:
                grouped_docs[doc_id] = {'pages': []}
            grouped_docs[doc_id]['pages'].append(row.to_dict())
    
    # Check similarity against each document
    best_match = None
    best_similarity = 0.0
    
    for doc_id, doc_data in grouped_docs.items():
        similarity = check_document_similarity(new_doc, doc_data)
        
        if similarity > best_similarity:
            best_similarity = similarity
            doc_data['document_id'] = doc_id  # Add document ID to the data
            best_match = doc_data
    
    # Determine if it's an exact duplicate or similar document
    if best_similarity >= EXACT_DUPLICATE_THRESHOLD:
        logger.info(f"Found exact duplicate: {best_match['document_id']} (similarity: {best_similarity:.4f})", 
                   extra={"icon": "🔍"})
        return True, best_match, best_similarity
    elif best_similarity >= SIMILAR_DOCUMENT_THRESHOLD:
        logger.info(f"Found similar document: {best_match['document_id']} (similarity: {best_similarity:.4f})", 
                   extra={"icon": "🔍"})
        return False, best_match, best_similarity
    
    return False, None, best_similarity

def deduplication_prompt(new_doc_id, existing_doc_id, similarity, db_connection=None):
    """
    Interactive prompt for handling duplicate documents.
    
    Args:
        new_doc_id: The identifier of the new document
        existing_doc_id: The identifier of the existing similar document
        similarity: The similarity score between the documents
        db_connection: Optional LanceDB connection
        
    Returns:
        str: The action to take ('keep_old', 'keep_new', 'keep_both', or 'detailed')
    """
    print("\n" + "="*60)
    print("🔍 DOCUMENT SIMILARITY DETECTED")
    print("="*60)
    print(f"📄 New document: {new_doc_id}")
    print(f"📜 Existing document: {existing_doc_id}")
    print(f"📊 Similarity score: {similarity:.4f}")
    print("="*60)
    
    print("\nWhat would you like to do?")
    print("1. Keep only the existing document (discard new)")
    print("2. Replace with the new document (discard old)")
    print("3. Keep both as separate documents")
    print("4. Perform detailed chunk-level analysis")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            if choice == '1':
                return "keep_old"
            elif choice == '2':
                return "keep_new" 
            elif choice == '3':
                return "keep_both"
            elif choice == '4':
                return "detailed"
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
        except (EOFError, KeyboardInterrupt):
            print("\n❌ Operation cancelled by user.")
            return "keep_old"  # Default to safe option
        except Exception as e:
            print(f"❌ Error reading input: {e}")
            return "keep_old"  # Default to safe option

def handle_document_similarity(new_doc_records: List[Dict], dedup_results: Dict) -> Tuple[bool, str]:
    """
    Handle document similarity detection by prompting the user for action.
    
    Args:
        new_doc_records: List of records for the new document
        dedup_results: Results from the deduplication check
        
    Returns:
        Tuple of (should_continue, action) where:
        - should_continue: Whether to proceed with saving
        - action: What action to take ('replace_old', 'keep_both', 'detailed', etc.)
    """
    if not new_doc_records:
        logger.error("No document records provided", extra={"icon": "❌"})
        return False, "skip"
    
    # Get document information
    new_doc_id = new_doc_records[0].get('pdf_identifier', 'unknown')
    old_doc_id = dedup_results.get('old_version_id', 'unknown')
    similarity = dedup_results.get('version_similarity', 0.0)
    
    logger.info(f"Detected new version: {old_doc_id} (similarity: {similarity:.4f}). Prompting user for action.", extra={"icon": "🔄"})
    
    # Prompt user for decision
    action = deduplication_prompt(new_doc_id, old_doc_id, similarity)
    
    if action == "keep_old":
        logger.info(f"User chose to keep existing document {old_doc_id}", extra={"icon": "✅"})
        return False, "skip"
        
    elif action == "keep_new":
        logger.info(f"User chose to replace {old_doc_id} with {new_doc_id}", extra={"icon": "🔄"})
        return True, "replace_old"
        
    elif action == "keep_both":
        logger.info(f"User chose to keep both documents", extra={"icon": "📋"})
        return True, "keep_both"
        
    elif action == "detailed":
        logger.info(f"User chose detailed chunk-level analysis", extra={"icon": "🔍"})
        return True, "detailed"
        
    else:
        logger.warning(f"Unknown action: {action}. Defaulting to skip.", extra={"icon": "⚠️"})
        return False, "skip"

def handle_duplicate_document(new_doc: Dict, duplicate_doc: Dict, similarity: float, db_connection=None) -> Dict:
    """
    Handle a duplicate document scenario based on user choice.
    
    Args:
        new_doc: Information about the new document
        duplicate_doc: Information about the duplicate document
        similarity: Similarity score between the documents
        db_connection: Optional existing LanceDB connection
        
    Returns:
        Dictionary with the action result
    """
    new_doc_id = new_doc.get('document_id', new_doc.get('pdf_identifier', 'new_document'))
    existing_doc_id = duplicate_doc.get('document_id', 'existing_document')
    
    # Get user choice
    choice = deduplication_prompt(new_doc_id, existing_doc_id, similarity, db_connection)
    
    result = {
        'action': choice,
        'new_doc_id': new_doc_id,
        'existing_doc_id': existing_doc_id,
        'similarity': similarity
    }
    
    db = db_connection or connect_to_lancedb()
    if not db:
        logger.error("Could not connect to database", extra={"icon": "❌"})
        return result
    
    # Perform the chosen action
    if choice == 'keep_old':
        logger.info(f"Keeping existing document {existing_doc_id}, discarding {new_doc_id}", 
                   extra={"icon": "✅"})
        # No action needed, just don't save the new document
        
    elif choice == 'keep_new':
        logger.info(f"Replacing {existing_doc_id} with {new_doc_id}", extra={"icon": "🔄"})
        # This is handled by the calling code - we'll save the new doc and remove old
        result['replace_old'] = True
        
    elif choice == 'keep_both':
        logger.info(f"Keeping both documents as separate entries", extra={"icon": "📋"})
        # This is handled by the calling code - we'll save both docs
        result['keep_both'] = True
        
    elif choice == 'detailed':
        logger.info(f"Performing detailed deduplication between {existing_doc_id} and {new_doc_id}", 
                   extra={"icon": "🔍"})
        
        # This would be implemented in a separate function
        # For now, just mark it for detailed handling
        result['perform_detailed_dedup'] = True
    
    return result

def process_document_update(new_doc: Dict, old_doc: Dict, missing_chunks: List) -> Dict:
    """
    Process an update to an existing document, handling missing chunks.
    
    Args:
        new_doc: Information about the new document
        old_doc: Information about the old document
        missing_chunks: List of chunks that are in the old document but not in the new one
        
    Returns:
        Dictionary with the result of the update process
    """
    new_doc_id = new_doc.get('document_id', 'new_document')
    old_doc_id = old_doc.get('document_id', 'old_document')
    
    # Use user_interaction module to handle the update
    action, chunks_to_keep = user_interaction.handle_document_update(
        new_doc_id, old_doc_id, missing_chunks
    )
    
    result = {
        'action': action,
        'new_doc_id': new_doc_id,
        'old_doc_id': old_doc_id,
        'kept_chunks': len(chunks_to_keep) if chunks_to_keep else 0
    }
    
    if action == 'replace':
        logger.info(f"Replacing {old_doc_id} with {new_doc_id} (discarding {len(missing_chunks)} chunks)", 
                   extra={"icon": "🔄"})
        result['chunks_to_keep'] = []
        
    elif action == 'merge':
        logger.info(f"Merging {old_doc_id} into {new_doc_id} (keeping {len(chunks_to_keep)} chunks)", 
                   extra={"icon": "🔄"})
        result['chunks_to_keep'] = chunks_to_keep
        
    elif action == 'keep':
        logger.info(f"Keeping both {old_doc_id} and {new_doc_id} as separate documents", 
                   extra={"icon": "📋"})
        result['chunks_to_keep'] = []
        
    # Display summary
    user_interaction.display_document_comparison(
        {'document_id': old_doc_id, 'total_chunks': len(missing_chunks)}, 
        {'document_id': new_doc_id, 'total_chunks': len(new_doc.get('chunks', []))}
    )
    
    return result

def main(new_doc_path: str):
    """Test function for the pipeline interaction module."""
    logger.info(f"Testing pipeline interaction with document: {new_doc_path}", extra={"icon": "🔄"})
    
    # This would normally be provided by the pipeline controller
    test_new_doc = {
        'document_id': os.path.basename(new_doc_path),
        'pages': [
            {'embedding': np.random.rand(1024), 'page_number': 1},
            {'embedding': np.random.rand(1024), 'page_number': 2}
        ]
    }
    
    # Test duplicate detection
    db = connect_to_lancedb()
    if db and DOCUMENTS_TABLE_NAME in db.table_names():
        is_duplicate, duplicate_doc, similarity = find_duplicate_documents(test_new_doc, db)
        
        if is_duplicate or (duplicate_doc and similarity >= SIMILAR_DOCUMENT_THRESHOLD):
            result = handle_duplicate_document(test_new_doc, duplicate_doc, similarity, db)
            logger.info(f"Duplicate handling result: {result}", extra={"icon": "✅"})
        else:
            logger.info(f"No duplicates found for {test_new_doc['document_id']}", extra={"icon": "✅"})
    else:
        logger.info("No database or documents table available for testing", extra={"icon": "ℹ️"})

if __name__ == "__main__":
    if len(sys.argv) > 1:
        doc_path = sys.argv[1]
        main(doc_path)
    else:
        logger.error("No document path provided for testing", extra={"icon": "❌"})
        print("Usage: python pipeline_interaction.py <document_path>")