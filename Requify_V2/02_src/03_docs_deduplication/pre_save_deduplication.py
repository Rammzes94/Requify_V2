"""
pre_save_deduplication.py

This script provides document-level deduplication for the document processing pipeline.
It performs the following operations:
1. Checks newly parsed documents against existing documents in the LanceDB database
2. Uses vector similarity with high thresholds (0.99+) to identify duplicate pages
3. Detects updated versions of existing documents through partial content matching
4. Compares document metadata and embeddings to identify similarities
5. Classifies pages as new, duplicates, or updates to existing content
6. Provides detailed information about duplicate pages and their source documents

The script works as a pre-processing step before saving content to the database,
ensuring only unique or updated content is saved while maintaining references to
duplicate or previous versions for traceability.
"""
# -------------------------------------------------------------------------------------
# Imports & Setup
# -------------------------------------------------------------------------------------
import os
import sys
import logging
import time
from typing import List, Dict, Tuple, Set, Optional
import numpy as np
import pandas as pd
import lancedb
from dotenv import load_dotenv

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import _00_utils
_00_utils.setup_project_directory()

# Setup logging with script prefix
class ScriptLogger(logging.LoggerAdapter):
    def __init__(self, logger, prefix):
        super().__init__(logger, {})
        self.prefix = prefix
        
    def process(self, msg, kwargs):
        return f"{self.prefix}{msg}", kwargs

logger = ScriptLogger(_00_utils.setup_logging(), "[Docs_Deduplication] ")

# Load environment variables
load_dotenv()

# -------------------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------------------
OUTPUT_DIR_BASE = "03_output"  # Define base output directory
LANCEDB_SUBDIR_NAME = "lancedb"  # Subdirectory for LanceDB within 03_output
LANCEDB_TABLE_NAME = "documents"
EMBEDDING_DIMENSION = 1024  # Dimension for e5-large models (must match stable_save_to_lancedb.py)
DUPLICATE_SIMILARITY_THRESHOLD = 0.99  # Cosine similarity threshold for duplicate pages
# LanceDB returns distance for cosine as 1 - similarity. So, distance_threshold = 1 - SIMILARITY_THRESHOLD
DISTANCE_THRESHOLD = 1.0 - DUPLICATE_SIMILARITY_THRESHOLD
TOP_K_RESULTS = 5  # Number of similar items to retrieve for each page query

# -------------------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------------------
def connect_to_lancedb(lancedb_path: str):
    """Connect to LanceDB and return the connection."""
    logger.info(f"Connecting to LanceDB at: {lancedb_path}", extra={"icon": "🔄"})
    if not os.path.exists(lancedb_path):
        logger.warning(f"LanceDB directory does not exist at {lancedb_path}. It will be created when saving.", extra={"icon": "⚠️"})
        return None
    
    try:
        db = lancedb.connect(lancedb_path)
        logger.info(f"Successfully connected to LanceDB", extra={"icon": "✅"})
        return db
    except Exception as e:
        logger.error(f"Failed to connect to LanceDB: {e}", extra={"icon": "❌"})
        return None

def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Normalize an embedding vector to unit length."""
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm

def calculate_cosine_similarity(embed1: np.ndarray, embed2: np.ndarray) -> float:
    """Calculate cosine similarity between two embedding vectors."""
    if len(embed1) != len(embed2):
        raise ValueError(f"Embedding dimensions don't match: {len(embed1)} vs {len(embed2)}")
    
    # Normalize the embeddings
    norm_embed1 = normalize_embedding(embed1)
    norm_embed2 = normalize_embedding(embed2)
    
    # Calculate cosine similarity
    return np.dot(norm_embed1, norm_embed2)

# -------------------------------------------------------------------------------------
# Main Deduplication Logic
# -------------------------------------------------------------------------------------
def check_document_duplicates(new_doc_data: List[Dict], db_connection=None):
    """
    Check if a newly parsed document has duplicate pages in the existing database.
    
    Args:
        new_doc_data: List of dictionaries containing page data for the new document
        db_connection: Optional existing LanceDB connection
        
    Returns:
        Tuple of (duplicate_pages, new_pages, update_pages) where:
            - duplicate_pages is a dict mapping page indexes to their duplicate info
            - new_pages is a list of indexes of pages that are new
            - update_pages is a dict mapping page indexes to existing records they should update
    """
    start_time = time.time()
    
    # Get the document ID from the first page
    if not new_doc_data:
        logger.warning("Empty document data provided", extra={"icon": "⚠️"})
        return {}, [], {}
    
    doc_id = new_doc_data[0].get('pdf_identifier', 'unknown')
    logger.info(f"Checking for duplicates of document: {doc_id}", extra={"icon": "🔄"})
    
    # Construct path to LanceDB
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    lancedb_path = os.path.join(project_root, OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME)
    
    # Connect to LanceDB if not already connected
    db = db_connection if db_connection else connect_to_lancedb(lancedb_path)
    
    # If database doesn't exist yet or table doesn't exist, all pages are new
    if not db or LANCEDB_TABLE_NAME not in db.table_names():
        logger.info(f"No existing database or table found. All {len(new_doc_data)} pages are new.", extra={"icon": "✅"})
        return {}, list(range(len(new_doc_data))), {}
    
    # Open the table
    table = db.open_table(LANCEDB_TABLE_NAME)
    
    # Prepare results
    duplicate_pages = {}  # {index: {'similar_id': id, 'similarity': score}}
    new_pages = []  # List of indices that are new
    update_pages = {}  # {index: {'record_id': id, 'is_newer': bool}}
    
    # Process each page in the new document
    for idx, page_data in enumerate(new_doc_data):
        page_num = page_data.get('page_number', idx + 1)
        embedding = page_data.get('embedding')
        
        if embedding is None:
            logger.warning(f"Page {page_num} has no embedding. Marking as new.", extra={"icon": "⚠️"})
            new_pages.append(idx)
            continue
        
        # Convert embedding to numpy array if it's a list
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        
        # Search for similar pages in database
        try:
            search_results = table.search(embedding).limit(TOP_K_RESULTS).to_df()
            
            if search_results.empty:
                logger.info(f"No similar pages found for page {page_num}. It's new.", extra={"icon": "✅"})
                new_pages.append(idx)
                continue
            
            # Check for duplicates or updates
            found_match = False
            for _, row in search_results.iterrows():
                distance = row['_distance']
                similarity = 1.0 - distance
                
                existing_id = row['pdf_identifier']
                existing_page = row.get('page_number', 'unknown')
                
                # If it's from the same document, it might be an update
                if existing_id == doc_id and existing_page == page_num:
                    # Check which is newer based on timestamp
                    existing_timestamp = pd.to_datetime(row.get('timestamp', '1900-01-01'))
                    new_timestamp = pd.to_datetime(page_data.get('timestamp', '2100-01-01'))
                    
                    # If new page is newer, mark for update
                    if new_timestamp > existing_timestamp:
                        update_pages[idx] = {
                            'record_id': row.name,  # Use row index as record ID
                            'is_newer': True
                        }
                        logger.info(f"Page {page_num} is a newer version of existing page. Marked for update.", extra={"icon": "🔄"})
                    else:
                        # Existing is newer or same age, mark as duplicate
                        duplicate_pages[idx] = {
                            'similar_id': f"{existing_id}_{existing_page}",
                            'similarity': similarity
                        }
                        logger.info(f"Page {page_num} is an older version of existing page. Skipping.", extra={"icon": "⏩"})
                    
                    found_match = True
                    break
                
                # Check if it's a duplicate of a different document
                if similarity >= DUPLICATE_SIMILARITY_THRESHOLD:
                    duplicate_pages[idx] = {
                        'similar_id': f"{existing_id}_{existing_page}",
                        'similarity': similarity
                    }
                    logger.info(f"Page {page_num} is a duplicate of {existing_id} page {existing_page} (similarity: {similarity:.4f}). Skipping.", extra={"icon": "⏩"})
                    found_match = True
                    break
            
            # If no match found, it's a new page
            if not found_match:
                logger.info(f"Page {page_num} has no close matches. It's new.", extra={"icon": "✅"})
                new_pages.append(idx)
                
        except Exception as e:
            logger.error(f"Error searching for similar pages for page {page_num}: {e}", extra={"icon": "❌"})
            # If error occurs, conservatively mark as new
            new_pages.append(idx)
    
    end_time = time.time()
    
    # Log summary
    logger.info(f"Duplicate check completed in {end_time - start_time:.2f} seconds.", extra={"icon": "🏁"})
    logger.info(f"Results for {doc_id}: {len(new_pages)} new pages, {len(duplicate_pages)} duplicates, {len(update_pages)} updates", extra={"icon": "📊"})
    
    return duplicate_pages, new_pages, update_pages

def get_document_pages_by_id(doc_id: str, db_connection=None):
    """
    Get all pages for a specific document ID from the database.
    
    Args:
        doc_id: PDF identifier to search for
        db_connection: Optional existing LanceDB connection
        
    Returns:
        DataFrame containing all pages for the document
    """
    # Construct path to LanceDB
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    lancedb_path = os.path.join(project_root, OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME)
    
    # Connect to LanceDB if not already connected
    db = db_connection if db_connection else connect_to_lancedb(lancedb_path)
    
    if not db or LANCEDB_TABLE_NAME not in db.table_names():
        logger.warning(f"No existing database or table found when searching for document {doc_id}", extra={"icon": "⚠️"})
        return pd.DataFrame()
    
    # Open the table
    table = db.open_table(LANCEDB_TABLE_NAME)
    
    try:
        # Query for all pages with the given document ID
        query = f"SELECT * FROM {LANCEDB_TABLE_NAME} WHERE pdf_identifier = '{doc_id}'"
        results = table.query(query).to_df()
        
        logger.info(f"Found {len(results)} existing pages for document {doc_id}", extra={"icon": "✅"})
        return results
    except Exception as e:
        logger.error(f"Error retrieving pages for document {doc_id}: {e}", extra={"icon": "❌"})
        return pd.DataFrame()

def check_for_document_version_update(new_doc_data: List[Dict], db_connection=None) -> Tuple[bool, float, Optional[str]]:
    """
    Check if a document is a new version of an existing document.
    
    Args:
        new_doc_data: List of dictionaries containing page data for the new document
        db_connection: Optional existing LanceDB connection
        
    Returns:
        Tuple of (is_new_version, avg_similarity, old_doc_id) where:
            - is_new_version is a boolean indicating if this appears to be a new version
            - avg_similarity is the average similarity between matching pages
            - old_doc_id is the ID of the old document version, if found
    """
    if not new_doc_data:
        return False, 0.0, None
    
    # Get the document ID and title from the first page
    doc_id = new_doc_data[0].get('pdf_identifier', 'unknown')
    doc_title = new_doc_data[0].get('document_title', '')
    
    # Don't check if doc_id is already in the database - that's handled separately
    
    # Construct path to LanceDB
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    lancedb_path = os.path.join(project_root, OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME)
    
    # Connect to LanceDB if not already connected
    db = db_connection if db_connection else connect_to_lancedb(lancedb_path)
    
    if not db or LANCEDB_TABLE_NAME not in db.table_names():
        return False, 0.0, None
    
    # Open the table
    table = db.open_table(LANCEDB_TABLE_NAME)
    
    # Get all unique document titles excluding the current document ID
    try:
        # For LanceDB 0.22.0, use pandas to filter
        all_docs = table.to_pandas()
        titles_df = all_docs[all_docs['pdf_identifier'] != doc_id][['pdf_identifier', 'document_title']].drop_duplicates()
    except Exception as e:
        logger.error(f"Error querying document titles: {e}", extra={"icon": "❌"})
        return False, 0.0, None
    
    if titles_df.empty:
        return False, 0.0, None
    
    # Find documents with similar titles
    similar_docs = []
    for _, row in titles_df.iterrows():
        other_id = row['pdf_identifier']
        other_title = row['document_title']
        
        # Simple string similarity (Jaccard index) for titles
        if not doc_title or not other_title:
            continue
            
        doc_title_words = set(doc_title.lower().split())
        other_title_words = set(other_title.lower().split())
        
        intersection = len(doc_title_words.intersection(other_title_words))
        union = len(doc_title_words.union(other_title_words))
        
        if union == 0:
            continue
            
        title_similarity = intersection / union
        
        if title_similarity > 0.6:  # Threshold for title similarity
            similar_docs.append((other_id, title_similarity))
    
    if not similar_docs:
        return False, 0.0, None
    
    # For each similar document, check content similarity
    best_match = None
    highest_avg_similarity = 0.0
    
    for other_id, title_sim in similar_docs:
        # Get sample pages from the other document
        try:
            # For LanceDB 0.22.0, use pandas to filter
            other_doc_pages = table.to_pandas()
            other_doc_pages = other_doc_pages[other_doc_pages['pdf_identifier'] == other_id].head(5)
        except Exception as e:
            logger.error(f"Error querying pages for document {other_id}: {e}", extra={"icon": "❌"})
            continue
        
        if other_doc_pages.empty:
            continue
        
        # Check content similarity between pages
        page_similarities = []
        
        for i, other_page in other_doc_pages.iterrows():
            other_embedding = other_page['embedding']
            if isinstance(other_embedding, list):
                other_embedding = np.array(other_embedding)
            
            # Compare with pages from new document
            for new_page in new_doc_data[:5]:  # Limit to first 5 pages for efficiency
                new_embedding = new_page.get('embedding')
                if new_embedding is None:
                    continue
                    
                if isinstance(new_embedding, list):
                    new_embedding = np.array(new_embedding)
                
                try:
                    similarity = calculate_cosine_similarity(new_embedding, other_embedding)
                    page_similarities.append(similarity)
                except Exception as e:
                    logger.error(f"Error calculating similarity: {e}", extra={"icon": "❌"})
        
        if page_similarities:
            avg_similarity = sum(page_similarities) / len(page_similarities)
            if avg_similarity > highest_avg_similarity:
                highest_avg_similarity = avg_similarity
                best_match = other_id
    
    # Determine if it's a new version
    is_new_version = highest_avg_similarity > 0.8 and highest_avg_similarity < 0.99
    
    if is_new_version:
        logger.info(f"Document {doc_id} appears to be a new version of {best_match} (avg similarity: {highest_avg_similarity:.4f})", extra={"icon": "🔄"})
    
    return is_new_version, highest_avg_similarity, best_match

# -------------------------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------------------------
def check_new_document(doc_data: List[Dict]) -> Dict:
    """
    Main function to check a new document against the database.
    
    Args:
        doc_data: List of dictionaries containing page data for the new document
        
    Returns:
        Dictionary with results of the check:
        {
            'duplicate_pages': {idx: info},
            'new_pages': [idx1, idx2, ...],
            'update_pages': {idx: info},
            'is_new_version': bool,
            'old_version_id': str or None,
            'version_similarity': float
        }
    """
    if not doc_data:
        logger.warning("Empty document data provided", extra={"icon": "⚠️"})
        return {
            'duplicate_pages': {},
            'new_pages': [],
            'update_pages': {},
            'is_new_version': False,
            'old_version_id': None,
            'version_similarity': 0.0
        }
    
    # Connect to LanceDB
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    lancedb_path = os.path.join(project_root, OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME)
    db = connect_to_lancedb(lancedb_path)
    
    # First check if this is a new version of an existing document
    is_new_version, version_similarity, old_version_id = check_for_document_version_update(doc_data, db)
    
    # Then check for page-level duplicates
    duplicate_pages, new_pages, update_pages = check_document_duplicates(doc_data, db)
    
    return {
        'duplicate_pages': duplicate_pages,
        'new_pages': new_pages,
        'update_pages': update_pages,
        'is_new_version': is_new_version,
        'old_version_id': old_version_id,
        'version_similarity': version_similarity
    }

if __name__ == "__main__":
    logger.info("This script is designed to be imported and used by the document processing pipeline.", extra={"icon": "ℹ️"})
    logger.info("It checks for duplicates before adding new documents to the database.", extra={"icon": "ℹ️"}) 