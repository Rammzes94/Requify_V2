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
import hashlib
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import lancedb
from dotenv import load_dotenv
from datetime import timedelta

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
OUTPUT_DIR_BASE = "_03_output"  # Define base output directory
LANCEDB_SUBDIR_NAME = "lancedb"  # Subdirectory for LanceDB within _03_output
LANCEDB_TABLE_NAME = "documents"
CHUNKS_TABLE_NAME = "document_chunks"
EMBEDDING_DIMENSION = 1024  # Dimension for e5-large models (must match stable_save_to_lancedb.py)
DUPLICATE_THRESHOLD = 0.995  # Cosine similarity threshold for duplicate pages
# LanceDB returns distance for cosine as 1 - similarity. So, distance_threshold = 1 - SIMILARITY_THRESHOLD
DISTANCE_THRESHOLD = 1.0 - DUPLICATE_THRESHOLD
SIMILAR_THRESHOLD = 0.90  # Threshold for similar pages
MIN_PAGES_TO_SAMPLE = 3  # Minimum number of pages to sample for comparison
MAX_PAGES_TO_SAMPLE = 5  # Maximum number of pages to sample for comparison
VERSION_SIMILARITY_THRESHOLD = 0.9  # Threshold to consider a document as a new version
INDEX_INITIALIZED_DOCS = False
INDEX_INITIALIZED_CHUNKS = False
CHUNK_DUPLICATION_THRESHOLD = 0.995  # Threshold for considering chunks as duplicates
CHUNK_SIMILARITY_THRESHOLD = 0.90  # Threshold for considering chunks as similar

# -------------------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------------------
def connect_to_lancedb(lancedb_path: str):
    """Connect to LanceDB and return the connection."""
    logger.info(f"Connecting to LanceDB at: {lancedb_path}", extra={"icon": "🔄"})
    if not os.path.exists(lancedb_path):
        logger.warning(
            f"LanceDB directory does not exist at {lancedb_path}. It will be created when saving.",
            extra={"icon": "⚠️"}
        )
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
        raise ValueError(
            f"Embedding dimensions don't match: {len(embed1)} vs {len(embed2)}"
        )
    norm_embed1 = normalize_embedding(embed1)
    norm_embed2 = normalize_embedding(embed2)
    return float(np.dot(norm_embed1, norm_embed2))


def create_chunk_hash(chunk_text: str) -> str:
    """
    Create a unique hash for a chunk of text.
    """
    return hashlib.md5(chunk_text.encode('utf-8')).hexdigest()


def ensure_index(db, table_name=None):
    """Ensure ANN index exists on the embedding column."""
    global INDEX_INITIALIZED_DOCS, INDEX_INITIALIZED_CHUNKS
    
    if not db:
        return
        
    if not table_name:
        table_name = LANCEDB_TABLE_NAME
        
    if table_name not in db.table_names():
        return
        
    # Check if this table's index is already initialized
    if (table_name == LANCEDB_TABLE_NAME and INDEX_INITIALIZED_DOCS) or \
       (table_name == CHUNKS_TABLE_NAME and INDEX_INITIALIZED_CHUNKS):
        return
    
    table = db.open_table(table_name)
    
    # Check if there are enough rows to build the index (minimum 256 required)
    row_count = len(table.to_pandas())
    
    if row_count < 256:
        logger.info(f"Not creating index: {table_name} has only {row_count} rows, minimum 256 required", extra={"icon": "⚠️"})
        return
        
    table.create_index(
        metric="cosine",
        vector_column_name="embedding"
    )
    
    if table_name == LANCEDB_TABLE_NAME:
        INDEX_INITIALIZED_DOCS = True
    elif table_name == CHUNKS_TABLE_NAME:
        INDEX_INITIALIZED_CHUNKS = True
        
    logger.info(f"Built ANN index on embedding column for {table_name}", extra={"icon": "✅"})

# -------------------------------------------------------------------------------------
# Chunk Deduplication Logic
# -------------------------------------------------------------------------------------
def check_chunk_duplicates(
    chunks_data: List[Dict], db_connection=None
) -> Tuple[Dict[int, Dict[str, object]], List[int], Dict[int, Dict[str, object]]]:
    """
    Check if newly created chunks have duplicates in the existing database.
    
    Returns:
        duplicate_chunks: dict mapping chunk idx -> {'similar_id', 'similarity', 'hash_match'}
        new_chunks: list of new chunk indices
        update_chunks: dict mapping chunk idx -> {'record_id', 'is_newer'}
    """
    start_time = time.time()
    if not chunks_data:
        logger.warning("Empty chunks data provided", extra={"icon": "⚠️"})
        return {}, [], {}
    
    doc_id = chunks_data[0].get('document_id', 'unknown')
    logger.info(f"Checking for duplicate chunks in document: {doc_id}", extra={"icon": "🔄"})
    
    # Connect to LanceDB
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    lancedb_path = os.path.join(project_root, OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME)
    db = db_connection or connect_to_lancedb(lancedb_path)
    
    if not db or CHUNKS_TABLE_NAME not in db.table_names():
        logger.info(
            f"No existing chunks table found. All {len(chunks_data)} chunks are new.",
            extra={"icon": "✅"}
        )
        return {}, list(range(len(chunks_data))), {}
    
    # Ensure index exists if we have enough chunks
    ensure_index(db, CHUNKS_TABLE_NAME)
    chunks_table = db.open_table(CHUNKS_TABLE_NAME)
    
    # Get a dataframe of all existing chunks for hash comparison
    existing_chunks_df = chunks_table.to_pandas()
    
    duplicate_chunks = {}
    new_chunks = []
    update_chunks = {}
    
    for idx, chunk_data in enumerate(chunks_data):
        chunk_id = chunk_data.get('chunk_id', f"chunk_{idx}")
        chunk_text = chunk_data.get('chunk_text', '')
        chunk_hash = create_chunk_hash(chunk_text)
        
        # Add hash to chunk data for future use
        chunks_data[idx]['chunk_hash'] = chunk_hash
        
        # First check for hash-based duplicates (exact matches)
        # Safely check for chunk_hash in dataframe columns
        if 'chunk_hash' in existing_chunks_df.columns:
            hash_matches = existing_chunks_df[existing_chunks_df['chunk_hash'] == chunk_hash]
            
            if not hash_matches.empty:
                # We have an exact duplicate based on hash
                match_id = hash_matches.iloc[0].get('chunk_id', '')
                duplicate_chunks[idx] = {
                    'similar_id': match_id,
                    'similarity': 1.0,  # Perfect match
                    'hash_match': True
                }
                logger.info(
                    f"Chunk {chunk_id} has an exact hash match with {match_id}. Skipping.",
                    extra={"icon": "⏩"}
                )
                continue
        else:
            logger.warning(f"No 'chunk_hash' column in existing chunks table. Skipping hash-based deduplication.", extra={"icon": "⚠️"})
        
        # If no hash match, check for vector similarity
        embedding = chunk_data.get('embedding')
        if embedding is None:
            logger.warning(f"Chunk {chunk_id} has no embedding. Marking as new.", extra={"icon": "⚠️"})
            new_chunks.append(idx)
            continue
            
        if isinstance(embedding, list):
            embedding = np.array(embedding)
            
        try:
            # Use ANN search if we have an index, otherwise use simple filtering
            if len(existing_chunks_df) >= 256:
                df = (
                    chunks_table.search(embedding)
                        .metric("cosine")
                        .limit(10)
                        .nprobes(32)
                        .refine_factor(5)
                        .to_df()
                )
            else:
                # For small tables, use regular filtering
                results = []
                for _, row in existing_chunks_df.iterrows():
                    exist_emb = row.get('embedding')
                    if exist_emb is not None:
                        sim = calculate_cosine_similarity(embedding, np.array(exist_emb))
                        if sim >= CHUNK_SIMILARITY_THRESHOLD:
                            results.append((row, sim))
                
                # Sort by similarity in descending order
                results.sort(key=lambda x: x[1], reverse=True)
                df = pd.DataFrame([r[0] for r in results[:10]])
                if not df.empty:
                    # Add distance as 1 - similarity to match LanceDB format
                    df['_distance'] = [1.0 - r[1] for r in results[:10]]
            
            if df.empty:
                logger.info(f"No similar chunks found for {chunk_id}. It's new.", extra={"icon": "✅"})
                new_chunks.append(idx)
                continue
                
            found = False
            for _, row in df.iterrows():
                sim = 1.0 - row.get('_distance', 0)
                existing_id = row.get('chunk_id', '')
                existing_doc = row.get('document_id', '')
                
                # Same-doc update check
                if existing_doc == doc_id and existing_id.split('_')[-1] == chunk_id.split('_')[-1]:
                    exist_ts = pd.to_datetime(row.get('timestamp', None))
                    new_ts = pd.to_datetime(chunk_data.get('timestamp', None))
                    if new_ts and exist_ts and new_ts > exist_ts:
                        update_chunks[idx] = {'record_id': row.name, 'is_newer': True}
                        logger.info(
                            f"Chunk {chunk_id} is a newer version. Marked for update.",
                            extra={"icon": "🔄"}
                        )
                    else:
                        duplicate_chunks[idx] = {
                            'similar_id': existing_id,
                            'similarity': sim,
                            'hash_match': False
                        }
                        logger.info(
                            f"Chunk {chunk_id} is an older version. Skipping.",
                            extra={"icon": "⏩"}
                        )
                    found = True
                    break
                    
                # Cross-doc duplicate based on similarity
                if sim >= CHUNK_DUPLICATION_THRESHOLD:
                    duplicate_chunks[idx] = {
                        'similar_id': existing_id,
                        'similarity': sim,
                        'hash_match': False
                    }
                    logger.info(
                        f"Chunk {chunk_id} is similar to {existing_id} (sim={sim:.4f}). Skipping.",
                        extra={"icon": "⏩"}
                    )
                    found = True
                    break
                    
            if not found:
                logger.info(f"Chunk {chunk_id} has no close matches. It's new.", extra={"icon": "✅"})
                new_chunks.append(idx)
                
        except Exception as e:
            logger.error(f"Error searching for chunk {chunk_id}: {e}", extra={"icon": "❌"})
            new_chunks.append(idx)
    
    elapsed = time.time() - start_time
    logger.info(
        f"Chunk deduplication completed in {elapsed:.2f}s: {len(new_chunks)} new, {len(duplicate_chunks)} duplicates, {len(update_chunks)} updates",
        extra={"icon": "📊"}
    )
    return duplicate_chunks, new_chunks, update_chunks

# -------------------------------------------------------------------------------------
# Main Deduplication Logic
# -------------------------------------------------------------------------------------
def check_document_duplicates(
    new_doc_data: List[Dict], db_connection=None
) -> Tuple[Dict[int, Dict[str, object]], List[int], Dict[int, Dict[str, object]]]:
    """
    Check if a newly parsed document has duplicate pages in the existing database.

    Returns:
        duplicate_pages: dict mapping page idx -> {{'similar_id', 'similarity'}}
        new_pages: list of new page indices
        update_pages: dict mapping page idx -> {{'record_id', 'is_newer'}}
    """
    start_time = time.time()
    if not new_doc_data:
        logger.warning("Empty document data provided", extra={"icon": "⚠️"})
        return {}, [], {}

    doc_id = new_doc_data[0].get('pdf_identifier', 'unknown')
    logger.info(f"Checking for duplicates of document: {doc_id}", extra={"icon": "🔄"})

    # Connect and index
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    lancedb_path = os.path.join(project_root, OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME)
    db = db_connection or connect_to_lancedb(lancedb_path)
    if not db or LANCEDB_TABLE_NAME not in db.table_names():
        logger.info(
            f"No existing database or table found. All {len(new_doc_data)} pages are new.",
            extra={"icon": "✅"}
        )
        return {}, list(range(len(new_doc_data))), {}

    ensure_index(db)
    table = db.open_table(LANCEDB_TABLE_NAME)

    duplicate_pages: Dict[int, Dict[str, object]] = {}
    new_pages: List[int] = []
    update_pages: Dict[int, Dict[str, object]] = {}

    for idx, page_data in enumerate(new_doc_data):
        page_num = page_data.get('page_number', idx + 1)
        embedding = page_data.get('embedding')
        if embedding is None:
            logger.warning(f"Page {page_num} has no embedding. Marking as new.", extra={"icon": "⚠️"})
            new_pages.append(idx)
            continue
        if isinstance(embedding, list):
            embedding = np.array(embedding)

        try:
            df = (
                table.search(embedding)
                     .metric("cosine")
                     .limit(MAX_PAGES_TO_SAMPLE)
                     .nprobes(32)
                     .refine_factor(5)
                     .to_df()
            )
            if df.empty:
                logger.info(f"No similar pages found for page {page_num}. It's new.", extra={"icon": "✅"})
                new_pages.append(idx)
                continue

            found = False
            for _, row in df.iterrows():
                sim = 1.0 - row['_distance']
                existing_id = row['pdf_identifier']
                existing_page = row.get('page_number', 'unknown')

                # Same-doc update check
                if existing_id == doc_id and existing_page == page_num:
                    exist_ts = pd.to_datetime(row.get('timestamp', None))
                    new_ts = pd.to_datetime(page_data.get('timestamp', None))
                    if new_ts and exist_ts and new_ts > exist_ts:
                        update_pages[idx] = {'record_id': row.name, 'is_newer': True}
                        logger.info(
                            f"Page {page_num} is a newer version. Marked for update.",
                            extra={"icon": "🔄"}
                        )
                    else:
                        duplicate_pages[idx] = {'similar_id': f"{existing_id}_{existing_page}", 'similarity': sim}
                        logger.info(
                            f"Page {page_num} is an older version. Skipping.",
                            extra={"icon": "⏩"}
                        )
                    found = True
                    break

                # Cross-doc duplicate
                if sim >= DUPLICATE_THRESHOLD:
                    duplicate_pages[idx] = {'similar_id': f"{existing_id}_{existing_page}", 'similarity': sim}
                    logger.info(
                        f"Page {page_num} duplicates {existing_id} page {existing_page} (sim={sim:.4f}). Skipping.",
                        extra={"icon": "⏩"}
                    )
                    found = True
                    break

            if not found:
                logger.info(f"Page {page_num} has no close matches. It's new.", extra={"icon": "✅"})
                new_pages.append(idx)

        except Exception as e:
            logger.error(f"Error searching for page {page_num}: {e}", extra={"icon": "❌"})
            new_pages.append(idx)

    elapsed = time.time() - start_time
    logger.info(
        f"Duplicate check completed in {elapsed:.2f}s: {len(new_pages)} new, {len(duplicate_pages)} dup, {len(update_pages)} updates",
        extra={"icon": "📊"}
    )
    return duplicate_pages, new_pages, update_pages


def get_document_pages_by_id(doc_id: str, db_connection=None) -> pd.DataFrame:
    """
    Get all pages for a specific document ID from the database.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    lancedb_path = os.path.join(project_root, OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME)
    db = db_connection or connect_to_lancedb(lancedb_path)
    if not db or LANCEDB_TABLE_NAME not in db.table_names():
        logger.warning(
            f"No existing database or table found for document {doc_id}",
            extra={"icon": "⚠️"}
        )
        return pd.DataFrame()
    ensure_index(db)
    table = db.open_table(LANCEDB_TABLE_NAME)
    try:
        df = table.to_pandas()
        res = df[df['pdf_identifier'] == doc_id]
        logger.info(f"Found {len(res)} pages for {doc_id}", extra={"icon": "✅"})
        return res
    except Exception as e:
        logger.error(f"Error retrieving {doc_id}: {e}", extra={"icon": "❌"})
        return pd.DataFrame()


def check_for_document_version_update(
    new_doc_data: List[Dict], db_connection=None
) -> Tuple[bool, float, Optional[str]]:
    """
    Check if a document is a new version of an existing document via ANN.
    """
    if not new_doc_data:
        return False, 0.0, None
    doc_id = new_doc_data[0].get('pdf_identifier', 'unknown')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    lancedb_path = os.path.join(project_root, OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME)
    db = db_connection or connect_to_lancedb(lancedb_path)
    if not db or LANCEDB_TABLE_NAME not in db.table_names():
        return False, 0.0, None
    ensure_index(db)
    table = db.open_table(LANCEDB_TABLE_NAME)

    sim_map: Dict[str, List[float]] = {}
    samples = new_doc_data[:min(MAX_PAGES_TO_SAMPLE, len(new_doc_data))]
    embeddings = [np.array(p['embedding']) for p in samples if p.get('embedding')]
    for emb in embeddings:
        try:
            df = (
                table.search(emb)
                     .metric("cosine")
                     .limit(MAX_PAGES_TO_SAMPLE)
                     .nprobes(32)
                     .refine_factor(5)
                     .to_df()
            )
            for _, row in df.iterrows():
                pid = row['pdf_identifier']
                if pid == doc_id:
                    continue
                sim = 1.0 - row['_distance']
                sim_map.setdefault(pid, []).append(sim)
        except Exception as e:
            logger.error(f"Error during version search: {e}", extra={"icon": "❌"})

    if not sim_map:
        return False, 0.0, None
    avg_sims = {pid: sum(v)/len(v) for pid, v in sim_map.items()}
    best_id, best_sim = max(avg_sims.items(), key=lambda x: x[1])
    is_new = VERSION_SIMILARITY_THRESHOLD <= best_sim < DUPLICATE_THRESHOLD
    if is_new:
        logger.info(
            f"Document {doc_id} appears to be new version of {best_id} (sim={best_sim:.4f})",
            extra={"icon": "🔄"}
        )
    return is_new, best_sim, best_id

# -------------------------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------------------------
def check_new_document(doc_data: List[Dict]) -> Dict[str, object]:
    """
    Main entry: deduplicate or detect version for a new document.
    """
    if not doc_data:
        logger.warning("Empty document data provided", extra={"icon": "⚠️"})
        return {
            'duplicate_pages': {}, 'new_pages': [], 'update_pages': {},
            'is_new_version': False, 'old_version_id': None, 'version_similarity': 0.0
        }
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    lancedb_path = os.path.join(project_root, OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME)
    db = connect_to_lancedb(lancedb_path)

    is_new, ver_sim, old_id = check_for_document_version_update(doc_data, db)
    if is_new and old_id:
        logger.info(
            f"Document is new version of {old_id}, adjusting page compare threshold.",
            extra={"icon": "🔄"}
        )
        original = get_document_pages_by_id(old_id, db)
        dup, new_pg, upd = {}, [], {}
        for idx, p in enumerate(doc_data):
            emb = p.get('embedding')
            if emb is None:
                new_pg.append(idx)
                continue
            emb_arr = np.array(emb) if isinstance(emb, list) else emb
            best_sim, best_row = 0.0, None
            for _, r in original.iterrows():
                other = np.array(r['embedding'])
                sim = calculate_cosine_similarity(emb_arr, other)
                if sim > best_sim:
                    best_sim, best_row = sim, r
            if best_sim >= SIMILAR_THRESHOLD:
                dup[idx] = {
                    'similar_id': f"{old_id}_{best_row.get('page_number', 'unknown')}",
                    'similarity': best_sim
                }
            else:
                new_pg.append(idx)
        return {
            'duplicate_pages': dup,
            'new_pages': new_pg,
            'update_pages': upd,
            'is_new_version': True,
            'old_version_id': old_id,
            'version_similarity': ver_sim
        }

    dup, new_pg, upd = check_document_duplicates(doc_data, db)
    return {
        'duplicate_pages': dup,
        'new_pages': new_pg,
        'update_pages': upd,
        'is_new_version': is_new,
        'old_version_id': old_id,
        'version_similarity': ver_sim
    }

if __name__ == "__main__":
    logger.info("This script is designed to be imported and used by the document processing pipeline.", extra={"icon": "ℹ️"})
    logger.info("It checks for duplicates before adding new documents to the database.", extra={"icon": "ℹ️"})
