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
import logging
import time
import hashlib
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import lancedb
from dotenv import load_dotenv
from datetime import timedelta
from src import config
from src.utils import setup_logging, get_logger, update_token_counters, get_token_usage, print_token_usage, reset_token_counters, setup_project_directory, generate_timestamp
setup_project_directory()

# Setup logging with script prefix
logger = get_logger("Pre_Save_Deduplication")

# Load environment variables
load_dotenv()

# -------------------------------------------------------------------------------------
# Deduplication constants are now in config.py. Only runtime state variables remain here.
# -------------------------------------------------------------------------------------

# Runtime state variables (do not move to config)
INDEX_INITIALIZED_DOCS = False
INDEX_INITIALIZED_CHUNKS = False

# Export constants used by other modules for backwards compatibility
SIMILAR_THRESHOLD = config.DEDUPLICATION_SIMILAR_THRESHOLD
DUPLICATE_THRESHOLD = config.DEDUPLICATION_DUPLICATE_THRESHOLD

# -------------------------------------------------------------------------------------
# Deduplication Logging Functions
# -------------------------------------------------------------------------------------
def log_document_comparison(doc_id: str, comparison_doc_id: str, similarity: float):
    """Log information about document comparison."""
    if similarity >= config.DEDUPLICATION_DUPLICATE_THRESHOLD:
        logger.info(
            f"📄 Document comparison: {doc_id} is a duplicate of {comparison_doc_id} "
            f"(similarity: {similarity:.4f})",
            extra={"icon": "♻️"}
        )
    elif similarity >= config.DEDUPLICATION_SIMILAR_THRESHOLD:
        logger.info(
            f"📄 Document comparison: {doc_id} is similar to {comparison_doc_id} "
            f"(similarity: {similarity:.4f})",
            extra={"icon": "🔄"}
        )
    else:
        logger.info(
            f"📄 Document comparison: {doc_id} is different from {comparison_doc_id} "
            f"(similarity: {similarity:.4f})",
            extra={"icon": "🆕"}
        )

def log_page_comparison(
    doc_id: str, 
    page_num: int, 
    comparison_doc_id: str, 
    comparison_page_num: int, 
    similarity: float
):
    """Log information about page comparison."""
    page_id = f"{doc_id} p{page_num}"
    comparison_page_id = f"{comparison_doc_id} p{comparison_page_num}"
    
    if similarity >= config.DEDUPLICATION_DUPLICATE_THRESHOLD:
        logger.info(
            f"📃 Page comparison: {page_id} is a duplicate of {comparison_page_id} "
            f"(similarity: {similarity:.4f})",
            extra={"icon": "♻️"}
        )
    elif similarity >= config.DEDUPLICATION_SIMILAR_THRESHOLD:
        logger.info(
            f"📃 Page comparison: {page_id} is similar to {comparison_page_id} "
            f"(similarity: {similarity:.4f})",
            extra={"icon": "🔄"}
        )
    else:
        logger.info(
            f"📃 Page comparison: {page_id} appears to be new "
            f"(similarity: {similarity:.4f} with {comparison_page_id})",
            extra={"icon": "🆕"}
        )

def log_chunk_comparison(
    chunk_id: str, 
    match_id: str = "", 
    similarity: float = 0.0, 
    is_duplicate: bool = False,
    match_type: str = "embedding",
    doc_id: str = "",
    comparison_doc_id: str = ""
):
    """Log information about chunk comparison."""
    doc_context = f"({doc_id})" if doc_id else ""
    comparison_doc_context = f"({comparison_doc_id})" if comparison_doc_id else ""
    
    if match_type == "hash":
        logger.info(
            f"🧩 Chunk comparison: {chunk_id} {doc_context} is an exact hash match with "
            f"{match_id} {comparison_doc_context}",
            extra={"icon": "♻️"}
        )
    elif is_duplicate or similarity >= config.DEDUPLICATION_DUPLICATE_THRESHOLD:
        logger.info(
            f"🧩 Chunk comparison: {chunk_id} {doc_context} is a duplicate of "
            f"{match_id} {comparison_doc_context} "
            f"(similarity: {similarity:.4f})",
            extra={"icon": "♻️"}
        )
    elif similarity >= config.DEDUPLICATION_CHUNK_SIMILARITY_THRESHOLD:
        logger.info(
            f"🧩 Chunk comparison: {chunk_id} {doc_context} is similar to "
            f"{match_id} {comparison_doc_context} "
            f"(similarity: {similarity:.4f})",
            extra={"icon": "🔄"}
        )
    else:
        logger.info(
            f"🧩 Chunk comparison: {chunk_id} {doc_context} appears to be new "
            f"(top similarity: {similarity:.4f} with {match_id})",
            extra={"icon": "🆕"}
        )

def log_document_deduplication_summary(
    doc_id: str,
    total_pages: int,
    duplicate_pages: int,
    similar_pages: int,
    new_pages: int,
    is_new_version: bool,
    old_version_id: Optional[str] = None,
    version_similarity: float = 0.0
):
    """Log a summary of document deduplication results."""
    if is_new_version and old_version_id:
        logger.info(
            f"📊 Document deduplication summary: {doc_id} is a new version of {old_version_id} "
            f"(similarity: {version_similarity:.4f})",
            extra={"icon": "🔄"}
        )
    elif duplicate_pages == total_pages:
        logger.info(
            f"📊 Document deduplication summary: {doc_id} is a complete duplicate "
            f"(all {total_pages} pages are duplicates)",
            extra={"icon": "♻️"}
        )
    elif duplicate_pages > 0 or similar_pages > 0:
        logger.info(
            f"📊 Document deduplication summary: {doc_id} has {new_pages} new pages, "
            f"{duplicate_pages} duplicate pages, and {similar_pages} similar pages "
            f"(total: {total_pages} pages)",
            extra={"icon": "🔄"}
        )
    else:
        logger.info(
            f"📊 Document deduplication summary: {doc_id} is completely new "
            f"(all {total_pages} pages are new)",
            extra={"icon": "🆕"}
        )

def log_chunk_deduplication_summary(
    total_chunks: int,
    duplicate_chunks: int,
    similar_chunks: int,
    new_chunks: int,
    processing_time: float = 0.0,
    doc_id: str = "",
    is_document_update: bool = False,
    updated_doc_id: Optional[str] = None
):
    """Log a summary of chunk deduplication results."""
    doc_prefix = f"{doc_id}: " if doc_id else ""
    
    if is_document_update and updated_doc_id:
        logger.info(
            f"📊 Chunk deduplication summary: {doc_prefix}updating chunks from {updated_doc_id} "
            f"({duplicate_chunks} duplicates, {similar_chunks} updates, {new_chunks} new chunks) "
            f"in {processing_time:.2f}s",
            extra={"icon": "🔄"}
        )
    elif duplicate_chunks == total_chunks:
        logger.info(
            f"📊 Chunk deduplication summary: {doc_prefix}only duplicate chunks "
            f"(all {total_chunks} chunks are duplicates) "
            f"in {processing_time:.2f}s",
            extra={"icon": "♻️"}
        )
    elif duplicate_chunks > 0 or similar_chunks > 0:
        logger.info(
            f"📊 Chunk deduplication summary: {doc_prefix}{new_chunks} new chunks, "
            f"{duplicate_chunks} duplicate chunks, and {similar_chunks} similar chunks "
            f"(total: {total_chunks} chunks) "
            f"in {processing_time:.2f}s",
            extra={"icon": "🔄"}
        )
    else:
        logger.info(
            f"📊 Chunk deduplication summary: {doc_prefix}all new chunks "
            f"({new_chunks} chunks) "
            f"in {processing_time:.2f}s",
            extra={"icon": "🆕"}
        )

def log_embedding_similarity(similarity: float, description: str = "Embedding comparison"):
    """Log information about embedding similarity."""
    if similarity >= config.DEDUPLICATION_DUPLICATE_THRESHOLD:
        logger.info(
            f"📏 {description} - Embedding similarity: {similarity:.4f} (DUPLICATE)",
            extra={"icon": "♻️"}
        )
    elif similarity >= config.DEDUPLICATION_SIMILAR_THRESHOLD:
        logger.info(
            f"📏 {description} - Embedding similarity: {similarity:.4f} (SIMILAR)",
            extra={"icon": "🔄"}
        )
    else:
        logger.info(
            f"📏 {description} - Embedding similarity: {similarity:.4f} (DIFFERENT)",
            extra={"icon": "🆕"}
        )

# -------------------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------------------
def connect_to_lancedb(lancedb_path: str, log_connection: bool = True):
    """
    Connect to LanceDB and return the connection.
    
    Args:
        lancedb_path: Path to the LanceDB directory
        log_connection: Whether to log connection messages (set to False for reused connections)
    """
    if log_connection:
        logger.info(f"Connecting to LanceDB at: {lancedb_path}", extra={"icon": "🔄"})
        
    if not os.path.exists(lancedb_path):
        logger.warning(
            f"LanceDB directory does not exist at {lancedb_path}. It will be created when saving.",
            extra={"icon": "⚠️"}
        )
        return None

    try:
        db = lancedb.connect(lancedb_path)
        if log_connection:
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
    similarity = float(np.dot(norm_embed1, norm_embed2))
    
    # Log the embedding similarity if verbose mode is enabled
    if config.DEDUPLICATION_VERBOSE_OUTPUT:
        log_embedding_similarity(similarity)
        
    return similarity


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
        table_name = config.DOCUMENTS_TABLE
        
    if table_name not in db.table_names():
        return
        
    # Check if this table's index is already initialized
    if (table_name == config.DOCUMENTS_TABLE and INDEX_INITIALIZED_DOCS) or \
       (table_name == config.DOCUMENT_CHUNKS_TABLE and INDEX_INITIALIZED_CHUNKS):
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
    
    if table_name == config.DOCUMENTS_TABLE:
        INDEX_INITIALIZED_DOCS = True
    elif table_name == config.DOCUMENT_CHUNKS_TABLE:
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
    
    if config.DEDUPLICATION_VERBOSE_OUTPUT:
        logger.info("\n" + "=" * 80, extra={"icon": "🧩"})
        logger.info(f"CHUNK DEDUPLICATION PROCESS FOR DOCUMENT: {doc_id}", extra={"icon": "🧩"})
        logger.info("=" * 80, extra={"icon": "🧩"})
    
    logger.info(f"Checking for duplicate chunks in document: {doc_id}", extra={"icon": "🔄"})
    
    # Connect to LanceDB or reuse existing connection
    if db_connection:
        db = db_connection
    else:
        # Connect to LanceDB
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
        lancedb_path = os.path.join(project_root, config.OUTPUT_DIR_BASE, config.LANCEDB_SUBDIR_NAME)
        db = connect_to_lancedb(lancedb_path, log_connection=True)
    
    if not db or config.DOCUMENT_CHUNKS_TABLE not in db.table_names():
        logger.info(
            f"No existing chunks table found. All {len(chunks_data)} chunks are new.",
            extra={"icon": "✅"}
        )
        if config.DEDUPLICATION_VERBOSE_OUTPUT:
            logger.info(f"✅ No existing chunks table found. All {len(chunks_data)} chunks are new.", extra={"icon": "✅"})
        return {}, list(range(len(chunks_data))), {}
    
    # Ensure index exists if we have enough chunks
    ensure_index(db, config.DOCUMENT_CHUNKS_TABLE)
    chunks_table = db.open_table(config.DOCUMENT_CHUNKS_TABLE)
    
    # Get a dataframe of all existing chunks for hash comparison
    existing_chunks_df = chunks_table.to_pandas()
    
    if config.DEDUPLICATION_VERBOSE_OUTPUT and not existing_chunks_df.empty:
        logger.info(f"📊 Found {len(existing_chunks_df)} existing chunks to compare against", extra={"icon": "🔍"})
    
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
                
                # Log the duplicate with enhanced visibility
                log_chunk_comparison(
                    chunk_id=chunk_id,
                    match_id=match_id,
                    similarity=1.0,
                    is_duplicate=True,
                    match_type="hash"
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
            
        # Attempt vector similarity search with robust error handling
        try:
            # First check if the schema has embedding column
            has_embedding_column = False
            for field in chunks_table.schema:
                if field.name == 'embedding':
                    has_embedding_column = True
                    break
                    
            if not has_embedding_column:
                logger.warning(f"No embedding column found in chunks table. Using direct comparison.", extra={"icon": "⚠️"})
                
                # Fall back to direct comparison without vector search
                best_match = None
                best_similarity = 0.0
                
                for _, row in existing_chunks_df.iterrows():
                    existing_emb = row.get('embedding')
                    if existing_emb is not None:
                        if isinstance(existing_emb, list):
                            existing_emb = np.array(existing_emb)
                            
                        try:
                            similarity = calculate_cosine_similarity(embedding, existing_emb)
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_match = row
                        except Exception:
                            continue
                            
                if best_match is not None and best_similarity >= config.DEDUPLICATION_CHUNK_SIMILARITY_THRESHOLD:
                    match_id = best_match.get('chunk_id', '')
                    
                    log_chunk_comparison(
                        chunk_id=chunk_id,
                        match_id=match_id,
                        similarity=best_similarity,
                        is_duplicate=best_similarity >= config.DEDUPLICATION_CHUNK_DUPLICATION_THRESHOLD
                    )
                    
                    if best_similarity >= config.DEDUPLICATION_CHUNK_DUPLICATION_THRESHOLD:
                        duplicate_chunks[idx] = {
                            'similar_id': match_id,
                            'similarity': best_similarity,
                            'hash_match': False
                        }
                    elif best_similarity >= config.DEDUPLICATION_CHUNK_SIMILARITY_THRESHOLD:
                        update_chunks[idx] = {
                            'similar_id': match_id,
                            'similarity': best_similarity
                        }
                    else:
                        new_chunks.append(idx)
                else:
                    new_chunks.append(idx)
                    if config.DEDUPLICATION_VERBOSE_OUTPUT:
                        logger.info(f"🆕 Chunk {chunk_id} has no similar chunks. Marking as new.", extra={"icon": "��"})
                continue
            
            # Use vector search with explicit column specification
            query_result = chunks_table.search(
                embedding, 
                vector_column_name="embedding"
            ).limit(5).to_pandas()
            
            if query_result.empty:
                if config.DEDUPLICATION_VERBOSE_OUTPUT:
                    logger.info(f"🆕 Chunk {chunk_id} has no similar chunks. Marking as new.", extra={"icon": "🆕"})
                new_chunks.append(idx)
                continue
                
            # Calculate similarity (1.0 - distance for cosine similarity)
            best_match = query_result.iloc[0]
            similarity = 1.0 - best_match['_distance']
            match_id = best_match.get('chunk_id', '')
            
            # Log detailed chunk comparison
            log_chunk_comparison(
                chunk_id=chunk_id,
                match_id=match_id,
                similarity=similarity,
                is_duplicate=similarity >= config.DEDUPLICATION_CHUNK_DUPLICATION_THRESHOLD
            )
            
            if similarity >= config.DEDUPLICATION_CHUNK_DUPLICATION_THRESHOLD:
                # This is a duplicate chunk
                duplicate_chunks[idx] = {
                    'similar_id': match_id,
                    'similarity': similarity,
                    'hash_match': False
                }
            elif similarity >= config.DEDUPLICATION_CHUNK_SIMILARITY_THRESHOLD:
                # This might be an updated version of an existing chunk
                update_chunks[idx] = {
                    'similar_id': match_id,
                    'similarity': similarity
                }
            else:
                # This is a new chunk
                new_chunks.append(idx)
                
        except Exception as e:
            logger.error(f"Error during vector search for chunk {chunk_id}: {str(e)}", extra={"icon": "❌"})
            if config.DEDUPLICATION_VERBOSE_OUTPUT:
                logger.error(f"Error during vector search: {str(e)}", extra={"icon": "❌"})
                logger.info(f"Marking chunk {chunk_id} as new due to search error.", extra={"icon": "🆕"})
            new_chunks.append(idx)
    
    # Log summary of chunk deduplication
    processing_time = time.time() - start_time
    log_chunk_deduplication_summary(
        total_chunks=len(chunks_data),
        duplicate_chunks=len(duplicate_chunks),
        similar_chunks=len(update_chunks),
        new_chunks=len(new_chunks),
        processing_time=processing_time
    )
    
    # Print more detailed summary in verbose mode
    if config.DEDUPLICATION_VERBOSE_OUTPUT:
        logger.info("\n" + "-" * 50, extra={"icon": "📊"})
        logger.info(f"CHUNK DEDUPLICATION SUMMARY:", extra={"icon": "📊"})
        logger.info(f"  • Total chunks processed: {len(chunks_data)}", extra={"icon": "📊"})
        logger.info(f"  • Exact duplicates found: {len(duplicate_chunks)}", extra={"icon": "📊"})
        logger.info(f"  • Similar chunks found: {len(update_chunks)}", extra={"icon": "📊"})
        logger.info(f"  • New unique chunks: {len(new_chunks)}", extra={"icon": "📊"})
        logger.info(f"  • Processing time: {processing_time:.2f} seconds", extra={"icon": "📊"})
        logger.info("-" * 50 + "\n", extra={"icon": "📊"})
    
    return duplicate_chunks, new_chunks, update_chunks

# -------------------------------------------------------------------------------------
# Main Deduplication Logic
# -------------------------------------------------------------------------------------
def check_document_duplicates(
    new_doc_data: List[Dict], db_connection=None
) -> Tuple[Dict[int, Dict[str, object]], List[int], Dict[int, Dict[str, object]]]:
    """
    Check if a new document has duplicates in the existing database.
    
    Returns:
        duplicate_pages: dict mapping page idx -> {'similar_id', 'similarity', 'hash_match'}
        new_pages: list of new page indices
        update_pages: dict mapping page idx -> {'record_id', 'is_newer'}
    """
    start_time = time.time()
    
    if not new_doc_data:
        logger.warning("Empty document data provided", extra={"icon": "⚠️"})
        return {}, [], {}
        
    doc_id = new_doc_data[0].get('document_id', 'unknown')
    
    logger.info(f"Checking for duplicate document: {doc_id}", extra={"icon": "🔄"})
    
    # Connect to LanceDB or reuse existing connection
    if db_connection:
        db = db_connection
    else:
        # Connect to LanceDB
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
        lancedb_path = os.path.join(project_root, config.OUTPUT_DIR_BASE, config.LANCEDB_SUBDIR_NAME)
        db = connect_to_lancedb(lancedb_path, log_connection=True)
    
    if not db or config.DOCUMENTS_TABLE not in db.table_names():
        logger.info(f"No existing documents table found. Document {doc_id} is new.", extra={"icon": "✅"})
        return {}, list(range(len(new_doc_data))), {}

    # Check if table has content
    table = db.open_table(config.DOCUMENTS_TABLE)
    existing_docs = table.to_pandas()
    if existing_docs.empty:
        logger.info(
            f"Empty documents table; all {len(new_doc_data)} pages are new.",
            extra={"icon": "✅"}
        )
        return {}, list(range(len(new_doc_data))), {}

    # Check if the new documents have valid embeddings
    has_valid_embeddings = True
    for idx, page_data in enumerate(new_doc_data):
        embedding = page_data.get("embedding")
        if embedding is None or (isinstance(embedding, list) and (not embedding or all(v == 0 for v in embedding))):
            has_valid_embeddings = False
            break
        if isinstance(embedding, np.ndarray) and (np.all(embedding == 0) or np.isnan(embedding).any()):
            has_valid_embeddings = False
            break
    
    if not has_valid_embeddings:
        logger.warning(
            f"Documents don't have valid embeddings yet. Deduplication is limited to metadata matching.",
            extra={"icon": "⚠️"}
        )
        # Return all pages as new since we can't do embedding-based deduplication yet
        return {}, list(range(len(new_doc_data))), {}

    ensure_index(db)

    # ── result containers ───────────────────────────────────────────────────────
    duplicate_pages: Dict[int, Dict[str, object]] = {}
    new_pages:      List[int]                     = []
    update_pages:   Dict[int, Dict[str, object]] = {}

    # ── main loop over pages ────────────────────────────────────────────────────
    for idx, page_data in enumerate(new_doc_data):
        page_num = page_data.get("page_number", idx + 1)
        embedding = page_data.get("embedding")

        # Enhanced embedding validation
        if embedding is None or not embedding:
            logger.warning(f"Page {page_num} has no embedding or empty embedding → new", extra={"icon": "⚠️"})
            new_pages.append(idx)
            continue
            
        if isinstance(embedding, list):
            if not embedding:  # Check again if it's an empty list
                logger.warning(f"Page {page_num} has empty list embedding → new", extra={"icon": "⚠️"})
                new_pages.append(idx)
                continue
            embedding = np.array(embedding)
            
        # Check for all zeros or NaNs in the embedding
        if np.all(embedding == 0) or np.isnan(embedding).any():
            logger.warning(f"Page {page_num} has invalid embedding (all zeros or NaNs) → new", extra={"icon": "⚠️"})
            new_pages.append(idx)
            continue

        # Log the query embedding for debugging
        if config.DEDUPLICATION_VERBOSE_OUTPUT:
            # Check if embedding is all zeros or contains NaNs
            is_all_zeros = np.all(embedding == 0)
            has_nans = np.isnan(embedding).any()
            # Remove first 5 and last 5 values preview, only log status
            logger.info(f"Querying with embedding for page {page_num}. " 
                       f"All zeros: {is_all_zeros}, Has NaNs: {has_nans}", 
                       extra={"icon": "🧬"})

        try:
            df = (
                table.search(embedding)           # brute-force scan if no ANN index
                     .metric("cosine")
                     .limit(config.DEDUPLICATION_MAX_PAGES_TO_SAMPLE)
                     .to_df()
            )
            if df.empty:
                logger.info(f"No matches for page {page_num} → new", extra={"icon": "✅"})
                new_pages.append(idx)
                # ── closest-match for empty result ──────────────────────────────
                logger.info(
                    f"🔍 No similar content found for page {page_num} in the database",
                    extra={"icon": "🔍"}
                )
                continue

            # Compute cosine similarity column once
            df["_sim"] = 1.0 - df["_distance"]

            # Handle cases where all similarities might be NaN.
            # This could happen if all _distance values from the search result are NaN.
            if df["_sim"].isna().all():
                logger.warning(
                    f"Page {page_num}: All similarity scores are NaN. No valid match found. Treating as new.",
                    extra={"icon": "⚠️"}
                )
                # Optionally log the problematic df for debugging if DEDUPLICATION_VERBOSE_OUTPUT is True
                if config.DEDUPLICATION_VERBOSE_OUTPUT:
                    try:
                        # Attempt to log the DataFrame; use to_string() for better readability if it's large
                        logger.info(f"DataFrame for page {page_num} with all NaN similarities ({len(df)} rows):\n{df.to_string()}", extra={"icon": "🐛"})
                    except Exception as log_df_error:
                        logger.info(f"DataFrame for page {page_num} with all NaN similarities ({len(df)} rows). Error during df logging: {log_df_error}", extra={"icon": "🐛"})
                
                new_pages.append(idx)
                # ── closest-match for NaN result ──────────────────────────────
                logger.info(
                    f"🔍 No valid similarity scores for page {page_num}",
                    extra={"icon": "🔍"}
                )
                continue
            
            # At this point, df["_sim"] has at least one non-NaN value.
            # idxmax() will skip NaNs and return the index LABEL of the maximum value.
            best_idx_label = df["_sim"].idxmax()
            
            # Use .loc to access the row using the index label, which is safer.
            best_row = df.loc[best_idx_label]
            
            best_sim = float(best_row["_sim"])
            best_id = best_row["pdf_identifier"]
            best_page = best_row.get("page_number", "unknown")

            # ── classification ────────────────────────────────────────────────
            found = False
            if best_id == doc_id and best_page == page_num:
                # This page already exists in the *same* document
                exist_ts = pd.to_datetime(best_row.get("timestamp"))
                new_ts   = pd.to_datetime(page_data.get("timestamp"))
                if new_ts and exist_ts and new_ts > exist_ts:
                    update_pages[idx] = {"record_id": best_row.name, "is_newer": True}
                    logger.info(
                        f"Page {page_num} is a newer version of itself.",
                        extra={"icon": "🔄"}
                    )
                else:
                    duplicate_pages[idx] = {
                        "similar_id": f"{best_id}_{best_page}",
                        "similarity": best_sim
                    }
                    logger.info(
                        f"Page {page_num} already exists unchanged.",
                        extra={"icon": "♻️"}
                    )
                found = True

            elif best_sim >= config.DEDUPLICATION_DUPLICATE_THRESHOLD:
                # Perfect / near-perfect duplicate in another document
                duplicate_pages[idx] = {
                    "similar_id": f"{best_id}_{best_page}",
                    "similarity": best_sim
                }
                logger.info(
                    f"Page {page_num} duplicates {best_id} page {best_page}.",
                    extra={"icon": "♻️"}
                )
                found = True

            # else: similarity < DEDUPLICATION_DUPLICATE_THRESHOLD  → keep as new

            if not found:
                new_pages.append(idx)

            # ── always emit closest-match log ──────────────────────────────────
            logger.info(
                f"🔍 Closest match for page {page_num}: "
                f"{best_id} page {best_page} (cosine ≈ {best_sim:.4f})",
                extra={"icon": "🔍"}
            )

        except Exception as e:
            logger.error(f"Search error on page {page_num}: {e}", extra={"icon": "❌"})
            new_pages.append(idx)
            # ── closest-match for error case ──────────────────────────────
            logger.info(
                f"🔍 Error finding match for page {page_num}: {str(e)}",
                extra={"icon": "🔍"}
            )

    # ── summary ────────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    logger.info(
        f"Duplicate check done in {elapsed:.2f}s → "
        f"{len(new_pages)} new, {len(duplicate_pages)} dup, {len(update_pages)} updates",
        extra={"icon": "📊"}
    )
    return duplicate_pages, new_pages, update_pages


def get_document_pages_by_id(doc_id: str, db_connection=None) -> pd.DataFrame:
    """
    Get all pages for a specific document ID from the database.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    lancedb_path = os.path.join(project_root, config.OUTPUT_DIR_BASE, config.LANCEDB_SUBDIR_NAME)
    db = db_connection or connect_to_lancedb(lancedb_path)
    if not db or config.DOCUMENTS_TABLE not in db.table_names():
        logger.warning(
            f"No existing database or table found for document {doc_id}",
            extra={"icon": "⚠️"}
        )
        return pd.DataFrame()
    ensure_index(db)
    table = db.open_table(config.DOCUMENTS_TABLE)
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
    Check if a document appears to be a new version of an existing document.
    
    Returns:
        is_update: Whether the document is a new version
        similarity: Similarity score between new and old version
        old_version_id: ID of the old version if found
    """
    if not new_doc_data:
        logger.warning("Empty document data provided", extra={"icon": "⚠️"})
        return False, 0.0, None
        
    doc_id = new_doc_data[0].get('document_id', 'unknown')
    
    # Connect to LanceDB or reuse existing connection
    if db_connection:
        db = db_connection
    else:
        # Connect to LanceDB
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
        lancedb_path = os.path.join(project_root, config.OUTPUT_DIR_BASE, config.LANCEDB_SUBDIR_NAME)
        db = connect_to_lancedb(lancedb_path, log_connection=True)
    
    if not db or config.DOCUMENTS_TABLE not in db.table_names():
        logger.info(f"No existing documents table found. Document {doc_id} is new.", extra={"icon": "✅"})
        return False, 0.0, None

    # Check if the new documents have valid embeddings
    has_valid_embeddings = True
    for page_data in new_doc_data:
        embedding = page_data.get("embedding")
        if embedding is None or (isinstance(embedding, list) and (not embedding or all(v == 0 for v in embedding))):
            has_valid_embeddings = False
            break
        if isinstance(embedding, np.ndarray) and (np.all(embedding == 0) or np.isnan(embedding).any()):
            has_valid_embeddings = False
            break
    
    if not has_valid_embeddings:
        logger.error(
            f"Document {doc_id} doesn't have valid embeddings. Cannot perform document version detection.",
            extra={"icon": "❌"}
        )
        return False, 0.0, None

    ensure_index(db)

    # Load all existing documents from the table
    table = db.open_table(config.DOCUMENTS_TABLE)
    existing_docs = table.to_pandas()

    # Get list of unique document IDs (excluding current doc)
    try:
        all_doc_ids = existing_docs['pdf_identifier'].unique().tolist()
        if doc_id in all_doc_ids:
            all_doc_ids.remove(doc_id)
        
        if not all_doc_ids:
            logger.info(f"No other documents to compare with {doc_id}", extra={"icon": "ℹ️"})
            return False, 0.0, None
        
        logger.info(f"Comparing {doc_id} with {len(all_doc_ids)} existing documents", extra={"icon": "🔍"})
    except Exception as e:
        logger.error(f"Error retrieving document IDs: {e}", extra={"icon": "❌"})
        return False, 0.0, None

    sim_map: Dict[str, List[float]] = {}
    samples = new_doc_data[:min(config.DEDUPLICATION_MAX_PAGES_TO_SAMPLE, len(new_doc_data))]
    
    # More robust embedding extraction
    embeddings = []
    for p in samples:
        emb = p.get('embedding')
        if emb is not None:
            if isinstance(emb, list) and emb:  # Ensure it's not an empty list
                emb_array = np.array(emb)
                if not np.all(emb_array == 0) and not np.isnan(emb_array).any():
                    embeddings.append(emb_array)
            elif isinstance(emb, np.ndarray) and not np.all(emb == 0) and not np.isnan(emb).any():
                embeddings.append(emb)
    
    if not embeddings:
        logger.warning(f"Document {doc_id} has no valid embeddings for comparison", extra={"icon": "⚠️"})
        return False, 0.0, None
        
    # Log number of sample pages being used
    logger.info(f"Using {len(embeddings)} page embeddings from {doc_id} for document comparison", extra={"icon": "📊"})
    
    # Process each embedding
    for i, emb in enumerate(embeddings):
        try:
            # Verify the embedding is valid for search
            if np.all(emb == 0) or np.isnan(emb).any():
                logger.warning(f"Skipping invalid embedding for page {i+1}: all zeros or contains NaNs", extra={"icon": "⚠️"})
                continue
                
            df = (
                table.search(emb)
                     .metric("cosine")
                     .limit(config.DEDUPLICATION_MAX_PAGES_TO_SAMPLE)
                     .to_df()
            )
            
            # Skip if no results or all distances are NaN
            if df.empty or df['_distance'].isna().all():
                logger.info(f"No valid matches found for page {i+1} embedding", extra={"icon": "ℹ️"})
                continue
                
            # Calculate similarities (1.0 - distance)
            df['_sim'] = 1.0 - df['_distance']
            
            # Find closest matches for each document ID
            for pid in all_doc_ids:
                doc_matches = df[df['pdf_identifier'] == pid]
                if not doc_matches.empty:
                    best_sim = doc_matches['_sim'].max()
                    if not pd.isna(best_sim):  # Ensure it's not NaN
                        sim_map.setdefault(pid, []).append(best_sim)
                        
        except Exception as e:
            logger.error(f"Error during version search for page {i+1}: {e}", extra={"icon": "❌"})

    if not sim_map:
        logger.info(f"No similarity data found between {doc_id} and existing documents", extra={"icon": "ℹ️"})
        return False, 0.0, None
        
    # Calculate average similarity for each document
    avg_sims = {pid: sum(v)/len(v) for pid, v in sim_map.items() if v}
    
    # Log all document similarities for debugging
    if config.DEDUPLICATION_VERBOSE_OUTPUT:
        for pid, sim in sorted(avg_sims.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"Document similarity: {doc_id} ↔ {pid}: {sim:.4f}", extra={"icon": "📏"})
    
    if not avg_sims:
        logger.info(f"No valid average similarities found for {doc_id}", extra={"icon": "ℹ️"})
        return False, 0.0, None
        
    # Get best match
    best_id, best_sim = max(avg_sims.items(), key=lambda x: x[1])
    
    # Always log the closest document match
    if best_sim >= config.DEDUPLICATION_DUPLICATE_THRESHOLD:
        logger.info(
            f"📚 DOCUMENT MATCH: {doc_id} is a DUPLICATE of {best_id} (similarity: {best_sim:.4f})",
            extra={"icon": "♻️"}
        )
    elif best_sim >= config.DEDUPLICATION_VERSION_SIMILARITY_THRESHOLD:
        logger.info(
            f"📚 DOCUMENT MATCH: {doc_id} is a NEW VERSION of {best_id} (similarity: {best_sim:.4f})",
            extra={"icon": "🔄"}
        )
    else:
        logger.info(
            f"📚 DOCUMENT MATCH: {doc_id} is DIFFERENT from closest document {best_id} (similarity: {best_sim:.4f})",
            extra={"icon": "🆕"}
        )
    
    # Determine if it's a new version (above threshold but below duplicate)
    is_new = config.DEDUPLICATION_VERSION_SIMILARITY_THRESHOLD <= best_sim < config.DEDUPLICATION_DUPLICATE_THRESHOLD
    
    # Always return the best match, even if below threshold
    return is_new, best_sim, best_id

# -------------------------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------------------------
def check_new_document(doc_data: List[Dict], db_connection=None) -> Dict[str, object]:
    """
    Check if a document is new, a duplicate, or an update of an existing document.
    
    Returns:
        result: Dict with document status and processing information
    """
    if not doc_data:
        logger.warning("Empty document data provided", extra={"icon": "⚠️"})
        return {
            "status": "error",
            "message": "Empty document data provided"
        }
    
    # Establish connection to the database (or reuse provided connection)
    if db_connection:
        db = db_connection
    else:
        # Connect to LanceDB
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
        lancedb_path = os.path.join(project_root, config.OUTPUT_DIR_BASE, config.LANCEDB_SUBDIR_NAME)
        db = connect_to_lancedb(lancedb_path, log_connection=True)
    
    # Get document ID and log the check
    doc_id = doc_data[0].get("pdf_identifier", "unknown")
    logger.info(f"Checking for duplicate/similar documents: {doc_id}", extra={"icon": "🔍"})
    
    # Check duplicate pages first with existing document database
    duplicate_pages, new_pages, update_pages = check_document_duplicates(doc_data, db)
    
    # Calculate summary counts
    total_pages = len(doc_data)
    duplicate_count = len(duplicate_pages)
    update_count = len(update_pages)
    new_count = len(new_pages)
    
    # Check if this is a complete duplicate (all pages are duplicates)
    is_complete_duplicate = duplicate_count == total_pages
    
    # Check if this might be a new version of an existing document
    is_new_version, version_similarity, old_version_id = check_for_document_version_update(doc_data, db)
    
    # Log detailed summary using enhanced logging
    log_document_deduplication_summary(
        doc_id=doc_id,
        total_pages=total_pages,
        duplicate_pages=duplicate_count,
        similar_pages=update_count,
        new_pages=new_count,
        is_new_version=is_new_version,
        old_version_id=old_version_id,
        version_similarity=version_similarity
    )
    
    # --- PATCH: If this is a new version (update), do NOT mark as duplicate ---
    is_duplicate = is_complete_duplicate and not is_new_version
    if is_new_version:
        logger.info(f"Not marking as duplicate because this is a new version of {old_version_id}", extra={"icon": "🔄"})
    # --- END PATCH ---
    
    # Return comprehensive results
    return {
        "status": "success",
        "message": "Duplicate check completed successfully",
        "is_duplicate": is_duplicate,
        "duplicate_pages": duplicate_pages,
        "new_pages": new_pages,
        "update_pages": update_pages,
        "is_new_version": is_new_version,
        "old_version_id": old_version_id,
        "version_similarity": version_similarity
    }

if __name__ == "__main__":
    logger.info("This script is designed to be imported and used by the document processing pipeline.", extra={"icon": "ℹ️"})
    logger.info("It checks for duplicates before adding new documents to the database.", extra={"icon": "ℹ️"})
