"""
inspect_lancedb_chunks.py

This script connects to the LanceDB database, retrieves chunks for two specified
documents, and calculates the similarity between chunks of the second document
and chunks of the first document. It helps in verifying how well context-aware
chunking has aligned chunk boundaries between document versions.
"""
import os
import sys
import lancedb
import pandas as pd
import numpy as np
import difflib
import re
from dotenv import load_dotenv

# Add the project root to sys.path to allow importing project modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import get_logger, setup_project_directory
from src._03_docs_deduplication.pre_save_deduplication import connect_to_lancedb # For DB connection
# Import the actual chunk similarity function from the agentic_chunking module
from src._02_parsing.agentic_chunking import compute_chunk_similarity

# Setup project directory and logging
setup_project_directory()
logger = get_logger("Inspect_LanceDB_Chunks")
load_dotenv()

# Constants
OUTPUT_DIR_BASE = "output"
LANCEDB_SUBDIR_NAME = "lancedb"
CHUNKS_TABLE_NAME = "document_chunks"

DOC_ID_1 = "fighter_jet_rocket_launcher_spec_2.pdf"
DOC_ID_2 = "fighter_jet_rocket_launcher_spec_2_changed_values.pdf"

SIMILARITY_THRESHOLD_FOR_MATCH = 0.75 # From agentic_chunking.py

def inspect_chunk_alignment():
    logger.info(f"🔬 Starting LanceDB chunk inspection for documents: '{DOC_ID_1}' and '{DOC_ID_2}'", extra={"icon": "🔬"})

    db_path = os.path.join(PROJECT_ROOT, OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME)
    db = connect_to_lancedb(db_path)

    if not db:
        logger.error(f"❌ Could not connect to LanceDB at {db_path}. Aborting.", extra={"icon": "❌"})
        return

    if CHUNKS_TABLE_NAME not in db.table_names():
        logger.error(f"❌ Chunks table '{CHUNKS_TABLE_NAME}' not found in the database. Aborting.", extra={"icon": "❌"})
        return

    chunks_table = db.open_table(CHUNKS_TABLE_NAME)
    all_chunks_df = chunks_table.to_pandas()

    if all_chunks_df.empty:
        logger.warning("⚠️ Chunks table is empty. Nothing to inspect.", extra={"icon": "⚠️"})
        return

    doc1_chunks_df = all_chunks_df[all_chunks_df['document_id'] == DOC_ID_1].sort_values(by='chunk_index').reset_index(drop=True)
    doc2_chunks_df = all_chunks_df[all_chunks_df['document_id'] == DOC_ID_2].sort_values(by='chunk_index').reset_index(drop=True)

    logger.info(f"Found {len(doc1_chunks_df)} chunks for DOC1 ('{DOC_ID_1}')", extra={"icon": "📄"})
    logger.info(f"Found {len(doc2_chunks_df)} chunks for DOC2 ('{DOC_ID_2}')", extra={"icon": "📄"})

    if doc1_chunks_df.empty or doc2_chunks_df.empty:
        logger.warning("⚠️ One or both documents have no chunks in the table. Cannot compare.", extra={"icon": "⚠️"})
        return

    overall_summary = []

    for idx2, chunk2_row in doc2_chunks_df.iterrows():
        chunk2_id = chunk2_row['chunk_id']
        chunk2_text = chunk2_row['chunk_text']
        logger.info(f"\nComparing DOC2 Chunk: {chunk2_id} (Index {chunk2_row['chunk_index']})", extra={"icon": "🔍"})
        
        best_match_in_doc1 = {"chunk_id": None, "similarity": 0.0, "chunk_index": -1}
        
        for idx1, chunk1_row in doc1_chunks_df.iterrows():
            chunk1_id = chunk1_row['chunk_id']
            chunk1_text = chunk1_row['chunk_text']
            
            similarity = compute_chunk_similarity(chunk2_text, chunk1_text) # Use imported function
            logger.info(f"  vs DOC1 Chunk: {chunk1_id} (Index {chunk1_row['chunk_index']}) -> Similarity: {similarity:.4f}", extra={"icon": "📏"})
            
            if similarity > best_match_in_doc1["similarity"]:
                best_match_in_doc1["chunk_id"] = chunk1_id
                best_match_in_doc1["similarity"] = similarity
                best_match_in_doc1["chunk_index"] = chunk1_row['chunk_index']

        logger.info(f"  🏆 Best match for DOC2 chunk '{chunk2_id}' is DOC1 chunk '{best_match_in_doc1['chunk_id']}' with similarity {best_match_in_doc1['similarity']:.4f}", extra={"icon": "🏆"})
        overall_summary.append({
            "doc2_chunk_id": chunk2_id,
            "doc2_chunk_index": chunk2_row['chunk_index'],
            "best_match_doc1_chunk_id": best_match_in_doc1['chunk_id'],
            "best_match_doc1_chunk_index": best_match_in_doc1['chunk_index'],
            "similarity_to_best_match": best_match_in_doc1['similarity']
        })

    logger.info("\n\n📊 Overall Chunk Alignment Summary:", extra={"icon": "📊"})
    num_doc2_chunks = len(doc2_chunks_df)
    well_aligned_chunks = 0
    potentially_misaligned_chunks = 0

    for summary_item in overall_summary:
        sim = summary_item['similarity_to_best_match']
        doc2_chunk_id = summary_item['doc2_chunk_id']
        doc1_chunk_id_match = summary_item['best_match_doc1_chunk_id']
        doc2_chunk_idx = summary_item['doc2_chunk_index']
        doc1_chunk_idx_match = summary_item['best_match_doc1_chunk_index']

        log_symbol = "✅" if sim >= SIMILARITY_THRESHOLD_FOR_MATCH else ("⚠️" if sim > 0 else "❌")
        logger.info(f"  {log_symbol} DOC2 Chunk '{doc2_chunk_id}' (Idx {doc2_chunk_idx}) "
                    f"best matches DOC1 Chunk '{doc1_chunk_id_match}' (Idx {doc1_chunk_idx_match}) "
                    f"with similarity: {sim:.4f}", extra={"icon": log_symbol})
        if sim >= SIMILARITY_THRESHOLD_FOR_MATCH :
            well_aligned_chunks +=1
        else:
            if sim > 0: # Only count as misaligned if there was *some* text to compare
                 potentially_misaligned_chunks +=1
            # Log the text of misaligned chunks for diagnosis
            if doc1_chunk_id_match: # Ensure there was a best match found
                chunk2_text_to_log = doc2_chunks_df[doc2_chunks_df['chunk_id'] == doc2_chunk_id]['chunk_text'].iloc[0]
                chunk1_text_to_log = doc1_chunks_df[doc1_chunks_df['chunk_id'] == doc1_chunk_id_match]['chunk_text'].iloc[0]
                logger.info(f'    MISALIGNED/NEW - DOC2 Chunk \'{doc2_chunk_id}\' Text (len {len(chunk2_text_to_log)}):\n      \'\'\'{chunk2_text_to_log[:300]}...\'\'\'', extra={"icon": "📄"})
                logger.info(f'    MISALIGNED/NEW - Matched DOC1 Chunk \'{doc1_chunk_id_match}\' Text (len {len(chunk1_text_to_log)}):\n      \'\'\'{chunk1_text_to_log[:300]}...\'\'\'', extra={"icon": "📜"})
            else: # Case where a doc2 chunk had no match at all in doc1 (similarity was 0 for all)
                chunk2_text_to_log = doc2_chunks_df[doc2_chunks_df['chunk_id'] == doc2_chunk_id]['chunk_text'].iloc[0]
                logger.info(f'    NO MATCH - DOC2 Chunk \'{doc2_chunk_id}\' Text (len {len(chunk2_text_to_log)}):\n      \'\'\'{chunk2_text_to_log[:300]}...\'\'\'', extra={"icon": "📄"})


    logger.info(f"\nTotal DOC2 Chunks: {num_doc2_chunks}", extra={"icon": "🔢"})
    logger.info(f"Chunks with best match similarity >= {SIMILARITY_THRESHOLD_FOR_MATCH}: {well_aligned_chunks} (Well Aligned / User Decision Triggered)", extra={"icon": "✅"})
    logger.info(f"Chunks with best match similarity < {SIMILARITY_THRESHOLD_FOR_MATCH} (and >0): {potentially_misaligned_chunks} (Potentially Misaligned or Truly New)", extra={"icon": "⚠️"})
    
    if potentially_misaligned_chunks > 0:
        logger.warning(f"Found {potentially_misaligned_chunks} chunk(s) in DOC2 that did not have a counterpart in DOC1 with similarity >= {SIMILARITY_THRESHOLD_FOR_MATCH}.", extra={"icon": "🚨"})
        logger.warning("This might indicate that context-aware chunking did not perfectly align all content, or these are genuinely new/significantly altered sections.", extra={"icon": "🤔"})

if __name__ == "__main__":
    inspect_chunk_alignment() 