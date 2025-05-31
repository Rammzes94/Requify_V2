"""
test_qwen_page_embeddings.py

This script compares page-level embeddings using both the default e5-large model 
and the Qwen2 model to demonstrate the differences in similarity scores.

It loads documents from LanceDB and compares their embeddings using both models
to show how the choice of embedding model affects similarity scores.
"""

import os
import sys
import time
import logging
import numpy as np
from dotenv import load_dotenv
import lancedb
from sentence_transformers import SentenceTransformer

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import setup_logging, get_logger, update_token_counters, get_token_usage, print_token_usage, reset_token_counters, setup_project_directory, generate_timestamp
from src import config
from src._03_docs_deduplication.pre_save_deduplication import calculate_cosine_similarity

setup_project_directory()

# Load environment variables
load_dotenv()

# Setup logging with script prefix
logger = get_logger("Test_Qwen_Page_Embeddings")

# Constants
OUTPUT_DIR_BASE = "output"
LANCEDB_SUBDIR_NAME = "lancedb"

# Original page-level embedding model
E5_MODEL_NAME = config.EMBEDDING_MODEL_NAME  # "intfloat/multilingual-e5-large-instruct"
E5_DIMENSION = config.EMBEDDING_DIMENSION    # 1024

# Document-level embedding model
QWEN_MODEL_NAME = config.DOC_EMBEDDING_MODEL_NAME  # "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
QWEN_DIMENSION = config.DOC_EMBEDDING_DIMENSION    # 1536

def connect_to_lancedb():
    """Connect to LanceDB and return the connection."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    lancedb_path = os.path.join(project_root, OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME)
    
    if not os.path.exists(lancedb_path):
        logger.error(f"LanceDB directory not found at {lancedb_path}", extra={"icon": "❌"})
        return None
        
    try:
        db = lancedb.connect(lancedb_path)
        logger.info(f"Connected to LanceDB at {lancedb_path}", extra={"icon": "✅"})
        return db
    except Exception as e:
        logger.error(f"Failed to connect to LanceDB: {e}", extra={"icon": "❌"})
        return None

def load_models():
    """Load both embedding models for comparison."""
    logger.info(f"Loading E5 model: {E5_MODEL_NAME}", extra={"icon": "🧠"})
    e5_model = SentenceTransformer(E5_MODEL_NAME)
    
    logger.info(f"Loading Qwen model: {QWEN_MODEL_NAME}", extra={"icon": "🧠"})
    qwen_model = SentenceTransformer(QWEN_MODEL_NAME, trust_remote_code=True)
    qwen_model.max_seq_length = config.DOC_EMBEDDING_MAX_SEQ_LENGTH
    
    return e5_model, qwen_model

def compare_document_embeddings(doc1_id, doc2_id):
    """Compare two documents using both embedding models."""
    db = connect_to_lancedb()
    if not db:
        return
        
    if "documents" not in db.table_names():
        logger.error("Documents table not found in database", extra={"icon": "❌"})
        return
        
    # Load the documents table
    doc_table = db.open_table("documents")
    docs_df = doc_table.to_pandas()
    
    # Filter the dataframe for the two document IDs
    doc1_rows = docs_df[docs_df["pdf_identifier"] == doc1_id]
    doc2_rows = docs_df[docs_df["pdf_identifier"] == doc2_id]
    
    if doc1_rows.empty:
        logger.error(f"Document {doc1_id} not found in database", extra={"icon": "❌"})
        return
        
    if doc2_rows.empty:
        logger.error(f"Document {doc2_id} not found in database", extra={"icon": "❌"})
        return
    
    # Load the models
    e5_model, qwen_model = load_models()
    
    # Get the content from both documents (assuming single page documents)
    doc1_content = doc1_rows.iloc[0].get("md_content", "")
    doc2_content = doc2_rows.iloc[0].get("md_content", "")
    
    if not doc1_content or not doc2_content:
        logger.error("Documents missing content", extra={"icon": "❌"})
        return
    
    # Compare using existing document embeddings (from database)
    logger.info("Comparing existing embeddings from database:", extra={"icon": "🔍"})
    
    # Get existing embeddings
    doc1_e5_embed = np.array(doc1_rows.iloc[0].get("embedding", []))
    doc2_e5_embed = np.array(doc2_rows.iloc[0].get("embedding", []))
    
    doc1_qwen_embed = np.array(doc1_rows.iloc[0].get("document_embedding", []))
    doc2_qwen_embed = np.array(doc2_rows.iloc[0].get("document_embedding", []))
    
    # Check if we have both types of embeddings
    if len(doc1_e5_embed) > 0 and len(doc2_e5_embed) > 0:
        e5_similarity = calculate_cosine_similarity(doc1_e5_embed, doc2_e5_embed)
        logger.info(f"E5 (page) embedding similarity: {e5_similarity:.4f}", extra={"icon": "📊"})
    else:
        logger.warning("Missing page-level embeddings in database", extra={"icon": "⚠️"})
    
    if len(doc1_qwen_embed) > 0 and len(doc2_qwen_embed) > 0:
        qwen_similarity = calculate_cosine_similarity(doc1_qwen_embed, doc2_qwen_embed)
        logger.info(f"Qwen (document) embedding similarity: {qwen_similarity:.4f}", extra={"icon": "📊"})
    else:
        logger.warning("Missing document-level embeddings in database", extra={"icon": "⚠️"})
    
    # Generate fresh embeddings using both models
    logger.info("\nGenerating fresh embeddings for comparison:", extra={"icon": "🔄"})
    
    # Generate E5 embeddings
    fresh_doc1_e5_embed = e5_model.encode(doc1_content, normalize_embeddings=True)
    fresh_doc2_e5_embed = e5_model.encode(doc2_content, normalize_embeddings=True)
    fresh_e5_similarity = calculate_cosine_similarity(fresh_doc1_e5_embed, fresh_doc2_e5_embed)
    
    # Generate Qwen embeddings
    fresh_doc1_qwen_embed = qwen_model.encode(doc1_content, normalize_embeddings=True)
    fresh_doc2_qwen_embed = qwen_model.encode(doc2_content, normalize_embeddings=True)
    fresh_qwen_similarity = calculate_cosine_similarity(fresh_doc1_qwen_embed, fresh_doc2_qwen_embed)
    
    # Log results
    logger.info(f"Fresh E5 embedding similarity: {fresh_e5_similarity:.4f}", extra={"icon": "📊"})
    logger.info(f"Fresh Qwen embedding similarity: {fresh_qwen_similarity:.4f}", extra={"icon": "📊"})
    
    # Print embedding dimensions
    logger.info(f"\nE5 embedding dimensions: {fresh_doc1_e5_embed.shape}", extra={"icon": "📏"})
    logger.info(f"Qwen embedding dimensions: {fresh_doc1_qwen_embed.shape}", extra={"icon": "📏"})
    
    # Check if content is identical
    if doc1_content == doc2_content:
        logger.info("Document contents are identical!", extra={"icon": "🔍"})
    else:
        logger.info("Document contents differ.", extra={"icon": "🔍"})
        
        # Count words in each document
        doc1_words = len(doc1_content.split())
        doc2_words = len(doc2_content.split())
        logger.info(f"Document 1: {doc1_words} words", extra={"icon": "📝"})
        logger.info(f"Document 2: {doc2_words} words", extra={"icon": "📝"})
        
        # Calculate percentage difference in word count
        word_diff_pct = abs(doc1_words - doc2_words) / max(doc1_words, doc2_words) * 100
        logger.info(f"Word count difference: {word_diff_pct:.2f}%", extra={"icon": "📊"})

if __name__ == "__main__":
    logger.info("Starting embedding comparison test", extra={"icon": "🚀"})
    
    # These are the document IDs we want to compare
    doc1_id = "fighter_jet_rocket_launcher_spec_2.pdf"
    doc2_id = "fighter_jet_rocket_launcher_spec_2_changed_values.pdf"
    
    # Check if command line arguments are provided
    if len(sys.argv) > 2:
        doc1_id = sys.argv[1]
        doc2_id = sys.argv[2]
        logger.info(f"Using command line arguments for document IDs: {doc1_id}, {doc2_id}", extra={"icon": "💬"})
    
    logger.info(f"Comparing documents: {doc1_id} and {doc2_id}", extra={"icon": "🔍"})
    compare_document_embeddings(doc1_id, doc2_id)
    logger.info("Embedding comparison complete", extra={"icon": "✅"}) 