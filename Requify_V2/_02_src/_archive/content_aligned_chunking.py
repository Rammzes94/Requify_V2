"""
content_aligned_chunking.py

This script combines document deduplication and intelligent chunking capabilities.
It performs the following operations:
1. Performs page-level embedding comparison to detect similar documents
2. Only proceeds to chunking if document is not an exact duplicate
3. For similar documents, ensures new chunks align with existing chunks where content matches
4. Maintains semantic consistency across document versions while properly handling new content
5. Works at both page level and chunk level to maximize deduplication effectiveness
"""

import os
import sys
import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import lancedb
from lancedb.pydantic import LanceModel, Vector
from sentence_transformers import SentenceTransformer
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.groq import Groq
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import pandas as pd
import uuid

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
import _00_utils
_00_utils.setup_project_directory()

# Load environment variables
load_dotenv()

# Setup logging with script prefix
class ScriptLogger(logging.LoggerAdapter):
    def __init__(self, logger, prefix):
        super().__init__(logger, {})
        self.prefix = prefix
        
    def process(self, msg, kwargs):
        return f"{self.prefix}{msg}", kwargs

logger = ScriptLogger(_00_utils.setup_logging(), "[Content_Chunking] ")

# -------------------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------------------
OUTPUT_DIR_BASE = "_03_output"
LANCEDB_SUBDIR_NAME = "lancedb"
DOCUMENTS_TABLE = "documents"  # Added to access the documents table
DOCUMENT_CHUNKS_TABLE = "document_chunks"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
EMBEDDING_DIMENSION = 1024  # Dimension for e5-large models
TARGET_CHUNK_SIZE = 150  # Updated target token count per chunk (midpoint of 100-200 range)
MIN_TOKENS = 100  # Minimum tokens per chunk
MAX_TOKENS = 200  # Maximum tokens per chunk
CHUNK_SIMILARITY_THRESHOLD = 0.92  # Threshold for aligning chunks
EXACT_PAGE_DUPLICATE_THRESHOLD = 0.98  # High threshold for exact page duplicates
SIMILAR_PAGE_THRESHOLD = 0.90  # Lower threshold for similar pages

# API keys from environment variables
api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize LLM models
active_chunking_model = Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key, temperature=0)

# -------------------------------------------------------------------------------------
# Models
# -------------------------------------------------------------------------------------
class DocumentChunk(BaseModel):
    """Model representing a single document chunk."""
    chunk_id: str
    document_id: str
    chunk_index: int
    start_offset: int
    end_offset: int
    chunk_text: str
    is_updated: bool = False
    previous_chunk_id: Optional[str] = None
    token_count: int

class ChunkingResults(BaseModel):
    """Model for the agent's chunking results."""
    chunks: List[Dict[str, Any]] = Field(..., description="List of chunks with metadata")
    rationale: str = Field(..., description="Explanation of chunking decisions")

class DocumentChunkDB(LanceModel):
    """LanceDB model for storing document chunks. Must match the schema from init_lancedb.py."""
    chunk_id: str
    document_id: str
    chunk_index: int
    start_offset: int
    end_offset: int
    chunk_text: str
    token_count: int
    embedding: Vector(EMBEDDING_DIMENSION)
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    is_updated: bool = False
    previous_chunk_id: Optional[str] = None
    timestamp: str
    aligned_with_chunk_id: Optional[str] = None
    aligned_with_document_id: Optional[str] = None

# -------------------------------------------------------------------------------------
# Agent Configuration
# -------------------------------------------------------------------------------------
chunking_agent = Agent(
    model=active_chunking_model,
    response_model=ChunkingResults,
    description="You are an expert document chunking agent that divides documents into semantically coherent chunks while maintaining alignment with existing chunks where content is similar."
)

# -------------------------------------------------------------------------------------
# Database Functions
# -------------------------------------------------------------------------------------
def connect_to_lancedb():
    """Connect to LanceDB and return the connection."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    lancedb_path = os.path.join(project_root, OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME)
    
    try:
        logger.info(f"Connecting to LanceDB database at {lancedb_path}", extra={"icon": "🔄"})
        os.makedirs(lancedb_path, exist_ok=True)
        db = lancedb.connect(lancedb_path)
        logger.info(f"Connected to LanceDB", extra={"icon": "✅"})
        return db
    except Exception as e:
        logger.error(f"Failed to connect to LanceDB: {e}", extra={"icon": "❌"})
        return None

def get_or_create_chunks_table(db):
    """Get the document chunks table from LanceDB or create it if it doesn't exist."""
    if not db:
        logger.error("No database connection provided", extra={"icon": "❌"})
        return None
        
    # Get table names and check if table exists
    table_names = db.table_names()
    
    # Try to open existing table
    if DOCUMENT_CHUNKS_TABLE in table_names:
        logger.info(f"Opening existing document chunks table", extra={"icon": "📂"})
        return db.open_table(DOCUMENT_CHUNKS_TABLE)
    
    # Table doesn't exist, create it
    logger.info(f"Creating new document chunks table", extra={"icon": "🆕"})
    table = db.create_table(DOCUMENT_CHUNKS_TABLE, schema=DocumentChunkDB)
    logger.info(f"Table created successfully", extra={"icon": "✅"})
    return table

def get_documents_table(db):
    """Get the documents table from LanceDB if it exists."""
    if not db:
        logger.error("No database connection provided", extra={"icon": "❌"})
        return None
        
    # Get table names and check if table exists
    table_names = db.table_names()
    
    # Try to open existing table
    if DOCUMENTS_TABLE in table_names:
        logger.info(f"Opening existing documents table", extra={"icon": "📂"})
        return db.open_table(DOCUMENTS_TABLE)
    
    logger.warning(f"Documents table does not exist", extra={"icon": "⚠️"})
    return None

# -------------------------------------------------------------------------------------
# Embedding Functions
# -------------------------------------------------------------------------------------
def load_embedding_model():
    """Load the embedding model."""
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info(f"Loaded embedding model: {EMBEDDING_MODEL_NAME}", extra={"icon": "✅"})
        return model
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}", extra={"icon": "❌"})
        return None

def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Normalize an embedding vector to unit length."""
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm

def calculate_similarity(embed1: np.ndarray, embed2: np.ndarray) -> float:
    """Calculate cosine similarity between two normalized embeddings."""
    return float(np.dot(normalize_embedding(embed1), normalize_embedding(embed2)))

def generate_embeddings(chunks: List[DocumentChunk], model) -> List[Tuple[DocumentChunk, np.ndarray]]:
    """Generate embeddings for a list of document chunks."""
    if not model:
        logger.error("No embedding model provided", extra={"icon": "❌"})
        return []
    
    chunk_embeddings = []
    for chunk in chunks:
        text = f"passage: {chunk.chunk_text}"
        try:
            embedding = model.encode(text)
            chunk_embeddings.append((chunk, embedding))
        except Exception as e:
            logger.error(f"Error generating embedding for chunk: {e}", extra={"icon": "❌"})
    
    logger.info(f"Generated embeddings for {len(chunk_embeddings)} chunks", extra={"icon": "✅"})
    return chunk_embeddings

def generate_page_embedding(page_text: str, model) -> np.ndarray:
    """Generate embedding for a page of text."""
    if not page_text.strip():
        logger.warning("Empty page text for embedding", extra={"icon": "⚠️"})
        return np.zeros(EMBEDDING_DIMENSION)
        
    # For e5 models, text needs to be prefixed with "passage: "
    text = f"passage: {page_text}"
    try:
        embedding = model.encode(text)
        return embedding
    except Exception as e:
        logger.error(f"Error generating page embedding: {e}", extra={"icon": "❌"})
        return np.zeros(EMBEDDING_DIMENSION)

def generate_chunk_embedding(chunk_text: str, embedding_model) -> np.ndarray:
    """
    Generate an embedding for a chunk of text.
    
    Args:
        chunk_text: The text to embed
        embedding_model: The embedding model to use
        
    Returns:
        The embedding vector as a numpy array
    """
    try:
        # Prefix for document chunks
        text = f"passage: {chunk_text.strip()}"
        
        # Generate embedding
        embedding = embedding_model.encode(text)
        return embedding
    except Exception as e:
        logger.error(f"Error generating chunk embedding: {e}", extra={"icon": "❌"})
        raise

# -------------------------------------------------------------------------------------
# Page-Level Deduplication
# -------------------------------------------------------------------------------------
def check_document_page_duplicates(document_pages: List[Dict], embedding_model) -> Tuple[bool, Optional[str], List[int], bool]:
    """
    Check document pages against existing pages in the database.
    
    Args:
        document_pages: List of dictionaries containing page data
        embedding_model: Embedding model to use
        
    Returns:
        Tuple of (is_complete_duplicate, similar_document_id, new_page_indices, is_similar_document)
    """
    logger.info(f"Checking {len(document_pages)} pages for duplicates", extra={"icon": "🔍"})
    
    db = connect_to_lancedb()
    if not db:
        return False, None, list(range(len(document_pages))), False
    
    # Get documents table
    docs_table = get_documents_table(db)
    if not docs_table:
        return False, None, list(range(len(document_pages))), False
        
    # Get the document ID from the first page (if available)
    current_doc_id = None
    if document_pages and 'pdf_identifier' in document_pages[0]:
        current_doc_id = document_pages[0]['pdf_identifier']
        logger.info(f"Current document ID: {current_doc_id}", extra={"icon": "ℹ️"})
    
    # Track duplicate and similar pages
    duplicate_pages = []
    new_page_indices = []
    similar_document_counts = {}
    is_similar_document = False
    
    # Check each page for duplicates
    for i, page_data in enumerate(document_pages):
        # Extract text to compare
        page_text = page_data.get('md_content', '')
        if not page_text:
            logger.warning(f"Page {i+1} has no text content", extra={"icon": "⚠️"})
            new_page_indices.append(i)
            continue
        
        # Generate embedding for page
        embedding = generate_page_embedding(page_text, embedding_model)
        
        # Search for similar pages
        try:
            results = docs_table.search(embedding).limit(10).to_df()
            
            if results.empty:
                logger.info(f"No similar pages found for page {i+1}", extra={"icon": "ℹ️"})
                new_page_indices.append(i)
                continue
                
            # Skip self-comparisons (same document) when looking for duplicates
            if current_doc_id:
                results = results[results['pdf_identifier'] != current_doc_id]
                
            if results.empty:
                logger.info(f"No similar pages found in other documents for page {i+1}", extra={"icon": "ℹ️"})
                new_page_indices.append(i)
                continue
                
            # Check for duplicates or similar pages
            found_duplicate = False
            for _, row in results.iterrows():
                similarity = 1.0 - row['_distance']
                doc_id = row['pdf_identifier']
                
                # Skip self comparisons
                if doc_id == current_doc_id:
                    continue
                
                logger.info(f"Found similar page in document {doc_id} (similarity: {similarity:.4f})", extra={"icon": "🔍"})
                
                # Exact duplicate threshold
                if similarity >= EXACT_PAGE_DUPLICATE_THRESHOLD:
                    found_duplicate = True
                    duplicate_pages.append(i)
                    similar_document_counts[doc_id] = similar_document_counts.get(doc_id, 0) + 1
                    logger.info(f"Page {i+1} is a duplicate of page in document {doc_id} (similarity: {similarity:.4f})", extra={"icon": "🔍"})
                    break
                    
                # Similar document threshold (but not exact)
                elif similarity >= SIMILAR_PAGE_THRESHOLD:
                    is_similar_document = True
                    similar_document_counts[doc_id] = similar_document_counts.get(doc_id, 0) + 1
                    logger.info(f"Page {i+1} is similar to page in document {doc_id} (similarity: {similarity:.4f})", extra={"icon": "🔍"})
            
            if not found_duplicate:
                new_page_indices.append(i)
                
        except Exception as e:
            logger.error(f"Error searching for similar pages: {e}", extra={"icon": "❌"})
            new_page_indices.append(i)
    
    # Determine if this is a complete duplicate
    is_complete_duplicate = len(duplicate_pages) == len(document_pages) and len(document_pages) > 0
    
    # Find the most similar document (for aligned chunking if needed)
    most_similar_doc = None
    max_matches = 0
    for doc_id, count in similar_document_counts.items():
        if count > max_matches:
            max_matches = count
            most_similar_doc = doc_id
    
    logger.info(f"Page-level results: {len(duplicate_pages)} duplicates, {len(new_page_indices)} new pages", extra={"icon": "📊"})
    if most_similar_doc:
        logger.info(f"Most similar document: {most_similar_doc} ({max_matches} matching pages)", extra={"icon": "🔍"})
    
    return is_complete_duplicate, most_similar_doc, new_page_indices, is_similar_document

# -------------------------------------------------------------------------------------
# Chunk-Level Similarity Detection
# -------------------------------------------------------------------------------------
def find_chunks_by_document_id(document_id: str) -> List[Dict]:
    """Find all chunks for a specific document ID."""
    logger.info(f"Retrieving chunks for document {document_id}", extra={"icon": "🔍"})
    
    db = connect_to_lancedb()
    if not db or DOCUMENT_CHUNKS_TABLE not in db.table_names():
        return []
    
    chunks_table = db.open_table(DOCUMENT_CHUNKS_TABLE)
    
    try:
        # Use pandas filtering for compatibility with LanceDB 0.22.0
        all_chunks = chunks_table.to_pandas()
        document_chunks = all_chunks[all_chunks['document_id'] == document_id]
        
        if document_chunks.empty:
            logger.info(f"No chunks found for document {document_id}", extra={"icon": "ℹ️"})
            return []
        
        similar_chunks = []
        for _, row in document_chunks.iterrows():
            similar_chunks.append({
                "chunk_id": row['chunk_id'],
                "document_id": row['document_id'],
                "chunk_text": row['chunk_text'],
                "chunk_index": row['chunk_index'],
                "similarity": 1.0,  # Treating these as exact matches for alignment
            })
        
        # Sort by chunk_index
        similar_chunks.sort(key=lambda x: x.get("chunk_index", 0))
        logger.info(f"Found {len(similar_chunks)} chunks for document {document_id}", extra={"icon": "✅"})
        return similar_chunks
    except Exception as e:
        logger.error(f"Error finding chunks for document {document_id}: {e}", extra={"icon": "❌"})
        return []

def find_similar_chunks(document_text: str, embedding_model, current_document_id: str = None) -> List[Dict]:
    """Find chunks from existing documents that are similar to parts of this document."""
    logger.info(f"Checking for similar chunks based on content", extra={"icon": "🔍"})
    
    db = connect_to_lancedb()
    if not db:
        return []
    
    # Check if chunks table exists
    if DOCUMENT_CHUNKS_TABLE not in db.table_names():
        logger.info(f"No document chunks table found. This appears to be a new document.", extra={"icon": "ℹ️"})
        return []
    
    # Simple text-based chunking for comparison
    paragraphs = [p for p in document_text.split("\n\n") if p.strip()]
    if not paragraphs:
        temp_chunks = [document_text[i:i+1000] for i in range(0, len(document_text), 1000)]
        paragraphs = [chunk for chunk in temp_chunks if chunk.strip()]
    
    # Get chunks table
    chunks_table = db.open_table(DOCUMENT_CHUNKS_TABLE)
    
    # Set similarity threshold
    threshold = 0.8
    
    similar_chunks = []
    
    # For each paragraph, find similar chunks
    for i, paragraph in enumerate(paragraphs):
        try:
            # Generate embedding for paragraph
            embedding = generate_chunk_embedding(paragraph, embedding_model)
            
            # Add filter condition if we have a current document ID
            where_clause = ""
            if current_document_id:
                where_clause = f"document_id != '{current_document_id}'"
            
            # Search for similar chunks - use only vector_column_name without metric
            try:
                search_query = chunks_table.search(embedding.tolist(), vector_column_name="embedding")
                
                # Apply the filter if needed
                if where_clause:
                    search_query = search_query.where(where_clause)
                    
                # Limit results
                search_query = search_query.limit(5)
                    
                # Execute the search
                results = search_query.to_pandas()
            except Exception as e:
                logger.error(f"Search query failed: {e}. Trying without vector_column_name specification.", extra={"icon": "⚠️"})
                # Fallback attempt without any special parameters
                search_query = chunks_table.search(embedding.tolist())
                if where_clause:
                    search_query = search_query.where(where_clause)
                results = search_query.limit(5).to_pandas()
            
            # Check if any chunks are above threshold
            if not results.empty:
                for _, row in results.iterrows():
                    # Ensure _distance exists
                    if "_distance" not in row:
                        logger.warning(f"Search result missing _distance field, skipping", extra={"icon": "⚠️"})
                        continue
                    
                    score = float(row["_distance"])
                    similarity = 1.0 - score  # Convert distance to similarity
                    
                    if similarity >= threshold:
                        # Ensure all required fields exist
                        required_fields = ["chunk_id", "document_id", "chunk_text", "start_offset", "end_offset"]
                        if not all(field in row for field in required_fields):
                            logger.warning(f"Search result missing required fields, skipping", extra={"icon": "⚠️"})
                            continue
                        
                        similar_chunk = {
                            "chunk_id": row["chunk_id"],
                            "document_id": row["document_id"],
                            "chunk_text": row["chunk_text"],
                            "start_offset": row["start_offset"],
                            "end_offset": row["end_offset"],
                            "similarity": similarity,
                            "para_index": i
                        }
                        similar_chunks.append(similar_chunk)
        except Exception as e:
            logger.error(f"Error searching for similar chunks: {e}", extra={"icon": "❌"})
            continue
    
    # Sort by similarity (highest first)
    similar_chunks.sort(key=lambda x: x["similarity"], reverse=True)
    
    logger.info(f"Found {len(similar_chunks)} similar chunks from other documents", extra={"icon": "🔍"})
    return similar_chunks

# -------------------------------------------------------------------------------------
# Document Chunking Functions
# -------------------------------------------------------------------------------------
def estimate_token_count(text: str) -> int:
    """
    Estimate the number of tokens in a piece of text.
    This is a rough approximation - about 4 characters per token for English text.
    """
    return len(text) // 4

def create_chunks_from_text(document_text: str) -> List[Dict]:
    """
    Create chunks from document text ensuring each chunk has 100-200 tokens.
    Combines smaller paragraphs to reach minimum token count and splits large ones.
    
    Args:
        document_text: The document text to chunk
        
    Returns:
        List of chunk dictionaries
    """
    logger.info(f"Creating chunks from document text (length: {len(document_text)})", extra={"icon": "✂️"})
    
    # Split text into paragraphs
    paragraphs = [p for p in document_text.split("\n\n") if p.strip()]
    
    # If no paragraphs, use character-based chunking
    if not paragraphs:
        logger.warning(f"No paragraphs found in document. Using character-based chunking.", extra={"icon": "⚠️"})
        # Calculate character counts that correspond to token ranges
        min_chars = MIN_TOKENS * 4
        max_chars = MAX_TOKENS * 4
        chunks_data = []
        
        # Create chunks of appropriate size
        for i in range(0, len(document_text), max_chars):
            chunk_text = document_text[i:i+max_chars]
            chunk_id = f"chunk_{len(chunks_data)+1}_{uuid.uuid4().hex[:8]}"
            chunks_data.append({
                "chunk_id": chunk_id,
                "chunk_index": len(chunks_data),
                "chunk_text": chunk_text,
                "start_offset": i,
                "end_offset": min(i+max_chars, len(document_text)),
                "token_count": estimate_token_count(chunk_text)
            })
        
        logger.info(f"Created {len(chunks_data)} chunks using character-based chunking", extra={"icon": "✅"})
        return chunks_data
    
    # Create optimized chunks from paragraphs
    chunks_data = []
    current_chunk = ""
    current_paragraphs = []
    
    for para in paragraphs:
        # Skip empty paragraphs
        if not para.strip():
            continue
            
        para_token_count = estimate_token_count(para)
        current_token_count = estimate_token_count(current_chunk)
        
        # Case 1: Adding this paragraph would make chunk too large
        if current_chunk and (current_token_count + para_token_count > MAX_TOKENS):
            # Save current chunk if it's large enough
            if current_token_count >= MIN_TOKENS:
                chunk_id = f"chunk_{len(chunks_data)+1}_{uuid.uuid4().hex[:8]}"
                # Get the absolute offsets from original text
                start_offset = document_text.find(current_paragraphs[0])
                end_offset = start_offset + len(current_chunk)
                
                chunks_data.append({
                    "chunk_id": chunk_id,
                    "chunk_index": len(chunks_data),
                    "chunk_text": current_chunk.strip(),
                    "start_offset": start_offset if start_offset >= 0 else 0,
                    "end_offset": end_offset,
                    "token_count": current_token_count
                })
                
                # Reset for new chunk
                current_chunk = para
                current_paragraphs = [para]
            else:
                # Current chunk is too small, add paragraph anyway
                current_chunk += "\n\n" + para
                current_paragraphs.append(para)
        
        # Case 2: Paragraph itself is too large
        elif para_token_count > MAX_TOKENS:
            # Save current chunk if needed
            if current_chunk and estimate_token_count(current_chunk) >= MIN_TOKENS:
                chunk_id = f"chunk_{len(chunks_data)+1}_{uuid.uuid4().hex[:8]}"
                start_offset = document_text.find(current_paragraphs[0])
                end_offset = start_offset + len(current_chunk)
                
                chunks_data.append({
                    "chunk_id": chunk_id,
                    "chunk_index": len(chunks_data),
                    "chunk_text": current_chunk.strip(),
                    "start_offset": start_offset if start_offset >= 0 else 0,
                    "end_offset": end_offset,
                    "token_count": estimate_token_count(current_chunk)
                })
                
                # Reset
                current_chunk = ""
                current_paragraphs = []
            
            # Split the large paragraph into smaller chunks
            # Using exact character counts to ensure we don't exceed MAX_TOKENS
            chars_per_chunk = MAX_TOKENS * 4 - 20  # Subtract a buffer to avoid going over
            para_offset = document_text.find(para)
            if para_offset < 0:
                para_offset = 0
                
            for j in range(0, len(para), chars_per_chunk):
                sub_chunk = para[j:j+chars_per_chunk]
                sub_chunk_tokens = estimate_token_count(sub_chunk)
                
                # If this is the last sub-chunk and it's too small, combine with previous
                if j + chars_per_chunk >= len(para) and sub_chunk_tokens < MIN_TOKENS and chunks_data:
                    # Try to combine with previous chunk if possible
                    prev_chunk = chunks_data[-1]
                    combined_text = prev_chunk["chunk_text"] + "\n\n" + sub_chunk
                    combined_tokens = estimate_token_count(combined_text)
                    
                    if combined_tokens <= MAX_TOKENS:
                        # Update previous chunk
                        prev_chunk["chunk_text"] = combined_text
                        prev_chunk["end_offset"] = para_offset + j + len(sub_chunk)
                        prev_chunk["token_count"] = combined_tokens
                    else:
                        # Must add as its own chunk
                        chunk_id = f"chunk_{len(chunks_data)+1}_{uuid.uuid4().hex[:8]}"
                        chunks_data.append({
                            "chunk_id": chunk_id,
                            "chunk_index": len(chunks_data),
                            "chunk_text": sub_chunk.strip(),
                            "start_offset": para_offset + j,
                            "end_offset": para_offset + j + len(sub_chunk),
                            "token_count": sub_chunk_tokens
                        })
                else:
                    # Add as normal chunk
                    chunk_id = f"chunk_{len(chunks_data)+1}_{uuid.uuid4().hex[:8]}"
                    chunks_data.append({
                        "chunk_id": chunk_id,
                        "chunk_index": len(chunks_data),
                        "chunk_text": sub_chunk.strip(),
                        "start_offset": para_offset + j,
                        "end_offset": para_offset + j + len(sub_chunk),
                        "token_count": sub_chunk_tokens
                    })
        
        # Case 3: Add to current chunk
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
            current_paragraphs.append(para)
    
    # Add the last chunk if it meets minimum size
    if current_chunk and estimate_token_count(current_chunk) >= MIN_TOKENS:
        chunk_id = f"chunk_{len(chunks_data)+1}_{uuid.uuid4().hex[:8]}"
        start_offset = document_text.find(current_paragraphs[0])
        end_offset = start_offset + len(current_chunk)
        
        chunks_data.append({
            "chunk_id": chunk_id,
            "chunk_index": len(chunks_data),
            "chunk_text": current_chunk.strip(),
            "start_offset": start_offset if start_offset >= 0 else 0,
            "end_offset": end_offset,
            "token_count": estimate_token_count(current_chunk)
        })
    elif current_chunk and chunks_data:
        # Try to combine with the last chunk if it's too small
        last_chunk = chunks_data[-1]
        combined_text = last_chunk["chunk_text"] + "\n\n" + current_chunk
        combined_tokens = estimate_token_count(combined_text)
        
        if combined_tokens <= MAX_TOKENS:
            # Update the last chunk
            last_chunk["chunk_text"] = combined_text
            last_chunk["end_offset"] = end_offset
            last_chunk["token_count"] = combined_tokens
    
    # Post-processing: check for any chunks still outside the target range
    final_chunks = []
    i = 0
    
    while i < len(chunks_data):
        chunk = chunks_data[i]
        
        # If chunk is too big, split it further
        if chunk["token_count"] > MAX_TOKENS:
            max_chars = MAX_TOKENS * 4 - 20  # Using a buffer
            chunk_text = chunk["chunk_text"]
            chunk_offset = chunk["start_offset"]
            
            for j in range(0, len(chunk_text), max_chars):
                sub_chunk = chunk_text[j:j+max_chars]
                sub_chunk_tokens = estimate_token_count(sub_chunk)
                
                chunk_id = f"chunk_{len(final_chunks)+1}_{uuid.uuid4().hex[:8]}"
                final_chunks.append({
                    "chunk_id": chunk_id,
                    "chunk_index": len(final_chunks),
                    "chunk_text": sub_chunk.strip(),
                    "start_offset": chunk_offset + j,
                    "end_offset": chunk_offset + j + len(sub_chunk),
                    "token_count": sub_chunk_tokens
                })
        # If chunk is too small and there's a next chunk, try to combine
        elif chunk["token_count"] < MIN_TOKENS and i < len(chunks_data) - 1:
            next_chunk = chunks_data[i + 1]
            combined_text = chunk["chunk_text"] + "\n\n" + next_chunk["chunk_text"]
            combined_tokens = estimate_token_count(combined_text)
            
            if combined_tokens <= MAX_TOKENS:
                # Create combined chunk
                chunk_id = f"chunk_{len(final_chunks)+1}_{uuid.uuid4().hex[:8]}"
                final_chunks.append({
                    "chunk_id": chunk_id,
                    "chunk_index": len(final_chunks),
                    "chunk_text": combined_text.strip(),
                    "start_offset": chunk["start_offset"],
                    "end_offset": next_chunk["end_offset"],
                    "token_count": combined_tokens
                })
                # Skip the next chunk since we combined it
                i += 1
            else:
                # If we can't combine, still add this small chunk
                # (this may happen at document boundaries)
                final_chunks.append(chunk)
        # If this chunk is too small but it's the last one, attach to previous if possible
        elif chunk["token_count"] < MIN_TOKENS and i == len(chunks_data) - 1 and final_chunks:
            prev_chunk = final_chunks[-1]
            combined_text = prev_chunk["chunk_text"] + "\n\n" + chunk["chunk_text"]
            combined_tokens = estimate_token_count(combined_text)
            
            if combined_tokens <= MAX_TOKENS:
                # Update the previous chunk
                prev_chunk["chunk_text"] = combined_text
                prev_chunk["end_offset"] = chunk["end_offset"]
                prev_chunk["token_count"] = combined_tokens
            else:
                # Must add as a small chunk
                final_chunks.append(chunk)
        else:
            # This chunk is fine as is
            final_chunks.append(chunk)
        
        i += 1
    
    # Check the token count of all chunks
    in_range_count = sum(1 for c in final_chunks if MIN_TOKENS <= c["token_count"] <= MAX_TOKENS)
    logger.info(f"Created {len(final_chunks)} chunks with {in_range_count} in target range {MIN_TOKENS}-{MAX_TOKENS}",
                extra={"icon": "✅"})
    return final_chunks

def chunk_with_alignment(document_text: str, similar_chunks: List[Dict]) -> List[Dict]:
    """Chunk the document text with alignment to similar chunks while ensuring token size constraints."""
    logger.info(f"Chunking document with alignment to {len(similar_chunks)} similar chunks", extra={"icon": "✂️"})
    
    # First create optimized chunks using the improved chunking method
    base_chunks = create_chunks_from_text(document_text)
    
    # Find the most similar chunk for each generated chunk
    for chunk in base_chunks:
        chunk_text = chunk["chunk_text"]
        best_similarity = CHUNK_SIMILARITY_THRESHOLD  # Only align if above threshold
        best_match = None
        
        for similar in similar_chunks:
            # Calculate similarity between this chunk and the candidate
            similarity = calculate_text_similarity(chunk_text, similar["chunk_text"])
            
            if similarity > best_similarity and similarity > CHUNK_SIMILARITY_THRESHOLD:
                best_similarity = similarity
                best_match = similar
        
        # If we found a good match, add alignment information
        if best_match:
            chunk["aligned_with_chunk_id"] = best_match["chunk_id"]
            chunk["aligned_with_document_id"] = best_match["document_id"]
            logger.info(f"Aligned chunk {chunk['chunk_index']+1} with chunk {best_match['chunk_id']} from {best_match['document_id']} (similarity: {best_similarity:.4f})", extra={"icon": "🔗"})
    
    logger.info(f"Created {len(base_chunks)} chunks with token-optimized alignment", extra={"icon": "✅"})
    return base_chunks

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate a simple similarity score between two text fragments."""
    # This is a very basic implementation
    # In a real system, you might use embeddings or more sophisticated methods
    
    # Normalize and tokenize texts
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    
    # Calculate Jaccard similarity
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    
    if union == 0:
        return 0.0
        
    return intersection / union

def save_document_chunks(document_id: str, chunks: List[Dict], embedding_model) -> bool:
    """
    Save document chunks to the database with embeddings.
    
    Args:
        document_id: ID of the document
        chunks: List of chunk dictionaries
        embedding_model: Model to use for generating embeddings
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Saving {len(chunks)} chunks for document {document_id}", extra={"icon": "💾"})
    
    db = connect_to_lancedb()
    if not db:
        return False
    
    # Create the chunks table if it doesn't exist
    if DOCUMENT_CHUNKS_TABLE not in db.table_names():
        logger.info(f"Creating {DOCUMENT_CHUNKS_TABLE} table", extra={"icon": "🔧"})
        
        # Use the DocumentChunkDB schema directly
        table = db.create_table(DOCUMENT_CHUNKS_TABLE, schema=DocumentChunkDB)
    else:
        # Open existing table
        table = db.open_table(DOCUMENT_CHUNKS_TABLE)
    
    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Generate embeddings for chunks
    chunk_records = []
    for chunk in chunks:
        # Generate embedding for chunk
        try:
            embedding = generate_chunk_embedding(chunk["chunk_text"], embedding_model)
            
            # Use provided token count or calculate if missing
            token_count = chunk.get("token_count", estimate_token_count(chunk["chunk_text"]))
            
            chunk_record = {
                "chunk_id": chunk["chunk_id"],
                "document_id": document_id,
                "chunk_index": chunk["chunk_index"],
                "chunk_text": chunk["chunk_text"],
                "start_offset": chunk["start_offset"],
                "end_offset": chunk["end_offset"],
                "embedding": embedding.tolist(),
                "aligned_with_chunk_id": chunk.get("aligned_with_chunk_id", ""),
                "aligned_with_document_id": chunk.get("aligned_with_document_id", ""),
                "token_count": token_count,
                "is_duplicate": False,
                "duplicate_of": "",  # Empty string instead of None
                "is_updated": False,
                "previous_chunk_id": "",  # Empty string instead of None
                "timestamp": timestamp
            }
            chunk_records.append(chunk_record)
        except Exception as e:
            logger.error(f"Error generating embedding for chunk {chunk['chunk_id']}: {e}", extra={"icon": "❌"})
    
    if not chunk_records:
        logger.error(f"No valid chunks to save for document {document_id}", extra={"icon": "❌"})
        return False
    
    # Create DocumentChunkDB instances from records for proper schema validation
    validated_records = []
    for record in chunk_records:
        try:
            # Convert empty strings to None for optional string fields
            for field in ["duplicate_of", "previous_chunk_id", "aligned_with_chunk_id", "aligned_with_document_id"]:
                if field in record and record[field] == "":
                    record[field] = None
                    
            # Create a validated instance
            validated_record = DocumentChunkDB.model_validate(record)
            validated_records.append(validated_record)
        except Exception as e:
            logger.error(f"Error validating chunk record: {e}", extra={"icon": "❌"})
    
    # Add chunks to table
    try:
        table.add(validated_records)
        logger.info(f"Successfully saved {len(validated_records)} chunks for document {document_id}", extra={"icon": "✅"})
        return True
    except Exception as e:
        logger.error(f"Error saving chunks to database: {e}", extra={"icon": "❌"})
        return False

# -------------------------------------------------------------------------------------
# Main Processing Function
# -------------------------------------------------------------------------------------
def process_document(document_text: str, document_id: str, document_pages: List[Dict] = None) -> bool:
    """
    Process a document with page-level similarity check and content-aligned chunking.
    
    Args:
        document_text: The full document text
        document_id: ID of the document
        document_pages: Optional list of page dictionaries with md_content field
    
    Returns:
        True if processing was successful, False otherwise
    """
    logger.info(f"Processing document {document_id} (length: {len(document_text)} chars)", extra={"icon": "🔄"})
    
    # Load embedding model
    embedding_model = load_embedding_model()
    if not embedding_model:
        logger.error(f"Failed to load embedding model", extra={"icon": "❌"})
        return False
    
    # If pages weren't provided, generate them now
    if not document_pages or len(document_pages) == 0:
        # This is simplified - in practice, you'd have your page splitting logic here
        document_pages = [{"md_content": document_text}]
    
    # Check for duplicate document at page level
    is_complete_duplicate, similar_document_id, new_page_indices, is_similar_document = check_document_page_duplicates(
        document_pages, embedding_model
    )
    
    # Continue with processing in all cases - don't abort for duplicates
    # (This allows us to test alignment functionality)
    
    # Look for similar chunks in other documents for alignment
    similar_chunks = find_similar_chunks(document_text, embedding_model, current_document_id=document_id)
    
    # Create chunks based on similar content
    if similar_chunks:
        logger.info(f"Found {len(similar_chunks)} chunks from other documents to align with", extra={"icon": "📊"})
        # Use aligned chunking based on the similar chunks found
        chunks = chunk_with_alignment(document_text, similar_chunks)
    else:
        # Otherwise do regular chunking for new content
        logger.info(f"No similar chunks found in other documents. Using regular chunking.", extra={"icon": "ℹ️"})
        chunks = create_chunks_from_text(document_text)
    
    # Add document_id to each chunk
    for chunk in chunks:
        chunk["document_id"] = document_id
    
    # Save chunks with embeddings
    result = save_document_chunks(document_id, chunks, embedding_model)
    
    return result

# Entry point for using the script directly
if __name__ == "__main__":
    # Simple CLI for testing
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        logger.info(f"Processing document: {file_path}", extra={"icon": "🔍"})
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                document_text = f.read()
            
            document_id = os.path.basename(file_path)
            
            # Create simple page dictionary for testing
            test_pages = [{"md_content": document_text}]
            
            success = process_document(document_text, document_id, test_pages)
            
            if success:
                logger.info(f"Document processed and saved successfully.", extra={"icon": "✅"})
            else:
                logger.error("Failed to process and save document.", extra={"icon": "❌"})
                
        except Exception as e:
            logger.error(f"Error processing document: {e}", extra={"icon": "❌"})
    else:
        logger.info("Usage: python content_aligned_chunking.py <file_path>", extra={"icon": "ℹ️"}) 