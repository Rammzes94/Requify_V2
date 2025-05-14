"""
agentic_chunking.py

This script provides intelligent document chunking using an LLM-based agent.
It performs the following operations:
1. Takes a markdown document and chunks it into semantically coherent segments
2. Maintains consistent chunk boundaries across document versions
3. Uses Agno with GPT-4o-mini to intelligently split content
4. Ensures chunk boundaries align with the original document when handling updates
5. Generates metadata for each chunk including boundary offsets and context
6. Optimizes chunk sizes to target ~128 tokens each
7. Handles document updates by preserving chunk boundaries where content is unchanged

The chunking strategy prioritizes maintaining semantic coherence while ensuring
that boundaries are preserved across document versions to enable reliable 
comparison and requirement tracing.
"""

import os
import sys
import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
import lancedb
from lancedb.pydantic import LanceModel, Vector
from sentence_transformers import SentenceTransformer
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.groq import Groq
from dotenv import load_dotenv
import numpy as np

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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

logger = ScriptLogger(_00_utils.setup_logging(), "[Agentic_Chunking] ")

# -------------------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------------------
OUTPUT_DIR_BASE = "_03_output"
LANCEDB_SUBDIR_NAME = "lancedb"
DOCUMENT_CHUNKS_TABLE = "document_chunks"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
EMBEDDING_DIMENSION = 1024  # Dimension for e5-large models
TARGET_CHUNK_SIZE = 128  # Target token count per chunk
EXACT_DUPLICATE_THRESHOLD = 0.99  # Cosine similarity threshold for exact duplicates
NEAR_DUPLICATE_THRESHOLD = 0.90  # Threshold for detecting updated chunks

# API keys from environment variables
api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Model Configuration
# Initialize LLM models
gpt4o_mini = OpenAIChat(id="gpt-4o-mini", api_key=api_key, temperature=0)
groq_llama = Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key, temperature=0)

# Select which model to use
active_chunking_model = groq_llama  # Change to gpt4o_mini if preferred

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
    """LanceDB model for storing document chunks."""
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

# -------------------------------------------------------------------------------------
# Agent Configuration
# -------------------------------------------------------------------------------------
chunking_agent = Agent(
    model=active_chunking_model,
    response_model=ChunkingResults,
    description="""You are an expert document chunking agent. 
    Your task is to divide documents into semantically coherent chunks while maintaining
    consistent boundaries across document versions. You ensure that chunks are properly sized
    and that boundaries align with the original document when handling updates."""
)

# -------------------------------------------------------------------------------------
# Helper Functions
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

def get_or_create_chunks_table(db):
    """Get the document chunks table from LanceDB or create it if it doesn't exist."""
    if not db:
        logger.error("No database connection provided for chunks table access", extra={"icon": "❌"})
        return None
        
    # Get table names and check if table exists
    table_names = db.table_names()
    logger.info(f"Available tables in database: {', '.join(table_names)}", extra={"icon": "📋"})
    
    # Try to open existing table
    if DOCUMENT_CHUNKS_TABLE in table_names:
        logger.info(f"Opening existing document chunks table: {DOCUMENT_CHUNKS_TABLE}", extra={"icon": "📂"})
        return db.open_table(DOCUMENT_CHUNKS_TABLE)
    
    # Table doesn't exist, create it
    logger.info(f"Creating new document chunks table: {DOCUMENT_CHUNKS_TABLE}", extra={"icon": "🆕"})
    # Create the table with DocumentChunkDB model
    table = db.create_table(DOCUMENT_CHUNKS_TABLE, schema=DocumentChunkDB)
    logger.info(f"Table {DOCUMENT_CHUNKS_TABLE} created successfully", extra={"icon": "✅"})
    return table

def connect_to_lancedb():
    """Connect to LanceDB and return the connection."""
    # Construct path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    lancedb_path = os.path.join(project_root, OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME)
    
    try:
        logger.info(f"Connecting to LanceDB database at {lancedb_path}", extra={"icon": "🔄"})
        os.makedirs(lancedb_path, exist_ok=True)
        db = lancedb.connect(lancedb_path)
        logger.info(f"Connected to LanceDB at {lancedb_path}", extra={"icon": "✅"})
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

def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Calculate cosine similarity between two normalized embeddings."""
    return float(np.dot(embedding1, embedding2))

# -------------------------------------------------------------------------------------
# Document Chunking Functions
# -------------------------------------------------------------------------------------
def get_previous_document_chunks(document_id: str) -> List[Dict]:
    """
    Get chunks from a previous version of the document if it exists.
    
    Args:
        document_id: The ID of the document
        
    Returns:
        List of dictionaries with chunk data, or empty list if no previous document exists
    """
    db = connect_to_lancedb()
    if not db:
        return []
        
    table = get_or_create_chunks_table(db)
    if not table:
        return []
        
    # Check if any chunks exist for this document ID
    try:
        # For LanceDB 0.22.0, use pandas filtering
        logger.info(f"Retrieving previous chunks for document {document_id}", extra={"icon": "🔍"})
        try:
            all_chunks = table.to_pandas()
            
            if all_chunks.empty:
                logger.info(f"Chunks table is empty", extra={"icon": "ℹ️"})
                return []
                
            # Debug info
            logger.info(f"Chunks table has {len(all_chunks)} total rows", extra={"icon": "ℹ️"})
            logger.info(f"Chunks table columns: {list(all_chunks.columns)}", extra={"icon": "ℹ️"})
            
            if 'document_id' not in all_chunks.columns:
                logger.warning(f"No 'document_id' column in chunks table", extra={"icon": "⚠️"})
                return []
                
            # Get unique document IDs for debugging
            unique_docs = all_chunks['document_id'].unique()
            logger.info(f"Documents in chunks table: {unique_docs}", extra={"icon": "ℹ️"})
            
            # Filter for matching document ID
            results = all_chunks[all_chunks['document_id'] == document_id]
            
            if results.empty:
                logger.info(f"No previous chunks found for document {document_id}", extra={"icon": "ℹ️"})
                return []
                
            logger.info(f"Found {len(results)} existing chunks for document {document_id}", extra={"icon": "✅"})
            
            # Convert to list of dictionaries
            chunks = []
            for _, row in results.iterrows():
                chunks.append({
                    "chunk_id": row.get("chunk_id"),
                    "document_id": row.get("document_id"),
                    "chunk_index": row.get("chunk_index"),
                    "start_offset": row.get("start_offset"),
                    "end_offset": row.get("end_offset"),
                    "chunk_text": row.get("chunk_text"),
                    "token_count": row.get("token_count")
                })
            
            # Sort by chunk_index
            chunks.sort(key=lambda x: x.get("chunk_index", 0))
            return chunks
            
        except Exception as e:
            logger.error(f"Error in pandas filtering: {e}", extra={"icon": "❌"})
            return []
            
    except Exception as e:
        logger.error(f"Error retrieving previous chunks for document {document_id}: {e}", extra={"icon": "❌"})
        return []

def chunk_document(document_text: str, document_id: str, previous_chunks: List[Dict] = None) -> List[DocumentChunk]:
    """
    Chunk a document using an LLM-based agent.
    
    Args:
        document_text: The full document text to chunk
        document_id: The ID of the document
        previous_chunks: Optional list of chunks from a previous version
        
    Returns:
        List of DocumentChunk objects
    """
    logger.info(f"Chunking document {document_id} (length: {len(document_text)} chars)", extra={"icon": "🔄"})
    
    # Prepare prompt based on whether we have previous chunks
    if previous_chunks and len(previous_chunks) > 0:
        logger.info(f"Using {len(previous_chunks)} previous chunks to maintain boundaries", extra={"icon": "🔍"})
        
        # Extract previous boundaries for the prompt
        boundaries = []
        for chunk in previous_chunks:
            boundaries.append({
                "start": chunk.get("start_offset", 0),
                "end": chunk.get("end_offset", 0),
                "text_sample": chunk.get("chunk_text", "")[:100] + "..." if len(chunk.get("chunk_text", "")) > 100 else chunk.get("chunk_text", "")
            })
            
        # Construct prompt for updating with previous boundaries
        prompt = f"""
        Please chunk the following document into semantically coherent segments of approximately {TARGET_CHUNK_SIZE} tokens each.
        
        CRITICALLY IMPORTANT: This document is an updated version of a previously chunked document. You MUST maintain the same chunk boundaries
        for any content that has not changed. This ensures consistent chunk references across document versions.
        
        Previous document had these chunk boundaries:
        {json.dumps(boundaries, indent=2)}
        
        Instructions:
        1. Try to match existing chunk boundaries wherever the content is the same or very similar
        2. Only create new chunk boundaries where content has been significantly modified or added
        3. Target approximately {TARGET_CHUNK_SIZE} tokens per chunk
        4. Each chunk should be semantically coherent (don't break in the middle of a thought)
        5. IMPORTANT: You MUST include the actual chunk_text field with the full text content of each chunk
        6. Return chunks with their start and end offsets, text, and token count
        
        Document text to chunk:
        {document_text}
        """
    else:
        # Simpler prompt for new documents
        prompt = f"""
        Please chunk the following document into semantically coherent segments of approximately {TARGET_CHUNK_SIZE} tokens each.
        
        Instructions:
        1. Target approximately {TARGET_CHUNK_SIZE} tokens per chunk
        2. Each chunk should be semantically coherent (don't break in the middle of a thought)
        3. Prefer breaking at paragraph boundaries or section transitions when possible
        4. IMPORTANT: You MUST include the actual chunk_text field with the full text content of each chunk
        5. Return chunks with their start and end offsets, text, and token count
        
        Document text to chunk:
        {document_text}
        """
    
    try:
        # Run the chunking agent
        response = chunking_agent.run(prompt)
        _00_utils.update_token_counters(response)
        
        # Access structured response
        results = response.content
        
        # Validate the response
        if not results.chunks:
            logger.error(f"Chunking agent did not return any chunks", extra={"icon": "❌"})
            return simple_document_chunking(document_text, document_id)
            
        # Check if chunks have text
        missing_text = [i for i, chunk in enumerate(results.chunks) if "chunk_text" not in chunk or not chunk.get("chunk_text", "")]
        if missing_text:
            logger.warning(f"Chunking agent returned {len(missing_text)} chunks without text. Using fallback.", extra={"icon": "⚠️"})
            return simple_document_chunking(document_text, document_id)
        
        # Convert to DocumentChunk objects
        document_chunks = []
        for i, chunk_data in enumerate(results.chunks):
            # Generate a unique chunk ID
            chunk_id = f"{document_id}_chunk_{i+1:04d}"
            
            # Validate chunk data has required fields
            if not chunk_data.get("chunk_text", ""):
                logger.warning(f"Chunk {i+1} is missing chunk_text. Skipping.", extra={"icon": "⚠️"})
                continue
                
            # Create DocumentChunk object
            document_chunk = DocumentChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                chunk_index=i,
                start_offset=chunk_data.get("start_offset", 0),
                end_offset=chunk_data.get("end_offset", len(chunk_data.get("chunk_text", ""))),
                chunk_text=chunk_data.get("chunk_text", ""),
                token_count=chunk_data.get("token_count", 0)
            )
            
            # Final validation
            if not document_chunk.chunk_text:
                logger.warning(f"Chunk {chunk_id} has empty text after creation. Skipping.", extra={"icon": "⚠️"})
                continue
                
            document_chunks.append(document_chunk)
            
        # If we have no valid chunks, use simple chunking
        if not document_chunks:
            logger.warning(f"No valid chunks were created. Using simple chunking.", extra={"icon": "⚠️"})
            return simple_document_chunking(document_text, document_id)
            
        logger.info(f"Successfully chunked document into {len(document_chunks)} chunks", extra={"icon": "✅"})
        return document_chunks
        
    except Exception as e:
        logger.error(f"Error chunking document: {e}", extra={"icon": "❌"})
        # Fall back to simple chunking if agent fails
        return simple_document_chunking(document_text, document_id)

def simple_document_chunking(document_text: str, document_id: str) -> List[DocumentChunk]:
    """
    Simple fallback chunking by paragraphs and fixed size.
    
    Args:
        document_text: The document text to chunk
        document_id: The ID of the document
        
    Returns:
        List of DocumentChunk objects
    """
    logger.info(f"Using simple chunking as fallback for document {document_id}", extra={"icon": "⚠️"})
    
    # Split by paragraphs
    paragraphs = [p for p in document_text.split("\n\n") if p.strip()]
    
    # Group paragraphs into chunks of approximately TARGET_CHUNK_SIZE tokens
    # This is a very simple approximation - 1 token ≈ 4 characters
    chunks = []
    current_chunk = []
    current_token_count = 0
    
    for para in paragraphs:
        para_token_count = len(para) // 4  # Simple approximation
        
        if current_token_count + para_token_count > TARGET_CHUNK_SIZE * 1.5 and current_chunk:
            # Save current chunk and start a new one
            chunk_text = "\n\n".join(current_chunk)
            chunk_token_count = current_token_count
            
            chunk_id = f"{document_id}_chunk_{len(chunks)+1:04d}"
            start_offset = document_text.find(current_chunk[0]) if current_chunk else 0
            end_offset = start_offset + len(chunk_text)
            
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                chunk_index=len(chunks),
                start_offset=start_offset,
                end_offset=end_offset,
                chunk_text=chunk_text,
                token_count=chunk_token_count
            ))
            
            current_chunk = [para]
            current_token_count = para_token_count
        else:
            current_chunk.append(para)
            current_token_count += para_token_count
    
    # Add the final chunk if it has content
    if current_chunk:
        chunk_text = "\n\n".join(current_chunk)
        chunk_token_count = current_token_count
        
        chunk_id = f"{document_id}_chunk_{len(chunks)+1:04d}"
        start_offset = document_text.find(current_chunk[0]) if current_chunk else 0
        end_offset = start_offset + len(chunk_text)
        
        chunks.append(DocumentChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            chunk_index=len(chunks),
            start_offset=start_offset,
            end_offset=end_offset,
            chunk_text=chunk_text,
            token_count=chunk_token_count
        ))
    
    logger.info(f"Created {len(chunks)} chunks using simple chunking", extra={"icon": "✅"})
    return chunks

# -------------------------------------------------------------------------------------
# Duplicate Detection Functions
# -------------------------------------------------------------------------------------
def generate_chunk_embeddings(chunks: List[DocumentChunk]) -> List[Tuple[DocumentChunk, np.ndarray]]:
    """
    Generate embeddings for a list of document chunks.
    
    Args:
        chunks: List of DocumentChunk objects
        
    Returns:
        List of tuples containing (chunk, embedding)
    """
    # Load the embedding model
    embedding_model = load_embedding_model()
    if not embedding_model:
        logger.error("Failed to load embedding model", extra={"icon": "❌"})
        return []
    
    # Generate embeddings for each chunk
    chunk_embeddings = []
    for chunk in chunks:
        # Prepare text for embedding
        # For e5 models, text needs to be prefixed with "passage: "
        text = f"passage: {chunk.chunk_text}"
        
        try:
            # Generate embedding
            embedding = embedding_model.encode(text)
            chunk_embeddings.append((chunk, embedding))
        except Exception as e:
            logger.error(f"Error generating embedding for chunk {chunk.chunk_id}: {e}", extra={"icon": "❌"})
    
    logger.info(f"Generated embeddings for {len(chunk_embeddings)} chunks", extra={"icon": "✅"})
    return chunk_embeddings

def check_chunk_duplicates(chunk_embeddings: List[Tuple[DocumentChunk, np.ndarray]]) -> Dict[str, Dict]:
    """
    Check for duplicate chunks using vector similarity.
    
    Args:
        chunk_embeddings: List of tuples containing (chunk, embedding)
        
    Returns:
        Dictionary mapping chunk IDs to dictionaries with duplication information
    """
    db = connect_to_lancedb()
    if not db:
        return {}
    
    # Get the document chunks table
    table = get_or_create_chunks_table(db)
    if not table or DOCUMENT_CHUNKS_TABLE not in db.table_names():
        return {}
    
    # Dictionary to store duplicate information
    duplicate_info = {}
    
    # Check each chunk against the database
    for chunk, embedding in chunk_embeddings:
        try:
            # Search for similar chunks in database
            search_results = table.search(embedding, query_type="vector").limit(5).to_df()
            
            if search_results.empty:
                logger.info(f"No similar chunks found for {chunk.chunk_id}", extra={"icon": "✅"})
                continue
            
            # Check for duplicates or near duplicates
            for _, row in search_results.iterrows():
                similarity = 1.0 - row['_distance']  # Convert distance to similarity
                
                if similarity >= EXACT_DUPLICATE_THRESHOLD:
                    # Found an exact duplicate
                    duplicate_info[chunk.chunk_id] = {
                        "is_duplicate": True,
                        "duplicate_of": row.get("chunk_id"),
                        "similarity": similarity,
                        "is_updated": False
                    }
                    logger.info(f"Chunk {chunk.chunk_id} is a duplicate of {row.get('chunk_id')} (similarity: {similarity:.4f})", extra={"icon": "⏭️"})
                    break
                elif similarity >= NEAR_DUPLICATE_THRESHOLD:
                    # Found a near duplicate (updated version)
                    duplicate_info[chunk.chunk_id] = {
                        "is_duplicate": False,
                        "is_updated": True,
                        "previous_chunk_id": row.get("chunk_id"),
                        "similarity": similarity
                    }
                    logger.info(f"Chunk {chunk.chunk_id} is an updated version of {row.get('chunk_id')} (similarity: {similarity:.4f})", extra={"icon": "🔄"})
                    break
            
        except Exception as e:
            logger.error(f"Error checking duplicates for chunk {chunk.chunk_id}: {e}", extra={"icon": "❌"})
    
    logger.info(f"Completed duplicate check: {len(duplicate_info)} duplicates or updates found", extra={"icon": "📊"})
    return duplicate_info

def remove_document_chunks(document_id: str) -> bool:
    """
    Remove all chunks for a document from the database.
    
    Args:
        document_id: The ID of the document
        
    Returns:
        True if successful, False otherwise
    """
    db = connect_to_lancedb()
    if not db:
        return False
    
    # Get the document chunks table
    table = get_or_create_chunks_table(db)
    if not table:
        return False
    
    # Get all chunks for this document to check how many there are
    all_chunks = table.to_pandas()
    if all_chunks.empty:
        logger.info(f"No chunks found for document {document_id}", extra={"icon": "ℹ️"})
        return True
    
    # Filter for this document's chunks
    document_chunks = all_chunks[all_chunks['document_id'] == document_id]
    if document_chunks.empty:
        logger.info(f"No chunks found for document {document_id}", extra={"icon": "ℹ️"})
        return True
    
    # Count chunks before deletion
    chunk_count = len(document_chunks)
    logger.info(f"Found {chunk_count} chunks to delete for document {document_id}", extra={"icon": "🔍"})
    
    # Delete chunks for this document
    db.drop_table(DOCUMENT_CHUNKS_TABLE)
    
    # Recreate the table
    table = db.create_table(DOCUMENT_CHUNKS_TABLE, schema=DocumentChunkDB)
    
    # Add back all chunks except those from the document being deleted
    other_chunks = all_chunks[all_chunks['document_id'] != document_id]
    if not other_chunks.empty:
        records = []
        for _, row in other_chunks.iterrows():
            record = {col: row[col] for col in row.index if col != "id"}
            records.append(record)
        
        table.add(records)
    
    logger.info(f"Successfully removed {chunk_count} chunks for document {document_id}", extra={"icon": "✅"})
    return True

# -------------------------------------------------------------------------------------
# Main Functions
# -------------------------------------------------------------------------------------
def process_document_chunks(document_text: str, document_id: str) -> Tuple[List[DocumentChunk], Dict[str, Dict], bool]:
    """
    Process a document, chunk it, and check for duplicates.
    
    Args:
        document_text: The full document text to process
        document_id: The ID of the document
        
    Returns:
        Tuple of (chunks, duplicate_info, is_update)
    """
    # Check if this document exists already
    previous_chunks = get_previous_document_chunks(document_id)
    is_update = len(previous_chunks) > 0
    
    # Chunk the document
    if is_update:
        logger.info(f"Document {document_id} is an update. Using previous chunks as reference.", extra={"icon": "🔄"})
        chunks = chunk_document(document_text, document_id, previous_chunks)
    else:
        logger.info(f"Document {document_id} is new. Chunking from scratch.", extra={"icon": "🆕"})
        chunks = chunk_document(document_text, document_id)
    
    # Generate embeddings for chunks
    chunk_embeddings = generate_chunk_embeddings(chunks)
    
    # Check for duplicates
    duplicate_info = check_chunk_duplicates(chunk_embeddings)
    
    # Return results
    return chunks, duplicate_info, is_update

def save_document_chunks(chunks: List[DocumentChunk], duplicate_info: Dict[str, Dict], is_update: bool = False) -> bool:
    """
    Save document chunks to LanceDB, handling duplicates and updates.
    
    Args:
        chunks: List of DocumentChunk objects
        duplicate_info: Dictionary mapping chunk IDs to duplication information
        is_update: Whether this is an update to an existing document
        
    Returns:
        True if successful, False otherwise
    """
    if not chunks:
        logger.warning("No chunks to save", extra={"icon": "⚠️"})
        return False
    
    # Get document ID from first chunk
    document_id = chunks[0].document_id
    logger.info(f"Preparing to save {len(chunks)} chunks for document {document_id}", extra={"icon": "🔄"})
    
    # Validate chunk text to ensure it's not empty
    empty_chunks = [c for c in chunks if not c.chunk_text]
    if empty_chunks:
        logger.warning(f"Found {len(empty_chunks)} chunks with empty text. Filling with placeholder text.", extra={"icon": "⚠️"})
        
        # Fill empty chunks with placeholder text
        for chunk in empty_chunks:
            # Generate a placeholder text based on token count
            token_count = chunk.token_count
            estimated_char_count = token_count * 4  # Rough estimate: 1 token ≈ 4 chars
            
            placeholder_text = f"""This is a placeholder for chunk {chunk.chunk_id} with approximately {token_count} tokens.
This placeholder was automatically generated because the original chunk text was empty."""
            
            # Make sure the text is roughly the right length
            while len(placeholder_text) < estimated_char_count:
                placeholder_text += " Additional placeholder content to reach the appropriate length."
                
            # Set the chunk text
            chunk.chunk_text = placeholder_text[:estimated_char_count]
            logger.info(f"Added placeholder text for chunk {chunk.chunk_id} ({len(chunk.chunk_text)} chars)", extra={"icon": "🔧"})
    
    # Get embeddings for chunks
    chunk_embeddings = generate_chunk_embeddings(chunks)
    if not chunk_embeddings:
        logger.error("Failed to generate embeddings for chunks", extra={"icon": "❌"})
        return False
    
    # Connect to LanceDB
    db = connect_to_lancedb()
    if not db:
        logger.error("Failed to connect to LanceDB", extra={"icon": "❌"})
        return False
    
    # If this is an update, first get all existing chunks except for this document
    existing_chunks_data = None
    if is_update and DOCUMENT_CHUNKS_TABLE in db.table_names():
        try:
            logger.info(f"Getting existing chunks to preserve during update", extra={"icon": "🔄"})
            existing_table = db.open_table(DOCUMENT_CHUNKS_TABLE)
            existing_df = existing_table.to_pandas()
            if not existing_df.empty:
                # Filter out chunks from this document
                existing_chunks_data = existing_df[existing_df['document_id'] != document_id]
                logger.info(f"Found {len(existing_chunks_data)} chunks from other documents to preserve", extra={"icon": "✅"})
        except Exception as e:
            logger.warning(f"Error getting existing chunks: {e}. Will only save new chunks.", extra={"icon": "⚠️"})
    
    # Drop the existing table and create a new one
    if DOCUMENT_CHUNKS_TABLE in db.table_names():
        logger.info(f"Dropping existing chunks table", extra={"icon": "🔄"})
        db.drop_table(DOCUMENT_CHUNKS_TABLE)
    
    # Create a fresh table
    logger.info(f"Creating fresh chunks table", extra={"icon": "🔄"})
    table = db.create_table(DOCUMENT_CHUNKS_TABLE, schema=DocumentChunkDB)
    
    # Prepare records for this document
    new_records = []
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    for i, (chunk, embedding) in enumerate(chunk_embeddings):
        # Check if this chunk is a duplicate or update
        dup_info = duplicate_info.get(chunk.chunk_id, {})
        
        # Validate that chunk_text is not empty
        if not chunk.chunk_text:
            logger.error(f"Chunk {chunk.chunk_id} has empty text even after filling placeholders!", extra={"icon": "❌"})
            continue
            
        # Debug log chunk text length
        logger.debug(f"Chunk {chunk.chunk_id} text length: {len(chunk.chunk_text)}", extra={"icon": "🔍"})
        
        # Create record as a dictionary (not using pydantic model directly)
        record = {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
            "start_offset": chunk.start_offset,
            "end_offset": chunk.end_offset,
            "chunk_text": chunk.chunk_text,
            "token_count": chunk.token_count,
            "embedding": embedding.tolist(),
            "is_duplicate": dup_info.get("is_duplicate", False),
            "duplicate_of": dup_info.get("duplicate_of", None),
            "is_updated": dup_info.get("is_updated", False),
            "previous_chunk_id": dup_info.get("previous_chunk_id", None),
            "timestamp": timestamp
        }
        
        new_records.append(record)
    
    # Add the new records
    logger.info(f"Adding {len(new_records)} chunks to fresh table", extra={"icon": "➕"})
    table.add(new_records)
    
    # If we have existing chunks from other documents, add them back
    if existing_chunks_data is not None and not existing_chunks_data.empty:
        logger.info(f"Adding back {len(existing_chunks_data)} chunks from other documents", extra={"icon": "➕"})
        # Convert DataFrame to list of dictionaries
        existing_records = []
        for _, row in existing_chunks_data.iterrows():
            record = {col: row[col] for col in row.index if col != 'id'}
            existing_records.append(record)
        
        table.add(existing_records)
    
    # Log success
    logger.info(f"Successfully saved chunks for document {document_id}", extra={"icon": "✅"})
    return True

def process_and_save_document(document_text: str, document_id: str) -> Tuple[bool, bool, Dict]:
    """
    Process a document, check for duplicates, and save chunks to LanceDB.
    
    Args:
        document_text: The full document text to process
        document_id: The ID of the document
        
    Returns:
        Tuple of (success, is_duplicate, document_info)
    """
    # Process the document
    chunks, duplicate_info, is_update = process_document_chunks(document_text, document_id)
    
    # Check if this document is a complete duplicate
    unique_chunks = [c for c in chunks if c.chunk_id not in duplicate_info or not duplicate_info[c.chunk_id].get("is_duplicate", False)]
    
    # If all chunks are duplicates, the document is a duplicate
    is_duplicate = len(unique_chunks) == 0 and len(chunks) > 0
    
    if is_duplicate:
        logger.info(f"Document {document_id} is a complete duplicate. Not saving to database.", extra={"icon": "⏭️"})
        return True, True, {"document_id": document_id, "duplicate_type": "exact", "duplicate_info": duplicate_info}
    
    # If some chunks are duplicates but not all, it's an update
    has_duplicates = any(duplicate_info.get(c.chunk_id, {}).get("is_duplicate", False) for c in chunks)
    has_updates = any(duplicate_info.get(c.chunk_id, {}).get("is_updated", False) for c in chunks)
    
    document_info = {
        "document_id": document_id,
        "is_update": is_update,
        "has_duplicates": has_duplicates,
        "has_updates": has_updates,
        "total_chunks": len(chunks),
        "duplicate_chunks": sum(1 for c in chunks if duplicate_info.get(c.chunk_id, {}).get("is_duplicate", False)),
        "updated_chunks": sum(1 for c in chunks if duplicate_info.get(c.chunk_id, {}).get("is_updated", False)),
        "new_chunks": sum(1 for c in chunks if c.chunk_id not in duplicate_info)
    }
    
    # Save chunks to LanceDB
    success = save_document_chunks(chunks, duplicate_info, is_update)
    
    if not success:
        logger.error(f"Failed to save chunks for document {document_id}", extra={"icon": "❌"})
        return False, False, document_info
    
    logger.info(f"Successfully saved chunks for document {document_id}", extra={"icon": "✅"})
    return True, False, document_info

if __name__ == "__main__":
    # Simple test function when run directly
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        logger.info(f"Processing document: {file_path}", extra={"icon": "🔍"})
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                document_text = f.read()
            
            document_id = os.path.basename(file_path)
            success, is_duplicate, document_info = process_and_save_document(document_text, document_id)
            
            if success:
                if is_duplicate:
                    logger.info(f"Document is a duplicate. Skipped saving.", extra={"icon": "⏭️"})
                else:
                    logger.info(f"Document processed and saved successfully.", extra={"icon": "✅"})
                    logger.info(f"Document info: {document_info}", extra={"icon": "📊"})
            else:
                logger.error("Failed to process and save document.", extra={"icon": "❌"})
                
        except Exception as e:
            logger.error(f"Error processing document: {e}", extra={"icon": "❌"})
    else:
        logger.info("Usage: python agentic_chunking.py <markdown_file_path>", extra={"icon": "ℹ️"}) 