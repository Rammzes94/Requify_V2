#!/usr/bin/env python3
"""
context_aware_chunking.py

This script implements an enhanced approach to document chunking that:
1. Uses existing chunks from similar documents as context for the LLM
2. Maintains chunk alignment between document versions
3. Intelligently identifies duplicate, updated, and new chunks
4. Uses an Agno agent to handle uncertain cases by prompting the user

The workflow significantly improves chunk alignment between document versions and
reduces manual review by only involving the user when truly necessary.
"""

import os
import sys
import json
import logging
import time
import difflib
import hashlib
import re  # Added for regex pattern matching
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.openai import OpenAIChat
import gc  # Add garbage collection
import torch

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import _00_utils
_00_utils.setup_project_directory()

# Import deduplication module - directly using pre_save_deduplication
from _03_docs_deduplication import pre_save_deduplication as dedup

# Setup logging with script prefix
logger = _00_utils.setup_logging()

# Create a consistent logger with prefix for better visibility


logger = _00_utils.get_logger("Context_Aware_Chunking")

# Utility function for consistent logging
def log_chunk_comparison(chunk_id: str, similar_chunk_id: str, similarity: float, decision: str, reason: str):
    """
    Log the comparison between two chunks with a detailed explanation of the decision.
    
    Args:
        chunk_id: ID of the current chunk
        similar_chunk_id: ID of the similar chunk found
        similarity: Similarity score between the chunks
        decision: Decision made (e.g., "keep_original", "keep_new", "both")
        reason: Reason for the decision
    """
    logger.info(f"🔍 Chunk Comparison: {chunk_id} vs {similar_chunk_id}", extra={"icon": "🔍"})
    logger.info(f"   Similarity: {similarity:.4f}", extra={"icon": "📏"})
    logger.info(f"   Decision: {decision}", extra={"icon": "⚖️"})
    logger.info(f"   Reason: {reason}", extra={"icon": "💬"})
    
    # Log a summary of the decision with an appropriate icon
    if decision == "keep_original":
        logger.info(f"🔄 Using existing chunk {similar_chunk_id} instead of {chunk_id}", extra={"icon": "🔄"})
    elif decision == "keep_new":
        logger.info(f"⬆️ Replacing {similar_chunk_id} with new chunk {chunk_id}", extra={"icon": "⬆️"})
    elif decision == "both":
        logger.info(f"➕ Keeping both {similar_chunk_id} and {chunk_id}", extra={"icon": "➕"})
    else:
        logger.info(f"❓ Undefined decision for {chunk_id} vs {similar_chunk_id}: {decision}", extra={"icon": "❓"})

def log_chunk_deduplication_summary(doc_id: str, total_chunks: int, duplicate_chunks: int, similar_chunks: int, new_chunks: int, is_document_update: bool = False, updated_doc_id: Optional[str] = None):
    """
    Log a summary of chunk deduplication results
    
    Args:
        doc_id: Document ID
        total_chunks: Total number of chunks processed
        duplicate_chunks: Number of exact duplicate chunks found
        similar_chunks: Number of similar/updated chunks found
        new_chunks: Number of new unique chunks
        is_document_update: Whether this is an update to an existing document
        updated_doc_id: ID of the document being updated (if applicable)
    """
    if is_document_update and updated_doc_id:
        update_text = f" (update of {updated_doc_id})"
    else:
        update_text = ""
        
    logger.info(f"📊 CHUNK DEDUPLICATION SUMMARY FOR {doc_id}{update_text}:", extra={"icon": "📊"})
    logger.info(f"  • Total chunks processed: {total_chunks}", extra={"icon": "🔢"})
    logger.info(f"  • Exact duplicates found: {duplicate_chunks}", extra={"icon": "♻️"})
    logger.info(f"  • Similar chunks found: {similar_chunks}", extra={"icon": "🔄"})
    logger.info(f"  • New unique chunks: {new_chunks}", extra={"icon": "🆕"})
    
    if duplicate_chunks > 0 or similar_chunks > 0:
        logger.info(f"  • Duplicate/similar chunk rate: {(duplicate_chunks + similar_chunks) / total_chunks:.1%}", extra={"icon": "📈"})
    
    if is_document_update:
        if new_chunks == 0 and similar_chunks == 0:
            logger.info(f"  • Document content is identical to {updated_doc_id}", extra={"icon": "✓"})
        elif similar_chunks > 0 and new_chunks == 0:
            logger.info(f"  • Document is a minor update of {updated_doc_id}", extra={"icon": "🔁"})
        elif new_chunks > 0:
            logger.info(f"  • Document is a major update of {updated_doc_id} with new content", extra={"icon": "⬆️"})

# Load environment variables
load_dotenv()

# Constants
MAX_CHAR_SIZE = 900  # Maximum allowed character size
TARGET_CHAR_SIZE = 700  # Target character size per chunk
MAX_SECTION_SIZE = 30000  # Maximum section size for processing with LLM
MAX_RETRIES = 2  # Maximum number of retries for LLM calls
SIMILARITY_THRESHOLD = 0.82  # Updated to match deduplication module
DUPLICATE_THRESHOLD = 0.995  # High threshold for automatic duplicates without LLM

OUTPUT_DIR_BASE = "_03_output"
LANCEDB_SUBDIR_NAME = "lancedb"
CHUNKS_TABLE_NAME = "document_chunks"

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct" # Use existing embedding model from reference chunks
EMBEDDING_DEVICE = "cpu" # Force CPU usage to avoid MPS memory issues
EMBEDDING_MAX_SEQ_LENGTH = 256 # Reduce sequence length to save memory
EMBEDDING_BATCH_SIZE = 8 # Control batch size for memory management

# Set to True to enable more detailed console output 
VERBOSE_CHUNKING_OUTPUT = os.environ.get("VERBOSE_CHUNKING_OUTPUT", "True").lower() == "true"

# Get OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI models
llm = OpenAIChat(id="gpt-4.1-mini", api_key=api_key)
decision_llm = OpenAIChat(id="gpt-4.1-mini", api_key=api_key)  # Higher capability model for decisions

# Pydantic models for LLM output validation
class ChunksOutputModel(BaseModel):
    chunks: List[str] = Field(
        ..., description="List of text chunks that preserve semantic coherence."
    )

class ChunkDecisionModel(BaseModel):
    decision: str = Field(
        ..., 
        description="Decision about the chunk: 'keep_new' (new chunk contains meaningful new information), 'keep_old' (old chunk is better or new chunk adds no value), or 'need_user_input' (truly uncertain which to keep)"
    )
    reason: str = Field(
        ..., 
        description="Clear reasoning explaining your decision, why one version is better or why user input is needed"
    )
    differences: Optional[List[str]] = Field(
        None, 
        description="Key differences between the chunks that influenced the decision"
    )

# ------------------------------------------------------------------------------
# Retrieve Existing Chunks
# ------------------------------------------------------------------------------
def get_similar_document_chunks(document_id: str, similar_doc_id: str) -> List[Dict[str, Any]]:
    """
    Retrieve all chunks from a similar document to provide context for chunking.
    
    Args:
        document_id: ID of the new document being processed
        similar_doc_id: ID of the similar existing document
        
    Returns:
        List of chunk records from the similar document
    """
    try:
        # Connect to LanceDB
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
        lancedb_path = os.path.join(project_root, OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME)
        
        db = dedup.connect_to_lancedb(lancedb_path)
        if not db or CHUNKS_TABLE_NAME not in db.table_names():
            logger.warning(f"No existing chunks table found for document {similar_doc_id}", extra={"icon": "⚠️"})
            return []
            
        # Query the database for chunks from the similar document
        chunks_table = db.open_table(CHUNKS_TABLE_NAME)
        df = chunks_table.to_pandas()
        
        # Filter for chunks belonging to the similar document
        similar_chunks = df[df['document_id'] == similar_doc_id].to_dict('records')
        
        if not similar_chunks:
            logger.warning(f"No chunks found for similar document {similar_doc_id}", extra={"icon": "⚠️"})
            return []
            
        logger.info(f"Retrieved {len(similar_chunks)} chunks from similar document {similar_doc_id}", extra={"icon": "✅"})
        return similar_chunks
        
    except Exception as e:
        logger.error(f"Error retrieving similar document chunks: {e}", extra={"icon": "❌"})
        return []

# ------------------------------------------------------------------------------
# Context-Aware Chunking
# ------------------------------------------------------------------------------
def context_aware_chunking(
    text: str, 
    document_id: str, 
    similar_chunks: List[Dict[str, Any]]
) -> List[str]:
    """
    Chunk text while maintaining alignment with existing chunks.
    
    Args:
        text: Text content to chunk
        document_id: ID of the document being processed
        similar_chunks: Existing chunks from a similar document
        
    Returns:
        List of aligned chunks
    """
    if not text.strip():
        logger.info("Empty document text. No chunks to create.", extra={"icon": "ℹ️"})
        return []
        
    if not similar_chunks:
        logger.info("No reference chunks provided. Using standard chunking.", extra={"icon": "ℹ️"})
        from _02_parsing.integrated_chunking import chunk_markdown
        return chunk_markdown(text)
    
    # Sort reference chunks by index for proper ordering
    similar_chunks.sort(key=lambda x: x.get('chunk_index', 0))
    
    # Create context for the LLM with the reference chunks
    reference_chunks_text = "\n\n".join([
        f"Chunk {i+1}:\n{chunk.get('chunk_text', '')}" 
        for i, chunk in enumerate(similar_chunks)
    ])
    
    # Enhanced prompt for context-aware chunking
    context_aware_prompt = """
    Your task is to chunk the NEW DOCUMENT TEXT in a way that aligns with the REFERENCE CHUNKS.
    
    Follow these rules carefully:
    1. Create chunks that maintain the same semantic boundaries as the reference chunks.
    2. If content is identical or very similar, create chunks that match the reference exactly.
    3. If sections are reordered but content is the same, try to maintain the original chunk boundaries.
    4. When content differs significantly, create appropriate new chunks.
    5. Each chunk should be roughly {target_size} characters (max {max_size} characters).
    6. Preserve coherent content within each chunk.
    7. For reordered sections, try to keep the same chunks as the reference document, even if in different order.
    
    CRITICAL: The goal is to maximize chunk alignment for comparison, even if sections are moved around.
    A successful chunking should make it easy to identify which parts are duplicates vs. which are new/changed.
    
    Format your response as: {{"chunks": ["chunk1", "chunk2", ...]}}
    """
    
    logger.info(f"Performing context-aware chunking with {len(similar_chunks)} reference chunks", extra={"icon": "🔄"})
    
    agent = Agent(
        model=llm,
        markdown=True,
        debug_mode=False,
        response_model=ChunksOutputModel,
        description=context_aware_prompt.format(
            target_size=TARGET_CHAR_SIZE, 
            max_size=MAX_CHAR_SIZE
        ),
        use_json_mode=True
    )
    
    # Prepare prompt content with both reference chunks and new text
    prompt_content = f"""
    # REFERENCE CHUNKS (from similar document):
    {reference_chunks_text}
    
    # NEW DOCUMENT TEXT (to be chunked):
    {text}
    """
    
    retry_count = 0
    chunks = []
    
    while retry_count <= MAX_RETRIES:
        try:
            response = agent.run(prompt_content)
            _00_utils.update_token_counters(response)
            
            data = response.content
            
            if isinstance(data, ChunksOutputModel):
                chunks = data.chunks
                break
            elif isinstance(data, dict) and 'chunks' in data:
                chunks = data['chunks']
                break
            elif isinstance(data, str):
                try:
                    parsed_data = json.loads(data)
                    if 'chunks' in parsed_data:
                        chunks = parsed_data['chunks']
                        break
                except json.JSONDecodeError:
                    logger.error(f"LLM returned invalid JSON string: {data[:100]}...", extra={"icon": "❌"})
            
            # If we haven't broken out of the loop, try again
            retry_count += 1
            if retry_count <= MAX_RETRIES:
                delay = 1 * (2 ** (retry_count - 1))
                logger.info(f"Waiting {delay} seconds before retry...", extra={"icon": "⏱️"})
                time.sleep(delay)
            else:
                logger.error(f"All {MAX_RETRIES} retries failed. Falling back to standard chunking.", extra={"icon": "❌"})
                from _02_parsing.integrated_chunking import chunk_markdown
                return chunk_markdown(text)
                
        except Exception as e:
            logger.error(f"Error during context-aware chunking: {e}", extra={"icon": "❌"})
            retry_count += 1
            
            if retry_count <= MAX_RETRIES:
                delay = 1 * (2 ** (retry_count - 1))
                logger.info(f"Waiting {delay} seconds before retry...", extra={"icon": "⏱️"})
                time.sleep(delay)
            else:
                logger.error(f"All {MAX_RETRIES} retries failed. Falling back to standard chunking.", extra={"icon": "❌"})
                from _02_parsing.integrated_chunking import chunk_markdown
                return chunk_markdown(text)
    
    # Filter out empty chunks
    chunks = [chunk for chunk in chunks if chunk.strip()]
    logger.info(f"Context-aware chunking produced {len(chunks)} chunks", extra={"icon": "✅"})
    
    return chunks

# ------------------------------------------------------------------------------
# Chunk Comparison & Decision Logic
# ------------------------------------------------------------------------------
def generate_chunk_hash(text: str) -> str:
    """Generate a hash for a chunk of text."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def compute_chunk_similarity(chunk1: str, chunk2: str) -> float:
    """Compute text similarity between two chunks using difflib."""
    return difflib.SequenceMatcher(None, chunk1, chunk2).ratio()

def evaluate_chunk_pair(new_chunk: str, old_chunk: str) -> Dict[str, Any]:
    """
    Evaluate a pair of chunks to determine if they're duplicates, similar, or different.
    
    Returns decision information with similarity score and decision.
    """
    # First calculate simple text similarity
    similarity = compute_chunk_similarity(new_chunk, old_chunk)
    
    # Create a decision info dictionary
    decision_info = {
        "similarity": similarity,
        "decision": "",
        "reason": "",
        "differences": []
    }
    
    # Automatic decision for high similarity (duplicate)
    if similarity >= DUPLICATE_THRESHOLD:
        decision_info["decision"] = "keep_old"
        decision_info["reason"] = f"Chunks are nearly identical (similarity: {similarity:.4f})"
        return decision_info
    
    # For medium similarity, use LLM to decide
    if similarity >= SIMILARITY_THRESHOLD:
        # Set up the decision prompt
        decision_prompt = """
        Compare these two document chunks and determine the best action:
        
        OLD CHUNK:
        '''
        {old_chunk}
        '''
        
        NEW CHUNK:
        '''
        {new_chunk}
        '''
        
        Comparison rules:
        1. If chunks contain identical content (even if formatting/wording differs slightly), keep the old chunk
        2. If new chunk contains meaningful new information or corrections, keep the new chunk
        3. If truly uncertain which is better, request user input
        
        Your objective is to maximize information quality while minimizing unnecessary redundancy.
        """
        
        # Create a Pydantic model for the expected output structure
        decision_agent = Agent(
            model=decision_llm,
            markdown=True,
            debug_mode=False,
            response_model=ChunkDecisionModel,
            description=decision_prompt.format(old_chunk=old_chunk, new_chunk=new_chunk),
            use_json_mode=True
        )
        
        try:
            response = decision_agent.run("")
            _00_utils.update_token_counters(response)
            
            data = response.content
            decision_info["decision"] = data.decision
            decision_info["reason"] = data.reason
            if data.differences:
                decision_info["differences"] = data.differences
                
            logger.info(
                f"LLM determined chunks with {similarity:.4f} similarity should be: {data.decision}",
                extra={"icon": "🧠"}
            )
            
        except Exception as e:
            logger.error(f"Error using LLM for chunk comparison: {e}", extra={"icon": "❌"})
            # Default to keeping the new chunk if LLM fails
            decision_info["decision"] = "keep_new"
            decision_info["reason"] = "LLM comparison failed, defaulting to keeping new chunk"
            
    else:
        # Low similarity - keep new chunk
        decision_info["decision"] = "keep_new"
        decision_info["reason"] = f"Chunks are significantly different (similarity: {similarity:.4f})"
        
    return decision_info

# ------------------------------------------------------------------------------
# User Interaction with Agno Agent
# ------------------------------------------------------------------------------
def chunk_comparison_tool(
    new_chunk: str, 
    old_chunk: str, 
    new_doc_id: str, 
    old_doc_id: str,
    reason: str, 
    differences: List[str]
) -> str:
    """
    Compare two chunks and get user decision on which to keep.
    
    Args:
        new_chunk: Text of the new chunk
        old_chunk: Text of the existing chunk
        new_doc_id: ID of the new document
        old_doc_id: ID of the old document
        reason: Reason for needing user input
        differences: Key differences between chunks
        
    Returns:
        User decision: 'keep_new', 'keep_old'
    """
    logger.info("\n" + "="*80, extra={"icon": "🔍"})
    logger.info("CHUNK COMPARISON NEEDED", extra={"icon": "🔍"})
    logger.info("="*80, extra={"icon": "🔍"})
    
    logger.info(f"\nREASON: {reason}", extra={"icon": "ℹ️"})
    
    logger.info("\nKEY DIFFERENCES:", extra={"icon": "📊"})
    for i, diff in enumerate(differences, 1):
        logger.info(f"  {i}. {diff}", extra={"icon": "🔄"})
    
    logger.info(f"\nOLD DOCUMENT: {old_doc_id}", extra={"icon": "📜"})
    logger.info("-" * 40, extra={"icon": "📜"})
    logger.info(old_chunk, extra={"icon": "📜"})
    logger.info("-" * 40, extra={"icon": "📜"})
    
    logger.info(f"\nNEW DOCUMENT: {new_doc_id}", extra={"icon": "📄"})
    logger.info("-" * 40, extra={"icon": "📄"})
    logger.info(new_chunk, extra={"icon": "📄"})
    logger.info("-" * 40, extra={"icon": "📄"})
    
    while True:
        choice = input("\nChoose which chunk to use (1=old, 2=new): ")
        if choice in ('1', '2'):
            return "keep_old" if choice == '1' else "keep_new"
        logger.warning("Invalid choice. Please enter 1 or 2.", extra={"icon": "⚠️"})

def prompt_user_for_chunk_decision(
    new_chunk: str, 
    old_chunk: Dict[str, Any],
    new_doc_id: str,
    decision_info: Dict[str, Any]
) -> str:
    """
    Prompt the user for a decision between two chunks.
    
    Args:
        new_chunk: Text of the new chunk
        old_chunk: Dict containing data about the old chunk
        new_doc_id: ID of the new document
        decision_info: Dictionary with decision reasoning and differences
        
    Returns:
        User decision: 'keep_new', 'keep_old'
    """
    # Check if we're in testing mode and should auto-select new
    auto_select = os.environ.get("REQUIFY_AUTO_SELECT_NEW", "false").lower() == "true"
    
    if auto_select:
        logger.info(f"Auto-selecting 'keep_new' for testing", extra={"icon": "🤖"})
        return "keep_new"
    
    # For normal operation, prompt the user
    logger.info(f"Prompting user for chunk decision", extra={"icon": "👤"})
    
    old_doc_id = old_chunk.get('document_id', 'unknown')
    old_chunk_text = old_chunk.get('chunk_text', '')
    reason = decision_info.get('reason', 'Reason not provided')
    differences = decision_info.get('differences', [])
    
    logger.info("\n" + "="*80, extra={"icon": "🔍"})
    logger.info("CHUNK COMPARISON NEEDED", extra={"icon": "🔍"})
    logger.info("="*80, extra={"icon": "🔍"})
    
    logger.info(f"\nREASON: {reason}", extra={"icon": "ℹ️"})
    
    logger.info("\nKEY DIFFERENCES:", extra={"icon": "📊"})
    for i, diff in enumerate(differences, 1):
        logger.info(f"  {i}. {diff}", extra={"icon": "🔄"})
    
    logger.info(f"\nOLD DOCUMENT: {old_doc_id}", extra={"icon": "📜"})
    logger.info("-" * 40, extra={"icon": "📜"})
    logger.info(old_chunk_text, extra={"icon": "📜"})
    logger.info("-" * 40, extra={"icon": "📜"})
    
    logger.info(f"\nNEW DOCUMENT: {new_doc_id}", extra={"icon": "📄"})
    logger.info("-" * 40, extra={"icon": "📄"})
    logger.info(new_chunk, extra={"icon": "📄"})
    logger.info("-" * 40, extra={"icon": "📄"})
    
    while True:
        choice = input("\nChoose which chunk to use (1=old, 2=new): ")
        if choice in ('1', '2'):
            return "keep_old" if choice == '1' else "keep_new"
        logger.warning("Invalid choice. Please enter 1 or 2.", extra={"icon": "⚠️"})

# ------------------------------------------------------------------------------
# Main Processing Logic
# ------------------------------------------------------------------------------
def process_document_with_context(
    document_text: str,
    document_id: str,
    similar_doc_id: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Process a document with context from a similar document for better chunking.
    
    Args:
        document_text: Text content of the document
        document_id: Identifier for the new document
        similar_doc_id: Identifier for a similar document to use as context
        
    Returns:
        List of processed chunks with metadata
        Dictionary mapping new chunk IDs to replaced chunk IDs
    """
    start_time = time.time()
    
    if not document_text.strip():
        logger.info("Empty document text. No chunks to create.", extra={"icon": "ℹ️"})
        return [], {}
    
    # Initialize tracking variables
    replaced_chunks = {}  # Maps new chunk ID -> old chunk ID for replacements
    duplicate_count = 0
    similar_count = 0
    new_count = 0
    
    logger.info(f"Processing document: {document_id}", extra={"icon": "🔄"})
    
    # If a similar document is provided, retrieve its chunks for context
    similar_chunks = []
    if similar_doc_id:
        logger.info(f"Using chunks from similar document {similar_doc_id} as context", extra={"icon": "🔄"})
        similar_chunks = get_similar_document_chunks(document_id, similar_doc_id)
        # Log that we're using context-aware chunking
        if similar_chunks:
            logger.info(
                f"Performing context-aware chunking with {len(similar_chunks)} reference chunks",
                extra={"icon": "🧩"}
            )
    
    # If no similar document or no chunks found, fall back to regular chunking
    chunks = []
    if not similar_chunks:
        # We don't have context, use standard chunking
        logger.info("No context chunks available, using standard chunking", extra={"icon": "ℹ️"})
        from _02_parsing.integrated_chunking import chunk_markdown
        raw_chunks = chunk_markdown(document_text)
    else:
        # Use context-aware chunking
        raw_chunks = context_aware_chunking(document_text, document_id, similar_chunks)

    # Check if chunking succeeded
    if not raw_chunks:
        logger.error("Chunking failed to produce any chunks", extra={"icon": "❌"})
        return [], {}
        
    logger.info(f"Created {len(raw_chunks)} initial chunks", extra={"icon": "✅"})
    
    # Create a list to hold processed chunks with full metadata
    processed_chunks = []
    current_offset = 0
    
    # Embedding model for chunk embeddings
    try:
        import torch
        from torch import Tensor
        from transformers import AutoTokenizer, AutoModel
        
        # Load the embedding model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
        model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
        
        # Force to CPU to avoid MPS memory issues
        if EMBEDDING_DEVICE == "cpu":
            model = model.to(EMBEDDING_DEVICE)
        
        logger.info(f"Loaded embedding model: {EMBEDDING_MODEL_NAME}", extra={"icon": "✅"})
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}", extra={"icon": "❌"})
        return [], {}
    
    # Process each chunk
    for i, chunk_text in enumerate(raw_chunks):
        # Find the actual position of this chunk in the document
        chunk_start = document_text.find(chunk_text, current_offset)
        if chunk_start == -1:
            # If not found from current offset, search from beginning (though this indicates an issue)
            chunk_start = document_text.find(chunk_text)
            if chunk_start == -1:
                logger.warning(
                    f"Could not find exact position for chunk {i+1}, using approximate position",
                    extra={"icon": "⚠️"}
                )
                chunk_start = current_offset
        
        chunk_end = chunk_start + len(chunk_text)
        current_offset = chunk_end
        
        # Generate chunk hash for deduplication
        chunk_hash = generate_chunk_hash(chunk_text)
        
        # Calculate token count
        token_count = len(tokenizer.encode(chunk_text))
        
        # Generate unique chunk ID
        chunk_id = f"{document_id}_chunk_{i+1:04d}"
        
        # Generate embedding for the chunk
        try:
            inputs = tokenizer(
                chunk_text, 
                padding=True, 
                truncation=True, 
                max_length=EMBEDDING_MAX_SEQ_LENGTH, 
                return_tensors="pt"
            )
            with torch.no_grad():
                embedding = model(**inputs).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                
            # Normalize the embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        except Exception as e:
            logger.error(f"Error generating embedding for chunk {i+1}: {e}", extra={"icon": "❌"})
            # Create a zero embedding as fallback
            embedding = np.zeros(1024)
        
        # Check for duplicates against reference chunks
        is_duplicate = False
        duplicate_of = ""
        is_updated = False
        previous_chunk_id = ""
        
        # Store the chunk data
        timestamp = _00_utils.generate_timestamp()
        
        # Compare against all reference chunks from the similar document
        if similar_chunks:
            best_similarity = 0.0
            best_chunk = None
            
            for ref_chunk in similar_chunks:
                # Compare embeddings
                ref_embedding = np.array(ref_chunk.get('embedding', []))
                if len(ref_embedding) > 0:
                    similarity = dedup.calculate_cosine_similarity(embedding, ref_embedding)
                    
                    # Log the comparison with enhanced logging
                    log_chunk_comparison(
                        chunk_id=chunk_id,
                        similar_chunk_id=ref_chunk.get('chunk_id', 'unknown'),
                        similarity=similarity,
                        decision="evaluating",
                        reason="Comparing chunks for similarity"
                    )
                    
                    # Track best match
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_chunk = ref_chunk
                    
                    # Check for duplicate or similar chunk
                    if similarity >= DUPLICATE_THRESHOLD:
                        # This is a duplicate chunk
                        is_duplicate = True
                        duplicate_of = ref_chunk.get('chunk_id', '')
                        duplicate_count += 1
                        break
                    elif similarity >= SIMILARITY_THRESHOLD:
                        # This might be an updated version of the chunk
                        decision_info = evaluate_chunk_pair(chunk_text, ref_chunk.get('chunk_text', ''))
                        
                        if decision_info["decision"] == "keep_old":
                            # Keep old chunk (it's a duplicate)
                            is_duplicate = True
                            duplicate_of = ref_chunk.get('chunk_id', '')
                            duplicate_count += 1
                            break
                        elif decision_info["decision"] == "keep_new":
                            # Keep new chunk but track relationship to old chunk
                            is_updated = True
                            previous_chunk_id = ref_chunk.get('chunk_id', '')
                            similar_count += 1
                            
                            # Record replacement
                            replaced_chunks[chunk_id] = previous_chunk_id
                            
                            # Log the decision
                            logger.info(
                                f"Chunk {chunk_id} replaces {previous_chunk_id} - Reason: {decision_info['reason']}",
                                extra={"icon": "🔄"}
                            )
                            break
            
            # If we didn't find a duplicate or similar chunk, it's a new chunk
            if not is_duplicate and not is_updated:
                new_count += 1
        else:
            # Without reference chunks, all chunks are new
            new_count += 1
            
        # Skip duplicate chunks in the output (we'll reference the original)
        if is_duplicate:
            logger.info(
                f"Skipping duplicate chunk {chunk_id}, referencing {duplicate_of} instead",
                extra={"icon": "♻️"}
            )
            continue
                
        # Create the chunk data structure
        chunk_data = {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "chunk_index": i,
            "start_offset": chunk_start,
            "end_offset": chunk_end,
            "chunk_text": chunk_text,
            "token_count": token_count,
            "embedding": embedding.tolist(),
            "chunk_hash": chunk_hash,
            "is_duplicate": is_duplicate,
            "duplicate_of": duplicate_of,
            "is_updated": is_updated,
            "previous_chunk_id": previous_chunk_id,
            "timestamp": timestamp,
            "aligned_with_chunk_id": "",
            "aligned_with_document_id": similar_doc_id or ""
        }
        
        processed_chunks.append(chunk_data)
    
    # Log a summary of the chunking results with enhanced logging
    log_chunk_deduplication_summary(
        doc_id=document_id,
        total_chunks=len(raw_chunks),
        duplicate_chunks=duplicate_count,
        similar_chunks=similar_count,
        new_chunks=new_count,
        is_document_update=(similar_doc_id is not None),
        updated_doc_id=similar_doc_id
    )
    
    # Clean up resources
    if 'model' in locals():
        del model
    if 'tokenizer' in locals():
        del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    duration = time.time() - start_time
    logger.info(
        f"Processed {len(processed_chunks)} chunks in {duration:.2f}s "
        f"({duplicate_count} duplicates, {similar_count} updated, {new_count} new)",
        extra={"icon": "✅"}
    )
    
    return processed_chunks, replaced_chunks

def save_chunks_to_db(chunks: List[Dict[str, Any]], replaced_chunks: Optional[Dict[str, str]] = None) -> bool:
    """
    Save processed chunks to the LanceDB chunks table.
    
    Args:
        chunks: List of chunk records to save
        replaced_chunks: Dictionary mapping original chunk IDs to new chunk IDs (for marking chunks as replaced)
        
    Returns:
        True if successful, False otherwise
    """
    if not chunks:
        logger.warning("No chunks to save to database", extra={"icon": "⚠️"})
        return True
        
    try:
        # Connect to LanceDB
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
        lancedb_path = os.path.join(project_root, OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME)
        
        db = dedup.connect_to_lancedb(lancedb_path)
        
        # Check if table exists
        table_exists = CHUNKS_TABLE_NAME in db.table_names()
        
        if table_exists:
            logger.info(f"Opened existing table: {CHUNKS_TABLE_NAME}", extra={"icon": "✅"})
            chunks_table = db.open_table(CHUNKS_TABLE_NAME)
            
            # Check table schema for required columns
            table_schema = chunks_table.schema
            existing_columns = {field.name for field in table_schema}
            required_columns = {
                'is_duplicate_marker', 
                'is_replaced', 
                'replaced_by'
            }
            
            missing_columns = required_columns - existing_columns
            
            # Get existing chunks data
            existing_df = chunks_table.to_pandas()
            
            # Add missing columns if needed
            if missing_columns:
                logger.info(f"Adding missing columns: {missing_columns}", extra={"icon": "🔄"})
                # Add required columns with default values
                for col in missing_columns:
                    if col in ['is_duplicate_marker', 'is_replaced']:
                        existing_df[col] = False
                    else:
                        existing_df[col] = ""
            
            # Prepare replacement updates - only query for chunks that need updates
            replacement_updates = []
            if replaced_chunks and not existing_df.empty:
                for old_chunk_id, new_chunk_id in replaced_chunks.items():
                    mask = existing_df['chunk_id'] == old_chunk_id
                    if any(mask):
                        # Get the row indexes that need updating
                        for idx in existing_df[mask].index:
                            # Create a modified copy of the existing row
                            updated_row = existing_df.loc[idx].copy()
                            updated_row['is_replaced'] = True
                            updated_row['replaced_by'] = new_chunk_id
                            replacement_updates.append(updated_row)
                        
                        logger.info(f"Marked chunk {old_chunk_id} as replaced by {new_chunk_id}", extra={"icon": "✅"})
                    else:
                        logger.warning(f"Could not find chunk {old_chunk_id} to mark as replaced", extra={"icon": "⚠️"})
            
            # Prepare new chunks dataframe
            if chunks:
                new_chunks_df = pd.DataFrame(chunks)
                
                # Ensure required columns exist in new chunks
                for col in required_columns:
                    if col not in new_chunks_df.columns:
                        if col in ['is_duplicate_marker', 'is_replaced']:
                            new_chunks_df[col] = False
                        else:
                            new_chunks_df[col] = ""
            
            # Combine existing chunks that need updates with new chunks
            if replacement_updates or missing_columns:
                # Convert replacement updates to DataFrame if any exist
                if replacement_updates:
                    updates_df = pd.DataFrame(replacement_updates)
                    
                    # Remove rows that will be updated (replaced chunks)
                    if replaced_chunks:
                        existing_df = existing_df[~existing_df['chunk_id'].isin(replaced_chunks.keys())]
                    
                    # Combine existing (non-replaced) + updates + new chunks
                    combined_data = pd.concat([existing_df, updates_df, new_chunks_df], ignore_index=True)
                else:
                    # No replacements, just schema updates and adding new chunks
                    combined_data = pd.concat([existing_df, new_chunks_df], ignore_index=True)
                
                # Drop and recreate table with all data
                db.drop_table(CHUNKS_TABLE_NAME)
                chunks_table = db.create_table(CHUNKS_TABLE_NAME, data=combined_data)
                
                if replacement_updates:
                    logger.info(f"Updated {len(replacement_updates)} chunks, added {len(new_chunks_df)} new chunks", extra={"icon": "✅"})
                else:
                    logger.info(f"Updated schema with {len(missing_columns)} new columns, added {len(new_chunks_df)} new chunks", extra={"icon": "✅"})
            else:
                # No replacements or schema updates, just add new chunks
                chunks_table.add(new_chunks_df)
                logger.info(f"Added {len(new_chunks_df)} new chunks", extra={"icon": "✅"})
        else:
            # First chunk added - create new table
            if chunks:
                logger.info(f"Creating new table: {CHUNKS_TABLE_NAME}", extra={"icon": "✅"})
                data = pd.DataFrame(chunks)
                
                # Ensure all required columns exist
                for col in ['is_duplicate_marker', 'is_replaced', 'replaced_by']:
                    if col not in data.columns:
                        if col in ['is_duplicate_marker', 'is_replaced']:
                            data[col] = False
                        else:
                            data[col] = ""
                    
                chunks_table = db.create_table(CHUNKS_TABLE_NAME, data=data)
                return True
            else:
                logger.warning("No chunks to save, table not created", extra={"icon": "⚠️"})
                return True
        
        # Create vector index if needed
        dedup.ensure_index(db, CHUNKS_TABLE_NAME)
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving chunks to database: {str(e)}", extra={"icon": "❌"})
        return False

def cleanup_memory():
    """
    Release memory after processing to prevent memory leaks.
    Especially important when running multiple document processing tasks in sequence.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # If using MPS (Apple Silicon), clear that cache too
    if hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
    
    logger.info("Memory cleaned up", extra={"icon": "🧹"})

# ------------------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------------------
def process_document(
    document_text: str, 
    document_id: str, 
    similar_document_id: Optional[str] = None
) -> bool:
    """
    Main entry point for processing a document.
    
    Args:
        document_text: Full text of the document
        document_id: ID of the document being processed
        similar_document_id: ID of a similar document (if known)
        
    Returns:
        True if processing was successful, False otherwise
    """
    logger.info(f"Processing document: {document_id}", extra={"icon": "🚀"})
    
    try:
        # Step 1: Generate chunks with context
        chunks, replaced_chunks = process_document_with_context(document_text, document_id, similar_document_id)
        
        if not chunks:
            logger.warning(f"No chunks were generated for document {document_id}", extra={"icon": "⚠️"})
            return False
        
        # Step 2: Save chunks to database
        if chunks:
            # Pass replaced_chunks along with chunks to handle both in one operation
            success = save_chunks_to_db(chunks, replaced_chunks)
        else:
            # If there are no chunks to save because all were duplicates, 
            # consider this a success (proper deduplication)
            duplicates_found = 0  # This would need to be passed from process_document_with_context
            if duplicates_found > 0:
                logger.info(f"All chunks were detected as duplicates - no new chunks to save", extra={"icon": "✅"})
                success = True
            else:
                logger.warning(f"No chunks were generated for document {document_id}", extra={"icon": "⚠️"})
                success = False
        
        return success
    finally:
        # Clean up memory regardless of success or failure
        cleanup_memory()

def main():
    """
    Test function for context-aware chunking.
    """
    test_file = os.path.join("_01_input", "raw", "sample_md.md")
    if not os.path.exists(test_file):
        # Create a simple test file
        test_content = """
        # Test Document
        
        ## Section 1
        This is a test document for context-aware chunking.
        It contains multiple sections with different content.
        
        ## Section 2
        The context-aware chunking system should:
        * Align chunks with existing documents
        * Detect meaningful changes
        * Prompt users only when necessary
        
        ## Section 3
        This is the final section of our test document.
        It shows how the system handles different content structures.
        """
        
        os.makedirs(os.path.dirname(test_file), exist_ok=True)
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
    
    with open(test_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    process_document(content, "test_document")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 

def create_chunk_groups(chunk_text: str, existing_chunks: List[Dict], doc_id: str, sim_threshold: float = 0.75) -> dict:
    """
    Groups new chunks with similar existing chunks.
    
    Args:
        chunk_text: The document text to be chunked
        existing_chunks: List of existing chunks from a similar document
        doc_id: Document ID for the new document
        sim_threshold: Threshold for similarity to consider chunks as matching
    
    Returns:
        Dictionary with grouped chunks
    """ 