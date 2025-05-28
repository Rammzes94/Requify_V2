#!/usr/bin/env python3
"""
agentic_chunking.py

This script implements a comprehensive approach to document chunking that:
1. Uses existing chunks from similar documents as context for the LLM (from context_aware_chunking)
2. Handles large documents efficiently by splitting into manageable sections (from integrated_chunking)
3. Recursively processes oversized chunks to ensure proper sizing (from integrated_chunking)
4. Maintains chunk alignment between document versions (from context_aware_chunking)
5. Intelligently identifies duplicate, updated, and new chunks (from context_aware_chunking)
6. Uses improved boundary detection for more coherent chunks (from agentic_chunking)

The script combines the best features from all three previous chunking implementations
while maintaining a clean, efficient, and maintainable structure.
"""

import os
import sys
import json
import logging
import time
import difflib
import hashlib
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.models.groq import Groq
import gc
import torch

# --- Start of sys.path modification ---
# Get the absolute path of the directory containing the current script (src/_02_parsing)
_current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path of the project root (parent of src)
_project_root = os.path.abspath(os.path.join(_current_script_dir, '..', '..'))

# Add the project root to sys.path if it's not already there
# This allows imports like `from src.module import ...`
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
# --- End of sys.path modification ---

# Now project-specific imports can be done
from src.utils import setup_logging, get_logger, update_token_counters, get_token_usage, print_token_usage, reset_token_counters, setup_project_directory, generate_timestamp
from src import config as project_config # Use an alias to avoid confusion with other 'config' variables

# Call setup_project_directory() early.
setup_project_directory()

# Load environment variables
load_dotenv()

# Import deduplication module
from src._03_docs_deduplication import pre_save_deduplication as dedup

# Setup logging with script prefix
logger = get_logger("Agentic_Chunking")

# Constants - now imported from project_config
MAX_CHAR_SIZE = project_config.MAX_CHAR_SIZE  # Maximum allowed character size for a chunk
TARGET_CHAR_SIZE = project_config.TARGET_CHAR_SIZE  # Target character size per chunk
MAX_SECTION_SIZE = project_config.MAX_SECTION_SIZE  # Maximum section size for processing with LLM
MAX_RETRIES = project_config.MAX_RETRIES  # Maximum number of retries for API calls
INITIAL_RETRY_DELAY = project_config.INITIAL_RETRY_DELAY  # Initial delay (in seconds) before retrying
SIMILARITY_THRESHOLD = project_config.SIMILARITY_THRESHOLD  # Threshold for similar chunks
DUPLICATE_THRESHOLD = project_config.DUPLICATE_THRESHOLD  # High threshold for automatic duplicates without LLM

# LanceDB settings - now imported from project_config
OUTPUT_DIR_BASE = project_config.OUTPUT_DIR_BASE
LANCEDB_SUBDIR_NAME = project_config.LANCEDB_SUBDIR_NAME
CHUNK_DIAGNOSTICS_DIR = project_config.CHUNK_DIAGNOSTICS_DIR

# Use embedding model settings from project_config
EMBEDDING_MODEL_NAME = project_config.EMBEDDING_MODEL_NAME
EMBEDDING_DEVICE = project_config.EMBEDDING_DEVICE
EMBEDDING_MAX_SEQ_LENGTH = project_config.EMBEDDING_MAX_SEQ_LENGTH
EMBEDDING_BATCH_SIZE = project_config.EMBEDDING_BATCH_SIZE

# API keys from project_config
api_key = project_config.OPENAI_API_KEY
groq_api_key = project_config.GROQ_API_KEY

# Get the model for chunking from the project_config
chunking_model_name = project_config.get_model_for_task("chunking")

# Initialize models
openai_text_model = OpenAIChat(id=chunking_model_name, api_key=api_key)
groq_text_model = Groq(id=chunking_model_name, api_key=groq_api_key)

# Select which model to use based on configuration
MODEL_PROVIDER = project_config.MODEL_PROVIDER.lower()

if MODEL_PROVIDER == "openai":
    active_model = openai_text_model
    logger.info(f"Using OpenAI model for chunking: {chunking_model_name}", extra={"icon": "🧠"})
else:  # Default to Groq
    active_model = groq_text_model
    logger.info(f"Using Groq model for chunking: {chunking_model_name}", extra={"icon": "🧠"})

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
    differences: List[str] = Field(
        ..., 
        description="List of concrete, specific differences between the chunks that influenced the decision. Examples: 'Weight changed from 300kg to 350kg', 'Added new safety information in paragraph 3', etc. You MUST provide at least 2-3 differences if they exist."
    )

# -----------------------------------------------------------------------------
# Core Chunking Functions (Enhanced from integrated_chunking.py)
# -----------------------------------------------------------------------------

def approx_token_count(text: str) -> int:
    """Approximate token count based on character length (1 token ~= 4 characters)"""
    return len(text) // 4

def is_sentence_boundary(text: str, pos: int) -> bool:
    """
    More robust check if a position in text is at a sentence boundary.
    Considers ., !, ?, followed by space, newline, or end of text.
    Also handles cases like quotes or parentheses around sentence-ending punctuation.
    
    From agentic_chunking.py for better boundary detection.
    """
    if pos <= 0 or pos >= len(text):
        return False

    # Character immediately before the potential split point
    prev_char = text[pos-1]
    
    # Character immediately after the potential split point (if exists)
    next_char = text[pos] if pos < len(text) else ''

    if prev_char in '.!?':
        # Common case: Punctuation followed by space or newline
        if next_char.isspace() or next_char == '\n' or pos == len(text):
            return True
        # Case: Punctuation followed by closing quote/parenthesis, then space/newline
        if next_char in ['"', "'", ")", "]", "}"]:
            if pos + 1 < len(text):
                after_quote_char = text[pos+1]
                if after_quote_char.isspace() or after_quote_char == '\n':
                    return True
    
    return False

def split_large_document(md_text: str) -> List[str]:
    """
    Split very large documents into processable sections.
    Enhanced version from integrated_chunking.py that includes better splitting logic.
    """
    if len(md_text) <= MAX_SECTION_SIZE:
        return [md_text]
        
    logger.info(f"🔄 Document exceeds size threshold ({len(md_text)} chars), performing initial splitting")
    
    # First try to split by headers (most semantically meaningful)
    header_matches = list(re.finditer(r'(?:\n|^)(#{1,6}\s+[^\n]+)', md_text))
    
    if len(header_matches) > 1:
        sections = []
        current_section = ""
        last_pos = 0
        
        for match in header_matches:
            if match.start() > 0:  # Skip the first header as it's the beginning
                section_text = md_text[last_pos:match.start()]
                if section_text:
                    if len(current_section) + len(section_text) <= MAX_SECTION_SIZE:
                        current_section += section_text
                    else:
                        if current_section:
                            sections.append(current_section)
                        current_section = section_text
                last_pos = match.start()
        
        # Add the final section
        final_section = md_text[last_pos:]
        if final_section:
            if len(current_section) + len(final_section) <= MAX_SECTION_SIZE:
                current_section += final_section
            else:
                if current_section:
                    sections.append(current_section)
                current_section = final_section
        
        if current_section:
            sections.append(current_section)
            
        if len(sections) > 1:
            logger.info(f"✅ Split document by headers into {len(sections)} sections")
            return sections
    
    # If header splitting didn't work well, try paragraphs
    sections = []
    current_section = ""
    
    paragraphs = md_text.split('\n\n')
    
    for para in paragraphs:
        if len(current_section) + len(para) + 2 <= MAX_SECTION_SIZE:
            if current_section:
                current_section += '\n\n' + para
            else:
                current_section = para
        else:
            if current_section:
                sections.append(current_section)
            current_section = para
            
    if current_section:
        sections.append(current_section)
    
    # If we still have sections that are too large, split them further
    final_sections = []
    for section in sections:
        if len(section) <= MAX_SECTION_SIZE:
            final_sections.append(section)
        else:
            # For extremely large sections, just break by size
            logger.info(f"⚠️ Found an extremely large section ({len(section)} chars), breaking it up")
            for i in range(0, len(section), MAX_SECTION_SIZE // 2):
                subsection = section[i:i + MAX_SECTION_SIZE // 2]
                if subsection:
                    final_sections.append(subsection)
    
    logger.info(f"✅ Split large document into {len(final_sections)} processable sections")
    return final_sections

def get_chunks_from_llm(md_text: str, context_chunks: Optional[List[Dict[str, Any]]] = None) -> List[str]:
    """
    Ask the LLM to chunk the text directly with retry logic.
    Enhanced with support for context-aware chunking when context_chunks is provided.
    
    CHANGE: The prompt now explicitly instructs the LLM to ONLY split, NEVER modify, rephrase, or omit any content, and to preserve the original text exactly in each chunk. This ensures perfect traceability for requirements extraction.
    """
    if not md_text.strip():
        return []
        
    logger.info(f"🔄 Asking LLM to chunk text of {len(md_text)} characters")
    
    # Determine whether to use context-aware or standard chunking
    if context_chunks:
        # Context-aware chunking - prepare reference chunks for the LLM
        reference_chunks_text = "\n\n".join([
            f"Chunk {i+1}:\n{chunk.get('chunk_text', '')}" 
            for i, chunk in enumerate(context_chunks)
        ])
        
        # Enhanced prompt for context-aware chunking
        prompt = project_config.CONTEXT_AWARE_CHUNKING_PROMPT
        
        # Prepare full prompt content with reference chunks
        full_prompt = f"""
        # REFERENCE CHUNKS (from similar document):
        {reference_chunks_text}
        
        # NEW DOCUMENT TEXT (to be chunked):
        {md_text}
        """
        
    else:
        # Standard chunking prompt (improved from integrated_chunking)
        prompt = project_config.STANDARD_CHUNKING_PROMPT
        
        full_prompt = md_text
    
    # Format the prompt with size parameters
    formatted_prompt = prompt.format(max_size=MAX_CHAR_SIZE, target_size=TARGET_CHAR_SIZE)
    
    # Create agent for chunking
    agent = Agent(
        model=active_model,
        markdown=True,
        debug_mode=False,
        response_model=ChunksOutputModel,
        description=formatted_prompt,
        use_json_mode=True
    )
    
    retry_count = 0
    chunks = []
    
    while retry_count <= MAX_RETRIES:
        try:
            # If this is a retry, log it
            if retry_count > 0:
                logger.info(f"🔄 Retry #{retry_count} - Asking LLM to chunk text again")
            logger.debug(f"[LLM_CALL] Description / System prompt:\n{agent.description[:2000]}{'... [truncated]' if len(agent.description) > 2000 else ''}\n\n[LLM_CALL] User Prompt:\n{full_prompt[:2000]}{'... [truncated]' if len(full_prompt) > 2000 else ''}")
            response = agent.run(full_prompt)
            logger.debug(f"[LLM_CALL] Output from chunking agent: {str(response.content)[:2000]}{'... [truncated]' if len(str(response.content)) > 2000 else ''}")
            update_token_counters(response, chunking_model_name)
            
            data = response.content
            
            if isinstance(data, ChunksOutputModel):
                chunks = data.chunks
                break  # Success
            elif isinstance(data, str):
                try:
                    parsed_data = json.loads(data)
                    if 'chunks' in parsed_data:
                        chunks = parsed_data['chunks']
                        break  # Success
                    else:
                        logger.error(f"❌ LLM returned JSON without 'chunks' key", extra={"icon": "❌"})
                except json.JSONDecodeError:
                    logger.error(f"❌ LLM returned invalid JSON string", extra={"icon": "❌"})
            elif isinstance(data, dict):
                if 'chunks' in data:
                    chunks = data['chunks']
                    break  # Success
                else:
                    logger.error(f"❌ LLM returned dict without 'chunks' key", extra={"icon": "❌"})
            else:
                logger.error(f"❌ LLM returned unexpected data type: {type(data)}", extra={"icon": "❌"})
            
            # If we haven't broken out of the loop, the response wasn't valid
            retry_count += 1
            
            if retry_count <= MAX_RETRIES:
                # Calculate exponential backoff delay
                delay = INITIAL_RETRY_DELAY * (2 ** (retry_count - 1))
                logger.info(f"⏱️ Waiting {delay} seconds before retry...", extra={"icon": "⏱️"})
                time.sleep(delay)
            else:
                logger.error(f"❌ All {MAX_RETRIES} retries failed. Returning empty chunk list.", extra={"icon": "❌"})
                
        except Exception as e:
            logger.error(f"❌ Error getting chunks from LLM: {e}", extra={"icon": "❌"})
            retry_count += 1
            
            if retry_count <= MAX_RETRIES:
                delay = INITIAL_RETRY_DELAY * (2 ** (retry_count - 1))
                logger.info(f"⏱️ Waiting {delay} seconds before retry...", extra={"icon": "⏱️"})
                time.sleep(delay)
            else:
                logger.error(f"❌ All {MAX_RETRIES} retries failed due to exceptions.", extra={"icon": "❌"})
                return []
    
    # Filter out empty chunks
    chunks = [chunk for chunk in chunks if chunk.strip()]
    
    if chunks:
        logger.info(f"✅ LLM provided {len(chunks)} chunks", extra={"icon": "✅"})
    else:
        logger.warning("⚠️ LLM returned no valid chunks", extra={"icon": "⚠️"})
    
    # Check if any chunk exceeds the maximum size and needs re-chunking
    oversized_chunks = [i for i, chunk in enumerate(chunks) if len(chunk) > MAX_CHAR_SIZE]
    if oversized_chunks:
        logger.info(f"⚠️ Found {len(oversized_chunks)} oversized chunks, re-chunking them", extra={"icon": "⚠️"})
        
        # Log details about the oversized chunks to help diagnose issues
        for i in oversized_chunks:
            chunk = chunks[i]
            logger.info(f"⚠️ Oversized chunk #{i+1}: {len(chunk)} chars (limit: {MAX_CHAR_SIZE})", extra={"icon": "⚠️"})
            # Log the first characters to help identify the chunk
            logger.debug(f"Chunk start: {chunk[:100]}...", extra={"icon": "🔍"})
            
        new_chunks = []
        for i, chunk in enumerate(chunks):
            if i in oversized_chunks:
                # For extremely large chunks, first try a simple paragraph split
                if len(chunk) > MAX_CHAR_SIZE * 1.5:
                    logger.info(f"🔪 Chunk #{i+1} is extremely oversized ({len(chunk)} chars), trying paragraph split first", extra={"icon": "🔪"})
                    para_chunks = []
                    paragraphs = chunk.split('\n\n')
                    current_para_chunk = ""
                    
                    # Combine paragraphs into chunks under the size limit
                    for para in paragraphs:
                        if len(current_para_chunk) + len(para) + 2 <= MAX_CHAR_SIZE:
                            if current_para_chunk:
                                current_para_chunk += '\n\n' + para
                            else:
                                current_para_chunk = para
                        else:
                            if current_para_chunk:
                                para_chunks.append(current_para_chunk)
                            current_para_chunk = para
                    
                    if current_para_chunk:
                        para_chunks.append(current_para_chunk)
                    
                    # Check if any paragraph chunks are still too large
                    if any(len(pc) > MAX_CHAR_SIZE for pc in para_chunks):
                        logger.info(f"🔄 Paragraph split still produced oversized chunks, using LLM for finer chunking", extra={"icon": "🔄"})
                        # Use LLM for fine-grained chunking
                        sub_chunks = get_chunks_from_llm(chunk)
                        new_chunks.extend(sub_chunks)
                    else:
                        logger.info(f"✅ Paragraph split successful, created {len(para_chunks)} chunks", extra={"icon": "✅"})
                        new_chunks.extend(para_chunks)
                else:
                    # Regular oversized chunk, use LLM to re-chunk
                    sub_chunks = get_chunks_from_llm(chunk)
                    new_chunks.extend(sub_chunks)
            else:
                new_chunks.append(chunk)
                
        chunks = new_chunks
        
        # Double-check that we don't still have oversized chunks
        still_oversized = [i for i, chunk in enumerate(chunks) if len(chunk) > MAX_CHAR_SIZE]
        if still_oversized:
            logger.warning(f"⚠️ Still have {len(still_oversized)} oversized chunks after re-chunking", extra={"icon": "⚠️"})
            # For persistent oversized chunks, break them at sentence boundaries as a last resort
            final_chunks = []
            for i, chunk in enumerate(chunks):
                if i in still_oversized:
                    logger.info(f"🔪 Breaking persistent oversized chunk #{i+1} at sentence boundaries", extra={"icon": "🔪"})
                    sentences = re.split(r'(?<=[.!?])\s+', chunk)
                    current_sentence_chunk = ""
                    
                    for sentence in sentences:
                        if len(current_sentence_chunk) + len(sentence) + 1 <= MAX_CHAR_SIZE:
                            if current_sentence_chunk:
                                current_sentence_chunk += ' ' + sentence
                            else:
                                current_sentence_chunk = sentence
                        else:
                            if current_sentence_chunk:
                                final_chunks.append(current_sentence_chunk)
                            current_sentence_chunk = sentence
                    
                    if current_sentence_chunk:
                        final_chunks.append(current_sentence_chunk)
                else:
                    final_chunks.append(chunk)
            
            chunks = final_chunks
        
        logger.info(f"✅ After re-chunking oversized chunks, we now have {len(chunks)} chunks", extra={"icon": "✅"})
    
    return chunks

def chunk_markdown(md_text: str, context_chunks: Optional[List[Dict[str, Any]]] = None) -> List[str]:
    """
    Main function to chunk markdown text with optional context awareness.
    
    Args:
        md_text: The markdown text to chunk
        context_chunks: Optional list of existing chunks to use as context for alignment
        
    Returns:
        List of chunked text strings
    """
    if not md_text.strip():
        logger.info("ℹ️ Document is empty. No chunks to create.", extra={"icon": "ℹ️"})
        return []
    
    if context_chunks:
        logger.info(f"🔄 Performing context-aware chunking with {len(context_chunks)} reference chunks", extra={"icon": "🔄"})
    else:
        logger.info(f"🔄 Chunking markdown text of {len(md_text)} characters using standard approach", extra={"icon": "🔄"})
    
    # Handle large documents by splitting into sections first
    document_sections = split_large_document(md_text)
    
    all_chunks = []
    for i, section in enumerate(document_sections):
        logger.info(f"🔄 Processing section {i+1}/{len(document_sections)}", extra={"icon": "🔄"})
        # Only use context for the first section if we have multiple sections
        section_context = context_chunks if i == 0 else None
        section_chunks = get_chunks_from_llm(section, section_context)
        all_chunks.extend(section_chunks)
    
    logger.info(f"✅ Completed chunking. Total chunks: {len(all_chunks)}", extra={"icon": "✅"})
    return all_chunks 

# -----------------------------------------------------------------------------
# Chunk Comparison & Evaluation Functions
# -----------------------------------------------------------------------------

def generate_chunk_hash(text: str) -> str:
    """Generate a hash for a chunk of text for deduplication."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def compute_chunk_similarity(chunk1: str, chunk2: str) -> float:
    """
    Compute text similarity between two chunks using a combination of methods:
    1. Standard sequence matching for direct similarity
    2. Set-based similarity for reordered content
    
    Returns a similarity score between 0 and 1
    """
    # Standard similarity using difflib
    seq_similarity = difflib.SequenceMatcher(None, chunk1, chunk2).ratio()
    
    # If chunks are almost identical by sequence, return the high sequence similarity directly.
    # This helps ensure that textually identical or very nearly identical chunks
    # achieve a similarity score high enough to be considered duplicates.
    if seq_similarity >= 0.99:
        return seq_similarity

    # Set-based similarity to handle reordered content
    # Tokenize the chunks into words and compare as sets
    words1 = set(re.findall(r'\b\w+\b', chunk1.lower()))
    words2 = set(re.findall(r'\b\w+\b', chunk2.lower()))
    
    # Calculate Jaccard similarity for word sets
    if not words1 or not words2:
        set_similarity = 0.0
    else:
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        set_similarity = intersection / union if union > 0 else 0.0
    
    # Combine the similarities with a bias toward sequence similarity
    # but giving some weight to set similarity for reordered content
    combined_similarity = (0.7 * seq_similarity) + (0.3 * set_similarity)
    
    return combined_similarity

def evaluate_chunk_pair(new_chunk: str, old_chunk: Dict[str, Any], actual_new_doc_id: str) -> Dict[str, Any]:
    """
    Evaluate a pair of chunks to determine if they're duplicates, similar, or different.
    
    Returns decision information with similarity score and decision.
    """
    # Extract necessary information from old_chunk
    old_chunk_text = old_chunk.get('chunk_text', '')
    
    # First calculate simple text similarity
    similarity = compute_chunk_similarity(new_chunk, old_chunk_text)
    
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
        if similarity == 1:
            decision_info["reason"] = f"Chunks are exactly identical (similarity: {similarity:.4f})"
        else:
            decision_info["reason"] = f"Chunks are nearly identical (similarity: {similarity:.4f})"
        return decision_info
    
    # For medium similarity, use LLM to decide
    if similarity >= SIMILARITY_THRESHOLD:
        # Set up the decision prompt
        decision_prompt = project_config.CHUNK_COMPARISON_PROMPT
        
        # Create an agent for the decision
        decision_agent = Agent(
            model=active_model,
            markdown=True,
            debug_mode=False,
            response_model=ChunkDecisionModel,
            description=decision_prompt.format(old_chunk=old_chunk_text, new_chunk=new_chunk),
            use_json_mode=True
        )
        
        try:
            response = decision_agent.run("")
            update_token_counters(response, chunking_model_name)
            
            data = response.content
            decision_info["decision"] = data.decision
            decision_info["reason"] = data.reason
            if data.differences:
                decision_info["differences"] = data.differences
                
            logger.info(
                f"LLM determined chunks with {similarity:.4f} similarity should be: {data.decision}",
                extra={"icon": "🧠"}
            )
            
            # If LLM recommends user input, prompt the user
            if data.decision == "need_user_input":
                logger.info(f"LLM suggested getting user input for chunk comparison", extra={"icon": "👨‍💻"})
                # Get user decision
                user_decision = prompt_user_for_chunk_decision(
                    new_chunk=new_chunk,
                    old_chunk=old_chunk,
                    new_doc_id=actual_new_doc_id,
                    decision_info=decision_info
                )
                decision_info["decision"] = user_decision
                decision_info["reason"] += " (User decision: " + user_decision + ")"
            
        except Exception as e:
            logger.error(f"Error using LLM for chunk comparison: {e}", extra={"icon": "❌"})
            # Default to keeping the new chunk if LLM fails
            decision_info["decision"] = "keep_new"
            decision_info["reason"] = "LLM comparison failed, defaulting to keeping new chunk"
            
    else:
        # Low similarity - this is a new chunk, not similar to any existing chunk
        decision_info["decision"] = "keep_new"
        decision_info["reason"] = f"Chunks are significantly different (similarity: {similarity:.4f})"
        
    return decision_info

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
    old_doc_id = old_chunk.get('document_id', 'unknown')
    old_chunk_text = old_chunk.get('chunk_text', '')
    reason = decision_info.get('reason', 'Reason not provided')
    differences = decision_info.get('differences', [])
    
    logger.info("================================================================================", extra={"icon": "🔍"})
    logger.info("CHUNK COMPARISON NEEDED", extra={"icon": "🔍"})
    logger.info("================================================================================", extra={"icon": "🔍"})
    
    logger.info(f"REASON: {reason}", extra={"icon": "ℹ️"})
    
    logger.info("KEY DIFFERENCES:", extra={"icon": "📊"})
    if differences:
        for i, diff in enumerate(differences, 1):
            logger.info(f"  {i}. {diff}", extra={"icon": "🔄"})
    else:
        logger.info("  (No specific differences detected)", extra={"icon": "⚠️"})
    
    logger.info(f"OLD DOCUMENT: {old_doc_id}", extra={"icon": "📜"})
    logger.info("----------------------------------------", extra={"icon": "📜"})
    logger.info(old_chunk_text, extra={"icon": "📜"})
    logger.info("----------------------------------------", extra={"icon": "📜"})
    
    logger.info(f"NEW DOCUMENT: {new_doc_id}", extra={"icon": "📄"})
    logger.info("----------------------------------------", extra={"icon": "📄"})
    logger.info(new_chunk, extra={"icon": "📄"})
    logger.info("----------------------------------------", extra={"icon": "📄"})
    
    while True:
        choice = input("\nChoose which chunk to use (1=old, 2=new): ")
        if choice in ('1', '2'):
            return "keep_old" if choice == '1' else "keep_new"
        logger.warning("Invalid choice. Please enter 1 or 2.", extra={"icon": "⚠️"})

def log_chunk_comparison(chunk_id: str, similar_chunk_id: str, similarity: float, decision: str, reason: str, is_evaluating: bool = False):
    """
    Log the comparison between two chunks with a detailed explanation of the decision.
    
    Args:
        chunk_id: ID of the current chunk
        similar_chunk_id: ID of the similar chunk found
        similarity: Similarity score between the chunks
        decision: Decision made (e.g., "keep_original", "keep_new", "both", "evaluating", "need_user_input")
        reason: Reason for the decision
        is_evaluating: If True, only log at DEBUG level (for intermediate evaluation messages)
    """
    # For intermediate evaluation messages (not final decisions), log at debug level
    if is_evaluating or decision == "evaluating":
        logger.debug(
            f"🔍 Chunk Comparison: {chunk_id} vs {similar_chunk_id} | Similarity: {similarity:.4f} | Status: Evaluating",
            extra={"icon": "🔍"}
        )
        return
    
    # For final decisions, log a structured message with all information
    logger.info(
        f"🔍 Chunk: {chunk_id} vs {similar_chunk_id} | Sim: {similarity:.4f} | Decision: {decision} | {reason}",
        extra={"icon": "🔍"}
    )
    
    # Log a summary of the decision with an appropriate icon (but only for final decisions)
    if decision == "keep_original" or decision == "keep_old":
        logger.info(f"🔄 Using existing chunk {similar_chunk_id} instead of {chunk_id}", extra={"icon": "🔄"})
    elif decision == "keep_new":
        logger.info(f"⬆️ Replacing {similar_chunk_id} with new chunk {chunk_id}", extra={"icon": "⬆️"})
    elif decision == "both":
        logger.info(f"➕ Keeping both {similar_chunk_id} and {chunk_id}", extra={"icon": "➕"})
    elif decision == "need_user_input":
        logger.info(f"👤 User input needed for {chunk_id} vs {similar_chunk_id}", extra={"icon": "👤"})
    elif decision != "evaluating":  # Don't log for evaluating as we've already handled it
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

# -----------------------------------------------------------------------------
# Document Processing Functions
# -----------------------------------------------------------------------------

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
        if not db or project_config.DOCUMENT_CHUNKS_TABLE not in db.table_names():
            logger.warning(f"No existing chunks table found for document {similar_doc_id}", extra={"icon": "⚠️"})
            return []
            
        # Query the database for chunks from the similar document
        chunks_table = db.open_table(project_config.DOCUMENT_CHUNKS_TABLE)
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
    
    # Chunk the document - with or without context
    raw_chunks = chunk_markdown(document_text, similar_chunks if similar_chunks else None)

    # Check if chunking succeeded
    if not raw_chunks:
        logger.error("Chunking failed to produce any chunks", extra={"icon": "❌"})
        return [], {}
        
    logger.info(f"Created {len(raw_chunks)} initial chunks", extra={"icon": "✅"})
    
    # Create a list to hold processed chunks with full metadata
    processed_chunks = []
    current_offset = 0
    
    # Load embedding model
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
                # Diagnostic: log and write chunk and document slice to temp file for analysis
                temp_dir = project_config.CHUNK_DIAGNOSTICS_DIR
                os.makedirs(temp_dir, exist_ok=True)
                doc_base = os.path.splitext(os.path.basename(document_id))[0]
                temp_filename = f"{doc_base}_chunk_{i+1:04d}_diagnostic.txt"
                temp_path = os.path.join(temp_dir, temp_filename)
                doc_slice = document_text[max(0, current_offset-100):current_offset+100]
                with open(temp_path, 'w', encoding='utf-8') as f:
                    f.write(f"CHUNK {i+1} (not found verbatim):\n{'-'*40}\n{chunk_text}\n\n")
                    f.write(f"Document slice around current_offset={current_offset}:\n{'-'*40}\n{doc_slice}\n")
                logger.info(f"Diagnostic written to {temp_path}", extra={"icon": "📝"})
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
        timestamp = generate_timestamp()
        
        # Compare against all reference chunks from the similar document
        if similar_chunks:
            best_similarity = 0.0
            best_chunk = None
            
            for ref_chunk in similar_chunks:
                # Compare embeddings
                ref_embedding = np.array(ref_chunk.get('embedding', []))
                if len(ref_embedding) > 0:
                    # Use our enhanced similarity measure
                    similarity = compute_chunk_similarity(chunk_text, ref_chunk.get('chunk_text', ''))
                    
                    # Log the initial comparison status
                    log_chunk_comparison(
                        chunk_id=chunk_id,
                        similar_chunk_id=ref_chunk.get('chunk_id', 'unknown'),
                        similarity=similarity,
                        decision="evaluating",
                        reason="Comparing chunks for similarity",
                        is_evaluating=True
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
                        
                        # Log the final decision for duplicate chunks
                        log_chunk_comparison(
                            chunk_id=chunk_id,
                            similar_chunk_id=ref_chunk.get('chunk_id', 'unknown'),
                            similarity=similarity,
                            decision="keep_old",
                            reason=f"Chunks are nearly identical (similarity: {similarity:.4f})",
                            is_evaluating=True
                        )
                        break
                    elif similarity >= SIMILARITY_THRESHOLD:
                        # This might be an updated version of the chunk
                        decision_info = evaluate_chunk_pair(
                            new_chunk=chunk_text,      # Text from the current new document's chunk
                            old_chunk=ref_chunk,       # Dict of the old document's chunk
                            actual_new_doc_id=document_id # Pass the ID of the new document being processed
                        )
                        
                        # Log the final decision based on the evaluation
                        log_chunk_comparison(
                            chunk_id=chunk_id,
                            similar_chunk_id=ref_chunk.get('chunk_id', 'unknown'),
                            similarity=similarity,
                            decision=decision_info["decision"],
                            reason=decision_info["reason"],
                            is_evaluating=True
                        )
                        
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
                            
                            # Log the decision with detailed reason
                            logger.info(
                                f"Chunk {chunk_id} replaces {previous_chunk_id} - Reason: {decision_info['reason']}",
                                extra={"icon": "🔄"}
                            )
                            break
            
            # If we didn't find a duplicate or similar chunk, it's a new chunk
            if not is_duplicate and not is_updated:
                new_count += 1
                
                # If we have a best match, log it even though it's below threshold
                log_chunk_comparison(
                    chunk_id=chunk_id,
                    similar_chunk_id=best_chunk.get('chunk_id', 'unknown'),
                    similarity=best_similarity,
                    decision="keep_new",
                    reason=f"New chunk with no similar content above threshold (best similarity: {best_similarity:.4f})",
                    is_evaluating=True
                )
            else:
                # No similar chunks at all
                logger.info(f"✨ Chunk {chunk_id} is completely new with no similar content", extra={"icon": "✨"})
        else:
            # Without reference chunks, all chunks are new
            new_count += 1
            
        # Now instead of skipping duplicate chunks, we'll always save them with the duplicate flag
        if is_duplicate:
            logger.info(
                f"Found duplicate chunk {chunk_id}, will save with reference to {duplicate_of}",
                extra={"icon": "♻️"}
            )
            # Note: We continue processing instead of 'continue' to save the chunk
                
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
            "aligned_with_document_id": similar_doc_id or "",
        }
        
        processed_chunks.append(chunk_data)
    
    # Log a summary of the chunking results
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
    
    # If using MPS (Apple Silicon), clear that cache too
    if torch.backends.mps.is_available(): # More robust check
        if hasattr(torch.mps, 'empty_cache'): # Still need to check for the attribute itself
            torch.mps.empty_cache()
    
    # Use directly instead of duplicating the code (replaced with our new helper function)
    cleanup_memory(log_message=True)
    
    duration = time.time() - start_time
    logger.info(
        f"Processed {len(processed_chunks)} chunks in {duration:.2f}s "
        f"({duplicate_count} duplicates, {similar_count} updated, {new_count} new)",
        extra={"icon": "✅"}
    )
    
    return processed_chunks, replaced_chunks

def update_replacement_references_in_df(df, old_chunk_id: str, new_chunk_id: str) -> None:
    """
    Update all replacement references in the DataFrame to point to the new chunk, flattening the chain.
    Args:
        df: DataFrame of all chunks
        old_chunk_id: The chunk being replaced
        new_chunk_id: The new chunk that replaces it
    """
    # Find all chunks that currently point to the old chunk
    chunks_pointing_to_old = df[df['replaced_by'] == old_chunk_id]
    if not chunks_pointing_to_old.empty:
        logger.info(f"Flattening {len(chunks_pointing_to_old)} replacement references from {old_chunk_id} to {new_chunk_id}", extra={"icon": "🔄"})
        df.loc[df['replaced_by'] == old_chunk_id, 'replaced_by'] = new_chunk_id
    
    # Update the old chunk being replaced
    df.loc[df['chunk_id'] == old_chunk_id, 'is_replaced'] = True
    df.loc[df['chunk_id'] == old_chunk_id, 'replaced_by'] = new_chunk_id

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
        table_exists = project_config.DOCUMENT_CHUNKS_TABLE in db.table_names()
        
        if table_exists:
            logger.info(f"Opened existing table: {project_config.DOCUMENT_CHUNKS_TABLE}", extra={"icon": "✅"})
            chunks_table = db.open_table(project_config.DOCUMENT_CHUNKS_TABLE)
            # Check table schema for required columns
            table_schema = chunks_table.schema
            existing_columns = {field.name for field in table_schema}
            required_columns = {
                'is_replaced', 
                'replaced_by'
            }
            missing_columns = required_columns - existing_columns
            # Load all current data to a DataFrame
            current_df = chunks_table.to_pandas()
            # Add missing columns if needed
            if missing_columns:
                logger.info(f"Adding missing columns: {missing_columns}", extra={"icon": "🔄"})
                for col in missing_columns:
                    if col == 'is_replaced':
                        current_df[col] = False
                    else:
                        current_df[col] = ""
            # Handle replacement flattening in-memory
            if replaced_chunks:
                if len(replaced_chunks) > 0:
                    logger.info(f"Flattening {len(replaced_chunks)} replacement chains", extra={"icon": "🔄"})
                    
                for new_chunk_id, old_chunk_id in replaced_chunks.items():
                    # Don't log individual flattening messages here, just do the updates silently
                    update_replacement_references_in_df(current_df, old_chunk_id, new_chunk_id)
                
                if len(replaced_chunks) > 0:
                    logger.info(f"Successfully flattened all replacement chains", extra={"icon": "✅"})
            # Add new chunks to the DataFrame
            new_chunks_df = pd.DataFrame(chunks) if chunks else pd.DataFrame()
            if not new_chunks_df.empty:
                for col in ['is_replaced', 'replaced_by']:
                    if col not in new_chunks_df.columns:
                        if col == 'is_replaced':
                            new_chunks_df[col] = False
                        else:
                            new_chunks_df[col] = ""
                combined_df = pd.concat([current_df, new_chunks_df], ignore_index=True)
                logger.info(f"Added {len(new_chunks_df)} new chunks to existing table (in-memory)", extra={"icon": "➕"})
            else:
                combined_df = current_df
                logger.info("No new chunks from current document to add.", extra={"icon": "ℹ️"})
            # Write the updated DataFrame to LanceDB ONCE
            db.drop_table(project_config.DOCUMENT_CHUNKS_TABLE)
            db.create_table(project_config.DOCUMENT_CHUNKS_TABLE, combined_df)
            logger.info(f"Recreated table '{project_config.DOCUMENT_CHUNKS_TABLE}' with all updated and new chunks.", extra={"icon": "✅"})
        else:
            # First chunk added - create new table
            if chunks:
                logger.info(f"Creating new table: {project_config.DOCUMENT_CHUNKS_TABLE}", extra={"icon": "✅"})
                data = pd.DataFrame(chunks)
                
                # Ensure all required columns exist
                for col in ['is_replaced', 'replaced_by']:
                    if col not in data.columns:
                        if col == 'is_replaced':
                            data[col] = False
                        else:
                            data[col] = ""
                    
                chunks_table = db.create_table(project_config.DOCUMENT_CHUNKS_TABLE, data=data)
                return True
            else:
                logger.warning("No chunks to save, table not created", extra={"icon": "⚠️"})
                return True
        
        # Create vector index if needed
        dedup.ensure_index(db, project_config.DOCUMENT_CHUNKS_TABLE)
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving chunks to database: {str(e)}", extra={"icon": "❌"})
        return False 

# -----------------------------------------------------------------------------
# Utility Functions & Main Entry Point
# -----------------------------------------------------------------------------

def analyze_chunks(chunks: List[str]) -> Dict[str, Any]:
    """Provide simple statistics about the chunks."""
    if not chunks:
        return {"chunk_count": 0}
    
    char_sizes = [len(chunk) for chunk in chunks]
    token_sizes = [approx_token_count(chunk) for chunk in chunks]
    
    return {
        "chunk_count": len(chunks),
        "char_sizes": {
            "min": min(char_sizes) if char_sizes else 0,
            "max": max(char_sizes) if char_sizes else 0,
            "avg": sum(char_sizes) / len(char_sizes) if char_sizes else 0
        },
        "token_sizes": {
            "min": min(token_sizes) if token_sizes else 0,
            "max": max(token_sizes) if token_sizes else 0,
            "avg": sum(token_sizes) / len(token_sizes) if token_sizes else 0
        }
    }

def cleanup_memory(log_message: bool = True):
    """
    Release memory after processing to prevent memory leaks.
    Especially important when running multiple document processing tasks in sequence.
    
    Args:
        log_message: Whether to log a message about cleaning up memory
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # If using MPS (Apple Silicon), clear that cache too
    if torch.backends.mps.is_available(): # More robust check
        if hasattr(torch.mps, 'empty_cache'): # Still need to check for the attribute itself
            torch.mps.empty_cache()
    
    if log_message:
        logger.info("Memory cleaned up", extra={"icon": "🧹"})

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
            # No chunks were generated which isn't normal except for empty documents
            if not document_text.strip():
                logger.info(f"Document {document_id} is empty, nothing to process", extra={"icon": "ℹ️"})
                return True
            else:
                logger.warning(f"No chunks were generated for document {document_id}", extra={"icon": "⚠️"})
                return False
        
        # Step 2: Save chunks to database
        if chunks:
            # Pass replaced_chunks along with chunks to handle both in one operation
            success = save_chunks_to_db(chunks, replaced_chunks)
            if not success:
                logger.error(f"Failed to save chunks to database for document {document_id}", extra={"icon": "❌"})
                return False
        else:
            # If there are no chunks to save, log this information
            logger.info(f"No new chunks to save for document {document_id}", extra={"icon": "ℹ️"})
        
        logger.info(f"Successfully processed document: {document_id}", extra={"icon": "✅"})
        return True
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {e}", extra={"icon": "❌"})
        return False
    finally:
        # Clean up memory regardless of success or failure, but don't log since it was likely already logged
        cleanup_memory(log_message=False)

def process_document_pages(document_pages: List[Dict[str, Any]], document_id: str) -> bool:
    """
    Process a document from its pages (for compatibility with the pipeline).
    
    Args:
        document_pages: List of page dictionaries with content
        document_id: ID of the document to process
        
    Returns:
        True if processing was successful, False otherwise
    """
    # Extract full text from pages
    document_text = ""
    for page in sorted(document_pages, key=lambda p: p.get('page_number', 0)):
        if 'md_content' in page:
            document_text += page['md_content'] + "\n\n"
    
    # Process the document using the main function
    return process_document(document_text, document_id)

def main():
    """
    Test function for chunking.
    """
    # Try to find a test file
    test_file = os.path.join("input", "raw", "test_document.txt")
    test_content = ""
    
    if os.path.exists(test_file):
        with open(test_file, 'r', encoding='utf-8') as f:
            test_content = f.read()
    else:
        # Create a simple test file if none exists
        test_content = """
        # Test Document
        
        ## Section 1
        This is a test document for context-aware chunking.
        It contains multiple sections with different content.
        
        ## Section 2
        The chunking system should:
        * Keep related content together
        * Maintain semantic coherence
        * Create appropriately sized chunks
        * Handle document updates efficiently
        
        ## Section 3
        This is the final section of our test document.
        It shows how the system handles different content structures.
        """
        
        # Write the test file for future use
        os.makedirs(os.path.dirname(test_file), exist_ok=True)
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
    
    logger.info(f"Testing chunking with document length: {len(test_content)} characters", extra={"icon": "🧪"})
    
    # Process the document
    chunks = chunk_markdown(test_content)
    
    if chunks:
        analysis = analyze_chunks(chunks)
        logger.info(f"Generated {len(chunks)} chunks", extra={"icon": "✅"})
        logger.info(f"Min/Max/Avg chars: {analysis['char_sizes']['min']}/{analysis['char_sizes']['max']}/{analysis['char_sizes']['avg']:.1f}", extra={"icon": "📊"})
        logger.info(f"Min/Max/Avg tokens: {analysis['token_sizes']['min']}/{analysis['token_sizes']['max']}/{analysis['token_sizes']['avg']:.1f}", extra={"icon": "📊"})
        
        # Output the chunks
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"\n--- Chunk {i} ---", extra={"icon": "📄"})
            logger.info(chunk[:150] + "..." if len(chunk) > 150 else chunk, extra={"icon": "📄"})
    else:
        logger.warning("No chunks were generated", extra={"icon": "⚠️"})
    
    return 0

if __name__ == "__main__":
    sys.exit(main())