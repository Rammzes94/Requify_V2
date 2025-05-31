"""
doc_embedding_utils.py

This module provides utilities for generating document-level embeddings,
including text preparation, tokenization, and embedding generation.

It handles proper tokenization and truncation of document text to ensure 
it fits within the model's context window while maximizing token usage.
"""

import os
import logging
from typing import List, Dict, Optional, Union
import numpy as np
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

from src.utils import setup_logging, get_logger, update_token_counters, get_token_usage, print_token_usage, reset_token_counters, setup_project_directory, generate_timestamp

# Setup logging
logger = get_logger("Doc_Embedding_Utils")

# Constants
DOC_EMBEDDING_MODEL_NAME = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
DOC_EMBEDDING_DIMENSION = 1024  # Dimension for the document embedding model
DOC_EMBEDDING_MAX_SEQ_LENGTH = 32768  # Maximum sequence length for the document embedding model

def prepare_document_text(all_records: List[Dict]) -> str:
    """
    Prepare document text by concatenating all page content and truncating if necessary.
    
    Args:
        all_records: List of page records with md_content fields
        
    Returns:
        Properly truncated document text ready for embedding
    """
    # Concatenate all page content with space separator
    try:
        # Filter out empty content and sort by page number if available
        valid_records = [r for r in all_records if r.get("md_content")]
        sorted_records = sorted(valid_records, key=lambda x: x.get("page_number", 9999))
        
        if not sorted_records:
            logger.warning("No valid content found in records", extra={"icon": "⚠️"})
            return ""
            
        # Concatenate the text with spaces
        full_text = " ".join([record.get("md_content", "") for record in sorted_records])
        
        logger.info(f"Created full document text ({len(full_text)} characters)", extra={"icon": "✅"})
        
        # Use Qwen tokenizer to truncate at token limit
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B", trust_remote_code=True)
        tokens = tokenizer.encode(full_text, add_special_tokens=False)
        
        tokens_count = len(tokens)
        if tokens_count > DOC_EMBEDDING_MAX_SEQ_LENGTH:
            # Truncate tokens and decode back to text
            logger.info(f"Document text exceeds token limit ({tokens_count} > {DOC_EMBEDDING_MAX_SEQ_LENGTH}). Truncating.", extra={"icon": "✂️"})
            truncated_tokens = tokens[:DOC_EMBEDDING_MAX_SEQ_LENGTH]
            truncated_text = tokenizer.decode(truncated_tokens)
            logger.info(f"Truncated document text to {DOC_EMBEDDING_MAX_SEQ_LENGTH} tokens", extra={"icon": "✅"})
            return truncated_text
        else:
            logger.info(f"Document text is within token limit ({tokens_count} <= {DOC_EMBEDDING_MAX_SEQ_LENGTH})", extra={"icon": "✅"})
            return full_text
            
    except Exception as e:
        logger.error(f"Error preparing document text: {str(e)}", extra={"icon": "❌"})
        return ""

def count_tokens(text: str) -> int:
    """
    Count the number of tokens in the given text using the Qwen2 tokenizer.
    
    Args:
        text: The text to count tokens for
        
    Returns:
        Number of tokens in the text
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B", trust_remote_code=True)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    except Exception as e:
        logger.error(f"Error counting tokens: {str(e)}", extra={"icon": "❌"})
        return 0

def get_document_embedder(device: str = "cpu") -> Optional[SentenceTransformer]:
    """
    Initialize and return the document-level embedding model.
    
    Args:
        device: Device to load the model on ('cpu' or 'cuda')
        
    Returns:
        Initialized SentenceTransformer model or None if initialization fails
    """
    try:
        logger.info(f"Loading document embedding model: {DOC_EMBEDDING_MODEL_NAME} on {device}", extra={"icon": "🧠"})
        model = SentenceTransformer(DOC_EMBEDDING_MODEL_NAME, device=device, trust_remote_code=True)
        model.max_seq_length = DOC_EMBEDDING_MAX_SEQ_LENGTH  # Explicitly set max sequence length
        logger.info(f"Document embedding model loaded successfully with max_seq_length={DOC_EMBEDDING_MAX_SEQ_LENGTH}", extra={"icon": "✅"})
        return model
    except Exception as e:
        logger.error(f"Error loading document embedding model: {str(e)}", extra={"icon": "❌"})
        return None

def generate_document_embedding(doc_text: str, model: SentenceTransformer) -> Optional[np.ndarray]:
    """
    Generate document embedding for the given text using the provided model.
    
    Args:
        doc_text: Document text to embed
        model: SentenceTransformer model to use for embedding
        
    Returns:
        Document embedding as numpy array or None if embedding fails
    """
    if not doc_text.strip():
        logger.warning("Empty document text, cannot generate embedding", extra={"icon": "⚠️"})
        return None
        
    try:
        logger.info(f"Generating document embedding for text ({len(doc_text)} chars)", extra={"icon": "🧬"})
        embedding = model.encode(doc_text, normalize_embeddings=True)
        logger.info(f"Generated document embedding successfully (shape: {embedding.shape})", extra={"icon": "✅"})
        return embedding
    except Exception as e:
        logger.error(f"Error generating document embedding: {str(e)}", extra={"icon": "❌"})
        return None 