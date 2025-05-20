"""
pre_save_reqs_deduplication.py

This script performs requirement-level deduplication in the requirements extraction pipeline.
It performs the following operations:
1. Checks newly extracted requirements against existing ones in the LanceDB database
2. Uses both vector similarity and LLM-based semantic analysis to identify duplicates
3. Applies high-precision duplicate detection to avoid storing redundant requirements
4. Handles edge cases with multiple similarity detection mechanisms
5. Uses a batch approach for efficient processing of multiple requirements
6. Provides confidence scores and detailed rationale for duplicate detection
7. Maintains references between duplicate requirements for traceability

The script integrates with the requirements extraction process to filter duplicates
before saving to the database, while preserving relationships between requirements
for downstream analysis and traceability reporting.
"""

import os
import sys
import logging
import time
import numpy as np
import pandas as pd
import lancedb
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple, Set

# Import Agno for LLM interaction
from agno.agent import Agent
from agno.models.groq import Groq
from agno.models.openai import OpenAIChat

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import _00_utils
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '_00_utils'))) # Removed redundant/incorrect path append
import config  # Import config from src
_00_utils.setup_project_directory()

# Load environment variables
load_dotenv()

# Setup logging with script prefix


logger = _00_utils.get_logger("Reqs_Deduplication")

# -------------------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------------------
OUTPUT_DIR_BASE = "_03_output"
LANCEDB_SUBDIR_NAME = "lancedb"
# Construct path relative to project root
LANCEDB_DIR_PATH = os.path.join(OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME)
REQUIREMENTS_TABLE_NAME = "requirements"
EMBEDDING_DIMENSION = config.EMBEDDING_DIMENSION  # Use from config

# Deduplication parameters
SIM_THRESHOLD = 0.97  # Cosine similarity threshold to consider as duplicate
TOP_K_RESULTS = 10  # Number of neighbors to compare per requirement

# LLM Parameters
GROQ_MODEL_ID = "llama-3.3-70b-versatile"
OPENAI_MODEL_ID = "gpt-4o-mini"

# Get API keys from config
api_key = config.OPENAI_API_KEY
groq_api_key = config.GROQ_API_KEY

# Select which model to use based on configuration
MODEL_PROVIDER = config.MODEL_PROVIDER.lower()

logger.info(f"Model provider for requirements deduplication: {MODEL_PROVIDER}", extra={"icon": "🧠"})

# -------------------------------------------------------------------------------------
# Pydantic Models for LLM Interaction
# -------------------------------------------------------------------------------------
class LLMDuplicateInfo(BaseModel):
    is_duplicate: bool = Field(..., description="Whether this requirement is a duplicate of an existing one.")
    duplicate_of: Optional[str] = Field(None, description="ID of the existing requirement this is a duplicate of, if any.")
    confidence: float = Field(..., description="Confidence level (0.0 to 1.0) in the duplicate determination.")
    rationale: str = Field(..., description="Brief explanation of why this is or is not a duplicate.")

class LLMDuplicateCheck(BaseModel):
    requirements: List[LLMDuplicateInfo] = Field(..., description="List of requirements with duplicate check results.")

# -------------------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------------------

def connect_to_lancedb(db_path: str):
    """Connect to LanceDB and return the connection object."""
    try:
        logger.info(f"Connecting to LanceDB at: {db_path}", extra={"icon": "🔄"})
        db = lancedb.connect(db_path)
        logger.info(f"Successfully connected to LanceDB. Available tables: {db.table_names()}", extra={"icon": "✅"})
        return db
    except Exception as e:
        logger.error(f"Error connecting to LanceDB at {db_path}: {e}", extra={"icon": "❌"})
        return None

def load_existing_requirements(db, doc_id=None):
    """
    Load existing requirements from the database, optionally filtered by document ID.
    
    Args:
        db: LanceDB connection
        doc_id: Optional document ID to filter by
        
    Returns:
        DataFrame of existing requirements
    """
    if not db or REQUIREMENTS_TABLE_NAME not in db.table_names():
        logger.warning(f"Requirements table '{REQUIREMENTS_TABLE_NAME}' not found", extra={"icon": "⚠️"})
        return pd.DataFrame()
    
    try:
        table = db.open_table(REQUIREMENTS_TABLE_NAME)
        
        # Query with or without document filter
        if doc_id:
            query = f"SELECT * FROM {REQUIREMENTS_TABLE_NAME} WHERE document_id = '{doc_id}'"
            logger.info(f"Loading existing requirements for document '{doc_id}'", extra={"icon": "🔍"})
        else:
            query = f"SELECT * FROM {REQUIREMENTS_TABLE_NAME}"
            logger.info(f"Loading all existing requirements", extra={"icon": "🔍"})
            
        existing_reqs = table.query(query).to_df()
        
        if existing_reqs.empty:
            logger.info(f"No existing requirements found{' for document ' + doc_id if doc_id else ''}", extra={"icon": "ℹ️"})
        else:
            logger.info(f"Loaded {len(existing_reqs)} existing requirements", extra={"icon": "✅"})
            
        return existing_reqs
    except Exception as e:
        logger.error(f"Error loading existing requirements: {e}", extra={"icon": "❌"})
        return pd.DataFrame()

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
# Deduplication Functions
# -------------------------------------------------------------------------------------

def check_vector_duplicates(new_reqs: List[Dict], existing_reqs: pd.DataFrame) -> Dict[int, Dict]:
    """
    Check for vector-based duplicates between new requirements and existing ones.
    
    Args:
        new_reqs: List of dictionaries containing new requirements
        existing_reqs: DataFrame of existing requirements
        
    Returns:
        Dictionary mapping new requirement indices to duplicate information
    """
    if existing_reqs.empty or not new_reqs:
        return {}
    
    # Ensure embeddings are available
    if 'embedding' not in existing_reqs.columns:
        logger.warning("Existing requirements have no embeddings. Cannot check for vector duplicates.", extra={"icon": "⚠️"})
        return {}
    
    # Prepare existing embeddings
    existing_embeddings = []
    valid_indices = []
    
    for i, row in existing_reqs.iterrows():
        emb = row.get('embedding')
        if emb is None:
            continue
            
        if isinstance(emb, list):
            emb = np.array(emb, dtype=np.float32)
            
        if emb.ndim == 0 or (emb.ndim == 1 and len(emb) != EMBEDDING_DIMENSION):
            continue
            
        existing_embeddings.append(normalize_embedding(emb))
        valid_indices.append(i)
    
    if not existing_embeddings:
        logger.warning("No valid embeddings found in existing requirements", extra={"icon": "⚠️"})
        return {}
        
    existing_embeddings_np = np.array(existing_embeddings)
    
    # Check each new requirement against existing ones
    duplicates = {}
    
    for idx, req in enumerate(new_reqs):
        emb = req.get('embedding')
        if emb is None:
            continue
            
        if isinstance(emb, list):
            emb = np.array(emb, dtype=np.float32)
            
        # Normalize the embedding
        norm_emb = normalize_embedding(emb)
        
        # Calculate similarities with all existing embeddings
        similarities = np.dot(existing_embeddings_np, norm_emb)
        
        # Check for duplicates
        max_sim_idx = np.argmax(similarities)
        max_sim = similarities[max_sim_idx]
        
        if max_sim >= SIM_THRESHOLD:
            existing_idx = valid_indices[max_sim_idx]
            duplicate_req = existing_reqs.iloc[existing_idx]
            
            duplicates[idx] = {
                'duplicate_id': duplicate_req.get('requirement_id', f"req_{existing_idx}"),
                'document_id': duplicate_req.get('document_id', 'unknown'),
                'similarity': float(max_sim),
                'existing_text': duplicate_req.get('description', ''),
                'check_llm': max_sim < 0.99  # If similarity is very high, no need for LLM check
            }
            
            logger.info(f"Requirement {idx} appears to be a duplicate (similarity: {max_sim:.4f})", extra={"icon": "🔍"})
            
    return duplicates

def check_llm_duplicates(new_reqs: List[Dict], existing_reqs: pd.DataFrame, vector_duplicates: Dict[int, Dict]) -> Dict[int, Dict]:
    """
    Use LLM to verify vector-based duplicates and catch semantic duplicates.
    
    Args:
        new_reqs: List of dictionaries containing new requirements
        existing_reqs: DataFrame of existing requirements
        vector_duplicates: Dictionary of already found vector-based duplicates
        
    Returns:
        Updated dictionary with LLM-verified duplicates
    """
    if not vector_duplicates:
        return {}
    
    # Initialize the LLM agent based on environment variable
    try:
        if MODEL_PROVIDER == "openai":
            llm = OpenAIChat(id=OPENAI_MODEL_ID, api_key=api_key)
            logger.info(f"Initialized OpenAI agent with model {OPENAI_MODEL_ID}", extra={"icon": "🤖"})
        else:  # Default to Groq
            llm = Groq(id=GROQ_MODEL_ID, api_key=groq_api_key)
            logger.info(f"Initialized Groq agent with model {GROQ_MODEL_ID}", extra={"icon": "🤖"})
        
        agent = Agent(llm)
    except Exception as e:
        logger.error(f"Error initializing LLM agent: {e}", extra={"icon": "❌"})
        # Return the vector duplicates without LLM verification
        return {k: v for k, v in vector_duplicates.items() if not v.get('check_llm', True)}
    
    # Prepare the candidates that need LLM verification
    llm_check_candidates = []
    
    for idx, dup_info in vector_duplicates.items():
        if dup_info.get('check_llm', True):
            new_req = new_reqs[idx]
            
            # Find the existing requirement from its ID
            existing_req_row = existing_reqs[existing_reqs['requirement_id'] == dup_info['duplicate_id']]
            if existing_req_row.empty:
                continue
                
            existing_req = existing_req_row.iloc[0]
            
            llm_check_candidates.append({
                'idx': idx,
                'new_req': new_req,
                'existing_req': existing_req,
                'similarity': dup_info['similarity']
            })
    
    if not llm_check_candidates:
        logger.info("No requirements need LLM verification", extra={"icon": "ℹ️"})
        return vector_duplicates
    
    # Process in batches to avoid overloading the LLM
    batch_size = 3
    verified_duplicates = {k: v for k, v in vector_duplicates.items() if not v.get('check_llm', True)}
    
    for i in range(0, len(llm_check_candidates), batch_size):
        batch = llm_check_candidates[i:i+batch_size]
        
        # Prepare the prompt for the LLM
        prompt = """
You are an expert requirements analyst. You need to determine whether the new requirements are duplicates of existing requirements.

For each pair, analyze both requirements carefully and determine if they express the same functional or non-functional requirement, 
even if the wording is different. Focus on the actual requirement, not the surrounding context.

A duplicate means they specify the same thing that needs to be implemented or tested. They are semantically equivalent.
They are NOT duplicates if they specify different features, behaviors, or constraints, even if they are related.

Below are pairs of requirements to analyze:
"""
        
        for j, candidate in enumerate(batch):
            new_req_desc = candidate['new_req'].get('description', '')
            existing_req_desc = candidate['existing_req'].get('description', '')
            
            prompt += f"\n**Pair {j+1}**\n"
            prompt += f"Existing Requirement ID: {candidate['existing_req'].get('requirement_id', 'unknown')}\n"
            prompt += f"Existing Requirement: {existing_req_desc}\n\n"
            prompt += f"New Requirement: {new_req_desc}\n"
            prompt += f"Vector Similarity: {candidate['similarity']:.4f}\n"
            prompt += f"---\n"
        
        # Query the LLM
        try:
            response = agent.capture(LLMDuplicateCheck, prompt)
            
            # Process the response
            if len(response.requirements) != len(batch):
                logger.warning(f"LLM returned {len(response.requirements)} results for {len(batch)} candidates", extra={"icon": "⚠️"})
            
            # Update the duplicates dict with LLM results
            for j, result in enumerate(response.requirements):
                if j >= len(batch):
                    break
                    
                candidate = batch[j]
                idx = candidate['idx']
                existing_req = candidate['existing_req']
                
                if result.is_duplicate:
                    verified_duplicates[idx] = {
                        'duplicate_id': existing_req.get('requirement_id', f"req_{existing_req.name}"),
                        'document_id': existing_req.get('document_id', 'unknown'),
                        'similarity': candidate['similarity'],
                        'existing_text': existing_req.get('description', ''),
                        'llm_verified': True,
                        'llm_confidence': result.confidence,
                        'llm_rationale': result.rationale
                    }
                    logger.info(f"LLM verified requirement {idx} as duplicate (confidence: {result.confidence:.2f})", extra={"icon": "✅"})
                else:
                    logger.info(f"LLM determined requirement {idx} is NOT a duplicate", extra={"icon": "❌"})
                    # Do not add to verified_duplicates
                    
        except Exception as e:
            logger.error(f"Error during LLM verification: {e}", extra={"icon": "❌"})
            # For errors, conservatively don't mark as duplicate
    
    return verified_duplicates

def check_requirements_duplicates(new_requirements: List[Dict], document_id: str = None) -> Tuple[List[Dict], Dict[int, Dict]]:
    """
    Main function to check for duplicates among new requirements.
    
    Args:
        new_requirements: List of dictionaries containing new requirements
        document_id: Optional ID of the document these requirements are from
        
    Returns:
        Tuple of (unique_requirements, duplicate_info) where:
            - unique_requirements is a list of requirements that are not duplicates
            - duplicate_info is a dictionary with information about duplicate requirements
    """
    start_time = time.time()
    
    if not new_requirements:
        logger.warning("No new requirements provided", extra={"icon": "⚠️"})
        return [], {}
    
    # Connect to LanceDB
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    lancedb_path = os.path.join(project_root, OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME)
    
    db = connect_to_lancedb(lancedb_path)
    
    if not db:
        logger.warning("Could not connect to LanceDB. Treating all requirements as new.", extra={"icon": "⚠️"})
        return new_requirements, {}
    
    # Load existing requirements for comparison
    existing_reqs = load_existing_requirements(db)
    
    if existing_reqs.empty:
        logger.info("No existing requirements found. All are new.", extra={"icon": "ℹ️"})
        return new_requirements, {}
    
    # Perform vector-based duplicate detection
    vector_duplicates = check_vector_duplicates(new_requirements, existing_reqs)
    
    # Verify with LLM if needed
    if vector_duplicates:
        verified_duplicates = check_llm_duplicates(new_requirements, existing_reqs, vector_duplicates)
    else:
        verified_duplicates = {}
    
    # Create list of unique requirements (those not in duplicates)
    unique_reqs = []
    for idx, req in enumerate(new_requirements):
        if idx not in verified_duplicates:
            # Add reference to original document even for unique requirements
            if document_id:
                req['document_id'] = document_id
            unique_reqs.append(req)
        else:
            logger.info(f"Skipping duplicate requirement {idx}", extra={"icon": "⏭️"})
    
    # Add reference to duplicates in unique requirements
    for idx, dup_info in verified_duplicates.items():
        # Find a unique requirement ID if duplicate_id format is known
        dup_id = dup_info.get('duplicate_id')
        
        # Add a reference field to the duplicate requirement
        if idx < len(new_requirements):
            new_requirements[idx]['is_duplicate'] = True
            new_requirements[idx]['duplicate_of'] = dup_id
            
            # If document_id is provided, add it
            if document_id:
                new_requirements[idx]['document_id'] = document_id
    
    end_time = time.time()
    logger.info(f"Duplicate check completed in {end_time - start_time:.2f} seconds", extra={"icon": "🏁"})
    logger.info(f"Found {len(verified_duplicates)} duplicates out of {len(new_requirements)} requirements", extra={"icon": "📊"})
    
    return unique_reqs, verified_duplicates

if __name__ == "__main__":
    logger.info("This script is designed to be imported and used by the requirements extraction pipeline.", extra={"icon": "ℹ️"})
    logger.info("It checks for duplicates before adding new requirements to the database.", extra={"icon": "ℹ️"}) 