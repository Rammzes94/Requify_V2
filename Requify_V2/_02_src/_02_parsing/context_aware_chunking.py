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

# Import deduplication module
from _03_docs_deduplication import pre_save_deduplication as dedup

# Load environment variables
load_dotenv()

# Setup logging with script prefix
class ScriptLogger(logging.LoggerAdapter):
    def __init__(self, logger, prefix):
        super().__init__(logger, {})
        self.prefix = prefix
        
    def process(self, msg, kwargs):
        return f"{self.prefix}{msg}", kwargs

logger = ScriptLogger(_00_utils.setup_logging(), "[Context_Chunking] ")

# Constants
MAX_CHAR_SIZE = 900  # Maximum allowed character size
TARGET_CHAR_SIZE = 700  # Target character size per chunk
MAX_SECTION_SIZE = 30000  # Maximum section size for processing with LLM
MAX_RETRIES = 2  # Maximum number of retries for LLM calls
SIMILARITY_THRESHOLD = 0.85  # Lowered threshold for considering content similar
DUPLICATE_THRESHOLD = 0.995  # High threshold for automatic duplicates without LLM

OUTPUT_DIR_BASE = "_03_output"
LANCEDB_SUBDIR_NAME = "lancedb"
CHUNKS_TABLE_NAME = "document_chunks"

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct" # Use existing embedding model from reference chunks
EMBEDDING_DEVICE = "cpu" # Force CPU usage to avoid MPS memory issues
EMBEDDING_MAX_SEQ_LENGTH = 256 # Reduce sequence length to save memory
EMBEDDING_BATCH_SIZE = 8 # Control batch size for memory management


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
    """
    Compute similarity between two chunks using a combination of measures.
    """
    # Simple cosine similarity using sentence-transformers would be more accurate but slower
    # For now, use a simpler approach with difflib for demo purposes
    return difflib.SequenceMatcher(None, chunk1, chunk2).ratio()

def evaluate_chunk_pair(new_chunk: str, old_chunk: str) -> Dict[str, Any]:
    """
    Use an LLM to evaluate a pair of chunks and determine which one to keep.
    
    Args:
        new_chunk: Text of the new chunk
        old_chunk: Text of the existing chunk
        
    Returns:
        Dictionary with decision and reasoning
    """
    import difflib
    import re
    
    # Generate a unified diff for visualization
    diff = list(difflib.ndiff(old_chunk.splitlines(), new_chunk.splitlines()))
    diff_text = "\n".join(diff)
    
    evaluation_prompt = """
    You are an expert document reviewer comparing two versions of technical content.
    Your task is to determine which version should be kept in the database.
    
    Review the chunks carefully and choose ONE of these options:
    
    1. 'keep_new' - Choose this ONLY if:
       - The new chunk contains meaningful ADDITIONAL information not present in the old chunk
       - The new chunk adds clarity without changing specifications or requirements
       - The old chunk has clear errors or omissions that are fixed in the new version
       - There is NEW content like additional steps, contact info, or safety information
       
    2. 'keep_old' - Choose this ONLY if:
       - The old chunk contains IMPORTANT information that is REMOVED in the new chunk
       - The new chunk REMOVES critical requirements, specifications, or safety information
       - The new chunk might introduce errors or ambiguities not present in the old version
       - The new chunk has DUPLICATE sections within itself
    
    3. 'need_user_input' - Choose this ONLY when:
       - Numerical specifications are CHANGED (not just reformatted)
       - Material requirements are substantively altered in meaning
       - Technical specifications are changed in a way that affects behavior or compliance
    
    IMPORTANT EVALUATION RULES:
    - Consider these as DUPLICATES (keep_old) if only formatting changes:
      * List format changed to table format with same data
      * Whitespace, indentation, or bullet style changes
      * Text reordering without content change
      * Paragraph merging or splitting without content change
      * Same meaning expressed with different words/synonyms
      * Units displayed differently but with equivalent values (e.g., "10 cm" vs "100 mm")
      * Numbers formatted differently (e.g., "1000" vs "1,000" vs "1.0 × 10³")
    
    - Look for NUMERICAL VALUE CHANGES specifically:
      * Different measurements, dimensions, weights, or capacities
      * Different tolerance ranges or thresholds
      * Changed version numbers, model numbers, or part numbers
      * Changed timelines, durations, or frequencies
      * Any numbers that affect physical properties, performance, or compliance
    
    - ADDITION OF NEW CONTENT should usually result in 'keep_new' unless the new content is redundant or contradictory
    
    - ADDED ITEMS such as:
      * New steps in a procedure
      * Additional contact information
      * New safety warnings
      * Extra details about operation
      Should always lead to 'keep_new' decision
      
    - Always check for REMOVED content - a critical security measure or specification 
      that's removed should result in choosing 'keep_old', BUT if non-critical info
      is removed while important NEW info is added, choose 'keep_new'
      
    - NUMERICAL CHANGES to specifications like dimensions, weights, capacities, 
      intervals, or version numbers should almost always trigger 'need_user_input'
      
    - Be careful to distinguish between COSMETIC changes (keep_old) and SUBSTANTIVE 
      changes (need_user_input)
    
    OLD CHUNK:
    ```
    {old_chunk}
    ```
    
    NEW CHUNK:
    ```
    {new_chunk}
    ```
    
    DIFF (showing changes):
    ```
    {diff_text}
    ```
    
    Return your decision and reasoning in this exact format:
    Decision: [keep_new/keep_old/need_user_input]
    Reason: [clear explanation of your reasoning]
    Differences: [list the key differences that influenced your decision]
    """
    
    agent = Agent(
        model=decision_llm,  
        markdown=True,
        debug_mode=False,
        response_model=ChunkDecisionModel,
        description=evaluation_prompt,
        use_json_mode=True
    )
    
    prompt_content = f"""
    # OLD CHUNK:
    ```
    {old_chunk}
    ```
    
    # NEW CHUNK:
    ```
    {new_chunk}
    ```
    
    # DIFFERENCES (DIFF):
    ```
    {diff_text}
    ```
    
    Based on these chunks, make your decision about which to keep.
    Pay special attention to:
    1. Whether content is actually DIFFERENT or just REFORMATTED
    2. If numerical specs or values have changed (needs user input)
    3. If IMPORTANT content was removed (keep old version) 
    4. If VALUABLE content was added (keep new version)
    5. If sections were just reordered (treat as duplicate)
    6. If formatting changed but content is semantically equivalent (treat as duplicate)
    7. If units of measurement changed but the actual values are equivalent
    8. If there are numerical differences that affect requirements or specifications
    """
    
    try:
        response = agent.run(prompt_content)
        _00_utils.update_token_counters(response)
        
        data = response.content
        
        if isinstance(data, ChunkDecisionModel):
            logger.info(f"LLM decision: {data.decision} - {data.reason[:100]}...", extra={"icon": "🤔"})
            return {
                "decision": data.decision,
                "reason": data.reason,
                "differences": data.differences or []
            }
        elif isinstance(data, dict):
            logger.info(f"LLM decision: {data.get('decision')} - {data.get('reason', '')[:100]}...", extra={"icon": "🤔"})
            return data
        else:
            logger.error(f"Unexpected response format from LLM", extra={"icon": "❌"})
            # Default to keeping new if we can't parse the response
            return {
                "decision": "keep_new",
                "reason": "Error parsing LLM response - defaulting to keeping new content",
                "differences": []
            }
            
    except Exception as e:
        logger.error(f"Error during chunk evaluation: {e}", extra={"icon": "❌"})
        # Default to keeping new if there's an error
        return {
            "decision": "keep_new",
            "reason": f"Error during evaluation: {str(e)} - defaulting to keeping new content",
            "differences": []
        }

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
    print("\n" + "="*80)
    print(f"CHUNK COMPARISON NEEDED")
    print("="*80)
    
    print(f"\nREASON: {reason}")
    
    if differences:
        print("\nKEY DIFFERENCES:")
        for i, diff in enumerate(differences, 1):
            print(f"  {i}. {diff}")
    
    print(f"\nOLD DOCUMENT: {old_doc_id}")
    print("-" * 40)
    print(old_chunk)
    print("-" * 40)
    
    print(f"\nNEW DOCUMENT: {new_doc_id}")
    print("-" * 40)
    print(new_chunk)
    print("-" * 40)
    
    while True:
        choice = input("\nWhich version would you like to keep? (1=old, 2=new): ")
        if choice == '1':
            return "keep_old"
        elif choice == '2':
            return "keep_new"
        else:
            print("Invalid choice. Please enter 1 or 2.")

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
    
    print("\n" + "="*80)
    print(f"CHUNK COMPARISON NEEDED")
    print("="*80)
    
    print(f"\nREASON: {reason}")
    
    if differences:
        print("\nKEY DIFFERENCES:")
        for i, diff in enumerate(differences, 1):
            print(f"  {i}. {diff}")
    
    print(f"\nOLD DOCUMENT: {old_doc_id}")
    print("-" * 40)
    print(old_chunk_text)
    print("-" * 40)
    
    print(f"\nNEW DOCUMENT: {new_doc_id}")
    print("-" * 40)
    print(new_chunk)
    print("-" * 40)
    
    while True:
        choice = input("\nWhich version would you like to keep? (1=old, 2=new): ")
        if choice == '1':
            return "keep_old"
        elif choice == '2':
            return "keep_new"
        else:
            print("Invalid choice. Please enter 1 or 2.")

# ------------------------------------------------------------------------------
# Main Processing Logic
# ------------------------------------------------------------------------------
def process_document_with_context(
    document_text: str,
    document_id: str,
    similar_doc_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Process a document with context-aware chunking and intelligent deduplication.
    
    Args:
        document_text: Full text of the document
        document_id: ID of the document being processed
        similar_doc_id: ID of a similar document for context (if available)
        
    Returns:
        List of processed chunk data (ready for database insertion)
    """
    logger.info(f"Processing document {document_id} with context-aware chunking", extra={"icon": "🚀"})
    
    start_time = time.time()
    
    # Step 1: Get reference chunks if a similar document is provided
    reference_chunks = []
    if similar_doc_id:
        reference_chunks = get_similar_document_chunks(document_id, similar_doc_id)
    
    # Step 2: Perform context-aware chunking
    chunks = context_aware_chunking(document_text, document_id, reference_chunks)
    
    if not chunks:
        logger.warning(f"No chunks were generated for document {document_id}", extra={"icon": "⚠️"})
        return []
    
    # Step 3: Process each chunk and compare with existing chunks
    from sentence_transformers import SentenceTransformer
    import datetime
    
    # Initialize embedding model for comparison
    logger.info("Loading embedding model", extra={"icon": "🧠"})

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE)
    
    # Additional memory optimizations
    embedding_model.max_seq_length = EMBEDDING_MAX_SEQ_LENGTH  # Reduce sequence length
    
    processed_chunks = []
    duplicates_found = 0
    updates_kept = 0
    user_decisions = 0
    
    # Process chunks in batches to manage memory usage
    chunk_batches = [chunks[i:i + EMBEDDING_BATCH_SIZE] for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE)]
    
    for batch_idx, chunk_batch in enumerate(chunk_batches):
        logger.info(f"Processing batch {batch_idx+1}/{len(chunk_batches)} ({len(chunk_batch)} chunks)", extra={"icon": "🔄"})
        
        # Compute embeddings for the entire batch at once
        batch_embeddings = embedding_model.encode(chunk_batch, normalize_embeddings=True)
        
        for i, (chunk_text, embedding) in enumerate(zip(chunk_batch, batch_embeddings)):
            # Original chunk index in the full list
            chunk_idx = batch_idx * EMBEDDING_BATCH_SIZE + i
            
            # Generate chunk metadata
            chunk_id = f"{document_id}_chunk_{chunk_idx+1}"
            chunk_hash = generate_chunk_hash(chunk_text)
            # Convert embedding to list if it's a numpy array
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            token_count = len(chunk_text) // 4  # Simple approximation
            
            # Check for similar existing chunks
            similar_chunk = None
            highest_similarity = 0.0
            
            # First pass: look for exact hash matches (most efficient)
            if reference_chunks:
                for ref_chunk in reference_chunks:
                    # Check hash-based exact match first
                    if ref_chunk.get('chunk_hash') == chunk_hash:
                        logger.info(f"Exact hash match for chunk {chunk_idx+1}", extra={"icon": "🔍"})
                        duplicates_found += 1
                        # Skip this chunk entirely - it's an exact duplicate
                        similar_chunk = None
                        break
                
                # If no exact match, look for similar chunks
                if similar_chunk is None:
                    for ref_chunk in reference_chunks:
                        # Check embedding similarity
                        ref_embedding = ref_chunk.get('embedding')
                        if ref_embedding is not None and isinstance(ref_embedding, (list, np.ndarray)):
                            # Convert to numpy arrays for dot product
                            if isinstance(ref_embedding, list):
                                ref_embedding = np.array(ref_embedding)
                            current_embedding = np.array(embedding)
                            
                            # Calculate cosine similarity
                            similarity = np.dot(ref_embedding, current_embedding) / (
                                np.linalg.norm(ref_embedding) * np.linalg.norm(current_embedding)
                            )
                            
                            if similarity > highest_similarity:
                                highest_similarity = similarity
                                similar_chunk = ref_chunk
            
            # Now evaluate if the similar chunk warrants keeping or replacing
            if similar_chunk and highest_similarity >= SIMILARITY_THRESHOLD:
                logger.info(f"Similar chunk found for chunk {chunk_idx+1} (similarity: {highest_similarity:.4f})", extra={"icon": "🔍"})
                
                # If it's not an exact duplicate but similar, use LLM to evaluate
                if highest_similarity < DUPLICATE_THRESHOLD:
                    # Have the LLM evaluate the chunks
                    eval_result = evaluate_chunk_pair(chunk_text, similar_chunk.get('chunk_text', ''))
                    
                    decision = eval_result.get('decision')
                    if decision == "keep_old":
                        logger.info(f"Decision: Keep old chunk {similar_chunk.get('chunk_id')} - {eval_result.get('reason', '')[:100]}...", extra={"icon": "⏩"})
                        # Skip adding this chunk to processed_chunks - we'll keep the old one
                        continue
                    elif decision == "need_user_input":
                        # This should be rare - ask the user to decide
                        user_decisions += 1
                        user_decision = prompt_user_for_chunk_decision(
                            chunk_text, 
                            similar_chunk,
                            document_id,
                            eval_result
                        )
                        
                        if user_decision == "keep_old":
                            logger.info(f"User decision: Keep old chunk {similar_chunk.get('chunk_id')}", extra={"icon": "👤"})
                            # Skip adding this chunk
                            continue
                        else:
                            logger.info(f"User decision: Keep new chunk (replacing old)", extra={"icon": "👤"})
                            updates_kept += 1
                    else:  # keep_new by default
                        logger.info(f"Decision: Keep new chunk - {eval_result.get('reason', '')[:100]}...", extra={"icon": "✅"})
                        updates_kept += 1
                else:
                    # Very high similarity but not exact hash match - likely just formatting differences
                    if highest_similarity > DUPLICATE_THRESHOLD:
                        # If it's a very high match, treat as duplicate without LLM evaluation
                        logger.info(f"Chunk {chunk_idx+1} nearly identical to existing chunk - skipping", extra={"icon": "⏩"})
                        continue
                        
                    # If similarity is high but not quite duplicate threshold, check for formatting or reordering
                    if SIMILARITY_THRESHOLD <= highest_similarity < DUPLICATE_THRESHOLD:
                        # Extract text normalized for comparison
                        normalized_old = ' '.join(similar_chunk.get('chunk_text', '').lower().split())
                        normalized_new = ' '.join(chunk_text.lower().split())
                        
                        # Check for simple reformatting cases: same words in different format
                        if sorted(normalized_old.split()) == sorted(normalized_new.split()):
                            logger.info(f"Chunk {chunk_idx+1} has same content in different order - treating as duplicate", extra={"icon": "⏩"})
                            continue
                            
                        # Check for whitespace/bullet differences - replacing bullet types and normalizing whitespace
                        normalized_old_content = re.sub(r'[-*•◦▪▫]', '-', normalized_old).replace('\n', ' ').replace('\t', ' ')
                        normalized_new_content = re.sub(r'[-*•◦▪▫]', '-', normalized_new).replace('\n', ' ').replace('\t', ' ')
                        
                        # Enhanced normalization to catch more formatting variants
                        # Remove common punctuation and normalize spacing
                        normalized_old_content = re.sub(r'[,.;:()[\]{}]', ' ', normalized_old_content)
                        normalized_old_content = re.sub(r'\s+', ' ', normalized_old_content).strip()
                        normalized_new_content = re.sub(r'[,.;:()[\]{}]', ' ', normalized_new_content)
                        normalized_new_content = re.sub(r'\s+', ' ', normalized_new_content).strip()
                        
                        # Normalize number formats (1,000 to 1000, etc.)
                        normalized_old_content = re.sub(r'(\d),(\d)', r'\1\2', normalized_old_content)
                        normalized_new_content = re.sub(r'(\d),(\d)', r'\1\2', normalized_new_content)
                        
                        # Normalize units for comparison (10mm to 10 mm, etc.)
                        normalized_old_content = re.sub(r'(\d)([a-zA-Z]+)', r'\1 \2', normalized_old_content)
                        normalized_new_content = re.sub(r'(\d)([a-zA-Z]+)', r'\1 \2', normalized_new_content)
                        
                        # Check for list vs table format - if content words are mostly the same
                        # Filter out common stopwords to focus on meaningful content words
                        stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when', 'at', 'from', 'by', 'for', 'with', 'about', 'to', 'in', 'on', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'of', 'this', 'that', 'these', 'those'}
                        old_words = set(word for word in re.findall(r'\b\w+\b', normalized_old_content.lower()) if word not in stopwords and not word.isdigit())
                        new_words = set(word for word in re.findall(r'\b\w+\b', normalized_new_content.lower()) if word not in stopwords and not word.isdigit())
                        
                        # Calculate word overlap ratio
                        if old_words and new_words:
                            # Calculate two different overlap metrics
                            overlap_ratio = len(old_words.intersection(new_words)) / max(len(old_words), len(new_words))
                            jaccard_similarity = len(old_words.intersection(new_words)) / len(old_words.union(new_words))
                            
                            # Get all numeric values from both texts for comparison
                            old_numbers = set(re.findall(r'\b\d+\.?\d*\b', normalized_old_content))
                            new_numbers = set(re.findall(r'\b\d+\.?\d*\b', normalized_new_content))
                            
                            # If the chunk has numbers, make sure they're the same
                            numbers_match = True
                            if old_numbers or new_numbers:
                                if old_numbers != new_numbers:
                                    numbers_match = False
                            
                            # Higher threshold than before (0.85 -> 0.90 for overlap ratio)
                            # But also check Jaccard similarity as an additional metric
                            if (overlap_ratio > 0.90 or jaccard_similarity > 0.80) and numbers_match:
                                logger.info(f"Chunk {chunk_idx+1} has high word overlap ({overlap_ratio:.2f}, jaccard {jaccard_similarity:.2f}) with existing - treating as duplicate", extra={"icon": "⏩"})
                                continue
        
        # Add the new chunk to our processed list
        chunk_record = {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "chunk_index": chunk_idx,
            "start_offset": 0,  # Simplified - would track actual positions in real system
            "end_offset": len(chunk_text),
            "chunk_text": chunk_text,
            "token_count": token_count,
            "embedding": embedding,
            "chunk_hash": chunk_hash,
            "is_duplicate": False,
            "duplicate_of": "",
            "is_updated": similar_chunk is not None and highest_similarity >= SIMILARITY_THRESHOLD,
            "previous_chunk_id": similar_chunk.get('chunk_id', '') if similar_chunk and highest_similarity >= SIMILARITY_THRESHOLD else "",
            "timestamp": datetime.datetime.now().isoformat(),
            "aligned_with_chunk_id": similar_chunk.get('chunk_id', '') if similar_chunk and highest_similarity >= SIMILARITY_THRESHOLD else "",
            "aligned_with_document_id": similar_chunk.get('document_id', '') if similar_chunk and highest_similarity >= SIMILARITY_THRESHOLD else "",
            "is_duplicate_marker": False  # Add this field for all normal chunks
        }
        processed_chunks.append(chunk_record)
        
        if similar_chunk and highest_similarity >= SIMILARITY_THRESHOLD:
            logger.info(f"Chunk {chunk_idx+1} added as update to existing chunk {similar_chunk.get('chunk_id')}", extra={"icon": "🔄"})
        else:
            logger.info(f"Chunk {chunk_idx+1} added as new chunk", extra={"icon": "✅"})
    
    end_time = time.time()
    logger.info(f"Document processing completed in {end_time - start_time:.2f} seconds", extra={"icon": "⏱️"})
    logger.info(
        f"Generated {len(processed_chunks)} chunks ({duplicates_found} duplicates skipped, {updates_kept} updates kept, {user_decisions} user decisions)",
        extra={"icon": "📊"}
    )
    
    # Handle case where all chunks are duplicates
    if len(chunks) > 0 and len(processed_chunks) == 0:
        logger.info(f"All {len(chunks)} chunks from document {document_id} are exact duplicates of existing chunks", extra={"icon": "🔄"})
        
        if similar_doc_id:
            logger.info(f"Document {document_id} appears to be a duplicate of {similar_doc_id}", extra={"icon": "⏩"})
            
            # Create a single marker chunk to indicate this is a complete duplicate
            # This helps with traceability while avoiding redundant storage
            marker_chunk = {
                "chunk_id": f"{document_id}_duplicate_marker",
                "document_id": document_id,
                "chunk_index": 0,
                "start_offset": 0,
                "end_offset": 0,
                "chunk_text": f"This document is a complete duplicate of {similar_doc_id}",
                "token_count": 0,
                "embedding": np.zeros(len(embedding)).tolist(),  # Empty embedding
                "chunk_hash": "duplicate_document_marker",
                "is_duplicate": True,
                "duplicate_of": similar_doc_id,
                "is_updated": False,
                "previous_chunk_id": "",
                "timestamp": datetime.datetime.now().isoformat(),
                "aligned_with_chunk_id": "",
                "aligned_with_document_id": similar_doc_id,
                "is_duplicate_marker": True  # Special flag to identify this as a marker chunk
            }
            processed_chunks.append(marker_chunk)
    
    return processed_chunks

def save_chunks_to_db(chunks: List[Dict[str, Any]]) -> bool:
    """
    Save processed chunks to the LanceDB chunks table.
    
    Args:
        chunks: List of chunk records to save
        
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
            
            # Check if need to add the is_duplicate_marker column
            table_schema = chunks_table.schema
            has_duplicate_marker = any(field.name == 'is_duplicate_marker' for field in table_schema)
            
            if not has_duplicate_marker:
                # Create new dataframe with updated schema
                logger.info("Adding is_duplicate_marker field to existing records", extra={"icon": "🔄"})
                df = chunks_table.to_pandas()
                df['is_duplicate_marker'] = False
                
                # Drop and recreate table
                db.drop_table(CHUNKS_TABLE_NAME)
                chunks_table = db.create_table(CHUNKS_TABLE_NAME, data=df)
        else:
            # First chunk added - create new table
            if chunks:
                logger.info(f"Creating new table: {CHUNKS_TABLE_NAME}", extra={"icon": "✅"})
                data = pd.DataFrame(chunks)
                
                # Ensure the is_duplicate_marker field exists
                if 'is_duplicate_marker' not in data.columns:
                    data['is_duplicate_marker'] = False
                    
                chunks_table = db.create_table(CHUNKS_TABLE_NAME, data=data)
                return True
            else:
                logger.warning("No chunks to save, table not created", extra={"icon": "⚠️"})
                return True
        
        # Add new chunks to the table
        if chunks:
            data = pd.DataFrame(chunks)
            chunks_table.add(data)
            logger.info(f"Added {len(chunks)} chunks to database", extra={"icon": "✅"})
        
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
        chunks = process_document_with_context(document_text, document_id, similar_document_id)
        
        if not chunks:
            logger.warning(f"No chunks were generated for document {document_id}", extra={"icon": "⚠️"})
            return False
        
        # Step 2: Save chunks to database
        if chunks:
            success = save_chunks_to_db(chunks)
        else:
            # If there are no chunks to save because all were duplicates, 
            # consider this a success (proper deduplication)
            if duplicates_found > 0:
                logger.info(f"All chunks ({duplicates_found}) were detected as duplicates - no new chunks to save", extra={"icon": "✅"})
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