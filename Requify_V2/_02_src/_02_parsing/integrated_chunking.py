"""
integrated_chunking.py

This script implements a simplified approach to document chunking using LLMs.
It directly asks the model to chunk the text while preserving semantic coherence,
and handles large documents by processing them in manageable sections.
"""

import os
import sys
import json
import logging
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.openai import OpenAIChat


# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import _00_utils
_00_utils.setup_project_directory()

# Load environment variables
load_dotenv()

# Define constants
MAX_CHAR_SIZE = 900  # Maximum allowed character size
TARGET_CHAR_SIZE = 700  # Target character size per chunk
MAX_SECTION_SIZE = 30000  # Maximum section size for processing with LLM
MAX_RETRIES = 2  # Maximum number of retries for LLM calls
INITIAL_RETRY_DELAY = 1  # Initial delay in seconds for retry backoff

# Setup logging with script prefix
class ScriptLogger(logging.LoggerAdapter):
    def __init__(self, logger, prefix):
        super().__init__(logger, {})
        self.prefix = prefix
        
    def process(self, msg, kwargs):
        return f"{self.prefix}{msg}", kwargs

logger = ScriptLogger(_00_utils.setup_logging(), "[Integrated_Chunking] ")

# Get OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI model
openai_text_model = OpenAIChat(id="gpt-4o-mini", api_key=api_key)

# Pydantic model for LLM output validation
class ChunksOutputModel(BaseModel):
    chunks: List[str] = Field(
        ..., description="List of text chunks that preserve semantic coherence."
    )

# Simplified chunking prompt
chunking_prompt = """
chunk the following text, preserving content that goes well together. a chunk shall have no more than {max_size} characters.
Format your response as: {{"chunks": ["chunk1", "chunk2", ...]}}
Do not add ANY extra text that is not in the original input text.
"""

def approx_token_count(text: str) -> int:
    """Approximate token count based on character length (1 token ~= 4 characters)"""
    return len(text) // 4

def split_large_document(md_text: str) -> List[str]:
    """Split very large documents into processable sections."""
    if len(md_text) <= MAX_SECTION_SIZE:
        return [md_text]
        
    logger.info(f"🔄 Document exceeds size threshold ({len(md_text)} chars), performing initial splitting")
    
    # First try to split by double newlines (paragraphs)
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

def get_chunks_from_llm(md_text: str) -> List[str]:
    """Ask the LLM to chunk the text directly with retry logic."""
    if not md_text.strip():
        return []
        
    logger.info(f"🔄 Asking LLM to chunk text of {len(md_text)} characters")
    
    prompt = chunking_prompt.format(max_size=MAX_CHAR_SIZE)
    agent = Agent(
        model=openai_text_model,
        markdown=True,
        debug_mode=False,
        response_model=ChunksOutputModel,
        description=prompt,
        use_json_mode=True
    )
    
    retry_count = 0
    chunks = []
    
    while retry_count <= MAX_RETRIES:
        try:
            # If this is a retry, log it
            if retry_count > 0:
                logger.info(f"🔄 Retry #{retry_count} - Asking LLM to chunk text again")
            
            response = agent.run(md_text)
            _00_utils.update_token_counters(response)
            
            data = response.content
            
            if isinstance(data, ChunksOutputModel):
                chunks = data.chunks
                break  # Success, exit the retry loop
            elif isinstance(data, str):
                try:
                    parsed_data = json.loads(data)
                    if 'chunks' in parsed_data:
                        chunks = parsed_data['chunks']
                        break  # Success, exit the retry loop
                    else:
                        logger.error(f"❌ LLM returned JSON without 'chunks' key: {data[:100]}...")
                except json.JSONDecodeError:
                    logger.error(f"❌ LLM returned invalid JSON string: {data[:100]}...")
            elif isinstance(data, dict):
                if 'chunks' in data:
                    chunks = data['chunks']
                    break  # Success, exit the retry loop
                else:
                    logger.error(f"❌ LLM returned dict without 'chunks' key: {str(data)[:100]}...")
            else:
                logger.error(f"❌ LLM returned unexpected data type: {type(data)}")
            
            # If we haven't broken out of the loop, the response wasn't valid
            retry_count += 1
            
            if retry_count <= MAX_RETRIES:
                # Calculate exponential backoff delay: 1s, then 3s
                delay = INITIAL_RETRY_DELAY * (2 ** (retry_count - 1))
                logger.info(f"⏱️ Waiting {delay} seconds before retry...")
                time.sleep(delay)
            else:
                logger.error(f"❌ All {MAX_RETRIES} retries failed. Returning empty chunk list.")
                
        except Exception as e:
            logger.error(f"❌ Error getting chunks from LLM: {e}", exc_info=True)
            retry_count += 1
            
            if retry_count <= MAX_RETRIES:
                delay = INITIAL_RETRY_DELAY * (2 ** (retry_count - 1))
                logger.info(f"⏱️ Waiting {delay} seconds before retry...")
                time.sleep(delay)
            else:
                logger.error(f"❌ All {MAX_RETRIES} retries failed due to exceptions. Returning empty chunk list.")
                return []
    
    # Filter out empty chunks
    chunks = [chunk for chunk in chunks if chunk.strip()]
    logger.info(f"✅ LLM provided {len(chunks)} chunks")
    
    # Check if any chunk exceeds the maximum size and needs re-chunking
    oversized_chunks = [i for i, chunk in enumerate(chunks) if len(chunk) > MAX_CHAR_SIZE]
    if oversized_chunks:
        logger.info(f"⚠️ Found {len(oversized_chunks)} oversized chunks, re-chunking them")
        new_chunks = []
        for i, chunk in enumerate(chunks):
            if i in oversized_chunks:
                # Re-chunk the oversized chunk
                sub_chunks = get_chunks_from_llm(chunk)
                new_chunks.extend(sub_chunks)
            else:
                new_chunks.append(chunk)
        chunks = new_chunks
        logger.info(f"✅ After re-chunking oversized chunks, we now have {len(chunks)} chunks")
    
    return chunks

def chunk_markdown(md_text: str) -> List[str]:
    """Main function to chunk markdown text."""
    if not md_text.strip():
        logger.info("ℹ️ Document is empty. No chunks to create.")
        return []
    
    logger.info(f"🔄 Chunking markdown text of {len(md_text)} characters using simplified LLM approach")
    
    # Handle large documents by splitting into sections first
    document_sections = split_large_document(md_text)
    
    all_chunks = []
    for i, section in enumerate(document_sections):
        logger.info(f"🔄 Processing section {i+1}/{len(document_sections)}")
        section_chunks = get_chunks_from_llm(section)
        all_chunks.extend(section_chunks)
    
    logger.info(f"✅ Completed chunking. Total chunks: {len(all_chunks)}")
    return all_chunks

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

def process_document(document_text: str, document_id: str, document_pages: List[Dict[str, Any]]) -> bool:
    """
    Process a document for chunking and save chunks to the database.
    
    Args:
        document_text: The full text of the document
        document_id: The document identifier
        document_pages: List of page data dictionaries
        
    Returns:
        True if processing was successful, False otherwise
    """
    try:
        start_time = time.time()
        logger.info(f"🚀 Processing document: {document_id}")
        
        # Perform chunking
        chunks = chunk_markdown(document_text)
        
        if not chunks:
            logger.warning(f"⚠️ No chunks generated for document: {document_id}")
            return False
        
        # Analyze the chunks
        analysis = analyze_chunks(chunks)
        logger.info(f"📊 Generated {len(chunks)} chunks")
        logger.info(f"   Min/Max/Avg chars: {analysis['char_sizes']['min']}/{analysis['char_sizes']['max']}/{analysis['char_sizes']['avg']:.1f}")
        logger.info(f"   Min/Max/Avg tokens: {analysis['token_sizes']['min']}/{analysis['token_sizes']['max']}/{analysis['token_sizes']['avg']:.1f}")
        
        # Save chunks to LanceDB
        import lancedb
        from sentence_transformers import SentenceTransformer
        import numpy as np
        import datetime
        
        # Connect to LanceDB
        db_path = os.path.join("_03_output", "lancedb")
        logger.info(f"🔄 Connecting to LanceDB at {db_path}")
        db = lancedb.connect(db_path)
        
        # Try to open existing chunks table
        try:
            chunks_table = db.open_table("document_chunks")
            logger.info(f"✅ Opened existing document_chunks table")
        except Exception as e:
            logger.error(f"❌ Failed to open document_chunks table: {str(e)}")
            return False
        
        # Load the embedding model
        logger.info(f"🔄 Loading embedding model")
        model_name = "intfloat/multilingual-e5-large-instruct"
        embedding_model = SentenceTransformer(model_name)
        
        # Get current timestamp
        current_timestamp = datetime.datetime.now().isoformat()
        
        # Prepare chunk data for insertion
        chunk_data = []
        for i, chunk_text in enumerate(chunks):
            # Generate an embedding for the chunk
            embedding = embedding_model.encode(chunk_text, normalize_embeddings=True).tolist()
            
            # Create unique chunk ID
            chunk_id = f"{document_id}_chunk_{i+1}"
            token_count = approx_token_count(chunk_text)
            
            # For simplified demo, we'll use character indices as offsets
            # In a real system, you'd track actual positions
            start_offset = 0
            end_offset = 0
            if i > 0:
                start_offset = sum(len(chunks[j]) for j in range(i))
            end_offset = start_offset + len(chunk_text)
            
            chunk_record = {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "chunk_index": i,
                "start_offset": start_offset,
                "end_offset": end_offset,
                "chunk_text": chunk_text,
                "token_count": token_count,
                "embedding": embedding,
                "is_duplicate": False,
                "duplicate_of": "",  # Empty string for non-duplicates
                "is_updated": False,
                "previous_chunk_id": "",
                "timestamp": current_timestamp,
                "aligned_with_chunk_id": "",
                "aligned_with_document_id": ""
            }
            chunk_data.append(chunk_record)
        
        # Add the chunks to the table
        logger.info(f"🔄 Adding {len(chunk_data)} chunks to LanceDB")
        chunks_table.add(chunk_data)
        logger.info(f"✅ Chunks successfully saved to LanceDB")
        
        end_time = time.time()
        logger.info(f"✅ Document processing completed in {end_time - start_time:.2f} seconds")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error processing document: {e}", exc_info=True)
        return False

def main():
    """
    Test function for chunking.
    """
    test_file = os.path.join("_01_input", "raw", "sample_md.md")
    if not os.path.exists(test_file):
        logger.error(f"❌ Test file not found: {test_file}")
        return 1
        
    with open(test_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
        
    logger.info(f"📄 Loaded test file: {test_file} ({len(md_content)} characters)")
    
    chunks = chunk_markdown(md_content)
    
    if chunks:
        logger.info("\n" + "="*40 + " CHUNK INSPECTION " + "="*40)
        for i, chunk in enumerate(chunks, 1):
            approx_tokens = approx_token_count(chunk)
            logger.info(f"\n--- Chunk {i} ({len(chunk)} chars, ~{approx_tokens} tokens) ---")
            logger.info(chunk[:100] + "..." if len(chunk) > 100 else chunk)
        logger.info("="*98)
        
        # Create test output
        output_dir = "chunks_output"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "test_chunks.md")
        
        with open(output_file, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks, 1):
                f.write(f"<!-- Chunk {i} - {len(chunk)} chars -->\n")
                f.write(chunk)
                f.write("\n\n---\n\n")
        
        logger.info(f"✅ Wrote test chunks to: {output_file}")
    else:
        logger.info("No chunks were produced.")
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 