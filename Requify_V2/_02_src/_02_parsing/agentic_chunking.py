import os
import sys
import json
import logging
import statistics
import re
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.openai import OpenAIChat


# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import _00_utils
_00_utils.setup_project_directory()

"""
This script implements an agentic chunking system using LLMs to intelligently split markdown text 
into appropriately sized chunks while preserving semantic coherence and maintaining context.
It uses an LLM-centric approach for determining split points.
"""

# Define constants
TARGET_TOKEN_SIZE = 128  # Target token size per chunk
MAX_TOKEN_SIZE = 180  # Maximum allowed token size
MIN_TOKEN_SIZE = 80  # Minimum allowed token size
TOKEN_TO_CHAR_RATIO = 4  # Approximate ratio of characters to tokens

# Large document handling constants
MAX_DOCUMENT_CHARS = 40000  # ~12K tokens threshold for initial splitting
MAX_SECTION_CHARS = 35000   # Maximum section size for processing with LLM

# Derive character targets from token targets
TARGET_CHAR_SIZE = TARGET_TOKEN_SIZE * TOKEN_TO_CHAR_RATIO
MAX_CHAR_SIZE = MAX_TOKEN_SIZE * TOKEN_TO_CHAR_RATIO
MIN_CHAR_SIZE = MIN_TOKEN_SIZE * TOKEN_TO_CHAR_RATIO

# Setup logging with script prefix
class ScriptLogger(logging.LoggerAdapter):
    def __init__(self, logger, prefix):
        super().__init__(logger, {})
        self.prefix = prefix
        
    def process(self, msg, kwargs):
        return f"{self.prefix}{msg}", kwargs

logger = ScriptLogger(_00_utils.setup_logging(), "[Agentic_Chunking] ")

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Define DocumentChunk class for external use
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

# Pydantic model to validate chunking output
class SplitPointsModel(BaseModel):
    split_points: list[int] = Field(
        ..., description="List of character positions where the document should be split."
    )

# Initialize the OpenAI text model for chunking
openai_text_model = OpenAIChat(id="gpt-4.1-mini", api_key=api_key)

# Create a prompt that asks for split points rather than actual chunks
split_points_description = (
    f"You are a document chunking expert. Your task is to identify optimal split points in a markdown document "
    f"that preserve semantic coherence and CRITICALLY IMPORTANT: never break mid-sentence or mid-paragraph unless the paragraph is extremely long."
    f"\n\nINSTRUCTIONS:"
    f"\n1. Analyze the entire document to identify paragraph breaks, section boundaries, and natural split points."
    f"\n2. The document is {{length}} characters long."
    f"\n3. Aim for chunks of approximately {TARGET_CHAR_SIZE} characters each (target)."
    f"\n4. Each chunk MUST be between {MIN_CHAR_SIZE}-{MAX_CHAR_SIZE} characters. This is a strict rule. If a semantic unit is too small, combine it with an adjacent one. If too large, find the best possible split within it that respects sentence boundaries."
    f"\n5. CRITICALLY IMPORTANT: NEVER split in the middle of a sentence. This is the HIGHEST PRIORITY rule. Always split at the end of a complete sentence."

    f"\n6. Respect markdown structure (headers should be kept with at least some of their content that follows)."

    f"\n7. Prefer to split at paragraph breaks (double newlines) whenever possible, as these are natural semantic boundaries."

    f"\n8. If a paragraph is too long to fit within {MAX_CHAR_SIZE} characters, you MUST split it. Find a sentence boundary within that paragraph for the split."

    f"\n9. Ensure each chunk forms a coherent unit that makes sense when read independently."

    f"\n10. Provide a list of character positions (0-indexed) where the document should be split. These are the positions *after* which a new chunk begins. Do not include 0 or the total length of the document as split points."

    f"\n\nGIVE ME ONLY A LIST OF CHARACTER POSITIONS where I should split the document."

    f"\nFor example, if the document has 2000 characters and should be split after character 500 and character 1050, return:"

    f"\n{{\"split_points\": [500, 1050]}}"

)

def approx_token_count(text: str) -> int:
    """
    Approximate token count based on character length.
    This is a rough estimate - 1 token is ~4 characters in English.
    """
    return len(text) // TOKEN_TO_CHAR_RATIO

def split_large_document(md_text: str) -> List[str]:
    """
    Performs initial splitting of very large documents that exceed the LLM's context window.
    """
    if len(md_text) <= MAX_DOCUMENT_CHARS:
        return [md_text]
        
    logger.info(f"🔪 Document exceeds size threshold ({len(md_text)} chars), performing initial splitting")
    
    section_boundaries = []
    header_matches = list(re.finditer(r'(?:\n|^)(#{1,6}\s+[^\n]+)', md_text))
    for match in header_matches[1:]:
        section_boundaries.append(match.start())
    
    paragraph_breaks = list(re.finditer(r'\n\n\n+', md_text))
    for match in paragraph_breaks:
        section_boundaries.append(match.start())
    
    section_boundaries.sort()
    
    sections = []
    start_pos = 0
    current_section_text = ""
    
    for boundary_pos in section_boundaries:
        if len(current_section_text) + (boundary_pos - start_pos) > MAX_SECTION_CHARS and len(current_section_text) > 0:
            sections.append(current_section_text)
            current_section_text = md_text[start_pos:boundary_pos]
        else:
            current_section_text += md_text[start_pos:boundary_pos]
        start_pos = boundary_pos
    
    current_section_text += md_text[start_pos:]
    if current_section_text:
        sections.append(current_section_text)
    
    final_sections = []
    for sec in sections:
        if len(sec) <= MAX_SECTION_CHARS:
            final_sections.append(sec)
        else:
            subsections = re.split(r'\n\n', sec)
            current_subsection_text = ""
            for sub_sec in subsections:
                if len(current_subsection_text) + len(sub_sec) + 2 > MAX_SECTION_CHARS and len(current_subsection_text) > 0: # +2 for \n\n
                    final_sections.append(current_subsection_text.strip())
                    current_subsection_text = sub_sec + "\n\n"
                else:
                    current_subsection_text += sub_sec + "\n\n"
            if current_subsection_text.strip():
                final_sections.append(current_subsection_text.strip())
    
    logger.info(f"✅ Split large document into {len(final_sections)} processable sections")
    return final_sections

def is_sentence_boundary(text: str, pos: int) -> bool:
    """
    More robust check if a position in text is at a sentence boundary.
    Considers ., !, ?, followed by space, newline, or end of text.
    Also handles cases like quotes or parentheses around sentence-ending punctuation.
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
            elif pos + 1 == len(text): # Ends with quote/paren
                return True
        return False
    return False

def find_nearest_sentence_boundary(text: str, target_pos: int, max_search_distance: int = 150) -> int:
    """
    Find the nearest valid sentence boundary to the target_pos.
    Searches backward first, then forward. If no boundary is found, returns -1.
    """
    # Ensure target_pos is within reasonable bounds of the text
    if target_pos < 0 or target_pos > len(text): # Allow target_pos == len(text) for end of doc
        logger.warning(f"Target position {target_pos} is out of text bounds (0-{len(text)}).")
        return -1

    if is_sentence_boundary(text, target_pos):
        return target_pos

    # Search backward
    for i in range(1, max_search_distance + 1):
        bwd_pos = target_pos - i
        if bwd_pos <= 0: # Stop if we go before the start of the text
            break
        if is_sentence_boundary(text, bwd_pos):
            logger.debug(f"Adjusted split from {target_pos} to {bwd_pos} (backward search)")
            return bwd_pos

    # Search forward
    for i in range(1, max_search_distance + 1):
        fwd_pos = target_pos + i
        if fwd_pos >= len(text): # Stop if we go past the end of the text
            break
        if is_sentence_boundary(text, fwd_pos):
            logger.debug(f"Adjusted split from {target_pos} to {fwd_pos} (forward search)")
            return fwd_pos
            
    logger.warning(f"⚠️ Could not find a sentence boundary near {target_pos} within {max_search_distance} chars. Original point problematic.")
    return -1 # Indicate failure to find a boundary

def get_llm_split_suggestions(md_text: str) -> List[int]:
    """
    Ask the LLM to suggest split points for the document.
    """
    logger.info(f"🔄 Asking LLM for suggested split points in a {len(md_text)} character document")
    
    try:
        prompt = split_points_description.replace("{length}", str(len(md_text)))
        temp_agent = Agent(
            model=openai_text_model,
            markdown=True, # The input is markdown
            debug_mode=False,
            response_model=SplitPointsModel,
            description=prompt,
            use_json_mode=True # Expecting JSON output
        )
        
        response = temp_agent.run(md_text)
        _00_utils.update_token_counters(response)
        
        data = response.content
        
        # LLM should directly return the Pydantic model instance if use_json_mode=True and response_model is set
        if isinstance(data, SplitPointsModel):
            split_points = data.split_points
        elif isinstance(data, str): # Fallback if JSON string is returned
            try:
                split_points = json.loads(data).get('split_points', [])
            except json.JSONDecodeError:
                logger.error(f"❌ LLM returned invalid JSON string: {data}")
                return []
        elif isinstance(data, dict): # Fallback if dict is returned
             split_points = data.get('split_points', [])
        else:
            logger.error(f"❌ LLM returned unexpected data type: {type(data)}")
            split_points = []
            
        split_points = [int(p) for p in split_points if isinstance(p, (int, float, str)) and str(p).isdigit()]
        split_points = [p for p in split_points if 0 < p < len(md_text)] # Ensure points are within bounds
        split_points = sorted(list(set(split_points))) # Remove duplicates and sort
        
        logger.info(f"✅ LLM suggested {len(split_points)} split points: {split_points}")
        return split_points
    
    except Exception as e:
        logger.error(f"❌ Error getting LLM split suggestions: {e}", exc_info=True)
        return []

def refine_and_enforce_split_points(md_text: str, llm_split_points: List[int]) -> List[int]:
    """
    Refine LLM split points:
    1. Ensure ALL points are at valid sentence boundaries. Discard if not.
    2. Iterate through sentence-boundary-aligned points to form initial chunks.
    3. If a chunk is too large, try to split it further using LLM or rule based sentence splitting.
    4. If a chunk is too small, consider merging *carefully* with the next, only if the combined chunk is not too large and maintains coherence.
       Prefer smaller valid chunks over forced merges that break semantics or create oversized chunks.
    """
    if not md_text.strip():
        return []

    # 1. Align all LLM points to sentence boundaries. Discard points where no valid boundary can be found.
    aligned_sentence_points = []
    for point in llm_split_points:
        adjusted_point = find_nearest_sentence_boundary(md_text, point)
        if adjusted_point != -1: # -1 indicates no boundary found
            if adjusted_point != point:
                 logger.info(f"🛠️ Adjusted LLM split point from {point} to {adjusted_point} to align with sentence boundary.")
            aligned_sentence_points.append(adjusted_point)
        else:
            logger.warning(f"🗑️ Discarding LLM split point {point} as no valid sentence boundary found nearby.")
    
    aligned_sentence_points = sorted(list(set(p for p in aligned_sentence_points if 0 < p < len(md_text)))) # must be within text
    
    if not aligned_sentence_points:
        logger.warning("⚠️ No valid LLM split points remained after sentence boundary alignment. Document may not be split as intended by LLM.")
        # If the whole document is too large, make a single split.
        if len(md_text) > MAX_CHAR_SIZE:
            logger.info(f"Document is {len(md_text)} chars, exceeds max {MAX_CHAR_SIZE}. Forcing one split in the middle.")
            potential_split = len(md_text) // 2
            adjusted_split = find_nearest_sentence_boundary(md_text, potential_split)
            if adjusted_split != -1 and 0 < adjusted_split < len(md_text):
                return [adjusted_split]
        return [] # Otherwise, return no splits

    logger.info(f"📌 Initial sentence-aligned points: {aligned_sentence_points}")

    # 2. Iterate and adjust for size
    final_splits = []
    current_chunk_start = 0

    for i in range(len(aligned_sentence_points) + 1): # +1 to handle the segment after the last split point
        current_chunk_end = aligned_sentence_points[i] if i < len(aligned_sentence_points) else len(md_text)
        
        # Ensure current_chunk_end is not before or at current_chunk_start
        if current_chunk_end <= current_chunk_start:
            if i == len(aligned_sentence_points) and current_chunk_start < len(md_text): # ensure last piece is captured
                 current_chunk_end = len(md_text)
            else:
                continue

        chunk_text = md_text[current_chunk_start:current_chunk_end]
        chunk_len_chars = len(chunk_text)
        # logger.debug(f"Processing tentative chunk: Start={current_chunk_start}, End={current_chunk_end}, Len={chunk_len_chars}")

        # Scenario A: Chunk is too large
        if chunk_len_chars > MAX_CHAR_SIZE:
            logger.info(f"📏 Chunk [{current_chunk_start}-{current_chunk_end}] is too large ({chunk_len_chars} > {MAX_CHAR_SIZE}). Attempting to sub-split.")
            # Sub-split this large chunk. The LLM should have ideally done this.
            # We will split it into smaller pieces, ensuring each new split is a sentence boundary.
            num_sub_chunks = (chunk_len_chars + TARGET_CHAR_SIZE -1) // TARGET_CHAR_SIZE # ceiling division
            
            if num_sub_chunks <=1: num_sub_chunks = 2 # ensure at least one split if too large

            sub_chunk_target_len = chunk_len_chars / num_sub_chunks
            
            temp_sub_split_start = current_chunk_start
            for k in range(1, num_sub_chunks): # Iterate to create num_sub_chunks-1 new split points
                potential_sub_split_abs = current_chunk_start + int(k * sub_chunk_target_len)
                
                # Ensure the potential split is within the current large chunk's bounds, not too close to its start/end
                if potential_sub_split_abs <= temp_sub_split_start + MIN_CHAR_SIZE // 2: 
                    continue 
                if potential_sub_split_abs >= current_chunk_end - MIN_CHAR_SIZE // 2:
                    break # No more useful splits can be made

                actual_sub_split = find_nearest_sentence_boundary(md_text, potential_sub_split_abs)
                
                if actual_sub_split != -1 and actual_sub_split > temp_sub_split_start and actual_sub_split < current_chunk_end:
                    # Check if this new split creates a tiny chunk from temp_sub_split_start
                    if (actual_sub_split - temp_sub_split_start) >= MIN_CHAR_SIZE:
                         final_splits.append(actual_sub_split)
                         temp_sub_split_start = actual_sub_split # Next sub-chunk starts from here
                    else:
                        logger.debug(f"Skipping sub-split at {actual_sub_split}, would create too small chunk from {temp_sub_split_start}")
                else:
                    logger.warning(f"Could not find suitable sub-split point near {potential_sub_split_abs} within chunk [{current_chunk_start}-{current_chunk_end}]")

            current_chunk_start = current_chunk_end # Move to the end of the processed large chunk for next iteration

        # Scenario B: Chunk is too small
        elif chunk_len_chars < MIN_CHAR_SIZE and current_chunk_start > 0 : # Don't merge the very first chunk if it's small
            # This logic attempts to merge a small chunk by *removing* the previous split point.
            # This should be done cautiously.
            if final_splits: # If there's a previous split to remove
                prev_split_point = final_splits[-1]
                # Consider text from before previous split point up to current_chunk_end
                combined_len_if_merged = current_chunk_end - (final_splits[-2] if len(final_splits) > 1 else 0)

                if combined_len_if_merged <= MAX_CHAR_SIZE:
                    logger.info(f"🤏 Chunk [{current_chunk_start}-{current_chunk_end}] is too small ({chunk_len_chars}). Attempting to merge by removing previous split at {prev_split_point}.")
                    # The current_chunk_start was the prev_split_point. By removing it,
                    # the previous chunk now extends to current_chunk_end.
                    final_splits.pop() # Remove the split that made this small chunk
                    # The current_chunk_start for the next iteration effectively becomes the start of this merged chunk.
                    # However, the loop structure sets current_chunk_start = current_chunk_end, so this needs careful thought.
                    # The effect is that the *previous* chunk gets extended.
                    # This small chunk [current_chunk_start:current_chunk_end] is now part of the previous one.
                    current_chunk_start = current_chunk_end # Continue to next segment
                else:
                    logger.info(f"Chunk [{current_chunk_start}-{current_chunk_end}] is too small but merging would make previous chunk too large. Keeping small chunk.")
                    if current_chunk_start < len(md_text) and current_chunk_start not in final_splits: # only add if it's a valid end of chunk.
                         # This means current_chunk_start was an original aligned_sentence_point
                         # And it formed a small chunk. We keep it as the end of that small chunk.
                         final_splits.append(current_chunk_start) # this makes current_chunk_start the end of the prev segment.
                    current_chunk_start = current_chunk_end

            else: # No previous split, this is the first segment and it's small
                logger.info(f"First chunk is too small ({chunk_len_chars}), but no previous chunk to merge with. Keeping it.")
                # This path implies current_chunk_start is 0. The end of this small chunk is current_chunk_end.
                # If current_chunk_end is a valid split point (from aligned_sentence_points), it will be added.
                # The loop will naturally progress. This small chunk will be [0:current_chunk_end].
                current_chunk_start = current_chunk_end

        # Scenario C: Chunk is within size limits
        else:
            logger.debug(f"Chunk [{current_chunk_start}-{current_chunk_end}] is within size limits ({chunk_len_chars}).")
            # The end of this valid chunk is current_chunk_end. If current_chunk_end is one of the
            # aligned_sentence_points, it should be added as a split.
            # If current_chunk_end is len(md_text), it's the end of doc, not a split point.
            if current_chunk_end < len(md_text): # only add if not end of document
                 final_splits.append(current_chunk_end)
            current_chunk_start = current_chunk_end
            
    # Deduplicate and sort, ensure points are valid
    final_splits = sorted(list(set(p for p in final_splits if 0 < p < len(md_text))))
    logger.info(f"✅ Final refined split points after all processing: {final_splits}")
    return final_splits

def create_chunks_from_split_points(md_text: str, split_points: List[int]) -> List[str]:
    """
    Create chunks from the text using the provided split points.
    """
    all_points = [0] + split_points + [len(md_text)]
    # Remove duplicate points that might have arisen from refinement
    all_points = sorted(list(set(all_points))) 
    
    chunks = []
    if len(all_points) < 2: # Not enough points to create any chunk
        logger.warning("⚠️ Not enough split points to create chunks. Returning original text as one chunk.")
        if md_text.strip(): # only add if not empty
            return [md_text.strip()]
        return []

    for i in range(len(all_points) - 1):
        start = all_points[i]
        end = all_points[i+1]
        
        if start >= end: # Skip if start is not before end (e.g. due to duplicate points)
            logger.debug(f"Skipping chunk creation for start={start}, end={end}")
            continue

        chunk = md_text[start:end].strip() # Strip whitespace from each chunk
        if chunk: # Only add non-empty chunks
            chunks.append(chunk)
            # Verify split point again, just in case.
            if end < len(md_text) and not is_sentence_boundary(md_text, end):
                 logger.warning(f"⚠️ Potential mid-sentence split detected AT THE END of chunk generation at position {end}. Text around: '{md_text[max(0,end-30):min(len(md_text),end+30)]}'")
    
    logger.info(f"Created {len(chunks)} chunks.")
    return chunks

def process_document_section(md_text: str) -> list[str]:
    """
    Split a single document section into chunks using the LLM-centric approach.
    """
    logger.info(f"🔄 Processing section of {len(md_text)} characters (~{approx_token_count(md_text)} tokens)")
    
    # Step 1: Get LLM suggested split points
    llm_split_points = get_llm_split_suggestions(md_text)
    
    # Step 2: Refine LLM's points to ensure sentence boundaries and enforce size limits.
    # This step is crucial and acts as a validator and fine-tuner for the LLM's output.
    # The LLM is primarily responsible for semantic chunking and adhering to rules.
    final_split_points = refine_and_enforce_split_points(md_text, llm_split_points)
    
    # Step 3: Create chunks from verified split points
    chunks = create_chunks_from_split_points(md_text, final_split_points)
    
    is_valid = validate_chunking(chunks, md_text)
    if not is_valid:
        logger.error(f"❌ Chunking validation failed - content may be lost or significantly altered.")
    
    analysis = analyze_chunks(chunks)
    if analysis.get("chunk_count", 0) > 0:
        logger.info(f"📊 Created {analysis['chunk_count']} chunks. Compliance: {analysis.get('target_compliance', {}).get('percent_compliant', 'N/A')}% within token range.")
        logger.info(f"   Min/Max/Avg chars: {analysis.get('char_sizes', {}).get('min','N/A')}/{analysis.get('char_sizes', {}).get('max','N/A')}/{analysis.get('char_sizes', {}).get('avg','N/A')}")
        logger.info(f"   Min/Max/Avg tokens: {analysis.get('token_sizes', {}).get('min','N/A')}/{analysis.get('token_sizes', {}).get('max','N/A')}/{analysis.get('token_sizes', {}).get('avg','N/A')}")
    else:
        logger.info("ℹ️ No chunks were created from this section.")

    return chunks

def chunk_markdown(md_text: str) -> list[str]:
    """
    Split markdown text into chunks, handling documents of any size.
    Relies primarily on LLM for splitting decisions.
    """
    logger.info(f"🔄 Chunking markdown text of {len(md_text)} characters (~{approx_token_count(md_text)} tokens) using LLM-centric approach.")
    
    if not md_text.strip():
        logger.info("ℹ️ Document is empty. No chunks to create.")
        return []

    document_sections = split_large_document(md_text)
    
    all_chunks = []
    for i, section in enumerate(document_sections):
        logger.info(f"🔄 Processing section {i+1}/{len(document_sections)}")
        if not section.strip():
            logger.debug(f"Skipping empty section {i+1}")
            continue
        section_chunks = process_document_section(section)
        all_chunks.extend(section_chunks)
    
    logger.info(f"✅ Completed LLM-centric chunking. Total chunks: {len(all_chunks)}")
    return all_chunks

def analyze_chunks(chunks: List[str]) -> Dict[str, Any]:
    """Analyze chunk sizes and provide statistics."""
    if not chunks:
        return {"error": "No chunks to analyze", "chunk_count": 0}
    
    char_sizes = [len(chunk) for chunk in chunks]
    token_sizes = [approx_token_count(chunk) for chunk in chunks]
    
    # Ensure lists are not empty before calculating statistics
    def safe_stat(func, data_list, default=0):
        if not data_list: return default
        if len(data_list) == 1 and func == statistics.stdev: return default # stdev needs at least 2 points
        try:
            return round(func(data_list), 2)
        except statistics.StatisticsError: # Handles cases like stdev with insufficient data
            return default

    return {
        "chunk_count": len(chunks),
        "char_sizes": {
            "min": safe_stat(min, char_sizes),
            "max": safe_stat(max, char_sizes),
            "avg": safe_stat(statistics.mean, char_sizes),
            "median": safe_stat(statistics.median, char_sizes),
            "std_dev": safe_stat(statistics.stdev, char_sizes),
        },
        "token_sizes": {
            "min": safe_stat(min, token_sizes),
            "max": safe_stat(max, token_sizes),
            "avg": safe_stat(statistics.mean, token_sizes),
            "median": safe_stat(statistics.median, token_sizes),
            "std_dev": safe_stat(statistics.stdev, token_sizes),
        },
        "target_compliance": {
            "too_small": sum(1 for t in token_sizes if t < MIN_TOKEN_SIZE),
            "too_large": sum(1 for t in token_sizes if t > MAX_TOKEN_SIZE),
            "within_range": sum(1 for t in token_sizes if MIN_TOKEN_SIZE <= t <= MAX_TOKEN_SIZE),
            "percent_compliant": safe_stat(lambda x: sum(1 for t in x if MIN_TOKEN_SIZE <= t <= MAX_TOKEN_SIZE) / len(x) * 100 if len(x) > 0 else 0, token_sizes),
        }
    }

def validate_chunking(chunks: List[str], md_text: str) -> bool:
    """
    Validate that the chunks, when combined, contain all the original text.
    This is a more lenient check, focusing on content preservation rather than exact whitespace.
    """
    if not md_text.strip() and not chunks: # Both empty, valid
        return True
    if not md_text.strip() and chunks: # Original empty, but chunks exist (should not happen if chunks are stripped)
        logger.warning("⚠️ Validation: Original text is empty but chunks were produced.")
        return False # Or True if empty chunks are acceptable
    if md_text.strip() and not chunks: # Original has text, but no chunks produced
        logger.warning("⚠️ Validation: Original text has content but no chunks were produced.")
        # This might be valid if the document is too small to be chunked according to rules
        if len(md_text) < MIN_CHAR_SIZE :
            logger.info("ℹ️ Document too small to be chunked according to rules, validation pass.")
            return True
        return False


    combined = ''.join(chunks) # Chunks are already stripped
    
    # Normalize by removing all whitespace and converting to lowercase for comparison
    cleaned_combined = ''.join(combined.split()).lower()
    cleaned_original = ''.join(md_text.split()).lower()
    
    if cleaned_combined == cleaned_original:
        logger.info(f"✅ Validation passed: Content matches after normalization.")
        return True

    # If not exact match, check length difference as a percentage
    len_diff = abs(len(cleaned_original) - len(cleaned_combined))
    # prevent division by zero if original is all whitespace
    original_meaningful_len = len(cleaned_original) if cleaned_original else 1 
    diff_percentage = (len_diff / original_meaningful_len) * 100

    logger.info(f"📏 Content comparison: original normalized={len(cleaned_original)} chars, combined normalized={len(cleaned_combined)} chars")
    logger.info(f"📏 Normalized length difference: {len_diff} chars, {diff_percentage:.2f}% of original")
    
    if diff_percentage < 2.0: # Allow up to 2% difference for minor LLM summarizations/phrasing changes at boundaries
        logger.warning(f"⚠️ Validation: Content has minor differences ({diff_percentage:.2f}%) after normalization. Passing with warning.")
        return True
    else:
        logger.error(f"❌ Validation failed: Significant content difference ({diff_percentage:.2f}%) after normalization.")
        # Log snippets of differences
        # (This part can be complex; a basic diff might be useful for debugging)
        # For now, just log a sample:
        logger.error(f"   Original sample: '{cleaned_original[:100]}...' Combin. sample: '{cleaned_combined[:100]}...' ")
        return False

def main():
    """
    Main function to test chunking on a sample markdown file.
    """
    _00_utils.reset_token_counters()
    
    input_file = os.path.join("_01_input", "raw", "sample_md.md")
    if not os.path.exists(input_file):
        logger.error(f"❌ Input file not found: {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        md_content = f.read()

    logger.info(f"📄 Loaded markdown file: {input_file} ({len(md_content)} characters)")

    try:
        chunks = chunk_markdown(md_content)
        
        logger.info("\n" + "="*40 + " CHUNK INSPECTION " + "="*40)
        if chunks:
            for i, chunk in enumerate(chunks, 1):
                approx_tokens = approx_token_count(chunk)
                logger.info(f"\n--- Chunk {i} ({len(chunk)} chars, ~{approx_tokens} tokens) ---")
                logger.info(chunk)
                if not chunk.strip():
                    logger.warning(f"   ⚠️ Chunk {i} is empty or only whitespace!")
                if approx_tokens < MIN_TOKEN_SIZE or approx_tokens > MAX_TOKEN_SIZE:
                    logger.warning(f"   ⚠️ Chunk {i} token count ({approx_tokens}) is outside range [{MIN_TOKEN_SIZE}-{MAX_TOKEN_SIZE}]")

        else:
            logger.info("No chunks were produced.")
        logger.info("="*98 + "\n")

        analysis = analyze_chunks(chunks)
        logger.info(f"📊 Final chunk analysis:")
        if analysis and "error" not in analysis:
            logger.info(f"  - Chunk count: {analysis['chunk_count']}")
            logger.info(f"  - Char sizes: min={analysis['char_sizes']['min']}, max={analysis['char_sizes']['max']}, avg={analysis['char_sizes']['avg']}")
            logger.info(f"  - Token sizes: min={analysis['token_sizes']['min']}, max={analysis['token_sizes']['max']}, avg={analysis['token_sizes']['avg']}")
            logger.info(f"  - Compliance: {analysis['target_compliance']['percent_compliant']}% within target range")
            logger.info(f"  - Out of range: {analysis['target_compliance']['too_small']} too small, {analysis['target_compliance']['too_large']} too large")
        else:
            logger.info(f"  - No analysis data available (Original was: {len(md_content)} chars)")


        # Create output directory (optional, for manual inspection if needed)
        output_dir = "chunks_output_agentic_test"
        os.makedirs(output_dir, exist_ok=True)
        combined_file = os.path.join(output_dir, "combined_chunks_agentic_test.md")
        with open(combined_file, "w", encoding="utf-8") as cf:
            for i, chunk in enumerate(chunks, 1):
                token_count = approx_token_count(chunk)
                cf.write(f"<!-- Chunk {i:03d} Start - {len(chunk)} chars / ~{token_count} tokens -->\n")
                cf.write(chunk)
                cf.write("\n\n---\n\n") # Delimiter
                cf.write(f"<!-- Chunk {i:03d} End -->\n\n")
        logger.info(f"✅ Wrote combined chunks to: {combined_file}")
        
        analysis_file = os.path.join(output_dir, "chunk_analysis_agentic_test.json")
        with open(analysis_file, "w", encoding="utf-8") as af:
            json.dump(analysis, af, indent=2)
        logger.info(f"✅ Wrote analysis to: {analysis_file}")
        
        logger.info("📊 Token usage summary for this run:")
        _00_utils.print_token_usage("gpt-4o-mini") # Assuming gpt-4o-mini is used
    
    except Exception as e:
        logger.error(f"❌ Failed to chunk markdown: {e}", exc_info=True)
        return 1 # Indicate error
    
    return 0 # Indicate success

if __name__ == "__main__":
    # Ensure project context is set up if running directly
    if os.path.basename(os.getcwd()) != 'Requify_V2':
        # Attempt to change to project root if possible, or warn
        project_root_guess = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        if os.path.basename(project_root_guess) == 'Requify_V2':
            os.chdir(project_root_guess)
            logger.info(f"Changed current working directory to: {os.getcwd()}")
        else:
            logger.warning(f"Script might not be running from project root. CWD: {os.getcwd()}")
            
    _00_utils.setup_project_directory() # Re-run in case CWD changed
    sys.exit(main())
