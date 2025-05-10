"""
clean_docs_table.py

This script connects to a LanceDB table containing PDF page data (including vector embeddings),
and performs a similarity search to find and report duplicate or very similar pages.
It iterates through each page, uses its embedding to query for similar pages,
and logs pairs of pages that exceed a defined similarity threshold.


"""
# -------------------------------------------------------------------------------------
# Imports & Setup
# -------------------------------------------------------------------------------------
import os
import sys
from dotenv import load_dotenv
import lancedb
from lancedb.pydantic import LanceModel, Vector
from typing import List, Optional, Set, Tuple
import pandas as pd

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import _00_utils
_00_utils.setup_project_directory()
logger = _00_utils.setup_logging() # Setup centralized logging

# Load environment variables
load_dotenv()

# -------------------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------------------
OUTPUT_DIR_BASE = "03_output"  # Define base output directory
LANCEDB_SUBDIR_NAME = "lancedb"  # Subdirectory for LanceDB within 03_output
LANCEDB_TABLE_NAME = "all_pdf_pages"
EMBEDDING_DIMENSION = 1024  # Dimension for e5-large models (must match stable_save_to_lancedb.py)
SIMILARITY_THRESHOLD = 0.95  # Cosine similarity threshold for considering pages as duplicates
# LanceDB returns distance for cosine as 1 - similarity. So, distance_threshold = 1 - SIMILARITY_THRESHOLD
DISTANCE_THRESHOLD = 1.0 - SIMILARITY_THRESHOLD
TOP_K_RESULTS = 5  # Number of similar items to retrieve for each page query



# -------------------------------------------------------------------------------------
# LanceDB Schema (must match the schema used for table creation)
# -------------------------------------------------------------------------------------
class PDFPage(LanceModel):
    pdf_identifier: str
    page_number: Optional[int]
    document_title: Optional[str]
    summary: Optional[str]
    hashtags: Optional[List[str]]
    md_content: Optional[str]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    processing_duration: Optional[float]
    error_flag: Optional[bool]
    timestamp: Optional[str]
    embedding: Vector(EMBEDDING_DIMENSION)
    image_b64: Optional[str]

# -------------------------------------------------------------------------------------
# Main Deduplication Logic
# -------------------------------------------------------------------------------------
def find_similar_pages():
    """
    Connects to LanceDB, loads page data, and finds similar pages based on vector embeddings.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..")) # Assumes script is in 02_src/03_deduplication
    lancedb_path = os.path.join(project_root, OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME) # Path for lancedb within 03_output

    logger.info(f"Attempting to connect to LanceDB at: {lancedb_path}", extra={"icon": "☁️"})
    if not os.path.exists(lancedb_path):
        logger.error(f"LanceDB directory not found at {lancedb_path}. Please run the ingestion script first.")
        return

    try:
        db = lancedb.connect(lancedb_path)
        logger.info(f"Successfully connected to LanceDB.")
    except Exception as e:
        logger.error(f"Failed to connect to LanceDB: {e}")
        return

    table_names = db.table_names()
    if LANCEDB_TABLE_NAME not in table_names:
        logger.error(f"Table '{LANCEDB_TABLE_NAME}' not found in LanceDB. Available tables: {table_names}")
        return

    try:
        table = db.open_table(LANCEDB_TABLE_NAME)
        logger.info(f"Opened table '{LANCEDB_TABLE_NAME}'. Total records: {len(table)}", extra={"icon": "📄"})
    except Exception as e:
        logger.error(f"Failed to open table '{LANCEDB_TABLE_NAME}': {e}")
        return

    # Fetch all necessary data. Using to_pandas() for easier iteration.
    # Select specific columns to reduce memory usage if the table is very large.
    # For this "simple comparison", we need id, page_number, and embedding.
    try:
        all_pages_df = table.to_pandas()
        # Ensure 'page_number' is treated as nullable integer if it comes as float from pandas
        if 'page_number' in all_pages_df.columns:
            all_pages_df['page_number'] = all_pages_df['page_number'].astype('Int64')

        logger.info(f"Loaded {len(all_pages_df)} pages from table into DataFrame.")
    except Exception as e:
        logger.error(f"Failed to load data from table to DataFrame: {e}")
        return

    if all_pages_df.empty:
        logger.info("No pages found in the table to process.")
        return

    processed_pairs: Set[Tuple[str, str]] = set()
    similar_pages_found_count = 0

    for index, row in all_pages_df.iterrows():
        current_pdf_id = row['pdf_identifier']
        current_page_num = row['page_number'] # This can be pd.NA
        current_embedding = row['embedding']

        # Create a unique ID for the current page for display and set operations
        # Handle pd.NA for page_number gracefully
        current_page_id_str = f"{current_pdf_id}_p{current_page_num if pd.notna(current_page_num) else 'N/A'}"
        logger.info(f"Processing page: {current_page_id_str}")

        try:
            # Search for similar vectors.
            # The distance for cosine is 1 - similarity.
            # We filter out the current page itself in the post-processing step.
            search_results_df = table.search(current_embedding)\
                                     .limit(TOP_K_RESULTS + 1)\
                                     .to_df() # Corrected line break
            
            # Ensure 'page_number' is treated as nullable integer in results as well
            if 'page_number' in search_results_df.columns:
                search_results_df['page_number'] = search_results_df['page_number'].astype('Int64')

        except Exception as e:
            logger.error(f"Error during similarity search for page {current_page_id_str}: {e}")
            continue

        if search_results_df.empty:
            logger.info(f"No search results for page {current_page_id_str}.")
            continue
            
        found_similar_for_current = False
        for _, sr_row in search_results_df.iterrows():
            similar_pdf_id = sr_row['pdf_identifier']
            similar_page_num = sr_row['page_number'] # This can be pd.NA
            distance = sr_row['_distance'] # LanceDB uses _distance for similarity score (1-cosine for cosine metric)
            similarity_score = 1.0 - distance

            # Create a unique ID for the similar page
            similar_page_id_str = f"{similar_pdf_id}_p{similar_page_num if pd.notna(similar_page_num) else 'N/A'}"

            # Skip if it's the same page
            if current_pdf_id == similar_pdf_id and current_page_num == similar_page_num:
                continue

            if similarity_score >= SIMILARITY_THRESHOLD:
                # Create a canonical representation of the pair to avoid duplicates like (A,B) and (B,A)
                pair = tuple(sorted((current_page_id_str, similar_page_id_str)))
                if pair not in processed_pairs:
                    logger.info(
                        f"Found similar pages: {current_page_id_str} and {similar_page_id_str} "
                        f"(Similarity: {similarity_score:.4f})"
                    )
                    processed_pairs.add(pair)
                    similar_pages_found_count +=1
                    found_similar_for_current = True
        
        if not found_similar_for_current:
             logger.info(f"No new similar pages found for {current_page_id_str} above threshold {SIMILARITY_THRESHOLD}.")


    logger.info(f"Deduplication check completed. Found {similar_pages_found_count} unique pairs of similar pages.", extra={"icon": "🏁"})

if __name__ == "__main__":
    find_similar_pages()
