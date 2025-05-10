import os
import sys
import logging
import time
import numpy as np
import pandas as pd
import lancedb
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional # Added List, Optional

# Import Agno for LLM interaction
from agno.agent import Agent
from agno.models.groq import Groq # Using Groq, can be changed to OpenAIChat if preferred
# import json # Removed json import

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # noqa
import _00_utils
_00_utils.setup_project_directory()

# Load environment variables
load_dotenv()

# Setup logging
logger = _00_utils.setup_logging()

# -------------------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------------------
OUTPUT_DIR_BASE = "03_output"
LANCEDB_SUBDIR_NAME = "lancedb"
# Construct path relative to project root
LANCEDB_DIR_PATH = os.path.join(OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME)
REQUIREMENTS_TABLE_NAME = "requirements"
EMBEDDING_DIMENSION = 1024  # As defined in extract_requirements.py

# Deduplication parameters
WINDOW_SIZE = 30  # Number of neighbors to compare per requirement
SIM_THRESHOLD = 0.995  # Cosine similarity threshold to consider as duplicate

# LLM Deduplication Parameters
LLM_WINDOW_SIZE = 30
LLM_WINDOW_OVERLAP = 10 # New window starts at index i + (LLM_WINDOW_SIZE - LLM_WINDOW_OVERLAP)
LLM_MODEL_ID = "llama-3.3-70b-versatile" # Configurable LLM model

# -------------------------------------------------------------------------------------
# Pydantic Models for LLM Interaction
# -------------------------------------------------------------------------------------
class LLMDuplicatePair(BaseModel):
    req_id_1: str = Field(..., description="Requirement ID of the first item in a duplicate pair.")
    req_id_2: str = Field(..., description="Requirement ID of the second item in a duplicate pair.")
    # reason: Optional[str] = Field(None, description="Reason why these are considered duplicates by the LLM.")

class LLMDuplicateReport(BaseModel):
    duplicate_pairs: List[LLMDuplicatePair] = Field(default_factory=list, description="List of identified duplicate pairs within the window.")

# -------------------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------------------

def connect_to_lancedb(db_path: str):
    """Connect to LanceDB and return the connection object."""
    try:
        logger.info(f"Attempting to connect to LanceDB at: {db_path}")
        db = lancedb.connect(db_path)
        logger.info(f"Successfully connected to LanceDB. Available tables: {db.table_names()}")
        return db
    except Exception as e:
        logger.error(f"Error connecting to LanceDB at {db_path}: {e}", exc_info=True)
        raise

def load_requirements_data(db: lancedb.LanceDBConnection, table_name: str) -> pd.DataFrame:
    """Load requirement_id, embedding, created_timestamp, title, section, and description from the LanceDB table."""
    try:
        if table_name not in db.table_names():
            logger.error(f"Table '{table_name}' not found in LanceDB.")
            return pd.DataFrame() # Return empty DataFrame

        table = db.open_table(table_name)
        logger.info(f"Opened table '{table_name}'. Fetching all records...")
        
        all_data = table.to_pandas()
        
        if all_data.empty:
            logger.warning(f"Table '{table_name}' is empty.")
            return pd.DataFrame()

        logger.info(f"Successfully loaded {len(all_data)} records from '{table_name}'.")

        # Remove full duplicates (same requirement_id AND same description)
        before = len(all_data)
        all_data = all_data.drop_duplicates(subset=["requirement_id", "description"], keep="first")
        after = len(all_data)
        if before != after:
            logger.info(f"Removed {before - after} full duplicates (same requirement_id AND description) from loaded data.")

        # Handle duplicate requirement_ids: keep the one with the latest timestamp
        if 'requirement_id' in all_data.columns and 'created_timestamp' in all_data.columns:
            if all_data['requirement_id'].duplicated().any():
                logger.warning(f"Duplicate requirement_ids found in source table '{table_name}'. Keeping the entry with the latest 'created_timestamp' for each duplicate ID.")
                # Convert timestamp to datetime if it's not already, to ensure correct sorting
                # Errors='coerce' will turn unparseable timestamps into NaT, which sort early (can be adjusted if needed)
                all_data['created_timestamp_dt'] = pd.to_datetime(all_data['created_timestamp'], errors='coerce')
                # Sort by requirement_id and then by timestamp descending, then keep the first (which is the latest)
                all_data = all_data.sort_values(['requirement_id', 'created_timestamp_dt'], ascending=[True, False]) \
                                     .drop_duplicates(subset=['requirement_id'], keep='first')
                all_data = all_data.drop(columns=['created_timestamp_dt']) # Drop helper column
                logger.info(f"After handling duplicate requirement_ids, {len(all_data)} unique records remain.")
        elif 'requirement_id' in all_data.columns and all_data['requirement_id'].duplicated().any():
            logger.warning(f"Duplicate requirement_ids found in source table '{table_name}' but 'created_timestamp' is missing. Keeping the first encountered entry for each duplicate ID.")
            all_data = all_data.drop_duplicates(subset=['requirement_id'], keep='first')
            logger.info(f"After handling duplicate requirement_ids (first entry kept), {len(all_data)} unique records remain.")

        required_columns = ["requirement_id", "embedding", "created_timestamp", "title", "section", "description"]
        missing_columns = [col for col in required_columns if col not in all_data.columns]

        if missing_columns:
            logger.error(f"Missing one or more required columns ({', '.join(missing_columns)}) in table '{table_name}'.")
            # Attempt to return available required columns at least
            available_required = [col for col in required_columns if col in all_data.columns]
            if not available_required: # If even requirement_id or embedding is missing, it's problematic
                 logger.error("Essential columns like 'requirement_id' or 'embedding' are missing. Cannot proceed with partial data.")
                 return pd.DataFrame()
            logger.warning(f"Proceeding with available columns: {available_required}")
            return all_data[available_required]
            
        return all_data[required_columns]
        
    except Exception as e:
        logger.error(f"Error loading data from table '{table_name}': {e}", exc_info=True)
        raise

# Restore normalize_embeddings function
def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalize a batch of embeddings to unit length."""
    logger.info("Normalizing embeddings...")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero for zero-vectors if any (replace norm with 1 to avoid NaN)
    norms[norms == 0] = 1.0
    normalized_embeddings = embeddings / norms
    logger.info("Embeddings normalized.")
    return normalized_embeddings

# Helper function to sort a dataframe by projection score (adapted from find_duplicate_requirements)
def sort_dataframe_by_projection(df: pd.DataFrame) -> pd.DataFrame:
    """Sorts the DataFrame by projection score of normalized embeddings."""
    logger.info("Attempting to sort DataFrame by projection score...")
    if df.empty:
        logger.warning("Input DataFrame for sorting is empty.")
        return pd.DataFrame()
    if 'embedding' not in df.columns:
        logger.warning("Input DataFrame for sorting is missing 'embedding' column.")
        return pd.DataFrame()
    
    # Ensure all required columns for detailed logging are present or handle their absence
    req_id_col_present = 'requirement_id' in df.columns

    embeddings_list = df['embedding'].tolist()
    
    processed_embeddings = []
    valid_indices = [] # Keep track of indices of valid embeddings

    for i, emb in enumerate(embeddings_list):
        try:
            emb_np = np.array(emb, dtype=np.float32)
            current_id_log = df.iloc[i]['requirement_id'] if req_id_col_present else f"index {i}"

            if emb_np.ndim == 0: 
                logger.warning(f"Scalar embedding found for {current_id_log}. Skipping this embedding.")
                continue 
            elif emb_np.ndim == 1 and emb_np.shape[0] == EMBEDDING_DIMENSION:
                processed_embeddings.append(emb_np)
                valid_indices.append(i)
            elif emb_np.ndim == 2 and emb_np.shape[0] == 1 and emb_np.shape[1] == EMBEDDING_DIMENSION:
                processed_embeddings.append(emb_np.reshape(EMBEDDING_DIMENSION))
                valid_indices.append(i)
            else: 
                logger.error(f"Unexpected embedding shape {emb_np.shape} for {current_id_log}. Expected ({EMBEDDING_DIMENSION},). Skipping this embedding.")
                continue
        except Exception as e:
            current_id_log_exc = df.iloc[i]['requirement_id'] if req_id_col_present and i < len(df) else f"index {i}"
            logger.error(f"Error processing embedding for {current_id_log_exc}: {e}. Skipping this embedding.")
            continue
    
    if not processed_embeddings:
        logger.error("No valid embeddings to process for sorting after validation.")
        return pd.DataFrame()

    embeddings_np = np.array(processed_embeddings).reshape(len(processed_embeddings), EMBEDDING_DIMENSION)
    
    # Filter the original DataFrame to keep only rows with valid embeddings
    df_filtered_for_sorting = df.iloc[valid_indices].copy()
    if df_filtered_for_sorting.empty:
        logger.error("DataFrame became empty after filtering for valid embeddings. Cannot sort.")
        return pd.DataFrame()

    normalized_embeddings_np = normalize_embeddings(embeddings_np) 

    logger.info("Calculating mean vector for semantic projection (for sorting)...")
    mean_vector = np.mean(normalized_embeddings_np, axis=0)
    if np.isnan(mean_vector).any(): # Check if mean_vector calculation resulted in NaNs
        logger.error("Mean vector contains NaN values. This might be due to all-zero embeddings or other data issues. Cannot proceed with projection.")
        return pd.DataFrame() # Or return df_filtered_for_sorting if projection is optional
        
    logger.info("Calculating projection scores (for sorting)...")
    projection_scores = np.dot(normalized_embeddings_np, mean_vector)
    
    df_filtered_for_sorting['projection_score'] = projection_scores

    logger.info("Sorting requirements by projection score...")
    df_sorted = df_filtered_for_sorting.sort_values(by='projection_score', ascending=False).reset_index(drop=True)
    logger.info(f"Successfully sorted {len(df_sorted)} requirements by projection score.")
    return df_sorted

# -------------------------------------------------------------------------------------
# LLM-based Deduplication Functions
# -------------------------------------------------------------------------------------
def format_requirements_for_llm(window_df: pd.DataFrame) -> str:
    """Formats a window of requirements into a string for the LLM prompt."""
    prompt_text_parts = []
    # Ensure required columns are present
    required_cols_for_llm = ['requirement_id', 'title', 'section', 'description']
    missing_cols = [col for col in required_cols_for_llm if col not in window_df.columns]
    if missing_cols:
        logger.error(f"LLM Formatting: Missing columns required for LLM prompt: {missing_cols}. Cannot format.")
        # Return an empty string or raise an error, depending on desired handling
        # For now, log and proceed, LLM might get less context.
        # A better approach might be to ensure these columns are always present or skip the window.

    for _, row in window_df.iterrows():
        # Use .get() with defaults to handle potentially missing columns gracefully if not strictly enforced above
        req_text = f"Requirement ID: {row.get('requirement_id', 'N/A')}\n" \
                   f"Title: {row.get('title', 'N/A')}\n" \
                   f"Section: {row.get('section', 'N/A')}\n" \
                   f"Description:\n{row.get('description', 'N/A')}\n---\n"
        prompt_text_parts.append(req_text)
    return "\n".join(prompt_text_parts)

def process_windows_with_llm(sorted_df: pd.DataFrame, agent: Agent, project_root: str) -> pd.DataFrame:
    """Processes sorted requirements in rolling windows using an LLM to find duplicates."""
    if sorted_df.empty:
        logger.info("LLM Processing: Input DataFrame is empty. No windows to process.")
        return pd.DataFrame(columns=['req_id_1', 'req_id_2'])

    all_llm_duplicate_pairs = []
    num_requirements = len(sorted_df)
    step_size = LLM_WINDOW_SIZE - LLM_WINDOW_OVERLAP
    if step_size <= 0:
        logger.error(f"LLM Window step_size must be positive. Calculated: {step_size}. Check LLM_WINDOW_SIZE and LLM_WINDOW_OVERLAP.")
        return pd.DataFrame(columns=['req_id_1', 'req_id_2'])

    required_cols_check = ['requirement_id', 'title', 'section', 'description']
    if not all(col in sorted_df.columns for col in required_cols_check):
        logger.error(f"LLM Processing: Input DataFrame missing one or more required columns: {required_cols_check}. Aborting LLM pass.")
        return pd.DataFrame(columns=['req_id_1', 'req_id_2'])

    for i in range(0, num_requirements, step_size):
        window_df = sorted_df.iloc[i : i + LLM_WINDOW_SIZE]
        if len(window_df) < 2: 
            logger.info(f"LLM Window {i // step_size + 1}: Too few items ({len(window_df)} remaining), skipping comparison for this window end.")
            continue # Skip if not enough items for a pair
        
        logger.info(f"🔄 LLM Processing: Window {i // step_size + 1} (Indices {i}-{min(i + LLM_WINDOW_SIZE -1, num_requirements -1)}), {len(window_df)} requirements.")

        requirements_text_for_prompt = format_requirements_for_llm(window_df)
        if not requirements_text_for_prompt: # If formatting failed due to missing columns
            logger.warning(f"LLM Window {i // step_size + 1}: Formatting requirements for LLM failed. Skipping this window.")
            continue
        
        prompt = (
            "You are an expert system requirements analyst. Review the following list of technical requirements "
            "and identify any pairs that are semantic duplicates - requirements that specify the same core functionality "
            "or constraint, even if worded differently.\n\n"
            "For each duplicate pair you identify, also select which requirement is the best written (most clear, complete, and precise). "
            "This best-written requirement should be considered the canonical version for that pair.\n\n"
            "Here are the requirements:\n\n"
            f"{requirements_text_for_prompt}\n\n"
            "Return your findings as a JSON object in this format:\n"
            "{\n"
            "  \"duplicate_pairs\": [\n"
            "    {\"req_id_1\": \"FIRST_REQUIREMENT_ID\", \"req_id_2\": \"SECOND_REQUIREMENT_ID\", \"canonical_id\": \"BEST_REQUIREMENT_ID\"}\n"
            "  ]\n"
            "}\n"
            "If no duplicates are found, return:\n"
            "{\n"
            "  \"duplicate_pairs\": []\n"
            "}"
        )

        try:
            response = agent.run(prompt) 
            
            if response and response.content and hasattr(response.content, 'duplicate_pairs'):
                window_duplicate_pairs = response.content.duplicate_pairs
                if window_duplicate_pairs:
                    logger.info(f"✅ LLM Window {i // step_size + 1}: Found {len(window_duplicate_pairs)} potential duplicate pairs.")
                    current_window_ids = set(window_df['requirement_id'].values)
                    for pair_idx, pair in enumerate(window_duplicate_pairs):
                        # Validate pair IDs returned by LLM
                        if not (hasattr(pair, 'req_id_1') and hasattr(pair, 'req_id_2')):
                            logger.warning(f"LLM Window {i // step_size + 1}, Pair {pair_idx+1}: LLM returned malformed pair object: {pair}. Skipping.")
                            continue

                        id1_from_llm = pair.req_id_1
                        id2_from_llm = pair.req_id_2

                        if id1_from_llm not in current_window_ids or id2_from_llm not in current_window_ids:
                            logger.warning(f"LLM Window {i // step_size + 1}, Pair {pair_idx+1}: LLM returned IDs ({id1_from_llm}, {id2_from_llm}) not present in the current window. Skipping this pair.")
                            continue
                        if id1_from_llm == id2_from_llm:
                            logger.warning(f"LLM Window {i // step_size + 1}, Pair {pair_idx+1}: LLM returned a pair with identical IDs ({id1_from_llm}). Skipping this pair.")
                            continue
                        
                        # Normalize pair order (e.g., lexicographically smaller ID first)
                        # This helps in removing duplicates if (A,B) and (B,A) are reported.
                        final_id1, final_id2 = sorted((id1_from_llm, id2_from_llm))
                        all_llm_duplicate_pairs.append({'req_id_1': final_id1, 'req_id_2': final_id2})
                else:
                    logger.info(f"ℹ️ LLM Window {i // step_size + 1}: No duplicates found by LLM in this window.")
            else:
                logger.warning(f"❌ LLM Window {i // step_size + 1}: Received no valid content or 'duplicate_pairs' attribute from LLM. Response object: {response}")

        except Exception as e:
            logger.error(f"❌ LLM Window {i // step_size + 1}: Error during LLM call or processing response: {e}", exc_info=True)
            # Optionally, save problematic prompt/window for debugging
            # debug_dir = os.path.join(project_root, OUTPUT_DIR_BASE, "llm_debug")
            # os.makedirs(debug_dir, exist_ok=True)
            # with open(os.path.join(debug_dir, f"window_{i // step_size + 1}_prompt.txt"), "w") as f_debug:
            #    f_debug.write(prompt)
            # with open(os.path.join(debug_dir, f"window_{i // step_size + 1}_data.csv"), "w") as f_data:
            #    window_df.to_csv(f_data, index=False)


    if not all_llm_duplicate_pairs:
        logger.info("ℹ️ LLM Processing: No duplicate pairs identified across all windows.")
        return pd.DataFrame(columns=['req_id_1', 'req_id_2'])

    llm_duplicates_df = pd.DataFrame(all_llm_duplicate_pairs).drop_duplicates().reset_index(drop=True)
    logger.info(f"✅ LLM Processing: Total unique duplicate pairs found by LLM across all windows: {len(llm_duplicates_df)}")
    
    return llm_duplicates_df

# -------------------------------------------------------------------------------------
# Main Deduplication Logic
# -------------------------------------------------------------------------------------

def find_duplicate_requirements(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]: # Modified return type
    """
    Detects and groups semantically duplicate requirements using cosine-sorted sliding window.
    Returns a DataFrame of duplicate pairs and the full DataFrame sorted by projection score.
    """
    if df.empty or 'embedding' not in df.columns:
        logger.warning("Input DataFrame is empty or missing 'embedding' column. Skipping deduplication.")
        return pd.DataFrame(columns=['req_id_1', 'req_id_2', 'similarity']), pd.DataFrame()

    start_time = time.time()
    
    # Step 1: Get original embeddings and then normalize them
    embeddings_list = df['embedding'].tolist()
    try:
        embeddings_np = np.array(embeddings_list, dtype=np.float32).reshape(len(embeddings_list), -1)
    except ValueError as e:
        logger.error(f"Error converting embeddings to NumPy array. Are all embeddings of the same dimension ({EMBEDDING_DIMENSION})? Error: {e}")
        for i, emb in enumerate(embeddings_list):
            if np.array(emb).ndim == 0 or np.array(emb).shape[0] != EMBEDDING_DIMENSION :
                logger.error(f"Problematic embedding at index {i}, ID {df.iloc[i]['requirement_id']}: shape {np.array(emb).shape}")
        return pd.DataFrame(columns=['req_id_1', 'req_id_2', 'similarity']), pd.DataFrame()

    normalized_embeddings_np = normalize_embeddings(embeddings_np)
    df['normalized_embedding'] = list(normalized_embeddings_np) # Store for later use

    # Step 2: Project Onto a Shared Semantic Axis (using NORMALIZED embeddings)
    logger.info("Calculating mean vector for semantic projection (using NORMALIZED embeddings)...")
    mean_vector = np.mean(normalized_embeddings_np, axis=0)
    
    logger.info("Calculating projection scores (using NORMALIZED embeddings)...")
    projection_scores = np.dot(normalized_embeddings_np, mean_vector)
    df['projection_score'] = projection_scores

    # Step 3: Sort Requirements by Projection
    logger.info("Sorting requirements by projection score...")
    df_sorted = df.sort_values(by='projection_score', ascending=False).reset_index(drop=True)

    # Step 4 & 5: Slide a Comparison Window and Compute Cosine Similarity
    # (using NORMALIZED embeddings from the sorted DataFrame)
    logger.info(f"Sliding window (size {WINDOW_SIZE}) comparison with threshold {SIM_THRESHOLD} (using cosine similarity on normalized embeddings)...")
    duplicate_pairs = []
    num_requirements = len(df_sorted)

    # Get normalized embeddings from the sorted DataFrame for quick access
    # This ensures we use the embeddings corresponding to the sorted order
    sorted_normalized_embeddings_np = np.array(df_sorted['normalized_embedding'].tolist(), dtype=np.float32).reshape(num_requirements, -1)

    for i in range(num_requirements):
        if i % 1000 == 0 and i > 0:
            logger.info(f"Processing requirement {i}/{num_requirements} in sliding window...")
            
        req1_id = df_sorted.at[i, 'requirement_id']
        req1_norm_emb = sorted_normalized_embeddings_np[i]
        
        window_end = min(i + 1 + WINDOW_SIZE, num_requirements)
        
        for j in range(i + 1, window_end):
            req2_id = df_sorted.at[j, 'requirement_id']
            req2_norm_emb = sorted_normalized_embeddings_np[j]
            
            # Cosine similarity for normalized vectors is the dot product
            similarity = np.dot(req1_norm_emb, req2_norm_emb)
            
            if similarity >= SIM_THRESHOLD:
                duplicate_pairs.append({
                    'req_id_1': req1_id,
                    'req_id_2': req2_id,
                    'similarity': similarity,
                    'timestamp_1': df_sorted.at[i, 'created_timestamp'],
                    'timestamp_2': df_sorted.at[j, 'created_timestamp'],
                    'description_1': df_sorted.at[i, 'description'],
                    'description_2': df_sorted.at[j, 'description'],
                    'norm_emb_1_sample': req1_norm_emb[:5].tolist(),  # Added embedding sample 1
                    'norm_emb_2_sample': req2_norm_emb[:5].tolist()   # Added embedding sample 2
                })
    
    end_time = time.time()
    logger.info(f"Sliding window comparison completed in {end_time - start_time:.2f} seconds.")
    logger.info(f"Found {len(duplicate_pairs)} potential duplicate pairs.")

    return pd.DataFrame(duplicate_pairs), df_sorted # Return both dataframes

# -------------------------------------------------------------------------------------
# Grouping and Reporting Logic
# -------------------------------------------------------------------------------------

def group_similar_requirements(duplicate_pairs_df: pd.DataFrame) -> list[set[str]]:
    """
    Groups requirement IDs into clusters based on the duplicate pairs found.
    Uses a connected components approach.
    """
    if duplicate_pairs_df.empty:
        return []

    adj = {} # Adjacency list for the graph
    all_req_ids = set(duplicate_pairs_df['req_id_1']).union(set(duplicate_pairs_df['req_id_2']))

    for req_id in all_req_ids:
        adj[req_id] = set()

    for _, row in duplicate_pairs_df.iterrows():
        req1, req2 = row['req_id_1'], row['req_id_2']
        adj[req1].add(req2)
        adj[req2].add(req1)

    clusters = []
    visited = set()

    for req_id in all_req_ids:
        if req_id not in visited:
            current_cluster = set()
            q = [req_id] # Queue for BFS
            visited.add(req_id)
            head = 0
            while head < len(q):
                u = q[head]
                head += 1
                current_cluster.add(u)
                for v_neighbor in adj.get(u, set()):
                    if v_neighbor not in visited:
                        visited.add(v_neighbor)
                        q.append(v_neighbor)
            if current_cluster: # Should always be true if req_id was in all_req_ids
                clusters.append(current_cluster)
    
    logger.info(f"Grouped {len(all_req_ids)} requirements into {len(clusters)} similarity clusters.")
    return clusters

# New function to determine canonical and mark duplicates
def determine_canonical_and_duplicates(clusters: list[set[str]], requirements_details_df: pd.DataFrame) -> pd.DataFrame: # Renamed requirements_df to requirements_details_df for clarity
    """
    Determines the canonical requirement for each cluster and marks others as duplicates.
    Returns a DataFrame with all requirements from clusters, marked appropriately.
    Uses requirements_details_df to fetch details like timestamp, title, etc.
    """
    if not clusters:
        logger.info("No clusters provided for canonical selection.")
        return pd.DataFrame()
    if requirements_details_df.empty:
        logger.info("No requirement details DataFrame provided for canonical selection.")
        return pd.DataFrame()

    # Ensure requirement_id is the index for quick lookup in the details DataFrame
    # Handle potential duplicates in requirements_details_df['requirement_id'] if they exist, though load_data should prevent this.
    if requirements_details_df['requirement_id'].duplicated().any():
        logger.warning("Duplicate requirement_ids found in requirements_details_df for canonical selection. Using first occurrence for lookup.")
        details_lookup_df = requirements_details_df.drop_duplicates(subset=['requirement_id'], keep='first').set_index('requirement_id')
    else:
        details_lookup_df = requirements_details_df.set_index('requirement_id')

    deduplication_map_list = []

    for cluster_num, cluster_ids_set in enumerate(clusters, 1):
        if not cluster_ids_set:
            logger.warning(f"Cluster {cluster_num} is empty. Skipping.")
            continue

        cluster_members_data = []
        valid_cluster_ids = set() # IDs for which details were found

        for req_id in cluster_ids_set:
            if req_id in details_lookup_df.index:
                member_details = details_lookup_df.loc[req_id]
                # Ensure 'created_timestamp' exists, use a far past date if not, for consistent sorting
                timestamp_val = member_details.get('created_timestamp')
                try:
                    # Attempt to parse. If already datetime, it's fine. If string, parse. If unparsable, NaT.
                    dt_val = pd.to_datetime(timestamp_val, errors='coerce') 
                except Exception: # Catch any parsing error
                    dt_val = pd.NaT
                
                if pd.isna(dt_val):
                    logger.warning(f"Requirement ID '{req_id}' in Cluster {cluster_num} has missing or unparsable timestamp ('{timestamp_val}'). Will be treated as older.")
                
                cluster_members_data.append({
                    'requirement_id': req_id,
                    'created_timestamp_dt': dt_val, # Store as datetime for sorting
                    'title': member_details.get('title', 'N/A'),
                    'section': member_details.get('section', 'N/A'),
                    'description': member_details.get('description', 'N/A')
                })
                valid_cluster_ids.add(req_id)
            else:
                logger.warning(f"Details for requirement_id '{req_id}' not found in provided DataFrame (details_lookup_df). Skipping from cluster {cluster_num}.")
        
        if not cluster_members_data:
            logger.warning(f"Cluster {cluster_num} became empty after trying to fetch member details. Skipping.")
            continue

        # Sort by timestamp (newest first, NaT treated as oldest), then by requirement_id (lexicographically smallest as tie-breaker)
        # Pandas sorts NaT first when ascending=True, and last when ascending=False. So, for descending time (ascending=False), NaT is last (good).
        # For requirement_id, ascending=True.
        cluster_members_df = pd.DataFrame(cluster_members_data)
        cluster_members_df = cluster_members_df.sort_values(by=['created_timestamp_dt', 'requirement_id'], ascending=[False, True])
        
        canonical_member_row = cluster_members_df.iloc[0]
        canonical_req_id = canonical_member_row['requirement_id']
        canonical_timestamp_log = canonical_member_row['created_timestamp_dt']
        if pd.notna(canonical_timestamp_log):
            canonical_timestamp_log = canonical_timestamp_log.strftime('%Y-%m-%d %H:%M:%S')
        else:
            canonical_timestamp_log = "N/A (missing/unparseable)"


        logger.info(f"Cluster {cluster_num} (size {len(cluster_members_df)}): Canonical chosen is '{canonical_req_id}' (Timestamp: {canonical_timestamp_log})")

        for _, member_row in cluster_members_df.iterrows(): # Iterate over the sorted DataFrame
            is_duplicate_status = (member_row['requirement_id'] != canonical_req_id)
            timestamp_str = member_row['created_timestamp_dt'].strftime('%Y-%m-%d %H:%M:%S') if pd.notna(member_row['created_timestamp_dt']) else 'N/A'
            
            deduplication_map_list.append({
                'requirement_id': member_row['requirement_id'],
                'title': member_row['title'],
                'section': member_row['section'],
                'description': member_row['description'], # Consider truncating for log/CSV if very long
                'created_timestamp': timestamp_str,
                'is_duplicate': is_duplicate_status,
                'canonical_id': canonical_req_id,
                'cluster_id': f"C{cluster_num}" # Add a prefix for clarity
            })

    if not deduplication_map_list:
        logger.info("No deduplication map entries generated.")
        return pd.DataFrame()
        
    return pd.DataFrame(deduplication_map_list)

# -------------------------------------------------------------------------------------
# LanceDB Update Function
# -------------------------------------------------------------------------------------
def update_lancedb_from_map(table: lancedb.table.LanceTable, deduplication_map_df: pd.DataFrame, pass_name: str = "Deduplication Pass"):
    """Updates the LanceDB table with is_duplicate and canonical_id fields from a map."""
    if deduplication_map_df.empty:
        logger.info(f"ℹ️ {pass_name}: Deduplication map is empty. No updates to LanceDB.")
        return 0

    logger.info(f"🔄 {pass_name}: Updating LanceDB table '{table.name}' with 'is_duplicate' and 'canonical_id' fields...")
    update_count = 0
    error_count = 0
    
    # Ensure 'is_duplicate' and 'canonical_id' are in the correct format if not already
    # This should ideally be handled before calling, but as a safeguard:
    if 'is_duplicate' in deduplication_map_df.columns:
        deduplication_map_df['is_duplicate'] = deduplication_map_df['is_duplicate'].astype(bool)
    if 'canonical_id' in deduplication_map_df.columns:
        deduplication_map_df['canonical_id'] = deduplication_map_df['canonical_id'].astype(str).fillna('')


    for _, map_row in deduplication_map_df.iterrows():
        req_id_to_update = map_row['requirement_id']
        is_dup_value = bool(map_row['is_duplicate']) 
        canon_id_value = str(map_row['canonical_id']) if pd.notna(map_row['canonical_id']) else '' 

        try:
            # Construct the WHERE clause carefully, especially if requirement_id can have special characters.
            # LanceDB's SQL syntax for string literals typically uses single quotes.
            # Escaping single quotes within the req_id_to_update itself would be necessary if they can occur.
            # For simplicity, assuming requirement_id does not contain single quotes here.
            # If it can, replace "'" with "''" in req_id_to_update before using in SQL.
            # Example: safe_req_id = req_id_to_update.replace("'", "''")
            # where_clause = f"requirement_id = '{safe_req_id}'"
            # where_clause = f"requirement_id = '{req_id_to_update}'"
            where_clause = f"requirement_id = '{req_id_to_update}'"

            table.update(
                values={"is_duplicate": is_dup_value, "canonical_id": canon_id_value},
                where=where_clause
            )
            update_count += 1
            if update_count % 100 == 0:
                logger.info(f"🔄 {pass_name}: Updated {update_count} records in LanceDB...")
        except Exception as e_update:
            logger.error(f"❌ {pass_name}: Error updating record '{req_id_to_update}' in LanceDB: {e_update}", exc_info=True)
            error_count +=1
    
    logger.info(f"✅ {pass_name}: Successfully updated {update_count} records in LanceDB '{table.name}' table.")
    if error_count > 0:
        logger.warning(f"⚠️ {pass_name}: Failed to update {error_count} records.")
    return update_count

# -------------------------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------------------------
def main():
    """Main function to execute the requirements deduplication process."""
    logger.info("🚀 Starting requirements deduplication process...")
    start_total_time = time.time()

    # Construct absolute path to LanceDB directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    absolute_lancedb_path = os.path.join(project_root, LANCEDB_DIR_PATH)

    db = None
    try:
        db = connect_to_lancedb(absolute_lancedb_path)
        if not db:
            logger.error("❌ Failed to connect to LanceDB. Exiting.")
            return

        # Open the table
        if REQUIREMENTS_TABLE_NAME not in db.table_names():
            logger.error(f"❌ Table '{REQUIREMENTS_TABLE_NAME}' not found in LanceDB. Cannot proceed.")
            return
        table = db.open_table(REQUIREMENTS_TABLE_NAME)
        logger.info(f"ℹ️ Opened table '{REQUIREMENTS_TABLE_NAME}' for schema checks and updates.")

        # Check and add columns if they don't exist
        current_schema = table.schema
        if 'is_duplicate' not in current_schema.names:
            logger.info("ℹ️ Column 'is_duplicate' not found. Adding it with default 'false'.")
            table.add_columns(transforms={"is_duplicate": "false"}) # LanceDB expects string 'false' for boolean
            logger.info("✅ Successfully added 'is_duplicate' column.")
        
        if 'canonical_id' not in current_schema.names:
            logger.info("ℹ️ Column 'canonical_id' not found. Adding it with default empty string.")
            table.add_columns(transforms={"canonical_id": "''"}) 
            logger.info("✅ Successfully added 'canonical_id' column.")

        requirements_df = load_requirements_data(db, REQUIREMENTS_TABLE_NAME)
        if requirements_df.empty:
            logger.info("ℹ️ No requirements data loaded from DB. Exiting.")
            return
            
        logger.info(f"ℹ️ Loaded {len(requirements_df)} requirements with columns: {requirements_df.columns.tolist()}")
        
        # --- PASS 1: Embedding-based Deduplication ---
        logger.info("🚀 --- Starting Pass 1: Embedding-based Deduplication ---")
        # Perform embedding deduplication - now returns duplicate pairs and the sorted df
        # Pass a copy as find_duplicate_requirements modifies the df by adding columns
        duplicate_pairs_df_pass1, sorted_requirements_df_pass1 = find_duplicate_requirements(requirements_df.copy()) 

        # Save the sorted list of requirements by projection score from Pass 1
        # This sorted_requirements_df_pass1 contains all original items, sorted.
        # It might include items that will be marked as duplicates by this pass.
        output_dir_for_results = os.path.join(project_root, OUTPUT_DIR_BASE, "deduplication_results")
        os.makedirs(output_dir_for_results, exist_ok=True) # Ensure directory exists

        if not sorted_requirements_df_pass1.empty:
            sorted_csv_path_pass1 = os.path.join(output_dir_for_results, "pass1_sorted_requirements_by_projection.csv")
            columns_to_save_sorted = ['requirement_id', 'projection_score', 'title', 'section', 'description', 'created_timestamp']
            actual_columns_to_save = [col for col in columns_to_save_sorted if col in sorted_requirements_df_pass1.columns]
            try:
                sorted_requirements_df_pass1[actual_columns_to_save].to_csv(sorted_csv_path_pass1, index=False)
                logger.info(f"✅ Pass 1: Full list of requirements sorted by projection score saved to: {sorted_csv_path_pass1}")
            except Exception as e:
                logger.error(f"❌ Pass 1: Error saving sorted requirements list to {sorted_csv_path_pass1}: {e}")
        else:
            logger.info("ℹ️ Pass 1: Sorted requirements DataFrame is empty, not saving.")

        deduplication_map_df_pass1 = pd.DataFrame() # Initialize empty
        if not duplicate_pairs_df_pass1.empty:
            logger.info(f"ℹ️ Pass 1: Identified {len(duplicate_pairs_df_pass1)} potential duplicate pairs using embeddings.")
            logger.info(f"--- Top 10 Embedding-based Duplicate Pairs (Pass 1) ---")
            logger.info(f"\n{duplicate_pairs_df_pass1.head(10).to_string()}")
            
            csv_output_path_pass1 = os.path.join(output_dir_for_results, "pass1_embedding_duplicate_pairs.csv")
            duplicate_pairs_df_pass1.to_csv(csv_output_path_pass1, index=False)
            logger.info(f"✅ Pass 1: Full list of embedding-based duplicate pairs saved to: {csv_output_path_pass1}")

            # --- Enriched export for Pass 1 ---
            # Merge to get descriptions and titles for both req_id_1 and req_id_2
            reqs_for_merge = requirements_df[['requirement_id', 'title', 'section', 'description']].copy()
            enriched_pass1 = duplicate_pairs_df_pass1.merge(
                reqs_for_merge.add_suffix('_1'), left_on='req_id_1', right_on='requirement_id_1', how='left'
            ).merge(
                reqs_for_merge.add_suffix('_2'), left_on='req_id_2', right_on='requirement_id_2', how='left'
            )
            # Drop the extra requirement_id_1 and requirement_id_2 columns
            enriched_pass1 = enriched_pass1.drop(columns=['requirement_id_1', 'requirement_id_2'])
            enriched_csv_path_pass1 = os.path.join(output_dir_for_results, "pass1_embedding_duplicate_pairs_with_desc.csv")
            enriched_pass1.to_csv(enriched_csv_path_pass1, index=False)
            logger.info(f"✅ Pass 1: Enriched duplicate pairs with descriptions saved to: {enriched_csv_path_pass1}")

            logger.info("🔄 Pass 1: Grouping embedding-based duplicate pairs into clusters...")
            clusters_pass1 = group_similar_requirements(duplicate_pairs_df_pass1)
            
            if clusters_pass1:
                logger.info(f"ℹ️ Pass 1: Identified {len(clusters_pass1)} clusters. Determining canonical versions...")
                # Pass the original requirements_df for details
                deduplication_map_df_pass1 = determine_canonical_and_duplicates(clusters_pass1, requirements_df.copy())

                if not deduplication_map_df_pass1.empty:
                    map_output_path_pass1 = os.path.join(output_dir_for_results, "pass1_embedding_deduplication_map.csv")
                    try:
                        deduplication_map_df_pass1.to_csv(map_output_path_pass1, index=False)
                        logger.info(f"✅ Pass 1: Embedding-based deduplication map saved to: {map_output_path_pass1}")
                        
                        # Update LanceDB table based on Pass 1
                        update_lancedb_from_map(table, deduplication_map_df_pass1, "Pass 1 Embedding Deduplication")
                    except Exception as e:
                        logger.error(f"❌ Pass 1: Error saving or applying embedding deduplication map: {e}", exc_info=True)
                else:
                    logger.info("ℹ️ Pass 1: Embedding-based deduplication map is empty.")
            else:
                logger.info("ℹ️ Pass 1: No clusters formed from embedding-based duplicate pairs.")
        else:
            logger.info("ℹ️ Pass 1: No embedding-based duplicate pairs found meeting the criteria.")
        logger.info("✅ --- Finished Pass 1: Embedding-based Deduplication ---")

        # --- Update requirements_df in-memory to reflect Pass 1 changes ---
        # This is crucial for the LLM pass to operate on the correct set of active requirements.
        # If 'is_duplicate' or 'canonical_id' were not in the original load, add them.
        if 'is_duplicate' not in requirements_df.columns:
            requirements_df['is_duplicate'] = False
        if 'canonical_id' not in requirements_df.columns:
            # Initialize canonical_id to self if not duplicate (will be set to False above)
            requirements_df['canonical_id'] = requirements_df.apply(
                lambda row: row['requirement_id'] if not row['is_duplicate'] else '', axis=1
            )
        
        if not deduplication_map_df_pass1.empty:
            # Create a map for easier update: requirement_id -> (is_duplicate, canonical_id)
            update_values_pass1 = deduplication_map_df_pass1.set_index('requirement_id')[['is_duplicate', 'canonical_id']].to_dict('index')
            for req_id, updates in update_values_pass1.items():
                idx_list = requirements_df.index[requirements_df['requirement_id'] == req_id].tolist()
                if idx_list:
                    for idx_item in idx_list: # Handle if req_id somehow not unique (though load_data tries to prevent)
                        requirements_df.loc[idx_item, 'is_duplicate'] = updates['is_duplicate']
                        requirements_df.loc[idx_item, 'canonical_id'] = updates['canonical_id']
            
            # For requirements not in the map (not part of any cluster in Pass 1),
            # ensure their 'is_duplicate' is False and 'canonical_id' is their own ID.
            mapped_ids_pass1 = set(deduplication_map_df_pass1['requirement_id'])
            unmapped_mask_pass1 = ~requirements_df['requirement_id'].isin(mapped_ids_pass1)
            requirements_df.loc[unmapped_mask_pass1, 'is_duplicate'] = False # Explicitly set unmapped to not duplicate
             # For unmapped, canonical ID should be its own ID if it wasn't set by a previous (non-existent) map.
            requirements_df.loc[unmapped_mask_pass1, 'canonical_id'] = requirements_df.loc[unmapped_mask_pass1, 'requirement_id']
        else: # No map from Pass 1, ensure all are marked active initially for LLM pass
            requirements_df['is_duplicate'] = False
            requirements_df['canonical_id'] = requirements_df['requirement_id']
        
        # --- PASS 2: LLM-based Deduplication ---
        logger.info("🚀 --- Starting Pass 2: LLM-based Deduplication ---")
        
        # Filter for active requirements for LLM pass
        requirements_for_llm_pass_df = requirements_df[~requirements_df['is_duplicate']].copy() # Use ~ for boolean negation

        if requirements_for_llm_pass_df.empty:
            logger.info("ℹ️ Pass 2: No active requirements left after Pass 1. Skipping LLM deduplication.")
        else:
            logger.info(f"ℹ️ Pass 2: Proceeding with {len(requirements_for_llm_pass_df)} active requirements for LLM pass.")
            
            # Sort these active requirements by projection score for windowing
            # This uses the `sort_dataframe_by_projection` helper function.
            # It needs 'embedding' and other detail columns.
            sorted_active_reqs_for_llm_df = sort_dataframe_by_projection(requirements_for_llm_pass_df)

            if sorted_active_reqs_for_llm_df.empty:
                logger.info("ℹ️ Pass 2: Active requirements DataFrame became empty after sorting attempt. Skipping LLM deduplication.")
            else:
                logger.info(f"ℹ️ Pass 2: Successfully sorted {len(sorted_active_reqs_for_llm_df)} active requirements for LLM processing.")
                # Initialize Agno LLM agent
                llm_api_key = os.getenv("GROQ_API_KEY") # Ensure this is in your .env
                if not llm_api_key:
                    logger.error("❌ Pass 2: GROQ_API_KEY not found in environment variables. Cannot initialize LLM agent. Skipping LLM pass.")
                else:
                    llm_model = Groq(id=LLM_MODEL_ID, api_key=llm_api_key)
                    
                    llm_dedup_agent = Agent(
                        model=llm_model,
                        markdown=False, 
                        debug_mode=True, 
                        description="AI agent for identifying duplicate system requirements.",
                        response_model=LLMDuplicateReport, 
                        use_json_mode=True 
                    )
                    logger.info(f"🤖 Pass 2: Initialized LLM Agent with model: {LLM_MODEL_ID}")
                    
                    # Process windows with LLM
                    llm_duplicate_pairs_df_pass2 = process_windows_with_llm(sorted_active_reqs_for_llm_df, llm_dedup_agent, project_root)
                    
                    if not llm_duplicate_pairs_df_pass2.empty:
                        logger.info(f"ℹ️ Pass 2: LLM identified {len(llm_duplicate_pairs_df_pass2)} potential duplicate pairs.")
                        llm_pairs_csv_path_pass2 = os.path.join(output_dir_for_results, "pass2_llm_duplicate_pairs.csv")
                        llm_duplicate_pairs_df_pass2.to_csv(llm_pairs_csv_path_pass2, index=False)
                        logger.info(f"✅ Pass 2: LLM-identified duplicate pairs saved to: {llm_pairs_csv_path_pass2}")

                        # --- Enriched export for Pass 2 ---
                        reqs_for_merge = requirements_df[['requirement_id', 'title', 'section', 'description']].copy()
                        enriched_pass2 = llm_duplicate_pairs_df_pass2.merge(
                            reqs_for_merge.add_suffix('_1'), left_on='req_id_1', right_on='requirement_id_1', how='left'
                        ).merge(
                            reqs_for_merge.add_suffix('_2'), left_on='req_id_2', right_on='requirement_id_2', how='left'
                        )
                        enriched_pass2 = enriched_pass2.drop(columns=['requirement_id_1', 'requirement_id_2'])
                        enriched_csv_path_pass2 = os.path.join(output_dir_for_results, "pass2_llm_duplicate_pairs_with_desc.csv")
                        enriched_pass2.to_csv(enriched_csv_path_pass2, index=False)
                        logger.info(f"✅ Pass 2: Enriched duplicate pairs with descriptions saved to: {enriched_csv_path_pass2}")

                        # Group and determine canonicals for these LLM-found pairs
                        logger.info("🔄 Pass 2: Grouping LLM-identified duplicate pairs into clusters...")
                        clusters_pass2 = group_similar_requirements(llm_duplicate_pairs_df_pass2)
                        
                        if clusters_pass2:
                            logger.info(f"ℹ️ Pass 2: Identified {len(clusters_pass2)} clusters from LLM results. Determining canonical versions...")
                            # Use the main `requirements_df` (which reflects Pass 1 updates) for full context
                            deduplication_map_df_pass2 = determine_canonical_and_duplicates(clusters_pass2, requirements_df.copy())

                            if not deduplication_map_df_pass2.empty:
                                map_output_path_pass2 = os.path.join(output_dir_for_results, "pass2_llm_deduplication_map.csv")
                                deduplication_map_df_pass2.to_csv(map_output_path_pass2, index=False)
                                logger.info(f"✅ Pass 2: LLM-based deduplication map saved to: {map_output_path_pass2}")
                                
                                # Update LanceDB with results from LLM pass
                                update_lancedb_from_map(table, deduplication_map_df_pass2, "Pass 2 LLM Deduplication")
                            else:
                                logger.info("ℹ️ Pass 2: LLM-based deduplication map is empty.")
                        else:
                            logger.info("ℹ️ Pass 2: No clusters formed from LLM-identified duplicate pairs.")
                    else:
                        logger.info("ℹ️ Pass 2: No duplicate pairs identified by the LLM.")
        logger.info("✅ --- Finished Pass 2: LLM-based Deduplication ---")

    except ConnectionRefusedError as ce:
        logger.error(f"❌ Critical Error: Connection to LanceDB refused at {absolute_lancedb_path}. Ensure LanceDB is running or accessible. Details: {ce}", exc_info=True)
    except lancedb.errors.LanceDBClientError as lce:
        logger.error(f"❌ Critical Error: LanceDB client error. This could be due to network issues, permissions, or DB corruption at {absolute_lancedb_path}. Details: {lce}", exc_info=True)
    except Exception as e:
        logger.error(f"❌ An unexpected error occurred during the main deduplication process: {e}", exc_info=True)
    finally:
        logger.info("🧹 Deduplication process finished. Resources should be auto-released.")
        # LanceDB connection closes automatically on db object deletion or script exit.

    end_total_time = time.time()
    logger.info(f"🎉🎉 Total deduplication process (all passes) finished in {end_total_time - start_total_time:.2f} seconds. 🎉🎉")

if __name__ == "__main__":
    main()
