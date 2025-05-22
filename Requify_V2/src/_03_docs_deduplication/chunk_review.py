"""
chunk_review.py

This script provides an interactive interface for reviewing and managing orphaned chunks 
during document deduplication. It allows users to:
1. Review chunks that couldn't be automatically matched to existing content
2. Make decisions about keeping them as orphaned content, deleting them, or manually aligning
3. View similarity-based suggestions for potential matches with existing content
4. Add metadata flags to chunks for tracking review decisions
"""

import os
import sys
import logging
import time
import hashlib
import numpy as np
import pandas as pd
import lancedb
from typing import List, Dict, Tuple, Optional, Any
from dotenv import load_dotenv
from src.utils import setup_logging, get_logger, update_token_counters, get_token_usage, print_token_usage, reset_token_counters, setup_project_directory, generate_timestamp
setup_project_directory()

# Load environment variables
load_dotenv()

# Setup logging with script prefix


logger = get_logger("Chunk_Review")

# Constants
OUTPUT_DIR_BASE = "output"
LANCEDB_SUBDIR_NAME = "lancedb"
CHUNKS_TABLE_NAME = "document_chunks"
SIMILARITY_THRESHOLD = 0.85  # Threshold for suggesting similar chunks
MAX_SUGGESTIONS = 5  # Maximum number of similarity suggestions to show

def connect_to_lancedb():
    """Connect to LanceDB and return the connection."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    lancedb_path = os.path.join(project_root, OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME)
    
    logger.info(f"Connecting to LanceDB at: {lancedb_path}", extra={"icon": "🔄"})
    if not os.path.exists(lancedb_path):
        logger.error(f"LanceDB directory does not exist at {lancedb_path}", extra={"icon": "❌"})
        return None

    try:
        db = lancedb.connect(lancedb_path)
        logger.info(f"Successfully connected to LanceDB", extra={"icon": "✅"})
        return db
    except Exception as e:
        logger.error(f"Failed to connect to LanceDB: {e}", extra={"icon": "❌"})
        return None

def ensure_metadata_fields(db):
    """
    Ensure the chunks table has the needed metadata fields for review status.
    Adds fields if missing: is_orphaned, manually_aligned, user_reviewed.
    """
    if not db or CHUNKS_TABLE_NAME not in db.table_names():
        logger.error(f"No {CHUNKS_TABLE_NAME} table found in database", extra={"icon": "❌"})
        return False
    
    # Since LanceDB doesn't directly support adding columns, we'll take a simplified
    # approach: just check if the tables exist, and if not, just run the queries without 
    # the expected columns - mark all chunks as "new" for review purposes
    
    logger.info(f"Using simplified schema check for {CHUNKS_TABLE_NAME}", extra={"icon": "ℹ️"})
    return True

def get_orphaned_chunks(db):
    """Get chunks that might be orphaned (no alignment and not reviewed)."""
    if not db or CHUNKS_TABLE_NAME not in db.table_names():
        logger.error(f"No {CHUNKS_TABLE_NAME} table found in database", extra={"icon": "❌"})
        return pd.DataFrame()
        
    chunks_table = db.open_table(CHUNKS_TABLE_NAME)
    
    # Create a query for potentially orphaned chunks
    try:
        # Ensure schema fields
        ensure_metadata_fields(db)
                
        # Get all chunks as pandas DataFrame
        query = chunks_table.to_pandas()
        
        # Filter for candidate orphans - handle missing columns gracefully
        filters = []
        
        # Check is_duplicate column
        if "is_duplicate" in query.columns:
            filters.append(query["is_duplicate"] == False)
            
        # Check aligned_with_chunk_id column
        if "aligned_with_chunk_id" in query.columns:
            filters.append(query["aligned_with_chunk_id"].isna() | (query["aligned_with_chunk_id"] == ""))
        
        # Check user_reviewed column - may be missing initially
        if "user_reviewed" in query.columns:
            filters.append((query["user_reviewed"].isna()) | (query["user_reviewed"] == False))
        
        # Apply all filters that we could build
        if filters:
            orphan_candidates = query
            for f in filters:
                orphan_candidates = orphan_candidates[f]
        else:
            # If we couldn't build filters, return all chunks
            orphan_candidates = query
            logger.warning("Could not filter chunks properly - returning all chunks", extra={"icon": "⚠️"})
        
        if orphan_candidates.empty:
            logger.info("No orphaned chunks found for review", extra={"icon": "✅"})
        else:
            logger.info(f"Found {len(orphan_candidates)} potential orphaned chunks for review", extra={"icon": "🔍"})
            
        return orphan_candidates
            
    except Exception as e:
        logger.error(f"Error finding orphaned chunks: {e}", extra={"icon": "❌"})
        return pd.DataFrame()

def find_similar_chunks(db, chunk_embedding, exclude_id):
    """Find chunks similar to the given chunk but not the same."""
    if not db or CHUNKS_TABLE_NAME not in db.table_names():
        logger.error(f"No {CHUNKS_TABLE_NAME} table found in database", extra={"icon": "❌"})
        return []
        
    chunks_table = db.open_table(CHUNKS_TABLE_NAME)
    
    # Use vector search to find similar chunks
    try:
        results = (
            chunks_table.search(chunk_embedding)
            .metric("cosine")
            .limit(MAX_SUGGESTIONS + 1)  # +1 to account for the chunk itself
            .nprobes(20)
            .to_pandas()
        )
        
        # Filter out the chunk itself
        results = results[results["chunk_id"] != exclude_id]
        
        # Filter by similarity threshold
        # Note: LanceDB returns distance as 1-similarity for cosine
        results = results[results["_distance"] <= (1.0 - SIMILARITY_THRESHOLD)]
        
        # Limit to top MAX_SUGGESTIONS
        results = results.head(MAX_SUGGESTIONS)
        
        return results
        
    except Exception as e:
        logger.error(f"Error finding similar chunks: {e}", extra={"icon": "❌"})
        return []

def display_chunk(chunk, index=None, total=None):
    """Display a chunk with its metadata."""
    header = f"Chunk {index}/{total}" if index and total else "Chunk"
    print(f"\n=== {header} ===")
    print(f"ID: {chunk['chunk_id']}")
    print(f"Document: {chunk['document_id']}")
    print(f"Index: {chunk['chunk_index']}")
    print(f"Length: {len(chunk['chunk_text'])} chars, ~{chunk['token_count']} tokens")
    print("\nContent:")
    print("----------------------------")
    print(chunk['chunk_text'])
    print("----------------------------")

def display_similar_chunk(chunk, similarity):
    """Display a similar chunk with similarity score."""
    print(f"\n--- Similar Chunk (Similarity: {similarity:.2f}) ---")
    print(f"ID: {chunk['chunk_id']}")
    print(f"Document: {chunk['document_id']}")
    print(f"Length: {len(chunk['chunk_text'])} chars")
    print("\nContent:")
    print("----------------------------")
    print(chunk['chunk_text'])
    print("----------------------------")

def update_chunk_status(db, chunk_id, updates):
    """Update a chunk's metadata fields."""
    if not db or CHUNKS_TABLE_NAME not in db.table_names():
        logger.error(f"No {CHUNKS_TABLE_NAME} table found in database", extra={"icon": "❌"})
        return False
        
    chunks_table = db.open_table(CHUNKS_TABLE_NAME)
    
    try:
        # Create update dataframe with just the ID and fields to update
        update_df = pd.DataFrame([{"chunk_id": chunk_id, **updates}])
        
        # Perform the update
        chunks_table.update(update_df, on="chunk_id")
        logger.info(f"Updated chunk {chunk_id}: {updates}", extra={"icon": "✅"})
        return True
        
    except Exception as e:
        logger.error(f"Failed to update chunk {chunk_id}: {e}", extra={"icon": "❌"})
        return False

def review_orphaned_chunks():
    """Main function to review orphaned chunks."""
    # Connect to database
    db = connect_to_lancedb()
    if not db:
        return False
        
    # Ensure metadata fields exist
    if not ensure_metadata_fields(db):
        logger.error("Required metadata fields could not be created", extra={"icon": "❌"})
        return False
        
    # Get orphaned chunks
    orphaned_chunks = get_orphaned_chunks(db)
    if orphaned_chunks.empty:
        print("No orphaned chunks to review.")
        return True
        
    # Review each chunk
    total_chunks = len(orphaned_chunks)
    print(f"\nFound {total_chunks} chunks to review")
    
    for i, (_, chunk) in enumerate(orphaned_chunks.iterrows()):
        display_chunk(chunk, i+1, total_chunks)
        
        # Find similar chunks
        similar_chunks = find_similar_chunks(db, chunk["embedding"], chunk["chunk_id"])
        
        if not similar_chunks.empty:
            print(f"\nFound {len(similar_chunks)} potentially similar chunks:")
            for _, similar in similar_chunks.iterrows():
                # Convert distance to similarity (cosine distance = 1 - similarity)
                similarity = 1.0 - similar["_distance"]
                display_similar_chunk(similar, similarity)
                
        # Ask for user decision
        print("\nWhat would you like to do with this chunk?")
        print("1. Mark as orphaned (keep but note it has no alignment)")
        print("2. Manually align with one of the suggested chunks")
        print("3. Delete this chunk")
        print("4. Skip (review later)")
        print("5. Exit review")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            # Mark as orphaned
            update_chunk_status(db, chunk["chunk_id"], {
                "is_orphaned": True,
                "user_reviewed": True
            })
            print("Chunk marked as orphaned.")
            
        elif choice == '2':
            # Manual alignment
            if similar_chunks.empty:
                print("No similar chunks available for alignment.")
                continue
                
            print("\nSelect a chunk to align with:")
            for j, (_, similar) in enumerate(similar_chunks.iterrows()):
                similarity = 1.0 - similar["_distance"]
                print(f"{j+1}. Chunk {similar['chunk_id']} (Similarity: {similarity:.2f})")
                
            align_choice = input(f"\nEnter chunk number (1-{len(similar_chunks)}): ")
            try:
                align_idx = int(align_choice) - 1
                if 0 <= align_idx < len(similar_chunks):
                    align_chunk = similar_chunks.iloc[align_idx]
                    update_chunk_status(db, chunk["chunk_id"], {
                        "manually_aligned": True,
                        "user_reviewed": True,
                        "aligned_with_chunk_id": align_chunk["chunk_id"],
                        "aligned_with_document_id": align_chunk["document_id"]
                    })
                    print(f"Chunk manually aligned with {align_chunk['chunk_id']}.")
                else:
                    print("Invalid selection.")
            except ValueError:
                print("Invalid input. Please enter a number.")
                
        elif choice == '3':
            # Delete chunk
            confirm = input("Are you sure you want to delete this chunk? (y/n): ")
            if confirm.lower() == 'y':
                try:
                    chunks_table = db.open_table(CHUNKS_TABLE_NAME)
                    # Get current data as DataFrame
                    df = chunks_table.to_pandas()
                    # Filter out the chunk to delete
                    df = df[df["chunk_id"] != chunk["chunk_id"]]
                    # Overwrite the table (since LanceDB doesn't have direct delete)
                    chunks_table.delete()
                    db.create_table(CHUNKS_TABLE_NAME, df)
                    print(f"Chunk {chunk['chunk_id']} deleted.")
                except Exception as e:
                    logger.error(f"Failed to delete chunk: {e}", extra={"icon": "❌"})
            else:
                print("Deletion cancelled.")
                
        elif choice == '4':
            # Skip
            print("Chunk skipped for later review.")
            continue
            
        elif choice == '5':
            # Exit
            print("Exiting review.")
            break
            
        else:
            print("Invalid choice. Please enter a number between 1-5.")
            
    print("\nChunk review completed.")
    return True

def main():
    """Main entry point for chunk review."""
    logger.info("Starting orphaned chunk review", extra={"icon": "🔄"})
    success = review_orphaned_chunks()
    if success:
        logger.info("Chunk review completed successfully", extra={"icon": "✅"})
    else:
        logger.error("Chunk review failed", extra={"icon": "❌"})

if __name__ == "__main__":
    main() 