"""
show_deduplication_results.py

This script shows the deduplication results that occurred during the most recent pipeline run.
It connects to the LanceDB database and identifies chunks that have deduplication relationships
(replaced, updated, or duplicate chunks) and displays them in a readable format with color-coded output.
"""

import os
import sys
import pandas as pd
import numpy as np
import lancedb
import argparse
from dotenv import load_dotenv
from colorama import Fore, Style, init

# Initialize colorama for cross-platform colored terminal output
init()

# Add the parent directory to the system path to allow importing modules from it
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)
import src._00_utils as _00_utils
_00_utils.setup_project_directory()

# Import constants directly from pre_save_deduplication
from src._03_docs_deduplication.pre_save_deduplication import (
    DUPLICATE_THRESHOLD, 
    SIMILAR_THRESHOLD,
    connect_to_lancedb,
    calculate_cosine_similarity
)

# Load environment variables
load_dotenv()

# Setup logging
logger = _00_utils.setup_logging()

# Constants
LANCEDB_SUBDIR = "lancedb"
OUTPUT_DIR = "output"
CHUNKS_TABLE = "document_chunks"
DOCUMENTS_TABLE = "documents"

def connect_to_lancedb():
    """Connect to LanceDB database."""
    lancedb_path = os.path.join(project_root, OUTPUT_DIR, LANCEDB_SUBDIR)
    logger.info(f"🔌 Connecting to LanceDB at: {lancedb_path}")
    
    if not os.path.exists(lancedb_path):
        logger.error(f"❌ LanceDB directory not found at: {lancedb_path}")
        return None
    
    try:
        db = lancedb.connect(lancedb_path)
        logger.info(f"✅ Connected to LanceDB successfully")
        return db
    except Exception as e:
        logger.error(f"❌ Failed to connect to LanceDB: {e}")
        return None

def get_similarity_color(similarity):
    """Return color code based on similarity value."""
    if similarity >= DUPLICATE_THRESHOLD:
        return Fore.GREEN  # High similarity (duplicate)
    elif similarity >= SIMILAR_THRESHOLD:
        return Fore.YELLOW  # Medium similarity (similar)
    else:
        return Fore.RED  # Low similarity (different)

def show_document_stats(db, target_doc=None):
    """Show document statistics in the database."""
    if DOCUMENTS_TABLE not in db.table_names():
        logger.warning(f"⚠️ Documents table '{DOCUMENTS_TABLE}' not found in database")
        return
    
    docs_table = db.open_table(DOCUMENTS_TABLE)
    docs_df = docs_table.to_pandas()
    
    if target_doc:
        docs_df = docs_df[docs_df['pdf_identifier'].str.contains(target_doc)]
        if docs_df.empty:
            print(f"No documents found matching '{target_doc}'")
            return
    
    unique_docs = docs_df['pdf_identifier'].unique()
    
    print(f"\n{Fore.CYAN}=== Documents in Database ({len(unique_docs)} documents) ==={Style.RESET_ALL}")
    for i, doc_id in enumerate(sorted(unique_docs), 1):
        doc_pages = docs_df[docs_df['pdf_identifier'] == doc_id]
        print(f"{i}. {doc_id}: {len(doc_pages)} pages")

def show_deduplication_stats(db, target_doc=None, show_text=False):
    """Show deduplication statistics from the chunks table."""
    if CHUNKS_TABLE not in db.table_names():
        logger.warning(f"⚠️ Chunks table '{CHUNKS_TABLE}' not found in database")
        return
    
    chunks_table = db.open_table(CHUNKS_TABLE)
    chunks_df = chunks_table.to_pandas()
    
    # Filter by target document if specified
    if target_doc:
        chunks_df = chunks_df[chunks_df['document_id'].str.contains(target_doc)]
        if chunks_df.empty:
            print(f"No chunks found for documents matching '{target_doc}'")
            return
    
    total_chunks = len(chunks_df)
    unique_docs = chunks_df['document_id'].unique()
    
    print(f"\n{Fore.CYAN}=== Deduplication Statistics ==={Style.RESET_ALL}")
    print(f"Total chunks in database: {total_chunks}")
    print(f"Documents with chunks: {len(unique_docs)}")
    
    # Count chunks by document
    doc_counts = chunks_df['document_id'].value_counts().sort_values(ascending=False)
    
    print(f"\n{Fore.CYAN}=== Chunk Counts by Document ==={Style.RESET_ALL}")
    for doc_id, count in doc_counts.items():
        print(f"{doc_id}: {count} chunks")
    
    # Find duplicate chunks by hash
    if 'chunk_hash' in chunks_df.columns:
        duplicate_hashes = chunks_df['chunk_hash'].value_counts()
        duplicate_hashes = duplicate_hashes[duplicate_hashes > 1].sort_values(ascending=False)
        
        if not duplicate_hashes.empty:
            print(f"\n{Fore.CYAN}=== Chunks with Same Hash ({len(duplicate_hashes)} hashes) ==={Style.RESET_ALL}")
            for hash_val, count in duplicate_hashes.items()[:10]:  # Show top 10
                print(f"Hash {hash_val[:8]}...: {count} chunks")
                if count > 5:
                    print("  (showing first 5 chunks)")
                    matching_chunks = chunks_df[chunks_df['chunk_hash'] == hash_val].head(5)
                else:
                    matching_chunks = chunks_df[chunks_df['chunk_hash'] == hash_val]
                
                for _, chunk in matching_chunks.iterrows():
                    print(f"  - {chunk['chunk_id']} (Doc: {chunk['document_id']})")
    
    # Find replaced chunks
    if 'is_replaced' in chunks_df.columns:
        replaced_chunks = chunks_df[chunks_df['is_replaced'] == True]
        
        if not replaced_chunks.empty:
            print(f"\n{Fore.CYAN}=== Replaced Chunks ({len(replaced_chunks)} chunks) ==={Style.RESET_ALL}")
            for _, chunk in replaced_chunks.iterrows():
                print(f"{Fore.RED}Original: {chunk['chunk_id']} (Doc: {chunk['document_id']}){Style.RESET_ALL}")
                print(f"{Fore.GREEN}Replaced by: {chunk['replaced_by']}{Style.RESET_ALL}")
                
                # Find the replacement chunk for more details
                replacement = chunks_df[chunks_df['chunk_id'] == chunk['replaced_by']]
                if not replacement.empty:
                    rep = replacement.iloc[0]
                    print(f"Replacement doc: {rep['document_id']}")
                    
                    # Calculate similarity if embeddings are available
                    if 'embedding' in chunk and 'embedding' in rep:
                        similarity = calculate_cosine_similarity(chunk['embedding'], rep['embedding'])
                        color = get_similarity_color(similarity)
                        print(f"Similarity: {color}{similarity:.4f}{Style.RESET_ALL}")
                    
                    # Show text snippets if requested
                    if show_text:
                        print(f"\nOriginal text:")
                        print(f"{Fore.RED}{chunk['chunk_text'][:200]}...{Style.RESET_ALL}" 
                              if len(chunk['chunk_text']) > 200 else f"{Fore.RED}{chunk['chunk_text']}{Style.RESET_ALL}")
                        print(f"\nReplacement text:")
                        print(f"{Fore.GREEN}{rep['chunk_text'][:200]}...{Style.RESET_ALL}" 
                              if len(rep['chunk_text']) > 200 else f"{Fore.GREEN}{rep['chunk_text']}{Style.RESET_ALL}")
                print("-" * 80)
    
    # Find updated chunks (with previous_chunk_id)
    if 'previous_chunk_id' in chunks_df.columns:
        updated_chunks = chunks_df[chunks_df['previous_chunk_id'].astype(str) != ""]
        
        if not updated_chunks.empty:
            print(f"\n{Fore.CYAN}=== Updated Chunks ({len(updated_chunks)} chunks) ==={Style.RESET_ALL}")
            for _, chunk in updated_chunks.iterrows():
                print(f"{Fore.GREEN}Current: {chunk['chunk_id']} (Doc: {chunk['document_id']}){Style.RESET_ALL}")
                print(f"{Fore.YELLOW}Previous: {chunk['previous_chunk_id']}{Style.RESET_ALL}")
                
                # Find the previous chunk for more details
                previous = chunks_df[chunks_df['chunk_id'] == chunk['previous_chunk_id']]
                if not previous.empty:
                    prev = previous.iloc[0]
                    print(f"Previous version doc: {prev['document_id']}")
                    
                    # Calculate similarity if embeddings are available
                    if 'embedding' in chunk and 'embedding' in prev:
                        similarity = calculate_cosine_similarity(chunk['embedding'], prev['embedding'])
                        color = get_similarity_color(similarity)
                        print(f"Similarity: {color}{similarity:.4f}{Style.RESET_ALL}")
                    
                    # Show text snippets if requested
                    if show_text:
                        print(f"\nCurrent text:")
                        print(f"{Fore.GREEN}{chunk['chunk_text'][:200]}...{Style.RESET_ALL}" 
                              if len(chunk['chunk_text']) > 200 else f"{Fore.GREEN}{chunk['chunk_text']}{Style.RESET_ALL}")
                        print(f"\nPrevious text:")
                        print(f"{Fore.YELLOW}{prev['chunk_text'][:200]}...{Style.RESET_ALL}" 
                              if len(prev['chunk_text']) > 200 else f"{Fore.YELLOW}{prev['chunk_text']}{Style.RESET_ALL}")
                print("-" * 80)
    
    # Find duplicate chunks
    if 'is_duplicate' in chunks_df.columns and 'duplicate_of' in chunks_df.columns:
        duplicate_chunks = chunks_df[chunks_df['is_duplicate'] == True]
        
        if not duplicate_chunks.empty:
            print(f"\n{Fore.CYAN}=== Duplicate Chunks ({len(duplicate_chunks)} chunks) ==={Style.RESET_ALL}")
            for _, chunk in duplicate_chunks.iterrows():
                print(f"{Fore.YELLOW}Duplicate: {chunk['chunk_id']} (Doc: {chunk['document_id']}){Style.RESET_ALL}")
                print(f"{Fore.GREEN}Original: {chunk['duplicate_of']}{Style.RESET_ALL}")
                
                # Find the original chunk for more details
                original = chunks_df[chunks_df['chunk_id'] == chunk['duplicate_of']]
                if not original.empty:
                    orig = original.iloc[0]
                    print(f"Original doc: {orig['document_id']}")
                    
                    # Calculate similarity if embeddings are available
                    if 'embedding' in chunk and 'embedding' in orig:
                        similarity = calculate_cosine_similarity(chunk['embedding'], orig['embedding'])
                        color = get_similarity_color(similarity)
                        print(f"Similarity: {color}{similarity:.4f}{Style.RESET_ALL}")
                    
                    # Show text snippets if requested
                    if show_text and 'chunk_text' in chunk and 'chunk_text' in orig:
                        # Only show text if they're different (in case hash match)
                        if chunk['chunk_text'] != orig['chunk_text']:
                            print(f"\nDuplicate text:")
                            print(f"{Fore.YELLOW}{chunk['chunk_text'][:200]}...{Style.RESET_ALL}" 
                                  if len(chunk['chunk_text']) > 200 else f"{Fore.YELLOW}{chunk['chunk_text']}{Style.RESET_ALL}")
                            print(f"\nOriginal text:")
                            print(f"{Fore.GREEN}{orig['chunk_text'][:200]}...{Style.RESET_ALL}" 
                                  if len(orig['chunk_text']) > 200 else f"{Fore.GREEN}{orig['chunk_text']}{Style.RESET_ALL}")
                        else:
                            print(f"Text is identical")
                print("-" * 80)

def show_document_relationships(db, target_doc=None):
    """Show document version relationships."""
    if DOCUMENTS_TABLE not in db.table_names() or CHUNKS_TABLE not in db.table_names():
        logger.warning("⚠️ Required tables not found in database")
        return
    
    chunks_table = db.open_table(CHUNKS_TABLE)
    chunks_df = chunks_table.to_pandas()
    
    # Get all documents with document relationships
    unique_docs = chunks_df['document_id'].unique()
    
    # Filter by target document if specified
    if target_doc:
        related_docs = set()
        target_chunks = chunks_df[chunks_df['document_id'].str.contains(target_doc)]
        
        # Add target docs
        related_docs.update(target_chunks['document_id'].unique())
        
        # Add documents that this document references
        for col in ['aligned_with_document_id', 'previous_chunk_id']:
            if col in target_chunks.columns:
                related_docs.update([doc for doc in target_chunks[col].unique() if doc and doc != ''])
        
        # Add documents that reference this document
        if 'aligned_with_document_id' in chunks_df.columns:
            referencing = chunks_df[chunks_df['aligned_with_document_id'].str.contains(target_doc, na=False)]
            related_docs.update(referencing['document_id'].unique())
        
        # Filter chunks to only include related documents
        chunks_df = chunks_df[chunks_df['document_id'].isin(related_docs)]
        unique_docs = list(related_docs)
    
    print(f"\n{Fore.CYAN}=== Document Relationships ==={Style.RESET_ALL}")
    
    # Count relationships between documents
    relationships = {}
    
    for doc_id in unique_docs:
        doc_chunks = chunks_df[chunks_df['document_id'] == doc_id]
        
        # Look for document relationships through aligned_with_document_id
        if 'aligned_with_document_id' in doc_chunks.columns:
            for aligned_doc in doc_chunks['aligned_with_document_id'].unique():
                if aligned_doc and aligned_doc != '':
                    key = (doc_id, aligned_doc)
                    if key not in relationships:
                        relationships[key] = {'count': 0, 'type': 'alignment'}
                    relationships[key]['count'] += len(doc_chunks[doc_chunks['aligned_with_document_id'] == aligned_doc])
        
        # Look for chunk relationships through previous_chunk_id
        if 'previous_chunk_id' in doc_chunks.columns:
            for prev_chunk_id in doc_chunks['previous_chunk_id'].unique():
                if prev_chunk_id and prev_chunk_id != '':
                    # Find which document this chunk belongs to
                    source_chunks = chunks_df[chunks_df['chunk_id'] == prev_chunk_id]
                    if not source_chunks.empty:
                        source_doc = source_chunks.iloc[0]['document_id']
                        if source_doc != doc_id:  # Only if from different document
                            key = (doc_id, source_doc)
                            if key not in relationships:
                                relationships[key] = {'count': 0, 'type': 'update'}
                            relationships[key]['count'] += len(doc_chunks[doc_chunks['previous_chunk_id'] == prev_chunk_id])
        
        # Look for duplicate relationships
        if 'duplicate_of' in doc_chunks.columns:
            for dup_chunk_id in doc_chunks['duplicate_of'].unique():
                if dup_chunk_id and dup_chunk_id != '':
                    # Find which document this chunk belongs to
                    source_chunks = chunks_df[chunks_df['chunk_id'] == dup_chunk_id]
                    if not source_chunks.empty:
                        source_doc = source_chunks.iloc[0]['document_id']
                        if source_doc != doc_id:  # Only if from different document
                            key = (doc_id, source_doc)
                            if key not in relationships:
                                relationships[key] = {'count': 0, 'type': 'duplicate'}
                            relationships[key]['count'] += len(doc_chunks[doc_chunks['duplicate_of'] == dup_chunk_id])
    
    # Display the relationships
    if relationships:
        for (doc1, doc2), info in sorted(relationships.items(), key=lambda x: (x[0][0], x[0][1])):
            rel_type = info['type']
            count = info['count']
            
            if rel_type == 'alignment':
                print(f"{doc1} → {doc2}: {count} aligned chunks")
            elif rel_type == 'update':
                print(f"{Fore.GREEN}{doc1}{Style.RESET_ALL} → {Fore.YELLOW}{doc2}{Style.RESET_ALL}: {count} updated chunks")
            elif rel_type == 'duplicate':
                print(f"{Fore.YELLOW}{doc1}{Style.RESET_ALL} → {Fore.GREEN}{doc2}{Style.RESET_ALL}: {count} duplicate chunks")
    else:
        print("No document relationships found")

def main():
    """Main function to show deduplication results."""
    parser = argparse.ArgumentParser(description="Show deduplication results from LanceDB")
    parser.add_argument("--doc", type=str, help="Filter results to a specific document (substring match)")
    parser.add_argument("--text", action="store_true", help="Show chunk text in the output")
    parser.add_argument("--relationships", action="store_true", help="Show document relationships")
    args = parser.parse_args()
    
    logger.info("🔍 Showing deduplication results from the most recent pipeline run")
    
    # Connect to LanceDB
    db = connect_to_lancedb()
    if not db:
        logger.error("❌ Failed to connect to database")
        return
    
    # Show tables in database
    logger.info(f"📋 Tables in database: {db.table_names()}")
    
    # Show document stats
    show_document_stats(db, args.doc)
    
    # Show document relationships if requested
    if args.relationships:
        show_document_relationships(db, args.doc)
    
    # Show deduplication stats
    show_deduplication_stats(db, args.doc, args.text)
    
    logger.info("✅ Deduplication results displayed")

if __name__ == "__main__":
    main() 