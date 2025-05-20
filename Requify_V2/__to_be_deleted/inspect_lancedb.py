#!/usr/bin/env python3
"""
inspect_lancedb.py

A utility script to inspect the contents of LanceDB tables for debugging purposes.
"""

import os
import sys
import lancedb
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple

# Path to LanceDB
LANCEDB_PATH = "_03_output/lancedb"
SIMILARITY_THRESHOLD = 0.9  # Threshold for considering chunks similar

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def main():
    """Main function to inspect LanceDB tables."""
    # Connect to LanceDB
    db = lancedb.connect(LANCEDB_PATH)
    
    # Get list of tables
    tables = db.table_names()
    print(f"Available tables: {tables}")
    
    # Inspect document_chunks table
    if "document_chunks" in tables:
        chunks_table = db.open_table("document_chunks")
        chunks_df = chunks_table.to_pandas()
        
        # Print column names
        print("\nDocument Chunks Table Columns:")
        for col in chunks_df.columns:
            print(f"  - {col}")
        
        # Print summary info
        print(f"\nDocument Chunks Table: {len(chunks_df)} rows")
        print(f"Documents: {chunks_df['document_id'].nunique()} unique documents")
        print(f"Document IDs: {chunks_df['document_id'].unique()}")
        
        print("\n--- DOCUMENT CHUNKS ---\n")
        for i, row in chunks_df.iterrows():
            print(f"Chunk {i+1}:")
            print(f"  Document: {row['document_id']}")
            print(f"  Chunk ID: {row['chunk_id']}")
            print(f"  Text snippet: {row['chunk_text'][:100]}...\n")
        
        # Check if embedding column exists before attempting similarity analysis
        embedding_column = None
        for possible_name in ['embedding', 'chunk_embedding', 'vector']:
            if possible_name in chunks_df.columns:
                embedding_column = possible_name
                break
        
        # Analyze chunk similarity across documents if embedding column exists
        if embedding_column and len(chunks_df['document_id'].unique()) > 1:
            unique_docs = chunks_df['document_id'].unique()
            print(f"\n--- CHUNK SIMILARITY ANALYSIS (using '{embedding_column}' column) ---\n")
            
            # Group chunks by document
            docs_chunks = {}
            for doc_id in unique_docs:
                docs_chunks[doc_id] = chunks_df[chunks_df['document_id'] == doc_id]
            
            # Compare chunks between documents
            for i in range(len(unique_docs) - 1):
                doc1_id = unique_docs[i]
                for j in range(i + 1, len(unique_docs)):
                    doc2_id = unique_docs[j]
                    
                    print(f"Comparing chunks between '{doc1_id}' and '{doc2_id}':")
                    similar_chunks_found = False
                    
                    # For each chunk in doc1, find similar chunks in doc2
                    doc1_chunks = docs_chunks[doc1_id]
                    doc2_chunks = docs_chunks[doc2_id]
                    
                    for _, chunk1 in doc1_chunks.iterrows():
                        best_similarity = 0
                        best_chunk = None
                        
                        for _, chunk2 in doc2_chunks.iterrows():
                            # Calculate similarity
                            sim = cosine_similarity(chunk1[embedding_column], chunk2[embedding_column])
                            
                            if sim > best_similarity:
                                best_similarity = sim
                                best_chunk = chunk2
                        
                        if best_similarity > SIMILARITY_THRESHOLD:
                            similar_chunks_found = True
                            print(f"  Chunk '{chunk1['chunk_id']}' is similar to '{best_chunk['chunk_id']}' (similarity: {best_similarity:.4f})")
                            print(f"    Doc1 text: {chunk1['chunk_text'][:100]}...")
                            print(f"    Doc2 text: {best_chunk['chunk_text'][:100]}...")
                            print()
                    
                    if not similar_chunks_found:
                        print(f"  No similar chunks found above threshold ({SIMILARITY_THRESHOLD})\n")
        elif not embedding_column and len(chunks_df['document_id'].unique()) > 1:
            print("\nSkipping similarity analysis - no embedding column found in the document_chunks table")
    
    # Inspect documents table
    if "documents" in tables:
        documents_table = db.open_table("documents")
        documents_df = documents_table.to_pandas()
        
        # Print column names
        print("\nDocuments Table Columns:")
        for col in documents_df.columns:
            print(f"  - {col}")
        
        print(f"\nDocuments Table: {len(documents_df)} rows")
        print(f"Documents: {documents_df['pdf_identifier'].nunique()} unique documents")
        print(f"Document IDs: {documents_df['pdf_identifier'].unique()}")
        
        # Check if embedding column exists before attempting similarity analysis
        embedding_column = None
        for possible_name in ['embedding', 'page_embedding', 'vector']:
            if possible_name in documents_df.columns:
                embedding_column = possible_name
                break
        
        # Analyze document similarity if embedding column exists
        if embedding_column and len(documents_df['pdf_identifier'].unique()) > 1:
            unique_docs = documents_df['pdf_identifier'].unique()
            print(f"\n--- DOCUMENT SIMILARITY ANALYSIS (using '{embedding_column}' column) ---\n")
            
            for i in range(len(unique_docs) - 1):
                doc1_id = unique_docs[i]
                for j in range(i + 1, len(unique_docs)):
                    doc2_id = unique_docs[j]
                    
                    # Get embeddings for both documents
                    doc1_embeddings = documents_df[documents_df['pdf_identifier'] == doc1_id][embedding_column].tolist()
                    doc2_embeddings = documents_df[documents_df['pdf_identifier'] == doc2_id][embedding_column].tolist()
                    
                    # Compare each page embedding
                    similarities = []
                    for emb1 in doc1_embeddings:
                        for emb2 in doc2_embeddings:
                            similarities.append(cosine_similarity(emb1, emb2))
                    
                    if similarities:
                        avg_sim = sum(similarities) / len(similarities)
                        max_sim = max(similarities)
                        print(f"Similarity between '{doc1_id}' and '{doc2_id}':")
                        print(f"  Average: {avg_sim:.4f}")
                        print(f"  Maximum: {max_sim:.4f}")
                        print()
        elif not embedding_column and len(documents_df['pdf_identifier'].unique()) > 1:
            print("\nSkipping similarity analysis - no embedding column found in the documents table")

if __name__ == "__main__":
    main() 