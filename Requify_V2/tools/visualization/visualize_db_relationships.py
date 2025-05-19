#!/usr/bin/env python3
"""
visualize_db_relationships.py

This script generates visualizations of document and chunk relationships in the LanceDB database.
It creates network graphs showing how documents are related through shared or similar chunks,
helping to understand the deduplication and version relationships.
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Any, Optional, Tuple, Set
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Add the project root directory to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# Import project utilities
from _02_src._00_utils import get_logger, setup_project_directory

# Load environment variables
load_dotenv()

# Set up logging
logger = get_logger("DB_Visualizer")

# Constants
OUTPUT_DIR = os.path.join(project_root, "_03_output")
LANCEDB_DIR = os.path.join(OUTPUT_DIR, "lancedb")
TEST_LANCEDB_DIR = os.path.join(project_root, "tests", "e2e", "_03_output", "lancedb")
VALIDATION_LANCEDB_DIR = os.path.join(project_root, "tools", "validation", "_03_output", "lancedb")
VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
MIN_SIMILARITY_THRESHOLD = 0.1  # Minimum similarity threshold for visualizing relationships

# Ensure visualizations directory exists
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

# Try to import visualization libraries, but make them optional
try:
    import matplotlib.pyplot as plt
    import networkx as nx
    VISUALIZATION_AVAILABLE = True
except ImportError:
    logger.warning("Matplotlib and/or NetworkX not available. Visualizations will be limited to JSON output.", extra={"icon": "⚠️"})
    VISUALIZATION_AVAILABLE = False

def connect_to_lancedb(db_path=None):
    """
    Connect to LanceDB and return the database object
    
    Args:
        db_path: Optional path to the LanceDB directory. If None, uses default path.
    """
    try:
        import lancedb
        
        # Use provided path or default
        lancedb_path = db_path or LANCEDB_DIR
        
        # Check if database directory exists
        if not os.path.exists(lancedb_path):
            logger.error(f"LanceDB directory not found: {lancedb_path}", extra={"icon": "❌"})
            return None
        
        # Connect to database
        logger.info(f"Connecting to LanceDB at: {lancedb_path}", extra={"icon": "🔄"})
        db = lancedb.connect(lancedb_path)
        return db
    except ImportError:
        logger.error("LanceDB not installed. Please install with 'pip install lancedb'", extra={"icon": "❌"})
        return None
    except Exception as e:
        logger.error(f"Error connecting to LanceDB: {str(e)}", extra={"icon": "❌"})
        return None

def get_document_relationships(db) -> Dict[str, Any]:
    """
    Analyze document relationships based on chunk similarity
    
    Returns:
        Dictionary with document relationship data
    """
    if not db:
        return {}
    
    # Check if required tables exist
    required_tables = ["documents", "document_chunks"]
    for table_name in required_tables:
        if table_name not in db.table_names():
            logger.error(f"Required table '{table_name}' not found in database", extra={"icon": "❌"})
            return {}
    
    # Load documents and chunks
    try:
        doc_table = db.open_table("documents")
        chunks_table = db.open_table("document_chunks")
        
        logger.info("Loading document data from LanceDB...", extra={"icon": "🔄"})
        docs_df = doc_table.to_pandas()
        logger.info("Loading chunk data from LanceDB...", extra={"icon": "🔄"})
        chunks_df = chunks_table.to_pandas()
        
        # Get unique document IDs
        doc_ids = set(docs_df['pdf_identifier'].unique())
        
        # Build document relationships based on shared chunks
        relationships = {}
        
        # Group chunks by document
        logger.info("Grouping chunks by document...", extra={"icon": "🔄"})
        doc_chunks = {}
        for doc_id in doc_ids:
            doc_chunks[doc_id] = chunks_df[chunks_df['document_id'] == doc_id]
        
        # For each document pair, calculate relationship metrics
        logger.info("Analyzing document relationships...", extra={"icon": "🔄"})
        for doc_id in doc_ids:
            relationships[doc_id] = {
                "similar_docs": [],
                "chunk_count": len(doc_chunks.get(doc_id, [])),
                "unique_chunks": 0,
                "shared_chunks": 0
            }
            
            # Skip documents with no chunks
            if doc_id not in doc_chunks or len(doc_chunks[doc_id]) == 0:
                continue
                
            # Check relationships with other documents
            for other_doc_id in doc_ids:
                if doc_id == other_doc_id or other_doc_id not in doc_chunks:
                    continue
                
                # Find shared chunks (based on common chunk IDs or high similarity)
                shared = 0
                doc_chunk_ids = set(doc_chunks[doc_id]['chunk_id'])
                other_chunk_ids = set(doc_chunks[other_doc_id]['chunk_id'])
                
                # Count exact matches (same chunk ID)
                exact_matches = len(doc_chunk_ids.intersection(other_chunk_ids))
                shared += exact_matches
                
                # Calculate relationship metrics
                if shared > 0:
                    similarity = shared / len(doc_chunks[doc_id])
                    relationships[doc_id]["similar_docs"].append({
                        "doc_id": other_doc_id,
                        "shared_chunks": shared,
                        "similarity": similarity
                    })
            
            # Sort similar docs by similarity (descending)
            relationships[doc_id]["similar_docs"].sort(key=lambda x: x["similarity"], reverse=True)
            
            # Count unique chunks (not shared with any other document)
            unique_chunks = 0
            for _, chunk in doc_chunks[doc_id].iterrows():
                chunk_id = chunk['chunk_id']
                is_unique = True
                
                for other_doc_id in doc_ids:
                    if doc_id == other_doc_id:
                        continue
                    
                    if other_doc_id in doc_chunks and chunk_id in set(doc_chunks[other_doc_id]['chunk_id']):
                        is_unique = False
                        break
                
                if is_unique:
                    unique_chunks += 1
            
            relationships[doc_id]["unique_chunks"] = unique_chunks
            relationships[doc_id]["shared_chunks"] = relationships[doc_id]["chunk_count"] - unique_chunks
        
        return {
            "documents": len(doc_ids),
            "chunks": len(chunks_df),
            "relationships": relationships
        }
        
    except Exception as e:
        logger.error(f"Error analyzing document relationships: {str(e)}", extra={"icon": "❌"})
        return {}

def create_network_graph(relationships: Dict[str, Any], output_file: str):
    """
    Create a network graph of document relationships
    
    Args:
        relationships: Dictionary with document relationship data
        output_file: Path to save the graph image
    """
    if not VISUALIZATION_AVAILABLE:
        logger.warning("Visualization libraries not available. Skipping graph creation.", extra={"icon": "⚠️"})
        return
    
    try:
        # Create graph
        G = nx.Graph()
        
        # Add nodes (documents)
        for doc_id, data in relationships.get('relationships', {}).items():
            # Use node size proportional to chunk count
            size = data.get('chunk_count', 1) * 100
            
            # Shorten document name for display
            display_name = Path(doc_id).stem
            if len(display_name) > 20:
                display_name = display_name[:17] + "..."
            
            G.add_node(doc_id, size=size, label=display_name)
        
        # Add edges (relationships)
        for doc_id, data in relationships.get('relationships', {}).items():
            for similar in data.get('similar_docs', []):
                other_doc_id = similar.get('doc_id')
                similarity = similar.get('similarity', 0)
                shared_chunks = similar.get('shared_chunks', 0)
                
                # Only add edges with significant similarity
                if similarity >= MIN_SIMILARITY_THRESHOLD:
                    # Scale edge width based on similarity
                    width = similarity * 5
                    G.add_edge(doc_id, other_doc_id, weight=similarity, width=width, 
                              label=f"{shared_chunks} chunks\n{similarity:.2f} sim")
        
        # Draw the graph
        plt.figure(figsize=(12, 10))
        
        # Calculate node positions using spring layout
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # Get node sizes
        node_sizes = [G.nodes[n].get('size', 300) for n in G.nodes()]
        
        # Get edge widths
        edge_widths = [G.edges[e].get('width', 1.0) for e in G.edges()]
        
        # Draw nodes and edges
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="lightblue", alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color="gray")
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
        
        # Add a title
        plt.title(f"Document Relationship Graph ({len(G.nodes())} documents, {len(G.edges())} relationships)")
        plt.axis("off")
        
        # Save the figure
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Network graph saved to {output_file}", extra={"icon": "📊"})
        
    except Exception as e:
        logger.error(f"Error creating network graph: {str(e)}", extra={"icon": "❌"})

def get_available_databases():
    """Return a list of available LanceDB databases in the project"""
    databases = []
    
    # Check main database
    if os.path.exists(LANCEDB_DIR):
        databases.append(("main", LANCEDB_DIR))
    
    # Check test database
    if os.path.exists(TEST_LANCEDB_DIR):
        databases.append(("test", TEST_LANCEDB_DIR))
    
    # Check validation database
    if os.path.exists(VALIDATION_LANCEDB_DIR):
        databases.append(("validation", VALIDATION_LANCEDB_DIR))
    
    return databases

def prompt_for_database():
    """Prompt the user to select a database"""
    databases = get_available_databases()
    
    if not databases:
        logger.error("No LanceDB databases found in the project", extra={"icon": "❌"})
        return None
    
    print("\nAvailable LanceDB databases:")
    for i, (name, path) in enumerate(databases):
        print(f"{i+1}. {name} ({path})")
    
    while True:
        try:
            choice = input("\nSelect a database (enter number or 'q' to quit): ")
            if choice.lower() == 'q':
                return None
            
            choice = int(choice)
            if 1 <= choice <= len(databases):
                return databases[choice-1][1]
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number or 'q'.")

def main():
    """Main entry point"""
    # Declare global variable at the beginning of the function
    global MIN_SIMILARITY_THRESHOLD
    
    # Setup project directory
    setup_project_directory()
    
    parser = argparse.ArgumentParser(description='Visualize document and chunk relationships in LanceDB')
    parser.add_argument('--output', type=str, help='Base name for output files (without extension)')
    parser.add_argument('--min-similarity', type=float, default=MIN_SIMILARITY_THRESHOLD, 
                        help=f'Minimum similarity threshold for visualizing relationships (default: {MIN_SIMILARITY_THRESHOLD})')
    parser.add_argument('--db-path', type=str, help='Path to LanceDB directory')
    parser.add_argument('--db-type', type=str, choices=['main', 'test', 'validation'], 
                        help='Type of database to use (main, test, or validation)')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode to select database')
    
    args = parser.parse_args()
    
    # Update similarity threshold if provided
    MIN_SIMILARITY_THRESHOLD = args.min_similarity
    
    # Determine which database to use
    db_path = None
    
    if args.interactive:
        # Interactive mode - prompt user for database selection
        db_path = prompt_for_database()
        if not db_path:
            logger.info("No database selected. Exiting.", extra={"icon": "🛑"})
            return 0
    elif args.db_path:
        # Use specified path
        db_path = args.db_path
    elif args.db_type:
        # Use specified database type
        if args.db_type == 'main':
            db_path = LANCEDB_DIR
        elif args.db_type == 'test':
            db_path = TEST_LANCEDB_DIR
        elif args.db_type == 'validation':
            db_path = VALIDATION_LANCEDB_DIR
    
    # Connect to database
    db = connect_to_lancedb(db_path)
    if not db:
        return 1
    
    # Get document relationships
    logger.info("Analyzing document relationships...", extra={"icon": "🔍"})
    relationships = get_document_relationships(db)
    
    if not relationships:
        logger.error("No document relationships found", extra={"icon": "❌"})
        return 1
    
    # Determine output files
    if args.output:
        base_name = args.output
    else:
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        db_type = args.db_type or "custom" if args.db_path else "main"
        base_name = f"doc_relationships_{db_type}_{timestamp}"
    
    # Save relationships as JSON
    json_file = os.path.join(VISUALIZATIONS_DIR, f"{base_name}.json")
    with open(json_file, 'w') as f:
        json.dump(relationships, f, indent=2)
    
    logger.info(f"Relationship data saved to {json_file}", extra={"icon": "💾"})
    
    # Create network graph
    if VISUALIZATION_AVAILABLE:
        logger.info("Creating network graph...", extra={"icon": "🔗"})
        graph_file = os.path.join(VISUALIZATIONS_DIR, f"{base_name}.png")
        create_network_graph(relationships, graph_file)
    
    # Print summary statistics
    doc_count = relationships.get('documents', 0)
    chunk_count = relationships.get('chunks', 0)
    
    logger.info(f"Summary: {doc_count} documents with {chunk_count} chunks", extra={"icon": "📋"})
    
    # Count relationships
    relationship_count = 0
    for doc_id, data in relationships.get('relationships', {}).items():
        relationship_count += len(data.get('similar_docs', []))
    
    logger.info(f"Found {relationship_count} document relationships", extra={"icon": "🔗"})
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 