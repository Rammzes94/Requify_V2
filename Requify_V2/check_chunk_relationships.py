"""
check_chunk_relationships.py

This script examines the chunks in the LanceDB database to identify any issues
with the relationships between original chunks and their replacements.
"""

import os
import sys
import pandas as pd
import lancedb
from dotenv import load_dotenv

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import _02_src._00_utils as _00_utils
_00_utils.setup_project_directory()

# Load environment variables
load_dotenv()

# Define important constants
OUTPUT_DIR = "_03_output"
LANCEDB_SUBDIR = "lancedb"
CHUNKS_TABLE_NAME = "document_chunks"

def main():
    """Check chunk relationships."""
    # Connect to LanceDB
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lancedb_path = os.path.join(script_dir, OUTPUT_DIR, LANCEDB_SUBDIR)
    print(f"Connecting to LanceDB at: {lancedb_path}")
    
    db = lancedb.connect(lancedb_path)
    
    # Get all chunks
    chunks_table = db.open_table(CHUNKS_TABLE_NAME)
    chunks_df = chunks_table.to_pandas()
    
    print(f"\n🔍 Total chunks in database: {len(chunks_df)}")
    
    # List all chunk IDs
    print("\n📋 All chunk IDs:")
    for i, chunk_id in enumerate(chunks_df['chunk_id']):
        print(f"  {i+1}. {chunk_id}")
    
    # Check for chunks marked as replaced
    replaced_chunks = chunks_df[chunks_df['is_replaced'] == True]
    print(f"\n🔄 Chunks marked as replaced: {len(replaced_chunks)}")
    
    if not replaced_chunks.empty:
        print("\nReplaced chunks and their replacements:")
        for _, chunk in replaced_chunks.iterrows():
            chunk_id = chunk['chunk_id']
            replaced_by = chunk['replaced_by']
            print(f"  - {chunk_id} -> {replaced_by}")
            
            # Check if the replacement chunk exists
            if replaced_by in chunks_df['chunk_id'].values:
                print(f"    ✅ Replacement chunk exists")
                # Check the reverse relationship
                replacement_chunk = chunks_df[chunks_df['chunk_id'] == replaced_by].iloc[0]
                if 'previous_chunk_id' in replacement_chunk and replacement_chunk['previous_chunk_id'] == chunk_id:
                    print(f"    ✅ Replacement chunk correctly references original")
                else:
                    print(f"    ❌ Replacement chunk does not reference original properly")
                    print(f"       Previous chunk ID: {replacement_chunk.get('previous_chunk_id', 'None')}")
            else:
                print(f"    ❌ Replacement chunk doesn't exist in database!")
    
    # Check for chunks with previous_chunk_id
    updated_chunks = chunks_df[chunks_df['previous_chunk_id'] != ""]
    print(f"\n🔄 Chunks with previous_chunk_id (updated chunks): {len(updated_chunks)}")
    
    if not updated_chunks.empty:
        print("\nUpdated chunks and their previous versions:")
        for _, chunk in updated_chunks.iterrows():
            chunk_id = chunk['chunk_id']
            previous_id = chunk['previous_chunk_id']
            print(f"  - {chunk_id} <- {previous_id}")
            
            # Check if the previous chunk exists
            if previous_id in chunks_df['chunk_id'].values:
                print(f"    ✅ Previous chunk exists")
                # Check the reverse relationship
                previous_chunk = chunks_df[chunks_df['chunk_id'] == previous_id].iloc[0]
                if 'replaced_by' in previous_chunk and previous_chunk['replaced_by'] == chunk_id:
                    print(f"    ✅ Previous chunk correctly shows replacement")
                else:
                    print(f"    ❌ Previous chunk does not show proper replacement")
                    print(f"       Replaced by: {previous_chunk.get('replaced_by', 'None')}")
            else:
                print(f"    ❌ Previous chunk doesn't exist in database!")
    
    # Check for inconsistencies in the relationships
    broken_relationships = []
    for _, chunk in chunks_df.iterrows():
        # Check replaced chunks
        if chunk['is_replaced'] and chunk['replaced_by']:
            replacement_id = chunk['replaced_by']
            if replacement_id not in chunks_df['chunk_id'].values:
                broken_relationships.append({
                    'chunk_id': chunk['chunk_id'],
                    'issue': f"Replacement chunk {replacement_id} not found",
                    'type': 'missing_replacement'
                })
        
        # Check updated chunks
        if chunk['previous_chunk_id']:
            previous_id = chunk['previous_chunk_id']
            if previous_id not in chunks_df['chunk_id'].values:
                broken_relationships.append({
                    'chunk_id': chunk['chunk_id'],
                    'issue': f"Previous chunk {previous_id} not found",
                    'type': 'missing_previous'
                })
    
    if broken_relationships:
        print("\n⚠️ Broken relationships found:")
        for rel in broken_relationships:
            print(f"  - {rel['chunk_id']}: {rel['issue']}")
    else:
        print("\n✅ No broken relationships found")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 