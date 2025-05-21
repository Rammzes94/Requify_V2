#!/usr/bin/env python3
"""
Simple script to check the LanceDB database tables and embeddings.
"""

import os
import sys
import lancedb
import numpy as np

# Connect to the LanceDB database
db_path = "output/lancedb"
print(f"Connecting to LanceDB at {db_path}")
db = lancedb.connect(db_path)

# Get list of tables
tables = db.table_names()
print(f"Tables in database: {tables}")

# Check each table
for table_name in tables:
    print(f"\n==== Examining table: {table_name} ====")
    table = db.open_table(table_name)
    
    # Get table schema
    schema = table.schema
    print(f"Schema fields: {[field.name for field in schema]}")
    
    # Get data
    try:
        df = table.to_pandas()
        print(f"Row count: {len(df)}")
        
        if len(df) > 0:
            print(f"Column names: {list(df.columns)}")
            
            # Check for embedding column
            if 'embedding' in df.columns:
                first_embedding = df.iloc[0]['embedding']
                print(f"First embedding type: {type(first_embedding)}")
                
                if isinstance(first_embedding, list):
                    print(f"Embedding length: {len(first_embedding)}")
                    print(f"First 5 values: {first_embedding[:5]}")
                    print(f"Last 5 values: {first_embedding[-5:]}")
                    
                    # Check for all zeros
                    all_zeros = all(v == 0 for v in first_embedding)
                    print(f"All zeros: {all_zeros}")
                    
                    # Check for NaN values 
                    has_nans = any(np.isnan(v) for v in first_embedding if isinstance(v, float))
                    print(f"Has NaNs: {has_nans}")
                elif isinstance(first_embedding, np.ndarray):
                    print(f"Embedding length: {len(first_embedding)}")
                    print(f"First 5 values: {first_embedding[:5]}")
                    print(f"All zeros: {np.all(first_embedding == 0)}")
                    print(f"Has NaNs: {np.isnan(first_embedding).any()}")
            else:
                print("No embedding column found")
                
            # If the table is 'documents', also check document identifiers
            if table_name == 'documents':
                print("\nDocument identifiers in database:")
                if 'pdf_identifier' in df.columns:
                    for doc_id in df['pdf_identifier'].unique():
                        doc_count = len(df[df['pdf_identifier'] == doc_id])
                        print(f"  - {doc_id}: {doc_count} pages")
    except Exception as e:
        print(f"Error examining table data: {e}")

print("\nDone checking database.") 