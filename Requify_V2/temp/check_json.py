#!/usr/bin/env python3
"""
Simple script to check the structure of JSON files and whether they contain embeddings.
"""

import os
import sys
import json
import glob

# Find all the combined JSON files
json_files = glob.glob("output/parsed_content/fighter_jet_*/fighter_jet_*_combined.json")

print(f"Found {len(json_files)} JSON files:")
for json_file in json_files:
    print(f"\nChecking {json_file}")
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        # Check top-level keys
        print(f"Top-level keys: {list(data.keys())}")
        
        # Check for embedding field
        has_embedding = 'embedding' in data
        print(f"Has top-level embedding field: {has_embedding}")
        
        # Check for pages field
        has_pages = 'pages' in data
        print(f"Has pages field: {has_pages}")
        
        # If it has pages, check page structure
        if has_pages:
            num_pages = len(data['pages'])
            print(f"Number of pages: {num_pages}")
            
            # Check first page structure
            first_page_key = list(data['pages'].keys())[0]
            first_page = data['pages'][first_page_key]
            print(f"First page keys: {list(first_page.keys())}")
            
            # Check if page has embedding
            has_page_embedding = 'embedding' in first_page
            print(f"First page has embedding field: {has_page_embedding}")
            
            if has_page_embedding:
                emb = first_page['embedding']
                print(f"Embedding type: {type(emb)}")
                if isinstance(emb, list):
                    print(f"Embedding length: {len(emb)}")
                    print(f"First 5 values: {emb[:5]}")
                    print(f"Last 5 values: {emb[-5:]}")
                    print(f"All zeros: {all(v == 0 for v in emb)}")
                else:
                    print(f"Embedding value: {emb}")
    except Exception as e:
        print(f"Error processing {json_file}: {e}")

print("\nDone checking JSON files.") 