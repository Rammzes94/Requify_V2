#!/usr/bin/env python3
"""
test_document_diff.py

This script compares the contents of two documents to identify differences.
It performs the following tasks:
1. Loads source JSON files created by the document processing pipeline
2. Extracts text content from both documents
3. Performs a line-by-line diff showing additions, deletions, and changes
4. Displays a color-coded comparison in the terminal
5. Works with document IDs or file paths for maximum flexibility

Used for debugging and validating document processing to see how different
versions of documents compare at the text level.
"""

import os
import sys
import json
import difflib
import argparse
import logging

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import setup_logging, get_logger, update_token_counters, get_token_usage, print_token_usage, reset_token_counters, setup_project_directory, generate_timestamp
setup_project_directory()

# Set up logging
logger = setup_logging()

# Create a consistent logger with prefix for better visibility


logger = get_logger("Test_Document_Diff")

def load_document_content(doc_path):
    """Load document content from a parsed JSON file"""
    try:
        with open(doc_path, 'r') as f:
            data = json.load(f)
        
        # Extract page contents and combine them
        content = ""
        for page_key, page_data in data.get('pages', {}).items():
            if 'md_content' in page_data:
                content += page_data['md_content'] + "\n\n"
        
        return content
    except Exception as e:
        logger.error(f"Error loading document content: {e}", extra={"icon": "❌"})
        return None

def get_document_path(doc_identifier):
    """
    Get the full path to a document based on ID or path.
    
    This handles both full file paths and document IDs.
    """
    # If it's already a full path to a file that exists, return it
    if os.path.isfile(doc_identifier):
        return doc_identifier
    
    # Otherwise, look for it in the standard parsed output location
    base_name = os.path.basename(doc_identifier)
    # Remove extension if present
    base_name = os.path.splitext(base_name)[0]
    
    # Check if it's in the standard output directory
    output_dir = os.path.join("output", "parsed_content", base_name)
    if os.path.exists(output_dir):
        # Look for combined_content.json
        json_path = os.path.join(output_dir, "combined_content.json")
        if os.path.exists(json_path):
            return json_path
    
    # If we couldn't find it, return the original and let the caller handle errors
    return doc_identifier

def compare_documents(doc1_path, doc2_path):
    """Compare the content of two documents and show the differences"""
    # Get full paths
    doc1_path = get_document_path(doc1_path)
    doc2_path = get_document_path(doc2_path)
    
    # Load document content
    content1 = load_document_content(doc1_path)
    content2 = load_document_content(doc2_path)
    
    if not content1 or not content2:
        logger.error("Failed to load one or both documents", extra={"icon": "❌"})
        return
    
    # Split content into lines
    lines1 = content1.splitlines()
    lines2 = content2.splitlines()
    
    # Generate diff
    diff = list(difflib.unified_diff(
        lines1, lines2, 
        fromfile=os.path.basename(doc1_path),
        tofile=os.path.basename(doc2_path),
        lineterm=''
    ))
    
    # Display diff with colors
    if diff:
        logger.info("\n========== DOCUMENT DIFFERENCES ==========\n", extra={"icon": "📊"})
        for line in diff:
            if line.startswith('+'):
                logger.info(f"\033[92m{line}\033[0m", extra={"icon": "➕"})  # Green for additions
            elif line.startswith('-'):
                logger.info(f"\033[91m{line}\033[0m", extra={"icon": "➖"})  # Red for deletions
            elif line.startswith('^'):
                logger.info(f"\033[96m{line}\033[0m", extra={"icon": "🔄"})  # Cyan for changes
            else:
                logger.info(line, extra={"icon": "ℹ️"})
        logger.info("\n=========================================\n", extra={"icon": "📊"})
    else:
        logger.info("\nNo differences found between documents!\n", extra={"icon": "✅"})

def main():
    parser = argparse.ArgumentParser(description="Compare the content of two documents")
    parser.add_argument('doc1', help="First document ID or path")
    parser.add_argument('doc2', help="Second document ID or path")
    args = parser.parse_args()
    
    compare_documents(args.doc1, args.doc2)

if __name__ == "__main__":
    main() 