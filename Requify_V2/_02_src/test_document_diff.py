"""
test_document_diff.py

This script extracts content from our test documents and shows the differences between them.
This helps demonstrate what changes our context-aware chunking system should detect.
"""

import os
import sys
import difflib
import argparse
import logging
from dotenv import load_dotenv

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import _00_utils
_00_utils.setup_project_directory()

# Load environment variables
load_dotenv()

# Setup logging
logger = _00_utils.setup_logging()

# Test document paths
DOC1_PATH = os.path.join("_01_input", "raw", "fighter_jet_rocket_launcher_spec_2.pdf")
DOC2_PATH = os.path.join("_01_input", "raw", "fighter_jet_rocket_launcher_spec_2_changed_values.pdf")

def extract_text_from_pdf(pdf_path):
    """Extract text content from a PDF file using our parsing module."""
    try:
        # Import PDF parsing modules
        from _02_parsing.stable_pdf_parsing import PDFProcessor, plain_agent, structured_agent, document_title_agent
        
        # Setup the processor
        processor = PDFProcessor(plain_agent, structured_agent, document_title_agent)
        
        # Convert to combined JSON
        combined_json_path = processor.pdf_to_structured_json(pdf_path)
        
        # Load the JSON to extract text
        import json
        with open(combined_json_path, 'r', encoding='utf-8') as f:
            doc_data = json.load(f)
        
        # Extract document text (combine all page content)
        document_text = ""
        
        # Single page with markdown content
        if 'md_content' in doc_data:
            document_text = doc_data.get('md_content', '')
        # Multi-page document
        elif 'pages' in doc_data:
            for page_key, page_info in sorted(doc_data.get('pages', {}).items()):
                document_text += page_info.get('md_content', '') + "\n\n"
        
        return document_text
        
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return ""

def show_document_diff():
    """Display differences between the two test documents."""
    # Get full paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    doc1_full_path = os.path.join(project_root, DOC1_PATH)
    doc2_full_path = os.path.join(project_root, DOC2_PATH)
    
    # Verify documents exist
    if not os.path.exists(doc1_full_path):
        logger.error(f"❌ First document not found at {doc1_full_path}")
        return False
    
    if not os.path.exists(doc2_full_path):
        logger.error(f"❌ Second document not found at {doc2_full_path}")
        return False
    
    # Extract text from both documents
    logger.info(f"Extracting text from first document: {DOC1_PATH}")
    doc1_text = extract_text_from_pdf(doc1_full_path)
    
    logger.info(f"Extracting text from second document: {DOC2_PATH}")
    doc2_text = extract_text_from_pdf(doc2_full_path)
    
    if not doc1_text or not doc2_text:
        logger.error("❌ Failed to extract text from one or both documents")
        return False
    
    # Generate and display differences
    logger.info("Generating diff between documents:")
    
    # Split text into lines
    doc1_lines = doc1_text.splitlines()
    doc2_lines = doc2_text.splitlines()
    
    # Get line differences
    diff = list(difflib.unified_diff(
        doc1_lines, 
        doc2_lines, 
        fromfile='Original Document', 
        tofile='Changed Values Document', 
        lineterm='', 
        n=3  # Context lines
    ))
    
    # Output the differences
    if diff:
        print("\n========== DOCUMENT DIFFERENCES ==========\n")
        for line in diff:
            if line.startswith('+'):
                # Added line - green
                print(f"\033[92m{line}\033[0m")
            elif line.startswith('-'):
                # Removed line - red
                print(f"\033[91m{line}\033[0m")
            elif line.startswith('@@'):
                # Line position indicator - cyan
                print(f"\033[96m{line}\033[0m")
            else:
                # Unchanged context - white
                print(line)
        print("\n=========================================\n")
    else:
        print("\nNo differences found between documents!\n")
    
    return True

def main():
    show_document_diff()
    return True

if __name__ == "__main__":
    sys.exit(0 if main() else 1) 