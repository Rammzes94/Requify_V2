"""
test_document_diff.py

This script compares two PDF documents to identify differences
between them, which helps us understand what changes 
the context-aware chunking system should detect.
"""

import os
import sys
import json
import difflib
from dotenv import load_dotenv

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import _02_src._00_utils as _00_utils
_00_utils.setup_project_directory()

# Load environment variables
load_dotenv()

# Setup logging
logger = _00_utils.setup_logging()

# Import the PDF processor
from _02_src._02_parsing.stable_pdf_parsing import PDFProcessor, plain_agent, structured_agent, document_title_agent

def extract_document_content(pdf_path):
    """Extract content from a PDF document"""
    processor = PDFProcessor(plain_agent, structured_agent, document_title_agent)
    json_path = processor.pdf_to_structured_json(pdf_path)
    
    with open(json_path, 'r') as f:
        doc_data = json.load(f)
    
    # Extract all text content from pages
    content = ""
    for page_key, page_info in sorted(doc_data.get('pages', {}).items()):
        content += page_info.get('md_content', '') + "\n\n"
    
    return content, doc_data

def compare_documents(doc1_path, doc2_path):
    """Compare two documents and display their differences"""
    logger.info(f"🔍 Comparing documents: {os.path.basename(doc1_path)} and {os.path.basename(doc2_path)}")
    
    # Extract content from both documents
    content1, data1 = extract_document_content(doc1_path)
    content2, data2 = extract_document_content(doc2_path)
    
    # Create a diff
    diff = list(difflib.unified_diff(
        content1.splitlines(),
        content2.splitlines(),
        fromfile=os.path.basename(doc1_path),
        tofile=os.path.basename(doc2_path),
        lineterm=''
    ))
    
    # Print the diff
    print("\n".join(diff))
    
    # Count changes
    additions = [line for line in diff if line.startswith('+') and not line.startswith('+++')]
    deletions = [line for line in diff if line.startswith('-') and not line.startswith('---')]
    
    logger.info(f"📊 Diff summary: {len(additions)} additions, {len(deletions)} deletions")
    
    # Print a summary of key changes
    print("\nKey changes detected:")
    for line in diff:
        if (line.startswith('+') or line.startswith('-')) and not (line.startswith('+++') or line.startswith('---')):
            if any(keyword in line.lower() for keyword in ['weight', 'dimensions', 'speed', 'capacity', 'range', 'kg', 'mm', 'cm', 'rate']):
                print(line)

if __name__ == "__main__":
    # Define the document paths
    doc1_path = os.path.join("_01_input", "raw", "fighter_jet_rocket_launcher_spec_2.pdf")
    doc2_path = os.path.join("_01_input", "raw", "fighter_jet_rocket_launcher_spec_2_changed_values.pdf")
    
    # Compare the documents
    compare_documents(doc1_path, doc2_path) 