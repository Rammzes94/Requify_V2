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
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
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
    logger.info(f"📄 Extracting content from {os.path.basename(pdf_path)}")
    processor = PDFProcessor(plain_agent, structured_agent, document_title_agent)
    json_path = processor.pdf_to_structured_json(pdf_path)
    
    with open(json_path, 'r') as f:
        doc_data = json.load(f)
    
    # Extract all text content from pages
    content = ""
    page_count = len(doc_data.get('pages', {}))
    logger.info(f"📑 Extracted {page_count} pages from {os.path.basename(pdf_path)}")
    
    for page_key, page_info in sorted(doc_data.get('pages', {}).items()):
        content += page_info.get('md_content', '') + "\n\n"
        logger.info(f"📃 Page {page_info.get('page_number', '?')}: {len(page_info.get('md_content', ''))} characters")
    
    return content, doc_data

def compare_documents(doc1_path, doc2_path):
    """Compare two documents and display their differences"""
    logger.info(f"🔍 Comparing documents: {os.path.basename(doc1_path)} and {os.path.basename(doc2_path)}")
    
    # Extract content from both documents
    content1, data1 = extract_document_content(doc1_path)
    content2, data2 = extract_document_content(doc2_path)
    
    logger.info(f"📊 Document sizes: {len(content1)} chars vs {len(content2)} chars")
    
    # Create a diff
    logger.info(f"🔄 Generating diff between documents")
    diff = list(difflib.unified_diff(
        content1.splitlines(),
        content2.splitlines(),
        fromfile=os.path.basename(doc1_path),
        tofile=os.path.basename(doc2_path),
        lineterm=''
    ))
    
    # Log the diff
    if diff:
        logger.info(f"📝 Differences found: {len(diff)} lines differ")
        for line in diff[:20]:  # Log first 20 diff lines
            if line.startswith('+') and not line.startswith('+++'):
                logger.info(f"➕ {line}")
            elif line.startswith('-') and not line.startswith('---'):
                logger.info(f"➖ {line}")
        if len(diff) > 20:
            logger.info(f"... and {len(diff) - 20} more differences")
    else:
        logger.info("📋 No differences found between documents")
    
    # Print the diff
    print("\n".join(diff))
    
    # Count changes
    additions = [line for line in diff if line.startswith('+') and not line.startswith('+++')]
    deletions = [line for line in diff if line.startswith('-') and not line.startswith('---')]
    
    logger.info(f"📊 Diff summary: {len(additions)} additions, {len(deletions)} deletions")
    
    # Print a summary of key changes
    key_changes = []
    for line in diff:
        if (line.startswith('+') or line.startswith('-')) and not (line.startswith('+++') or line.startswith('---')):
            if any(keyword in line.lower() for keyword in ['weight', 'dimensions', 'speed', 'capacity', 'range', 'kg', 'mm', 'cm', 'rate']):
                key_changes.append(line)
    
    if key_changes:
        logger.info(f"🔑 {len(key_changes)} key specification changes detected")
        print("\nKey changes detected:")
        for line in key_changes:
            print(line)
            if line.startswith('+'):
                logger.info(f"➕ Key spec added: {line[1:]}")
            else:
                logger.info(f"➖ Key spec removed: {line[1:]}")

if __name__ == "__main__":
    # Define the document paths
    doc1_path = os.path.join(project_root, "_01_input", "raw", "fighter_jet_rocket_launcher_spec_2.pdf")
    doc2_path = os.path.join(project_root, "_01_input", "raw", "fighter_jet_rocket_launcher_spec_2_changed_values.pdf")
    
    # Compare the documents
    compare_documents(doc1_path, doc2_path) 