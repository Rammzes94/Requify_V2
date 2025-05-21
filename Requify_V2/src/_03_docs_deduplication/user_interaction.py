"""
user_interaction.py

This script handles user interaction for document processing decisions.
It provides functions for:
1. Prompting users about how to handle document updates
2. Confirming actions before modifying the database
3. Handling document merge conflicts when content is removed in newer versions
4. Offering command-line interfaces for user decision making
5. Displaying summary information about document changes

These functions are used when automatic decisions cannot be made and
human intervention is required to determine the correct course of action.
"""

import os
import sys
import logging
from typing import Dict, List, Tuple, Optional, Any

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import _00_utils
_00_utils.setup_project_directory()

# Setup logging with script prefix


logger = _00_utils.get_logger("User_Interaction")

# -------------------------------------------------------------------------------------
# User Interaction Functions
# -------------------------------------------------------------------------------------

def prompt_for_update_choice(document_id: str, old_version_id: str, missing_chunks: List[Dict]) -> str:
    """
    Prompt the user for how to handle an updated document where content is missing.
    
    Args:
        document_id: The ID of the new document
        old_version_id: The ID of the old document
        missing_chunks: List of chunks that are in the old document but not the new one
        
    Returns:
        User choice: 'replace', 'merge', or 'keep'
    """
    logger.info(f"Document {document_id} appears to be an update of {old_version_id}, but some content is missing", extra={"icon": "⚠️"})
    logger.info(f"There are {len(missing_chunks)} chunks in the old document that don't have a match in the new one", extra={"icon": "ℹ️"})
    
    # Show sample of missing content
    if missing_chunks:
        print("\n=== Sample of Missing Content ===")
        for i, chunk in enumerate(missing_chunks[:3]):  # Show up to 3 samples
            print(f"\nChunk {i+1}/{min(3, len(missing_chunks))}:")
            print(chunk.get('chunk_text', '')[:200] + '...' if len(chunk.get('chunk_text', '')) > 200 else chunk.get('chunk_text', ''))
        
        if len(missing_chunks) > 3:
            print(f"\n... and {len(missing_chunks) - 3} more missing chunks")
    
    # Prompt for user choice
    print("\nHow would you like to handle this update?")
    print("1. Replace old document completely (discard missing content)")
    print("2. Merge documents (keep missing content from old version)")
    print("3. Keep both as separate documents")
    
    while True:
        choice = input("\nEnter your choice (1-3): ")
        if choice == '1':
            return 'replace'
        elif choice == '2':
            return 'merge'
        elif choice == '3':
            return 'keep'
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

def confirm_action(action_description: str) -> bool:
    """
    Ask the user to confirm an action.
    
    Args:
        action_description: Description of the action to confirm
        
    Returns:
        True if user confirms, False otherwise
    """
    confirmation = input(f"\n{action_description} (y/n): ")
    return confirmation.lower() in ['y', 'yes']

def display_document_comparison(old_doc_info: Dict, new_doc_info: Dict) -> None:
    """
    Display a comparison between old and new document versions.
    
    Args:
        old_doc_info: Dictionary with information about the old document
        new_doc_info: Dictionary with information about the new document
    """
    print("\n=== Document Comparison ===")
    print(f"Old document: {old_doc_info.get('document_id')}")
    print(f"  - Total chunks: {old_doc_info.get('total_chunks', 0)}")
    
    print(f"\nNew document: {new_doc_info.get('document_id')}")
    print(f"  - Total chunks: {new_doc_info.get('total_chunks', 0)}")
    print(f"  - New chunks: {new_doc_info.get('new_chunks', 0)}")
    print(f"  - Updated chunks: {new_doc_info.get('updated_chunks', 0)}")
    print(f"  - Duplicate chunks: {new_doc_info.get('duplicate_chunks', 0)}")
    
    # Calculate differences
    old_total = old_doc_info.get('total_chunks', 0)
    new_total = new_doc_info.get('total_chunks', 0)
    print(f"\nSize difference: {new_total - old_total} chunks ({'+' if new_total > old_total else ''}{((new_total / old_total) - 1) * 100:.1f}%)")

def handle_document_update(document_id: str, old_version_id: str, missing_chunks: List[Dict]) -> Tuple[str, List[Dict]]:
    """
    Handle a document update with potentially missing content.
    
    Args:
        document_id: The ID of the new document
        old_version_id: The ID of the old document
        missing_chunks: List of chunks that are in the old document but not the new one
        
    Returns:
        Tuple of (action, chunks_to_keep) where:
            - action is one of: 'replace', 'merge', 'keep'
            - chunks_to_keep is a list of chunks to keep from the old document (for 'merge' only)
    """
    # Display information about the update
    logger.info(f"Handling update from {old_version_id} to {document_id}", extra={"icon": "🔄"})
    
    if not missing_chunks:
        logger.info(f"No missing content - replacing old version with new version", extra={"icon": "✅"})
        return 'replace', []
    
    # Get user choice
    choice = prompt_for_update_choice(document_id, old_version_id, missing_chunks)
    
    if choice == 'replace':
        if confirm_action(f"Confirm replacing {old_version_id} with {document_id} (discarding missing content)"):
            logger.info(f"User confirmed replacing old version with new version", extra={"icon": "✅"})
            return 'replace', []
        else:
            logger.info(f"User cancelled replacement", extra={"icon": "❌"})
            return 'keep', []
    
    elif choice == 'merge':
        logger.info(f"Merging documents - keeping missing content from old version", extra={"icon": "🔄"})
        return 'merge', missing_chunks
    
    else:  # choice == 'keep'
        logger.info(f"Keeping both versions as separate documents", extra={"icon": "📋"})
        return 'keep', []

def handle_document_removal(document_id: str, removal_reason: str) -> bool:
    """
    Handle confirmation for document removal.
    
    Args:
        document_id: The ID of the document to remove
        removal_reason: Reason for removing the document
        
    Returns:
        True if removal is confirmed, False otherwise
    """
    logger.info(f"Proposed removal of document {document_id}: {removal_reason}", extra={"icon": "⚠️"})
    
    if confirm_action(f"Confirm removal of document {document_id}"):
        logger.info(f"User confirmed removal of document {document_id}", extra={"icon": "✅"})
        return True
    else:
        logger.info(f"User cancelled removal of document {document_id}", extra={"icon": "❌"})
        return False

# Main function for testing
if __name__ == "__main__":
    print("User interaction module - testing mode")
    
    # Test document update handling
    test_doc_id = "test_document.pdf"
    test_old_id = "old_document.pdf"
    test_missing = [
        {"chunk_id": "chunk1", "chunk_text": "This is some example text from the first missing chunk."},
        {"chunk_id": "chunk2", "chunk_text": "This is some example text from the second missing chunk."},
        {"chunk_id": "chunk3", "chunk_text": "This is some example text from the third missing chunk."},
        {"chunk_id": "chunk4", "chunk_text": "This is some example text from the fourth missing chunk."}
    ]
    
    action, chunks = handle_document_update(test_doc_id, test_old_id, test_missing)
    print(f"\nResult: action={action}, kept_chunks={len(chunks)}") 