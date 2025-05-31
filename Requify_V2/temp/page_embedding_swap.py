"""
page_embedding_swap.py

This script creates a temporary version of stable_save_to_lancedb.py that uses 
the Qwen model for both document-level and page-level embeddings.

This allows testing how similarity scores change when the same embedding model 
is used for both document and page level comparisons.
"""

import os
import sys
import re
import shutil
from dotenv import load_dotenv

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import setup_logging, get_logger, setup_project_directory, generate_timestamp

setup_project_directory()
load_dotenv()

# Setup logging
logger = get_logger("Page_Embedding_Swap")

# Constants
ORIGINAL_FILE = os.path.join("src", "_02_parsing", "stable_save_to_lancedb.py")
BACKUP_FILE = os.path.join("temp", "stable_save_to_lancedb_backup.py")
MODIFIED_FILE = os.path.join("temp", "stable_save_to_lancedb_qwen.py")

def backup_original_file():
    """Create a backup of the original file."""
    try:
        if not os.path.exists(ORIGINAL_FILE):
            logger.error(f"Original file not found at {ORIGINAL_FILE}", extra={"icon": "❌"})
            return False
            
        shutil.copy2(ORIGINAL_FILE, BACKUP_FILE)
        logger.info(f"Created backup at {BACKUP_FILE}", extra={"icon": "💾"})
        return True
    except Exception as e:
        logger.error(f"Error creating backup: {e}", extra={"icon": "❌"})
        return False

def create_modified_version():
    """Create a modified version using Qwen for page embeddings."""
    try:
        if not os.path.exists(BACKUP_FILE):
            logger.error(f"Backup file not found at {BACKUP_FILE}", extra={"icon": "❌"})
            return False
            
        # Read the backup file
        with open(BACKUP_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Modify the embedding model settings to use Qwen for page-level embeddings
        modified_content = content
        
        # Replace the page-level embedding model with Qwen
        # First, let's add imports we need for Qwen
        import_pattern = r"from src.utils import setup_logging, get_logger.*"
        import_replacement = (
            "from src.utils import setup_logging, get_logger, update_token_counters, get_token_usage, "
            "print_token_usage, reset_token_counters, setup_project_directory, generate_timestamp\n"
            "from src.utils.doc_embedding_utils import get_document_embedder, prepare_document_text, generate_document_embedding"
        )
        modified_content = re.sub(import_pattern, import_replacement, modified_content)
        
        # Now replace the EMBEDDING_MODEL_NAME constant
        model_pattern = r"EMBEDDING_MODEL_NAME = config\.EMBEDDING_MODEL_NAME"
        model_replacement = "EMBEDDING_MODEL_NAME = config.DOC_EMBEDDING_MODEL_NAME  # Using Qwen for page embeddings"
        modified_content = re.sub(model_pattern, model_replacement, modified_content)
        
        # Replace the dimension
        dimension_pattern = r"EMBEDDING_DIMENSION = config\.EMBEDDING_DIMENSION"
        dimension_replacement = "EMBEDDING_DIMENSION = config.DOC_EMBEDDING_DIMENSION  # 1536 for Qwen"
        modified_content = re.sub(dimension_pattern, dimension_replacement, modified_content)
        
        # Update the SentenceTransformer initialization to include trust_remote_code=True
        embedder_pattern = r"text_embedder = SentenceTransformer\(EMBEDDING_MODEL_NAME\)"
        embedder_replacement = "text_embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=True)\n        text_embedder.max_seq_length = config.DOC_EMBEDDING_MAX_SEQ_LENGTH"
        modified_content = re.sub(embedder_pattern, embedder_replacement, modified_content)
        
        # Write the modified content to the new file
        with open(MODIFIED_FILE, 'w', encoding='utf-8') as f:
            f.write(modified_content)
            
        logger.info(f"Created modified version at {MODIFIED_FILE}", extra={"icon": "✅"})
        return True
    except Exception as e:
        logger.error(f"Error creating modified version: {e}", extra={"icon": "❌"})
        return False

def create_test_script():
    """Create a script to test the modified version."""
    test_script_path = os.path.join("temp", "run_test_with_qwen.py")
    script_content = """
import os
import sys
import subprocess
from dotenv import load_dotenv

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import setup_logging, get_logger, setup_project_directory

setup_project_directory()
load_dotenv()

# Setup logging
logger = get_logger("Run_Test_With_Qwen")

def main():
    # The document to process
    document_path = os.path.join("input", "raw", "fighter_jet_rocket_launcher_spec_2_changed_values.pdf")
    
    # Path to the modified script
    script_path = os.path.join("temp", "stable_save_to_lancedb_qwen.py")
    
    # Run the modified script with the document
    logger.info(f"Running modified script with Qwen for page-level embeddings", extra={"icon": "🚀"})
    cmd = [sys.executable, script_path, document_path]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        logger.info(f"Return code: {result.returncode}", extra={"icon": "🔍"})
        
        if result.stdout:
            logger.info(f"Output:\\n{result.stdout}", extra={"icon": "📝"})
            
        if result.stderr:
            logger.warning(f"Errors:\\n{result.stderr}", extra={"icon": "⚠️"})
            
        logger.info(f"Test completed", extra={"icon": "✅"})
    except Exception as e:
        logger.error(f"Error running test: {e}", extra={"icon": "❌"})

if __name__ == "__main__":
    main()
"""
    
    try:
        with open(test_script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        logger.info(f"Created test script at {test_script_path}", extra={"icon": "✅"})
        return True
    except Exception as e:
        logger.error(f"Error creating test script: {e}", extra={"icon": "❌"})
        return False

def create_runner_script():
    """Create a simple runner script to use the temporary version."""
    runner_script_path = os.path.join("temp", "process_with_qwen.py")
    script_content = """
import os
import sys
import importlib.util
from dotenv import load_dotenv

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import setup_logging, get_logger, setup_project_directory

setup_project_directory()
load_dotenv()

# Setup logging
logger = get_logger("Process_With_Qwen")

def main():
    if len(sys.argv) < 2:
        logger.error("Please provide a document path", extra={"icon": "❌"})
        print("Usage: python process_with_qwen.py <document_path>")
        return
        
    document_path = sys.argv[1]
    if not os.path.exists(document_path):
        logger.error(f"Document not found at {document_path}", extra={"icon": "❌"})
        return
        
    # Import the modified module
    spec = importlib.util.spec_from_file_location("qwen_lancedb", 
                                                 os.path.join("temp", "stable_save_to_lancedb_qwen.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Run the main function with the document path
    logger.info(f"Processing document with Qwen embeddings: {document_path}", extra={"icon": "🚀"})
    success = module.main(document_path)
    
    if success:
        logger.info(f"Processing completed successfully", extra={"icon": "✅"})
    else:
        logger.error(f"Processing failed", extra={"icon": "❌"})

if __name__ == "__main__":
    main()
"""
    
    try:
        with open(runner_script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        logger.info(f"Created runner script at {runner_script_path}", extra={"icon": "✅"})
        return True
    except Exception as e:
        logger.error(f"Error creating runner script: {e}", extra={"icon": "❌"})
        return False

def main():
    """Main execution function."""
    logger.info("Starting page embedding swap process", extra={"icon": "🚀"})
    
    # Create backup first
    if not backup_original_file():
        return
        
    # Create modified version
    if not create_modified_version():
        return
        
    # Create test script
    if not create_test_script():
        return
        
    # Create runner script
    if not create_runner_script():
        return
        
    logger.info("Setup complete. You can now run the following commands:", extra={"icon": "✅"})
    logger.info("1. To test with a specific document:", extra={"icon": "🔍"})
    logger.info("   python temp/process_with_qwen.py <document_path>", extra={"icon": "💻"})
    logger.info("2. To run an automated test with the changed values document:", extra={"icon": "🔄"})
    logger.info("   python temp/run_test_with_qwen.py", extra={"icon": "💻"})
    logger.info("Remember to restore the original if needed!", extra={"icon": "⚠️"})

if __name__ == "__main__":
    main() 