
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
