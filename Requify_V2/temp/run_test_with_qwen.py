
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
            logger.info(f"Output:\n{result.stdout}", extra={"icon": "📝"})
            
        if result.stderr:
            logger.warning(f"Errors:\n{result.stderr}", extra={"icon": "⚠️"})
            
        logger.info(f"Test completed", extra={"icon": "✅"})
    except Exception as e:
        logger.error(f"Error running test: {e}", extra={"icon": "❌"})

if __name__ == "__main__":
    main()
