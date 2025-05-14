"""
test_pipeline.py

A script to run the full document processing pipeline.
"""

import os
import sys
import logging
import importlib.util
import time
import lancedb

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import the pipeline controller using importlib
controller_path = os.path.join(os.path.dirname(__file__), "_02_src", "pipeline_controller.py")
spec = importlib.util.spec_from_file_location("pipeline_controller", controller_path)
pipeline_controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline_controller)

# Get the required functions and constants
process_document = pipeline_controller.process_document
STEP_EXTRACT_REQS = pipeline_controller.STEP_EXTRACT_REQS  # Full pipeline

# Import FileHash model for database reset
admin_path = os.path.join(os.path.dirname(__file__), "_02_src", "_00_lancedb_admin", "init_lancedb.py")
spec = importlib.util.spec_from_file_location("init_lancedb", admin_path)
lancedb_admin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lancedb_admin)
FileHash = lancedb_admin.FileHash

def run_full_pipeline():
    """Run the complete pipeline for the test document."""
    test_file = "_01_input/raw/fighter_jet_rocket_launcher_spec_2.pdf"
    
    if not os.path.exists(test_file):
        logger.error(f"Test file not found: {test_file}")
        return False
    
    # Run the full pipeline
    logger.info(f"Running full pipeline for: {test_file}")
    start_time = time.time()
    
    # Run the complete pipeline
    success = process_document(test_file, max_step=STEP_EXTRACT_REQS)
    
    end_time = time.time()
    
    if success:
        logger.info(f"✅ Full pipeline completed successfully in {end_time - start_time:.2f} seconds")
    else:
        logger.error("❌ Pipeline failed - check logs for details")
    
    return success

if __name__ == "__main__":
    run_full_pipeline() 