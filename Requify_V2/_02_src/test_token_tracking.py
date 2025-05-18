#!/usr/bin/env python3
"""
test_token_tracking.py

This script tests the token tracking functionality in _00_utils.py.
It simulates token usage with different models and demonstrates the reporting capabilities.
"""

import os
import sys
import random
from dotenv import load_dotenv

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import _02_src._00_utils as utils
utils.setup_project_directory()

# Load environment variables
load_dotenv()

# Set up logging
logger = utils.setup_logging()

def simulate_agent_response(input_tokens, output_tokens):
    """Simulate an agent response with token metrics."""
    class MockResponse:
        def __init__(self, input_tokens, output_tokens):
            self.metrics = {
                'input_tokens': [input_tokens],
                'output_tokens': [output_tokens]
            }
    
    return MockResponse(input_tokens, output_tokens)

def test_token_tracking():
    """Test token tracking with a simple test case."""
    print("Starting token tracking test")
    
    # Reset counters for this test
    utils.reset_token_counters()
    
    # Test with gpt-4o-mini model
    model = "gpt-4o-mini"
    
    # Simulate a request
    input_tokens = 1000
    output_tokens = 200
    
    response = simulate_agent_response(input_tokens, output_tokens)
    utils.update_token_counters(response, model)
    
    print(f"Simulated {model} request: {input_tokens} input tokens, {output_tokens} output tokens")
    
    # Print summary
    utils.print_token_usage(model)
    
    # Display overall status
    utils.display_token_usage_status()
    
    print("Token tracking test completed")

if __name__ == "__main__":
    test_token_tracking() 