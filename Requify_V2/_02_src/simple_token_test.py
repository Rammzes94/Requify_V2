#!/usr/bin/env python3
"""
simple_token_test.py

This script provides a very simple test of the token tracking functionality.
"""

import os
import sys

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import _02_src._00_utils as utils

# Simulate a response with tokens
class MockResponse:
    def __init__(self, input_tokens, output_tokens):
        self.metrics = {
            'input_tokens': [input_tokens],
            'output_tokens': [output_tokens]
        }

# Initialize
utils.setup_project_directory()
utils.reset_token_counters()

# Simulate some token usage
response = MockResponse(1000, 200)
utils.update_token_counters(response, "gpt-4o-mini")
print(f"Added tokens: 1000 input, 200 output")

# Display usage
utils.print_token_usage("gpt-4o-mini")

# Try to display status
try:
    utils.display_token_usage_status()
except Exception as e:
    print(f"Error displaying status: {str(e)}")

print("Test completed.") 