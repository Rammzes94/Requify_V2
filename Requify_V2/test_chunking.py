#!/usr/bin/env python3
"""
test_chunking.py

A simple test script to verify that our chunking functionality works properly
by generating chunks from a test string and analyzing the results.
"""

import sys
import os
sys.path.append(os.path.abspath('_02_src'))
from _02_src._02_parsing import integrated_chunking
import logging

logging.basicConfig(level=logging.INFO)

# Test with a medium-sized text
test_text = 'This is a test paragraph with some content. ' * 100  # ~3400 chars
print(f'Testing with text length: {len(test_text)} chars')

# Run the chunking
chunks = integrated_chunking.chunk_markdown(test_text)

# Display results
print(f'\nResults: Generated {len(chunks)} chunks')
for i, chunk in enumerate(chunks):
    print(f'Chunk {i+1} length: {len(chunk)} chars')

if len(chunks) <= 1 and len(test_text) > integrated_chunking.TARGET_CHAR_SIZE:
    print("\nWarning: Text exceeded target size but only generated one chunk!")
    print(f"Target size: {integrated_chunking.TARGET_CHAR_SIZE}, Actual text: {len(test_text)}")
else:
    print("\nChunking successful - multiple chunks created as expected.") 