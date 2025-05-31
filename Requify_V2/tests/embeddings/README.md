# Document Embedding Tests

This directory contains tests for the document-level embedding functionality in Requify V2.

## Overview

The document-level embedding feature enhances deduplication by representing entire documents as single vectors, enabling fast whole-document similarity checks.

## Test Files

- `test_doc_embedding.py` - Main test script for document embedding functionality
- `run_embedding_tests.sh` - Shell script to run common test scenarios

## Usage

Run the tests from the project root directory:

```bash
# List available documents
./tests/embeddings/run_embedding_tests.sh --list

# Process a specific document
./tests/embeddings/run_embedding_tests.sh --process fighter_jet_unique_reordered.pdf

# Compare similarity between two documents
./tests/embeddings/run_embedding_tests.sh --similarity fighter_jet_unique_reordered.pdf fighter_jet_rocket_launcher_spec_2.pdf

# Compare all document pairs
./tests/embeddings/run_embedding_tests.sh --compare-all
```

## Implementation Details

- Model: Alibaba-NLP/gte-Qwen2-1.5B-instruct (1536-dimension embeddings)
- Max sequence length: 32,768 tokens
- Core implementation in `src/utils/doc_embedding_utils.py`
- Document embedding stored in LanceDB alongside page-level embeddings 