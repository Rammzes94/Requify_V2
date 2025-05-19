# Requify Testing Framework

This directory contains the testing framework for the Requify document processing system.

## Overview

The testing framework focuses on validating the document processing pipeline, particularly:

1. Document deduplication
2. Version handling
3. Content similarity detection
4. Chunking and requirement extraction

## Test Structure

- `chunking/` - Tests for the document chunking implementation
  - `test_consolidated_chunking.py` - Tests for the consolidated chunking module
  - `test_files/` - Sample documents for chunking tests
- `e2e/` - End-to-end tests that validate the entire pipeline
  - `test_scenarios.py` - Main script with test scenarios and execution logic
- `utils/` - Utility scripts for testing
  - `check_test_files.py` - Verifies test file availability
  - `run_test_sequence.py` - Helpers for running test sequences
  - `simple_token_test.py` - Tests for token tracking
  - `test_token_tracking.py` - Additional token tracking tests

## Main Test Categories

### Chunking Tests

The chunking tests validate the core document chunking functionality:

1. Basic chunking of small documents
2. Standard chunking of larger documents
3. Context-aware chunking for document version detection
4. Handling of duplicate and similar chunks

To run chunking tests:

```bash
python tests/chunking/test_consolidated_chunking.py
```

### End-to-End Pipeline Tests

The framework includes several predefined test scenarios:

1. **Baseline Document Ingest & Rerun** - Tests basic document ingestion and exact duplicate detection
2. **Value Change Variant** - Validates handling of documents with modified values
3. **Extra End Content** - Tests detection of documents with additional content
4. **Unique Original vs. Unique Reordered** - Tests similarity detection in reordered content
5. **Language Variant** - Tests handling of documents with similar content in different language patterns
6. **Changed vs. Reordered** - A complex test combining modified values and reordered content

## Running Tests

### Running Individual Scenarios

```bash
python tests/run_tests.py <scenario_id>
```

For example, to run scenario 1:

```bash
python tests/run_tests.py 1
```

### Running All Scenarios

```bash
python tests/run_tests.py all
```

Or use the shell script:

```bash
bash tests/run_all_tests.sh
```

### Running a Quick Subset

```bash
python tests/run_tests.py subset
```

## Test Results

Test results are stored in:

- `_03_output/test_results/` - JSON result files and summary reports
- `_03_output/test_results/reports/` - HTML reports

## Interpreting Results

Each test scenario verifies:

- Document deduplication
- Chunk detection status (new, duplicate, updated)
- Requirements relationship

## Adding New Tests

To add a new test scenario:

1. Add your test documents to `_01_input/raw/` or the appropriate test files directory
2. For pipeline tests, define a new scenario in `tests/e2e/test_scenarios.py`
3. For chunking tests, add new cases to `tests/chunking/test_consolidated_chunking.py`
4. Update the relevant test lists and configuration

## Verification Logic

The test framework verifies several aspects of the pipeline:

1. **Hash-based Duplicate Detection**: Checks if identical documents are flagged by the hash checker
2. **Chunk Deduplication**: Verifies that similar chunks are properly identified across documents
3. **Version Detection**: Tests if related document versions are correctly linked
4. **Content Change Detection**: Confirms that changes in content are properly identified and handled

Each test scenario clears the database before running to ensure a clean state.

## Troubleshooting

If tests are failing, check:

1. The log output in the console for specific errors
2. The detailed JSON results for exactly which expectations weren't met
3. That the test document files exist in the expected location
4. The database connection and state by running the database viewer utility 