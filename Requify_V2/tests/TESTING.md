# Requify Testing Framework Documentation

## Overview

The Requify testing framework is designed to verify the document processing pipeline's ability to handle various document scenarios. The tests focus on validating:

1. Document deduplication detection
2. Version tracking and handling
3. Content similarity detection across documents
4. Chunking and requirement extraction processes
5. Database relationship visualization

## Test Framework Organization

The testing framework has been organized into a clean, maintainable structure:

```
Requify_V2/
├── input/raw/               # Test input documents
├── src/                     # Main product code (to be shipped)
├── output/                  # Output storage
│   ├── test_results/            # Test results storage
│   └── visualizations/          # Visualization outputs
├── tests/                       # All test-related code
│   ├── chunking/                # Tests for chunking functionality
│   │   └── test_files/          # Test files for chunking tests
│   ├── e2e/                     # End-to-end tests
│   │   └── test_scenarios.py    # Test scenario definitions and runner
│   ├── utils/                   # Test utilities
│   ├── README.md                # Testing documentation
│   ├── run_all_tests.sh         # Script to run all tests
│   └── run_tests.py             # Python test runner with CLI
└── tools/                       # Utility tools
    ├── test_utils/              # Test utilities
    │   └── results_reporter.py  # HTML report generator
    └── visualization/           # Visualization tools
        └── visualize_db_relationships.py  # Database relationship visualization tool
```

## Test Scenarios

The framework includes several predefined scenarios to test various aspects of the system:

1. **Baseline Document Ingest & Rerun** - Tests hash-based exact duplicate detection
2. **Value Change Variant** - Tests handling of documents with modified values
3. **Extra End Content** - Tests handling of documents with additional content
4. **Unique Original vs. Unique Reordered** - Tests similarity detection with reordered content
5. **Language Variant** - Tests handling of documents with similar content but different phrasing
6. **Changed vs. Reordered (Combined)** - Complex test combining multiple types of changes

## Running Tests

### Individual Scenarios

```bash
python tests/run_tests.py <scenario_id>
```

### All Scenarios

```bash
python tests/run_tests.py all
# Or
bash tests/run_all_tests.sh
```

### Quick Subset

```bash
python tests/run_tests.py subset
```

## Database Visualization

To visualize document and chunk relationships in the test database:

```bash
python tools/visualization/visualize_db_relationships.py --db-type test
```

Or to interactively select a database:

```bash
python tools/visualization/visualize_db_relationships.py --interactive
```

## Test Results and Reporting

Results are saved in:
- JSON files in `output/test_results/`
- Human-readable summary files
- HTML reports in `output/test_results/reports/`
- Relationship visualizations in `output/visualizations/`

To generate a report from the latest test run:

```bash
python tests/run_tests.py report
```

## Verification Logic

The test framework verifies each scenario by:

1. Clearing the database before each scenario
2. Running the pipeline for each document in the scenario
3. Analyzing logs to extract chunk statistics (new, duplicate, updated)
4. Comparing actual results against expected outcomes
5. Generating detailed reports of failures and successes

## Adding New Tests

To add a new test scenario:

1. Add appropriate test files to `input/raw/`
2. Define a new scenario in `tests/e2e/test_scenarios.py`
3. Specify the expected behavior for each step
4. Update the total scenario count in `run_all_tests.sh`

## Multiple Database Environments

The testing framework supports multiple database environments:

1. **Main Database** - Production database with real documents
2. **Test Database** - Used for automated test scenarios
3. **Validation Database** - Used for quality assurance and benchmark testing

Each environment maintains isolation to prevent test data from contaminating production data.

## Technology Used

- Python for test scripting
- Bash for automation
- LanceDB for database operations
- NetworkX and Matplotlib for relationship visualization
- HTML/CSS for reporting 