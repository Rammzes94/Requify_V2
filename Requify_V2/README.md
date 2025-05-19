# Requify_V2: Document Requirements Extraction System

Requify_V2 is a powerful system for extracting structured requirements from technical documents. It processes documents individually, performs content deduplication, and generates high-quality, structured requirements.

## System Overview

Requify_V2 processes documents through a pipeline:

1. **Document Parsing**: Converts PDF documents into structured JSON with text and images
2. **Deduplication & Storage**: Checks for duplicate pages and stores unique content in LanceDB
3. **Content-Aware Chunking**: Performs intelligent document chunking with context awareness
4. **Requirements Extraction**: Extracts structured requirements from documents
5. **Requirements Deduplication**: Identifies and filters duplicate requirements

The system maintains full traceability between documents and requirements, with robust metadata tracking.

## Getting Started

### Prerequisites

- Python 3.10+
- Dependencies in `requirements.txt`
- API keys for OpenAI/Groq/Mistral in `.env` file

### Installation

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Setup the `.env` file with required API keys:
   ```
   OPENAI_API_KEY=your_api_key
   # Or alternative model provider
   GROQ_API_KEY=your_api_key
   MISTRAL_API_KEY=your_api_key
   ```

## Usage

### Single Document Processing

Process a document through the entire pipeline:

```bash
python pipeline_runner.py --doc_path _01_input/raw/your_document.pdf
```

### Controlling Pipeline Steps

You can control how far the pipeline runs by specifying the maximum step:

```bash
python pipeline_runner.py --doc_path _01_input/raw/your_document.pdf --max_step 4
```

Available steps:
1. Hash-based duplicate check only
2. Parse document only
3. Parse + Deduplication check only (no DB save)
4. Parse + Save to LanceDB (no chunking)
5. Parse + Save to LanceDB + Chunk document
6. Complete pipeline with requirements extraction (default)

### Dry Run Mode

To check a document without modifying the database:

```bash
python pipeline_runner.py --doc_path _01_input/raw/your_document.pdf --dry_run
```

### Visualizing Document Relationships

To visualize document and chunk relationships in the database:

```bash
python tools/visualization/visualize_db_relationships.py
```

For interactive database selection:

```bash
python tools/visualization/visualize_db_relationships.py --interactive
```

For a specific database type:

```bash
python tools/visualization/visualize_db_relationships.py --db-type test
```

Output visualizations are saved in `_03_output/visualizations/`.

## Directory Structure

- `_01_input/`: Input documents
  - `raw/`: Raw input documents
- `_02_src/`: Source code
  - `_00_lancedb_admin/`: LanceDB administration utilities
  - `_00_utils/`: Utility functions and configuration
  - `_01_ingestion/`: Document ingestion modules
  - `_02_parsing/`: Document parsing and chunking modules
  - `_03_docs_deduplication/`: Document deduplication modules
  - `_04_extract_reqs/`: Requirements extraction modules
  - `_05_reqs_deduplication/`: Requirements deduplication modules
- `_03_output/`: Output data
  - `lancedb/`: LanceDB vector database
  - `parsed_content/`: Parsed document content
  - `pdf_images/`: Images extracted from PDFs
  - `test_results/`: Test execution results
  - `visualizations/`: Database relationship visualizations
- `tests/`: Testing framework
  - `chunking/`: Tests for document chunking
  - `e2e/`: End-to-end pipeline tests
  - `utils/`: Test utilities
  - `TESTING.md`: Testing documentation
- `tools/`: Utility tools
  - `test_utils/`: Test utilities
  - `validation/`: Validation utilities
  - `visualization/`: Visualization tools

## Multiple Database Environments

Requify_V2 supports multiple database environments:

1. **Main Database** (`_03_output/lancedb/`)
   - Production database with real documents
   
2. **Test Database** (`tests/e2e/_03_output/lancedb/`)
   - Used for automated test scenarios
   - Isolated from production data
   
3. **Validation Database** (`tools/validation/_03_output/lancedb/`)
   - Used for quality assurance and benchmark testing

## Key Components

### Consolidated Chunking

The system uses a consolidated chunking implementation that combines:

1. Efficient processing of large documents by splitting into sections
2. Context-aware chunking for maintaining alignment between document versions
3. Intelligent duplicate/similar chunk detection
4. Memory-efficient processing with proper cleanup

### Database Relationship Visualization

The system includes visualization tools for exploring document relationships:

1. Network graphs showing how documents are related through similar chunks
2. Document similarity metrics and statistics
3. Interactive selection of different database environments

### Testing

Comprehensive testing is available in the `tests/` directory:

```bash
# Run chunking tests
python tests/chunking/test_consolidated_chunking.py

# Run end-to-end pipeline tests
python tests/run_tests.py all
```

See `tests/TESTING.md` for more details on testing.

## Configuration

Configuration is managed through `.env` file in the project root.

## License

[Your License Information] 