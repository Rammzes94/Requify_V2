# Requify_V2: Document Requirements Extraction System

Requify_V2 is a powerful system for extracting structured requirements from technical documents. It processes documents individually, performs content deduplication, and generates high-quality, structured requirements.

## System Overview

Requify_V2 processes documents through a pipeline:

1. **Document Parsing**: Converts PDF documents into structured JSON with text and images
2. **Deduplication & Storage**: Checks for duplicate pages and stores unique content in LanceDB
3. **Requirements Extraction**: Extracts structured requirements from documents
4. **Requirements Deduplication**: Identifies and filters duplicate requirements

The system maintains full traceability between documents and requirements, with robust metadata tracking.

## Getting Started

### Prerequisites

- Python 3.10+
- Dependencies in `requirements.txt`
- API keys for OpenAI/Groq in `.env` file

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

## Usage

### Single Document Processing

Process a document through the entire pipeline:

```bash
python _02_src/pipeline_controller.py --input _01_input/raw/your_document.pdf
```

### Processing Documents in a Directory

Process one document from a directory (will choose the first PDF found):

```bash
python _02_src/pipeline_controller.py --input-dir _01_input/raw
```

### Individual Pipeline Steps

If needed, you can run individual steps:

1. Parse a PDF:
   ```bash
   python _02_src/_02_parsing/stable_pdf_parsing.py
   ```

2. Save parsed document to LanceDB:
   ```bash
   python _02_src/_02_parsing/stable_save_to_lancedb.py
   ```

3. Extract requirements:
   ```bash
   python _02_src/_04_extract_reqs/extract_requirements.py
   ```

## Directory Structure

- `_01_input/`: Input documents
  - `raw/`: Raw input documents
  - `processed/`: Processed documents (temporary storage)
- `_02_src/`: Source code
  - `pipeline_controller.py`: Main pipeline controller
  - `_01_ingestion/`: Document ingestion modules
  - `_02_parsing/`: Document parsing modules
  - `_03_docs_deduplication/`: Document deduplication modules
  - `_04_extract_reqs/`: Requirements extraction modules
  - `_05_reqs_deduplication/`: Requirements deduplication modules
- `_03_output/`: Output data
  - `lancedb/`: LanceDB vector database
  - `parsed_content/`: Parsed document content
  - `pdf_images/`: Images extracted from PDFs

## Configuration

Configuration is managed through `.env` file in the project root.

## License

[Your License Information] 