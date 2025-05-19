# Requirements Extraction Agent – Functional Specification

## Product Summary
The Requirements Extraction Agent streamlines the process of converting large volumes of technical and business documents into structured, searchable requirements. It addresses the challenge of manual extraction and management of requirements from diverse file types by leveraging vision-capable LLMs and automated embedding pipelines. Users ingest documents from local directories or OneDrive, immediately remove exact duplicates via hash checks, and then deduplicate and update content via embedding similarity in LanceDB. The system maintains consistency of chunk boundaries across document versions to ensure reliable comparison and requirement extraction.

## 1. User Roles
- Usable by systems engineers, software engineers, product managers, and other stakeholders in complex systems/software engineering.

## 2. Supported Input & Ingestion
- **File Types**: PDF, DOCX, XLSX, PPTX, PPT, DOC, TXT, etc.  
- **Source Locations**: 
  - **Implemented**: Local directories
  - **Planned**: OneDrive cloud storage
- **Ingestion Modes**:  
  - **Implemented**: Command line interface with configurable pipeline steps
  - **Planned**: Web UI with "Ingest Now" button for initial load of large file sets

## 3. Preliminary Hash-based Duplicate Removal
- **Process**  
  1. Scan target directory, compute content hash (e.g. MD5/SHA) for each file.  
  2. Immediately drop any file whose hash matches an existing record—no further processing.  
  3. Log each removal event for audit.
- **Implementation**: Fully implemented with both MD5 and SHA256 hash computation for robust duplicate detection.

## 4. Embedding Similarity Scan
- **Vector Store**: LanceDB  
- **Process**  
  1. After hash-filtering, extract every page as base64-encoded PNG + page embedding. Store in a LanceDB "documents" table.  
  2. For each new page, run a cosine-similarity query (threshold ≥ 0.9) against existing page embeddings.  
     - **If duplicate**: report back "duplicate file" and abort ingestion.  
     - **If updated version**: tag as "update" and proceed with chunking, preserving knowledge of the original document's chunk boundaries.  
  3. Only unique or updated files proceed to the next stage.
- **Implementation**: Fully implemented with configurable similarity thresholds.

## 5. Parsing & Embedding Pipeline
- **Vision-LLM** parses PDF pages directly; DOC/DOCX/PPT/PPTX are converted → PDF → PNG.  
- Extract text via OCR/LLM from both raw documents and page images.  
- Store each page's base64 image and raw text in the "documents" table.
- **Implementation**: 
  - PDF parsing with image extraction is fully implemented
  - Conversion pipeline for other formats is partially implemented

## 6. Agentic Chunking & Consistent Boundaries
- **Chunking Agent**: Agno with GPT-4o-mini  
- **Process**  
  1. Feed full markdown text of the new or updated document to the chunking agent with instructions to produce ~128-token semantic chunks.  
  2. Record chunk boundary positions (character offsets or sentence indices).  
  3. Generate embeddings for each chunk and store in LanceDB "document_chunks" table, along with boundary metadata.  
  4. For each new chunk, run a cosine-similarity search against existing chunks (threshold ≥ 0.9):  
     - **If exact duplicate**: skip storing the chunk.  
     - **If near-duplicate (update)**: mark chunk as "updated", but reuse original requirement links for identical text.  
- **Implementation**: 
  - Fully implemented with memory-efficient processing for large documents
  - Boundary consistency is maintained through alignment with previous document versions

## 7. Database Relationship Visualization
- **Process**
  1. Analyze document relationships based on shared chunks
  2. Generate network graphs showing how documents are related
  3. Calculate similarity metrics between documents
  4. Provide detailed JSON data and visual representation
- **Implementation**:
  - Fully implemented with interactive database selection
  - Support for multiple database environments (main, test, validation)
  - Configurable similarity thresholds for visualization

## 8. Fixed Extraction Schema
Each final chunk processed for requirement extraction will yield fields:  
1. AI Summary  
2. Description  
3. Verification Method  
4. Test Environment  
5. Acceptance Criteria  
6. Source Text  
7. Source Chunk ID (for traceability)
- **Implementation**: Schema is implemented and enforced throughout the pipeline.

## 9. Multiple Database Environments
- **Main Database**: Production environment with real documents
- **Test Database**: Isolated environment for automated testing
- **Validation Database**: Quality assurance and benchmarking environment
- **Implementation**: Fully implemented with proper isolation between environments.

## 10. Tech Stack & Agent Tools
- **Agent Framework**: Agno (with agentic RAG patterns)  
- **Language Model**: 
  - **Implemented**: Support for OpenAI, Groq, and Mistral models
  - **Planned**: Additional model providers
- **Vector Store**: LanceDB for page & chunk embeddings  
- **Chunking Tools**: Custom Agno agent prompts for semantic splits  
- **Duplication Tools**: Cosine-similarity queries in LanceDB  
- **Visualization Tools**: NetworkX and Matplotlib for relationship graphs
- **UI**: 
  - **Implemented**: Command-line interface
  - **Planned**: Web-based dashboard (e.g. React/Next.js) to monitor ingestion, deduplication decisions, and chunk boundaries

---

**Future Enhancements**
1. Web-based user interface with dashboard
2. OneDrive integration for cloud document ingestion
3. Additional document format support
4. Enhanced relationship visualization with interactive filtering
5. Real-time monitoring of pipeline progress

---

**Questions / Clarifications**  
1. Is recording chunk boundary offsets (e.g. character indices or sentence numbers) sufficient to guarantee reproducible chunking across versions?  
2. How would you like "update" notifications surfaced in the UI when an updated document is detected?  
