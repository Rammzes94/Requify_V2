# Requirements Extraction Agent – Functional Specification

## Product Summary
The Requirements Extraction Agent streamlines the process of converting large volumes of technical and business documents into structured, searchable requirements. It addresses the challenge of manual extraction and management of requirements from diverse file types by leveraging vision-capable LLMs and automated embedding pipelines. Users ingest documents from local directories or OneDrive, immediately remove exact duplicates via hash checks, and then deduplicate and update content via embedding similarity in LanceDB. The system maintains consistency of chunk boundaries across document versions to ensure reliable comparison and requirement extraction.

## 1. User Roles
- Usable by systems engineers, software engineers, product managers, and other stakeholders in complex systems/software engineering.

## 2. Supported Input & Ingestion
- **File Types**: PDF, DOCX, XLSX, PPTX, PPT, DOC, TXT, etc.  
- **Source Locations**: Local directories and OneDrive cloud storage.  
- **Ingestion Modes**:  
  - **Bulk Import**: “Ingest Now” button for initial load of large file sets.  
  - **Manual Add**: On-demand ingestion of individual files.

## 3. Preliminary Hash-based Duplicate Removal
- **Process**  
  1. Scan target directory, compute content hash (e.g. MD5/SHA) for each file.  
  2. Immediately drop any file whose hash matches an existing record—no further processing.  
  3. Log each removal event for audit.

## 4. Embedding Similarity Scan
- **Vector Store**: LanceDB  
- **Process**  
  1. After hash-filtering, extract every page as base64-encoded PNG + page embedding. Store in a LanceDB “documents” table.  
  2. For each new page, run a cosine-similarity query (threshold ≥ 0.9) against existing page embeddings.  
     - **If duplicate**: report back “duplicate file” and abort ingestion.  
     - **If updated version**: tag as “update” and proceed with chunking, preserving knowledge of the original document’s chunk boundaries.  
  3. Only unique or updated files proceed to the next stage.

## 5. Parsing & Embedding Pipeline
- **Vision-LLM** parses PDF pages directly; DOC/DOCX/PPT/PPTX are converted → PDF → PNG.  
- Extract text via OCR/LLM from both raw documents and page images.  
- Store each page’s base64 image and raw text in the “documents” table.

## 6. Agentic Chunking & Consistent Boundaries
- **Chunking Agent**: Agno with GPT-4o-mini  
- **Process**  
  1. Feed full markdown text of the new or updated document to the chunking agent with instructions to produce ~128-token semantic chunks.  
  2. Record chunk boundary positions (character offsets or sentence indices).  
  3. Generate embeddings for each chunk and store in LanceDB “document_chunks” table, along with boundary metadata.  
  4. For each new chunk, run a cosine-similarity search against existing chunks (threshold ≥ 0.9):  
     - **If exact duplicate**: skip storing the chunk.  
     - **If near-duplicate (update)**: mark chunk as “updated”, but reuse original requirement links for identical text.  
- **Consistency Concern**: to ensure similar docs chunk similarly, we record and reuse the chunking prompt and boundary markers from the original ingestion—please confirm if capturing offsets is sufficient, or if you’d prefer a more deterministic algorithm (e.g. sliding window with overlap).

## 7. Fixed Extraction Schema
Each final chunk processed for requirement extraction will yield fields:  
1. AI Summary  
2. Description  
3. Verification Method  
4. Test Environment  
5. Acceptance Criteria  
6. Source Text  
7. Source Chunk ID (for traceability)

## 8. Tech Stack & Agent Tools
- **Agent Framework**: Agno (with agentic RAG patterns)  
- **Language Model**: GPT-4o-mini for vision and text tasks  
- **Vector Store**: LanceDB for page & chunk embeddings  
- **Chunking Tools**: custom Agno agent prompts for semantic splits  
- **Duplication Tools**: cosine-similarity queries in LanceDB  
- **UI**: Web-based dashboard (e.g. React/Next.js) to monitor ingestion, deduplication decisions, and chunk boundaries  

---

**Questions / Clarifications**  
1. Is recording chunk boundary offsets (e.g. character indices or sentence numbers) sufficient to guarantee reproducible chunking across versions?  
2. How would you like “update” notifications surfaced in the UI when an updated document is detected?  
