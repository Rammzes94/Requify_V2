# Requirements Extraction Agent – Functional Specification

## Product Summary
The Requirements Extraction Agent streamlines the process of converting large volumes of technical and business documents into structured, searchable requirements. It addresses the challenge of manual extraction and management of requirements from diverse file types by leveraging vision-capable LLMs and automated embedding pipelines. Users ingest documents from local directories or OneDrive, automatically detect and resolve duplicates via both metadata-based and embedding-based scans, and maintain full traceability through audit logs and metadata. The system empowers engineering teams to review, edit, and export requirements seamlessly into their downstream tools, accelerating development workflows and ensuring high-quality deliverables.

## 1. User Roles
- Usable by systems engineers, software engineers, product managers, and other stakeholders in complex systems/software engineering.

## 2. Supported Input & Ingestion
- **File Types**: PDF, DOCX, XLSX, PPTX, PPT, DOC, TXT, etc.  
- **Source Locations**: Local directories and OneDrive cloud storage.  
- **Ingestion Modes**:  
  - **Incremental Document Processing**: Process one document at a time to ensure consistency and proper duplicate detection.
  - **Manual Add**: On-demand addition of individual files with immediate deduplication.

## 3. Preliminary Duplicate Scan
- **Tools**: `scan_directory`, `find_similar_files`, `determine_newer_file`, `compare_file_contents`.  
- **Process**:  
  1. Scan a target directory to gather file metadata (path, size, hash, timestamps).  
  2. Perform metadata-based filtering: identical hashes or exact sizes are auto-removed.  
  3. Group files by extension and name similarity to identify obvious duplicates.  
  4. Automatically mark and unselect duplicates, keeping the most up-to-date version by timestamp or size.  
- **Output**: JSON of filtered file list for downstream parsing.

## 4. Incremental Embedding Similarity Scan
- **Vector Store**: LanceDB for efficient vector-based similarity search.
- **Process**:  
  1. Pre-check each new document against existing content before saving.
  2. Embed each page and compare with existing vectors to detect duplicates.
  3. Use cosine similarity with threshold ≥0.99 to identify duplicate pages.
  4. Skip duplicate pages, add only unique content to the database.
  5. For updated versions of existing documents, overwrite the old versions.
- **Integration**: This check ensures only unique content is processed for requirement extraction.

## 5. Parsing & Embedding Pipeline
- **Vision-LLM** for direct PDF parsing.  
- **Conversion**: DOC/DOCX/PPT/PPTX → PDF → PNG.  
- **Extraction**: Text from raw documents and generated images.  
- **Embeddings**: Store both text embeddings and vision embeddings for each chunk in LanceDB.
- **Process Flow**: Parse one document at a time → Check for duplicates → Save to database → Extract requirements.

## 6. Fixed Extraction Schema
Every extracted requirement shall include:  
1. AI Summary  
2. Description  
3. Verification Method  
4. Test Environment  
5. Acceptance Criteria  
6. Source Text

## 7. Unified Database (LanceDB)
Each requirement record stores:  
- **Requirement ID** (unique)  
- **Project/Folder ID**  
- **File Identifier** (filename or internal file ID)  
- **Page Number** (for PDF/image sources)  
- **Original Context** (exact text or image segment)  
- **Embedding Vector** (stored in LanceDB)  
- **Similarity Score** (for deduplication)  
- **Ingestion Batch ID**  
- **Creation Timestamp**  
- **Last Updated Timestamp**  
- **Status** (`active` / `deleted` / `review_needed`)  
- **Duplicate Status** (`is_duplicate` flag, `canonical_id` reference)
- **User Action Log** (who edited/approved/deleted, when)  
- **User-Defined Tags** (optional)

## 8. Duplicate Detection & Resolution
- **Page-Level Duplicates**: cosine similarity ≥ 0.99 threshold.  
- **Requirements-Level Duplicates**: vector similarity ≥ 0.97 + LLM verification.
- **Behavior**:  
  - **Pre-Save Document Check**: Detect duplicate pages before adding to database.
  - **Pre-Save Requirements Check**: Detect duplicate requirements before adding to database.
  - Automatically keep the newest version; mark older as duplicates and reference the canonical version.
- **Edge Cases**: LLM flags uncertain duplicates as needing review.

## 9. Manual Curation & Batch Operations
- Inline edit and delete of individual requirements.  
- Bulk-select actions (delete, tag, mark reviewed).  
- Deleted items remain marked so they won't reappear on re-ingestion.

## 10. Traceability & Audit
- Maintain full history of edits/deletions so that once removed or altered, the requirement won't be re-suggested.  
- Audit log for all user actions and system events.

## 11. Exports & Integration
- **Format**: JSON export (selectable fields and filters).  
- **Configuration**: Export settings defined via environment variables (dotenv).

## 12. Logs & Notifications
- **Logged Events**: ingestion successes/failures, parsing errors, deduplication actions, LLM "review_needed" flags.  
- **UI**: Log viewer with filters (timestamp, event type).  
- **Email**: Optional notifications on ingestion completion and review-needed events.

## 13. Configuration Management
- All thresholds, file-type mappings, export settings, and notification rules managed via a `.env` file.  
- Config reloaded on service restart.

## 14. Tech Stack & Agent Tools
- **Agent Framework**: pydantic_ai (with built-in support for agentic RAG patterns).  
- **Extraction Engine**: Vision-capable LLM via pydantic_ai, embedding storage and retrieval on LanceDB.  
- **LightRAG** layered for lightweight retrieval tasks.  
- **RAG Tools Provided**: keyword search, hashtag search, basic dense retrieval (text + vision embeddings), LightRAG index queries.  
- **Directory Scan Tools**: `scan_directory`, `find_similar_files`, `determine_newer_file`, `compare_file_contents`.  
- **Database**: LanceDB (primary vector store).  
- **UI**: Web-based dashboard (e.g., React/Next.js)

## 15. Quality & Testing
- The project shall achieve **100% code coverage** via automated test suites.  
- Tests shall include unit tests, integration tests, and end-to-end scenarios for all major workflows.  
