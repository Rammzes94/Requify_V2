import React from 'react';

const PipelineVisualization = () => {
  // Styles for the pipeline components
  const styles = {
    container: {
      fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      padding: '20px',
      maxWidth: '1000px',
      margin: '0 auto',
      backgroundColor: '#f8f9fa',
      borderRadius: '8px',
      boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)'
    },
    title: {
      textAlign: 'center',
      color: '#2c3e50',
      marginBottom: '30px'
    },
    pipeline: {
      display: 'flex',
      flexDirection: 'column',
      gap: '20px'
    },
    stage: {
      backgroundColor: 'white',
      border: '1px solid #e1e4e8',
      borderRadius: '8px',
      padding: '15px',
      boxShadow: '0 2px 4px rgba(0, 0, 0, 0.05)'
    },
    stageHeader: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'flex-start',
      marginBottom: '10px',
      borderBottom: '1px solid #e1e4e8',
      paddingBottom: '10px'
    },
    stageIcon: {
      marginRight: '10px',
      fontSize: '24px',
      width: '32px',
      height: '32px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      backgroundColor: '#e3f2fd',
      borderRadius: '50%'
    },
    stageTitle: {
      fontWeight: 'bold',
      fontSize: '18px',
      color: '#2c3e50'
    },
    stageDescription: {
      fontSize: '14px',
      lineHeight: '1.5',
      color: '#4a5568'
    },
    stageDetails: {
      fontSize: '14px',
      backgroundColor: '#f5f7fa',
      padding: '10px 15px',
      borderRadius: '6px',
      marginTop: '10px'
    },
    listItem: {
      margin: '8px 0',
      display: 'flex',
      alignItems: 'flex-start'
    },
    listItemBullet: {
      color: '#4299e1',
      marginRight: '8px',
      fontWeight: 'bold'
    },
    arrow: {
      textAlign: 'center',
      fontSize: '24px',
      color: '#a0aec0',
      margin: '0'
    },
    footer: {
      textAlign: 'center',
      marginTop: '20px',
      fontSize: '12px',
      color: '#718096'
    },
    databases: {
      display: 'flex',
      justifyContent: 'space-around',
      flexWrap: 'wrap',
      gap: '10px',
      marginTop: '30px'
    },
    database: {
      backgroundColor: '#ebf8ff',
      borderRadius: '6px',
      padding: '12px',
      width: '22%',
      minWidth: '200px',
      boxSizing: 'border-box',
      border: '1px solid #bee3f8'
    },
    dbTitle: {
      fontWeight: 'bold',
      marginBottom: '8px',
      color: '#2b6cb0',
      fontSize: '16px',
      display: 'flex',
      alignItems: 'center'
    },
    dbContent: {
      fontSize: '13px',
      color: '#4a5568'
    },
    modules: {
      fontSize: '12px',
      color: '#718096',
      fontStyle: 'italic',
      marginTop: '8px'
    }
  };

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>Requify Document Processing Pipeline</h1>
      
      <div style={styles.pipeline}>
        {/* Stage 1: File Ingestion & Hash Deduplication */}
        <div style={styles.stage}>
          <div style={styles.stageHeader}>
            <div style={styles.stageIcon}>📄</div>
            <div style={styles.stageTitle}>Stage 1: File Ingestion & Hash Deduplication</div>
          </div>
          <div style={styles.stageDescription}>
            Performs content-based file deduplication to prevent redundant processing.
          </div>
          <div style={styles.stageDetails}>
            <div style={styles.listItem}>
              <span style={styles.listItemBullet}>•</span>
              <span>Calculates MD5 and SHA256 hashes for each input file</span>
            </div>
            <div style={styles.listItem}>
              <span style={styles.listItemBullet}>•</span>
              <span>Checks against existing hashes to detect exact duplicates</span>
            </div>
            <div style={styles.listItem}>
              <span style={styles.listItemBullet}>•</span>
              <span>Stores file metadata and hashes in LanceDB</span>
            </div>
            <div style={styles.modules}>Modules: file_hash_deduplication.py</div>
          </div>
        </div>
        
        <div style={styles.arrow}>↓</div>
        
        {/* Stage 2: Document Parsing */}
        <div style={styles.stage}>
          <div style={styles.stageHeader}>
            <div style={styles.stageIcon}>🔍</div>
            <div style={styles.stageTitle}>Stage 2: Document Parsing</div>
          </div>
          <div style={styles.stageDescription}>
            Converts documents to structured text and generates metadata.
          </div>
          <div style={styles.stageDetails}>
            <div style={styles.listItem}>
              <span style={styles.listItemBullet}>•</span>
              <span>Converts PDFs to high-resolution images</span>
            </div>
            <div style={styles.listItem}>
              <span style={styles.listItemBullet}>•</span>
              <span>Uses vision-enabled LLMs to extract text and tables</span>
            </div>
            <div style={styles.listItem}>
              <span style={styles.listItemBullet}>•</span>
              <span>Generates document title, page summaries, and hashtags</span>
            </div>
            <div style={styles.listItem}>
              <span style={styles.listItemBullet}>•</span>
              <span>Produces structured JSON output for further processing</span>
            </div>
            <div style={styles.modules}>Modules: stable_pdf_parsing.py, stable_excel_parsing.py</div>
          </div>
        </div>
        
        <div style={styles.arrow}>↓</div>
        
        {/* Stage 3: Document-Level Deduplication */}
        <div style={styles.stage}>
          <div style={styles.stageHeader}>
            <div style={styles.stageIcon}>🔄</div>
            <div style={styles.stageTitle}>Stage 3: Document-Level Deduplication</div>
          </div>
          <div style={styles.stageDescription}>
            Performs semantic similarity checking to identify duplicate documents.
          </div>
          <div style={styles.stageDetails}>
            <div style={styles.listItem}>
              <span style={styles.listItemBullet}>•</span>
              <span>Computes embedding vectors for document pages</span>
            </div>
            <div style={styles.listItem}>
              <span style={styles.listItemBullet}>•</span>
              <span>Checks for exact duplicates (similarity ≥ 0.98)</span>
            </div>
            <div style={styles.listItem}>
              <span style={styles.listItemBullet}>•</span>
              <span>Identifies similar documents (similarity ≥ 0.90)</span>
            </div>
            <div style={styles.listItem}>
              <span style={styles.listItemBullet}>•</span>
              <span>Interacts with user for document merge decisions</span>
            </div>
            <div style={styles.modules}>Modules: pre_save_deduplication.py, pipeline_interaction.py</div>
          </div>
        </div>
        
        <div style={styles.arrow}>↓</div>
        
        {/* Stage 4: LanceDB Storage */}
        <div style={styles.stage}>
          <div style={styles.stageHeader}>
            <div style={styles.stageIcon}>💾</div>
            <div style={styles.stageTitle}>Stage 4: Document Storage in LanceDB</div>
          </div>
          <div style={styles.stageDescription}>
            Stores parsed document data in the LanceDB vector database.
          </div>
          <div style={styles.stageDetails}>
            <div style={styles.listItem}>
              <span style={styles.listItemBullet}>•</span>
              <span>Saves page content, metadata, and embeddings</span>
            </div>
            <div style={styles.listItem}>
              <span style={styles.listItemBullet}>•</span>
              <span>Constructs vector search indices for similarity queries</span>
            </div>
            <div style={styles.listItem}>
              <span style={styles.listItemBullet}>•</span>
              <span>Handles document update cases by storing version information</span>
            </div>
            <div style={styles.modules}>Modules: stable_save_to_lancedb.py, init_lancedb.py</div>
          </div>
        </div>
        
        <div style={styles.arrow}>↓</div>
        
        {/* Stage 5: Content-Aware Chunking */}
        <div style={styles.stage}>
          <div style={styles.stageHeader}>
            <div style={styles.stageIcon}>✂️</div>
            <div style={styles.stageTitle}>Stage 5: Content-Aware Chunking</div>
          </div>
          <div style={styles.stageDescription}>
            Segments documents into semantic chunks while maintaining alignment with similar documents.
          </div>
          <div style={styles.stageDetails}>
            <div style={styles.listItem}>
              <span style={styles.listItemBullet}>•</span>
              <span>For new documents: Creates optimal chunks (100-200 tokens) preserving semantic boundaries</span>
            </div>
            <div style={styles.listItem}>
              <span style={styles.listItemBullet}>•</span>
              <span>For similar documents: Aligns chunks with existing content for better version comparison</span>
            </div>
            <div style={styles.listItem}>
              <span style={styles.listItemBullet}>•</span>
              <span>Identifies duplicate, updated, and new chunks</span>
            </div>
            <div style={styles.listItem}>
              <span style={styles.listItemBullet}>•</span>
              <span>Maintains relationships between aligned chunks</span>
            </div>
            <div style={styles.modules}>Modules: context_aware_chunking.py, integrated_chunking.py</div>
          </div>
        </div>
        
        <div style={styles.arrow}>↓</div>
        
        {/* Stage 6: Requirements Extraction */}
        <div style={styles.stage}>
          <div style={styles.stageHeader}>
            <div style={styles.stageIcon}>📋</div>
            <div style={styles.stageTitle}>Stage 6: Requirements Extraction</div>
          </div>
          <div style={styles.stageDescription}>
            Extracts atomic requirements from document chunks and analyzes them.
          </div>
          <div style={styles.stageDetails}>
            <div style={styles.listItem}>
              <span style={styles.listItemBullet}>•</span>
              <span>Extracts individual requirements with titles, descriptions, and rationale</span>
            </div>
            <div style={styles.listItem}>
              <span style={styles.listItemBullet}>•</span>
              <span>Analyzes whether each requirement is software-related</span>
            </div>
            <div style={styles.listItem}>
              <span style={styles.listItemBullet}>•</span>
              <span>Maintains traceability to source chunks and documents</span>
            </div>
            <div style={styles.listItem}>
              <span style={styles.listItemBullet}>•</span>
              <span>Generates embeddings for semantic searching</span>
            </div>
            <div style={styles.modules}>Modules: extract_requirements.py</div>
          </div>
        </div>
        
        <div style={styles.arrow}>↓</div>
        
        {/* Stage 7: Requirements Deduplication */}
        <div style={styles.stage}>
          <div style={styles.stageHeader}>
            <div style={styles.stageIcon}>🔍</div>
            <div style={styles.stageTitle}>Stage 7: Requirements Deduplication</div>
          </div>
          <div style={styles.stageDescription}>
            Identifies and manages duplicate requirements across documents.
          </div>
          <div style={styles.stageDetails}>
            <div style={styles.listItem}>
              <span style={styles.listItemBullet}>•</span>
              <span>Uses vector similarity for initial duplicate detection</span>
            </div>
            <div style={styles.listItem}>
              <span style={styles.listItemBullet}>•</span>
              <span>Employs LLM for semantic verification of potential duplicates</span>
            </div>
            <div style={styles.listItem}>
              <span style={styles.listItemBullet}>•</span>
              <span>Maintains references between duplicate and original requirements</span>
            </div>
            <div style={styles.modules}>Modules: pre_save_reqs_deduplication.py</div>
          </div>
        </div>
      </div>
      
      {/* Database Tables */}
      <h2 style={{...styles.title, marginTop: '40px'}}>LanceDB Database Tables</h2>
      <div style={styles.databases}>
        <div style={styles.database}>
          <div style={styles.dbTitle}>
            <span style={{marginRight: '8px'}}>📄</span>
            <span>documents</span>
          </div>
          <div style={styles.dbContent}>
            Document pages with content, metadata, and embeddings for page-level similarity
          </div>
        </div>
        
        <div style={styles.database}>
          <div style={styles.dbTitle}>
            <span style={{marginRight: '8px'}}>🧩</span>
            <span>document_chunks</span>
          </div>
          <div style={styles.dbContent}>
            Semantic chunks with alignment data between document versions
          </div>
        </div>
        
        <div style={styles.database}>
          <div style={styles.dbTitle}>
            <span style={{marginRight: '8px'}}>📋</span>
            <span>requirements</span>
          </div>
          <div style={styles.dbContent}>
            Extracted requirements with analysis and traceability to source chunks
          </div>
        </div>
        
        <div style={styles.database}>
          <div style={styles.dbTitle}>
            <span style={{marginRight: '8px'}}>🔐</span>
            <span>file_hashes</span>
          </div>
          <div style={styles.dbContent}>
            File metadata and cryptographic hashes for file-level deduplication
          </div>
        </div>
      </div>
      
      <div style={styles.footer}>
        Requify Pipeline Controller manages flow between stages and can stop at any point for testing
      </div>
    </div>
  );
};

export default PipelineVisualization;