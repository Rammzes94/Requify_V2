import os
import sys
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.models.groq import Groq
import lancedb
import sentence_transformers
from lancedb.pydantic import LanceModel, Vector
from dotenv import load_dotenv
import traceback

"""
extract_requirements.py

This script extracts atomic requirements from document chunks stored in LanceDB.
It performs the following operations:
1. Connects to LanceDB and retrieves document chunks
2. Uses LLMs to extract atomic requirements from each chunk's content
3. Generates embeddings for each requirement
4. Checks for duplicate requirements using the pre_save_reqs_deduplication module
5. Saves unique requirements to the requirements table in LanceDB
6. Maintains traceability between requirements and their source chunks
7. Reuses existing requirements for duplicate chunks

The script supports both OpenAI and Groq models for text processing and can be 
configured via environment variables. For document updates, it preserves 
requirements from identical chunks while processing only new or modified content.
"""

# -------------------------------------------------------------------------------------
# Setup and Configuration
# -------------------------------------------------------------------------------------

# Setup project directory and load environment variables
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import _00_utils
from _00_utils import update_token_counters, print_token_usage
import config
_00_utils.setup_project_directory()
load_dotenv()

# Configure logging with script prefix first
logger = _00_utils.setup_logging()

logger = _00_utils.get_logger("Extract_Requirements")

# Import the deduplication module for requirements
try:
    from _05_reqs_deduplication.pre_save_reqs_deduplication import check_requirements_duplicates
    DEDUPLICATION_AVAILABLE = True
except ImportError as e:
    logger.warning("Could not import pre_save_reqs_deduplication module. Duplicate detection will be disabled.", extra={"icon": "⚠️"})
    DEDUPLICATION_AVAILABLE = False

# Suppress all HTTP request logs and third-party library logs
logging.getLogger('httpx').setLevel(logging.ERROR)
logging.getLogger('httpcore').setLevel(logging.ERROR)
logging.getLogger('agno').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('requests').setLevel(logging.ERROR)

# -------------------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------------------

# Get API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Get the model for requirement extraction from the config
req_extraction_model_name = config.get_model_for_task("requirement_extraction")

# Model Configuration
# Initialize models for requirements extraction
openai_model = OpenAIChat(id=req_extraction_model_name, api_key=api_key, temperature=0)

# Uncomment to use Groq models instead
groq_model = Groq(id=req_extraction_model_name, api_key=groq_api_key, temperature=0)

# Select which model to use based on configuration
MODEL_PROVIDER = config.MODEL_PROVIDER.lower()

if MODEL_PROVIDER == "openai":
    active_model = openai_model
    logger.info(f"Using OpenAI model for requirements extraction: {req_extraction_model_name}", extra={"icon": "🧠"})
else:  # Default to Groq
    active_model = groq_model
    logger.info(f"Using Groq model for requirements extraction: {req_extraction_model_name}", extra={"icon": "🧠"})

# LanceDB settings
OUTPUT_DIR_BASE = config.OUTPUT_DIR_BASE  # Use from config
LANCEDB_SUBDIR_NAME = "lancedb"  # Subdirectory for LanceDB within _03_output
LANCEDB_DIR_PATH = os.path.join(OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME) # Construct path relative to project root
LANCEDB_URI = LANCEDB_DIR_PATH  # Alias for consistency
DOCUMENT_CHUNKS_TABLE = "document_chunks"
DOCUMENTS_TABLE = "documents"
REQUIREMENTS_TABLE = "requirements"
EMBEDDING_DIMENSION = config.EMBEDDING_DIMENSION  # Use from config
EMBEDDING_MODEL = config.EMBEDDING_MODEL_NAME  # Use from config

# -------------------------------------------------------------------------------------
# Prompt Templates
# -------------------------------------------------------------------------------------

EXTRACT_REQUIREMENTS_PROMPT = """Extract all atomic requirements from the following text. 
Focus on sections that define technical specifications, technical requirements, and mandatory elements - things focused on the product.

For the following input, identify ALL individual requirements and break them down into atomic units.
Each atomic requirement should be a single, testable statement. You have to give me all requirements!
Extract all technical prescriptions, specifications, and mandatory elements.

For each requirement, provide:
1. The requirement ID, section, title, and description.
2. A rationale explaining why it is considered a technical product requirement. This should focus on the technical aspects that affect the physical vehicle or its systems.
3. The 'source_text', which is the exact text segment from the input TEXT that directly led to this requirement's extraction.

The input may not have a single technical product requirement in it. In that case, just give back nothing. 
We only care about requirements for the product!

TEXT:
{text}"""

ANALYZE_REQUIREMENT_PROMPT = """
Analyze this requirement and determine if it is software-related (will affect software systems):

"{requirement_description}"
    
Be concise in your rationale.
"""

BATCH_ANALYZE_REQUIREMENTS_PROMPT = """
Analyze each of the following requirements and determine if it is software-related (will affect software systems).

{requirements_text}

For each requirement, provide:
1. Whether it is software-related (true/false)
2. Your confidence level (0.0 to 1.0) 
3. A concise rationale for your decision

Keep each analysis brief and focused.
"""

# -------------------------------------------------------------------------------------
# Agent Descriptions
# -------------------------------------------------------------------------------------

REQUIREMENTS_AGENT_DESCRIPTION = """You are a specialized agent for extracting atomic requirements from technical documents.
You are good at making sure to take the entire input into account and do not stop after a few examples.
You focus only on technical product requirements!"""

SOFTWARE_AGENT_DESCRIPTION = """You are a specialized agent for determining if requirements are related to software.
You have expertise in software systems and understanding how requirements translate to software implementations."""

# -------------------------------------------------------------------------------------
# Data Models
# -------------------------------------------------------------------------------------

# Pydantic models for structured outputs
class AtomicRequirement(BaseModel):
    """An individual requirement extracted from the document."""
    id: str
    title: str
    description: str
    source_text: str
    requirement_rationale: str
    section: str
    page_number: Optional[int]
    source_chunk_id: Optional[str] = None

class RequirementsExtractor(BaseModel):
    """Collection of extracted requirements from a document."""
    document_name: str = Field(..., description="Name of the document")
    document_id: str = Field(..., description="ID of the document")
    requirements: List[AtomicRequirement] = Field(..., description="List of atomic requirements extracted from the document")
    total_count: int = Field(..., description="Total number of requirements extracted")
    rationale: str = Field(..., description="A brief rationale why no requirements were selected, if applicable")

class SoftwareRequirementAnalysis(BaseModel):
    """Analysis of whether a requirement is software-related or not."""
    is_software_related: bool = Field(..., description="Boolean indicating if the requirement is related to software")
    confidence: float = Field(..., description="Confidence level (0.0 to 1.0) of the analysis")
    rationale: str = Field(..., description="Explanation of why this requirement is or is not software-related")

class BatchSoftwareAnalysis(BaseModel):
    """Batch analysis of multiple requirements."""
    analyses: List[SoftwareRequirementAnalysis] = Field(..., description="List of software requirement analyses")

class Requirement(LanceModel):
    """LanceDB model for storing requirements."""
    requirement_id: str
    document_id: str
    page_number: Optional[int]
    section: str
    title: str
    description: str
    requirement_rationale: str
    is_software_related: bool
    software_confidence: float
    software_rationale: str
    source_text: str  # This is the only field we'll use for source text
    source_chunk_id: str # ID of the chunk this requirement came from
    created_timestamp: str
    embedding: Vector(EMBEDDING_DIMENSION)
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None

# -------------------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------------------

def create_agent(response_model, description):
    """Create an agent with the specified model and description."""
    return Agent(
        model=active_model,
        description=description,
        response_model=response_model,
    )

def extract_requirements_from_text(text: str, document_id: str, document_name: str, chunk_id: str, requirement_count_offset: int = 0) -> RequirementsExtractor:
    """Extract requirements from text."""
    logger.info(f"🔄 Extracting requirements from chunk {chunk_id}")
    
    # Initialize the agent with structured outputs
    requirements_agent = create_agent(
        response_model=RequirementsExtractor,
        description=REQUIREMENTS_AGENT_DESCRIPTION
    )
    
    try:
        # Format the prompt without printing it
        prompt = EXTRACT_REQUIREMENTS_PROMPT.format(text=text)
        
        # Extract requirements using the agent
        response = requirements_agent.run(prompt)
        update_token_counters(response, req_extraction_model_name)
        
        # Ensure the response is a RequirementsExtractor object
        if isinstance(response, RunResponse):
            # Access the content field, which contains the structured data
            requirements_data = response.content
        else:
            requirements_data = response
        
        # Log the results
        if hasattr(requirements_data, 'requirements') and len(requirements_data.requirements) > 0:
            logger.info(f"✅ Successfully extracted {len(requirements_data.requirements)} requirements from chunk {chunk_id}")
        else:
            logger.info(f"ℹ️ No requirements found in chunk {chunk_id}")
            
        # Add offset to requirement IDs if provided
        if requirement_count_offset > 0 and hasattr(requirements_data, 'requirements'):
            for req in requirements_data.requirements:
                # Assuming ID is numeric and can be padded
                try:
                    num_id = int(req.id)
                    req.id = f"{num_id + requirement_count_offset:04d}"
                except ValueError:
                    # If ID is not numeric, append the offset
                    req.id = f"{req.id}-{requirement_count_offset}"
        
        # Add source chunk ID to each requirement
        if hasattr(requirements_data, 'requirements'):
            for req in requirements_data.requirements:
                req.source_chunk_id = chunk_id
        
        return requirements_data
    
    except Exception as e:
        logger.error(f"❌ Error extracting requirements from chunk {chunk_id}: {e}")
        # Return empty result in case of error
        return RequirementsExtractor(
            document_name=document_name,
            document_id=document_id,
            requirements=[],
            total_count=0,
            rationale=f"Error during extraction: {str(e)}"
        )

def analyze_requirement(requirement: AtomicRequirement, agent: Agent) -> SoftwareRequirementAnalysis:
    """Analyze a single requirement to determine if it's software-related."""
    prompt = ANALYZE_REQUIREMENT_PROMPT.format(requirement_description=requirement.description)
    
    try:
        response = agent.run(prompt)
        update_token_counters(response, req_extraction_model_name)
        
        if isinstance(response, RunResponse):
            return response.content
        return response
    except Exception as e:
        logger.error(f"❌ Error analyzing requirement {requirement.id}: {e}")
        # Return conservative default in case of error
        return SoftwareRequirementAnalysis(
            is_software_related=False,
            confidence=0.5,
            rationale=f"Error during analysis: {str(e)}"
        )

def connect_to_lancedb():
    """Connect to LanceDB and return the connection."""
    # Construct path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    lancedb_path = os.path.join(project_root, LANCEDB_DIR_PATH)
    
    try:
        logger.info(f"Connecting to LanceDB database at {lancedb_path}", extra={"icon": "🔄"})
        db = lancedb.connect(lancedb_path)
        logger.info(f"Connected to LanceDB at {lancedb_path}", extra={"icon": "✅"})
        return db
    except Exception as e:
        logger.error(f"Failed to connect to LanceDB: {e}", extra={"icon": "❌"})
        return None

def get_or_create_requirements_table(db):
    """Get the requirements table from LanceDB or create it if it doesn't exist."""
    if not db:
        logger.error("❌ [LanceDB] No database connection provided for requirements table access")
        return None
        
    try:
        logger.info(f"🔍 [LanceDB] Checking for existing table '{REQUIREMENTS_TABLE}' in database")
        table_names = db.table_names()
        logger.info(f"📋 [LanceDB] Available tables in database: {', '.join(table_names) if table_names else 'None'}")
        
        if REQUIREMENTS_TABLE in table_names:
            logger.info(f"✅ [LanceDB] Found existing requirements table: '{REQUIREMENTS_TABLE}'")
            try:
                logger.info(f"🔄 [LanceDB] Attempting to open table '{REQUIREMENTS_TABLE}'...")
                table = db.open_table(REQUIREMENTS_TABLE)
                
                if table is None:
                    logger.error(f"❌ [LanceDB] Failed to open table '{REQUIREMENTS_TABLE}' - table object is None")
                    return None
                
                logger.info(f"✅ [LanceDB] Successfully opened table '{REQUIREMENTS_TABLE}' for writing")
                return table
            except Exception as e:
                logger.error(f"❌ [LanceDB] Error accessing existing table '{REQUIREMENTS_TABLE}': {str(e)}")
                logger.error(f"❌ [LanceDB] Traceback: {traceback.format_exc()}")
                return None
        else:
            # Create the table if it doesn't exist
            logger.info(f"🆕 [LanceDB] Creating new requirements table: '{REQUIREMENTS_TABLE}'")
            try:
                table = db.create_table(REQUIREMENTS_TABLE, schema=Requirement)
                logger.info(f"✅ [LanceDB] Successfully created table '{REQUIREMENTS_TABLE}'")
                return table
            except Exception as e:
                logger.error(f"❌ [LanceDB] Error creating table '{REQUIREMENTS_TABLE}': {str(e)}")
                logger.error(f"❌ [LanceDB] Traceback: {traceback.format_exc()}")
                return None
            
    except Exception as e:
        logger.error(f"❌ [LanceDB] Error setting up requirements table: {str(e)}")
        logger.error(f"❌ [LanceDB] Traceback: {traceback.format_exc()}")
        return None

def analyze_requirements_batch(requirements: List[AtomicRequirement], agent: Agent) -> Dict[str, SoftwareRequirementAnalysis]:
    """Analyze requirements in batch to determine if they're software-related."""
    if not requirements:
        return {}
        
    logger.info(f"🔄 Analyzing {len(requirements)} requirements in batch for software relation...")
    
    # Prepare the requirements text for batch analysis
    requirements_text = ""
    for i, req in enumerate(requirements):
        requirements_text += f"Requirement {i+1} - ID: {req.id}, Title: {req.title}\n"
        requirements_text += f"Description: {req.description}\n\n"
    
    prompt = BATCH_ANALYZE_REQUIREMENTS_PROMPT.format(requirements_text=requirements_text)
    
    try:
        # Run batch analysis
        response = agent.run(prompt)
        update_token_counters(response, req_extraction_model_name)
        
        if isinstance(response, RunResponse):
            batch_result = response.content
        else:
            batch_result = response
        
        # Map results back to requirement IDs
        results = {}
        for i, analysis in enumerate(batch_result.analyses):
            if i < len(requirements):
                req_id = requirements[i].id
                results[req_id] = analysis
                
        logger.info(f"✅ Batch analysis complete for {len(results)} requirements")
        return results
    
    except Exception as e:
        logger.error(f"❌ Error during batch analysis: {e}")
        
        # Fall back to individual analysis
        logger.info("🔄 Falling back to individual requirement analysis...")
        results = {}
        for req in requirements:
            results[req.id] = analyze_requirement(req, agent)
            
        return results

def load_embedding_model():
    """Load the embedding model."""
    try:
        # Set up sentence-transformers logging to include our script prefix
        st_logger = logging.getLogger('sentence_transformers')
        for handler in st_logger.handlers:
            st_logger.removeHandler(handler)
        st_logger.addHandler(logging.StreamHandler())
        st_logger.setLevel(logging.INFO)
        # Wrap the sentence-transformers logger with our ScriptLogger
        st_logger = ScriptLogger(st_logger, "[Extract_Requirements] ")
        
        model = sentence_transformers.SentenceTransformer(EMBEDDING_MODEL)
        return model
    except Exception as e:
        logger.error(f"❌ Error loading embedding model: {e}")
        return None

def get_document_chunks(db, document_id: str):
    """
    Get all chunks for a document from the LanceDB document_chunks table.
    
    Args:
        db: LanceDB connection
        document_id: The document ID to get chunks for
    
    Returns:
        DataFrame containing all chunks for the document
    """
    if not db:
        logger.error("❌ No database connection")
        return None
    
    try:
        # Check if table exists
        logger.info(f"Checking if document_chunks table exists in database")
        table_names = db.table_names()
        logger.info(f"Available tables in database: {', '.join(table_names)}")
        
        if DOCUMENT_CHUNKS_TABLE not in table_names:
            logger.error(f"❌ Document chunks table '{DOCUMENT_CHUNKS_TABLE}' does not exist")
            return None
        
        # Open table
        logger.info(f"Opening document_chunks table...")
        table = db.open_table(DOCUMENT_CHUNKS_TABLE)
        
        # Query chunks for document
        # For LanceDB 0.22.0, use pandas filtering
        try:
            logger.info(f"Reading all chunks from table and filtering for document_id: {document_id}")
            all_chunks = table.to_pandas()
            
            if all_chunks.empty:
                logger.warning(f"⚠️ Chunks table is empty - no chunks found at all")
                return None
                
            # Debug - show the columns in the table
            logger.info(f"Columns in chunks table: {list(all_chunks.columns)}")
            
            # Debug - check if document_id column exists
            if 'document_id' not in all_chunks.columns:
                logger.error(f"❌ 'document_id' column not found in chunks table. Available columns: {list(all_chunks.columns)}")
                return None
                
            # Debug - Show all unique document IDs
            unique_docs = all_chunks['document_id'].unique()
            logger.info(f"Documents in chunks table: {unique_docs}")
            
            results = all_chunks[all_chunks['document_id'] == document_id]
            
            if results.empty:
                logger.warning(f"⚠️ No chunks found for document '{document_id}'")
                return None
            
            logger.info(f"✅ Found {len(results)} chunks for document '{document_id}'")
            return results
        except Exception as e:
            logger.error(f"❌ Error querying document chunks: {e}")
            return None
    
    except Exception as e:
        logger.error(f"❌ Error querying document chunks: {e}")
        return None

def get_document_info(db, document_id: str):
    """
    Get document info from the LanceDB documents table.
    
    Args:
        db: LanceDB connection
        document_id: The document ID to get info for
    
    Returns:
        DataFrame containing document info (first page is sufficient)
    """
    if not db:
        logger.error("❌ No database connection")
        return None
    
    try:
        # Check if table exists
        if DOCUMENTS_TABLE not in db.table_names():
            logger.error(f"❌ Documents table '{DOCUMENTS_TABLE}' does not exist")
            return None
        
        # Open table
        table = db.open_table(DOCUMENTS_TABLE)
        
        # Query first page of document using to_pandas() for LanceDB 0.22.0
        try:
            all_docs = table.to_pandas()
            results = all_docs[all_docs['pdf_identifier'] == document_id]
            
            if results.empty:
                logger.warning(f"⚠️ No document info found for '{document_id}'")
                return None
            
            return results.iloc[0]
        except KeyError as ke:
            logger.error(f"❌ KeyError accessing column in documents table: {ke}")
            if 'all_docs' in locals() and hasattr(all_docs, 'columns'):
                logger.error(f"ℹ️ Available columns in documents table: {list(all_docs.columns)}")
            else:
                logger.error("ℹ️ Could not retrieve column list from documents table.")
            return None
        except Exception as e:
            logger.error(f"❌ Error querying document info: {e}")
            logger.error(f"Traceback: {traceback.format_exc() if hasattr(traceback, 'format_exc') else 'N/A'}")
            return None
    
    except Exception as e:
        logger.error(f"❌ Error querying document info: {e}")
        return None

def get_existing_requirements_for_chunk(db, chunk_id: str):
    """
    Get existing requirements for a specific chunk.
    
    Args:
        db: LanceDB connection
        chunk_id: The chunk ID to get requirements for
    
    Returns:
        DataFrame containing requirements for the chunk
    """
    if not db:
        logger.error("❌ No database connection")
        return None
    
    # Check if table exists
    if REQUIREMENTS_TABLE not in db.table_names():
        logger.info(f"ℹ️ Requirements table '{REQUIREMENTS_TABLE}' does not exist yet")
        return None
    
    try:
        # Open table
        table = db.open_table(REQUIREMENTS_TABLE)
        
        # Query requirements for chunk
        # For LanceDB 0.22.0, use pandas filtering
        try:
            all_reqs = table.to_pandas()
            results = all_reqs[all_reqs['source_chunk_id'] == chunk_id]
            
            if results.empty:
                logger.info(f"ℹ️ No existing requirements found for chunk '{chunk_id}'")
                return None
            
            logger.info(f"✅ Found {len(results)} existing requirements for chunk '{chunk_id}'")
            return results
        except Exception as e:
            logger.error(f"❌ Error querying existing requirements: {e}")
            return None
    
    except Exception as e:
        logger.error(f"❌ Error querying existing requirements: {e}")
        return None

def process_single_document(doc_id: str):
    """
    Process a single document for requirements extraction.
    
    Args:
        doc_id: The document ID (filename) to process
    """
    logger.info(f"🔄 Processing document: {doc_id}")
    
    # Step 1: Connect to LanceDB
    db = connect_to_lancedb()
    if not db:
        logger.error(f"❌ Failed to connect to LanceDB database")
        return
    
    # Step 2: Get document info and check tables exist
    document_info = get_document_info(db, doc_id)
    if document_info is None:
        logger.error(f"❌ Document {doc_id} not found in database")
        return
    
    document_title = document_info.get('document_title', doc_id)
    logger.info(f"✅ Found document info: {document_title}")
    
    # Step 3: Get document chunks
    chunks_df = get_document_chunks(db, doc_id)
    if chunks_df is None or chunks_df.empty:
        logger.error(f"❌ No chunks found for document {doc_id}")
        return
    
    # Step 4: Load embedding model
    logger.info(f"🔄 Loading embedding model '{EMBEDDING_MODEL}'...")
    embedding_model = load_embedding_model()
    if not embedding_model:
        logger.error(f"❌ Failed to load embedding model '{EMBEDDING_MODEL}'")
        return
    logger.info(f"✅ Successfully loaded embedding model: {EMBEDDING_MODEL}")
    
    # Step 5: Get requirements table
    logger.info(f"🔄 Accessing requirements table '{REQUIREMENTS_TABLE}'...")
    requirements_table = get_or_create_requirements_table(db)
    if not requirements_table:
        logger.error(f"❌ Failed to get requirements table '{REQUIREMENTS_TABLE}'")
        return
    logger.info(f"✅ Successfully accessed requirements table '{REQUIREMENTS_TABLE}'")
    
    # Track all extracted requirements for batch processing
    all_requirements = []
    
    # Track how many requirements we save in total
    total_requirements = 0
    
    # Create a software analysis agent for later batch processing
    software_agent = create_agent(
        response_model=BatchSoftwareAnalysis,
        description=SOFTWARE_AGENT_DESCRIPTION
    )
    
    try:
        # Sort chunks by index
        sorted_chunks = chunks_df.sort_values(by="chunk_index")
        
        # Process each chunk
        for index, chunk in sorted_chunks.iterrows():
            chunk_id = chunk.get('chunk_id')
            chunk_text = chunk.get('chunk_text', '')
            chunk_index = chunk.get('chunk_index', index)
            is_duplicate = chunk.get('is_duplicate', False)
            duplicate_of = chunk.get('duplicate_of')
            
            logger.info(f"🔄 Processing chunk {chunk_index+1}/{len(sorted_chunks)}: {chunk_id}")
            
            # If this chunk is a duplicate, get requirements from the original chunk
            if is_duplicate and duplicate_of:
                logger.info(f"ℹ️ Chunk {chunk_id} is a duplicate of {duplicate_of}, reusing requirements")
                existing_reqs = get_existing_requirements_for_chunk(db, duplicate_of)
                
                if existing_reqs is not None and not existing_reqs.empty:
                    # Update source_chunk_id to point to the new chunk
                    for i, req in existing_reqs.iterrows():
                        req_data = req.to_dict()
                        req_data['source_chunk_id'] = chunk_id
                        all_requirements.append(req_data)
                    
                    logger.info(f"✅ Reused {len(existing_reqs)} requirements from duplicate chunk")
                    continue
                else:
                    logger.warning(f"⚠️ No requirements found for original chunk {duplicate_of}, will extract new ones")
            
            # Check if there are already requirements for this chunk
            existing_reqs = get_existing_requirements_for_chunk(db, chunk_id)
            if existing_reqs is not None and not existing_reqs.empty:
                logger.info(f"ℹ️ {len(existing_reqs)} existing requirements found for chunk {chunk_id}, skipping extraction")
                continue
            
            # Extract requirements from this chunk
            if not chunk_text:
                logger.warning(f"⚠️ Empty chunk text for {chunk_id}, skipping")
                continue
                
            logger.info(f"🔄 Extracting requirements from chunk {chunk_id} (content length: {len(chunk_text)} chars)")
            
            try:
                # Extract requirements from this chunk
                requirements_data = extract_requirements_from_text(
                    text=chunk_text,
                    document_id=doc_id,
                    document_name=document_title,
                    chunk_id=chunk_id
                )
                
                # Process extracted requirements
                if hasattr(requirements_data, 'requirements') and requirements_data.requirements:
                    req_count = len(requirements_data.requirements)
                    logger.info(f"🔄 Found {req_count} requirements in chunk {chunk_id}, preparing to process...")
                    
                    # Create requirement records for each extracted requirement
                    for i, req in enumerate(requirements_data.requirements):
                        try:
                            # Create embedding for the requirement
                            logger.info(f"🔄 Creating embedding for requirement {i+1}/{req_count}: {req.id} - '{req.title}'")
                            req_text = f"{req.title}\n{req.description}"
                            # Prepend instruction for e5 models
                            embedding_text = f"passage: {req_text}"
                            embedding = embedding_model.encode(embedding_text).tolist()
                            
                            # Create a record for LanceDB
                            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                            
                            # Correctly set the source_chunk_id field
                            req_data = {
                                "requirement_id": req.id,
                                "document_id": doc_id,
                                "page_number": None,  # Will be updated in batch processing
                                "section": req.section,
                                "title": req.title,
                                "description": req.description,
                                "requirement_rationale": req.requirement_rationale,
                                "is_software_related": False,  # Default values
                                "software_confidence": 0.0,    
                                "software_rationale": "",    
                                "source_text": req.source_text,
                                "source_chunk_id": req.source_chunk_id or chunk_id,
                                "created_timestamp": timestamp,
                                "embedding": embedding,
                                "is_duplicate": False,
                                "duplicate_of": None
                            }
                            
                            # Add to list of all requirements for batch processing
                            all_requirements.append(req_data)
                        except Exception as e:
                            logger.error(f"❌ Error processing requirement {req.id}: {str(e)}")
                            logger.error(f"❌ Traceback: {traceback.format_exc()}")
                            continue
                else:
                    logger.info(f"ℹ️ No requirements found in chunk {chunk_id}")
            except Exception as e:
                logger.error(f"❌ Error processing chunk {chunk_id}: {str(e)}")
                logger.error(f"❌ Traceback: {traceback.format_exc()}")
        
        # Check for duplicate requirements if deduplication module is available
        if DEDUPLICATION_AVAILABLE and all_requirements:
            logger.info(f"🔄 Checking for duplicate requirements in {len(all_requirements)} extracted requirements...")
            unique_requirements, duplicate_info = check_requirements_duplicates(all_requirements, doc_id)
            
            logger.info(f"✅ Deduplication complete: {len(unique_requirements)} unique, {len(duplicate_info)} duplicates")
            
            # Batch process requirements for software relation
            if unique_requirements:
                # Convert requirements to AtomicRequirement objects for the agent
                atomic_reqs = []
                for req_data in unique_requirements:
                    atomic_req = AtomicRequirement(
                        id=req_data.get("requirement_id", ""),
                        title=req_data.get("title", ""),
                        description=req_data.get("description", ""),
                        source_text=req_data.get("source_text", ""),
                        requirement_rationale=req_data.get("requirement_rationale", ""),
                        section=req_data.get("section", ""),
                        page_number=req_data.get("page_number"),
                        source_chunk_id=req_data.get("source_chunk_id", "")
                    )
                    atomic_reqs.append(atomic_req)
                
                # Analyze for software relation
                logger.info(f"🔄 Analyzing {len(atomic_reqs)} requirements for software relation...")
                software_results = analyze_requirements_batch(atomic_reqs, software_agent)
                
                # Update requirements with software analysis
                for req_data in unique_requirements:
                    req_id = req_data.get("requirement_id")
                    if req_id in software_results:
                        analysis = software_results[req_id]
                        req_data["is_software_related"] = analysis.is_software_related
                        req_data["software_confidence"] = analysis.confidence
                        req_data["software_rationale"] = analysis.rationale
            
            # Save unique requirements to LanceDB
            logger.info(f"🔄 Saving {len(unique_requirements)} requirements to LanceDB...")
            saved_count = 0
            
            for req_data in unique_requirements:
                try:
                    req_id = req_data.get('requirement_id')
                    logger.info(f"🔄 Saving requirement {req_id} to LanceDB...")
                    
                    # Add to LanceDB
                    requirements_table.add([req_data])
                    saved_count += 1
                    logger.info(f"✅ Successfully saved requirement {req_id}")
                except Exception as e:
                    logger.error(f"❌ Error saving requirement {req_data.get('requirement_id')}: {str(e)}")
                    logger.error(f"❌ Traceback: {traceback.format_exc()}")
            
            # Save duplicates with the duplicate flag
            if duplicate_info:
                logger.info(f"🔄 Saving {len(duplicate_info)} duplicate requirements with duplicate flag...")
                
                for idx, dup_info in duplicate_info.items():
                    idx = int(idx)
                    if idx < len(all_requirements):
                        req_data = all_requirements[idx]
                        req_data["is_duplicate"] = True
                        req_data["duplicate_of"] = dup_info.get("duplicate_id")
                        
                        try:
                            req_id = req_data.get('requirement_id')
                            logger.info(f"🔄 Saving duplicate requirement {req_id} (duplicate of {dup_info.get('duplicate_id')})...")
                            
                            # Add to LanceDB
                            requirements_table.add([req_data])
                            saved_count += 1
                            logger.info(f"✅ Successfully saved duplicate requirement {req_id}")
                        except Exception as e:
                            logger.error(f"❌ Error saving duplicate requirement: {str(e)}")
            
            total_requirements = saved_count
        else:
            # If no deduplication, save all requirements directly
            logger.info(f"🔄 Saving {len(all_requirements)} requirements to LanceDB without deduplication...")
            saved_count = 0
            
            for req_data in all_requirements:
                try:
                    req_id = req_data.get('requirement_id')
                    logger.info(f"🔄 Saving requirement {req_id} to LanceDB...")
                    
                    # Add to LanceDB
                    requirements_table.add([req_data])
                    saved_count += 1
                    logger.info(f"✅ Successfully saved requirement {req_id}")
                except Exception as e:
                    logger.error(f"❌ Error saving requirement {req_data.get('requirement_id')}: {str(e)}")
                    logger.error(f"❌ Traceback: {traceback.format_exc()}")
            
            total_requirements = saved_count
    
    except Exception as e:
        logger.error(f"❌ Error during document processing: {str(e)}")
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
    
    # Log completion
    logger.info(f"✅ Extraction complete. Successfully processed {total_requirements} requirements from document '{doc_id}'")
    print_token_usage() # Add token usage printing here

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Usage: python extract_requirements.py <document_id>", extra={"icon": "❌"})
        sys.exit(1)
    
    document_id = sys.argv[1]
    process_single_document(document_id)