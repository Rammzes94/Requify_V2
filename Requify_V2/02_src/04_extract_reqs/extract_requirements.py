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

This script extracts atomic requirements from parsed document content stored in LanceDB.
It performs the following operations:
1. Connects to LanceDB and retrieves document pages
2. Uses LLMs to extract atomic requirements from each page's content
3. Generates embeddings for each requirement
4. Checks for duplicate requirements using the pre_save_reqs_deduplication module
5. Saves unique requirements to the requirements table in LanceDB

The script supports both OpenAI and Groq models for text processing and can be 
configured via environment variables. It maintains the relationship between 
requirements and their source documents.
"""

# -------------------------------------------------------------------------------------
# Setup and Configuration
# -------------------------------------------------------------------------------------

# Setup project directory and load environment variables
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import _00_utils
from _00_utils import setup_project_directory, update_token_counters, print_token_usage
setup_project_directory()
load_dotenv()

# Import the deduplication module for requirements
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '05_reqs_deduplication')))
try:
    import pre_save_reqs_deduplication as reqs_dedup
except ImportError:
    print("Warning: Could not import pre_save_reqs_deduplication module. Duplicate detection will be disabled.")
    reqs_dedup = None

# Configure logging with script prefix
logger = _00_utils.setup_logging()

class ScriptLogger(logging.LoggerAdapter):
    def __init__(self, logger, prefix):
        super().__init__(logger, {})
        self.prefix = prefix
        
    def process(self, msg, kwargs):
        return f"{self.prefix}{msg}", kwargs

logger = ScriptLogger(_00_utils.setup_logging(), "[Extract_Requirements] ")

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

# Model Configuration
# Initialize models for requirements extraction
openai_model = OpenAIChat(id="gpt-4o-mini", api_key=api_key, temperature=0)

# Uncomment to use Groq models instead
groq_model = Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key, temperature=0)

# Select which model to use
#active_model = openai_model
active_model = groq_model

# LanceDB settings
OUTPUT_DIR_BASE = "03_output"  # Define base output directory
LANCEDB_SUBDIR_NAME = "lancedb"  # Subdirectory for LanceDB within 03_output
LANCEDB_DIR_PATH = os.path.join(OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME) # Construct path relative to project root
LANCEDB_URI = LANCEDB_DIR_PATH  # Alias for consistency
SOURCE_TABLE_NAME = "documents"
TARGET_TABLE_NAME = "requirements"
EMBEDDING_DIMENSION = 1024  # Maintain same dimension as source table
EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"  # Default embedding model

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

def extract_requirements_from_text(text: str, document_id: str, document_name: str, requirement_count_offset: int = 0) -> RequirementsExtractor:
    """Extract requirements from text."""
    logger.info("🔄 Extracting requirements from text chunk...")
    
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
        _00_utils.update_token_counters(response)
        
        # Ensure the response is a RequirementsExtractor object
        if isinstance(response, RunResponse):
            # Access the content field, which contains the structured data
            requirements_data = response.content
        else:
            requirements_data = response
        
        # Log the results
        if hasattr(requirements_data, 'requirements') and len(requirements_data.requirements) > 0:
            logger.info(f"✅ Successfully extracted {len(requirements_data.requirements)} requirements")
        else:
            logger.info("ℹ️ No requirements found in this text chunk")
            
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
        
        return requirements_data
    
    except Exception as e:
        logger.error(f"❌ Error extracting requirements: {e}")
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
        _00_utils.update_token_counters(response)
        
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
        logger.info(f"🔄 Connecting to LanceDB database at {lancedb_path}...")
        db = lancedb.connect(lancedb_path)
        logger.info(f"✅ Connected to LanceDB at {lancedb_path}")
        return db
    except Exception as e:
        logger.error(f"❌ Failed to connect to LanceDB: {e}")
        return None

def get_or_create_requirements_table(db):
    """Get the requirements table from LanceDB. Table must exist already."""
    if not db:
        logger.error("❌ [LanceDB] No database connection provided for requirements table access")
        return None
        
    try:
        logger.info(f"🔍 [LanceDB] Checking for existing table '{TARGET_TABLE_NAME}' in database")
        table_names = db.table_names()
        logger.info(f"📋 [LanceDB] Available tables in database: {', '.join(table_names) if table_names else 'None'}")
        
        if TARGET_TABLE_NAME in table_names:
            logger.info(f"✅ [LanceDB] Found existing requirements table: '{TARGET_TABLE_NAME}'")
            try:
                logger.info(f"🔄 [LanceDB] Attempting to open table '{TARGET_TABLE_NAME}'...")
                table = db.open_table(TARGET_TABLE_NAME)
                
                if table is None:
                    logger.error(f"❌ [LanceDB] Failed to open table '{TARGET_TABLE_NAME}' - table object is None")
                    return None
                
                logger.info(f"🔍 [LanceDB] Validating table schema for '{TARGET_TABLE_NAME}'")
                schema = table.schema
                
                # Fixed: PyArrow Schema uses field() method to get fields by index, not a fields attribute
                field_names = []
                for i in range(len(schema)):
                    field = schema.field(i)
                    field_names.append(field.name)
                
                logger.info(f"📊 [LanceDB] Table schema contains {len(field_names)} fields: {', '.join(field_names)}")
                
                # Check for required fields
                required_fields = ["requirement_id", "document_id", "title", "description", "embedding"]
                missing_fields = [field for field in required_fields if field not in field_names]
                
                if missing_fields:
                    logger.warning(f"⚠️ [LanceDB] Table '{TARGET_TABLE_NAME}' is missing some expected fields: {', '.join(missing_fields)}")
                else:
                    logger.info(f"✅ [LanceDB] Table '{TARGET_TABLE_NAME}' contains all required fields")
                
                # Make sure we have a valid table object with the 'add' method
                if not hasattr(table, 'add'):
                    logger.error(f"❌ [LanceDB] Table object for '{TARGET_TABLE_NAME}' is missing 'add' method - not a valid LanceTable")
                    return None
                
                logger.info(f"✅ [LanceDB] Successfully opened table '{TARGET_TABLE_NAME}' for writing")
                return table
            except Exception as e:
                logger.error(f"❌ [LanceDB] Error accessing existing table '{TARGET_TABLE_NAME}': {str(e)}")
                logger.error(f"❌ [LanceDB] Traceback: {traceback.format_exc()}")
                return None
        else:
            logger.error(f"❌ [LanceDB] Table '{TARGET_TABLE_NAME}' does not exist in the database and must be created externally")
            logger.error(f"❌ [LanceDB] Available tables: {', '.join(table_names) if table_names else 'None'}")
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
        _00_utils.update_token_counters(response)
        
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
        model = sentence_transformers.SentenceTransformer(EMBEDDING_MODEL)
        return model
    except Exception as e:
        logger.error(f"❌ Error loading embedding model: {e}")
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
    
    # Step 2: Get all pages for the document
    logger.info(f"🔄 Opening source table '{SOURCE_TABLE_NAME}' to retrieve document pages...")
    try:
        table = db.open_table(SOURCE_TABLE_NAME)
    except Exception as e:
        logger.error(f"❌ Failed to open source table '{SOURCE_TABLE_NAME}': {e}")
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return
        
    logger.info(f"🔄 Querying records for document ID: {doc_id}...")
    try:
        # Get all pages for the document from LanceDB 
        records = table.to_pandas()
        doc_pages = records[records["pdf_identifier"] == doc_id]
        
        if doc_pages.empty:
            logger.error(f"❌ No pages found for document ID '{doc_id}' in table '{SOURCE_TABLE_NAME}'")
            return
        
        logger.info(f"✅ Found {len(doc_pages)} pages for document '{doc_id}'")
    except Exception as e:
        logger.error(f"❌ Error querying document pages from LanceDB: {e}")
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return
        
    # Step 3: Load embedding model
    logger.info(f"🔄 Loading embedding model '{EMBEDDING_MODEL}'...")
    embedding_model = load_embedding_model()
    if not embedding_model:
        logger.error(f"❌ Failed to load embedding model '{EMBEDDING_MODEL}'")
        return
    logger.info(f"✅ Successfully loaded embedding model: {EMBEDDING_MODEL}")
    
    # Step 4: Get requirements table
    logger.info(f"🔄 Accessing requirements table '{TARGET_TABLE_NAME}'...")
    requirements_table = get_or_create_requirements_table(db)
    
    # DEBUG: Check if we actually got a table back and examine its schema
    if requirements_table is not None:
        try:
            schema = requirements_table.schema
            logger.info(f"🔍 Requirements table schema: {schema}")
            
            # Fixed: PyArrow Schema uses field() method to get fields by index, not a fields attribute
            field_names = []
            for i in range(len(schema)):
                field = schema.field(i)
                field_names.append(field.name)
                
            logger.info(f"🔍 Available fields in table: {field_names}")
        except Exception as e:
            logger.error(f"❌ Error examining table schema: {e}")
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
    
    # The table is correctly returned but somehow the condition is failing
    # Force it to True if we have a valid LanceTable object
    table_exists = requirements_table is not None and hasattr(requirements_table, 'add')
    
    if not table_exists:
        logger.error(f"❌ Failed to get requirements table '{TARGET_TABLE_NAME}'")
        return
    logger.info(f"✅ Successfully accessed requirements table '{TARGET_TABLE_NAME}'")
    
    # Track all extracted requirements for batch processing
    all_requirements = []
    
    # Track how many requirements we save in total
    total_requirements = 0
    
    try:
        # For each page in the document
        for index, page in doc_pages.iterrows():
            page_number = page.get('page_number')
            logger.info(f"🔄 Processing page {page_number}/{len(doc_pages)} of document '{doc_id}'")
            
            # Get the content field - try different column names depending on LanceDB schema
            page_content = page.get('page_content', page.get('md_content', page.get('processed_content', '')))
            if not page_content:
                logger.warning(f"⚠️ No content found for page {page_number} of document '{doc_id}'")
                continue
                
            logger.info(f"🔄 Extracting requirements from page {page_number} (content length: {len(page_content)} chars)")
            
            try:
                # Extract requirements from this page
                requirements_data = extract_requirements_from_text(
                    text=page_content,
                    document_id=doc_id,
                    document_name=doc_id
                )
                
                # Process extracted requirements
                if hasattr(requirements_data, 'requirements') and requirements_data.requirements:
                    req_count = len(requirements_data.requirements)
                    logger.info(f"🔄 Found {req_count} requirements on page {page_number}, preparing to process...")
                    
                    # Create requirement records for each extracted requirement
                    for i, req in enumerate(requirements_data.requirements):
                        try:
                            # Create embedding for the requirement
                            logger.info(f"🔄 Creating embedding for requirement {i+1}/{req_count}: {req.id} - '{req.title}'")
                            req_text = f"{req.title}\n{req.description}"
                            embedding = embedding_model.encode([req_text])[0].tolist()
                            
                            # Create a record for LanceDB
                            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                            
                            # Use only source_text
                            req_data = {
                                "requirement_id": req.id,
                                "document_id": doc_id,
                                "page_number": page_number,
                                "section": req.section,
                                "title": req.title,
                                "description": req.description,
                                "requirement_rationale": req.requirement_rationale,
                                "is_software_related": False,  # Default values
                                "software_confidence": 0.0,    
                                "software_rationale": "",    
                                "source_text": req.source_text,  # Just use source_text
                                "created_timestamp": timestamp,
                                "embedding": embedding,
                                "is_duplicate": False,
                                "duplicate_of": None
                            }
                            
                            # Add to list of all requirements for deduplication
                            all_requirements.append(req_data)
                        except Exception as e:
                            logger.error(f"❌ Error processing requirement {req.id}: {str(e)}")
                            logger.error(f"❌ Traceback: {traceback.format_exc()}")
                            continue
                else:
                    logger.info(f"ℹ️ No requirements found on page {page_number} of document '{doc_id}'")
            except Exception as e:
                logger.error(f"❌ Error processing page {page_number}: {str(e)}")
                logger.error(f"❌ Traceback: {traceback.format_exc()}")
        
        # Check for duplicate requirements if deduplication module is available
        if reqs_dedup and all_requirements:
            logger.info(f"🔄 Checking for duplicate requirements in {len(all_requirements)} extracted requirements...")
            unique_requirements, duplicate_info = reqs_dedup.check_requirements_duplicates(all_requirements, doc_id)
            
            logger.info(f"✅ Deduplication complete: {len(unique_requirements)} unique, {len(duplicate_info)} duplicates")
            
            # Save unique requirements to LanceDB
            logger.info(f"🔄 Saving {len(all_requirements)} requirements to LanceDB...")
            saved_count = 0
            
            for req_data in all_requirements:
                try:
                    req_id = req_data.get('requirement_id')
                    
                    # If it's a duplicate, log it but still save with the duplicate flag
                    if req_id and any(duplicate_info.get(i, {}).get('duplicate_id') == req_id for i in duplicate_info):
                        logger.info(f"ℹ️ Requirement {req_id} is a duplicate, saving with duplicate flag")
                    
                    logger.info(f"🔄 Saving requirement {req_id} to LanceDB table '{TARGET_TABLE_NAME}'...")
                    
                    # Add to LanceDB
                    requirements_table.add([req_data])
                    saved_count += 1
                    logger.info(f"✅ Successfully saved requirement {req_id}")
                except Exception as e:
                    logger.error(f"❌ Error saving requirement {req_data.get('requirement_id')}: {str(e)}")
                    logger.error(f"❌ Traceback: {traceback.format_exc()}")
            
            total_requirements = saved_count
        else:
            # If no deduplication, save all requirements directly
            logger.info(f"🔄 Saving {len(all_requirements)} requirements to LanceDB without deduplication...")
            saved_count = 0
            
            for req_data in all_requirements:
                try:
                    req_id = req_data.get('requirement_id')
                    logger.info(f"🔄 Saving requirement {req_id} to LanceDB table '{TARGET_TABLE_NAME}'...")
                    
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
    logger.info(f"✅ Extraction complete. Successfully processed {total_requirements} requirements from {len(doc_pages)} pages of document '{doc_id}'")
    _00_utils.print_token_usage() # Add token usage printing here