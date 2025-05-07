import os
import sys
import logging
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
import lancedb
from lancedb.pydantic import LanceModel, Vector
from dotenv import load_dotenv

# -------------------------------------------------------------------------------------
# Setup and Configuration
# -------------------------------------------------------------------------------------

# Setup project directory and load environment variables
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from _00_utils import setup_project_directory, update_token_counters, print_token_usage
setup_project_directory()
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

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

# LanceDB settings
LANCEDB_DIR_PATH = "lancedb"
SOURCE_TABLE_NAME = "all_pdf_pages"
TARGET_TABLE_NAME = "requirements"
EMBEDDING_DIMENSION = 1024  # Maintain same dimension as source table

# Document filter - set to None to process all documents
# To process all documents, change to: TARGET_DOCUMENT = None
TARGET_DOCUMENT = "fighter_jet_rocket_launcher_spec_3_language_variant.pdf"

# Define cost per million tokens
INPUT_COST_PER_MILLION = 0.15  # per million input tokens for 4o-mini
OUTPUT_COST_PER_MILLION = 0.60  # per million output tokens for 4o-mini

# Initialize token counters as global variables - using the ones from _00_utils

# -------------------------------------------------------------------------------------
# Prompt Templates
# -------------------------------------------------------------------------------------

EXTRACT_REQUIREMENTS_PROMPT = """Extract all atomic requirements from the following text. 
Focus on sections that define technical specifications, technical requirements, and mandatory elements - things focused on the product.

For the following input, identify ALL individual requirements and break them down into atomic units.
Each atomic requirement should be a single, testable statement. You have to give me all requirements!
Extract all technical prescriptions, specifications, and mandatory elements.

For each requirement, provide a rationale explaining why it is considered a technical product requirement. This should focus on the technical aspects that affect the physical vehicle or its systems.

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
    """Model for a single atomic requirement extracted from a regulation."""
    id: str = Field(..., description="Unique identifier for this requirement (e.g., '0001')")
    section: str = Field(..., description="The reference number from the document (e.g., 'x.y.z', 5.2.3; must include all digits)")
    title: str = Field(..., description="Short title summarizing the requirement")
    description: str = Field(..., description="Full text of the requirement")
    rationale: str = Field(..., description="Explanation of why this is considered a technical product requirement")

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
    source_text: str
    created_timestamp: str
    embedding: Vector(EMBEDDING_DIMENSION)

# -------------------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------------------

def create_agent(response_model, description):
    """Create an agent with the specified model and description."""
    return Agent(
        model=OpenAIChat(id="gpt-4o-mini", api_key=api_key, temperature=0),
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
        update_token_counters(response)
        
        extracted_reqs = response.content
        
        # Update document information
        extracted_reqs.document_id = document_id
        extracted_reqs.document_name = document_name
            
        # Update requirement IDs to be sequential across chunks
        if extracted_reqs.total_count > 0:
            for i, req in enumerate(extracted_reqs.requirements):
                new_id = f"{document_id}-{(i + 1 + requirement_count_offset):04d}"
                req.id = new_id
        
        return extracted_reqs
    except Exception as e:
        logger.error(f"❌ Error extracting requirements: {e}")
        # Return an empty requirements object in case of error
        return RequirementsExtractor(
            document_name=document_name,
            document_id=document_id,
            requirements=[],
            total_count=0,
            rationale=f"Error occurred during extraction: {str(e)}"
        )

def analyze_requirement(requirement: AtomicRequirement, agent: Agent) -> SoftwareRequirementAnalysis:
    """Analyze a single requirement to determine if it's software-related."""
    try:
        logger.info(f"🔍 Analyzing requirement {requirement.id} - {requirement.description}")
        prompt = ANALYZE_REQUIREMENT_PROMPT.format(requirement_description=requirement.description)
        
        response = agent.run(prompt)
        update_token_counters(response)
        return response.content
    except Exception as e:
        logger.error(f"❌ Error analyzing requirement: {e}")
        # Return a default analysis in case of error
        return SoftwareRequirementAnalysis(
            is_software_related=False,
            confidence=0.0,
            rationale=f"Error occurred during analysis: {str(e)}"
        )

def connect_to_lancedb():
    """Connect to LanceDB and return the connection object."""
    try:
        logger.info("🔄 Connecting to LanceDB...")
        db = lancedb.connect(LANCEDB_DIR_PATH)
        logger.info(f"✅ Connected to LanceDB. Available tables: {db.table_names()}")
        return db
    except Exception as e:
        logger.error(f"❌ Error connecting to LanceDB: {e}")
        raise

def get_or_create_requirements_table(db):
    """Get or create the requirements table."""
    try:
        if TARGET_TABLE_NAME in db.table_names():
            logger.info(f"✅ Opening existing table: {TARGET_TABLE_NAME}")
            return db.open_table(TARGET_TABLE_NAME)
        else:
            logger.info(f"🔄 Creating new table: {TARGET_TABLE_NAME}")
            return db.create_table(TARGET_TABLE_NAME, schema=Requirement)
    except Exception as e:
        logger.error(f"❌ Error getting or creating table: {e}")
        raise

def save_requirements_to_lancedb(requirements_list, source_text_map, embedding_map, table):
    """Save requirements to LanceDB table."""
    if not requirements_list:
        logger.info("ℹ️ No requirements to save.")
        return
    
    logger.info(f"🔄 Saving {len(requirements_list)} requirements to LanceDB...")
    records = []
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    for req in requirements_list:
        req_id = req["id"]
        source_text = source_text_map.get(req_id, "")
        embedding = embedding_map.get(req_id, None)
        
        if not embedding:
            logger.warning(f"⚠️ No embedding found for requirement {req_id}")
            continue
        
        # Create record for LanceDB using direct column values
        record = Requirement(
            requirement_id=req_id,
            document_id=req["document_id"],
            page_number=req["page_number"],
            section=req["section"],
            title=req["title"],
            description=req["description"],
            requirement_rationale=req["requirement_rationale"],
            is_software_related=req["software_analysis"].is_software_related,
            software_confidence=req["software_analysis"].confidence,
            software_rationale=req["software_analysis"].rationale,
            source_text=source_text,
            created_timestamp=timestamp,
            embedding=embedding
        )
        records.append(record)
    
    try:
        table.add(records)
        logger.info(f"✅ Successfully saved {len(records)} requirements to LanceDB")
    except Exception as e:
        logger.error(f"❌ Error saving requirements to LanceDB: {e}")
        raise

# -------------------------------------------------------------------------------------
# Main Process Functions
# -------------------------------------------------------------------------------------

def analyze_requirements_batch(requirements: List[AtomicRequirement], agent: Agent) -> Dict[str, SoftwareRequirementAnalysis]:
    """Analyze a batch of requirements to determine if they're software-related."""
    if not requirements:
        return {}
    
    # Prepare the requirements text for the prompt
    requirements_text = ""
    for i, req in enumerate(requirements, 1):
        requirements_text += f"Requirement {i}: \"{req.description}\"\n\n"
    
    try:
        prompt = BATCH_ANALYZE_REQUIREMENTS_PROMPT.format(requirements_text=requirements_text)
        
        response = agent.run(prompt)
        update_token_counters(response)
        
        batch_analysis = response.content
        
        # Map the analyses back to the requirements by ID
        results = {}
        for i, req in enumerate(requirements):
            if i < len(batch_analysis.analyses):
                results[req.id] = batch_analysis.analyses[i]
            else:
                # Fallback in case of mismatch
                logger.warning(f"⚠️ Missing analysis for requirement {req.id}, using default")
                results[req.id] = SoftwareRequirementAnalysis(
                    is_software_related=False,
                    confidence=0.0,
                    rationale="Analysis not provided"
                )
                
        return results
        
    except Exception as e:
        logger.error(f"❌ Error analyzing requirements batch: {e}")
        # Return default analyses in case of error
        return {req.id: SoftwareRequirementAnalysis(
            is_software_related=False,
            confidence=0.0,
            rationale=f"Error occurred during batch analysis: {str(e)}"
        ) for req in requirements}

def process_document_page(page_data, embedder):
    """Process a single document page to extract requirements."""
    document_id = page_data['pdf_identifier']
    page_number = page_data.get('page_number', 0)
    md_content = page_data.get('md_content', '')
    
    if not md_content:
        logger.warning(f"⚠️ No content to process for {document_id} page {page_number}")
        return [], {}
    
    # Use title if available, otherwise create a descriptive name
    document_name = page_data.get('title', f"Document {document_id}")
    
    # Extract requirements from the content
    extracted_reqs = extract_requirements_from_text(
        text=md_content, 
        document_id=document_id,
        document_name=document_name
    )
    
    if extracted_reqs.total_count == 0:
        logger.info(f"ℹ️ No requirements found in {document_id} page {page_number}")
        return [], {}
    
    # Analyze requirements for software relevance in batches of 5
    software_agent = create_agent(
        response_model=BatchSoftwareAnalysis,
        description=SOFTWARE_AGENT_DESCRIPTION
    )
    
    analyzed_requirements = []
    source_text_map = {}
    
    # Process requirements in batches of 5
    batch_size = 5
    requirement_batches = [extracted_reqs.requirements[i:i+batch_size] 
                          for i in range(0, len(extracted_reqs.requirements), batch_size)]
    
    for batch in requirement_batches:
        # Log the requirements being analyzed
        for req in batch:
            logger.info(f"🔍 Analyzing requirement {req.id} - {req.description}")
            
        # Analyze the batch
        batch_analyses = analyze_requirements_batch(batch, software_agent)
        
        # Process the results
        for req in batch:
            software_analysis = batch_analyses.get(req.id)
            
            if not software_analysis:
                logger.warning(f"⚠️ No analysis found for {req.id}, using default")
                software_analysis = SoftwareRequirementAnalysis(
                    is_software_related=False,
                    confidence=0.0,
                    rationale="Analysis not provided"
                )
            
            # Store the analyzed requirement
            analyzed_req = {
                "id": req.id,
                "document_id": document_id,
                "page_number": page_number,
                "section": req.section,
                "title": req.title,
                "description": req.description,
                "requirement_rationale": req.rationale,
                "software_analysis": software_analysis
            }
            analyzed_requirements.append(analyzed_req)
            
            # Store source text mapping
            source_text_map[req.id] = md_content
            
            logger.info(f"  ✅ Requirement {req.id}: {'Software-related' if software_analysis.is_software_related else 'Not software-related'} (Confidence: {software_analysis.confidence:.2f})")
    
    logger.info(f"✅ Processed {len(analyzed_requirements)} requirements from {document_id} page {page_number}")
    return analyzed_requirements, source_text_map

def main():
    """Main function to extract requirements from LanceDB and save them."""
    start_time = time.time()
    
    try:
        # Connect to LanceDB
        db = connect_to_lancedb()
        
        # Open source table
        if SOURCE_TABLE_NAME not in db.table_names():
            logger.error(f"❌ Source table '{SOURCE_TABLE_NAME}' not found in LanceDB")
            return
        
        source_table = db.open_table(SOURCE_TABLE_NAME)
        
        # Get or create target table
        target_table = get_or_create_requirements_table(db)
        
        # Load and initialize embedding model - reuse the same one from source table
        try:
            from sentence_transformers import SentenceTransformer
            embedding_model_name = "intfloat/multilingual-e5-large-instruct"
            logger.info(f"🔄 Loading embedding model: {embedding_model_name}")
            embedder = SentenceTransformer(embedding_model_name)
        except Exception as e:
            logger.error(f"❌ Error loading embedding model: {e}")
            return
        
        # Fetch data from source table
        source_data = source_table.to_pandas()
        
        # Filter the data if TARGET_DOCUMENT is specified
        if TARGET_DOCUMENT:
            logger.info(f"📌 Filtering documents: Only processing '{TARGET_DOCUMENT}'")
            source_data = source_data[source_data['pdf_identifier'] == TARGET_DOCUMENT]
            logger.info(f"📊 Found {len(source_data)} pages for '{TARGET_DOCUMENT}'")
        
        # Check if we have data
        if len(source_data) == 0:
            logger.warning("⚠️ No data found in source table after filtering")
            return
        
        logger.info(f"✅ Loaded {len(source_data)} document pages from source table")
        
        # Process each document page
        all_requirements = []
        all_source_text_maps = {}
        all_embedding_maps = {}
        
        for idx, (_, row) in enumerate(source_data.iterrows(), 1):
            page_data = row.to_dict()
            logger.info(f"🔄 Processing document {page_data['pdf_identifier']} page {page_data.get('page_number', 'unknown')}")
            
            # Extract and analyze requirements
            requirements, source_text_map = process_document_page(page_data, embedder)
            
            # Create embeddings for each requirement
            embedding_map = {}
            for req in requirements:
                # Use the instruction format with "passage:" prefix for embedding
                text_to_embed = f"passage: {req['description']}"
                embedding = embedder.encode(text_to_embed).tolist()
                embedding_map[req['id']] = embedding
            
            # Store requirements and mappings
            all_requirements.extend(requirements)
            all_source_text_maps.update(source_text_map)
            all_embedding_maps.update(embedding_map)
            
            # Log progress - fix the counter to show current/total correctly
            logger.info(f"✅ Processed {idx}/{len(source_data)} pages. Total requirements so far: {len(all_requirements)}")
        
        # Save all requirements to LanceDB
        save_requirements_to_lancedb(all_requirements, all_source_text_maps, all_embedding_maps, target_table)
        
        # Create index on embedding column if enough data is available
        if len(all_requirements) >= 256:
            logger.info("🔄 Creating vector index...")
            target_table.create_index(vector_column_name="embedding", metric="cosine", replace=True)
        else:
            logger.info(f"ℹ️ Skipping vector index creation - not enough records (need at least 256, got {len(all_requirements)})")
        
        # Log completion information
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        logger.info("\n--- SUMMARY ---")
        logger.info(f"✅ Successfully processed {len(source_data)} document pages")
        logger.info(f"✅ Extracted {len(all_requirements)} requirements")
        software_count = sum(1 for req in all_requirements if req['software_analysis'].is_software_related)
        percentage = (software_count / len(all_requirements) * 100) if len(all_requirements) > 0 else 0
        logger.info(f"✅ Software-related requirements: {software_count} ({percentage:.1f}%)")
        logger.info(f"✅ Total processing time: {elapsed_time:.2f} seconds")
        
        # Log token usage using utility function
        print_token_usage("gpt-4o-mini")
        
    except Exception as e:
        logger.error(f"❌ An error occurred: {e}")


if __name__ == "__main__":
    main()