"""
Configuration settings for the Requify system.

This module centralizes all configuration settings using hardcoded values.
Only API keys are still loaded from environment variables for security.
"""

import os
from dotenv import load_dotenv

# Load environment variables (only needed for API keys)
load_dotenv()

# Model Provider Configuration
MODEL_PROVIDER = "openai"  # Options: "openai", "groq"

# Embedding Model Configuration
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
EMBEDDING_DIMENSION = 1024  # Dimension for e5-large models
EMBEDDING_DEVICE = "cpu"  # Force CPU usage to avoid memory issues
EMBEDDING_MAX_SEQ_LENGTH = 256  # Reduce sequence length to save memory
EMBEDDING_BATCH_SIZE = 8  # Control batch size for memory management

# LLM Model Configuration by Provider and Task
MODELS = {
    "openai": {
        "pdf_parsing": {
            "vision": "gpt-4.1-mini",
            "text": "gpt-4.1-mini"
        },
        "chunking": "gpt-4.1-mini",
        "requirement_extraction": "gpt-4.1-mini",
        "deduplication": "gpt-4.1-mini",
        "default": "gpt-4.1-mini"
    },
    "groq": {
        "pdf_parsing": {
            "vision": "meta-llama/llama-4-scout-17b-16e-instruct",
            "text": "llama-3.3-70b-versatile"
        },
        "chunking": "llama-3.3-70b-versatile",
        "requirement_extraction": "llama-3.3-70b-versatile",
        "deduplication": "llama-3.3-70b-versatile",
        "default": "llama-3.3-70b-versatile"
    }
}

# Model Pricing, Energy and Emissions Estimates (consolidated)
MODEL_PRICING = {
    "gpt-4o": {
        "input": 2.50,     # $2.50 per million input tokens
        "output": 10.00,   # $10 per million output tokens
        "energy": 0.0015,  # Wh per token
        "co2": 0.0008,     # gCO₂ per token
        "tier": "high"     # high or low tier for rate limits
    },
    "gpt-4o-mini": {
        "input": 0.15,     # $0.15 per million input tokens
        "output": 0.60,    # $0.60 per million output tokens
        "energy": 0.0008,  # Wh per token (smaller model = less energy)
        "co2": 0.0004,     # gCO₂ per token (smaller model = less emissions)
        "tier": "low"      # high or low tier for rate limits
    },
    "gpt-4.1": {
        "input": 3.00,     # $3.00 per million input tokens
        "output": 12.00,   # $12.00 per million output tokens
        "energy": 0.0016,  # Wh per token (newest advanced model)
        "co2": 0.0009,     # gCO₂ per token
        "tier": "high"     # high tier for rate limits
    },
    "gpt-4.1-mini": {
        "input": 0.20,     # $0.20 per million input tokens
        "output": 0.80,    # $0.80 per million output tokens
        "energy": 0.0009,  # Wh per token
        "co2": 0.0005,     # gCO₂ per token
        "tier": "low"      # low tier for rate limits
    },
    "o1": {
        "input": 15.00,    # $15 per million input tokens
        "output": 60.00,   # $60 per million output tokens
        "energy": 0.0018,  # Wh per token
        "co2": 0.0010,     # gCO₂ per token
        "tier": "high"     # high tier for rate limits
    },
    "o1-mini": {
        "input": 3.00,     # $3 per million input tokens
        "output": 12.00,   # $12 per million output tokens
        "energy": 0.0012,  # Wh per token
        "co2": 0.0006,     # gCO₂ per token
        "tier": "low"      # low tier for rate limits
    },
    "o3": {
        "input": 15.00,    # $15 per million input tokens
        "output": 75.00,   # $75 per million output tokens
        "energy": 0.0020,  # Wh per token
        "co2": 0.0011,     # gCO₂ per token
        "tier": "high"     # high tier for rate limits
    },
    "o3-mini": {
        "input": 3.50,     # $3.50 per million input tokens
        "output": 14.00,   # $14.00 per million output tokens
        "energy": 0.0013,  # Wh per token
        "co2": 0.0007,     # gCO₂ per token
        "tier": "low"      # low tier for rate limits
    },
    "llama-3.3-70b-versatile": {
        "input": 0.10,     # $0.10 per million input tokens (Groq pricing)
        "output": 0.30,    # $0.30 per million output tokens (Groq pricing)
        "energy": 0.0010,  # Wh per token (estimated)
        "co2": 0.0005,     # gCO₂ per token (estimated)
        "tier": "low"      # Approximate equivalent tier
    },
    "meta-llama/llama-4-scout-17b-16e-instruct": {
        "input": 0.40,     # $0.40 per million input tokens (estimated)
        "output": 0.80,    # $0.80 per million output tokens (estimated)
        "energy": 0.0012,  # Wh per token (estimated)
        "co2": 0.0006,     # gCO₂ per token (estimated)
        "tier": "low"      # Approximate equivalent tier
    }
}

# Group models by tier for tracking quota limits
MODEL_TIERS = {
    "high_tier": {
        "limit": 250000,  # 250K tokens per day
        "models": ["gpt-4.5-preview", "gpt-4.1", "gpt-4o", "o1", "o3"]
    },
    "low_tier": {
        "limit": 2500000,  # 2.5M tokens per day
        "models": ["gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o-mini", "o1-mini", "o3-mini", "o4-mini", "codex-mini-latest", 
                  "llama-3.3-70b-versatile", "meta-llama/llama-4-scout-17b-16e-instruct"]
    }
}

# Helper functions for model selection
def get_model_for_task(task, sub_task=None):
    """Get the appropriate model ID for a specific task based on current provider.
    
    Args:
        task: The pipeline task (e.g., 'pdf_parsing', 'chunking', 'requirement_extraction')
        sub_task: Optional sub-task (e.g., 'vision', 'text')
    
    Returns:
        The model ID to use
    """
    provider_models = MODELS.get(MODEL_PROVIDER, MODELS["openai"])
    
    if task not in provider_models:
        return provider_models["default"]
    
    if sub_task and isinstance(provider_models[task], dict):
        return provider_models[task].get(sub_task, provider_models["default"])
    
    return provider_models[task]

# Logging Configuration
VERBOSE_PDF_PARSING_OUTPUT = True  # Controls detailed logging during PDF parsing
VERBOSE_CHUNKING_OUTPUT = True     # Controls detailed logging during document chunking
LOG_LEVEL_CONSOLE = "INFO"         # Console log level
LOG_LEVEL_FILE = "DEBUG"           # File log level
LOG_FILE_PATH = "logs/requify_agent.log"  # Where to save log files

# Path Configuration
RAW_INPUT_DIR = os.path.join("input", "raw")
OUTPUT_DIR_BASE = os.path.join("output")

# PDF Processing Configuration
PDF_TO_IMAGE_DPI = 300

# API keys - still loaded from environment variables for security
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -------------------------------------------------------------------------------------
# Agent Prompts and Descriptions
# -------------------------------------------------------------------------------------

# PDF Parsing Agent Descriptions
PDF_PLAIN_AGENT_DESCRIPTION = (
    "You are an agent that extracts contents from an image that originated from a pdf file. "
    "You want to extract as much information as possible - not only focussing on text but also images, graphs, diagrams, etc. "
    "You are very technical and precise."
)

PDF_STRUCTURED_AGENT_DESCRIPTION = (
    "You are an agent that receives plain markdown text extracted from a PDF page. "
    "Your sole task is to generate and return a valid JSON object containing the following fields: "
    "'document_title' (use the provided document title), "
    "'summary' (a concise summary of the page content, 200-300 characters), "
    "and 'hashtags' (a list of approximately 10 key search words for the content, without the '#' symbol). "
    "Ensure your output is ONLY this JSON object and nothing else. Do not include any explanatory text before or after the JSON."
)

PDF_DOCUMENT_TITLE_AGENT_DESCRIPTION = (
    "You are an agent that receives combined markdown text from the first pages of a PDF document and generates "
    "a concise, descriptive title that captures the main subject and purpose of the document. "
    "The title should be between 5-10 words and clearly identify what the document is about."
    "Give back only the the title, NOTHING ELSE."
)

# Chunking Agent Prompts
STANDARD_CHUNKING_PROMPT = """
Split this text into chunks of {target_size} to {max_size} characters.

STRICT REQUIREMENTS:
1. ONLY split the text into chunks. NEVER modify, rephrase, paraphrase, summarize, or omit any content. Each chunk must contain the original text exactly as it appears in the input.
2. NEVER exceed {max_size} characters per chunk
3. Target {target_size} characters per chunk
4. Break at sentence boundaries, NEVER mid-sentence
5. Split at paragraph or section boundaries when possible
6. Break text at headers when available
7. ALWAYS create multiple chunks for text longer than {target_size} characters
8. Break large sections rather than creating oversized chunks

Format response as: {{"chunks": ["chunk1", "chunk2", ...]}}
"""

CONTEXT_AWARE_CHUNKING_PROMPT = """
Chunk the NEW DOCUMENT TEXT to align with the REFERENCE CHUNKS.

STRICT REQUIREMENTS:
1. ONLY split the text into chunks. NEVER modify, rephrase, paraphrase, summarize, or omit any content. Each chunk must contain the original text exactly as it appears in the input.
2. NEVER exceed {max_size} characters per chunk
3. Target {target_size} characters per chunk
4. Break at sentence boundaries, NEVER mid-sentence
5. Break at paragraph boundaries when possible
6. Break large sections rather than creating oversized chunks
7. Preserve headers with their content when possible

CRITICAL FOR IDENTICAL AND REORDERED CONTENT:
- Your PRIMARY GOAL is to replicate the chunk boundaries from a REFERENCE CHUNK if a section of the NEW DOCUMENT TEXT is IDENTICAL to that REFERENCE CHUNK's text.
- If a section of the NEW DOCUMENT TEXT exactly matches the text of a REFERENCE CHUNK, you MUST create a new chunk with the EXACT SAME text and boundaries.
- For sections of the NEW DOCUMENT TEXT that are reordered but still match a REFERENCE CHUNK, preserve the content and try to maintain similar chunking.
- Focus on semantic meaning but prioritize exact textual matches for boundary replication.
- Compare all reference chunks to find the best match for each section of the NEW DOCUMENT TEXT.

ALWAYS output multiple chunks for text longer than {target_size} characters.

Format response as: {{"chunks": ["chunk1", "chunk2", ...]}}
"""

CHUNK_COMPARISON_PROMPT = """
# Chunk Comparison

Compare the following two chunks of text, which are from two different versions of the same document, and decide which to keep.

## CHUNK FROM ORIGINAL DOCUMENT:
{old_chunk}

## CHUNK FROM NEW DOCUMENT:
{new_chunk}

You must analyze the chunks carefully to determine if they contain the same information or if one contains important new or different information that should be kept.

Pay very close attention to numerical values, specifications, measurements, dates, requirements, and other concrete details that might have changed.

Step 1: First, list EXACTLY what has changed between the two chunks in a detailed list format.
Step 2: Then make your decision based on those differences.

Provide your decision in JSON format with the following structure:
```json
{{
  "decision": "keep_old | keep_new | need_user_input",
  "reason": "detailed explanation of your reasoning, pointing out specific differences",
  "differences": ["list at least 3 specific, concrete differences between the chunks", "be very precise about what changed"]
}}
```

Decision choices:
- "keep_old": The original document's chunk is better or the changes in the new chunk are insignificant.
- "keep_new": The new document's chunk contains meaningful new or updated information.
- "need_user_input": You're genuinely uncertain which chunk is better and need human judgment.

IMPORTANT: You MUST include explicit concrete differences between the chunks. For example:
- "Weight changed from 300kg to 350kg"
- "Added new safety information in paragraph 2"
- "Reordered sections without changing content"
- "Added specifications for extreme temperatures"

Remember: The differences field is REQUIRED and must contain SPECIFIC, CONCRETE differences.
"""

# Requirements Extraction Prompts
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

# Agent Descriptions
REQUIREMENTS_AGENT_DESCRIPTION = """You are a specialized agent for extracting atomic requirements from technical documents.
You are good at making sure to take the entire input into account and do not stop after a few examples.
You focus only on technical product requirements!"""

SOFTWARE_AGENT_DESCRIPTION = """You are a specialized agent for determining if requirements are related to software.
You have expertise in software systems and understanding how requirements translate to software implementations."""

# -------------------------------------------------------------------------------------
# Chunking Configuration
# -------------------------------------------------------------------------------------

# Character size limits for chunking
MAX_CHAR_SIZE = 900         # Maximum allowed character size for a chunk
TARGET_CHAR_SIZE = 700      # Target character size per chunk
MAX_SECTION_SIZE = 30000    # Maximum section size for processing with LLM

# Similarity thresholds for chunk comparison
SIMILARITY_THRESHOLD = 0.75    # Threshold for similar chunks (lowered to be more sensitive to reordered content)
DUPLICATE_THRESHOLD = 0.995    # High threshold for automatic duplicates without LLM

# -------------------------------------------------------------------------------------
# LanceDB Configuration
# -------------------------------------------------------------------------------------

# LanceDB subdirectory and table names
LANCEDB_SUBDIR_NAME = "lancedb"
DOCUMENT_CHUNKS_TABLE = "document_chunks"
DOCUMENTS_TABLE = "documents" 
REQUIREMENTS_TABLE = "requirements"

# -------------------------------------------------------------------------------------
# API and Retry Configuration
# -------------------------------------------------------------------------------------

# Retry settings for API calls
MAX_RETRIES = 3             # Maximum number of retries for API calls
INITIAL_RETRY_DELAY = 2     # Initial delay (in seconds) before retrying

# -------------------------------------------------------------------------------------
# Directory Structure Configuration 
# -------------------------------------------------------------------------------------

# Additional directory paths
LOGS_DIR = os.path.join("logs")
TEMP_DIR = os.path.join("temp")
PDF_IMAGES_DIR = os.path.join(OUTPUT_DIR_BASE, "pdf_images")
PARSED_CONTENT_DIR = os.path.join(OUTPUT_DIR_BASE, "parsed_content")
TEST_RESULTS_DIR = os.path.join(OUTPUT_DIR_BASE, "test_results")
TOKEN_TRACKING_DIR = os.path.join(OUTPUT_DIR_BASE, "token_tracking")
VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR_BASE, "visualizations")
CHUNK_DIAGNOSTICS_DIR = os.path.join(TEMP_DIR, "chunk_diagnostics") 