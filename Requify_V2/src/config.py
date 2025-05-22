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