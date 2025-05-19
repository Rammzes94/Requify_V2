"""
Configuration settings for the Requify system.

This module centralizes all configuration settings using hardcoded values.
Only API keys are still loaded from environment variables for security.
"""

import os
from dotenv import load_dotenv

# Load environment variables (only needed for API keys)
load_dotenv()

# Model Selection Configuration
MODEL_PROVIDER = "openai"  # Options: "openai", "groq"

# Logging Configuration
VERBOSE_PDF_PARSING_OUTPUT = True
VERBOSE_CHUNKING_OUTPUT = True
LOG_LEVEL = "INFO"
LOG_TO_CONSOLE = True
LOG_TO_FILE = False
LOG_FILE_PATH = "logs/requify_agent.log"

# Path Configuration
RAW_INPUT_DIR = os.path.join("_01_input", "raw")
OUTPUT_DIR_BASE = os.path.join("_03_output")

# PDF Processing Configuration
PDF_TO_IMAGE_DPI = 300

# API keys - still loaded from environment variables for security
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY") 