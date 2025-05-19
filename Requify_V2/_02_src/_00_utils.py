import os
import sys
import pathlib
import pdfplumber
import re
import logging
import time
from datetime import datetime, date
import json
import csv
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import numpy as np


# Initialize token counters as global variables
total_input_tokens = 0
total_output_tokens = 0
model_token_usage = defaultdict(lambda: {"input": 0, "output": 0})

# Model pricing constants
MODEL_PRICING = {
    "gpt-4o": {
        "input": 2.50,    # $2.50 per million input tokens
        "output": 10.00,  # $10 per million output tokens
        "energy": 0.00125, # Wh per token
        "co2": 0.0006     # gCO₂ per token
    },
    "gpt-4o-mini": {
        "input": 0.15,    # $0.15 per million input tokens
        "output": 0.60,   # $0.60 per million output tokens
        "energy": 0.0008, # Wh per token (smaller model = less energy)
        "co2": 0.0004     # gCO₂ per token (smaller model = less emissions)
    },
    "o1": {
        "input": 15.00,   # $15 per million input tokens
        "output": 60.00,  # $60 per million output tokens
        "energy": 0.0015, # Wh per token (more advanced model = more energy)
        "co2": 0.0008     # gCO₂ per token (more advanced model = more emissions)
    }
}

# OpenAI token usage limits (daily)
OPENAI_TOKEN_LIMITS = {
    "high_tier": {
        "limit": 250000,  # 250K tokens per day
        "models": ["gpt-4.5-preview", "gpt-4.1", "gpt-4o", "o1", "o3"]
    },
    "low_tier": {
        "limit": 2500000,  # 2.5M tokens per day
        "models": ["gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o-mini", "o1-mini", "o3-mini", "o4-mini", "codex-mini-latest"]
    }
}

# Token tracking file paths
TOKEN_TRACKING_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_03_output", "token_tracking")
DAILY_USAGE_FILE = os.path.join(TOKEN_TRACKING_DIR, "daily_token_usage.csv")
MODEL_USAGE_FILE = os.path.join(TOKEN_TRACKING_DIR, "model_token_usage.json")
REPORTS_DIR = os.path.join(TOKEN_TRACKING_DIR, "reports")

# Ensure directories exist
os.makedirs(TOKEN_TRACKING_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def update_token_counters(response, model_id="gpt-4o-mini"):
    """
    Update token counters based on API response metrics.
    
    Args:
        response: The response object from an agent.run() call
        model_id: The model ID used for the request
    """
    global total_input_tokens, total_output_tokens, model_token_usage
    
    # Ensure token tracking data is loaded
    if not model_token_usage:
        load_token_tracking_data()
    
    # Update the global token counters
    if hasattr(response, 'metrics'):
        metrics = response.metrics
        input_tokens_added = 0
        output_tokens_added = 0
        
        if 'input_tokens' in metrics and isinstance(metrics['input_tokens'], list) and metrics['input_tokens']:
            input_tokens_added = metrics['input_tokens'][0]
            total_input_tokens += input_tokens_added
            
        if 'output_tokens' in metrics and isinstance(metrics['output_tokens'], list) and metrics['output_tokens']:
            output_tokens_added = metrics['output_tokens'][0]
            total_output_tokens += output_tokens_added
        
        # Update model-specific tracking
        if model_id in model_token_usage:
            model_token_usage[model_id]["input"] += input_tokens_added
            model_token_usage[model_id]["output"] += output_tokens_added
        else:
            model_token_usage[model_id] = {
                "input": input_tokens_added,
                "output": output_tokens_added
            }
        
        # Try to get the global logger, but handle case where it's not configured
        try:
            logger = logging.getLogger()
            if logger.handlers:  # Check if logger is configured
                logger.info(f"Token counters updated: {total_input_tokens} input, {total_output_tokens} output", extra={"icon": "🔢"})
            else:
                print(f"Token counters updated: {total_input_tokens} input, {total_output_tokens} output")
        except Exception:
            # If there's any issue with logging, fall back to print
            print(f"Token counters updated: {total_input_tokens} input, {total_output_tokens} output")
        
        # Save updated token usage to file
        save_token_usage(model_id)

def get_token_usage():
    """
    Get the current token usage statistics.
    
    Returns:
        dict: Dictionary containing token usage information
    """
    # Ensure token tracking data is loaded
    if not model_token_usage:
        load_token_tracking_data()
    
    limits = check_token_limits()
    
    return {
        "session": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens
        },
        "models": dict(model_token_usage),
        "limits": limits
    }

def print_token_usage(model_id="gpt-4o-mini"):
    """
    Print a summary of token usage, estimated cost, energy usage, and CO2 emissions.
    Also generates a usage report if appropriate.
    
    Args:
        model_id (str): The model ID to use for pricing calculation
    """
    # Ensure token tracking data is loaded
    if not model_token_usage:
        load_token_tracking_data()
    
    if model_id not in MODEL_PRICING:
        # Try to get the global logger, but handle case where it's not configured
        try:
            logger = logging.getLogger()
            if logger.handlers:  # Check if logger is configured
                logger.warning(f"Unknown model '{model_id}'. Defaulting to gpt-4o-mini pricing.", extra={"icon": "⚠️"})
            else:
                print(f"Warning: Unknown model '{model_id}'. Defaulting to gpt-4o-mini pricing.")
        except Exception:
            # If there's any issue with logging, fall back to print
            print(f"Warning: Unknown model '{model_id}'. Defaulting to gpt-4o-mini pricing.")
        model_id = "gpt-4o-mini"
        
    pricing = MODEL_PRICING[model_id]
    
    # Calculate cost
    input_cost = (total_input_tokens / 1000000) * pricing["input"]
    output_cost = (total_output_tokens / 1000000) * pricing["output"]
    estimated_cost = input_cost + output_cost
    
    # Calculate energy and CO2 based on model-specific values
    total_tokens = total_input_tokens + total_output_tokens
    energy_usage = total_tokens * pricing["energy"]
    co2_emissions = total_tokens * pricing["co2"]
    
    # Check token limits
    limits = check_token_limits()
    
    # Try to get the global logger, but handle case where it's not configured
    try:
        logger = logging.getLogger()
        if logger.handlers:  # Check if logger is configured
            logger.info(f"\nToken usage for {model_id}:", extra={"icon": "📊"})
            logger.info(f"  - Input tokens: {total_input_tokens}", extra={"icon": "📊"})
            logger.info(f"  - Output tokens: {total_output_tokens}", extra={"icon": "📊"})
            logger.info(f"  - Total tokens: {total_tokens}", extra={"icon": "📊"})
            logger.info(f"  - Estimated cost: ${estimated_cost:.4f}", extra={"icon": "💰"})
            logger.info(f"  - Energy consumption: {energy_usage:.6f} Wh ({energy_usage/1000:.6f} kWh)", extra={"icon": "⚡"})
            logger.info(f"  - CO₂ emissions: {co2_emissions:.6f} gCO₂ ({co2_emissions/1000:.6f} kgCO₂)", extra={"icon": "🌱"})
            
            # Log limit status
            tier = "high_tier" if model_id in OPENAI_TOKEN_LIMITS["high_tier"]["models"] else "low_tier"
            logger.info(f"  - Daily limit: {limits[tier]['usage']} / {limits[tier]['limit']} tokens ({limits[tier]['percentage']:.1f}%)", extra={"icon": "⚠️" if limits[tier]['warning'] else "✅"})
        else:
            print(f"\nToken usage for {model_id}:")
            print(f"  - Input tokens: {total_input_tokens}")
            print(f"  - Output tokens: {total_output_tokens}")
            print(f"  - Total tokens: {total_tokens}")
            print(f"  - Estimated cost: ${estimated_cost:.4f}")
            print(f"  - Energy consumption: {energy_usage:.6f} Wh ({energy_usage/1000:.6f} kWh)")
            print(f"  - CO₂ emissions: {co2_emissions:.6f} gCO₂ ({co2_emissions/1000:.6f} kgCO₂)")
    except Exception:
        # If there's any issue with logging, fall back to print
        print(f"\nToken usage for {model_id}:")
        print(f"  - Input tokens: {total_input_tokens}")
        print(f"  - Output tokens: {total_output_tokens}")
        print(f"  - Total tokens: {total_tokens}")
        print(f"  - Estimated cost: ${estimated_cost:.4f}")
        print(f"  - Energy consumption: {energy_usage:.6f} Wh ({energy_usage/1000:.6f} kWh)")
        print(f"  - CO₂ emissions: {co2_emissions:.6f} gCO₂ ({co2_emissions/1000:.6f} kgCO₂)")
    
    # Save token usage
    save_token_usage(model_id)
    
    # Generate report if needed
    if should_generate_report():
        generate_token_usage_report()

def reset_token_counters():
    """Reset the token counters to zero for the current session only."""
    global total_input_tokens, total_output_tokens
    total_input_tokens = 0
    total_output_tokens = 0
    
    logger = logging.getLogger()
    if logger.handlers:
        logger.info("Token counters reset for current session", extra={"icon": "🔄"})
    else:
        print("Token counters reset for current session")

def setup_project_directory():
    """
    Smart directory handling to ensure consistent working directory.
    
    In interactive mode (Jupyter/IPython), changes to the project root folder 
    (identified by presence of .env, venv, myenv, .gitignore, or requirements.txt).
    In script mode, maintains normal execution behavior.
    
    Returns:
        str: The current working directory after any necessary adjustments
    """
    
    # Check if we're running in interactive mode (like IPython or Jupyter)
    is_interactive = not hasattr(sys, 'ps1') and sys.argv[0] == '' or 'ipykernel' in sys.modules

    # Handle interactive mode 
    if is_interactive:
        current_dir = os.getcwd()
        
        # List of files/folders that indicate project root
        root_indicators = ['.env', 'venv', 'myenv', '.gitignore', 'requirements.txt']
        
        # Go up directory levels until we find root indicators
        test_dir = current_dir
        while test_dir != os.path.dirname(test_dir):  # Stop at filesystem root
            # Check if any root indicators exist in this directory
            if any(os.path.exists(os.path.join(test_dir, indicator)) for indicator in root_indicators):
                # Found the project root
                if test_dir != current_dir:
                    os.chdir(test_dir)
                    # Use the logger if it exists
                    logger = logging.getLogger()
                    if logger.handlers:
                        logger.info(f"Interactive mode: Changed working directory to: {test_dir}", extra={"icon": "📂"})
                    else:
                        print(f"Interactive mode: Changed working directory to: {test_dir}")
                else:
                    # Use the logger if it exists
                    logger = logging.getLogger()
                    if logger.handlers:
                        logger.info(f"Interactive mode: Already in project root: {test_dir}", extra={"icon": "📂"})
                    else:
                        print(f"Interactive mode: Already in project root: {test_dir}")
                break
            
            # Move up one directory level
            test_dir = os.path.dirname(test_dir)
            
            # If we've reached the filesystem root without finding indicators
            if test_dir == os.path.dirname(test_dir):
                # Use the logger if it exists
                logger = logging.getLogger()
                if logger.handlers:
                    logger.warning(f"Interactive mode: Could not find project root. Staying in: {current_dir}", extra={"icon": "⚠️"})
                else:
                    print(f"Interactive mode: Could not find project root. Staying in: {current_dir}")
                break
    else:
        # For normal script execution, do nothing special
        # Comment out the print statement as it's unnecessary logging
        # print(f"Running script from: {os.getcwd()}")
        pass
    return os.getcwd()


# -------------------------------------------------------------------------------------
# [Logging Setup]
# -------------------------------------------------------------------------------------

# Filter to ensure 'icon' is present in all log records
class EnsureIconFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'icon'):
            # Module-specific default icons
            module_icons = {
                'pipeline_controller': '🚧',
                'pipeline_runner': '🚀',
                'stable_pdf_parsing': '📄',
                'stable_excel_parsing': '📊',
                'context_aware_chunking': '🧩',
                'extract_requirements': '📝',
                'pre_save_deduplication': '♻️',
                'pipeline_interaction': '🔁',
                'file_hash_deduplication': '🔐',
                'lancedb': '🛢️',
                'init_lancedb': '🏗️',
                'reset_lancedb': '🔄',
                'test_document_diff': '⚖️',
                'util_lancedb_viewer': '👁️'
            }
            
            # Try to get the module name from the record
            module_name = record.module.lower() if hasattr(record, 'module') else ''
            
            # Look for a module-specific icon
            for key, icon in module_icons.items():
                if key in module_name:
                    record.icon = icon
                    return True
            
            # If no module-specific icon found, use level-based default
            level_name = record.levelname
            if level_name == 'INFO':
                record.icon = 'ℹ️'
            elif level_name == 'WARNING':
                record.icon = '⚠️'
            elif level_name == 'ERROR':
                record.icon = '❌'
            elif level_name == 'CRITICAL':
                record.icon = '💥'
            elif level_name == 'DEBUG':
                record.icon = '🐞'
            else:
                record.icon = '📌' # A more noticeable default
        return True

# Custom adapter to add icons to log messages if not already present
class IconAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        # Ensure 'extra' dictionary exists and has an 'icon' key
        extra = kwargs.get('extra', {})
        
        # Determine icon based on log level and message content if not provided
        if 'icon' not in extra:
            # Get level name from level number if available, otherwise from logger's effective level
            level_name = logging.getLevelName(kwargs.get('levelno', self.logger.getEffectiveLevel()))

            # Base level icons
            level_icons = {
                'DEBUG': '🐞',
                'INFO': 'ℹ️',
                'WARNING': '⚠️',
                'ERROR': '❌',
                'CRITICAL': '💥'
            }
            
            # Context-specific icons based on message content
            message_lower = msg.lower()

            # Aviation/Aerospace Themed Icons
            if "fighter" in message_lower or "jet" in message_lower:
                extra['icon'] = '✈️'
            elif "rocket" in message_lower or "launch" in message_lower:
                extra['icon'] = '🚀'
            elif "missile" in message_lower:
                extra['icon'] = '🎯'
            elif "radar" in message_lower:
                extra['icon'] = '📡'
            elif "aircraft" in message_lower or "plane" in message_lower:
                extra['icon'] = '✈️'
            elif "pilot" in message_lower or "flight" in message_lower:
                extra['icon'] = '👨‍✈️'
            elif "navigation" in message_lower:
                extra['icon'] = '🧭'
            elif "altitude" in message_lower:
                extra['icon'] = '🏔️'
            elif "specification" in message_lower:
                extra['icon'] = '📋'
            # Database related
            elif "database" in message_lower or "db" in message_lower:
                if "connect" in message_lower or "connecting" in message_lower:
                    extra['icon'] = '🔌'
                elif "query" in message_lower or "searching" in message_lower:
                    extra['icon'] = '🔍'
                elif "table" in message_lower:
                    if "create" in message_lower:
                        extra['icon'] = '📝'
                    elif "drop" in message_lower:
                        extra['icon'] = '🗑️'
                    else:
                        extra['icon'] = '📋'
                elif "index" in message_lower:
                    extra['icon'] = '📇'
                elif "backup" in message_lower:
                    extra['icon'] = '💾'
                elif "restore" in message_lower:
                    extra['icon'] = '📤'
                else:
                    extra['icon'] = '🛢️'
                    
            # File operations
            elif "file" in message_lower:
                if "read" in message_lower or "loading" in message_lower or "open" in message_lower:
                    extra['icon'] = '📂'
                elif "write" in message_lower or "saving" in message_lower:
                    extra['icon'] = '📥'
                elif "delete" in message_lower or "remove" in message_lower:
                    extra['icon'] = '🗑️'
                elif "duplicate" in message_lower:
                    extra['icon'] = '🔄'
                elif "hash" in message_lower:
                    extra['icon'] = '🔐'
                else:
                    extra['icon'] = '📄'
                    
            # Processing status
            elif "start" in message_lower or "begin" in message_lower or "initializing" in message_lower:
                extra['icon'] = '🚀'
            elif "process" in message_lower or "running" in message_lower or "executing" in message_lower:
                extra['icon'] = '⚙️'
            elif "complete" in message_lower or "finish" in message_lower or "done" in message_lower or "success" in message_lower:
                extra['icon'] = '✅'
            elif "timeout" in message_lower or "too long" in message_lower:
                extra['icon'] = '⏱️'
            elif "retry" in message_lower:
                extra['icon'] = '🔁'
                
            # Document processing specific
            elif "pdf" in message_lower:
                extra['icon'] = '📑'
            elif "image" in message_lower or "png" in message_lower or "jpg" in message_lower:
                extra['icon'] = '🖼️'
            elif "embedding" in message_lower or "vector" in message_lower:
                extra['icon'] = '🧮'
            elif "chunk" in message_lower:
                if "duplicate" in message_lower:
                    extra['icon'] = '♻️'
                elif "similar" in message_lower:
                    extra['icon'] = '👯'
                else:
                    extra['icon'] = '🧩'
            elif "token" in message_lower:
                if "count" in message_lower:
                    extra['icon'] = '🔢'
                else:
                    extra['icon'] = '🎫'
                    
            # Pipeline stages
            elif "pipeline" in message_lower:
                if "step" in message_lower:
                    extra['icon'] = '📶'
                elif "error" in message_lower or "fail" in message_lower:
                    extra['icon'] = '📛'
                else:
                    extra['icon'] = '🔄'
            elif "extract" in message_lower and "requirement" in message_lower:
                extra['icon'] = '📋'
            elif "parse" in message_lower:
                extra['icon'] = '📝'
            elif "deduplication" in message_lower or "deduplicate" in message_lower:
                if "document" in message_lower:
                    extra['icon'] = '📚'
                else:
                    extra['icon'] = '🔍'
                    
            # AI/ML Related
            elif "model" in message_lower:
                if "load" in message_lower:
                    extra['icon'] = '🤖'
                elif "predict" in message_lower or "inference" in message_lower:
                    extra['icon'] = '🔮'
                elif "train" in message_lower:
                    extra['icon'] = '🏋️'
                else:
                    extra['icon'] = '🧠'
            elif "llm" in message_lower or "language model" in message_lower:
                extra['icon'] = '🤖'
            elif "gpt" in message_lower:
                extra['icon'] = '🧠'
            elif "openai" in message_lower:
                extra['icon'] = '🔮'
            elif "groq" in message_lower:
                extra['icon'] = '⚡'
            elif "anthropic" in message_lower or "claude" in message_lower:
                extra['icon'] = '🦜'
            elif "agent" in message_lower:
                extra['icon'] = '🕵️'
            elif "prompt" in message_lower:
                extra['icon'] = '💬'
            elif "ai" in message_lower:
                extra['icon'] = '🧠'
            elif "neural" in message_lower:
                extra['icon'] = '🔄'
            elif "transformer" in message_lower:
                extra['icon'] = '🧿'
                    
            # Statistics and metrics
            elif "stat" in message_lower or "metric" in message_lower:
                extra['icon'] = '📊'
            elif "summary" in message_lower:
                extra['icon'] = '📈'
            elif "report" in message_lower:
                extra['icon'] = '📜'
            elif "time" in message_lower or "duration" in message_lower:
                extra['icon'] = '⏱️'
            elif "progress" in message_lower:
                extra['icon'] = '⏳'
                
            # Memory management
            elif "memory" in message_lower:
                if "clean" in message_lower:
                    extra['icon'] = '🧹'
                elif "leak" in message_lower:
                    extra['icon'] = '💧'
                else:
                    extra['icon'] = '🧠'
            elif "cache" in message_lower:
                extra['icon'] = '💽'
                
            # Common actions
            elif "check" in message_lower or "verify" in message_lower:
                extra['icon'] = '✓'
            elif "compare" in message_lower:
                extra['icon'] = '⚖️'
            elif "detect" in message_lower:
                extra['icon'] = '🔎'
            elif "skip" in message_lower:
                extra['icon'] = '⏭️'
            elif "found" in message_lower:
                extra['icon'] = '🔍'
            elif "add" in message_lower:
                extra['icon'] = '➕'
            elif "remove" in message_lower:
                extra['icon'] = '➖'
            elif "update" in message_lower:
                extra['icon'] = '🔄'
            elif "change" in message_lower:
                extra['icon'] = '📝'
            elif "match" in message_lower:
                extra['icon'] = '🎯'
            elif "replace" in message_lower:
                extra['icon'] = '🔀'
            elif "merge" in message_lower:
                extra['icon'] = '🔄'
            
            # Security and protection
            elif "secure" in message_lower or "security" in message_lower:
                extra['icon'] = '🔒'
            elif "encrypt" in message_lower:
                extra['icon'] = '🔐'
            elif "authenticate" in message_lower or "auth" in message_lower:
                extra['icon'] = '🔑'
            elif "permission" in message_lower:
                extra['icon'] = '🛡️'
            elif "firewall" in message_lower:
                extra['icon'] = '🧱'
            elif "password" in message_lower:
                extra['icon'] = '🔐'
                
            # Hardware and infrastructure
            elif "server" in message_lower:
                extra['icon'] = '🖥️'
            elif "cpu" in message_lower or "processor" in message_lower:
                extra['icon'] = '⚡'
            elif "gpu" in message_lower:
                extra['icon'] = '🎮'
            elif "ram" in message_lower or "memory" in message_lower:
                extra['icon'] = '🧠'
            elif "disk" in message_lower or "storage" in message_lower:
                extra['icon'] = '💾'
            elif "network" in message_lower:
                extra['icon'] = '🌐'
            elif "cloud" in message_lower:
                extra['icon'] = '☁️'
                
            # Specialized content
            elif "requirement" in message_lower:
                extra['icon'] = '📋'
            elif "similarity" in message_lower:
                extra['icon'] = '🔄'
            elif "directory" in message_lower or "folder" in message_lower:
                extra['icon'] = '📁'
            elif "path" in message_lower:
                extra['icon'] = '🧭'
            elif "config" in message_lower or "settings" in message_lower:
                extra['icon'] = '⚙️'
            elif "batch" in message_lower:
                extra['icon'] = '📦'
            elif "document" in message_lower:
                extra['icon'] = '📄'
            elif "page" in message_lower:
                extra['icon'] = '📃'
            elif "version" in message_lower:
                extra['icon'] = '🏷️'
            elif "schema" in message_lower:
                extra['icon'] = '📐'
            elif "id" in message_lower or "identifier" in message_lower:
                extra['icon'] = '🔑'
            elif "create" in message_lower:
                extra['icon'] = '🆕'
            
            # If no specific icon found, use the default for the log level
            if 'icon' not in extra:
                extra['icon'] = level_icons.get(level_name, 'ℹ️')

        kwargs['extra'] = extra
        return msg, kwargs

# Centralized ScriptLogger for all modules to use
class ScriptLogger(logging.LoggerAdapter):
    """
    ScriptLogger adds a consistent prefix to all log messages.
    This is centralized here so all modules can use the same logger format.
    
    Usage:
        logger = ScriptLogger(setup_logging(), "[ModuleName] ")
    """
    def __init__(self, logger, prefix):
        super().__init__(logger, {})
        self.prefix = prefix
        
    def process(self, msg, kwargs):
        return f"{self.prefix}{msg}", kwargs

def setup_logging():
    """
    Configures logging for the project based on .env settings.
    Returns a logger instance with an IconAdapter.
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    # Make icons more prominent in log format by placing them at the start
    log_format = os.getenv("LOG_FORMAT", '%(asctime)s [%(levelname)s] %(icon)s (%(module)s:%(lineno)d) - %(message)s')
    log_date_format = os.getenv("LOG_DATE_FORMAT", '%Y-%m-%d %H:%M:%S')
    log_to_console = os.getenv("LOG_TO_CONSOLE", "True").lower() == "true"
    log_to_file = os.getenv("LOG_TO_FILE", "False").lower() == "true"
    log_file_path = os.getenv("LOG_FILE_PATH", "logs/requify_agent.log")
    log_file_mode = os.getenv("LOG_FILE_MODE", "a")

    handlers = []
    ensure_icon_filter = EnsureIconFilter() # Create an instance of the filter

    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.addFilter(ensure_icon_filter) # Add filter to handler
        handlers.append(console_handler)
    
    if log_to_file:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        file_handler = logging.FileHandler(log_file_path, mode=log_file_mode)
        file_handler.addFilter(ensure_icon_filter) # Add filter to handler
        handlers.append(file_handler)

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format=log_format,
        datefmt=log_date_format,
        handlers=handlers
    )
    
    # Suppress INFO logs from specific modules
    logging.getLogger("httpx._client").setLevel(logging.WARNING)
    for logger_name in logging.root.manager.loggerDict:
        if logger_name.startswith("agno") or logger_name.startswith("groq"):
            logging.getLogger(logger_name).setLevel(logging.WARNING)

    logger_name = os.path.splitext(os.path.basename(sys.argv[0]))[0] if sys.argv[0] else "interactive"
    base_logger = logging.getLogger(logger_name)
    icon_logger = IconAdapter(base_logger, {})
    return icon_logger

def get_logger(script_name):
    """
    Create a script-specific logger with a consistent prefix format.
    
    Args:
        script_name: A short name for the script or module
        
    Returns:
        A ScriptLogger that adds the script name as a prefix to all messages
    """
    base_logger = setup_logging()
    return ScriptLogger(base_logger, f"[{script_name}] ")

def generate_timestamp():
    """
    Generate a timestamp string in the format YYYY-MM-DD HH:MM:SS.
    
    Returns:
        str: Formatted timestamp
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def load_token_tracking_data():
    """
    Load token usage data from tracking files if they exist.
    Updates the global model_token_usage dictionary.
    """
    global model_token_usage
    today = date.today().isoformat()
    
    # Load model usage data if it exists
    if os.path.exists(MODEL_USAGE_FILE):
        try:
            with open(MODEL_USAGE_FILE, 'r') as f:
                loaded_data = json.load(f)
                # Convert to defaultdict
                model_token_usage = defaultdict(lambda: {"input": 0, "output": 0})
                for model, usage in loaded_data.items():
                    model_token_usage[model] = usage
            logger = logging.getLogger()
            if logger.handlers:
                logger.info(f"Loaded token usage data for {len(model_token_usage)} models", extra={"icon": "📊"})
        except Exception as e:
            logger = logging.getLogger()
            if logger.handlers:
                logger.warning(f"Failed to load token usage data: {str(e)}", extra={"icon": "⚠️"})
    
    # Create daily usage file if it doesn't exist
    if not os.path.exists(DAILY_USAGE_FILE):
        with open(DAILY_USAGE_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['date', 'model', 'input_tokens', 'output_tokens', 'total_tokens', 'cost'])

def save_token_usage(model_id="gpt-4o-mini"):
    """
    Save current token usage to tracking files and update daily usage.
    
    Args:
        model_id (str): The model ID to use for tracking
    """
    global model_token_usage, total_input_tokens, total_output_tokens
    today = date.today().isoformat()
    
    # Ensure token tracking data is loaded
    if not model_token_usage:
        load_token_tracking_data()
    
    # Update the model's token usage for this session
    input_tokens = total_input_tokens
    output_tokens = total_output_tokens
    
    if model_id in model_token_usage:
        model_token_usage[model_id]["input"] += input_tokens
        model_token_usage[model_id]["output"] += output_tokens
    else:
        model_token_usage[model_id] = {"input": input_tokens, "output": output_tokens}
    
    # Save updated model usage data
    try:
        with open(MODEL_USAGE_FILE, 'w') as f:
            json.dump(dict(model_token_usage), f, indent=2)
    except Exception as e:
        logger = logging.getLogger()
        if logger.handlers:
            logger.warning(f"Failed to save model token usage: {str(e)}", extra={"icon": "⚠️"})
    
    # Calculate cost
    pricing = MODEL_PRICING.get(model_id, MODEL_PRICING["gpt-4o-mini"])
    input_cost = (input_tokens / 1000000) * pricing["input"]
    output_cost = (output_tokens / 1000000) * pricing["output"]
    total_cost = input_cost + output_cost
    
    # Append to daily usage file
    try:
        # Check if we already have an entry for this model today using plain CSV reading
        updated_existing = False
        if os.path.exists(DAILY_USAGE_FILE):
            # Load the current CSV file
            rows = []
            with open(DAILY_USAGE_FILE, 'r', newline='') as f:
                reader = csv.reader(f)
                header = next(reader)  # Skip header
                for row in reader:
                    if len(row) >= 6:  # Ensure row has enough columns
                        row_date = row[0]
                        row_model = row[1]
                        if row_date == today and row_model == model_id:
                            # Update this row
                            row_input = int(row[2]) + input_tokens
                            row_output = int(row[3]) + output_tokens
                            row_total = row_input + row_output
                            row_cost = float(row[5]) + total_cost
                            rows.append([row_date, row_model, row_input, row_output, row_total, row_cost])
                            updated_existing = True
                        else:
                            rows.append(row)
                    else:
                        rows.append(row)  # Keep malformed rows
            
            if updated_existing:
                # Write back the updated CSV
                with open(DAILY_USAGE_FILE, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    writer.writerows(rows)
        
        if not updated_existing:
            # Append new entry
            with open(DAILY_USAGE_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([today, model_id, input_tokens, output_tokens, input_tokens + output_tokens, total_cost])
    except Exception as e:
        logger = logging.getLogger()
        if logger.handlers:
            logger.warning(f"Failed to update daily token usage: {str(e)}", extra={"icon": "⚠️"})
    
    # Check if we should generate a report
    should_generate = should_generate_report()
    if should_generate:
        try:
            # Try to import visualization packages at runtime only if needed
            import matplotlib.pyplot as plt
            import pandas as pd
            import numpy as np
            generate_token_usage_report()
        except ImportError:
            logger = logging.getLogger()
            if logger.handlers:
                logger.warning("Could not generate token usage report: matplotlib or pandas not available", extra={"icon": "⚠️"})

def should_generate_report():
    """
    Check if we should generate a new report based on time or usage patterns.
    For simplicity, we'll generate a report once per day when token usage is saved.
    
    Returns:
        bool: True if a report should be generated
    """
    today = date.today().isoformat()
    report_file = os.path.join(REPORTS_DIR, f"token_report_{today}.pdf")
    
    # Generate a report if we haven't generated one today
    return not os.path.exists(report_file)

def check_token_limits():
    """
    Check if token usage is approaching OpenAI limits.
    
    Returns:
        dict: Dictionary with limit status for high and low tier models
    """
    today = date.today().isoformat()
    high_tier_usage = 0
    low_tier_usage = 0
    
    if os.path.exists(DAILY_USAGE_FILE):
        try:
            with open(DAILY_USAGE_FILE, 'r', newline='') as f:
                reader = csv.reader(f)
                header = next(reader)  # Skip header
                for row in reader:
                    if len(row) >= 5:  # Ensure row has enough columns
                        row_date = row[0]
                        row_model = row[1]
                        
                        if row_date == today:
                            total_tokens = int(row[4])  # total_tokens column
                            
                            # Check which tier this model belongs to
                            if row_model in OPENAI_TOKEN_LIMITS["high_tier"]["models"]:
                                high_tier_usage += total_tokens
                            elif row_model in OPENAI_TOKEN_LIMITS["low_tier"]["models"]:
                                low_tier_usage += total_tokens
        except Exception as e:
            logger = logging.getLogger()
            if logger.handlers:
                logger.warning(f"Error checking token limits: {str(e)}", extra={"icon": "⚠️"})
    
    high_tier_limit = OPENAI_TOKEN_LIMITS["high_tier"]["limit"]
    low_tier_limit = OPENAI_TOKEN_LIMITS["low_tier"]["limit"]
    
    high_tier_percentage = (high_tier_usage / high_tier_limit) * 100 if high_tier_limit > 0 else 0
    low_tier_percentage = (low_tier_usage / low_tier_limit) * 100 if low_tier_limit > 0 else 0
    
    return {
        "high_tier": {
            "usage": high_tier_usage,
            "limit": high_tier_limit,
            "percentage": high_tier_percentage,
            "warning": high_tier_percentage > 80
        },
        "low_tier": {
            "usage": low_tier_usage,
            "limit": low_tier_limit,
            "percentage": low_tier_percentage,
            "warning": low_tier_percentage > 80
        }
    }

def generate_token_usage_report():
    """
    Generate a comprehensive report of token usage, with visualizations if packages are available.
    Saves the report in the reports directory.
    """
    today = date.today().isoformat()
    report_file = os.path.join(REPORTS_DIR, f"token_report_{today}.txt")
    html_report_file = os.path.join(REPORTS_DIR, f"token_summary_{today}.html")
    
    # Skip if report already exists
    if os.path.exists(report_file):
        return
    
    try:
        if not os.path.exists(DAILY_USAGE_FILE):
            return
        
        # Check if we have visualization packages available
        viz_available = False
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            import numpy as np
            viz_available = True
        except ImportError:
            pass
        
        # Load the data
        data = []
        models = set()
        with open(DAILY_USAGE_FILE, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            for row in reader:
                if len(row) >= 6:  # Ensure row has enough columns
                    data.append({
                        'date': row[0],
                        'model': row[1],
                        'input_tokens': int(row[2]),
                        'output_tokens': int(row[3]),
                        'total_tokens': int(row[4]),
                        'cost': float(row[5])
                    })
                    models.add(row[1])
        
        # Skip if no data
        if not data:
            return
        
        # If visualization packages are available, create PDF report
        if viz_available:
            try:
                pdf_report_file = os.path.join(REPORTS_DIR, f"token_report_{today}.pdf")
                
                # Convert to pandas DataFrame
                df = pd.DataFrame(data)
                
                # Create visualizations
                plt.figure(figsize=(20, 15))
                
                # 1. Daily token usage by model
                plt.subplot(2, 2, 1)
                pivot_df = df.pivot_table(
                    index='date', 
                    columns='model', 
                    values='total_tokens',
                    aggfunc='sum'
                ).fillna(0)
                pivot_df.plot(kind='bar', ax=plt.gca())
                plt.title('Daily Token Usage by Model')
                plt.xlabel('Date')
                plt.ylabel('Tokens')
                plt.xticks(rotation=45)
                plt.legend(title='Model')
                
                # 2. Model usage distribution (pie chart)
                plt.subplot(2, 2, 2)
                model_totals = df.groupby('model')['total_tokens'].sum()
                plt.pie(model_totals, labels=model_totals.index, autopct='%1.1f%%')
                plt.title('Token Usage Distribution by Model')
                
                # 3. Usage trend over time
                plt.subplot(2, 2, 3)
                daily_totals = df.groupby('date')['total_tokens'].sum()
                daily_totals.plot(kind='line', marker='o')
                plt.title('Total Token Usage Trend')
                plt.xlabel('Date')
                plt.ylabel('Total Tokens')
                plt.xticks(rotation=45)
                
                # 4. Limit status visualization
                plt.subplot(2, 2, 4)
                limits = check_token_limits()
                
                categories = ['High-Tier Models', 'Low-Tier Models']
                usage = [limits['high_tier']['usage'], limits['low_tier']['usage']]
                limits_values = [limits['high_tier']['limit'], limits['low_tier']['limit']]
                
                x = np.arange(len(categories))
                width = 0.35
                
                ax = plt.gca()
                usage_bars = ax.bar(x - width/2, usage, width, label='Usage')
                limit_bars = ax.bar(x + width/2, limits_values, width, label='Limit', alpha=0.5)
                
                # Add percentage labels
                for i, (usage_val, limit_val) in enumerate(zip(usage, limits_values)):
                    percentage = (usage_val / limit_val) * 100 if limit_val > 0 else 0
                    plt.text(i - width/2, usage_val + 0.05 * max(limits_values), 
                            f"{percentage:.1f}%", ha='center', va='bottom')
                
                ax.set_xticks(x)
                ax.set_xticklabels(categories)
                ax.legend()
                
                plt.title('Daily Token Usage vs. Limits')
                plt.ylabel('Tokens')
                
                # Add report title and date
                plt.suptitle(f'Token Usage Report - {today}', fontsize=16)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                
                # Save as PDF
                plt.savefig(pdf_report_file)
            except Exception as e:
                logger = logging.getLogger()
                if logger.handlers:
                    logger.warning(f"Failed to generate visualization report: {str(e)}", extra={"icon": "⚠️"})
        
        # Always create a text report
        with open(report_file, 'w') as f:
            f.write(f"TOKEN USAGE REPORT - {today}\n")
            f.write("="*80 + "\n\n")
            
            # Get limits
            limits = check_token_limits()
            
            # Summary 
            today_data = [item for item in data if item['date'] == today]
            today_total_tokens = sum(item['total_tokens'] for item in today_data)
            today_total_cost = sum(item['cost'] for item in today_data)
            
            f.write("TODAY'S SUMMARY:\n")
            f.write(f"  Total Tokens: {today_total_tokens:,}\n")
            f.write(f"  Total Cost: ${today_total_cost:.4f}\n\n")
            
            # Model breakdown
            f.write("MODEL BREAKDOWN:\n")
            for model in sorted(models):
                model_data = [item for item in today_data if item['model'] == model]
                if model_data:
                    model_tokens = sum(item['total_tokens'] for item in model_data)
                    model_cost = sum(item['cost'] for item in model_data)
                    f.write(f"  {model}:\n")
                    f.write(f"    Tokens: {model_tokens:,}\n")
                    f.write(f"    Cost: ${model_cost:.4f}\n")
            f.write("\n")
            
            # Limits
            f.write("TOKEN LIMITS:\n")
            f.write("  High-Tier Models:\n")
            f.write(f"    Usage: {limits['high_tier']['usage']:,} / {limits['high_tier']['limit']:,} tokens\n")
            f.write(f"    Percentage: {limits['high_tier']['percentage']:.1f}%\n")
            if limits['high_tier']['warning']:
                f.write("    WARNING: Approaching daily limit!\n")
            
            f.write("  Low-Tier Models:\n")
            f.write(f"    Usage: {limits['low_tier']['usage']:,} / {limits['low_tier']['limit']:,} tokens\n")
            f.write(f"    Percentage: {limits['low_tier']['percentage']:.1f}%\n")
            if limits['low_tier']['warning']:
                f.write("    WARNING: Approaching daily limit!\n")
            
            f.write("\n")
            f.write(f"Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Create HTML report
        with open(html_report_file, 'w') as f:
            f.write("<html><head>")
            f.write("<style>")
            f.write("body { font-family: Arial, sans-serif; margin: 20px; }")
            f.write("h1, h2 { color: #333366; }")
            f.write("table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }")
            f.write("th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }")
            f.write("th { background-color: #f2f2f2; }")
            f.write("tr:hover { background-color: #f5f5f5; }")
            f.write(".warning { color: red; font-weight: bold; }")
            f.write(".progress-container { width: 100%; background-color: #f1f1f1; border-radius: 5px; }")
            f.write(".progress-bar { height: 20px; border-radius: 5px; }")
            f.write(".progress-bar.high { background-color: #4CAF50; }")
            f.write(".progress-bar.medium { background-color: #FFEB3B; }")
            f.write(".progress-bar.critical { background-color: #F44336; }")
            f.write("</style>")
            f.write("</head><body>")
            
            f.write(f"<h1>Token Usage Summary - {today}</h1>")
            
            # Today's summary
            f.write("<h2>Today's Summary</h2>")
            f.write("<table>")
            f.write("<tr><th>Total Tokens</th><th>Total Cost</th></tr>")
            f.write(f"<tr><td>{today_total_tokens:,}</td><td>${today_total_cost:.4f}</td></tr>")
            f.write("</table>")
            
            # Model breakdown
            f.write("<h2>Model Breakdown</h2>")
            f.write("<table>")
            f.write("<tr><th>Model</th><th>Tokens</th><th>Cost</th></tr>")
            for model in sorted(models):
                model_data = [item for item in today_data if item['model'] == model]
                if model_data:
                    model_tokens = sum(item['total_tokens'] for item in model_data)
                    model_cost = sum(item['cost'] for item in model_data)
                    f.write(f"<tr><td>{model}</td><td>{model_tokens:,}</td><td>${model_cost:.4f}</td></tr>")
            f.write("</table>")
            
            # Limits
            f.write("<h2>Token Limits</h2>")
            
            # High-tier models
            high_pct = limits['high_tier']['percentage']
            high_class = "critical" if high_pct > 80 else "medium" if high_pct > 50 else "high"
            
            f.write("<h3>High-Tier Models (GPT-4o, GPT-4.1, O1, O3)</h3>")
            f.write(f"<p>{limits['high_tier']['usage']:,} / {limits['high_tier']['limit']:,} tokens ({high_pct:.1f}%)</p>")
            f.write("<div class='progress-container'>")
            f.write(f"<div class='progress-bar {high_class}' style='width:{min(100, high_pct)}%'></div>")
            f.write("</div>")
            if limits['high_tier']['warning']:
                f.write("<p class='warning'>WARNING: Approaching daily limit!</p>")
            
            # Low-tier models
            low_pct = limits['low_tier']['percentage']
            low_class = "critical" if low_pct > 80 else "medium" if low_pct > 50 else "high"
            
            f.write("<h3>Low-Tier Models (GPT-4o-mini, etc.)</h3>")
            f.write(f"<p>{limits['low_tier']['usage']:,} / {limits['low_tier']['limit']:,} tokens ({low_pct:.1f}%)</p>")
            f.write("<div class='progress-container'>")
            f.write(f"<div class='progress-bar {low_class}' style='width:{min(100, low_pct)}%'></div>")
            f.write("</div>")
            if limits['low_tier']['warning']:
                f.write("<p class='warning'>WARNING: Approaching daily limit!</p>")
            
            f.write(f"<p><em>Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>")
            f.write("</body></html>")
        
        logger = logging.getLogger()
        if logger.handlers:
            logger.info(f"Generated token usage text report: {report_file}", extra={"icon": "📊"})
            logger.info(f"Generated token summary HTML: {html_report_file}", extra={"icon": "📊"})
            
            # Log warnings if approaching limits
            if limits['high_tier']['warning']:
                logger.warning(f"HIGH-TIER MODEL USAGE AT {limits['high_tier']['percentage']:.1f}% OF DAILY LIMIT!", extra={"icon": "⚠️"})
            if limits['low_tier']['warning']:
                logger.warning(f"LOW-TIER MODEL USAGE AT {limits['low_tier']['percentage']:.1f}% OF DAILY LIMIT!", extra={"icon": "⚠️"})
    
    except Exception as e:
        logger = logging.getLogger()
        if logger.handlers:
            logger.error(f"Failed to generate token usage report: {str(e)}", extra={"icon": "❌"})
            logger.error(f"Exception details: {repr(e)}", extra={"icon": "❌"})

def display_token_usage_status():
    """
    Display current token usage status including limits and visualization.
    Can be called from the command line.
    """
    today = date.today().isoformat()
    
    # Ensure token tracking data is loaded
    load_token_tracking_data()
    
    # Check if we have data
    if not os.path.exists(DAILY_USAGE_FILE):
        print(f"No token usage data found at {DAILY_USAGE_FILE}")
        return
    
    try:
        # Read the CSV data
        data = []
        with open(DAILY_USAGE_FILE, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            for row in reader:
                if len(row) >= 6:  # Ensure row has enough columns
                    data.append({
                        'date': row[0],
                        'model': row[1],
                        'input_tokens': int(row[2]),
                        'output_tokens': int(row[3]),
                        'total_tokens': int(row[4]),
                        'cost': float(row[5])
                    })
        
        print("\n" + "="*80)
        print(f"TOKEN USAGE STATUS - {today}")
        print("="*80)
        
        # Display today's usage
        print("\nToday's Usage:")
        today_data = [item for item in data if item['date'] == today]
        if not today_data:
            print("  No usage recorded today")
        else:
            today_total_tokens = sum(item['total_tokens'] for item in today_data)
            today_total_cost = sum(item['cost'] for item in today_data)
            print(f"  Total Tokens: {today_total_tokens:,}")
            print(f"  Estimated Cost: ${today_total_cost:.4f}")
            
            # Display by model
            print("\n  Breakdown by Model:")
            models = set(item['model'] for item in today_data)
            for model in sorted(models):
                model_data = [item for item in today_data if item['model'] == model]
                model_tokens = sum(item['total_tokens'] for item in model_data)
                model_cost = sum(item['cost'] for item in model_data)
                print(f"    - {model}: {model_tokens:,} tokens (${model_cost:.4f})")
        
        # Check limits
        limits = check_token_limits()
        print("\nDaily Limits Status:")
        
        high_tier = limits['high_tier']
        low_tier = limits['low_tier']
        
        # Helper function for progress bar
        def get_progress_bar(percentage, width=40):
            filled = int(width * percentage / 100)
            bar = '█' * filled + '░' * (width - filled)
            return bar
        
        # High-tier models
        print(f"  High-Tier Models (GPT-4o, GPT-4.1, O1, O3):")
        print(f"    {high_tier['usage']:,} / {high_tier['limit']:,} tokens ({high_tier['percentage']:.1f}%)")
        print(f"    {get_progress_bar(high_tier['percentage'])}")
        if high_tier['warning']:
            print(f"    ⚠️  WARNING: Approaching daily limit!")
        
        # Low-tier models
        print(f"\n  Low-Tier Models (GPT-4o-mini, etc.):")
        print(f"    {low_tier['usage']:,} / {low_tier['limit']:,} tokens ({low_tier['percentage']:.1f}%)")
        print(f"    {get_progress_bar(low_tier['percentage'])}")
        if low_tier['warning']:
            print(f"    ⚠️  WARNING: Approaching daily limit!")
        
        # Reports info
        print("\nReports:")
        report_file = os.path.join(REPORTS_DIR, f"token_report_{today}.txt")
        html_report_file = os.path.join(REPORTS_DIR, f"token_summary_{today}.html")
        pdf_report_file = os.path.join(REPORTS_DIR, f"token_report_{today}.pdf")
        
        if os.path.exists(report_file):
            print(f"  Text Report: {report_file}")
        else:
            print("  Text Report: Not generated yet")
            
        if os.path.exists(html_report_file):
            print(f"  HTML Summary: {html_report_file}")
        else:
            print("  HTML Summary: Not generated yet")
            
        if os.path.exists(pdf_report_file):
            print(f"  PDF Report: {pdf_report_file}")
        else:
            print("  PDF Report: Not generated yet (requires matplotlib and pandas)")
        
        print("\nTo generate a new report, call generate_token_usage_report()")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"Error displaying token usage status: {str(e)}")

if __name__ == "__main__":
    # If run as a script, display the token usage status
    setup_project_directory()
    display_token_usage_status()

