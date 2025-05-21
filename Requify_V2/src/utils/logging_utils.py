"""
Logging utilities for the Requify_V2 project.

Provides functions for setting up logging with custom icon formatting.
"""

import os
import sys
import logging

# Map of message content keywords to icons
MESSAGE_ICONS = {
    # Aviation/Aerospace
    "fighter": "✈️",
    "jet": "✈️",
    "rocket": "🚀",
    "launch": "🚀",
    "missile": "🎯",
    "radar": "📡",
    "aircraft": "✈️",
    "plane": "✈️",
    "pilot": "👨‍✈️",
    "flight": "👨‍✈️",
    "navigation": "🧭",
    "altitude": "🏔️",
    "specification": "📋",
    
    # Database related
    "database": "🛢️",
    "db": "🛢️",
    "connect": "🔌",
    "connecting": "🔌",
    "query": "🔍",
    "searching": "🔍",
    "table": "📋",
    "create table": "📝",
    "drop table": "🗑️",
    "index": "📇",
    "backup": "💾",
    "restore": "📤",
    
    # File operations
    "file": "📄",
    "read file": "📂",
    "loading file": "📂",
    "open file": "📂",
    "write file": "📥",
    "saving file": "📥",
    "delete file": "🗑️",
    "remove file": "🗑️",
    "duplicate file": "🔄",
    "hash file": "🔐",
    
    # Processing status
    "start": "🚀",
    "begin": "🚀",
    "initializing": "🚀",
    "process": "⚙️",
    "running": "⚙️",
    "executing": "⚙️",
    "complete": "✅",
    "finish": "✅",
    "done": "✅",
    "success": "✅",
    "timeout": "⏱️",
    "too long": "⏱️",
    "retry": "🔁",
    
    # Document processing specific
    "pdf": "📑",
    "image": "🖼️",
    "png": "🖼️", 
    "jpg": "🖼️",
    "embedding": "🧮",
    "vector": "🧮",
    "chunk": "🧩",
    "duplicate chunk": "♻️",
    "similar chunk": "👯",
    "token count": "🔢",
    "token": "🎫",
    
    # Pipeline stages
    "pipeline": "🔄",
    "pipeline step": "📶",
    "pipeline error": "📛",
    "pipeline fail": "📛",
    "extract requirement": "📋",
    "parse": "📝",
    "deduplication": "🔍",
    "deduplicate": "🔍",
    "document deduplication": "📚",
    
    # AI/ML Related
    "model": "🧠",
    "load model": "🤖",
    "predict": "🔮",
    "inference": "🔮",
    "train model": "🏋️",
    "llm": "🤖",
    "language model": "🤖",
    "gpt": "🧠",
    "openai": "🔮",
    "groq": "⚡",
    "anthropic": "🦜",
    "claude": "🦜",
    "agent": "🕵️",
    "prompt": "💬",
    "ai": "🧠",
    "neural": "🔄",
    "transformer": "🧿",
    
    # Statistics and metrics
    "stat": "📊",
    "metric": "📊",
    "summary": "📈",
    "report": "📜",
    "time": "⏱️",
    "duration": "⏱️",
    "progress": "⏳",
    
    # Memory management
    "memory": "🧠",
    "clean memory": "🧹",
    "memory leak": "💧",
    "cache": "💽",
    
    # Common actions
    "check": "✓",
    "verify": "✓",
    "compare": "⚖️",
    "detect": "🔎",
    "skip": "⏭️",
    "found": "🔍",
    "add": "➕",
    "remove": "➖",
    "update": "🔄",
    "change": "📝",
    "match": "🎯",
    "replace": "🔀",
    "merge": "🔄",
    
    # Security and protection
    "secure": "🔒",
    "security": "🔒",
    "encrypt": "🔐",
    "authenticate": "🔑",
    "auth": "🔑",
    "permission": "🛡️",
    "firewall": "🧱",
    "password": "🔐",
    
    # Hardware and infrastructure
    "server": "🖥️",
    "cpu": "⚡",
    "processor": "⚡",
    "gpu": "🎮",
    "ram": "🧠",
    "disk": "💾",
    "storage": "💾",
    "network": "🌐",
    "cloud": "☁️",
    
    # Specialized content
    "requirement": "📋",
    "similarity": "🔄",
    "directory": "📁",
    "folder": "📁",
    "path": "🧭",
    "config": "⚙️",
    "settings": "⚙️",
    "batch": "📦",
    "document": "📄",
    "page": "📃",
    "version": "🏷️",
    "schema": "📐",
    "id": "🔑",
    "identifier": "🔑",
    "create": "🆕",
}

# Default icons for log levels
LEVEL_ICONS = {
    'DEBUG': '🐞',
    'INFO': 'ℹ️',
    'WARNING': '⚠️',
    'ERROR': '❌',
    'CRITICAL': '💥'
}

# Module-specific default icons
MODULE_ICONS = {
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

class IconFilter(logging.Filter):
    """Filter that adds an icon attribute to log records based on content and level."""
    
    def filter(self, record):
        if not hasattr(record, 'icon'):
            # First check for module-specific icon
            module_name = record.module.lower() if hasattr(record, 'module') else ''
            
            for key, icon in MODULE_ICONS.items():
                if key in module_name:
                    record.icon = icon
                    return True
            
            # If no module match, check message content for keyword matches
            message = record.getMessage().lower()
            for keyword, icon in MESSAGE_ICONS.items():
                if keyword.lower() in message:
                    record.icon = icon
                    return True
            
            # Default to level-based icon
            level_name = record.levelname
            record.icon = LEVEL_ICONS.get(level_name, '📌')
            
        return True

class ScriptLogger(logging.LoggerAdapter):
    """
    ScriptLogger adds a consistent prefix to all log messages.
    
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
    Returns a logger instance with icon filtering.
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = os.getenv("LOG_FORMAT", '%(asctime)s [%(levelname)s] %(icon)s - %(message)s')
    log_date_format = os.getenv("LOG_DATE_FORMAT", '%Y-%m-%d %H:%M:%S')
    log_to_console = os.getenv("LOG_TO_CONSOLE", "True").lower() == "true"
    log_to_file = os.getenv("LOG_TO_FILE", "False").lower() == "true"
    log_file_path = os.getenv("LOG_FILE_PATH", "logs/requify_agent.log")
    log_file_mode = os.getenv("LOG_FILE_MODE", "a")

    handlers = []
    icon_filter = IconFilter()

    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.addFilter(icon_filter)
        handlers.append(console_handler)
    
    if log_to_file:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        file_handler = logging.FileHandler(log_file_path, mode=log_file_mode)
        file_handler.addFilter(icon_filter)
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
    return logging.getLogger(logger_name)

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