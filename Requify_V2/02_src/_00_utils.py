import os
import sys
import pathlib
import pdfplumber
import re
import logging


# Initialize token counters as global variables
total_input_tokens = 0
total_output_tokens = 0

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

def update_token_counters(response):
    """
    Update token counters based on API response metrics.
    
    Args:
        response: The response object from an agent.run() call
    """
    global total_input_tokens, total_output_tokens
    if hasattr(response, 'metrics'):
        metrics = response.metrics
        if 'input_tokens' in metrics and isinstance(metrics['input_tokens'], list) and metrics['input_tokens']:
            total_input_tokens += metrics['input_tokens'][0]
            
        if 'output_tokens' in metrics and isinstance(metrics['output_tokens'], list) and metrics['output_tokens']:
            total_output_tokens += metrics['output_tokens'][0]
        print("Token counters updated: " + str(total_input_tokens) + " input, " + str(total_output_tokens) + " output")

def get_token_usage():
    """
    Get the current token usage statistics.
    
    Returns:
        dict: Dictionary containing token usage information
    """
    return {
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens
    }





def print_token_usage(model_id="gpt-4o-mini"):
    """
    Print a summary of token usage, estimated cost, energy usage, and CO2 emissions.
    
    Args:
        model_id (str): The model ID to use for pricing calculation
    """
    if model_id not in MODEL_PRICING:
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
    
    print(f"\nToken usage for {model_id}:")
    print(f"  - Input tokens: {total_input_tokens}")
    print(f"  - Output tokens: {total_output_tokens}")
    print(f"  - Total tokens: {total_tokens}")
    print(f"  - Estimated cost: ${estimated_cost:.4f}")
    print(f"  - Energy consumption: {energy_usage:.6f} Wh ({energy_usage/1000:.6f} kWh)")
    print(f"  - CO₂ emissions: {co2_emissions:.6f} gCO₂ ({co2_emissions/1000:.6f} kgCO₂)")

def reset_token_counters():
    """Reset the token counters to zero."""
    global total_input_tokens, total_output_tokens
    total_input_tokens = 0
    total_output_tokens = 0



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
                    print(f"Interactive mode: Changed working directory to: {test_dir}")
                else:
                    print(f"Interactive mode: Already in project root: {test_dir}")
                break
            
            # Move up one directory level
            test_dir = os.path.dirname(test_dir)
            
            # If we've reached the filesystem root without finding indicators
            if test_dir == os.path.dirname(test_dir):
                print(f"Interactive mode: Could not find project root. Staying in: {current_dir}")
                break
    else:
        # For normal script execution, do nothing special
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
            # Determine a generic default icon based on level if possible,
            # otherwise a truly generic one.
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
                record.icon = ' ' # A space or an empty string if preferred
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

            if level_name == 'INFO':
                if "found" in msg.lower() and "similar" in msg.lower():
                    extra['icon'] = '🎯'
                elif "processing" in msg.lower() or "querying" in msg.lower():
                    extra['icon'] = '🔄'
                elif "connected" in msg.lower() or "loaded" in msg.lower() or "completed" in msg.lower() or "successfully" in msg.lower():
                    extra['icon'] = '✅'
                elif "skipped" in msg.lower() or "no new similar" in msg.lower() or "no pages found" in msg.lower() or "no search results" in msg.lower(): # Added more skip/empty conditions
                    extra['icon'] = '⏩'
                else:
                    extra['icon'] = 'ℹ️'  # Default for INFO
            elif level_name == 'WARNING':
                extra['icon'] = '⚠️'
            elif level_name == 'ERROR':
                extra['icon'] = '❌'
            elif level_name == 'CRITICAL':
                extra['icon'] = '💥'
            elif level_name == 'DEBUG':
                extra['icon'] = '🐞'
            else:
                extra['icon'] = 'ℹ️' # Default icon for unknown levels

        kwargs['extra'] = extra
        return msg, kwargs

def setup_logging():
    """
    Configures logging for the project based on .env settings.
    Returns a logger instance with an IconAdapter.
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = os.getenv("LOG_FORMAT", '%(asctime)s [%(levelname)s] (%(module)s:%(lineno)d) %(icon)s - %(message)s')
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

