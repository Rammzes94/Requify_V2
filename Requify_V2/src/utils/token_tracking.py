"""
Token tracking utilities for the Requify_V2 project.

Provides functions for tracking token usage, calculating costs,
and displaying usage statistics.
"""

import os
import logging
import json
import csv
from datetime import date
from collections import defaultdict

# Import configuration from parent directory
from src import config

MODEL_PRICING = config.MODEL_PRICING
MODEL_TIERS = config.MODEL_TIERS

# Initialize token counters as global variables
total_input_tokens = 0
total_output_tokens = 0
model_token_usage = defaultdict(lambda: {"input": 0, "output": 0})

# Token tracking file paths
TOKEN_TRACKING_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                "_03_output", "token_tracking")
DAILY_USAGE_FILE = os.path.join(TOKEN_TRACKING_DIR, "daily_token_usage.csv")
MODEL_USAGE_FILE = os.path.join(TOKEN_TRACKING_DIR, "model_token_usage.json")

# Ensure directories exist
os.makedirs(TOKEN_TRACKING_DIR, exist_ok=True)

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
                logger.info(f"Token counters updated: {input_tokens_added} input, {output_tokens_added} output for {model_id}")
            else:
                print(f"Token counters updated: {input_tokens_added} input, {output_tokens_added} output for {model_id}")
        except Exception:
            # If there's any issue with logging, fall back to print
            print(f"Token counters updated: {input_tokens_added} input, {output_tokens_added} output for {model_id}")
        
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
                logger.warning(f"Unknown model '{model_id}'. Defaulting to gpt-4o-mini pricing.")
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
            logger.info(f"\nToken usage for {model_id}:")
            logger.info(f"  - Input tokens: {total_input_tokens}")
            logger.info(f"  - Output tokens: {total_output_tokens}")
            logger.info(f"  - Total tokens: {total_tokens}")
            logger.info(f"  - Estimated cost: ${estimated_cost:.4f}")
            logger.info(f"  - Energy consumption: {energy_usage:.6f} Wh ({energy_usage/1000:.6f} kWh)")
            logger.info(f"  - CO₂ emissions: {co2_emissions:.6f} gCO₂ ({co2_emissions/1000:.6f} kgCO₂)")
            
            # Log limit status
            tier = "high_tier" if model_id in MODEL_TIERS["high_tier"]["models"] else "low_tier"
            logger.info(f"  - Daily limit: {limits[tier]['usage']} / {limits[tier]['limit']} tokens ({limits[tier]['percentage']:.1f}%)")
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

def reset_token_counters():
    """Reset the token counters to zero for the current session only."""
    global total_input_tokens, total_output_tokens
    total_input_tokens = 0
    total_output_tokens = 0
    
    logger = logging.getLogger()
    if logger.handlers:
        logger.info("Token counters reset for current session")
    else:
        print("Token counters reset for current session")

def load_token_tracking_data():
    """
    Load token usage data from tracking files if they exist.
    Updates the global model_token_usage dictionary.
    """
    global model_token_usage
    
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
                logger.info(f"Loaded token usage data for {len(model_token_usage)} models")
        except Exception as e:
            logger = logging.getLogger()
            if logger.handlers:
                logger.warning(f"Failed to load token usage data: {str(e)}")
    
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
            logger.warning(f"Failed to save model token usage: {str(e)}")
    
    # Calculate cost based on model_id using MODEL_PRICING from config
    pricing = MODEL_PRICING.get(model_id, MODEL_PRICING.get("gpt-4o-mini"))
    input_cost = (input_tokens / 1000000) * pricing["input"]
    output_cost = (output_tokens / 1000000) * pricing["output"]
    total_cost = input_cost + output_cost
    
    # Append to daily usage file
    try:
        # Check if we already have an entry for this model today
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
            logger.warning(f"Failed to update daily token usage: {str(e)}")

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
                            
                            # Use the MODEL_TIERS constant to determine tier
                            if row_model in MODEL_TIERS["high_tier"]["models"]:
                                high_tier_usage += total_tokens
                            elif row_model in MODEL_TIERS["low_tier"]["models"]:
                                low_tier_usage += total_tokens
                            # For unknown models, try to check if we have pricing data with tier info
                            elif row_model in MODEL_PRICING and MODEL_PRICING[row_model].get("tier") == "high":
                                high_tier_usage += total_tokens
                            else:
                                # Default to low tier for unknown models
                                low_tier_usage += total_tokens
        except Exception as e:
            logger = logging.getLogger()
            if logger.handlers:
                logger.warning(f"Error checking token limits: {str(e)}")
    
    high_tier_limit = MODEL_TIERS["high_tier"]["limit"]
    low_tier_limit = MODEL_TIERS["low_tier"]["limit"]
    
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

def display_token_usage_status():
    """
    Display current token usage status including limits and visualization in the console.
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
            
        print("\n" + "="*80)
    
    except Exception as e:
        print(f"Error displaying token usage status: {str(e)}")

if __name__ == "__main__":
    # If run as a script, display the token usage status
    from .directory_utils import setup_project_directory
    
    setup_project_directory()
    display_token_usage_status()