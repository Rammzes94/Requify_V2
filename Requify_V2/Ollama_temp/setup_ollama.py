#!/usr/bin/env python3
"""
setup_ollama.py

This script helps set up the required dependencies and Ollama models for image analysis.
It checks if Ollama is installed, installs the necessary Python packages,
and pulls the required Ollama models.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

# Get the current directory
SCRIPT_DIR = Path(__file__).parent

# Vision models we want to install
VISION_MODELS = [
    "llava",           # Most reliable vision model
    "bakllava",        # Alternative vision model
    "llama3.2-vision"  # Meta's vision model
]

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)

def run_command(command, check=True):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(command, shell=True, check=check, 
                               capture_output=True, text=True)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return e

def check_ollama_installed():
    """Check if Ollama is installed and running."""
    print_header("Checking if Ollama is installed")
    
    # Check if Ollama is in PATH
    result = run_command("which ollama" if platform.system() != "Windows" else "where ollama", check=False)
    
    if result.returncode == 0:
        print("✅ Ollama is installed")
        
        # Check if Ollama server is running
        print("\nChecking if Ollama server is running...")
        server_check = run_command("curl -s http://localhost:11434/api/tags", check=False)
        
        if server_check.returncode == 0:
            print("✅ Ollama server is running")
            return True
        else:
            print("❌ Ollama server is not running")
            print("\nPlease start the Ollama server before continuing.")
            print("You can start it by running the 'ollama serve' command in a separate terminal.")
            return False
    else:
        print("❌ Ollama is not installed")
        print("\nPlease install Ollama first:")
        print("- macOS/Linux: curl -fsSL https://ollama.com/install.sh | sh")
        print("- Windows: Visit https://ollama.com/download")
        return False

def install_python_dependencies():
    """Install required Python packages."""
    print_header("Installing Python dependencies")
    
    required_packages = [
        "agno",
        "pillow"
    ]
    
    for package in required_packages:
        print(f"Installing {package}...")
        result = run_command(f"pip install {package}")
        if result.returncode == 0:
            print(f"✅ Successfully installed {package}")
        else:
            print(f"❌ Failed to install {package}")
            return False
    
    return True

def pull_ollama_models():
    """Pull the required Ollama models."""
    print_header("Pulling Ollama vision models")
    
    for model in VISION_MODELS:
        print(f"\nPulling {model} model...")
        print(f"This may take a while depending on your internet connection and if you've downloaded the model before.")
        result = run_command(f"ollama pull {model}")
        
        if result.returncode == 0:
            print(f"✅ Successfully pulled {model} model")
        else:
            print(f"❌ Failed to pull {model} model")
            print("Continuing with other models...")
    
    # List available models
    print("\nListing available models:")
    run_command("ollama list")
    
    return True

def main():
    """Main function to set up the environment."""
    print_header("Ollama Vision Setup")
    
    # Check if Ollama is installed and running
    if not check_ollama_installed():
        return False
    
    # Install Python dependencies
    if not install_python_dependencies():
        return False
    
    # Pull Ollama models
    if not pull_ollama_models():
        return False
    
    print_header("Setup Complete!")
    print("You can now run the image analysis scripts:")
    print(f"- python {SCRIPT_DIR}/image_analysis.py")
    print(f"- python {SCRIPT_DIR}/multi_model_analysis.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 