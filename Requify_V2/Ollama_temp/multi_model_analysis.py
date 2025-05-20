#!/usr/bin/env python3
"""
multi_model_analysis.py

This script demonstrates using Agno with multiple Ollama vision models to analyze images.
It tries different models to see which ones work best for the image analysis task.
"""

import os
import sys
import time
from pathlib import Path
from agno.agent import Agent
from agno.media import Image
from agno.models.ollama import Ollama

# Get the current directory where the script is located
SCRIPT_DIR = Path(__file__).parent

# List of vision models to try
VISION_MODELS = [
    "llava",         # Most common vision model
    "bakllava",      # Alternative vision model
    "gemma3",        # Google's vision model
    "llama3.2-vision" # Meta's vision model
]

def analyze_image_with_model(image_path, prompt, model_name):
    """
    Analyze an image using a specific Ollama vision model.
    
    Args:
        image_path: Path to the image file
        prompt: Prompt to send to the model along with the image
        model_name: Name of the Ollama model to use
        
    Returns:
        The model's response or None if the model fails
    """
    print(f"\n--- Trying model: {model_name} ---")
    
    try:
        # Create an agent with the specified Ollama vision model
        agent = Agent(
            model=Ollama(id=model_name),
            markdown=True,
        )
        
        # Process the image
        print("Getting response from model...")
        start_time = time.time()
        response = agent.run(
            prompt,
            images=[Image(filepath=image_path)],
        )
        end_time = time.time()
        
        # Calculate processing time
        process_time = end_time - start_time
        print(f"Model responded in {process_time:.2f} seconds")
        
        return {
            "model": model_name,
            "content": response.content,
            "process_time": process_time
        }
    
    except Exception as e:
        print(f"Error with model {model_name}: {str(e)}")
        return None

def find_best_working_model(image_path, prompt):
    """
    Try multiple models and find the best one that works.
    
    Args:
        image_path: Path to the image file
        prompt: Prompt to send to the model
        
    Returns:
        List of successful model responses
    """
    results = []
    
    for model_name in VISION_MODELS:
        result = analyze_image_with_model(image_path, prompt, model_name)
        if result:
            results.append(result)
    
    return results

if __name__ == "__main__":
    # Path to the image file
    image_path = SCRIPT_DIR / "Example.jpg"
    
    # Make sure the image exists
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)
    
    # Define the prompt
    prompt = "Describe in detail what you see in this image. What objects, people, or scenes are present?"
    
    print(f"Analyzing image: {image_path}")
    print(f"Prompt: {prompt}")
    
    # Try multiple models
    results = find_best_working_model(image_path, prompt)
    
    # Print results
    if results:
        print("\n--- Image Analysis Results ---")
        for i, result in enumerate(results, 1):
            print(f"\nResult {i} - Model: {result['model']} (processed in {result['process_time']:.2f}s)")
            print("-" * 50)
            print(result['content'])
            print("-" * 50)
    else:
        print("\nNo models successfully analyzed the image. Make sure Ollama is running and has vision models installed.")
        print("You can install models using: ollama pull <model_name>")
        print("For example: ollama pull llava") 