#!/usr/bin/env python3
"""
image_analysis.py

This script demonstrates using Agno with Ollama to analyze images using a multimodal model.
It loads an image and asks Ollama's vision model to describe the image contents.
"""

import os
import sys
from pathlib import Path
from agno.agent import Agent
from agno.media import Image
from agno.models.ollama import Ollama

# Get the current directory where the script is located
SCRIPT_DIR = Path(__file__).parent

def analyze_image(image_path, prompt):
    """
    Analyze an image using Ollama's vision model.
    
    Args:
        image_path: Path to the image file
        prompt: Prompt to send to the model along with the image
        
    Returns:
        The model's response
    """
    print(f"Analyzing image: {image_path}")
    print(f"Prompt: {prompt}")
    
    # Create an agent with Ollama's vision model
    agent = Agent(
        model=Ollama(id="llava"),  # llava is Ollama's most common vision model
        markdown=True,
    )
    
    # Process the image
    print("\nGetting response from model...")
    response = agent.run(
        prompt,
        images=[Image(filepath=image_path)],
    )
    
    return response.content

if __name__ == "__main__":
    # Path to the image file
    image_path = SCRIPT_DIR / "Example.jpg"
    
    # Make sure the image exists
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)
    
    # Analyze the image
    prompt = "Describe in detail what you see in this image. What objects, people, or scenes are present?"
    result = analyze_image(image_path, prompt)
    
    # Print the result
    print("\n--- Image Analysis Results ---")
    print(result) 