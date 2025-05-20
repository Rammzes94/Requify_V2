#!/usr/bin/env python3
"""
image_text_extractor.py

This script extracts text from images using Ollama's vision capabilities through the Agno framework.
It's optimized for OCR-like functionality to extract and structure textual content from images.
"""

import os
import sys
import argparse
from pathlib import Path
from agno.agent import Agent
from agno.media import Image
from agno.models.ollama import Ollama

# Constants
DEFAULT_MODEL = "llava:7b"
DEFAULT_PROMPT = """
Extract and transcribe ALL text visible in this image.
Format your response as plain text.
Preserve the layout and structure of the text as shown in the image.
Include ALL text elements (titles, headings, paragraphs, captions, buttons, menus, etc).
Do not include descriptions of the image or non-text elements.
Do not include your own observations or interpretations.
"""

def extract_text_from_image(image_path, prompt, model_name=DEFAULT_MODEL):
    """
    Extract text from an image using Ollama's vision model.
    
    Args:
        image_path: Path to the image file
        prompt: Prompt to send to the model along with the image
        model_name: Name of the Ollama model to use (default: llava:7b)
        
    Returns:
        The extracted text from the image
    """
    # Verify the image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)
    
    print(f"Processing image: {image_path}")
    print(f"Using model: {model_name}")
    
    try:
        # Create an agent with Ollama's vision model
        agent = Agent(
            model=Ollama(id=model_name),
            markdown=True,
        )
        
        # Process the image
        print("\nExtracting text from image...")
        response = agent.run(
            prompt,
            images=[Image(filepath=image_path)],
        )
        
        return response.content
    except Exception as e:
        print(f"Error extracting text: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure Ollama is installed and running")
        print(f"2. Verify that the '{model_name}' model is installed (run 'ollama list')")
        print(f"3. If the model is missing, install it with 'ollama pull {model_name}'")
        print("4. Check if your image format is supported (JPG, PNG, etc.)")
        sys.exit(1)

def main():
    """Parse command-line arguments and extract text from the image."""
    parser = argparse.ArgumentParser(description="Extract text from images using Ollama's vision models")
    
    # Add command-line arguments
    parser.add_argument("--image", "-i", type=str, help="Path to the image file to process")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL, 
                        help=f"Ollama model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--prompt", "-p", type=str, 
                        default=DEFAULT_PROMPT,
                        help="Prompt to send to the model for text extraction")
    parser.add_argument("--output", "-o", type=str, help="Output file path for extracted text (optional)")
    
    args = parser.parse_args()
    
    # If no image path provided, check for example.jpg in current directory
    if not args.image:
        # Check in this priority:
        # 1. example.jpg in current working directory
        # 2. Example.jpg in current working directory
        # 3. example.jpg in script directory
        # 4. Example.jpg in script directory

        # Current working directory
        cwd = Path.cwd()
        script_dir = Path(__file__).parent
        
        possible_paths = [
            cwd / "example.jpg",  # User's own example.jpg in current directory
            cwd / "Example.jpg",
            script_dir / "example.jpg",
            script_dir / "Example.jpg"
        ]
        
        image_path = None
        for path in possible_paths:
            if path.exists():
                image_path = str(path)
                print(f"Using image: {image_path}")
                break
                
        if not image_path:
            print("Error: No image specified and no example.jpg found.")
            print("Please specify an image with --image or place example.jpg in the current directory.")
            sys.exit(1)
    else:
        image_path = args.image
    
    # Extract text from the image
    extracted_text = extract_text_from_image(image_path, args.prompt, args.model)
    
    # Write to output file if specified
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(extracted_text)
            print(f"\nExtracted text saved to: {args.output}")
        except Exception as e:
            print(f"Error writing to output file: {str(e)}")
    
    # Print the extracted text
    print("\n--- Extracted Text ---")
    print(extracted_text)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 