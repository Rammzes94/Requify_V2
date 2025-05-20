# Ollama Vision Image Analysis

This folder contains scripts for analyzing images using Ollama's vision-capable models through the Agno framework.

## Overview

These scripts demonstrate how to use locally running Ollama models to analyze images. Unlike cloud-based solutions, Ollama runs entirely on your computer, offering privacy and no usage fees.

## Prerequisites

- Python 3.8 or higher
- Ollama installed and running (https://ollama.com/download)
- At least one vision-capable Ollama model

## Quick Start

1. **Setup your environment:**

```bash
python setup_ollama.py
```

This will check if Ollama is installed, install the required Python packages, and download the necessary vision models.

2. **Run the basic image analysis script:**

```bash
python image_analysis.py
```

3. **Or try multiple models to see which works best:**

```bash
python multi_model_analysis.py
```

## Available Scripts

- **setup_ollama.py** - Sets up the necessary dependencies and models
- **image_analysis.py** - Simple script for analyzing an image using the LLaVA model
- **multi_model_analysis.py** - Advanced script that tries multiple vision models

## Supported Vision Models

The scripts are configured to work with these Ollama vision models:

- **llava** - The most widely used vision model, good for general image analysis
- **bakllava** - An alternative vision model
- **gemma3** - Google's multimodal model (if available)
- **llama3.2-vision** - Meta's vision model

## Custom Usage

To analyze your own images, replace `Example.jpg` with your image file, or modify the image path in the scripts.

You can also customize the prompts in the scripts to ask different questions about the images.

## Troubleshooting

If you encounter issues:

1. Make sure Ollama is installed and running
2. Verify that you've downloaded at least one vision model with `ollama list`
3. If a model is missing, download it with `ollama pull llava` (or another model name)
4. Check for error messages that might indicate what's wrong

## Resources

- [Ollama Documentation](https://github.com/ollama/ollama/tree/main/docs)
- [Agno Documentation](https://docs.agno.com)
- [Ollama Models Library](https://ollama.com/library) 