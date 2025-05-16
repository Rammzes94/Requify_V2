# ---------------------------------------------------------------------
# PDF Parsing Script - stable_pdf_parsing.py
# ---------------------------------------------------------------------
"""
stable_pdf_parsing.py

This script parses PDF documents and extracts their content in a structured format.
It performs the following operations:
1. Converts PDF pages to high-resolution images
2. Uses vision-enabled LLMs to extract text, tables, and descriptions of non-text elements
3. Processes extracted content into markdown format
4. Generates metadata such as document title, page summaries, and relevant hashtags
5. Creates a combined JSON output with all extracted information
6. Saves page images as encoded base64 strings for later reference

The script supports both OpenAI and Groq models for vision and text processing tasks.
It uses PyMuPDF for PDF manipulation and handles pages sequentially to avoid memory issues.
"""
# ---------------------------------------------------------------------
# Section 1: Imports and Setup
# ---------------------------------------------------------------------
import os
import sys
import time
import traceback
import json
import base64 # Added for image encoding
from datetime import datetime, timezone
from typing import List, Optional, Tuple # Added Tuple
import logging

import fitz  # PyMuPDF
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.media import Image
from agno.models.openai import OpenAIChat
from agno.models.groq import Groq

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import _00_utils

# Setup centralized logging with script prefix
logger = _00_utils.setup_logging()
logger = logging.LoggerAdapter(logger, {"prefix": "[PDF_Parsing] "})

class ScriptLogger(logging.LoggerAdapter):
    def __init__(self, logger, prefix):
        super().__init__(logger, {})
        self.prefix = prefix
        
    def process(self, msg, kwargs):
        return f"{self.prefix}{msg}", kwargs

logger = ScriptLogger(_00_utils.setup_logging(), "[PDF_Parsing] ")

# Load environment variables
load_dotenv()

# ---------------------------------------------------------------------
# Section 2: Configuration Constants
# ---------------------------------------------------------------------

# Default paths
RAW_INPUT_DIR = os.path.join("_01_input", "raw")
# Define the main output directory for this script's products
OUTPUT_DIR_BASE = os.path.join("_03_output") 
PROCESSED_OUTPUT_BASE_DIR = os.path.join(OUTPUT_DIR_BASE) # Changed to use OUTPUT_DIR_BASE
FILTERED_FILES_JSON = os.path.join(OUTPUT_DIR_BASE, "filtered_files_by_extension.json") # Changed to read from OUTPUT_DIR_BASE
# DPI setting for PDF to image conversion
PDF_TO_IMAGE_DPI = 300
# Note: We process files sequentially to avoid pickling errors with agent objects

# API keys from environment variables
api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Model Configuration
# Initialize OpenAI models for vision and text processing
openai_vision_model = OpenAIChat(id="gpt-4o", api_key=api_key)
openai_text_model = OpenAIChat(id="gpt-4o-mini", api_key=api_key)

# Groq models (commented out by default)
groq_vision_model = Groq(id="meta-llama/llama-4-scout-17b-16e-instruct", api_key=groq_api_key)
groq_text_model = Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key)

# Select which models to use
#active_vision_model = openai_vision_model
#active_text_model = openai_text_model
active_vision_model = groq_vision_model
active_text_model = groq_text_model

# ---------------------------------------------------------------------
# Section 3: Pydantic Models for AI bot and for final extracted data
# ---------------------------------------------------------------------
class AgnoPageMetadata(BaseModel):
    document_title: str = Field(..., description="Title for the entire document, derived from first 10 pages.")
    summary: str = Field(..., description="Short summary of the page content, 200-300 characters.")
    hashtags: List[str] = Field(..., description="Key search words for the content (without '#').")

# PageExtractedData inherits from AgnoPageMetadata to include all the metadata fields (document_title, summary, hashtags)
# while adding additional fields specific to the extracted page data
class PageExtractedData(AgnoPageMetadata):
    pdf_identifier: str = Field(..., description="Identifier/source filename of the PDF file.")
    page_number: int = Field(..., description="Page number in the PDF file.")
    md_content: str = Field(..., description="Markdown content extracted from the page.")
    image_b64: Optional[str] = Field(None, description="Base64 encoded string of the page image (PNG).")
    input_tokens: int = Field(..., description="Number of tokens provided as input.")
    output_tokens: int = Field(..., description="Number of tokens generated as output.")
    processing_duration: float = Field(..., description="Duration (in seconds) of processing the page.")
    error_flag: bool = Field(..., description="Indicates if an error occurred during processing.")
    timestamp: str = Field(..., description="ISO timestamp when processing completed.")

# ---------------------------------------------------------------------
# Section 4: Agent Initialization
# ---------------------------------------------------------------------
# Plain agent for PDF parsing (no structured outputs)
plain_agent = Agent(
    model=active_vision_model,
    markdown=True,
    debug_mode=False,
    description=(
        "You are an agent that extracts contents from an image that originated from a pdf file. "
        "You want to extract as much information as possible - not only focussing on text but also images, graphs, diagrams, etc. "
        "You are very technical and precise."
    )
)

# Structured agent to generate metadata from extracted markdown.
structured_agent = Agent(
    model=active_text_model,
    markdown=False,
    debug_mode=False,
    description=(
        "You are an agent that receives plain markdown text extracted from a PDF page. "
        "Your sole task is to generate and return a valid JSON object containing the following fields: "
        "'document_title' (use the provided document title), "
        "'summary' (a concise summary of the page content, 200-300 characters), "
        "and 'hashtags' (a list of approximately 10 key search words for the content, without the '#' symbol). "
        "Ensure your output is ONLY this JSON object and nothing else. Do not include any explanatory text before or after the JSON."
    ),
    response_model=AgnoPageMetadata,
    use_json_mode=True  # Ensure the LLM is constrained to output valid JSON
)

# Add a new agent for document title generation
document_title_agent = Agent(
    model=active_text_model,
    markdown=False,
    debug_mode=False,
    description=(
        "You are an agent that receives combined markdown text from the first pages of a PDF document and generates "
        "a concise, descriptive title that captures the main subject and purpose of the document. "
        "The title should be between 5-10 words and clearly identify what the document is about."
        "Give back only the the title, NOTHING ELSE."
    )
)

# ---------------------------------------------------------------------
# Section 4: PDF Processing Controller Class
# ---------------------------------------------------------------------
class PDFProcessor:
    def __init__(self, plain_agent: Agent, structured_agent: Agent, document_title_agent: Agent):
        self.plain_agent = plain_agent
        self.structured_agent = structured_agent
        self.document_title_agent = document_title_agent
    # -----------------------------------------------------------------
    # Method: Convert PDF to Images
    # -----------------------------------------------------------------
    def pdf_to_images(self, pdf_path, dpi=PDF_TO_IMAGE_DPI, output_images_dir=None) -> List[Tuple[str, Optional[str]]]:
        logger.info(f"Processing '{pdf_path}' to images...")
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file '{pdf_path}' does not exist.")
            return []
        try:
            pdf = fitz.open(pdf_path)
            image_data_list = []
            if output_images_dir is None:
                base_name = os.path.splitext(os.path.basename(pdf_path))[0]
                # Use the central processed directory, now under _03_output
                output_images_dir = os.path.join(PROCESSED_OUTPUT_BASE_DIR, "pdf_images", base_name)
            os.makedirs(output_images_dir, exist_ok=True)
            for i, page in enumerate(pdf):
                zoom = dpi / 72  # Default PDF DPI is 72
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                page_name = f"page_{i+1:03d}"
                image_path = os.path.join(output_images_dir, f"{page_name}.png")
                image_b64_string = None
                try:
                    pix.save(image_path)
                    logger.info(f"Saved image for page {i+1} to {image_path}")
                    # Read the saved image and encode to base64
                    with open(image_path, "rb") as image_file:
                        image_b64_string = base64.b64encode(image_file.read()).decode('utf-8')
                    logger.info(f"Encoded image for page {i+1} to base64.")
                except Exception as e:
                    logger.error(f"Error saving or encoding image for page {i+1}: {e}")
                
                image_data_list.append((image_path, image_b64_string))

            logger.info(f"Generated and encoded {len(image_data_list)} images from PDF")
            return image_data_list
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            traceback.print_exc()
            return []

    def generate_document_title(self, md_contents, pdf_identifier):
        """Generate a title for the entire document based on page summaries"""
        logger.info(f"Generating document title for {pdf_identifier} from page summaries...")
        try:
            # First generate summaries for each page if we have content
            page_summaries = []
            for i, content in enumerate(md_contents[:10]):  # Only use first 10 pages
                if not content:
                    continue
                    
                # Generate a quick summary for each page
                try:
                    prompt = f"Summarize this page content in 1-2 sentences:\n\n{content[:1000]}"  # Limit content size
                    response = self.structured_agent.run(prompt)
                    _00_utils.update_token_counters(response) # Added token counting
                    summary = response.content
                    if isinstance(summary, str):
                        page_summaries.append(summary)
                    else:
                        # If we got a structured response, try to extract summary
                        summary = summary.get('summary', '') if hasattr(summary, 'get') else str(summary)
                        page_summaries.append(summary)
                except Exception as e:
                    logger.error(f"Error generating summary for page {i+1}: {str(e)}")
            
            # Combine summaries for title generation
            combined_summaries = "\n\n".join(page_summaries)
            
            if not combined_summaries:
                logger.warning(f"No summaries available for document title generation: {pdf_identifier}")
                return f"Document {pdf_identifier}"
                
            # Generate title from summaries
            prompt = (
                "Based on the following summaries from the first pages of a document, "
                "create a concise and descriptive title that captures the main subject "
                "and purpose of the document. The title should be between 5-10 words.\n\n"
                f"Page summaries:\n{combined_summaries}"
            )
            
            response = self.document_title_agent.run(prompt)
            _00_utils.update_token_counters(response) # Added token counting
            document_title = response.content.strip()
            logger.info(f"Generated document title from summaries: {document_title}")
            return document_title
        except Exception as e:
            logger.error(f"Error generating document title: {str(e)}")
            return f"Document {pdf_identifier}"

    # -----------------------------------------------------------------
    # Method: Process PDF to Structured JSON
    # -----------------------------------------------------------------
    def pdf_to_structured_json(self, pdf_path, output_dir=None): # output_dir will be constructed, not taken as default
        try:
            # Ensure the base output directory for parsed content exists under _03_output
            # output_dir for specific PDF will be a subfolder of this
            pdf_base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            # Construct specific output_dir for this PDF's parsed content
            specific_output_dir = os.path.join(PROCESSED_OUTPUT_BASE_DIR, "parsed_content", pdf_base_name)
            os.makedirs(specific_output_dir, exist_ok=True)

            image_data_list = self.pdf_to_images(pdf_path) # pdf_to_images now uses PROCESSED_OUTPUT_BASE_DIR for its output
            if not image_data_list:
                logger.error("No images were created or encoded from the PDF. Process failed.")
                return None
            logger.info(f"Successfully converted PDF to {len(image_data_list)} images and encoded them.")
            pdf_identifier = os.path.basename(pdf_path)
            
            # First, extract markdown content from each page and capture token usage
            md_extraction_results = []
            for i, (image_path, _) in enumerate(image_data_list):
                page_input_tokens = 0
                page_output_tokens = 0
                page_md_content_extracted = ""
                extraction_error = False
                try:
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    output_md_path = os.path.join(specific_output_dir, f"{base_name}.md")
                    
                    response = self.plain_agent.run(
                        "Extract the image contents. Do not say anything extra, such as 'Here is the content:'. "
                        "First, decide which elements are text only - here you will just extract it exactly as it is. "
                        "Preserve Tables in Markdown format. Make sure tables are wide enough - add about 10 spaces extra always. "
                        "If there are pictures, graphs, diagrams or other non-text elements, describe them in great technical detail (100-300 words). "
                        "Focus on the key takeaway of what is shown. Ensure to keep the order of the elements as they are in the input. "
                        "Do not summarize what is shown in the input, other than pictures, graphs, diagrams or other non-text elements. "
                        "Text will be used as input for further processing, so structure it as well as possible.",
                        images=[Image(filepath=image_path)]
                    )
                    _00_utils.update_token_counters(response) # Added token counting

                    tokens_metrics = response.metrics if hasattr(response, 'metrics') else {}
                    page_input_tokens = tokens_metrics.get('input_tokens', [0])[0] if tokens_metrics.get('input_tokens') else 0
                    page_output_tokens = tokens_metrics.get('output_tokens', [0])[0] if tokens_metrics.get('output_tokens') else 0
                    
                    page_md_content_extracted = response.content
                    with open(output_md_path, "w", encoding="utf-8") as f:
                        f.write(page_md_content_extracted)
                    logger.info(f"Successfully extracted markdown for {base_name}")
                except Exception as e:
                    logger.error(f"Error extracting markdown for {image_path}: {str(e)}")
                    extraction_error = True
                
                md_extraction_results.append({
                    'content': page_md_content_extracted,
                    'input_tokens': page_input_tokens,
                    'output_tokens': page_output_tokens,
                    'error': extraction_error
                })
            
            # Generate document title from page summaries (pass only content strings)
            md_content_strings_for_title = [res['content'] for res in md_extraction_results]
            document_title = self.generate_document_title(md_content_strings_for_title, pdf_identifier)
            
            # Process each page with the document title
            logger.info("Processing pages sequentially...")
            contents_dict = {}
            
            for i, (image_path, image_b64_content) in enumerate(image_data_list):
                page_number = i + 1
                key = f"page_{page_number:03d}"
                
                # Use the extracted markdown and tokens from the first pass
                extraction_result = md_extraction_results[i] if i < len(md_extraction_results) else {'content': "", 'input_tokens': 0, 'output_tokens': 0, 'error': True}
                page_md_content = extraction_result['content']
                plain_agent_input_tokens = extraction_result['input_tokens']
                plain_agent_output_tokens = extraction_result['output_tokens']
                current_image_b64 = image_b64_content
                
                # Process page directly
                try:
                    logger.info(f"Processing page {page_number}...")
                    
                    # Fallback for markdown extraction is removed. 
                    # We now rely on the md_content from md_extraction_results.
                    # plain_agent_input_tokens and plain_agent_output_tokens are from the first pass.

                    # Process the markdown content to get structured data
                    start_time = time.time()
                    prompt = (
                        "Given the markdown text below and the document title, extract and return a JSON object with the following fields: "
                        f"document_title (use: '{document_title}'), summary (a short summary of this specific page within 300 characters), "
                        "and hashtags (a list of key search words without the '#' symbol). The Hashtags shall focus on the technical details, not highlevel."
                        "Markdown Input:\n" + page_md_content
                    )
                    
                    response_structured = self.structured_agent.run(prompt)
                    _00_utils.update_token_counters(response_structured) # Added token counting
                    processing_duration = time.time() - start_time
                    
                    json_output = response_structured.content
                    if isinstance(json_output, str):
                        json_output = json.loads(json_output)
                        
                    # Ensure document_title is correct
                    if isinstance(json_output, dict):
                        json_output["document_title"] = document_title
                        meta_data = AgnoPageMetadata.model_validate(json_output)
                    else:
                        # If it's already a Pydantic model, create a new one with updated document_title
                        meta_data = AgnoPageMetadata(
                            document_title=document_title,
                            summary=getattr(json_output, "summary", ""),
                            hashtags=getattr(json_output, "hashtags", [])
                        )
                    
                    tokens_metrics_structured = response_structured.metrics if hasattr(response_structured, 'metrics') else {}
                    input_tokens_structured = tokens_metrics_structured.get('input_tokens', [0])[0] if tokens_metrics_structured.get('input_tokens') else 0
                    output_tokens_structured = tokens_metrics_structured.get('output_tokens', [0])[0] if tokens_metrics_structured.get('output_tokens') else 0
                    
                    # Combine tokens from the first-pass plain_agent and the structured_agent
                    total_input_tokens = plain_agent_input_tokens + input_tokens_structured
                    total_output_tokens = plain_agent_output_tokens + output_tokens_structured
                    
                    page_data = PageExtractedData(
                        **meta_data.model_dump(),
                        pdf_identifier=pdf_identifier,
                        page_number=page_number,
                        md_content=page_md_content,
                        image_b64=current_image_b64,
                        input_tokens=total_input_tokens, # Use combined tokens
                        output_tokens=total_output_tokens, # Use combined tokens
                        processing_duration=processing_duration,
                        error_flag=False, # Or consider extraction_result['error']
                        timestamp=datetime.now(tz=timezone.utc).isoformat()
                    )
                    contents_dict[key] = page_data
                    logger.info(f"Successfully processed page {page_number}")
                    
                except Exception as e:
                    logger.error(f"Error processing page {page_number}: {str(e)}")
                    # Create fallback data for this page
                    fallback_data = PageExtractedData(
                        document_title=document_title,
                        summary="",
                        hashtags=[],
                        pdf_identifier=pdf_identifier,
                        page_number=page_number,
                        md_content=page_md_content, # Content from first pass
                        image_b64=current_image_b64,
                        input_tokens=plain_agent_input_tokens, # Tokens from first pass plain_agent
                        output_tokens=plain_agent_output_tokens, # Tokens from first pass plain_agent
                        processing_duration=0.0,
                        error_flag=True,
                        timestamp=datetime.now(tz=timezone.utc).isoformat()
                    )
                    contents_dict[key] = fallback_data
                
            combined_data = {
                "pdf_identifier": pdf_identifier,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "pages": {key: page_data.model_dump() for key, page_data in sorted(contents_dict.items())}
            }
            combined_file = os.path.join(specific_output_dir, f"{os.path.splitext(pdf_identifier)[0]}_combined.json") # Save combined JSON to specific_output_dir
            with open(combined_file, "w", encoding="utf-8") as f:
                json.dump(combined_data, f, indent=4)
            logger.info(f"All extracted page data combined and saved to {combined_file}")
            return combined_file
        except Exception as e:
            logger.error(f"Critical error in pdf_to_structured_json: {e}")
            traceback.print_exc()
            return None

# ---------------------------------------------------------------------
# Section 5: Main Entry Point
# ---------------------------------------------------------------------

# Removed if __name__ == '__main__' block as this script is imported as a module.
