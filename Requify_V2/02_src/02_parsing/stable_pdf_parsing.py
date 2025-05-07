# ---------------------------------------------------------------------
# Section 1: Imports and Setup
# ---------------------------------------------------------------------
import os
import sys
import time
import traceback
import json
import logging
from datetime import datetime, timezone
from typing import List, Optional

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
_00_utils.setup_project_directory()

# Load environment variables
load_dotenv()

# ---------------------------------------------------------------------
# Section 2: Configuration Constants
# ---------------------------------------------------------------------
# Target PDF file to process (set to None to process all PDFs in the input directory)
TARGET_PDF_FILE = "fighter_jet_rocket_launcher_spec_3_language_variant.pdf"
# TARGET_PDF_FILE = None  # Uncomment to process all PDFs

# Default paths
RAW_INPUT_DIR = os.path.join("01_input", "raw")
PROCESSED_OUTPUT_BASE_DIR = os.path.join("01_input", "processed")
FILTERED_FILES_JSON = os.path.join("02_src", "filtered_files_by_extension.json")
# DPI setting for PDF to image conversion
PDF_TO_IMAGE_DPI = 300
# Note: We process files sequentially to avoid pickling errors with agent objects

# ---------------------------------------------------------------------
# Section 3: Pydantic Models for AI bot and for final extracted data
# ---------------------------------------------------------------------
class AgnoPageMetadata(BaseModel):
    document_title: str = Field(..., description="Title for the entire document, derived from first 10 pages.")
    summary: str = Field(..., description="Short summary of the page content, 200-300 characters.")
    hashtags: List[str] = Field(..., description="Key search words for the content (without '#').")

class PageExtractedData(AgnoPageMetadata):
    pdf_identifier: str = Field(..., description="Identifier/source filename of the PDF file.")
    page_number: int = Field(..., description="Page number in the PDF file.")
    md_content: str = Field(..., description="Markdown content extracted from the page.")
    input_tokens: int = Field(..., description="Number of tokens provided as input.")
    output_tokens: int = Field(..., description="Number of tokens generated as output.")
    processing_duration: float = Field(..., description="Duration (in seconds) of processing the page.")
    error_flag: bool = Field(..., description="Indicates if an error occurred during processing.")
    timestamp: str = Field(..., description="ISO timestamp when processing completed.")

# ---------------------------------------------------------------------
# Section 4: Agent Initialization
# ---------------------------------------------------------------------
# Note: The api_key variable is used by both agent types as in your original code.
api_key = os.getenv("OPENAI_API_KEY")
api_key = os.getenv("GROQ_API_KEY")

# Initialize agents for vision and text processing
vision_model = OpenAIChat(id="gpt-4o", api_key=api_key)
text_model = OpenAIChat(id="gpt-4o-mini", api_key=api_key)

# Overwrite with Groq models as in the original code
vision_model = Groq(id="meta-llama/llama-4-scout-17b-16e-instruct", api_key=api_key)
text_model = Groq(id="llama-3.3-70b-versatile", api_key=api_key)

# Plain agent for PDF parsing (no structured outputs)
plain_agent = Agent(
    model=vision_model,
    markdown=True,
    debug_mode=False,
    structured_outputs=False,
    description=(
        "You are an agent that extracts contents from an image that originated from a pdf file. "
        "You want to extract as much information as possible - not only focussing on text but also images, graphs, diagrams, etc. "
        "You are very technical and precise."
    )
)

# Structured agent to generate metadata from extracted markdown.
structured_agent = Agent(
    model=text_model,
    markdown=False,
    debug_mode=False,
    structured_outputs=True,
    description=(
        "You are an agent that receives plain markdown text extracted from a PDF page and must generate "
        "a JSON object with the following fields: title (a short title for the page), summary (a short summary within 300 characters), "
        "and hashtags (a list of key search words without the '#' symbol, about 10). "
    ),
    response_model=AgnoPageMetadata,
    use_json_mode=False
)

# Add a new agent for document title generation
document_title_agent = Agent(
    model=text_model,
    markdown=False,
    debug_mode=False,
    structured_outputs=False,
    description=(
        "You are an agent that receives combined markdown text from the first pages of a PDF document and generates "
        "a concise, descriptive title that captures the main subject and purpose of the document. "
        "The title should be between 5-10 words and clearly identify what the document is about."
    )
)

# ---------------------------------------------------------------------
# Section 4: PDF Processing Controller Class
# ---------------------------------------------------------------------
class PDFProcessor:
    def __init__(self, plain_agent: Agent, structured_agent: Agent, document_title_agent: Agent):
        # Setup logging configuration
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.plain_agent = plain_agent
        self.structured_agent = structured_agent
        self.document_title_agent = document_title_agent

    # -----------------------------------------------------------------
    # Method: Convert PDF to Images
    # -----------------------------------------------------------------
    def pdf_to_images(self, pdf_path, dpi=PDF_TO_IMAGE_DPI, output_images_dir=None):
        logging.info(f"🔄 Processing '{pdf_path}' to images...")
        if not os.path.exists(pdf_path):
            logging.error(f"❌ PDF file '{pdf_path}' does not exist.")
            return []
        try:
            pdf = fitz.open(pdf_path)
            image_paths = []
            if output_images_dir is None:
                base_name = os.path.splitext(os.path.basename(pdf_path))[0]
                # Use the central processed directory
                output_images_dir = os.path.join(PROCESSED_OUTPUT_BASE_DIR, "pdf_images", base_name)
            os.makedirs(output_images_dir, exist_ok=True)
            for i, page in enumerate(pdf):
                zoom = dpi / 72  # Default PDF DPI is 72
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                page_name = f"page_{i+1:03d}"
                image_path = os.path.join(output_images_dir, f"{page_name}.png")
                pix.save(image_path)
                logging.info(f"✅ Saved image for page {i+1} to {image_path}")
                image_paths.append(image_path)
            logging.info(f"✅ Generated {len(image_paths)} images from PDF")
            return image_paths
        except Exception as e:
            logging.error(f"❌ Error converting PDF to images: {e}")
            traceback.print_exc()
            return []

    def generate_document_title(self, md_contents, pdf_identifier):
        """Generate a title for the entire document based on page summaries"""
        logging.info(f"🔄 Generating document title for {pdf_identifier} from page summaries...")
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
                    summary = response.content
                    if isinstance(summary, str):
                        page_summaries.append(summary)
                    else:
                        # If we got a structured response, try to extract summary
                        summary = summary.get('summary', '') if hasattr(summary, 'get') else str(summary)
                        page_summaries.append(summary)
                except Exception as e:
                    logging.error(f"❌ Error generating summary for page {i+1}: {str(e)}")
            
            # Combine summaries for title generation
            combined_summaries = "\n\n".join(page_summaries)
            
            if not combined_summaries:
                logging.warning(f"⚠️ No summaries available for document title generation: {pdf_identifier}")
                return f"Document {pdf_identifier}"
                
            # Generate title from summaries
            prompt = (
                "Based on the following summaries from the first pages of a document, "
                "create a concise and descriptive title that captures the main subject "
                "and purpose of the document. The title should be between 5-10 words.\n\n"
                f"Page summaries:\n{combined_summaries}"
            )
            
            response = self.document_title_agent.run(prompt)
            document_title = response.content.strip()
            logging.info(f"✅ Generated document title from summaries: {document_title}")
            return document_title
        except Exception as e:
            logging.error(f"❌ Error generating document title: {str(e)}")
            return f"Document {pdf_identifier}"

    # -----------------------------------------------------------------
    # Method: Process PDF to Structured JSON
    # -----------------------------------------------------------------
    def pdf_to_structured_json(self, pdf_path, output_dir="02_src/02_parsing/parsed_content"):
        try:
            os.makedirs(output_dir, exist_ok=True)
            image_paths = self.pdf_to_images(pdf_path)
            if not image_paths:
                logging.error("❌ No images were created from the PDF. Process failed.")
                return None
            logging.info(f"✅ Successfully converted PDF to {len(image_paths)} images")
            pdf_identifier = os.path.basename(pdf_path)
            
            # First, extract markdown content from each page
            md_contents = []
            for i, image_path in enumerate(image_paths):
                try:
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    output_md_path = os.path.join(output_dir, f"{base_name}.md")
                    
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
                    
                    md_content = response.content
                    with open(output_md_path, "w", encoding="utf-8") as f:
                        f.write(md_content)
                    md_contents.append(md_content)
                    logging.info(f"✅ Successfully extracted markdown for {base_name}")
                except Exception as e:
                    logging.error(f"❌ Error extracting markdown for {image_path}: {str(e)}")
                    md_contents.append("")
            
            # Generate document title from page summaries
            document_title = self.generate_document_title(md_contents, pdf_identifier)
            
            # Process each page with the document title
            logging.info(f"🔄 Processing pages sequentially...")
            contents_dict = {}
            
            for i, image_path in enumerate(image_paths):
                page_number = i + 1
                key = f"page_{page_number:03d}"
                
                # Use the extracted markdown if available, otherwise will be an empty string
                page_md_content = md_contents[i] if i < len(md_contents) else ""
                
                # Process page directly
                try:
                    logging.info(f"🔄 Processing page {page_number}...")
                    
                    # Skip markdown extraction if it's already done above
                    if not page_md_content:
                        try:
                            base_name = os.path.splitext(os.path.basename(image_path))[0]
                            output_md_path = os.path.join(output_dir, f"{base_name}.md")
                            
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
                            
                            tokens_metrics = response.metrics if hasattr(response, 'metrics') else {}
                            
                            input_tokens = tokens_metrics.get('input_tokens', [0])[0] if tokens_metrics.get('input_tokens') else 0
                            output_tokens = tokens_metrics.get('output_tokens', [0])[0] if tokens_metrics.get('output_tokens') else 0
                            
                            page_md_content = response.content
                            with open(output_md_path, "w", encoding="utf-8") as f:
                                f.write(page_md_content)
                        except Exception as e:
                            logging.error(f"❌ Error extracting markdown for {image_path}: {str(e)}")
                            input_tokens = 0
                            output_tokens = 0
                    else:
                        input_tokens = 0
                        output_tokens = 0
                    
                    # Process the markdown content to get structured data
                    start_time = time.time()
                    prompt = (
                        "Given the markdown text below and the document title, extract and return a JSON object with the following fields: "
                        f"document_title (use: '{document_title}'), summary (a short summary of this specific page within 300 characters), "
                        "and hashtags (a list of key search words without the '#' symbol). "
                        "Markdown Input:\n" + page_md_content
                    )
                    
                    response = self.structured_agent.run(prompt)
                    processing_duration = time.time() - start_time
                    
                    json_output = response.content
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
                    
                    tokens_metrics = response.metrics if hasattr(response, 'metrics') else {}
                    input_tokens_structured = tokens_metrics.get('input_tokens', [0])[0] if tokens_metrics.get('input_tokens') else 0
                    output_tokens_structured = tokens_metrics.get('output_tokens', [0])[0] if tokens_metrics.get('output_tokens') else 0
                    total_input_tokens = input_tokens + input_tokens_structured
                    total_output_tokens = output_tokens + output_tokens_structured
                    
                    page_data = PageExtractedData(
                        **meta_data.model_dump(),
                        pdf_identifier=pdf_identifier,
                        page_number=page_number,
                        md_content=page_md_content,
                        input_tokens=total_input_tokens,
                        output_tokens=total_output_tokens,
                        processing_duration=processing_duration,
                        error_flag=False,
                        timestamp=datetime.now(tz=timezone.utc).isoformat()
                    )
                    contents_dict[key] = page_data
                    logging.info(f"✅ Successfully processed page {page_number}")
                    
                except Exception as e:
                    logging.error(f"❌ Error processing page {page_number}: {str(e)}")
                    # Create fallback data for this page
                    fallback_data = PageExtractedData(
                        document_title=document_title,
                        summary="",
                        hashtags=[],
                        pdf_identifier=pdf_identifier,
                        page_number=page_number,
                        md_content=page_md_content if page_md_content else "",
                        input_tokens=0,
                        output_tokens=0,
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
            combined_file = os.path.join(output_dir, f"{os.path.splitext(pdf_identifier)[0]}_combined.json")
            with open(combined_file, "w", encoding="utf-8") as f:
                json.dump(combined_data, f, indent=4)
            logging.info(f"✅ All extracted page data combined and saved to {combined_file}")
            return combined_file
        except Exception as e:
            logging.error(f"❌ Critical error in pdf_to_structured_json: {e}")
            traceback.print_exc()
            return None

# ---------------------------------------------------------------------
# Section 5: Main Entry Point
# ---------------------------------------------------------------------
if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    processor = PDFProcessor(plain_agent, structured_agent, document_title_agent)
    
    if TARGET_PDF_FILE is None:
        # Process all PDFs from the filtered files list
        logging.info("🔄 Processing all PDFs in the filtered files list...")
        
        if not os.path.exists(FILTERED_FILES_JSON):
            logging.error(f"❌ Filtered files list not found: {FILTERED_FILES_JSON}")
            sys.exit(1)
            
        try:
            with open(FILTERED_FILES_JSON, "r") as f:
                pre_filtered = json.load(f)
                
            if ".pdf" not in pre_filtered or not pre_filtered[".pdf"]:
                logging.warning("⚠️ No PDF files found in the filtered files list.")
                sys.exit(0)
                
            logging.info(f"✅ Found {len(pre_filtered['.pdf'])} PDF files to process")
            
            for pdf_info in pre_filtered[".pdf"]:
                pdf_file = os.path.normpath(pdf_info["filename"])
                if not os.path.exists(pdf_file):
                    logging.error(f"❌ PDF file not found: {pdf_file}")
                    continue
                logging.info(f"🔄 Processing PDF: {pdf_file}")
                
                # Prepare output directory
                base_name = os.path.splitext(os.path.basename(pdf_file))[0]
                subfolder_path = os.path.join(PROCESSED_OUTPUT_BASE_DIR, "parsed_content", base_name)
                os.makedirs(subfolder_path, exist_ok=True)
                
                # Process the PDF
                output_file = processor.pdf_to_structured_json(
                    pdf_file, 
                    output_dir=subfolder_path
                )
                
                # Report results
                if output_file and os.path.exists(output_file):
                    logging.info(f"✅ Process completed successfully. Output saved to: {output_file}")
                else:
                    logging.error(f"❌ Process failed for {pdf_file}: No output file was generated.")
                    
        except Exception as e:
            logging.error(f"❌ Error processing files: {str(e)}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Process a single specified PDF file
        logging.info(f"🔄 Processing single PDF file: {TARGET_PDF_FILE}")
        
        # Construct the full path to the target PDF file
        pdf_file = os.path.join(RAW_INPUT_DIR, TARGET_PDF_FILE)
        
        # Check if file exists
        if not os.path.exists(pdf_file):
            logging.error(f"❌ Target PDF file not found: {pdf_file}")
            sys.exit(1)
            
        logging.info(f"🔄 Processing PDF: {pdf_file}")
        
        # Prepare output directory
        base_name = os.path.splitext(os.path.basename(pdf_file))[0]
        subfolder_path = os.path.join(PROCESSED_OUTPUT_BASE_DIR, "parsed_content", base_name)
        os.makedirs(subfolder_path, exist_ok=True)
        
        # Process the PDF
        output_file = processor.pdf_to_structured_json(
            pdf_file, 
            output_dir=subfolder_path
        )
        
        # Report results
        if output_file and os.path.exists(output_file):
            logging.info(f"✅ Process completed successfully. Output saved to: {output_file}")
        else:
            logging.error(f"❌ Process failed for {pdf_file}: No output file was generated.")