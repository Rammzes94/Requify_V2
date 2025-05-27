"""
Test script for gte-Qwen2-1.5B-instruct embedding model.

This script loads the Alibaba-NLP/gte-Qwen2-1.5B-instruct model from sentence-transformers,
encodes example queries and documents, and computes similarity scores between them.
Results are logged with icons. This script is for temporary/experimental use only.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    import _00_utils
    _00_utils.setup_project_directory()
except ImportError:
    pass  # If _00_utils is not available, skip project setup

# Load environment variables
load_dotenv()

# =====================
# Constants
# =====================
MODEL_NAME = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
MAX_SEQ_LENGTH = 8192
QUERIES = [
    "how much protein should a female eat",
    "summit define",
]
DOCUMENTS = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
]

# =====================
# Logging Setup
# =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# =====================
# Main Logic
# =====================
try:
    from sentence_transformers import SentenceTransformer
    import torch

    device = "cpu"
    logging.info("🔄 Loading model: %s on device: %s", MODEL_NAME, device)
    model = SentenceTransformer(MODEL_NAME, device=device, trust_remote_code=True)
    model.max_seq_length = MAX_SEQ_LENGTH
    logging.info("✅ Model loaded successfully.")

    # Encode the queries and documents
    logging.info("🔄 Encoding queries and documents on CPU...")
    query_embeddings = model.encode(QUERIES, prompt_name="query", device=device)
    document_embeddings = model.encode(DOCUMENTS, device=device)
    logging.info("✅ Encodings complete.")

    # Compute the (cosine) similarity scores using matrix multiplication
    logging.info("🔄 Computing similarity scores...")
    scores = (query_embeddings @ document_embeddings.T) * 100
    logging.info("✅ Similarity scores computed.")
    logging.info("🔎 Similarity scores (as percentage): %s", scores.tolist())

except Exception as e:
    logging.error("❌ Error during embedding or similarity computation: %s", e) 