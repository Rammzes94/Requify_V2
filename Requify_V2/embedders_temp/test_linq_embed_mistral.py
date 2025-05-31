"""
Test script for gte-Qwen2-1.5B-instruct embedding model.

This script loads the Alibaba-NLP/gte-Qwen2-1.5B-instruct model from sentence-transformers,
encodes example queries and documents, and computes similarity scores between them.
Results are logged with icons. This script is for temporary/experimental use only.


{
  "Model Name": "gte-Qwen2-1.5B-instruct",
  "Link": "https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct",
  "Contextual Note": "⚠️ NA",
  "Model ID": 11,
  "Parameters": "1B",
  "Sequence Length": 8960,
  "Context Length": 32768,
  "Average Score": 59.45,
  "MTEB Scores": {
    "Classification": 52.69,
    "Clustering": 62.51,
    "Pair Classification": 58.32,
    "Retrieval": 52.05
  },
  "MTEB Average": 0.74,
  "BEIR Score": 24.02,
  "BBQ Score": 81.58,
  "SQuAD Score": 62.58,
  "HellaSwag Score": 60.78,
  "ARC Score": 71.61
}



"""

import os
import sys
import logging
from dotenv import load_dotenv

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import setup_logging, get_logger, update_token_counters, get_token_usage, print_token_usage, reset_token_counters, setup_project_directory, generate_timestamp
setup_project_directory()

# Load environment variables
load_dotenv()

# =====================
# Constants
# =====================
MODEL_NAME = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
MAX_SEQ_LENGTH = 32768  # manually set if you need more than 8960

MODEL_NAME = "intfloat/multilingual-e5-large-instruct"



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

from sentence_transformers import SentenceTransformer
import torch

device = "cpu"
logging.info("🔄 Loading model: %s on device: %s", MODEL_NAME, device)
model = SentenceTransformer(MODEL_NAME, device=device, trust_remote_code=True)
model.max_seq_length = MAX_SEQ_LENGTH
logging.info("✅ Model loaded successfully.")



# Get the tokenizer max sequence length
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True)
print(tokenizer.model_max_length)


# =====================
# Token Counting
# =====================

from transformers import AutoTokenizer

def count_tokens_qwen(text: str) -> int:
    """
    Counts the number of tokens in the given text using the Qwen2 tokenizer.

    Returns:
        int: Number of tokens.
    """
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B", trust_remote_code=True)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)

# Example usage
if __name__ == "__main__":
    with open("embedders_temp/page_001.md", "r", encoding="utf-8") as f:
        text = f.read()

    token_count = count_tokens_qwen(text)
    print(f"Token count: {token_count}")




# =====================
# Embed All TXT Files & Log Pairwise Similarities
# =====================

import glob
import numpy as np
import time

# Find all .txt files in embedders_temp
TXT_DIR = os.path.dirname(os.path.abspath(__file__))
txt_files = glob.glob(os.path.join(TXT_DIR, '*.txt'))

if not txt_files:
    logging.warning("❌ No .txt files found in embedders_temp.")
else:
    logging.info(f"🔄 Found {len(txt_files)} .txt files for embedding.")
    file_texts = []
    file_names = []
    file_times = []
    total_start = time.time()
    for file_path in txt_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_texts.append(f.read())
                file_names.append(os.path.basename(file_path))
        except Exception as e:
            logging.error(f"❌ Failed to read {file_path}: {e}")

    # Embed all files one by one to time each
    embeddings = []
    for idx, text in enumerate(file_texts):
        start = time.time()
        emb = model.encode([text], device=device)
        elapsed = time.time() - start
        embeddings.append(emb[0])
        file_times.append(elapsed)
        logging.info(f"✅ Embedding done for '{file_names[idx]}' in {elapsed:.2f} seconds.")
    total_elapsed = time.time() - total_start
    logging.info(f"✅ All file embeddings complete. Total time: {total_elapsed:.2f} seconds.")

    # Compute pairwise cosine similarities
    def cosine_similarity(a, b):
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    n = len(file_names)
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            icon = "♻️" if sim >= 0.99 else ("🔄" if sim >= 0.90 else "🆕")
            logging.info(f"{icon} Similarity between '{file_names[i]}' and '{file_names[j]}': {sim:.4f}")


