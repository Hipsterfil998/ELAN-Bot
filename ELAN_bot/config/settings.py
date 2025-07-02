"""
Configuration settings for ELAN-Bot application.
"""

import os
from pathlib import Path

# Base paths - adjusted for HF Spaces structure
BASE_DIR = Path(__file__).parent.parent

# API Configuration - Hugging Face Spaces compatible
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN", "")

# Model Configuration
DEFAULT_ENCODER_MODEL = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_LLM_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_TOKENIZER_MODEL = "gpt-4o-mini"

# Vector Database Configuration - HF Spaces compatible  
# Database is directly in the project root, not in data/ subdirectory
QDRANT_CLIENT_PATH = str(BASE_DIR / "qdrant_data")
# Ensure the qdrant directory exists
Path(QDRANT_CLIENT_PATH).mkdir(parents=True, exist_ok=True)

# Collection name matches your actual data structure
# Note: Qdrant automatically manages collections/ subdirectory
COLLECTION_NAME = "ELAN_docs_pages"  # This will map to qdrant_data/collections/ELAN_docs_pages/
SEARCH_LIMIT = 3

# Text Processing Configuration
CHUNK_SIZE = 2048
MAX_TOKENS = 4096
TEMPERATURE = 0.1
ANSWER_MAX_TOKENS = 500
ANSWER_TARGET_WORDS = 120

# UI Configuration
APP_TITLE = "ELAN-Bot"
APP_DESCRIPTION = """Hello there!👋\nI'm ELAN-Bot, a virtual assistant designed to help you with the ELAN annotation software. You can ask me questions about:\n
- 📚 software usage: how to use ELAN and its main features
- 💻 XML code: modify the EAF file by providing me with the copy-pasted XML code and some instructions (e.g --> instructions: change the participant name from Eleonora to Gianni. [EAF file])\n
Software usage functionality available in English, Spanish, Italian, French and German
Based on Llama 3.3 70B"""

APP_EXAMPLES = [
    "How can I add a new tier in ELAN?",
    "¿Cómo puedo exportar anotaciones en formato txt?",
    "Come posso cercare all'interno delle annotazioni?"
]

TEXTBOX_PLACEHOLDER = "Ask me anything about ELAN..."