"""Configuration for the Event RAG Agent."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"

# FAISS Configuration
FAISS_INDEX_PATH = "faiss_event_index"

# Text Splitting Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval Configuration
TOP_K_DOCUMENTS = 3

# LLM Configuration
LLM_TEMPERATURE = 0
LLM_MAX_TOKENS = 1024
LLM_TIMEOUT = 30
