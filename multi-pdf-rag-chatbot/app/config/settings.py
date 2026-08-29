from pathlib import Path
from dotenv import load_dotenv
import os

# Load .env
load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Load .env
load_dotenv()

# Project Root
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Project Name
PROJECT_NAME = os.getenv("PROJECT_NAME")

# Embedding
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))

# Retrieval
TOP_K = int(os.getenv("TOP_K"))

# Paths
PDF_DIR = BASE_DIR / os.getenv("PDF_DIR")
VECTOR_DB_DIR = BASE_DIR / os.getenv("VECTOR_DB")

VECTOR_DB_EXISTS = VECTOR_DB_DIR.exists()