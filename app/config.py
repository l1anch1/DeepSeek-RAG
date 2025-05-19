import os
from pathlib import Path

# Device settings
DEVICE = os.getenv("DEVICE", "cpu")

# Data directories
BASE_DIR = str(Path(__file__).resolve().parent.parent)
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", os.path.join(DATA_DIR, "documents"))
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", os.path.join(DATA_DIR, "vectordb"))

# LLM model settings
USE_OPENAI = True
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")  
OPENAI_LLM_MODEL = os.environ.get("OPENAI_LLM_MODEL", "gpt-4o") 
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "deepseek-r1:14b")

# Embedding settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-zh-v1.5")
MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.1"))
MODEL_TOP_P = float(os.getenv("MODEL_TOP_P", "0.6"))
MODEL_NUM_CTX = int(os.getenv("MODEL_NUM_CTX", "8192"))
MODEL_NUM_PREDICT = int(os.getenv("MODEL_NUM_PREDICT", "2048"))
MODEL_NUM_THREAD = int(os.getenv("MODEL_NUM_THREAD", "12"))

# Chunking settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Retrieval settings
SEARCH_K = int(
    os.getenv("SEARCH_K", "8")
)  # Number of documents to return during retrieval
