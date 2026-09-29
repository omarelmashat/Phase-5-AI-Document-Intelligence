
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().with_name(".env"))


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default

AI_PROVIDER = os.getenv("AI_PROVIDER", "cohere").lower() 
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", AI_PROVIDER).lower()

COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
HF_API_KEY = os.getenv("HF_API_KEY", "")

COHERE_CHAT_MODEL = os.getenv("COHERE_CHAT_MODEL", "command-r-08-2024")
HF_CHAT_MODEL = os.getenv("HF_CHAT_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
COHERE_EMBED_MODEL = os.getenv("COHERE_EMBED_MODEL", "embed-english-v3.0")
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

LLM_TEMPERATURE = _float("LLM_TEMPERATURE", 0.0) 
LLM_MAX_TOKENS = _int("LLM_MAX_TOKENS", 800)

API_MAX_RETRIES = _int("API_MAX_RETRIES", 3)
API_RETRY_DELAY_SECONDS = _float("API_RETRY_DELAY_SECONDS", 1.0)
API_TIMEOUT_SECONDS = _int("API_TIMEOUT_SECONDS", 60)
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE_MB = _int("MAX_FILE_SIZE_MB", 20)
MAX_PDF_PAGES = _int("MAX_PDF_PAGES", 20)

CHUNK_SIZE = _int("CHUNK_SIZE", 800)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 100)
TOP_K = _int("TOP_K", 4)
EMBED_BATCH_SIZE = _int("EMBED_BATCH_SIZE", 64) 
MAX_DISTANCE = _float("MAX_DISTANCE", 1.3)
HISTORY_LIMIT = _int("HISTORY_LIMIT", 6)  
CLASSIFY_CHARS = _int("CLASSIFY_CHARS", 3000) 
MAX_PROMPT_CHARS = _int("MAX_PROMPT_CHARS", 60000) 
TEMP_ROOT_PREFIX = "docintel_"
