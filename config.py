import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chromadb"
SQLITE_DB_PATH = DATA_DIR / "rag_app.db"
STANDALONE_DIR = DATA_DIR / "standalone_exports"

# Create required directories
for directory in [DATA_DIR, UPLOADS_DIR, CHROMA_DIR, STANDALONE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# LLM & Embedding configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()

# Ollama settings (local, no API key needed)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b").strip()

# Determine active LLM provider: 'openai', 'anthropic', 'ollama', or 'offline'
configured_provider = os.getenv("LLM_PROVIDER", "auto").strip().lower()
if configured_provider == "auto":
    if OPENAI_API_KEY:
        LLM_PROVIDER = "openai"
    elif ANTHROPIC_API_KEY:
        LLM_PROVIDER = "anthropic"
    else:
        LLM_PROVIDER = "ollama"  # Try local Ollama before falling back to offline
else:
    LLM_PROVIDER = configured_provider

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")

# Chunking & Retrieval settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "4"))
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "pdf_knowledge_base")

# Server settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
MAX_CONCURRENT_USERS = int(os.getenv("MAX_CONCURRENT_USERS", "50"))
