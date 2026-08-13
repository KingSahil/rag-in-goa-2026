import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    sarvam_key: str = os.getenv("SARVAM_API_KEY", "")
    sarvam_chat_model: str = os.getenv("SARVAM_CHAT_MODEL", "sarvam-30b")
    embed_model: str = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
    index_dir: str = os.getenv("INDEX_DIR", "../indexes")
    top_k_dense: int = int(os.getenv("TOP_K_DENSE", "8"))
    top_k_bm25: int = int(os.getenv("TOP_K_BM25", "8"))
    top_k_final: int = int(os.getenv("TOP_K_FINAL", "5"))
    gen_max_tokens: int = int(os.getenv("GEN_MAX_TOKENS", "96"))
    gen_timeout_s: float = float(os.getenv("GEN_TIMEOUT_S", "0.18"))
    rag_timeout_s: float = float(os.getenv("RAG_TIMEOUT_S", "0.20"))

settings = Settings()
