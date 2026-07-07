import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass  # not frozen: tests monkeypatch fields like vector_store_path
class Settings:
    # --- LLM via OpenRouter ---
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "openrouter/free")

    # --- Embeddings via OpenRouter (OpenAI-compatible) ---
    openrouter_embed_model: str = os.getenv(
        "OPENROUTER_EMBED_MODEL",
        "openai/text-embedding-3-large",
    )
    openrouter_base_url: str = os.getenv(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1",
    )

    # --- Embeddings (legacy local, optional) ---
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    # --- Vector store ---
    vector_store_path: str = os.getenv("VECTOR_STORE_PATH", "./vector_store")
    vector_store_collection: str = os.getenv("VECTOR_STORE_COLLECTION", "rag_docs")

    # --- Chunking ---
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    # --- Retrieval ---
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "4"))
    not_found_distance_threshold: float = float(
        os.getenv("NOT_FOUND_DISTANCE_THRESHOLD", "1.1")
    )

    # --- App ---
    data_dir: str = os.getenv("DATA_DIR", "./data/sample_inputs")
    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = int(os.getenv("APP_PORT", "8000"))


settings = Settings()