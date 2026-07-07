import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from src.config import settings

_embeddings_singleton: Optional[OpenAIEmbeddings] = None


def get_embeddings() -> OpenAIEmbeddings:
    """
    Singleton embeddings instance using OpenRouter's OpenAI-compatible API.

    Avoids local Torch/torchaudio; all embedding computation is hosted. [web:53][web:78]
    """
    global _embeddings_singleton
    if _embeddings_singleton is None:
        _embeddings_singleton = OpenAIEmbeddings(
            model=settings.openrouter_embed_model,
            openai_api_key=settings.openrouter_api_key,
            openai_api_base=settings.openrouter_base_url,
        )
    return _embeddings_singleton


def build_vectorstore(chunks: List[Document]) -> FAISS:
    """Build a fresh FAISS index from chunks and persist it to disk."""
    if not chunks:
        raise ValueError("Cannot build a vector store from zero chunks.")

    embeddings = get_embeddings()
    store = FAISS.from_documents(chunks, embeddings)
    os.makedirs(settings.vector_store_path, exist_ok=True)
    store.save_local(
        settings.vector_store_path,
        index_name=settings.vector_store_collection,
    )
    return store


def load_vectorstore() -> FAISS:
    """Load a previously persisted FAISS index from disk."""
    index_file = os.path.join(
        settings.vector_store_path,
        f"{settings.vector_store_collection}.faiss",
    )
    if not os.path.exists(index_file):
        raise FileNotFoundError(
            f"No vector store found at {index_file}. Run ingestion first "
            f"(python -m src.ingest)."
        )
    embeddings = get_embeddings()
    return FAISS.load_local(
        settings.vector_store_path,
        embeddings,
        index_name=settings.vector_store_collection,
        allow_dangerous_deserialization=True,
    )


def vectorstore_exists() -> bool:
    index_file = os.path.join(
        settings.vector_store_path,
        f"{settings.vector_store_collection}.faiss",
    )
    return os.path.exists(index_file)