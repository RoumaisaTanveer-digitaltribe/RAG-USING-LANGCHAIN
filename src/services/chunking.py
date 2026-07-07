
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks, tagging each with a chunk_id."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # Assign a stable, human-readable chunk id per source file so citations
    # like "hr_policy.txt#chunk-3" are reproducible and debuggable.
    counters = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        counters[source] = counters.get(source, -1) + 1
        chunk.metadata["chunk_id"] = counters[source]

    return chunks
