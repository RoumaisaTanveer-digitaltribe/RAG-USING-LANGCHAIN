
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_document(file_path: str) -> List[Document]:
    """Load a single file and return LangChain Documents with source metadata."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}' for {path.name}. "
            f"Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if ext == ".pdf":
        loader = PyPDFLoader(str(path))
        docs = loader.load()  # one Document per page, page numbers pre-populated
        for i, d in enumerate(docs):
            d.metadata["source"] = path.name
            # PyPDFLoader gives 0-indexed pages; normalize to human-friendly 1-indexed.
            d.metadata["page"] = d.metadata.get("page", i) + 1
    else:
        loader = TextLoader(str(path), encoding="utf-8")
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = path.name
            d.metadata["page"] = None  # plain text has no page concept

    return docs


def load_documents_from_dir(dir_path: str) -> List[Document]:
    """Load every supported file in a directory (non-recursive)."""
    dir_p = Path(dir_path)
    if not dir_p.exists():
        raise FileNotFoundError(f"Data directory not found: {dir_path}")

    all_docs: List[Document] = []
    files = sorted(
        f for f in dir_p.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise ValueError(
            f"No supported documents ({sorted(SUPPORTED_EXTENSIONS)}) found in {dir_path}"
        )

    for f in files:
        all_docs.extend(load_document(str(f)))

    return all_docs
