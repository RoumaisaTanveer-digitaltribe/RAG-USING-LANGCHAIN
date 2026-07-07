
import argparse
import sys

from src.config import settings
from src.services.document_loader import load_documents_from_dir
from src.services.chunking import chunk_documents
from src.services.vectorstore import build_vectorstore


def run_ingestion(data_dir: str) -> int:
    print(f"[ingest] Loading documents from: {data_dir}")
    documents = load_documents_from_dir(data_dir)
    print(f"[ingest] Loaded {len(documents)} document page(s)/file(s).")

    print("[ingest] Splitting into chunks...")
    chunks = chunk_documents(documents)
    print(f"[ingest] Produced {len(chunks)} chunks "
          f"(chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap}).")

    print(f"[ingest] Embedding with '{settings.embedding_model}' and building FAISS index...")
    build_vectorstore(chunks)
    print(f"[ingest] Vector store saved to: {settings.vector_store_path}")
    return len(chunks)


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG vector store.")
    parser.add_argument(
        "--data-dir", default=settings.data_dir,
        help="Directory containing PDF/TXT/MD source documents.",
    )
    args = parser.parse_args()
    try:
        run_ingestion(args.data_dir)
    except Exception as exc:
        print(f"[ingest] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
