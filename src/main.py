
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field

from src.config import settings
from src.services.vectorstore import vectorstore_exists
from src.services.rag_chain import answer_question
from src.ingest import run_ingestion
from src.services.document_loader import SUPPORTED_EXTENSIONS

app = FastAPI(
    title="RAG Chatbot API",
    description="Document-grounded Q&A chatbot built with LangChain + FAISS.",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question.")


class SourceItem(BaseModel):
    file: str
    page: int | None = None
    chunk_id: int | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    confidence: str


class IngestResponse(BaseModel):
    status: str
    chunks_indexed: int


@app.get("/health")
def health():
    return {"status": "ok", "vector_store_ready": vectorstore_exists()}


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    if not vectorstore_exists():
        raise HTTPException(
            status_code=409,
            detail="Vector store not built yet. Call POST /ingest first "
                   "(or run `python -m src.ingest`).",
        )
    try:
        result = answer_question(payload.question)
    except RuntimeError as exc:
        # e.g. missing GOOGLE_API_KEY
        raise HTTPException(status_code=500, detail=str(exc))
    return result.to_dict()


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    try:
        n_chunks = run_ingestion(settings.data_dir)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "success", "chunks_indexed": n_chunks}


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )
    dest_dir = Path(settings.data_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / file.filename
    with dest_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return {
        "status": "uploaded",
        "file": file.filename,
        "note": "Call POST /ingest to (re)index the vector store with this new file.",
    }
