
import pytest
from fastapi.testclient import TestClient

from src.services.document_loader import load_document, load_documents_from_dir
from src.services.chunking import chunk_documents
from src.services.vectorstore import build_vectorstore, load_vectorstore, vectorstore_exists
from src.services import rag_chain
from src.services.rag_chain import _confidence_from_scores, answer_question, NOT_FOUND_PHRASE


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def test_load_txt_document(sample_data_dir):
    docs = load_document(f"{sample_data_dir}/remote_work_policy.txt")
    assert len(docs) == 1
    assert docs[0].metadata["source"] == "remote_work_policy.txt"
    assert "Remote Work Policy" in docs[0].page_content


def test_load_pdf_document_has_page_numbers(sample_data_dir):
    docs = load_document(f"{sample_data_dir}/hr_policy.pdf")
    assert len(docs) >= 2
    assert all(d.metadata["page"] >= 1 for d in docs)  # 1-indexed
    assert docs[0].metadata["source"] == "hr_policy.pdf"


def test_load_documents_from_dir(sample_data_dir):
    docs = load_documents_from_dir(sample_data_dir)
    sources = {d.metadata["source"] for d in docs}
    assert "hr_policy.pdf" in sources
    assert "remote_work_policy.txt" in sources


def test_load_unsupported_extension_raises(tmp_path):
    bad_file = tmp_path / "notes.docx"
    bad_file.write_text("hello")
    with pytest.raises(ValueError):
        load_document(str(bad_file))


def test_load_missing_dir_raises():
    with pytest.raises(FileNotFoundError):
        load_documents_from_dir("./this_dir_does_not_exist")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def test_chunking_preserves_source_and_assigns_chunk_id(sample_data_dir):
    docs = load_documents_from_dir(sample_data_dir)
    chunks = chunk_documents(docs)
    assert len(chunks) >= len(docs)
    for c in chunks:
        assert "source" in c.metadata
        assert "chunk_id" in c.metadata


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

def test_build_and_load_vectorstore(sample_data_dir, isolated_vector_store_dir):
    docs = load_documents_from_dir(sample_data_dir)
    chunks = chunk_documents(docs)
    build_vectorstore(chunks)

    assert vectorstore_exists()
    store = load_vectorstore()
    results = store.similarity_search("annual leave", k=2)
    assert len(results) == 2


def test_build_vectorstore_empty_raises(isolated_vector_store_dir):
    with pytest.raises(ValueError):
        build_vectorstore([])


# ---------------------------------------------------------------------------
# Confidence scoring (pure function, deterministic)
# ---------------------------------------------------------------------------

def test_confidence_high_for_small_distance():
    assert _confidence_from_scores([0.2]) == "high"


def test_confidence_not_found_for_large_distance():
    assert _confidence_from_scores([5.0]) == "not_found"


def test_confidence_not_found_for_empty_scores():
    assert _confidence_from_scores([]) == "not_found"


def test_confidence_medium_band():
    # threshold default 1.1 -> medium band is [0.605, 0.88)
    assert _confidence_from_scores([0.7]) == "medium"


# ---------------------------------------------------------------------------
# RAG chain (LLM mocked)
# ---------------------------------------------------------------------------

def test_answer_question_returns_sources_when_grounded(
    monkeypatch, sample_data_dir, isolated_vector_store_dir
):
    docs = load_documents_from_dir(sample_data_dir)
    chunks = chunk_documents(docs)
    build_vectorstore(chunks)

    monkeypatch.setattr(
        rag_chain, "_make_llm",
        lambda: type("FakeLLM", (), {
            "invoke": lambda self, msgs: type("R", (), {"content": "18 annual leaves per year."})()
        })()
    )
    # Fake (non-semantic) embeddings don't produce realistic distances, so force
    # the "grounded" branch here; real embeddings handle this in production.
    monkeypatch.setattr(rag_chain, "_confidence_from_scores", lambda scores: "high")

    result = answer_question("How many annual leaves are employees allowed?")
    assert result.confidence in ("high", "medium", "low")
    assert len(result.sources) > 0
    assert result.sources[0].file in ("hr_policy.pdf", "remote_work_policy.txt")


def test_answer_question_llm_signals_not_found(
    monkeypatch, sample_data_dir, isolated_vector_store_dir
):
    docs = load_documents_from_dir(sample_data_dir)
    chunks = chunk_documents(docs)
    build_vectorstore(chunks)

    monkeypatch.setattr(
        rag_chain, "_make_llm",
        lambda: type("FakeLLM", (), {
            "invoke": lambda self, msgs: type("R", (), {"content": NOT_FOUND_PHRASE})()
        })()
    )

    result = answer_question("What is the CEO's favorite color?")
    assert result.confidence == "not_found"
    assert result.sources == []


def test_answer_question_missing_api_key_raises(
    monkeypatch, sample_data_dir, isolated_vector_store_dir
):
    docs = load_documents_from_dir(sample_data_dir)
    chunks = chunk_documents(docs)
    build_vectorstore(chunks)

    from src.config import settings
    monkeypatch.setattr(settings, "google_api_key", "")
    # Force a "grounded enough" retrieval so we reach the LLM construction step.
    monkeypatch.setattr(rag_chain, "_confidence_from_scores", lambda scores: "high")

    with pytest.raises(RuntimeError):
        answer_question("How many annual leaves are employees allowed?")


# ---------------------------------------------------------------------------
# FastAPI endpoints
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    from src.main import app
    return TestClient(app)


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "status" in resp.json()


def test_ask_without_vectorstore_returns_409(client, isolated_vector_store_dir):
    resp = client.post("/ask", json={"question": "anything"})
    assert resp.status_code == 409


def test_ask_with_empty_question_returns_422(client):
    resp = client.post("/ask", json={"question": ""})
    assert resp.status_code == 422


def test_upload_rejects_unsupported_extension(client, tmp_path):
    bad_file = tmp_path / "bad.docx"
    bad_file.write_bytes(b"not a real docx")
    with bad_file.open("rb") as f:
        resp = client.post("/upload", files={"file": ("bad.docx", f, "application/octet-stream")})
    assert resp.status_code == 400
