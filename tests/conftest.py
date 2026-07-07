"""
Shared pytest fixtures.

Tests must run offline and without a real GOOGLE_API_KEY (CI has neither
network access to Hugging Face nor a real key). We achieve this by:
  - monkeypatching src.services.vectorstore.get_embeddings() to return a
    deterministic fake embedder (no download required),
  - monkeypatching src.services.rag_chain._make_llm() to return a fake LLM
    with a scripted .invoke() response.
"""
import shutil
from pathlib import Path

import pytest
from langchain_community.embeddings import DeterministicFakeEmbedding

TEST_VECTOR_STORE_DIR = Path(__file__).parent / "_tmp_vector_store"


class FakeLLMResponse:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    """Drop-in stand-in for ChatGoogleGenerativeAI used in tests."""

    def __init__(self, scripted_answer: str = "This is a grounded test answer."):
        self.scripted_answer = scripted_answer

    def invoke(self, messages):
        return FakeLLMResponse(self.scripted_answer)


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    """Replace the real (network-downloaded) HF embeddings with a fast fake one."""
    from src.services import vectorstore as vs_module

    fake = DeterministicFakeEmbedding(size=384)
    monkeypatch.setattr(vs_module, "get_embeddings", lambda: fake)
    monkeypatch.setattr(vs_module, "_embeddings_singleton", fake)
    yield


@pytest.fixture()
def isolated_vector_store_dir(monkeypatch):
    """Point the app at a scratch vector-store directory for this test only."""
    from src.config import settings

    if TEST_VECTOR_STORE_DIR.exists():
        shutil.rmtree(TEST_VECTOR_STORE_DIR)
    TEST_VECTOR_STORE_DIR.mkdir(parents=True)

    monkeypatch.setattr(settings, "vector_store_path", str(TEST_VECTOR_STORE_DIR))
    yield str(TEST_VECTOR_STORE_DIR)

    if TEST_VECTOR_STORE_DIR.exists():
        shutil.rmtree(TEST_VECTOR_STORE_DIR)


@pytest.fixture()
def sample_data_dir():
    return str(Path(__file__).parent.parent / "data" / "sample_inputs")
