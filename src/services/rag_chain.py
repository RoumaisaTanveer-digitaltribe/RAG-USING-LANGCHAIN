from dataclasses import dataclass, field
from typing import List, Dict, Any

from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import settings
from src.services.vectorstore import load_vectorstore

NOT_FOUND_PHRASE = "NOT_FOUND_IN_DOCUMENTS"


SYSTEM_PROMPT = f"""You are a document Q&A assistant. You must answer ONLY using the
CONTEXT provided below, which was retrieved from the user's uploaded documents.

Rules:
1. Use only facts stated in the CONTEXT. Do not use outside knowledge, do not guess,
   and do not fill in gaps with assumptions.
2. If the CONTEXT does not contain enough information to answer the question,
   respond with EXACTLY this token and nothing else: {NOT_FOUND_PHRASE}
3. Keep answers concise and factual (1-4 sentences unless the question needs a list).
4. Do not mention "the context" or "the documents" in your answer -- just answer
   naturally, as if you already knew the information.
"""


@dataclass
class SourceRef:
    file: str
    page: Any = None
    chunk_id: Any = None
    snippet: str = ""


@dataclass
class RagResult:
    answer: str
    sources: List[SourceRef] = field(default_factory=list)
    confidence: str = "low"  # "high" | "medium" | "low" | "not_found"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": [
                {"file": s.file, "page": s.page, "chunk_id": s.chunk_id, "content": s.snippet}
                for s in self.sources
            ],
            "confidence": self.confidence,
        }


def _format_context(docs_with_scores) -> str:
    blocks = []
    for i, (doc, _score) in enumerate(docs_with_scores):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        loc = f"{source}" + (
            f", page {page}"
            if page
            else f", chunk {doc.metadata.get('chunk_id')}"
        )
        blocks.append(f"[Excerpt {i + 1} | {loc}]\n{doc.page_content}")
    return "\n\n".join(blocks)


def _confidence_from_scores(scores: List[float]) -> str:
    """
    FAISS (default index) returns L2 distance: lower = more similar.
    Thresholds are heuristic and tunable via NOT_FOUND_DISTANCE_THRESHOLD.
    """
    if not scores:
        return "not_found"
    best = min(scores)
    if best > settings.not_found_distance_threshold:
        return "not_found"
    if best < settings.not_found_distance_threshold * 0.55:
        return "high"
    if best < settings.not_found_distance_threshold * 0.8:
        return "medium"
    return "low"


def retrieve(question: str) -> List:
    """Retrieve top-k (document, distance_score) pairs from the vector store."""
    store = load_vectorstore()
    return store.similarity_search_with_score(question, k=settings.retrieval_top_k)


def _make_llm() -> ChatOpenRouter:
    if not settings.openrouter_api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file."
        )

    return ChatOpenRouter(
        model=settings.openrouter_model,          # e.g. "openrouter/free" [web:42][web:46]
        temperature=0,
        max_tokens=500,
        api_key=settings.openrouter_api_key,
    )


def answer_question(question: str) -> RagResult:
    """Full RAG pipeline: retrieve -> ground-check -> generate -> cite."""
    docs_with_scores = retrieve(question)
    scores = [score for _doc, score in docs_with_scores]
    confidence = _confidence_from_scores(scores)

    if confidence == "not_found" or not docs_with_scores:
        return RagResult(
            answer="I couldn't find an answer to this in the uploaded documents.",
            sources=[],
            confidence="not_found",
        )

    context = _format_context(docs_with_scores)
    llm = _make_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"CONTEXT:\n{context}\n\nQUESTION: {question}"),
    ]
    response = llm.invoke(messages)
    answer_text = response.content.strip()

    if NOT_FOUND_PHRASE in answer_text:
        return RagResult(
            answer="I couldn't find an answer to this in the uploaded documents.",
            sources=[],
            confidence="not_found",
        )

    sources = [
        SourceRef(
            file=doc.metadata.get("source", "unknown"),
            page=doc.metadata.get("page"),
            chunk_id=doc.metadata.get("chunk_id"),
            snippet=doc.page_content[:300],
        )
        for doc, _score in docs_with_scores
    ]

    return RagResult(answer=answer_text, sources=sources, confidence=confidence)