# RAG Chatbot (LangChain + FAISS + OpenRouter)

A document-grounded Q&A chatbot that answers **only** from the documents you provide. Every response includes the source document (and page number for PDFs). If the answer is not present in the uploaded documents, the chatbot returns a **"not found"** response instead of generating unsupported information.

---

## Workflow

```text
Load Documents
      │
      ▼
Split into Chunks
      │
      ▼
Generate Embeddings
      │
      ▼
Store in FAISS
      │
      ├──────────────────────────────────────────────┐
      │                                              │
      ▼                                              │
User Question ─────────► Retrieve Top-K Chunks ◄─────┘
                              │
                              ▼
          OpenRouter-hosted LLM generates answer
            strictly from retrieved context
                              │
                              ▼
         Return Answer + Source Citations
         (or "Not Found" fallback)
```

---

# Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Framework** | LangChain | Document loading, chunking, retrieval pipeline |
| **Vector Store** | FAISS | Local vector database |
| **Embeddings** | OpenRouter Embedding Model | Hosted embeddings through OpenAI-compatible API |
| **LLM** | OpenRouter Chat Model | Grounded answer generation |
| **Backend API** | FastAPI | REST API with interactive Swagger docs |

---

# Project Structure

```text
rag-chatbot/
│
├── src/
│   ├── config.py                 # Environment configuration
│   ├── ingest.py                 # Document ingestion pipeline
│   ├── main.py                   # FastAPI application
│   │
│   └── services/
│       ├── document_loader.py    # PDF/TXT loading
│       ├── chunking.py           # Text splitting
│       ├── vectorstore.py        # FAISS + Embeddings
│       └── rag_chain.py          # Retrieval & grounded generation
│
├── tests/
│   ├── conftest.py
│   ├── test_basic.py
│   └── run_test_questions.py
│
├── data/
│   ├── sample_inputs/
│   └── sample_outputs/
│
├── screenshots/
│
└── .github/
    └── workflows/
        └── ci.yml
```

---

# Setup

**Requirements**

- Python 3.10+

Clone the repository:

```bash
git clone <repository-url>
cd rag-chatbot
```

Create a virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS/Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Copy the template:

```bash
cp .env.example .env
```

Example `.env`:

```text
# OpenRouter
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxx
OPENROUTER_MODEL=openrouter/free

# Embeddings
OPENROUTER_EMBED_MODEL=openai/text-embedding-3-large
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# RAG Configuration
VECTOR_STORE_PATH=./vector_store
VECTOR_STORE_COLLECTION=rag_docs

CHUNK_SIZE=1000
CHUNK_OVERLAP=150

RETRIEVAL_TOP_K=4
NOT_FOUND_DISTANCE_THRESHOLD=1.1

DATA_DIR=./data/sample_inputs

APP_HOST=0.0.0.0
APP_PORT=8000
```

> The default configuration uses OpenRouter-hosted embeddings, so no local Torch, SentenceTransformers, or GPU setup is required.

---

# Running the Project

## 1. Build the Vector Store

```bash
python -m src.ingest
```

This loads the documents, splits them into chunks, generates embeddings, and stores everything in FAISS.

---

## 2. Start the API

```bash
uvicorn src.main:app --reload --port 8000
```

Open:

```
http://localhost:8000/docs
```

to access the Swagger UI.

---

## 3. Ask Questions

Example:

```bash
curl -X POST http://localhost:8000/ask \
-H "Content-Type: application/json" \
-d '{"question":"How many annual leaves are employees allowed?"}'
```

---

## 4. Upload Additional Documents

```bash
curl -X POST http://localhost:8000/upload \
-F "file=@/path/to/document.pdf"
```

Rebuild the vector store:

```bash
curl -X POST http://localhost:8000/ingest
```

---

# Testing

Run unit tests:

```bash
pytest tests/test_basic.py -v
```

Run end-to-end evaluation:

```bash
python tests/run_test_questions.py
```

Requirements:

- Documents already ingested
- Valid `OPENROUTER_API_KEY`

---

# Sample Request

```json
{
  "question": "How many annual leaves are employees allowed?"
}
```

---

# Sample Response

```json
{
  "answer": "Employees are allowed 18 annual leaves per year, with up to 6 unused days carried forward to the next year.",
  "sources": [
    {
      "file": "hr_policy.pdf",
      "page": 2,
      "chunk_id": 1
    }
  ],
  "confidence": "high"
}
```

---

# Not Found Response

```json
{
  "answer": "I couldn't find an answer to this in the uploaded documents.",
  "sources": [],
  "confidence": "not_found"
}
```

---

# How Grounding Works

## 1. Retrieval

The user's question is embedded using the configured OpenRouter embedding model.

FAISS retrieves the **Top-K** most relevant chunks together with similarity scores.

---

## 2. Relevance Check

If the closest chunk exceeds the configured distance threshold:

```
NOT_FOUND_DISTANCE_THRESHOLD
```

the chatbot immediately returns the **Not Found** response without calling the language model.

---

## 3. Grounded Generation

Otherwise, the retrieved chunks are sent to the OpenRouter chat model with strict instructions to answer **only from the provided context**.

If sufficient evidence still does not exist, the model emits a sentinel token and the API again returns the fallback response.

---

## 4. Source Citations

Every answer includes:

- Document filename
- PDF page number (when available)
- Chunk ID

---

## 5. Confidence Score

Confidence is computed from retrieval distance rather than the LLM's own confidence.

Possible values:

- `high`
- `medium`
- `low`
- `not_found`

---

# Screenshots

<img width="1202" height="469" alt="image" src="https://github.com/user-attachments/assets/1880c34c-0eb9-4085-828a-56648b880f67" />

<img width="1148" height="363" alt="image" src="https://github.com/user-attachments/assets/9a4881bf-6adb-484a-8d47-5f37693e993d" />
<img width="1342" height="591" alt="image" src="https://github.com/user-attachments/assets/e5b9f675-560b-4de9-816b-dfb67926e747" />
<img width="1199" height="407" alt="image" src="https://github.com/user-attachments/assets/236dbe3d-935a-40af-a3aa-e3e4a20deb8d" />




---

# Features

- PDF and TXT document support
- Source and page metadata preservation
- Recursive document chunking
- OpenRouter-hosted embeddings
- FAISS local vector database
- Grounded retrieval-augmented generation
- Two-layer hallucination prevention
- Source citations for every answer
- Distance-based confidence estimation
- FastAPI REST endpoints
- Offline unit tests
- End-to-end evaluation script
- GitHub Actions CI integration

---

# Current Limitations

- No reranking after FAISS retrieval
- No multi-turn conversation memory
- No streaming responses
- Confidence thresholds are heuristic and may require tuning for different datasets
- API endpoints do not include authentication
- Large document collections have not been extensively benchmarked
- Document ingestion currently runs synchronously

---

# API Key Notes

The OpenRouter API key is loaded from the `.env` file and should never be committed to version control.

The `openrouter/free` model allows free access to supported providers, although responses may vary slightly because OpenRouter dynamically routes requests across available free models.
