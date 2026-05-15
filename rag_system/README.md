# Production-Style RAG System

This project is a complete Retrieval-Augmented Generation system built with FastAPI, Streamlit, ChromaDB, SentenceTransformers, and OpenRouter for chat completions. Users can upload documents, index them into a persistent vector database, and ask grounded questions with citations and retrieval transparency.

## Features

- Document ingestion for PDF, TXT, Markdown, and DOCX
- Recursive chunking with overlap for semantic continuity
- SentenceTransformer embeddings using `all-MiniLM-L6-v2`
- Persistent ChromaDB vector storage with metadata filtering
- Hybrid retrieval combining vector search and BM25
- Optional reranking with `cross-encoder/ms-marco-MiniLM-L-6-v2`
- OpenRouter-based chat completions with streaming, retries, and fallback models
- Query rewriting and short-term conversation memory
- Grounded prompt construction with hallucination guardrails
- Streamlit chat UI with citations, confidence, and upload workflow
- Lightweight RAG evaluation metrics
- File validation, upload limits, safe prompt handling, and rate limiting
- Docker and docker-compose support

## Architecture

```text
Upload -> Loader -> Recursive Chunker -> Embedding Service -> Chroma Vector Store
                                                                  |
Question -> Query Rewriter -> Hybrid Retriever -> Reranker -> Prompt Builder -> OpenRouter LLM
                                                                  |
                                                          Citations + Confidence
```

## Project Structure

```text
rag_system/
├── app/
│   ├── api/
│   ├── chunking/
│   ├── embeddings/
│   ├── frontend/
│   ├── ingestion/
│   ├── prompting/
│   ├── retrieval/
│   ├── utils/
│   ├── config.py
│   ├── evaluation.py
│   ├── main.py
│   └── schemas.py
├── chroma_store/
├── tests/
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── README.md
└── requirements.txt
```

## Key Modules

### `app/retrieval/llm_service.py`

This is the reusable OpenRouter client. It:

- Sends chat completion requests to `https://openrouter.ai/api/v1/chat/completions`
- Reads `OPENROUTER_API_KEY` from `.env`
- Supports model switching
- Retries on rate limits and transient server failures
- Falls back across configured models
- Streams token output for chat responses
- Raises clear errors for invalid or missing API keys

### `app/retrieval/openrouter_config.py`

This module centralizes OpenRouter settings such as:

- Endpoint
- Primary model
- Fallback models
- Timeout
- Retry count
- Retry backoff
- Optional `HTTP-Referer` and `X-Title` headers

## OpenRouter Setup

### 1. Create an OpenRouter account

Go to the OpenRouter website and sign up for an account. Complete any billing or provider-access setup required for the models you want to use.

### 2. Generate an API key

After signing in, open your OpenRouter dashboard and create an API key.

### 3. Place the API key in `.env`

Copy the example environment file and add your key:

```bash
copy .env.example .env
```

Set:

```env
OPENROUTER_API_KEY=your_api_key_here
```

You can also configure model selection:

```env
OPENROUTER_PRIMARY_MODEL=openai/gpt-4o-mini
OPENROUTER_FALLBACK_MODELS=anthropic/claude-3-haiku,meta-llama/llama-3-70b-instruct
```

## Supported Example Models

- `openai/gpt-4o-mini`
- `anthropic/claude-3-haiku`
- `meta-llama/llama-3-70b-instruct`

## Environment Configuration

Important `.env` values:

```env
LLM_PROVIDER=openrouter
LLM_MODEL=openai/gpt-4o-mini
LLM_BASE_URL=https://openrouter.ai/api/v1/chat/completions
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1/chat/completions
OPENROUTER_PRIMARY_MODEL=openai/gpt-4o-mini
OPENROUTER_FALLBACK_MODELS=anthropic/claude-3-haiku,meta-llama/llama-3-70b-instruct
LLM_TIMEOUT_SECONDS=60
LLM_MAX_RETRIES=3
LLM_RETRY_BACKOFF_SECONDS=1.5
```

## Local Setup

### 1. Install dependencies

```bash
cd rag_system
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
copy .env.example .env
```

Add your `OPENROUTER_API_KEY` and adjust the selected models if needed.

### 3. Run the backend

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Run the frontend

```bash
streamlit run app/frontend/streamlit_app.py
```

## Docker

```bash
docker compose up --build
```

Backend: `http://localhost:8000`

Frontend: `http://localhost:8501`

## API Endpoints

- `POST /upload`: upload and index a document
- `POST /query`: retrieve grounded answers
- `GET /documents`: list indexed documents
- `DELETE /documents/{id}`: remove a document and its vectors
- `GET /health`: health check

## Example API Usage

### Direct OpenRouter call in Python

```python
import os
import requests

API_KEY = os.getenv("OPENROUTER_API_KEY")

response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    },
    json={
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "Explain retrieval-augmented generation."}],
    },
    timeout=60,
)

print(response.json())
```

### Query the RAG backend

```bash
curl -X POST http://localhost:8000/query ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"Summarize the uploaded policy.\",\"top_k\":5,\"use_hybrid\":true,\"use_reranking\":true}"
```

## Example Questions

- "What does the uploaded policy say about paid leave?"
- "Summarize the onboarding steps from all uploaded documents."
- "Which document mentions quarterly revenue guidance?"
- "What are the differences between section 2 and section 5?"

## Troubleshooting

### Invalid API key

Symptoms:

- `OpenRouter rejected the API key`
- `401` or `403` responses

Fix:

- Verify `OPENROUTER_API_KEY` in `.env`
- Confirm the key is active in your OpenRouter dashboard
- Restart the backend after updating `.env`

### Rate limits

Symptoms:

- `429` responses
- slow or retried responses

Fix:

- Reduce request frequency
- Increase `LLM_MAX_RETRIES`
- Switch the primary model or reorder fallback models

### Model unavailable

Symptoms:

- request failures for a specific provider/model

Fix:

- Set a different `OPENROUTER_PRIMARY_MODEL`
- Add backup entries to `OPENROUTER_FALLBACK_MODELS`

### No answer returned

Symptoms:

- empty or fallback answers

Fix:

- confirm documents were indexed successfully
- inspect retrieved chunks in the Streamlit source panel
- increase `top_k`
- check whether the source documents actually contain the answer

## Testing

```bash
pytest
```

## Notes

- The current evaluation metrics are lightweight local heuristics for regression checks.
- Conversation memory and semantic caching are in-memory by default.
- You can point the same client pattern at a different OpenRouter model without changing the RAG pipeline.
