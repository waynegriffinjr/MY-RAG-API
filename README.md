# RAG API

A FastAPI service for retrieving relevant passages from local documents and
generating answers grounded in those passages with Ollama. ChromaDB stores the
document chunks and their embeddings on disk.

## Requirements

- Python 3.14 (the version configured by this project)
- [Ollama](https://ollama.com/) running locally
- The `llama3.2:1b` model available in Ollama

The API currently uses fixed settings: Ollama at `http://localhost:11434`, the
`llama3.2:1b` model, a `./docs` input directory, and a `./rag_db` ChromaDB
directory. These paths are relative to the directory from which you start the
API.

## Setup

From the project directory:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install fastapi "uvicorn[standard]" requests chromadb
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

Install and start Ollama, then download the model:

```bash
ollama pull llama3.2:1b
ollama serve
```

If Ollama is already running as a background service, the `ollama serve`
command is not needed.

## Add documents and start the API

Put `.txt` and `.md` files directly in the project's `docs/` directory. The
ingestion endpoint reads files in filename order, splits each file at blank
lines, and stores each non-empty paragraph as a separate chunk. Other file
types and nested directories are not read.

Start the API from the project directory:

```bash
uvicorn my_rag_api:app --reload
```

The service listens at `http://127.0.0.1:8000`. Interactive API documentation
is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs), and
the OpenAPI schema is at `/openapi.json`.

In Swagger UI, call **POST `/ingest`** before **POST `/ask`**. Ingestion can
also be triggered with:

```bash
curl -X POST http://127.0.0.1:8000/ingest
```

Ingestion uses upsert semantics, so repeating it updates chunks with matching
IDs. It does not remove old chunks when a source file or paragraph is deleted.
The database persists under `rag_db/` between API restarts.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Basic service-running message |
| `POST` | `/ask` | Retrieve relevant chunks and generate a grounded answer |
| `POST` | `/ingest` | Load `.txt` and `.md` paragraph chunks from `docs/` |
| `GET` | `/stats` | Return the stored chunk count, model name, and database path |
| `GET` | `/health` | Report ChromaDB and Ollama connectivity |

### Ask a question

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What does the semantic search pipeline do?",
    "n_results": 3,
    "max_distance": 1.2
  }'
```

`question` is required and cannot be empty or whitespace. `n_results` is
optional (default `3`, range `1`–`20`), and `max_distance` is optional (default
`1.2`, must be greater than `0`). Chunks whose ChromaDB distance exceeds
`max_distance` are excluded.

Example response:

```json
{
  "answer": "The pipeline loads documents, chunks them, creates embeddings, and stores them for retrieval.",
  "sources": [
    {
      "text": "Ingestion phase (run once): Load raw documents...",
      "source": "semantic_search_pipeline.txt",
      "distance": 0.42
    }
  ],
  "confidence": "high"
}
```

Confidence is based on the closest retrieved chunk's distance: `high` below
`0.50`, `medium` below `0.75`, and `low` otherwise. If no chunks have been
ingested or no chunks meet the distance threshold, the endpoint returns a
message with an empty `sources` list and `low` confidence. The model is
instructed to answer only from the retrieved context and cite source documents.

### Ingest documents

```bash
curl -X POST http://127.0.0.1:8000/ingest
```

The response reports `chunks_ingested` and a message. If `docs/` does not
exist, the endpoint returns HTTP `500`; if it contains no supported document
chunks, it reports zero ingested chunks.

### Check status

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/stats
```

`/health` returns `status`, `chromadb`, `ollama`, and `document_count`.
Overall health is `ok` only when ChromaDB is accessible and Ollama responds to
its health check. `/stats` reports the configured model and database path as
well as the current chunk count.

## Errors and browser access

- Invalid `/ask` input is rejected with HTTP `422`.
- Ollama connection failures, timeouts, or HTTP errors produce HTTP `503`;
  invalid Ollama responses produce HTTP `502`.
- The API's CORS middleware allows browser origins `http://localhost:8000`
  and `http://localhost:8501`.

## Tests

Run the retrieval checks from the project directory with the dependencies
installed:

```bash
python test_retrieve.py
```

Author: Wayne Griffin Jr.