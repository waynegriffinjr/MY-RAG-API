from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import requests
import chromadb
import os

app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation API",
    version="1.0.0" 
    )

# ── Configuration ─────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "llama3.2:1b"
DB_PATH = "./rag_db"
DOCS_DIR = "./docs"

# ── ChromaDB setup ─────────────────────────────────────────────────────────
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(
    "documents", configuration={"hnsw": {"space": "cosine"}}
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ───────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1, # Returns 422 Unprocessable Entity Automatically for Invalid Entries
        description="Question to ask the RAG system"
    )
    n_results: int = Field(default=3, ge=1, le=20)
    max_distance: float = Field(default=1.2, gt=0)
    
    @field_validator("question")
    @classmethod
    def question_must_not_be_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question cannot be empty or whitespace") # Returns 422
        return value

class SourceChunk(BaseModel):
    text: str
    source: str
    distance: float

class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    confidence: str

class IngestResponse(BaseModel):
    chunks_ingested: int
    message: str


class HealthResponse(BaseModel):
    status: str
    chromadb: str
    ollama: str
    document_count: int


# Temporary Endpoint
@app.get("/")
def root():
    return {"message": "RAG API is running"}

# ── RAG helpers ────────────────────────────────────────────────────────────
# (Paste / adapt functions from your my_rag.py / rag_pipeline.py)

SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Answer ONLY from the provided context. "
    "If the context doesn't contain the answer, say you don't have enough "
    "information. Cite source documents by name. Keep responses under 200 words."
)


def load_documents(directory: str) -> list[dict]:
    """Load all .txt and .md files from directory, chunk by paragraph."""
    docs = []

    # Iterate sorted os.listdir(directory)
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(('.txt', '.md')):
            # For each .txt / .md file, open and read it
            with open(os.path.join(directory, filename), 'r') as f:
                content = f.read()

            # Split content by "\n\n" to get paragraphs, strip each one
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

            # For each non-empty paragraph, append:
            for i, para in enumerate(paragraphs):
                docs.append({
                    "text": para,
                    "id": f"{filename}_{i}",
                    "metadata": {"source": filename, "chunk_index": str(i)}
                })

    # Return the list
    return docs


def retrieve(
    collection, 
    query: str, 
    n_results: int = 3,
    max_distance: float = 1.2
) -> list[dict]:
    """Query ChromaDB, return top chunks as list of dicts, return list of {"text", "metadata", "distance"}."""
    
    # Guard against an empty collection
    if collection.count() == 0:
        return []
    
    # Retrieve the requested number of chunks
    results = collection.query(
        query_texts=[query], 
        n_results=min(n_results, collection.count())
    )
    # Build and return a list of {"text", "metadata", "distance"} dicts
    chunks = []
    for i in range(len(results['documents'][0])):
        chunks.append({
            "text": results['documents'][0][i],
            "metadata": results['metadatas'][0][i],
            "distance": results['distances'][0][i]
        })
        
    # Keep only the chunks within the distance threshold
    filtered_chunks = [
        chunk for chunk in chunks
        if chunk["distance"] <= max_distance
    ]
    
    return filtered_chunks


def compute_confidence(chunks: list[dict]) -> str:
    """Guardrail 2 — return 'high', 'medium', or 'low' based on best distance."""
    if not chunks:
        return "low"

    best_distance = min(chunk["distance"] for chunk in chunks)
    if best_distance < 0.50:
        return "high"
    if best_distance < 0.75:
        return "medium"
    return "low"

def build_messages(question: str, retrieved_chunks: list[dict]) -> list[dict]:
    """Build the RAG messages list: system (with context) + user question."""
    context_parts = []
    
    # Format each chunk as "[Source: <source>]\n<text>"
    for chunk in retrieved_chunks:
        source = chunk['metadata']['source']
        context_parts.append(f"[Source: {source}\n{chunk['text']}]")
    
    # Join chunks with "\n\n---\n\n"
    context = "\n\n---\n\n".join(context_parts)
    
    # Append context to SYSTEM_PROMPT
    system_with_context = f"{SYSTEM_PROMPT}\nCONTEXT:\n{context}"
    
    return [
        {"role": "system", "content": system_with_context},
        {"role": "user", "content": question}
    ]


def call_ollama(messages: list[dict], stream: bool = False) -> str:
    """Call Ollama (non-streaming for structured use). Return response text."""
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat", 
            json={
                "model": LLM_MODEL,
                "messages": messages,
                "stream": False
            }, 
            timeout=120
        )

        response.raise_for_status()
       
        result = response.json()    
        return result["message"]["content"]
    
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=503,
            detail="Ollama request timed out."
        )
    
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="ERROR: Cannot connect to Ollama. Is it running? (ollama serve)"
        )
        
    except requests.exceptions.HTTPError:
        raise HTTPException(
            status_code=503,
            detail="Ollama returned an HTTP error. Check the model configuration."
        )

    except (ValueError, KeyError, TypeError):
        raise HTTPException(
            status_code=502,
            detail="Ollama returned an invalid response."
        )
        


def check_ollama_health() -> bool:
    """Return True if Ollama is reachable, False otherwise."""
    try:
        response = requests.get(
            f"{OLLAMA_URL}/api/tags",
            timeout=3
        )
        return response.status_code == 200
    
    except requests.exceptions.RequestException:
        return False


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Retrieve context and generate a grounded answer."""
    
    # Check whether any documents have been ingested
    if collection.count() == 0:
        return AskResponse(
            answer="No documents habe been ingested yet. Use POST /ingest first.",
            sources=[],
            confidence="low"
        )
    
    # Retrieve relevant chunks.
    chunks = retrieve(
        collection,
        req.question,
        req.n_results,
        req.max_distance
    )
    
    # Handle questions with no sufficiently relevant chunks.
    if not chunks:
        return AskResponse(
            answer="No relevant documents found for this question.",
            sources=[],
            confidence="low"
        )
        
    # Build the prompt using the retrieved context.
    messages = build_messages(req.question, chunks)
    
    # Generate an answer using Ollama.
    answer = call_ollama(messages)
    
    # Convert retrieved chunks into validated source objects.
    sources = [
        SourceChunk(
            text=chunk["text"],
            source=chunk["metadata"].get("source", "unknown"),
            distance=chunk["distance"]
        )
        for chunk in chunks
    ]
    
    # Calculate confidence based on retrieval distance.
    confidence = compute_confidence(chunks)
    
    return AskResponse(
        answer=answer,
        sources=sources,
        confidence=confidence
    )



@app.post("/ingest", response_model=IngestResponse)
def ingest():
    """Load documents from docs/ into ChromaDB."""

    try:
        documents = load_documents(DOCS_DIR)
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="The documents directory does not exist."
        )

    if not documents:
        return IngestResponse(
            chunks_ingested=0,
            message="No .txt or .md document chunks found in the docs directory."
        )

    collection.upsert(
        ids=[doc["id"] for doc in documents],
        documents=[doc["text"] for doc in documents],
        metadatas=[doc["metadata"] for doc in documents]
    )

    return IngestResponse(
        chunks_ingested=len(documents),
        message=f"Successfully ingested {len(documents)} document chunks."
    )



@app.get("/stats")
def stats():
    """Return document count and model info."""
    return {"document_count": collection.count(), "model": LLM_MODEL, "db_path": DB_PATH}


@app.get("/health", response_model=HealthResponse)
def health():
    """Check ChromaDB and Ollama availability."""

    try:
        document_count = collection.count()
        chromadb_status = "ok"
    except Exception:
        document_count = 0
        chromadb_status = "error"

    ollama_status = (
        "connected" if check_ollama_health() else "disconnected"
    )

    overall_status = (
        "ok"
        if chromadb_status == "ok" and ollama_status == "connected"
        else "error"
    )

    return HealthResponse(
        status=overall_status,
        chromadb=chromadb_status,
        ollama=ollama_status,
        document_count=document_count
    )