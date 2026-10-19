RAG API with Health Checks
Objective: Build a FastAPI RAG API with production-ready features: health checks, configurable parameters, and proper error handling.

Time: 35 minutes

What you’ll do:

Build my_rag_api.py with these endpoints:
POST /ask - Accept a question, return a grounded answer with sources and confidence
POST /ingest - Load documents from the docs/ folder into ChromaDB
GET /stats - Return document count and model info
GET /health - Check if ChromaDB is accessible and Ollama is running
Include Pydantic schemas for all request/response models
Add CORS middleware for frontend connectivity
Handle these error cases:
Ollama not running → 503 Service Unavailable
No documents ingested → clear message in /ask response
Invalid question (empty string) → 422 validation error via Pydantic
Test all endpoints using Swagger UI at /docs
Deliverable: A working FastAPI RAG API that you can test in Swagger UI. All 4 endpoints functional, with proper error handling.

Why this exercise? This API is the backend for your module project. Getting it working and tested now means Week 2 focuses on Docker, not debugging the API.