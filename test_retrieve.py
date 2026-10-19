
import chromadb
from my_rag_api import retrieve

client = chromadb.Client()
collection = client.create_collection(
    name="test_documents",
    configuration={"hnsw": {"space": "cosine"}}
)

# Test 1: Empty collection
result = retrieve(collection, "deployment")
assert result == []
print("Test 1 passed: empty collection")

# Add two documents for retrieval tests
collection.add(
    ids=["doc1", "doc2"],
    documents=[
        "FastAPI serves HTTP endpoints.",
        "The garden needs water."
    ],
    metadatas=[
        {"source": "api.md"},
        {"source": "garden.txt"}
    ]
)

# Test 2: Retrieval returns chunks
result = retrieve(
    collection,
    "FastAPI HTTP endpoints",
    n_results=2,
    max_distance=2.0
)
assert len(result) > 0
assert all("text" in chunk and "distance" in chunk for chunk in result)
print("Test 2 passed: retrieval returns chunks")

# Test 3: Threshold excludes distant chunks
result = retrieve(
    collection,
    "FastAPI HTTP endpoints",
    n_results=2,
    max_distance=0.0
)
assert result == []
print("Test 3 passed: distance threshold filters results")
