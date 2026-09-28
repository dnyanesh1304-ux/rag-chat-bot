import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH = "chroma_db"
COLLECTION_NAME = "rag_documents"

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(COLLECTION_NAME)

# User question
question = "What does CGTMSE stand for?"

# Convert question to embedding
question_embedding = model.encode([question]).tolist()

# Search ChromaDB
results = collection.query(
    query_embeddings=question_embedding,
    n_results=2
)

print("\nQUESTION:")
print(question)

print("\nRETRIEVED CHUNKS:")
for i, document in enumerate(results["documents"][0], start=1):
    print(f"\n--- Result {i} ---")
    print(document)
