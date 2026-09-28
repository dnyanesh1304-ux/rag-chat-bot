import fitz
import chromadb
from sentence_transformers import SentenceTransformer

PDF_PATH = "documents/rag_test_cgtmse_sample.pdf"
DB_PATH = "chroma_db"
COLLECTION_NAME = "rag_documents"

# Load PDF
doc = fitz.open(PDF_PATH)

full_text = ""
for page in doc:
    full_text += page.get_text() + "\n"

# Create chunks
chunk_size = 500
overlap = 100

chunks = []
start = 0

while start < len(full_text):
    end = start + chunk_size
    chunk = full_text[start:end].strip()

    if chunk:
        chunks.append(chunk)

    start += chunk_size - overlap

print(f"Created {len(chunks)} chunks")

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create local Chroma database
client = chromadb.PersistentClient(path=DB_PATH)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)

# Create embeddings
print("Creating embeddings...")

embeddings = model.encode(chunks).tolist()

# Store chunks + embeddings
ids = [f"chunk_{i}" for i in range(len(chunks))]

collection.upsert(
    ids=ids,
    documents=chunks,
    embeddings=embeddings
)

print(f"Stored {len(chunks)} chunks in ChromaDB")
print("Vector database created successfully!")
