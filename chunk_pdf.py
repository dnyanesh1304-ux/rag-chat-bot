import fitz

pdf_path = "documents/rag_test_cgtmse_sample.pdf"

doc = fitz.open(pdf_path)

# Extract all text
full_text = ""

for page in doc:
    full_text += page.get_text() + "\n"

# Split text into chunks
chunk_size = 500
overlap = 100

chunks = []

start = 0

while start < len(full_text):
    end = start + chunk_size
    chunk = full_text[start:end]
    chunks.append(chunk)
    start += chunk_size - overlap

print(f"Total chunks: {len(chunks)}")

for i, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {i} ---")
    print(chunk)
