import fitz

pdf_path = "documents/rag_test_cgtmse_sample.pdf"

doc = fitz.open(pdf_path)

print(f"Number of pages: {len(doc)}")

for page_number, page in enumerate(doc, start=1):
    text = page.get_text()
    print(f"\n--- Page {page_number} ---")
    print(text)
