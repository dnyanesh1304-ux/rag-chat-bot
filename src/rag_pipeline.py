"""
RAG Pipeline Module
Handles PDF ingestion, text extraction, chunking, embedding, ChromaDB persistence,
and semantic similarity retrieval.
"""

import os
import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional

class RAGPipeline:
    def __init__(
        self,
        db_path: str = "chroma_db",
        collection_name: str = "rag_documents",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize Embedding Model
        print(f"[RAG] Initializing SentenceTransformer model: {embedding_model_name}")
        self.model = SentenceTransformer(embedding_model_name)

        # Initialize ChromaDB Client
        os.makedirs(self.db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        print(f"[RAG] Connected to ChromaDB collection '{self.collection_name}' with {self.collection.count()} existing chunks.")

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extracts text per page from PDF using PyMuPDF (fitz).
        Returns list of dicts: [{'page': 1, 'text': '...'}]
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

        doc = fitz.open(pdf_path)
        pages_data = []

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if text and text.strip():
                pages_data.append({
                    "page": page_num,
                    "text": text.strip()
                })
        doc.close()
        return pages_data

    def chunk_text(self, text: str, page_num: int = 1, source: str = "unknown") -> List[Dict[str, Any]]:
        """
        Splits text into sliding chunks with defined size and overlap.
        Preserves original chunking logic (500 size / 100 overlap).
        """
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            chunk_content = text[start:end].strip()

            if chunk_content:
                chunks.append({
                    "text": chunk_content,
                    "page": page_num,
                    "source": source,
                    "char_start": start,
                    "char_end": min(end, text_len)
                })

            start += self.chunk_size - self.chunk_overlap

        return chunks

    def ingest_pdf(self, pdf_path: str, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Extracts, chunks, embeds, and indexes a PDF document into ChromaDB.
        """
        if filename is None:
            filename = os.path.basename(pdf_path)

        pages = self.extract_text_from_pdf(pdf_path)
        if not pages:
            return {
                "success": False,
                "message": "No extractable text found in PDF.",
                "chunks_count": 0,
                "filename": filename
            }

        all_chunks = []
        for p in pages:
            page_chunks = self.chunk_text(p["text"], page_num=p["page"], source=filename)
            all_chunks.extend(page_chunks)

        if not all_chunks:
            return {
                "success": False,
                "message": "Could not generate chunks from PDF text.",
                "chunks_count": 0,
                "filename": filename
            }

        chunk_texts = [c["text"] for c in all_chunks]
        embeddings = self.model.encode(chunk_texts).tolist()

        ids = []
        metadatas = []
        documents = []

        # Generate unique chunk IDs based on source and index
        # To avoid collisions with existing sample chunks:
        base_id = os.path.splitext(filename)[0].replace(" ", "_").replace("-", "_")
        for i, chunk in enumerate(all_chunks):
            chunk_id = f"{base_id}_p{chunk['page']}_c{i}"
            ids.append(chunk_id)
            documents.append(chunk["text"])
            metadatas.append({
                "source": filename,
                "page": chunk["page"],
                "char_start": chunk["char_start"],
                "char_end": chunk["char_end"],
                "chunk_index": i
            })

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        return {
            "success": True,
            "filename": filename,
            "pages": len(pages),
            "chunks_count": len(all_chunks),
            "total_collection_chunks": self.collection.count()
        }

    def retrieve(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Embeds the user query and retrieves top-k matching chunks from ChromaDB.
        """
        if not query or not query.strip():
            return []

        query_embedding = self.model.encode([query]).tolist()
        count = self.collection.count()
        if count == 0:
            return []

        actual_n = min(n_results, count)
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=actual_n,
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
            ids = results["ids"][0] if results.get("ids") else [f"doc_{i}" for i in range(len(docs))]

            for i in range(len(docs)):
                meta = metas[i] if i < len(metas) and metas[i] is not None else {}
                dist = distances[i] if i < len(distances) else 0.0
                # Chroma uses L2 distance or cosine distance; convert to similarity score 0-1
                # Lower distance = higher similarity
                similarity = max(0.0, min(1.0, 1.0 - (dist / 2.0))) if dist <= 2.0 else 0.1

                retrieved.append({
                    "id": ids[i],
                    "text": docs[i],
                    "metadata": meta,
                    "distance": round(float(dist), 4),
                    "similarity": round(float(similarity), 4),
                    "source": meta.get("source", "Indexed Document"),
                    "page": meta.get("page", 1)
                })

        return retrieved

    def get_stats(self) -> Dict[str, Any]:
        """
        Returns collection summary and document sources.
        """
        count = self.collection.count()
        sources = set()

        if count > 0:
            try:
                # Sample or fetch all metadatas to extract distinct sources
                all_data = self.collection.get(include=["metadatas"])
                if all_data and "metadatas" in all_data and all_data["metadatas"]:
                    for m in all_data["metadatas"]:
                        if m and "source" in m:
                            sources.add(m["source"])
            except Exception as e:
                print(f"[RAG] Warning fetching sources: {e}")

        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "distinct_sources": sorted(list(sources)) if sources else ["rag_test_cgtmse_sample.pdf"]
        }
