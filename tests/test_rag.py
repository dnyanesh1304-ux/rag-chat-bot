"""
Test Suite for RAG Document Bot and Telemetry Pipeline
Verifies:
- PDF text extraction & chunking
- ChromaDB vector storage and semantic retrieval
- Telemetry logging, token counting, and cost formulas
- FastAPI REST endpoints
"""

import os
import unittest
from src.rag_pipeline import RAGPipeline
from src.telemetry import TelemetryTracker
from src.omniroute_client import OmniRouteClient, MODEL_PRICING

class TestRAGPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rag = RAGPipeline(
            db_path="chroma_db",
            collection_name="rag_documents"
        )
        cls.telemetry_db = "logs/test_telemetry.db"
        cls.telemetry = TelemetryTracker(db_path=cls.telemetry_db)
        cls.client = OmniRouteClient()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.telemetry_db):
            try:
                os.remove(cls.telemetry_db)
            except Exception:
                pass

    def test_01_existing_retrieval(self):
        """Verifies vector retrieval from existing ChromaDB sample documents."""
        query = "What does CGTMSE stand for?"
        results = self.rag.retrieve(query, n_results=2)
        self.assertGreaterEqual(len(results), 1, "Should retrieve at least 1 chunk")
        first_doc = results[0]["text"]
        self.assertIn("CGTMSE", first_doc, "Retrieved chunk must contain 'CGTMSE'")
        self.assertIn("similarity", results[0], "Result must contain similarity score")
        self.assertGreater(results[0]["similarity"], 0.0, "Similarity score must be positive")

    def test_02_chunking_logic(self):
        """Verifies chunk_size and overlap constraints."""
        sample_text = "A" * 1200
        chunks = self.rag.chunk_text(sample_text, page_num=1, source="test.pdf")
        self.assertGreater(len(chunks), 1, "1200 chars should produce multiple chunks")
        self.assertEqual(len(chunks[0]["text"]), 500, "First chunk must be 500 chars")
        # Check overlap
        self.assertEqual(chunks[1]["char_start"], 400, "Second chunk start should respect 100 char overlap")

    def test_03_telemetry_logging_and_kpis(self):
        """Verifies SQLite call logging, token ledger, and metric calculations."""
        # Log a sample call
        row_id = self.telemetry.log_call(
            query="Test Query",
            model_requested="no-think/agy/claude-sonnet-4-6",
            model_resolved="claude-sonnet-4-6",
            prompt_tokens=1000,
            completion_tokens=200,
            total_tokens=1200,
            latency_ms=450.5,
            estimated_cost_usd=0.006,
            chunks_retrieved=2,
            success=True,
            response_snippet="Test Answer"
        )
        self.assertIsNotNone(row_id, "Should return valid row id")

        # Fetch metrics summary
        summary = self.telemetry.get_metrics_summary()
        self.assertGreaterEqual(summary["totals"]["total_requests"], 1)
        self.assertGreaterEqual(summary["totals"]["total_tokens"], 1200)
        self.assertGreater(summary["totals"]["total_cost_usd"], 0.0)
        self.assertTrue(len(summary["model_comparison"]) > 0)
        self.assertEqual(summary["model_comparison"][0]["model"], "no-think/agy/claude-sonnet-4-6")

    def test_04_cost_calculator(self):
        """Verifies accurate cost calculation across free, fast, and high-tier models."""
        # 1000 prompt tokens, 500 completion tokens on Sonnet ($3.00/M in, $15.00/M out)
        # In: 1000 * 3 / 1,000,000 = 0.003
        # Out: 500 * 15 / 1,000,000 = 0.0075
        # Total = 0.0105
        sonnet_cost = self.client.calculate_cost("no-think/agy/claude-sonnet-4-6", 1000, 500)
        self.assertAlmostEqual(sonnet_cost, 0.0105, places=4)

        # Free tier model should always cost $0.0
        free_cost = self.client.calculate_cost("auto/best-free", 5000, 2000)
        self.assertEqual(free_cost, 0.0)

    def test_05_fastapi_app_routes(self):
        """Verifies FastAPI health and documents endpoints."""
        from fastapi.testclient import TestClient
        from src.main import app

        client = TestClient(app)
        
        # Test /api/health
        health_res = client.get("/api/health")
        self.assertEqual(health_res.status_code, 200)
        self.assertIn("vector_db", health_res.json())

        # Test /api/models
        models_res = client.get("/api/models")
        self.assertEqual(models_res.status_code, 200)
        self.assertIn("models", models_res.json())

        # Test /api/documents
        docs_res = client.get("/api/documents")
        self.assertEqual(docs_res.status_code, 200)
        self.assertIn("distinct_sources", docs_res.json())

        # Test /api/metrics
        metrics_res = client.get("/api/metrics")
        self.assertEqual(metrics_res.status_code, 200)
        self.assertIn("totals", metrics_res.json())

if __name__ == "__main__":
    unittest.main()
