"""
Telemetry & Monitoring Module
Instruments every LLM request, recording tokens, latency, cost calculations,
model quality/success indicators, and storing metrics in persistent SQLite.
"""

import sqlite3
import os
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

class TelemetryTracker:
    def __init__(self, db_path: str = "logs/telemetry.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS call_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    query TEXT NOT NULL,
                    model_requested TEXT NOT NULL,
                    model_resolved TEXT,
                    prompt_tokens INTEGER DEFAULT 0,
                    completion_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    latency_ms REAL NOT NULL,
                    estimated_cost_usd REAL DEFAULT 0.0,
                    chunks_retrieved INTEGER DEFAULT 0,
                    success INTEGER NOT NULL,
                    error_message TEXT,
                    response_snippet TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON call_logs (timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_logs_model ON call_logs (model_requested)
            """)
            conn.commit()

    def log_call(
        self,
        query: str,
        model_requested: str,
        model_resolved: Optional[str],
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        latency_ms: float,
        estimated_cost_usd: float,
        chunks_retrieved: int,
        success: bool,
        error_message: Optional[str] = None,
        response_snippet: Optional[str] = None
    ) -> int:
        """
        Records a single RAG generation / LLM call.
        """
        now = datetime.now().isoformat() + "Z"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO call_logs (
                    timestamp, query, model_requested, model_resolved,
                    prompt_tokens, completion_tokens, total_tokens,
                    latency_ms, estimated_cost_usd, chunks_retrieved,
                    success, error_message, response_snippet
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now,
                query[:500],
                model_requested,
                model_resolved or model_requested,
                int(prompt_tokens or 0),
                int(completion_tokens or 0),
                int(total_tokens or (prompt_tokens + completion_tokens)),
                round(float(latency_ms), 2),
                round(float(estimated_cost_usd), 6),
                int(chunks_retrieved or 0),
                1 if success else 0,
                error_message,
                (response_snippet or "")[:500]
            ))
            conn.commit()
            return cursor.lastrowid

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Aggregates dashboard metrics:
        - Total requests, success rate
        - Total cost ($)
        - Latency stats (average, min, max)
        - Token breakdown
        - Model breakdown (comparison of cost & speed)
        - Optimization recommendations
        """
        with self._get_connection() as conn:
            # Overall totals
            totals_row = conn.execute("""
                SELECT 
                    COUNT(*) as total_calls,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_calls,
                    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed_calls,
                    SUM(prompt_tokens) as sum_prompt_tokens,
                    SUM(completion_tokens) as sum_completion_tokens,
                    SUM(total_tokens) as sum_total_tokens,
                    SUM(estimated_cost_usd) as sum_cost_usd,
                    AVG(latency_ms) as avg_latency_ms,
                    MIN(latency_ms) as min_latency_ms,
                    MAX(latency_ms) as max_latency_ms
                FROM call_logs
            """).fetchone()

            total_calls = totals_row["total_calls"] or 0
            successful_calls = totals_row["successful_calls"] or 0
            failed_calls = totals_row["failed_calls"] or 0
            total_prompt_tokens = totals_row["sum_prompt_tokens"] or 0
            total_comp_tokens = totals_row["sum_completion_tokens"] or 0
            total_tokens = totals_row["sum_total_tokens"] or 0
            total_cost = totals_row["sum_cost_usd"] or 0.0
            avg_latency = totals_row["avg_latency_ms"] or 0.0

            # Per-model breakdown
            model_rows = conn.execute("""
                SELECT 
                    model_requested,
                    COUNT(*) as count,
                    SUM(estimated_cost_usd) as model_cost,
                    AVG(latency_ms) as avg_latency,
                    SUM(total_tokens) as total_tokens,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count
                FROM call_logs
                GROUP BY model_requested
                ORDER BY count DESC
            """).fetchall()

            model_comparison = []
            for r in model_rows:
                cnt = r["count"]
                m_cost = r["model_cost"] or 0.0
                m_lat = r["avg_latency"] or 0.0
                model_comparison.append({
                    "model": r["model_requested"],
                    "requests": cnt,
                    "cost_usd": round(m_cost, 4),
                    "avg_cost_per_query": round(m_cost / cnt, 5) if cnt > 0 else 0,
                    "avg_latency_ms": round(m_lat, 1),
                    "total_tokens": r["total_tokens"] or 0,
                    "success_rate": round((r["success_count"] / cnt) * 100, 1) if cnt > 0 else 0
                })

            # Cost optimization recommendation
            recommendations = []
            if total_calls > 0:
                # Check if high-cost models were used
                expensive_models = [m for m in model_comparison if "opus" in m["model"].lower() or "best-coding" in m["model"].lower()]
                if expensive_models:
                    recommendations.append({
                        "type": "cost_saving",
                        "title": "Switch to Fast Routing for General Q&A",
                        "description": "Switching routine retrieval queries from Opus/Coding models to 'auto/fast' or 'auto/best-fast' can reduce API costs by up to 85% with negligible change in grounding accuracy."
                    })
                recommendations.append({
                    "type": "chunking_efficiency",
                    "title": "Token Density Optimization",
                    "description": f"Average prompt size is {round(total_prompt_tokens / total_calls) if total_calls else 0} tokens. Chunks with top semantic similarity (k=2 or 3) provide optimal context density."
                })

            # Recent 25 logs
            recent_logs_rows = conn.execute("""
                SELECT 
                    id, timestamp, query, model_requested, model_resolved,
                    prompt_tokens, completion_tokens, total_tokens,
                    latency_ms, estimated_cost_usd, chunks_retrieved,
                    success, error_message, response_snippet
                FROM call_logs
                ORDER BY id DESC
                LIMIT 25
            """).fetchall()

            recent_logs = []
            for row in recent_logs_rows:
                recent_logs.append({
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "query": row["query"],
                    "model": row["model_requested"],
                    "resolved_model": row["model_resolved"],
                    "tokens": {
                        "prompt": row["prompt_tokens"],
                        "completion": row["completion_tokens"],
                        "total": row["total_tokens"]
                    },
                    "latency_ms": row["latency_ms"],
                    "cost_usd": row["estimated_cost_usd"],
                    "chunks": row["chunks_retrieved"],
                    "success": bool(row["success"]),
                    "error": row["error_message"],
                    "snippet": row["response_snippet"]
                })

            return {
                "totals": {
                    "total_requests": total_calls,
                    "successful_requests": successful_calls,
                    "failed_requests": failed_calls,
                    "success_rate_percent": round((successful_calls / total_calls * 100), 1) if total_calls else 100.0,
                    "total_tokens": total_tokens,
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_comp_tokens,
                    "total_cost_usd": round(total_cost, 5),
                    "avg_latency_ms": round(avg_latency, 1)
                },
                "model_comparison": model_comparison,
                "recommendations": recommendations,
                "recent_logs": recent_logs
            }
