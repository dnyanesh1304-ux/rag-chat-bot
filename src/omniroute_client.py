"""
OmniRoute LLM Client & RAG Grounding Engine
Communicates with the local OmniRoute proxy (http://localhost:20128/v1),
constructs grounded prompts with citations, calculates model-specific costs,
and instruments latency and tokens.
"""

import os
import time
import json
import requests
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# Pricing table: (input_cost_per_million, output_cost_per_million) in USD
TIER_PRICING = {
    "opus": (15.00, 75.00),
    "coding": (15.00, 75.00),
    "sonnet": (3.00, 15.00),
    "smart": (3.00, 15.00),
    "fast": (0.25, 1.00),
    "haiku": (0.80, 4.00),
    "flash": (0.15, 0.60),
    "cheap": (0.15, 0.60),
    "free": (0.00, 0.00)
}

DEFAULT_PRICING = (1.00, 3.00)
MODEL_PRICING = TIER_PRICING

class OmniRouteClient:
    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        timeout: int = 60
    ):
        self.api_base = (api_base or os.getenv("OMNIROUTE_API_BASE", "http://localhost:20128/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("OMNIROUTE_API_KEY", "")
        self.default_model = default_model or os.getenv("DEFAULT_MODEL", "no-think/agy/claude-sonnet-4-6")
        self.timeout = timeout

    def calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Calculates estimated cost in USD based on model keyword & pricing tier.
        """
        m_lower = model.lower()
        pricing = DEFAULT_PRICING

        if "free" in m_lower:
            pricing = TIER_PRICING["free"]
        elif "opus" in m_lower or "best-coding" in m_lower:
            pricing = TIER_PRICING["opus"]
        elif "sonnet" in m_lower or "best-chat" in m_lower or "smart" in m_lower:
            pricing = TIER_PRICING["sonnet"]
        elif "haiku" in m_lower:
            pricing = TIER_PRICING["haiku"]
        elif "flash" in m_lower or "cheap" in m_lower:
            pricing = TIER_PRICING["flash"]
        elif "fast" in m_lower:
            pricing = TIER_PRICING["fast"]

        in_price_per_m, out_price_per_m = pricing
        cost = (prompt_tokens * in_price_per_m / 1_000_000) + (completion_tokens * out_price_per_m / 1_000_000)
        return round(cost, 6)

    def check_connection(self) -> Dict[str, Any]:
        """
        Pings OmniRoute /v1/models to verify server readiness.
        """
        try:
            url = f"{self.api_base}/models"
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("id") for m in data.get("data", []) if "id" in m]
                return {
                    "connected": True,
                    "status_code": resp.status_code,
                    "models_count": len(models),
                    "sample_models": models[:10]
                }
            return {
                "connected": False,
                "status_code": resp.status_code,
                "error": resp.text[:200]
            }
        except Exception as e:
            return {
                "connected": False,
                "status_code": None,
                "error": str(e)
            }

    def get_available_models(self) -> List[str]:
        """
        Returns list of curated models.
        """
        recommended_models = [
            "no-think/agy/claude-sonnet-4-6",
            "no-think/agy/claude-opus-4-6-thinking",
            "auto/fast",
            "auto/best-chat",
            "auto/best-coding",
            "auto/best-free"
        ]
        return recommended_models

    def generate_grounded_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Builds grounded context from retrieved chunks and queries OmniRoute.
        Instruments latency, tokens, cost, and success status.
        """
        active_model = model or self.default_model

        # Build context block with citations
        context_lines = []
        citations = []
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            source = chunk.get("source", "Document")
            page = chunk.get("page", 1)
            similarity = chunk.get("similarity", 0.0)
            citation_tag = f"[Source {idx}: {source}, Page {page}]"
            context_lines.append(f"{citation_tag}\n{chunk.get('text', '')}")
            citations.append({
                "index": idx,
                "source": source,
                "page": page,
                "similarity": similarity,
                "snippet": chunk.get("text", "")[:250]
            })

        context_text = "\n\n".join(context_lines) if context_lines else "No relevant documents found."

        system_prompt = (
            "You are an expert AI document assistant. Answer the user question accurately, "
            "helpfully, and truthfully based ONLY on the provided context retrieved from the indexed documents.\n\n"
            "GUIDELINES:\n"
            "1. Answer strictly using facts present in the retrieved context.\n"
            "2. If the answer cannot be determined or found in the context, explicitly inform the user "
            "that the indexed documents do not contain sufficient information to answer the question.\n"
            "3. Cite the relevant source tags (e.g., [Source 1: filename, Page X]) in your answer.\n"
            "4. Keep your answer clear, direct, and well-structured with bullet points where appropriate."
        )

        user_content = f"CONTEXT:\n{context_text}\n\nUSER QUESTION:\n{query}"

        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": active_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "max_tokens": 800,
            "temperature": 0.2,
            "stream": False
        }

        start_time = time.perf_counter()
        endpoint = f"{self.api_base}/chat/completions"

        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=self.timeout)
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            if resp.status_code == 200:
                answer = ""
                resolved_model = active_model
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0

                content_type = resp.headers.get("content-type", "")

                # Non-streaming JSON response
                if "application/json" in content_type:
                    res_json = resp.json()
                    choices = res_json.get("choices", [])
                    if choices:
                        answer = choices[0].get("message", {}).get("content", "")
                    resolved_model = res_json.get("model", active_model)
                    usage = res_json.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
                else:
                    # Fallback SSE parsing if streamed
                    raw_lines = resp.text.splitlines()
                    for line in raw_lines:
                        if line.startswith("data: ") and not line.endswith("[DONE]"):
                            try:
                                chunk_json = json.loads(line[6:])
                                delta = chunk_json.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if delta:
                                    answer += delta
                                if "model" in chunk_json:
                                    resolved_model = chunk_json["model"]
                            except Exception:
                                pass

                # Fallback token estimation if usage omitted by upstream
                if total_tokens == 0:
                    prompt_tokens = max(1, len(user_content) // 4)
                    completion_tokens = max(1, len(answer) // 4)
                    total_tokens = prompt_tokens + completion_tokens

                cost_usd = self.calculate_cost(resolved_model, prompt_tokens, completion_tokens)

                return {
                    "success": True,
                    "answer": answer.strip(),
                    "model_requested": active_model,
                    "model_resolved": resolved_model,
                    "citations": citations,
                    "latency_ms": round(latency_ms, 2),
                    "tokens": {
                        "prompt": prompt_tokens,
                        "completion": completion_tokens,
                        "total": total_tokens
                    },
                    "cost_usd": cost_usd,
                    "error": None
                }
            else:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                err_msg = f"OmniRoute returned HTTP {resp.status_code}: {resp.text[:300]}"
                return {
                    "success": False,
                    "answer": f"Error contacting AI model service ({resp.status_code}). Please verify your configuration.",
                    "model_requested": active_model,
                    "model_resolved": active_model,
                    "citations": citations,
                    "latency_ms": round(latency_ms, 2),
                    "tokens": {"prompt": 0, "completion": 0, "total": 0},
                    "cost_usd": 0.0,
                    "error": err_msg
                }

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "success": False,
                "answer": f"Network error contacting OmniRoute at {self.api_base}: {str(e)}",
                "model_requested": active_model,
                "model_resolved": active_model,
                "citations": citations,
                "latency_ms": round(latency_ms, 2),
                "tokens": {"prompt": 0, "completion": 0, "total": 0},
                "cost_usd": 0.0,
                "error": str(e)
            }
