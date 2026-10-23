"""
OmniRoute LLM Client & RAG Grounding Engine
Communicates with the local OmniRoute proxy (http://localhost:20128/v1),
constructs grounded prompts with citations, calculates model-specific costs,
and instruments latency and tokens.

Includes:
- Automatic fallback across OmniRoute models
- Direct Groq API fallback (free tier, always available)
"""

import os
import time
import json
import requests
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(PROJECT_ROOT, ".env")
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
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
    "free": (0.00, 0.00),
    "groq": (0.05, 0.10),
}

DEFAULT_PRICING = (1.00, 3.00)
MODEL_PRICING = TIER_PRICING

# OmniRoute fallback chain
OMNIROUTE_FALLBACKS = [
    "auto/best-chat",
    "auto/fast",
    "auto/smart",
    "auto/cheap",
    "auto/best-free",
    "ddgw/gpt-4o-mini",
    "ddgw/claude-3-5-haiku-20241022",
    "ddgw/mistral-small-2501",
]

# Groq free-tier models (direct API, always available)
# qwen3.8-27b is placed first for maximum speed and reliable direct content generation
GROQ_MODELS = [
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "allam-2-7b",
]

GROQ_API_BASE = "https://api.groq.com/openai/v1"


class OmniRouteClient:
    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        timeout: int = 15
    ):
        self.api_base = (api_base or os.getenv("OMNIROUTE_API_BASE", "http://localhost:20128/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("OMNIROUTE_API_KEY", "")
        self.default_model = default_model or os.getenv("DEFAULT_MODEL", "groq/qwen/qwen3.8-27b")
        self.timeout = timeout

        # Groq direct fallback
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.groq_enabled = bool(self.groq_api_key)
        if self.groq_enabled:
            print("[OmniRoute] ✓ Groq direct fallback enabled (free tier)")
        else:
            print("[OmniRoute] ⚠ No GROQ_API_KEY set — Groq fallback disabled. Set it in .env for reliable access.")

    def calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Calculates estimated cost in USD based on model keyword & pricing tier.
        """
        m_lower = model.lower()
        pricing = DEFAULT_PRICING

        if "free" in m_lower:
            pricing = TIER_PRICING["free"]
        elif "groq" in m_lower or "qwen" in m_lower or "llama" in m_lower or "gemma" in m_lower or "gpt-oss" in m_lower or "allam" in m_lower:
            pricing = TIER_PRICING["groq"]
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
        Tests connectivity to OmniRoute and Groq.
        """
        result = {"connected": False, "status_code": None, "error": None}

        # Check OmniRoute
        try:
            resp = requests.get(f"{self.api_base}/models", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("id") for m in data.get("data", []) if "id" in m]
                result = {
                    "connected": True,
                    "status_code": resp.status_code,
                    "models_count": len(models),
                    "sample_models": models[:10]
                }
            else:
                result["error"] = resp.text[:200]
                result["status_code"] = resp.status_code
        except Exception as e:
            result["error"] = str(e)

        # Add Groq status
        result["groq_fallback"] = self.groq_enabled
        if self.groq_enabled:
            try:
                gresp = requests.get(
                    f"{GROQ_API_BASE}/models",
                    headers={"Authorization": f"Bearer {self.groq_api_key}"},
                    timeout=5
                )
                result["groq_connected"] = gresp.status_code == 200
            except Exception:
                result["groq_connected"] = False

        # Consider healthy if either is available
        if not result.get("connected") and (result.get("groq_connected") or self.groq_enabled):
            result["connected"] = True

        return result

    def get_available_models(self) -> List[str]:
        """
        Returns list of curated models.
        """
        models = []
        if self.groq_enabled:
            models.extend([f"groq/{m}" for m in GROQ_MODELS])
        models.extend([
            "auto/best-chat",
            "auto/fast",
            "auto/smart",
            "auto/best-coding",
            "auto/best-free",
        ])
        return models

    def _call_omniroute(self, model: str, messages: List[Dict], max_tokens: int = 800, temperature: float = 0.2) -> Dict:
        """Call OmniRoute API."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }

        resp = requests.post(f"{self.api_base}/chat/completions", headers=headers, json=payload, timeout=self.timeout)
        return {"status_code": resp.status_code, "headers": dict(resp.headers), "body": resp.text}

    def _call_groq(self, model: str, messages: List[Dict], max_tokens: int = 1500, temperature: float = 0.2) -> Dict:
        """Call Groq API directly (free tier, always available)."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.groq_api_key}"
        }

        # Strip 'groq/' prefix if present
        clean_model = model[5:] if model.startswith("groq/") else model
        if clean_model not in GROQ_MODELS:
            clean_model = GROQ_MODELS[0]

        payload = {
            "model": clean_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }

        resp = requests.post(f"{GROQ_API_BASE}/chat/completions", headers=headers, json=payload, timeout=30)
        return {"status_code": resp.status_code, "headers": dict(resp.headers), "body": resp.text}

    def _parse_response(self, raw: Dict, model_name: str) -> Dict:
        """Parse a successful API response from either OmniRoute or Groq."""
        answer = ""
        resolved_model = model_name
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        # Try direct JSON parse first
        parsed_json = None
        try:
            parsed_json = json.loads(raw.get("body", ""))
        except Exception:
            parsed_json = None

        if isinstance(parsed_json, dict) and "choices" in parsed_json:
            choices = parsed_json.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                answer = msg.get("content", "") or ""
                # Some reasoning models place thinking in 'reasoning'
                if not answer.strip() and msg.get("reasoning"):
                    answer = msg["reasoning"]
            resolved_model = parsed_json.get("model", model_name)
            usage = parsed_json.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
        else:
            # Fall back to SSE stream parsing if response was streamed
            raw_lines = raw.get("body", "").splitlines()
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

        return {
            "answer": answer,
            "resolved_model": resolved_model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def generate_grounded_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Builds grounded context from retrieved chunks and queries LLM.
        Priority: User-selected model → OmniRoute fallbacks → Groq direct fallback.
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

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        # === PHASE 1: Direct Groq execution if user selected a groq/ model ===
        if active_model.startswith("groq/") and self.groq_enabled:
            return self._try_groq(active_model, messages, citations, active_model, query)

        # Try OmniRoute with fallbacks
        models_to_try = [active_model]
        for fb in OMNIROUTE_FALLBACKS:
            if fb not in models_to_try:
                models_to_try.append(fb)

        overall_start = time.perf_counter()
        last_error = None

        for attempt_idx, try_model in enumerate(models_to_try):
            start_time = time.perf_counter()
            try:
                print(f"[OmniRoute] Attempt {attempt_idx + 1}: trying '{try_model}'...")
                raw = self._call_omniroute(try_model, messages)
                latency_ms = (time.perf_counter() - start_time) * 1000.0

                if raw["status_code"] == 200:
                    parsed = self._parse_response(raw, try_model)
                    if parsed["answer"].strip():
                        if parsed["total_tokens"] == 0:
                            parsed["prompt_tokens"] = max(1, len(user_content) // 4)
                            parsed["completion_tokens"] = max(1, len(parsed["answer"]) // 4)
                            parsed["total_tokens"] = parsed["prompt_tokens"] + parsed["completion_tokens"]

                        total_latency = (time.perf_counter() - overall_start) * 1000.0
                        cost_usd = self.calculate_cost(parsed["resolved_model"], parsed["prompt_tokens"], parsed["completion_tokens"])
                        print(f"[OmniRoute] ✓ Success with '{parsed['resolved_model']}' in {latency_ms:.0f}ms")

                        return {
                            "success": True,
                            "answer": parsed["answer"].strip(),
                            "model_requested": active_model,
                            "model_resolved": parsed["resolved_model"],
                            "citations": citations,
                            "latency_ms": round(total_latency, 2),
                            "tokens": {"prompt": parsed["prompt_tokens"], "completion": parsed["completion_tokens"], "total": parsed["total_tokens"]},
                            "cost_usd": cost_usd,
                            "error": None
                        }
                    else:
                        last_error = f"'{try_model}' returned empty answer"
                else:
                    last_error = f"'{try_model}' HTTP {raw['status_code']}: {raw['body'][:200]}"
                    print(f"[OmniRoute] ✗ {last_error}")
                    continue

            except requests.exceptions.Timeout:
                last_error = f"'{try_model}' timed out"
                print(f"[OmniRoute] ✗ {last_error}")
                continue
            except Exception as e:
                last_error = f"'{try_model}' error: {str(e)}"
                print(f"[OmniRoute] ✗ {last_error}")
                continue

        # === PHASE 2: All OmniRoute failed — try Groq direct ===
        if self.groq_enabled:
            print(f"[OmniRoute] All OmniRoute models failed. Falling back to Groq direct API...")
            return self._try_groq(GROQ_MODELS[0], messages, citations, active_model, query)

        # === Everything failed ===
        total_latency = (time.perf_counter() - overall_start) * 1000.0
        return {
            "success": False,
            "answer": "Unable to get a response. All OmniRoute providers are down and no Groq API key is configured. "
                      "Add GROQ_API_KEY to your .env file for reliable access (free at console.groq.com).",
            "model_requested": active_model,
            "model_resolved": active_model,
            "citations": citations,
            "latency_ms": round(total_latency, 2),
            "tokens": {"prompt": 0, "completion": 0, "total": 0},
            "cost_usd": 0.0,
            "error": f"All providers failed. Last: {last_error}"
        }

    def _try_groq(self, groq_model: str, messages: List[Dict], citations: List, original_model: str, query: str) -> Dict:
        """Try Groq direct API as fallback or direct choice."""
        start_time = time.perf_counter()
        clean_pref = groq_model[5:] if groq_model.startswith("groq/") else groq_model
        
        candidates = []
        if clean_pref in GROQ_MODELS:
            candidates.append(clean_pref)
        for gm in GROQ_MODELS:
            if gm not in candidates:
                candidates.append(gm)

        for gm in candidates:
            try:
                print(f"[Groq] Trying '{gm}'...")
                raw = self._call_groq(gm, messages, max_tokens=1500)
                latency_ms = (time.perf_counter() - start_time) * 1000.0

                if raw["status_code"] == 200:
                    parsed = self._parse_response(raw, f"groq/{gm}")
                    if parsed["answer"].strip():
                        if parsed["total_tokens"] == 0:
                            parsed["prompt_tokens"] = max(1, len(query) // 4)
                            parsed["completion_tokens"] = max(1, len(parsed["answer"]) // 4)
                            parsed["total_tokens"] = parsed["prompt_tokens"] + parsed["completion_tokens"]

                        cost_usd = self.calculate_cost(gm, parsed["prompt_tokens"], parsed["completion_tokens"])
                        print(f"[Groq] ✓ Success with '{gm}' in {latency_ms:.0f}ms")

                        return {
                            "success": True,
                            "answer": parsed["answer"].strip(),
                            "model_requested": original_model,
                            "model_resolved": f"groq/{gm}",
                            "citations": citations,
                            "latency_ms": round(latency_ms, 2),
                            "tokens": {"prompt": parsed["prompt_tokens"], "completion": parsed["completion_tokens"], "total": parsed["total_tokens"]},
                            "cost_usd": cost_usd,
                            "error": None
                        }
                    else:
                        print(f"[Groq] '{gm}' returned 200 but answer was empty")
                        continue
                else:
                    err = raw["body"][:200]
                    print(f"[Groq] ✗ '{gm}' HTTP {raw['status_code']}: {err}")
                    continue

            except Exception as e:
                print(f"[Groq] ✗ '{gm}' error: {str(e)}")
                continue

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return {
            "success": False,
            "answer": "Groq API request failed. Please check your GROQ_API_KEY in .env.",
            "model_requested": original_model,
            "model_resolved": "groq/failed",
            "citations": citations,
            "latency_ms": round(latency_ms, 2),
            "tokens": {"prompt": 0, "completion": 0, "total": 0},
            "cost_usd": 0.0,
            "error": "All Groq models failed"
        }
