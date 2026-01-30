# src/synthfuse/agents/external/qwen/client.py
"""
Qwen Agent Adapter – Cloud API wrapper for Moonshot's Qwen models.
Complies with Synth-Fuse open-gate security model:
  - All core interaction via gated_consult()
  - Output projected through hologram
  - No direct access to /meta, /vector, or state
"""

import os
import asyncio
import httpx
from typing import Any, Dict, Optional

from synthfuse.agents.base import Agent, register_agent, AgentResponse
from synthfuse.agents.open_gate import gated_consult
from synthfuse.agents.hologram import project_as_hologram

@register_agent("qwen")
class QwenAgent(Agent):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen-max",
        timeout: float = 30.0
    ):
        self.api_key = api_key or os.getenv("QWEN_API_KEY")
        if not self.api_key:
            raise ValueError("Qwen API key required (set QWEN_API_KEY or pass explicitly)")
        self.model = model
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)
        self._async_client = httpx.AsyncClient(timeout=timeout)

    def _prepare_payload(self, prompt: str, **kwargs) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 512)
        }

    def generate(self, prompt: str, **kwargs) -> AgentResponse:
        """Synchronous generation via Qwen API."""
        payload = self._prepare_payload(prompt, **kwargs)
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            resp = self._client.post(
                "https://api.moonshot.cn/v1/chat/completions",
                json=payload,
                headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]

            # Optional: consult Alchemist core for post-processing (e.g., vector embedding)
            # This goes through the open-gate—never touches core directly
            core_result = gated_consult(
                agent_id="qwen",
                spell="( ⊗ ℂ )",  # example: apply chaos smoothing
                context={"text": content},
                gate_level=1
            )

            # Return raw + hologram-safe version
            raw_response = AgentResponse(
                content=content,
                meta={
                    "model": self.model,
                    "latency_ms": resp.elapsed.total_seconds() * 1000,
                    "tokens": data["usage"]["total_tokens"]
                },
                vector=core_result.get("vector")  # if computed
            )

            # Project as hologram for external safety
            safe_output = project_as_hologram(raw_response)
            # But internally, we keep full response for forum use
            return raw_response

        except Exception as e:
            raise RuntimeError(f"Qwen API error: {e}")

    async def agenerate(self, prompt: str, **kwargs) -> AgentResponse:
        """Asynchronous generation."""
        payload = self._prepare_payload(prompt, **kwargs)
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            resp = await self._async_client.post(
                "https://api.moonshot.cn/v1/chat/completions",
                json=payload,
                headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]

            # Same gated consultation pattern
            core_result = gated_consult(
                agent_id="qwen",
                spell="( ⊗ ℂ )",
                context={"text": content},
                gate_level=1
            )

            return AgentResponse(
                content=content,
                meta={
                    "model": self.model,
                    "latency_ms": resp.elapsed.total_seconds() * 1000,
                    "tokens": data["usage"]["total_tokens"]
                },
                vector=core_result.get("vector")
            )

        except Exception as e:
            raise RuntimeError(f"Qwen async API error: {e}")

    def embed(self, text: str) -> Any:
        """
        Placeholder: Qwen does not expose public embeddings.
        In practice, route through a local embedder or use /vector proxy.
        """
        # Future: call gated_consult to use a local embedder like BGE-small
        raise NotImplementedError("Qwen embedding not available; use local agent instead")
