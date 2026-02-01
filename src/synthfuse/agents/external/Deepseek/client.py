# src/synthfuse/agents/external/deepseek/client.py
"""
DeepSeek Agent Adapter – Cloud API wrapper for DeepSeek models.
API: https://api.deepseek.com
"""

import os
import asyncio
import httpx
from typing import Any, Dict, Optional

from synthfuse.agents.base import Agent, register_agent, AgentResponse
from synthfuse.agents.open_gate import gated_consult

@register_agent("deepseek")
class DeepSeekAgent(Agent):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-coder",
        timeout: float = 30.0
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API key required (set DEEPSEEK_API_KEY or pass explicitly)")
        self.model = model
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)
        self._async_client = httpx.AsyncClient(timeout=timeout)

    def _prepare_payload(self, prompt: str, **kwargs) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.2),
            "max_tokens": kwargs.get("max_tokens", 1024)
        }

    def generate(self, prompt: str, **kwargs) -> AgentResponse:
        payload = self._prepare_payload(prompt, **kwargs)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            resp = self._client.post(
                "https://api.deepseek.com/chat/completions",
                json=payload,
                headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]

            core_result = gated_consult(
                agent_id="deepseek",
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
            raise RuntimeError(f"DeepSeek API error: {e}")

    async def agenerate(self, prompt: str, **kwargs) -> AgentResponse:
        payload = self._prepare_payload(prompt, **kwargs)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            resp = await self._async_client.post(
                "https://api.deepseek.com/chat/completions",
                json=payload,
                headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]

            core_result = gated_consult(
                agent_id="deepseek",
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
            raise RuntimeError(f"DeepSeek async API error: {e}")

    def embed(self, text: str) -> Any:
        # DeepSeek does not currently offer public embedding API
        raise NotImplementedError("DeepSeek embedding not available")
