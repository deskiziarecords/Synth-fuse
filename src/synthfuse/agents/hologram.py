# src/synthfuse/agents/hologram.py
"""
The "hologram": what external systems see.
Never exposes internal state, memory, or constitution.
"""

import jax.numpy as jnp
from .base import AgentResponse

def project_as_hologram(response: AgentResponse) -> dict:
    """Convert raw agent output into safe, signed hologram."""
    return {
        "content": _sanitize_content(response.content),
        "embedding": response.vector.tolist() if response.vector is not None else None,
        "meta": _filter_safe_meta(response.meta),
        # No internal paths, no memory IDs, no rollback tokens
    }

def _sanitize_content(content):
    if isinstance(content, jnp.ndarray):
        return {"shape": content.shape, "dtype": str(content.dtype)}
    return str(content)[:1000]  # truncate to prevent leakage

def _filter_safe_meta(meta):
    SAFE_KEYS = {"latency_ms", "model", "steps", "device"}
    return {k: v for k, v in meta.items() if k in SAFE_KEYS}
