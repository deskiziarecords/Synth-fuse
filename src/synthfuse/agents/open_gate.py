# src/synthfuse/agents/open_gate.py
"""
Enforces Hotpatch-style separation:
  - Yellow (agents) → may request
  - Red (core) → never directly modified
All agent access to /meta, /vector, or state must pass through this gate.
"""

from typing import Any, Dict
from ..meta.constitution import assert_system_integrity
from ..security.open_gate import verify_caller  # from your Hotpatch-inspired module

def gated_consult(
    agent_id: str,
    spell: str,
    context: Dict[str, Any],
    gate_level: int = 0
) -> Dict[str, Any]:
    """
    Only entry point for agents to consult the Alchemist core.
    - Validates caller identity
    - Checks system integrity
    - Returns only hologram-safe output
    """
    # 1. Verify agent is registered and sandboxed
    if not verify_caller(agent_id):
        raise PermissionError("Untrusted agent")

    # 2. Assert core hasn't been tampered with
    assert_system_integrity()

    # 3. Execute in isolated context (no side effects)
    from ..alchemj.compiler import compile_and_run
    result = compile_and_run(spell, context, sandboxed=True)

    # 4. Strip internal state; return only public hologram
    return {
        "result": result.public_output,
        "telemetry": result.safe_metrics,
        "vector": result.latent_embedding  # optional, from /vector
    }
