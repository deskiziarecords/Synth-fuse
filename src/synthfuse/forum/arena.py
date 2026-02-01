# src/synthfuse/forum/arena.py
from typing import List, Dict, Any
from synthfuse.agents.base import Agent
from synthfuse.agents.hologram import project_as_hologram
from .protocols.turn import TurnManager
from .protocols.consensus import measure_consensus
from .protocols.memory import DebateMemory
from .hologram import project_debate_outcome

async def run_debate(
    prompt: str,
    agents: List[Agent],
    max_rounds: int = 3
) -> Dict[str, Any]:
    """
    Run a multi-agent debate with semantic consensus tracking.
    Returns a hologram-safe summary.
    """
    memory = DebateMemory(prompt)
    turn_mgr = TurnManager(agents, max_rounds)

    # Initial responses
    initial_embeddings = []
    for agent in agents:
        resp = await agent.agenerate(prompt)
        safe_resp = project_as_hologram(resp)
        memory.add_turn(agent.__class__.__name__, safe_resp["content"], resp.vector)
        if resp.vector is not None:
            initial_embeddings.append(resp.vector)

    # Measure initial alignment
    consensus = measure_consensus(initial_embeddings)

    # If already aligned, return early
    if consensus > 0.85:
        return project_debate_outcome(memory, consensus, "converged")

    # Multi-round refinement
    for agent, round_num in turn_mgr:
        # Build context from memory
        context = "\n".join([
            f"{t['agent']}: {t['message']}" 
            for t in memory.turns[-3:]  # last 3 turns
        ])
        new_prompt = f"Given this discussion:\n{context}\n\nRefine your answer to: {prompt}"
        
        resp = await agent.agenerate(new_prompt)
        safe_resp = project_as_hologram(resp)
        memory.add_turn(agent.__class__.__name__, safe_resp["content"], resp.vector)

        # Update consensus
        recent_embeddings = [t["embedding"] for t in memory.turns[-len(agents):]]
        consensus = measure_consensus(recent_embeddings)

        if consensus > 0.9:
            break

    return project_debate_outcome(memory, consensus, "refined")
