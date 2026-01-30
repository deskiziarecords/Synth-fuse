# src/synthfuse/forum/hologram.py
def project_debate_outcome(memory, consensus: float, status: str) -> dict:
    """Return only public, non-sensitive debate summary."""
    return {
        "status": status,
        "consensus_score": round(consensus, 3),
        "final_statements": [
            {"agent": turn["agent"], "summary": turn["message"][:200]}
            for turn in memory.turns[-len(set(t["agent"] for t in memory.turns)):]
        ],
        "topic_embedding_shape": memory.topic_vec.shape if memory.topic_vec is not None else None
    }
