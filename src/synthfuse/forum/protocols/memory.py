# src/synthfuse/forum/protocols/memory.py
from typing import List, Dict
import jax.numpy as jnp
from synthfuse.vector.lazy_tensor_scp import store_vector, retrieve_similar

class DebateMemory:
    """Short-term vector memory for current debate."""
    def __init__(self, topic: str):
        self.topic_vec = None  # will be set on first embed
        self.turns: List[Dict] = []

    def add_turn(self, agent_name: str, message: str, embedding: jnp.ndarray):
        if self.topic_vec is None:
            self.topic_vec = embedding
        self.turns.append({
            "agent": agent_name,
            "message": message,
            "embedding": embedding
        })
        # Optional: store in /vector for long-term recall
        store_vector(f"debate_{agent_name}", embedding)
