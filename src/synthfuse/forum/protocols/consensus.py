# src/synthfuse/forum/protocols/consensus.py
import jax.numpy as jnp
from scipy.spatial.distance import cosine

def semantic_similarity(vec1: jnp.ndarray, vec2: jnp.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    return 1.0 - cosine(vec1, vec2)

def measure_consensus(embeddings: List[jnp.ndarray]) -> float:
    """Average pairwise similarity across all agents."""
    if len(embeddings) < 2:
        return 1.0
    total = 0.0
    count = 0
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            total += semantic_similarity(embeddings[i], embeddings[j])
            count += 1
    return total / count if count > 0 else 0.0
