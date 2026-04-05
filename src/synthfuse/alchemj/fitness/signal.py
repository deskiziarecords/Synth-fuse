import jax.numpy as jnp
import time
from typing import Callable
from ..fitness import ProblemResult

def evaluate_signal(fn: Callable, data: jnp.ndarray, labels: jnp.ndarray) -> ProblemResult:
    """Evaluates an ALCHEM-J spell for signal processing (e.g., classification accuracy)."""
    key = jax.random.PRNGKey(0)
    params = {}

    start_time = time.time()
    try:
        preds = fn(key, data, params)
        accuracy = jnp.mean((preds > 0.5) == labels)
        score = float(accuracy)
    except Exception:
        return ProblemResult(score=-1e9, time=time.time() - start_time, diverged=True)

    end_time = time.time()
    return ProblemResult(score=score, time=end_time - start_time)
