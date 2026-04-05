import jax
import jax.numpy as jnp
import time
from typing import Callable
from ..fitness import ProblemResult

def evaluate_optimization(fn: Callable, f: Callable, x0: jnp.ndarray, iters: int = 100) -> ProblemResult:
    """Evaluates an ALCHEM-J spell as an optimization algorithm."""
    key = jax.random.PRNGKey(int(time.time()))
    x = x0
    params = {}

    start_time = time.time()
    try:
        for _ in range(iters):
            x = fn(key, x, params)
            if jnp.any(jnp.isnan(x)) or jnp.any(jnp.isinf(x)):
                return ProblemResult(score=-1e9, time=time.time() - start_time, diverged=True)
    except Exception:
        return ProblemResult(score=-1e9, time=time.time() - start_time, diverged=True)

    end_time = time.time()
    score = -float(f(x)) # minimize → maximize score
    return ProblemResult(score=score, time=end_time - start_time)
