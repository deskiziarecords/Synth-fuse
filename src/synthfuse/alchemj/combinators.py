"""
ALCHEM-J Combinators
Functional Higher-Order Operators for Fusion Calculus
"""
from typing import Callable, Dict, Any, List
import jax
import jax.numpy as jnp
import jax.random as jr

PyTree = Any
StepFn = Callable[[jax.Array, PyTree, Dict[str, Any]], PyTree]

def fuse_seq(steps: List[StepFn]) -> StepFn:
    """
    Function Composition: steps[n] ∘ steps[n-1] ∘ ... ∘ steps[0]
    """
    def seq_step(key: jax.Array, x: PyTree, params: Dict[str, Any]) -> PyTree:
        current_x = x
        keys = jr.split(key, len(steps))
        for i, step in enumerate(steps):
            current_x = step(keys[i], current_x, params)
        return current_x
    return seq_step

def fuse_loop(step: StepFn, max_iter: int = 100) -> StepFn:
    """
    Fixed-Point Iteration: fⁿ(x)
    """
    def loop_step(key: jax.Array, x: PyTree, params: Dict[str, Any]) -> PyTree:
        # scan over max_iter
        def body(carry, k):
            x_val = carry
            return step(k, x_val, params), None

        keys = jr.split(key, max_iter)
        final_x, _ = jax.lax.scan(body, x, keys)
        return final_x
    return loop_step

def fuse_map(step: StepFn) -> StepFn:
    """
    Vmap: Apply step to a batch of inputs
    """
    return jax.vmap(step, in_axes=(0, 0, None))
