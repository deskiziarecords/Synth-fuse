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

def fuse_cond(cond_fn: Callable[[PyTree], bool], step: StepFn) -> StepFn:
    """
    Conditional Dynamics: 1_c(x) f
    """
    def cond_step(key: jax.Array, x: PyTree, params: Dict[str, Any]) -> PyTree:
        return jax.lax.cond(
            cond_fn(x),
            lambda val: step(key, val, params),
            lambda val: val,
            x
        )
    return cond_step

def fuse_parallel(step_a: StepFn, step_b: StepFn) -> StepFn:
    """
    Parallel Fusion: (f + g)(x)
    """
    def parallel_step(key: jax.Array, x: PyTree, params: Dict[str, Any]) -> PyTree:
        k1, k2 = jr.split(key)
        out_a = step_a(k1, x, params)
        out_b = step_b(k2, x, params)
        return jax.tree.map(jnp.add, out_a, out_b)
    return parallel_step

def fuse_meta(step: StepFn, meta_fn: StepFn) -> StepFn:
    """
    Meta-Dynamics: M(f)
    """
    def meta_step(key: jax.Array, x: PyTree, params: Dict[str, Any]) -> PyTree:
        k1, k2 = jr.split(key)
        x_next = step(k1, x, params)
        return meta_fn(k2, x_next, params)
    return meta_step
