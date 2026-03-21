"""
ISO-VNS Chaotic Perturbation
Spell:  (𝕀 ∘ ℂ ⊗ 𝕍)(chaos_beta=3.8, k_max=5)
Fuses ISO swarm with chaotic logistic map and Variable Neighbourhood Search.
"""
import jax
import jax.numpy as jnp
from synthfuse.alchemj.registry import register
from synthfuse.alchemj import compile_spell
from synthfuse.alchemj.combinators import fuse_seq, fuse_parallel

# ISO and Chaos are already in registry.py, we need VNS.

@register("𝕍")
def vns_step(key: jax.Array, x: jax.Array, params: dict) -> jax.Array:
    """
    Variable Neighbourhood Search perturbation.
    k_max: max neighbourhood size
    """
    k_max = params.get("k_max", 5)
    k = jax.random.randint(key, (), 1, k_max + 1)
    # perturbation proportional to k
    noise = jax.random.normal(key, x.shape) * k * params.get("perturb_scale", 0.1)
    return x + noise

_SPELL = "(𝕀 ⊕ ℂ ⊕ 𝕍)(r={beta}, k_max={k}, perturb_scale={scale})"

def make(pop: int = 128, dims: int = 100, **hyper):
    hp = {
        "beta": 3.8,
        "k": 5,
        "scale": 0.1,
    }
    hp.update(hyper)

    step_fn = compile_spell(_SPELL.format(
        beta=hp["beta"],
        k=hp["k"],
        scale=hp["scale"]
    ))

    key = jax.random.PRNGKey(42)
    state = jax.random.normal(key, (pop, dims))

    return step_fn, state
