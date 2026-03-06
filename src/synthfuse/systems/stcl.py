"""
Semantic-Thermodynamic Compression Loop (STCL)
Minimises free-energy  ℱ = Λ - β·C   where
  Λ = semantic information content  (I_concept - I_surface)
  C = compressed bit-cost
Loop self-stabilises by alternating:
  1. semantic-field update  (increases Λ)
  2. thermodynamic cooling   (decreases C)
  3. free-energy descent     (gradient on manifold)
"""
import jax
import jax.numpy as jnp
import chex
from typing import Any, Callable
from synthfuse.alchemj import compile_spell

PyTree = Any


# ------------------------------------------------------------------
# 1.  State container
# ------------------------------------------------------------------
@chex.dataclass
class STCLState:
    representation: PyTree     # current latent code  z
    surface_bits: jax.Array    # bit-count of z
    concept_energy: jax.Array  # Λ(z)
    temperature: float         # β = 1/T
    clock: int


# ------------------------------------------------------------------
# 2.  Semantic field  Λ(z)  (Λ = I_concept - I_surface)
# ------------------------------------------------------------------
def semantic_field(z: PyTree, anchor: PyTree) -> float:
    """
    Λ(ℓ) = I_concept(ℓ) - I_surface(ℓ)
    I_concept: invariant meaning (cosine similarity to semantic anchor)
    I_surface: representational redundancy (entropy of representation)
    """
    flat_z, _ = jax.flatten_util.ravel_pytree(z)
    flat_a, _ = jax.flatten_util.ravel_pytree(anchor)

    # I_concept: 1.0 = identical to anchor
    i_concept = jnp.dot(flat_z, flat_a) / (
        jnp.linalg.norm(flat_z) * jnp.linalg.norm(flat_a) + 1e-8
    )

    # I_surface: entropy of the vector (proxy for representational redundancy)
    # Lower entropy -> higher redundancy -> higher I_surface
    probs = jax.nn.softmax(flat_z)
    entropy = -jnp.sum(probs * jnp.log(probs + 1e-8))
    i_surface = 1.0 / (entropy + 1.0)

    return jnp.array(i_concept - i_surface)


# ------------------------------------------------------------------
# 3.  Compression cost  C(z)  (bits)
# ------------------------------------------------------------------
def compression_cost(z: PyTree, quant: float = 1e-3) -> int:
    """
    Simple zero-run-length + quantisation estimate
    Returns bit count (int)
    """
    flat, _ = jax.flatten_util.ravel_pytree(z)
    q = jnp.round(flat / quant)
    # zero-run-length
    zeros = q == 0
    runs = jnp.split(zeros, jnp.where(zeros[:-1] != zeros[1:])[0] + 1)
    run_bits = sum(r.size * (1 + jnp.ceil(jnp.log2(r.size + 1))) for r in runs)
    non_zero_bits = jnp.sum(q != 0) * 16  # 16-bit per non-zero
    return int(run_bits + non_zero_bits)


# ------------------------------------------------------------------
# 4.  Thermodynamic cooling  (quench)
# ------------------------------------------------------------------
def thermodynamic_cool(z: PyTree, cool_rate: float = 0.99) -> PyTree:
    """
    Multiplicative cooling:  z ← z * cool_rate  (manifold preserving)
    """
    return tree_map(lambda a: a * cool_rate, z)


# ------------------------------------------------------------------
# 5.  Free-energy manifold gradient
# ------------------------------------------------------------------
def free_energy_grad(z: PyTree, anchor: PyTree, beta: float, quant: float) -> PyTree:
    """
    ∇_z ℱ = ∇_z Λ - β ∇_z C
    Minimising ℱ(ℓ) = Λ(ℓ) - β·C(ℓ)
    Compression is free-energy minimisation.
    """
    # ∇Λ (We want to MAXIMISE Λ, but the goal is to MINIMISE free energy ℱ = Λ - βC ?)
    # Re-reading: "Computation is defined as: St+1 = argmin (F = E_semantic + λ C_topology)"
    # Prompt says: "min ℱ(ℓ) = Λ(ℓ) - β·C(ℓ)"
    # Wait, in the prompt: "Λ(ℓ) = I_concept - I_surface".
    # Usually we want to maximise info.
    # Prompt Page 7: "min Eℓ∼S [f~(ℓ) - αΛ(ℓ) + βC(ℓ)]"
    # Prompt Page 6: "F(ℓ) = Λ(ℓ) - β·C(ℓ)"  ... and "Compression is free-energy minimization."
    # If we want to MINIMIZE F, and compression cost C is positive, -βC helps if C is large? No.
    # Usually F = E - TS. If E is energy (to be min) and S is entropy (to be max).
    # Let's follow "min [f~ - αΛ + βC]" from page 7.

    # We'll treat Λ as something we want to MAXIMISE (negative in objective).
    # Objective: -Λ + βC

    # ∇(-Λ)
    grad_neg_lambda = jax.grad(lambda z_val: -semantic_field(z_val, anchor))(z)

    # ∇C (finite-diff)
    eps = 1e-4
    def calc_c(z_val):
        return compression_cost(z_val, quant)

    grad_c = jax.grad(lambda z_val: calc_c(z_val))(z) # Using jax.grad if C was differentiable, but it's not.
    # Using finite diff for C
    grad_c = jax.tree.map(
        lambda a: (calc_c(jax.tree.map(lambda v: v + eps, z)) -
                   calc_c(jax.tree.map(lambda v: v - eps, z))) / (2 * eps),
        z
    )

    # combine: ∇ℱ = -∇Λ + β∇C
    return jax.tree.map(lambda gnl, gc: gnl + beta * gc, grad_neg_lambda, grad_c)


# ------------------------------------------------------------------
# 6.  Single STCL step (ready for JIT)
# ------------------------------------------------------------------
@jax.jit
def stcl_step(key: jax.Array, state: STCLState, params: dict) -> STCLState:
    """
    Params: anchor (PyTree), lr, cool_rate, quant
    """
    anchor = params["anchor"]
    lr = params.get("lr", 0.01)
    cool_rate = params.get("cool_rate", 0.99)
    quant = params.get("quant", 1e-3)

    # 1. semantic field update (gradient ascent on Λ)
    grad_f = free_energy_grad(state.representation, anchor, state.temperature, quant)
    new_z = jax.tree.map(lambda z, g: z + lr * g, state.representation, grad_f)

    # 2. thermodynamic cooling (reduce C)
    new_z = thermodynamic_cool(new_z, cool_rate)

    # 3. recompute observables
    new_lambda = semantic_field(new_z, anchor)
    new_bits = compression_cost(new_z, quant)
    new_free_energy = new_lambda - state.temperature * new_bits

    return STCLState(
        representation=new_z,
        surface_bits=new_bits,
        concept_energy=new_lambda,
        temperature=state.temperature,
        clock=state.clock + 1,
    )


# ------------------------------------------------------------------
# 7.  Public factory
# ------------------------------------------------------------------
def make_stcl(anchor: PyTree, z_init: PyTree, temp: float = 1.0) -> tuple[Callable, STCLState]:
    """
    Returns (jit_step, init_state) ready for Synth-Fuse pipeline
    """
    init_state = STCLState(
        representation=z_init,
        surface_bits=compression_cost(z_init),
        concept_energy=semantic_field(z_init, anchor),
        temperature=temp,
        clock=0,
    )
    return stcl_step, init_state

class SemanticThermodynamicLoop:
    """Legacy wrapper for STCL."""
    def __init__(self, beta=1.0):
        self.beta = beta
    def step(self, z, anchor):
        return stcl_step(None, STCLState(z, 0, 0, self.beta, 0), {"anchor": anchor})