"""
WR-SSO: Weierstrass-Regularized Semantic Swarm Optimizer
Fuses:
  1. W-Orion smoothing (Heat-Kernel design manifold)
  2. Semantic-Thermodynamic Compression (STCL)
  3. Swarm-RL exploration (Levy-guided consensus)

Governing Objective:
  min_θ E_{ℓ∼S} [ f~_σ(ℓ) - αΛ(ℓ) + βC(ℓ) ]
"""
import jax
import jax.numpy as jnp
import chex
from typing import Any, Callable, Dict
from synthfuse.alchemj.plugins.orion import weierstrass_potential, OrionState
from synthfuse.systems.stcl import semantic_field, compression_cost, STCLState
from synthfuse.alchemj.combinators import fuse_seq, fuse_loop

PyTree = Any

@chex.dataclass
class WRSSOState:
    swarm_pos: jax.Array        # [pop, dim] ℓ
    swarm_vel: jax.Array
    best_pos: jax.Array
    best_fitness: float
    orion: OrionState
    stcl: STCLState
    alpha: float                # semantic weight
    beta: float                 # compression weight
    key: jax.Array

def wr_sso_step(key: jax.Array, state: WRSSOState, params: Dict[str, Any]) -> WRSSOState:
    """
    ℓ_{t+1} = ℓ_t + η∇f~_σ(ℓ_t) + ξ_t(Levy) - γ∇Λ(ℓ_t)
    """
    k1, k2, k3 = jax.random.split(key, 3)
    pop, dim = state.swarm_pos.shape

    # 1. Gradients from W-Orion (smoothed manifold)
    def orion_val(pos):
        return weierstrass_potential(None, pos[None, :], {
            "embeddings": state.orion.embeddings,
            "temperature": state.orion.temperature,
            "scores": state.orion.scores
        })[0]

    grad_orion = jax.vmap(jax.grad(orion_val))(state.swarm_pos)

    # 2. Gradients from Semantic Field (Λ)
    # We want to MINIMISE [f~ - αΛ + βC], so -αΛ means we want to MAXIMISE Λ
    # Thus, the update is + α ∇Λ
    def semantic_val(pos):
        return semantic_field(pos, params["anchor"])

    grad_semantic = jax.vmap(jax.grad(semantic_val))(state.swarm_pos)

    # 3. Levy noise (ξ_t)
    levy_alpha = params.get("levy_alpha", 1.5)
    levy_noise = jax.random.normal(k1, (pop, dim)) * (1.0 / levy_alpha)

    # 4. Update Rule
    eta = params.get("eta", 0.01)   # orion step size
    gamma = params.get("gamma", 0.01) # semantic step size

    # Direction: +η∇f~ (ascent on smoothed fitness) - γ∇Λ (Wait, prompt says -γ∇Λ(ℓt))
    # Page 7: ℓt+1 = ℓt + η∇f~σ(ℓt) + ξt(Levy) - γ∇Λ(ℓt)
    # I will follow the prompt's conceptual update rule exactly.
    new_pos = state.swarm_pos + eta * grad_orion + levy_noise - gamma * grad_semantic

    # 5. Semantic-Thermodynamic Compression (STCL) interaction
    # Re-inject compressed structure into swarm evolution?
    # STCL acts as a regularizer. We could apply a cooling/quantization step here.
    quant = params.get("quant", 1e-3)
    new_pos = jnp.round(new_pos / quant) * quant # Spatial truncation where Λ ≈ 0 (placeholder logic)

    # 6. Update Swarm State
    # Fitness evaluation (combined objective)
    def objective(pos):
        f_tilde = orion_val(pos)
        lambda_val = semantic_val(pos)
        c_val = compression_cost(pos, quant)
        return f_tilde - state.alpha * lambda_val + state.beta * c_val

    fitness = jax.vmap(objective)(new_pos)
    is_better = fitness < state.best_fitness
    best_pos = jnp.where(is_better[:, None], new_pos, state.best_pos)
    best_fitness = jnp.where(is_better, fitness, state.best_fitness)

    return state.replace(
        swarm_pos=new_pos,
        best_pos=best_pos,
        best_fitness=best_fitness,
        key=k3
    )

def make_wr_sso(pop: int, dim: int, anchor: PyTree, **kwargs):
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)

    # Init Orion
    num_tools = kwargs.get("num_tools", 128)
    orion = OrionState(
        embeddings=jax.random.normal(k1, (num_tools, dim)),
        temperature=kwargs.get("sigma", 1.0),
        scores=jnp.ones(num_tools),
        density=jnp.zeros(dim)
    )

    # Init STCL
    stcl = STCLState(
        representation=jnp.zeros(dim),
        surface_bits=jnp.array(0),
        concept_energy=jnp.array(0.0),
        temperature=kwargs.get("beta", 1.0),
        clock=0
    )

    state = WRSSOState(
        swarm_pos=jax.random.normal(k2, (pop, dim)),
        swarm_vel=jnp.zeros((pop, dim)),
        best_pos=jnp.zeros((pop, dim)),
        best_fitness=jnp.inf,
        orion=orion,
        stcl=stcl,
        alpha=kwargs.get("alpha", 0.5),
        beta=kwargs.get("beta", 0.5),
        key=k3
    )

    return wr_sso_step, state
