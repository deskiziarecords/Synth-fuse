"""
ALCHEM-J Operator Registry
Maps symbols → pure JAX functions with uniform signature
Φ(key, x, params) -> new_x
"""
from typing import Callable, Dict, Any
import jax
import jax.numpy as jnp
import jax.random as jr
from functools import partial

PyTree = Any
StepFn = Callable[[jax.Array, PyTree, Dict[str, Any]], PyTree]

# ----- internal table ---------------------------------------------------------
_REGISTRY: Dict[str, StepFn] = {}


# ----- public decorator -------------------------------------------------------
def register(symbol: str):
    """@register('𝕀') def iso_step(key,x,p): ..."""
    # Relax to allow multi-codepoint symbols (e.g. 𝓜𝓐)
    if not symbol:
        raise ValueError("Symbol cannot be empty")
    def decorator(fn: StepFn):
        _REGISTRY[symbol] = fn
        return fn
    return decorator


# ----- lookup -----------------------------------------------------------------
def get(symbol: str) -> StepFn:
    if symbol not in _REGISTRY:
        raise KeyError(f"Symbol {symbol} not registered")
    return _REGISTRY[symbol]

class GlobalRegistry:
    def resolve_best_fit(self, spell, context):
        return {}

# ----- built-ins (loaded automatically) ---------------------------------------
@register("𝕀")
def _iso(key, x, p):
    """Dummy ISO/RIME step: x + ε·N(0,1)"""
    eps = p.get("eps", 0.01)
    return x + eps * jr.normal(key, jax.tree.map(jnp.shape, x))


@register("ℝ")
def _rl(key, x, p):
    """Dummy RL policy update: identity (real impl in plugins.rl)"""
    return x


@register("𝕃")
def _levy(key, x, p):
    """Lévy noise injection (simplified symmetric α-stable)"""
    alpha = p.get("alpha", 1.5)
    scale = p.get("scale", 0.1)
    # α-stable with scale ~ N(0,1) for demo
    noise = jr.normal(key, jax.tree.map(jnp.shape, x))
    return jax.tree.map(lambda n: scale * n * (1 / alpha), noise)


@register("𝕊")
def _svd_proj(key, x, p):
    """Low-rank SVD projection (matrix leaves only)"""
    rank = p.get("rank", min(8, *(x.shape[-2:])))
    U, S, Vt = jnp.linalg.svd(x, full_matrices=False)
    return (U[:, :rank] * S[:rank]) @ Vt[:rank, :]


@register("ℤ")
def _zeta(key, x, p):
    """Zeta-transform placeholder: identity"""
    return x


@register("ℂ")
def _chaos(key, x, p):
    """Logistic chaos map (element-wise)"""
    r = p.get("r", 3.8)
    return jax.tree.map(lambda a: r * a * (1 - a), x)


@register("𝜑")
def _meta(key, x, p):
    """Meta-gradient correction placeholder: identity"""
    return x


# ----- Systemic Primitives (v0.3.0 Economic Expansion) ------------------------

@register("Δ$")
def _recursive_debt(key, x, p):
    """Recursive Debt: mandated growth via interest rate beta."""
    beta = p.get("interest_rate", 0.05)
    return x * (1.0 + beta)


@register("𝕄")
def _market_sgd(key, x, p):
    """Market SGD: maximizing V_total over the manifold."""
    lr = p.get("market_lr", 0.01)
    v_total = jnp.sum(x) # Simplified market value
    grad = jax.grad(lambda a: jnp.sum(a))(x)
    return x + lr * grad


@register("𝕀𝕞𝕞")
def _immune_trigger(key, x, p):
    """Immune Trigger: threshold-based systemic response."""
    risk = p.get("systemic_risk", 0.0)
    delta = p.get("solvency_delta", 1.0)
    # If risk > threshold, activate 'bailout' state (clamp)
    threshold = 0.8
    triggered = (risk > threshold) | (delta < 0.1)
    return jnp.where(triggered, jnp.ones_like(x) * 0.5, x)


@register("§")
def _institutional_invariant(key, x, p):
    """Fiduciary duty and legal constraints anchor."""
    # Anchors x to a specific institutional baseline
    baseline = p.get("fiduciary_baseline", jnp.mean(x))
    return (x + baseline) / 2.0


@register("ℕ")
def _narrative_control(key, x, p):
    """Narrative Control: perception filtering and supply-chain re-routing."""
    # Simulates a reset of x (market sentiment) toward a desired narrative
    narrative_center = p.get("narrative_goal", 1.0)
    bias = p.get("media_bias", 0.1)
    return x * (1 - bias) + narrative_center * bias