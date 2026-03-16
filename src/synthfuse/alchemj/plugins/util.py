"""
Utility primitives: Lévy, chaos, zeta, delta, etc.
All obey:  StepFn(key, x, params) -> new_x
"""
import jax
import jax.numpy as jnp
import jax.random as jr
from synthfuse.alchemj.registry import register

tree_map = jax.tree.map

# ------------------------------------------------------------------
# 𝕃  –  Symmetric α-stable Lévy noise (simplified)
# ------------------------------------------------------------------
@register("𝕃")
def levy_noise(key: jax.Array, x: jax.Array, params: dict) -> jax.Array:
    """
    Params: alpha (tail, 0<α≤2), scale (c>0), beta (skew, [-1,1]), loc
    Returns x + levy_sample (same shape)
    """
    alpha = params.get("alpha", 1.5)
    scale = params.get("scale", 0.1)
    beta = params.get("beta", 0.0)   # symmetry
    loc = params.get("loc", 0.0)

    # Chambers-Mallows-Stuck method (α≠1)
    def _levy(key, shape):
        w = jr.exponential(key, shape)
        phi = jr.uniform(key, shape, minval=-jnp.pi/2 + 1e-5, maxval=jnp.pi/2 - 1e-5)
        if alpha == 1.0:
            z = jnp.tan(phi)
            eta = beta * jnp.log(jnp.abs(w)) + jnp.pi/2
            return loc + scale * (z + eta)
        else:
            cos_phi = jnp.cos(phi)
            z = jnp.tan(phi)
            alpha_phi = alpha * phi
            eta = -beta * jnp.tan(jnp.pi*alpha/2) if (alpha != 1.0) else 0.0

            # Use jnp.power for stability
            term1 = jnp.sin(alpha_phi) / cos_phi
            term2 = jnp.cos((1.0 - alpha) * phi) / w

            levy = jnp.sign(term1) * jnp.power(jnp.abs(term1), 1.0/alpha) * \
                   jnp.power(jnp.abs(term2), (1.0 - alpha) / alpha)

            return loc + scale * (levy + eta)

    keys = jr.split(key, 10)  # cheap split
    noise = tree_map(lambda arr: _levy(keys[0], arr.shape), x)
    return tree_map(jnp.add, x, noise)

# ------------------------------------------------------------------
# ℂ  –  Logistic chaotic map (element-wise)
# ------------------------------------------------------------------
@register("ℂ")
def chaos_logistic(key: jax.Array, x: jax.Array, params: dict) -> jax.Array:
    """
    Params: r (bifurcation param, ~3.57-4.0), n (iterations)
    """
    r = params.get("r", 3.8)
    n = params.get("n", 1)

    def _step(z, _):
        return r * z * (1 - z), None

    final, _ = jax.lax.scan(_step, x, jnp.arange(n))
    return final

# ------------------------------------------------------------------
 # ℤ  –  Zeta-transform (Dirichlet series, truncated)
 # ------------------------------------------------------------------
@register("ℤ")
def zeta_transform(key: jax.Array, x: jax.Array, params: dict) -> jax.Array:
    """
    Params: s (complex exponent), max_terms
    Returns Σ_{k=1}^{max_terms} x_k / k^s   (along last axis)
    """
    s = params.get("s", 2.0)
    max_terms = params.get("max_terms", x.shape[-1])
    k = jnp.arange(1, max_terms + 1, dtype=x.dtype)
    coeffs = 1.0 / (k ** s)  # [max_terms]

    if x.ndim == 1:
        return jnp.dot(x[:max_terms], coeffs)
    return jnp.tensordot(x[..., :max_terms], coeffs, axes=1)

# ------------------------------------------------------------------
 # Δ  –  Delta compression (residual + quantise)
 # ------------------------------------------------------------------
@register("Δ")
def delta_compress(key: jax.Array, x: jax.Array, params: dict) -> jax.Array:
    """
    Params: baseline (PyTree), quant (bin width), signed
    Returns compressed residual:  q = round( (x - b) / quant )
    """
    baseline = params["baseline"]
    quant = params.get("quant", 1e-3)
    signed = params.get("signed", True)

    residual = tree_map(jnp.subtract, x, baseline)
    q = tree_map(lambda r: jnp.round(r / quant), residual)
    if signed:
        return q
    else:
        return tree_map(lambda z: jnp.abs(z), q)

# ------------------------------------------------------------------
 # ℛ  –  Random spherical perturbation (unit-norm)
 # ------------------------------------------------------------------
@register("ℛ")
def spherical_noise(key: jax.Array, x: jax.Array, params: dict) -> jax.Array:
    """
    Params: radius (perturbation size)
    Returns x + radius * uniform random on sphere (same norm)
    """
    radius = params.get("radius", 0.01)
    flat, rebuild = jax.flatten_util.ravel_pytree(x)
    d = flat.shape[0]
    # sample isotropic direction
    dir_key, _ = jr.split(key)
    direction = jr.normal(dir_key, (d,))
    direction = direction / (jnp.linalg.norm(direction) + 1e-8)
    perturb = radius * direction
    return rebuild(flat + perturb)
