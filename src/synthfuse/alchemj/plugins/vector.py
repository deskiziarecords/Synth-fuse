"""
ALCHEM-J Vector & Hyper-Efficiency Plugins
Symbols: 𝕊 (SVD/SCP), ℤ (Zeta), 𝜑 (Meta-grad)
"""
import jax
import jax.numpy as jnp
from synthfuse.alchemj.registry import register
from synthfuse.solvers.scp import truncated_svd
from synthfuse.tools.foundation.math_utils import zeta_transform

@register("𝕊")
def _svd_scp(key, x, p):
    """
    Spectral Compression Parser (SCP) Operator.
    Formula: x_approx = U_k @ diag(Σ_k) @ V_k_T
    """
    rank = p.get("rank", 64)
    # Ensure x is a matrix for SVD
    orig_shape = x.shape
    if x.ndim == 1:
        x = x.reshape(1, -1)

    U_k, Σ_k, V_k_T = truncated_svd(x, k=rank)
    reconstructed = (U_k * Σ_k) @ V_k_T

    return reconstructed.reshape(orig_shape)

@register("ℤ")
def _zeta_op(key, x, p):
    """
    Zeta-transform Operator for collision detection signatures.
    """
    # x is histogram of masks or features
    # Pad to power of 2 for fast_zeta_transform if needed
    n = x.shape[0]
    next_pow2 = 1 << (n - 1).bit_length()
    if n < next_pow2:
        x_padded = jnp.pad(x, (0, next_pow2 - n))
    else:
        x_padded = x

    res = zeta_transform(x_padded)
    return res[:n]

@register("𝜑")
def _meta_grad_op(key, x, p):
    """
    Meta-gradient correction (AMGDL-based).
    Adjusts x (typically gradients) using meta-parameters in p.
    """
    meta_grad = p.get("meta_grad", 0.0)
    meta_lr = p.get("meta_lr", 0.01)

    # Correction: x_new = x * exp(-meta_lr * meta_grad)
    correction = jnp.exp(-meta_lr * meta_grad)
    return x * correction
