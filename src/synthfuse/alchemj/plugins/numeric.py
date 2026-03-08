"""
Pure-JAX numerical primitives for ALCHEM-J
All obey:  StepFn(key, x, params) -> new_x
"""
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from synthfuse.alchemj.registry import register

tree_map = jax.tree.map

# ------------------------------------------------------------------
# 𝕊  –  SVD low-rank projection (matrix leaves only)
# ------------------------------------------------------------------
@register("𝕊")
def svd_lowrank(key: jax.Array, x: jax.Array, params: dict) -> jax.Array:
    """
    Params: rank, threshold
    """
    rank = min(params.get("rank", 8), *(x.shape[-2:]))
    threshold = params.get("threshold", 1e-6)
    U, S, Vt = jnp.linalg.svd(x, full_matrices=False)
    mask = S > threshold
    k = min(rank, int(mask.sum()))
    return (U[:, :k] * S[:k]) @ Vt[:k, :]

# ------------------------------------------------------------------
# 𝕊𝕊 –  Strassen O(n^log2(7)) matmul (powers of 2 only)
# ------------------------------------------------------------------
@register("𝕊𝕊")
def strassen_mul(key: jax.Array, x: jax.Array, params: dict) -> jax.Array:
    """
    Params: block (minimum size to switch to Strassen)
    """
    block = params.get("block", 128)
    n = x.shape[0]
    if n <= block or n & (n - 1):
        # fall back to GEMM
        return x @ x
    half = n // 2
    a11, a12, a21, a22 = _partition(x, half)
    b11, b12, b21, b22 = _partition(x, half)  # assume A=B for square

    m1 = strassen_mul(key, a11 + a22, b11 + b22, params)
    m2 = strassen_mul(key, a21 + a22, b11, params)
    m3 = strassen_mul(key, a11, b12 - b22, params)
    m4 = strassen_mul(key, a22, b21 - b11, params)
    m5 = strassen_mul(key, a11 + a12, b22, params)
    m6 = strassen_mul(key, a21 - a11, b11 + b12, params)
    m7 = strassen_mul(key, a12 - a22, b21 + b22, params)

    c11 = m1 + m4 - m5 + m7
    c12 = m3 + m5
    c21 = m2 + m4
    c22 = m1 - m2 + m3 + m6

    return _combine(c11, c12, c21, c22)

def _partition(x, half):
    return x[:half, :half], x[:half, half:], x[half:, :half], x[half:, half:]

def _combine(c11, c12, c21, c22):
    top = jnp.hstack([c11, c12])
    bot = jnp.hstack([c21, c22])
    return jnp.vstack([top, bot])

# ------------------------------------------------------------------
# ℂ𝕙 –  Sparse Cholesky solve  (assume x is RHS, A in params)
# ------------------------------------------------------------------
@register("ℂ𝕙")
def sparse_cholesky(key: jax.Array, x: jax.Array, params: dict) -> jax.Array:
    """
    Params: A (positive-definite matrix), reorder=True
    Returns  A^{-1} x  via sparse Cholesky
    """
    A = params["A"]
    reorder = params.get("reorder", True)
    if reorder:
        perm = jsp.linalg.cholesky(A, permute_l=True)[1]
        A = A[perm][:, perm]
        x = x[perm]
    L = jnp.linalg.cholesky(A)
    y = jsp.linalg.solve_triangular(L, x, lower=True)
    z = jsp.linalg.solve_triangular(L.T, y, lower=False)
    if reorder:
        z = z[jnp.argsort(perm)]
    return z

# ------------------------------------------------------------------
# ℕ𝕋𝕂 –  NTK projection (vector leaves)
# ------------------------------------------------------------------
@register("ℕ𝕋𝕂")
def ntk_project(key: jax.Array, x: jax.Array, params: dict) -> jax.Array:
    """
    Params: basis (matrix whose rows=NTK eigenvectors), rank
    Projects x onto top-rank NTK subspace.
    """
    basis = params["basis"]  # shape [R, D]
    rank = min(params.get("rank", 8), basis.shape[0])
    B = basis[:rank]  # [rank, D]
    coeffs = B @ x  # [rank]
    return B.T @ coeffs  # back to D

# ------------------------------------------------------------------
# 𝔽 –  Fisher-Rao geodesic step
# ------------------------------------------------------------------
@register("𝔽")
def fisher_step(key: jax.Array, x: jax.Array, params: dict) -> jax.Array:
    """
    Params: grad, fisher_diag (diagonal FIM), lr
    Performs natural-gradient update:  θ ← θ - lr F^{-1} g
    """
    grad = params["grad"]
    fisher_diag = params["fisher_diag"]
    lr = params.get("lr", 1e-3)
    # stable inversion with damping
    damp = 1e-8
    inv_f = 1.0 / (fisher_diag + damp)
    update = -lr * inv_f * grad
    return x + update

# ------------------------------------------------------------------
# Ι – Algorithm Iota: ℓ4-norm maximization with Bloch signal models
# ------------------------------------------------------------------
@register("Ι")
def algorithm_iota(key: jax.Array, x: jax.Array, params: dict) -> jax.Array:
    """
    Absolute fourth power maximization for MRI quantitative mapping.
    Bloch signal model proxy: maximizes peakiness of reconstruction.
    """
    # x represents reconstruction weights or signal components
    # ℓ4-norm maximization via gradient ascent
    lr = params.get("lr", 0.01)
    def l4_norm(v):
        return jnp.sum(jnp.abs(v)**4)

    grad = jax.grad(l4_norm)(x)
    return x + lr * grad

# ------------------------------------------------------------------
# ρ – Algorithm Rho: RGF Schur Complement with Triple-Color warm starts
# ------------------------------------------------------------------
@register("ρ")
def algorithm_rho(key: jax.Array, x: jax.Array, params: dict) -> jax.Array:
    """
    Selected matrix inversion for demand forecasting.
    RGF Schur Complement approximation.
    """
    # x is a covariance or interaction matrix
    # Triple-Color warm start (mocked as a biased initialization)
    bias = params.get("warm_start_bias", 0.1)
    warm_x = x + bias * jr.normal(key, x.shape)

    # Selected inversion (Schur Complement proxy)
    # Using JAX's solver as a base for the Schur complement
    # In practice, this would be a more specialized sparse RGF solver
    return jnp.linalg.inv(warm_x + jnp.eye(x.shape[0]) * 1e-6)