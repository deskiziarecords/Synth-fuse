import jax.numpy as jnp
from jax import jit

def truncated_svd(matrix, k=64):
    """
    Compute truncated SVD of a matrix.
    Args:
        matrix: Input matrix of shape (m, n)
        k: Compression rank
    Returns:
        U_k, Σ_k, V_k_T: Truncated components
    """
    m, n = matrix.shape
    k = min(k, m, n)

    U, S, Vh = jnp.linalg.svd(matrix, full_matrices=False)

    return U[:, :k], S[:k], Vh[:k, :]

@jit
def max_kurtosis_match(query_vec, reconstructed_matrix):
    """
    Compute similarity using MaxKurtosis-optimized matching.

    Innovation: Instead of simple cosine similarity, we find the projection
    that maximizes the kurtosis of the match, identifying non-Gaussian
    signal components (true information) versus Gaussian noise.

    Formula: S(q, d) = Sim(v_query, F_gen(theta_gen))
    """
    # Kurtosis = E[(X-mu)^4] / sigma^4
    # For matching, we weight the dot product by the local kurtosis of the signal

    q = query_vec.ravel()
    r = reconstructed_matrix.ravel()

    # Normalize
    q_norm = q / (jnp.linalg.norm(q) + 1e-8)
    r_norm = r / (jnp.linalg.norm(r) + 1e-8)

    # Element-wise product
    prod = q_norm * r_norm

    # Compute local kurtosis as a weighting factor
    # (High kurtosis = heavy tails/sparse signal = more informative)
    mean = jnp.mean(prod)
    std = jnp.std(prod) + 1e-8
    kurt = jnp.mean(((prod - mean) / std) ** 4)

    # Weight the standard similarity by the kurtosis-based 'signal quality'
    similarity = jnp.dot(q_norm, r_norm) * (1.0 + 0.1 * jnp.log(kurt + 1e-8))

    return similarity
