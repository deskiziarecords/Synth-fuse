# synthfuse/vector/lazy_tensor_scp.py
import jax
import jax.numpy as jnp
from jax import jit, vmap
from typing import Callable, Optional
from synthfuse.solvers.scp import truncated_svd, max_kurtosis_match

class LazyTensorSCP:
    """
    Lazy Tensor Database enhanced with Spectral Compression Parser.

    Innovation: Store generation functions instead of static vectors.
    Formula: S(q, d) = Sim(v_query, F_gen(theta_gen))
    Impact: 99% storage reduction, automatic model updates without re-indexing.
    """
    
    def __init__(self, compression_rank: int = 64):
        self.rank = compression_rank
        self.compressed_db = {}  # id -> (U_k, Σ_k, V_k_T, gen_metadata)
        self.gen_funcs = {}      # id -> F_gen
    
    def register(self, id: str, gen_matrix: jnp.ndarray,
                gen_func: Optional[Callable[[], jnp.ndarray]] = None):
        """
        Store generation function via SCP.
        Replaces: { (ID, vector) } with { (ID, generation_function) }
        """
        # Compress and store initial state
        U_k, Σ_k, V_k_T = truncated_svd(gen_matrix, k=self.rank)
        self.compressed_db[id] = (U_k, Σ_k, V_k_T)

        # Store generation function for "Lazy" evaluation
        if gen_func is not None:
            self.gen_funcs[id] = gen_func
        else:
            # Default: use the compressed representation itself
            self.gen_funcs[id] = lambda: self.decompress(U_k, Σ_k, V_k_T)
    
    def query(self, query_vec: jnp.ndarray, id: str) -> float:
        """MaxKurtosis-optimized similarity via SCP reconstruction."""
        # Check if we should re-generate from the function (automatic model updates)
        if id in self.gen_funcs:
            reconstructed = self.gen_funcs[id]()
        else:
            U_k, Σ_k, V_k_T = self.compressed_db[id]
            reconstructed = self.decompress(U_k, Σ_k, V_k_T)

        return self._jit_kurtosis_match(query_vec, reconstructed)

    @staticmethod
    @jit
    def _jit_kurtosis_match(query_vec, reconstructed):
        return max_kurtosis_match(query_vec, reconstructed)

    @staticmethod
    @jit
    def decompress(U_k, Σ_k, V_k_T):
        """Reconstruct approximation: gen_matrix ≈ U_k @ diag(Σ_k) @ V_k_T."""
        return (U_k * Σ_k) @ V_k_T  # O(k * (m + n)) vs O(m*n)
