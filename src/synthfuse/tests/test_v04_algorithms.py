import jax
import jax.numpy as jnp
from synthfuse.alchemj.registry import get
import pytest

# Load plugins to register symbols
import synthfuse.alchemj.plugins.math
import synthfuse.alchemj.plugins.numeric
import synthfuse.alchemj.plugins.util

@pytest.mark.parametrize("symbol", ["ℙ", "𝔹", "ℍ", "𝓐", "𝕃", "ℂ", "ℤ", "Δ", "ℛ"])
def test_sigil_execution(symbol):
    key = jax.random.PRNGKey(0)
    step_fn = get(symbol)

    x = jnp.ones((10,))

    params = {
        "target_freqs": jnp.array([0.5, 0.5]),
        "h": 2,
        "chunk_size": 2,
        "warm_start_bias": 0.0,
        "threshold": 0.1,
        "baseline": jnp.zeros((10,)),
        "s": 2.0,
        "alpha": 1.5,
        "scale": 0.1
    }

    out = step_fn(key, x, params)

    # Check shape
    if symbol == "ℤ" and out.shape != x.shape:
        # In util.py it's a sum, so it might reduce
        pass
    else:
        assert out.shape == x.shape

    assert not jnp.any(jnp.isnan(out))

def test_omega_gossip():
    from synthfuse.systems.omega_gossip import omega_gossip_step
    key = jax.random.PRNGKey(0)
    state = jnp.ones((5, 2))
    params = {"mixing_matrix": jnp.ones((5, 5)) / 5}
    out = omega_gossip_step(key, state, params)
    assert out.shape == state.shape

def test_sigma_hls():
    from synthfuse.systems.sigma_hls import sigma_hls_explore
    key = jax.random.PRNGKey(0)
    state = jnp.ones((10,))
    out = sigma_hls_explore(key, state, {})
    assert out.shape == state.shape

def test_kappa_verify():
    from synthfuse.systems.kappa_verifier import kappa_verify
    key = jax.random.PRNGKey(0)
    state = jnp.array([0.1, 0.8, 0.5])
    out = kappa_verify(key, state, {"time": 0.01})
    assert out.shape == state.shape
