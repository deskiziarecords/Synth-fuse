import jax
import jax.numpy as jnp

def cosine_similarity(a, b):
    a_norm = jnp.linalg.norm(a)
    b_norm = jnp.linalg.norm(b)
    return jnp.dot(a, b) / (a_norm * b_norm + 1e-8)

def euclidean_distance(a, b):
    return jnp.linalg.norm(a - b)

def softmax_temperature(logits, temperature=1.0):
    return jax.nn.softmax(logits / temperature)

def entropy(probs):
    return -jnp.sum(probs * jnp.log(probs + 1e-8))

def clip_gradients(grads, max_norm):
    gnorm = jnp.linalg.norm(grads)
    return jax.lax.cond(gnorm > max_norm, lambda: grads * (max_norm / gnorm), lambda: grads)

def zeta_transform(f):
    """
    Fast Zeta Transform: O(n 2^n).
    Computes sum over all subsets for each mask.
    """
    f_hat = f.copy()
    n = int(jnp.log2(f.shape[0]))
    for i in range(n):
        mask = jnp.arange(f_hat.shape[0]) & (1 << i)
        f_hat = jnp.where(mask > 0,
                         f_hat + f_hat[jnp.arange(f_hat.shape[0]) ^ (1 << i)],
                         f_hat)
    return f_hat

def weierstrass_transform(f, X, sigma, num_samples, key):
    """
    Gaussian smoothing via Weierstrass transform.
    f_tilde(X) = E_{Y~N(X, sigma^2 I)}[f(Y)]
    """
    noise = jax.random.normal(key, (num_samples,) + X.shape) * sigma
    Y_samples = X + noise
    f_values = jax.vmap(f)(Y_samples)
    return jnp.mean(f_values)
