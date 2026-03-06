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

def zeta_transform(x):
    # Placeholder for zeta transform
    return x

def weierstrass_transform(x, s=1.0):
    # Placeholder for weierstrass transform (Gaussian convolution)
    return x
