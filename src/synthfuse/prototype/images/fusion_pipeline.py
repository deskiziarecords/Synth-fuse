# src/synthfuse/prototype/images/fusion_pipeline.py
"""
Tensor-Graph Fusion Pipeline – End-to-end differentiable image generation.
Fully fused: preprocess → iterative refinement → postprocess in one JIT graph.
Designed for batch inference, edge deployment, and spell composition.
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable, Dict, Any


# ───────────────────────────────────────────────
# Core Pure Functions (No Side Effects)
# ───────────────────────────────────────────────

def default_preprocess(image: jax.Array) -> jax.Array:
    """Normalize uint8 [0,255] → float32 [-1, 1]."""
    return image.astype(jnp.float32) / 127.5 - 1.0


def default_postprocess(x: jax.Array) -> jax.Array:
    """Convert float32 [-1, 1] → uint8 [0, 255]."""
    x = jnp.clip((x + 1.0) * 127.5, 0, 255)
    return x.astype(jnp.uint8)


def default_diffusion_step(
    params: Dict[str, Any],
    x: jax.Array,
    t: int,
    rng: jax.Array
) -> jax.Array:
    """
    Placeholder diffusion step.
    Replace with real score model or noise scheduler in advanced versions.
    """
    noise_scale = params.get("noise_scale", 0.1)
    noise = jax.random.normal(rng, x.shape) * noise_scale
    return x + noise


# ───────────────────────────────────────────────
# Fused Pipeline (Single Compiled Graph)
# ───────────────────────────────────────────────

@partial(jax.jit, static_argnums=(2, 3, 4))
def fused_image_pipeline(
    params: Dict[str, Any],
    rng: jax.Array,
    num_steps: int,
    preprocess_fn: Callable = default_preprocess,
    postprocess_fn: Callable = default_postprocess,
    step_fn: Callable = default_diffusion_step,
    image: jax.Array = None
) -> jax.Array:
    """
    Fully fused end-to-end image generation pipeline.
    
    Args:
        params: Model parameters (e.g., {"noise_scale": 0.1})
        rng: PRNG key
        num_steps: Number of iterative refinement steps
        preprocess_fn: Input normalization (default: uint8 → [-1,1])
        postprocess_fn: Output conversion (default: [-1,1] → uint8)
        step_fn: Core iterative operation (e.g., diffusion, optimization)
        image: Input tensor of shape (H, W, C), dtype uint8
    
    Returns:
        Generated image of same shape, dtype uint8
    """
    # Preprocess
    x = preprocess_fn(image)

    # Iterative refinement loop
    def _body(i, state):
        x, r = state
        r_i = jax.random.fold_in(r, i)
        x = step_fn(params, x, i, r_i)
        return (x, r)

    x, _ = jax.lax.fori_loop(0, num_steps, _body, (x, rng))

    # Postprocess
    return postprocess_fn(x)


# ───────────────────────────────────────────────
# Batched Version (For Scalable Inference)
# ───────────────────────────────────────────────

batch_fused_pipeline = jax.vmap(
    fused_image_pipeline,
    in_axes=(None, 0, None, None, None, None, 0)
)


# ───────────────────────────────────────────────
# Utility: Latent Embedding (For /vector Integration)
# ───────────────────────────────────────────────

def embed_image(image: jax.Array) -> jax.Array:
    """
    Simple global average pooling embedding.
    Shape: (H, W, C) → (C,)
    Replace with CLIP or custom encoder as needed.
    """
    if image.ndim != 3:
        raise ValueError("Expected (H, W, C) image")
    return jnp.mean(image.astype(jnp.float32), axis=(0, 1))
