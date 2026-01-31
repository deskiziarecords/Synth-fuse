# src/synthfuse/agents/local/tensor_fusion.py
"""
Tensor-Graph Fusion Agent – On-device, JAX-native image generator.
Fully fused: preprocess → diffusion loop → postprocess in one JIT graph.
Runs entirely in the red layer (no external calls).
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Any, Dict

from synthfuse.agents.base import LocalAgent, register_agent, AgentResponse


@partial(jax.jit, static_argnums=(2,))
def _fused_image_generator(
    params: Dict[str, Any],
    rng: jax.Array,
    num_steps: int,
    image: jax.Array
) -> jax.Array:
    """
    End-to-end fused pipeline compiled as a single JAX graph.
    Input:  [H, W, C] uint8 image (e.g., noise or sketch)
    Output: [H, W, C] uint8 generated image
    """
    # Preprocess: normalize to [-1, 1]
    x = image.astype(jnp.float32) / 127.5 - 1.0

    # Diffusion loop (placeholder: Gaussian smoothing)
    def body_fn(i, state):
        x, r = state
        r_i = jax.random.fold_in(r, i)
        noise_scale = params.get("noise_scale", 0.1)
        noise = jax.random.normal(r_i, x.shape) * noise_scale
        x = x + noise
        return (x, r)

    x, _ = jax.lax.fori_loop(0, num_steps, body_fn, (x, rng))

    # Postprocess: back to [0, 255] uint8
    x = jnp.clip((x + 1.0) * 127.5, 0, 255)
    return x.astype(jnp.uint8)


@register_agent("tensor_fusion")
class TensorFusionAgent(LocalAgent):
    """
    Local agent for real-time, on-device generative fusion.
    Ideal for edge deployment, browser (via IREE), or spell embedding.
    """

    def __init__(
        self,
        steps: int = 20,
        noise_scale: float = 0.1,
        seed: int = 0
    ):
        self.steps = steps
        self.params = {"noise_scale": noise_scale}
        self.seed = seed

    def generate(self, prompt: jax.Array, **kwargs) -> AgentResponse:
        """
        Generate image from input tensor.
        prompt: jnp.ndarray of shape (H, W, C), dtype uint8
        """
        if not isinstance(prompt, jax.Array):
            raise TypeError("Prompt must be a JAX array (e.g., jnp.zeros((256,256,3), dtype=jnp.uint8))")

        rng = jax.random.PRNGKey(kwargs.get("seed", self.seed))
        output = _fused_image_generator(self.params, rng, self.steps, prompt)

        return AgentResponse(
            content=output,
            meta={
                "steps": self.steps,
                "noise_scale": self.params["noise_scale"],
                "device": str(output.device()),
                "latency_ms": None  # measured externally
            },
            vector=self.embed(output)  # auto-embed for forum use
        )

    async def agenerate(self, prompt: jax.Array, **kwargs) -> AgentResponse:
        # Local agents are synchronous; async is passthrough
        return self.generate(prompt, **kwargs)

    def embed(self, image: jax.Array) -> jnp.ndarray:
        """
        Simple latent embedding: global average pool + flatten.
        Replace with CLIP or custom encoder in advanced versions.
        Shape: (C,) → can be expanded to (D,) via MLP if needed.
        """
        if image.ndim != 3:
            raise ValueError("Expected (H, W, C) image")
        # Global average pooling
        emb = jnp.mean(image.astype(jnp.float32), axis=(0, 1))  # (C,)
        return emb
