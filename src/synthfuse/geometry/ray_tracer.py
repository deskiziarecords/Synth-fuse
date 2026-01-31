# src/synthfuse/geometry/ray_tracer.py
import jax
import jax.numpy as jnp
from .primitives import Ray

@jax.jit
def trace_rays_through_field(
    rays: Ray,
    field_fn: callable,  # e.g., neural SDF: (B, 3) -> (B,)
    num_steps: int = 64
) -> jnp.ndarray:
    """Differentiable ray marching through implicit field."""
    origins = rays.pos  # (..., 3)
    directions = rays.dir  # (..., 3)
    
    # Flatten for vmap
    flat_origins = origins.reshape(-1, 3)
    flat_dirs = directions.reshape(-1, 3)
    
    def march_step(carry, _):
        t, points = carry
        new_points = flat_origins + flat_dirs * t[:, None]
        sdf_vals = field_fn(new_points)  # (N,)
        t = t + jnp.maximum(sdf_vals, 0.01)  # safe step
        return (t, new_points), sdf_vals

    t0 = jnp.zeros(flat_origins.shape[0])
    (_, final_points), _ = jax.lax.scan(
        march_step, (t0, flat_origins), None, length=num_steps
    )
    
    return final_points.reshape(origins.shape)
