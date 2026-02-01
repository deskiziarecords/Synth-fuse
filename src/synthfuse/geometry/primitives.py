# src/synthfuse/geometry/primitives.py
import jax.numpy as jnp
from typing import Any

# Lazy import to keep core lightweight
def _import_v3d():
    try:
        import visu3d as v3d
        return v3d
    except ImportError:
        raise ImportError("Install 'synthfuse[geometry]' for 3D support.")

class Ray:
    """JAX-compatible wrapper around visu3d.Ray."""
    def __init__(self, pos: jnp.ndarray, dir: jnp.ndarray):
        self.pos = pos
        self.dir = dir

    @classmethod
    def from_v3d(cls, v3d_ray):
        v3d = _import_v3d()
        ray_jax = v3d_ray.as_jax()
        return cls(ray_jax.pos, ray_jax.dir)

    def as_v3d(self):
        v3d = _import_v3d()
        return v3d.Ray(pos=self.pos, dir=self.dir)

    @property
    def shape(self):
        return self.pos.shape[:-1]

    def __getitem__(self, idx):
        return Ray(self.pos[idx], self.dir[idx])
