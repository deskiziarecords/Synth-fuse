# src/synthfuse/geometry/camera_utils.py
"""
Camera utilities for Synth-Fuse: batched, differentiable, visu3d-native.
Enables swarm/RL-guided viewpoint exploration in 3D space.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Union
from functools import partial

# Lazy import to keep core lightweight
def _import_v3d():
    try:
        import visu3d as v3d
        return v3d
    except ImportError:
        raise ImportError("Install 'synthfuse[geometry]' for 3D camera support.")


@partial(jax.jit, static_argnums=(1, 2))
def sample_spherical_poses(
    key: jax.Array,
    num_poses: int,
    radius_range: Tuple[float, float] = (2.0, 5.0),
    elevation_range: Tuple[float, float] = (jnp.pi / 6, jnp.pi / 2),  # 30°–90°
    azimuth_range: Tuple[float, float] = (0.0, 2 * jnp.pi)
) -> jnp.ndarray:
    """
    Sample camera positions on a spherical shell around origin.
    Returns: (num_poses, 3) array of world positions.
    """
    r_key, el_key, az_key = jax.random.split(key, 3)
    
    r = jax.random.uniform(r_key, (num_poses,), minval=radius_range[0], maxval=radius_range[1])
    el = jax.random.uniform(el_key, (num_poses,), minval=elevation_range[0], maxval=elevation_range[1])
    az = jax.random.uniform(az_key, (num_poses,), minval=azimuth_range[0], maxval=azimuth_range[1])
    
    # Spherical → Cartesian
    x = r * jnp.sin(el) * jnp.cos(az)
    y = r * jnp.sin(el) * jnp.sin(az)
    z = r * jnp.cos(el)
    
    return jnp.stack([x, y, z], axis=-1)  # (N, 3)


def create_cameras_from_poses(
    poses: jnp.ndarray,
    target: jnp.ndarray = jnp.array([0.0, 0.0, 0.0]),
    resolution: Tuple[int, int] = (256, 256),
    focal_px: float = 128.0
) -> "v3d.Camera":
    """
    Create a batch of visu3d cameras looking at a common target.
    Args:
        poses: (..., 3) camera positions
        target: (3,) world point all cameras look at
        resolution: (H, W)
        focal_px: focal length in pixels
    Returns:
        v3d.Camera with shape matching poses.shape[:-1]
    """
    v3d = _import_v3d()
    
    # Broadcast target to match poses
    target = jnp.broadcast_to(target, poses.shape)
    
    # Create pinhole spec
    spec = v3d.PinholeCamera.from_focal(
        resolution=resolution,
        focal_in_px=focal_px
    )
    
    # Build cameras
    cams = v3d.Camera.from_look_at(
        spec=spec,
        pos=poses,
        target=target
    )
    
    return cams


@jax.jit
def perturb_camera_poses(
    poses: jnp.ndarray,
    noise_scale: float = 0.1,
    key: Optional[jax.Array] = None
) -> jnp.ndarray:
    """
    Differentiable perturbation of camera poses (for swarm exploration).
    Input: (..., 3)
    Output: (..., 3)
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    noise = jax.random.normal(key, poses.shape) * noise_scale
    return poses + noise


def get_rays_from_trajectory(
    trajectory: jnp.ndarray,
    target: jnp.ndarray = jnp.array([0.0, 0.0, 0.0]),
    resolution: Tuple[int, int] = (64, 64),
    focal_px: float = 32.0
) -> "v3d.Ray":
    """
    Convert a camera trajectory into a batch of rays.
    Args:
        trajectory: (T, 3) or (B, T, 3) camera positions over time
        target: (3,) scene center
        resolution: (H, W)
        focal_px: focal length
    Returns:
        v3d.Ray with shape (*trajectory.shape[:-1], H, W)
    """
    cams = create_cameras_from_poses(
        trajectory, target=target, resolution=resolution, focal_px=focal_px
    )
    rays = cams.rays()  # (*traj_shape, H, W)
    return rays.as_jax()  # Ensure JAX arrays


# ───────────────────────────────────────
# Utility for Spell Integration
# ───────────────────────────────────────

def make_camera_state(
    key: jax.Array,
    num_views: int = 8,
    **kwargs
) -> dict:
    """Initialize camera state for use in Synth-Fuse recipes."""
    poses = sample_spherical_poses(key, num_views, **kwargs)
    return {
        "poses": poses,           # (N, 3)
        "target": jnp.zeros(3),   # fixed for now
        "resolution": (256, 256),
        "focal_px": 128.0
    }
