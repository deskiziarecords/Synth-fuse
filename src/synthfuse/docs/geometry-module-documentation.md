# Synth-Fuse Geometry Module Documentation

## Overview
The `synthfuse.geometry` module provides a set of utilities for working with camera poses and ray tracing in a 3D environment. It is specifically designed for differentiable rendering tasks and allows for manipulation of camera positions, creation of camera objects using the `visu3d` library, and tracing rays through implicit fields, like neural signed distance fields (SDFs).

### Directory Structure
- **`__init__.py`**: Initializes the package.
- **`camera_utils.py`**: Contains functions and classes for working with camera poses, creating cameras, perturbing camera positions, and converting camera trajectories into rays.
- **`primitives.py`**: Defines a JAX-compatible wrapper for the `visu3d.Ray` class, allowing for seamless integration with JAX.
- **`ray_tracer.py`**: Implements ray tracing through implicit fields using JAX.

---

## Module Components

### `camera_utils.py`

#### Description:
This module provides utilities for sampling camera poses, creating camera objects, perturbing camera positions, and getting rays from camera trajectories.

#### Functions:

- **`sample_spherical_poses`**
  ```python
  def sample_spherical_poses(
        key: jax.Array,
        num_poses: int,
        radius_range: Tuple[float, float] = (2.0, 5.0),
        elevation_range: Tuple[float, float] = (jnp.pi / 6, jnp.pi / 2),
        azimuth_range: Tuple[float, float] = (0.0, 2 * jnp.pi)
  ) -> jnp.ndarray
  ```
  Samples camera positions uniformly on the surface of a sphere.
  
  - **Parameters:**
      - `key`: Random key for JAX random number generation.
      - `num_poses`: Number of poses to sample.
      - `radius_range`: Tuple specifying the minimum and maximum radius of the spherical shell.
      - `elevation_range`: Tuple specifying the vertical angle range for pose sampling (in radians).
      - `azimuth_range`: Tuple specifying the horizontal angle range for pose sampling (in radians).
  
  - **Returns:** A `(num_poses, 3)` JAX array of spherical coordinates converted into Cartesian coordinates.

- **`create_cameras_from_poses`**
  ```python
  def create_cameras_from_poses(
        poses: jnp.ndarray,
        target: jnp.ndarray = jnp.array([0.0, 0.0, 0.0]),
        resolution: Tuple[int, int] = (256, 256),
        focal_px: float = 128.0
  ) -> "v3d.Camera"
  ```
  Creates a batch of `visu3d` camera objects pointing at a common target.
  
  - **Parameters:**
      - `poses`: A `(N, 3)` array of camera positions.
      - `target`: A `(3,)` array representing the world point that all cameras will look at.
      - `resolution`: Tuple indicating the resolution (height, width) of the camera.
      - `focal_px`: Focal length in pixels.
  
  - **Returns:** A `visu3d.Camera` object whose shape matches `poses.shape[:-1]`.

- **`perturb_camera_poses`**
  ```python
  def perturb_camera_poses(
        poses: jnp.ndarray,
        noise_scale: float = 0.1,
        key: Optional[jax.Array] = None
  ) -> jnp.ndarray
  ```
  Applies a differentiable perturbation to the camera poses for exploration.
  
  - **Parameters:**
      - `poses`: Input camera poses as a `(N, 3)` array.
      - `noise_scale`: Scale of the Gaussian noise to apply.
      - `key`: Optional random key for generating noise.
  
  - **Returns:** The perturbed camera poses.

- **`get_rays_from_trajectory`**
  ```python
  def get_rays_from_trajectory(
        trajectory: jnp.ndarray,
        target: jnp.ndarray = jnp.array([0.0, 0.0, 0.0]),
        resolution: Tuple[int, int] = (64, 64),
        focal_px: float = 32.0
  ) -> "v3d.Ray"
  ```
  Converts a camera trajectory into a batch of rays.
  
  - **Parameters:**
      - `trajectory`: An array of camera positions over time with shape `(T, 3)` or `(B, T, 3)`.
      - `target`: A `(3,)` array representing the scene center.
      - `resolution`: The resolution (height, width) of the camera.
      - `focal_px`: Focal length in pixels.
  
  - **Returns:** A `visu3d.Ray` object with shape matching the trajectory.

- **`make_camera_state`**
  ```python
  def make_camera_state(
        key: jax.Array,
        num_views: int = 8,
        **kwargs
  ) -> dict
  ```
  Initializes camera state for use in Synth-Fuse recipes.
  
  - **Parameters:**
      - `key`: Random key for JAX-based randomness.
      - `num_views`: Number of camera poses to generate.
      - `**kwargs`: Additional parameters passed to `sample_spherical_poses`.
  
  - **Returns:** A dictionary with keys "poses", "target", "resolution", and "focal_px".

---

### `primitives.py`

#### Description:
This module defines a class for rays that is compatible with JAX and integrates with the `visu3d` library.

#### Classes:

- **`Ray`**
  ```python
  class Ray:
      def __init__(self, pos: jnp.ndarray, dir: jnp.ndarray):
          ...
      @classmethod
      def from_v3d(cls, v3d_ray):
          ...
      def as_v3d(self):
          ...
      @property
      def shape(self):
          ...
      def __getitem__(self, idx):
          ...
  ```
  
  Represents a ray with its position and direction vectors.

  - **Constructor Parameters:**
      - `pos`: The starting position of the ray (a JAX array).
      - `dir`: The direction vector of the ray (a JAX array).
  
  - **Methods:**
      - `from_v3d`: Static method to convert a `visu3d.Ray` object to a `Ray`.
      - `as_v3d`: Converts the JAX `Ray` back to a `visu3d.Ray`.
      - `shape`: Property that returns the shape of the ray.
      - `__getitem__`: Allows indexing to retrieve sub-rays from the ray object.

---

### `ray_tracer.py`

#### Description:
This module contains functions to perform ray tracing through a specified implicit field using differentiable programming techniques.

#### Functions:

- **`trace_rays_through_field`**
  ```python
  @jax.jit
  def trace_rays_through_field(
        rays: Ray,
        field_fn: callable,
        num_steps: int = 64
  ) -> jnp.ndarray
  ```
  Performs a differentiable ray marching procedure through an implicit field.
  
  - **Parameters:**
      - `rays`: A `Ray` object containing positions and directional vectors.
      - `field_fn`: A callable that evaluates the implicit field (e.g., a neural SDF function that takes coordinates as input and outputs signed distance values).
      - `num_steps`: The number of steps to take during ray marching.
  
  - **Returns:** A JAX array representing the final points where the rays intersect the implicit field after marching.

---

This documentation should serve as a comprehensive guide for the `synthfuse.geometry` module, providing essential details about its components and usage.

---
