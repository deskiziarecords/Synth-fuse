# src/synthfuse/agents/local/ray_fusion.py
from synthfuse.agents.base import LocalAgent, register_agent, AgentResponse
from synthfuse.geometry.ray_tracer import trace_rays_through_field
from synthfuse.geometry.primitives import Ray
import jax.numpy as jnp

@register_agent("ray_fusion")
class RayFusionAgent(LocalAgent):
    def generate(self, prompt: jnp.ndarray, **kwargs) -> AgentResponse:
        # prompt = (H, W, 3) camera pose or noise
        v3d = _import_v3d()
        cam = v3d.Camera.from_look_at(
            spec=v3d.PinholeCamera.from_focal(resolution=prompt.shape[:2], focal_in_px=50.),
            pos=[2, 2, 2],
            target=[0, 0, 0]
        )
        rays_v3d = cam.rays()
        rays = Ray.from_v3d(rays_v3d)

        # Dummy SDF: sphere
        def sdf(points):
            return jnp.linalg.norm(points, axis=-1) - 1.0

        hit_points = trace_rays_through_field(rays, sdf)
        depth = jnp.linalg.norm(hit_points - rays.pos, axis=-1)

        return AgentResponse(
            content=depth,
            meta={"shape": depth.shape},
            vector=jnp.mean(depth)
        )

    def embed(self, x):
        return jnp.array([float(jnp.mean(x))])
