✅ Key Design Principles
Principle
	
Implementation
Pure & Side-Effect Free
	
No I/O, no global state
Fully Fused
	
One jax.jit graph → single kernel launch
Composable
	
Swap step_fn, preprocess_fn, etc. via arguments
Batch-Native
	
batch_fused_pipeline for high-throughput
Vector-Ready
	
embed_image() enables /vector integration
Spell-Compatible
	
Can be wrapped in agents/local/tensor_fusion.py
----
🧪 Usage Examples

```python
import jax.numpy as jnp
from src.synthfuse.prototype.images.fusion_pipeline import fused_image_pipeline

params = {"noise_scale": 0.05}
rng = jax.random.PRNGKey(42)
image = jnp.zeros((256, 256, 3), dtype=jnp.uint8)

output = fused_image_pipeline(
    params, rng, num_steps=30, image=image
)
print(output.shape)  # (256, 256, 3)



```
--------
## Batch Interface

```python
batch_images = jnp.zeros((8, 256, 256, 3), dtype=jnp.uint8)
batch_rngs = jax.random.split(jax.random.PRNGKey(0), 8)

batch_output = batch_fused_pipeline(
    params, batch_rngs, num_steps=30, image=batch_images
)
print(batch_output.shape)  # (8, 256, 256, 3)
```
