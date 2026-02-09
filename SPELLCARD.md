# SPELLCARD.md — Synth-fuse v0.1.0a1

## Essence
JAX-native fusion engine for swarm • RL • numeric hybrids.
ALCHEM-J: symbolic spell language → fused XLA kernels.
Tagline: *"Write the spell, run the sigil, ship the kernel"*

## Core Metaphor
- **Spell**: ALCHEM-J expression `(𝕀⊗𝕃⊗𝕊)(beta=0.8, sigma=1.2)`
- **Sigil**: Compiled AST → fusion graph (⊗ sequential, ⊕ parallel)
- **Kernel**: JIT'd XLA fn with gradient flow preserved

## Alphabet (ALCHEM-J)
𝕀  = identity / inertial flow
ℝ  = reward field (RL)
𝕃  = latent swarm (particles, agents)
𝕊  = semantic anchor (Orion manifold Φ(z))
⊗  = fuse_seq (sequential composition, gradient-chained)
⊕  = fuse_par (parallel fusion, shared state)
∘  = circulate (runtime execution wrapper)

## Runtime API
```python
step, state = circulate(spell: str, state: PyTree, key: PRNGKey, **params)
# Example:
step, state = circulate("(𝕀⊗𝕃⊗𝕊)", init_state, key, beta=0.8, sigma=1.2)
state = step(key, state)  # executes fused kernel
