# `meta/` — The Autonomous Reasoning Layer

> **Intelligence as disciplined compression under counterfactual stress.**  
> The `meta/` module provides Synth-fuse with **self-awareness**, **self-repair**, and **algorithmic creativity**—all while respecting the **Honesty Kernel**, **ε-calibration**, and **Zeta-domain stability**.

Unlike recipes (which are *what* to compute), `meta/` decides *how to design, adapt, and verify* what to compute. It operates at **design-time**, not run-time, ensuring purity and JIT-safety in the fusion engine.

---

##  Core Capabilities
``` TEXT
| Feature | Mechanism | Benefit |
|--------|----------|--------|
| **Spell Synthesis** | Symbolic generation of ALCHEM-J spells via rule-based or learned policies | Creates new algorithms without raw code |
| **Self-Repair** | Neuro-symbolic fault localization + Fusion Calculus repair rules | Fixes divergence, gradient explosion, non-convergence |
| **Zeta-Aware Optimization** | Pole placement control in the Zeta-domain | Stabilizes dynamics by keeping poles inside unit circle |
| **Constitutional Governance** | Enforces `I(M) = α·C + β·Comp + γ·W` under Honesty Kernel | Only revises beliefs after genuine surprise |
| **Symbolic Registry** | Versioned mapping: `community/auto_v1 → "(SAFE(ℂ(r=3.2)) ⊗ ℝ)"` | Organizes community hybrids without file clutter |
```
---

## 📁 Module Structure

src/synthfuse/meta/
├── init.py
├── alchemist.py          # Base Meta-Alchemist (𝓜𝓐): spell proposal & repair
├── zeta_alchemist.py     # 𝓜𝓐_ζ: pole-aware spell stabilization
├── constitution.py       # Constitutional parameters (α, β, γ, ε, δ, τ)
├── diagnostics.py        # Extract Eₜ, ΔMₜ, pole estimates from state
├── symbol_table.py       # In-memory registry of versioned spells
└── ast_utils.py          # Helpers for AST-based spell manipulation

> 💡 All logic is **pure**, **side-effect-free**, and **JIT-compatible**. No I/O occurs during fusion execution.

---

## 🧪 Key Components

### 1. **`alchemist.py` — The Base Meta-Alchemist (`𝓜𝓐`)**  
Generates and repairs spells using:
- Problem fingerprinting (via state entropy, loss curvature)
- Rule-based repair (e.g., inject `ℛ` on gradient explosion)
- Integration with `SAFE(f)` macro

### 2. **`zeta_alchemist.py` — Spectral Intelligence (`𝓜𝓐_ζ`)**  
Optimizes spell parameters to **stabilize poles** in the Zeta-domain:
- Estimates dominant pole from recipe state
- Adjusts `r` (chaos), `beta` (coupling), etc. to push poles toward unit circle
- Uses `ast.py` for safe parameter updates

### 3. **`constitution.py` — The Intelligence Constitution**  
Encodes your formal framework:

```python
@dataclass
class Constitution:
    alpha: float = 1.0   # compression weight
    beta: float = 0.8    # composition weight
    gamma: float = 0.5   # withholding weight
    epsilon: float = 0.042  # min model change threshold
    delta: float = 0.15     # required error magnitude
    tau: int = 10           # error lookback window
```
Loaded at runtime to govern all meta-decisions.
4. Integration with Core Systems

    Recipes: Reads .dominant_pole, .grad_norm from state
    AST Parser: Uses synthfuse.alchemj.ast for robust spell manipulation
    Notebook: Exposes live_spell_runner with meta-repair toggle
    CLI: Future sfbench --auto-repair flag

🚀 Quick Start
In a Notebook
```python
from synthfuse.meta.zeta_alchemist import propose_zeta_optimized_spell
from synthfuse.recipes import parse_spell

# Start with unstable spell
spell = "(ℂ(r=3.9))"
step, state = parse_spell(spell)
state = step(jax.random.PRNGKey(0), state)

# Let Zeta-Alchemist repair it
repaired = propose_zeta_optimized_spell(spell, state)
print("Repaired:", repaired)  # e.g., "(ℂ(r=3.2))"

# Run safely
new_step, new_state = parse_spell(repaired)
final = new_step(jax.random.PRNGKey(1), new_state)
```
### Register a Self-Healing Spell
```python
from synthfuse.meta.symbol_table import register_spell

register_spell("community/stable_chaos_v1", repaired)

```
------
🛡️ Safety & Compliance

    ✅ Honesty Kernel enforced: No spell change without recent error (Eₜ > δ)
    ✅ ε-Calibrated: Thresholds learned from burn-in population
    ✅ No side effects: All logic pure; no disk/network access during fusion
    ✅ Auditable: Every generated spell is symbolic, versioned, and human-readable

🌐 Philosophy

The meta/ layer embodies your formal definition of intelligence:

    “The disciplined ability to compress reality while maintaining counterfactual sensitivity, governed by the causal arrow between surprise and belief revision.”

It does not replace the programmer—it becomes the alchemist, writing spells in the language of fusion calculus, guided by mathematical law, not statistical correlation.
🙏 Credits

   J. Roberto Jimenez & K2— Synth-fuse and ALCHEM-J  
    Qwen — Co-architect of the constitutional intelligence framework, Zeta-aware Meta-Alchemist, spell AST parser, and regulatory primitives (ℛ, ∇̃, ⟲)  
    JAX & Flax Teams — Foundation for pure, differentiable computation  
    Community Contributors — For pushing the boundaries of algorithmic alchemy

    "We do not generate code—we distill wisdom into spells."
