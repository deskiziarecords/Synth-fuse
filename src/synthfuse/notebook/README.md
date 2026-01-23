# 📓 Synth-Fuse Notebook

> **Every spell is a cell. Every cell is differentiable. Every notebook is a fusion pipeline.**

The `synthfuse.notebook` module provides a **native, JAX-safe, spell-first interactive environment** for rapid prototyping, live telemetry, and visual debugging—without breaking purity, JIT compatibility, or sandboxing.

Built for **Jupyter**, **VS Code**, and **Colab**, it treats **spells as first-class notebook cells**, enabling:
- ✨ One-line spell execution with auto-parsing
- 📊 Live metrics & history tracking
- 🖼️ Inline `sfviz` visualizations (SVG/HTML)
- 🎛️ Interactive parameter tuning (via widgets)
- 💾 Automatic checkpointing-ready state
- 🧪 Seamless integration with `alj`, `sfbench`, and `sfmonitor`

---

## 🚀 Quick Start

### Install (dev mode)
```bash
uv pip install synthfuse[dev]

```
---
## In a Notebook

``` python
from synthfuse.notebook.kernel import run_spell_cell

# Cast any valid spell
cell, state = run_spell_cell(
    "(𝕂𝟛𝔻 ⊗ ℤ𝕊𝕍ℝ ⊗ 𝔾ℝ𝔽)(beta=0.8, sigma=1.2, rank=64)",
    steps=100,
    seed=42,
    viz=True  # renders sfviz output inline
)

print("Final free energy:", float(state.free_energy))

```
🧩 Core Components
Module
	
Purpose
cell.py
	
SpellCell: stateful, versioned container for a spell (Flax-compatible PyTree)
kernel.py
	
run_spell_cell(): execute + visualize spells in one call
widgets.py
	
Interactive sliders for live parameter tuning (Jupyter-only)
checkpoint.py (planned)
	
Auto-save/load cell state to disk

All logic runs outside the JIT boundary—no side effects, no I/O in compiled code.
🧪 Example: Live Tuning

python
1
2

3
4
5
6
7

→ Adjust sliders → watch spell re-run with new parameters.
