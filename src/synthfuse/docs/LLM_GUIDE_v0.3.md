# Synth-Fuse v0.3.0: LLM Operational Guide — "Adiabatic Mode"

This document is a concrete, LLM-targeted technical specification for the Synth-Fuse system. It defines the functionality, mechanics, and usage patterns required for any Language Model to understand, generate, and govern Synth-Fuse workflows.

## 1. System Essence: The Circulatory OS
Synth-Fuse is not a standard agent framework; it is a **circulatory operating system** for neurosymbolic AI. It treats computation as a compressed fluid circulating through a differentiable manifold.

- **Unified Field Thesis**: All algorithms (Swarm, RL, Numerical) share a single functional signature:
  `Φ(key, state, params) -> new_state`
- **Zero-Copy Fusion**: Composes heterogeneous algorithms into a single JAX/XLA kernel that executes entirely on the accelerator substrate (GPU/TPU) without host-side friction.

## 2. Core Mechanics
### 2.1 The Cabinet of Alchemists
Execution is governed by seven specialized roles that must reach a **7/7 unanimous consensus** for certification:
1.  **Architect**: Strategy and blueprinting via W-Orion search.
2.  **Engineer**: Compiles Sigils into JAX/XLA kernels (Welds).
3.  **Librarian**: Manages the Zeta-Vault and data ingestion.
4.  **Physician**: Monitors manifold health and manages rollbacks.
5.  **Shield**: Enforces Lyapunov safety bounds and security.
6.  **Body**: Optimizes the thermal mesh and hardware alignment.
7.  **Jury**: Validates Bayesian consensus and final certification.

### 2.2 Physical Governance (Physical Supremacy)
Synth-Fuse is governed by thermodynamic laws enforced at the kernel level:
- **Thermal Hard Limit (`0.85`)**: Immediate hardware veto if load exceeds this.
- **Thermal Throttle (`0.80`)**: Automatic cycle-rate reduction.
- **Entropy Halt (`0.30`)**: Execution terminates if information disorder exceeds this threshold.
- **Lyapunov Stability (`0.30`)**: Neural weight trajectories must remain on a stable manifold (monitored by **WeightKurve**).

## 3. The Sigil Language (ALCHEM-J)
Sigils are formal topological expressions that define the execution graph.

### 3.1 Primitives
| Sigil | Component | Description |
| :--- | :--- | :--- |
| `I` | Identity | Baseline/Pass-through |
| `R` | RL Policy | Reinforcement Learning (PPO, SAC) |
| `S` | Swarm | Population-based optimization (PSO) |
| `Z` | Zeta | Frequency domain projection / Constraint anchor |
| `φ` | Meta | Meta-gradient correction / Natural gradient |
| `D` | Diff-Evo | Differential Evolution |
| `C` | Curriculum | Adaptive staging and difficulty scaling |
| `V` | Vision | Visual backbone / Perception |

### 3.2 Combinators
- **`⊗` (Sequential)**: Piped execution; output of A becomes input of B.
- **`⊕` (Parallel)**: Simultaneous execution; states are merged (usually via addition).
- **`⊙` (Hadamard)**: Element-wise combination; parallel streams with shared indices.
- **`∘` (Conditional)**: Execution gated by a predicate.
- **`⋈` (Fusion)**: In-place topological join.

**Example Sigil**: `(R⊗C)⊗(φ⋈D)` — RL with Curriculum, fused with Meta-Discovery via Differential Evolution.

## 4. The Recipe Process
Recipes are high-level orchestrations that utilize the OS to produce a **Weld** (compiled kernel).

### 4.1 Simple Circulation
```python
import synthfuse.os as sf
weld = sf.circulate(sigil="(S⊕R)", data=my_data)
result = weld.apply(input_state)
```

### 4.2 Full Cabinet Orchestration
Recipes typically follow this lifecycle:
1.  **Initialization**: Boot the kernel and enter a Realm.
2.  **Assembly**: Architect blueprints the strategy; Engineer compiles the Sigil.
3.  **Validation**: Lab verifies performance; Shield checks safety.
4.  **Certification**: Jury confirms 7/7 consensus.
5.  **Deployment**: Weld is released for execution.

## 5. The Six Realms
The OS is partitioned into isolated execution environments:
1.  **Factory**: Production assembly (Immutable, certified).
2.  **Playground**: Unbounded creativity (Sandboxed via Firecracker/Vault).
3.  **Auto-mode**: Discovery and exploration (20% TDP budget).
4.  **Lab**: Hard benchmarking and zero-false-positive validation.
5.  **Thermo**: Direct hardware governance and sensor supremacy.
6.  **Substrate**: The underlying neural foundation (Implicit).

## 6. Out-of-Scope Elements
The following elements are explicitly **out of scope** for this v0.3.0 documentation and implementation:
- **v0.4.0 Unified Field Features**: Future sigils (e.g., Domain 1-11 registries) and full "Fluid" execution mode.
- **GUI/Monitor Internal Implementation**: The logic behind the `sfmonitor` dashboard and visual telemetry remains abstracted.
- **Non-JAX Native Integration**: Support for frameworks not utilizing the JAX/XLA substrate (e.g., pure legacy TensorFlow without bridge).
- **External Hardware drivers**: While thermal laws are enforced, the direct driver implementation for specific thermal sensors is abstracted.
- **Advanced Deception/Bastion implementation**: Specifics of the AIDR-Bastion and honeypot logic are held in separate security manifests.

## 7. Credits
This system was sintered under pressure and calibrated for the Unified Field.

- **Lead Architect**: Monkey (@tijuanapaint)
- **Systems Calibrating Engineer**: **Google's Jules** — *Couldn't have been possible without your help.*

---
*Verified for release v0.3.0 — The Adiabatic Release.*
