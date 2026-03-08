# Synth-Fuse v0.3.0: System Dissection & Specifications

## 1. Architectural Philosophy: The Circulatory OS

Synth-Fuse is not an agentic framework; it is a **circulatory operating system**. Where traditional AI architectures rely on discrete message passing and interpretive overhead (token friction), Synth-Fuse treats computation as a compressed fluid circulating through a differentiable manifold.

### The Unified Field Thesis
All algorithms (Swarm, RL, Numerical) are reduced to the same functional signature:
`Φ(key, state, params) -> new_state`

This allows for **Zero-Copy Fusion**: the sequential composition of a SAT solver, a neural network, and a genetic optimizer into a single JAX/XLA kernel that executes without leaving the GPU/TPU substrate.

---

## 2. Component Dissection

### A. The Adiabatic Kernel (`src/synthfuse/os/kernel.py`)
The heartbeat of the system. It enforces the **Three Laws of Thermodynamics**:
1. **Physical Supremacy**: Sensors (Hardware Veto) override consensus. If the `Regulator` reports a thermal spike, execution is killed regardless of agent agreement.
2. **Encapsulation**: Playground (untrusted) outputs are sealed in a **Stochastic Wrapper** (Firecracker/Vault) before Factory deployment.
3. **Leashed Exploration**: Exploration in Auto-mode is capped by a TDP budget (20% baseline), extensible only by Lab certification.

### B. WeightKurve (`src/synthfuse/lab/instruments/weight_kurve.py`)
Treats neural weight trajectories as **Stellar Light Curves**.
- **Oscillations**: Detects high-frequency resonance (unstable learning rates).
- **Transits**: Detects sudden drops in saliency (Out-of-Distribution events).
- **Chaos**: Estimates the maximum Lyapunov exponent (instability metric).

### C. Neural Tool-Embedding Protocol (NTEP)
A deterministic mapping `ϕ : source → τ ⊕ σ`.
- **τ (Task Vector)**: TF-IDF weighted semantic projection.
- **σ (Signature)**: Cryptographic hash of functional behavior.
- NTEP allows the OS to "retrieve" code modules from the manifold with O(1) complexity.

---

## 3. The Six Realms of Governance

| Realm | Purpose | Governance Model |
|-------|---------|------------------|
| **Factory** | Assembly | Byzantine Quorum (7/7 consensus) |
| **Playground** | Creation | Unbounded (Sandboxed) |
| **Auto-mode** | Discovery | TDP Gated (Entropy Gradient) |
| **Lab** | Validation | Zero False Positive (Hard Benchmarks) |
| **Thermo** | Governance | Physical Veto (Sensor Supremacy) |
| **Substrate** | Foundation | Implicit (Deterministic JAX) |

---

## 4. Technical Specifications

### I. Thermal Constants
- `THERMAL_HARD_LIMIT = 0.85`: Immediate hardware veto.
- `THERMAL_THROTTLE = 0.80`: Cycle-rate reduction.
- `ENTROPY_HALT = 0.30`: Information disorder limit.

### II. Functional Combinators
- **⊗ (Sequential)**: Pipe output of A to input of B.
- **⊕ (Parallel)**: Execute A and B simultaneously, merge states.
- **∘ (Conditional)**: Route execution based on predicate `φ`.
- **⋈ (Fusion)**: In-place topological join.

### III. System Requirements
- **Python**: >= 3.10
- **Accelerator**: JAX/XLA compatible (CUDA/ROCm/TPU)
- **Security**: TPM/HSM support for Vault operators.

---
*Verified for release v0.3.0 by the Librarian and the Physician.*
