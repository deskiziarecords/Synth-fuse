# 🧪 Synth-Fuse LLM Recipe Guide

Welcome to the **Synth-Fuse Alchemical Kitchen**. This document serves as the authoritative guide for Large Language Models (LLMs) to understand, construct, and execute **Recipes** within the Synth-Fuse (v0.5.0+) ecosystem.

---

## 1. What is a Recipe?

In Synth-Fuse, a **Recipe** (or **Spellcard**) is a formal blueprint that bridges human-readable intent with **ALCHEM-J** (the symbolic spell language) and **JAX-native** execution kernels.

A Recipe is composed of:
1.  **Essence**: The core "why" and "what" of the computational workflow.
2.  **Sigil**: The symbolic representation using ALCHEM-J operators (⊗, ⊕, ∘, ⋈).
3.  **Alphabet Mapping**: A table defining what each Unicode symbol represents in the specific context.
4.  **The Spell**: The actual ALCHEM-J string invoked by the `circulate` runtime.
5.  **Denotational Semantics**: The mathematical/logical flow of the transformation.
6.  **Runtime API**: A Python example showing how to initialize and execute the spell.

---

## 2. The 5 Foundational Recipes

Below are five distinct recipes illustrating the range of Synth-Fuse capabilities, from repository ingestion to stochastic swarm optimization.

### ✦ 1. Ingest — The Forager's Grimoire
**Sigil:** `(𝕋⊗𝔽⊗ℂ⊗𝕍⊗𝕃)`
**Tagline:** *"Harvest source, distill essence, feed the machine"*

| Symbol | Operator | Meaning in ingest Context |
| :--- | :--- | :--- |
| **𝕋** | Traverse | File-system walker + web crawler — builds source tree |
| **𝔽** | Filter | Glob include/exclude patterns, git diff/log selectors |
| **ℂ** | Compress | Tree-sitter structural extraction (signatures sans bodies) |
| **𝕍** | Validate | Token counting + VRAM estimation + model compatibility |
| **𝕃** | LLM-bridge | OpenAI-compatible API piping (Ollama, etc.) |

**The Spell:**
```alchem-j
(𝕋(paths=["./src", "https://example.com"], depth=3)
 ⊗ 𝔽(include="**/*.go", exclude="**/*_test.go")
 ⊗ ℂ(language=go, preserve=signatures)
 ⊗ 𝕍(model="llama3.1:8b", quant="q4_k_m", memory=8)
 ⊗ 𝕃(api="ollama", prompt="explain this code"))
```

**Semantics:** `⟦(𝕋⊗𝔽⊗ℂ⊗𝕍⊗𝕃)⟧ ≜ λ(k, x, p). 𝕃(k, 𝕍(k, ℂ(k, 𝔽(k, 𝕋(k, x, p), p), p), p), p)`

---

### ✦ 2. Renoun — The Codex Transmutation
**Sigil:** `(𝔻⊗𝕼⊗ℝ⊗𝕊)`
**Tagline:** *"Source becomes structure, structure becomes content"*

| Symbol | Operator | Meaning in renoun Context |
| :--- | :--- | :--- |
| **𝔻** | Directory | File-system as database — new Directory({ path, loader }) |
| **𝕼** | Query | Structured extraction — getFile(), getFiles(), type inference |
| **ℝ** | Render | MDX/TSX execution — getContent(), component hydration |
| **𝕊** | Schema | Validation layer — Zod-powered frontmatter/export validation |

**The Spell:**
```alchem-j
(𝔻(path="content", loader=mdx)
 ⊗ 𝕼(filter=schema, sort=date)
 ⊗ ℝ(hydrate=true, components=ui)
 ⊗ 𝕊(validator=zod, strict=true))
```

---

### ✦ 3. Extension.js — Cross-Browser Manifest
**Sigil:** `(𝔼⊗𝕋⊗𝔹)`
**Tagline:** *"One source, many masks, instant reload"*

| Symbol | Operator | Meaning in Context |
| :--- | :--- | :--- |
| **𝔼** | Extension | The core manifest/compiler that transforms source into browser output |
| **𝕋** | Template | Framework bindings (React, Vue, Svelte, TypeScript) |
| **𝔹** | Browser | Multi-target emission (Chromium, Gecko, WebKit engines) |

**Runtime API:**
```python
from synthfuse import circulate

step, state = circulate(
    "(𝔼⊗𝕋⊗𝔹)",
    init_state={"src": "./my-extension/"},
    key=jax.random.PRNGKey(0),
    template="react",
    browsers=["chrome", "firefox"],
    hmr=True
)
```

---

### ✦ 4. Jina-Serve — Conceptual Spellcard
**Sigil:** `(𝔼⊗𝔽⊗𝔻⊗ℂ)`
**Tagline:** *"Define the Executor, weave the Flow, serve the Intelligence"*

| Symbol | Operator | Meaning |
| :--- | :--- | :--- |
| **𝔼** | Executor | The fundamental compute unit (e.g., a model, a preprocessor). |
| **𝔽** | Flow | The orchestration layer (DAG of Executors). |
| **𝔻** | Deployment | The runtime environment (scaling, resources, protocol). |
| **ℂ** | Client | The entry point for streaming results. |

**Advanced Multi-Modal Spell:** `(𝔼(Text) ⊕ 𝔼(Image)) ⊗ 𝔽(Fusion) ⊗ 𝔻(K8s)`

---

### ✦ 5. RISO-Lévy Swarm — Tier 1 Hybrid
**Sigil:** `(𝕀⊗𝕃(α=1.5))`
**Tagline:** *"Stochastic escape from local minima"*

**Essence:** Fuses deterministic ISO/RIME updates (invariant state optimization) with stochastic Lévy random vectors.

**Implementation (Cabinet of Alchemists):**
```python
async def main():
    cabinet = CabinetOrchestrator()
    await cabinet.initialize()

    # 𝕀 = ISO/RIME update (Deterministic)
    # ⊗ = fuse_seq (Sequential fusion)
    # 𝕃 = Lévy random vector (Stochastic exploration)
    sigil = "(𝕀(rate=0.01) ⊗ 𝕃(α=1.5))"

    result_state = await cabinet.process_sigil(sigil, initial_state, key=key)
```

---

## 3. Recipe Template for LLMs

To create a new Recipe, follow this Markdown structure:

```markdown
# Recipe: [Name]

**Sigil:** ([Symbol] [Combinator] [Symbol])
**Tagline:** "[Catchy phrase summarizing utility]"

## Essence
[Describe the purpose, mechanism, and target goal of the recipe.]

## Alphabet Mapping
| Symbol | Operator | Meaning |
| :--- | :--- | :--- |
| **[X]** | [Name] | [Functional Description] |

## The Spell (ALCHEM-J)
`([Expression])`

## Denotational Semantics
⟦[Sigil]⟧ ≜ λ(k, x, p). [Function Logic]

## Runtime Execution
```python
# Python snippet using synthfuse.circulate or CabinetOrchestrator
```
```

---
*Calibrated by Google's Jules — Systems Calibrating Engineer.*
