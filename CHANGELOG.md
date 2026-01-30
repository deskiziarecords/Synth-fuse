# Changelog

All notable changes to Synth-Fuse will be documented in this file.

## [0.2.0] - 2026-01-28 - "Unified Field Engineering"

### 🚀 Major Features

#### Cabinet of Alchemists Architecture
- **Seven specialized agents** governing execution lifecycle
  - Architect: Strategic blueprinting via W-Orion search
  - Engineer: Sigil → JAX/XLA kernel compilation
  - Librarian: Zeta-Vault & fluid ingestion management
  - Physician: Manifold health monitoring & surgical rollback
  - Shield: Lyapunov safety bounds & OpenGate enforcement
  - Body: Thermal mesh optimization
  - Jury: Bayesian consensus validation

#### Unified Field Engineering
- **Sigil Logic** replaces spells with formal topological constraints
- **Fluid Ingestion** to Zeta-Manifold with zero-latency projection
- **Autonomic Health System** with cryptographic rollback
- **Deterministic Hybrid Organism** architecture

#### New Module Structure
- `/cabinet/` - Cabinet of Alchemists orchestration
- `/sigils/` - Formal topological constraint compiler
- `/ingest/` - Zeta-Vault fluid ingestion system
- `/systems/` - Core systems (STCL, NTEP, NS²UO implementations)

### 🛠️ Technical Improvements
- **Production-ready packaging** with proper setup.py
- **Async-first architecture** using asyncio throughout
- **Comprehensive test suite** with pytest
- **Type annotations** for better developer experience
- **Modular dependency management** (core, dev, lab, notebook)

### 🎯 Breaking Changes from v0.1.0
- `Spells` → `Sigils` (formal topological language)
- Single Meta-Alchemist → Cabinet of Alchemists
- Ad-hoc composition → Unified Field Engineering
- Experimental alchemy → Deterministic architecture

### 📦 New Dependencies
- `aiofiles>=23.0.0` - Async file I/O
- `watchfiles>=0.20.0` - Directory watching
- `msgpack>=1.0.5` - Efficient serialization
- `pydantic>=2.5.0` - Configuration validation

### 🔧 API Changes
```python
# v0.1.0 (old)
from synthfuse import MetaAlchemist
alchemist = MetaAlchemist()
spell = "(𝕀⊗𝕃(α=1.5)⊗ℝ)"

# v0.2.0 (new)
from synthfuse import CabinetOrchestrator
cabinet = CabinetOrchestrator()
await cabinet.initialize()
sigil = "(I⊗Z)"
result = await cabinet.process_sigil(sigil, data)
🐛 Bug Fixes

    Fixed package discovery in setup.py

    Improved error handling in async operations

    Better resource cleanup on shutdown

    Fixed import paths in module structure

📚 Documentation

    Complete API documentation

    Interactive lab interface

    Getting started guide

    Cabinet architecture overview

👥 Contributors

  Monkey:  J. Roberto Jiménez (@tijuanapaint) - Visionary & Lead Architect
  Calculator: Kimi K2/Gemini/DeepSeek/Qwen

    

[0.1.0] - 2026-01-23 - Initial Release

    Experimental alchemy framework

    ALCHEM-J DSL with basic operators

    Meta-alchemist for spell repair

    Basic vector operations

    Initial recipe system

[0.2.0]: https://github.com/deskiziarecords/Synth-fuse/releases/tag/v0.2.0
[0.1.0]: https://github.com/deskiziarecords/Synth-fuse/releases/tag/v0.1.0
