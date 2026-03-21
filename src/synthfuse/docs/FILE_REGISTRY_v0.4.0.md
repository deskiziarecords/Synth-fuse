# File Registry — v0.4.0 Unified Field Upgrade
Session ID: sf-2026-03-07-unified-field

This registry lists all files created or significantly modified during the transition from v0.2.0 Cabinet architecture to v0.4.0 Unified Field OS.

## 核心 (Core) — OS Kernel
- `src/synthfuse/os/kernel.py`: Core SynthFuseOS v0.4 implementation.
- `src/synthfuse/os/__init__.py`: Exporting new kernel functions (boot, os, shutdown).
- `src/synthfuse/session_logger.py`: Cryptographic session provenance logging.

## 领域 (Realms) — The Six Realms
- `src/synthfuse/realms/factory.py`: Production assembly logic.
- `src/synthfuse/realms/playground.py`: Creative sandbox and Stochastic Wrapper.
- `src/synthfuse/realms/automode.py`: Leashed exploration within thermal boundaries.
- `src/synthfuse/realms/lab.py`: Hard validation and WeightKurve certification.
- `src/synthfuse/realms/thermo.py`: Physical governance and sensor veto.
- `src/synthfuse/realms/__init__.py`: Realm exports.

## 仪器 (Instruments) — Physical Layer
- `src/synthfuse/lab/instruments/base.py`: LabInstrument base class.
- `src/synthfuse/lab/instruments/weight_kurve.py`: Updated WeightKurve with thermal stress signatures.

## 基础架构 (Infrastructure) — Aligned Components
- `src/synthfuse/systems/ntep.py`: Updated with NTEP class wrapper.
- `src/synthfuse/systems/archiver.py`: New Archiver class for uncertainty resolution.
- `src/synthfuse/systems/__init__.py`: Updated exports (NTEP, Archiver).
- `src/synthfuse/forum/arena.py`: New Arena class replacing procedural logic.
- `src/synthfuse/forum/__init__.py`: Updated exports (Arena).
- `src/synthfuse/meta/regulator.py`: Updated with Regulator class wrapper.
- `src/synthfuse/meta/__init__.py`: Updated exports (Regulator).

## 文档 (Documentation)
- `src/synthfuse/docs/SESSION_SIGILS_v0.4.0.md`: Complete sigil compendium.
- `src/synthfuse/docs/FILE_REGISTRY_v0.4.0.md`: This file.

---
*Verified by Physician and Librarian — v0.4.0*
