#!/usr/bin/env python3
"""
Synth-Fuse v0.2.0 - Cabinet of Alchemists Bootstrap
"""

import sys
import asyncio
from pathlib import Path
from typing import Optional

from synthfuse.cabinet.cabinet_orchestrator import CabinetOrchestrator
from synthfuse.ingest.watcher import LibrarianWatcher


def print_banner():
    """Display the Cabinet startup banner."""
    banner = """
    ┌────────────────────────────────────────────────────────────┐
    │                  CABINET OF ALCHEMISTS                     │
    │                    Synth-Fuse v0.2.0                       │
    │              Unified Field Engineering System              │
    └────────────────────────────────────────────────────────────┘
    
    Status: ONLINE
    Zeta-Vault: ACTIVE
    OpenGate: CERTIFIED
    
    Roles:
      • Architect   - Strategic blueprinting via W-Orion search
      • Engineer    - Sigil → JAX/XLA kernel compilation
      • Librarian   - Zeta-Vault & fluid ingestion
      • Physician   - Manifold health monitoring
      • Shield      - Lyapunov safety enforcement
      • Body        - Thermal mesh optimization
      • Jury        - Bayesian consensus validation
    
    Watching: ./ingest/raw/ for native data ingestion
    """
    print(banner)


async def main_async():
    """Async main entry point."""
    print_banner()
    
    # Create ingest directory if it doesn't exist
    ingest_dir = Path.cwd() / "ingest" / "raw"
    ingest_dir.mkdir(parents=True, exist_ok=True)
    print(f"📚 Librarian watching: {ingest_dir.absolute()}")
    
    # Initialize Cabinet
    try:
        cabinet = CabinetOrchestrator()
        await cabinet.initialize()
        
        # Start ingestion watcher
        watcher = LibrarianWatcher(ingest_dir, cabinet.librarian)
        watcher_task = asyncio.create_task(watcher.watch())
        
        # Keep running until interrupted
        print("\n🚀 Cabinet operational. Press Ctrl+C to shutdown.")
        await asyncio.Future()  # Run forever
        
    except KeyboardInterrupt:
        print("\n🛑 Cabinet shutdown initiated...")
    except Exception as e:
        print(f"❌ Cabinet startup failed: {e}")
        return 1
    
    return 0


def main():
    """Main CLI entry point."""
    try:
        return asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n👋 Cabinet shutdown complete.")
        return 0
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
