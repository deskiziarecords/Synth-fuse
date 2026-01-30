# src/synthfuse/__init__.py

__version__ = "0.2.0"
__author__ = "J. Roberto Jiménez"
__email__ = "your_email@example.com"
__license__ = "OpenGate Integrity License"

from .cabinet.cabinet_orchestrator import CabinetOrchestrator
from .sigils.compiler import SigilCompiler
from .ingest.manager import IngestionManager

from .alchemj import (
    parse_spell,
    compile_spell,
    execute_spell,
)

__all__ = [
    "CabinetOrchestrator",
    "SigilCompiler", 
    "IngestionManager",
    "parse_spell",
    "compile_spell",
    "execute_spell",
]

def start_engine():
    """Starts the Cabinet, Librarian, and Physician in sync."""
    print("Synth-Fuse v0.2.0: Cabinet of Alchemists is ONLINE.")
    # Initialize the primary orchestrator
    cabinet = CabinetOrchestrator()
    # In a real environment, you'd start the background loop here
    # return cabinet.start_autonomous_loop() 
    return cabinet
