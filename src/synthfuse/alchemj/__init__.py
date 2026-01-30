# src/synthfuse/alchemj/__init__.py

from .compiler import compile_to_hlo
from .registry import GlobalRegistry

# THE BRIDGE: Map 'parse_spell' to the new 'parse_sigil' 
# This fixes the ImportError while maintaining the new naming convention.

def parse_sigil(sigil_string):
    """
    The new standard for parsing topological constraints.
    (Formerly parse_spell)
    """
    # Logic to break down (𝕀 ⊗ ℝ) ⊕ ℤ
    print(f"Parsing Sigil: {sigil_string}")
    return {"op": "weld", "constraints": sigil_string}

# Export for backward compatibility with the current Lab version
parse_spell = parse_sigil
