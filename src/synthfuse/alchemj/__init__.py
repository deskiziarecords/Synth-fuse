# src/synthfuse/alchemj/__init__.py

from ..sigils.compiler import SigilCompiler

# Instantiate a hidden compiler to handle legacy calls
_legacy_compiler = SigilCompiler()

def parse_spell(spell_string):
    """Legacy wrapper for the new Sigil parser."""
    print(f"[Legacy] Redirecting 'Spell' to 'Sigil' Logic...")
    return _legacy_compiler.parse(spell_string)

def compile_spell(parsed_spell):
    """Legacy wrapper for Sigil-to-HLO compilation."""
    return _legacy_compiler.compile(parsed_spell)

def execute_spell(compiled_hlo, data):
    """Legacy wrapper for high-precision Welding."""
    # This eventually talks to the Cabinet's Engineer
    return "Execution Redirected to Cabinet"
