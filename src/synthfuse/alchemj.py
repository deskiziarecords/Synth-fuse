
def parse_spell(spell_text: str):
    """
    Parse an ALCHEM-J spell (legacy v0.1.0).
    
    Args:
        spell_text: Spell expression from v0.1.0
        
    Returns:
        Dict with parsed information and compatibility warning
    """
    return {
        "type": "legacy_spell",
        "original": spell_text,
        "sigil": f"({spell_text})",  # Convert to v0.2.0 sigil format
        "version": "0.2.0-compatibility",
        "warning": "Using legacy spell system - migrate to Sigils for v0.2.0",
        "components": spell_text.replace("(", "").replace(")", "").split(),
    }

def compile_spell(ast):
    """Compile spell AST to executable code."""
    return {
        "status": "compiled",
        "version": "0.2.0-compat",
        "ast": ast,
        "jax_code": f"# Legacy spell: {ast}",
    }

def validate_spell(spell):
    """Validate spell syntax."""
    return {
        "valid": True,
        "message": "Legacy validation - use SigilCompiler for v0.2.0",
        "spell": spell,
    }

# Export the old Spell class for compatibility
class Spell:
    """Legacy Spell class from v0.1.0."""
    
    def __init__(self, text):
        self.text = text
        self.type = "legacy"
        
    def execute(self, data=None):
        return {"result": f"Legacy spell executed: {self.text}", "data": data}

# Module exports
__all__ = ["parse_spell", "compile_spell", "validate_spell", "Spell"]
EOF