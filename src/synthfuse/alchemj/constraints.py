from typing import Dict, Set, Tuple

# Semantic Types for ALCHEM-J Sigils
TYPES = {
    "𝕀": "Update",
    "ℝ": "Update",
    "𝕃": "Noise",
    "𝕊": "Projection",
    "ℂ": "Transform",
    "ℤ": "Transform",
    "𝜑": "Condition",
    "Δ$": "Update",
    "𝕄": "Update",
    "𝕀𝕞𝕞": "Condition",
    "§": "Projection",
    "ℕ": "Transform",
    "𝕊𝕡": "Projection",
    "𝔹": "Projection",
    "Ω": "Transform",
    "Σ": "Transform",
    "κ": "Transform"
}

# ⊗ (Sequential) validity rules: (input_type, output_type)
VALID_SEQ: Set[Tuple[str, str]] = {
    ("Update", "Noise"),
    ("Noise", "Update"),
    ("Update", "Projection"),
    ("Projection", "Update"),
    ("Transform", "Update"),
    ("Update", "Transform"),
    ("Projection", "Noise"),
    ("Noise", "Projection"),
}

# ⊕ (Parallel) validity rules: (left_type, right_type)
VALID_PAR: Set[Tuple[str, str]] = {
    ("Update", "Update"),
    ("Transform", "Transform"),
    ("Noise", "Noise"),
    ("Projection", "Projection"),
}

# ∘ (Conditional) validity rules: (predicate_type, branch_type)
VALID_COND: Set[Tuple[str, str]] = {
    ("Condition", "Update"),
    ("Condition", "Transform"),
    ("Condition", "Noise"),
    ("Condition", "Projection"),
}

# Operator Frequency Constraints
MAX_COUNTS = {
    "𝕃": 2,
    "𝕊": 1,
    "𝕊𝕡": 1,
    "𝔹": 1,
}

# Forbidden Patterns (symbol → following symbols)
FORBIDDEN_FOLLOW = {
    "𝕃": ["𝕃"], # No Noise → Noise
    "𝕊": ["𝕊"], # No Projection → Projection loops
}

def get_node_type(node) -> str:
    """Infer semantic type of an AST node (imported here to avoid circularity)."""
    from .ast import Primitive, Combinator
    if isinstance(node, Primitive):
        return TYPES.get(node.symbol, "Transform")
    elif isinstance(node, Combinator):
        if node.op == "∘":
            return get_node_type(node.right)
        return get_node_type(node.left)
    return "Transform"

def is_valid_ast(node) -> bool:
    """Recursively validate an AST against semantic rules."""
    from .ast import Primitive, Combinator

    if isinstance(node, Primitive):
        return node.symbol in TYPES

    if isinstance(node, Combinator):
        if not (is_valid_ast(node.left) and is_valid_ast(node.right)):
            return False

        t1 = get_node_type(node.left)
        t2 = get_node_type(node.right)

        if node.op == "⊗":
            if (t1, t2) not in VALID_SEQ:
                return False
            # Check forbidden follow patterns
            if isinstance(node.left, Primitive) and isinstance(node.right, Primitive):
                 if node.right.symbol in FORBIDDEN_FOLLOW.get(node.left.symbol, []):
                     return False
        elif node.op == "⊕":
            if (t1, t2) not in VALID_PAR:
                return False
        elif node.op == "∘":
            if t1 != "Condition":
                return False

        return True

    return False
