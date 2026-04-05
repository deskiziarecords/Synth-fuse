import random
from typing import Dict, List, Tuple, Any, Optional
from .ast import Expr, Primitive, Combinator
from .registry import _REGISTRY
from .constraints import TYPES, VALID_SEQ, VALID_PAR, VALID_COND

class ProbabilisticGrammar:
    """A grammar that learns transition probabilities from successful spells."""
    def __init__(self, initial_weight: float = 1.0):
        self.weights: Dict[Tuple[str, str, str], float] = {}
        self.initial_weight = initial_weight
        self._initialize_weights()

    def _initialize_weights(self):
        """Seed weights based on initial validity rules."""
        options = ["⊗", "⊕", "∘"]
        # In a real implementation, we would iterate over all pairs of types/sigils.
        # For brevity, we start with uniform weights for known valid rules.
        pass

    def extract_patterns(self, node: Expr) -> List[Tuple[str, str, str]]:
        """Recursively extract (left_symbol, op, right_symbol) patterns."""
        patterns = []
        if isinstance(node, Combinator):
            # Use symbols if primitives, or "Expr" if sub-combinators
            left_id = node.left.symbol if isinstance(node.left, Primitive) else "Expr"
            right_id = node.right.symbol if isinstance(node.right, Primitive) else "Expr"
            patterns.append((left_id, node.op, right_id))
            patterns.extend(self.extract_patterns(node.left))
            patterns.extend(self.extract_patterns(node.right))
        return patterns

    def update(self, successful_spells: List[Expr], learning_rate: float = 0.1):
        """Update probabilities based on the provided list of spells."""
        for spell in successful_spells:
            patterns = self.extract_patterns(spell)
            for p in patterns:
                self.weights[p] = self.weights.get(p, self.initial_weight) + learning_rate

    def sample_comb(self, left: Expr, right: Expr, options: List[str]) -> str:
        """Sample a combinator based on learned probabilities and constraints."""
        from .evolution import get_node_type # Avoid circular import

        t1 = get_node_type(left)
        t2 = get_node_type(right)

        left_id = left.symbol if isinstance(left, Primitive) else "Expr"
        right_id = right.symbol if isinstance(right, Primitive) else "Expr"

        probs = []
        valid_options = []

        for c in options:
            # Check validity first
            is_valid = False
            if c == "⊗":
                is_valid = (t1, t2) in VALID_SEQ
            elif c == "⊕":
                is_valid = (t1, t2) in VALID_PAR
            elif c == "∘":
                is_valid = t1 == "Condition"

            if is_valid:
                valid_options.append(c)
                probs.append(self.weights.get((left_id, c, right_id), self.initial_weight))

        if not valid_options:
            return random.choice(options)

        total = sum(probs)
        r = random.uniform(0, total)
        upto = 0
        for i, p in enumerate(probs):
            if upto + p >= r:
                return valid_options[i]
            upto += p
        return valid_options[-1]
