import random
from typing import List, Dict, Any, Optional, Callable
from .ast import Expr, Primitive, Combinator, ast_to_spell, tree_size
from .registry import _REGISTRY
from .constraints import TYPES, VALID_SEQ, VALID_PAR, VALID_COND, MAX_COUNTS, FORBIDDEN_FOLLOW, is_valid_ast

OPS = list(_REGISTRY.keys())
COMBS = ["⊗", "⊕", "∘"]

def random_spell(depth: int = 3) -> Expr:
    """Generate a random valid spell AST."""
    if depth == 0:
        return Primitive(symbol=random.choice(OPS), params={})

    if random.random() < 0.4:
        return Primitive(symbol=random.choice(OPS), params={})

    return Combinator(
        op=random.choice(COMBS),
        left=random_spell(depth - 1),
        right=random_spell(depth - 1)
    )

def mutate_op(node: Expr) -> Expr:
    """Mutate a primitive operator."""
    if isinstance(node, Primitive):
        return Primitive(symbol=random.choice(OPS), params=node.params)
    elif isinstance(node, Combinator):
        if random.random() < 0.5:
            return Combinator(node.op, mutate_op(node.left), node.right)
        else:
            return Combinator(node.op, node.left, mutate_op(node.right))
    return node

def mutate_comb(node: Expr) -> Expr:
    """Mutate a combinator operator."""
    if isinstance(node, Combinator):
        if random.random() < 0.2:
            return Combinator(random.choice(COMBS), node.left, node.right)
        if random.random() < 0.5:
            return Combinator(node.op, mutate_comb(node.left), node.right)
        else:
            return Combinator(node.op, node.left, mutate_comb(node.right))
    return node

def mutate_subtree(node: Expr, max_depth: int = 2, constrained: bool = False) -> Expr:
    """Replace a subtree with a new random one."""
    if random.random() < 0.3:
        if constrained:
            return constrained_spell(depth=max_depth)
        return random_spell(depth=max_depth)

    if isinstance(node, Combinator):
        if random.random() < 0.5:
            return Combinator(node.op, mutate_subtree(node.left, max_depth, constrained), node.right)
        else:
            return Combinator(node.op, node.left, mutate_subtree(node.right, max_depth, constrained))
    return node

def safe_mutate(node: Expr, mutator: Callable[[Expr], Expr], constrained: bool = False) -> Expr:
    """Apply a mutation and ensure the result is valid if constrained."""
    if not constrained:
        return mutator(node)

    for _ in range(5):
        new_node = mutator(node)
        if is_valid_ast(new_node):
            return new_node
    return node

def get_node_type(node: Expr) -> str:
    """Infer semantic type of an AST node."""
    if isinstance(node, Primitive):
        return TYPES.get(node.symbol, "Transform")
    elif isinstance(node, Combinator):
        if node.op == "∘":
            return get_node_type(node.right)
        return get_node_type(node.left) # simplistic heuristic
    return "Transform"

def is_valid_combination(left: Expr, right: Expr, op: str) -> bool:
    """Check if a combination of two AST nodes is valid according to constraints."""
    t1 = get_node_type(left)
    t2 = get_node_type(right)

    if op == "⊗":
        return (t1, t2) in VALID_SEQ
    elif op == "⊕":
        return (t1, t2) in VALID_PAR
    elif op == "∘":
        return t1 == "Condition"
    return True

def constrained_spell(depth: int = 3) -> Expr:
    """Generate a random valid spell AST according to semantic constraints."""
    if depth == 0:
        # Filter symbols that are not Conditions for leaf if depth 0 (avoid single Condition)
        valid_leaves = [s for s in OPS if TYPES.get(s) != "Condition"]
        return Primitive(symbol=random.choice(valid_leaves or OPS), params={})

    if random.random() < 0.4:
        valid_leaves = [s for s in OPS if TYPES.get(s) != "Condition"]
        return Primitive(symbol=random.choice(valid_leaves or OPS), params={})

    op = random.choice(COMBS)
    left = constrained_spell(depth - 1)
    right = constrained_spell(depth - 1)

    # Selection pressure for valid combinations
    for _ in range(10):
        if is_valid_combination(left, right, op):
            break
        op = random.choice(COMBS)
        left = constrained_spell(depth - 1)
        right = constrained_spell(depth - 1)

    if not is_valid_combination(left, right, op):
        # Fallback to left if still invalid
        return left

    return Combinator(op=op, left=left, right=right)

def crossover(a: Expr, b: Expr) -> Expr:
    """Swap subtrees between two individuals."""
    if random.random() < 0.4:
        return b

    if isinstance(a, Combinator) and isinstance(b, Combinator):
        if random.random() < 0.5:
            return Combinator(a.op, crossover(a.left, b.left), a.right)
        else:
            return Combinator(a.op, a.left, crossover(a.right, b.right))
    elif isinstance(a, Combinator):
        if random.random() < 0.5:
            return Combinator(a.op, crossover(a.left, b), a.right)
        else:
            return Combinator(a.op, a.left, crossover(a.right, b))

    return a

class EvolutionEngine:
    """Evolutionary Programming over ALCHEM-J ASTs with curriculum learning."""
    def __init__(
        self,
        fitness_fn: Callable[[Expr, int], float],
        curriculum: Optional[Callable[[int], Dict[str, Any]]] = None
    ):
        self.fitness_fn = fitness_fn
        self.curriculum = curriculum

    def evolve(self, pop_size: int = 50, generations: int = 20, max_depth: int = 3, constrained: bool = False) -> Expr:
        """Main evolution loop."""
        # 1. Initial Population
        if constrained:
            population = [constrained_spell(depth=max_depth) for _ in range(pop_size)]
        else:
            population = [random_spell(depth=max_depth) for _ in range(pop_size)]

        for gen in range(generations):
            # Curriculum step
            params = self.curriculum(gen) if self.curriculum else {}

            # 2. Evaluation (fitness_fn can handle two-stage or curriculum)
            scored = [(self.fitness_fn(s, gen), s) for s in population]
            # Sort by fitness (descending)
            scored.sort(reverse=True, key=lambda x: x[0])

            # 3. Selection (top 50% survive)
            survivors = [s for _, s in scored[:pop_size // 2]]

            # 4. Reproduction
            new_pop = survivors.copy()
            while len(new_pop) < pop_size:
                a, b = random.sample(survivors, 2)
                # Crossover
                child = crossover(a, b)

                # Mutation
                r = random.random()
                if r < 0.3:
                    child = safe_mutate(child, mutate_op, constrained)
                elif r < 0.6:
                    child = safe_mutate(child, mutate_comb, constrained)
                else:
                    child = safe_mutate(child, lambda n: mutate_subtree(n, max_depth, constrained), constrained)

                new_pop.append(child)

            population = new_pop

            # (Optional) Log progress
            # print(f"Gen {gen}: Best score = {scored[0][0]}")

        # Final best individual
        final_best = max(population, key=lambda s: self.fitness_fn(s, generations))
        return final_best
