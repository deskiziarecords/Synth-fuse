import random
from typing import List, Dict, Any, Optional, Callable
from .ast import Expr, Primitive, Combinator, ast_to_spell
from .registry import _REGISTRY

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

def mutate_subtree(node: Expr, max_depth: int = 2) -> Expr:
    """Replace a subtree with a new random one."""
    if random.random() < 0.3:
        return random_spell(depth=max_depth)

    if isinstance(node, Combinator):
        if random.random() < 0.5:
            return Combinator(node.op, mutate_subtree(node.left, max_depth), node.right)
        else:
            return Combinator(node.op, node.left, mutate_subtree(node.right, max_depth))
    return node

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
    """Evolutionary Programming over ALCHEM-J ASTs."""
    def __init__(self, fitness_fn: Callable[[Expr], float]):
        self.fitness_fn = fitness_fn

    def evolve(self, pop_size: int = 50, generations: int = 20, max_depth: int = 3) -> Expr:
        """Main evolution loop."""
        # 1. Initial Population
        population = [random_spell(depth=max_depth) for _ in range(pop_size)]

        for gen in range(generations):
            # 2. Evaluation
            scored = [(self.fitness_fn(s), s) for s in population]
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
                    child = mutate_op(child)
                elif r < 0.6:
                    child = mutate_comb(child)
                else:
                    child = mutate_subtree(child, max_depth=max_depth)

                new_pop.append(child)

            population = new_pop

            # (Optional) Log progress
            # print(f"Gen {gen}: Best score = {scored[0][0]}")

        # Final best individual
        final_best = max(population, key=lambda s: self.fitness_fn(s))
        return final_best
