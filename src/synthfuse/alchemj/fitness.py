import time
import jax.numpy as jnp
from dataclasses import dataclass
from typing import List, Callable, Any, Optional, Dict
from .ast import Expr, ast_to_spell, tree_size

@dataclass
class ProblemResult:
    score: float
    time: float
    diverged: bool = False

class FitnessCache:
    def __init__(self):
        self._cache: Dict[str, float] = {}

    def get(self, spell: str) -> Optional[float]:
        return self._cache.get(spell)

    def set(self, spell: str, value: float):
        self._cache[spell] = value

class MultiComponentFitness:
    """Calculates fitness as a weighted sum of Performance, Efficiency, Stability, and Complexity."""
    def __init__(
        self,
        problem_fn: Callable[[Callable], ProblemResult],
        weights: Optional[Dict[str, float]] = None,
        num_runs: int = 3,
        cache: Optional[FitnessCache] = None
    ):
        self.problem_fn = problem_fn
        self.weights = weights or {
            "performance": 1.0,
            "efficiency": -0.2,
            "stability": -0.3,
            "complexity": -0.1
        }
        self.num_runs = num_runs
        self.cache = cache

    def __call__(self, ast: Expr, gen: int = 0) -> float:
        spell = ast_to_spell(ast)
        if self.cache:
            cached = self.cache.get(spell)
            if cached is not None:
                return cached

        try:
            from .compiler import compile_spell
            fn = compile_spell(spell)

            results: List[ProblemResult] = []
            for _ in range(self.num_runs):
                res = self.problem_fn(fn)
                results.append(res)

            scores = jnp.array([r.score for r in results])
            times = jnp.array([r.time for r in results])
            diverged = any(r.diverged for r in results)

            if diverged:
                return -1e9

            perf = float(jnp.mean(scores))
            var = float(jnp.var(scores))
            efficiency = float(jnp.mean(times))
            complexity = float(tree_size(ast))

            total_fitness = (
                self.weights.get("performance", 1.0) * perf +
                self.weights.get("efficiency", -0.2) * efficiency +
                self.weights.get("stability", -0.3) * var +
                self.weights.get("complexity", -0.1) * complexity
            )

            if self.cache:
                self.cache.set(spell, total_fitness)

            return total_fitness

        except Exception:
            return -1e9
