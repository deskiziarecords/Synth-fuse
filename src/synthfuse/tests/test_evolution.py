import pytest
import jax
import jax.numpy as jnp
from synthfuse.alchemj.ast import ast_to_spell
from synthfuse.alchemj.compiler import compile_spell
from synthfuse.alchemj.evolution import EvolutionEngine

def test_evolution_discovery():
    # Toy problem: find a spell that performs a specific transformation.
    # Target: x * 1.05 (Recursive Debt with interest_rate=0.05)

    key = jax.random.PRNGKey(42)
    x_test = jnp.array([1.0, 2.0, 3.0])
    target = x_test * 1.05

    def fitness(ast, gen):
        spell = ast_to_spell(ast)
        try:
            fn = compile_spell(spell)
            out = fn(key, x_test, {})
            # fitness = -MSE
            mse = jnp.mean((out - target)**2)
            return -float(mse)
        except Exception:
            return -1e9

    engine = EvolutionEngine(fitness)
    # Smaller population for fast test
    best_ast = engine.evolve(pop_size=20, generations=10, max_depth=2)

    best_spell = ast_to_spell(best_ast)
    print(f"Best spell found: {best_spell}")

    # Check if we found something reasonable
    best_fitness = fitness(best_ast, 10)
    assert best_fitness > -1.0 # Should at least find something better than catastrophic failure
