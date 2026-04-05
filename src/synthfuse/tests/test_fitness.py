import pytest
import jax.numpy as jnp
from synthfuse.alchemj.ast import parse_spell_ast, tree_size
from synthfuse.alchemj.fitness import MultiComponentFitness, ProblemResult, FitnessCache

def test_multi_component_fitness():
    def mock_problem(fn):
        # A mock evaluator that gives a better score for a specific "identity" function
        return ProblemResult(score=10.0, time=0.01)

    cache = FitnessCache()
    fitness = MultiComponentFitness(mock_problem, cache=cache)

    ast = parse_spell_ast("𝕀")
    score = fitness(ast, 0) # gen argument added

    # score = 1.0*10.0 - 0.2*0.01 - 0.3*0.0 - 0.1*1.0 = 10.0 - 0.002 - 0.1 = 9.898
    assert abs(score - 9.898) < 1e-5

    # Check cache
    assert cache.get("𝕀") == score

def test_tree_size():
    ast1 = parse_spell_ast("𝕀")
    assert tree_size(ast1) == 1

    ast2 = parse_spell_ast("𝕀 ⊗ 𝕃")
    assert tree_size(ast2) == 3 # 𝕀, 𝕃, ⊗

    ast3 = parse_spell_ast("(𝕀 ⊗ 𝕃) ⊕ 𝕊")
    assert tree_size(ast3) == 5 # 𝕀, 𝕃, ⊗, 𝕊, ⊕
