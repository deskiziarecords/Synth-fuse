import pytest
from synthfuse.alchemj.ast import parse_spell_ast
from synthfuse.alchemj.constraints import is_valid_ast, get_node_type
from synthfuse.alchemj.evolution import constrained_spell, EvolutionEngine

def test_ast_validation():
    # Valid: Update ⊗ Noise
    assert is_valid_ast(parse_spell_ast("𝕀 ⊗ 𝕃"))

    # Valid: Update ⊕ Update
    assert is_valid_ast(parse_spell_ast("𝕀 ⊕ ℝ"))

    # Valid: Condition ∘ Update
    assert is_valid_ast(parse_spell_ast("𝜑 ∘ 𝕀"))

    # Invalid: Noise ⊗ Noise (Forbidden follow)
    assert not is_valid_ast(parse_spell_ast("𝕃 ⊗ 𝕃"))

    # Invalid: Noise ⊕ Update (Invalid parallel)
    assert not is_valid_ast(parse_spell_ast("𝕃 ⊕ 𝕀"))

    # Invalid: Update ∘ Condition (Invalid conditional)
    assert not is_valid_ast(parse_spell_ast("𝕀 ∘ 𝜑"))

def test_constrained_generation():
    for _ in range(20):
        ast = constrained_spell(depth=3)
        assert is_valid_ast(ast), f"Generated invalid AST: {ast}"

def test_constrained_evolution():
    def dummy_fitness(ast, gen):
        return 0.0

    engine = EvolutionEngine(dummy_fitness)
    # Evolve a small population with constraints
    best_ast = engine.evolve(pop_size=10, generations=5, constrained=True)

    assert is_valid_ast(best_ast)
