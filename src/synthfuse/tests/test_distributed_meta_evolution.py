import pytest
import jax
import jax.numpy as jnp
from synthfuse.alchemj.ast import parse_spell_ast, ast_to_spell
from synthfuse.alchemj.evolution import EvolutionEngine, constrained_spell
from synthfuse.alchemj.distributed import MultiprocessingEvaluator
from synthfuse.alchemj.meta_grammar import ProbabilisticGrammar

def _test_fitness(ast, gen):
    # Must be at module level to be picklable for multiprocessing test
    from synthfuse.alchemj.compiler import compile_spell
    from synthfuse.alchemj.ast import ast_to_spell
    import jax
    import jax.numpy as jnp
    spell = ast_to_spell(ast)
    try:
        fn = compile_spell(spell)
        out = fn(jax.random.PRNGKey(gen), jnp.array([1.0, 2.0, 3.0]), {})
        mse = jnp.mean((out - jnp.array([1.05, 2.1, 3.15]))**2)
        return -float(mse)
    except Exception:
        return -1e9

def test_distributed_evolution():
    engine = EvolutionEngine(_test_fitness, evaluator=MultiprocessingEvaluator(max_workers=2))

    # Run a tiny evolution to verify it doesn't crash and uses workers
    best_ast = engine.evolve(pop_size=4, generations=2, max_depth=1)
    assert best_ast is not None

def test_meta_grammar_adaptation():
    grammar = ProbabilisticGrammar()

    # Simulate finding some "good" spells
    good_spells = [
        parse_spell_ast("𝕀 ⊗ 𝕃"),
        parse_spell_ast("ℝ ⊗ 𝕃")
    ]

    # Initial weights check (should be default initial_weight)
    w_initial = grammar.weights.get(("𝕀", "⊗", "𝕃"), 1.0)

    grammar.update(good_spells, learning_rate=1.0)

    # Weights should increase
    w_new = grammar.weights.get(("𝕀", "⊗", "𝕃"))
    assert w_new > w_initial

    # Sampling should favor these
    options = ["⊗", "⊕", "∘"]
    # 𝕀 ⊗ 𝕃 is valid, others might not be or have lower weight
    # We just check if it returns a valid one
    c = grammar.sample_comb(parse_spell_ast("𝕀"), parse_spell_ast("𝕃"), options)
    assert c in options
