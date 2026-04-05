"""
Spell → AST → JAX callable
"""
from lark import Lark, Transformer, v_args
from pathlib import Path
from typing import Callable, Any
import jax
import jax.numpy as jnp
from functools import lru_cache
from .registry import get

PyTree = Any
StepFn = Callable[[jax.Array, PyTree, dict], PyTree]

# ---------- load grammar once -----------------------------------------------
_GRAMMAR = Lark.open(Path(__file__).with_name("grammar.lark"), parser="lalr")


# ---------- AST → JAX transformer -------------------------------------------
@v_args(inline=True)
class _AST2Jax(Transformer):
    def __init__(self):
        self._lambda_counter = 0

    # terminals
    def prim(self, symbol, params):
        op = get(str(symbol))
        # bind params early (static)
        return lambda key, x, p: op(key, x, params)

    def seq(self, left, _, right):
        return lambda key, x, p: right(key, left(key, x, p), p)

    def par(self, left, _, right):
        # tree-additive parallel fusion
        return lambda key, x, p: jax.tree.map(
            jnp.add, left(key, x, p), right(key, x, p)
        )

    def guard(self, left, _, right):
        # left is pred_tree (primitive returning bool PyTree or lambda)
        def fn(key, x, p):
            mask = left(key, x, p)  # bool or float > 0 → True
            return jax.tree.map(
                lambda o, m: jnp.where(m, o, x), right(key, x, p), mask
            )
        return fn

    def lambda_expr(self, params, body):
        # compile-time lambda: params are names, body is stepfn
        # we simply ignore them for now (full λ-calculus TBD)
        return body

    def paren(self, child):
        return child

    # ------- containers -----------------------------------------------
    def dict(self, *keyvals):
        return dict(keyvals)

    def keyval(self, key, value):
        # key arrives as a token (string literal) – strip quotes
        return str(key).strip('"'), value

    def list(self, *items):
        return list(items)

    def params_list(self, arg_list):
        return arg_list or {}

    def no_params(self):
        return {}

    def arg_list(self, *args):
        return dict(args)

    def arg(self, name, value):
        return str(name), value

    # ------- terminals -------------------------------------------------
    def number(self, tok):
        # Lark gives str – cast to Python number
        try:
            return int(tok)
        except ValueError:
            return float(tok)

    def string(self, tok):
        return str(tok).strip('"')

    def boolean(self, tok):
        return str(tok).lower() == "true"

    def name(self, tok):
        return str(tok)


# ---------- public API -------------------------------------------------------
@lru_cache(maxsize=1024)
def compile_spell(source: str) -> StepFn:
    """source string → JIT-ready StepFn"""
    tree = _GRAMMAR.parse(source)
    stepfn = _AST2Jax().transform(tree)
    return jax.jit(stepfn)
