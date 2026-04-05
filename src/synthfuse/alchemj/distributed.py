import concurrent.futures
from typing import List, Callable, Any, Dict
from .ast import Expr, Primitive, Combinator

def serialize_ast(node: Expr) -> Dict[str, Any]:
    """Convert AST to a JSON-serializable dictionary."""
    if isinstance(node, Primitive):
        return {"type": "prim", "symbol": node.symbol, "params": node.params}
    elif isinstance(node, Combinator):
        return {
            "type": "comb",
            "op": node.op,
            "left": serialize_ast(node.left),
            "right": serialize_ast(node.right)
        }
    raise ValueError(f"Unknown AST node type: {type(node)}")

def deserialize_ast(data: Dict[str, Any]) -> Expr:
    """Convert a dictionary back to an AST node."""
    if data["type"] == "prim":
        return Primitive(symbol=data["symbol"], params=data["params"])
    elif data["type"] == "comb":
        return Combinator(
            op=data["op"],
            left=deserialize_ast(data["left"]),
            right=deserialize_ast(data["right"])
        )
    raise ValueError(f"Unknown serialized data type: {data['type']}")

def _worker_wrapper(args):
    fitness_fn, ast, gen = args
    return fitness_fn(ast, gen)

class MultiprocessingEvaluator:
    """Evaluates a population of spells using Python multiprocessing."""
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers

    def evaluate(self, population: List[Expr], fitness_fn: Callable[[Expr, int], float], gen: int) -> List[float]:
        # Note: fitness_fn must be picklable.
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            args_list = [(fitness_fn, ast, gen) for ast in population]
            results = list(executor.map(_worker_wrapper, args_list))
        return results

class LocalEvaluator:
    """Standard sequential evaluator."""
    def evaluate(self, population: List[Expr], fitness_fn: Callable[[Expr, int], float], gen: int) -> List[float]:
        return [fitness_fn(s, gen) for s in population]
