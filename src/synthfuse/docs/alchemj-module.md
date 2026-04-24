# ALCHEM-J Code Documentation

This documentation describes the various components of the ALCHEM-J framework, which is a symbolic fusion system designed for generating neural dynamics based on a combination of formal grammar, evolutionary algorithms, and JAX for numerical operations.

## File Structure Overview

- **ast.py**: Implements the Abstract Syntax Tree (AST) representation and parsing mechanisms for symbolic expressions.
- **distributed.py**: Contains evaluator classes for distributed and local evaluations of fitness functions.
- **compiler.py**: Transforms the AST into JAX-compatible callable functions.
- **combinators.py**: Provides functional combinators for defining operations within the system.
- **constraints.py**: Enforces semantic and structural constraints on the AST.
- **evolution.py**: Implements evolutionary strategies for optimizing spells and generating expression trees.
- **fitness.py**: Defines fitness evaluation strategies for spells, including caching mechanisms.
- **grammar.lark**: Defines the syntax grammar for parsing scripts written in ALCHEM-J.
- **meta_grammar.py**: Implements a probabilistic grammar that learns patterns from successful ASTs.
- **registry.py**: Manages the registration and lookup of operators in the ALCHEM-J ecosystem.
- **optimization.py**: Provides functions for running optimization problems using ALCHEM-J spells.
- **signal.py**: Evaluates ALCHEM-J spells in the context of signal processing tasks.
- **numeric.py**: Contains numerical primitives designed for use within the ALCHEM-J framework.
- **orion.py**: Implements the Weierstrass-transform neural gravity solver.
- **rl.py**: Contains reinforcement learning primitives for policy updates and value function approximations.
- **sat.py**: Implements SAT solver functionalities within ALCHEM-J.
- **sql.py**: Parses SQL queries into vector operations for manipulation.
- **swarm.py**: Implements intelligent swarm optimization techniques.
- **util.py**: A collection of utility primitives for various mathematical operations.
- **vector.py**: Contains vector manipulation plugins specific to low-dimensional data processing.

## Detailed Documentation of Each Component

### ast.py
**Overview**: This module serves as the foundational parser for the ALCHEM-J language, defining the grammar for fusion expressions and providing a structure for the AST.

#### Key Classes:
- **Param**: Represents a named argument with a value, where the value can be of any type (int, float, string).
- **Primitive**: Represents a single operation with a symbol and a dictionary of parameters.
- **Combinator**: Represents operations that combine expressions (e.g., addition, multiplication).
- **Lexer**: Tokenizes input strings into meaningful symbols, operations, and literals.
- **Parser**: Converts tokens into an abstract syntax tree (AST).

#### Key Functions:
- `parse_spell_ast(spell: str) -> Expr`: Converts a spell string into an AST.
- `ast_to_spell(ast_node: Expr) -> str`: Converts an AST back into a spell string.

### distributed.py
**Overview**: Manages the evaluation of a population of AST expressions in multiple processes for fitness calculations.

#### Key Classes:
- **MultiprocessingEvaluator**: Evaluates expressions across multiple processes for efficiency.
- **LocalEvaluator**: Serially evaluates expressions for direct computation.

#### Key Functions:
- `serialize_ast(node: Expr) -> Dict[str, Any]`: Converts an AST node to a JSON-serializable dictionary.
- `deserialize_ast(data: Dict[str, Any]) -> Expr`: Converts a dictionary back to an AST node.

### compiler.py
**Overview**: Compiles ALCHEM-J sentences into executable JAX functions.

#### Key Classes:
- None explicit but uses Lark grammar for parsing.

#### Key Functions:
- `compile_spell(source: str) -> StepFn`: Converts a source string into a JIT-ready JAX callable function.

### combinators.py
**Overview**: Implements higher-order functions for composing operations within the system.

#### Key Functions:
- `fuse_seq(steps: List[StepFn]) -> StepFn`: Combines multiple steps sequentially.
- `fuse_par(step_a: StepFn, step_b: StepFn) -> StepFn`: Combines operations in parallel.

### constraints.py
**Overview**: Enforces semantic validity rules for different types of operators and combinations.

#### Key Functions:
- `get_node_type(node) -> str`: Infers the semantic type of an AST node.
- `is_valid_ast(node) -> bool`: Validates an entire AST against defined rules.

### evolution.py
**Overview**: Implements evolutionary algorithms for optimizing spell populations through mutation, crossover, and selection processes.

#### Key Functions:
- `random_spell(depth: int = 3) -> Expr`: Generates a random valid spell AST based on a specified depth.
- `evolve(...)`: Main evolution loop function that iteratively evolves a population of spells based on fitness evaluations.

### fitness.py
**Overview**: Contains methods for calculating the fitness of spells by measuring performance across several metrics.

#### Key Classes:
- **ProblemResult**: Encapsulates results of fitness evaluations including scores and execution time.
- **FitnessCache**: Caches fitness evaluations to avoid redundant calculations.

### grammar.lark
**Overview**: Defines the formal grammar for parsing the ALCHEM-J language, using Lark syntax.

### meta_grammar.py
**Overview**: Implements a dynamic learning mechanism for transition probabilities in grammar based on previous successful spells.

### registry.py
**Overview**: Manages and stores mappings between symbols and their corresponding function implementations.

### optimization.py
**Overview**: Provides utilities to evaluate spells as optimization processes.

### signal.py
**Overview**: Evaluates ALCHEM-J spells specifically for tasks in the signal processing domain, such as signal classification.

### numeric.py
**Overview**: Contains pure-JAX numerical primitives that operate within the framework.

### orion.py
**Overview**: Implements a gravity-based solver using Weierstrass transformations.

### rl.py
**Overview**: Implements reinforcement learning algorithms, including PPO and DQN steps.

### sat.py
**Overview**: Provides functionalities for solving SAT problems using expressions defined in ALCHEM-J.

### sql.py
**Overview**: Translates SQL queries into operations that can be processed via vectors.

### swarm.py
**Overview**: Implements swarm intelligence optimization techniques, including inertia and social behaviors.

### util.py
**Overview**: Provides various utility primitives crucial for mathematical computations.

### vector.py
**Overview**: Contains numerous vector operations, including dimensionality reduction and other transformation techniques.

## Additional Notes
This framework is fundamentally structured upon functional programming principles, leveraging JAX's efficient computation capabilities. Each component builds upon the previously defined structures to allow for extensibility and modular design, paving the way for complex dynamic systems to be constructed through symbolic representation and evolutionary optimization techniques.

---
