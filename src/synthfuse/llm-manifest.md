## Synth-Fuse LLM Manifest v0.2.0
Unified Field Edition - Complete LLM Reference
File Location: src/synthfuse/llm-manifest.md
Version: 0.2.0-unified-field
Last Updated: 2026-02-10
Purpose: Self-contained guide for LLMs to generate valid Synth-Fuse recipes, spells, and welds
1. Executive Summary
Synth-Fuse is a JAX-native neuro-symbolic fusion runtime that translates high-level mathematical intent into hardware-optimized computational kernels. 
It operates through a Cabinet of 7 specialized roles that process symbolic expressions (Sigils) into compiled JAX code (Welds).

Core Philosophy: Intelligence as efficient circulation—like red blood cells delivering oxygen, Synth-Fuse delivers capability without bloat, friction, or thermal waste.
Key Constraint: All operations must achieve unanimous consensus across the Cabinet (7/7 roles agree) to be certified.
---
## 2. Architecture Overview:

┌─────────────────────────────────────────────────────────────┐
│                      USER LAYER                              │
│  Recipe → Sigil/Spell → circulate() → process_sigil()       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    CABINET ORCHESTRATOR                      │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │
│  │Architect│ │ Engineer│ │Librarian│ │Physician│           │
│  │  🏛️    │ │   🔧   │ │   📚   │ │   🩺   │           │
│  │Strategy │ │Compile  │ │ Ingest  │ │ Health  │           │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘           │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                       │
│  │  Shield │ │   Body  │ │   Jury  │                       │
│  │   🛡️   │ │   🌡️   │ │   ⚖️   │                       │
│  │ Safety  │ │ Thermal │ │Consensus│                       │
│  └─────────┘ └─────────┘ └─────────┘                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT LAYER                              │
│  Weld (JAX Kernel) + Vitals Report + Lyapunov Certificate   │
└─────────────────────────────────────────────────────────────┘

## 3. Fundamental Concepts
3.1 Sigils (Symbolic Intent)
Sigils are topological constraint expressions using mathematical notation:

| Symbol | Name             | Meaning                       | Use Case                                     |
| ------ | ---------------- | ----------------------------- | -------------------------------------------- |
| `⊗`    | Tensor Product   | Fuse into unified field       | Full integration (swarm + RL weights shared) |
| `⊕`    | Direct Sum       | Alternation/selection         | Switch between modes (explore vs exploit)    |
| `⊙`    | Hadamard Product | Element-wise combination      | Parallel streams (vision + policy)           |
| `I`    | Identity         | Baseline/no-op                | Placeholder or pure passthrough              |
| `Z`    | Zero-Point       | Constraint anchor             | Hard constraints, stability bounds           |
| `R`    | RL Policy        | Reinforcement learning        | PPO, SAC, or other policy gradients          |
| `S`    | Swarm            | Population-based optimization | PSO, Differential Evolution                  |
| `D`    | Diff-Evo         | Differential evolution        | Numeric optimization                         |
| `C`    | Curriculum       | Adaptive staging              | Progressive difficulty training              |
| `V`    | Vision           | Visual backbone               | ViT, DETR, Scenic wrappers                   |

### Valid Sigil Patterns:

    (I⊗Z) - Identity with zero-point constraint (baseline)
    (S⊕R) - Swarm alternates with RL (exploration/exploitation)
    (D⊗C) - Diff-Evo fused with curriculum (hard landscapes)
    (V⊙R) - Vision element-wise with policy (end-to-end perception)
    (S⊕R)⊗C - Swarm/RL fusion + curriculum (complex training)

### Invalid Patterns (will fail Cabinet consensus):

    ((A⊗B)⊗C)⊗D - Nesting > 3 levels (thermal runaway risk)
    (V⊗S) without projection - Type mismatch (vision + swarm incompatible)
    (R⊗R) - Redundant fusion (no value added)

## 3.2 Welds (Compiled Output)
A Weld is the final output: a JAX-native kernel, configuration, or rewrite. Characteristics:

    JIT-compiled via XLA
    TPU/GPU/CPU portable
    Cryptographically attested (Lyapunov certificate)
    Thermodynamically coherent (entropy < max_entropy)

## 3.3 Spells (Legacy)
Pre-v0.2.0 term for Sigils. Still supported via alchemj module for backward compatibility.
4. Recipe Patterns (Copy-Paste Templates)
Pattern A: One-Liner Circulation (Simplest)
Use when: Quick optimization, known data structure, single objective

``phyton
from synthfuse import circulate

# Basic numeric optimization
weld = circulate(
    manifold=my_dataframe,  # pandas DataFrame or JAX array
    objective="minimize_latency_under_compliance_constraint",
    substrate="cuda"        # "cuda" | "tpu" | "cpu"
)

# Execute
result = weld.apply(input_data)
``
**Available objectives:**

    minimize_latency_under_compliance_constraint
    maximize_throughput_under_stability
    optimize_entropy_under_thermal_bounds
    minimize_cost_under_service_level

Returns: Compiled JAX kernel with .apply() method
Pattern B: Full Cabinet Workflow (Maximum Control)
Use when: Complex multi-stage processing, health monitoring required, custom constraints

``phyton
import asyncio
from synthfuse import start_engine
from typing import Dict, Any

async def my_recipe(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Custom recipe with full Cabinet orchestration.
    Sigil: (S⊕R) - Swarm + RL alternation for supply chain optimization.
    """
    
    # 1. Initialize Cabinet (starts all 7 roles)
    cabinet = start_engine()
    success = await cabinet.initialize()
    
    if not success:
        raise RuntimeError("Cabinet failed to initialize")
    
    # 2. Define configuration
    config = {
        'max_entropy': 0.25,      # Stricter than default 0.3
        'max_thermal_load': 0.75,  # Lower than default 0.8
        'timeout': 45.0           # Extended timeout
    }
    
    # 3. Update cabinet config (optional)
    cabinet.max_entropy = config['max_entropy']
    cabinet.max_thermal_load = config['max_thermal_load']
    cabinet.timeout = config['timeout']
    
    # 4. Process through Cabinet
    result = await cabinet.process_sigil(
        sigil="(S⊕R)",  # Swarm alternates with RL
        data={
            'demand_matrix': data['demand'],
            'supplier_capacity': data['capacity'],
            'cost_structure': data['costs'],
            'constraints': {
                'max_total_cost': data['budget'],
                'min_service_level': 0.95,
                'risk_tolerance': 0.1
            }
        }
    )
    
    # 5. Validate consensus
    if not result['consensus_reached']:
        # Check which roles dissented
        verdict = result['verdict']
        raise RuntimeError(f"Cabinet consensus failed: {verdict}")
    
    # 6. Check vitals
    if result['entropy'] > config['max_entropy']:
        print(f"Warning: High entropy {result['entropy']}")
    
    if result['thermal_load'] > config['max_thermal_load']:
        print(f"Warning: Thermal throttling {result['thermal_load']}")
    
    # 7. Extract outputs
    return {
        'weld': result['compilation']['jax_code'],      # The compiled kernel
        'blueprint': result['blueprint'],               # Strategic plan from Architect
        'vitals': {
            'entropy': result['entropy'],               # Information disorder (lower better)
            'thermal_load': result['thermal_load'],     # Compute stress (lower better)
            'duration_ms': result['duration_seconds'] * 1000,
            'data_size': result['data_size']
        },
        'certified': True,  # Only True if consensus reached and bounds satisfied
        'timestamp': result['timestamp']
    }

# Usage
# result = asyncio.run(my_recipe(my_data))
``
---

### Pattern C: Custom Recipe Class (Reusable Component)
Use when: Creating library of reusable recipes, 3rd-party extensions

``phyton
# File: src/synthfuse/recipes/portfolio_optimizer.py
import asyncio
from typing import Dict, Any, Optional
from synthfuse.cabinet.cabinet_orchestrator import CabinetOrchestrator

class PortfolioOptimizationRecipe:
    """
    Lego-style recipe for financial portfolio optimization.
    Fuses Differential Evolution with Curriculum learning.
    
    Sigil: (D⊗C)
    """
    
    DEFAULT_CONFIG = {
        'max_entropy': 0.2,        # Finance requires low entropy
        'max_thermal_load': 0.7,
        'timeout': 60.0,
        'risk_aversion': 0.5       # Custom parameter
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.cabinet = CabinetOrchestrator(config=self.config)
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the Cabinet."""
        self._initialized = await self.cabinet.initialize()
        return self._initialized
    
    async def solve(self, 
                    returns: Any, 
                    cov_matrix: Any,
                    constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize portfolio weights.
        
        Args:
            returns: Expected returns vector (JAX array)
            cov_matrix: Covariance matrix (JAX array)
            constraints: {
                'target_return': float,
                'max_volatility': float,
                'max_position_size': float,
                'sector_limits': Dict[str, float]
            }
        """
        if not self._initialized:
            raise RuntimeError("Recipe not initialized. Call initialize() first.")
        
        # Validate inputs
        if returns.shape[0] != cov_matrix.shape[0]:
            raise ValueError("Returns and covariance dimensions mismatch")
        
        # Process through Cabinet
        result = await self.cabinet.process_sigil(
            sigil="(D⊗C)",  # Diff-Evo + Curriculum for non-convex optimization
            data={
                'returns': returns,
                'covariance': cov_matrix,
                'constraints': constraints,
                'risk_aversion': self.config['risk_aversion']
            }
        )
        
        # Post-processing
        if result['consensus_reached']:
            return {
                'weights': result['compilation']['jax_code'],  # Optimization kernel
                'expected_return': None,  # To be computed by executing weld
                'volatility': None,
                'sharpe_ratio': None,
                'vitals': {
                    'entropy': result['entropy'],
                    'thermal_load': result['thermal_load'],
                    'duration_ms': result['duration_seconds'] * 1000
                },
                'certified': (
                    result['entropy'] <= self.config['max_entropy'] and
                    result['thermal_load'] <= self.config['max_thermal_load']
                )
            }
        else:
            raise RuntimeError(f"Optimization failed: {result['verdict']}")
    
    async def emergency_stop(self):
        """Emergency shutdown if thermal runaway detected."""
        return await self.cabinet.emergency_shutdown()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current Cabinet health."""
        return self.cabinet.get_status()

# Usage Example:
# recipe = PortfolioOptimizationRecipe({'risk_aversion': 0.7})
# await recipe.initialize()
# result = await recipe.solve(returns, cov_matrix, constraints)
``
## Pattern D: Legacy Spell Interface (Backward Compatibility)
Use when: Maintaining old code, simple parsing needs
``phyton
from synthfuse.alchemj import parse_spell, compile_spell, execute_spell

# 1. Parse high-level intent (legacy syntax)
parsed = parse_spell("""
    OPTIMIZE portfolio 
    USING differential_evolution 
    WITH curriculum_adaptation
    CONSTRAINTS [entropy < 0.2, thermal < 0.7]
    YIELD jax_kernel
""")

# 2. Compile to HLO (High Level Operations)
compiled = compile_spell(parsed)
# Returns: {'sigil': '(D⊗C)', 'ast': '...', 'status': 'compiled'}

# 3. Execute (delegates to Cabinet's Engineer role)
result = execute_spell(compiled, data=my_portfolio_data)
# Note: In current version, this redirects to Cabinet

``
## 5. Batch Processing
Process multiple sigils efficiently:

``phyton
async def batch_recipe(datasets: List[Dict]) -> List[Dict]:
    cabinet = start_engine()
    await cabinet.initialize()
    
    # Define sigils for different optimization strategies
    sigils = [
        "(D⊗C)",    # Diff-Evo + Curriculum (conservative)
        "(S⊕R)",    # Swarm + RL (adaptive)
        "(I⊗Z)"     # Baseline (control)
    ]
    
    # Process all combinations in parallel
    results = await cabinet.process_batch(sigils, shared_data)
    
    # Filter successful welds
    certified_welds = [
        r for r in results 
        if r.get('consensus_reached') and 'compilation' in r
    ]
    
    return certified_welds
    ``
   ## 6. Safety & Emergency Procedures
6.1 Health Monitoring

``phyton
# Check Cabinet vitals before heavy processing
status = cabinet.get_status()

if status['average_entropy'] > 0.25:
    print("Warning: Elevated entropy detected - consider cooling period")

if status['average_thermal_load'] > 0.75:
    print("Critical: Thermal stress high - initiate emergency shutdown")
    await cabinet.emergency_shutdown()

``
### 6.2 Emergency Shutdown

``phyton
# Immediate halt with graceful degradation
shutdown_report = await cabinet.emergency_shutdown()

print(f"Uptime: {shutdown_report['uptime_seconds']}s")
print(f"Processed: {shutdown_report['processed_count']} sigils")
for role, result in shutdown_report['roles'].items():
    print(f"  {role}: {result['status']}")

``
### 6.3 Circuit Breakers (Automatic)
The Cabinet has built-in protections:

    Entropy threshold: Abort if entropy > max_entropy (default 0.3)
    Thermal limit: Throttle if thermal_load > max_thermal_load (default 0.8)
    Timeout: Fail if operation exceeds timeout seconds (default 30.0)
    Consensus failure: Abort if Jury returns non-unanimous verdict

---

## 7. Configuration Reference
7.1 CabinetOrchestrator Parameters

``table

| Parameter          | Type  | Default | Range   | Description                              |
| ------------------ | ----- | ------- | ------- | ---------------------------------------- |
| `max_entropy`      | float | 0.3     | 0.0-1.0 | Maximum information disorder allowed     |
| `max_thermal_load` | float | 0.8     | 0.0-1.0 | Maximum compute stress before throttling |
| `timeout`          | float | 30.0    | >0      | Operation timeout in seconds             |


''

7.2 circulate() Parameters

