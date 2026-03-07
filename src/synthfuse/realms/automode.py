"""
Realm 3: AUTO-MODE 🔬

Leashed exploration—20% TDP base, Lab-granted extensions.
Uses: meta/meta_alchemist.py, meta/regulator.py
Sigil: (R⊗C)⊗(φ⋈D)
"""

from typing import Any, Dict, List, Optional
import jax.numpy as jnp
from synthfuse.lab.instruments.weight_kurve import WeightKurve

class AutoModeRealm:
    """
    Auto-mode provides leashed exploration within thermal boundaries.
    """

    def __init__(self, os):
        self.os = os
        try:
            self.explorer = os.load_module('synthfuse.meta.meta_alchemist').MetaAlchemist()
        except:
            self.explorer = None
        self.regulator = os.regulator
        self.thermal_credit = os.PhysicalLaw.BASE_TDP_BUDGET  # 20%
        self.checkpoint_interval = 100  # iterations

    def explore(self, objective: str):
        """
        Leashed: Infinite only if thermally neutral.
        Checkpointing gated by entropy gradient ∇S.
        """
        self.os._context.log(f"AUTO-MODE: Exploring {objective}")

        iteration = 0
        results = []
        while self.thermal_credit > 0:
            # Exploration step
            if not self.explorer:
                break
            delta = self.explorer.step(objective)
            iteration += 1
            results.append(delta)

            # Entropy gradient checkpoint
            if iteration % self.checkpoint_interval == 0:
                nabla_s = self._entropy_gradient()
                if abs(nabla_s) > 0.05:
                    self.os._context.log(f"AUTO-MODE: Checkpoint at ∇S={nabla_s:.3f}")
                    extension = self._validate_with_lab(delta)
                    if extension:
                        self.thermal_credit += extension

            # Deduct cost
            cost = getattr(delta, 'thermal_cost', 0.01)
            self.thermal_credit -= cost

            # Thermally neutral check
            if self._is_thermally_neutral():
                self.thermal_credit = float('inf')  # Unlimited, monitored
                self.os._context.log("AUTO-MODE: Thermally neutral—unlimited exploration")

        self.os._context.log("AUTO-MODE: Thermal credit exhausted")
        return results

    def explore_with_kurve_feedback(self, objective: str):
        """
        Use weight dynamics to guide exploration.
        """
        self.os._context.log(f"AUTO-MODE: Exploring {objective} with Kurve feedback")
        
        iteration = 0
        while self.thermal_credit > 0:
            # Exploration step
            if not self.explorer:
                break
            result = self.explorer.step(objective)
            
            # Analyze weight trajectory
            if isinstance(result, dict) and 'weight_history' in result:
                kurve = WeightKurve.from_training_run(
                    result['weight_history'],
                    jnp.arange(len(result['weight_history'])),
                    'exploration_layer'
                )
                sig = kurve.analyze()
                
                # High chaos = checkpoint and validate
                if sig.lyapunov_estimate > 0.5:
                    self.os._context.log(
                        f"AUTO-MODE: High chaos detected (λ={sig.lyapunov_estimate:.3f})"
                    )
                    # self.checkpoint("chaos_detected", result)
                
                # High stress = request Lab extension with caution
                if sig.thermal_stress > 0.7:
                    self.os._context.log(
                        f"AUTO-MODE: Thermal stress elevated, requesting validation"
                    )
                    extension = self._validate_with_lab(result)
                    if not extension:
                        break  # Halt exploration
            
            cost = getattr(result, 'thermal_cost', 0.01) if not isinstance(result, dict) else 0.01
            self.thermal_credit -= cost
            iteration += 1

    def _entropy_gradient(self) -> float:
        """Calculate ∇S from history."""
        return 0.0  # Placeholder

    def _validate_with_lab(self, delta):
        """Mandatory Lab validation for extension."""
        lab = self.os.enter_realm(self.os.Realm.LAB)
        return lab.grant_extension(delta)

    def _is_thermally_neutral(self) -> bool:
        """δT ≈ 0, no waste heat."""
        return self.os._context.thermal.is_neutral()
