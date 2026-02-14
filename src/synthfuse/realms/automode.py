# In src/synthfuse/realms/automode.py

from synthfuse.lab.instruments.weight_kurve import WeightKurve

class AutoModeRealm:
    def explore_with_kurve_feedback(self, objective):
        """
        Use weight dynamics to guide exploration.
        """
        explorer = self.explorer  # meta_alchemist
        
        while self.thermal_credit > 0:
            # Exploration step
            result = explorer.step(objective)
            
            # Analyze weight trajectory
            if 'weight_history' in result:
                kurve = WeightKurve.from_training_run(
                    result['weight_history'],
                    result['steps'],
                    'exploration_layer'
                )
                sig = kurve.analyze()
                
                # High chaos = checkpoint and validate
                if sig.lyapunov_estimate > 0.5:
                    self.os._context.log(
                        f"AUTO-MODE: High chaos detected (λ={sig.lyapunov_estimate:.3f})"
                    )
                    self.checkpoint("chaos_detected", result)
                
                # High stress = request Lab extension with caution
                if sig.thermal_stress > 0.7:
                    self.os._context.log(
                        f"AUTO-MODE: Thermal stress elevated, requesting validation"
                    )
                    extension = self._validate_with_lab(result, sig)
                    if not extension:
                        break  # Halt exploration
            
            self.thermal_credit -= result.thermal_cost
