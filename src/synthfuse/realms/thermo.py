"""
Realm 5: THERMO-EFFICIENCY 🌡️

Physical governance—sensor veto supreme.
Uses: forum/arena.py, meta/regulator.py, systems/thermo_mesh.py
Sigil: ((I⊗Z)⊗S)⊙(F⊕R)
"""

from typing import Dict, Any, List, Optional
from synthfuse.lab.instruments.weight_kurve import WeightKurveThermoSensor

class ThermoRealm:
    """
    Thermo-efficiency realm enforces physical supremacy over consensus.
    """

    def __init__(self, os):
        self.os = os
        self.forum = os.forum
        self.regulator = os.regulator
        try:
            self.mesh = os.load_module('synthfuse.systems.thermo_mesh')
        except:
            self.mesh = None

    def deliberate(self, proposal: Dict) -> Dict:
        """
        Forum debate with Hardware Veto.
        Consensus is heuristic. Sensors are ground truth.
        """
        self.os._context.log(f"THERMO: Deliberating {proposal.get('id', 'unnamed')}")

        # Phase 1: LLM debate (Forum Arena)
        consensus = self.forum.debate(proposal, roles=7)
        self.os._context.log(f"THERMO: Forum consensus {consensus.vote}/7")

        # Phase 2: Physical reality check
        physical = self._sample_instruments()

        # Phase 3: Sensor Veto (Physical Supremacy)
        if physical['thermal_load'] > self.os.PhysicalLaw.THERMAL_HARD_LIMIT:
            self.os._trigger_veto(
                f"Consensus {consensus.vote}/7, "
                f"but thermal sensors read {physical['thermal_load']:.2f}"
            )
            return {
                'status': 'VETOED_BY_PHYSICS',
                'consensus': consensus.vote,
                'reality': physical,
                'message': 'Consensus reached, but Physical Reality disagreed.'
            }

        return {
            'status': 'CERTIFIED',
            'consensus': consensus.vote,
            'physical': physical,
            'certified': True
        }

    def deploy_weight_sensors(self, model_layers: List[str]):
        """
        Deploy Kurve sensors on high-thermal layers.
        """
        sensors = {}
        for layer_id in model_layers:
            sensor = WeightKurveThermoSensor(layer_id, self.os._context)
            sensors[layer_id] = sensor
        return sensors
    
    def deliberation_with_kurve(self, proposal: Dict, model_sensors: Dict[str, WeightKurveThermoSensor]):
        """
        Include weight dynamics in Thermo deliberation.
        """
        # Collect sensor alerts
        alerts = []
        for layer_id, sensor in model_sensors.items():
            alert = sensor.sample(sensor.history[-1] if sensor.history else None)
            if alert:
                alerts.append(alert.to_dict())
        
        # If any layer shows thermal stress, elevate scrutiny
        if alerts:
            proposal['weight_dynamics'] = alerts
            proposal['thermal_risk'] = max(a['thermal_stress'] for a in alerts)
        
        # Continue to standard deliberation
        return self.deliberate(proposal)

    def _sample_instruments(self) -> Dict:
        """Sample physical reality."""
        sample = self.regulator.sample()
        return {
            'thermal_load': sample.get('thermal_load', 0.0),
            'entropy': sample.get('entropy', 0.0),
            'hsm_attested': True
        }
