# In src/synthfuse/realms/thermo.py

from synthfuse.lab.instruments.weight_kurve import WeightKurveThermoSensor

class ThermoRealm:
    def deploy_weight_sensors(self, model_layers):
        """
        Deploy Kurve sensors on high-thermal layers.
        """
        sensors = {}
        for layer_id in model_layers:
            sensor = WeightKurveThermoSensor(layer_id, self.os._context)
            sensors[layer_id] = sensor
        return sensors
    
    def deliberation_with_kurve(self, proposal, model_sensors):
        """
        Include weight dynamics in Thermo deliberation.
        """
        # Collect sensor alerts
        alerts = []
        for layer_id, sensor in model_sensors.items():
            alert = sensor.sample(sensor.history[-1] if sensor.history else None)
            if alert:
                alerts.append(alert)
        
        # If any layer shows thermal stress, elevate scrutiny
        if alerts:
            proposal['weight_dynamics'] = alerts
            proposal['thermal_risk'] = max(a['thermal_stress'] for a in alerts)
        
        # Continue to standard deliberation
        return self.deliberate(proposal)
