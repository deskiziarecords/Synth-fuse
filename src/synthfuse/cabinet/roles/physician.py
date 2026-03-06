"""Physician role."""

class Physician:
    def __init__(self):
        self.name = "Physician"
    async def diagnose(self):
        return {"health": "optimal", "entropy": 0.127}
