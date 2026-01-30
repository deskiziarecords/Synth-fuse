"""Physician role."""

class Physician:
    def __init__(self):
        self.name = "Physician"
    async def execute(self):
        return {"role": self.name, "status": "executed"}
