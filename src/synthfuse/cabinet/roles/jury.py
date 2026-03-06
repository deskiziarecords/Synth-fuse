"""Jury role."""

class Jury:
    def __init__(self):
        self.name = "Jury"
    async def deliberate(self, evidence):
        return {"verdict": "unanimous", "confidence": 0.95}
