"""Jury role."""

class Jury:
    def __init__(self):
        self.name = "Jury"
    async def execute(self):
        return {"role": self.name, "status": "executed"}
