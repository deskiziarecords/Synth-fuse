"""Shield role."""

class Shield:
    def __init__(self):
        self.name = "Shield"
    async def execute(self):
        return {"role": self.name, "status": "executed"}
