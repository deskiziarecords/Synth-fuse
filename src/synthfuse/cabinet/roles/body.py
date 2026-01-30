"""Body role."""

class Body:
    def __init__(self):
        self.name = "Body"
    async def execute(self):
        return {"role": self.name, "status": "executed"}
