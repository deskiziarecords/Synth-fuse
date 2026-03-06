"""Body role."""

class Body:
    def __init__(self):
        self.name = "Body"
    async def thermoregulate(self, load):
        return {"load": load, "cooling": "active"}
