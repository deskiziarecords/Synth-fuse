"""Shield role."""

class Shield:
    def __init__(self):
        self.name = "Shield"
    async def protect(self, bounds):
        return {"bounds": bounds, "safe": True}
