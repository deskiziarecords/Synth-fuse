"""Librarian role."""

class Librarian:
    def __init__(self):
        self.name = "Librarian"
    async def ingest(self, data):
        return {"items": len(str(data)), "hash": "abc123"}
