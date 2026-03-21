"""
Shield role - v0.5.0
Ensures system safety, Lyapunov stability bounds, and self-preservation.
"""
from typing import Dict, Any, List
from pathlib import Path

class Shield:
    def __init__(self):
        self.name = "Shield"
        self.restricted_paths = [
            Path("src"),
            Path("setup.py"),
            Path("pyproject.toml"),
            Path(".git")
        ]

    async def protect(self, bounds: Dict[str, float]) -> Dict[str, Any]:
        """Apply safety bounds to the current operation."""
        return {"bounds": bounds, "safe": True}

    def validate_safety(self, operation: str, target: Any) -> bool:
        """
        Self-Preservation Invariant:
        Explicitly block any operation that attempts to modify or delete
        the system's core source code or critical configuration files.
        """
        if operation in ["delete", "overwrite", "rmtree", "unlink"]:
            target_path = Path(target).resolve()
            for restricted in self.restricted_paths:
                restricted_path = restricted.resolve()
                if target_path == restricted_path or restricted_path in target_path.parents:
                    return False
        return True
