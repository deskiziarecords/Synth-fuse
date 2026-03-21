"""
SessionLogger - v0.4.0
Archives session provenance and state.
"""
import hashlib
import json
import time
from pathlib import Path

class SessionLogger:
    def __init__(self, context):
        self.context = context

    def write(self) -> tuple[Path, str]:
        """Write session log to disk."""
        log_dir = Path("session_logs")
        log_dir.mkdir(exist_ok=True)

        timestamp = int(time.time())
        log_path = log_dir / f"session_{self.context.session_id}_{timestamp}.md"

        content = f"# Session Log: {self.context.session_id}\n\n"
        content += f"Operator: {self.context.operator}\n"
        content += f"Boot time: {self.context.boot_time_ms}\n\n"
        content += "## Provenance\n"
        for entry in self.context.provenance:
            content += f"- {entry}\n"

        log_path.write_text(content)

        session_hash = hashlib.sha256(content.encode()).hexdigest()

        return log_path, session_hash
