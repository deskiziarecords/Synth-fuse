import pytest
import asyncio
from pathlib import Path
from synthfuse.os.kernel import SynthFuseOS, Realm, ThermalViolation
from synthfuse.cabinet.cabinet_orchestrator import CabinetOrchestrator

class TestErasureProtection:
    @pytest.fixture
    def kernel(self):
        k = SynthFuseOS()
        k.boot()
        return k

    def test_safe_write_restricted_path(self, kernel):
        """Assert that safe_write prevents overwriting src/ files."""
        target = Path("src/synthfuse/__init__.py")
        with pytest.raises(ThermalViolation) as excinfo:
            kernel.safe_write(target, "malicious content")
        assert "Safety Violation" in str(excinfo.value)

        # Verify file content is unchanged
        with open(target, 'r') as f:
            content = f.read()
            assert "malicious content" not in content

    def test_safe_delete_restricted_path(self, kernel):
        """Assert that safe_delete prevents deleting src/ files."""
        target = Path("src/synthfuse/os/kernel.py")
        with pytest.raises(ThermalViolation) as excinfo:
            kernel.safe_delete(target)
        assert "Safety Violation" in str(excinfo.value)

        # Verify file still exists
        assert target.exists()

    def test_global_os_remove_restricted_path(self, kernel):
        """Assert that monkey-patched os.remove prevents deleting src/ files."""
        import os
        target = Path("src/synthfuse/__init__.py")
        with pytest.raises(PermissionError) as excinfo:
            os.remove(target)
        assert "Synth-Fuse OS Veto" in str(excinfo.value)
        assert target.exists()

    def test_safe_write_allowed_path(self, kernel, tmp_path):
        """Assert that safe_write allows writing to non-restricted paths."""
        target = tmp_path / "safe_file.txt"
        kernel.safe_write(target, "safe content")

        assert target.exists()
        assert target.read_text() == "safe content"

    @pytest.mark.asyncio
    async def test_jury_high_entropy_veto(self, kernel):
        """Assert that Jury role flags high-entropy destructive operations."""
        evidence = {
            'diagnosis': {'entropy': 0.8},
            'intent': {'destructive': True}
        }
        result = await kernel.cabinet.roles['jury'].deliberate(evidence)
        assert result['verdict'] == "dissent"
        assert result['confidence'] < 0.5

if __name__ == "__main__":
    pytest.main([__file__])
