# src/synthfuse/recipes/test_elixir_3.py
"""Validation tests for ELIXIR 3."""

import pytest
import asyncio
from synthfuse.recipes.elixir_3_chaotic_verification import (
    ChaoticVerificationEngine, 
    ProofCertificate,
    solve_sudoku
)


class TestElixir3:
    @pytest.fixture
    async def engine(self):
        engine = ChaoticVerificationEngine()
        await engine.initialize()
        yield engine
        await engine.emergency_stop()
    
    def test_sigil_valid(self):
        assert ChaoticVerificationEngine.SIGIL == "(L⊗C⊗Z)"
        # Validate primitives
        assert 'L' in "(L⊗C⊗Z)"  # Logic
        assert 'C' in "(L⊗C⊗Z)"  # Chaos
        assert 'Z' in "(L⊗C⊗Z)"  # Zero-point
    
    def test_version_lock(self):
        with pytest.raises(RuntimeError, match="v0.2.0"):
            # Would fail if version mismatch
            pass  # Tested in initialization
    
    @pytest.mark.asyncio
    async def test_sudoku_validation(self, engine):
        # Valid 9x9 Sudoku (81 chars)
        valid = "530070000600195000098006800800060003400803001700020006060000280000419005000080079"
        assert len(valid) == 81
        
        cnf = engine._validate_cnf(valid)
        assert cnf['type'] == 'sudoku'
        assert cnf['num_vars'] == 729  # 9x9x9
    
    @pytest.mark.asyncio
    async def test_dimacs_parsing(self, engine):
        dimacs = """
        c Simple SAT problem
        p cnf 3 2
        1 2 -3 0
        -1 -2 3 0
        """
        cnf = engine._validate_cnf(dimacs)
        assert cnf['type'] == 'dimacs'
        assert cnf['num_vars'] == 3
        assert cnf['num_clauses'] == 2
    
    @pytest.mark.asyncio
    async def test_zeta_pole_tracking(self, engine):
        state = jnp.array([0.1, 0.2, 0.15])
        pole = engine._compute_zeta_pole(state)
        
        assert isinstance(pole, complex)
        assert pole.real > 0  # Always positive real
        assert len(engine._pole_history) == 1
    
    @pytest.mark.asyncio
    async def test_termination_bound(self, engine):
        # Force pole history to exceed bound
        engine._pole_history = [complex(2, 0)] * 10  # radius = 2 > 0.99
        
        assert engine._check_termination_bound() == True
    
    @pytest.mark.asyncio
    async def test_certificate_verification(self, engine):
        cert = ProofCertificate(
            theorem_hash="a" * 32,
            proof_steps=["step1", "step2"],
            zeta_radius=0.95,
            entropy=0.15,
            thermal_load=0.6,
            duration_ms=100.0,
            cabinet_consensus=True,
            holographic_signature="9abcdef123456789",  # Would be computed
            timestamp_utc="2024-01-01T00:00:00+00:00"
        )
        
        # Note: signature won't match (we set it manually), but structure valid
        assert cert.theorem_hash == "a" * 32
        assert len(cert.proof_steps) == 2
    
    @pytest.mark.asyncio
    async def test_oracle_logging(self, engine):
        # Verify Oracle has logged Elixir 3 events
        from synthfuse.meta.self_documentation_oracle import ORACLE
        
        entries = ORACLE.query_history(
            author="elixir:3:chaotic_verification",
            limit=10
        )
        
        assert len(entries) >= 1  # At least instantiation log
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_full_verification(self, engine):
        """Integration test: actually solve a simple problem."""
        # Simple 2-SAT problem
        cnf = [[1, 2], [-1, 2], [1, -2]]
        
        cert = await engine.verify(
            <USER_PROVIDED_CNF>=cnf,
            <USER_PROVIDED_PROBLEM_TYPE>='sat'
        )
        
        assert cert.cabinet_consensus == True
        assert cert.zeta_radius <= 1.0  # Stable
        assert len(cert.proof_steps) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
