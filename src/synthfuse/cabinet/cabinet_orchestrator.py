# src/synthfuse/cabinet/cabinet_orchestrator.py
"""Cabinet Orchestrator - Unified Field Engineering v0.2.0"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

# Cabinet Role Imports - Updated for v0.2.0 Sigil system
try:
    from .roles.architect import Architect
    from .roles.engineer import Engineer
    from .roles.librarian import Librarian
    from .roles.physician import Physician
    from .roles.shield import Shield
    from .roles.body import Body
    from .roles.jury import Jury
    ROLES_AVAILABLE = True
except ImportError:
    # Create minimal role classes if files don't exist
    ROLES_AVAILABLE = False
    
    class RoleBase:
        def __init__(self, name, emoji):
            self.name = name
            self.emoji = emoji
            self.status = "ready"
        
        async def process(self, sigil, data):
            return {"role": self.name, "action": "processed", "sigil": sigil}
    
    class Architect(RoleBase):
        def __init__(self):
            super().__init__("🏛️ Architect", "🏛️")
        
        async def blueprint(self, strategy="W-Orion"):
            return {"strategy": strategy, "coordinates": {"x": 0, "y": 0, "z": 1}}
    
    class Engineer(RoleBase):
        def __init__(self):
            super().__init__("🔧 Engineer", "🔧")
        
        async def compile(self, sigil):
            return {"sigil": sigil, "jax_code": f"# Compiled: {sigil}"}
    
    class Librarian(RoleBase):
        def __init__(self):
            super().__init__("📚 Librarian", "📚")
        
        async def ingest(self, data):
            return {"items": len(str(data)), "hash": "abc123"}
    
    class Physician(RoleBase):
        def __init__(self):
            super().__init__("🩺 Physician", "🩺")
        
        async def diagnose(self):
            return {"health": "optimal", "entropy": 0.127}
    
    class Shield(RoleBase):
        def __init__(self):
            super().__init__("🛡️ Shield", "🛡️")
        
        async def protect(self, bounds):
            return {"bounds": bounds, "safe": True}
    
    class Body(RoleBase):
        def __init__(self):
            super().__init__("🌡️ Body", "🌡️")
        
        async def thermoregulate(self, load):
            return {"load": load, "cooling": "active"}
    
    class Jury(RoleBase):
        def __init__(self):
            super().__init__("⚖️ Jury", "⚖️")
        
        async def deliberate(self, evidence):
            return {"verdict": "unanimous", "confidence": 0.95}

class CabinetOrchestrator:
    """
    Orchestrates the Cabinet of Alchemists - Unified Field Engineering v0.2.0
    
    The Cabinet consists of seven specialized agents governing execution:
    1. 🏛️ Architect - Strategic blueprinting via W-Orion search
    2. 🔧 Engineer - Sigil → JAX/XLA kernel compilation
    3. 📚 Librarian - Zeta-Vault & fluid ingestion management
    4. 🩺 Physician - Manifold health monitoring & surgical rollback
    5. 🛡️ Shield - Lyapunov safety bounds & OpenGate enforcement
    6. 🌡️ Body - Thermal mesh optimization
    7. ⚖️ Jury - Bayesian consensus validation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Cabinet Orchestrator.
        
        Args:
            config: Optional configuration dictionary
                - max_entropy: Maximum allowed entropy (default: 0.3)
                - max_thermal_load: Maximum thermal load (default: 0.8)
                - timeout: Operation timeout in seconds (default: 30.0)
        """
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Configuration
        self.config = config or {}
        self.max_entropy = self.config.get('max_entropy', 0.3)
        self.max_thermal_load = self.config.get('max_thermal_load', 0.8)
        self.timeout = self.config.get('timeout', 30.0)
        
        # Initialize roles
        self.roles = {
            "architect": Architect(),
            "engineer": Engineer(),
            "librarian": Librarian(),
            "physician": Physician(),
            "shield": Shield(),
            "body": Body(),
            "jury": Jury(),
        }
        
        # State
        self.status = "initialized"
        self.version = "0.2.0"
        self.start_time = datetime.now()
        self.processed_count = 0
        self.entropy_history = []
        self.thermal_history = []
        
        self.logger.info(f"Cabinet Orchestrator v{self.version} created")
        self.logger.info(f"Roles available: {ROLES_AVAILABLE}")
        
    async def initialize(self) -> bool:
        """
        Initialize the Cabinet and all roles.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Cabinet of Alchemists...")
            
            # Initialize each role
            initialization_tasks = []
            for name, role in self.roles.items():
                if hasattr(role, 'initialize'):
                    task = role.initialize()
                    if asyncio.iscoroutine(task):
                        initialization_tasks.append(task)
                    else:
                        self.logger.debug(f"Role {name} synchronous init")
                else:
                    self.logger.debug(f"Role {name} has no initialize method")
            
            # Wait for all async initializations
            if initialization_tasks:
                await asyncio.gather(*initialization_tasks, return_exceptions=True)
            
            self.status = "online"
            self.logger.info(f"✅ Cabinet initialized: Unified Field Engineering v{self.version}")
            self.logger.info(f"📊 Roles: {list(self.roles.keys())}")
            
            return True
            
        except Exception as e:
            self.status = "error"
            self.logger.error(f"❌ Cabinet initialization failed: {e}")
            return False
        
    async def process_sigil(self, sigil: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a Sigil through the entire Cabinet workflow.
        
        Args:
            sigil: Sigil expression (e.g., "(I⊗Z)", "(R⊕S)")
            data: Input data for processing
            
        Returns:
            Dict containing processing results and metrics
        """
        if self.status != "online":
            raise RuntimeError("Cabinet not operational. Call initialize() first.")
        
        self.processed_count += 1
        process_id = f"sigil_{self.processed_count:06d}"
        
        self.logger.info(f"Processing {process_id}: {sigil}")
        
        try:
            # Start timing
            start_time = datetime.now()
            
            # 1. Architect: Create strategic blueprint
            blueprint = await self.roles["architect"].blueprint(strategy="W-Orion")
            
            # 2. Engineer: Compile Sigil
            compilation = await self.roles["engineer"].compile(sigil)
            
            # 3. Librarian: Ingest data
            ingestion = await self.roles["librarian"].ingest(data)
            
            # 4. Physician: Monitor health
            diagnosis = await self.roles["physician"].diagnose()
            
            # 5. Shield: Apply safety bounds
            protection = await self.roles["shield"].protect(
                bounds={"entropy": self.max_entropy, "thermal": self.max_thermal_load}
            )
            
            # 6. Body: Thermal regulation
            thermal = await self.roles["body"].thermoregulate(load=0.18)
            
            # 7. Jury: Reach consensus
            evidence = {
                "blueprint": blueprint,
                "compilation": compilation,
                "ingestion": ingestion,
                "diagnosis": diagnosis,
                "protection": protection,
                "thermal": thermal,
            }
            verdict = await self.roles["jury"].deliberate(evidence)
            
            # Calculate metrics
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            entropy = diagnosis.get('entropy', 0.127)
            thermal_load = thermal.get('load', 0.18)
            consensus = verdict.get('verdict') == 'unanimous'
            
            # Update history
            self.entropy_history.append(entropy)
            self.thermal_history.append(thermal_load)
            
            # Build result
            result = {
                "process_id": process_id,
                "sigil": sigil,
                "duration_seconds": duration,
                "entropy": entropy,
                "thermal_load": thermal_load,
                "consensus_reached": consensus,
                "cabinet_version": self.version,
                "timestamp": end_time.isoformat(),
                
                # Role outputs
                "blueprint": blueprint,
                "compilation": compilation,
                "ingestion": ingestion,
                "diagnosis": diagnosis,
                "protection": protection,
                "thermal": thermal,
                "verdict": verdict,
                
                # Metadata
                "data_size": len(str(data)),
                "processed_count": self.processed_count,
            }
            
            self.logger.info(f"✅ {process_id} completed in {duration:.3f}s")
            self.logger.info(f"   Entropy: {entropy:.3f}, Thermal: {thermal_load:.2%}, Consensus: {consensus}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ {process_id} failed: {e}")
            return {
                "process_id": process_id,
                "sigil": sigil,
                "error": str(e),
                "status": "failed",
                "consensus_reached": False,
                "timestamp": datetime.now().isoformat(),
            }
    
    async def emergency_shutdown(self) -> Dict[str, Any]:
        """
        Perform emergency shutdown of the Cabinet.
        
        Returns:
            Dict with shutdown status
        """
        self.logger.warning("🛑 Emergency shutdown initiated")
        
        shutdown_results = {}
        for name, role in self.roles.items():
            try:
                if hasattr(role, 'shutdown'):
                    result = await role.shutdown() if asyncio.iscoroutinefunction(role.shutdown) else role.shutdown()
                    shutdown_results[name] = result
                else:
                    shutdown_results[name] = {"status": "no_shutdown_method"}
            except Exception as e:
                shutdown_results[name] = {"status": "error", "error": str(e)}
        
        self.status = "offline"
        
        result = {
            "status": "shutdown",
            "cabinet_status": self.status,
            "roles": shutdown_results,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "processed_count": self.processed_count,
            "timestamp": datetime.now().isoformat(),
        }
        
        self.logger.info("✅ Cabinet shutdown complete")
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current Cabinet status.
        
        Returns:
            Dict with status information
        """
        avg_entropy = sum(self.entropy_history[-10:]) / len(self.entropy_history[-10:]) if self.entropy_history else 0
        avg_thermal = sum(self.thermal_history[-10:]) / len(self.thermal_history[-10:]) if self.thermal_history else 0
        
        return {
            "status": self.status,
            "version": self.version,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "processed_count": self.processed_count,
            "average_entropy": avg_entropy,
            "average_thermal_load": avg_thermal,
            "entropy_history_length": len(self.entropy_history),
            "thermal_history_length": len(self.thermal_history),
            "roles_available": list(self.roles.keys()),
            "roles_functional": ROLES_AVAILABLE,
            "config": {
                "max_entropy": self.max_entropy,
                "max_thermal_load": self.max_thermal_load,
                "timeout": self.timeout,
            }
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get detailed performance metrics.
        
        Returns:
            Dict with metrics
        """
        return {
            "performance": {
                "total_processed": self.processed_count,
                "entropy_trend": self.entropy_history[-20:] if self.entropy_history else [],
                "thermal_trend": self.thermal_history[-20:] if self.thermal_history else [],
            },
            "health": {
                "status": self.status,
                "roles_online": len(self.roles),
                "avg_response_time": "N/A",  # Could be calculated with timing data
            },
            "system": {
                "version": self.version,
                "start_time": self.start_time.isoformat(),
                "current_time": datetime.now().isoformat(),
            }
        }
    
    async def process_batch(self, sigils: List[str], data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process multiple Sigils in batch.
        
        Args:
            sigils: List of Sigil expressions
            data: Shared input data
            
        Returns:
            List of results for each Sigil
        """
        tasks = [self.process_sigil(sigil, data) for sigil in sigils]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "sigil": sigils[i],
                    "error": str(result),
                    "status": "failed",
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def __repr__(self) -> str:
        return f"CabinetOrchestrator(v{self.version}, status={self.status}, roles={len(self.roles)})"
    
    def __str__(self) -> str:
        status = self.get_status()
        return (
            f"Cabinet of Alchemists v{self.version}\n"
            f"Status: {status['status']}\n"
            f"Uptime: {status['uptime_seconds']:.1f}s\n"
            f"Processed: {status['processed_count']} Sigils\n"
            f"Roles: {', '.join(status['roles_available'])}"
        )

# Convenience function for quick testing
async def test_cabinet():
    """Quick test function for the Cabinet."""
    cabinet = CabinetOrchestrator()
    print("Testing Cabinet Orchestrator v0.2.0...")
    print(f"Created: {cabinet}")
    
    # Initialize
    success = await cabinet.initialize()
    print(f"Initialized: {success}")
    
    if success:
        # Get status
        status = cabinet.get_status()
        print(f"Status: {json.dumps(status, indent=2)}")
        
        # Process a test Sigil
        result = await cabinet.process_sigil("(I⊗Z)", {"test": [1, 2, 3, 4, 5]})
        print(f"Test result keys: {list(result.keys())}")
        
        # Shutdown
        shutdown = await cabinet.emergency_shutdown()
        print(f"Shutdown: {shutdown['status']}")
    
    return success

if __name__ == "__main__":
    # Run test if file is executed directly
    asyncio.run(test_cabinet())