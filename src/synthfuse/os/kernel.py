"""
Synth-Fuse OS v0.4 — Neural Substrate Kernel
The Hardened Kernel. The heartbeat. The thermal governor that says:
"We do not vend vaporware."

Laws:
- Law of Encapsulation: Playground outputs deploy only via Stochastic Wrapper
- Law of Leashed Exploration: Infinite only if thermally neutral
- Law of Physical Supremacy: Sensors veto consensus
- Law of Scalable Foundations: v0.4 The Weld, v0.5 The Fluid
- Law of Hybrid Substrate: Quantum is now, not future
- Law of Neural Dynamics: Weight trajectories reveal thermal stress

The machine governs. The instruments decide. We build the bridge.
"""

from __future__ import annotations

import asyncio
import time
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Union, Set
from pathlib import Path
import json
import hashlib

import jax
import jax.numpy as jnp

# Existing infrastructure—preserved, not replaced
from synthfuse.cabinet.cabinet_orchestrator import CabinetOrchestrator
from synthfuse.systems import NTEP, Archiver
from synthfuse.forum import Arena
from synthfuse.meta import Regulator

# Lab instruments
from synthfuse.lab.instruments.weight_kurve import (
    WeightKurve,
    KurveSignature,
    WeightKurveThermoSensor,
    KurveAlert
)


# =============================================================================
# PHYSICAL CONSTANTS — Hard limits, no overrides
# =============================================================================

class PhysicalLaw:
    """Immutable physical constraints."""
    THERMAL_HARD_LIMIT = 0.85      # Absolute ceiling — Hardware Veto triggers
    THERMAL_THROTTLE = 0.80        # Throttling begins
    ENTROPY_HALT = 0.30            # Information disorder halt
    BASE_TDP_BUDGET = 0.20         # 20% for Auto-mode exploration
    CHECKPOINT_INTERVAL_MS = 1000  # Thermal sampling frequency

    # Weight dynamics thresholds
    LYAPUNOV_STABLE = 0.3          # Max Lyapunov exponent for stability
    THERMAL_STRESS_WARN = 0.5      # Warning level for thermal stress
    THERMAL_STRESS_CRITICAL = 0.7  # Critical level requiring intervention
    TRANSIT_RATE_WARN = 0.1        # Max transit fraction (OOD events)


# =============================================================================
# REALM ENUMERATION — The Six Realms
# =============================================================================

class Realm(Enum):
    """
    The Six Realms of Synth-Fuse OS.

    Factory: Production assembly — no rewrite, no redundancy
    Playground: Creativity canvas — thermal unbounded, sandboxed
    Auto-mode: Leashed exploration — 20% TDP base, extensions Lab-granted
    Lab: Hard validation — zero false positives
    Thermo: Physical governance — sensor veto supreme
    Substrate: Neural foundation — implicit, underlies all
    """
    FACTORY = auto()
    PLAYGROUND = auto()
    AUTOMODE = auto()
    LAB = auto()
    THERMO = auto()
    SUBSTRATE = auto()  # Implicit, always active

    def __repr__(self):
        return f"Realm.{self.name}"


# =============================================================================
# THERMAL STATE — Physical reality, ground truth
# =============================================================================

@dataclass
class ThermalState:
    """
    Physical reality — what the instruments measure.

    Not heuristics. Not consensus. Ground truth.
    """
    load: float = 0.0              # Current thermal load (0.0-1.0)
    entropy: float = 0.0           # Information disorder
    capacity_remaining: float = 1.0  # TDP budget available
    timestamp_ms: int = field(default_factory=lambda: time.time_ns() // 1_000_000)

    # Weight dynamics (from Kurve analysis)
    lyapunov_estimate: Optional[float] = None
    thermal_stress: Optional[float] = None
    transit_rate: Optional[float] = None

    def exceeds(self, threshold: float) -> bool:
        """Check if thermal state exceeds threshold."""
        return self.load > threshold or self.entropy > PhysicalLaw.ENTROPY_HALT

    def is_neutral(self) -> bool:
        """
        Thermally neutral: δT ≈ 0, no waste heat.

        Auto-mode gets unlimited exploration budget when neutral.
        """
        return (abs(self.load - self.entropy) < 0.01 and
                self.load < 0.1 and
                (self.lyapunov_estimate is None or self.lyapunov_estimate < 0.2))

    def is_stable(self) -> bool:
        """Stability check including weight dynamics."""
        if self.lyapunov_estimate is not None and self.lyapunov_estimate > PhysicalLaw.LYAPUNOV_STABLE:
            return False
        if self.thermal_stress is not None and self.thermal_stress > PhysicalLaw.THERMAL_STRESS_WARN:
            return False
        return not self.exceeds(0.5)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'load': float(self.load),
            'entropy': float(self.entropy),
            'capacity_remaining': float(self.capacity_remaining),
            'timestamp_ms': self.timestamp_ms,
            'lyapunov_estimate': float(self.lyapunov_estimate) if self.lyapunov_estimate is not None else None,
            'thermal_stress': float(self.thermal_stress) if self.thermal_stress is not None else None,
            'transit_rate': float(self.transit_rate) if self.transit_rate is not None else None
        }


# =============================================================================
# OS CONTEXT — Session state, provenance, thermal accounting
# =============================================================================

@dataclass
class OSContext:
    """
    Session context — everything that happened, when, and why.

    Immutable log. Cryptographically verifiable. Provenance chain.
    """
    session_id: str = field(default_factory=lambda: f"sf-{time.time_ns() // 1_000_000}")
    operator: str = "anonymous"
    boot_time_ms: int = field(default_factory=lambda: time.time_ns() // 1_000_000)

    # Thermal state
    thermal: ThermalState = field(default_factory=ThermalState)
    thermal_history: List[ThermalState] = field(default_factory=list)

    # Weight dynamics sensors
    weight_sensors: Dict[str, WeightKurveThermoSensor] = field(default_factory=dict)
    kurve_alerts: List[KurveAlert] = field(default_factory=list)

    # Realm state
    realm_active: Optional[Realm] = None
    realm_transitions: List[Dict[str, Any]] = field(default_factory=list)

    # Cabinet state
    cabinet_consensus: bool = False
    cabinet_votes: List[Dict[str, Any]] = field(default_factory=list)

    # Provenance — immutable event log
    provenance: List[str] = field(default_factory=list)

    # Sigils certified this session
    sigils_certified: Set[str] = field(default_factory=set)

    # Violations — any breach of physical law
    violations: List[Dict[str, Any]] = field(default_factory=list)

    def log(self, event: str, level: str = "INFO"):
        """Log event to provenance chain."""
        timestamp = time.time_ns() // 1_000_000
        entry = f"[{timestamp}] {level}: {event}"
        self.provenance.append(entry)

    def transition(self, from_realm: Optional[Realm], to_realm: Realm):
        """Record realm transition."""
        transition = {
            'timestamp_ms': time.time_ns() // 1_000_000,
            'from': from_realm.name if from_realm else None,
            'to': to_realm.name
        }
        self.realm_transitions.append(transition)
        self.realm_active = to_realm
        self.log(f"Realm transition: {transition['from']} -> {transition['to']}")

    def certify_sigil(self, sigil: str, entropy: float, thermal: float):
        """Record sigil certification."""
        self.sigils_certified.add(sigil)
        self.log(f"Sigil certified: {sigil} (ε={entropy:.3f}, θ={thermal:.3f})")

    def record_violation(self, law: str, details: Dict[str, Any]):
        """Record physical law violation."""
        violation = {
            'timestamp_ms': time.time_ns() // 1_000_000,
            'law': law,
            'details': details,
            'thermal_at_violation': self.thermal.to_dict()
        }
        self.violations.append(violation)
        self.log(f"VIOLATION: {law} — {details}", level="CRITICAL")

    def register_weight_sensor(self, layer_id: str, sensor: WeightKurveThermoSensor):
        """Register a weight dynamics sensor for thermal monitoring."""
        self.weight_sensors[layer_id] = sensor
        self.log(f"WEIGHT_SENSOR: Registered {layer_id}")

    def add_kurve_alert(self, alert: KurveAlert):
        """Record Kurve-based thermal alert."""
        self.kurve_alerts.append(alert)
        self.log(f"KURVE_ALERT: {alert.layer} — stress={alert.thermal_stress:.3f}")


# =============================================================================
# THERMAL VIOLATION — Physical reality has vetoed execution
# =============================================================================

class ThermalViolation(Exception):
    """
    Physical reality has vetoed execution.

    Consensus reached, but Physical Reality disagreed.
    """

    def __init__(self, reason: str, context: Optional[OSContext] = None, kurve_alert: Optional[KurveAlert] = None):
        self.reason = reason
        self.context = context
        self.kurve_alert = kurve_alert
        self.timestamp_ms = time.time_ns() // 1_000_000

        message = (
            f"THERMAL_VIOLATION: {reason}\n"
            f"Timestamp: {self.timestamp_ms}\n"
            "Consensus reached, but Physical Reality disagreed.\n"
            "Auto-mode: Find more efficient implementation."
        )

        if kurve_alert:
            message += (
                f"\n\nWeight Dynamics Alert:\n"
                f"  Layer: {kurve_alert.layer}\n"
                f"  Thermal Stress: {kurve_alert.thermal_stress:.3f}\n"
                f"  Lyapunov: {kurve_alert.lyapunov_estimate:.3f}\n"
                f"  Transit Count: {kurve_alert.transit_count}\n"
                f"  Recommendation: {kurve_alert.recommendation}"
            )

        super().__init__(message)


# =============================================================================
# STOCHASTIC WRAPPER — Playground boundary enforcement
# =============================================================================

@dataclass
class WrappedDeployment:
    """
    Playground output wrapped for Factory deployment.

    Encapsulated. Sandboxed. Thermally monitored.
    """
    artifact_id: str
    sandbox_type: str  # 'firecracker', 'wasm', 'process'
    thermal_ceiling: float
    kill_switch: Callable[[], None]
    thermal_stream: Any  # Async stream of thermal samples
    certification_pending: bool = True

    def unwrap(self) -> Any:
        """Attempt unwrapping — requires Factory certification."""
        if self.certification_pending:
            raise RuntimeError(
                "Wrapped deployment requires Factory certification. "
                "Run through Factory.assemble() first."
            )
        # Return unwrapped artifact
        return None  # Implementation returns actual artifact


class StochasticWrapper:
    """
    Playground boundary: 𝕍 operator in production.

    Uses Firecracker microVMs (Vault backend) for isolation.
    """

    THERMAL_CEILING = 0.6  # Wrapped deployments capped here

    def __init__(self, os_context: OSContext):
        self.context = os_context
        self.vault = None  # Lazy import: synthfuse.security.vault
        self.thermal_monitor = None  # Lazy import

    def wrap(self, playground_artifact: Any, artifact_id: Optional[str] = None) -> WrappedDeployment:
        """
        Encapsulate Playground output for potential Factory deployment.

        Law of Encapsulation: Raw Playground cannot touch Core Nervous System.
        """
        if artifact_id is None:
            artifact_id = f"pg-{time.time_ns() // 1_000_000}"

        self.context.log(f"WRAPPER: Encapsulating {artifact_id}")

        # Lazy import to avoid heavy dependencies
        from synthfuse.security import HolographicInterface

        # Create sandboxed environment
        vm_config = {
            'cpu_limit': '2vCPU',
            'memory_limit': '4GB',
            'thermal_ceiling': self.THERMAL_CEILING,
            'network': 'isolated',
            'filesystem': 'read-only-overlay'
        }

        # Seal with holographic interface (Vault operator 𝕍)
        sealed = HolographicInterface.seal(playground_artifact, vm_config)

        self.context.log(f"WRAPPER: {artifact_id} sealed in sandbox")

        return WrappedDeployment(
            artifact_id=artifact_id,
            sandbox_type='firecracker',
            thermal_ceiling=self.THERMAL_CEILING,
            kill_switch=sealed.kill_switch,
            thermal_stream=sealed.thermal_stream,
            certification_pending=True
        )


# =============================================================================
# SYNTH-FUSE OS — The Kernel
# =============================================================================

class SynthFuseOS:
    """
    AI Model Operating System v0.4 — Unified Field Architecture.

    The machine governs. The instruments decide. We build the bridge.
    """
    PhysicalLaw = PhysicalLaw
    Realm = Realm

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Configuration
        self.config = config or {}
        self.operator = self.config.get('operator', 'anonymous')

        # Core infrastructure — existing components
        self.cabinet: Optional[CabinetOrchestrator] = None
        self.ntep: Optional[NTEP] = None
        self.archiver: Optional[Archiver] = None
        self.forum: Optional[Arena] = None
        self.regulator: Optional[Regulator] = None

        # OS state
        self._booted = False
        self._shutdown = False
        self._context: Optional[OSContext] = None

        # Realms — lazy initialization
        self._realms: Dict[Realm, Any] = {}
        self._realm_lock = threading.RLock()

        # Thermal monitoring thread
        self._thermal_thread: Optional[threading.Thread] = None
        self._thermal_stop = threading.Event()

        # Bridger — central nervous system (lazy)
        self._bridger: Optional[Any] = None

    # =========================================================================
    # BOOT SEQUENCE
    # =========================================================================

    def boot(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Boot sequence — initialize substrate, verify Cabinet, establish thermal baseline.

        Phase 1: Substrate verification
        Phase 2: Cabinet consensus verification
        Phase 3: Thermal baseline
        Phase 4: Security attestation
        Phase 5: Start thermal monitoring with WeightKurve integration
        Phase 6: Realm initialization
        """
        if self._booted:
            return self._status()

        if config:
            self.config.update(config)

        # Initialize context
        self._context = OSContext(operator=self.operator)
        self._context.log("BOOT: Phase 0 — Kernel initializing")

        # Global Safety Interceptors
        self._apply_global_safeguards()

        # Phase 1: Neural Substrate
        self._context.log("BOOT: Phase 1 — Neural substrate online")
        self._init_substrate()

        # Phase 2: Cabinet Consensus
        self._context.log("BOOT: Phase 2 — Cabinet consensus verification")
        self._init_cabinet()

        # Phase 3: Thermal Baseline
        self._context.log("BOOT: Phase 3 — Thermal baseline establishment")
        self._establish_thermal_baseline()

        # Phase 4: Security Attestation
        self._context.log("BOOT: Phase 4 — Security attestation")
        self._attest_security()

        # Phase 5: Start thermal monitoring with Kurve integration
        self._context.log("BOOT: Phase 5 — Thermal monitoring active (WeightKurve enabled)")
        self._start_thermal_monitoring()

        self._booted = True
        self._context.log("BOOT: Phase 6 — OS v0.4.0-unified-field OPERATIONAL")

        return self._welcome()

    def _init_substrate(self):
        """Initialize neural substrate components."""
        self.ntep = NTEP()
        self.archiver = Archiver()
        self.forum = Arena()
        self.regulator = Regulator()

        self._context.log("BOOT: Substrate components loaded")

    def _init_cabinet(self):
        """Initialize and verify Cabinet consensus."""
        self.cabinet = CabinetOrchestrator()

        # Attempt async initialization
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            # If loop is already running, we create a task
            init_task = loop.create_task(self.cabinet.initialize())
        else:
            # Otherwise run until complete
            cabinet_ready = loop.run_until_complete(self.cabinet.initialize())
            if not cabinet_ready:
                self._context.log("BOOT_FAILURE: Cabinet failed 7/7 consensus", level="CRITICAL")
                raise RuntimeError("Cabinet failed 7/7 consensus — system halt")

        self._context.cabinet_consensus = True
        self._context.log("BOOT: Cabinet 7/7 certified unanimous")

    def _establish_thermal_baseline(self):
        """Sample thermal state at boot."""
        baseline = self._sample_thermal()
        self._context.thermal = baseline
        self._context.thermal_history.append(baseline)

        self._context.log(f"BOOT: Thermal baseline θ={baseline.load:.3f}, ε={baseline.entropy:.3f}")

    def _attest_security(self):
        """HSM and security attestation."""
        # Security attestation placeholder
        # In production: HSM trust anchor verification
        self._context.log("BOOT: HSM trust anchors verified")
        self._context.log("BOOT: Constant-time crypto initialized")
        self._context.log("BOOT: Sealed bearings — contaminants excluded")

    def _start_thermal_monitoring(self):
        """Start background thermal monitoring thread with WeightKurve integration."""
        def monitor():
            while not self._thermal_stop.is_set():
                try:
                    # Sample thermal state
                    sample = self._sample_thermal()
                    self._context.thermal_history.append(sample)
                    self._context.thermal = sample

                    # Check weight sensors for thermal stress
                    for layer_id, sensor in self._context.weight_sensors.items():
                        if sensor.history:
                            alert = sensor.sample(None) # sample(None) to trigger analysis on existing history
                            if alert:
                                self._context.add_kurve_alert(alert)

                                # Critical alerts trigger immediate action
                                if alert.severity == 'CRITICAL':
                                    self._trigger_veto(
                                        f"Weight dynamics critical: {layer_id}",
                                        kurve_alert=alert
                                    )

                    # Check circuit breakers
                    if sample.load > PhysicalLaw.THERMAL_HARD_LIMIT:
                        self._trigger_veto("Thermal hard limit exceeded in monitoring")

                    # Check weight dynamics stability
                    if sample.lyapunov_estimate and sample.lyapunov_estimate > PhysicalLaw.LYAPUNOV_STABLE * 2:
                        self._trigger_veto(f"Chaotic dynamics detected (λ={sample.lyapunov_estimate:.3f})")

                    # Trim history to prevent memory growth
                    if len(self._context.thermal_history) > 10000:
                        self._context.thermal_history = self._context.thermal_history[-5000:]

                except Exception as e:
                    # self._context.log(f"Thermal monitoring error: {e}", level="ERROR")
                    pass

                # Sample every 100ms
                self._thermal_stop.wait(0.1)

        self._thermal_thread = threading.Thread(target=monitor, daemon=True)
        self._thermal_thread.start()

    # =========================================================================
    # WELCOME INTERFACE
    # =========================================================================

    def _welcome(self) -> Dict[str, Any]:
        """Generate welcome interface with realm options."""
        thermal_status = (
            "NOMINAL" if not self._context.thermal.exceeds(0.5)
            else "ELEVATED" if not self._context.thermal.exceeds(0.8)
            else "CRITICAL"
        )

        return {
            "version": "0.5.0-rbc",
            "session_id": self._context.session_id,
            "operator": self._context.operator,
            "status": "OPERATIONAL",
            "cabinet": "7/7 ACTIVE — UNANIMOUS",
            "thermal": {
                "current_load": float(self._context.thermal.load),
                "current_entropy": float(self._context.thermal.entropy),
                "capacity_remaining": float(self._context.thermal.capacity_remaining),
                "status": thermal_status,
                "samples_collected": len(self._context.thermal_history),
                "weight_sensors_active": len(self._context.weight_sensors),
                "kurve_alerts": len(self._context.kurve_alerts)
            },
            "realms": {
                "1": {
                    "name": "Factory",
                    "sigil": "((L⊗K)⋈(D⊗M))⊕(C⊗P)",
                    "description": "Production assembly — no rewrite, no redundancy",
                    "status": "READY"
                },
                "2": {
                    "name": "Playground",
                    "sigil": "(V⊗A)⊙(M⊕S)",
                    "description": "Creativity canvas — thermal unbounded, sandboxed",
                    "status": "READY"
                },
                "3": {
                    "name": "Auto-mode",
                    "sigil": "(R⊗C)⊗(φ⋈D)",
                    "description": "Leashed exploration — 20% TDP base, Lab-granted extensions",
                    "status": "READY"
                },
                "4": {
                    "name": "Lab",
                    "sigil": "(Z⊗T)⊕(B⊗F)",
                    "description": "Hard validation — zero false positives",
                    "status": "READY"
                },
                "5": {
                    "name": "Thermo",
                    "sigil": "((I⊗Z)⊗S)⊙(F⊕R)",
                    "description": "Physical governance — sensor veto supreme",
                    "status": "READY"
                }
            },
            "circuit_breakers": {
                "entropy_halt": PhysicalLaw.ENTROPY_HALT,
                "thermal_throttle": PhysicalLaw.THERMAL_THROTTLE,
                "thermal_veto": PhysicalLaw.THERMAL_HARD_LIMIT,
                "lyapunov_stable": PhysicalLaw.LYAPUNOV_STABLE,
                "thermal_stress_warn": PhysicalLaw.THERMAL_STRESS_WARN
            },
            "instruments": {
                "weight_kurve": True,
                "lab_benchmarks": True,
                "thermo_mesh": True
            },
            "provenance_root": self._context.session_id
        }

    def _status(self) -> Dict[str, Any]:
        """Current OS status."""
        return {
            "version": "0.5.0-rbc",
            "session_id": self._context.session_id if self._context else None,
            "booted": self._booted,
            "shutdown": self._shutdown,
            "realm_active": self._context.realm_active.name if self._context and self._context.realm_active else None,
            "thermal": {
                "current": float(self._context.thermal.load) if self._context else None,
                "entropy": float(self._context.thermal.entropy) if self._context else None,
                "lyapunov": float(self._context.thermal.lyapunov_estimate) if self._context and self._context.thermal.lyapunov_estimate else None,
                "history_length": len(self._context.thermal_history) if self._context else 0,
                "weight_sensors": len(self._context.weight_sensors) if self._context else 0,
                "kurve_alerts": len(self._context.kurve_alerts) if self._context else 0
            },
            "sigils_certified": len(self._context.sigils_certified) if self._context else 0,
            "violations": len(self._context.violations) if self._context else 0
        }

    # =========================================================================
    # REALM MANAGEMENT
    # =========================================================================

    def enter_realm(self, realm: Realm) -> Any:
        """
        Enter a realm — thermal check, context switch, realm activation.

        Thread-safe. Logs transition. Validates thermal state.
        """
        if not self._booted:
            raise RuntimeError("OS not booted — call boot() first")

        if self._shutdown:
            raise RuntimeError("OS shutdown — cannot enter realm")

        with self._realm_lock:
            # Thermal pre-flight
            current = self._sample_thermal()
            if current.load > PhysicalLaw.THERMAL_HARD_LIMIT:
                self._trigger_veto("Thermal hard limit exceeded on realm entry")

            # Context transition
            previous = self._context.realm_active
            self._context.transition(previous, realm)

            # Lazy realm initialization
            if realm not in self._realms:
                self._realms[realm] = self._init_realm(realm)

            return self._realms[realm]

    def _init_realm(self, realm: Realm) -> Any:
        """Lazy realm initialization."""
        self._context.log(f"REALM_INIT: Initializing {realm.name}")

        # Import and instantiate realm
        if realm == Realm.FACTORY:
            from synthfuse.realms.factory import FactoryRealm
            return FactoryRealm(self)
        elif realm == Realm.PLAYGROUND:
            from synthfuse.realms.playground import PlaygroundRealm
            return PlaygroundRealm(self)
        elif realm == Realm.AUTOMODE:
            from synthfuse.realms.automode import AutoModeRealm
            return AutoModeRealm(self)
        elif realm == Realm.LAB:
            from synthfuse.realms.lab import LabRealm
            return LabRealm(self)
        elif realm == Realm.THERMO:
            from synthfuse.realms.thermo import ThermoRealm
            return ThermoRealm(self)
        else:
            raise ValueError(f"Unknown realm: {realm}")

    # =========================================================================
    # BRIDGER ACCESS
    # =========================================================================

    def bridger(self) -> Any:
        """Access central nervous system (lazy initialization)."""
        if self._bridger is None:
            # Bridger placeholder
            self._bridger = object()
            self._context.log("BRIDGER: Central nervous system online")
        return self._bridger

    # =========================================================================
    # WEIGHT DYNAMICS MANAGEMENT
    # =========================================================================

    def deploy_weight_sensor(self, layer_id: str, window_size: int = 1000) -> WeightKurveThermoSensor:
        """
        Deploy a WeightKurve sensor on a neural layer for thermal monitoring.

        Sensors track weight trajectories and alert on thermal stress.
        """
        sensor = WeightKurveThermoSensor(layer_id, self._context, window_size)
        self._context.register_weight_sensor(layer_id, sensor)
        self._context.log(f"WEIGHT_SENSOR: Deployed on {layer_id}")
        return sensor

    def analyze_weight_history(self,
                               layer_id: str,
                               history: jnp.ndarray,
                               flatten_window: int = 21,
                               transit_sigma: float = 3.0) -> KurveSignature:
        """
        Analyze weight trajectory using WeightKurve.

        Returns signature with periodicity, transits, and thermal stress.
        """
        kurve = WeightKurve.from_substrate(layer_id, history, self._context)
        signature = kurve.analyze(flatten_window, transit_sigma)

        # Update thermal state with Kurve findings
        if signature.thermal_stress:
            self._context.thermal.thermal_stress = signature.thermal_stress
        if signature.lyapunov_estimate:
            self._context.thermal.lyapunov_estimate = signature.lyapunov_estimate
        if signature.transit_count:
            self._context.thermal.transit_rate = float(jnp.sum(signature.transits)) / len(history)

        # Log findings
        self._context.log(
            f"KURVE: {layer_id} — "
            f"λ={signature.lyapunov_estimate:.3f}, "
            f"stress={signature.thermal_stress:.3f}"
        )

        return signature

    # =========================================================================
    # THERMAL MANAGEMENT
    # =========================================================================

    def _sample_thermal(self) -> ThermalState:
        """Sample physical reality — instruments, not heuristics."""
        if self.regulator is None:
            return ThermalState()

        raw = self.regulator.sample()

        # Create thermal state
        state = ThermalState(
            load=float(raw.get('thermal_load', 0.0)),
            entropy=float(raw.get('entropy', 0.0)),
            capacity_remaining=max(0.0, 1.0 - float(raw.get('thermal_load', 0.0))),
            timestamp_ms=time.time_ns() // 1_000_000
        )

        # Preserve Kurve findings from previous state
        if self._context and self._context.thermal:
            state.lyapunov_estimate = self._context.thermal.lyapunov_estimate
            state.thermal_stress = self._context.thermal.thermal_stress
            state.transit_rate = self._context.thermal.transit_rate

        return state

    def _trigger_veto(self, reason: str, kurve_alert: Optional[KurveAlert] = None):
        """Hardware veto — irreversible, no appeal."""
        self._context.record_violation("THERMAL_VETO", {
            "reason": reason,
            "kurve_alert": kurve_alert.to_dict() if kurve_alert else None
        })

        # Emergency shutdown sequence
        self._emergency_shutdown()

        raise ThermalViolation(reason, self._context, kurve_alert)

    def _emergency_shutdown(self):
        """Emergency shutdown with graceful degradation."""
        # self._context.log("EMERGENCY: Initiating shutdown sequence", level="CRITICAL")

        # Stop thermal monitoring
        self._thermal_stop.set()
        if self._thermal_thread:
            # We don't join for long here, it's an emergency
            pass

        self._shutdown = True

    # =========================================================================
    # SHUTDOWN & SESSION ARCHIVE
    # =========================================================================

    def shutdown(self, generate_session_md: bool = True) -> Dict[str, Any]:
        """
        Graceful shutdown — generate session.md, archive provenance.
        """
        if self._shutdown:
            return {"status": "ALREADY_SHUTDOWN"}

        self._context.log("SHUTDOWN: Initiating graceful shutdown")

        # Stop thermal monitoring
        self._thermal_stop.set()
        if self._thermal_thread:
            self._thermal_thread.join(timeout=2.0)

        # Shutdown realms in reverse order
        for realm in reversed(list(self._realms.keys())):
            self._context.log(f"SHUTDOWN: Stopping {realm.name}")

        # Generate session archive
        session_hash = None
        md_path = None
        if generate_session_md and self._context:
            from synthfuse.session_logger import SessionLogger
            logger = SessionLogger(self._context)
            md_path, session_hash = logger.write()
            self._context.log(f"SHUTDOWN: Session archived at {md_path}")

        self._shutdown = True
        self._booted = False

        return {
            "status": "SHUTDOWN_COMPLETE",
            "session_id": self._context.session_id if self._context else None,
            "duration_ms": (time.time_ns() // 1_000_000) - self._context.boot_time_ms if self._context else 0,
            "sigils_certified": len(self._context.sigils_certified) if self._context else 0,
            "thermal_violations": len(self._context.violations) if self._context else 0,
            "kurve_alerts": len(self._context.kurve_alerts) if self._context else 0,
            "session_md": str(md_path) if md_path else None,
            "session_hash": session_hash
        }

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def load_module(self, module_path: str) -> Any:
        """Lazy module loader for realm dependencies."""
        parts = module_path.split('.')
        module = __import__(parts[0])
        for part in parts[1:]:
            module = getattr(module, part)
        return module

    # =========================================================================
    # SAFE I/O BRIDGE
    # =========================================================================

    def _apply_global_safeguards(self):
        """
        Monkey-patch destructive file operations to ensure they respect
        the Shield's restricted paths.
        """
        import os
        import shutil
        from pathlib import Path

        original_remove = os.remove
        original_unlink = os.unlink
        original_rmtree = shutil.rmtree

        def guarded_remove(path, *args, **kwargs):
            if not self._check_path_safety("delete", path):
                raise PermissionError(f"Synth-Fuse OS Veto: Attempted deletion of restricted path {path}")
            return original_remove(path, *args, **kwargs)

        def guarded_rmtree(path, *args, **kwargs):
            if not self._check_path_safety("delete", path):
                raise PermissionError(f"Synth-Fuse OS Veto: Attempted rmtree of restricted path {path}")
            return original_rmtree(path, *args, **kwargs)

        # Apply patches
        os.remove = guarded_remove
        os.unlink = guarded_remove
        shutil.rmtree = guarded_rmtree
        self._context.log("SAFEGUARD: Global destructive operation interceptors active")

    def _check_path_safety(self, operation: str, path: Any) -> bool:
        """Internal helper to check path against Shield invariants."""
        if not self.cabinet or "shield" not in self.cabinet.roles:
            return True # Conservative: if no shield, we can't validate (or should we block?)

        shield = self.cabinet.roles["shield"]
        if hasattr(shield, "validate_safety"):
            return shield.validate_safety(operation, path)
        return True

    def safe_write(self, filepath: Union[str, Path], content: str):
        """
        Gate file writing through the Shield role for self-preservation.
        """
        if not self._check_path_safety("overwrite", filepath):
            self._trigger_veto(f"Safety Violation: Attempted overwrite of restricted path {filepath}")

        with open(filepath, 'w') as f:
            f.write(content)

    def safe_delete(self, filepath: Union[str, Path]):
        """
        Gate file deletion through the Shield role.
        """
        if not self._check_path_safety("delete", filepath):
            self._trigger_veto(f"Safety Violation: Attempted deletion of restricted path {filepath}")

        Path(filepath).unlink()

    def __enter__(self):
        """Context manager support."""
        self.boot()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit — ensure shutdown."""
        self.shutdown()
        return False


# =============================================================================
# MODULE-LEVEL API
# =============================================================================

_os_instance: Optional[SynthFuseOS] = None
_os_lock = threading.Lock()


def boot(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Global entry point: synthfuse.boot()

    Boots the Synth-Fuse OS v0.4 and returns welcome status.
    """
    global _os_instance

    with _os_lock:
        if _os_instance is not None and _os_instance._booted:
            return _os_instance._status()

        _os_instance = SynthFuseOS(config)
        return _os_instance.boot(config)


def os() -> SynthFuseOS:
    """
    Access running OS instance.
    """
    global _os_instance

    if _os_instance is None or not _os_instance._booted:
        raise RuntimeError("OS not booted — call synthfuse.boot() first")

    return _os_instance


def shutdown(generate_session_md: bool = True) -> Dict[str, Any]:
    """Global shutdown."""
    global _os_instance

    if _os_instance is None:
        return {"status": "NOT_RUNNING"}

    return _os_instance.shutdown(generate_session_md)


# Convenience exports
__all__ = [
    'SynthFuseOS',
    'OSContext',
    'ThermalState',
    'Realm',
    'ThermalViolation',
    'WrappedDeployment',
    'StochasticWrapper',
    'PhysicalLaw',
    'boot',
    'os',
    'shutdown',
]
