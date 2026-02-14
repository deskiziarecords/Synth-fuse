"""
WeightKurve - Stellar Dynamics for Neural Weights
Lab Instrument v0.4.0

Analyzes weight trajectories as stellar light curves.
Flux = weight/gradient magnitude. Time = step/epoch.
Detects: oscillations (periodicity), transits (OOD), drift (thermal stress).
"""

import jax
import jax.numpy as jnp
from typing import Optional, Tuple, List
from dataclasses import dataclass

from synthfuse.lab.instruments.base import LabInstrument


@dataclass
class KurveSignature:
    """Extracted features from weight trajectory."""
    periodicity: Optional[Tuple[float, float]]  # (frequency, power)
    transits: jnp.ndarray  # Boolean mask of OOD events
    drift_rate: float  # Low-frequency trend magnitude
    thermal_stress: float  # Derived: high drift + high freq = stress
    lyapunov_estimate: float  # Chaos indicator from spectrum


class WeightKurve(LabInstrument):
    """
    Lab instrument: Analyzes neural weight trajectories as stellar light curves.
    
    Sigil: (Z⊙S) - Zero-point observation of Swarm dynamics
    """
    
    INSTRUMENT_ID = "weight_kurve_v1"
    THERMAL_COST = 0.05  # Per-analysis thermal load
    
    def __init__(self, 
                 time: jnp.ndarray, 
                 flux: jnp.ndarray, 
                 label: str = "Substrate Layer",
                 context=None):
        self.time = time
        self.flux = flux
        self.label = label
        self.context = context  # OS context for thermal logging
        
        # Validate JAX arrays
        assert isinstance(time, jnp.ndarray), "Time must be JAX array"
        assert isinstance(flux, jnp.ndarray), "Flux must be JAX array"
        
    @classmethod
    def from_substrate(cls, 
                       tool_id: str, 
                       history_buffer: jnp.ndarray,
                       context=None) -> 'WeightKurve':
        """
        Factory: Convert LazyTensor history into Kurve.
        
        Called by Neural Substrate when weight history exceeds
        thermal gradient threshold.
        """
        time = jnp.arange(len(history_buffer))
        return cls(
            time=time, 
            flux=history_buffer, 
            label=f"Layer::{tool_id}",
            context=context
        )
    
    @classmethod
    def from_training_run(cls,
                          weights_history: List[jnp.ndarray],
                          step_indices: jnp.ndarray,
                          layer_id: str) -> 'WeightKurve':
        """
        Factory: Full training run analysis.
        """
        # Concatenate or select representative layer
        flux = jnp.concatenate([w.flatten()[:1000] for w in weights_history[-5:]])
        time = jnp.arange(len(flux))
        return cls(time, flux, f"Training::{layer_id}")
    
    def flatten(self, window_size: int = 21) -> 'WeightKurve':
        """
        Remove low-frequency 'background' drift (global LR decay).
        Returns detrended Kurve for periodicity analysis.
        """
        # Savitzky-Golay style smoothing for trend
        kernel = jnp.ones(window_size) / window_size
        trend = jnp.convolve(self.flux, kernel, mode='same')
        
        # Avoid division by zero
        trend = jnp.where(trend == 0, 1e-8, trend)
        
        flattened = self.flux / trend
        
        if self.context:
            self.context.log(f"KURVE: Flattened {self.label} (window={window_size})")
        
        return WeightKurve(
            self.time, 
            flattened, 
            f"{self.label} (Flattened)",
            self.context
        )
    
    def to_periodogram(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Detect cyclic weight patterns via FFT.
        
        Returns:
            frequencies: FFT frequency bins
            power_spectrum: PSD at each frequency
        """
        # Real FFT for efficiency
        fourier = jnp.fft.rfft(self.flux)
        power = jnp.abs(fourier) ** 2
        freqs = jnp.fft.rfftfreq(len(self.flux), d=1.0)
        
        # Exclude DC component (index 0)
        power = power.at[0].set(0)
        
        if self.context:
            dominant_freq = freqs[jnp.argmax(power)]
            self.context.log(f"KURVE: Dominant frequency {dominant_freq:.4f} in {self.label}")
        
        return freqs, power
    
    def detect_transit(self, sigma: float = 3.0) -> jnp.ndarray:
        """
        Find 'transits'—sudden drops in weight saliency.
        
        Transits indicate:
        - OOD data points
        - Gradient starvation
        - Catastrophic forgetting events
        
        Returns boolean mask where transits occur.
        """
        mean_flux = jnp.mean(self.flux)
        std_flux = jnp.std(self.flux)
        threshold = mean_flux - (sigma * std_flux)
        
        transit_mask = self.flux < threshold
        
        transit_count = jnp.sum(transit_mask)
        if self.context:
            self.context.log(f"KURVE: Detected {transit_count} transits in {self.label}")
        
        return transit_mask
    
    def detect_oscillation(self, 
                          min_period: int = 5,
                          max_period: int = 100) -> Optional[Tuple[float, float]]:
        """
        Detect periodic oscillations in weight updates.
        
        Oscillations indicate:
        - Learning rate too high (instability)
        - Sharp loss landscape (poor conditioning)
        - Batch noise resonance
        
        Returns (period, amplitude) or None.
        """
        freqs, power = self.to_periodogram()
        
        # Find dominant non-zero frequency
        valid_idx = jnp.where((freqs > 0) & (freqs < 1.0/min_period))[0]
        if len(valid_idx) == 0:
            return None
        
        dominant_idx = valid_idx[jnp.argmax(power[valid_idx])]
        dominant_freq = freqs[dominant_idx]
        dominant_power = power[dominant_idx]
        
        period = 1.0 / dominant_freq
        
        if min_period <= period <= max_period:
            return (float(period), float(dominant_power))
        
        return None
    
    def compute_lyapunov_estimate(self) -> float:
        """
        Estimate maximum Lyapunov exponent from power spectrum.
        
        High-frequency power dominance suggests chaos/instability.
        """
        freqs, power = self.to_periodogram()
        
        # Weighted average frequency (spectral centroid)
        spectral_centroid = jnp.sum(freqs * power) / jnp.sum(power)
        
        # Normalize to [0, 1] as chaos indicator
        max_freq = jnp.max(freqs)
        lyap_estimate = float(spectral_centroid / max_freq)
        
        return lyap_estimate
    
    def analyze(self, 
                flatten_window: int = 21,
                transit_sigma: float = 3.0) -> KurveSignature:
        """
        Full Lab analysis—signature extraction for Cabinet certification.
        
        Thermal cost: 0.05 per analysis. Logged to OS context.
        """
        if self.context:
            self.context.thermal.load += self.THERMAL_COST
        
        # Pipeline: Flatten → Periodogram → Detect
        flat = self.flatten(flatten_window)
        
        periodicity = flat.detect_oscillation()
        transits = self.detect_transit(transit_sigma)
        lyapunov = flat.compute_lyapunov_estimate()
        
        # Drift rate from original (non-flattened)
        drift = jnp.std(jnp.diff(self.flux))
        
        # Thermal stress: composite metric
        stress = drift * (1.0 + lyapunov)
        
        signature = KurveSignature(
            periodicity=periodicity,
            transits=transits,
            drift_rate=float(drift),
            thermal_stress=float(stress),
            lyapunov_estimate=lyapunov
        )
        
        if self.context:
            self.context.log(
                f"KURVE: Analysis complete for {self.label} "
                f"(stress={stress:.3f}, lyapunov={lyapunov:.3f})"
            )
        
        return signature
    
    def to_thermo_alert(self, threshold_stress: float = 0.5) -> Optional[dict]:
        """
        Convert analysis to Thermo realm alert if stress exceeds bounds.
        
        Called by Lab realm when thermal_stress > threshold.
        """
        sig = self.analyze()
        
        if sig.thermal_stress > threshold_stress:
            return {
                'instrument': self.INSTRUMENT_ID,
                'layer': self.label,
                'thermal_stress': sig.thermal_stress,
                'lyapunov': sig.lyapunov_estimate,
                'transit_count': int(jnp.sum(sig.transits)),
                'recommendation': 'REDUCE_LR' if sig.periodicity else 'CHECK_DATA',
                'sigil': '(Z⊙S)'  # Observation of Swarm
            }
        
        return None


# =============================================================================
# THERMO REALM INTEGRATION
# =============================================================================

class WeightKurveThermoSensor:
    """
    Thermo realm wrapper: Continuous weight monitoring.
    
    Deployed by Neural Substrate when layer thermal_load > 0.6.
    """
    
    def __init__(self, layer_id: str, os_context):
        self.layer_id = layer_id
        self.context = os_context
        self.history = []
        self.kurve = None
        
    def sample(self, weights: jnp.ndarray):
        """
        Periodic sample from training loop.
        """
        self.history.append(weights)
        
        # Maintain rolling window
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
        
        # Analyze every 100 steps
        if len(self.history) % 100 == 0:
            buffer = jnp.concatenate([w.flatten()[:100] for w in self.history[-10:]])
            self.kurve = WeightKurve.from_substrate(
                self.layer_id, 
                buffer,
                self.context
            )
            
            alert = self.kurve.to_thermo_alert(threshold_stress=0.5)
            if alert:
                # Trigger Thermo realm deliberation
                self.context.log(f"THERMO_SENSOR: Alert from {self.layer_id}")
                return alert
        
        return None
