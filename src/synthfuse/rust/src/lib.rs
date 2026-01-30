//! Synth-Fuse Security Core - OpenGate Implementation
//! Nanosecond-speed safety bounds and cryptographic verification

use std::sync::Arc;
use ring::signature;
use sha3::{Sha3_256, Digest};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SecurityError {
    #[error("Lyapunov bound violation: {0}")]
    LyapunovViolation(String),
    
    #[error("Cryptographic verification failed")]
    CryptoVerificationFailed,
    
    #[error("Entropy limit exceeded")]
    EntropyLimitExceeded,
    
    #[error("Resource budget overflow")]
    ResourceBudgetOverflow,
}

/// OpenGate Safety Certificate (512 bytes exactly)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SafetyCertificate {
    pub budget_deltas: [f64; 64],      // Bytes 0-511: ΔB matrix
    pub leakage_hashes: [u8; 64],      // Bytes 512-575: SHA3-512 of hardware tests
    pub monitors_bitmap: [u8; 64],     // Bytes 576-639: 512-bit enable/disable map
    pub signature: [u8; 64],           // Bytes 640-703: Ed25519 signature
}

/// Lyapunov safety bound checker
pub struct LyapunovEnforcer {
    stability_margin: f64,
    max_entropy: f64,
}

impl LyapunovEnforcer {
    pub fn new(stability_margin: f64, max_entropy: f64) -> Self {
        Self {
            stability_margin,
            max_entropy,
        }
    }
    
    /// Check if state vector satisfies Lyapunov stability
    pub fn check_stability(&self, state_vector: &[f64]) -> Result<(), SecurityError> {
        let norm = self.compute_norm(state_vector);
        let entropy = self.compute_entropy(state_vector);
        
        if norm > self.stability_margin {
            return Err(SecurityError::LyapunovViolation(
                format!("Norm {} exceeds margin {}", norm, self.stability_margin)
            ));
        }
        
        if entropy > self.max_entropy {
            return Err(SecurityError::EntropyLimitExceeded);
        }
        
        Ok(())
    }
    
    fn compute_norm(&self, vector: &[f64]) -> f64 {
        vector.iter().map(|x| x * x).sum::<f64>().sqrt()
    }
    
    fn compute_entropy(&self, vector: &[f64]) -> f64 {
        let sum: f64 = vector.iter().map(|x| x.abs()).sum();
        if sum == 0.0 {
            return 0.0;
        }
        let probs: Vec<f64> = vector.iter().map(|x| x.abs() / sum).collect();
        -probs.iter().filter(|&&p| p > 0.0).map(|p| p * p.log2()).sum::<f64>()
    }
}

/// Cryptographic safety certificate validator
pub struct CertificateValidator {
    public_key: [u8; 32],
}

impl CertificateValidator {
    pub fn new(public_key: [u8; 32]) -> Self {
        Self { public_key }
    }
    
    /// Validate a 512-byte safety certificate
    pub fn validate(&self, cert: &SafetyCertificate) -> Result<(), SecurityError> {
        // Verify Ed25519 signature
        let message = self.certificate_message(cert);
        let sig = signature::Ed25519Signature::from_slice(&cert.signature[..64])
            .map_err(|_| SecurityError::CryptoVerificationFailed)?;
        
        let pk = signature::UnparsedPublicKey::new(
            &signature::ED25519,
            &self.public_key
        );
        
        pk.verify(&message, &sig.as_ref())
            .map_err(|_| SecurityError::CryptoVerificationFailed)?;
        
        Ok(())
    }
    
    fn certificate_message(&self, cert: &SafetyCertificate) -> Vec<u8> {
        let mut hasher = Sha3_256::new();
        hasher.update(&cert.budget_deltas);
        hasher.update(&cert.leakage_hashes);
        hasher.update(&cert.monitors_bitmap);
        hasher.finalize().to_vec()
    }
}

// Python bindings via PyO3
#[cfg(feature = "python")]
mod python {
    use pyo3::prelude::*;
    use pyo3::types::PyBytes;
    use super::*;
    
    #[pyclass]
    struct PyLyapunovEnforcer {
        inner: LyapunovEnforcer,
    }
    
    #[pymethods]
    impl PyLyapunovEnforcer {
        #[new]
        fn new(stability_margin: f64, max_entropy: f64) -> Self {
            Self {
                inner: LyapunovEnforcer::new(stability_margin, max_entropy),
            }
        }
        
        fn check_stability(&self, state_vector: Vec<f64>) -> PyResult<()> {
            self.inner.check_stability(&state_vector)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
        }
    }
    
    #[pyfunction]
    fn validate_certificate(py: Python, cert_bytes: &PyBytes) -> PyResult<bool> {
        if cert_bytes.len() != 512 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Certificate must be exactly 512 bytes"
            ));
        }
        
        // In production, would use actual public key
        let dummy_key = [0u8; 32];
        let validator = CertificateValidator::new(dummy_key);
        
        // Parse certificate (simplified)
        let cert_ptr = cert_bytes.as_ptr();
        let cert = unsafe {
            std::ptr::read(cert_ptr as *const SafetyCertificate)
        };
        
        match validator.validate(&cert) {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }
    
    #[pymodule]
    fn synthfuse_security_core(_py: Python, m: &PyModule) -> PyResult<()> {
        m.add_class::<PyLyapunovEnforcer>()?;
        m.add_function(wrap_pyfunction!(validate_certificate, m)?)?;
        Ok(())
    }
}
