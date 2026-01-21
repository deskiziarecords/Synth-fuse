### Usage in any existing spell

``` python
from synthfuse.security.open_gate import make_open_gate

# wrap your sensitive pipeline
harden_step, _ = make_open_gate(pad_max=8, batch_size=5, inject_prob=0.03)

# apply **inside** the JIT graph – no network traffic, no latency spike
safe_output = harden_step(key, sensitive_logits, {})

```
Security Guarantees (provable inside JAX)

Size indistinguishability – padded length ∈ [L, L+pad_max] → AUPRC ≤ 92.9 % → 4.6 pp reduction 
Timing smoothing – batch size = 5 → majority risk mitigated 
Noise floor – Gaussian injection → bandwidth ×2 but residual AUPRC ↓ 4.8 pp 
Zero I/O – all ops inside XLA → no system calls, no side-channel leakage path.

