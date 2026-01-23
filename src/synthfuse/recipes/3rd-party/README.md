

## Synth-Fuse 3rd-Party Recipes

This folder contains drop-in Synth-Fuse recipes for third-party algorithms, each fused into a single JIT-compiled spell. Below is a quick reference for each recipe, including the algorithm name, Colab one-liner, and benchmark command.

---

## **1. Knowledge3D**
**Algorithm:** 3-D Knowledge Graph Reasoning (RotatE-style in 3-D)
**Colab One-Liner:**
```python
# !pip install synthfuse
from synthfuse.recipes import knowledge3d
import jax

step, state = knowledge3d.make_knowledge3d(n_entities=1000, dim=64, lr=0.01, temp=0.7)
key = jax.random.PRNGKey(0)
for i in range(100):
    key, sub = jax.random.split(key)
    state = step(sub, state)
print("Final avg entity norm:", float(jnp.mean(jnp.linalg.norm(state.entity_embed, axis=1))))
```
**Benchmark:**
```bash
uv run sfbench "(𝕂𝟛𝔻 ⊗ 𝔾ℝ𝔸𝔻 ⊗ 𝕊𝟛𝔻)(lr=0.01, temp=0.7, smooth_sigma=1.0, rank=32)" --bench knowledge3d-link-pred --n_entities=10000
```

---
## **2. HFT-Fusion**
**Algorithm:** High-Frequency Trading (tick data, order-book, trade signal)
**Colab One-Liner:**
```python
# !pip install synthfuse ml-hft juliacall
from synthfuse.recipes import hft
import jax
import yfinance as yf

ticks = yf.download("AAPL", start="2020-01-01", end="2020-01-02", interval="1m")["Close"].values
ticks_jax = jnp.array(ticks).reshape(-1, 1)
step, state = hft.make_hft(ticks_jax, window=100, threshold=0.01, latency=1e-6, levels=10, spread=0.01, signal="ma_cross", confidence=0.95)
key = jax.random.PRNGKey(42)
for i in range(5):
    key, sub = jax.random.split(key)
    state = step(sub, state)
print("Final trade signal:", float(state["trade"]))
```
**Benchmark:**
```bash
uv run sfbench "(𝙷𝙵𝚃 ⊗ 𝚃𝙸𝙲𝙺 ⊗ 𝙱𝙾𝙾𝙺)(window=100, threshold=0.01, latency=1e-6, levels=10, spread=0.01, signal=ma_cross, confidence=0.95)" --bench aapl-ticks --interval=1m --start=2020-01-01 --end=2020-01-02
```

---
---

## **3. Linfa-Fusion**
**Algorithm:** Classical ML (SVM, k-means, GMM) via Rust-Linfa
**Colab One-Liner:**
```python
# !pip install synthfuse linfa-py
from synthfuse.recipes import linfa
import jax

X = jax.random.normal(jax.PRNGKey(0), (1000, 32))
y = jax.random.randint(jax.PRNGKey(1), (1000,), 0, 2)
step, state = linfa.make_linfa(X, y, kernel="rbf", k=5, n_comp=3)
key = jax.random.PRNGKey(42)
for i in range(10):
    key, sub = jax.random.split(key)
    state = step(sub, state)
print("Final SVM support vectors:", int(jnp.sum(state["support_mask"])))
```
**Benchmark:**
```bash
uv run sfbench "(𝕃𝕀ℕ𝔽𝔸 ⊗ 𝕂𝕄𝔼𝔸ℕ𝕊 ⊗ 𝔾𝔸𝕌𝕊𝕊)(kernel=rbf, k=5, n_comp=3)" --bench iris --n=150
```

---

## **4. MLJ-Fusion**
**Algorithm:** Julia MLJ (model, tuning, validation)
**Colab One-Liner:**
```python
# !pip install synthfuse juliacall mlj python-call
from synthfuse.recipes import mlj
import jax

X = jax.random.normal(jax.PRNGKey(0), (150, 4))
y = jax.random.randint(jax.PRNGKey(1), (150,), 0, 3)
step, state = mlj.make_mlj(X, y, model="rf", tune="grid", folds=5)
key = jax.random.PRNGKey(42)
for i in range(5):
    key, sub = jax.random.split(key)
    state = step(sub, state)
print("CV mean RMS:", float(state["cv_mean"]))
```
**Benchmark:**
```bash
uv run sfbench "(𝕄𝕃𝕁 ⊗ 𝕋𝕌ℕ𝔼 ⊗ 𝕍𝔸𝕃)(model=rf, tune=grid, folds=5)" --bench iris --n=150
```

---

## **5. HF-Fusion**
**Algorithm:** Hugging-Face Transformers (BERT, BPE, MLM)
**Colab One-Liner:**
```python
# !pip install synthfuse transformers torch jax[dlpack]
from synthfuse.recipes import hf
import jax

texts = ["Hello world", "Synth-Fuse rocks"]
step, state = hf.make_hf(texts, model="bert-base-uncased", max_len=128, temp=0.1)
key = jax.random.PRNGKey(42)
for i in range(5):
    key, sub = jax.random.split(key)
    state = step(sub, state)
print("Final MLM loss:", float(state["loss"]))
```
**Benchmark:**
```bash
uv run sfbench "(𝕋ℝ𝔸ℕ𝕊 ⊗ 𝔹ℙ𝔼 ⊗ 𝕄𝕃𝕄)(model=bert-base, max_len=128, temp=0.1)" --bench glue-mrpc --batch=32
```

---

## **6. TFQ-Fusion**
**Algorithm:** TensorFlow Quantum (quantum circuits, expectation, measurement)
**Colab One-Liner:**
```python
# !pip install synthfuse tensorflow-quantum jax[dlpack]
from synthfuse.recipes import tfq
import jax

step, state = tfq.make_tfq(n_qubits=4, depth=3, shots=1024, hamiltonian="0.5 * Z(0) * Z(1)", basis="Z")
key = jax.random.PRNGKey(42)
for i in range(5):
    key, sub = jax.random.split(key)
    state = step(sub, state)
print("Final expectation:", float(state["expectation"]))
```
**Benchmark:**
```bash
uv run sfbench "(ℚ𝕌𝔸ℕ𝕋𝕌𝙼 ⊗ ℚℂ𝙸ℝ𝕔 ⊗ ℚ𝙼𝙴𝙰𝚂)(n_qubits=4, depth=3, shots=1024)" --bench vqe-energy --n_qubits=4
```

---

## **7. PyBroker-Fusion**
**Algorithm:** Algorithmic Trading (strategy, backtest, return)
**Colab One-Liner:**
```python
# !pip install synthfuse pybroker
from synthfuse.recipes import pybroker
import jax
import yfinance as yf

prices = yf.download("AAPL", start="2020-01-01", end="2022-01-01")["Close"].values
prices_jax = jnp.array(prices)
step, state = pybroker.make_pybroker(prices_jax, strategy="ma_cross", lookback=20, risk_free=0.02)
key = jax.random.PRNGKey(42)
for i in range(5):
    key, sub = jax.random.split(key)
    state = step(sub, state)
print("Final Sharpe:", float(state["sharpe"]))
```
**Benchmark:**
```bash
uv run sfbench "(ℙ𝔹ℝ𝕆𝕂𝔼ℝ ⊗ 𝕊𝕋ℝ𝔸𝕋 ⊗ ℝ𝔼𝕋𝕌ℝℕ)(strategy=ma_cross, lookback=20, risk_free=0.02)" --bench aapl-daily --start=2020-01-01 --end=2022-01-01
```

---

## **8. Jubatus-Fusion**
**Algorithm:** Online ML (SVM, k-means, GMM)
**Colab One-Liner:**
```python
# !pip install synthfuse juliacall jubatus
from synthfuse.recipes import jubatus
import jax
import yfinance as yf

prices = yf.download("AAPL", start="2020-01-01", end="2022-01-01")["Close"].values
prices_jax = jnp.array(prices).reshape(-1, 1)
step, state = jubatus.make_jubatus(prices_jax, k=5, comp=3, C=1.0, kernel="rbf")
key = jax.random.PRNGKey(42)
for i in range(5):
    key, sub = jax.random.split(key)
    state = step(sub, state)
print("Final k-means inertia:", float(jnp.var(prices_jax - state["centroids"][state["labels"]])))
```
**Benchmark:**
```bash
uv run sfbench "(𝕁𝕌𝔹𝔸𝚃𝚄𝚂 ⊗ 𝙺𝙼𝙴𝙰𝙽𝚂 ⊗ 𝙶𝙰𝚄𝚂𝚂 ⊗ 𝚂𝚅𝙼)(model=fv_converter, k=5, comp=3, C=1.0, kernel=rbf)" --bench aapl-stream --start=2020-01-01 --end=2022-01-01
```

---

## **9. TFF-Fusion**
**Algorithm:** TensorFlow Federated (federated compute, aggregation, client update)
**Colab One-Liner:**
```python
# !pip install synthfuse tensorflow-federated juliacall
from synthfuse.recipes import tff
import jax

step, state = tff.make_tff(n_clients=10, rounds=50, lr=0.1, strategy="mean", agg="mean")
key = jax.random.PRNGKey(42)
for i in range(50):
    key, sub = jax.random.split(key)
    state = step(sub, state)
print("Final global model norm:", float(jnp.linalg.norm(state["global_model"])))
```
**Benchmark:**
```bash
uv run sfbench "(𝔽𝔼𝔻𝔼ℝ𝔸𝕥𝔼𝔻 ⊗ 𝔄𝔾𝔾ℝ𝔼𝔾𝔸𝕥𝔼 ⊗ 𝕌ℙ𝔻𝔸𝕥𝔼)(n_clients=10, rounds=50, lr=0.1)" --bench federated-mnist --n_clients=10
```

---

## **10. Rasa-Fusion**
**Algorithm:** Conversational AI (intent, entity, dialogue policy)
**Colab One-Liner:**
```python
# !pip install synthfuse rasa juliacall
from synthfuse.recipes import rasa
import jax

texts = ["Hello", "Book a table at 8pm", "Cancel my reservation"]
step, state = rasa.make_rasa(texts, intent="affirm", entity="time", policy="MemoizationPolicy")
key = jax.random.PRNGKey(42)
for i in range(5):
    key, sub = jax.random.split(key)
    state = step(sub, state)
print("Final intent confidence:", float(state["intent_probs"][0]))
```
**Benchmark:**
```bash
uv run sfbench "(𝚁𝙰𝚂𝙰 ⊗ ℕ𝙻𝚄 ⊗ 𝙳𝙸𝙰𝙻𝙾𝙶)(intent=affirm, entity=time, policy=MemoizationPolicy)" --bench rasa-dialog --texts="hello,book a table,cancel"
```

---

## **11. FATE-Fusion**
**Algorithm:** Federated AI (federated learning, secure aggregation, homomorphic encryption)
**Colab One-Liner:**
```python
# !pip install synthfuse fate-client juliacall
from synthfuse.recipes import fate
import jax

step, state = fate.make_fate(n_clients=10, rounds=50, lr=0.1, secure=True, encrypt=True)
key = jax.random.PRNGKey(42)
for i in range(50):
    key, sub = jax.random.split(key)
    state = step(sub, state)
print("Final global model norm:", float(jnp.linalg.norm(state["global_model"])))
```
**Benchmark:**
```bash
uv run sfbench "(𝔽𝔸𝕋𝔼 ⊗ 𝔽𝔼𝔻𝔼ℝ𝔸𝕥𝔼𝔻 ⊗ 𝕊𝔼ℂ𝕌ℝ𝔼)(n_clients=10, rounds=50, lr=0.1, secure=True, encrypt=True)" --bench federated-mnist-secure --n_clients=10
```

---

## **12. LightGBM-Fusion**
**Algorithm:** Gradient Boosting (tree split, gradient boost, leaf update)
**Colab One-Liner:**
```python
# !pip install synthfuse lightgbm juliacall
from synthfuse.recipes import lightgbm
import jax
import sklearn.datasets

X, y = sklearn.datasets.make_classification(n_samples=1000, n_features=20, n_classes=2)
X_jax = jnp.array(X)
y_jax = jnp.array(y)
step, state = lightgbm.make_lightgbm(X_jax, y_jax, n_trees=100, lr=0.1, max_depth=6)
key = jax.random.PRNGKey(42)
for i in range(100):
    key, sub = jax.random.split(key)
    state = step(sub, state)
print("Final AUC:", float(jnp.mean(state["preds"] > 0.5)))
```
**Benchmark:**
```bash
uv run sfbench "(𝕃𝙸𝙶𝙷𝚃 ⊗ 𝙶𝙱𝙳𝚃 ⊗ 𝙱𝙾𝙾𝚂𝚃)(n_trees=100, lr=0.1, max_depth=6, lambda_l2=1.0)" --bench higgs --n=10000
```

---

## **13. MTK-Fusion**
**Algorithm:** Symbolic-Numeric PDEs (ModelingToolkit.jl)
**Colab One-Liner:**
```python
# !pip install synthfuse modelingtoolkit juliacall
from synthfuse.recipes import mtk
import jax

step, state = mtk.make_mtk(eq="heat", dim=2, order=4, dt=0.01, tspan=(0.0, 1.0), method="Tsit5")
key = jax.random.PRNGKey(42)
for i in range(100):
    key, sub = jax.random.split(key)
    state = step(sub, state)
print("Final solution max:", float(jnp.max(state["u"])))
```
**Benchmark:**
```bash
uv run sfbench "(𝕄𝕋𝕂 ⊗ 𝔻𝔼ℝ𝕀𝕍 ⊗ 𝕊𝕆𝕃𝕍𝔼)(eq=heat, dim=2, order=4, dt=0.01, tspan=(0.0, 1.0), method=Tsit5)" --bench heat-pde --grid=64x64
```

---

## **14. BRPC-Fusion**
**Algorithm:** High-Performance RPC (RPC, streaming, zero-copy transport)
**Colab One-Liner:**
```python
# !pip install synthfuse brpc juliacall
from synthfuse.recipes import brpc
import jax

step, state = brpc.make_brpc(host="127.0.0.1", port=8080, streaming=True, zero_copy=True)
key = jax.random.PRNGKey(42)
for i in range(5):
    key, sub = jax.random.split(key)
    state = step(sub, state)
print("Final zero-copy buffer norm:", float(jnp.linalg.norm(state["zc_buffer"])))
```
**Benchmark:**
```bash
uv run sfbench "(𝙱𝚁𝙿𝙲 ⊗ 𝚁𝙿𝙲 ⊗ 𝚂𝚃𝚁𝙴𝙰𝙼)(host=127.0.0.1, port=8080, streaming=True, zero_copy=True)" --bench rpc-echo --payload=1MB
```

---

## **15. Topological-Fusion**
**Algorithm:** Topological Data Analysis (persistent homology, betti curves, persistence landscapes)
**Colab One-Liner:**
```python
# !pip install synthfuse pytorch-topological juliacall
from synthfuse.recipes import topological
import jax

X = jax.random.normal(jax.random.PRNGKey(0), (100, 3))
step, state = topological.make_topological(X, dim=2, max_dim=3, coeff=2, landscape_size=10)
key = jax.random.PRNGKey(42)
for i in range(5):
    key, sub = jax.random.split(key)
    state = step(sub, state)
print("Final persistence landscape norm:", float(jnp.linalg.norm(state["landscapes"])))
```
**Benchmark:**
```bash
uv run sfbench "(𝕋𝕆ℙ𝕆 ⊗ ℍ𝙾𝙼𝙾𝕃𝙾𝙶𝕐 ⊗ ℙ𝙴ℝ𝕊𝕀𝕊𝕋𝔼ℕℂ𝔼)(dim=2, max_dim=3, coeff=2, landscape_size=10)" --bench topological-circle --n=1000
```

---




### **Usage Notes**
- Each recipe is a drop-in replacement for the original algorithm, compiled into a single JIT kernel.
- Benchmarks are runnable via `uv run sfbench` with the provided spell and parameters.
- For Colab, install dependencies as shown in the one-liners.
