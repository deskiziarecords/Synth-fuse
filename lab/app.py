import streamlit as st
import asyncio
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import os
import sys

# Add src to path if needed
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from synthfuse.cabinet.cabinet_orchestrator import CabinetOrchestrator
except ImportError:
    # Fallback for local dev/testing
    st.error("Synth-Fuse package not found in path. Ensure you are running from the repo root.")
    st.stop()

# --- Page Config ---
st.set_page_config(
    page_title="Synth-Fuse Lab | Unified Field OS",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3e4150;
    }
    .sigil-card {
        background-color: #161b22;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #58a6ff;
        margin-bottom: 10px;
    }
    .terminal {
        font-family: 'Courier New', Courier, monospace;
        background-color: #000;
        color: #0f0;
        padding: 10px;
        border-radius: 5px;
        height: 300px;
        overflow-y: scroll;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State ---
if 'cabinet' not in st.session_state:
    st.session_state.cabinet = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'is_online' not in st.session_state:
    st.session_state.is_online = False

def log_event(message, type="info"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    emoji = "🔵" if type == "info" else "🟢" if type == "success" else "🔴"
    st.session_state.logs.append(f"[{timestamp}] {emoji} {message}")
    if len(st.session_state.logs) > 100:
        st.session_state.logs.pop(0)

# --- Sidebar ---
with st.sidebar:
    st.image("https://raw.githubusercontent.com/deskiziarecords/Synth-fuse/main/docs/assets/logo.png", width=200) # Placeholder or actual logo
    st.title("Synth-Fuse OS v0.5.0")
    st.subheader("RBC Circulatory Intelligence")

    st.divider()

    if not st.session_state.is_online:
        if st.button("🚀 Boot System", use_container_width=True):
            with st.spinner("Initializing Unified Field OS..."):
                cabinet = CabinetOrchestrator()
                success = asyncio.run(cabinet.initialize())
                if success:
                    st.session_state.cabinet = cabinet
                    st.session_state.is_online = True
                    log_event("Cabinet of Alchemists initialized", "success")
                    st.rerun()
                else:
                    st.error("Failed to initialize Cabinet")
    else:
        st.success("System Online")
        if st.button("🛑 Emergency Shutdown", use_container_width=True, type="primary"):
            asyncio.run(st.session_state.cabinet.emergency_shutdown())
            st.session_state.is_online = False
            st.session_state.cabinet = None
            log_event("System shutdown initiated", "info")
            st.rerun()

    st.divider()

    st.info("""
    **Current Realm**: Lab
    **Architecture**: Unified Field
    **Status**: Adiabatic Mode
    """)

# --- Main Interface ---
st.title("🏛️ Cabinet of Alchemists")
st.write("Governing Unified Field Engineering through the 42-Sigil Registry.")

if not st.session_state.is_online:
    st.warning("Please boot the system from the sidebar to begin.")

    # Showcase Sigils even when offline
    st.subheader("Available Sigil Registry Examples")
    cols = st.columns(3)
    with cols[0]:
        st.info("**Factory**: `((L⊗K)⋈(D⊗M))⊕(C⊗P)`")
    with cols[1]:
        st.info("**Thermo**: `((I⊗Z)⊗S)⊙(F⊕R)`")
    with cols[2]:
        st.info("**Economic**: `(𝕄 ⊗ Δ$) ⊕ (𝕀𝕞𝕞 ⊙ §)`")
else:
    # Online Dashboard
    status = st.session_state.cabinet.get_status()

    # Top Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Status", status['status'].upper())
    m2.metric("Sigils Processed", status['processed_count'])
    m3.metric("Avg Entropy", f"{status['average_entropy']:.3f}")
    m4.metric("Thermal Load", f"{status['average_thermal_load']:.1%}")

    tab1, tab2, tab3 = st.tabs(["🔮 Sigil Processor", "📊 Real-time Telemetry", "📚 Registry & Logs"])

    with tab1:
        st.header("Process Computational Sigil")

        col_inp, col_res = st.columns([1, 1])

        with col_inp:
            sigil_input = st.text_input("Enter Sigil Expression", "(I⊗Z)⊗S")
            data_input = st.text_area("Input Data (JSON)", value='{"input": [1.0, 2.0, 3.0], "context": "lab_test"}', height=100)

            if st.button("✨ Execute Sigil", use_container_width=True):
                try:
                    data = json.loads(data_input)
                    with st.spinner(f"Processing {sigil_input}..."):
                        result = asyncio.run(st.session_state.cabinet.process_sigil(sigil_input, data))
                        st.session_state.history.append(result)
                        log_event(f"Processed {sigil_input}", "success")
                except Exception as e:
                    st.error(f"Execution Error: {e}")
                    log_event(f"Error processing {sigil_input}: {e}", "error")

        with col_res:
            if st.session_state.history:
                latest = st.session_state.history[-1]
                if "error" in latest:
                    st.error(f"Process Failed: {latest['error']}")
                else:
                    st.success(f"Execution Complete: {latest['process_id']}")
                    st.json(latest)
            else:
                st.info("No sigils processed in this session.")

    with tab2:
        st.header("Manifold Telemetry")

        if len(st.session_state.history) > 1:
            df = pd.DataFrame([
                {
                    "step": i,
                    "entropy": h.get("entropy", 0),
                    "thermal": h.get("thermal_load", 0),
                    "duration": h.get("duration_seconds", 0)
                }
                for i, h in enumerate(st.session_state.history)
            ])

            c1, c2 = st.columns(2)

            with c1:
                fig_e = px.line(df, x="step", y="entropy", title="Entropy Trajectory", template="plotly_dark")
                fig_e.update_traces(line_color='#00d1ff')
                st.plotly_chart(fig_e, use_container_width=True)

            with c2:
                fig_t = px.area(df, x="step", y="thermal", title="Thermal Load Over Time", template="plotly_dark")
                fig_t.update_traces(line_color='#ff4b4b')
                st.plotly_chart(fig_t, use_container_width=True)

            # Mock Manifold Visualization
            st.subheader("WeightKurve Manifold Projection")
            z = np.random.randn(10, 10)
            fig_m = go.Figure(data=[go.Surface(z=z, colorscale='Viridis')])
            fig_m.update_layout(title='Neural Weight Surface Topology', autosize=False,
                              width=800, height=500, margin=dict(l=65, r=50, b=65, t=90),
                              template="plotly_dark")
            st.plotly_chart(fig_m, use_container_width=True)
        else:
            st.info("Insufficient data for telemetry. Process more sigils to see live charts.")

    with tab3:
        st.header("System Logs")
        log_text = "\n".join(reversed(st.session_state.logs))
        st.markdown(f'<div class="terminal">{log_text}</div>', unsafe_allow_html=True)

        st.divider()
        st.header("42-Sigil Registry Explorer")
        st.dataframe(pd.DataFrame([
            {"Domain": "Core", "Name": "Factory", "Sigil": "((L⊗K)⋈(D⊗M))⊕(C⊗P)"},
            {"Domain": "Core", "Name": "Thermo", "Sigil": "((I⊗Z)⊗S)⊙(F⊕R)"},
            {"Domain": "Economic", "Name": "Sovereign", "Sigil": "(𝕄 ⊗ Δ$) ⊕ (𝕀𝕞𝕞 ⊙ §)"},
            {"Domain": "Security", "Name": "Bastion", "Sigil": "(V⊗Z)⊗(R⊕S)"},
        ]))

st.divider()
st.caption("Synth-Fuse Unified Field OS v0.5.0 | Calibrated by Google's Jules | OpenGate Integrity")
