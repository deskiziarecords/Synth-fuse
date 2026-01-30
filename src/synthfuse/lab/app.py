import streamlit as st
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

st.set_page_config(
    page_title="Synth-Fuse Lab",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Synth-Fuse v0.2.0 - Unified Field Engineering")
st.markdown("### Interactive Lab Interface")

# Status section
st.header("Cabinet Status")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Status", "Offline", delta=None)
with col2:
    st.metric("Entropy", "0.000", delta=None)
with col3:
    st.metric("Thermal Load", "0.0%", delta=None)

# Sigil testing
st.header("Sigil Testing")
sigil_input = st.text_input("Enter Sigil", value="(I⊗Z)")
data_input = st.text_area("Input Data (JSON)", value='{"test": [1, 2, 3]}')

if st.button("Process Sigil"):
    try:
        st.info(f"Processing Sigil: {sigil_input}")
        st.json({"status": "Processing", "sigil": sigil_input})
    except Exception as e:
        st.error(f"Error: {e}")

# Cabinet controls
st.header("Cabinet Controls")
col1, col2 = st.columns(2)

with col1:
    if st.button("🚀 Initialize Cabinet", type="primary"):
        st.success("Cabinet initialization requested...")

with col2:
    if st.button("🛑 Emergency Shutdown"):
        st.warning("Emergency shutdown sequence initiated...")

# Logs
st.header("System Logs")
st.code("""
[INFO] Cabinet Orchestrator v0.2.0
[INFO] Python 3.10.0
[INFO] Ready to initialize...
""", language="log")

st.sidebar.title("Navigation")
st.sidebar.markdown("""
- **🏛️ Cabinet Dashboard**
- **🔧 Sigil Compiler**
- **📊 System Metrics**
- **⚙️ Settings**
""")

st.sidebar.info("""
Synth-Fuse v0.2.0
Unified Field Engineering
""")
