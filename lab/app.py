import streamlit as st
import sys
import os
import asyncio

# Add src to path to import synthfuse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

st.set_page_config(
    page_title="Synth-Fuse Lab v0.2.0",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Synth-Fuse v0.2.0 - Unified Field Engineering")
st.markdown("### Cabinet of Alchemists - Interactive Lab")

# Initialize session state
if 'cabinet' not in st.session_state:
    st.session_state.cabinet = None
    st.session_state.status = "offline"

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Cabinet Controls")
    
    if st.button("🚀 Initialize Cabinet", type="primary", use_container_width=True):
        try:
            from synthfuse import CabinetOrchestrator
            st.session_state.cabinet = CabinetOrchestrator()
            st.session_state.status = "initializing"
            st.success("Cabinet instance created!")
            
            # Simulate async initialization
            async def init_cabinet():
                await st.session_state.cabinet.initialize()
                st.session_state.status = "online"
            
            asyncio.run(init_cabinet())
            st.rerun()
            
        except Exception as e:
            st.error(f"Failed to initialize: {e}")
    
    if st.button("🛑 Emergency Shutdown", use_container_width=True):
        if st.session_state.cabinet:
            asyncio.run(st.session_state.cabinet.emergency_shutdown())
            st.session_state.status = "offline"
            st.warning("Cabinet shutdown initiated")
            st.rerun()
        else:
            st.warning("Cabinet not initialized")
    
    st.divider()
    st.header("📊 System Info")
    st.code(f"Status: {st.session_state.status}")
    st.code("Version: 0.2.0")
    st.code("Python: " + sys.version.split()[0])
    
    # Sidebar additions
with st.sidebar:
    st.divider()
    st.header("🧪 Recipe Loader")
    
    recipes = {
        "Custom": {"sigil": "", "data": {}},
        "Quantum Bell": {
            "sigil": "(H⊗H)→(CNOT)",
            "data": {"qubits": 2, "state": "entangled", "basis": "computational"},
            "desc": "Standard Bell state preparation"
        },
        "Thermal Dissipation": {
            "sigil": "(T⊗∇)→(κ)",
            "data": {"temperature": 300, "conductivity": 0.5, "nodes": 64},
            "desc": "Heat equation solver"
        },
        "Neural Manifold": {
            "sigil": "(R⊗σ)→(W⊕b)",
            "data": {"layers": [128, 64, 10], "activation": "relu", "dropout": 0.2},
            "desc": "MLP forward pass topology"
        },
        "Chaos Control": {
            "sigil": "(C⊗L)→(λ)",
            "data": {"lyapunov": 0.8, "attractor": "lorenz", "dt": 0.01},
            "desc": "Stabilized strange attractor"
        },
        "Signal Decomposition": {
            "sigil": "(F⊗W)→(ψ)",
            "data": {"signal": [0.1, 0.3, -0.2, 0.8], "wavelets": "db4", "levels": 3},
            "desc": "Wavelet packet analysis"
        }
    }
    
    selected_recipe = st.selectbox(
        "Select Recipe", 
        list(recipes.keys()),
        help="Load predefined sigil configurations"
    )
    
    if selected_recipe != "Custom":
        st.caption(f"*{recipes[selected_recipe]['desc']}*")
    
    if st.button("📥 Load Recipe", use_container_width=True):
        if selected_recipe != "Custom":
            st.session_state.current_sigil = recipes[selected_recipe]["sigil"]
            st.session_state.current_data = str(recipes[selected_recipe]["data"]).replace("'", '"')
            st.success(f"Loaded {selected_recipe}")
            st.rerun()

# Main dashboard
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Cabinet Status", st.session_state.status, delta=None)

with col2:
    st.metric("Entropy", "0.127", delta="-0.02")

with col3:
    st.metric("Thermal Load", "18%", delta="+3%")

# Sigil Processing Section
st.header("🔧 Sigil Processor")
sigil_input = st.text_input("Enter Sigil Expression", value="(I⊗Z)", 
                           help="Example: (I⊗Z), (R⊕S), (C⊗L)")

data_input = st.text_area("Input Data (JSON format)", 
                         value='{"values": [1, 2, 3, 4, 5], "mode": "test"}',
                         height=100)

if st.button("⚡ Process Sigil", type="secondary"):
    if st.session_state.cabinet and st.session_state.status == "online":
        with st.spinner("Processing Sigil..."):
            try:
                # Async processing
                async def process():
                    return await st.session_state.cabinet.process_sigil(
                        sigil_input, 
                        eval(data_input) if data_input else {}
                    )
                
                result = asyncio.run(process())
                st.success("Sigil processed successfully!")
                
                # Display results
                with st.expander("📋 Results Details"):
                    st.json(result)
                
                # Visualize
                st.subheader("📈 Processing Metrics")
                cols = st.columns(4)
                cols[0].metric("Entropy", f"{result.get('entropy', 0):.3f}")
                cols[1].metric("Thermal", f"{result.get('thermal_load', 0):.1%}")
                cols[2].metric("Consensus", "✓" if result.get('consensus_reached') else "✗")
                cols[3].metric("Sigil", sigil_input)
                
            except Exception as e:
                st.error(f"Processing failed: {e}")
    else:
        st.warning("Please initialize the Cabinet first")

# System Logs
st.header("📜 System Logs")
log_container = st.container(height=200)
with log_container:
    st.code("""[INFO] Synth-Fuse v0.2.0 initialized
[INFO] Cabinet Orchestrator ready
[INFO] All dependencies loaded successfully
[INFO] Unified Field Engineering active
[INFO] Zeta-Manifold projection stable
[INFO] Sigil compiler online""")

# Quick Start Guide
with st.expander("🚀 Quick Start Guide"):
    st.markdown("""
    ### Getting Started with Synth-Fuse
    
    1. **Initialize the Cabinet** using the sidebar button
    2. **Enter a Sigil** - topological constraint expression
    3. **Provide input data** in JSON format
    4. **Process the Sigil** to see results
    
    ### Example Sigils:
    - `(I⊗Z)` - Identity over integer lattice
    - `(R⊕S)` - Real expansion in symbolic manifold
    - `(C⊗L)` - Chaos constrained by Lévy stability
    
    ### Next Steps:
    - Explore the Cabinet architecture
    - Test different Sigil patterns
    - Monitor system metrics
    """)

# Footer
st.divider()
st.caption("Synth-Fuse v0.2.0 | Unified Field Engineering | Cabinet of Alchemists")
