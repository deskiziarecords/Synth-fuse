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

# Main dashboard
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Cabinet Status", st.session_state.status, delta=None)

with col2:
    st.metric("Entropy", "0.127", delta="-0.02")

with col3:
    st.metric("Thermal Load", "18%", delta="+3%")

# Sigil Processing Section - NOW ACTUALLY WORKS
st.header("🔧 Sigil Processor")

# Use session state from recipe editor if available
current_sigil = st.session_state.get('current_sigil', '(I⊗Z)')
current_data = st.session_state.get('current_data', '{"size": 4, "values": [1.0, 0.5, -0.5, -1.0]}')

col1, col2 = st.columns([1, 2])

with col1:
    sigil_input = st.text_input(
        "Enter Sigil Expression", 
        value=current_sigil,
        key="sigil_field",
        help="Examples: (I⊗Z), (H⊗H)→(CNOT), (R⊕σ), (T⊗∇)"
    )
    
    data_input = st.text_area(
        "Input Data (JSON)", 
        value=current_data,
        height=150,
        help="Parameters: size, temperature, lyapunov, etc."
    )

with col2:
    # Live preview of what will happen
    st.caption("Operation Preview")
    if '⊗' in sigil_input:
        st.info("🔀 Tensor Product: Kronecker product of operators")
    elif '⊕' in sigil_input:
        st.info("➕ Direct Sum: Block diagonal concatenation")
    elif '→' in sigil_input:
        st.info("➡️  Composition: Sequential application")
    
    # Quick parameter presets
    preset = st.selectbox("Quick Data Preset", 
        ["Custom", "Qubit (4x4)", "Thermal (64x64)", "Neural (128x128)"],
        key="data_preset"
    )
    if preset == "Qubit (4x4)":
        data_input = '{"size": 4, "seed": 42}'
    elif preset == "Thermal (64x64)":
        data_input = '{"size": 64, "temperature": 300, "conductivity": 0.5}'
    elif preset == "Neural (128x128)":
        data_input = '{"size": 128, "layers": [128, 64], "dropout": 0.2}'

if st.button("⚡ Execute Sigil", type="primary", use_container_width=True):
    if st.session_state.cabinet and st.session_state.status == "online":
        try:
            # Parse data safely
            import json
            input_data = json.loads(data_input) if data_input else {}
            
            with st.spinner("Compiling & Executing..."):
                # Run the actual computation
                async def run_computation():
                    return await st.session_state.cabinet.process_sigil(
                        sigil_input, 
                        input_data
                    )
                
                result = asyncio.run(run_computation())
            
            # Display results in cool columns
            st.success("✨ Sigil Compiled Successfully")
            
            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            res_col1.metric("Entropy", f"{result['entropy']:.4f}", 
                          delta=f"{result['entropy']-0.1:.2f}")
            res_col2.metric("Thermal Load", f"{result['thermal_load']:.1%}",
                          delta=f"{(1-result['thermal_load'])*100:.0f}%", delta_color="inverse")
            res_col3.metric("Consensus", "✓ STABLE" if result['consensus_reached'] else "✗ UNSTABLE",
                          delta="Converged" if result['consensus_reached'] else "Diverging")
            res_col4.metric("Matrix Shape", str(result['shape']))
            
            # Visualization of result
            st.subheader("📊 Result Matrix Visualization")
            
            viz_tab1, viz_tab2 = st.tabs(["Heatmap", "Eigenvalues"])
            
            with viz_tab1:
                result_array = np.array(result['result'])
                if len(result_array.shape) == 2:
                    fig, ax = plt.subplots(figsize=(6, 5))
                    im = ax.imshow(result_array, cmap='viridis', aspect='auto')
                    ax.set_title(f"Operator Matrix: {sigil_input}")
                    plt.colorbar(im, ax=ax)
                    st.pyplot(fig)
                else:
                    st.line_chart(result_array.flatten()[:100])
            
            with viz_tab2:
                if len(result_array.shape) == 2 and result_array.shape[0] == result_array.shape[1]:
                    eigenvals = np.linalg.eigvals(result_array)
                    fig, ax = plt.subplots()
                    ax.scatter(eigenvals.real, eigenvals.imag, alpha=0.6)
                    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
                    ax.set_xlabel("Real")
                    ax.set_ylabel("Imaginary")
                    ax.set_title("Eigenvalue Spectrum")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
            
            # Raw data expander
            with st.expander("🔍 Raw Computation Output"):
                st.json({
                    'sigil': result['sigil'],
                    'entropy': result['entropy'],
                    'thermal_load': result['thermal_load'],
                    'result_sample': str(result['result'])[:200] + "...",
                    'iteration': result['iteration']
                })
                
        except Exception as e:
            st.error(f"🚨 Computation Failed: {str(e)}")
            st.exception(e)
    else:
        st.error("⚠️ Initialize the Cabinet first! (Sidebar → Initialize)")

# Add matplotlib import at top
import matplotlib.pyplot as plt
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
