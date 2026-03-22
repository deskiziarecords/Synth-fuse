import streamlit as st
import sys
import os
from pathlib import Path

# Add src to sys.path if synthfuse is not importable
try:
    import synthfuse
except ImportError:
    src_path = str(Path(__file__).parent.parent / "src")
    if src_path not in sys.path:
        sys.path.append(src_path)

import asyncio
import jax
import jax.numpy as jnp
from synthfuse.os import boot, Realm, os as sf_os
from synthfuse.agents.hub import list_agents, load_agent

st.set_page_config(page_title="Synth-Fuse Lab", page_icon="🩸", layout="wide")

st.title("Synth-Fuse Laboratory Dashboard 🩸")

# Boot OS on start
if 'os_status' not in st.session_state:
    with st.spinner("Booting Adiabatic Kernel..."):
        st.session_state.os_status = boot()

# Sidebar: Kernel Status
st.sidebar.header("Kernel Diagnostics")
if st.session_state.os_status:
    status = st.session_state.os_status
    st.sidebar.success(f"OS Version: {status.get('version', 'unknown')}")

    thermal = status.get('thermal', {})
    st.sidebar.metric("Thermal Load", f"{thermal.get('current_load', 0.0):.2f}")
    st.sidebar.metric("Entropy", f"{thermal.get('current_entropy', 0.0):.2f}")
    st.sidebar.metric("Capacity Remaining", f"{thermal.get('capacity_remaining', 1.0):.2f}")

# Main Lab Area
tab1, tab2, tab3 = st.tabs(["Alchemist Console", "Forum Arena", "Systemic Hydraulics"])

with tab1:
    st.header("Alchemist Console")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Agent Hub")
        agents = list_agents()
        selected_agent = st.selectbox("Select Agent", agents, index=agents.index("ferros") if "ferros" in agents else 0)

        if selected_agent == "ferros":
            st.info("Ferros (Ferrolearn) connected. Sigil: (S⊗I)⊕(D⊙C)")
            swarm_size = st.slider("Swarm Size", 64, 512, 128)
            dimensions = st.slider("Dimensions", 32, 256, 64)
            agent_config = {"swarm_size": swarm_size, "dimensions": dimensions}
        else:
            agent_config = {}

        prompt = st.text_area("Input Command (Spell/Sigil)", "Optimize multi-agent swarm for thermal neutrality")

        if st.button("Cast Transmogrifai"):
            try:
                agent = load_agent(selected_agent, **agent_config)
                with st.spinner(f"Agent {selected_agent} executing..."):
                    response = agent.generate(prompt)
                    st.session_state.last_response = response
                    st.balloons()
            except Exception as e:
                st.error(f"Execution failed: {e}")

    with col2:
        st.subheader("Result Projection")
        if 'last_response' in st.session_state:
            resp = st.session_state.last_response
            st.write(resp.content)
            st.json(resp.metadata)
            if resp.vector is not None:
                st.write("Latent Embedding Vector:")
                st.line_chart(resp.vector)

with tab2:
    st.header("Forum Arena")
    st.write("Specialist debate pending...")

with tab3:
    st.header("Systemic Hydraulics")
    st.write("Monitoring recursive debt Δ$ and solvency integral δ...")
