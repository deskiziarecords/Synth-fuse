# src/synthfuse/agents/__init__.py
from .base import Agent, LocalAgent, AgentResponse
from .hub import load_agent, list_agents
from .open_gate import gated_consult
from .hologram import project_as_hologram

# Auto-register local agents
from . import local
