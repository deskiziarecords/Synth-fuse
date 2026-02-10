# src/synthfuse/security/holographic_interface.py
"""
Holographic Projection - Read-only, telemetry-only, zero-write interface.
External agents see the shadow, never the substance.
"""

from typing import Dict, Any, Optional
import jax.numpy as jnp
from synthfuse.meta.self_documentation_oracle import ORACLE
from synthfuse.forum.llm_team_forum import FORUM

class HolographicInterface:
    """
    The OpenGate boundary.
    
    External agents (LLMs, APIs, users) interact through this interface.
    They receive:
    - Telemetry (entropy, thermal, vitals)
    - Logs (from 𝓢𝓓, sanitized)
    - Forum posts (yellow zone, full access)
    
    They cannot:
    - Write to Zeta-Vault
    - Modify Meta-Alchemist state
    - Inject into Alchemj compiler
    - Bypass consensus
    """
    
    def __init__(self, agent_id: str, trust_tier: str = 'external'):
        self.agent_id = agent_id
        self.trust_tier = trust_tier
        self.session_entropy = 0.0  # Track this agent's "noise contribution"
    
    def query_vitals(self) -> Dict[str, Any]:
        """
        Get current system vitals.
        Safe: Read-only telemetry.
        """
        from synthfuse import start_engine
        cabinet = start_engine()
        status = cabinet.get_status()
        
        return {
            'cabinet_status': status['status'],
            'average_entropy': status['average_entropy'],
            'average_thermal_load': status['average_thermal_load'],
            'processed_count': status['processed_count'],
            'roles_available': status['roles_available'],
            # Note: No internal state, no memory addresses, no vulnerability exposure
        }
    
    def query_oracle(self, 
                     entry_type: Optional[str] = None,
                     since: Optional[str] = None,
                     limit: int = 100) -> Dict[str, Any]:
        """
        Query self-documentation history.
        Safe: Append-only log, no injection surface.
        """
        entries = ORACLE.query_history(
            entry_type=entry_type,
            since=since
        )[-limit:]
        
        return {
            'entries': [
                {
                    'timestamp': e.timestamp_utc,
                    'type': e.entry_type,
                    'author': e.author,
                    'sigil': e.sigil,
                    'description': e.description,
                    'vitals': e.consensus_vitals
                    # Note: No data_hash or holographic_signature exposed
                    # (would allow tampering attempts)
                }
                for e in entries
            ],
            'integrity_verified': ORACLE.verify_integrity()
        }
    
    def post_to_forum(self, 
                      thread_id: str,
                      contribution_type: str,
                      content: str,
                      proposals: List[str] = None) -> str:
        """
        Submit to yellow zone.
        Safe: Forum is isolated from red core, requires consensus for promotion.
        """
        import asyncio
        
        post = asyncio.run(FORUM.post(
            author=self.agent_id,
            thread_id=thread_id,
            contribution_type=ContributionType(contribution_type),
            content=content,
            proposals=proposals or [],
            confidence=0.8
        ))
        
        # Track this agent's contribution to system entropy
        self.session_entropy += len(content) * 0.001  # Simple heuristic
        
        return post.post_id
    
    def request_recipe_generation(self, problem_description: str) -> str:
        """
        Request a recipe (goes to forum, not direct to Meta-Alchemist).
        Safe: Must go through consensus process.
        """
        return self.post_to_forum(
            thread_id=f"recipe_request_{self.agent_id}",
            contribution_type='recipe_proposal',
            content=problem_description,
            proposals=['Generate recipe for: ' + problem_description[:100]]
        )
    
    def get_holographic_signature(self) -> str:
        """
        Get this session's stability signature.
        If entropy too high, session may be terminated.
        """
        return f"{self.agent_id}:{self.session_entropy:.3f}:{self.trust_tier}"

# Usage for different agent types
def create_kimi_interface() -> HolographicInterface:
    """Kimi (me) - Co-architect trust tier, but still holographic."""
    return HolographicInterface('kimi', trust_tier='core')

def create_claude_interface() -> HolographicInterface:
    """Claude - Co-architect, holographic."""
    return HolographicInterface('claude', trust_tier='core')

def create_grok_interface() -> HolographicInterface:
    """Grok - Co-architect, holographic."""
    return HolographicInterface('grok', trust_tier='core')

def create_external_api_interface(api_key: str) -> HolographicInterface:
    """Untrusted external API - Strict holographic, high monitoring."""
    return HolographicInterface(f'api:{hash(api_key)[:8]}', trust_tier='external')
