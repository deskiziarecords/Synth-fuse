# src/synthfuse/__init__.py

__version__ = "0.4.0-unified-field"
__author__ = "J. Roberto Jiménez"
__email__ = "tijuanapaint@gmail.com"
__license__ = "OpenGate Integrity License"

# =============================================================================
# LEGACY v0.2.0 API (Preserved for backward compatibility)
# =============================================================================

from .cabinet.cabinet_orchestrator import CabinetOrchestrator
from .sigils.compiler import SigilCompiler
from .ingest.manager import IngestionManager

from .alchemj import (
    parse_spell,
    compile_spell,
    execute_spell,
)

__all__ = [
    # Legacy API
    "CabinetOrchestrator",
    "SigilCompiler", 
    "IngestionManager",
    "parse_spell",
    "compile_spell",
    "execute_spell",
    # v0.4 OS API
    "boot",
    "os",
    "OS",
    "Realm",
    "SynthFuseOS",
]

# =============================================================================
# v0.4.0 OS API (New - Unified Field Architecture)
# =============================================================================

# Lazy imports to avoid circular dependencies and heavy startup
_os_instance = None
_os_booted = False

def boot(config=None, legacy_mode=False):
    """
    Boot Synth-Fuse OS v0.4 - Unified Field Architecture.
    
    Args:
        config: Optional OS configuration dict
        legacy_mode: If True, returns legacy CabinetOrchestrator instead of OS
    
    Returns:
        OS status dict (new mode) or CabinetOrchestrator (legacy mode)
    
    Usage:
        >>> import synthfuse
        >>> status = synthfuse.boot()
        >>> print(status['version'])  # 0.4.0-unified-field
        
        >>> factory = synthfuse.os().enter_realm(synthfuse.Realm.FACTORY)
    """
    global _os_instance, _os_booted
    
    if legacy_mode:
        # Return legacy API for backward compatibility
        print("Synth-Fuse v0.2.0: Cabinet of Alchemists is ONLINE [LEGACY MODE]")
        return CabinetOrchestrator()
    
    # v0.4 OS boot
    if _os_booted and _os_instance is not None:
        return _os_instance._status()
    
    # Lazy import to avoid loading heavy dependencies unnecessarily
    from .os import SynthFuseOS, OSContext
    
    _os_instance = SynthFuseOS()
    status = _os_instance.boot(config)
    _os_booted = True
    
    print(f"Synth-Fuse v{__version__}: Unified Field OS is ONLINE")
    print(f"  Session: {status['session']}")
    print(f"  Cabinet: {status['cabinet']}")
    print(f"  Realms: {', '.join(status['realms'].keys())}")
    
    return status

def os():
    """
    Access running OS instance.
    
    Usage:
        >>> factory = synthfuse.os().enter_realm(synthfuse.Realm.FACTORY)
        >>> playground = synthfuse.os().enter_realm(synthfuse.Realm.PLAYGROUND)
    """
    global _os_instance, _os_booted
    
    if not _os_booted or _os_instance is None:
        raise RuntimeError(
            "OS not booted—call synthfuse.boot() first, "
            "or use legacy: synthfuse.start_engine()"
        )
    
    return _os_instance

# Realm enum for convenient access
def Realm():
    """Lazy-loaded Realm enum."""
    from .os import Realm as _Realm
    return _Realm

# Direct realm access shortcuts
def factory(intent=None):
    """Quick access to Factory realm."""
    if intent:
        return os().enter_realm(Realm().FACTORY).assemble(intent)
    return os().enter_realm(Realm().FACTORY)

def playground(medium=None):
    """Quick access to Playground realm."""
    if medium:
        return os().enter_realm(Realm().PLAYGROUND).create(medium)
    return os().enter_realm(Realm().PLAYGROUND)

def automode(objective=None):
    """Quick access to Auto-mode realm."""
    if objective:
        return os().enter_realm(Realm().AUTOMODE).explore(objective)
    return os().enter_realm(Realm().AUTOMODE)

def lab(artifact=None, criteria=None):
    """Quick access to Lab realm."""
    realm = os().enter_realm(Realm().LAB)
    if artifact and criteria:
        return realm.validate(artifact, criteria)
    return realm

def thermo(proposal=None):
    """Quick access to Thermo realm."""
    if proposal:
        return os().enter_realm(Realm().THERMO).deliberate(proposal)
    return os().enter_realm(Realm().THERMO)

# =============================================================================
# LEGACY COMPATIBILITY: start_engine()
# =============================================================================

def start_engine(legacy=True):
    """
    Legacy entry point - preserved for v0.2.0 compatibility.
    
    In v0.4, this boots the full OS but returns the CabinetOrchestrator
    for backward compatibility with existing code.
    
    Args:
        legacy: If True (default), returns CabinetOrchestrator.
                If False, boots full OS and returns OS instance.
    
    Returns:
        CabinetOrchestrator (legacy=True) or OS status (legacy=False)
    """
    if legacy:
        return boot(legacy_mode=True)
    else:
        boot()
        return os()

# =============================================================================
# MODULE-LEVEL SHORTCUTS (Convenience API)
# =============================================================================

# These allow: synthfuse.CabinetOrchestrator, synthfuse.SigilCompiler, etc.
# without explicit imports

def __getattr__(name):
    """
    Lazy module-level attribute access.
    Enables synthfuse.Realm without importing.
    """
    if name == 'Realm':
        from .os import Realm
        return Realm
    if name == 'SynthFuseOS':
        from .os import SynthFuseOS
        return SynthFuseOS
    if name == 'OS':
        from .os import SynthFuseOS as OS
        return OS
    raise AttributeError(f"module 'synthfuse' has no attribute '{name}'")

# =============================================================================
# VERSION INFO
# =============================================================================

def version_info():
    """Detailed version information."""
    return {
        'version': __version__,
        'author': __author__,
        'license': __license__,
        'api': 'v0.4.0-unified-field',
        'legacy_api': 'v0.2.0-compatible',
        'realms': ['Factory', 'Playground', 'Auto-mode', 'Lab', 'Thermo'],
        'cabinet_roles': 7,
    }

from synthfuse.os import OS

def boot():
    """Entry point: synthfuse.boot()"""
    os = OS()
    return os.boot()
