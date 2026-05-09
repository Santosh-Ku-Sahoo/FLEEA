# FLEEA Interfaces Package
#
# Provides multiple frontends for the FLEEA agent:
#   - VoiceInterface (voice_interface.py) — continuous voice loop
#   - CLI             (app.py)            — terminal REPL
#   - Streamlit       (streamlit_app.py)  — web UI
#
# Lazy imports to avoid circular / runpy warnings when running
# sub-modules directly (python -m interfaces.voice_interface).


def __getattr__(name: str):
    if name == "VoiceInterface":
        from interfaces.voice_interface import VoiceInterface
        return VoiceInterface
    raise AttributeError(f"module 'interfaces' has no attribute {name!r}")


__all__ = ["VoiceInterface"]
