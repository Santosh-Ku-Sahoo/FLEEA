"""
FLEEA Streamlit Web Interface — Chat UI for the AI agent.

This is a thin presentation layer that reuses the existing backend:
    - FLEEABrain (agent/brain.py) for all LLM + tool logic
    - SessionManager, ExecutionLogger, SafetyManager, ToolRegistry
    - Phase 2 memory subsystems (ShortTermMemory, VectorStore,
      MemoryRetriever, ProfileManager)

All dependency wiring mirrors app.py — no backend logic is duplicated.

Usage:
    cd FLEEA
    .\\venv\\Scripts\\streamlit run streamlit_app.py
"""

from __future__ import annotations

import asyncio
import logging
import logging.handlers
import os
from pathlib import Path
from typing import Any

import streamlit as st

# ── Page config (MUST be first Streamlit call) ─────────────────────
st.set_page_config(
    page_title="FLEEA — AI Agent",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── FLEEA imports ──────────────────────────────────────────────────
from agent.brain import FLEEABrain
from agent.logger import ExecutionLogger
from agent.safety import SafetyManager
from agent.session import SessionManager
from config.settings import Settings
from database.models import DatabaseManager
from tools.tool_registry import build_default_registry

# Phase 2 memory (optional — graceful if missing)
try:
    from memory.short_term import ShortTermMemory
    from memory.retriever import MemoryRetriever
    from memory.profile_manager import ProfileManager
    from memory.vector_store import VectorStore
    _MEMORY_AVAILABLE = True
except ImportError:
    _MEMORY_AVAILABLE = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LOGGING (same as app.py — writes to logs/fleea.log)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _configure_logging(level: str) -> None:
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "fleea.log"),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(name)-22s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root = logging.getLogger("fleea")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not root.handlers:
        root.addHandler(handler)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DEPENDENCY WIRING (cached in session_state across reruns)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _init_backend() -> dict[str, Any]:
    """
    Wire all backend dependencies exactly like app.py._main().

    Returns a dict of all components needed by the UI.
    Called once and cached in st.session_state["backend"].
    """
    settings = Settings()  # type: ignore[call-arg]
    _configure_logging(settings.LOG_LEVEL)

    db = DatabaseManager(settings.db_path_resolved)
    logger = ExecutionLogger(db)
    session_mgr = SessionManager(db)
    safety = SafetyManager(settings)
    registry = build_default_registry(settings)

    # Phase 2 memory wiring
    short_term = None
    retriever = None
    profile_mgr = None
    vector_store = None

    if _MEMORY_AVAILABLE:
        try:
            persist_dir = str(Path(__file__).resolve().parent / "memory" / "chroma_db")
            vector_store = VectorStore(persist_dir=persist_dir)
            retriever = MemoryRetriever(vector_store)
            profile_mgr = ProfileManager(db)
            short_term = ShortTermMemory(max_turns=10)
        except Exception as exc:
            logging.getLogger("fleea.streamlit").warning(
                "Memory subsystem init failed (non-fatal): %s", exc
            )

    brain = FLEEABrain(
        settings=settings,
        tool_registry=registry,
        logger=logger,
        safety=safety,
        session_manager=session_mgr,
        short_term_memory=short_term,
        memory_retriever=retriever,
        profile_manager=profile_mgr,
        vector_store=vector_store,
    )

    # Start session
    session = session_mgr.create_session()

    return {
        "brain": brain,
        "session_mgr": session_mgr,
        "logger": logger,
        "profile_mgr": profile_mgr,
        "retriever": retriever,
        "short_term": short_term,
        "vector_store": vector_store,
        "session": session,
        "settings": settings,
    }


def get_backend() -> dict[str, Any]:
    """Get or initialize the backend singleton in session_state."""
    if "backend" not in st.session_state:
        st.session_state["backend"] = _init_backend()
    return st.session_state["backend"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ASYNC HELPER (Streamlit is sync, Brain.think() is async)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_async(coro):
    """Run an async coroutine from Streamlit's sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CUSTOM CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_CSS = """
<style>
    /* Global font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    header { visibility: hidden; }
    footer { visibility: hidden; }

    /* Chat message bubbles */
    .stChatMessage {
        border-radius: 12px !important;
        margin-bottom: 4px !important;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    }
    [data-testid="stSidebar"] * {
        color: #c9d1d9 !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #58a6ff !important;
    }

    /* Session badge */
    .session-badge {
        background: linear-gradient(135deg, #1a1b2e 0%, #16213e 100%);
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 16px;
        text-align: center;
    }
    .session-badge .label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #8b949e;
        margin-bottom: 4px;
    }
    .session-badge .value {
        font-size: 14px;
        font-weight: 600;
        color: #58a6ff;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Info panels in sidebar */
    .info-panel {
        background: #0d1117;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 12px;
        margin-top: 8px;
        font-size: 13px;
        line-height: 1.6;
    }
    .info-panel code {
        color: #79c0ff;
    }
    .info-panel .key {
        color: #8b949e;
    }
    .info-panel .val {
        color: #c9d1d9;
        font-weight: 500;
    }

    /* Header area */
    .fleea-header {
        text-align: center;
        padding: 20px 0 10px 0;
    }
    .fleea-header h1 {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 2px;
    }
    .fleea-header .subtitle {
        font-size: 14px;
        color: #8b949e;
    }
</style>
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def render_sidebar(backend: dict[str, Any]) -> None:
    """Render the sidebar with session info, actions, and memory panels."""

    session = backend["session"]
    profile_mgr = backend.get("profile_mgr")
    retriever = backend.get("retriever")
    short_term = backend.get("short_term")
    brain: FLEEABrain = backend["brain"]

    with st.sidebar:
        # ── Branding ──────────────────────────────────────────────
        st.markdown("## 🤖 FLEEA")
        st.caption("Future Learning Executive & Everyday Assistant")

        # ── Session badge ─────────────────────────────────────────
        session_id = getattr(session, "short_id", "unknown")
        st.markdown(
            f"""<div class="session-badge">
                <div class="label">Session</div>
                <div class="value">{session_id}</div>
            </div>""",
            unsafe_allow_html=True,
        )

        st.divider()

        # ── Actions ───────────────────────────────────────────────
        st.markdown("### Actions")

        if st.button("🗑️  Clear Chat", use_container_width=True):
            brain.clear_history()
            if short_term:
                short_term.clear()
            st.session_state["messages"] = []
            st.rerun()

        st.divider()

        # ── Profile Panel ─────────────────────────────────────────
        st.markdown("### 👤 User Profile")
        if profile_mgr:
            profile = profile_mgr.get_profile()
            if profile:
                html_lines = []
                for category, entries in sorted(profile.items()):
                    title = category.replace("_", " ").title()
                    html_lines.append(f"<strong>{title}</strong>")
                    for key, value in sorted(entries.items()):
                        display_key = key.replace("_", " ")
                        html_lines.append(
                            f'<span class="key">{display_key}:</span> '
                            f'<span class="val">{value}</span>'
                        )
                    html_lines.append("")

                st.markdown(
                    f'<div class="info-panel">{"<br>".join(html_lines)}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.caption("No profile data yet. Chat to build your profile!")
        else:
            st.caption("Profile system not available.")

        st.divider()

        # ── Memory Panel ──────────────────────────────────────────
        st.markdown("### 🧠 Memory")
        if retriever:
            stats = retriever.get_stats()
            st.markdown(
                f"""<div class="info-panel">
                    <span class="key">Stored memories:</span>
                    <span class="val">{stats.get('store_count', 0)}</span><br>
                    <span class="key">Threshold:</span>
                    <span class="val">{stats.get('threshold', 0):.2f}</span><br>
                    <span class="key">Max per query:</span>
                    <span class="val">{stats.get('max_memories', 0)}</span>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.caption("Memory system not available.")

        if short_term:
            stm_stats = short_term.get_stats()
            st.markdown(
                f"""<div class="info-panel">
                    <span class="key">Short-term:</span>
                    <span class="val">{stm_stats['message_count']}/{stm_stats['max_messages']} msgs</span><br>
                    <span class="key">Turns:</span>
                    <span class="val">{stm_stats['turn_count']}/{stm_stats['max_turns']}</span>
                </div>""",
                unsafe_allow_html=True,
            )

        st.divider()

        # ── System Status ─────────────────────────────────────────
        st.markdown("### ⚙️ System")
        status = brain.get_status()
        st.markdown(
            f"""<div class="info-panel">
                <span class="key">Model:</span>
                <span class="val">{status['model']}</span><br>
                <span class="key">Messages:</span>
                <span class="val">{status['message_count']}</span><br>
                <span class="key">Tools:</span>
                <span class="val">{', '.join(status['registered_tools']) or '—'}</span><br>
                <span class="key">Prompt size:</span>
                <span class="val">{status['system_prompt_length']:,} chars</span>
            </div>""",
            unsafe_allow_html=True,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN CHAT UI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main() -> None:
    """Main Streamlit entry point."""

    # ── Inject CSS ────────────────────────────────────────────────
    st.markdown(_CSS, unsafe_allow_html=True)

    # ── Initialize backend ────────────────────────────────────────
    try:
        backend = get_backend()
    except Exception as exc:
        st.error(f"**Failed to initialize FLEEA backend:** {exc}")
        st.info("Make sure your `.env` file is configured with valid API keys.")
        st.stop()

    brain: FLEEABrain = backend["brain"]

    # ── Initialize message history ────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # ── Sidebar ───────────────────────────────────────────────────
    render_sidebar(backend)

    # ── Header ────────────────────────────────────────────────────
    st.markdown(
        """<div class="fleea-header">
            <h1>🤖 FLEEA</h1>
            <div class="subtitle">Future Learning Executive & Everyday Assistant</div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Chat history ──────────────────────────────────────────────
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])

    # ── Welcome message (if empty) ────────────────────────────────
    if not st.session_state["messages"]:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(
                "Welcome! I'm **FLEEA**, your local-first AI assistant. "
                "Ask me anything — I can search the web, remember your preferences, "
                "and help you get things done.\n\n"
                "*Use the sidebar to view your profile, memory stats, and system status.*"
            )

    # ── Chat input ────────────────────────────────────────────────
    if user_input := st.chat_input("Ask FLEEA anything..."):
        # Display user message
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_input)

        # Get brain response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                try:
                    response = run_async(brain.think(user_input))
                except Exception as exc:
                    response = f"⚠️ Error: {exc}"

            st.markdown(response)

        # Store assistant response
        st.session_state["messages"].append({"role": "assistant", "content": response})

        # Rerun to update sidebar stats
        st.rerun()


if __name__ == "__main__":
    main()
