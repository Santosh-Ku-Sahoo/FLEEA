"""
FLEEA Voice UI Server — Flask + SocketIO visual interface.

Runs the VoiceInterface loop in a background thread and emits
real-time state events to the browser for the 3D ring animation.

States pushed to browser:
    idle       — waiting / ready
    listening  — mic open, recording
    thinking   — Brain.think() running
    speaking   — TTS playing
    error      — recoverable error

Usage:
    python -m interfaces.voice_ui_server
    # Opens http://localhost:5050 automatically
"""

from __future__ import annotations

# ── Eventlet Monkey Patching (Must be FIRST) ───────────────────────
import os
try:
    import eventlet
    eventlet.monkey_patch()
except ImportError:
    pass

import asyncio
import logging
import logging.handlers
import os
import sys
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from threading import Thread
from typing import Any

# ── project root on path ───────────────────────────────────────────
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from flask import Flask, render_template_string, request, jsonify
import bcrypt
from flask_socketio import SocketIO

from agent.brain import FLEEABrain
from agent.logger import ExecutionLogger
from agent.safety import SafetyManager
from agent.session import SessionManager
from config.settings import Settings
from database.models import DatabaseManager
from tools.tool_registry import build_default_registry
try:
    from voice.stt import SpeechToText
    from voice.tts import TextToSpeech
    _VOICE = True
except ImportError:
    SpeechToText = None  # type: ignore[misc,assignment]
    TextToSpeech = None  # type: ignore[misc,assignment]
    _VOICE = False

try:
    from memory.short_term import ShortTermMemory
    from memory.retriever import MemoryRetriever
    from memory.profile_manager import ProfileManager
    from memory.vector_store import VectorStore
    _MEM = True
except ImportError:
    _MEM = False

_log = logging.getLogger("fleea.voice_ui")
_EXIT_COMMANDS = frozenset({"exit", "quit", "stop", "goodbye", "bye"})

# ── Flask app ──────────────────────────────────────────────────────
# Serve Vite build if it exists
_DIST_DIR = Path(_PROJECT_ROOT) / "web" / "dist"
app = Flask(__name__, static_folder=str(_DIST_DIR), static_url_path="")
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "fleea-secure-session-key")

# SocketIO (async mode depends on eventlet presence)
_ASYNC_MODE = "eventlet" if "eventlet" in sys.modules else "threading"
sio = SocketIO(app, cors_allowed_origins="*", async_mode=_ASYNC_MODE)

# Database Manager for API routes
settings = Settings()  # type: ignore[call-arg]
db_manager = DatabaseManager(settings.db_path_resolved)

_state: dict[str, Any] = {
    "status": "idle",
    "transcript": "",
    "response": "",
    "running": False,
}

_backend: dict[str, Any] | None = None

def _push(status: str, **extra: str) -> None:
    """Emit a state update to all connected browser clients."""
    _state["status"] = status
    _state.update(extra)
    sio.emit("state", {"status": status, **extra})


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def catch_all(path):
    """Serve index.html for all non-API routes to support React routing."""
    if _DIST_DIR.exists():
        # If the requested path exists as a static file, serve it
        if path and (_DIST_DIR / path).exists():
            return app.send_static_file(path)
        # Otherwise serve index.html (SPA routing)
        return app.send_static_file("index.html")
    
    # Fallback for dev environment without build folder
    return "<h1>FLEEA Dashboard</h1><p>Frontend not built. Run <code>npm run build</code> in the <code>web</code> directory.</p>"


def _get_or_create_backend() -> dict[str, Any]:
    global _backend
    if _backend is not None:
        return _backend

    settings = Settings()  # type: ignore[call-arg]
    db = DatabaseManager(settings.db_path_resolved)
    logger = ExecutionLogger(db)
    session_mgr = SessionManager(db)
    safety = SafetyManager(settings)
    registry = build_default_registry(settings)

    short_term = retriever = profile_mgr = vector_store = None
    if _MEM:
        try:
            persist_dir = str(Path(_PROJECT_ROOT) / "memory" / "chroma_db")
            vector_store = VectorStore(persist_dir=persist_dir)
            retriever = MemoryRetriever(vector_store)
            profile_mgr = ProfileManager(db)
            short_term = ShortTermMemory(max_turns=10)
        except Exception as e:
            _log.warning("Memory init failed: %s", e)

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
    session = session_mgr.create_session()
    stt = SpeechToText(model_size="small", language="en") if _VOICE else None
    tts = TextToSpeech(rate=145, voice_index=1) if _VOICE else None
    
    _backend = {
        "brain": brain, "session_mgr": session_mgr, "logger": logger,
        "session": session, "stt": stt, "tts": tts,
        "profile_mgr": profile_mgr, "short_term": short_term, "vector_store": vector_store
    }
    return _backend


def _voice_loop(username: str = "User") -> None:
    """Background thread: continuous listen → think → speak."""
    _state["running"] = True
    try:
        backend = _get_or_create_backend()
    except Exception as exc:
        _push("error", message=str(exc))
        _state["running"] = False
        return

    brain: FLEEABrain = backend["brain"]
    stt: SpeechToText = backend["stt"]
    tts: TextToSpeech = backend["tts"]

    _push("idle", transcript="", response="")
    try:
        tts.speak(f"Hello {username}, how can I assist you?")
    except Exception as e:
        _log.error("Failed initial greeting: %s", e)
        _push("error", message=f"Audio Error: {e}")
        _state["running"] = False
        return

    while _state["running"]:
        try:
            # ── Listen ────────────────────────────────────────────
            _push("listening", transcript="", response="")
            text = ""
            for _ in range(2):
                try:
                    text = stt.listen(duration=5.0)
                    break
                except Exception:
                    pass

            if not text:
                _push("idle", transcript="", response="")
                time.sleep(0.3)
                continue

            _push("listening", transcript=text, response="")

            # ── Exit check ────────────────────────────────────────
            clean_text = text.lower().strip()
            for char in ".!?,;":
                clean_text = clean_text.replace(char, "")
            
            if clean_text in _EXIT_COMMANDS:
                _push("idle", transcript=text, response="Goodbye!")
                tts.speak("Goodbye! Have a great day.")
                break

            # ── Think ─────────────────────────────────────────────
            _push("thinking", transcript=text, response="")
            response = asyncio.run(brain.think(text))

            # ── Speak ─────────────────────────────────────────────
            _push("speaking", transcript=text, response=response)
            tts.speak(response)

            _push("idle", transcript=text, response=response)

        except KeyboardInterrupt:
            break
        except Exception as exc:
            _log.error("Loop error: %s", exc)
            _push("error", message=str(exc))
            time.sleep(1)
            _push("idle")

    _state["running"] = False
    _push("idle", transcript="", response="Session ended.")


@sio.on("connect")
def on_connect():
    sio.emit("state", {"status": _state["status"],
                        "transcript": _state.get("transcript", ""),
                        "response": _state.get("response", "")})


@sio.on("start")
def on_start(data: dict[str, Any] = None):
    if data is None:
        data = {}
    username = data.get("username", "User")
    if not _state["running"]:
        _state["running"] = True
        Thread(target=_voice_loop, args=(username,), daemon=True).start()


@sio.on("stop")
def on_stop():
    _state["running"] = False
    _push("idle", transcript="", response="Stopped.")


@sio.on("text_chat")
def on_text_chat(data: dict[str, Any]):
    """Handle text input from the web dashboard."""
    text = data.get("text", "").strip()
    if not text:
        return

    clean_text = text.lower().strip()
    for char in ".!?,;":
        clean_text = clean_text.replace(char, "")
    
    if clean_text in _EXIT_COMMANDS:
        on_stop()
        return

    def process_text():
        _push("thinking", transcript=text, response="")
        try:
            backend = _get_or_create_backend()
            brain: FLEEABrain = backend["brain"]
            response = asyncio.run(brain.think(text))
            _push("idle", transcript=text, response=response)
            
            # Broadcast updated system status after a turn
            on_get_status()
        except Exception as exc:
            _log.error("Text chat error: %s", exc)
            _push("error", transcript=text, response=f"Error: {exc}")

    Thread(target=process_text, daemon=True).start()


@sio.on("greet")
def on_greet(data: dict[str, Any]):
    name = data.get("name", "User")
    is_returning = data.get("isReturning", False)
    greeting = f"Welcome back, {name}. How can I assist you today?" if is_returning else f"Welcome, {name}. I am FLEEA. How can I assist you today?"
    
    _push("speaking", transcript="", response=greeting)
    
    def speak_greeting():
        try:
            backend = _get_or_create_backend()
            tts = backend["tts"]
            tts.speak(greeting)
            _push("idle", transcript="", response=greeting)
        except Exception as e:
            _log.error("Greeting error: %s", e)
            _push("idle", transcript="", response="Error playing greeting.")
            
    Thread(target=speak_greeting, daemon=True).start()


@sio.on("get_status")
def on_get_status():
    """Return system, profile, and memory stats for the sidebar."""
    try:
        backend = _get_or_create_backend()
        
        # Build memory stats
        memory_stats = {"total_entries": 0, "short_term_count": 0}
        vs = backend.get("vector_store")
        st = backend.get("short_term")
        if vs and hasattr(vs, "_collection") and vs._collection is not None:
            memory_stats["total_entries"] = vs._collection.count()
        if st and hasattr(st, "get_context"):
            memory_stats["short_term_count"] = len(st.get_context()) // 2
            
        # Build profile data
        profile_data = {}
        pm = backend.get("profile_mgr")
        if pm:
            p = pm.get_profile()
            profile_data = {"name": p.get("name"), "preferences": p.get("preferences")}

        # Build hardware status
        stt = backend.get("stt")
        tts = backend.get("tts")
        status = {
            "stt_available": stt.is_available() if stt else False,
            "tts_available": tts.is_available if tts else False,
            "brain_status": backend["brain"].get_status()
        }

        sio.emit("system_status", {
            "profile": profile_data,
            "memory_stats": memory_stats,
            "status": status
        })
    except Exception as exc:
        _log.warning("Could not get status: %s", exc)


# ── REST API Routes (Auth & Admin) ─────────────────────────────────

@app.route("/api/auth/signup", methods=["POST"])
def auth_signup():
    data = request.get_json() or {}
    user_id = data.get("userId", "").strip()
    name = data.get("name", "").strip()
    email = data.get("email", "").strip()
    password = data.get("password", "")

    if not all([user_id, name, email, password]):
        return jsonify({"error": "All fields are required"}), 400

    hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    
    success = db_manager.create_user(user_id, name, email, hashed_pw)
    if not success:
        return jsonify({"error": "Username already taken"}), 409

    return jsonify({"message": "User created successfully"}), 201


@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    data = request.get_json() or {}
    user_id = data.get("userId", "").strip()
    password = data.get("password", "")

    if not user_id or not password:
        return jsonify({"error": "Missing username or password"}), 400

    user = db_manager.authenticate_user(user_id)
    if not user:
        return jsonify({"error": "Username not found"}), 404

    if not bcrypt.checkpw(password.encode("utf-8"), user["password_hash"].encode("utf-8")):
        return jsonify({"error": "Invalid password"}), 401

    return jsonify({
        "userId": user["user_id"],
        "name": user["name"],
        "email": user["email"],
        "role": user.get("role", "user"),
        "lastLogin": datetime.now(timezone.utc).isoformat()
    }), 200


@app.route("/api/auth/profile", methods=["PUT"])
def update_profile():
    data = request.json
    user_id = data.get("userId")
    name = data.get("name")
    email = data.get("email")

    if not all([user_id, name, email]):
        return jsonify({"error": "Missing fields"}), 400

    success = db_manager.update_user(user_id, name, email)
    if success:
        return jsonify({"message": "Profile updated successfully"}), 200
    return jsonify({"error": "Failed to update profile or email already taken"}), 400


@app.route("/api/admin/users", methods=["GET"])
def admin_get_users():
    _log.info("API: GET /api/admin/users - Request received")
    try:
        users = db_manager.get_all_users()
        _log.info("API: GET /api/admin/users - Found %d users", len(users))
        return jsonify(users), 200
    except Exception as e:
        _log.error("API: GET /api/admin/users - Error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/users/<user_id>", methods=["DELETE"])
def admin_delete_user(user_id):
    # Fetch user first to check role
    user = db_manager.authenticate_user(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    if user.get("role") == "admin":
        return jsonify({"error": "Admin users cannot be deleted"}), 403
        
    success = db_manager.delete_user(user_id)
    if success:
        return jsonify({"message": "User deleted"}), 200
    return jsonify({"error": "Delete failed"}), 500


# ── Entry point ────────────────────────────────────────────────────

def main() -> None:
    log_dir = os.path.join(_PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure Log Rotation
    log_file = os.path.join(log_dir, "fleea.log")
    handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3
    )
    handler.setFormatter(logging.Formatter("%(asctime)s  %(name)-20s  %(levelname)-7s  %(message)s"))
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    root_logger.addHandler(handler)

    port = 5050
    print(f"\n  [FLEEA] Web Dashboard API running on http://localhost:{port}\n")
    print("  Use 'npm run dev' in the 'web' folder for development UI.")

    # Production eventlet monkey patch warning
    import warnings
    warnings.filterwarnings("ignore", module="werkzeug")
    sio.run(app, host="0.0.0.0", port=port, debug=False, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()

