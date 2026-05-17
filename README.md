# FLEEA — Future Learning Executive & Everyday Assistant

> A private, local-first AI agent that can **Answer · Act · Remember · Speak**.

![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Gemini](https://img.shields.io/badge/LLM-Gemini%202.0%20Flash-orange?logo=google)
![React](https://img.shields.io/badge/Frontend-React%2019%20+%20Vite-61dafb?logo=react)
![Phase](https://img.shields.io/badge/Phase-Production%20Ready-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## What is FLEEA?

FLEEA is a modular, desktop-based AI agent built with a **production-grade architecture**. It goes beyond a chatbot — it plans, executes tools, remembers context, and talks back to you through a J.A.R.V.I.S-themed web dashboard.

| Capability | Status |
|---|---|
| 🧠 Gemini-powered reasoning with native tool calling | ✅ Live |
| 🔍 Real-time web search via Tavily | ✅ Live |
| 💾 Long-term vector memory (ChromaDB + sentence-transformers) | ✅ Live |
| 🧬 Short-term conversation memory (sliding window) | ✅ Live |
| 👤 Persistent user profile learning | ✅ Live |
| 🎤 Voice input (faster-whisper STT) | ✅ Live |
| 🔊 Voice output (pyttsx3 TTS — thread-safe) | ✅ Live |
| 🌐 Mobile-First Responsive Dashboard (React 19 + Tailwind 4) | ✅ Live |
| 🔐 Database Neural Auth — secure SQLite user profiles + bcrypt | ✅ Live |
| 🛡️ Admin Dashboard — manage users securely | ✅ Live |
| 🤖 Autonomous multi-step task execution loop | ✅ Live |
| 🔒 Three-tier safety gate on all actions | ✅ Live |
| 📊 Token + cost tracking on every API call | ✅ Live |
| 💬 Real-time SocketIO text chat with Markdown rendering | ✅ Live |
| 📱 Adaptive Sidebar Drawers for mobile/tablet viewports | ✅ Live |

---

## Architecture

```text
┌──────────────────────── Interfaces ─────────────────────────────┐
│  cli.py (CLI REPL)         voice_ui_server.py (Flask+SocketIO) │
│  voice_interface.py        web/ (React 19 + Vite 8 + Tailwind) │
└──────────────────────────────┬──────────────────────────────────┘
                               │  SocketIO (bidirectional)
                       ┌───────▼───────┐
                       │  FLEEABrain   │  ← Gemini 2.0 Flash
                       │ (agent/brain) │    manual tool calling
                       └───────┬───────┘
          ┌────────────┼────────────┐
          ▼            ▼            ▼
    ToolRegistry   SafetyMgr   SessionMgr
      WebSearch    (3-tier)    (SQLite)
          │                        │
          ▼                        ▼
  ┌───────────────┐        ExecutionLogger
  │ Memory System │          (tokens/cost)
  │  VectorStore  │
  │  ShortTerm    │
  │  Retriever    │
  │  ProfileMgr   │
  └───────────────┘
          │
  ┌───────▼────────┐
  │  Voice I/O     │
  │  STT (Whisper) │
  │  TTS (pyttsx3) │
  └────────────────┘
```

---

## Project Structure

```
FLEEA/
├── agent/                       # Core intelligence
│   ├── brain.py                 # Agentic loop — Gemini + tool calling
│   ├── safety.py                # Three-tier action classification
│   ├── session.py               # Session lifecycle management
│   ├── logger.py                # Execution + token/cost logging
│   ├── planner.py               # Multi-step task planning
│   ├── executor.py              # Task execution with retry
│   ├── task_manager.py          # Persistent task state machine
│   ├── autonomous_loop.py       # Orchestrated autonomous pipeline
│   ├── memory.py                # Memory integration stub
│   └── profile.py               # User preference store
│
├── memory/                      # Memory subsystem
│   ├── vector_store.py          # ChromaDB persistent vector store
│   ├── retriever.py             # Semantic memory retrieval
│   ├── short_term.py            # Sliding-window conversation buffer
│   ├── profile_manager.py       # Structured user profile learning
│   └── chroma_db/               # Persisted vector data (auto-generated)
│
├── voice/                       # Voice I/O
│   ├── stt.py                   # Speech-to-text (faster-whisper)
│   ├── tts.py                   # Text-to-speech (pyttsx3, thread-safe)
│   ├── voice_interface.py       # Continuous listen→think→speak loop
│   └── wake_word.py             # Wake-word detection (placeholder)
│
├── tools/                       # Tool implementations
│   ├── base_tool.py             # Abstract tool interface
│   ├── tool_registry.py         # Centralized tool routing
│   ├── web_search.py            # Tavily web search tool
│   ├── calendar_tool.py         # Calendar integration (placeholder)
│   ├── file_tool.py             # File operations (placeholder)
│   ├── computer_control.py      # Desktop automation (placeholder)
│   └── task_manager.py          # Task manager tool (placeholder)
│
├── interfaces/                  # Interface adapters
│   ├── voice_interface.py       # Voice mode entry point
│   └── voice_ui_server.py       # Flask + SocketIO web backend
│
├── web/                         # React + Vite Web Dashboard
│   ├── src/
│   │   ├── main.jsx             # React entry point
│   │   ├── App.jsx              # Root component (auth gate + layout)
│   │   ├── App.css              # App-level styles
│   │   ├── index.css            # Global design tokens + Tailwind
│   │   ├── components/
│   │   │   ├── LoginScreen.jsx  # Neural Auth login/signup UI
│   │   │   ├── RingVisualizer.jsx # 3D rotating ring animation
│   │   │   ├── ChatPanel.jsx    # SocketIO-powered text chat
│   │   │   ├── AdminPanel.jsx   # Admin user management view
│   │   │   └── Sidebar.jsx      # Profile, memory stats, system status
│   │   └── hooks/
│   │       └── useSocket.js     # SocketIO client hook
│   ├── index.html               # HTML shell (Outfit font)
│   ├── vite.config.js           # Vite + React + Tailwind config
│   ├── package.json             # npm dependencies
│   └── dist/                    # Production build (auto-generated)
│
├── database/                    # SQLite schema + manager
│   ├── models.py                # ORM-style database operations
│   └── fleea.db                 # SQLite data file (auto-generated)
│
├── config/                      # Settings & prompts
│   ├── settings.py              # Pydantic-based
│   └── prompts.py               # System prompt templates
│
├── tests/                       # Test suite
├── logs/                        # Rotating log files
├── app.py                       # WSGI / Vercel entry point (Flask)
├── cli.py                       # CLI REPL entry point
├── requirements.txt             # Python dependencies
└── .env                         # API keys (never committed)
```

---

---

## Cloud Deployment (Render / GitHub)

FLEEA is pre-configured for automated deployment via **Render Blueprints**.

### 1. Push to GitHub
Fork this repository and push your changes to your `master` branch.

### 2. Deploy on Render
1. Create a new **Blueprint** service on [Render](https://render.com).
2. Connect your GitHub repository.
3. Set these **Environment Variables**:
   - `GOOGLE_API_KEY`: Your Gemini API Key.
   - `TAVILY_API_KEY`: Your Tavily API Key.
   - `DB_PATH`: `/data/fleea.db` (ensures persistence).
4. **Persistent Disk**: Render will mount a 1GB disk at `/data` automatically based on `render.yaml`.

### 3. Voice Support
When deployed to the cloud, FLEEA automatically switches to **Web Voice Fallback**. Use any modern browser (Chrome, Edge, Safari) to speak with the agent directly from the dashboard.

---

## Local Installation

### 1. Prerequisites

| Requirement | Notes |
|---|---|
| Python **3.11+** | Core runtime |
| Node.js **18+** & npm | For the Web Dashboard |
| [Google AI Studio API key](https://aistudio.google.com/apikey) | Free — powers the Gemini LLM |
| [Tavily API key](https://tavily.com) | Free tier: 1,000 searches/month |

### 2. Clone & Setup

```bash
# Clone the repo
git clone https://github.com/Santosh-Ku-Sahoo/FLEEA.git
cd FLEEA

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# Install Python dependencies
pip install -r requirements.txt

# Install Web Dashboard dependencies
cd web
npm install
cd ..
```

### 3. Configure

Create a `.env` file in the project root with your API keys:

```env
# ── LLM Provider ─────────────────────────────────
GOOGLE_API_KEY=AIza...

# ── Search ────────────────────────────────────────
TAVILY_API_KEY=tvly-...

# ── Agent Behavior (optional tuning) ─────────────
DEFAULT_MODEL=gemini-2.0-flash
MAX_RETRIES=3
MAX_TOOL_ITERATIONS=10
API_TIMEOUT_SECONDS=60

# ── Logging ───────────────────────────────────────
LOG_LEVEL=INFO
```

### 4. Run

#### 🌐 Web Dashboard (Recommended)

Start both the backend API and frontend dev server:

```bash
# Terminal 1 — Start Flask + SocketIO backend
python -m interfaces.voice_ui_server
# → Backend API running on http://localhost:5050

# Terminal 2 — Start Vite frontend
cd web
npm run dev
# → Dashboard available at http://localhost:5173
```

Open the dashboard URL in your browser. You'll see the **Neural Auth** login screen. Create a new account to test out the persistent database authentication. Administrative users have access to the **Admin Dashboard** for user management.

> [!TIP]
> Use the **Sign Up** link on the login screen to register your first local identity. All data is stored in the local `database/fleea.db` file.

#### 💻 Terminal (CLI REPL)

```bash
python cli.py
```

#### 🎤 Voice Mode (Headless)

```bash
python -m voice.voice_interface
# Options:
#   --model     tiny|base|small|medium  (default: base)
#   --duration  <seconds>               (default: 5.0)
#   --language  <code>                  (default: en)
```

---

## Production Deployment

FLEEA can be deployed as a unified single-process application where the Python backend serves the optimized React frontend.

### 1. Build & Prepare
Run the deployment script to build the frontend and verify dependencies:
```bash
python scripts/deploy.py
```

### 2. Launch
Once the build is complete, you only need to run the backend:
```bash
python -m interfaces.voice_ui_server
```
The dashboard will be served at **http://localhost:5050** (both `localhost` and `127.0.0.1` work — the server binds in dual-stack IPv4+IPv6 mode).

> [!IMPORTANT]
> In production mode (like on Render), the server must use the `eventlet` worker class for Gunicorn to handle high-performance concurrent WebSocket connections and bypass HTTP polling load balancer issues. Example: `gunicorn --worker-class eventlet -w 1 ...`

---

## Usage

### CLI Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/clear` | Clear conversation history |
| `/history` | Show conversation history |
| `/status` | Session stats (tokens, cost, uptime) |
| `/logs` | Recent tool execution logs |
| `/exit` | Exit with session summary |

### Web Dashboard

The dashboard provides a full J.A.R.V.I.S-themed experience:

| Feature | Description |
|---------|-------------|
| 🔐 **Neural Auth** | Secure SQLite-backed authentication with password hashing (bcrypt) |
| 🛡️ **Role-Based Access** | Admin vs. User roles; sensitive features like the Admin Panel are protected |
| 💎 **3D Ring Visualizer** | Animated rotating rings that react to agent state (Idle → Listening → Thinking → Speaking) |
| 💬 **Real-time Chat** | SocketIO-powered text chat with live Markdown rendering |
| 🎤 **Voice Controls** | Start/Stop buttons for hands-free speech interaction |
| 📱 **Responsive UI** | Mobile-first layout with collapsible drawer sidebars |
| 👤 **Profile Sidebar** | Live user profile, memory stats, and system hardware status |
| ⚙️ **System Monitor** | STT/TTS engine availability, brain status indicators |
| 🛂 **Admin Dashboard** | Manage all users with integrated deletion protection for administrative accounts |

### Voice Mode

Say **"exit"**, **"quit"**, or **"goodbye"** to end the session gracefully. All other speech is transcribed, sent to the brain, and spoken back to you.

---

## Memory System

FLEEA maintains three layers of memory that work automatically on every request:

| Layer | Storage | Scope | Implementation |
|-------|---------|-------|----------------|
| **Short-term** | Recent conversation turns (sliding window) | Per-session | `memory/short_term.py` |
| **Vector store** | Semantically indexed facts & summaries | Persistent | ChromaDB via `memory/vector_store.py` |
| **User profile** | Structured preferences, habits, facts | Persistent | `memory/profile_manager.py` |

Memory retrieval is fully automatic — relevant context is injected into every Brain call without manual commands.

---

## Safety

Every tool call passes through a three-tier safety gate:

| Level | Behaviour |
|-------|-----------|
| ✅ **SAFE** | Execute immediately (e.g., web search, memory read) |
| ⚠️ **NEEDS_APPROVAL** | Print intended action, wait for `y/n` confirmation |
| 🚫 **BLOCKED** | Never execute (e.g., system wipes, mass deletions) |

Unknown tools default to **NEEDS_APPROVAL** — fail-safe by design.

---

## Tech Stack

### Backend (Python)

| Component | Technology |
|-----------|-----------|
| LLM | Google Gemini 2.0 Flash via `google-genai` |
| Web Search | Tavily API via `httpx` |
| Configuration | Pydantic + pydantic-settings + python-dotenv |
| Database | SQLite (built-in) |
| Vector DB | ChromaDB + sentence-transformers (all-MiniLM-L6-v2) |
| STT | faster-whisper (CTranslate2-backed Whisper) |
| TTS | pyttsx3 (OS-native voices, thread-safe) |
| Web Server | Flask + Flask-SocketIO |
| Terminal UX | Rich |

### Frontend (JavaScript)

| Component | Technology |
|-----------|-----------|
| Framework | React 19 |
| Build Tool | Vite 8 |
| Styling | Tailwind CSS 4 |
| Real-time | Socket.IO Client |
| Icons | Lucide React |
| Markdown | react-markdown |
| Font | Outfit (Google Fonts) |

---

## Roadmap

- [x] **Phase 1** — Core Brain · Web Search · Logging · Safety · SQLite
- [x] **Phase 2** — Long-term Vector Memory · Short-term Buffer · User Profile
- [x] **Phase 3** — Voice I/O (STT + TTS) · Listen→Think→Speak Loop
- [x] **Phase 4** — Autonomous Multi-step Task Loop (Planner → Executor → Reflector)
- [x] **Phase 5** — React/Vite Web Dashboard · Neural Auth · SocketIO Real-time
- [ ] **Phase 6** — Calendar & Task Management integrations
- [ ] **Phase 7** — Computer Control (desktop automation)
- [ ] **Phase 8** — Plugin System · Wake-word detection

---

## Design Principles

1. **Architecture before intelligence** — infrastructure must be rock-solid before adding smarts
2. **Safety before automation** — never execute dangerous actions without explicit approval
3. **Thread safety matters** — COM-based TTS, async LLM calls, and React UI all coexist cleanly
4. **Explicit over implicit** — dependency injection everywhere, no globals, no singletons
5. **Log everything** — every tool call, every token, every cost
6. **Memory is first-class** — agents without memory are just expensive autocomplete
7. **Local-first privacy** — all data stays on your machine; no telemetry, no cloud sync
8. **Mathematical Robustness** — handling hardware-level audio edge cases (NaN/Overflow) gracefully
9. **Concurrency First** — non-blocking AI reasoning via background threading for a smooth UI experience

---

## Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feat/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feat/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Troubleshooting

### "Nothing Showing" (Black Screen) on Dashboard
If you open the dashboard and see only a black screen or it fails to load:

1. **Both URLs work**: The server binds to `::` (dual-stack), so both `http://localhost:5050` and `http://127.0.0.1:5050` are supported on Windows, macOS, and Linux.
2. **Hard Refresh**: If you've just updated the code, your browser may have cached an old JS bundle. Press `Ctrl + F5` (Windows) or `Cmd + Shift + R` (Mac) to clear the cache.
3. **Build the Frontend**: Ensure you have run `python scripts/deploy.py` or `npm run build` in the `web` directory so the `dist` folder exists.

### Chat Input Doesn't Work on Render (WebSocket Closed)
If you deploy to Render and the chat interface stays stuck on "READY" when you type a message (and you see a `WebSocket connection to ... failed` error in the browser console):
1. **Check `render.yaml`**: Ensure your start command uses the `eventlet` worker class for Gunicorn (`gunicorn --worker-class eventlet -w 1 ...`), NOT `gevent`. The `gevent` worker will instantly reject WebSocket upgrades unless you use a specialized websocket worker class.

### Audio / Voice Issues
- **STT**: Ensure you have a working microphone and have granted permissions in your OS.
- **TTS**: If no sound plays, check your system's default audio output. FLEEA uses `pyttsx3` which relies on native OS voices.

---

*Built with systems thinking. Not hype.*
]]>
