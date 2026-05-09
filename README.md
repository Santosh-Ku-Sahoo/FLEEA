# FLEEA — Future Learning Executive & Everyday Assistant

> A private, local-first AI agent for personal productivity and daily task automation.

![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Phase](https://img.shields.io/badge/Phase-1%20Core%20Brain-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## What is FLEEA?

FLEEA is a modular, desktop-based AI assistant that can **Answer + Act + Remember + Automate**.

Unlike generic chatbots, FLEEA is designed as a real AI agent system with:
- 🧠 **Claude-powered brain** with native tool calling
- 🔍 **Real-time web search** via Tavily API
- 🔒 **Safety-first architecture** — dangerous actions require explicit approval
- 📊 **Token + cost tracking** on every API call
- 📝 **Full execution logging** for debugging and observability
- 🏗️ **Modular architecture** — add new tools without touching the core

## Architecture

```
User (Terminal) → app.py → FLEEABrain
                              ↓
                         Claude API ←→ ToolRegistry
                              ↓              ↓
                         SafetyLayer    WebSearchTool
                              ↓
                      ExecutionLogger → SQLite
```

## Project Structure

```
FLEEA/
├── agent/           # Core intelligence
│   ├── brain.py     # Agentic loop with tool calling
│   ├── safety.py    # Action classification + approval
│   ├── session.py   # Session lifecycle management
│   ├── logger.py    # Execution + token/cost logging
│   ├── planner.py   # Multi-step planning (Phase 2)
│   ├── memory.py    # Vector memory (Phase 2)
│   └── profile.py   # User preferences
├── tools/           # Tool implementations
│   ├── base_tool.py       # Abstract tool interface
│   ├── tool_registry.py   # Centralized tool routing
│   ├── web_search.py      # Tavily web search
│   ├── calendar_tool.py   # Google Calendar (Phase 3)
│   ├── task_manager.py    # Task system (Phase 3)
│   ├── computer_control.py # Desktop automation (Phase 4)
│   └── file_tool.py       # File operations (Phase 4)
├── voice/           # Voice I/O (Phase 5)
├── database/        # SQLite schema + manager
├── config/          # Settings + prompts
├── logs/            # Log storage
├── app.py           # Terminal REPL entry point
├── requirements.txt
└── .env             # API keys (not committed)
```

## Quick Start

### 1. Prerequisites
- Python 3.11+
- Anthropic API key ([console.anthropic.com](https://console.anthropic.com))
- Tavily API key ([tavily.com](https://tavily.com) — free tier: 1000 searches/month)

### 2. Setup

```bash
# Clone / navigate to the project
cd FLEEA

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure

Edit `.env` with your API keys:

```env
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...
```

### 4. Run

```bash
python app.py
```

## Usage

### Chat
Simply type your message and press Enter:
```
You → What's the latest news about AI regulations?
```

FLEEA will automatically search the web when needed and synthesize results.

### Commands
| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/clear` | Clear conversation history |
| `/history` | Show conversation history |
| `/status` | Show session stats (tokens, cost, duration) |
| `/logs` | Show recent tool execution logs |
| `/exit` | Exit with session summary |

## Safety

Every tool call passes through a three-tier safety gate:

| Level | Action |
|-------|--------|
| ✅ **SAFE** | Execute immediately (e.g., web search) |
| ⚠️ **NEEDS_APPROVAL** | Show intended action, wait for confirmation |
| 🚫 **BLOCKED** | Never execute (e.g., `rm -rf /`) |

Unknown tools default to **NEEDS_APPROVAL** (fail-safe).

## Roadmap

- [x] **Phase 1** — Core Brain + Web Search + Logging + Safety
- [ ] **Phase 2** — Memory (ChromaDB + structured recall)
- [ ] **Phase 3** — Calendar + Task Management
- [ ] **Phase 4** — Computer Control (desktop automation)
- [ ] **Phase 5** — Voice I/O (Whisper + ElevenLabs)
- [ ] **Phase 6** — Web Dashboard + Plugin System

## Design Principles

1. **Architecture before intelligence** — build the infrastructure right
2. **Safety before automation** — never execute dangerous actions without approval
3. **Explicit over implicit** — dependency injection, no singletons, no globals
4. **Log everything** — every tool call, every token, every cost
5. **Phase discipline** — don't start Phase N+1 until Phase N works perfectly

---

Built with systems thinking. Not hype.
