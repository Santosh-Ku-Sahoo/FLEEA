"""
FLEEA Application — Vercel / WSGI entry point.

Exposes the Flask `app` instance for deployment platforms like Vercel
that auto-detect a top-level ``app`` variable in ``app.py``.

For the CLI terminal REPL, run ``python cli.py`` instead.
"""

from interfaces.voice_ui_server import app  # noqa: F401

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5050)
