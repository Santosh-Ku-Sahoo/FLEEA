"""Validation test for tools/web_search.py — unit tests without live API calls."""
import os, sys, types, asyncio, json

_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _project_root)

from config.settings import Settings

# Need tools.base_tool available for web_search.py's import
import tools.base_tool  # this works — tools/ is a real package on sys.path

# Load web_search.py directly (bypass tools/__init__.py which may have stale imports)
_ws_path = os.path.join(_project_root, "tools", "web_search.py")
_mod = types.ModuleType("tools.web_search")
_mod.__file__ = _ws_path
sys.modules["tools.web_search"] = _mod
with open(_ws_path, encoding="utf-8") as _f:
    exec(compile(_f.read(), _ws_path, "exec"), _mod.__dict__)

WebSearchTool = _mod.WebSearchTool
SearchResult = _mod.SearchResult
_format_results = _mod._format_results
_extract_domain = _mod._extract_domain
_truncate = _mod._truncate

settings = Settings()
tool = WebSearchTool(settings)

print("=== FLEEA tools/web_search.py Validation ===\n")

# ── 1. Interface compliance ──
assert tool.name == "web_search"
assert len(tool.description) > 50
assert tool.requires_approval is False
print("  OK  Interface: name, description, requires_approval")

# ── 2. Schema structure ──
schema = tool.get_schema()
assert schema["name"] == "web_search"
assert "input_schema" in schema
props = schema["input_schema"]["properties"]
assert "query" in props
assert "max_results" in props
assert schema["input_schema"]["required"] == ["query"]
print("  OK  Schema: Claude-compatible structure")

# ── 3. Empty query validation ──
result = asyncio.run(tool.execute(query=""))
assert not result.success
assert "empty" in result.error.lower()

result2 = asyncio.run(tool.execute(query="   "))
assert not result2.success

result3 = asyncio.run(tool.execute())
assert not result3.success
print("  OK  Input validation: empty/whitespace/missing query rejected")

# ── 4. SearchResult dataclass ──
sr = SearchResult(
    title="Test Article",
    summary="This is a test summary.",
    source="example.com",
    url="https://example.com/article",
    confidence=0.95,
)
d = sr.to_dict()
assert d["title"] == "Test Article"
assert d["summary"] == "This is a test summary."
assert d["source"] == "example.com"
assert d["url"] == "https://example.com/article"
assert d["confidence"] == 0.95
print("  OK  SearchResult: dataclass + to_dict()")

# ── 5. Domain extraction ──
test_domains = [
    ("https://www.nytimes.com/2025/article", "nytimes.com"),
    ("https://docs.python.org/3/library", "docs.python.org"),
    ("https://en.wikipedia.org/wiki/AI", "en.wikipedia.org"),
    ("https://www.bbc.co.uk/news", "bbc.co.uk"),
    ("http://example.com", "example.com"),
    ("", ""),
    ("not-a-url", ""),
]
for url, expected in test_domains:
    result = _extract_domain(url)
    assert result == expected, f"Domain extraction failed for '{url}': got '{result}', expected '{expected}'"
print(f"  OK  Domain extraction: {len(test_domains)} cases")

# ── 6. Text truncation ──
assert _truncate("short", 100) == "short"
assert _truncate("a" * 600, 500) == "a" * 497 + "..."
assert len(_truncate("a" * 600, 500)) == 500
assert _truncate("   padded   ", 100) == "padded"
print("  OK  Truncation: short pass-through, long truncated, stripped")

# ── 7. Result formatting with mock Tavily response ──
mock_tavily = {
    "query": "latest AI news",
    "answer": "AI is advancing rapidly in 2025.",
    "results": [
        {
            "title": "AI Breakthrough 2025",
            "url": "https://www.techcrunch.com/ai-breakthrough",
            "content": "Researchers have achieved a major milestone in artificial intelligence...",
            "score": 0.956,
        },
        {
            "title": "OpenAI Launches GPT-5",
            "url": "https://www.theverge.com/openai-gpt5",
            "content": "OpenAI has announced its latest language model...",
            "score": 0.892,
        },
        {
            "title": "",  # Missing title
            "url": "https://example.com/no-title",
            "content": "",  # Empty content
            "score": 0.0,  # Zero score
        },
    ],
}

formatted = _format_results("latest AI news", mock_tavily)
assert formatted["query"] == "latest AI news"
assert formatted["answer"] == "AI is advancing rapidly in 2025."
assert formatted["result_count"] == 3

r0 = formatted["results"][0]
assert r0["title"] == "AI Breakthrough 2025"
assert r0["source"] == "techcrunch.com"  # www. stripped
assert r0["url"] == "https://www.techcrunch.com/ai-breakthrough"
assert r0["confidence"] == 0.956

r2 = formatted["results"][2]
assert r2["title"] == ""  # empty string preserved (not replaced by fallback)
assert r2["source"] == "example.com"
assert r2["confidence"] == 0.0
print("  OK  Result formatting: titles, sources, confidence, edge cases")

# ── 8. Empty results handling ──
empty_tavily = {"query": "obscure topic", "results": []}
formatted_empty = _format_results("obscure topic", empty_tavily)
assert formatted_empty["result_count"] == 0
assert formatted_empty["results"] == []
assert formatted_empty["answer"] == ""
print("  OK  Empty results: handled gracefully")

# ── 9. Invalid result items ──
bad_tavily = {
    "query": "test",
    "results": [
        "not a dict",    # should be skipped
        42,              # should be skipped
        {"title": "Valid", "url": "https://valid.com", "content": "ok", "score": 0.5},
    ],
}
formatted_bad = _format_results("test", bad_tavily)
assert formatted_bad["result_count"] == 1  # only the valid dict survives
print("  OK  Invalid items: non-dict results skipped")

# ── 10. ToolResult.to_content_string() integration ──
from tools.base_tool import ToolResult
tr = ToolResult(success=True, data=formatted)
content = tr.to_content_string()
assert "AI Breakthrough 2025" in content
assert "techcrunch.com" in content
# Verify it's valid JSON
parsed = json.loads(content)
assert parsed["result_count"] == 3

tr_err = ToolResult(success=False, error="API timeout")
assert "Error: API timeout" in tr_err.to_content_string()
print("  OK  ToolResult integration: content string for Claude")

print("\nAll checks passed.")
