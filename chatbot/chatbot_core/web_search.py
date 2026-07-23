from __future__ import annotations

import html
import re
import urllib.parse
import urllib.request
from typing import Any


USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
MAX_MARKDOWN_BYTES = 160_000


def search_web(query: str, limit: int = 4) -> list[dict[str, str]]:
    text = (query or "").strip()
    if not text:
        return []
    enriched = f"{text} 网络梗 解释 来源"
    providers = [
        ("sogou", "https://r.jina.ai/http://www.sogou.com/web?" + urllib.parse.urlencode({"query": enriched})),
        ("bing", "https://r.jina.ai/http://www.bing.com/search?" + urllib.parse.urlencode({"q": enriched})),
    ]
    results: list[dict[str, str]] = []
    seen: set[str] = set()
    for provider, url in providers:
        try:
            markdown = fetch_markdown(url)
        except Exception:
            continue
        for row in parse_markdown_results(markdown, provider):
            key = normalize_key(row.get("title", ""), row.get("url", ""))
            if not key or key in seen:
                continue
            seen.add(key)
            results.append(row)
            if len(results) >= limit:
                return results
    return results


def fetch_markdown(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=18) as response:
        return response.read(MAX_MARKDOWN_BYTES).decode("utf-8", errors="ignore")


def parse_markdown_results(markdown: str, provider: str) -> list[dict[str, str]]:
    rows = parse_heading_results(markdown, provider)
    if rows:
        return rows
    return parse_related_queries(markdown, provider)


def parse_heading_results(markdown: str, provider: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    pattern = re.compile(r"^#{2,3}\s+\[(.*?)\]\((.*?)\)\s*$", re.M)
    matches = list(pattern.finditer(markdown or ""))
    for index, match in enumerate(matches):
        title = clean_markdown(match.group(1))
        url = clean_url(match.group(2))
        if not useful_title(title):
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else min(len(markdown), start + 900)
        snippet = clean_snippet(markdown[start:end])
        if not snippet and not title:
            continue
        rows.append({"provider": provider, "title": title, "url": url, "snippet": snippet})
        if len(rows) >= 5:
            break
    return rows


def parse_related_queries(markdown: str, provider: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    related_start = markdown.find("相关推荐")
    source = markdown[related_start:] if related_start >= 0 else markdown
    pattern = re.compile(r"\[([^\[\]]{2,40})\]\((https?://[^)]+)\)")
    for title, url in pattern.findall(source):
        title = clean_markdown(title)
        if not useful_title(title):
            continue
        rows.append(
            {
                "provider": provider,
                "title": title,
                "url": clean_url(url),
                "snippet": "搜索相关问题，说明这个词可能需要按网络语境理解。",
            }
        )
        if len(rows) >= 5:
            break
    return rows


def web_context_for_prompt(results: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "source": "live_web_search",
        "result_count": len(results),
        "results": [
            {
                "provider": row.get("provider", ""),
                "title": row.get("title", "")[:80],
                "snippet": row.get("snippet", "")[:220],
                "url": row.get("url", "")[:220],
            }
            for row in results[:4]
        ],
        "rule": "把这些当作临时联网证据。证据弱时说像是/可能是，不要硬编成确定梗源。",
    }


def clean_markdown(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_snippet(text: str) -> str:
    lines = []
    for raw in (text or "").splitlines():
        line = clean_markdown(raw)
        if not line or line.startswith("[") or line.startswith("*"):
            continue
        if any(skip in line.lower() for skip in ["javascript:", "privacy", "terms of use", "skip to content"]):
            continue
        lines.append(line)
        if len(" ".join(lines)) >= 240:
            break
    return " ".join(lines)[:260]


def clean_url(url: str) -> str:
    text = html.unescape(url or "").strip()
    parsed = urllib.parse.urlparse(text)
    query = urllib.parse.parse_qs(parsed.query)
    if parsed.netloc.endswith("bing.com") and "u" in query:
        encoded = query["u"][0]
        if encoded.startswith("a1"):
            try:
                return urllib.parse.unquote(encoded[2:])
            except Exception:
                return text
    return text


def useful_title(title: str) -> bool:
    lowered = (title or "").lower()
    if len(title.strip()) < 2:
        return False
    blocked = [
        "youtube",
        "google accounts",
        "apps on google play",
        "privacy policy",
        "terms of use",
        "search",
        "images",
        "videos",
    ]
    return not any(item in lowered for item in blocked)


def normalize_key(title: str, url: str) -> str:
    title_key = re.sub(r"\s+", "", (title or "").lower())
    host = urllib.parse.urlparse(url or "").netloc.lower()
    return f"{host}:{title_key}"[:160]
