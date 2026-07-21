#!/usr/bin/env python3
"""
Normalize exported WeChat HTML-in-JS records into chatbot-ready datasets.

The exporter stores pages as msg-*.js files containing a JavaScript array of
HTML snippets. This script parses those snippets without executing JavaScript.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote


TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
MSG_RE = re.compile(r'<div class="msg\s+([^"]*?)"\s+msgid="([^"]+)"(?:\s+msgtype="([^"]+)")?')
SPEAKER_RE = re.compile(
    r'<span class="dspname\s+(?:left|right)"\s+wxId="([^"]*)">(.*?)</span>\s*'
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})",
    re.S,
)
PRE_RE = re.compile(r"<pre>(.*?)</pre>", re.S)
ATTR_RE = re.compile(r'\b(rawUrl|src|href)="([^"]+)"')

PII_PATTERNS = {
    "phone": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    "email": re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    "id_card": re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
}

SENSITIVE_KEYWORDS = {
    "intimacy": ["性", "av", "喷", "摸", "黄", "涩", "本子"],
    "mental_health": ["抑郁", "焦虑", "自杀", "想死", "崩溃"],
    "location": ["地址", "宿舍", "学校", "深圳", "家里"],
}


def extract_js_array(source: str) -> list[str]:
    marker = "var msgArray"
    marker_pos = source.find(marker)
    if marker_pos < 0:
        return []
    start = source.find("[", marker_pos)
    if start < 0:
        return []

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(source)):
        char = source[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return json.loads(source[start : idx + 1])
    raise ValueError("Could not find the end of msgArray")


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    value = html.unescape(value)
    value = re.sub(r"<br\s*/?>", "\n", value, flags=re.I)
    value = re.sub(r"<[^>]+>", "", value)
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_asset_path(raw: str, records_dir: Path) -> str:
    decoded = unquote(html.unescape(raw))
    name = Path(decoded.replace("\\", "/")).name
    if name:
        candidate = records_dir / name
        if candidate.exists():
            return str(candidate)
    return decoded


def classify_message(class_tokens: list[str], msgtype: str | None, text: str, assets: list[str]) -> str:
    if "chat-notice" in class_tokens:
        return "notice"
    if "media" in class_tokens:
        if msgtype == "image" or any(a.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif")) for a in assets):
            return "image"
        if any(a.lower().endswith((".mp4", ".mov", ".avi")) for a in assets):
            return "video"
        if any(a.lower().endswith((".mp3", ".wav", ".amr", ".silk")) for a in assets):
            return "audio"
        return "media"
    if msgtype == "47" or text == "[表情]":
        return "sticker"
    if msgtype == "1":
        return "text"
    return msgtype or "unknown"


def detect_flags(text: str) -> list[str]:
    flags: set[str] = set()
    for name, pattern in PII_PATTERNS.items():
        if pattern.search(text):
            flags.add(f"pii:{name}")
    lower = text.lower()
    for name, keywords in SENSITIVE_KEYWORDS.items():
        if any(keyword.lower() in lower for keyword in keywords):
            flags.add(f"sensitive:{name}")
    return sorted(flags)


def parse_message(fragment: str, records_dir: Path) -> dict[str, Any] | None:
    msg_match = MSG_RE.search(fragment)
    if not msg_match:
        return None

    class_tokens = [token for token in msg_match.group(1).split() if token]
    side = "left" if "left" in class_tokens else "right" if "right" in class_tokens else "system"
    speaker_match = SPEAKER_RE.search(fragment)
    pre_match = PRE_RE.search(fragment)
    attr_values = [m.group(2) for m in ATTR_RE.finditer(fragment)]
    assets = sorted({normalize_asset_path(value, records_dir) for value in attr_values if "_files/" in unquote(value)})
    text = clean_text(pre_match.group(1) if pre_match else "")

    timestamp = None
    wxid = None
    display_name = None
    if speaker_match:
        wxid = html.unescape(speaker_match.group(1))
        display_name = clean_text(speaker_match.group(2))
        timestamp = speaker_match.group(3)
    else:
        ts_match = TIMESTAMP_RE.search(fragment)
        timestamp = ts_match.group(0) if ts_match else None

    msgtype = msg_match.group(3)
    content_type = classify_message(class_tokens, msgtype, text, assets)
    speaker_role = "system"
    if side == "right":
        speaker_role = "user"
    elif side == "left":
        speaker_role = "target"

    return {
        "message_id": int(msg_match.group(2)),
        "timestamp": timestamp,
        "side": side,
        "speaker_role": speaker_role,
        "wxid": wxid,
        "display_name": display_name,
        "raw_msgtype": msgtype,
        "content_type": content_type,
        "text": text,
        "assets": assets,
        "safety_flags": detect_flags(text),
    }


def iter_messages(records_dir: Path) -> list[dict[str, Any]]:
    data_dir = records_dir / "Data"
    messages: list[dict[str, Any]] = []
    for file_path in sorted(data_dir.glob("msg-*.js"), key=lambda p: int(re.search(r"msg-(\d+)", p.name).group(1))):
        source = file_path.read_text(encoding="utf-8")
        for fragment in extract_js_array(source):
            parsed = parse_message(fragment, records_dir)
            if parsed:
                parsed["source_file"] = file_path.name
                messages.append(parsed)
    messages.sort(key=lambda m: (m["timestamp"] or "", m["message_id"]))
    return messages


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_chunks(messages: list[dict[str, Any]], max_gap_minutes: int = 30, max_messages: int = 40) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    last_time: datetime | None = None

    def flush() -> None:
        if not current:
            return
        text_lines = [
            f"{m['speaker_role']}[{m['content_type']}]: {m['text'] or '[media]'}"
            for m in current
            if m["content_type"] != "notice"
        ]
        chunk_text = "\n".join(text_lines)
        chunks.append(
            {
                "chunk_id": f"chunk_{len(chunks) + 1:06d}",
                "start_message_id": current[0]["message_id"],
                "end_message_id": current[-1]["message_id"],
                "start_time": current[0]["timestamp"],
                "end_time": current[-1]["timestamp"],
                "message_count": len(current),
                "target_message_count": sum(1 for m in current if m["speaker_role"] == "target"),
                "user_message_count": sum(1 for m in current if m["speaker_role"] == "user"),
                "content_types": dict(Counter(m["content_type"] for m in current)),
                "safety_flags": sorted({flag for m in current for flag in m["safety_flags"]}),
                "text": chunk_text,
                "embedding_text": chunk_text[:6000],
            }
        )

    for msg in messages:
        current_time = datetime.strptime(msg["timestamp"], "%Y-%m-%d %H:%M:%S") if msg["timestamp"] else None
        gap_too_large = False
        if last_time and current_time:
            gap_too_large = (current_time - last_time).total_seconds() > max_gap_minutes * 60
        if current and (gap_too_large or len(current) >= max_messages):
            flush()
            current = []
        current.append(msg)
        if current_time:
            last_time = current_time
    flush()
    return chunks


def build_sft_pairs(messages: list[dict[str, Any]], context_window: int = 12) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for idx, msg in enumerate(messages):
        if msg["speaker_role"] != "target" or msg["content_type"] not in {"text", "sticker"}:
            continue
        if not msg["text"] and msg["content_type"] != "sticker":
            continue
        context = [
            {
                "role": m["speaker_role"],
                "content_type": m["content_type"],
                "text": m["text"],
                "timestamp": m["timestamp"],
            }
            for m in messages[max(0, idx - context_window) : idx]
            if m["speaker_role"] in {"user", "target"} and m["content_type"] in {"text", "sticker"}
        ]
        if not any(m["role"] == "user" for m in context):
            continue
        pairs.append(
            {
                "pair_id": f"sft_{len(pairs) + 1:06d}",
                "target_message_id": msg["message_id"],
                "timestamp": msg["timestamp"],
                "context": context,
                "assistant_response": {
                    "content_type": msg["content_type"],
                    "text": msg["text"] if msg["content_type"] == "text" else "[表情]",
                },
                "safety_flags": sorted({flag for m in messages[max(0, idx - context_window) : idx + 1] for flag in m["safety_flags"]}),
            }
        )
    return pairs


def build_sticker_events(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for idx, msg in enumerate(messages):
        if msg["content_type"] != "sticker":
            continue
        prev = [
            {
                "speaker_role": m["speaker_role"],
                "content_type": m["content_type"],
                "text": m["text"],
                "timestamp": m["timestamp"],
            }
            for m in messages[max(0, idx - 6) : idx]
            if m["speaker_role"] in {"user", "target"} and m["content_type"] in {"text", "sticker"}
        ]
        events.append(
            {
                "event_id": f"sticker_{len(events) + 1:06d}",
                "message_id": msg["message_id"],
                "timestamp": msg["timestamp"],
                "speaker_role": msg["speaker_role"],
                "context": prev,
                "assets": msg["assets"],
                "inferred_intent": None,
            }
        )
    return events


def inventory_assets(records_dir: Path) -> dict[str, Any]:
    files = [p for p in records_dir.rglob("*") if p.is_file()]
    ext_counts = Counter(p.suffix.lower() or "[none]" for p in files)
    folders = {}
    for folder in ["Emoji", "Portrait"]:
        folder_path = records_dir / folder
        folders[folder] = {
            "exists": folder_path.exists(),
            "file_count": sum(1 for p in folder_path.rglob("*") if p.is_file()) if folder_path.exists() else 0,
        }
    return {
        "total_files": len(files),
        "extension_counts": dict(sorted(ext_counts.items())),
        "folders": folders,
    }


def build_manifest(records_dir: Path, messages: list[dict[str, Any]], chunks: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    speakers = defaultdict(lambda: {"count": 0, "roles": Counter(), "display_names": Counter()})
    for msg in messages:
        key = msg["wxid"] or msg["speaker_role"]
        speakers[key]["count"] += 1
        speakers[key]["roles"][msg["speaker_role"]] += 1
        if msg["display_name"]:
            speakers[key]["display_names"][msg["display_name"]] += 1

    return {
        "source_dir": str(records_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "message_count": len(messages),
        "date_range": {
            "start": next((m["timestamp"] for m in messages if m["timestamp"]), None),
            "end": next((m["timestamp"] for m in reversed(messages) if m["timestamp"]), None),
        },
        "content_type_counts": dict(Counter(m["content_type"] for m in messages)),
        "speaker_role_counts": dict(Counter(m["speaker_role"] for m in messages)),
        "safety_flag_counts": dict(Counter(flag for m in messages for flag in m["safety_flags"])),
        "speakers": {
            key: {
                "count": value["count"],
                "roles": dict(value["roles"]),
                "display_names": dict(value["display_names"]),
            }
            for key, value in speakers.items()
        },
        "chunk_count": len(chunks),
        "sft_pair_count": len(pairs),
        "asset_inventory": inventory_assets(records_dir),
        "assumptions": [
            "Messages on the right side are treated as user.",
            "Messages on the left side are treated as target persona.",
            "System notices are retained in messages.jsonl but excluded from RAG/SFT text by default.",
        ],
    }


def write_readme(path: Path) -> None:
    path.write_text(
        """# WeChat Prepared Dataset

This folder contains normalized exports for building a consent-based chatbot.

Files:

- `messages.jsonl`: one normalized message per line. This is the source of truth.
- `messages.csv`: spreadsheet-friendly overview of the same messages.
- `rag_chunks.jsonl`: conversation chunks for embeddings/vector search.
- `sft_pairs.jsonl`: target-speaker response examples for later supervised fine-tuning or few-shot evaluation.
- `sticker_events.jsonl`: sticker-send events with recent context for future sticker intent labeling.
- `manifest.json`: counts, date range, speakers, assumptions, and asset inventory.

Role assumptions:

- `right` side -> `user`
- `left` side -> `target`
- notices -> `system`

Recommended next steps:

1. Review `manifest.json` to confirm the target/user mapping.
2. Manually inspect a small sample before using the data for modeling.
3. Add consent, deletion, and privacy controls before any product use.
4. Generate embeddings from `rag_chunks.jsonl`.
5. Use `sft_pairs.jsonl` for evaluation first; only fine-tune after the RAG version works.
""",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    records_dir = args.records_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    messages = iter_messages(records_dir)
    chunks = build_chunks(messages)
    pairs = build_sft_pairs(messages)
    sticker_events = build_sticker_events(messages)
    manifest = build_manifest(records_dir, messages, chunks, pairs)

    write_jsonl(out_dir / "messages.jsonl", messages)
    write_jsonl(out_dir / "rag_chunks.jsonl", chunks)
    write_jsonl(out_dir / "sft_pairs.jsonl", pairs)
    write_jsonl(out_dir / "sticker_events.jsonl", sticker_events)

    with (out_dir / "messages.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "message_id",
                "timestamp",
                "speaker_role",
                "side",
                "wxid",
                "display_name",
                "content_type",
                "text",
                "assets",
                "safety_flags",
                "source_file",
            ],
        )
        writer.writeheader()
        for msg in messages:
            row = dict(msg)
            row["assets"] = json.dumps(row["assets"], ensure_ascii=False)
            row["safety_flags"] = json.dumps(row["safety_flags"], ensure_ascii=False)
            writer.writerow({key: row.get(key) for key in writer.fieldnames})

    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(out_dir / "README.md")

    digest = hashlib.sha256(json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    print(json.dumps({"out_dir": str(out_dir), "message_count": len(messages), "chunk_count": len(chunks), "sft_pair_count": len(pairs), "digest": digest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
