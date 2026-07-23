from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, time, timedelta
from pathlib import Path
from statistics import median
from typing import Any

from .config import AppConfig
from .dataset import Dataset


TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
STYLE_ROLE = "user"
OTHER_ROLE = "target"
FAST_REPLY_SECONDS = 10 * 60
SLOW_REPLY_SECONDS = 60 * 60
TOPIC_GAP_SECONDS = 45 * 60
SLEEP_START = time(3, 0)
SLEEP_END = time(11, 0)
BEHAVIOR_VERSION = "0.3"


def load_or_build_behavior(config: AppConfig, dataset: Dataset) -> dict[str, Any]:
    path = config.persona_dir / "behavior_analysis.json"
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("version") == BEHAVIOR_VERSION:
            return cached
    analysis = analyze_behavior(dataset)
    config.persona_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    (config.persona_dir / "behavior_analysis.md").write_text(render_behavior_md(analysis), encoding="utf-8", newline="\n")
    return analysis


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, TIME_FORMAT)
    except ValueError:
        return None


def text_of(msg: dict[str, Any]) -> str:
    text = (msg.get("text") or "").strip()
    if text:
        return text
    return f"[{msg.get('content_type') or 'media'}]"


def hour_bucket(ts: datetime | None) -> str:
    if ts is None:
        return "unknown"
    hour = ts.hour
    if 0 <= hour < 6:
        return "late_night"
    if 6 <= hour < 12:
        return "morning"
    if 12 <= hour < 18:
        return "afternoon"
    return "evening"


def gap_label(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 60:
        return "<1m"
    if seconds < 5 * 60:
        return "1-5m"
    if seconds < 30 * 60:
        return "5-30m"
    if seconds < 2 * 3600:
        return "30m-2h"
    if seconds < 8 * 3600:
        return "2-8h"
    return ">8h"


def active_gap_seconds(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None
    if end <= start:
        return 0.0
    total = (end - start).total_seconds()
    sleeping = sleeping_overlap_seconds(start, end)
    return max(0.0, total - sleeping)


def sleeping_overlap_seconds(start: datetime, end: datetime) -> float:
    total = 0.0
    current_day = start.date()
    while current_day <= end.date():
        sleep_start = datetime.combine(current_day, SLEEP_START)
        sleep_end = datetime.combine(current_day, SLEEP_END)
        overlap_start = max(start, sleep_start)
        overlap_end = min(end, sleep_end)
        if overlap_end > overlap_start:
            total += (overlap_end - overlap_start).total_seconds()
        current_day += timedelta(days=1)
    return total


def analyze_behavior(dataset: Dataset) -> dict[str, Any]:
    messages = [
        msg
        for msg in dataset.iter_messages()
        if msg.get("speaker_role") in {STYLE_ROLE, OTHER_ROLE}
        and msg.get("content_type") in {"text", "sticker", "image", "49"}
        and parse_time(msg.get("timestamp")) is not None
    ]
    messages.sort(key=lambda msg: (msg.get("timestamp") or "", msg.get("message_id") or 0))

    initiations: list[dict[str, Any]] = []
    target_reply_outcomes: list[dict[str, Any]] = []
    topic_gap_values: list[float] = []
    hour_counts: Counter[str] = Counter()
    no_reply_counter: Counter[str] = Counter()
    no_reply_examples: list[dict[str, Any]] = []

    prev_msg: dict[str, Any] | None = None
    for idx, msg in enumerate(messages):
        role = msg.get("speaker_role")
        ts = parse_time(msg.get("timestamp"))
        prev_ts = parse_time(prev_msg.get("timestamp")) if prev_msg else None
        raw_gap = (ts - prev_ts).total_seconds() if ts and prev_ts else None
        gap = active_gap_seconds(prev_ts, ts)

        if role == STYLE_ROLE and (prev_msg is None or (gap is not None and gap >= TOPIC_GAP_SECONDS)):
            initiations.append(
                {
                    "message_id": msg.get("message_id"),
                    "timestamp": msg.get("timestamp"),
                    "active_gap_seconds": gap,
                    "raw_gap_seconds": raw_gap,
                    "gap_label": gap_label(gap),
                    "raw_gap_label": gap_label(raw_gap),
                    "hour_bucket": hour_bucket(ts),
                    "content_type": msg.get("content_type"),
                    "text": text_of(msg)[:80],
                }
            )
            if gap is not None:
                topic_gap_values.append(gap)
            hour_counts[hour_bucket(ts)] += 1

        if role == OTHER_ROLE:
            next_style = None
            for later in messages[idx + 1 : min(len(messages), idx + 80)]:
                if later.get("speaker_role") == STYLE_ROLE:
                    next_style = later
                    break
                later_ts = parse_time(later.get("timestamp"))
                if later_ts and ts and (later_ts - ts).total_seconds() > 8 * 3600:
                    break
            delay = None
            if next_style:
                next_ts = parse_time(next_style.get("timestamp"))
                delay = (next_ts - ts).total_seconds() if next_ts and ts else None
            outcome = "fast_reply" if delay is not None and delay <= FAST_REPLY_SECONDS else "slow_or_no_reply"
            if delay is None or delay > SLOW_REPLY_SECONDS:
                source_text = text_of(msg)
                category = classify_ignorable_source(msg, source_text)
                no_reply_counter[category] += 1
                if len(no_reply_examples) < 24 and source_text and source_text != "[sticker]":
                    no_reply_examples.append(
                        {
                            "category": category,
                            "timestamp": msg.get("timestamp"),
                            "source_text": source_text[:80],
                            "delay_label": gap_label(delay),
                        }
                    )
            target_reply_outcomes.append(
                {
                    "message_id": msg.get("message_id"),
                    "timestamp": msg.get("timestamp"),
                    "content_type": msg.get("content_type"),
                    "text_len": len(text_of(msg)),
                    "reply_delay_seconds": delay,
                    "outcome": outcome,
                }
            )
        prev_msg = msg

    delays = [x["reply_delay_seconds"] for x in target_reply_outcomes if x["reply_delay_seconds"] is not None]
    slow_or_no = sum(1 for x in target_reply_outcomes if x["outcome"] == "slow_or_no_reply")
    fast = sum(1 for x in target_reply_outcomes if x["outcome"] == "fast_reply")

    return {
        "version": BEHAVIOR_VERSION,
        "style_role": STYLE_ROLE,
        "style_aliases": ["backup", "ZyjT82011"],
        "topic_initiation": {
            "definition": "backup/user message that starts after >=45 minutes of active silence; daily sleep window 03:00-11:00 is excluded from gap time",
            "sleep_window_excluded": "03:00-11:00",
            "count": len(initiations),
            "median_active_gap_minutes": round(median(topic_gap_values) / 60, 1) if topic_gap_values else None,
            "median_gap_minutes": round(median(topic_gap_values) / 60, 1) if topic_gap_values else None,
            "gap_distribution": dict(Counter(item["gap_label"] for item in initiations)),
            "hour_distribution": dict(hour_counts),
            "examples": initiations[:20],
        },
        "response_to_other_side": {
            "definition": "whether backup/user replies after target/other-side messages",
            "fast_reply_count": fast,
            "slow_or_no_reply_count": slow_or_no,
            "slow_or_no_reply_ratio": round(slow_or_no / max(1, len(target_reply_outcomes)), 3),
            "median_reply_delay_minutes": round(median(delays) / 60, 1) if delays else None,
            "no_reply_categories": dict(no_reply_counter.most_common()),
            "no_reply_examples": no_reply_examples,
        },
        "runtime_hints": [
            "Do not answer every incoming message if it is a low-content acknowledgement, sticker-only, or topic-ending fragment.",
            "When proactively starting a topic, prefer short hooks, questions, or abrupt topic shifts.",
            "Single-symbol replies are allowed but should not dominate ordinary conversation.",
        ],
    }


def classify_ignorable_source(msg: dict[str, Any], text: str) -> str:
    ctype = msg.get("content_type")
    if ctype == "sticker":
        return "sticker_only"
    if ctype in {"image", "49"}:
        return "media_or_link"
    stripped = text.strip()
    if len(stripped) <= 2:
        return "very_short"
    if stripped in {"嗯", "好", "哦", "啊", "？", "?", "。", "6", "哈哈", "hhh"}:
        return "ack_or_reaction"
    if any(token in stripped for token in ["安安", "睡了", "晚安", "不说了", "算了"]):
        return "conversation_closing"
    if len(stripped) <= 6:
        return "short_fragment"
    return "other"


def render_behavior_md(analysis: dict[str, Any]) -> str:
    topic = analysis["topic_initiation"]
    response = analysis["response_to_other_side"]
    lines = [
        "# Behavior Analysis",
        "",
        "Primary style role: backup/user",
        "",
        "## Topic Initiation",
        "",
        f"- Count: {topic['count']}",
        f"- Median active gap: {topic.get('median_active_gap_minutes', topic.get('median_gap_minutes'))} minutes",
        f"- Sleep window excluded: {topic.get('sleep_window_excluded', '03:00-11:00')}",
        f"- Gap distribution: {topic['gap_distribution']}",
        f"- Hour distribution: {topic['hour_distribution']}",
        "",
        "## Slow Or No Reply",
        "",
        f"- Slow/no reply ratio: {response['slow_or_no_reply_ratio']}",
        f"- Median reply delay: {response['median_reply_delay_minutes']} minutes",
        f"- Categories: {response['no_reply_categories']}",
        "",
    ]
    return "\n".join(lines)
