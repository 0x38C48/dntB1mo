from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import ENGINE  # noqa: E402


QUESTION_CUES = ("?", "？", "吗", "嘛", "什么", "啥", "怎么", "哪", "几")
GENERIC_QUESTIONS = {
    "你说呢",
    "怎么了",
    "又问",
    "说啊",
    "干嘛",
    "咋了",
    "什么",
    "你说",
    "啊？",
}
BAD_PHRASES = (
    "你继续",
    "看记录",
    "聊天记录",
    "检索",
    "证据",
    "模型",
    "机器人",
    "我换个说法",
    "换个说法",
    "稳不稳",
    "能不能看到",
)


@dataclass
class EvalCase:
    pair_id: str
    timestamp: str
    history: list[dict[str, str]]
    user_message: str
    expected: str


def iter_pairs(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def trailing_user_turn(context: list[dict[str, Any]]) -> tuple[list[dict[str, str]], str] | None:
    if not context:
        return None
    idx = len(context) - 1
    trailing: list[str] = []
    while idx >= 0 and context[idx].get("role") == "user":
        text = str(context[idx].get("text") or "").strip()
        if text and text != "[表情]":
            trailing.append(text)
        idx -= 1
    if not trailing:
        return None
    trailing.reverse()
    history: list[dict[str, str]] = []
    for item in context[: idx + 1]:
        text = str(item.get("text") or "").strip()
        if not text or text == "[表情]":
            continue
        role = "assistant" if item.get("role") == "target" else "user"
        history.append({"role": role, "content": text})
    return history[-48:], "\n".join(trailing[-4:])


def case_from_pair(pair: dict[str, Any]) -> EvalCase | None:
    response = pair.get("assistant_response") or {}
    expected = str(response.get("text") or "").strip()
    if response.get("content_type") != "text" or not expected or expected == "[表情]":
        return None
    turn = trailing_user_turn(pair.get("context") or [])
    if not turn:
        return None
    history, user_message = turn
    if not user_message or len(user_message) > 80:
        return None
    return EvalCase(
        pair_id=str(pair.get("pair_id") or ""),
        timestamp=str(pair.get("timestamp") or ""),
        history=history,
        user_message=user_message,
        expected=expected,
    )


def bucket_name(case: EvalCase) -> str:
    msg = case.user_message
    if "\n" in msg:
        return "multi_user"
    if any(cue in msg for cue in QUESTION_CUES):
        return "question"
    if any(cue in msg for cue in ["吃", "饭", "困", "睡", "醒", "游戏", "原神", "星铁", "玩"]):
        return "food_sleep_game"
    if any(cue in msg for cue in ["什么东西", "说什么", "啥玩意", "不是"]):
        return "repair_like"
    if len(msg) <= 3:
        return "short"
    return "other"


def build_cases(path: Path, limit: int, seed: int, scan_limit: int) -> list[EvalCase]:
    buckets: dict[str, list[EvalCase]] = {
        "short": [],
        "multi_user": [],
        "question": [],
        "food_sleep_game": [],
        "repair_like": [],
        "other": [],
    }
    rng = random.Random(seed)
    seen = 0
    max_per_bucket = max(4, limit)
    for pair in iter_pairs(path):
        seen += 1
        if scan_limit and seen > scan_limit:
            break
        case = case_from_pair(pair)
        if not case:
            continue
        bucket = bucket_name(case)
        values = buckets[bucket]
        if len(values) < max_per_bucket:
            values.append(case)
        else:
            replacement = rng.randrange(seen)
            if replacement < max_per_bucket:
                values[replacement] = case

    selected: list[EvalCase] = []
    per_bucket = max(1, limit // len(buckets))
    for values in buckets.values():
        rng.shuffle(values)
        selected.extend(values[:per_bucket])
    if len(selected) < limit:
        rest: list[EvalCase] = []
        for values in buckets.values():
            rest.extend(case for case in values if case not in selected)
        rng.shuffle(rest)
        selected.extend(rest[: limit - len(selected)])
    return selected[:limit]


def normalize(text: str) -> str:
    return re.sub(r"[\s?？!！。，、…~]+", "", text or "")


def has_question(text: str) -> bool:
    return any(cue in text for cue in QUESTION_CUES)


def length_bucket(text: str) -> str:
    n = len(re.sub(r"\s+", "", text or ""))
    if n <= 3:
        return "micro"
    if n <= 8:
        return "short"
    if n <= 18:
        return "medium"
    return "long"


def judge_case(case: EvalCase, reply: str, mode: str, elapsed: float) -> dict[str, Any]:
    expected_norm = normalize(case.expected)
    reply_norm = normalize(reply)
    ratio = SequenceMatcher(None, expected_norm, reply_norm).ratio() if expected_norm and reply_norm else 0.0
    reply_lines = [line.strip() for line in reply.splitlines() if line.strip()]
    expected_lines = [line.strip() for line in case.expected.splitlines() if line.strip()]
    issues: list[str] = []

    if any(phrase in reply for phrase in BAD_PHRASES):
        issues.append("bad_meta_or_hard_fallback")
    if len(reply_lines) > 3:
        issues.append("too_many_bubbles")
    if length_bucket(case.expected) in {"micro", "short"} and length_bucket(reply) == "long":
        issues.append("too_long_vs_record")
    if not has_question(case.expected) and has_question(reply) and normalize(reply) in {normalize(x) for x in GENERIC_QUESTIONS}:
        issues.append("generic_question_deflection")
    if not has_question(case.expected) and has_question(reply) and len(case.user_message) > 3:
        issues.append("asks_when_record_answers")
    if len(set(reply_lines)) < len(reply_lines):
        issues.append("duplicate_bubbles")
    if "local_fallback" in mode:
        issues.append("api_fallback")
    if elapsed < 0.25 and mode == "sophnet_chat_completions":
        issues.append("suspicious_fast_api")

    return {
        "pair_id": case.pair_id,
        "timestamp": case.timestamp,
        "user_message": case.user_message,
        "expected": case.expected,
        "reply": reply,
        "mode": mode,
        "elapsed_sec": round(elapsed, 3),
        "similarity": round(ratio, 3),
        "expected_bucket": length_bucket(case.expected),
        "reply_bucket": length_bucket(reply),
        "expected_bubbles": len(expected_lines),
        "reply_bubbles": len(reply_lines),
        "issues": issues,
    }


def run_eval(limit: int, seed: int, out_dir: Path, scan_limit: int) -> Path:
    cases = build_cases(ROOT.parent / "wechat_prepared" / "sft_pairs.jsonl", limit, seed, scan_limit)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"self_eval_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    print(f"cases={len(cases)} scan_limit={scan_limit}", flush=True)
    with report_path.open("w", encoding="utf-8") as handle:
        for index, case in enumerate(cases, 1):
            print(f"{index:03d} start {case.pair_id}", flush=True)
            started = time.perf_counter()
            result = ENGINE.reply(case.user_message, case.history, [], "auto")
            reply = ENGINE.apply_consecutive_style(
                str(result.get("reply") or ""),
                case.user_message,
                case.history,
                str(result.get("mode") or ""),
                result.get("memories") or [],
                str(result.get("emotion") or "casual"),
            )
            elapsed = time.perf_counter() - started
            row = judge_case(case, reply, str(result.get("mode") or ""), elapsed)
            row["index"] = index
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"{index:03d} {row['mode']} {row['elapsed_sec']}s "
                f"sim={row['similarity']} issues={','.join(row['issues']) or '-'}",
                flush=True,
            )
    return report_path


def summarize(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    issue_counts: dict[str, int] = {}
    mode_counts: dict[str, int] = {}
    for row in rows:
        mode_counts[row["mode"]] = mode_counts.get(row["mode"], 0) + 1
        for issue in row["issues"]:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
    worst = sorted(
        rows,
        key=lambda row: (len(row["issues"]), -row["similarity"], row["reply_bucket"] == "long"),
        reverse=True,
    )[:12]
    return {
        "count": len(rows),
        "issue_counts": issue_counts,
        "mode_counts": mode_counts,
        "avg_similarity": round(sum(row["similarity"] for row in rows) / max(1, len(rows)), 3),
        "worst": worst,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scan-limit", type=int, default=12000)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "eval_reports")
    args = parser.parse_args()
    path = run_eval(args.limit, args.seed, args.out_dir, args.scan_limit)
    summary = summarize(path)
    summary_path = path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"REPORT={path}")
    print(f"SUMMARY={summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:5000])


if __name__ == "__main__":
    main()
