from __future__ import annotations

import json
import re
from collections import Counter
from statistics import median
from typing import Any

from .config import AppConfig
from .dataset import Dataset
from .textfix import fix_text


TONE_VERSION = "0.3"
DEFAULT_STYLE_ROLE = "user"
BLOCKED_TOP_PHRASES = {
    "嗯",
    "嗯嗯",
    "?",
    "？",
    "啊",
    "啊？",
    "嗯？",
    "好",
    "哼",
    "你继续",
    "然后呢",
    "怎么",
}

TONE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "sleepy": {
        "label": "困/低能量",
        "cues": ["困", "睡", "晚安", "安安", "熬夜", "累", "起不来", "没睡醒", "睡醒", "困告"],
    },
    "annoyed": {
        "label": "烦/嫌弃/边界",
        "cues": ["别", "不要", "烦", "服了", "逆天", "无语", "讨厌", "滚", "冷暴力", "闭嘴", "离谱"],
    },
    "soft": {
        "label": "安抚/照顾",
        "cues": ["抱抱", "摸摸", "乖", "别难受", "别生气", "没事", "早点休息", "多睡", "吃饭", "心疼"],
    },
    "playful": {
        "label": "接梗/玩笑",
        "cues": ["笑死", "哈哈", "绷", "乐", "6", "666", "嘻嘻", "可爱", "草", "xswl"],
    },
    "excited": {
        "label": "惊讶/兴奋",
        "cues": ["我靠", "卧槽", "wc", "哇", "牛", "真的假的", "好耶", "绝了"],
    },
    "curious": {
        "label": "疑问/追问",
        "cues": ["什么", "啥", "怎么", "为什么", "哪", "几", "吗", "？", "?"],
    },
    "engaged": {
        "label": "连续接话",
        "cues": ["然后", "所以", "确实", "那", "但是", "不过", "啊"],
    },
    "casual": {
        "label": "普通闲聊",
        "cues": [],
    },
}


def load_or_build_tone_profiles(config: AppConfig, dataset: Dataset) -> dict[str, Any]:
    path = config.persona_dir / "tone_profiles.json"
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("version") == TONE_VERSION:
            return cached
    profiles = build_tone_profiles(config, dataset)
    config.persona_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8")
    return profiles


def build_tone_profiles(config: AppConfig, dataset: Dataset) -> dict[str, Any]:
    style_role = read_style_role(config)
    buckets: dict[str, list[str]] = {tone: [] for tone in TONE_DEFINITIONS}
    all_style_lines: list[str] = []
    consecutive_runs: list[int] = []
    current_run = 0
    previous_role = ""

    for msg in dataset.iter_messages():
        role = str(msg.get("speaker_role") or msg.get("role") or "")
        if role == style_role:
            current_run = current_run + 1 if previous_role == style_role else 1
        elif current_run:
            consecutive_runs.append(current_run)
            current_run = 0
        previous_role = role

        if role != style_role or msg.get("content_type") != "text":
            continue
        text = normalize_line(msg.get("text"))
        if not is_usable_line(text):
            continue
        all_style_lines.append(text)
        tones = classify_line_tones(text)
        for tone in tones:
            buckets[tone].append(text)

    if current_run:
        consecutive_runs.append(current_run)

    tone_payload = {
        tone: summarize_tone(tone, lines, all_style_lines)
        for tone, lines in buckets.items()
    }
    return {
        "version": TONE_VERSION,
        "source": "wechat_prepared/messages.jsonl",
        "style_role": style_role,
        "record_message_count": len(all_style_lines),
        "consecutive_style": {
            "run_count": len(consecutive_runs),
            "multi_run_ratio": round(sum(1 for value in consecutive_runs if value >= 2) / max(1, len(consecutive_runs)), 3),
            "median_run_len": median(consecutive_runs) if consecutive_runs else 1,
            "avg_run_len": round(sum(consecutive_runs) / max(1, len(consecutive_runs)), 2),
        },
        "tones": tone_payload,
        "runtime_policy": [
            "先选当前语气，再看最近消息有没有承接关系；语气只决定表达方式，不改事实。",
            "当前情绪的 record_summary、top_phrases、examples 与 retrieved_memories 合并使用。",
            "不要复读 top_phrases；只学习长度、停顿、反问方式和气泡数量。",
        ],
    }


def read_style_role(config: AppConfig) -> str:
    path = config.persona_dir / "facts.json"
    if not path.exists():
        return DEFAULT_STYLE_ROLE
    try:
        facts = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return DEFAULT_STYLE_ROLE
    return str(facts.get("style_role") or DEFAULT_STYLE_ROLE)


def normalize_line(value: object) -> str:
    text = fix_text(str(value or "")).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def is_usable_line(text: str) -> bool:
    if not text or text in {"[表情]", "[图片]", "[动画表情]"}:
        return False
    if len(text) > 90:
        return False
    if "撤回了一条消息" in text:
        return False
    return True


def classify_line_tones(text: str) -> list[str]:
    lowered = text.lower()
    hits = [
        tone
        for tone, spec in TONE_DEFINITIONS.items()
        if tone != "casual" and any(str(cue).lower() in lowered for cue in spec.get("cues", []))
    ]
    if not hits:
        return ["casual"]
    if len(text) <= 2 and "curious" in hits and len(hits) > 1:
        hits.remove("curious")
    return hits[:3]


def summarize_tone(tone: str, lines: list[str], fallback_lines: list[str]) -> dict[str, Any]:
    source = lines or fallback_lines[:]
    lengths = [len(re.sub(r"\s+", "", line)) for line in source] or [1]
    short_ratio = sum(1 for value in lengths if value <= 8) / len(lengths)
    question_ratio = sum(1 for line in source if "?" in line or "？" in line) / max(1, len(source))
    symbol_ratio = sum(1 for line in source if re.fullmatch(r"[?？!！。…~6]+", line)) / max(1, len(source))
    phrase_counter = Counter(line for line in source if 1 <= len(line) <= 12 and line not in BLOCKED_TOP_PHRASES)
    ending_counter = Counter(line[-2:] for line in source if len(line) >= 2)
    top_phrases = [text for text, _ in phrase_counter.most_common(14)]
    examples = pick_diverse_examples(source, limit=10)
    line_hint = "偏单泡泡"
    if short_ratio >= 0.62:
        line_hint = "短泡泡为主"
    if question_ratio >= 0.28:
        line_hint += "，常用反问续话"
    if tone in {"playful", "excited", "engaged"}:
        line_hint += "，适合拆成两三句"
    if tone == "sleepy":
        line_hint += "，少主动展开"
    return {
        "label": TONE_DEFINITIONS[tone]["label"],
        "cues": TONE_DEFINITIONS[tone]["cues"],
        "stats": {
            "sample_count": len(lines),
            "median_chars": median(lengths),
            "avg_chars": round(sum(lengths) / len(lengths), 1),
            "short_ratio": round(short_ratio, 3),
            "question_ratio": round(question_ratio, 3),
            "symbol_ratio": round(symbol_ratio, 3),
        },
        "record_summary": build_summary(tone, lengths, short_ratio, question_ratio, symbol_ratio),
        "style": build_style_hint(tone),
        "bubbles": line_hint,
        "avoid": build_avoid_hint(tone),
        "top_phrases": top_phrases,
        "common_endings": [text for text, _ in ending_counter.most_common(8)],
        "examples": examples,
    }


def pick_diverse_examples(lines: list[str], limit: int) -> list[str]:
    examples: list[str] = []
    seen_prefix: set[str] = set()
    for line, _ in Counter(lines).most_common(160):
        if line in BLOCKED_TOP_PHRASES:
            continue
        prefix = re.sub(r"[?？!！。…~]+", "", line)[:3]
        if prefix in seen_prefix and len(line) > 2:
            continue
        seen_prefix.add(prefix)
        examples.append(line)
        if len(examples) >= limit:
            break
    return examples


def build_summary(tone: str, lengths: list[int], short_ratio: float, question_ratio: float, symbol_ratio: float) -> str:
    median_len = median(lengths)
    parts = [f"记录中位长度约{median_len}字"]
    parts.append("短句占比高" if short_ratio >= 0.55 else "不全是短句")
    if question_ratio >= 0.22:
        parts.append("疑问/反问明显")
    if symbol_ratio >= 0.08:
        parts.append("允许单符号但不能滥用")
    tone_note = {
        "sleepy": "整体低能量，常收束话题。",
        "annoyed": "偏嫌弃和边界，不靠长篇说教。",
        "soft": "安抚直接但不鸡汤。",
        "playful": "接梗快，重点是反应而不是解释。",
        "excited": "先短促惊讶，再补一点追问。",
        "curious": "常用短反问让对方展开。",
        "engaged": "连续接话时会挑重点，不逐条答卷。",
        "casual": "日常闲聊以随手接话和轻微吐槽为主。",
    }[tone]
    return "；".join(parts) + "；" + tone_note


def build_style_hint(tone: str) -> str:
    return {
        "sleepy": "懒一点、短一点，可以说困/睡/安安，别突然兴奋开大话题。",
        "annoyed": "可以短促否定、嫌弃、转移，像熟人拌嘴，不要上纲上线。",
        "soft": "先接住对方，再给一小句照顾或安抚，不要心理咨询腔。",
        "playful": "先接梗或笑点，再用半句吐槽/反问续上。",
        "excited": "用短惊讶词起手，第二句追问具体点。",
        "curious": "追问要贴着上一句，别只发问号。",
        "engaged": "把连续消息合成一个意思，先回应核心，再短短补一句。",
        "casual": "像刚看到消息，随手回，不解释，不端着。",
    }[tone]


def build_avoid_hint(tone: str) -> str:
    base = "不要说检索/记录/证据，不要暴露机器身份，不要套“你继续/然后呢”。"
    extra = {
        "sleepy": "不要一边说困一边主动展开复杂话题。",
        "annoyed": "不要连续攻击，也不要突然道歉式自我修正。",
        "soft": "不要长篇温柔和人生建议。",
        "playful": "不要百科解释梗，不要每次都用“乐”。",
        "excited": "不要满屏感叹号。",
        "curious": "不要连续单问号。",
        "engaged": "不要逐条编号回答。",
        "casual": "不要客服腔和报告腔。",
    }[tone]
    return f"{base}{extra}"
