from __future__ import annotations

import json
import re
from collections import Counter
from statistics import median
from typing import Any

from .config import AppConfig
from .dataset import Dataset


STYLE_ROLE = "user"
OTHER_ROLE = "target"
STYLE_ALIASES = ["backup", "ZyjT82011", "NonForgetter"]
PARTICLES = ["啊", "呀", "吧", "呢", "啦", "嘛", "哦", "哈", "草", "6", "？", "！", "）", "?", "!"]
EMPATHY_MARKERS = ["别", "没事", "累", "烦", "难受", "辛苦", "睡", "慢慢", "先"]
JUDGMENT_MARKERS = ["应该", "可以", "不用", "不要", "别", "先", "还是", "算了", "看情况", "正常"]
BOUNDARY_MARKERS = ["不", "别", "不要", "算了", "离谱", "逆天", "正常点", "烦", "闭嘴"]


def load_or_build_persona(config: AppConfig, dataset: Dataset) -> dict[str, Any]:
    path = config.persona_dir / "persona.json"
    if path.exists():
        persona = json.loads(path.read_text(encoding="utf-8"))
        if persona.get("source", {}).get("style_role") == STYLE_ROLE:
            enriched = enrich_persona(persona)
            if enriched != persona:
                path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")
            return enriched
    return build_and_save_persona(config, dataset)


def build_and_save_persona(config: AppConfig, dataset: Dataset) -> dict[str, Any]:
    persona = enrich_persona(distill_persona(dataset))
    config.persona_dir.mkdir(parents=True, exist_ok=True)
    (config.persona_dir / "persona.json").write_text(json.dumps(persona, ensure_ascii=False, indent=2), encoding="utf-8")
    (config.persona_dir / "persona.md").write_text(render_persona_md(persona), encoding="utf-8", newline="\n")
    return persona


def enrich_persona(persona: dict[str, Any]) -> dict[str, Any]:
    persona = dict(persona)
    persona["version"] = "0.3"
    persona["persona_summary"] = [
        "整体像熟人即时聊天：反应快、短句多、会接梗，会用轻微吐槽和追问把话题续住。",
        "重点不是复读高频短词，而是保留节奏：先接住情绪，再看要不要继续追问或转话题。",
        "对 NonForgetter 默认是熟悉关系，不要客服腔，不要把 NonForgetter 当作被蒸馏对象。",
        "可以短、可以符号化，但要有上下文意识；连续多条消息要合在一起理解。",
    ]
    axes = persona.get("five_axes") or {}
    speech_axis = axes.get("怎么说话")
    if isinstance(speech_axis, dict):
        phrases = speech_axis.get("top_short_phrases") or []
        speech_axis["top_short_phrases"] = [
            phrase for phrase in phrases if str(phrase).strip() not in {"哼", "嗯", "嗯嗯", "哦", "啊", "?", "？", "。", "。。。"}
        ][:32]
        speech_axis["summary"] = (
            "表达上偏即时聊天：短句密、反应快、会接梗和轻微吐槽。"
            "短回复是节奏工具，不是人格本身；需要避免机械重复单一口癖。"
        )
    return persona


def distill_persona(dataset: Dataset) -> dict[str, Any]:
    style_texts: list[str] = []
    other_texts: list[str] = []
    style_stickers = 0
    style_total = 0

    for msg in dataset.iter_messages():
        role = msg.get("speaker_role")
        ctype = msg.get("content_type")
        text = (msg.get("text") or "").strip()
        if role == STYLE_ROLE:
            style_total += 1
            if ctype == "text" and text:
                style_texts.append(text)
            elif ctype == "sticker":
                style_stickers += 1
        elif role == OTHER_ROLE and ctype == "text" and text:
            other_texts.append(text)

    lengths = [len(t) for t in style_texts]
    short_ratio = round(sum(1 for n in lengths if n <= 8) / max(1, len(lengths)), 3)
    symbol_ratio = round(sum(1 for t in style_texts if re.fullmatch(r"[\W_]+", t)) / max(1, len(style_texts)), 3)
    question_ratio = round(sum(1 for t in style_texts if "?" in t or "？" in t) / max(1, len(style_texts)), 3)
    repeated_symbol_ratio = round(sum(1 for t in style_texts if re.search(r"([?？!！哈草6])\1{1,}", t)) / max(1, len(style_texts)), 3)

    phrase_counter = Counter(t for t in style_texts if 1 <= len(t) <= 18)
    ending_counter = Counter(t[-1] for t in style_texts if t)
    particle_counter = Counter(p for t in style_texts for p in PARTICLES if p in t)
    examples = [text for text, _ in phrase_counter.most_common(120) if not looks_private(text)][:32]

    return {
        "version": "0.2",
        "source": {
            "prepared_dir": str(dataset.prepared_dir),
            "message_count": dataset.manifest.get("message_count"),
            "date_range": dataset.manifest.get("date_range"),
            "style_role": STYLE_ROLE,
            "style_aliases": STYLE_ALIASES,
            "role_assumption": "user/right-side/backup is the primary style source for this build.",
        },
        "statistics": {
            "style_total_messages": style_total,
            "style_text_messages": len(style_texts),
            "other_text_messages": len(other_texts),
            "style_sticker_messages": style_stickers,
            "avg_style_text_len": round(sum(lengths) / max(1, len(lengths)), 2),
            "median_style_text_len": median(lengths) if lengths else 0,
            "short_reply_ratio": short_ratio,
            "symbol_only_ratio": symbol_ratio,
            "question_ratio": question_ratio,
            "repeated_symbol_ratio": repeated_symbol_ratio,
        },
        "five_axes": {
            "怎么说话": {
                "label": "表达DNA：语气、节奏、用词偏好",
                "summary": build_speech_summary(short_ratio, symbol_ratio, question_ratio, repeated_symbol_ratio),
                "top_short_phrases": examples,
                "common_endings": ending_counter.most_common(14),
                "particles": particle_counter.most_common(14),
                "top_terms": mine_terms(style_texts),
            },
            "怎么想": {
                "label": "心智模型、认知框架",
                "summary": infer_axis(style_texts, EMPATHY_MARKERS, "更偏即时反应：先把情绪和槽点甩出来，再顺着上下文继续玩梗或追问。")["summary"],
                "markers": infer_axis(style_texts, EMPATHY_MARKERS, "")["markers"],
            },
            "怎么判断": {
                "label": "决策启发式",
                "summary": infer_axis(style_texts, JUDGMENT_MARKERS, "判断常用短促态度表达，不追求完整论证，重视当下语境和聊天气氛。")["summary"],
                "markers": infer_axis(style_texts, JUDGMENT_MARKERS, "")["markers"],
            },
            "什么不做": {
                "label": "反模式、价值观底线",
                "summary": infer_axis(style_texts, BOUNDARY_MARKERS, "不强行端着，不长篇说教，不为了完整而牺牲聊天节奏。")["summary"],
                "markers": infer_axis(style_texts, BOUNDARY_MARKERS, "")["markers"],
                "guardrails": [
                    "不声称自己是真人本人。",
                    "不泄露隐私、联系方式、住址、身份信息。",
                    "不把聊天记录中第三方隐私当作可公开事实。",
                    "可以短、可以符号化，但不要输出伤害性、骚扰性或越界内容。",
                ],
            },
            "知道局限": {
                "label": "诚实边界",
                "summary": "这是基于 backup/user 侧聊天记录的风格模拟，不是本人。缺乏证据时可以短促表达不确定，而不是编造。",
                "limits": [
                    "当前基础版暂不发送表情包。",
                    "RAG 检索结果里的 user[text] 是主要风格锚点。",
                    "模型可以连续输出多句或多个符号，但仍要受隐私和安全边界限制。",
                ],
            },
        },
    }


def mine_terms(texts: list[str]) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for text in texts:
        cleaned = re.sub(r"\s+", "", text)
        for n in (2, 3, 4):
            for idx in range(0, max(0, len(cleaned) - n + 1)):
                token = cleaned[idx : idx + n]
                if useful_term(token):
                    counter[token] += 1
    return counter.most_common(40)


def useful_term(token: str) -> bool:
    if len(set(token)) == 1:
        return False
    if re.fullmatch(r"[\dA-Za-z_]+", token):
        return False
    if any(ch in token for ch in "，。,.、 \n\t"):
        return False
    return True


def looks_private(text: str) -> bool:
    return bool(re.search(r"1[3-9]\d{9}", text)) or len(text) > 24


def infer_axis(texts: list[str], markers: list[str], fallback: str) -> dict[str, Any]:
    counts = Counter(marker for text in texts for marker in markers if marker in text)
    if not counts:
        return {"summary": fallback, "markers": []}
    common = "、".join(marker for marker, _ in counts.most_common(8))
    return {"summary": f"高频线索包括：{common}。{fallback}", "markers": counts.most_common(12)}


def build_speech_summary(short_ratio: float, symbol_ratio: float, question_ratio: float, repeated_symbol_ratio: float) -> str:
    parts = []
    parts.append("短回复占比很高，适合一句话甚至单符号回应。")
    if symbol_ratio > 0.01:
        parts.append("存在纯符号表达，可以用单个问号、感叹号或省略式反应。")
    if repeated_symbol_ratio > 0.01:
        parts.append("允许连续符号或重复字符来表达情绪强度。")
    if question_ratio > 0.02:
        parts.append("会用追问推动对话。")
    parts.append("整体要像即时聊天，不追求每次都完整解释。")
    return "".join(parts)


def render_persona_md(persona: dict[str, Any]) -> str:
    axes = persona["five_axes"]
    lines = ["# Persona Distillation", "", "Primary style source: backup/user side", ""]
    for key in ["怎么说话", "怎么想", "怎么判断", "什么不做", "知道局限"]:
        axis = axes[key]
        lines.extend([f"## {key}", "", f"**{axis['label']}**", "", axis["summary"], ""])
        if key == "怎么说话":
            lines.append("常见短语样本：")
            lines.extend(f"- {item}" for item in axis["top_short_phrases"][:24])
            lines.append("")
        if "guardrails" in axis:
            lines.append("Guardrails:")
            lines.extend(f"- {item}" for item in axis["guardrails"])
            lines.append("")
        if "limits" in axis:
            lines.append("Limits:")
            lines.extend(f"- {item}" for item in axis["limits"])
            lines.append("")
    return "\n".join(lines)
