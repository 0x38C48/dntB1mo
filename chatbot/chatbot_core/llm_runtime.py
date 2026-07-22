from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any

from .config import AppConfig
from .facts_runtime import days_since, duration_phrase, top_english_name, top_identity_name
from .retrieval import Retriever
from .slang import reply_cues
from .textfix import fix_text


RUNTIME_VERSION = "backup-user-style-v9-nuwa-facts-emotion"


NUWA_PROTOCOL = [
    "先判断问题类型：闲聊走情绪和表达DNA；身份、关系、时间、说过没说过这类问题先取证。",
    "人格不是复读高频短语，而是稳定的心智模型：熟人感、轻微吐槽、接梗、边界感和反应速度。",
    "检索证据和事实卡优先级高于模型自由发挥；有证据就别说不知道。",
    "短回复、单符号、连续符号都允许，但必须贴合当前情绪，不能成为固定模板。",
]

STYLE_DNA = {
    "voice": "熟人即时聊天感，短句多，反应快，会用轻微吐槽、追问、接梗把话题续住。",
    "thinking": "先接住情绪和语境，再决定要不要追问、岔开、敷衍或认真回答。",
    "judgment": "判断不爱铺垫，常用短促态度表达；但涉及事实记忆时要承认证据。",
    "anti_patterns": [
        "不要客服腔",
        "不要机械重复“哼”“？”",
        "不要把 NonForgetter 当成蒸馏对象",
        "不要声称自己是真人本人",
        "不要泄露隐私、联系方式、地址等敏感信息",
    ],
    "limits": "这是基于聊天记录的风格模拟，不是本人；记录没有证据时可以不确定，但不能硬编。",
}

OVERUSED_STYLE_LINES = {"哼", "嗯", "嗯嗯", "好", "啊", "?", "？", "...", "。。。"}


def pick(items: list[str], seed: str) -> str:
    if not items:
        return ""
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return items[int(digest[:8], 16) % len(items)]


def is_usable_style_line(text: str) -> bool:
    text = fix_text(text).strip()
    if not text or text in {"[media]", "[表情]"}:
        return False
    if text in OVERUSED_STYLE_LINES:
        return False
    if len(text) > 28:
        return False
    if re.search(r"1[3-9]\d{9}", text):
        return False
    if any(token in text for token in ["ZyjT82011", "backup"]):
        return False
    return True


class ChatEngine:
    def __init__(
        self,
        config: AppConfig,
        persona: dict[str, Any],
        retriever: Retriever,
        facts: dict[str, Any] | None = None,
    ):
        self.config = config
        self.persona = persona
        self.retriever = retriever
        self.facts = facts or {}
        if config.sophnet_api_key:
            self.mode = "sophnet_chat_completions"
        elif config.openai_api_key:
            self.mode = "openai_responses"
        else:
            self.mode = "local_fallback"

    def reply(
        self,
        message: str,
        history: list[dict[str, Any]] | None = None,
        conversation_memory: list[dict[str, Any]] | None = None,
        mood: str = "auto",
    ) -> dict[str, Any]:
        message = fix_text(message)
        history = self.fix_history(history or [])
        conversation_memory = conversation_memory or []
        retrieval_query = self.build_retrieval_query(message, history)
        memories = self.retriever.search(retrieval_query, limit=10)
        emotion = self.resolve_emotion(message, history, mood)

        fact_reply = self.fact_first_reply(message, history, memories)
        if fact_reply:
            return {
                "reply": fact_reply,
                "mode": f"{self.mode}_fact_route",
                "emotion": emotion,
                "facts": self.public_facts(),
                "memories": memories,
            }

        if self.is_memory_dispute_question(message) and self.has_memory_evidence(message, history, memories):
            return {
                "reply": self.memory_evidence_reply(message, history, memories),
                "mode": f"{self.mode}_evidence_route",
                "emotion": emotion,
                "facts": self.public_facts(),
                "memories": memories,
            }

        if self.should_route_locally(message):
            return {
                "reply": self.local_reply(message, memories, conversation_memory, emotion),
                "mode": f"{self.mode}_local_route",
                "emotion": emotion,
                "facts": self.public_facts(),
                "memories": memories,
            }

        prompt = self.build_prompt(message, history[-28:], memories, conversation_memory, emotion)
        if self.config.sophnet_api_key:
            try:
                text = self.call_sophnet_api(prompt)
                return {
                    "reply": self.polish_model_reply(text, message, memories, history, conversation_memory, emotion),
                    "mode": self.mode,
                    "emotion": emotion,
                    "facts": self.public_facts(),
                    "memories": memories,
                }
            except Exception as exc:
                return {
                    "reply": self.local_reply(message, memories, conversation_memory, emotion),
                    "mode": "local_fallback_after_sophnet_error",
                    "api_error": str(exc),
                    "emotion": emotion,
                    "facts": self.public_facts(),
                    "memories": memories,
                }
        if self.config.openai_api_key:
            try:
                text = self.call_model_api(prompt)
                return {
                    "reply": self.polish_model_reply(text, message, memories, history, conversation_memory, emotion),
                    "mode": self.mode,
                    "emotion": emotion,
                    "facts": self.public_facts(),
                    "memories": memories,
                }
            except Exception as exc:
                return {
                    "reply": self.local_reply(message, memories, conversation_memory, emotion),
                    "mode": "local_fallback_after_api_error",
                    "api_error": str(exc),
                    "emotion": emotion,
                    "facts": self.public_facts(),
                    "memories": memories,
                }
        return {
            "reply": self.local_reply(message, memories, conversation_memory, emotion),
            "mode": self.mode,
            "emotion": emotion,
            "facts": self.public_facts(),
            "memories": memories,
        }

    def build_prompt(
        self,
        message: str,
        history: list[dict[str, Any]],
        memories: list[dict[str, Any]],
        conversation_memory: list[dict[str, Any]],
        emotion: str,
    ) -> str:
        axes = self.persona.get("five_axes", {})
        persona_brief = {
            key: {
                "label": value.get("label"),
                "summary": fix_text(value.get("summary", "")),
                "guardrails": value.get("guardrails", []),
                "limits": value.get("limits", []),
                "top_short_phrases": self.filter_style_phrases(value.get("top_short_phrases", []))[:10],
            }
            for key, value in axes.items()
            if isinstance(value, dict)
        }
        recent_assistant_replies = [
            item.get("content", "")
            for item in history[-12:]
            if item.get("role") == "assistant" and item.get("content")
        ][-6:]
        payload = {
            "task": "生成自然中文即时聊天回复。主要模拟 backup/user 侧风格，但不要机械复读。",
            "nuwa_protocol": NUWA_PROTOCOL,
            "current_user_identity": {
                "display_name": "NonForgetter",
                "instruction": "当前聊天对象是 NonForgetter；你模拟的是 backup。不要把 NonForgetter 当成蒸馏对象。",
            },
            "style_dna": STYLE_DNA,
            "emotion_state": emotion,
            "identity_and_timeline_facts": self.public_facts(),
            "persona_five_axes": persona_brief,
            "recent_history": history,
            "conversation_evidence": self.extract_conversation_evidence(message, history),
            "recent_assistant_replies": recent_assistant_replies,
            "conversation_memory": conversation_memory,
            "retrieved_memories": memories,
            "slang_and_homophone_cues": reply_cues(message),
            "user_message": message,
            "output_rules": [
                "只输出回复正文，不解释检索过程。",
                "可以很短，也可以分成连续几句，但要针对当前这句，不要套模板。",
                "身份/名字/时间/是否说过的问题必须优先使用 identity_and_timeline_facts 和 retrieved_memories。",
                "有明确证据时不要说“不知道/没印象”。",
                "top_short_phrases 只是风格参考，不是复读清单。",
                "避免重复 recent_assistant_replies 中的句式；意思接近也要换角度。",
                "遇到多行 user_message，代表 NonForgetter 连续发了多条消息，要整体理解后回复。",
                "遇到谐音梗、网络梗、拼音缩写时按语境接住情绪和笑点，不要机械解释。",
                "不要声称自己是真人本人。",
            ],
        }
        return json.dumps(payload, ensure_ascii=False)

    def call_model_api(self, prompt: str) -> str:
        payload = {
            "model": self.config.openai_model,
            "input": [
                {
                    "role": "system",
                    "content": "You are a concise Chinese chat style simulator. Use evidence before facts. Avoid repetition.",
                },
                {"role": "user", "content": prompt},
            ],
            "max_output_tokens": 260,
        }
        request = urllib.request.Request(
            f"{self.config.openai_base_url.rstrip('/')}/responses",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.openai_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
        text = self.extract_response_text(data)
        return text.strip() or self.local_reply("", [], [], "auto")

    def call_sophnet_api(self, prompt: str) -> str:
        payload = {
            "model": self.config.sophnet_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You simulate backup's concise Chinese chat style. Evidence and local facts outrank improvisation. Avoid fixed catchphrases.",
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        request = urllib.request.Request(
            f"{self.config.sophnet_base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.sophnet_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
        text = self.extract_chat_completion_text(data)
        return text.strip() or self.local_reply("", [], [], "auto")

    @staticmethod
    def extract_response_text(data: dict[str, Any]) -> str:
        if data.get("output_text"):
            return str(data["output_text"])
        parts: list[str] = []
        for item in data.get("output", []):
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"}:
                    parts.append(str(content.get("text", "")))
        return "\n".join(part for part in parts if part)

    @staticmethod
    def extract_chat_completion_text(data: dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict) and item.get("text"))
        return ""

    def local_reply(
        self,
        message: str,
        memories: list[dict[str, Any]],
        conversation_memory: list[dict[str, Any]] | None = None,
        emotion: str = "auto",
    ) -> str:
        stripped = message.strip()
        if not stripped:
            return pick(["嗯？", "怎么了", "在"], emotion)
        if self.is_slang_meaning_question(stripped):
            return self.slang_meaning_reply(stripped)
        if self.is_user_identity_question(stripped):
            return pick(["NonForgetter啊", "你啊，NonForgetter", "这还要问啊"], stripped + emotion)
        if self.is_bot_identity_question(stripped):
            return self.fact_first_reply(stripped, [], memories) or pick(["backup啊", "你不是一直这么叫吗", "我吗，backup"], stripped + emotion)
        if self.is_time_question(stripped):
            return self.time_reply(stripped)

        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if len(lines) > 1:
            joined = " / ".join(lines[-4:])
            return pick(["等下\n我先看完", "你这几句得放一起看", "我懂你意思了", "你这是连着问我啊"], joined + emotion)

        cues = reply_cues(stripped)
        if cues:
            if any(token in stripped.lower() for token in ["xswl", "笑死", "绷", "哈哈", "草"]):
                return pick(["6", "绷不住了", "什么东西啊", "你别太会找乐子"], stripped + emotion)
            if any(token in stripped for token in ["寄", "丸辣", "完辣", "麻了", "破防", "摆烂"]):
                return pick(["完了", "麻了", "先别寄", "这也太离谱了"], stripped + emotion)
            if any(token in stripped for token in ["乐乐", "乐子"]):
                return pick(["乐子来了是吧", "你又开始了", "这也能找乐子", "乐什么啊"], stripped + emotion)

        candidates = self.extract_style_lines(memories)
        if self.is_question_like(stripped):
            return pick(candidates + ["怎么说", "什么", "你说", "啊？", "我想想"], stripped + emotion)
        if emotion == "annoyed":
            return pick(candidates + ["行行行", "你又来了", "我服了", "别太离谱"], stripped)
        if emotion == "soft":
            return pick(candidates + ["好嘛", "那先这样", "我听着", "慢慢说"], stripped)
        if emotion == "excited":
            return pick(candidates + ["我靠", "真的假的", "快说", "有点意思"], stripped)
        return pick(candidates + ["6", "我靠", "你继续", "有点意思", "先听你说"], stripped + emotion)

    def polish_model_reply(
        self,
        text: str,
        message: str,
        memories: list[dict[str, Any]],
        history: list[dict[str, Any]] | None = None,
        conversation_memory: list[dict[str, Any]] | None = None,
        emotion: str = "auto",
    ) -> str:
        cleaned = fix_text(text).strip().strip("“”\"")
        if not cleaned:
            return self.local_reply(message, memories, conversation_memory, emotion)
        if re.fullmatch(r"[?？]{1,8}", cleaned):
            return self.softened_question_reply(message, memories)
        if cleaned in OVERUSED_STYLE_LINES:
            return self.local_reply(message, memories, conversation_memory, emotion)
        if self.is_repeated_reply(cleaned, history or []):
            return self.repair_repeated_reply(message, emotion)
        if self.is_memory_dispute_question(message) and self.looks_like_denial(cleaned) and self.has_memory_evidence(message, history or [], memories):
            return self.memory_evidence_reply(message, history or [], memories)
        if self.is_bot_identity_question(message) and self.looks_like_denial(cleaned):
            return self.fact_first_reply(message, history or [], memories) or cleaned
        return cleaned

    def fact_first_reply(self, message: str, history: list[dict[str, Any]], memories: list[dict[str, Any]]) -> str | None:
        stripped = message.strip()
        if self.is_bot_identity_question(stripped):
            name = top_identity_name(self.facts)
            english = top_english_name(self.facts)
            if name and english:
                return pick(
                    [
                        f"{name}吧，英文名好像是 {english}",
                        f"林...不是，记录里最稳的是{name}，{english}也提过",
                        f"{name}，然后 {english} 这个英文名你也提过",
                    ],
                    stripped,
                )
            if name:
                return pick([f"{name}吧", f"记录里我叫{name}", f"你不是猜到{name}了吗"], stripped)
        if self.is_time_question(stripped):
            return self.time_reply(stripped)
        if self.is_memory_dispute_question(stripped) and self.has_memory_evidence(stripped, history, memories):
            return self.memory_evidence_reply(stripped, history, memories)
        return None

    def time_reply(self, message: str) -> str:
        start = self.facts.get("relationship", {}).get("known_since") or self.facts.get("date_range", {}).get("start")
        days = days_since(start, datetime.now())
        phrase = duration_phrase(days)
        date = (start or "2024-10-13")[:10]
        return pick(
            [
                f"从{date}算的话，{phrase}了",
                f"记录里最早是{date}，到现在差不多{phrase}",
                f"{date}开始的吧，已经{phrase}了",
            ],
            message,
        )

    def public_facts(self) -> dict[str, Any]:
        start = self.facts.get("relationship", {}).get("known_since") or self.facts.get("date_range", {}).get("start")
        return {
            "persona_display": self.facts.get("persona_display", "backup"),
            "current_user": self.facts.get("current_user", "NonForgetter"),
            "top_name": top_identity_name(self.facts),
            "top_english_name": top_english_name(self.facts),
            "known_since": start,
            "known_duration_days": days_since(start, datetime.now()),
            "name_evidence": (self.facts.get("identity", {}).get("name_candidates") or [])[:3],
            "english_name_evidence": (self.facts.get("identity", {}).get("english_name_candidates") or [])[:3],
        }

    @staticmethod
    def fix_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {**item, "content": fix_text(item.get("content", ""))}
            for item in history
        ]

    @staticmethod
    def resolve_emotion(message: str, history: list[dict[str, Any]], mood: str) -> str:
        if mood and mood != "auto":
            return mood
        text = message.lower()
        if any(token in text for token in ["困", "睡", "晚安", "安安"]):
            return "sleepy"
        if any(token in text for token in ["烦", "难受", "emo", "哭", "累"]):
            return "soft"
        if any(token in text for token in ["？", "?", "什么意思", "怎么", "为什么"]):
            return "curious"
        if any(token in text for token in ["笑死", "哈哈", "乐子", "绷", "6"]):
            return "playful"
        if any(token in text for token in ["别", "烦", "服了", "逆天"]):
            return "annoyed"
        if history and len([h for h in history[-8:] if h.get("role") == "user"]) >= 4:
            return "engaged"
        return "casual"

    def softened_question_reply(self, message: str, memories: list[dict[str, Any]]) -> str:
        digest = hashlib.sha256(message.encode("utf-8")).hexdigest()
        allow_symbol = int(digest[:2], 16) % 100 < 16
        if allow_symbol:
            return pick(["?", "？？", "啊？"], message)
        candidates = [line for line in self.extract_style_lines(memories) if not re.fullmatch(r"[?？!！。，、…~]+", line)]
        return pick(candidates + ["什么", "怎么说", "你说", "啊"], message)

    def repair_repeated_reply(self, message: str, emotion: str) -> str:
        stripped = message.strip()
        if self.is_bot_identity_question(stripped):
            return self.fact_first_reply(stripped, [], []) or pick(["backup啊", "你不是知道吗", "我吗"], stripped)
        if self.is_slang_meaning_question(stripped):
            return self.slang_meaning_reply(stripped)
        if self.is_meaning_question(stripped):
            return self.meaning_reply(stripped)
        return pick(["我换个说法", "刚刚那句不算", "等下，我重说", "不是那个意思"], stripped + emotion)

    @staticmethod
    def is_repeated_reply(reply: str, history: list[dict[str, Any]]) -> bool:
        raw_reply = re.sub(r"\s+", "", reply or "")
        normalized_reply = ChatEngine.normalize_for_repeat(reply)
        recent = [
            (re.sub(r"\s+", "", item.get("content", "")), ChatEngine.normalize_for_repeat(item.get("content", "")))
            for item in history[-12:]
            if item.get("role") == "assistant"
        ][-6:]
        for old_raw, old in recent:
            if not old_raw:
                continue
            if raw_reply == old_raw or SequenceMatcher(None, raw_reply, old_raw).ratio() >= 0.72:
                return True
            if len(normalized_reply) < 4 or not old:
                continue
            if normalized_reply in old or old in normalized_reply:
                return True
            if SequenceMatcher(None, normalized_reply, old).ratio() >= 0.68:
                return True
        return False

    @staticmethod
    def normalize_for_repeat(text: str) -> str:
        text = re.sub(r"[\s?？!！。，、…~]+", "", text or "")
        for filler in ["刚不是说了嘛", "我还能咋办", "你骚的", "哼", "嗯"]:
            text = text.replace(filler, "")
        return text

    @staticmethod
    def is_memory_dispute_question(message: str) -> bool:
        return any(token in message for token in ["说过", "没说过", "记得", "记错", "是不是", "绝对没有", "有记录"])

    @staticmethod
    def looks_like_denial(reply: str) -> bool:
        return any(token in reply for token in ["不知道", "没印象", "没说过", "记错", "没有吧", "不记得", "我咋知道"])

    @staticmethod
    def has_memory_evidence(message: str, history: list[dict[str, Any]], memories: list[dict[str, Any]]) -> bool:
        haystack = "\n".join(
            [message]
            + [str(item.get("content", "")) for item in history[-12:]]
            + [str(memory.get("text", "")) for memory in memories[:6]]
        )
        if any(token in haystack for token in ["林薇艺", "lily", "Lily", "姓林", "英文名"]):
            return True
        return any(token in haystack for token in ["乐乐", "说过", "记得"])

    @staticmethod
    def memory_evidence_reply(message: str, history: list[dict[str, Any]], memories: list[dict[str, Any]]) -> str:
        haystack = "\n".join(
            [message]
            + [str(item.get("content", "")) for item in history[-12:]]
            + [str(memory.get("text", "")) for memory in memories[:4]]
        )
        if "林薇艺" in haystack or "姓林" in haystack:
            return "有，林薇艺这个我刚才没接上"
        if "lily" in haystack.lower():
            return "有，lily这个英文名记录里也有"
        if "乐乐" in haystack:
            return "有，乐乐这个要按梗和上下文看"
        return "有相关的，我刚才没接上"

    @staticmethod
    def filter_style_phrases(phrases: list[Any]) -> list[str]:
        result = []
        seen = set()
        for phrase in phrases:
            text = fix_text(str(phrase)).strip()
            if is_usable_style_line(text) and text not in seen:
                seen.add(text)
                result.append(text)
        return result

    @staticmethod
    def should_route_locally(message: str) -> bool:
        return (
            ChatEngine.is_capability_question(message)
            or ChatEngine.is_meaning_question(message)
            or ChatEngine.is_time_question(message)
        )

    @staticmethod
    def build_retrieval_query(message: str, history: list[dict[str, Any]]) -> str:
        recent_user_text = [
            str(item.get("content", ""))
            for item in history[-10:]
            if item.get("role") == "user" and item.get("content")
        ][-5:]
        expansions = []
        if ChatEngine.is_bot_identity_question(message):
            expansions.extend(["名字 姓 林 薇 艺 英文名 lily 猜名字"])
        if ChatEngine.is_time_question(message):
            expansions.extend(["认识 多久 开始 聊天 第一条"])
        return "\n".join([message, *recent_user_text, *expansions])

    @staticmethod
    def extract_conversation_evidence(message: str, history: list[dict[str, Any]]) -> list[dict[str, str]]:
        if not any(token in message for token in ["说过", "没说过", "记得", "记错", "是不是", "姓", "名字", "绝对没有"]):
            return []
        rows: list[dict[str, str]] = []
        for item in history[-18:]:
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            if any(token in content for token in ["乐乐", "姓", "林", "名字", "英文名", "lily", "说过", "记得"]):
                rows.append({"role": str(item.get("role", "")), "content": content})
        return rows[-8:]

    @staticmethod
    def is_capability_question(message: str) -> bool:
        return any(token in message for token in ["帮我做什么", "能做什么", "会做什么", "可以帮我"])

    @staticmethod
    def is_question_like(message: str) -> bool:
        return "?" in message or "？" in message or any(token in message for token in ["什么", "怎么", "为什么", "吗"])

    @staticmethod
    def is_bot_identity_question(message: str) -> bool:
        return any(token in message for token in ["你是谁", "你叫什么", "你叫啥", "你叫什么名字", "你名字", "英文名"])

    @staticmethod
    def is_user_identity_question(message: str) -> bool:
        return any(token in message for token in ["我是谁", "我是你的谁", "我是你的什么人", "我是什么人", "我是你什么人"])

    @staticmethod
    def is_time_question(message: str) -> bool:
        return any(token in message for token in ["聊了多久", "认识多久", "认识多长", "多久了", "什么时候认识", "哪天认识"])

    @staticmethod
    def is_meaning_question(message: str) -> bool:
        return any(token in message for token in ["什么意思", "啥意思", "什么含义", "何意"])

    @staticmethod
    def is_slang_meaning_question(message: str) -> bool:
        return any(token in message for token in ["乐乐", "乐子", "梗", "网络梗"])

    @staticmethod
    def meaning_reply(message: str) -> str:
        if "乐乐" in message:
            return pick(["乐子那个乐吧，不是单纯名字", "就是找乐子/看乐子的那个梗", "偏网络梗，差不多是好笑、找乐子的意思"], message)
        return pick(["字面意思吧", "你说哪个词", "这个要看你前面怎么用的"], message)

    @staticmethod
    def slang_meaning_reply(message: str) -> str:
        if "乐乐" in message:
            return pick(["乐子那个乐吧，不是单纯名字", "就是找乐子/看乐子的那个梗", "偏网络梗，差不多是好笑、找乐子的意思"], message)
        if "乐子" in message:
            return pick(["看热闹找乐子的意思", "就是拿来好笑/看戏的那个乐子", "乐子人那个乐子"], message)
        return pick(["是梗，得看上下文", "按网络梗理解，不是字面硬翻", "这个要结合前后文看"], message)

    @staticmethod
    def extract_style_lines(memories: list[dict[str, Any]]) -> list[str]:
        candidates: list[str] = []
        seen: set[str] = set()
        for memory in memories:
            for line in fix_text(memory.get("text") or "").splitlines():
                if not line.startswith("user[text]:"):
                    continue
                text = line.split(":", 1)[1].strip()
                if is_usable_style_line(text) and text not in seen:
                    seen.add(text)
                    candidates.append(text)
        return candidates[:18]
