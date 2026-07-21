from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from difflib import SequenceMatcher
from typing import Any

from .config import AppConfig
from .retrieval import Retriever
from .slang import reply_cues


RUNTIME_VERSION = "backup-user-style-v8-slang-evidence"


PERSONA_NARRATIVE = [
    "人格核心不是高频口癖，而是聊天姿态：反应快、句子短、会接梗，会用轻微吐槽和追问把对话续住。",
    "对 NonForgetter 要像熟人聊天，不要把对方当陌生用户，也不要用客服式总结。",
    "短回复可以用，但不要机械复读某个高频词；同一种句式连续出现要主动换说法。",
    "如果 NonForgetter 连续发多条消息，先把这些话当成同一轮表达理解，再决定是否逐点回应。",
]

OVERUSED_STYLE_LINES = {
    "哼",
    "嗯",
    "嗯嗯",
    "哦",
    "好",
    "啊",
    "?",
    "？",
    "。",
    "。。。",
    "...",
}


def pick(items: list[str], seed: str) -> str:
    if not items:
        return ""
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return items[int(digest[:8], 16) % len(items)]


def is_usable_style_line(text: str) -> bool:
    text = text.strip()
    if not text or text in {"[media]", "[表情]"}:
        return False
    if text in OVERUSED_STYLE_LINES:
        return False
    if len(text) > 24:
        return False
    if re.search(r"1[3-9]\d{9}", text):
        return False
    if any(token in text for token in ["ZyjT82011", "backup"]):
        return False
    return True


class ChatEngine:
    def __init__(self, config: AppConfig, persona: dict[str, Any], retriever: Retriever):
        self.config = config
        self.persona = persona
        self.retriever = retriever
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
    ) -> dict[str, Any]:
        history = history or []
        conversation_memory = conversation_memory or []
        retrieval_query = self.build_retrieval_query(message, history)
        memories = self.retriever.search(retrieval_query, limit=10)
        if self.is_memory_dispute_question(message) and self.has_memory_evidence(message, history, memories):
            return {
                "reply": self.memory_evidence_reply(message, history, memories),
                "mode": f"{self.mode}_evidence_route",
                "memories": memories,
            }
        if self.should_route_locally(message):
            return {
                "reply": self.local_reply(message, memories, conversation_memory),
                "mode": f"{self.mode}_local_route",
                "memories": memories,
            }

        prompt = self.build_prompt(message, history[-24:], memories, conversation_memory)
        if self.config.sophnet_api_key:
            try:
                text = self.call_sophnet_api(prompt)
                return {
                    "reply": self.polish_model_reply(text, message, memories, history, conversation_memory),
                    "mode": self.mode,
                    "memories": memories,
                }
            except Exception as exc:
                return {
                    "reply": self.local_reply(message, memories, conversation_memory),
                    "mode": "local_fallback_after_sophnet_error",
                    "api_error": str(exc),
                    "memories": memories,
                }
        if self.config.openai_api_key:
            try:
                text = self.call_model_api(prompt)
                return {
                    "reply": self.polish_model_reply(text, message, memories, history, conversation_memory),
                    "mode": self.mode,
                    "memories": memories,
                }
            except Exception as exc:
                return {
                    "reply": self.local_reply(message, memories, conversation_memory),
                    "mode": "local_fallback_after_api_error",
                    "api_error": str(exc),
                    "memories": memories,
                }
        return {"reply": self.local_reply(message, memories, conversation_memory), "mode": self.mode, "memories": memories}

    def build_prompt(
        self,
        message: str,
        history: list[dict[str, Any]],
        memories: list[dict[str, Any]],
        conversation_memory: list[dict[str, Any]],
    ) -> str:
        axes = self.persona.get("five_axes", {})
        persona_brief = {
            key: {
                "label": value.get("label"),
                "summary": value.get("summary"),
                "guardrails": value.get("guardrails", []),
                "limits": value.get("limits", []),
                "top_short_phrases": self.filter_style_phrases(value.get("top_short_phrases", []))[:12],
            }
            for key, value in axes.items()
        }
        recent_assistant_replies = [
            item.get("content", "")
            for item in history[-12:]
            if item.get("role") == "assistant" and item.get("content")
        ][-6:]
        return json.dumps(
            {
                "task": "生成自然的中文即时聊天回复。主要模仿 retrieved_memories 里 user[text]/backup 侧的语言风格，但不要机械复读。",
                "current_user_identity": {
                    "display_name": "NonForgetter",
                    "instruction": "把正在聊天的人当作 NonForgetter。不要误把 NonForgetter 当作蒸馏对象，也不要把 backup 当作当前用户。",
                },
                "persona_narrative": PERSONA_NARRATIVE,
                "persona_five_axes": persona_brief,
                "recent_history": history,
                "conversation_evidence": self.extract_conversation_evidence(message, history),
                "recent_assistant_replies": recent_assistant_replies,
                "conversation_memory": conversation_memory,
                "retrieved_memories": memories,
                "slang_and_homophone_cues": reply_cues(message),
                "user_message": message,
                "output_rules": [
                    "只输出回复正文，不要解释检索过程。",
                    "可以很短，但要针对当前这句话，不要套模板。",
                    "top_short_phrases 只是风格参考，不是复读清单。",
                    "不要复读 recent_assistant_replies 里的句式；意思接近时也要换角度。",
                    "身份类问题要稳定：当前聊天的人是 NonForgetter；你模拟的是 backup/乐乐。",
                    "遇到多行 user_message，代表 NonForgetter 连续发送了多条消息，要整体理解后回复。",
                    "遇到谐音梗、网络梗、拼音缩写时，按语境接住情绪和笑点，不要机械解释。",
                    "不要泄露隐私、联系方式、住址、身份信息。",
                    "不要声称自己是真人本人。",
                ],
            },
            ensure_ascii=False,
        )

    def call_model_api(self, prompt: str) -> str:
        payload = {
            "model": self.config.openai_model,
            "input": [
                {
                    "role": "system",
                    "content": "You are a concise Chinese chat style simulator. Avoid repetition. The current user is NonForgetter.",
                },
                {"role": "user", "content": prompt},
            ],
            "max_output_tokens": 240,
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
        return text.strip() or self.local_reply("", [])

    def call_sophnet_api(self, prompt: str) -> str:
        payload = {
            "model": self.config.sophnet_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a concise Chinese chat style simulator. Primary style source is backup/user-side messages. Do not repeat recent wording. The current user is NonForgetter. Do not claim to be the real person.",
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
        return text.strip() or self.local_reply("", [])

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
    ) -> str:
        stripped = message.strip()
        if not stripped:
            return "嗯？"
        if self.is_user_identity_question(stripped):
            return pick(["NonForgetter啊", "你是 NonForgetter 啊", "这还要问啊，NonForgetter"], stripped)
        if self.is_bot_identity_question(stripped):
            return pick(["backup啊", "那就乐乐吧", "你刚不是叫我乐乐吗"], stripped)
        if self.is_slang_meaning_question(stripped):
            return self.slang_meaning_reply(stripped)
        if self.is_meaning_question(stripped):
            return self.meaning_reply(stripped)

        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if len(lines) > 1:
            joined = " / ".join(lines[-3:])
            if any(token in stripped for token in ["为什么", "咋", "怎么", "？", "?"]):
                return pick(["等下\n我先看完", "你这是连着问我啊\n我想想", "你这几句要放一起看"], joined)
            return pick(["我懂你意思了", "等下\n你这几句是连着的吧", "我先把你这几句当一件事看"], joined)

        if self.is_capability_question(stripped):
            return pick(["陪你聊天啊", "你说，我听着", "乱七八糟的也可以说"], stripped)
        if any(word in stripped for word in ["累", "烦", "难受", "崩溃", "不开心"]):
            return pick(["先别硬撑，缓一下。", "怎么了，你说。", "好啦，先别把自己逼太紧。"], stripped)
        if stripped in {"hi", "hello", "你好", "在吗", "在不在"}:
            return pick(["在呢", "嗯？", "怎么啦"], stripped)
        if any(token in stripped for token in ["讲讲", "说说", "展开", "详细"]):
            return pick(["你先说是哪一个", "你具体说哪个", "等下，你说清楚一点"], stripped)
        if any(token in stripped for token in ["无聊", "随便", "话题", "聊什么"]):
            return pick(["那讲八卦\n我要听", "随便啊\n你今天有没有什么离谱的", "玩不玩\n或者讲点逆天的"], stripped)

        cues = reply_cues(stripped)
        if cues:
            lowered = stripped.lower()
            if any(token in lowered for token in ["xswl", "笑死", "笑鼠", "绷", "蚌埠住", "草", "艹"]):
                return pick(["6", "绷不住了", "哈", "什么东西啊"], stripped)
            if any(token in stripped for token in ["寄", "丸辣", "完辣", "麻了", "破防", "摆烂"]):
                return pick(["完了", "麻了", "先别寄", "这也太离谱了"], stripped)
            if any(token in stripped for token in ["离谱", "逆天", "抽象"]):
                return pick(["逆天", "有点抽象", "这什么", "6"], stripped)

        candidates = self.extract_style_lines(memories)
        if "?" in stripped or "？" in stripped:
            return pick(candidates + ["怎么说", "什么", "啊？"], stripped)
        return pick(candidates + ["6", "哈", "我靠", "你继续", "有点意思"], stripped)

    def polish_model_reply(
        self,
        text: str,
        message: str,
        memories: list[dict[str, Any]],
        history: list[dict[str, Any]] | None = None,
        conversation_memory: list[dict[str, Any]] | None = None,
    ) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return self.local_reply(message, memories, conversation_memory)
        if re.fullmatch(r"[?？]{1,8}", cleaned):
            return self.softened_question_reply(message, memories)
        if cleaned in OVERUSED_STYLE_LINES:
            return self.local_reply(message, memories, conversation_memory)
        if self.is_repeated_reply(cleaned, history or []):
            return self.repair_repeated_reply(message)
        if self.is_memory_dispute_question(message) and self.looks_like_denial(cleaned) and self.has_memory_evidence(message, history or [], memories):
            return self.memory_evidence_reply(message, history or [], memories)
        if self.is_capability_question(message) and cleaned in {"不知道", "不清楚", "说不清楚"}:
            return self.local_reply(message, memories, conversation_memory)
        return cleaned

    def softened_question_reply(self, message: str, memories: list[dict[str, Any]]) -> str:
        digest = hashlib.sha256(message.encode("utf-8")).hexdigest()
        allow_symbol = int(digest[:2], 16) % 100 < 20
        if allow_symbol:
            return pick(["?", "？？", "啊？"], message)
        candidates = [line for line in self.extract_style_lines(memories) if not re.fullmatch(r"[?？!！。.,，、…~～]+", line)]
        return pick(candidates + ["什么", "怎么说", "你说", "啊"], message)

    def repair_repeated_reply(self, message: str) -> str:
        stripped = message.strip()
        if self.is_user_identity_question(stripped):
            return pick(["NonForgetter啊", "你啊，NonForgetter", "你是 NonForgetter，这还要问"], stripped)
        if self.is_bot_identity_question(stripped):
            return pick(["backup啊", "那就叫乐乐", "你不是刚给我取名了吗"], stripped)
        if self.is_slang_meaning_question(stripped):
            return self.slang_meaning_reply(stripped)
        if self.is_meaning_question(stripped):
            return self.meaning_reply(stripped)
        return pick(["我换个说法", "刚刚那句不算", "等下，我重说", "不是那个意思"], stripped)

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
        text = re.sub(r"[\s?？!！。.,，、…~～]", "", text or "")
        for filler in ["刚不是说了嘛", "刚不是说了", "我还能咋办", "你赐的", "嘛", "呗"]:
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
            + [str(memory.get("text", "")) for memory in memories[:5]]
        )
        if "姓林" in haystack or ("姓" in haystack and "林" in haystack):
            return True
        return any(token in haystack for token in ["乐乐", "说过", "记得"])

    @staticmethod
    def memory_evidence_reply(message: str, history: list[dict[str, Any]], memories: list[dict[str, Any]]) -> str:
        haystack = "\n".join(
            [message]
            + [str(item.get("content", "")) for item in history[-12:]]
            + [str(memory.get("text", "")) for memory in memories[:3]]
        )
        if "姓林" in haystack or ("姓" in haystack and "林" in haystack):
            return "有，姓林这个我刚才没接上"
        if "乐乐" in haystack:
            return "有，乐乐这个要按梗和上下文看"
        return "有相关的，我刚才没接上"

    @staticmethod
    def filter_style_phrases(phrases: list[Any]) -> list[str]:
        result = []
        seen = set()
        for phrase in phrases:
            text = str(phrase).strip()
            if is_usable_style_line(text) and text not in seen:
                seen.add(text)
                result.append(text)
        return result

    @staticmethod
    def should_route_locally(message: str) -> bool:
        return (
            ChatEngine.is_capability_question(message)
            or ChatEngine.is_identity_question(message)
            or ChatEngine.is_user_identity_question(message)
            or ChatEngine.is_meaning_question(message)
        )

    @staticmethod
    def build_retrieval_query(message: str, history: list[dict[str, Any]]) -> str:
        recent_user_text = [
            str(item.get("content", ""))
            for item in history[-8:]
            if item.get("role") == "user" and item.get("content")
        ][-4:]
        return "\n".join([message, *recent_user_text])

    @staticmethod
    def extract_conversation_evidence(message: str, history: list[dict[str, Any]]) -> list[dict[str, str]]:
        if not any(token in message for token in ["说过", "没说过", "记得", "记错", "是不是", "姓", "绝对没有"]):
            return []
        rows: list[dict[str, str]] = []
        for item in history[-16:]:
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            if any(token in content for token in ["乐乐", "姓", "林", "说过", "记得", "名字", "什么人"]):
                rows.append({"role": str(item.get("role", "")), "content": content})
        return rows[-8:]

    @staticmethod
    def is_capability_question(message: str) -> bool:
        return any(token in message for token in ["帮我做什么", "能做什么", "会做什么", "可以帮我"])

    @staticmethod
    def is_identity_question(message: str) -> bool:
        return ChatEngine.is_bot_identity_question(message) or any(token in message for token in ["叫什么", "什么名字", "叫啥"])

    @staticmethod
    def is_bot_identity_question(message: str) -> bool:
        return any(token in message for token in ["你是谁", "你叫什么", "你叫啥", "你叫什么名字"])

    @staticmethod
    def is_user_identity_question(message: str) -> bool:
        return any(token in message for token in ["我是谁", "我是你的谁", "我是你的什么人", "我是什么人", "我是你什么人"])

    @staticmethod
    def is_meaning_question(message: str) -> bool:
        return any(token in message for token in ["什么意思", "啥意思", "什么含义", "何意味"])

    @staticmethod
    def is_slang_meaning_question(message: str) -> bool:
        return any(token in message for token in ["乐乐", "乐子", "梗", "网络梗"])

    @staticmethod
    def meaning_reply(message: str) -> str:
        if "乐乐" in message:
            return pick(["就是你刚给我安的名字吧", "字面上就是乐那个乐，听着挺随便的", "大概就是个名字，没啥高深意思"], message)
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
            for line in (memory.get("text") or "").splitlines():
                if not line.startswith("user[text]:"):
                    continue
                text = line.split(":", 1)[1].strip()
                if is_usable_style_line(text) and text not in seen:
                    seen.add(text)
                    candidates.append(text)
        return candidates[:16]
