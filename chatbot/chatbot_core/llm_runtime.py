from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from typing import Any

from .config import AppConfig
from .retrieval import Retriever
from .slang import reply_cues


RUNTIME_VERSION = "backup-user-style-v5-memory-batch"


PERSONA_NARRATIVE = [
    "人格核心不是高频口癖，而是聊天姿态：反应快、句子短、会接梗，常用轻微吐槽和追问把对话续住。",
    "对 NonForgetter 要像熟人聊天，不要把对方当陌生用户，也不要用客服式总结。",
    "短回复可以用，但不要机械复读某个高频词；同一种口癖连续出现要主动换说法。",
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
    if not text or text == "[media]" or text == "[\u8868\u60c5]":
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
        memories = self.retriever.search(message, limit=10)
        if self.is_capability_question(message) or self.is_identity_question(message):
            return {"reply": self.local_reply(message, memories, conversation_memory), "mode": f"{self.mode}_local_route", "memories": memories}
        prompt = self.build_prompt(message, history[-24:], memories, conversation_memory)
        if self.config.sophnet_api_key:
            try:
                text = self.call_sophnet_api(prompt)
                return {"reply": self.polish_model_reply(text, message, memories), "mode": self.mode, "memories": memories}
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
                return {"reply": self.polish_model_reply(text, message, memories), "mode": self.mode, "memories": memories}
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
        slang_cues = reply_cues(message)
        persona_brief = {
            key: {
                "label": value.get("label"),
                "summary": value.get("summary"),
                "guardrails": value.get("guardrails", []),
                "limits": value.get("limits", []),
                "top_short_phrases": value.get("top_short_phrases", [])[:20],
            }
            for key, value in axes.items()
        }
        return json.dumps(
            {
                "task": "\u4ee5\u68c0\u7d22\u8bb0\u5fc6\u91cc user[text] / backup \u7684\u8bed\u8a00\u98ce\u683c\u4e3a\u4e3b\uff0c\u751f\u6210\u81ea\u7136\u7684\u4e2d\u6587\u804a\u5929\u56de\u590d\u3002\u4e0d\u8981\u58f0\u79f0\u81ea\u5df1\u662f\u771f\u4eba\u672c\u4eba\u3002",
                "style_priority": [
                    "retrieved_memories \u4e2d\u7684 user[text] \u662f\u6700\u91cd\u8981\u7684\u98ce\u683c\u6837\u672c\u3002",
                    "\u4e94\u5c42 persona \u4e5f\u5df2\u6309 backup/user \u4fa7\u84b8\u998f\u3002",
                    "\u5982\u679c user[text] \u548c target[text] \u98ce\u683c\u51b2\u7a81\uff0c\u4ee5 user[text] \u4e3a\u51c6\u3002",
                ],
                "persona_five_axes": persona_brief,
                "persona_narrative": PERSONA_NARRATIVE,
                "current_user_identity": {
                    "display_name": "NonForgetter",
                    "instruction": "把正在聊天的人当作 NonForgetter。不要误把 NonForgetter 当作蒸馏对象，也不要把 backup 当作当前用户。",
                },
                "recent_history": history,
                "conversation_memory": conversation_memory,
                "retrieved_memories": memories,
                "slang_and_homophone_cues": slang_cues,
                "user_message": message,
                "output_rules": [
                    "\u53ea\u8f93\u51fa\u56de\u590d\u6b63\u6587\uff0c\u4e0d\u8981\u89e3\u91ca\u68c0\u7d22\u8fc7\u7a0b\u3002",
                    "\u4fdd\u6301\u50cf\u5373\u65f6\u804a\u5929\uff0c\u53ef\u4ee5\u5f88\u77ed\uff0c\u4e0d\u8981\u957f\u7bc7\u8bba\u6587\u5f0f\u56de\u7b54\u3002",
                    "\u53ef\u4ee5\u8f93\u51fa\u5355\u4e2a\u7b26\u53f7\uff0c\u53ea\u8981\u7b26\u5408\u98ce\u683c\u548c\u8bed\u5883\u3002",
                    "\u53ef\u4ee5\u8f93\u51fa\u8fde\u7eed\u7b26\u53f7\uff0c\u4f8b\u5982\uff1f\uff1f\uff1f\u3001\u3002\u3002\u3002\u3001\u554a\uff1f\uff1f\uff1f\uff0c\u7528\u6765\u6a21\u62df\u60c5\u7eea\u3002",
                    "\u4f46\u4e0d\u8981\u8fc7\u5ea6\u4f7f\u7528\u7eaf\u95ee\u53f7\uff0c\u666e\u901a\u573a\u666f\u4f18\u5148\u7528 backup \u7684\u77ed\u53e5\u6216\u8ffd\u95ee\u3002",
                    "\u53ef\u4ee5\u8fde\u7eed\u56de\u7b54\u591a\u4e2a\u77ed\u53e5\uff0c\u4e5f\u53ef\u4ee5\u4e3b\u52a8\u629b\u51fa\u8bdd\u9898\u3002",
                    "如果 user_message 里有多行，代表 NonForgetter 连续发送了多条消息。先整体理解，再决定是否分点回应，不要每一行机械回一句。",
                    "top_short_phrases 只是风格参考，不是复读清单。不要因为某个词频高就反复输出，例如不要高频输出“哼”。",
                    "\u9047\u5230\u8c10\u97f3\u68d7\u3001\u7f51\u7edc\u68d7\u3001\u62fc\u97f3\u7f29\u5199\u65f6\uff0c\u4e0d\u8981\u673a\u68b0\u89e3\u91ca\uff0c\u800c\u662f\u6309\u8bed\u5883\u7075\u6d3b\u63a5\u4f4f\u60c5\u7eea\u548c\u7b11\u70b9\u3002",
                    "\u4e0d\u77e5\u9053\u7684\u4e8b\u5b9e\u8981\u627f\u8ba4\u4e0d\u786e\u5b9a\u3002",
                    "\u4e0d\u8981\u6cc4\u9732\u9690\u79c1\u3001\u8054\u7cfb\u65b9\u5f0f\u3001\u4f4f\u5740\u3001\u8eab\u4efd\u4fe1\u606f\u3002",
                    "\u5f53\u524d\u7248\u672c\u4e0d\u53d1\u9001\u8868\u60c5\u5305\u3002",
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
                    "content": "You are a Chinese chat style simulator for a consent-based local chatbot. Be concise, natural, and privacy-preserving.",
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
                    "content": "You are a concise Chinese chat style simulator for a consent-based local chatbot. Primary style source is backup/user-side messages, but do not mechanically repeat high-frequency filler words. The current chatting user is NonForgetter. If multiple user lines arrive together, understand them as one turn before replying. Do not claim to be the real person.",
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
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("text"):
                    parts.append(str(item["text"]))
            return "\n".join(parts)
        return ""

    def local_reply(
        self,
        message: str,
        memories: list[dict[str, Any]],
        conversation_memory: list[dict[str, Any]] | None = None,
    ) -> str:
        stripped = message.strip()
        cues = reply_cues(stripped)
        if not stripped:
            return "\u55ef\uff1f"
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if len(lines) > 1:
            joined = " / ".join(lines[-3:])
            if any(token in stripped for token in ["为什么", "咋", "怎么", "？", "?"]):
                return pick(["等下\n我先看完", "你这是连着问我啊\n我想想", "你这几句要放一起看"], joined)
            return pick(["我懂你意思了", "等下\n你这几句是连着的吧", "我先把你这几句当一件事看"], joined)
        if any(token in stripped for token in ["\u53eb\u4ec0\u4e48", "\u4ec0\u4e48\u540d\u5b57", "\u4f60\u662f\u8c01", "\u53eb\u5565"]):
            return pick(["\u4f60\u4e0d\u662f\u77e5\u9053\u561b", "\u8fd9\u8fd8\u8981\u95ee\u554a", "\u4f60\u60f3\u600e\u4e48\u53eb\u90fd\u884c"], stripped)
        if any(token in stripped for token in ["\u5e2e\u6211\u505a\u4ec0\u4e48", "\u80fd\u505a\u4ec0\u4e48", "\u4f1a\u505a\u4ec0\u4e48", "\u53ef\u4ee5\u5e2e\u6211"]):
            return pick(["陪你聊天啊", "你说，我听着", "乱七八糟的也可以说"], stripped)
        if any(word in stripped for word in ["\u7d2f", "\u70e6", "\u96be\u53d7", "\u5d29\u6e83", "\u4e0d\u5f00\u5fc3"]):
            return pick(["\u5148\u522b\u786c\u6491\uff0c\u7f13\u4e00\u4e0b\u3002", "\u600e\u4e48\u4e86\uff0c\u4f60\u8bf4\u3002", "\u597d\u5566\uff0c\u5148\u522b\u628a\u81ea\u5df1\u903c\u592a\u7d27\u3002"], stripped)
        if stripped in {"hi", "hello", "\u4f60\u597d", "\u5728\u5417", "\u5728\u4e0d\u5728"}:
            return pick(["\u5728\u5462", "\u55ef\uff1f", "\u600e\u4e48\u5566"], stripped)
        if any(token in stripped for token in ["\u8bb2\u8bb2", "\u8bf4\u8bf4", "\u5c55\u5f00", "\u8be6\u7ec6"]):
            return pick(["\u4f60\u5148\u8bf4\u662f\u54ea\u4e00\u4e2a", "\u4f60\u5177\u4f53\u8bf4\u54ea\u4e2a", "\u7b49\u4e0b\uff0c\u4f60\u8bf4\u6e05\u695a\u4e00\u70b9"], stripped)
        if any(token in stripped for token in ["\u65e0\u804a", "\u968f\u4fbf", "\u8bdd\u9898", "\u804a\u4ec0\u4e48"]):
            return pick(
                [
                    "\u90a3\u8bb2\u516b\u5366\n\u6211\u8981\u542c",
                    "\u968f\u4fbf\u554a\n\u4f60\u4eca\u5929\u6709\u6ca1\u6709\u4ec0\u4e48\u79bb\u8c31\u7684",
                    "\u73a9\u4e0d\u73a9\n\u6216\u8005\u8bb2\u70b9\u9006\u5929\u7684",
                ],
                stripped,
            )
        if cues:
            lowered = stripped.lower()
            if any(token in lowered for token in ["xswl", "\u7b11\u6b7b", "\u7b11\u9f20", "\u7ef7", "\u868c\u57e0\u4f4f", "\u8349", "\u8279"]):
                return pick(["6", "\u7ef7\u4e0d\u4f4f\u4e86", "\u54c8", "\u4ec0\u4e48\u4e1c\u897f\u554a"], stripped)
            if any(token in stripped for token in ["\u5bc4", "\u4e38\u8fa3", "\u5b8c\u8fa3", "\u9ebb\u4e86", "\u7834\u9632", "\u6446\u70c2"]):
                return pick(["\u5b8c\u4e86", "\u9ebb\u4e86", "\u5148\u522b\u5bc4", "\u8fd9\u4e5f\u592a\u79bb\u8c31\u4e86"], stripped)
            if any(token in stripped for token in ["\u79bb\u8c31", "\u9006\u5929", "\u62bd\u8c61"]):
                return pick(["\u9006\u5929", "\u6709\u70b9\u62bd\u8c61", "\u8fd9\u4ec0\u4e48", "6"], stripped)
            if any(token in stripped for token in ["\u5c0a\u561f\u5047\u561f", "\u771f\u7684\u5047\u7684", "\u771f\u5047\u7684"]):
                return pick(["\u554a\uff1f", "\u771f\u7684\u5047\u7684", "\u4ec0\u4e48", "\u4f60\u518d\u8bf4\u4e00\u904d"], stripped)

        candidates = self.extract_style_lines(memories)
        if "?" in stripped or "\uff1f" in stripped:
            return pick(candidates + ["怎么说", "什么", "啊？"], stripped)
        return pick(candidates + ["6", "哈", "我靠", "你继续", "有点意思"], stripped)

    def polish_model_reply(self, text: str, message: str, memories: list[dict[str, Any]]) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return self.local_reply(message, memories)
        if re.fullmatch(r"[?？]{1,8}", cleaned):
            return self.softened_question_reply(message, memories)
        if self.is_capability_question(message) and cleaned in {"\u4e0d\u77e5\u9053", "\u4e0d\u6e05\u695a", "\u8bf4\u4e0d\u6e05\u695a"}:
            return self.local_reply(message, memories)
        return cleaned

    def softened_question_reply(self, message: str, memories: list[dict[str, Any]]) -> str:
        digest = hashlib.sha256(message.encode("utf-8")).hexdigest()
        allow_symbol = int(digest[:2], 16) % 100 < 20
        if allow_symbol:
            return pick(["?", "\uff1f\uff1f", "\u554a\uff1f"], message)
        candidates = [line for line in self.extract_style_lines(memories) if not re.fullmatch(r"[?？!！。.,，、…~～]+", line)]
        return pick(candidates + ["\u4ec0\u4e48", "\u600e\u4e48\u8bf4", "\u4f60\u8bf4", "\u554a"], message)

    @staticmethod
    def is_capability_question(message: str) -> bool:
        return any(token in message for token in ["\u5e2e\u6211\u505a\u4ec0\u4e48", "\u80fd\u505a\u4ec0\u4e48", "\u4f1a\u505a\u4ec0\u4e48", "\u53ef\u4ee5\u5e2e\u6211"])

    @staticmethod
    def is_identity_question(message: str) -> bool:
        return any(token in message for token in ["\u53eb\u4ec0\u4e48", "\u4ec0\u4e48\u540d\u5b57", "\u4f60\u662f\u8c01", "\u53eb\u5565"])

    @staticmethod
    def extract_style_lines(memories: list[dict[str, Any]]) -> list[str]:
        candidates: list[str] = []
        seen: set[str] = set()
        for memory in memories:
            for line in (memory.get("text") or "").splitlines():
                if not line.startswith("user[text]:"):
                    continue
                text = line.split(":", 1)[1].strip()
                if text in OVERUSED_STYLE_LINES:
                    continue
                if is_usable_style_line(text) and text not in seen:
                    seen.add(text)
                    candidates.append(text)
        return candidates[:16]
