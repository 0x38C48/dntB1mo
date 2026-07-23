from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any

from .config import AppConfig
from .facts_runtime import days_since, top_english_name, top_identity_name
from .retrieval import Retriever
from .slang import reply_cues
from .textfix import fix_text
from .web_search import search_web, web_context_for_prompt


RUNTIME_VERSION = "backup-user-style-v17-grounded-preferences"

DEFAULT_MAX_REPLY_CHARS = 28
FACT_MAX_REPLY_CHARS = 18
BUBBLE_MAX_CHARS = 12


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


def self_match_any(text: str, tokens: list[str]) -> bool:
    return any(token in text for token in tokens)


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


def compact_reply(text: str, max_chars: int = DEFAULT_MAX_REPLY_CHARS) -> str:
    text = fix_text(text).strip().strip("“”\"")
    if not text:
        return ""
    text = re.sub(r"^(根据|从|按)(聊天记录|记录|事实卡|检索结果)[，,:：]?", "", text)
    text = re.sub(r"(我查到|可以看到|证据显示)[，,:：]?", "", text)
    text = re.sub(r"你是不是在[^，。！？?\n]{0,12}", "你又来了", text)
    text = re.sub(r"你复读机[^，。！？?\n]{0,12}", "复读机啊", text)
    text = re.sub(r"你今天跟这[^，。！？?\n]{0,16}", "过不去了是吧", text)
    text = text.replace("又嘻嘻，", "嘻嘻")
    text = text.replace("又来了，什么又", "什么又")
    text = text.replace("那咋了那咋了，", "那咋了")
    text = re.sub(r"[ \t\r\f\v]+", " ", text).strip()

    parts = [part.strip() for part in re.split(r"[\n。；;，,！!？?]+", text) if part.strip()]
    if not parts:
        return text[:min(max_chars, BUBBLE_MAX_CHARS)]

    bubbles: list[str] = []
    for part in parts:
        short = shrink_bubble(part)
        if short and short not in bubbles:
            bubbles.append(short)
        if len(bubbles) >= 2:
            break
    if not bubbles:
        bubbles = [text[:BUBBLE_MAX_CHARS]]

    joined = "\n".join(bubbles)
    if len(joined) <= max_chars:
        return joined
    return "\n".join(bubbles[:1])


def compact_explanation(text: str, max_bubbles: int = 3, bubble_chars: int = 24) -> str:
    cleaned = fix_text(text).strip().strip("“”\"")
    if not cleaned:
        return ""
    cleaned = re.sub(r"^(联网查了下|搜了下|根据搜索结果)[，,:：\s]*", "", cleaned)
    pieces = [part.strip() for part in re.split(r"[\n。；;！!]+", cleaned) if part.strip()]
    bubbles: list[str] = []
    for piece in pieces:
        piece = re.sub(r"\s+", " ", piece)
        if len(piece) > bubble_chars:
            split_parts = re.split(r"[，,、：:]", piece)
            for sub in split_parts:
                sub = sub.strip()
                if not sub:
                    continue
                bubbles.append(sub[:bubble_chars])
                if len(bubbles) >= max_bubbles:
                    return "\n".join(bubbles)
        else:
            bubbles.append(piece)
        if len(bubbles) >= max_bubbles:
            break
    return "\n".join(bubbles[:max_bubbles]) if bubbles else cleaned[:bubble_chars]


def shrink_bubble(text: str) -> str:
    text = text.strip()
    if len(text) <= BUBBLE_MAX_CHARS:
        return text
    for marker in ("是吧", "吧", "啊", "嘛", "呢", "啦", "了", "？", "?"):
        idx = text.find(marker)
        if idx >= 0 and idx + len(marker) <= BUBBLE_MAX_CHARS + 2:
            return text[: idx + len(marker)]
    return text[:BUBBLE_MAX_CHARS]


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
        fact_domain = self.classify_fact_domain(message)
        retrieval_query = self.build_retrieval_query(message, history)
        memories = self.retriever.search(retrieval_query, limit=10)
        memories = self.rerank_memories_for_question(message, memories)
        temporary_facts = self.extract_temporary_facts(message, memories, fact_domain)
        emotion = self.resolve_emotion(message, history, mood)
        web_results = search_web(message, limit=4) if self.should_web_lookup(message) else []

        fact_reply = self.fact_first_reply(message, history, memories)
        if fact_reply:
            if self.is_recent_chat_memory_question(message):
                fact_text = fact_reply
            else:
                fact_text = fact_reply if self.allows_long_reply(message) else compact_reply(fact_reply, FACT_MAX_REPLY_CHARS)
            return {
                "reply": fact_text,
                "mode": f"{self.mode}_fact_route",
                "emotion": emotion,
                "facts": self.public_facts(),
                "memories": memories,
                "web_results": web_results,
            }

        if self.is_memory_dispute_question(message) and self.has_memory_evidence(message, history, memories):
            return {
                "reply": compact_reply(self.memory_evidence_reply(message, history, memories), FACT_MAX_REPLY_CHARS),
                "mode": f"{self.mode}_evidence_route",
                "emotion": emotion,
                "facts": self.public_facts(),
                "memories": memories,
                "web_results": web_results,
            }

        if web_results:
            return {
                "reply": self.web_meaning_reply(message, web_results),
                "mode": "web_search_route",
                "emotion": emotion,
                "facts": self.public_facts(),
                "memories": memories,
                "web_results": web_results,
            }

        if self.should_route_locally(message):
            return {
                "reply": compact_reply(self.local_reply(message, memories, conversation_memory, emotion), DEFAULT_MAX_REPLY_CHARS),
                "mode": f"{self.mode}_local_route",
                "emotion": emotion,
                "facts": self.public_facts(),
                "memories": memories,
                "web_results": web_results,
            }

        prompt = self.build_prompt(message, history[-48:], memories, conversation_memory, emotion, fact_domain, temporary_facts, web_results)
        if self.config.sophnet_api_key:
            try:
                text = self.call_sophnet_api(prompt)
                return {
                    "reply": self.polish_model_reply(text, message, memories, history, conversation_memory, emotion),
                    "mode": self.mode,
                    "emotion": emotion,
                    "facts": self.public_facts(),
                    "memories": memories,
                    "web_results": web_results,
                }
            except Exception as exc:
                return {
                    "reply": compact_reply(self.local_reply(message, memories, conversation_memory, emotion), DEFAULT_MAX_REPLY_CHARS),
                    "mode": "local_fallback_after_sophnet_error",
                    "api_error": str(exc),
                    "emotion": emotion,
                    "facts": self.public_facts(),
                    "memories": memories,
                    "web_results": web_results,
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
                    "web_results": web_results,
                }
            except Exception as exc:
                return {
                    "reply": compact_reply(self.local_reply(message, memories, conversation_memory, emotion), DEFAULT_MAX_REPLY_CHARS),
                    "mode": "local_fallback_after_api_error",
                    "api_error": str(exc),
                    "emotion": emotion,
                    "facts": self.public_facts(),
                    "memories": memories,
                    "web_results": web_results,
                }
        return {
            "reply": compact_reply(self.local_reply(message, memories, conversation_memory, emotion), DEFAULT_MAX_REPLY_CHARS),
            "mode": self.mode,
            "emotion": emotion,
            "facts": self.public_facts(),
            "memories": memories,
            "web_results": web_results,
        }

    def build_prompt(
        self,
        message: str,
        history: list[dict[str, Any]],
        memories: list[dict[str, Any]],
        conversation_memory: list[dict[str, Any]],
        emotion: str,
        fact_domain: str,
        temporary_facts: dict[str, Any],
        web_results: list[dict[str, str]] | None = None,
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
            "fact_domain": fact_domain,
            "identity_and_timeline_facts": self.public_facts(),
            "temporary_facts_from_retrieval": temporary_facts,
            "fact_retrieval_policy": self.facts.get("retrieval_policy", {}),
            "persona_five_axes": persona_brief,
            "recent_history": history,
            "conversation_evidence": self.extract_conversation_evidence(message, history),
            "recent_assistant_replies": recent_assistant_replies,
            "conversation_memory": conversation_memory,
            "retrieved_memories": memories,
            "web_search": web_context_for_prompt(web_results or []),
            "recent_dialogue_state": self.recent_dialogue_state(history),
            "style_samples_from_backup": self.extract_style_lines(memories)[:12],
            "slang_and_homophone_cues": reply_cues(message),
            "user_message": message,
            "output_rules": [
                "默认像微信短消息：1-2个短泡泡，单泡泡尽量不超过12个中文字符。",
                "除非用户明确要求详细，不要解释来龙去脉，不要写完整报告句。",
                "能说“又来了”就不要说“你是不是又在……”。",
                "避免固定长模板：“你是不是…是吧”“你今天跟…过不去了吧”这类句式少用。",
                "事实只点到即可，比如“林薇艺\\nlily”或“2024-10-13”。",
                "优先保留个人语气：吐槽、反问、停顿、轻微敷衍可以有，别太端着。",
                "style_samples_from_backup 的权重高于抽象总结；学节奏和用词，不要逐字复读。",
                "只输出回复正文，不解释检索过程。",
                "可以很短，也可以分成连续几句，但要针对当前这句，不要套模板。",
                "身份/名字/时间/是否说过的问题必须优先使用 identity_and_timeline_facts 和 retrieved_memories。",
                "生日、情感经历、情绪起伏等事实问题必须先使用 temporary_facts_from_retrieval；证据不足只能说像是/没稳，不要编。",
                "如果 temporary_facts_from_retrieval 已经足够回答，只做语气压缩，不要改事实。",
                "有明确证据时不要说“不知道/没印象”。",
                "top_short_phrases 只是风格参考，不是复读清单。",
                "避免重复 recent_assistant_replies 中的句式；意思接近也要换角度。",
                "遇到多行 user_message，代表 NonForgetter 连续发了多条消息，要整体理解后回复。",
                "遇到“刚刚/刚才/上一句/之前我说了什么/你回答了什么”，必须优先看 recent_dialogue_state，不要装作没看到。",
                "遇到谐音梗、网络梗、拼音缩写时按语境接住情绪和笑点，不要机械解释。",
                "如果 web_search.result_count > 0，说明已经联网；解释梗时优先综合 web_search，证据弱就说像是/可能是。",
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
                    "content": "You simulate backup's concise Chinese WeChat style. Reply in 1-2 micro chat bubbles, usually under 12 Chinese chars each. Evidence outranks improvisation, but never sound like a report.",
                },
                {"role": "user", "content": prompt},
            ],
            "max_output_tokens": 80,
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
                    "content": "You simulate backup's concise Chinese WeChat style. Reply in 1-2 micro chat bubbles, usually under 12 Chinese chars each. Evidence outranks improvisation, but never sound like a report.",
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
            return compact_reply(self.memory_evidence_reply(message, history or [], memories), FACT_MAX_REPLY_CHARS)
        if self.is_bot_identity_question(message) and self.looks_like_denial(cleaned):
            return compact_reply(self.fact_first_reply(message, history or [], memories) or cleaned, FACT_MAX_REPLY_CHARS)
        max_chars = 90 if self.allows_long_reply(message) else DEFAULT_MAX_REPLY_CHARS
        return compact_reply(cleaned, max_chars)

    def polish_web_reply(self, text: str, message: str, web_results: list[dict[str, str]]) -> str:
        cleaned = fix_text(text).strip().strip("“”\"")
        if not cleaned or self.looks_like_denial(cleaned):
            return self.web_meaning_reply(message, web_results)
        return compact_explanation(cleaned)

    def fact_first_reply(self, message: str, history: list[dict[str, Any]], memories: list[dict[str, Any]]) -> str | None:
        stripped = message.strip()
        if self.is_recent_chat_memory_question(stripped):
            return self.recent_chat_reply(stripped, history)
        if self.is_identity_correction(stripped):
            return "哦对\n我串了"
        if self.is_user_identity_question(stripped):
            return self.user_identity_reply(stripped, history, memories)
        if self.is_bot_identity_question(stripped):
            name = top_identity_name(self.facts)
            english = top_english_name(self.facts)
            if self.is_bot_surname_question(stripped):
                if name and name[0]:
                    return f"姓{name[0]}"
                return "姓什么没稳"
            if name and english:
                return pick(
                    [
                        f"{name}\n{english}",
                        f"{name}吧",
                        f"{english}也有",
                    ],
                    stripped,
                )
            if name:
                return pick([name, f"{name}吧", f"就{name}"], stripped)
        if self.is_birthday_question(stripped):
            return self.birthday_reply(stripped)
        if self.is_relationship_question(stripped):
            return self.relationship_reply(stripped)
        if self.is_soothing_question(stripped):
            return self.soothing_reply(stripped)
        if self.is_topic_question(stripped):
            return self.shared_topic_reply(stripped)
        if self.is_emotion_history_question(stripped):
            return self.emotion_history_reply(stripped)
        if self.is_preference_question(stripped):
            return self.category_fact_reply(stripped, "preferences", memories)
        if self.is_nickname_question(stripped):
            return self.category_fact_reply(stripped, "nicknames")
        if self.is_habit_question(stripped):
            return self.category_fact_reply(stripped, "habits")
        if self.is_wake_time_question(stripped):
            return self.wake_time_reply(memories)
        if self.is_time_question(stripped):
            return self.time_reply(stripped)
        if self.is_memory_dispute_question(stripped) and self.has_memory_evidence(stripped, history, memories):
            return self.memory_evidence_reply(stripped, history, memories)
        return None

    @staticmethod
    def recent_chat_reply(message: str, history: list[dict[str, Any]]) -> str:
        def recent(role: str, limit: int = 3) -> list[str]:
            rows = [
                str(item.get("content", "")).strip()
                for item in history
                if item.get("role") == role and str(item.get("content", "")).strip()
            ]
            return rows[-limit:]

        user_rows = recent("user")
        assistant_rows = recent("assistant")
        if any(token in message for token in ["你刚刚", "你刚才", "你上一句", "你前面", "你回答", "她回答", "backup回答"]):
            if not assistant_rows:
                return "我还没回啥"
            return "我刚回：\n" + assistant_rows[-1][:48]
        if any(token in message for token in ["我刚刚", "我刚才", "我上一句", "我前面", "之前我", "前面我"]):
            if not user_rows:
                return "你还没说啥"
            return "你刚说：\n" + user_rows[-1][:48]
        rows = []
        for item in history[-6:]:
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            speaker = "你" if item.get("role") == "user" else "我"
            rows.append(f"{speaker}: {content[:34]}")
        return "刚才大概是：\n" + "\n".join(rows[-4:]) if rows else "刚才没啥"

    def user_identity_reply(self, message: str, history: list[dict[str, Any]], memories: list[dict[str, Any]]) -> str:
        user_name = self.extract_user_identity_from_history(history, "name") or self.extract_user_identity_from_memories(memories, "name")
        user_surname = self.extract_user_identity_from_history(history, "surname") or self.extract_user_identity_from_memories(memories, "surname")
        if any(token in message for token in ["姓什么", "我姓啥", "我姓什么", "我什么姓"]):
            if user_surname:
                return f"你姓{user_surname}"
            return "你没说稳\n别拿我的套你"
        if any(token in message for token in ["英文名", "english name", "English name"]):
            english = self.extract_user_identity_from_history(history, "english")
            if english:
                return english
            return "你英文名\n我没记稳"
        if user_name:
            return user_name
        return "NonForgetter吧\n真名没稳"

    @staticmethod
    def extract_user_identity_from_history(history: list[dict[str, Any]], kind: str) -> str | None:
        patterns = {
            "surname": [
                r"我姓([\u4e00-\u9fff])",
                r"我.*?姓氏是([\u4e00-\u9fff])",
                r"记住.*?我姓([\u4e00-\u9fff])",
            ],
            "name": [
                r"我叫([\u4e00-\u9fffA-Za-z0-9_\-]{1,16})",
                r"我的名字是([\u4e00-\u9fffA-Za-z0-9_\-]{1,16})",
                r"记住.*?我叫([\u4e00-\u9fffA-Za-z0-9_\-]{1,16})",
            ],
            "english": [
                r"我(?:的)?英文名(?:字)?(?:是|叫)([A-Za-z][A-Za-z0-9_\-]{1,20})",
                r"my name is ([A-Za-z][A-Za-z0-9_\-]{1,20})",
            ],
        }
        for item in reversed(history):
            if item.get("role") != "user":
                continue
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            for pattern in patterns.get(kind, []):
                match = re.search(pattern, content, flags=re.I)
                if match:
                    value = match.group(1).strip()
                    if value and value not in {"什么", "啥", "谁", "知道吗"}:
                        return value
        return None

    @staticmethod
    def extract_user_identity_from_memories(memories: list[dict[str, Any]], kind: str) -> str | None:
        patterns = {
            "surname": [
                r"target\[text\]:\s*我姓([\u4e00-\u9fff])",
                r"target\[text\]:\s*我.*?姓氏是([\u4e00-\u9fff])",
                r"user\[text\]:\s*你姓([\u4e00-\u9fff])",
            ],
            "name": [
                r"target\[text\]:\s*我叫([\u4e00-\u9fffA-Za-z0-9_\-]{1,16})",
                r"target\[text\]:\s*我的名字是([\u4e00-\u9fffA-Za-z0-9_\-]{1,16})",
                r"user\[text\]:\s*你叫([\u4e00-\u9fffA-Za-z0-9_\-]{1,16})",
            ],
            "english": [
                r"target\[text\]:\s*我(?:的)?英文名(?:字)?(?:是|叫)([A-Za-z][A-Za-z0-9_\-]{1,20})",
                r"user\[text\]:\s*你(?:的)?英文名(?:字)?(?:是|叫)?([A-Za-z][A-Za-z0-9_\-]{1,20})",
            ],
        }
        blocked = {"林", "林薇艺", "lily", "Lily", "backup", "NonForgetter"}
        for memory in memories:
            text = fix_text(memory.get("text") or "")
            for pattern in patterns.get(kind, []):
                for match in re.finditer(pattern, text, flags=re.I):
                    value = match.group(1).strip()
                    if value and value not in blocked and value not in {"什么", "啥", "谁", "知道吗"}:
                        return value
        return None

    def birthday_reply(self, message: str) -> str | None:
        person = "backup"
        if any(token in message for token in ["我生日", "我的生日", "我哪天", "我几号"]):
            person = "NonForgetter"
        if any(token in message for token in ["你生日", "你的生日", "你哪天", "你几号"]):
            person = "backup"
        candidates = (self.facts.get("birthdays", {}).get(person) or [])[:3]
        if not candidates:
            return "记录里没稳"
        top = candidates[0]
        if top.get("score", 0) <= 1 and len(candidates) > 1:
            return f"{top.get('value')}\n但没很稳"
        return f"{top.get('value')}\n像是这个"

    def relationship_reply(self, message: str) -> str | None:
        categories = self.facts.get("relationship_history", {}).get("categories") or []
        if not categories:
            return None
        values = [item.get("value") for item in categories[:5]]
        if self.allows_long_reply(message):
            return "挺熟\n会拉扯\n会吵也会哄\n照顾也多"
        if "conflict" in values and "repair" in values:
            return "有拉扯\n也有哄"
        if "care" in values and "affection" in values:
            return "挺熟的\n有照顾"
        if values:
            return "记录里有\n得看片段"
        return None

    def emotion_history_reply(self, message: str) -> str | None:
        emotions = self.facts.get("emotion_patterns", {}).get("backup") or []
        if not emotions:
            return None
        values = [item.get("value") for item in emotions[:4]]
        if self.allows_long_reply(message):
            return "困累很多\n会低落\n也会生气\n但能接梗"
        if "low" in values and "angry" in values:
            return "起伏挺大\n会低落会炸"
        if "sleepy_tired" in values:
            return "常见是困累"
        if values:
            return "有波动\n记录里能看"
        return None

    def category_fact_reply(self, message: str, section: str, memories: list[dict[str, Any]] | None = None) -> str | None:
        memories = memories or []
        if section == "preferences":
            if self.is_play_preference_question(message):
                return self.play_preference_reply(message, memories)
            if self.is_food_preference_question(message):
                return self.food_preference_reply(message, memories)
            if any(token in message for token in ["喜欢什么", "喜欢啥", "有什么喜欢", "偏好"]):
                return "太泛了\n问吃的还是玩的"
        person = "backup"
        if section == "nicknames":
            if any(token in message for token in ["叫我", "喊我", "称呼我"]):
                person = "NonForgetter"
            elif any(token in message for token in ["叫你", "喊你", "称呼你"]):
                person = "backup"
        else:
            if any(token in message for token in ["我", "我的", "我平时", "我喜欢", "我讨厌"]):
                person = "NonForgetter"
            if any(token in message for token in ["你", "你的", "你平时", "你喜欢", "你讨厌"]):
                person = "backup"
        rows = (self.facts.get(section, {}).get(person) or [])[:4]
        if not rows:
            return "记录里没稳"
        labels = [self.display_fact_label(row.get("value", "")) for row in rows[:2]]
        if len(labels) == 1:
            return f"像是{labels[0]}"
        return f"{labels[0]}\n还有{labels[1]}"

    def play_preference_reply(self, message: str, memories: list[dict[str, Any]]) -> str:
        person = "NonForgetter" if any(token in message for token in ["我喜欢", "我爱玩", "我玩什么"]) else "backup"
        labels = self.extract_play_labels(memories, person)
        if labels:
            if len(labels) == 1:
                return f"只翻稳{labels[0]}\n别的没稳"
            if len(labels) == 2:
                return f"{labels[0]}\n还有{labels[1]}"
            return f"{labels[0]}、{labels[1]}\n{labels[2]}也有"
        if self.has_shared_topic("games"):
            return "记录里像二游\n具体要再翻"
        return "没翻到稳的\n别硬猜"

    def rerank_memories_for_question(self, message: str, memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.is_play_preference_question(message):
            return memories
        person = "NonForgetter" if any(token in message for token in ["我喜欢", "我爱玩", "我玩什么"]) else "backup"
        scored = [
            (self.play_memory_score(memory, person), float(memory.get("score") or 0), index, memory)
            for index, memory in enumerate(memories)
        ]
        scored.sort(key=lambda item: (item[0], item[1], -item[2]), reverse=True)
        return [memory for _, _, _, memory in scored]

    @staticmethod
    def play_memory_score(memory: dict[str, Any], person: str) -> int:
        role_prefix = "target[text]:" if person == "NonForgetter" else "user[text]:"
        score = 0
        for line in fix_text(memory.get("text") or "").splitlines():
            line = line.strip()
            if not line.startswith(role_prefix):
                continue
            content = line.split(":", 1)[1].strip()
            labels = ChatEngine.extract_play_labels([{"text": line}], person)
            if labels and ChatEngine.is_positive_play_context(content):
                score += 10 + len(labels)
        return score

    def food_preference_reply(self, message: str, memories: list[dict[str, Any]]) -> str:
        labels = self.extract_food_labels(memories)
        if labels:
            return "\n".join(labels[:2])
        return "吃的没稳\n只看到常聊吃饭"

    @staticmethod
    def extract_play_labels(memories: list[dict[str, Any]], person: str) -> list[str]:
        role_prefix = "target[text]:" if person == "NonForgetter" else "user[text]:"
        aliases: list[tuple[str, tuple[str, ...]]] = [
            ("原神", ("原神", "尘歌壶", "跑图")),
            ("星铁", ("星铁", "崩铁", "星穹铁道")),
            ("绝区零", ("绝区零", "zzz", "ZZZ")),
            ("鸣潮", ("鸣潮",)),
            ("Steam", ("steam", "Steam")),
            ("瓦", ("玩瓦", "打瓦", "瓦罗兰特", "无畏契约")),
            ("MC", ("mc", "MC", "我的世界")),
        ]
        counts: dict[str, int] = {label: 0 for label, _ in aliases}
        for memory in memories:
            for line in fix_text(memory.get("text") or "").splitlines():
                line = line.strip()
                if not line.startswith(role_prefix):
                    continue
                content = line.split(":", 1)[1].strip()
                for label, terms in aliases:
                    if any(term in content for term in terms) and ChatEngine.is_positive_play_context(content):
                        counts[label] += 1
        ranked = sorted(((count, label) for label, count in counts.items() if count > 0), reverse=True)
        return [label for _, label in ranked[:4]]

    @staticmethod
    def is_positive_play_context(content: str) -> bool:
        negative = ["没玩", "不玩", "没打", "不打", "不好玩", "不想玩", "玩不来", "没玩过", "不玩啦"]
        if any(token in content for token in negative):
            return False
        activity = ["玩", "打", "登", "上号", "任务", "跑图", "尘歌壶", "抽", "通关", "开黑", "退", "级"]
        return any(token in content for token in activity)

    @staticmethod
    def extract_food_labels(memories: list[dict[str, Any]]) -> list[str]:
        aliases: list[tuple[str, tuple[str, ...]]] = [
            ("咖啡/瑞幸", ("瑞幸", "咖啡")),
            ("不太想吃食堂", ("不想吃食堂", "食堂")),
            ("吃饭", ("吃饭", "晚饭", "午饭")),
        ]
        counts: dict[str, int] = {label: 0 for label, _ in aliases}
        for memory in memories:
            for line in fix_text(memory.get("text") or "").splitlines():
                if not line.strip().startswith("user[text]:"):
                    continue
                content = line.split(":", 1)[1].strip()
                for label, terms in aliases:
                    if any(term in content for term in terms):
                        counts[label] += 1
        ranked = sorted(((count, label) for label, count in counts.items() if count > 0), reverse=True)
        return [label for _, label in ranked[:3]]

    def has_shared_topic(self, value: str) -> bool:
        return any(item.get("value") == value for item in self.facts.get("shared_topics") or [])

    def shared_topic_reply(self, message: str) -> str | None:
        rows = (self.facts.get("shared_topics") or [])[:4]
        if not rows:
            return "记录里没稳"
        labels = [self.display_fact_label(row.get("value", "")) for row in rows[:3]]
        return "\n".join(labels[:3])

    def soothing_reply(self, message: str) -> str | None:
        rows = (self.facts.get("soothing_patterns") or [])[:3]
        if not rows:
            return "记录里没稳"
        labels = [self.display_fact_label(row.get("value", "")) for row in rows[:2]]
        return "\n".join(labels)

    @staticmethod
    def display_fact_label(value: str) -> str:
        labels = {
            "like": "有喜欢的会直说",
            "dislike": "不喜欢也会说",
            "food": "吃的",
            "media_game": "游戏二游",
            "money_gift": "钱和礼物",
            "sleep_wake": "睡觉起床",
            "study_work": "学习考试",
            "meal": "吃饭",
            "family": "家里人",
            "health": "身体状态",
            "games": "游戏",
            "sleep": "睡觉",
            "study_exam": "学习考试",
            "anime_music": "动画和歌",
            "family_daily": "家里日常",
            "sleep_rest": "让你睡觉",
            "comfort": "抱抱安慰",
            "deescalate": "先哄住",
            "feed_care": "催吃饭",
        }
        return labels.get(value, value)

    def time_reply(self, message: str) -> str:
        start = self.facts.get("relationship", {}).get("known_since") or self.facts.get("date_range", {}).get("start")
        days = days_since(start, datetime.now())
        date = (start or "2024-10-13")[:10]
        return pick(
            [
                date,
                f"{date}\n挺久了",
                f"从{date}算",
            ],
            message,
        )

    def wake_time_reply(self, memories: list[dict[str, Any]]) -> str:
        samples: list[tuple[float, str]] = []
        for memory in memories:
            timestamp = str(memory.get("start_time") or "")
            chunk_hour = self.hour_from_timestamp(timestamp)
            for raw_line in fix_text(memory.get("text") or "").splitlines():
                if not raw_line.startswith("user[text]:"):
                    continue
                line = raw_line.split(":", 1)[1].strip()
                if not self.is_wake_evidence_line(line):
                    continue
                explicit = self.hour_from_text(line)
                if explicit is not None and self.looks_like_sleep_time_line(line):
                    continue
                if explicit is not None:
                    samples.append((explicit, line))
                    continue
                if chunk_hour is not None:
                    if any(token in line for token in ["还没起", "起不来", "没睡醒"]):
                        samples.append((max(chunk_hour + 1, 11), line))
                    elif any(token in line for token in ["才起床", "刚起床", "刚睡醒", "睡醒了", "醒了", "起床"]):
                        samples.append((chunk_hour, line))

        if not samples:
            return "记录里没准点\n但不像六点"

        weighted = sorted(samples, key=lambda item: item[0])
        median = weighted[len(weighted) // 2][0]
        if median < 9 and any(hour >= 11 for hour, _ in weighted):
            median = sorted(hour for hour, _ in weighted if hour >= 10)[0]
        if median >= 15:
            guess = "中午到下午吧"
        elif median >= 12:
            guess = "十二点前后吧"
        elif median >= 10.5:
            guess = "十一二点吧"
        else:
            guess = "十点多吧"
        return f"{guess}\n不像六点"

    @staticmethod
    def is_wake_evidence_line(line: str) -> bool:
        if any(token in line for token in ["你起", "你醒", "你什么时候", "叫醒了她", "我爸醒", "他起床", "ropz"]):
            return False
        return any(
            token in line
            for token in ["起床", "睡醒", "醒了", "没睡醒", "还没起", "起不来", "才起床", "刚起床", "几点醒", "几点起"]
        )

    @staticmethod
    def looks_like_sleep_time_line(line: str) -> bool:
        return any(token in line for token in ["几点睡", "什么时候睡", "睡啊", "睡觉", "睡着", "晚安"])

    @staticmethod
    def hour_from_timestamp(timestamp: str) -> float | None:
        try:
            return float(datetime.strptime(timestamp[:19], "%Y-%m-%d %H:%M:%S").hour)
        except ValueError:
            return None

    @staticmethod
    def hour_from_text(line: str) -> float | None:
        match = re.search(r"(\d{1,2}|[一二两三四五六七八九十]{1,3})点(半|多)?", line)
        if not match:
            return None
        hour = ChatEngine.parse_cn_hour(match.group(1))
        if hour is None:
            return None
        if match.group(2) == "半":
            hour += 0.5
        elif match.group(2) == "多":
            hour += 0.3
        if any(token in line for token in ["下午", "中午"]) and hour < 8:
            hour += 12
        return hour

    @staticmethod
    def parse_cn_hour(value: str) -> float | None:
        if value.isdigit():
            hour = int(value)
            return float(hour) if 0 <= hour <= 24 else None
        digits = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
        if value == "十":
            return 10.0
        if value.startswith("十") and len(value) == 2:
            return float(10 + digits.get(value[1], 0))
        if value.endswith("十") and len(value) == 2:
            return float(digits.get(value[0], 0) * 10)
        if "十" in value:
            left, right = value.split("十", 1)
            return float(digits.get(left, 1) * 10 + digits.get(right, 0))
        if value in digits:
            return float(digits[value])
        return None

    def public_facts(self) -> dict[str, Any]:
        start = self.facts.get("relationship", {}).get("known_since") or self.facts.get("date_range", {}).get("start")
        birthdays = {
            key: [
                {
                    "value": item.get("value"),
                    "score": item.get("score"),
                    "evidence": (item.get("evidence") or [])[:2],
                }
                for item in values[:3]
            ]
            for key, values in (self.facts.get("birthdays") or {}).items()
        }
        relationship = [
            {
                "value": item.get("value"),
                "score": item.get("score"),
                "evidence": (item.get("evidence") or [])[:2],
            }
            for item in (self.facts.get("relationship_history", {}).get("categories") or [])[:6]
        ]
        emotions = {
            key: [
                {
                    "value": item.get("value"),
                    "score": item.get("score"),
                    "evidence": (item.get("evidence") or [])[:2],
                }
                for item in values[:5]
            ]
            for key, values in (self.facts.get("emotion_patterns") or {}).items()
        }
        preferences = self.slim_section("preferences")
        habits = self.slim_section("habits")
        nicknames = self.slim_section("nicknames")
        topics = [
            {
                "value": item.get("value"),
                "score": item.get("score"),
                "evidence": (item.get("evidence") or [])[:2],
            }
            for item in (self.facts.get("shared_topics") or [])[:8]
        ]
        soothing = [
            {
                "value": item.get("value"),
                "score": item.get("score"),
                "evidence": (item.get("evidence") or [])[:2],
            }
            for item in (self.facts.get("soothing_patterns") or [])[:6]
        ]
        return {
            "persona_display": self.facts.get("persona_display", "backup"),
            "current_user": self.facts.get("current_user", "NonForgetter"),
            "top_name": top_identity_name(self.facts),
            "top_english_name": top_english_name(self.facts),
            "known_since": start,
            "known_duration_days": days_since(start, datetime.now()),
            "name_evidence": (self.facts.get("identity", {}).get("name_candidates") or [])[:3],
            "english_name_evidence": (self.facts.get("identity", {}).get("english_name_candidates") or [])[:3],
            "birthdays": birthdays,
            "relationship_brief": relationship,
            "emotion_brief": emotions,
            "preference_brief": preferences,
            "habit_brief": habits,
            "nickname_brief": nicknames,
            "shared_topics": topics,
            "soothing_patterns": soothing,
        }

    def slim_section(self, section: str) -> dict[str, list[dict[str, Any]]]:
        return {
            key: [
                {
                    "value": item.get("value"),
                    "score": item.get("score"),
                    "evidence": (item.get("evidence") or [])[:2],
                }
                for item in values[:6]
            ]
            for key, values in (self.facts.get(section) or {}).items()
        }

    @staticmethod
    def fix_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {**item, "content": fix_text(item.get("content", ""))}
            for item in history
        ]

    @staticmethod
    def recent_dialogue_state(history: list[dict[str, Any]]) -> dict[str, Any]:
        recent = [
            {
                "role": str(item.get("role", "")),
                "content": str(item.get("content", "")).strip()[:160],
            }
            for item in history[-24:]
            if str(item.get("content", "")).strip()
        ]
        user_messages = [item["content"] for item in recent if item["role"] == "user"][-8:]
        assistant_replies = [item["content"] for item in recent if item["role"] == "assistant"][-8:]
        return {
            "last_turns": recent,
            "last_user_messages": user_messages,
            "last_assistant_replies": assistant_replies,
            "rule": "这是当前不断线聊天的短期记忆。用户追问刚才/之前说了什么时，直接引用这里，不要去猜。",
        }

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
    def web_meaning_reply(message: str, web_results: list[dict[str, str]]) -> str:
        titles = " / ".join(row.get("title", "") for row in web_results[:3] if row.get("title"))
        haystack = f"{message} {titles}"
        if any(token in haystack for token in ["别这么说", "别说这种话", "别说xx这种话", "别说 xx 这种话"]):
            return "查了下\n像劝停吐槽梗\n不算固定出处"
        if any(token in titles for token in ["你还别说", "真别说", "别说了", "下次别说"]):
            return "像那类梗\n让人别继续说\n带点吐槽"
        if web_results:
            title = fix_text(web_results[0].get("title", "")).strip("_ ")[:22]
            return f"搜到像这个\n{title}\n得看上下文"
        if "乐子" in message or "乐乐" in message:
            return "乐子那个乐\n看热闹的意思"
        return "没搜稳\n可能只是语境梗"

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
    def is_identity_correction(message: str) -> bool:
        return any(token in message for token in ["那是你的名字", "那tm是你的名字", "那是你名字", "这是你的名字", "不是我的名字"]) or (
            "林薇艺" in message and any(token in message for token in ["你", "不是我", "你的"])
        )

    @staticmethod
    def is_recent_chat_memory_question(message: str) -> bool:
        recent_tokens = ["刚刚", "刚才", "上一句", "前一句", "前面", "之前", "上面"]
        memory_tokens = ["说了什么", "说什么", "说过什么", "讲了什么", "聊了什么", "回答了什么", "回了什么", "你回答", "她回答"]
        if any(token in message for token in recent_tokens) and any(token in message for token in memory_tokens):
            return True
        return any(token in message for token in ["我刚刚说的啥", "我刚才说的啥", "你刚刚回的啥", "你刚才回的啥"])

    @staticmethod
    def should_web_lookup(message: str) -> bool:
        if len(message.strip()) > 80:
            return False
        lookup_tokens = ["是什么梗", "啥梗", "什么梗", "网络梗", "出自哪里", "出处", "热梗", "梗图"]
        meaning_tokens = ["什么意思", "啥意思", "什么含义"]
        if any(token in message for token in lookup_tokens):
            return True
        return any(token in message for token in meaning_tokens) and any(token in message for token in ["梗", "网络", "弹幕", "贴吧", "b站", "B站", "微博", "抖音"])

    @staticmethod
    def is_play_preference_question(message: str) -> bool:
        return any(token in message for token in ["喜欢玩什么", "喜欢玩啥", "爱玩什么", "爱玩啥", "平时玩什么", "平时玩啥", "玩什么游戏", "玩啥游戏", "玩什么"])

    @staticmethod
    def is_food_preference_question(message: str) -> bool:
        return any(token in message for token in ["喜欢吃什么", "喜欢吃啥", "爱吃什么", "爱吃啥", "喜欢喝什么", "爱喝什么", "吃什么", "喝什么"])

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
            return "有\n林薇艺"
        if "lily" in haystack.lower():
            return "有\nlily"
        if "乐乐" in haystack:
            return "乐乐那个梗吧"
        return "有\n我刚漏了"

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
            or ChatEngine.is_wake_time_question(message)
            or ChatEngine.is_birthday_question(message)
            or ChatEngine.is_preference_question(message)
            or ChatEngine.is_habit_question(message)
            or ChatEngine.is_nickname_question(message)
            or ChatEngine.is_topic_question(message)
            or ChatEngine.is_soothing_question(message)
        )

    @staticmethod
    def allows_long_reply(message: str) -> bool:
        return any(token in message for token in ["详细", "讲清楚", "展开", "分析", "解释一下", "说清楚"])

    @staticmethod
    def build_retrieval_query(message: str, history: list[dict[str, Any]]) -> str:
        recent_user_text = [
            str(item.get("content", ""))
            for item in history[-10:]
            if item.get("role") == "user" and item.get("content")
        ][-5:]
        expansions = []
        domain = ChatEngine.classify_fact_domain(message)
        if ChatEngine.is_user_identity_question(message):
            expansions.extend(["NonForgetter target 我姓 我叫 我的名字 我的英文名 你姓 你叫 你名字"])
        if ChatEngine.is_bot_identity_question(message):
            expansions.extend(["名字 姓 林 薇 艺 英文名 lily 猜名字"])
        if ChatEngine.is_time_question(message):
            expansions.extend(["认识 多久 开始 聊天 第一条"])
        if domain == "birthday":
            expansions.extend(["生日 生快 生日快乐 几号 哪天 出生 过生日 今天生日 明天生日"])
        if domain == "relationship":
            expansions.extend(
                [
                    "感情 情感经历 关系 喜欢 爱 想你 抱抱 陪你",
                    "吵架 生气 冷暴力 对不起 道歉 和好 原谅 别生气 不理",
                ]
            )
        if domain == "emotion":
            expansions.extend(["情绪 心情 难受 伤心 生气 哭 emo 破防 委屈 崩溃 困 累 起伏"])
        if domain == "preference":
            expansions.extend(["喜欢 不喜欢 讨厌 想吃 爱吃 好看 好听 想玩 想要 偏好"])
        if ChatEngine.is_play_preference_question(message):
            expansions.extend(["玩什么 游戏 原神 星铁 崩铁 绝区零 zzz 鸣潮 steam 瓦罗兰特 无畏契约 mc 尘歌壶 跑图"])
        if ChatEngine.is_food_preference_question(message):
            expansions.extend(["吃什么 喝什么 吃饭 晚饭 瑞幸 咖啡 食堂 饿了 代餐"])
        if domain == "habit":
            expansions.extend(["习惯 平时 睡觉 起床 熬夜 吃饭 上课 考试 家里 身体 不舒服"])
        if domain == "nickname":
            expansions.extend(["叫你 叫我 喊你 喊我 名字 昵称 外号 宝宝 姐 哥 猫"])
        if domain == "topic":
            expansions.extend(["平时聊 常聊 话题 游戏 原神 星铁 吃饭 睡觉 学校 动画 音乐 家里"])
        if domain == "soothing":
            expansions.extend(["怎么哄 安慰 抱抱 别难受 别生气 早点休息 多睡 吃饭 乖"])
        if ChatEngine.is_wake_time_question(message):
            expansions.extend(
                [
                    "几点起床 几点醒 睡醒 醒了 起床 起了 还没起床 还没起 没睡醒 起不来",
                    "十一点多 十二点 一点多 中午 下午 才起床 刚睡醒 自然醒",
                ]
            )
        return "\n".join([message, *recent_user_text, *expansions])

    @staticmethod
    def classify_fact_domain(message: str) -> str:
        if ChatEngine.is_bot_identity_question(message) or ChatEngine.is_user_identity_question(message):
            return "identity"
        if ChatEngine.is_birthday_question(message):
            return "birthday"
        if ChatEngine.is_relationship_question(message):
            return "relationship"
        if ChatEngine.is_soothing_question(message):
            return "soothing"
        if ChatEngine.is_topic_question(message):
            return "topic"
        if ChatEngine.is_emotion_history_question(message):
            return "emotion"
        if ChatEngine.is_wake_time_question(message):
            return "habit"
        if ChatEngine.is_preference_question(message):
            return "preference"
        if ChatEngine.is_habit_question(message):
            return "habit"
        if ChatEngine.is_nickname_question(message):
            return "nickname"
        if ChatEngine.is_time_question(message):
            return "time"
        if ChatEngine.is_memory_dispute_question(message):
            return "memory_dispute"
        if ChatEngine.is_slang_meaning_question(message):
            return "slang"
        return "open_chat"

    @staticmethod
    def extract_temporary_facts(message: str, memories: list[dict[str, Any]], domain: str) -> dict[str, Any]:
        terms = ChatEngine.domain_terms(domain)
        evidence: list[dict[str, str]] = []
        for memory in memories:
            chunk_id = str(memory.get("chunk_id") or "")
            timestamp = str(memory.get("start_time") or "")
            for raw_line in fix_text(memory.get("text") or "").splitlines():
                if not raw_line.startswith(("user[text]:", "target[text]:")):
                    continue
                role, text = raw_line.split(":", 1)
                text = text.strip()
                if not text:
                    continue
                if terms and not any(term in text for term in terms):
                    continue
                evidence.append(
                    {
                        "chunk_id": chunk_id,
                        "timestamp": timestamp,
                        "role": role.removesuffix("[text]"),
                        "text": text[:160],
                    }
                )
                if len(evidence) >= 10:
                    break
            if len(evidence) >= 10:
                break
        return {
            "domain": domain,
            "evidence_count": len(evidence),
            "evidence": evidence,
            "rule": "这些只是本轮临时事实；回答必须以它们和 facts.json 为准，不能自行补确定事实。",
        }

    @staticmethod
    def domain_terms(domain: str) -> list[str]:
        return {
            "birthday": ["生日", "生快", "生日快乐", "几号", "哪天", "出生", "过生日"],
            "relationship": ["喜欢", "爱", "想你", "抱抱", "陪你", "吵架", "生气", "冷暴力", "对不起", "和好", "原谅", "别生气", "不理"],
            "emotion": ["难受", "伤心", "生气", "哭", "emo", "破防", "委屈", "崩溃", "困", "累", "没睡醒", "烦"],
            "preference": ["喜欢", "不喜欢", "讨厌", "想吃", "爱吃", "好看", "好听", "想玩", "想要"],
            "habit": ["习惯", "平时", "睡", "起床", "睡醒", "醒了", "还没起", "起不来", "没睡醒", "熬夜", "吃饭", "上课", "考试", "家里", "不舒服"],
            "nickname": ["叫你", "叫我", "喊你", "喊我", "昵称", "名字", "宝宝", "姐", "哥", "猫"],
            "topic": ["原神", "星铁", "游戏", "吃饭", "睡觉", "学校", "动画", "歌", "家里"],
            "soothing": ["抱抱", "安慰", "别难受", "别生气", "早点休息", "多睡", "吃饭", "乖"],
            "memory_dispute": ["说过", "记得", "记错", "没说过", "绝对没有"],
            "identity": ["名字", "姓", "我叫", "你叫", "英文名", "NonForgetter", "林薇艺", "lily"],
        }.get(domain, [])

    @staticmethod
    def extract_conversation_evidence(message: str, history: list[dict[str, Any]]) -> list[dict[str, str]]:
        if not any(token in message for token in ["说过", "没说过", "记得", "记错", "是不是", "姓", "名字", "绝对没有", "刚才", "刚刚", "上一句", "前面", "之前", "回答"]):
            return []
        rows: list[dict[str, str]] = []
        for item in history[-30:]:
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            if self_match_any(content, ["乐乐", "姓", "林", "名字", "英文名", "lily", "说过", "记得"]) or any(token in message for token in ["刚才", "刚刚", "上一句", "前面", "之前", "回答"]):
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
        if ChatEngine.is_user_identity_question(message):
            return False
        return any(token in message for token in ["你是谁", "你叫什么", "你叫啥", "你叫什么名字", "你名字", "你姓什么", "你姓啥", "你的英文名", "你英文名"])

    @staticmethod
    def is_bot_surname_question(message: str) -> bool:
        return any(token in message for token in ["你姓什么", "你姓啥", "你的姓", "你什么姓"])

    @staticmethod
    def is_user_identity_question(message: str) -> bool:
        return any(
            token in message
            for token in [
                "我是谁",
                "我是你的谁",
                "我是你的什么人",
                "我是什么人",
                "我是你什么人",
                "我姓什么",
                "我姓啥",
                "我什么姓",
                "我叫什么",
                "我叫啥",
                "我的名字",
                "我名字",
                "我的英文名",
                "我英文名",
            ]
        )

    @staticmethod
    def is_time_question(message: str) -> bool:
        return any(token in message for token in ["聊了多久", "认识多久", "认识多长", "多久了", "什么时候认识", "哪天认识"])

    @staticmethod
    def is_birthday_question(message: str) -> bool:
        return any(token in message for token in ["生日", "生快", "几号出生", "哪天出生", "出生日期"])

    @staticmethod
    def is_relationship_question(message: str) -> bool:
        return any(
            token in message
            for token in [
                "感情经历",
                "情感经历",
                "我们关系",
                "什么关系",
                "关系怎么样",
                "喜欢过",
                "爱过",
                "吵过",
                "吵架",
                "和好",
                "冷暴力",
                "分手",
                "情感",
            ]
        )

    @staticmethod
    def is_emotion_history_question(message: str) -> bool:
        return any(
            token in message
            for token in [
                "情绪起伏",
                "情绪",
                "心情",
                "低落",
                "难受",
                "生气",
                "哭过",
                "emo",
                "破防",
                "委屈",
                "崩溃",
            ]
        )

    @staticmethod
    def is_preference_question(message: str) -> bool:
        return any(token in message for token in ["喜欢什么", "讨厌什么", "不喜欢什么", "爱吃什么", "想吃什么", "偏好", "喜欢吃", "喜欢玩"])

    @staticmethod
    def is_habit_question(message: str) -> bool:
        return any(token in message for token in ["平时", "习惯", "作息", "几点睡", "吃饭", "上课", "考试", "家里", "身体", "不舒服"])

    @staticmethod
    def is_nickname_question(message: str) -> bool:
        return any(token in message for token in ["叫我什么", "叫你什么", "喊我什么", "喊你什么", "昵称", "外号", "怎么称呼"])

    @staticmethod
    def is_topic_question(message: str) -> bool:
        return any(token in message for token in ["聊什么", "常聊", "平时聊", "话题", "聊得最多"])

    @staticmethod
    def is_soothing_question(message: str) -> bool:
        return any(token in message for token in ["怎么哄", "怎么安慰", "哄我", "安慰我", "我难受怎么办", "我生气怎么办"])

    @staticmethod
    def is_wake_time_question(message: str) -> bool:
        return any(
            token in message
            for token in [
                "几点起床",
                "几点起",
                "几点醒",
                "什么时候起床",
                "什么时候醒",
                "一般几点起",
                "一般几点醒",
                "起床时间",
                "睡到几点",
            ]
        )

    @staticmethod
    def is_meaning_question(message: str) -> bool:
        return any(token in message for token in ["什么意思", "啥意思", "什么含义", "何意"])

    @staticmethod
    def is_slang_meaning_question(message: str) -> bool:
        return any(token in message for token in ["乐乐", "乐子", "梗", "网络梗"])

    @staticmethod
    def meaning_reply(message: str) -> str:
        if "乐乐" in message:
            return pick(["乐子那个乐", "找乐子的乐吧", "不是单纯名字"], message)
        return pick(["字面意思吧", "你说哪个词", "这个要看你前面怎么用的"], message)

    @staticmethod
    def slang_meaning_reply(message: str) -> str:
        if "乐乐" in message:
            return pick(["乐子那个乐", "找乐子的乐吧", "不是单纯名字"], message)
        if "乐子" in message:
            return pick(["看乐子的意思", "拿来好笑的", "乐子人那个乐子"], message)
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
