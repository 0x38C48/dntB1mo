from __future__ import annotations

import hashlib
import json
import re
import time
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


RUNTIME_VERSION = "backup-user-style-v29-proactive-topic-tails"

DEFAULT_MAX_REPLY_CHARS = 28
FACT_MAX_REPLY_CHARS = 18
FACT_STYLE_MAX_REPLY_CHARS = 32
BUBBLE_MAX_CHARS = 12

CONSECUTIVE_STYLE_PROFILE = {
    "observed_multi_run_ratio": 0.487,
    "micro_opening_multi_ratio": 0.578,
    "open_chat_multi_ratio": 0.446,
    "avg_run_len": 1.88,
}

EMOTION_PROMPT_PROFILES = {
    "casual": {
        "record_summary": "记录里最多见，短句为主，中位长度约4字；常见是接一句、轻轻吐槽或抛回去。",
        "style": "不热情解释，不端着；可以短促、随手、像刚看到消息。",
        "avoid": "不要写成长句安慰或报告。",
        "bubbles": "1-2条，第二条可以是干嘛/然后呢/你说。",
    },
    "playful": {
        "record_summary": "主要是6、666、哈、笑死、绷不住这类短促接梗，平均也很短。",
        "style": "先接梗，再轻微反问或补刀；允许单符号和短连发。",
        "avoid": "不要解释笑点，不要把梗讲成百科。",
        "bubbles": "常见2条：笑点反应 + 一句追问/吐槽。",
    },
    "annoyed": {
        "record_summary": "不爽时不是长篇指责，更像短促否定、嫌弃、让话题停一下。",
        "style": "少字、冷一点、可用别/服了/算了/你又来，但别持续攻击。",
        "avoid": "不要说教，不要大段分析情绪。",
        "bubbles": "1-2条，第二条偏收束或转移。",
    },
    "soft": {
        "record_summary": "软一点时也不鸡汤，更多是短安抚、接住、让对方别硬撑。",
        "style": "轻一点、贴近一点，允许好啦/没事/我听着/先这样。",
        "avoid": "不要心理咨询腔，不要过度温柔。",
        "bubbles": "1-2条，必要时补一句照顾性短句。",
    },
    "sleepy": {
        "record_summary": "困/睡相关通常更短、更懒，倾向结束话题或催睡。",
        "style": "懒散、短、低能量；可以说困/去睡/别硬撑。",
        "avoid": "不要突然兴奋，不要主动开复杂话题。",
        "bubbles": "1条居多，最多补一句。",
    },
    "excited": {
        "record_summary": "兴奋不是长篇夸张，常见是wc/哇/牛/真的假的这类突然抬高。",
        "style": "先短促惊讶，再追问一点。",
        "avoid": "不要写满感叹号，不要过度表演。",
        "bubbles": "2条较自然：惊讶 + 追问。",
    },
    "curious": {
        "record_summary": "疑问多用什么/啥/怎么说，通常不是完整提问句。",
        "style": "短反问、让对方展开；对事实问题仍先查证但口语化。",
        "avoid": "不要每次只发一个问号。",
        "bubbles": "1-2条，第二条可追问具体点。",
    },
    "engaged": {
        "record_summary": "对方连续发很多时，会整体接住，但仍然保持短泡泡。",
        "style": "先表示听懂，再挑一个点追问或回应。",
        "avoid": "不要逐条答题。",
        "bubbles": "2-3条，像连续看完后补话。",
    },
}


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
        active_topic_state = self.latest_active_topic_state(conversation_memory)
        candidate_topic_state = self.topic_session_from_text(message, source="chat")
        active_topic_closing = active_topic_state is not None and self.should_close_active_topic(message, active_topic_state)
        topic_scope_state = (
            active_topic_state
            if active_topic_state
            and not self.should_break_active_topic(message)
            and not active_topic_closing
            else None
        )
        if (
            active_topic_closing
            and not self.should_break_active_topic(message)
            and candidate_topic_state.get("category") != "open"
            and not self.is_explicit_topic_end(message)
        ):
            topic_scope_state = candidate_topic_state
        retrieval_query = self.build_retrieval_query(message, history, topic_scope_state)
        memories = self.retriever.search(retrieval_query, limit=10)
        memories = self.rerank_memories_for_question(message, memories, topic_scope_state)
        temporary_facts = self.extract_temporary_facts(message, memories, fact_domain)
        emotion = self.resolve_emotion(message, history, mood)
        dialogue_act = self.classify_dialogue_act(message)
        web_results = search_web(message, limit=4) if self.should_web_lookup(message) else []
        active_topic_blocked = self.should_break_active_topic(message) or active_topic_closing
        prompt_conversation_memory = (
            self.without_active_topic(conversation_memory) if active_topic_blocked else conversation_memory
        )

        repair_reply = self.conversation_repair_reply(message, history)
        if repair_reply:
            return {
                "reply": repair_reply,
                "mode": f"{self.mode}_repair_route",
                "emotion": emotion,
                "facts": self.public_facts(),
                "memories": memories,
                "web_results": web_results,
            }

        topic_reply = self.active_topic_reply(message, history, conversation_memory)
        if topic_reply:
            return {
                "reply": topic_reply,
                "mode": f"{self.mode}_topic_continuation_route",
                "emotion": emotion,
                "facts": self.public_facts(),
                "memories": memories,
                "web_results": web_results,
            }

        fact_reply = self.fact_first_reply(message, history, memories) if self.should_answer_fact_first(message) else None
        if fact_reply:
            if self.is_recent_chat_memory_question(message):
                fact_text = fact_reply
            else:
                fact_text = self.humanize_fact_reply(message, fact_reply, memories, history)
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

        if self.is_first_person_preference_statement(message):
            return {
                "reply": self.preference_statement_reply(message),
                "mode": f"{self.mode}_chat_act_route",
                "emotion": emotion,
                "facts": self.public_facts(),
                "memories": memories,
                "web_results": web_results,
            }

        food_chat_reply = self.food_chat_reply(message)
        if food_chat_reply:
            return {
                "reply": food_chat_reply,
                "mode": f"{self.mode}_chat_act_route",
                "emotion": emotion,
                "facts": self.public_facts(),
                "memories": memories,
                "web_results": web_results,
            }

        close_topic_reply = (
            self.close_topic_reply(message, active_topic_state)
            if active_topic_closing
            and (candidate_topic_state.get("category") == "open" or self.is_explicit_topic_end(message))
            and not self.should_break_active_topic(message)
            else None
        )
        if close_topic_reply:
            return {
                "reply": close_topic_reply,
                "mode": f"{self.mode}_topic_close_route",
                "emotion": emotion,
                "facts": self.public_facts(),
                "memories": memories,
                "web_results": web_results,
            }

        new_topic_reply = (
            self.new_topic_seed_reply(message, topic_scope_state)
            if topic_scope_state
            and candidate_topic_state.get("category") != "open"
            and str(topic_scope_state.get("topic") or "") == str(candidate_topic_state.get("topic") or "")
            else None
        )
        if new_topic_reply:
            return {
                "reply": new_topic_reply,
                "mode": f"{self.mode}_topic_seed_route",
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

        prompt = self.build_prompt(
            message,
            history[-48:],
            memories,
            prompt_conversation_memory,
            emotion,
            fact_domain,
            temporary_facts,
            web_results,
            dialogue_act,
            topic_scope_state,
        )
        if self.config.sophnet_api_key:
            try:
                text = self.call_sophnet_api(prompt)
                return {
                    "reply": self.polish_model_reply(text, message, memories, history, prompt_conversation_memory, emotion),
                    "mode": self.mode,
                    "emotion": emotion,
                    "facts": self.public_facts(),
                    "memories": memories,
                    "web_results": web_results,
                }
            except Exception as exc:
                return {
                    "reply": compact_reply(self.local_reply(message, memories, prompt_conversation_memory, emotion), DEFAULT_MAX_REPLY_CHARS),
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
                    "reply": self.polish_model_reply(text, message, memories, history, prompt_conversation_memory, emotion),
                    "mode": self.mode,
                    "emotion": emotion,
                    "facts": self.public_facts(),
                    "memories": memories,
                    "web_results": web_results,
                }
            except Exception as exc:
                return {
                    "reply": compact_reply(self.local_reply(message, memories, prompt_conversation_memory, emotion), DEFAULT_MAX_REPLY_CHARS),
                    "mode": "local_fallback_after_api_error",
                    "api_error": str(exc),
                    "emotion": emotion,
                    "facts": self.public_facts(),
                    "memories": memories,
                    "web_results": web_results,
                }
        return {
            "reply": compact_reply(self.local_reply(message, memories, prompt_conversation_memory, emotion), DEFAULT_MAX_REPLY_CHARS),
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
        dialogue_act: str = "open_chat",
        active_topic_scope: dict[str, Any] | None = None,
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
            "emotion_prompt_profile": self.emotion_prompt_profile(emotion),
            "active_topic_scope": active_topic_scope,
            "fact_domain": fact_domain,
            "dialogue_act": dialogue_act,
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
                "默认像微信短消息：通常1-2个短泡泡，接话/吐槽/情绪上来时可以2-3个，单泡泡尽量不超过12个中文字符。",
                "除非用户明确要求详细，不要解释来龙去脉，不要写完整报告句。",
                "能说“又来了”就不要说“你是不是又在……”。",
                "避免固定长模板：“你是不是…是吧”“你今天跟…过不去了吧”这类句式少用。",
                "事实只点到即可，比如“林薇艺\\nlily”或“2024-10-13”。",
                "优先保留个人语气：吐槽、反问、停顿、轻微敷衍可以有，别太端着。",
                "emotion_prompt_profile 是当前情绪下的说话方式约束；它影响语气、气泡数、是否追问，但不能覆盖事实证据。",
                "如果 active_topic_scope 不为空，说明当前日常话题尚未结束；除非用户明显换题/结束/问事实插问，否则思考方向和检索记忆都围绕该话题。",
                "style_samples_from_backup 的权重高于抽象总结；学节奏和用词，不要逐字复读。",
                "只输出回复正文，不解释检索过程。",
                "可以很短，也可以分成连续几句，但要针对当前这句，不要套模板。",
                "身份/名字/时间/是否说过的问题必须优先使用 identity_and_timeline_facts 和 retrieved_memories。",
                "生日、情感经历、情绪起伏等事实问题必须先使用 temporary_facts_from_retrieval；证据不足时用真人记忆口吻说“我不记得/我想想/好像/只记得一点”，不要说“记录/检索/看到/证据/稳不稳”。",
                "如果 temporary_facts_from_retrieval 已经足够回答，只做语气压缩，不要改事实。",
                "有明确证据时不要说“不知道/没印象”。",
                "top_short_phrases 只是风格参考，不是复读清单。",
                "避免重复 recent_assistant_replies 中的句式；意思接近也要换角度。",
                "遇到多行 user_message，代表 NonForgetter 连续发了多条消息，要整体理解后回复。",
                "如果 dialogue_act 是 casual_statement 或 preference_statement，先接住对方这句话，不要像问卷一样回答事实标签。",
                "如果 dialogue_act 是 repair_request，用“不是/算了/哎呀/你别管/那咋了”这类含糊转移，不要说“我换个说法/我重说/我接歪了/我生成错了”。",
                "遇到“刚刚/刚才/上一句/之前我说了什么/你回答了什么”，必须优先看 recent_dialogue_state，用“你刚说/我刚回”回答，不要说“看到记录”。",
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
                    "content": "You simulate backup's concise Chinese WeChat style. Reply in micro chat bubbles, usually 1-2 but often 2-3 when the turn feels like quick consecutive messages. Keep each bubble short. Evidence outranks improvisation, but never sound like a report.",
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
                    "content": "You simulate backup's concise Chinese WeChat style. Reply in micro chat bubbles, usually 1-2 but often 2-3 when the turn feels like quick consecutive messages. Keep each bubble short. Evidence outranks improvisation, but never sound like a report.",
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

    def apply_consecutive_style(
        self,
        reply: str,
        message: str,
        history: list[dict[str, Any]] | None = None,
        mode: str = "",
        memories: list[dict[str, Any]] | None = None,
        emotion: str = "casual",
    ) -> str:
        text = fix_text(reply).strip()
        if not text or self.allows_long_reply(message):
            return text
        if "topic_close" in mode or "topic_seed" in mode:
            return text

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return text
        if len(lines) >= 3:
            return "\n".join(lines[:3])

        probability = self.consecutive_probability(message, mode, lines, history or [], emotion)
        seed = "|".join([message, text, mode, emotion, str(len(history or []))])
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        roll = int(digest[:8], 16) / 0xFFFFFFFF
        if roll >= probability:
            return "\n".join(lines)

        extra = self.consecutive_tail(message, "\n".join(lines), mode, memories or [], emotion)
        if not extra:
            return "\n".join(lines)
        extra_lines = [line.strip() for line in extra.splitlines() if line.strip()]
        for line in extra_lines:
            if line and line not in lines and len(lines) < 3 and not self.is_near_duplicate_line(line, lines):
                lines.append(line)
        return "\n".join(lines)

    def consecutive_probability(
        self,
        message: str,
        mode: str,
        lines: list[str],
        history: list[dict[str, Any]],
        emotion: str,
    ) -> float:
        text = message.strip()
        if "web" in mode:
            base = 0.22
        elif any(token in mode for token in ["fact", "evidence", "memory"]):
            base = 0.34
        elif "local_proactive" in mode:
            topic_state = self.topic_session_from_text("\n".join(lines), source="proactive")
            if topic_state.get("category") == "sleep":
                base = 0.22
            else:
                base = 0.44
        elif any(token in mode for token in ["topic_continuation", "chat_act", "repair"]):
            base = 0.56
        else:
            base = CONSECUTIVE_STYLE_PROFILE["open_chat_multi_ratio"]

        proactive_sleep = "local_proactive" in mode and self.topic_session_from_text("\n".join(lines), source="proactive").get("category") == "sleep"
        if (len(text) <= 3 or self.is_minimal_topic_ack(text)) and not proactive_sleep:
            base = max(base, CONSECUTIVE_STYLE_PROFILE["micro_opening_multi_ratio"])
        if "\n" in message:
            base += 0.06
        if emotion in {"playful", "annoyed", "engaged", "curious"}:
            base += 0.06
        if len(lines) == 2:
            base *= 0.35
        recent_user_count = sum(1 for item in history[-8:] if item.get("role") == "user")
        if recent_user_count >= 4:
            base += 0.05
        return max(0.08, min(0.68, base))

    def consecutive_tail(
        self,
        message: str,
        reply: str,
        mode: str,
        memories: list[dict[str, Any]],
        emotion: str,
    ) -> str:
        seed = message + "|" + reply + "|" + mode + "|" + emotion
        if "local_proactive" in mode:
            return self.proactive_consecutive_tail(reply, seed)
        if any(token in mode for token in ["fact", "evidence", "memory"]):
            if self.is_bot_identity_question(message) or self.is_user_identity_question(message):
                return pick(["又来", "别装", "你猜"], seed)
            if self.is_time_question(message):
                return pick(["你还问", "好久了", "差不多"], seed)
            if self.is_wake_time_question(message):
                return pick(["差不多", "我记得", "大概吧"], seed)
            if self.is_birthday_question(message):
                return pick(["别诈我", "应该吧", "我记得"], seed)
            return pick(["差不多", "我印象是", "别问了"], seed)

        if self.is_question_like(message):
            return pick(["你说呢", "怎么了", "又问"], seed)
        if emotion == "annoyed":
            return pick(["服了", "别太离谱", "你又来"], seed)
        if emotion == "soft":
            return pick(["先这样", "慢慢来", "我听着"], seed)
        if emotion == "playful":
            return pick(["笑死", "又乐", "6"], seed)
        if len(message.strip()) <= 3:
            return pick(["然后呢", "说啊", "干嘛"], seed)

        return pick(["然后呢", "你继续", "说啊", "嗯哼", "干嘛"], seed)

    def proactive_consecutive_tail(self, reply: str, seed: str) -> str:
        topic_state = self.topic_session_from_text(reply, source="proactive")
        category = str(topic_state.get("category") or "open")
        first_line = next((line.strip() for line in fix_text(reply).splitlines() if line.strip()), "")
        if category == "sleep":
            if any(token in first_line for token in ["醒", "醒着"]):
                return pick(["还醒着吗", "别装睡", "在不在"], seed)
            return pick(["你又不睡", "别熬了", "去睡啊"], seed)
        if category == "food":
            return pick(["别又不吃", "吃了没", "快去吃"], seed)
        if category == "game":
            return pick(["还在打吗", "打完没", "别打太晚"], seed)
        if category == "presence":
            return pick(["在不在", "干嘛去了", "说话"], seed)
        if category == "music":
            return pick(["听啥", "又循环了", "好听吗"], seed)
        if category == "school":
            return pick(["几点去", "别迟到", "去不去"], seed)
        if category == "media":
            return pick(["看啥", "好看吗", "又看什么"], seed)
        if category == "care":
            return pick(["怎么了", "别硬撑", "还难受吗"], seed)
        return pick(["干嘛", "在不在", "你说"], seed)

    @staticmethod
    def is_near_duplicate_line(line: str, existing: list[str]) -> bool:
        normalized = ChatEngine.normalize_for_repeat(line)
        for old in existing:
            old_normalized = ChatEngine.normalize_for_repeat(old)
            if not normalized or not old_normalized:
                continue
            if normalized == old_normalized:
                return True
            if SequenceMatcher(None, normalized, old_normalized).ratio() >= 0.72:
                return True
        return False

    def topic_memory_update(
        self,
        message: str,
        result: dict[str, Any],
        conversation_memory: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        mode = str(result.get("mode") or "")
        current = self.latest_active_topic_state(conversation_memory)
        if current and self.should_close_active_topic(message, current):
            new_state = self.topic_session_from_text(message, source="chat")
            if new_state.get("category") != "open" and not self.is_explicit_topic_end(message):
                return {"action": "remember", **new_state}
            return {"action": "clear"}

        if any(token in mode for token in ["fact", "evidence", "memory", "web", "repair"]):
            return None

        if "topic_continuation" in mode and current:
            updated = dict(current)
            updated["action"] = "remember"
            updated["source"] = "continuation"
            updated["turns"] = int(updated.get("turns") or 0) + 1
            return updated

        candidate = self.topic_session_from_text(message, source="chat")
        if candidate.get("category") != "open":
            candidate["action"] = "remember"
            return candidate
        return None

    def conversation_repair_reply(self, message: str, history: list[dict[str, Any]]) -> str | None:
        if not self.is_conversation_repair_question(message):
            return None
        last_assistant = next(
            (
                str(item.get("content", "")).strip()
                for item in reversed(history)
                if item.get("role") == "assistant" and str(item.get("content", "")).strip()
            ),
            "",
        )
        if last_assistant:
            return pick(["不是", "哎呀\n算了", "你别管", "好吧\n当我没说", "那咋了", "没事"], message + last_assistant)
        return pick(["什么", "啊？", "不是"], message)

    def active_topic_reply(
        self,
        message: str,
        history: list[dict[str, Any]],
        conversation_memory: list[dict[str, Any]],
    ) -> str | None:
        topic_state = self.latest_active_topic_state(conversation_memory)
        topic = str(topic_state.get("topic") or "") if topic_state else ""
        if not topic or self.should_break_active_topic(message) or self.should_close_active_topic(message, topic_state or {}):
            return None
        if not self.last_assistant_matches_topic(history, topic):
            return None

        stripped = message.strip()
        if not stripped:
            return None
        if self.is_topic_prompt_request(stripped):
            return self.expand_active_topic(topic, stripped)
        if self.is_minimal_topic_ack(stripped) or self.is_topic_continuation_like(stripped):
            return self.continue_active_topic(topic, stripped)
        return None

    @staticmethod
    def without_active_topic(conversation_memory: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [item for item in conversation_memory if item.get("kind") != "active_topic"]

    @classmethod
    def latest_active_topic_state(cls, conversation_memory: list[dict[str, Any]]) -> dict[str, Any] | None:
        now = time.time()
        for item in conversation_memory:
            if item.get("kind") != "active_topic":
                continue
            try:
                created_at = float(item.get("created_at") or 0)
            except (TypeError, ValueError):
                created_at = 0
            if created_at and now - created_at > 20 * 60:
                continue

            content = str(item.get("content", "")).strip()
            if not content:
                continue
            try:
                payload = json.loads(content)
                if not isinstance(payload, dict):
                    payload = {"topic": content}
            except (json.JSONDecodeError, TypeError):
                payload = {"topic": content}

            topic = str(payload.get("topic") or "").strip()
            if not topic:
                continue
            if payload.get("status") == "closed":
                continue

            category = str(payload.get("category") or "").strip()
            keywords = [str(word).strip() for word in payload.get("keywords") or [] if str(word).strip()]
            inferred = cls.topic_session_from_text(topic, source=str(payload.get("source") or "memory"))
            if not category or category == "open":
                category = inferred["category"]
            if not keywords:
                keywords = inferred["keywords"]
            return {
                "topic": topic[:80],
                "category": category or "open",
                "keywords": keywords[:12],
                "source": str(payload.get("source") or "memory"),
                "turns": int(payload.get("turns") or 0),
                "created_at": created_at,
            }
        return None

    @staticmethod
    def latest_active_topic(conversation_memory: list[dict[str, Any]]) -> str | None:
        state = ChatEngine.latest_active_topic_state(conversation_memory)
        return str(state.get("topic") or "") if state else None

    @staticmethod
    def topic_session_from_text(text: str, source: str = "chat") -> dict[str, Any]:
        raw = fix_text(text or "").strip()
        first_line = next((line.strip() for line in raw.splitlines() if line.strip()), raw)
        probes = {
            "food": ["吃饭", "吃", "饭", "饿", "外卖", "食堂", "喝", "咖啡"],
            "game": ["游戏", "打完", "原神", "星铁", "崩铁", "绝区零", "zzz", "鸣潮", "steam", "瓦", "玩"],
            "presence": ["在吗", "在不在", "干嘛", "不说话", "没声", "醒着"],
            "sleep": ["困", "睡", "醒", "晚安", "安安", "起床", "睡觉"],
            "music": ["听歌", "歌", "循环"],
            "school": ["学校", "上课", "考试", "作业", "迟到"],
            "media": ["动画", "番", "看什么", "视频", "电影"],
            "care": ["难受", "烦", "哭", "抱抱", "生气", "别难受"],
        }
        for category, words in probes.items():
            hits = [word for word in words if word.lower() in first_line.lower()]
            if hits:
                return {"topic": first_line[:80], "category": category, "keywords": list(dict.fromkeys(hits + words[:5])), "source": source}
        return {"topic": first_line[:80], "category": "open", "keywords": [], "source": source}

    @staticmethod
    def last_assistant_matches_topic(history: list[dict[str, Any]], topic: str) -> bool:
        first_topic = next((line.strip() for line in topic.splitlines() if line.strip()), topic.strip())
        assistants = [
            str(item.get("content", "")).strip()
            for item in history[-8:]
            if item.get("role") == "assistant" and str(item.get("content", "")).strip()
        ]
        if not assistants:
            return False
        recent = assistants[-4:]
        return any(
            first_topic in item or item in first_topic or SequenceMatcher(None, item, first_topic).ratio() >= 0.55
            for item in recent
        )

    def should_break_active_topic(self, message: str) -> bool:
        if self.should_web_lookup(message) or self.is_conversation_repair_question(message):
            return True
        checkers = [
            self.is_recent_chat_memory_question,
            self.is_identity_correction,
            self.is_user_identity_question,
            self.is_bot_identity_question,
            self.is_birthday_question,
            self.is_relationship_question,
            self.is_soothing_question,
            self.is_topic_question,
            self.is_emotion_history_question,
            self.is_preference_question,
            self.is_nickname_question,
            self.is_habit_question,
            self.is_wake_time_question,
            self.is_time_question,
            self.is_memory_dispute_question,
        ]
        return any(checker(message) for checker in checkers)

    def should_close_active_topic(self, message: str, topic_state: dict[str, Any]) -> bool:
        stripped = fix_text(message).strip()
        if not stripped:
            return False
        if self.is_explicit_topic_end(stripped):
            return True
        if self.is_topic_prompt_request(stripped) or self.is_minimal_topic_ack(stripped):
            return False
        current_category = str(topic_state.get("category") or "open")
        new_state = self.topic_session_from_text(stripped)
        new_category = str(new_state.get("category") or "open")
        if new_category != "open" and current_category != "open" and new_category != current_category:
            return True
        if len(stripped) > 40 and new_category == "open" and not self.topic_message_matches(stripped, topic_state):
            return True
        return False

    @staticmethod
    def is_explicit_topic_end(message: str) -> bool:
        return any(token in fix_text(message).strip() for token in ["算了", "不聊", "别聊", "换个", "换话题", "先这样", "没事了", "睡了", "晚安", "安安", "拜拜", "88"])

    @staticmethod
    def topic_message_matches(message: str, topic_state: dict[str, Any]) -> bool:
        haystack = fix_text(message).lower()
        keywords = [str(word).lower() for word in topic_state.get("keywords") or [] if str(word).strip()]
        if any(word and word in haystack for word in keywords):
            return True
        category = str(topic_state.get("category") or "")
        category_cues = {
            "food": ["吃", "饭", "饿", "喝", "外卖"],
            "game": ["玩", "打", "游戏", "原神", "星铁", "绝区零"],
            "presence": ["在", "干嘛", "说话", "没声"],
            "sleep": ["困", "睡", "醒", "晚安"],
            "music": ["歌", "听"],
            "school": ["学校", "上课", "考试"],
            "media": ["看", "动画", "番", "视频"],
            "care": ["难受", "烦", "哭", "抱"],
        }
        return any(cue in haystack for cue in category_cues.get(category, []))

    @staticmethod
    def is_minimal_topic_ack(message: str) -> bool:
        stripped = message.strip()
        return stripped in {
            "在",
            "嗯",
            "嗯嗯",
            "好",
            "啊",
            "哦",
            "噢",
            "行",
            "可以",
            "还在",
            "没",
            "没有",
            "还没",
            "不知道",
            "?",
            "？",
        }

    @staticmethod
    def is_topic_continuation_like(message: str) -> bool:
        stripped = message.strip()
        if len(stripped) > 40:
            return False
        if re.search(r"(什么|啥|怎么|哪|几点|多久|生日|名字|是谁|为什么|吗)$", stripped):
            return any(token in stripped for token in ["吃什么", "吃啥", "玩什么", "玩啥", "干嘛", "做什么"])
        return True

    @staticmethod
    def is_topic_prompt_request(message: str) -> bool:
        stripped = message.strip()
        return stripped in {"怎么了", "咋了", "干嘛", "什么事", "啥事", "怎么", "？", "?"} or any(
            token in stripped for token in ["怎么了", "咋了", "什么事", "啥事"]
        )

    def expand_active_topic(self, topic: str, message: str) -> str:
        first_line = next((line.strip() for line in topic.splitlines() if line.strip()), topic.strip())
        seed = first_line + "|" + message
        if any(token in first_line for token in ["吃饭", "吃了吗", "饿"]):
            return pick(["问你吃没\n别又不吃", "你吃饭没啊\n这还问", "怕你又饿着"], seed)
        if any(token in first_line for token in ["游戏", "打完", "原神", "星铁", "玩"]):
            return pick(["问你还打不打\n别装没看见", "你游戏打完没\n我就问问", "想问你还在不在打"], seed)
        if any(token in first_line for token in ["在吗", "干嘛", "不说话"]):
            return pick(["问你呢\n你在干嘛", "你突然没声了\n干嘛去了", "没什么\n就看你在不在"], seed)
        if any(token in first_line for token in ["困", "睡", "醒"]):
            return pick(["看你是不是又困了", "你是不是又要睡", "问你醒着没"], seed)
        if any(token in first_line for token in ["听歌", "歌"]):
            return pick(["想问你听啥", "你又在循环什么", "问你听歌没"], seed)
        if any(token in first_line for token in ["学校", "去学校"]):
            return pick(["问你去不去学校", "今天要去吗", "别又迟到"], seed)
        if "突然想起来" in first_line:
            return pick(["突然想起你之前说的\n有点好笑", "想起个事\n但是忘一半了", "没什么\n突然想到你"], seed)
        return pick(["没什么\n就问问", "想起你了\n不行啊", "问你在干嘛"], seed)

    def continue_active_topic(self, topic: str, message: str) -> str:
        first_line = next((line.strip() for line in topic.splitlines() if line.strip()), topic.strip())
        seed = first_line + "|" + message
        if any(token in first_line for token in ["吃饭", "吃了吗", "饿"]):
            if any(token in message for token in ["什么", "啥"]):
                return pick(["你想吃啥", "随便吃点", "别问我啊"], seed)
            if any(token in message for token in ["没", "还没", "没有"]):
                return pick(["那去吃啊", "别饿着", "快点去"], seed)
            return pick(["可以啊", "那吃呗", "也行"], seed)

        if any(token in first_line for token in ["游戏", "打完", "原神", "星铁", "玩"]):
            if any(token in message for token in ["在", "还在"]):
                return pick(["还在啊", "打完没", "玩啥呢"], seed)
            if any(token in message for token in ["什么", "啥"]):
                return pick(["你不是会玩", "又问我", "你想玩啥"], seed)
            return pick(["那你继续", "别打太晚", "行吧"], seed)

        if any(token in first_line for token in ["在吗", "干嘛", "不说话"]):
            if "在" in message:
                return pick(["那你干嘛呢", "在就说话", "嗯哼"], seed)
            return pick(["然后呢", "你继续", "说啊"], seed)

        if any(token in first_line for token in ["困", "睡"]):
            if any(token in message for token in ["困", "睡", "累"]):
                return pick(["那睡会", "别硬撑", "去睡啊"], seed)
            return pick(["还不困啊", "那你干嘛", "行"], seed)

        if any(token in first_line for token in ["听歌", "歌"]):
            if any(token in message for token in ["什么", "啥"]):
                return pick(["你听的啥", "随便听点", "又让我选"], seed)
            return pick(["好听吗", "你又循环了", "听吧"], seed)

        if any(token in first_line for token in ["学校", "去学校"]):
            return pick(["去不去啊", "几点去", "别迟到"], seed)

        return pick(["然后呢", "你继续", "说啊", "嗯哼"], seed)

    def new_topic_seed_reply(self, message: str, topic_state: dict[str, Any] | None) -> str | None:
        if not topic_state:
            return None
        category = str(topic_state.get("category") or "open")
        seed = message + "|" + category
        if category == "food":
            if any(token in message for token in ["吃什么", "吃啥"]):
                return pick(["你想吃啥", "随便吃点", "别问我啊"], seed)
            return pick(["你吃没", "又饿了啊", "吃饭啊"], seed)
        if category == "game":
            if any(token in message for token in ["打完", "还在"]):
                return pick(["还没啊", "没打完", "你又催"], seed)
            return pick(["玩啥呢", "你又开了", "什么游戏"], seed)
        if category == "sleep":
            return pick(["又困了啊", "那去睡", "别硬撑"], seed)
        if category == "presence":
            return pick(["在", "干嘛", "你说"], seed)
        if category == "music":
            return pick(["听啥", "又循环了", "好听吗"], seed)
        if category == "school":
            return pick(["要去吗", "几点去", "别迟到"], seed)
        if category == "media":
            return pick(["看啥", "又看什么", "好看吗"], seed)
        if category == "care":
            return pick(["怎么了", "别硬撑", "我听着"], seed)
        return None

    @staticmethod
    def close_topic_reply(message: str, topic_state: dict[str, Any] | None) -> str:
        category = str((topic_state or {}).get("category") or "open")
        seed = message + "|" + category
        if any(token in message for token in ["睡了", "晚安", "安安"]):
            return pick(["安安", "晚安安", "那睡吧\n安安", "别熬了\n安安"], seed)
        if any(token in message for token in ["换个", "换话题"]):
            return pick(["行\n换啥", "那换", "你说"], seed)
        if any(token in message for token in ["算了", "不聊", "别聊", "先这样"]):
            return pick(["行吧", "那先这样", "好"], seed)
        return pick(["行", "好吧", "那咋了"], seed)

    @staticmethod
    def preference_statement_reply(message: str) -> str:
        if any(token in message for token in ["喜欢玩", "爱玩", "挺喜欢玩"]):
            return pick(["那你玩啥", "玩啥啊", "那还挺好"], message)
        if any(token in message for token in ["喜欢吃", "爱吃", "想吃"]):
            return pick(["吃啥", "那吃呗", "你又饿了"], message)
        return pick(["嗯哼", "然后呢", "你继续"], message)

    @staticmethod
    def food_chat_reply(message: str) -> str | None:
        stripped = message.strip()
        if stripped in {"吃什么", "吃啥", "吃点什么", "吃啥啊"}:
            return pick(["你想吃啥", "随便吃点", "别问我啊"], stripped)
        return None

    def humanize_fact_reply(
        self,
        message: str,
        fact_reply: str,
        memories: list[dict[str, Any]],
        history: list[dict[str, Any]],
    ) -> str:
        answer = fix_text(fact_reply).strip()
        if not answer:
            return answer
        if self.allows_long_reply(message):
            return answer
        if self.is_recent_chat_memory_question(message):
            return answer

        compact_answer = self.fact_answer_atom(answer)
        if not compact_answer:
            return answer
        if any(token in compact_answer for token in ["不记得", "想不起来", "不太确定"]):
            return pick(
                [
                    f"{compact_answer}\n别逼我",
                    f"{compact_answer}\n我想想",
                    f"{compact_answer}\n真忘了",
                ],
                message + compact_answer,
            )

        if self.is_bot_identity_question(message):
            return pick(
                [
                    f"又问这个\n{compact_answer}",
                    f"{compact_answer}\n你猜的那个",
                    f"不是说过\n{compact_answer}",
                ],
                message + compact_answer,
            )
        if self.is_user_identity_question(message):
            return pick(
                [
                    f"你啊\n{compact_answer}",
                    f"{compact_answer}\n还问",
                    f"我记得是\n{compact_answer}",
                ],
                message + compact_answer,
            )
        if self.is_birthday_question(message):
            return pick(
                [
                    f"好像{compact_answer}",
                    f"{compact_answer}\n别又诈我",
                    f"我记得是\n{compact_answer}",
                ],
                message + compact_answer,
            )
        if self.is_time_question(message):
            compact_answer = self.casual_time_atom(message, compact_answer)
            return pick(
                [
                    f"挺久了\n{compact_answer}",
                    f"那会吧\n{compact_answer}",
                    f"你还问\n{compact_answer}",
                ],
                message + compact_answer,
            )
        if self.is_wake_time_question(message):
            return pick(
                [
                    f"大概{compact_answer}",
                    f"{compact_answer}\n差不多",
                    f"我记得差不多\n{compact_answer}",
                ],
                message + compact_answer,
            )
        if self.is_preference_question(message) or self.is_topic_question(message):
            return pick(
                [
                    f"像是{compact_answer}",
                    f"{compact_answer}\n这类吧",
                    f"我印象里\n{compact_answer}",
                ],
                message + compact_answer,
            )
        if self.is_relationship_question(message) or self.is_emotion_history_question(message):
            return pick(
                [
                    f"{compact_answer}\n反正挺明显",
                    f"大概就{compact_answer}",
                    f"我印象是\n{compact_answer}",
                ],
                message + compact_answer,
            )
        if self.is_memory_dispute_question(message):
            return pick(
                [
                    f"{compact_answer}\n你又不信",
                    f"有印象\n{compact_answer}",
                    f"差不多这个\n{compact_answer}",
                ],
                message + compact_answer,
            )

        style_lines = self.extract_style_lines(memories)[:4]
        return pick(
            [f"我记得是\n{compact_answer}", f"应该是\n{compact_answer}", f"{compact_answer}\n差不多"] + style_lines,
            message + compact_answer + str(len(history)),
        )

    @staticmethod
    def casual_time_atom(message: str, answer: str) -> str:
        if not any(token in message for token in ["多久", "多长"]):
            return answer
        match = re.search(r"(20\d{2})-(\d{1,2})(?:-\d{1,2})?", answer)
        if not match:
            return answer
        year = int(match.group(1)) % 100
        month = int(match.group(2))
        return f"{year}年{month}月那会"

    @staticmethod
    def fact_answer_atom(answer: str) -> str:
        compacted = compact_reply(answer, FACT_MAX_REPLY_CHARS)
        lines = [line.strip() for line in compacted.splitlines() if line.strip()]
        if not lines:
            return compacted
        if len(lines) == 1:
            return lines[0]
        joined = "，".join(lines[:2])
        return joined if len(joined) <= FACT_MAX_REPLY_CHARS else lines[0]

    def fact_first_reply(self, message: str, history: list[dict[str, Any]], memories: list[dict[str, Any]]) -> str | None:
        stripped = message.strip()
        if self.is_recent_chat_memory_question(stripped):
            return self.recent_chat_reply(stripped, history)
        if self.is_identity_correction(stripped):
            return "哦不对\n是我的"
        if self.is_user_identity_question(stripped):
            return self.user_identity_reply(stripped, history, memories)
        if self.is_bot_identity_question(stripped):
            name = top_identity_name(self.facts)
            english = top_english_name(self.facts)
            if self.is_bot_surname_question(stripped):
                if name and name[0]:
                    return f"姓{name[0]}"
                return "你姓什么\n我不记得"
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
            return "我不记得你说过\n哦不对"
        if any(token in message for token in ["英文名", "english name", "English name"]):
            english = self.extract_user_identity_from_history(history, "english")
            if english:
                return english
            return "你英文名\n我不记得"
        if user_name:
            return user_name
        return "NonForgetter吧\n真名我不记得"

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
            return "我不记得"
        top = candidates[0]
        if top.get("score", 0) <= 1 and len(candidates) > 1:
            return f"{top.get('value')}\n但不太确定"
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
            return "有印象\n但得想想"
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
            return "有波动\n我有印象"
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
            return "我不记得"
        labels = [self.display_fact_label(row.get("value", "")) for row in rows[:2]]
        if len(labels) == 1:
            return f"像是{labels[0]}"
        return f"{labels[0]}\n还有{labels[1]}"

    def play_preference_reply(self, message: str, memories: list[dict[str, Any]]) -> str:
        person = "NonForgetter" if any(token in message for token in ["我喜欢", "我爱玩", "我玩什么"]) else "backup"
        labels = self.extract_play_labels(memories, person)
        if labels:
            if len(labels) == 1:
                return f"只记得{labels[0]}\n别的想不起来"
            if len(labels) == 2:
                return f"{labels[0]}\n还有{labels[1]}"
            return f"{labels[0]}、{labels[1]}\n{labels[2]}也有"
        if self.has_shared_topic("games"):
            return "像二游\n具体想不起来"
        return "具体想不起来\n别让我硬猜"

    def rerank_memories_for_question(
        self,
        message: str,
        memories: list[dict[str, Any]],
        topic_state: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if topic_state and not self.should_break_active_topic(message):
            scoped = self.scope_memories_to_topic(memories, topic_state)
            if scoped:
                memories = scoped
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
    def scope_memories_to_topic(memories: list[dict[str, Any]], topic_state: dict[str, Any]) -> list[dict[str, Any]]:
        keywords = [str(word).strip().lower() for word in topic_state.get("keywords") or [] if str(word).strip()]
        category = str(topic_state.get("category") or "")
        if not keywords and category == "open":
            return memories
        category_bonus = {
            "food": ["吃", "饭", "饿", "喝", "外卖", "食堂"],
            "game": ["玩", "打", "游戏", "原神", "星铁", "绝区零", "崩铁", "steam"],
            "presence": ["在吗", "干嘛", "说话", "没声"],
            "sleep": ["困", "睡", "醒", "晚安", "起床"],
            "music": ["歌", "听歌", "循环"],
            "school": ["学校", "上课", "考试", "迟到"],
            "media": ["看", "动画", "番", "视频"],
            "care": ["难受", "烦", "哭", "抱抱", "生气"],
        }.get(category, [])
        probes = list(dict.fromkeys([*keywords, *category_bonus]))
        if not probes:
            return memories
        scored: list[tuple[int, float, int, dict[str, Any]]] = []
        for index, memory in enumerate(memories):
            text = fix_text(memory.get("text") or "").lower()
            hit_count = sum(1 for probe in probes if probe and probe.lower() in text)
            scored.append((hit_count, float(memory.get("score") or 0), -index, memory))
        ranked = sorted(scored, key=lambda item: (item[0], item[1], item[2]), reverse=True)
        hits = [memory for hit, _, _, memory in ranked if hit > 0]
        if len(hits) >= 3:
            return hits[:10]
        return [memory for _, _, _, memory in ranked]

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
        return "吃的想不起来\n只记得老聊吃饭"

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
            return "我不记得"
        labels = [self.display_fact_label(row.get("value", "")) for row in rows[:3]]
        return "\n".join(labels[:3])

    def soothing_reply(self, message: str) -> str | None:
        rows = (self.facts.get("soothing_patterns") or [])[:3]
        if not rows:
            return "我不记得"
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
            return "准点想不起来\n大概中午前后"

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
        return guess

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
    def emotion_prompt_profile(emotion: str) -> dict[str, str]:
        return EMOTION_PROMPT_PROFILES.get(emotion) or EMOTION_PROMPT_PROFILES["casual"]

    @staticmethod
    def resolve_emotion(message: str, history: list[dict[str, Any]], mood: str) -> str:
        if mood and mood != "auto":
            return mood
        text = message.lower()
        if any(token in text for token in ["困", "睡", "晚安", "安安"]):
            return "sleepy"
        if any(token in text for token in ["别", "服了", "逆天", "无语"]):
            return "annoyed"
        if any(token in text for token in ["烦", "难受", "emo", "哭", "累"]):
            return "soft"
        if any(token in text for token in ["？", "?", "什么意思", "怎么", "为什么"]):
            return "curious"
        if any(token in text for token in ["笑死", "哈哈", "乐子", "绷", "6"]):
            return "playful"
        if any(token in text for token in ["我靠", "卧槽", "真的假的", "好耶", "哇", "牛"]):
            return "excited"
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
        return pick(["不是", "算了", "你别管", "哎呀", "好吧", "那咋了", "没事"], stripped + emotion)

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
            return f"像这个\n{title}\n得看上下文"
        if "乐子" in message or "乐乐" in message:
            return "乐子那个乐\n看热闹的意思"
        return "不太确定\n可能只是语境梗"

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
    def is_conversation_repair_question(message: str) -> bool:
        if ChatEngine.should_web_lookup(message):
            return False
        return any(
            token in message
            for token in [
                "你在说什么",
                "你说什么",
                "你说啥",
                "说什么啊",
                "什么东西",
                "这啥玩意",
                "啥玩意",
                "你是不是乱说",
                "你接的什么",
            ]
        )

    @staticmethod
    def is_first_person_preference_statement(message: str) -> bool:
        if ChatEngine.is_question_like(message):
            return False
        return message.startswith("我") and any(
            token in message
            for token in [
                "喜欢玩",
                "爱玩",
                "挺喜欢玩",
                "喜欢吃",
                "爱吃",
                "想吃",
            ]
        )

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
    def build_retrieval_query(
        message: str,
        history: list[dict[str, Any]],
        topic_state: dict[str, Any] | None = None,
    ) -> str:
        recent_user_text = [
            str(item.get("content", ""))
            for item in history[-10:]
            if item.get("role") == "user" and item.get("content")
        ][-5:]
        expansions = []
        domain = ChatEngine.classify_fact_domain(message)
        if topic_state:
            topic_text = str(topic_state.get("topic") or "")
            topic_category = str(topic_state.get("category") or "")
            topic_keywords = " ".join(str(word) for word in topic_state.get("keywords") or [])
            expansions.extend(
                [
                    f"当前话题 {topic_text}",
                    f"话题类别 {topic_category}",
                    f"话题关键词 {topic_keywords}",
                ]
            )
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
    def should_answer_fact_first(message: str) -> bool:
        if ChatEngine.is_recent_chat_memory_question(message) or ChatEngine.is_identity_correction(message):
            return True
        if ChatEngine.is_user_identity_question(message) or ChatEngine.is_bot_identity_question(message):
            return True
        if ChatEngine.is_time_question(message) or ChatEngine.is_wake_time_question(message):
            return True
        if ChatEngine.is_memory_dispute_question(message):
            return True
        if ChatEngine.is_birthday_question(message) or ChatEngine.is_relationship_question(message):
            return ChatEngine.is_question_like(message)
        if ChatEngine.is_soothing_question(message) or ChatEngine.is_topic_question(message):
            return ChatEngine.is_question_like(message)
        if ChatEngine.is_emotion_history_question(message):
            return ChatEngine.is_question_like(message)
        if ChatEngine.is_preference_question(message):
            return True
        if ChatEngine.is_nickname_question(message) or ChatEngine.is_habit_question(message):
            return ChatEngine.is_question_like(message)
        return False

    @staticmethod
    def classify_dialogue_act(message: str) -> str:
        if ChatEngine.is_conversation_repair_question(message):
            return "repair_request"
        if ChatEngine.is_recent_chat_memory_question(message):
            return "recent_memory_question"
        if ChatEngine.should_web_lookup(message):
            return "web_knowledge_question"
        if ChatEngine.should_answer_fact_first(message):
            return "fact_question"
        if ChatEngine.is_first_person_preference_statement(message):
            return "preference_statement"
        if ChatEngine.is_question_like(message):
            return "open_question"
        if len(message.strip()) <= 2:
            return "minimal_ack"
        return "casual_statement"

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
        if not ChatEngine.is_question_like(message):
            return False
        return any(token in message for token in ["喜欢什么", "讨厌什么", "不喜欢什么", "爱吃什么", "想吃什么", "偏好", "喜欢吃", "喜欢玩", "爱玩", "玩什么", "玩啥"])

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
