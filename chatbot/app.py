from __future__ import annotations

import hashlib
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from chatbot_core.config import AppConfig
from chatbot_core.behavior_runtime import load_or_build_behavior
from chatbot_core.dataset import Dataset
from chatbot_core.facts_runtime import load_or_build_facts
from chatbot_core.llm_runtime import RUNTIME_VERSION, ChatEngine
from chatbot_core.persona_runtime import load_or_build_persona
from chatbot_core.retrieval import Retriever
from chatbot_core.store import ChatStore


ROOT = Path(__file__).resolve().parent
STATIC_ROOT = ROOT / "static"
WALLPAPER_PATH = Path(r"C:\Users\NonFor\Pictures\25240.jpg")
CONFIG = AppConfig.from_env(ROOT)
DATASET = Dataset(CONFIG.prepared_dir)
PERSONA = load_or_build_persona(CONFIG, DATASET)
BEHAVIOR = load_or_build_behavior(CONFIG, DATASET)
FACTS = load_or_build_facts(CONFIG, DATASET)
RETRIEVER = Retriever(DATASET.load_chunks())
ENGINE = ChatEngine(CONFIG, PERSONA, RETRIEVER, FACTS)
STORE = ChatStore(CONFIG.db_path)


def conversation_id_from(value: object) -> str:
    text = str(value or "default").strip()
    return text[:80] or "default"


def json_bytes(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def seeded_pick(items: list[str], seed: str) -> str:
    if not items:
        return "在吗"
    digest = hashlib.sha1(seed.encode("utf-8", errors="ignore")).hexdigest()
    return items[int(digest[:8], 16) % len(items)]


def proactive_reply(conversation_id: str) -> str:
    history = STORE.load_messages(conversation_id, limit=20)
    last_text = history[-1]["content"] if history else ""
    candidates = [
        "你在干嘛",
        "醒着吗",
        "吃饭没",
        "困",
        "你又不说话",
        "我突然想起来",
        "在吗",
    ]
    topic_templates = {
        "吃饭": ["吃饭没", "你吃饭了吗"],
        "睡觉": ["困", "你不会又睡了吧"],
        "游戏": ["你游戏打完没", "还在打吗"],
        "原神": ["原神打了吗"],
        "星铁": ["星铁打了吗"],
        "学校": ["你今天去学校吗"],
        "动画": ["你又看什么"],
        "歌": ["你在听歌吗"],
    }
    for item in FACTS.get("shared_topics") or []:
        topic = str(item.get("value") or "")
        candidates.extend(topic_templates.get(topic, []))
    unique = list(dict.fromkeys(candidates))
    return seeded_pick(unique, f"{conversation_id}|{len(history)}|{last_text}")


class Handler(BaseHTTPRequestHandler):
    server_version = "WechatPersonaChatbot/0.1"

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")

    def send_json(self, payload: object, status: int = 200) -> None:
        body = json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_static(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_json({"error": "not_found"}, 404)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_static(STATIC_ROOT / "index.html")
            return
        if parsed.path.startswith("/static/"):
            safe_name = parsed.path.removeprefix("/static/").replace("\\", "/")
            if ".." in safe_name:
                self.send_json({"error": "bad_path"}, 400)
                return
            self.send_static(STATIC_ROOT / safe_name)
            return
        if parsed.path == "/api/wallpaper":
            self.send_static(WALLPAPER_PATH if WALLPAPER_PATH.exists() else STATIC_ROOT / "assets" / "cmd-bg.jpg")
            return
        if parsed.path == "/api/status":
            self.send_json(
                {
                    "ok": True,
                    "mode": ENGINE.mode,
                    "runtime_version": RUNTIME_VERSION,
                    "sophnet_model": CONFIG.sophnet_model,
                    "rag": "slang_homophone_web_and_recent_context",
                    "message_count": DATASET.manifest.get("message_count"),
                    "chunk_count": DATASET.manifest.get("chunk_count"),
                    "date_range": DATASET.manifest.get("date_range"),
                    "behavior_version": BEHAVIOR.get("version"),
                    "facts_version": FACTS.get("version"),
                    "database": str(CONFIG.db_path),
                }
            )
            return
        if parsed.path == "/api/persona":
            self.send_json(PERSONA)
            return
        if parsed.path == "/api/search":
            query = parse_qs(parsed.query).get("q", [""])[0]
            self.send_json({"results": RETRIEVER.search(query, limit=10)})
            return
        if parsed.path == "/api/behavior":
            self.send_json(BEHAVIOR)
            return
        if parsed.path == "/api/facts":
            self.send_json(ENGINE.public_facts())
            return
        if parsed.path == "/api/history":
            conversation_id = conversation_id_from(parse_qs(parsed.query).get("conversation_id", ["default"])[0])
            self.send_json(
                {
                    "conversation_id": conversation_id,
                    "messages": STORE.load_messages(conversation_id),
                    "memories": STORE.load_memories(conversation_id),
                }
            )
            return
        self.send_json({"error": "not_found"}, 404)

    def do_POST(self) -> None:
        try:
            if self.path == "/api/chat":
                payload = self.read_json()
                conversation_id = conversation_id_from(payload.get("conversation_id"))
                incoming = payload.get("messages")
                if isinstance(incoming, list):
                    parts = [str(item).strip() for item in incoming if str(item).strip()]
                    message = "\n".join(parts)
                else:
                    parts = [str(payload.get("message", "")).strip()]
                    message = parts[0]
                history = STORE.load_messages(conversation_id, limit=160)
                if not message:
                    self.send_json({"error": "empty_message"}, 400)
                    return
                for part in parts:
                    STORE.add_message(conversation_id, "user", part)
                    STORE.remember_from_user_text(conversation_id, part)
                context = STORE.load_memories(conversation_id)
                result = ENGINE.reply(message, history, context, str(payload.get("mood") or "auto"))
                result["reply"] = ENGINE.apply_consecutive_style(
                    str(result.get("reply", "")),
                    message,
                    history,
                    str(result.get("mode") or ""),
                    result.get("memories") or [],
                    str(result.get("emotion") or "casual"),
                )
                STORE.add_message(conversation_id, "assistant", result.get("reply", ""))
                result["conversation_id"] = conversation_id
                result["conversation_memory"] = context
                self.send_json(result)
                return
            if self.path == "/api/clear":
                payload = self.read_json()
                conversation_id = conversation_id_from(payload.get("conversation_id"))
                STORE.clear_conversation(conversation_id)
                self.send_json({"ok": True, "conversation_id": conversation_id})
                return
            if self.path == "/api/proactive":
                payload = self.read_json()
                conversation_id = conversation_id_from(payload.get("conversation_id"))
                history = STORE.load_messages(conversation_id, limit=80)
                reply = proactive_reply(conversation_id)
                reply = ENGINE.apply_consecutive_style(
                    reply,
                    reply,
                    history,
                    "local_proactive",
                    [],
                    str(payload.get("mood") or "casual"),
                )
                STORE.add_message(conversation_id, "assistant", reply)
                STORE.remember_active_topic(conversation_id, reply)
                self.send_json(
                    {
                        "reply": reply,
                        "mode": "local_proactive",
                        "conversation_id": conversation_id,
                        "memories": [],
                    }
                )
                return
            self.send_json({"error": "not_found"}, 404)
        except Exception as exc:
            self.send_json({"error": "server_error", "detail": str(exc)}, 500)


def main() -> None:
    server = ThreadingHTTPServer((CONFIG.host, CONFIG.port), Handler)
    print(f"Serving WeChat Persona Chatbot on http://{CONFIG.host}:{CONFIG.port}")
    print(f"Mode: {ENGINE.mode}")
    server.serve_forever()


if __name__ == "__main__":
    main()
