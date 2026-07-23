from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_local_env(root: Path) -> dict[str, str]:
    path = root / ".env.local"
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip().lstrip("\ufeff")] = value.strip().strip('"').strip("'")
    return values


def env_value(local_env: dict[str, str], key: str, default: str | None = None) -> str | None:
    return os.getenv(key) or local_env.get(key) or default


@dataclass(frozen=True)
class AppConfig:
    root: Path
    project_root: Path
    prepared_dir: Path
    persona_dir: Path
    db_path: Path
    host: str
    port: int
    openai_api_key: str | None
    openai_model: str
    openai_base_url: str
    sophnet_api_key: str | None
    sophnet_model: str
    sophnet_base_url: str

    @classmethod
    def from_env(cls, root: Path) -> "AppConfig":
        project_root = root.parent
        local_env = load_local_env(root)
        return cls(
            root=root,
            project_root=project_root,
            prepared_dir=Path(env_value(local_env, "WECHAT_PREPARED_DIR", str(project_root / "wechat_prepared"))),
            persona_dir=Path(env_value(local_env, "PERSONA_DIR", str(root / "persona"))),
            db_path=Path(env_value(local_env, "CHATBOT_DB_PATH", str(root / "chat_memory.db"))),
            host=env_value(local_env, "CHATBOT_HOST", "127.0.0.1") or "127.0.0.1",
            port=int(env_value(local_env, "CHATBOT_PORT", "8765") or "8765"),
            openai_api_key=env_value(local_env, "OPENAI_API_KEY"),
            openai_model=env_value(local_env, "OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini",
            openai_base_url=env_value(local_env, "OPENAI_BASE_URL", "https://api.openai.com/v1") or "https://api.openai.com/v1",
            sophnet_api_key=env_value(local_env, "SOPHNET_API_KEY"),
            sophnet_model=env_value(local_env, "SOPHNET_MODEL", "GLM-5.2") or "GLM-5.2",
            sophnet_base_url=env_value(local_env, "SOPHNET_BASE_URL", "https://www.sophnet.com/api/open-apis/v1") or "https://www.sophnet.com/api/open-apis/v1",
        )
