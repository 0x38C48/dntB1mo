from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    root: Path
    project_root: Path
    prepared_dir: Path
    persona_dir: Path
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
        return cls(
            root=root,
            project_root=project_root,
            prepared_dir=Path(os.getenv("WECHAT_PREPARED_DIR", project_root / "wechat_prepared")),
            persona_dir=Path(os.getenv("PERSONA_DIR", root / "persona")),
            host=os.getenv("CHATBOT_HOST", "127.0.0.1"),
            port=int(os.getenv("CHATBOT_PORT", "8765")),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            sophnet_api_key=os.getenv("SOPHNET_API_KEY"),
            sophnet_model=os.getenv("SOPHNET_MODEL", "DeepSeek-V4-Flash"),
            sophnet_base_url=os.getenv("SOPHNET_BASE_URL", "https://www.sophnet.com/api/open-apis/v1"),
        )
