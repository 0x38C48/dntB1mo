from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chatbot_core.config import AppConfig
from chatbot_core.dataset import Dataset
from chatbot_core.persona_runtime import build_and_save_persona


def main() -> None:
    config = AppConfig.from_env(ROOT)
    dataset = Dataset(config.prepared_dir)
    persona = build_and_save_persona(config, dataset)
    stats = persona["statistics"]
    print("Persona built:")
    print(f"- style text messages: {stats['style_text_messages']}")
    print(f"- avg style text length: {stats['avg_style_text_len']}")
    print(f"- short reply ratio: {stats['short_reply_ratio']}")
    print(f"- output: {config.persona_dir}")


if __name__ == "__main__":
    main()
