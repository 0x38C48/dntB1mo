from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .textfix import fix_obj


class Dataset:
    def __init__(self, prepared_dir: Path):
        self.prepared_dir = prepared_dir
        self.manifest = self._read_json(prepared_dir / "manifest.json")

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        return fix_obj(json.loads(path.read_text(encoding="utf-8")))

    @staticmethod
    def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield fix_obj(json.loads(line))

    def iter_messages(self) -> Iterable[dict[str, Any]]:
        yield from self.iter_jsonl(self.prepared_dir / "messages.jsonl")

    def load_chunks(self) -> list[dict[str, Any]]:
        path = self.prepared_dir / "rag_chunks.jsonl"
        if not path.exists():
            return []
        return list(self.iter_jsonl(path))
