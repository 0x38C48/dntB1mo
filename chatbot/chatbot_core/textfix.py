from __future__ import annotations

from typing import Any


def fix_text(value: Any) -> str:
    """Repair common UTF-8-as-GBK mojibake without touching healthy text."""
    if value is None:
        return ""
    text = str(value)
    if not text:
        return ""
    if not looks_mojibake(text):
        return text
    try:
        fixed = text.encode("gbk").decode("utf-8")
    except UnicodeError:
        return text
    return fixed if fixed else text


def looks_mojibake(text: str) -> bool:
    markers = ("涔", "愪", "鎬", "庝", "箞", "鍟", "锛", "銆", "鑱", "妯", "棣")
    return sum(1 for marker in markers if marker in text) >= 2


def fix_obj(value: Any) -> Any:
    if isinstance(value, str):
        return fix_text(value)
    if isinstance(value, list):
        return [fix_obj(item) for item in value]
    if isinstance(value, dict):
        return {key: fix_obj(item) for key, item in value.items()}
    return value
