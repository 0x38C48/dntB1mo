from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from .slang import expand_for_retrieval


def tokenize(text: str) -> list[str]:
    text = expand_for_retrieval(text or "").lower()
    ascii_tokens = re.findall(r"[a-z0-9_]{2,}", text)
    chinese = re.findall(r"[\u4e00-\u9fff]+", text)
    grams: list[str] = []
    for span in chinese:
        grams.append(span)
        grams.extend(span[i : i + 2] for i in range(max(0, len(span) - 1)))
        grams.extend(span[i : i + 3] for i in range(max(0, len(span) - 2)))
    return ascii_tokens + grams


class Retriever:
    def __init__(self, chunks: list[dict[str, Any]]):
        self.chunks = chunks
        self.doc_tokens: list[Counter[str]] = []
        self.df: Counter[str] = Counter()
        for chunk in chunks:
            counts = Counter(tokenize(chunk.get("embedding_text") or chunk.get("text") or ""))
            self.doc_tokens.append(counts)
            for token in counts:
                self.df[token] += 1
        self.n_docs = max(1, len(chunks))

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        q_tokens = Counter(tokenize(query))
        if not q_tokens:
            return []
        scored: list[tuple[float, int]] = []
        for idx, counts in enumerate(self.doc_tokens):
            score = 0.0
            doc_len = sum(counts.values()) or 1
            for token, q_count in q_tokens.items():
                tf = counts.get(token, 0)
                if not tf:
                    continue
                idf = math.log((self.n_docs + 1) / (self.df[token] + 0.5))
                score += q_count * (tf / math.sqrt(doc_len)) * idf
            if score > 0:
                scored.append((score, idx))
        scored.sort(reverse=True)
        results = []
        for score, idx in scored[:limit]:
            chunk = self.chunks[idx]
            results.append(
                {
                    "score": round(score, 4),
                    "chunk_id": chunk.get("chunk_id"),
                    "start_time": chunk.get("start_time"),
                    "end_time": chunk.get("end_time"),
                    "text": (chunk.get("text") or "")[:1400],
                    "safety_flags": chunk.get("safety_flags", []),
                }
            )
        return results
