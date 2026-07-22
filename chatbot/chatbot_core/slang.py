from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlangHit:
    group: str
    matched: tuple[str, ...]
    aliases: tuple[str, ...]
    reply_cues: tuple[str, ...]


SLANG_GROUPS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "meme_fun": (
        ("乐乐", "乐子", "乐子人", "找乐子", "看乐子", "取乐", "乐"),
        (
            "把“乐乐/乐子”按网络梗处理，可能是好笑、看热闹、找乐子，不要只当普通人名",
            "优先结合上下文判断是在取名、开玩笑还是解释梗",
        ),
    ),
    "laugh": (
        ("笑死", "笑鼠", "xswl", "绷不住", "蚌埠住", "绷", "哈哈", "草", "艹", "6"),
        ("接住笑点，用短句、重复符号或轻微吐槽回应",),
    ),
    "absurd": (
        ("离谱", "逆天", "抽象", "地狱笑话", "怪", "牛"),
        ("按荒诞/吐槽语境理解，回答可以跳一点，不要机械解释梗本身",),
    ),
    "done": (
        ("寄", "寄了", "丸辣", "完辣", "完了", "麻了", "破防", "开摆", "摆烂"),
        ("按崩溃、放弃、被打击的情绪理解，可短促共情，也可顺势吐槽",),
    ),
    "true_or_fake": (
        ("尊嘟假嘟", "真的假的", "真假", "真的假的啊"),
        ("按怀疑/震惊语气理解，可以追问或短促反应",),
    ),
    "thanks": (
        ("栓q", "3q", "thank you", "谢谢", "谢了"),
        ("按轻松感谢语境理解，可以简短回应，不要太正式",),
    ),
    "sad": (
        ("难受想哭", "难受", "想哭", "emo", "抑郁", "玉玉"),
        ("按低落语境理解，更像朋友聊天，轻一点安抚",),
    ),
    "surprise": (
        ("我超", "卧槽", "我靠", "woc", "wc", "啊？", "？？"),
        ("按震惊/情绪上扬理解，可用短句和连续标点模拟反应",),
    ),
    "game": (
        ("原神", "ys", "崩铁", "星铁", "sr", "zzz", "绝区零", "鸣潮", "mc", "米哈游"),
        ("优先按游戏/二游上下文召回记忆，回答可以带一点玩家式吐槽",),
    ),
    "relationship": (
        ("cp", "磕", "kswl", "好嗑", "官配", "拉郎"),
        ("按CP/关系梗理解，可以轻微起哄或追问细节",),
    ),
}


HOMOPHONE_GROUPS: tuple[tuple[str, ...], ...] = (
    ("吗", "嘛", "妈", "马"),
    ("呢", "捏"),
    ("了", "啦", "辣", "喇"),
    ("吧", "罢", "八"),
    ("寄", "寄寄", "寄了"),
)


def _contains(text: str, term: str) -> bool:
    lowered = term.lower()
    return lowered in text or term in text


def analyze_slang(text: str) -> list[SlangHit]:
    raw = text or ""
    lowered = raw.lower()
    hits: list[SlangHit] = []
    for group, (terms, cues) in SLANG_GROUPS.items():
        matched = tuple(term for term in terms if _contains(lowered, term))
        if matched:
            hits.append(SlangHit(group=group, matched=matched, aliases=terms, reply_cues=cues))

    for variants in HOMOPHONE_GROUPS:
        matched = tuple(term for term in variants if term in raw)
        if matched:
            hits.append(
                SlangHit(
                    group="homophone_particle",
                    matched=matched,
                    aliases=variants,
                    reply_cues=("允许把语气词、谐音字当作同类语气召回",),
                )
            )
    return hits


def expand_for_retrieval(text: str) -> str:
    hits = analyze_slang(text)
    if not hits:
        return text or ""
    terms: list[str] = [text or ""]
    for hit in hits:
        terms.extend(hit.aliases)
        terms.extend(hit.matched)
    return " ".join(dict.fromkeys(term for term in terms if term))


def reply_cues(text: str) -> list[str]:
    cues: list[str] = []
    for hit in analyze_slang(text):
        cues.extend(hit.reply_cues)
    return list(dict.fromkeys(cues))[:8]
