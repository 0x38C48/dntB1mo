from __future__ import annotations

import json
import re
from collections import Counter, deque
from datetime import datetime
from typing import Any

from .config import AppConfig
from .dataset import Dataset


TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
STYLE_ROLE = "user"
OTHER_ROLE = "target"
COMMON_SURNAMES = set(
    "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜"
    "戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳鲍史唐费"
    "廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄和"
    "穆萧尹姚邵湛汪祁毛禹狄米贝明臧计伏成戴谈宋庞熊纪舒屈项祝董梁杜阮"
    "蓝闵席季麻强贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田胡凌霍虞万支"
    "柯昝管卢莫经房裘缪干解应宗丁宣邓郁单杭洪包诸左石崔吉龚程邢裴陆荣"
    "翁荀羊於惠甄曲家封芮羿储靳汲邴糜松井段富巫乌焦巴弓牧隗山谷车侯宓"
    "蓬全郗班仰秋仲伊宫宁仇栾暴甘厉戎祖武符刘景詹束龙叶幸司韶郜黎蓟薄"
    "印宿白怀蒲台从鄂索咸籍赖卓蔺屠蒙池乔阳胥能苍双闻莘党翟谭贡劳逄姬"
    "申扶堵冉宰郦雍郤璩桑桂濮牛寿通边扈燕冀浦尚农温别庄晏柴瞿阎充慕连"
    "茹习宦艾鱼容向古易慎戈廖庾终暨居衡步都耿满弘匡国文寇广禄阙东欧殳"
    "沃利蔚越夔隆师巩厍聂晁勾敖融冷訾辛阚那简饶空曾毋沙乜养鞠须丰巢关"
)


def load_or_build_facts(config: AppConfig, dataset: Dataset) -> dict[str, Any]:
    path = config.persona_dir / "facts.json"
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("version") == "0.1":
            return cached
    facts = build_facts(dataset)
    config.persona_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
    (config.persona_dir / "facts.md").write_text(render_facts_md(facts), encoding="utf-8", newline="\n")
    return facts


def build_facts(dataset: Dataset) -> dict[str, Any]:
    manifest_range = dataset.manifest.get("date_range") or {}
    messages = [
        msg
        for msg in dataset.iter_messages()
        if msg.get("speaker_role") in {STYLE_ROLE, OTHER_ROLE}
        and msg.get("content_type") == "text"
        and (msg.get("text") or "").strip()
    ]
    messages.sort(key=lambda msg: (msg.get("timestamp") or "", msg.get("message_id") or 0))

    name_counter: Counter[str] = Counter()
    name_evidence: dict[str, list[dict[str, str]]] = {}
    english_counter: Counter[str] = Counter()
    english_evidence: dict[str, list[dict[str, str]]] = {}
    window: deque[dict[str, Any]] = deque(maxlen=24)

    for msg in messages:
        text = str(msg.get("text") or "").strip()
        role = str(msg.get("speaker_role") or "")
        ts = str(msg.get("timestamp") or "")

        for candidate in re.findall(r"([\u4e00-\u9fff]{2,4})(?:女士|同学|小姐|先森|先生)", text):
            if role == OTHER_ROLE and useful_name(candidate):
                name_counter[candidate] += 3
                add_evidence(name_evidence, candidate, ts, role, text)

        if role == STYLE_ROLE and ("我英文名" in text or "英文名" in text):
            for previous in list(window)[-8:]:
                if previous.get("speaker_role") != OTHER_ROLE:
                    continue
                prev_text = str(previous.get("text") or "")
                for candidate in re.findall(r"[a-zA-Z]{3,16}", prev_text):
                    if candidate.lower() not in {"the", "and", "you", "for", "with", "that"}:
                        english_counter[candidate.lower()] += 2
                        add_evidence(
                            english_evidence,
                            candidate.lower(),
                            str(previous.get("timestamp") or ""),
                            str(previous.get("speaker_role") or ""),
                            prev_text,
                        )

        if role == STYLE_ROLE and "英文名" in text:
            for candidate in re.findall(r"[a-zA-Z]{3,16}", text):
                lowered = candidate.lower()
                if lowered not in {"the", "and", "you", "for", "with", "that", "this", "what", "wtf", "hhh"}:
                    english_counter[lowered] += 1
                    add_evidence(english_evidence, lowered, ts, role, text)

        if role == OTHER_ROLE and ("名字" in text or "姓什么" in text):
            for candidate in re.findall(r"[\u4e00-\u9fff]{2,4}", text):
                if useful_name(candidate):
                    name_counter[candidate] += 1
                    add_evidence(name_evidence, candidate, ts, role, text)

        window.append(msg)

    name_candidates = [
        {
            "value": value,
            "score": score,
            "evidence": name_evidence.get(value, [])[:5],
        }
        for value, score in name_counter.most_common(8)
        if score >= 2
    ]
    english_candidates = [
        {
            "value": value,
            "score": score,
            "evidence": english_evidence.get(value, [])[:5],
        }
        for value, score in english_counter.most_common(5)
        if score >= 2
    ]
    return {
        "version": "0.1",
        "style_role": STYLE_ROLE,
        "current_user": "NonForgetter",
        "persona_display": "backup",
        "date_range": {
            "start": manifest_range.get("start"),
            "end": manifest_range.get("end"),
        },
        "identity": {
            "name_candidates": name_candidates,
            "english_name_candidates": english_candidates,
        },
        "relationship": {
            "known_since": manifest_range.get("start"),
            "basis": "wechat_prepared manifest date_range.start",
        },
        "runtime_rules": [
            "身份和时间问题先看事实卡，再决定是否用RAG/API润色。",
            "事实有证据时不要说不知道；事实不确定时用熟人语气承认只记得线索。",
            "不要把当前聊天对象搞反：当前用户是 NonForgetter，模拟对象是 backup。",
        ],
    }


def useful_name(candidate: str) -> bool:
    blocked = {
        "什么",
        "名字",
        "自己",
        "这个",
        "那个",
        "好名字",
        "改名字",
        "我名字",
        "你名字",
        "英文名",
    }
    if candidate in blocked:
        return False
    if len(candidate) < 2 or len(candidate) > 4:
        return False
    if candidate[0] not in COMMON_SURNAMES:
        return False
    if any(token in candidate for token in ["高中", "名字", "国榜", "阿离", "看到", "我的"]):
        return False
    return True


def add_evidence(bucket: dict[str, list[dict[str, str]]], key: str, timestamp: str, role: str, text: str) -> None:
    rows = bucket.setdefault(key, [])
    if len(rows) >= 8:
        return
    rows.append({"timestamp": timestamp, "role": role, "text": text[:160]})


def days_since(value: str | None, now: datetime | None = None) -> int | None:
    if not value:
        return None
    try:
        start = datetime.strptime(value[:19], TIME_FORMAT)
    except ValueError:
        return None
    now = now or datetime.now()
    return max(0, (now - start).days)


def duration_phrase(days: int | None) -> str:
    if days is None:
        return "我只能查到记录里从 2024-10-13 开始"
    years = days // 365
    months = (days % 365) // 30
    if years:
        return f"{years}年{months}个月左右，{days}天"
    if months:
        return f"{months}个月左右，{days}天"
    return f"{days}天"


def top_identity_name(facts: dict[str, Any]) -> str | None:
    candidates = facts.get("identity", {}).get("name_candidates") or []
    return candidates[0].get("value") if candidates else None


def top_english_name(facts: dict[str, Any]) -> str | None:
    candidates = facts.get("identity", {}).get("english_name_candidates") or []
    return candidates[0].get("value") if candidates else None


def render_facts_md(facts: dict[str, Any]) -> str:
    name = top_identity_name(facts) or "-"
    english = top_english_name(facts) or "-"
    start = facts.get("date_range", {}).get("start") or "-"
    days = duration_phrase(days_since(start))
    return "\n".join(
        [
            "# Runtime Facts",
            "",
            f"- persona display: {facts.get('persona_display')}",
            f"- current user: {facts.get('current_user')}",
            f"- top name candidate: {name}",
            f"- top english name candidate: {english}",
            f"- known since: {start}",
            f"- duration now: {days}",
        ]
    )
