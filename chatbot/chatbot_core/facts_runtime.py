from __future__ import annotations

import json
import re
from collections import Counter, deque
from datetime import datetime, timedelta
from typing import Any

from .config import AppConfig
from .dataset import Dataset


TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
FACTS_VERSION = "0.2"
STYLE_ROLE = "user"
OTHER_ROLE = "target"
ROLE_LABELS = {STYLE_ROLE: "backup", OTHER_ROLE: "NonForgetter"}
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

EMOTION_LEXICON = {
    "low": ["难受", "伤心", "哭", "emo", "委屈", "崩溃", "破防", "心碎", "烦死", "累死", "要似", "想似"],
    "angry": ["生气", "气死", "烦", "火大", "讨厌", "服了", "别骂", "不爽", "冷暴力"],
    "sleepy_tired": ["困", "睡", "晚安", "熬夜", "累", "起不来", "没睡醒", "睡醒", "起床气"],
    "happy_playful": ["开心", "笑死", "哈哈", "嘻嘻", "乐", "好耶", "绷不住", "可爱"],
    "care_soft": ["抱抱", "摸摸", "心疼", "乖", "别难受", "别生气", "早点休息", "多睡", "吃饭"],
}

RELATIONSHIP_LEXICON = {
    "care": ["早点休息", "晚安", "多睡", "别熬", "吃饭", "心疼", "担心", "抱抱", "陪你", "别难受"],
    "affection": ["喜欢你", "爱你", "想你", "亲亲", "贴贴", "宝宝", "宝贝", "老婆", "男票", "女朋友", "对象"],
    "conflict": ["吵架", "生气", "冷暴力", "别骂", "讨厌", "不理", "伤心", "委屈", "破防", "绝交"],
    "repair": ["对不起", "道歉", "原谅", "和好", "别生气", "不生气", "没事了", "哄", "别哭"],
    "boundary": ["不要", "别这样", "不可以", "不准", "算了", "别问", "不想说", "不需要和我说"],
    "jealousy": ["吃醋", "嫉妒", "占有欲", "前任", "男朋友", "女朋友", "男票", "对象"],
}

RELATIVE_DAY_WORDS = {"今天": 0, "明天": 1, "昨天": -1}


def load_or_build_facts(config: AppConfig, dataset: Dataset) -> dict[str, Any]:
    path = config.persona_dir / "facts.json"
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("version") == FACTS_VERSION:
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
    birthday_counter: dict[str, Counter[str]] = {STYLE_ROLE: Counter(), OTHER_ROLE: Counter()}
    birthday_evidence: dict[str, dict[str, list[dict[str, str]]]] = {STYLE_ROLE: {}, OTHER_ROLE: {}}
    emotion_counter: dict[str, Counter[str]] = {STYLE_ROLE: Counter(), OTHER_ROLE: Counter()}
    emotion_evidence: dict[str, dict[str, list[dict[str, str]]]] = {STYLE_ROLE: {}, OTHER_ROLE: {}}
    relationship_counter: Counter[str] = Counter()
    relationship_evidence: dict[str, list[dict[str, str]]] = {}
    timeline: list[dict[str, str]] = []
    window: deque[dict[str, Any]] = deque(maxlen=24)

    for msg in messages:
        text = str(msg.get("text") or "").strip()
        role = str(msg.get("speaker_role") or "")
        ts = str(msg.get("timestamp") or "")

        collect_identity(text, role, ts, window, name_counter, name_evidence, english_counter, english_evidence)
        collect_birthdays(text, role, ts, birthday_counter, birthday_evidence)
        collect_emotions(text, role, ts, emotion_counter, emotion_evidence)
        collect_relationship(text, role, ts, relationship_counter, relationship_evidence, timeline)
        window.append(msg)

    return {
        "version": FACTS_VERSION,
        "style_role": STYLE_ROLE,
        "current_user": "NonForgetter",
        "persona_display": "backup",
        "date_range": {
            "start": manifest_range.get("start"),
            "end": manifest_range.get("end"),
        },
        "identity": {
            "name_candidates": ranked_candidates(name_counter, name_evidence, minimum=2, limit=8),
            "english_name_candidates": ranked_candidates(english_counter, english_evidence, minimum=2, limit=5),
        },
        "birthdays": {
            ROLE_LABELS[role]: ranked_candidates(counter, birthday_evidence[role], minimum=1, limit=6)
            for role, counter in birthday_counter.items()
        },
        "relationship": {
            "known_since": manifest_range.get("start"),
            "basis": "wechat_prepared manifest date_range.start",
        },
        "relationship_history": {
            "categories": ranked_candidates(relationship_counter, relationship_evidence, minimum=2, limit=8),
            "timeline": dedupe_timeline(timeline, limit=24),
        },
        "emotion_patterns": {
            ROLE_LABELS[role]: ranked_candidates(counter, emotion_evidence[role], minimum=2, limit=8)
            for role, counter in emotion_counter.items()
        },
        "retrieval_policy": {
            "classify_first": "先把问题分到 identity/birthday/relationship/emotion/habit/time/memory_dispute/slang/open_chat。",
            "persistent_facts": "身份、生日候选、关系模式、情绪模式先查 facts.json。",
            "temporary_facts": "具体追问再用 domain query 扩展检索前 10 条，把命中的短证据当临时事实。",
            "api_budget": "证据足够的短事实本地回答；证据不足或需要语气整合时才调用 API。",
            "anti_hallucination": "没有证据时说“记录里没看到/像是”，不要编确定日期、关系或情绪。",
        },
        "runtime_rules": [
            "身份、生日、时间、关系经历、情绪起伏问题先看事实卡和临时检索证据。",
            "事实有证据时不要说不知道；事实不确定时用熟人语气承认只记得线索。",
            "不要把当前聊天对象搞反：当前用户是 NonForgetter，模拟对象是 backup。",
            "API 只负责语气整合，不负责凭空补事实。",
        ],
    }


def collect_identity(
    text: str,
    role: str,
    ts: str,
    window: deque[dict[str, Any]],
    name_counter: Counter[str],
    name_evidence: dict[str, list[dict[str, str]]],
    english_counter: Counter[str],
    english_evidence: dict[str, list[dict[str, str]]],
) -> None:
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


def collect_birthdays(
    text: str,
    role: str,
    ts: str,
    birthday_counter: dict[str, Counter[str]],
    birthday_evidence: dict[str, dict[str, list[dict[str, str]]]],
) -> None:
    if "生日" not in text and "生快" not in text:
        return
    subjects = birthday_subject_roles(text, role)
    dates = date_candidates_from_text(text, ts)
    if not dates and any(token in text for token in ["生日快乐", "生快", "生日呀", "过生日"]):
        inferred = date_from_timestamp(ts)
        if inferred:
            dates = [inferred]
    for subject in subjects:
        for date in dates:
            birthday_counter[subject][date] += 3 if explicit_date_mentioned(text) else 1
            add_evidence(birthday_evidence[subject], date, ts, role, text)


def collect_emotions(
    text: str,
    role: str,
    ts: str,
    emotion_counter: dict[str, Counter[str]],
    emotion_evidence: dict[str, dict[str, list[dict[str, str]]]],
) -> None:
    for emotion, tokens in EMOTION_LEXICON.items():
        hits = [token for token in tokens if token in text]
        if not hits:
            continue
        emotion_counter[role][emotion] += len(hits)
        add_evidence(emotion_evidence[role], emotion, ts, role, text)


def collect_relationship(
    text: str,
    role: str,
    ts: str,
    relationship_counter: Counter[str],
    relationship_evidence: dict[str, list[dict[str, str]]],
    timeline: list[dict[str, str]],
) -> None:
    for category, tokens in RELATIONSHIP_LEXICON.items():
        hits = [token for token in tokens if token in text]
        if not hits:
            continue
        relationship_counter[category] += len(hits)
        add_evidence(relationship_evidence, category, ts, role, text)
        if len(text) >= 2 and ts:
            timeline.append(
                {
                    "timestamp": ts,
                    "category": category,
                    "role": role,
                    "text": text[:120],
                }
            )


def ranked_candidates(
    counter: Counter[str],
    evidence: dict[str, list[dict[str, str]]],
    minimum: int = 1,
    limit: int = 8,
) -> list[dict[str, Any]]:
    return [
        {
            "value": value,
            "score": score,
            "evidence": evidence.get(value, [])[:5],
        }
        for value, score in counter.most_common(limit)
        if score >= minimum
    ]


def birthday_subject_roles(text: str, role: str) -> list[str]:
    other = OTHER_ROLE if role == STYLE_ROLE else STYLE_ROLE
    subjects: list[str] = []
    if any(token in text for token in ["我生日", "我的生日", "自己生日", "本小姐生日"]):
        subjects.append(role)
    if any(token in text for token in ["你生日", "你的生日", "给你过生日", "祝你生日", "你过生日"]):
        subjects.append(other)
    if any(token in text for token in ["生日快乐", "生快"]):
        subjects.append(other)
    if not subjects and any(token in text for token in ["生日", "过生日"]):
        subjects.append(role)
    return list(dict.fromkeys(subjects))


def explicit_date_mentioned(text: str) -> bool:
    return bool(re.search(r"(\d{1,2}|[一二两三四五六七八九十]{1,3})[月/-](\d{1,2}|[一二两三四五六七八九十]{1,3})", text))


def date_candidates_from_text(text: str, ts: str) -> list[str]:
    dates: list[str] = []
    for match in re.finditer(r"(\d{1,2})[月/-](\d{1,2})[日号]?", text):
        normalized = normalize_month_day(match.group(1), match.group(2))
        if normalized:
            dates.append(normalized)
    for match in re.finditer(r"([一二两三四五六七八九十]{1,3})月([一二两三四五六七八九十]{1,3})[日号]?", text):
        month = parse_cn_number(match.group(1))
        day = parse_cn_number(match.group(2))
        normalized = normalize_month_day(month, day)
        if normalized:
            dates.append(normalized)
    for word, offset in RELATIVE_DAY_WORDS.items():
        if word in text:
            date = relative_date(ts, offset)
            if date:
                dates.append(date)
    return list(dict.fromkeys(dates))


def normalize_month_day(month: object, day: object) -> str | None:
    try:
        month_i = int(month)
        day_i = int(day)
    except (TypeError, ValueError):
        return None
    if 1 <= month_i <= 12 and 1 <= day_i <= 31:
        return f"{month_i:02d}-{day_i:02d}"
    return None


def relative_date(ts: str, offset_days: int) -> str | None:
    try:
        value = datetime.strptime(ts[:19], TIME_FORMAT) + timedelta(days=offset_days)
    except ValueError:
        return None
    return value.strftime("%m-%d")


def date_from_timestamp(ts: str) -> str | None:
    try:
        value = datetime.strptime(ts[:19], TIME_FORMAT)
    except ValueError:
        return None
    return value.strftime("%m-%d")


def parse_cn_number(value: str) -> int | None:
    digits = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if value == "十":
        return 10
    if value.startswith("十") and len(value) == 2:
        return 10 + digits.get(value[1], 0)
    if value.endswith("十") and len(value) == 2:
        return digits.get(value[0], 0) * 10
    if "十" in value:
        left, right = value.split("十", 1)
        return digits.get(left, 1) * 10 + digits.get(right, 0)
    return digits.get(value)


def dedupe_timeline(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (row.get("timestamp", "")[:10], row.get("category", ""), row.get("text", "")[:24])
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
        if len(result) >= limit:
            break
    return result


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
    birthday_rows = []
    for person, data in (facts.get("birthdays") or {}).items():
        top = data[0]["value"] if data else "-"
        birthday_rows.append(f"- {person} birthday candidate: {top}")
    relationship = facts.get("relationship_history", {}).get("categories") or []
    emotions = facts.get("emotion_patterns", {}).get("backup") or []
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
            *birthday_rows,
            f"- relationship categories: {', '.join(item['value'] for item in relationship[:5]) or '-'}",
            f"- backup emotion patterns: {', '.join(item['value'] for item in emotions[:5]) or '-'}",
        ]
    )
