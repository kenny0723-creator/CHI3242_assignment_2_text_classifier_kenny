"""依關鍵詞規則標註候選句，並產生可保留人工判斷的複核樣本。

輸出檔及複核欄位沿用舊有 ai_* 命名；本腳本不呼叫任何 AI 服務。
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "data" / "candidates.csv"
LABELS = ROOT / "data" / "ai_labels.csv"
MANUAL = ROOT / "data" / "manual_check.csv"
RANDOM_STATE = 42

ABSTRACT_WORDS = tuple(dict.fromkeys((
    "认为", "觉得", "希望", "应该", "意义", "价值", "精神", "灵魂", "命运",
    "思想", "感情", "道理", "象征", "仿佛", "似乎", "大约", "本来", "向来",
    "永远", "从来", "人生", "世界", "社会", "制度", "礼教", "革命", "自由",
    "平等", "仁义", "道德", "绝望", "寂寞", "悲哀", "空虚", "奇怪", "可怕",
    "可怜", "可笑", "讽刺", "批判", "揭露",
)))
CONCRETE_WORDS = (
    "走", "坐", "站", "拿", "放", "看", "听", "说", "吃", "喝", "穿", "脸",
    "手", "眼", "嘴", "头", "脚", "门", "窗", "桌", "椅", "碗", "杯", "灯",
    "船", "河", "树", "花", "草", "狗", "猫", "天空", "月亮", "太阳", "风",
    "雨", "雪", "街道", "屋子", "酒店", "柜台", "银元", "铜钱", "馒头", "茴香豆",
)
QUOTES = ("“", "”", "「", "」", "『", "』", "\"", "＂")
PRONOUNS = ("我", "我们", "你", "你们", "他", "他们", "她", "她们", "咱", "咱们")


def count_terms(sentence: str, terms: tuple[str, ...]) -> int:
    return sum(sentence.count(term) for term in terms)


def label_sentence(sentence: str) -> tuple[str, int, int]:
    n_abstract = count_terms(sentence, ABSTRACT_WORDS)
    n_concrete = count_terms(sentence, CONCRETE_WORDS)
    if n_abstract > n_concrete:
        return "abstract", n_abstract, n_concrete
    if n_concrete > n_abstract:
        return "concrete", n_abstract, n_concrete

    if any(mark in sentence for mark in QUOTES):
        return "concrete", n_abstract, n_concrete
    pronoun_count = count_terms(sentence, PRONOUNS)
    density = pronoun_count / max(len(sentence), 1)
    if density >= 0.03:
        return "abstract", n_abstract, n_concrete
    return "uncertain", n_abstract, n_concrete


def make_stratified_sample(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    target = min(len(frame), max(40, math.ceil(len(frame) * 0.10)))
    rng = random.Random(RANDOM_STATE)
    selected_ids: set[str] = set()
    groups = {
        label: group.to_dict("records")
        for label, group in frame.groupby("label", sort=True)
    }
    for records in groups.values():
        rng.shuffle(records)
        selected_ids.update(
            row["id"] for row in records[:math.floor(target * len(records) / len(frame))]
        )

    remaining = [
        row["id"] for records in groups.values() for row in records
        if row["id"] not in selected_ids
    ]
    rng.shuffle(remaining)
    selected_ids.update(remaining[: target - len(selected_ids)])
    sample = frame[frame["id"].isin(selected_ids)].copy()
    return sample.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def build_manual_table(frame: pd.DataFrame) -> pd.DataFrame:
    review_frame = frame[frame["label"] != "uncertain"]
    sample = make_stratified_sample(review_frame)
    manual = sample[["id", "sentence", "label"]].rename(columns={"label": "ai_label"})
    manual["my_label"] = ""
    manual["note"] = ""

    # 重跑時保留已完成的人工複核，避免抽樣模板覆蓋既有標註
    if MANUAL.exists():
        previous = pd.read_csv(MANUAL, dtype={"id": str}, encoding="utf-8-sig")
        previous["my_label"] = previous["my_label"].fillna("")
        previous["note"] = previous["note"].fillna("")
        reviewed = previous[previous["my_label"].str.strip().ne("")]
        lookup = reviewed.set_index("id")[["my_label", "note"]].to_dict("index")
        for index, row_id in manual["id"].items():
            if row_id in lookup:
                manual.at[index, "my_label"] = lookup[row_id]["my_label"]
                manual.at[index, "note"] = lookup[row_id]["note"]

        extra = reviewed[~reviewed["id"].isin(manual["id"])]
        if not extra.empty:
            manual = pd.concat(
                [manual, extra[["id", "sentence", "ai_label", "my_label", "note"]]],
                ignore_index=True,
            )
        filled = int(manual["my_label"].str.strip().ne("").sum())
        print(f"已保留人工複核標籤：{filled} 句")
    return manual


def main() -> None:
    frame = pd.read_csv(CANDIDATES, dtype={"id": str})
    annotations = frame["sentence"].map(label_sentence)
    frame["label"] = annotations.map(lambda value: value[0])
    frame["n_abstract"] = annotations.map(lambda value: value[1])
    frame["n_concrete"] = annotations.map(lambda value: value[2])
    frame[["id", "source", "sentence", "label", "n_abstract", "n_concrete"]].to_csv(
        LABELS, index=False, encoding="utf-8-sig"
    )

    manual = build_manual_table(frame)
    manual.to_csv(MANUAL, index=False, encoding="utf-8-sig")

    counts = frame["label"].value_counts().to_dict()
    uncertain_ratio = counts.get("uncertain", 0) / max(len(frame), 1)
    print(f"總句數：{len(frame)}")
    print(f"各標籤句數：{counts}")
    print(f"uncertain 比例：{uncertain_ratio:.2%}")
    print(f"人工抽查句數：{len(manual)}（已標註候選句 {int((frame['label'] != 'uncertain').sum())} 句的至少 10%，最低目標 40）")
    print(f"人工抽查各標籤句數：{manual['ai_label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
