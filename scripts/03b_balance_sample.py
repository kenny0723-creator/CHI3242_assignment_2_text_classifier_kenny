"""套用人工複核修正，再建立 concrete/abstract 平衡訓練樣本。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data" / "ai_labels.csv"
MANUAL = ROOT / "data" / "manual_check.csv"
OUTPUT = ROOT / "data" / "ai_labels_balanced.csv"
CLASS_ORDER = ["concrete", "abstract"]
SAMPLE_SIZE_PER_CLASS = 400
MIN_PER_CLASS = 200
RANDOM_STATE = 42


def apply_manual_review(frame: pd.DataFrame) -> pd.DataFrame:
    manual = pd.read_csv(MANUAL, dtype={"id": str}, encoding="utf-8-sig")
    required = {"id", "my_label"}
    if not required.issubset(manual.columns):
        raise ValueError(f"manual_check.csv 必須包含欄位：{sorted(required)}")
    duplicates = manual[manual["id"].duplicated()]["id"].unique()
    if len(duplicates) > 0:
        raise ValueError(f"manual_check.csv 有重複 id：{sorted(duplicates)}")
    manual["my_label"] = manual["my_label"].fillna("").astype(str).str.strip()
    invalid = ~manual["my_label"].isin(CLASS_ORDER)
    if invalid.any():
        ids = manual.loc[invalid, "id"].tolist()
        raise ValueError(
            "請先為複核表每句填入 concrete 或 abstract；"
            f"尚未完成或標籤無效的 id：{ids[:10]}"
        )
    unknown_ids = set(manual["id"]) - set(frame["id"])
    if unknown_ids:
        raise ValueError(
            f"manual_check.csv 含有不在 ai_labels.csv 的 id：{sorted(unknown_ids)[:10]}"
        )
    corrections = manual.set_index("id")["my_label"].to_dict()
    corrected = frame["id"].map(corrections).fillna(frame["label"])
    n_corrected = int((corrected != frame["label"]).sum())
    print(f"套用人工複核修正：{n_corrected} 句（總複核 {len(corrections)} 句）")
    frame = frame.copy()
    frame["label"] = corrected
    return frame


def main() -> None:
    if not MANUAL.exists():
        print(
            f"STOP: 找不到人工複核表 {MANUAL.name}。"
            "請先在 03 產生的複核表填入 my_label，再執行本腳本。"
        )
        sys.exit(1)

    frame = pd.read_csv(LABELS, dtype={"id": str}, encoding="utf-8-sig")
    frame = apply_manual_review(frame)
    eligible = frame[frame["label"].isin(CLASS_ORDER)].copy()
    counts = eligible["label"].value_counts().reindex(CLASS_ORDER, fill_value=0)
    print(f"label source: {LABELS.name}; distribution: {counts.to_dict()}")

    if (counts < MIN_PER_CLASS).any():
        print(
            f"STOP: a class has fewer than {MIN_PER_CLASS} rows; "
            f"counts={counts.to_dict()}. "
            "Label more sentences or adjust review before retraining."
        )
        sys.exit(1)

    sample_count = min(SAMPLE_SIZE_PER_CLASS, int(counts.min()))
    parts = [
        eligible[eligible["label"] == label].sample(n=sample_count, random_state=RANDOM_STATE)
        for label in CLASS_ORDER
    ]
    balanced = (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
    balanced.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    balanced_counts = balanced["label"].value_counts().reindex(CLASS_ORDER, fill_value=0)
    print(f"balanced distribution (1:1): {balanced_counts.to_dict()}")
    print(f"balanced total: {len(balanced)}")


if __name__ == "__main__":
    main()
