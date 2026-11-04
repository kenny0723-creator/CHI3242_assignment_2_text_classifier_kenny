from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
LABELS = DATA_DIR / "ai_labels_balanced.csv"
LABEL_ORDER = ["concrete", "abstract"]


def main() -> None:
    frame = pd.read_csv(LABELS, dtype={"id": str})

    frame = frame[frame["label"].isin(LABEL_ORDER)].copy()
    counts = frame["label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
    if (counts == 0).any():
        raise ValueError(f"平衡標註資料缺少類別：{counts.to_dict()}")
    if counts.nunique() != 1:
        raise ValueError(f"輸入平衡檔類別數量不一致：{counts.to_dict()}")

    frame = frame.sample(frac=1, random_state=42).reset_index(drop=True)
    train, test = train_test_split(
        frame,
        test_size=0.2,
        stratify=frame["label"],
        random_state=42,
    )
    frame.to_csv(DATA_DIR / "train_labeled.csv", index=False, encoding="utf-8-sig")
    train.to_csv(DATA_DIR / "train.csv", index=False, encoding="utf-8-sig")
    test.to_csv(DATA_DIR / "test.csv", index=False, encoding="utf-8-sig")
    print(f"train_labeled.csv：{len(frame)}；{frame['label'].value_counts().to_dict()}")
    print(f"train.csv：{len(train)}；{train['label'].value_counts().to_dict()}")
    print(f"test.csv：{len(test)}；{test['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
