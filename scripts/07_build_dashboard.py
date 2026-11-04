from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
DOCS_DIR = ROOT / "docs"
DOCS_DATA = DOCS_DIR / "data"
LABELS = ["concrete", "abstract"]


def records(path: Path, columns: list[str] | None = None) -> list[dict]:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if columns:
        frame = frame[columns]
    return json.loads(frame.fillna("").to_json(orient="records", force_ascii=False))


def main() -> None:
    metrics = json.loads((OUTPUT_DIR / "metrics.json").read_text(encoding="utf-8"))
    manual = pd.read_csv(
        DATA_DIR / "manual_check.csv",
        dtype={"id": str},
        encoding="utf-8-sig",
    )
    manual["ai_label"] = manual["ai_label"].fillna("").astype(str).str.strip()
    manual["my_label"] = manual["my_label"].fillna("").astype(str).str.strip()
    reviewed = manual[
        manual["ai_label"].isin(LABELS) & manual["my_label"].isin(LABELS)
    ].copy()
    disagreements = reviewed[reviewed["ai_label"] != reviewed["my_label"]]
    agreement = (
        float((reviewed["ai_label"] == reviewed["my_label"]).mean())
        if len(reviewed)
        else None
    )

    data = {
        "metrics": metrics,
        "manual": {
            "sample_size": int(len(manual)),
            "reviewed_size": int(len(reviewed)),
            "agreement": agreement,
            "ai_distribution": manual["ai_label"].value_counts().to_dict(),
            "human_distribution": reviewed["my_label"].value_counts().to_dict(),
            "disagreement_count": int(len(disagreements)),
            "rows": records(
                DATA_DIR / "manual_check.csv",
                ["id", "sentence", "ai_label", "my_label", "note"],
            ),
        },
        "test_predictions": records(
            OUTPUT_DIR / "test_predictions.csv",
            ["id", "sentence", "true", "pred", "prob_concrete", "prob_abstract"],
        ),
        "analysis_predictions": records(
            OUTPUT_DIR / "analysis_predictions.csv",
            ["id", "sentence", "pred", "prob_concrete", "prob_abstract"],
        ),
    }

    DOCS_DATA.mkdir(parents=True, exist_ok=True)
    (DOCS_DATA / "dashboard-data.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for source, target in (
        (OUTPUT_DIR / "analysis_predictions.csv", DOCS_DATA / "analysis_predictions.csv"),
        (OUTPUT_DIR / "test_predictions.csv", DOCS_DATA / "test_predictions.csv"),
        (DATA_DIR / "manual_check.csv", DOCS_DATA / "manual_check.csv"),
    ):
        shutil.copyfile(source, target)
    print(f"互動頁資料已建立：{(DOCS_DATA / 'dashboard-data.json').relative_to(ROOT)}")
    print(
        f"人工複核一致率："
        f"{'尚無資料' if agreement is None else f'{agreement:.2%}'} "
        f"({len(reviewed)}/{len(manual)})"
    )
    print(f"複核分歧：{len(disagreements)} 句")


if __name__ == "__main__":
    main()
