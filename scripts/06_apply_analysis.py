from __future__ import annotations

import re
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_TEXT = ROOT / "data" / "raw" / "秋河_(郁達夫).txt"
SENTENCE_OUTPUT = ROOT / "data" / "analysis_sentences.csv"
PREDICTION_OUTPUT = ROOT / "output" / "analysis_predictions.csv"
SENTENCE_SPLIT = re.compile(r"(?<=[。！？；])")
PUNCTUATION_ONLY = re.compile(r"^[\s，。！？；：、,.!?;:…—\-\"'“”‘’（）()《》【】\[\]·]+$")
def clean_sentences(text: str) -> list[str]:
    result: list[str] = []
    for part in SENTENCE_SPLIT.split(text):
        sentence = part.strip()
        non_text_line = (
            "作品赏析" in sentence
            or sentence.startswith("作者")
            or bool(re.fullmatch(r"[—－-]{3,}", sentence))
        )
        if (
            not sentence
            or PUNCTUATION_ONLY.fullmatch(sentence)
            or len(sentence) < 5
            or len(sentence) > 120
            or non_text_line
        ):
            continue
        result.append(sentence)
    return result


def main() -> None:
    if not RAW_TEXT.exists():
        source = ROOT / "秋河_(郁達夫).txt"
        if not source.exists():
            raise FileNotFoundError(f"找不到〈秋河〉原始文本：{RAW_TEXT}")
        RAW_TEXT.parent.mkdir(parents=True, exist_ok=True)
        RAW_TEXT.write_bytes(source.read_bytes())

    sentences = clean_sentences(RAW_TEXT.read_text(encoding="utf-8-sig"))
    frame = pd.DataFrame({
        "id": [f"A{index:05d}" for index in range(1, len(sentences) + 1)],
        "sentence": sentences,
    })
    frame.to_csv(SENTENCE_OUTPUT, index=False, encoding="utf-8-sig")

    vectorizer = joblib.load(ROOT / "output" / "vec_tfidf.joblib")
    model = joblib.load(ROOT / "output" / "model_tfidf.joblib")
    features = vectorizer.transform(frame["sentence"].fillna("").astype(str))
    zero_feature_rows = int((features.getnnz(axis=1) == 0).sum())
    print(
        f"向量化診斷：{zero_feature_rows}/{len(frame)} 句沒有任何已知特徵；"
        "這類句子的機率只反映模型截距。"
    )
    probabilities = model.predict_proba(features)
    classes = list(model.classes_)
    concrete_index = classes.index("concrete")
    abstract_index = classes.index("abstract")
    prediction = pd.DataFrame({
        "id": frame["id"],
        "sentence": frame["sentence"],
        "pred": model.predict(features),
        "prob_concrete": probabilities[:, concrete_index],
        "prob_abstract": probabilities[:, abstract_index],
    })
    prediction.to_csv(PREDICTION_OUTPUT, index=False, encoding="utf-8-sig")

    counts = prediction["pred"].value_counts()
    total = max(len(prediction), 1)
    mid_confidence = prediction["prob_concrete"].between(0.4, 0.6).sum()
    print(f"〈秋河〉有效句數：{len(prediction)}")
    for label in ("concrete", "abstract"):
        print(f"{label}：{counts.get(label, 0)} ({counts.get(label, 0) / total:.2%})")
    print(f"prob_concrete 平均/中位數：{prediction['prob_concrete'].mean():.4f} / {prediction['prob_concrete'].median():.4f}")
    print(f"prob_concrete 落在 0.4–0.6：{mid_confidence} ({mid_confidence / total:.2%})")
    prediction["distance_from_0_5"] = (prediction["prob_concrete"] - 0.5).abs()
    lowest = prediction.nsmallest(10, "distance_from_0_5")
    print("信心最低 10 句（列印完整句子及兩類機率）：")
    for row in lowest.itertuples(index=False):
        print(
            f"{row.id} [{row.pred}; concrete={row.prob_concrete:.4f}; "
            f"abstract={row.prob_abstract:.4f}] {row.sentence}"
        )


if __name__ == "__main__":
    main()
