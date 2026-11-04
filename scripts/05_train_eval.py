from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
LABELS = ["concrete", "abstract"]


def probability_columns(model: LogisticRegression, features) -> tuple[list[float], list[float]]:
    probabilities = model.predict_proba(features)
    classes = list(model.classes_)
    concrete_index = classes.index("concrete")
    abstract_index = classes.index("abstract")
    return probabilities[:, concrete_index].tolist(), probabilities[:, abstract_index].tolist()


def evaluate(model: LogisticRegression, features, truth) -> dict:
    predicted = model.predict(features)
    matrix = confusion_matrix(truth, predicted, labels=LABELS)
    report = classification_report(
        truth, predicted, labels=LABELS, output_dict=True, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": LABELS,
        "classification_report": report,
    }


def feature_coverage(features) -> dict[str, int]:
    nonzero_rows = int((features.getnnz(axis=1) > 0).sum())
    return {
        "rows_with_features": nonzero_rows,
        "rows_without_features": int(features.shape[0] - nonzero_rows),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv", dtype={"id": str})
    test = pd.read_csv(DATA_DIR / "test.csv", dtype={"id": str})
    train_text = train["sentence"].fillna("").astype(str)
    test_text = test["sentence"].fillna("").astype(str)

    vectorizer_a = CountVectorizer(max_features=200)
    x_train_a = vectorizer_a.fit_transform(train_text)
    x_test_a = vectorizer_a.transform(test_text)
    model_a = LogisticRegression(max_iter=1000, random_state=42)
    model_a.fit(x_train_a, train["label"])

    vectorizer_b = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    x_train_b = vectorizer_b.fit_transform(train_text)
    x_test_b = vectorizer_b.transform(test_text)
    model_b = LogisticRegression(max_iter=1000, random_state=42)
    model_b.fit(x_train_b, train["label"])

    vectorizer_c = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        max_features=10000,
    )
    x_train_c = vectorizer_c.fit_transform(train_text)
    x_test_c = vectorizer_c.transform(test_text)
    model_c = LogisticRegression(max_iter=1000, random_state=42)
    model_c.fit(x_train_c, train["label"])

    models = {
        "A_CountVectorizer": (model_a, x_test_a),
        "B_Word_TfidfVectorizer": (model_b, x_test_b),
        "C_CharNgram_TfidfVectorizer": (model_c, x_test_c),
    }
    selected_name = "C_CharNgram_TfidfVectorizer"
    selected_model, selected_features = models[selected_name]

    results = {
        "dataset": {
            "train_size": int(len(train)),
            "test_size": int(len(test)),
            "label_order": LABELS,
            "feature_coverage": {
                "A_CountVectorizer_train": feature_coverage(x_train_a),
                "A_CountVectorizer_test": feature_coverage(x_test_a),
                "B_Word_TfidfVectorizer_train": feature_coverage(x_train_b),
                "B_Word_TfidfVectorizer_test": feature_coverage(x_test_b),
                "C_CharNgram_TfidfVectorizer_train": feature_coverage(x_train_c),
                "C_CharNgram_TfidfVectorizer_test": feature_coverage(x_test_c),
            },
        },
        "models": {
            "A_CountVectorizer": evaluate(model_a, x_test_a, test["label"]),
            "B_Word_TfidfVectorizer": evaluate(model_b, x_test_b, test["label"]),
            "C_CharNgram_TfidfVectorizer": evaluate(model_c, x_test_c, test["label"]),
        },
        "selection": selected_name,
        "model_settings": {
            "A_CountVectorizer": {
                "vectorizer": "CountVectorizer",
                "max_features": 200,
                "classifier": "LogisticRegression",
                "max_iter": 1000,
                "random_state": 42,
            },
            "B_Word_TfidfVectorizer": {
                "vectorizer": "TfidfVectorizer",
                "max_features": 2000,
                "ngram_range": [1, 2],
                "classifier": "LogisticRegression",
                "max_iter": 1000,
                "random_state": 42,
            },
            "C_CharNgram_TfidfVectorizer": {
                "vectorizer": "TfidfVectorizer",
                "analyzer": "char",
                "ngram_range": [2, 5],
                "max_features": 10000,
                "classifier": "LogisticRegression",
                "max_iter": 1000,
                "random_state": 42,
            },
        },
        "notes": (
            "C 使用字元級 TF-IDF n-gram (2,5)，不依賴中文分詞器；"
            "繁體/簡體字形依原文保留。"
        ),
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    concrete, abstract = probability_columns(selected_model, selected_features)
    predictions = pd.DataFrame({
        "id": test["id"],
        "sentence": test["sentence"],
        "true": test["label"],
        "pred": selected_model.predict(selected_features),
        "prob_concrete": concrete,
        "prob_abstract": abstract,
    })
    predictions.to_csv(
        OUTPUT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig"
    )
    joblib.dump(selected_model, OUTPUT_DIR / "model_tfidf.joblib")
    joblib.dump(vectorizer_c, OUTPUT_DIR / "vec_tfidf.joblib")

    print("模型比較（accuracy / macro precision / macro recall / macro F1）")
    print(f"特徵覆蓋率診斷：{results['dataset']['feature_coverage']}")
    for model_name, values in results["models"].items():
        metrics = values["classification_report"]["macro avg"]
        print(
            f"{model_name}: {values['accuracy']:.4f} / "
            f"{metrics['precision']:.4f} / {metrics['recall']:.4f} / {metrics['f1-score']:.4f}"
        )
        print(f"{model_name} 完整分類報告：")
        print(classification_report(
            test["label"],
            models[model_name][0].predict(models[model_name][1]),
            labels=LABELS,
            zero_division=0,
            digits=4,
        ))
        print(f"{model_name} 混淆矩陣（列=true，欄=pred；{LABELS}）：")
        for row in values["confusion_matrix"]:
            print(row)
    print(f"選用模型：{selected_name}")
    print("選用模型混淆矩陣（列=true，欄=pred；concrete, abstract）：")
    for row in results["models"][selected_name]["confusion_matrix"]:
        print(row)


if __name__ == "__main__":
    main()
