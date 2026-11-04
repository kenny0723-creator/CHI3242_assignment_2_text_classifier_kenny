from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT = ROOT / "data" / "candidates.csv"
SOURCES = [
    ("狂人日记", "kuangren.txt"),
    ("孔乙己", "kongyiji.txt"),
    ("药", "yao.txt"),
    ("明天", "mingtian.txt"),
    ("一件小事", "yijianxiaoshi.txt"),
    ("头发的故事", "toufa.txt"),
    ("风波", "fengbo.txt"),
    ("故乡", "guxiang.txt"),
    ("阿Ｑ正传", "aq.txt"),
    ("端午节", "duanwu.txt"),
    ("白光", "baiguang.txt"),
    ("兔和猫", "tuhemiao.txt"),
    ("鸭的喜剧", "yadexiju.txt"),
    ("社戏", "shexi.txt"),
]
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
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    source_counts: dict[str, int] = {}
    for source, filename in SOURCES:
        path = RAW_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"找不到篇目檔案，請先執行 01：{path}")
        count = 0
        for sentence in clean_sentences(path.read_text(encoding="utf-8")):
            if sentence in seen:
                continue
            seen.add(sentence)
            count += 1
            rows.append({
                "id": f"C{len(rows) + 1:05d}",
                "source": source,
                "sentence": sentence,
            })
        source_counts[source] = count

    frame = pd.DataFrame(rows, columns=["id", "source", "sentence"])
    frame = frame.sample(frac=1, random_state=42).reset_index(drop=True)
    frame.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"候選句總數：{len(frame)}")
    print("篇目分佈：" + ", ".join(f"{key}={value}" for key, value in source_counts.items()))


if __name__ == "__main__":
    main()
