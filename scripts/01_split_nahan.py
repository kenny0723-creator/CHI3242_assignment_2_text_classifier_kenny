from __future__ import annotations

import shutil
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
SOURCE_NAME = "呐喊.txt"

STORIES = {
    "狂人日记": ("kuangren.txt", "某君昆仲"),
    "孔乙己": ("kongyiji.txt", "鲁镇的酒店的格局"),
    "药": ("yao.txt", "秋天的后半夜"),
    "明天": ("mingtian.txt", "原来鲁镇是僻静地方"),
    "一件小事": ("yijianxiaoshi.txt", "我从乡下跑到京城里"),
    "头发的故事": ("toufa.txt", "星期日的早晨"),
    "风波": ("fengbo.txt", "临河的土场上"),
    "故乡": ("guxiang.txt", "我冒了严寒"),
    "阿Ｑ正传": ("aq.txt", "第一章序"),
    "端午节": ("duanwu.txt", "方玄绰近来爱说"),
    "白光": ("baiguang.txt", "陈士成看过县考的榜"),
    "兔和猫": ("tuhemiao.txt", "住在我们后进院子里的三太太"),
    "鸭的喜剧": ("yadexiju.txt", "俄国的盲诗人爱罗先珂君带了"),
    "社戏": ("shexi.txt", "在倒数上去的二十年中"),
}
ALL_TITLES = {
    "狂人日记", "孔乙己", "药", "明天", "一件小事", "头发的故事",
    "风波", "故乡", "阿Ｑ正传", "端午节", "白光", "兔和猫",
    "鸭的喜剧", "社戏",
}
ENDING_DATE = re.compile(r"^一九[〇零一二三四五六七八九0-9]+年[〇零一二三四五六七八九十0-9]+月$")
MIN_STORY_CHARS = 1500
MAX_STORY_CHARS = 12000
ACCEPTED_OUT_OF_RANGE = {"一件小事", "阿Ｑ正传"}


def ensure_raw_source() -> Path:
    target = RAW_DIR / SOURCE_NAME
    if target.exists():
        return target
    source = ROOT / SOURCE_NAME
    if not source.exists():
        raise FileNotFoundError(f"找不到原始文本：{source}")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def trim_story(lines: list[str], title: str, opening: str) -> str:
    title_indices = [
        index for index, line in enumerate(lines)
        if line.strip() in ALL_TITLES
    ]
    positions = [index for index in title_indices if lines[index].strip() == title]
    if len(positions) != 1:
        raise ValueError(f"篇名「{title}」應恰好出現一次，實際為 {len(positions)} 次")

    start_title = positions[0]
    next_title = next(
        (index for index in title_indices if index > start_title),
        len(lines),
    )
    section = lines[start_title + 1:next_title]
    opening_index = next(
        (index for index, line in enumerate(section) if opening in line),
        None,
    )
    if opening_index is None:
        raise ValueError(f"「{title}」找不到正文起始語句：{opening}")

    body = section[opening_index:]
    dated_lines = [
        index for index, line in enumerate(body)
        if ENDING_DATE.fullmatch(line.strip())
    ]
    if dated_lines:
        body = body[:dated_lines[-1] + 1]
    else:
        ending_markers = [
            index for index, line in enumerate(body)
            if line.strip() == "* * *" or line.strip().startswith("———")
        ]
        if not ending_markers:
            raise ValueError(f"「{title}」正文末尾找不到日期或作品分隔符")
        body = body[:ending_markers[-1]]

    text = "\n".join(body).strip()
    if not text:
        raise ValueError(f"「{title}」切分後正文為空")
    return text + "\n"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    source_path = ensure_raw_source()
    lines = source_path.read_text(encoding="utf-8-sig").splitlines()
    print(f"來源：{source_path.relative_to(ROOT)}")

    extracted: dict[str, tuple[str, str]] = {}
    out_of_range: list[str] = []
    for title, (filename, opening) in STORIES.items():
        text = trim_story(lines, title, opening)
        chars = len(text.strip())
        extracted[title] = (filename, text)
        status = "正常" if MIN_STORY_CHARS <= chars <= MAX_STORY_CHARS else "超出預期範圍"
        print(f"{title}：{chars:,} 字（{status}）")
        if not MIN_STORY_CHARS <= chars <= MAX_STORY_CHARS:
            out_of_range.append(title)

    unexpected = [title for title in out_of_range if title not in ACCEPTED_OUT_OF_RANGE]
    if unexpected:
        names = "、".join(unexpected)
        raise SystemExit(
            f"停止：{names} 的字數不在 {MIN_STORY_CHARS}–{MAX_STORY_CHARS} 範圍；"
            "未寫入或覆蓋任何篇目檔案，請先確認原始文本或調整檢查門檻。"
        )
    if out_of_range:
        print(
            "已依使用者確認保留自然篇幅例外：" + "、".join(out_of_range)
        )

    for title, (filename, text) in extracted.items():
        output_path = RAW_DIR / filename
        output_path.write_text(text, encoding="utf-8", newline="\n")
        print(f"已輸出：{title} -> {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
