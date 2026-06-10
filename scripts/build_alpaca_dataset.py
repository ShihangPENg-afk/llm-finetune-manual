from __future__ import annotations

import json
import re
from pathlib import Path


INSTRUCTION_TEMPLATES = [
    "请根据给定资料，概括该段内容的核心信息。",
    "请阅读下面的资料，并提取其中的关键信息。",
    "请根据以下内容，总结这一段的重点。",
    "请说明这段资料主要讲了什么。",
]

MIN_TEXT_LEN = 160
MAX_SAMPLES = 150


NOISE_PATTERNS = [
    r"第\s*\d+\s*页",
    r"Page\s*\d+",
    r"Synthetic PDF corpus",
    r"中文纯文本数据集 PDF 转换版",
    r"数据集说明",
]


def remove_noise_phrases(text: str) -> str:
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = remove_noise_phrases(text)
    return text


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？；.!?;])\s*", text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def choose_summary_sentences(text: str, max_sentences: int = 2) -> str:
    sentences = split_sentences(text)

    # 过滤过短、像标题或碎片的句子
    candidates = []
    for s in sentences:
        if len(s) < 18:
            continue
        if re.fullmatch(r"[\d\W]+", s):
            continue
        candidates.append(s)

    if not candidates:
        return text[:220].strip()

    # 优先选前面较完整的句子，但不超过两句
    chosen = candidates[:max_sentences]
    summary = " ".join(chosen).strip()

    if len(summary) > 260:
        summary = summary[:260].strip()

    return summary


def build_samples(chunks: list[dict], max_samples: int = MAX_SAMPLES) -> list[dict]:
    samples = []
    seen_inputs = set()

    for item in chunks:
        raw_text = item.get("text", "")
        text = clean_text(raw_text)

        if len(text) < MIN_TEXT_LEN:
            continue
        if text in seen_inputs:
            continue

        seen_inputs.add(text)

        instruction = INSTRUCTION_TEMPLATES[len(samples) % len(INSTRUCTION_TEMPLATES)]
        output = choose_summary_sentences(text, max_sentences=2)

        if len(output) < 40:
            continue

        sample = {
            "instruction": instruction,
            "input": text,
            "output": output,
        }
        samples.append(sample)

        if len(samples) >= max_samples:
            break

    return samples


def main():
    chunk_path = Path("data/processed/chunks.json")
    out_path = Path("data/processed/alpaca_train.json")

    if not chunk_path.exists():
        raise FileNotFoundError(f"找不到 chunk 文件: {chunk_path}")

    chunks = json.loads(chunk_path.read_text(encoding="utf-8"))
    samples = build_samples(chunks)

    out_path.write_text(
        json.dumps(samples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"✅ 原始 chunk 数量: {len(chunks)}")
    print(f"✅ 共生成 {len(samples)} 条 Alpaca 样本")
    print(f"✅ instruction 模板数: {len(INSTRUCTION_TEMPLATES)}")
    print(f"✅ 输出文件: {out_path}")


if __name__ == "__main__":
    main()