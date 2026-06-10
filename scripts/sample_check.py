from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path


NOISE_PATTERNS = [
    r"第\s*\d+\s*页",
    r"Page\s*\d+",
    r"Synthetic PDF corpus",
    r"中文纯文本数据集 PDF 转换版",
    r"数据集说明",
]


def has_noise(text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in NOISE_PATTERNS)


def main():
    path = Path("data/processed/alpaca_train.json")
    if not path.exists():
        raise FileNotFoundError(f"找不到数据集文件: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not data:
        print("⚠️ 数据集为空")
        return

    print(f"✅ 总样本数: {len(data)}")

    counter = Counter(item["instruction"] for item in data)
    print("✅ instruction 分布:")
    for k, v in counter.items():
        print(f"  - {k}: {v}")

    noise_count = sum(has_noise(item["input"]) for item in data)
    prefix_count = sum(item["input"].startswith(item["output"]) for item in data)

    print(f"✅ 含明显噪音样本数: {noise_count}")
    print(f"✅ output 为 input 前缀的样本数: {prefix_count}")

    picked = random.sample(data, min(10, len(data)))

    for i, item in enumerate(picked, 1):
        print("=" * 80)
        print(f"[样本 {i}]")
        print("instruction:")
        print(item["instruction"])
        print("\ninput:")
        print(item["input"][:250])
        print("\noutput:")
        print(item["output"][:250])


if __name__ == "__main__":
    main()