#!/usr/bin/env python3
"""可选脚本：7B CPU 推理极慢，仅作参考，不作为 Day3 强制验收项。"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2-7B-Instruct"
LORA_DIR = Path("outputs/qwen2_7b_lora_cpu")

QUESTIONS = [
    "这份资料主要讲什么？",
    "什么是需求分析？",
    "什么是特征工程？",
    "什么是 DNS 解析？",
    "什么是模型治理？",
]


def load_model(phase: str):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="cpu",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    if phase == "after":
        model = PeftModel.from_pretrained(model, str(LORA_DIR))
    model.eval()
    return tokenizer, model


def run_eval(phase: str) -> None:
    if phase == "after" and not LORA_DIR.exists():
        raise FileNotFoundError(f"找不到 LoRA 目录: {LORA_DIR}")

    print(f"Phase: {phase}（CPU float32，单题可能需 30–60+ 分钟）", flush=True)
    tokenizer, model = load_model(phase)

    for i, q in enumerate(QUESTIONS, 1):
        messages = [{"role": "user", "content": q}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt")
        print(f"Generating Q{i}...", flush=True)
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=32, use_cache=False)
        ans = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        print(f"[{phase} Q{i}] {q}\nAnswer: {ans}\n{'=' * 80}", flush=True)
        del inputs, outputs
        gc.collect()


def main():
    parser = argparse.ArgumentParser(
        description="可选：对比微调前后 5 条固定问题（CPU 7B 推理极慢）"
    )
    parser.add_argument(
        "--phase",
        choices=["before", "after"],
        default="before",
        help="before=基座模型，after=加载 LoRA 权重",
    )
    args = parser.parse_args()
    run_eval(args.phase)


if __name__ == "__main__":
    main()
