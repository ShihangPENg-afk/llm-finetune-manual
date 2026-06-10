#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

# CPU 可用训练脚本
llamafactory-cli train configs/qwen2_7b_lora_cpu.yaml