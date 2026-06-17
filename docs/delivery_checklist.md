# llm-finetune-for-manufacturing 验收清单

> 最后核对：2026-06-10  
> 关联文档：[README.md](../README.md) · [README.en.md](../README.en.md) · [experiment_record.md](experiment_record.md)  
> 关联主项目：[rag-agentic-system](https://github.com/ShihangPENg-afk/rag-agentic-system)（LoRA 尚未接入）

---

## 验收状态总览

| 类别 | 状态 |
|------|------|
| 数据处理（PDF → Alpaca） | 已完成 |
| LLaMA-Factory 配置与训练 | 已完成 |
| 文档与实验记录 | 已完成 |
| 效果评测与 rag-agentic-system 接入 | 未完成（见第 9 节） |

---

## 1. PDF 切块是否完成

- [x] `data/raw_pdfs/` 下共 **5** 份 PDF（`manual1.pdf` ~ `manual5.pdf`）
- [x] `scripts/extract_chunks.py` 可正常执行
- [x] 输出 `data/processed/chunks.json`，共 **132** 个 chunk

**验收命令：**

```bash
python scripts/extract_chunks.py data/raw_pdfs/
python -c "import json; print(len(json.load(open('data/processed/chunks.json'))))"
# 预期输出：132
```

---

## 2. Alpaca 数据集是否生成

- [x] `scripts/build_alpaca_dataset.py` 可正常执行
- [x] 输出 `data/processed/alpaca_train.json`，共 **132** 条样本
- [x] 每条样本包含 `instruction` / `input` / `output` 三元组
- [x] **4** 种 instruction 模板轮换分配（各 33 条）

**验收命令：**

```bash
python scripts/build_alpaca_dataset.py
python -c "import json; d=json.load(open('data/processed/alpaca_train.json')); print(len(d), d[0].keys())"
# 预期输出：132 dict_keys(['instruction', 'input', 'output'])
```

---

## 3. sample_check 是否通过

- [x] `scripts/sample_check.py` 存在且可执行
- [x] 总样本数：**132**
- [x] instruction 分布均衡（4 种模板各 33 条）
- [x] 含明显噪音样本数：**0**
- [x] 随机抽查 10 条，字段完整、内容可读

**验收命令：**

```bash
python scripts/sample_check.py
```

**备注：** `output 为 input 前缀的样本数: 4` 为摘要构造策略所致，属预期现象，不影响数据集可用性。

---

## 4. dataset_info.json 是否符合 LLaMA-Factory 要求

- [x] 文件路径：`data/processed/dataset_info.json`
- [x] 数据集名称：`manual_alpaca`
- [x] `file_name` 指向同目录下 `alpaca_train.json`
- [x] `formatting` 为 `alpaca`
- [x] `columns` 映射正确：
  - `prompt` → `instruction`
  - `query` → `input`
  - `response` → `output`
- [x] 训练配置 `configs/qwen2_7b_lora_cpu.yaml` 中 `dataset_dir: data/processed`、`dataset: manual_alpaca` 与之一致

**验收要点：**

```json
{
  "manual_alpaca": {
    "file_name": "alpaca_train.json",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
```

---

## 5. CPU LoRA YAML 是否存在

- [x] 配置文件：`configs/qwen2_7b_lora_cpu.yaml`
- [x] 基座模型：`Qwen/Qwen2-7B-Instruct`
- [x] 微调方式：`finetuning_type: lora`
- [x] 运行环境：CPU（`bf16: false`, `fp16: false`）
- [x] 训练收敛：`max_samples: 50`，`num_train_epochs: 1`

---

## 6. 训练脚本是否存在

- [x] 启动脚本：`scripts/train_qwen2_7b_lora_cpu.sh`
- [x] 调用方式：`llamafactory-cli train configs/qwen2_7b_lora_cpu.yaml`
- [x] 可选评测脚本：`scripts/eval_before_after_cpu.py`

**验收命令：**

```bash
bash scripts/train_qwen2_7b_lora_cpu.sh
```

---

## 7. 训练输出是否存在

- [x] 输出目录：`outputs/qwen2_7b_lora_cpu/`（已被 `.gitignore` 忽略，本地已生成）
- [x] LoRA 权重：`adapter_model.safetensors`（约 40 MB）
- [x] LoRA 配置：`adapter_config.json`
- [x] 检查点：`checkpoint-50/`
- [x] loss 日志：`trainer_log.jsonl`
- [x] loss 曲线：`training_loss.png`
- [x] 训练汇总：`train_results.json`（平均 train_loss ≈ 0.42）

**训练结果摘要（2026-06-09）：**

| 指标 | 值 |
|------|-----|
| 是否成功启动 | 是 |
| 训练 step 数 | 50 |
| 训练耗时 | 约 4h14m |
| 最终 loss | 0.158（step 50） |
| adapter 文件 | 已生成 |

---

## 8. README 是否写明 CPU 限制

- [x] 说明实验环境为 Intel MacBook Pro CPU，无 GPU/MPS
- [x] 说明 `max_samples: 50`、1 epoch 仅为流程验证
- [x] 说明本地 CPU 硬件约束下不建议完整训练与推理评测
- [x] 说明 LoRA 模型尚未接入 rag-agentic-system
- [x] 说明后续应迁移至 GPU 服务器完成正式训练

**对应章节：** [README.md](../README.md) 中的「训练配置」「当前限制」「与 rag-agentic-system 的关系」。

---

## 9. 当前未完成项

| 项目 | 状态 | 说明 |
|------|------|------|
| before/after 效果对比 | 未完成 | `eval_before_after_cpu.py` 已就绪；16GB CPU 下 7B 推理不具可行性且易 OOM |
| 全量样本正式训练 | 未完成 | 当前仅训练 50/132 条、1 epoch，不足以获得可用领域模型 |
| GPU 环境迁移 | 未完成 | 需在 GPU 服务器上使用全量数据与更多 epoch 重训 |
| LoRA 接入 rag-agentic-system | 未完成 | rag-agentic-system 仍使用 DashScope 在线 API，未加载本仓库 adapter 权重 |
| 训练输出提交 Git | 不适用 | `outputs/` 已被 `.gitignore` 忽略，权重仅保留在本地 |
| 验证集与 eval 指标 | 未完成 | 训练配置未划分验证集，无 `eval_loss` / `eval_accuracy` |

---

## 验收结论

满足 **第 1–8 节** 全部勾选项后，可认为 llm-finetune-for-manufacturing **CPU 流程验证验收通过**。第 9 节未完成项作为后续迭代 backlog，不影响「PDF → Alpaca → LoRA 训练 → 权重保存」链路已跑通的结论。
