# llm-finetune-for-manufacturing

> 项目仓库：https://github.com/ShihangPENg-afk/llm-finetune-for-manufacturing  
> English: [README.en.md](README.en.md)

独立的 **LoRA 微调实验仓库**，用于验证从 PDF 技术手册到 Alpaca 格式数据集，再到 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) LoRA 微调的完整流程。

本仓库的目标不是提供生产级模型，而是在本地 CPU 环境下验证「数据处理 → 数据集注册 → LoRA 训练 → 权重保存」完整工作流，为后续在 GPU 服务器上进行正式训练提供可复现基线。

---

## 关联 GitHub 仓库

| 仓库 | GitHub | 说明 |
|------|--------|------|
| **llm-finetune-for-manufacturing** | https://github.com/ShihangPENg-afk/llm-finetune-for-manufacturing | 本仓库：PDF → LoRA 微调实验 |
| **rag-agent** | https://github.com/ShihangPENg-afk/rag-agent | Agentic RAG 主应用（LoRA **尚未接入**） |
| **industrial-health-demo** | https://github.com/ShihangPENg-afk/industrial-health-demo | 工业 ML 推理 API（与本仓库无代码依赖） |

---

## 与 rag-agent 的关系

| 项目 | GitHub | 职责 | 当前状态 |
|------|--------|------|----------|
| **rag-agent** | https://github.com/ShihangPENg-afk/rag-agent | Agentic RAG 问答服务：PDF 上传、FAISS 检索、LangGraph Agent、RAGAS 评估 | 主应用仓库（工程化 POC） |
| **llm-finetune-for-manufacturing** | https://github.com/ShihangPENg-afk/llm-finetune-for-manufacturing | PDF → Alpaca 数据集 → Qwen2-7B LoRA 微调 | 本仓库 |

**两者关系：**

- 同属 PDF 知识处理链路的不同阶段，但**代码、依赖与部署完全独立**，各自维护独立的虚拟环境。
- **当前 LoRA 微调模型尚未接入 rag-agent**。rag-agent 的生成节点与 RAGAS 评估仍调用 DashScope 在线 API（`qwen-plus`），未加载本仓库 `outputs/` 下的 adapter 权重。
- rag-agent 的 RAGAS 指标（faithfulness、answer_relevancy 等）**仅反映在线 API 下的问答表现**，与本仓库的训练 loss 或微调前后对比结果无直接关联。

---

## 数据处理流程

```
PDF（data/raw_pdfs/）
    │
    ▼  scripts/extract_chunks.py
chunks.json（文本切块）
    │
    ▼  scripts/build_alpaca_dataset.py
alpaca_train.json（Alpaca 三元组）
    │
    ▼  dataset_info.json
LLaMA-Factory 数据集注册（manual_alpaca）
```

| 步骤 | 脚本 | 输入 | 输出 | 说明 |
|------|------|------|------|------|
| PDF 切块 | `scripts/extract_chunks.py` | `data/raw_pdfs/*.pdf` | `data/processed/chunks.json` | PyMuPDF 抽取文本，按段落合并为 ~800 字符 chunk |
| 构造 Alpaca 数据 | `scripts/build_alpaca_dataset.py` | `chunks.json` | `data/processed/alpaca_train.json` | 清洗噪音、去重、长度过滤，4 种 instruction 模板轮换 |
| 数据集注册 | — | `alpaca_train.json` | `data/processed/dataset_info.json` | 向 LLaMA-Factory 注册 `manual_alpaca` 数据集 |

---

## 数据集信息

| 属性 | 值 |
|------|-----|
| 样本数 | **132**（本地实验：5 份 PDF） |
| instruction 模板数 | **4**（轮换分配，避免单一指令过拟合） |
| 数据格式 | Alpaca（`instruction` / `input` / `output`） |
| 数据集名称 | `manual_alpaca` |

`dataset_info.json` 通过 `prompt` / `query` / `response` 映射到 Alpaca 字段：

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

instruction 模板示例：

- 请根据给定资料，概括该段内容的核心信息。
- 请阅读下面的资料，并提取其中的关键信息。
- 请根据以下内容，总结这一段的重点。
- 请说明这段资料主要讲了什么。

---

## 训练配置

| 配置项 | 值 |
|--------|-----|
| 基座模型 | `Qwen/Qwen2-7B-Instruct` |
| 微调框架 | [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)（本地验证 `>=0.9.3`） |
| 微调方式 | LoRA（`rank: 4`, `alpha: 8`） |
| 训练环境 | Intel MacBook Pro CPU（x86_64，16GB 内存，无 GPU/MPS） |
| 数据集 | `manual_alpaca` |
| `max_samples` | **50**（从 132 条中抽样，降低 CPU 训练耗时） |
| `num_train_epochs` | **1** |
| 精度 | float32（`bf16` / `fp16` 关闭，CPU 更稳定） |
| 配置文件 | `configs/qwen2_7b_lora_cpu.yaml` |
| 训练输出 | `outputs/qwen2_7b_lora_cpu/`（已被 `.gitignore` 忽略） |

**定位说明：** 本地 CPU 环境仅用于**流程可复现验证**，而非追求模型效果。50 条样本、1 个 epoch 的配置足以确认数据管线与训练脚本可正常工作；正式训练应在 GPU 服务器上完成。

> **Clone 后数据处理：** 仓库不提交 `data/processed/*.json` 与 `data/raw_pdfs/*.pdf`。首次运行前须将 PDF 放入 `data/raw_pdfs/`，再执行 `extract_chunks.py` 与 `build_alpaca_dataset.py` 生成 `chunks.json`、`alpaca_train.json` 与 `dataset_info.json`。文档中的 132 条样本数来自本地 5 份合成/演示 PDF（`manual1.pdf` ~ `manual5.pdf`）的实验结果。

---

## 运行命令

克隆本仓库：

```bash
git clone https://github.com/ShihangPENg-afk/llm-finetune-for-manufacturing.git
cd llm-finetune-for-manufacturing
```

在项目根目录执行：

```bash
# 0. 安装依赖
pip install -r requirements.txt

# 1. PDF 切块
python scripts/extract_chunks.py data/raw_pdfs/

# 2. 构造 Alpaca 数据集
python scripts/build_alpaca_dataset.py

# 3. 抽样检查数据质量
python scripts/sample_check.py

# 4. 启动 CPU LoRA 训练
bash scripts/train_qwen2_7b_lora_cpu.sh

# 5. before/after 评测（可选；本地 CPU 上 7B 推理不具可行性）
python scripts/eval_before_after_cpu.py --phase before
python scripts/eval_before_after_cpu.py --phase after
```

运行完成后：

| 产物 | 路径 |
|------|------|
| 文本切块 | `data/processed/chunks.json` |
| Alpaca 微调数据 | `data/processed/alpaca_train.json` |
| LLaMA-Factory 数据集配置 | `data/processed/dataset_info.json` |
| LoRA 权重 | `outputs/qwen2_7b_lora_cpu/adapter_model.safetensors`（本地生成，不提交 Git） |

---

## 当前限制

- **本地硬件约束**：实验在 Intel MacBook Pro CPU（16GB、无 GPU）上完成。7B 基座首次下载约 1h+；50 step 训练约 4h；全量 CPU 推理易 OOM 或单题耗时过长，**不适合**本地完整训练与评测。
- **流程验证定位**：当前 `max_samples: 50`、1 epoch 仅为 workflow 验证，不足以获得可用领域模型。
- **LoRA 模型尚未接入 rag-agent**：微调权重保存在本仓库 `outputs/`，rag-agent 仍使用 DashScope 在线 API，两者尚未打通。
- **无生产鉴权、未云部署**：训练与权重均为本地实验产物。

---

## 文档

| 文档 | 说明 |
|------|------|
| [docs/experiment_record.md](docs/experiment_record.md) | 实验记录（硬件、loss、adapter 输出） |
| [docs/delivery_checklist.md](docs/delivery_checklist.md) | CPU 流程验收清单 |

---

## 后续计划

- 将数据管线与训练配置迁移至 GPU 服务器，使用全量 132 条样本与更多 epoch 完成正式训练
- 在 GPU 环境完成 before/after 效果评测
- 评估是否将 LoRA adapter 接入 rag-agent 生成节点（**当前尚未接入**）

---

## 目录结构

```
data/
  raw_pdfs/              # 原始 PDF（*.pdf 已被 .gitignore 忽略，本地放置）
  processed/
    chunks.json          # 文本切块（本地生成，.gitignore）
    alpaca_train.json    # Alpaca 微调数据（本地生成，.gitignore）
    dataset_info.json    # LLaMA-Factory 数据集配置（本地生成，.gitignore）
configs/
  qwen2_7b_lora_cpu.yaml # CPU LoRA 训练配置
scripts/
  extract_chunks.py      # PDF 切块
  build_alpaca_dataset.py
  sample_check.py        # 数据抽样检查
  train_qwen2_7b_lora_cpu.sh
  eval_before_after_cpu.py
outputs/                 # 训练输出（LoRA 权重、日志，已被 .gitignore 忽略）
docs/
  experiment_record.md   # 实验记录
  delivery_checklist.md  # 验收清单
LICENSE
requirements.txt
README.md                # 中文（本文件）
README.en.md             # 英文
```
