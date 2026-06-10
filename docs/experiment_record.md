# LoRA 微调实验记录

> 实验日期：2026-06-09  
> 仓库：[llm-finetune-manual](../README.md)  
> 关联主项目：[rag-agent](../../rag-agent)（当前 LoRA 模型尚未接入）

---

## 1. 实验目标

验证从 PDF 技术手册到 LoRA 微调权重的完整流程是否可在本机跑通，具体包括：

1. **数据处理**：PDF 抽取文本 → 切块 → 构造 Alpaca 格式数据集 → 注册 LLaMA-Factory 数据集
2. **模型训练**：在 CPU 环境下使用 LLaMA-Factory 对 `Qwen/Qwen2-7B-Instruct` 进行 LoRA 微调
3. **产物确认**：训练过程产生 loss 日志，最终保存 adapter 权重文件

本实验**不追求训练效果**，仅确认链路可行，为后续迁移至 GPU 环境完成正式训练提供基线。

---

## 2. 硬件环境

| 项目 | 配置 |
|------|------|
| 设备 | Intel MacBook Pro |
| 处理器 | x86_64 CPU |
| 内存 | 16 GB |
| GPU | **无 NVIDIA GPU**（亦无 Apple MPS 加速） |
| 操作系统 | macOS |
| 训练模式 | 纯 CPU，float32 |

---

## 3. 数据集信息

数据来源：`data/raw_pdfs/` 下的 5 份 PDF 技术手册。

| 指标 | 数值 |
|------|------|
| PDF 数量 | **5**（`manual1.pdf` ~ `manual5.pdf`） |
| chunk 数 | **132**（`data/processed/chunks.json`） |
| Alpaca 样本数 | **132**（`data/processed/alpaca_train.json`） |
| instruction 模板数 | **4**（轮换分配，避免单一指令过拟合） |
| 数据集名称 | `manual_alpaca` |
| 字段格式 | Alpaca：`instruction` / `input` / `output` |
| LLaMA-Factory 映射 | `prompt` → instruction，`query` → input，`response` → output |

数据处理脚本：

- PDF 切块：`scripts/extract_chunks.py`
- Alpaca 构造：`scripts/build_alpaca_dataset.py`
- 数据集配置：`data/processed/dataset_info.json`

---

## 4. LLaMA-Factory 配置

| 配置项 | 值 |
|--------|-----|
| 基座模型 | `Qwen/Qwen2-7B-Instruct` |
| 微调方式 | LoRA（`rank: 4`, `alpha: 8`, `dropout: 0.05`, `target: all`） |
| 运行设备 | **CPU**（`bf16` / `fp16` 均关闭） |
| 数据集 | `manual_alpaca` |
| `max_samples` | **50**（从 132 条中抽样，降低 CPU 训练耗时） |
| `num_train_epochs` | **1** |
| `per_device_train_batch_size` | 1 |
| `learning_rate` | 5.0e-5 |
| `cutoff_len` | 1024 |
| `template` | qwen |
| 配置文件 | `configs/qwen2_7b_lora_cpu.yaml` |
| 启动命令 | `bash scripts/train_qwen2_7b_lora_cpu.sh` |

可训练参数：10,092,544（占总参数 7,625,709,056 的 0.13%）。

---

## 5. 训练结果

### 5.1 是否成功启动

**是。** 训练正常启动并完成，无致命错误。

- 首次运行需从 Hugging Face 下载基座权重（约 1h13m）
- 训练 50 step 总耗时：**4h14m21s**（平均每 step 约 5 min）
- 非阻断警告：`Sliding Window Attention is enabled but not implemented for sdpa`；未配置验证集，无 `eval_loss`

### 5.2 是否生成 loss 输出

**是。** loss 持续下降，训练侧拟合正常。

| 记录位置 | 说明 |
|----------|------|
| `outputs/qwen2_7b_lora_cpu/trainer_log.jsonl` | 逐步 loss 日志 |
| `outputs/qwen2_7b_lora_cpu/training_loss.png` | loss 曲线图 |
| `outputs/qwen2_7b_lora_cpu/train_results.json` | 汇总指标 |

关键 loss 数值：

| step | loss |
|------|------|
| 5 | 0.9789 |
| 25 | 0.3606 |
| 50 | 0.1580 |
| 平均 train_loss | **0.4229** |

### 5.3 是否生成 adapter 文件

**是。** 权重已保存至 `outputs/qwen2_7b_lora_cpu/`。

| 文件 | 说明 |
|------|------|
| `adapter_model.safetensors` | LoRA 权重（约 40 MB） |
| `adapter_config.json` | LoRA 配置 |
| `checkpoint-50/` | 第 50 step 检查点 |
| `tokenizer.json` 等 | 配套 tokenizer 文件 |

### 5.4 Before / After 评测

**未完成。** 评测脚本 `scripts/eval_before_after_cpu.py` 已就绪，但 16GB CPU 环境下 7B 全量推理存在内存与速度瓶颈（单题可能需 30–60+ 分钟），before/after 对比留待 GPU 环境执行。

---

## 6. 当前限制

1. **CPU 训练 7B 模型极慢**：50 step 即需约 4 小时，不适合大规模调参或完整训练。
2. **CPU 推理瓶颈**：16GB 内存下 7B 全量推理易 OOM 或极慢，before/after 评测难以在本机完成。
3. **训练规模受限**：`max_samples=50`、1 epoch 仅为流程验证，不足以获得可用领域模型。
4. **实验定位**：当前实验用于**流程验证**，loss 下降仅说明模型在 50 条样本上拟合，不代表领域问答效果提升。
5. **尚未接入 rag-agent**：微调权重保存在本仓库 `outputs/`，rag-agent 仍使用 DashScope 在线 API，两者未打通。

---

## 7. 后续计划

| 优先级 | 计划 | 说明 |
|--------|------|------|
| 1 | **迁移到 GPU 环境** | 使用全量 132 条样本、更多 epoch 完成正式训练，缩短训练与推理时间 |
| 2 | **做 before/after 对比** | 在 GPU 或 32GB+ 内存环境运行 `eval_before_after_cpu.py`，对比基座模型与 LoRA 模型的领域回答 |
| 3 | **可选接入 rag-agent** | 评估微调效果后，将 LoRA adapter 接入 rag-agent 生成节点，替换或补充 DashScope 在线 API |

相关文档：

- 仓库说明：[README.md](../README.md)
- 训练配置：[configs/qwen2_7b_lora_cpu.yaml](../configs/qwen2_7b_lora_cpu.yaml)
- rag-agent 项目：[rag-agent/README.md](../../rag-agent/README.md)
