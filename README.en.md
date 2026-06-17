# llm-finetune-for-manufacturing

> Repository: https://github.com/ShihangPENg-afk/llm-finetune-for-manufacturing  
> 中文说明：[README.md](README.md)

Standalone **LoRA fine-tuning experiment repo** for validating the full pipeline from PDF technical manuals to Alpaca-format datasets, then to [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) LoRA fine-tuning.

This repo does **not** aim to ship a production-ready model. It verifies the end-to-end workflow—data processing → dataset registration → LoRA training → weight export—on a local CPU setup, as a reproducible baseline before formal training on a GPU server.

---

## Related GitHub Repositories

| Repository | GitHub | Description |
|------------|--------|-------------|
| **llm-finetune-for-manufacturing** | https://github.com/ShihangPENg-afk/llm-finetune-for-manufacturing | This repo: PDF → LoRA fine-tuning experiment |
| **rag-agentic-system** | https://github.com/ShihangPENg-afk/rag-agentic-system | Agentic RAG main app (LoRA **not integrated yet**) |
| **predictive-maintenance-mini** | https://github.com/ShihangPENg-afk/predictive-maintenance-mini | Industrial ML inference API (no code dependency on this repo) |

---

## Relationship with rag-agentic-system

| Project | GitHub | Role | Status |
|---------|--------|------|--------|
| **rag-agentic-system** | https://github.com/ShihangPENg-afk/rag-agentic-system | Agentic RAG Q&A: PDF upload, FAISS retrieval, LangGraph agent, RAGAS eval | Main application repo (engineering POC) |
| **llm-finetune-for-manufacturing** | https://github.com/ShihangPENg-afk/llm-finetune-for-manufacturing | PDF → Alpaca dataset → Qwen2-7B LoRA fine-tuning | This repo |

**How they relate:**

- Both sit on the same PDF knowledge pipeline at different stages, but **code, dependencies, and deployment are fully independent**, each with its own virtual environment.
- **The LoRA fine-tuned model is not integrated into rag-agentic-system yet.** rag-agentic-system’s generation node and RAGAS evaluation still call the DashScope online API (`qwen-plus`); they do not load adapter weights from this repo’s `outputs/`.
- rag-agentic-system RAGAS metrics (faithfulness, answer_relevancy, etc.) **only reflect online API Q&A performance** and are not directly comparable to training loss or before/after results in this repo.

---

## Data Processing Pipeline

```
PDF (data/raw_pdfs/)
    │
    ▼  scripts/extract_chunks.py
chunks.json (text chunks)
    │
    ▼  scripts/build_alpaca_dataset.py
alpaca_train.json (Alpaca triplets)
    │
    ▼  dataset_info.json
LLaMA-Factory dataset registration (manual_alpaca)
```

| Step | Script | Input | Output | Notes |
|------|--------|-------|--------|-------|
| PDF chunking | `scripts/extract_chunks.py` | `data/raw_pdfs/*.pdf` | `data/processed/chunks.json` | PyMuPDF text extraction; ~800-char chunks |
| Build Alpaca data | `scripts/build_alpaca_dataset.py` | `chunks.json` | `data/processed/alpaca_train.json` | Noise cleanup, dedup, length filter; 4 instruction templates rotated |
| Dataset registration | — | `alpaca_train.json` | `data/processed/dataset_info.json` | Registers `manual_alpaca` for LLaMA-Factory |

---

## Dataset Info

| Attribute | Value |
|-----------|-------|
| Sample count | **132** (from local experiment with 5 PDFs) |
| Instruction templates | **4** (rotated to avoid single-instruction overfitting) |
| Format | Alpaca (`instruction` / `input` / `output`) |
| Dataset name | `manual_alpaca` |

`dataset_info.json` maps Alpaca fields via `prompt` / `query` / `response`:

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

Instruction template examples:

- Summarize the core information in the given material.
- Read the material below and extract key information.
- Summarize the main points of the following content.
- Explain what this material is mainly about.

*(Templates in the repo are in Chinese, matching the source PDF language.)*

---

## Training Configuration

| Item | Value |
|------|-------|
| Base model | `Qwen/Qwen2-7B-Instruct` |
| Framework | [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) (tested with `>=0.9.3`) |
| Method | LoRA (`rank: 4`, `alpha: 8`) |
| Environment | Intel MacBook Pro CPU (x86_64, 16GB RAM, no GPU/MPS) |
| Dataset | `manual_alpaca` |
| `max_samples` | **50** (sampled from 132 rows to reduce CPU training time) |
| `num_train_epochs` | **1** |
| Precision | float32 (`bf16` / `fp16` disabled for CPU stability) |
| Config file | `configs/qwen2_7b_lora_cpu.yaml` |
| Output | `outputs/qwen2_7b_lora_cpu/` (ignored by `.gitignore`) |

**Scope:** Local CPU is for **workflow verification only**, not model quality. 50 samples and 1 epoch are enough to confirm the pipeline and training script; formal training should run on a GPU server.

> **After clone:** `data/processed/*.json` and `data/raw_pdfs/*.pdf` are not committed. Place your own PDF manuals under `data/raw_pdfs/`, then run `extract_chunks.py` and `build_alpaca_dataset.py` to generate `chunks.json`, `alpaca_train.json`, and `dataset_info.json`. The 132-sample count in docs comes from a local run with 5 synthetic/demo PDFs (`manual1.pdf`–`manual5.pdf`).

---

## Run Commands

Clone this repository:

```bash
git clone https://github.com/ShihangPENg-afk/llm-finetune-for-manufacturing.git
cd llm-finetune-for-manufacturing
```

From the project root:

```bash
# 0. Install dependencies
pip install -r requirements.txt

# 1. PDF chunking
python scripts/extract_chunks.py data/raw_pdfs/

# 2. Build Alpaca dataset
python scripts/build_alpaca_dataset.py

# 3. Sample quality check
python scripts/sample_check.py

# 4. Start CPU LoRA training
bash scripts/train_qwen2_7b_lora_cpu.sh

# 5. Before/after eval (optional; 7B CPU inference is not practical locally)
python scripts/eval_before_after_cpu.py --phase before
python scripts/eval_before_after_cpu.py --phase after
```

After a successful run:

| Artifact | Path |
|----------|------|
| Text chunks | `data/processed/chunks.json` |
| Alpaca training data | `data/processed/alpaca_train.json` |
| LLaMA-Factory dataset config | `data/processed/dataset_info.json` |
| LoRA weights | `outputs/qwen2_7b_lora_cpu/adapter_model.safetensors` (local only, not in Git) |

---

## Current Limitations

- **Local hardware:** Experiments ran on Intel MacBook Pro CPU (16GB, no GPU). First base-model download ~1h+; 50 training steps ~4h; full 7B CPU inference often OOMs or takes 30–60+ min per question—**not suitable** for full local training or eval.
- **Workflow-only scope:** `max_samples: 50` and 1 epoch verify the pipeline only; **not sufficient for a usable domain model**.
- **LoRA not in rag-agentic-system:** Weights live under this repo’s `outputs/`; rag-agentic-system still uses DashScope online API—**not connected yet**.
- **No production auth or cloud deploy:** Training and weights are local experiment artifacts.

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/experiment_record.md](docs/experiment_record.md) | Experiment log (hardware, loss, adapter output) |
| [docs/delivery_checklist.md](docs/delivery_checklist.md) | Acceptance checklist for the CPU workflow |

---

## Roadmap

- Move data pipeline and training config to a GPU server; train on all 132 samples with more epochs
- Run before/after evaluation on GPU
- Evaluate integrating the LoRA adapter into rag-agentic-system’s generation node (**not integrated yet**)

---

## Directory Layout

```
data/
  raw_pdfs/              # Source PDFs (*.pdf gitignored; local only)
  processed/
    chunks.json          # Text chunks (generated locally, gitignored)
    alpaca_train.json    # Alpaca data (generated locally, gitignored)
    dataset_info.json    # LLaMA-Factory config (generated locally, gitignored)
configs/
  qwen2_7b_lora_cpu.yaml # CPU LoRA training config
scripts/
  extract_chunks.py
  build_alpaca_dataset.py
  sample_check.py
  train_qwen2_7b_lora_cpu.sh
  eval_before_after_cpu.py
outputs/                 # Training output (gitignored; local only)
docs/
  experiment_record.md
  delivery_checklist.md
LICENSE
requirements.txt
README.md                # Chinese
README.en.md             # English (this file)
```
